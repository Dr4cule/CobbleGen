"""FastAPI backend for the CobbleGen web UI."""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from modules.tts_engine import list_tts_voices
from webapp.jobs import MANAGER

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
_KNOWN_EMOTIONS = {"neutral", "calm", "happy", "sad", "angry", "fearful", "fear", "disgust", "surprise"}

app = FastAPI(title="CobbleGen", version="1.0.0")


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("._-")
    return cleaned or f"item_{int(time.time())}"


def _unique_path(directory: Path, filename: str) -> Path:
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    candidate = directory / filename
    index = 2
    while candidate.exists():
        candidate = directory / f"{stem}_{index}{suffix}"
        index += 1
    return candidate


def _media_files(directory: Path, extensions: set[str]) -> list[dict[str, Any]]:
    if not directory.exists():
        return []
    items = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in extensions:
            items.append(
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "size_mb": round(path.stat().st_size / 1024 / 1024, 1),
                }
            )
    return items


# ----------------------------------------------------------------------------
# Settings schema exposed to the UI. Each maps to a config value / env override.
# ----------------------------------------------------------------------------
SETTINGS_FIELDS = [
    {"key": "STORY_TARGET_WPM", "label": "Narration speed (WPM)", "type": "int", "min": 80, "max": 200, "group": "Narration"},
    {"key": "NVIDIA_DYNAMIC_EMOTION", "label": "Emotional narration (per script)", "type": "bool", "group": "Narration"},
    {"key": "MAX_REEL_DURATION", "label": "Max duration (s, 0=unlimited)", "type": "int", "min": 0, "max": 600, "group": "Narration"},
    {"key": "NARRATION_VOLUME", "label": "Narration volume", "type": "float", "min": 0.0, "max": 2.0, "group": "Audio"},
    {"key": "BACKGROUND_MUSIC_ENABLED", "label": "Background music", "type": "bool", "group": "Audio"},
    {"key": "BACKGROUND_MUSIC_VOLUME", "label": "Music volume", "type": "float", "min": 0.0, "max": 1.0, "group": "Audio"},
    {"key": "SUBTITLE_FONT_SIZE", "label": "Subtitle size", "type": "int", "min": 30, "max": 120, "group": "Subtitles"},
    {"key": "SUBTITLE_HIGHLIGHT_COL", "label": "Highlight colour", "type": "str", "group": "Subtitles"},
    {"key": "WORDS_PER_CAPTION_GROUP", "label": "Words per caption", "type": "int", "min": 1, "max": 8, "group": "Subtitles"},
    {"key": "SHOW_INTRO_CARD", "label": "Show intro card", "type": "bool", "group": "Scenes"},
    {"key": "IMAGE_DISPLAY_SEC", "label": "Image on-screen (s)", "type": "float", "min": 1.0, "max": 8.0, "group": "Scenes"},
    {"key": "UNSPLASH_MAX_IMAGES", "label": "Max scene images (0=all)", "type": "int", "min": 0, "max": 12, "group": "Scenes"},
    {"key": "VIDEO_BITRATE", "label": "Video bitrate", "type": "str", "group": "Output"},
    {"key": "VIDEO_RENDER_PREFER_GPU", "label": "Prefer GPU (NVENC)", "type": "bool", "group": "Output"},
]
SETTINGS_KEYS = {field["key"] for field in SETTINGS_FIELDS}


def _current_setting_value(key: str) -> Any:
    return getattr(config, key, "")


# ----------------------------------------------------------------------------
# Request models
# ----------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    story_text: str | None = None
    story_name: str | None = None
    story_file: str | None = None  # existing file in stories dir
    voice: str | None = None
    music_file: str | None = None
    footage_file: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>CobbleGen</h1><p>UI not built.</p>", status_code=200)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "tts_backend": config.TTS_BACKEND,
        "ollama_hosts": list(config.OLLAMA_HOSTS),
        "footage_count": len(_media_files(config.FOOTAGE_DIR, VIDEO_EXTENSIONS)),
        "music_count": len(_media_files(config.MUSIC_DIR, AUDIO_EXTENSIONS)),
    }


@app.get("/api/assets")
def assets() -> dict[str, Any]:
    stories = []
    if config.STORIES_DIR.exists():
        for path in sorted(config.STORIES_DIR.glob("*.txt")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            stories.append(
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "preview": text.strip()[:160],
                    "words": len(text.split()),
                }
            )
    return {
        "stories": stories,
        "footage": _media_files(config.FOOTAGE_DIR, VIDEO_EXTENSIONS),
        "music": _media_files(config.MUSIC_DIR, AUDIO_EXTENSIONS),
    }


_VOICE_CACHE: dict[str, Any] = {}


@app.get("/api/voices")
def voices() -> dict[str, Any]:
    cache_key = config.TTS_BACKEND
    if cache_key in _VOICE_CACHE:
        return _VOICE_CACHE[cache_key]
    try:
        raw = list_tts_voices()
    except Exception as exc:  # noqa: BLE001
        return {"backend": config.TTS_BACKEND, "voices": [], "error": str(exc)}
    voices_out = [
        {
            "id": v.get("ShortName") or v.get("Name") or "",
            "gender": v.get("Gender", ""),
            "locale": v.get("Locale", ""),
        }
        for v in raw
        if v.get("ShortName") or v.get("Name")
    ]
    result = {"backend": config.TTS_BACKEND, "voices": voices_out, "current": config.NVIDIA_TTS_VOICE if "nvidia" in config.TTS_BACKEND else config.TTS_VOICE}
    _VOICE_CACHE[cache_key] = result
    return result


_PREVIEW_TEXT = "Here's a quick taste of how your reel narration will sound."
_PREVIEW_CACHE: dict[str, Path] = {}
_PREVIEW_LOCK = threading.Lock()
PREVIEW_DIR = BASE_DIR / "temp_preview"


@app.get("/api/voice-preview")
def voice_preview(voice: str | None = None) -> FileResponse:
    """Generate a short narration sample for the given voice and return audio.

    Runs in its own temp directory (never the job temp dir) and serialises with
    a lock, so previewing a voice can never collide with a reel being rendered.
    """
    from modules import tts_engine

    cache_key = voice or "__default__"
    cached = _PREVIEW_CACHE.get(cache_key)
    if cached and cached.exists():
        return FileResponse(str(cached), media_type=_audio_mime(cached))

    voice_attr = "NVIDIA_TTS_VOICE" if "nvidia" in config.TTS_BACKEND.lower() else "TTS_VOICE"
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    with _PREVIEW_LOCK:
        original_temp = tts_engine.TEMP_DIR
        original_voice = getattr(tts_engine, voice_attr)
        tts_engine.TEMP_DIR = PREVIEW_DIR
        if voice:
            setattr(tts_engine, voice_attr, voice)
        try:
            audio_path, _ = tts_engine.generate_speech(_PREVIEW_TEXT)
            preview_path = PREVIEW_DIR / f"voice_preview_{_safe_name(cache_key)}{audio_path.suffix}"
            shutil.copyfile(audio_path, preview_path)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"Voice preview failed: {exc}")
        finally:
            tts_engine.TEMP_DIR = original_temp
            setattr(tts_engine, voice_attr, original_voice)

    _PREVIEW_CACHE[cache_key] = preview_path
    return FileResponse(str(preview_path), media_type=_audio_mime(preview_path))


def _audio_mime(path: Path) -> str:
    return {".mp3": "audio/mpeg", ".wav": "audio/wav"}.get(path.suffix.lower(), "audio/wav")


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    fields = []
    for field in SETTINGS_FIELDS:
        fields.append({**field, "value": _current_setting_value(field["key"])})
    return {"fields": fields}


def _voice_env_key() -> str:
    return "NVIDIA_TTS_VOICE" if "nvidia" in config.TTS_BACKEND.lower() else "TTS_VOICE"


@app.post("/api/generate")
def generate(req: GenerateRequest) -> dict[str, Any]:
    config.STORIES_DIR.mkdir(parents=True, exist_ok=True)

    if req.story_file:
        story_path = config.STORIES_DIR / Path(req.story_file).name
        if not story_path.exists():
            raise HTTPException(404, f"Story file not found: {req.story_file}")
        story_name = story_path.name
    elif req.story_text and req.story_text.strip():
        if len(req.story_text.strip()) < 40:
            raise HTTPException(400, "Story is too short - paste at least a couple of sentences.")
        stem = _safe_name(req.story_name or f"reel_{int(time.time())}")[:60]
        if not stem.endswith(".txt"):
            stem += ".txt"
        story_path = _unique_path(config.STORIES_DIR, stem)
        story_path.write_text(req.story_text.strip(), encoding="utf-8")
        story_name = story_path.name
    else:
        raise HTTPException(400, "Provide either story_text or story_file.")

    overrides: dict[str, Any] = {}
    # Whitelisted setting overrides
    for key, value in (req.settings or {}).items():
        if key in SETTINGS_KEYS and value not in (None, ""):
            overrides[key] = value
    if req.voice:
        overrides[_voice_env_key()] = req.voice
        # If the user explicitly picked an emotion variant (e.g. "...Mia.Angry"),
        # honour that fixed emotion instead of switching emotions per chunk.
        if req.voice.rsplit(".", 1)[-1].lower() in _KNOWN_EMOTIONS:
            overrides["NVIDIA_DYNAMIC_EMOTION"] = "false"
    if req.music_file:
        music_path = config.MUSIC_DIR / Path(req.music_file).name
        if music_path.exists():
            overrides["BACKGROUND_MUSIC_FILE"] = str(music_path)
    if req.footage_file:
        overrides["FORCE_FOOTAGE_FILE"] = Path(req.footage_file).name

    job = MANAGER.submit(story_name, story_path, overrides)
    return {"job_id": job.id, "job": job.to_dict()}


@app.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    return {"jobs": MANAGER.list_jobs()}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = MANAGER.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    ok = MANAGER.cancel(job_id)
    if not ok:
        raise HTTPException(400, "Job cannot be cancelled")
    return {"ok": True}


@app.get("/api/jobs/{job_id}/stream")
async def stream_job(job_id: str) -> StreamingResponse:
    job = MANAGER.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    async def event_generator():
        last_version = -1
        while True:
            snapshot = job.to_dict()
            if snapshot["version"] != last_version:
                last_version = snapshot["version"]
                yield f"data: {json.dumps(snapshot)}\n\n"
            if snapshot["status"] in {"done", "error", "cancelled"}:
                break
            await asyncio.sleep(0.4)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/library")
def library() -> dict[str, Any]:
    reels = []
    if config.OUTPUT_DIR.exists():
        for path in sorted(config.OUTPUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
            meta_path = path.with_name(path.stem + "_meta.txt")
            meta = _parse_meta(meta_path) if meta_path.exists() else {}
            reels.append(
                {
                    "name": path.name,
                    "size_mb": round(path.stat().st_size / 1024 / 1024, 1),
                    "mtime": path.stat().st_mtime,
                    "title": meta.get("TITLE", path.stem),
                    "description": meta.get("DESCRIPTION", ""),
                    "hashtags": meta.get("HASHTAGS", ""),
                    "hook": meta.get("HOOK", ""),
                }
            )
    return {"reels": reels}


def _parse_meta(meta_path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    lines = meta_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    current_key: str | None = None
    buffer: list[str] = []
    known = {"TITLE", "DESCRIPTION", "HASHTAGS", "HOOK", "INTRO", "OUTRO", "STORY SOURCE", "FOOTAGE", "BACKGROUND MUSIC", "PHOTO CREDITS", "OUTPUT VIDEO"}
    for line in lines:
        if line.strip() in known:
            if current_key is not None:
                result[current_key] = "\n".join(buffer).strip()
            current_key = line.strip()
            buffer = []
        else:
            buffer.append(line)
    if current_key is not None:
        result[current_key] = "\n".join(buffer).strip()
    return result


@app.get("/api/video/{name}")
def serve_video(name: str) -> FileResponse:
    path = config.OUTPUT_DIR / Path(name).name
    if not path.exists() or path.suffix.lower() != ".mp4":
        raise HTTPException(404, "Video not found")
    return FileResponse(str(path), media_type="video/mp4", filename=path.name)


@app.delete("/api/library/{name}")
def delete_reel(name: str) -> dict[str, Any]:
    path = config.OUTPUT_DIR / Path(name).name
    if not path.exists():
        raise HTTPException(404, "Reel not found")
    path.unlink()
    meta_path = path.with_name(path.stem + "_meta.txt")
    if meta_path.exists():
        meta_path.unlink()
    return {"ok": True}


_UPLOAD_TARGETS = {
    "footage": (lambda: config.FOOTAGE_DIR, VIDEO_EXTENSIONS),
    "music": (lambda: config.MUSIC_DIR, AUDIO_EXTENSIONS),
    "stories": (lambda: config.STORIES_DIR, {".txt"}),
}


@app.post("/api/upload/{kind}")
async def upload_asset(kind: str, file: UploadFile = File(...)) -> dict[str, Any]:
    if kind not in _UPLOAD_TARGETS:
        raise HTTPException(400, f"Unknown upload kind: {kind}")
    dir_getter, extensions = _UPLOAD_TARGETS[kind]
    target_dir = dir_getter()
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in extensions:
        raise HTTPException(400, f"Unsupported file type {suffix} for {kind}")
    dest = _unique_path(target_dir, _safe_name(Path(file.filename or "upload").name))
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return {"ok": True, "name": dest.name, "size": dest.stat().st_size}


@app.delete("/api/asset/{kind}/{name}")
def delete_asset(kind: str, name: str) -> dict[str, Any]:
    if kind not in _UPLOAD_TARGETS:
        raise HTTPException(400, f"Unknown asset kind: {kind}")
    dir_getter, _ = _UPLOAD_TARGETS[kind]
    path = dir_getter() / Path(name).name
    if not path.exists():
        raise HTTPException(404, "Asset not found")
    path.unlink()
    return {"ok": True}


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
