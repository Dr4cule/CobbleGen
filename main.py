from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from config import (
    FOOTAGE_DIR,
    INTRO_NARRATION_ENABLED,
    MIN_REEL_DURATION,
    MUSIC_DIR,
    OUTRO_NARRATION_ENABLED,
    OUTRO_TEXT,
    OUTPUT_DIR,
    STORIES_DIR,
    TEMP_DIR,
)
from modules.image_fetcher import fetch_images_for_scenes
from modules.state_manager import StateManager
from modules.story_processor import process_story_file
from modules.subtitle_generator import generate_ass_subtitles
from modules.tts_engine import generate_speech, get_audio_duration, list_tts_voices
from modules.video_editor import assemble_video, select_background_music, select_footage_clip


LOGGER = logging.getLogger(__name__)
STEP_COUNT = 7

STEP_LABELS = {
    1: "Preparing workspace",
    2: "Analysing story with AI",
    3: "Generating narration",
    4: "Building karaoke subtitles",
    5: "Fetching scene images",
    6: "Selecting footage",
    7: "Rendering final reel",
}


def emit_event(event: str, **fields: Any) -> None:
    """Emit a single-line JSON progress event on stdout for the web UI.

    The line is prefixed with a sentinel so a parent process can distinguish
    structured progress from ordinary log output. Safe to call when no parent
    is listening; it simply prints to stdout.
    """
    payload = {"event": event, **fields}
    try:
        sys.stdout.write("@@REEL@@ " + json.dumps(payload) + "\n")
        sys.stdout.flush()
    except (OSError, ValueError):
        pass


def emit_step(step: int) -> None:
    emit_event("step", step=step, total=STEP_COUNT, label=STEP_LABELS.get(step, f"Step {step}"))


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return cleaned or "reel"


def _normalise_inline_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _build_intro_narration(meta: dict[str, Any]) -> str:
    configured = _normalise_inline_text(str(meta.get("intro_text") or ""))
    if configured:
        return configured

    hook = _normalise_inline_text(str(meta.get("hook") or "This story gets messy fast.")).rstrip(".!?")
    if not hook:
        return "This one starts with a problem."

    fallback_openers = [
        f"{hook}.",
        f"{hook}. That is where it starts going wrong.",
        f"{hook}. Keep watching, because it gets worse.",
        f"{hook}. Nobody was ready for what came next.",
    ]
    return fallback_openers[sum(ord(char) for char in hook) % len(fallback_openers)]


def _build_outro_narration(meta: dict[str, Any]) -> str:
    configured = _normalise_inline_text(str(meta.get("outro_text") or OUTRO_TEXT))
    if configured:
        return configured

    title = _normalise_inline_text(str(meta.get("title") or "this story"))
    hook = _normalise_inline_text(str(meta.get("hook") or "this one"))
    fallback_outros = [
        f"Would you have handled {title.lower()} the same way?",
        f"Tell me if you would have made the same call after {hook.lower()}.",
        "Drop your verdict in the comments.",
        "Would you have done anything differently?",
    ]
    seed = sum(ord(char) for char in f"{title}|{hook}")
    return fallback_outros[seed % len(fallback_outros)]


def _ensure_directories() -> None:
    for directory in (FOOTAGE_DIR, MUSIC_DIR, STORIES_DIR, OUTPUT_DIR, TEMP_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _clean_temp_dir() -> None:
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


def _story_candidates(args: argparse.Namespace, state_manager: StateManager) -> list[Path]:
    if args.story:
        path = Path(args.story).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Story file does not exist: {path}")
        if not args.reprocess and state_manager.is_processed(path):
            LOGGER.info("Skipping already-processed story: %s", path)
            return []
        return [path]

    if args.all:
        stories = sorted(STORIES_DIR.glob("*.txt"))
        if args.reprocess:
            return stories
        return [story for story in stories if not state_manager.is_processed(story)]

    return []


def _write_metadata_file(
    story_path: Path,
    output_video: Path,
    meta: dict[str, Any],
    footage_path: Path,
    clip_start: float,
    music_path: Path | None,
    scenes_with_images: list[dict[str, Any]],
    intro_text: str,
    outro_text: str,
) -> Path:
    safe_title = _safe_filename(str(meta["title"]))[:80]
    metadata_path = OUTPUT_DIR / f"{story_path.stem}_{safe_title}_meta.txt"

    hashtags = " ".join(f"#{tag.lstrip('#')}" for tag in meta["hashtags"])
    credits = sorted({scene["photographer"] for scene in scenes_with_images if scene.get("photographer")})
    credit_text = "\n".join(credits) if credits else "None"

    contents = "\n".join(
        [
            "TITLE",
            str(meta["title"]),
            "",
            "DESCRIPTION",
            str(meta["description"]),
            "",
            "HASHTAGS",
            hashtags,
            "",
            "HOOK",
            str(meta["hook"]),
            "",
            "INTRO",
            intro_text or "None",
            "",
            "OUTRO",
            outro_text or "None",
            "",
            "STORY SOURCE",
            str(story_path.resolve()),
            "",
            "FOOTAGE",
            f"{footage_path.name} @ {clip_start:.3f}s",
            "",
            "BACKGROUND MUSIC",
            music_path.name if music_path else "None",
            "",
            "PHOTO CREDITS",
            credit_text,
            "",
            "OUTPUT VIDEO",
            str(output_video.resolve()),
            "",
        ]
    )
    metadata_path.write_text(contents, encoding="utf-8")
    return metadata_path


def run_pipeline_for_story(story_path: Path, state_manager: StateManager) -> dict[str, Any]:
    """Run the full reel-generation pipeline for one story file.

    Args:
        story_path: Source story text file to process.
        state_manager: Shared JSON state manager.

    Returns:
        A dictionary describing the generated output artifacts.
    """
    emit_step(1)
    LOGGER.info("[%d/%d] Cleaning temporary workspace for %s", 1, STEP_COUNT, story_path.name)
    _clean_temp_dir()

    emit_step(2)
    LOGGER.info("[%d/%d] Processing story text with AI", 2, STEP_COUNT)
    story_result = process_story_file(story_path)
    final_text = story_result["final_text"]
    speech_text = story_result.get("speech_text", final_text)
    meta = story_result["meta"]
    if not final_text.strip():
        raise ValueError(f"Story file is empty after processing: {story_path}")

    intro_text = _build_intro_narration(meta) if INTRO_NARRATION_ENABLED else ""
    outro_text = _build_outro_narration(meta) if OUTRO_NARRATION_ENABLED else ""
    speech_text = " ".join(segment for segment in (intro_text, str(speech_text).strip(), outro_text) if segment)
    emit_event("meta", title=str(meta.get("title") or ""), hook=str(meta.get("hook") or ""))

    emit_step(3)
    LOGGER.info("[%d/%d] Generating narration audio", 3, STEP_COUNT)
    audio_path, word_timestamps = generate_speech(str(speech_text))
    audio_duration = get_audio_duration(audio_path)
    if audio_duration < MIN_REEL_DURATION:
        LOGGER.warning(
            "Narration is shorter than MIN_REEL_DURATION (%ss < %ss). Proceeding anyway.",
            round(audio_duration, 2),
            MIN_REEL_DURATION,
        )

    emit_step(4)
    LOGGER.info("[%d/%d] Building karaoke subtitles", 4, STEP_COUNT)
    subtitle_path = generate_ass_subtitles(word_timestamps)

    emit_step(5)
    LOGGER.info("[%d/%d] Fetching contextual images", 5, STEP_COUNT)
    scenes_with_images = fetch_images_for_scenes(meta["scenes"])

    emit_step(6)
    LOGGER.info("[%d/%d] Selecting and preparing footage", 6, STEP_COUNT)
    footage_path, clip_start, base_clip = select_footage_clip(state_manager, audio_duration)
    music_path = select_background_music()

    safe_title = _safe_filename(str(meta["title"]))[:80]
    output_video = OUTPUT_DIR / f"{story_path.stem}_{safe_title}.mp4"

    emit_step(7)
    LOGGER.info("[%d/%d] Assembling final reel", 7, STEP_COUNT)
    assemble_video(
        base_clip=base_clip,
        audio_path=audio_path,
        subtitle_path=subtitle_path,
        scenes_with_images=scenes_with_images,
        hook_text=intro_text or str(meta["hook"]),
        background_music_path=music_path,
        output_path=output_video,
    )

    metadata_path = _write_metadata_file(
        story_path=story_path,
        output_video=output_video,
        meta=meta,
        footage_path=footage_path,
        clip_start=clip_start,
        music_path=music_path,
        scenes_with_images=scenes_with_images,
        intro_text=intro_text,
        outro_text=outro_text,
    )
    state_manager.mark_processed(story_path)

    emit_event(
        "done",
        output_video=output_video.name,
        metadata=metadata_path.name,
        title=str(meta.get("title") or ""),
    )

    return {
        "story_path": story_path,
        "output_video": output_video,
        "metadata_path": metadata_path,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the reel pipeline.

    Args:
        None.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate short-form Reddit story reels.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--story", help="Process one specific story file.")
    selection.add_argument("--all", action="store_true", help="Process all unprocessed story files.")
    parser.add_argument("--reset-footage", action="store_true", help="Clear footage usage history.")
    parser.add_argument("--reset-stories", action="store_true", help="Clear processed story history.")
    parser.add_argument("--reprocess", action="store_true", help="Ignore processed-story state.")
    parser.add_argument("--list-voices", action="store_true", help="List available voices for the configured TTS backend and exit.")
    return parser.parse_args()


def main() -> int:
    """Run the CLI entrypoint for the Reddit reel pipeline.

    Args:
        None.

    Returns:
        Process exit code.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    args = parse_args()
    _ensure_directories()
    state_manager = StateManager()

    if args.list_voices:
        for voice in list_tts_voices():
            LOGGER.info("%s | %s | %s", voice.get("ShortName"), voice.get("Gender"), voice.get("Locale"))
        return 0

    if args.reset_footage:
        state_manager.reset_footage()
        LOGGER.info("Cleared footage usage history.")

    if args.reset_stories:
        state_manager.reset_processed_stories()
        LOGGER.info("Cleared processed story history.")

    if not args.story and not args.all:
        if args.reset_footage or args.reset_stories:
            return 0
        LOGGER.error("Choose either --story PATH or --all, or use a reset/list command.")
        return 1

    try:
        stories = _story_candidates(args, state_manager)
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        return 1

    if not stories:
        LOGGER.info("No stories to process.")
        return 0

    successes: list[dict[str, Any]] = []
    failures: list[tuple[Path, str]] = []

    for story_path in stories:
        try:
            result = run_pipeline_for_story(story_path, state_manager)
            successes.append(result)
            LOGGER.info("Finished %s -> %s", story_path.name, result["output_video"].name)
        except Exception as exc:
            LOGGER.exception("Pipeline failed for %s", story_path)
            emit_event("error", story=story_path.name, message=str(exc))
            failures.append((story_path, str(exc)))

    LOGGER.info("Summary: %d succeeded, %d failed.", len(successes), len(failures))
    for result in successes:
        LOGGER.info("Success: %s -> %s", result["story_path"].name, result["output_video"].resolve())
    for story_path, error in failures:
        LOGGER.error("Failure: %s -> %s", story_path.name, error)

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
