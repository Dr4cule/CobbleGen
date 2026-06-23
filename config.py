from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"


def _load_dotenv(env_file: Path) -> None:
    if not env_file.exists():
        return

    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


_load_dotenv(ENV_FILE)


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    return int(raw_value) if raw_value is not None else default


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    return float(raw_value) if raw_value is not None else default


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    raw_value = os.getenv(name)
    if not raw_value:
        return default.expanduser().resolve()

    if os.name == "nt" and raw_value.startswith("/"):
        return default.expanduser().resolve()

    value = Path(raw_value)
    if not value.is_absolute():
        value = BASE_DIR / value
    return value.expanduser().resolve()


FOOTAGE_DIR = _env_path("FOOTAGE_DIR", BASE_DIR / "footage")
MUSIC_DIR = _env_path("MUSIC_DIR", BASE_DIR / "music")
STORIES_DIR = _env_path("STORIES_DIR", BASE_DIR / "stories")
OUTPUT_DIR = _env_path("OUTPUT_DIR", BASE_DIR / "output")
TEMP_DIR = _env_path("TEMP_DIR", BASE_DIR / "temp")
STATE_FILE = _env_path("STATE_FILE", BASE_DIR / "pipeline_state.json")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY")
GEMINI_MODEL = _env_str("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_MODEL_FALLBACKS = tuple(
    model.strip()
    for model in _env_str("GEMINI_MODEL_FALLBACKS", "gemini-2.5-pro,gemini-2.5-flash").split(",")
    if model.strip()
)
OLLAMA_BASE_URL = _env_str("OLLAMA_BASE_URL", "http://master:11434").rstrip("/")


OLLAMA_MODEL = _env_str("OLLAMA_MODEL", "nemotron-3-super:cloud")


def _ollama_endpoints() -> tuple[tuple[str, str], ...]:
    """Resolve ordered (host_url, model) pairs to try, with failover.

    Reads OLLAMA_HOSTS (comma-separated) when present, otherwise falls back to
    OLLAMA_BASE_URL plus slave. Each entry may pin a model with ``url|model``
    syntax, so failover can target a host that serves a different model.
    Bare host names are upgraded to full http URLs on :11434. Inference always
    runs on these remote hosts (master/slave), never locally.
    """
    raw = _env_str("OLLAMA_HOSTS", "")
    candidates = [item.strip() for item in raw.split(",") if item.strip()]
    if not candidates:
        candidates = [
            f"{OLLAMA_BASE_URL}|{OLLAMA_MODEL}",
            f"http://slave:11434|{OLLAMA_MODEL}",
        ]

    endpoints: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        if "|" in candidate:
            host_part, model_part = candidate.split("|", 1)
            model = model_part.strip() or OLLAMA_MODEL
        else:
            host_part, model = candidate, OLLAMA_MODEL
        url = host_part.strip()
        url = url if "://" in url else f"http://{url}"
        if ":" not in url.split("://", 1)[1]:
            url = f"{url}:11434"
        url = url.rstrip("/")
        key = (url, model)
        if key not in seen:
            seen.add(key)
            endpoints.append(key)
    return tuple(endpoints)


OLLAMA_ENDPOINTS = _ollama_endpoints()
OLLAMA_HOSTS = tuple(url for url, _model in OLLAMA_ENDPOINTS)
ENABLE_OLLAMA_FALLBACK = _env_bool("ENABLE_OLLAMA_FALLBACK", True)
OLLAMA_REQUEST_TIMEOUT = _env_int("OLLAMA_REQUEST_TIMEOUT", 300)

TTS_BACKEND = _env_str("TTS_BACKEND", "nvidia_magpie")
TTS_VOICE = _env_str("TTS_VOICE", "en-US-GuyNeural")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_RIVA_SERVER = _env_str("NVIDIA_RIVA_SERVER", "grpc.nvcf.nvidia.com:443")
NVIDIA_RIVA_USE_SSL = _env_bool("NVIDIA_RIVA_USE_SSL", True)
NVIDIA_MAGPIE_FUNCTION_ID = _env_str("NVIDIA_MAGPIE_FUNCTION_ID", "877104f7-e885-42b9-8de8-f6e4c6303969")
NVIDIA_TTS_LANGUAGE_CODE = _env_str("NVIDIA_TTS_LANGUAGE_CODE", "en-US")
NVIDIA_TTS_VOICE = _env_str("NVIDIA_TTS_VOICE", "Magpie-Multilingual.HI-IN.HouZhen")
NVIDIA_TTS_SAMPLE_RATE_HZ = _env_int("NVIDIA_TTS_SAMPLE_RATE_HZ", 22050)
NVIDIA_TTS_MAX_CHARS = _env_int("NVIDIA_TTS_MAX_CHARS", 350)
# Per-chunk emotional narration. When enabled, each narration chunk is analysed
# for emotional content and the matching NVIDIA Magpie emotion voice variant
# (e.g. "...Mia.Angry") is used, so the delivery follows the script's mood.
# Falls back to NVIDIA_DEFAULT_EMOTION (or the plain voice) when a given emotion
# is unavailable for the chosen speaker.
# Emotion mode for NVIDIA Magpie:
#   "dominant" - detect ONE emotion for the whole reel (consistent voice, the
#                default; avoids timbre jumps between chunks)
#   "dynamic"  - switch emotion per chunk (more expressive, less consistent)
#   "off"      - always use NVIDIA_DEFAULT_EMOTION
NVIDIA_EMOTION_MODE = _env_str("NVIDIA_EMOTION_MODE", "dominant").strip().lower()
# Back-compat: NVIDIA_DYNAMIC_EMOTION=false forces "off".
NVIDIA_DYNAMIC_EMOTION = _env_bool("NVIDIA_DYNAMIC_EMOTION", True)
if not NVIDIA_DYNAMIC_EMOTION:
    NVIDIA_EMOTION_MODE = "off"
NVIDIA_DEFAULT_EMOTION = _env_str("NVIDIA_DEFAULT_EMOTION", "neutral")
# Normalise every chunk to a consistent loudness before stitching so the volume
# does not jump between chunks (a common cause of "inconsistent" NVIDIA audio).
NVIDIA_TTS_NORMALIZE_CHUNKS = _env_bool("NVIDIA_TTS_NORMALIZE_CHUNKS", True)
NVIDIA_TTS_TARGET_LUFS = _env_float("NVIDIA_TTS_TARGET_LUFS", -18.0)
NVIDIA_TTS_MAX_ATTEMPTS = _env_int("NVIDIA_TTS_MAX_ATTEMPTS", 2)
NVIDIA_TTS_CHUNK_CROSSFADE_MS = _env_int("NVIDIA_TTS_CHUNK_CROSSFADE_MS", 25)
NVIDIA_TTS_GRPC_MAX_MESSAGE_MB = _env_int("NVIDIA_TTS_GRPC_MAX_MESSAGE_MB", 64)

# --- ElevenLabs TTS (high-quality, very consistent voice; emotion inferred from
# the script automatically). Free tier is limited (~10k characters/month), so it
# is offered as a selectable engine rather than the default. ---
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVEN_API_KEY")
ELEVENLABS_VOICE_ID = _env_str("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")  # George
ELEVENLABS_MODEL = _env_str("ELEVENLABS_MODEL", "eleven_multilingual_v2")
ELEVENLABS_OUTPUT_FORMAT = _env_str("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")
ELEVENLABS_MAX_CHARS = _env_int("ELEVENLABS_MAX_CHARS", 9000)
ELEVENLABS_STABILITY = _env_float("ELEVENLABS_STABILITY", 0.45)
ELEVENLABS_SIMILARITY = _env_float("ELEVENLABS_SIMILARITY", 0.8)
ELEVENLABS_STYLE = _env_float("ELEVENLABS_STYLE", 0.45)
ELEVENLABS_SPEAKER_BOOST = _env_bool("ELEVENLABS_SPEAKER_BOOST", True)
ELEVENLABS_BASE_URL = _env_str("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io").rstrip("/")
VOX_CPM_URL = _env_str("VOX_CPM_URL", "http://master:8001/tts")
VOX_CPM_SPEAKER = _env_str("VOX_CPM_SPEAKER", "")
VOX_CPM_PROMPT_TEXT = _env_str("VOX_CPM_PROMPT_TEXT", "")
VOX_CPM_PROMPT_WAV = _env_str("VOX_CPM_PROMPT_WAV", "")
VOX_CPM_MAX_CHARS = _env_int("VOX_CPM_MAX_CHARS", 80)
PARLER_TTS_URL = _env_str("PARLER_TTS_URL", VOX_CPM_URL)
PARLER_SPEAKER_ID = _env_str("PARLER_SPEAKER_ID", _env_str("DEFAULT_SPEAKER_ID", VOX_CPM_SPEAKER or "aurora"))
PARLER_EMOTION = _env_str("PARLER_EMOTION", _env_str("DEFAULT_EMOTION", "dramatic"))
PARLER_VOICE_DESCRIPTION = _env_str("PARLER_VOICE_DESCRIPTION", "")
PARLER_TEMPERATURE = _env_float("PARLER_TEMPERATURE", 0.45)
PARLER_MAX_NEW_TOKENS = _env_int("PARLER_MAX_NEW_TOKENS", 2048)
PARLER_MIN_NEW_TOKENS = _env_int("PARLER_MIN_NEW_TOKENS", 384)
PARLER_TOKENS_PER_SECOND = _env_float("PARLER_TOKENS_PER_SECOND", 90.0)
PARLER_TOKEN_MARGIN = _env_float("PARLER_TOKEN_MARGIN", 1.25)
PARLER_LOCAL_CHUNKING_ENABLED = _env_bool("PARLER_LOCAL_CHUNKING_ENABLED", True)
PARLER_LOCAL_CHUNK_CHARS = _env_int("PARLER_LOCAL_CHUNK_CHARS", 160)
PARLER_MAX_ATTEMPTS = _env_int("PARLER_MAX_ATTEMPTS", 3)
PARLER_CHUNK_CROSSFADE_MS = _env_int("PARLER_CHUNK_CROSSFADE_MS", 30)
TTS_AUDIO_QUALITY_GUARD = _env_bool("TTS_AUDIO_QUALITY_GUARD", True)
TTS_STATIC_ZCR_THRESHOLD = _env_float("TTS_STATIC_ZCR_THRESHOLD", 0.24)
TTS_STATIC_RMS_DB_THRESHOLD = _env_float("TTS_STATIC_RMS_DB_THRESHOLD", -50.0)
TTS_STATIC_MAX_SECONDS = _env_int("TTS_STATIC_MAX_SECONDS", 3)
# A window only counts as static-like if its short-frame energy is FLAT (low
# coefficient of variation) AND it has no inter-word silences. Real speech --
# even loud, consonant-heavy narration with a high zero-crossing rate --
# modulates energy per syllable, so it stays well above this floor and is not
# flagged. Raise to catch more static; lower to be more forgiving.
TTS_STATIC_ENERGY_COV_MAX = _env_float("TTS_STATIC_ENERGY_COV_MAX", 0.18)
TTS_REQUEST_TIMEOUT = _env_int("TTS_REQUEST_TIMEOUT", 900)
USE_WHISPER_ALIGN = _env_bool("USE_WHISPER_ALIGN", True)
WHISPER_MODEL = _env_str("WHISPER_MODEL", "base")

UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
UNSPLASH_MAX_IMAGES = _env_int("UNSPLASH_MAX_IMAGES", 0)
UNSPLASH_SEARCH_URL = _env_str("UNSPLASH_SEARCH_URL", "https://api.unsplash.com/search/photos")
UNSPLASH_REQUEST_DELAY = _env_float("UNSPLASH_REQUEST_DELAY", 0.25)
UNSPLASH_PER_PAGE = _env_int("UNSPLASH_PER_PAGE", 5)
UNSPLASH_ORIENTATION = _env_str("UNSPLASH_ORIENTATION", "portrait")

OUTPUT_WIDTH = _env_int("OUTPUT_WIDTH", 1080)
OUTPUT_HEIGHT = _env_int("OUTPUT_HEIGHT", 1920)
OUTPUT_FPS = _env_int("OUTPUT_FPS", 30)
VIDEO_BITRATE = _env_str("VIDEO_BITRATE", "5M")
AUDIO_BITRATE = _env_str("AUDIO_BITRATE", "192k")
VIDEO_CODEC = _env_str("VIDEO_CODEC", "libx264")
VIDEO_RENDER_PREFER_GPU = _env_bool("VIDEO_RENDER_PREFER_GPU", True)
AUDIO_CODEC = _env_str("AUDIO_CODEC", "aac")
AUDIO_MASTERING_ENABLED = _env_bool("AUDIO_MASTERING_ENABLED", True)
AUDIO_MASTER_FILTER = _env_str(
    "AUDIO_MASTER_FILTER",
    "highpass=f=70,lowpass=f=8500,dynaudnorm=f=150:g=9,alimiter=limit=0.97",
)
BACKGROUND_MUSIC_ENABLED = _env_bool("BACKGROUND_MUSIC_ENABLED", True)
BACKGROUND_MUSIC_FILE = _env_str("BACKGROUND_MUSIC_FILE", "")
BACKGROUND_MUSIC_VOLUME = _env_float("BACKGROUND_MUSIC_VOLUME", 0.16)
NARRATION_VOLUME = _env_float("NARRATION_VOLUME", 1.0)
MUSIC_FADE_SEC = _env_float("MUSIC_FADE_SEC", 1.25)

SUBTITLE_FONT_SIZE = _env_int("SUBTITLE_FONT_SIZE", 68)
SUBTITLE_COLOR = _env_str("SUBTITLE_COLOR", "white")
SUBTITLE_OUTLINE_COLOR = _env_str("SUBTITLE_OUTLINE_COLOR", "black")
SUBTITLE_HIGHLIGHT_COL = _env_str("SUBTITLE_HIGHLIGHT_COL", "yellow")
SUBTITLE_OUTLINE_WIDTH = _env_int("SUBTITLE_OUTLINE_WIDTH", 3)
WORDS_PER_CAPTION_GROUP = _env_int("WORDS_PER_CAPTION_GROUP", 4)
SUBTITLE_V_MARGIN = max(_env_int("SUBTITLE_V_MARGIN", 120), 170)
SUBTITLE_STYLE_NAME = _env_str("SUBTITLE_STYLE_NAME", "Default")
SUBTITLE_MARGIN_H = _env_int("SUBTITLE_MARGIN_H", 60)
SUBTITLE_MAX_CHARS_PER_LINE = _env_int("SUBTITLE_MAX_CHARS_PER_LINE", 18)
SUBTITLE_MAX_LINES = _env_int("SUBTITLE_MAX_LINES", 2)

MAX_REEL_DURATION = _env_int("MAX_REEL_DURATION", 180)
MIN_REEL_DURATION = _env_int("MIN_REEL_DURATION", 20)
IMAGE_DISPLAY_SEC = _env_float("IMAGE_DISPLAY_SEC", 3.5)
IMAGE_FADE_SEC = _env_float("IMAGE_FADE_SEC", 0.45)
IMAGE_OPACITY = _env_float("IMAGE_OPACITY", 0.88)
IMAGE_BLUR_BG = _env_bool("IMAGE_BLUR_BG", True)
SHOW_INTRO_CARD = _env_bool("SHOW_INTRO_CARD", True)
INTRO_DURATION = _env_float("INTRO_DURATION", 2.5)
INTRO_NARRATION_ENABLED = _env_bool("INTRO_NARRATION_ENABLED", True)
OUTRO_NARRATION_ENABLED = _env_bool("OUTRO_NARRATION_ENABLED", True)
OUTRO_TEXT = _env_str("OUTRO_TEXT", "")
MIN_CLIP_GAP = _env_float("MIN_CLIP_GAP", 5.0)
# Optional: force a specific footage file (filename in FOOTAGE_DIR or absolute path).
# Used by the web UI footage picker. Empty means auto-select least-used footage.
FORCE_FOOTAGE_FILE = _env_str("FORCE_FOOTAGE_FILE", "")

STORY_TARGET_WPM = _env_int("STORY_TARGET_WPM", 145)
DEFAULT_SCENE_COUNT = _env_int("DEFAULT_SCENE_COUNT", 4)
MIN_SCENE_COUNT = _env_int("MIN_SCENE_COUNT", 3)
MAX_SCENE_COUNT = _env_int("MAX_SCENE_COUNT", 8)
SECONDS_PER_SCENE_IMAGE = _env_float("SECONDS_PER_SCENE_IMAGE", 22.0)
ENABLE_EXPRESSIVE_NARRATION = _env_bool("ENABLE_EXPRESSIVE_NARRATION", True)
NARRATION_STYLE_HINT = _env_str(
    "NARRATION_STYLE_HINT",
    "emotionally engaging American storyteller with natural pauses and stronger dramatic beats",
)
NARRATION_MAX_SENTENCE_WORDS = _env_int("NARRATION_MAX_SENTENCE_WORDS", 22)

DEFAULT_BOLD_FONTFILE = (
    r"C:\Windows\Fonts\arialbd.ttf"
    if os.name == "nt"
    else "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
)

SUBTITLE_FONT = _env_str("SUBTITLE_FONT", "Arial" if os.name == "nt" else "DejaVu Sans")
INTRO_TEXT_FONTFILE = _env_str("INTRO_TEXT_FONTFILE", DEFAULT_BOLD_FONTFILE)
INTRO_TEXT_SIZE = _env_int("INTRO_TEXT_SIZE", 54)
INTRO_TEXT_COLOR = _env_str("INTRO_TEXT_COLOR", "white")
INTRO_TEXT_BORDER_COLOR = _env_str("INTRO_TEXT_BORDER_COLOR", "black")
INTRO_TEXT_BORDER_WIDTH = _env_int("INTRO_TEXT_BORDER_WIDTH", 3)
INTRO_BLUR_SIGMA = _env_float("INTRO_BLUR_SIGMA", 6.0)
INTRO_DARKEN_MULTIPLIER = _env_float("INTRO_DARKEN_MULTIPLIER", 0.5)
INTRO_MAX_CHARS_PER_LINE = _env_int("INTRO_MAX_CHARS_PER_LINE", 24)
INTRO_LINE_SPACING = _env_int("INTRO_LINE_SPACING", 16)
IMAGE_BG_BLUR_SIGMA = _env_float("IMAGE_BG_BLUR_SIGMA", 8.0)

SCENE_CAPTION_FONTFILE = _env_str("SCENE_CAPTION_FONTFILE", DEFAULT_BOLD_FONTFILE)
SCENE_CAPTION_FONT_SIZE = _env_int("SCENE_CAPTION_FONT_SIZE", 44)
SCENE_CAPTION_TEXT_COLOR = _env_str("SCENE_CAPTION_TEXT_COLOR", "white")
SCENE_CAPTION_BOX_COLOR = _env_str("SCENE_CAPTION_BOX_COLOR", "black@0.45")
SCENE_CAPTION_BORDER_WIDTH = _env_int("SCENE_CAPTION_BORDER_WIDTH", 2)
SCENE_CAPTION_Y = _env_int("SCENE_CAPTION_Y", 150)
SCENE_CAPTION_MAX_CHARS_PER_LINE = _env_int("SCENE_CAPTION_MAX_CHARS_PER_LINE", 24)

VOX_PROMPT_SAMPLE_RATE = _env_int("VOX_PROMPT_SAMPLE_RATE", 16000)
VOX_PROMPT_CHANNELS = _env_int("VOX_PROMPT_CHANNELS", 1)
VOX_SILENCE_THRESHOLD = _env_int("VOX_SILENCE_THRESHOLD", 350)
VOX_SILENCE_PAD_MS = _env_int("VOX_SILENCE_PAD_MS", 120)
ENABLE_TTS_FALLBACK = _env_bool("ENABLE_TTS_FALLBACK", True)
TTS_FALLBACK_BACKEND = _env_str("TTS_FALLBACK_BACKEND", "nvidia_magpie")

# Optional: disable Gemini calls for offline or debugging runs.
DISABLE_GEMINI = _env_bool("DISABLE_GEMINI", True)
