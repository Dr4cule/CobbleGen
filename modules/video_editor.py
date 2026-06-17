from __future__ import annotations

import logging
import random
import re
import subprocess
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import (
    AUDIO_BITRATE,
    AUDIO_CODEC,
    AUDIO_MASTER_FILTER,
    AUDIO_MASTERING_ENABLED,
    BACKGROUND_MUSIC_ENABLED,
    BACKGROUND_MUSIC_FILE,
    BACKGROUND_MUSIC_VOLUME,
    FOOTAGE_DIR,
    FORCE_FOOTAGE_FILE,
    IMAGE_BG_BLUR_SIGMA,
    IMAGE_BLUR_BG,
    IMAGE_DISPLAY_SEC,
    IMAGE_FADE_SEC,
    IMAGE_OPACITY,
    INTRO_BLUR_SIGMA,
    INTRO_DARKEN_MULTIPLIER,
    INTRO_DURATION,
    INTRO_TEXT_BORDER_COLOR,
    INTRO_TEXT_BORDER_WIDTH,
    INTRO_TEXT_COLOR,
    INTRO_TEXT_FONTFILE,
    INTRO_LINE_SPACING,
    INTRO_MAX_CHARS_PER_LINE,
    INTRO_TEXT_SIZE,
    MIN_CLIP_GAP,
    MUSIC_DIR,
    MUSIC_FADE_SEC,
    NARRATION_VOLUME,
    OUTPUT_FPS,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    SCENE_CAPTION_BORDER_WIDTH,
    SCENE_CAPTION_BOX_COLOR,
    SCENE_CAPTION_FONTFILE,
    SCENE_CAPTION_FONT_SIZE,
    SCENE_CAPTION_MAX_CHARS_PER_LINE,
    SCENE_CAPTION_TEXT_COLOR,
    SCENE_CAPTION_Y,
    SHOW_INTRO_CARD,
    TEMP_DIR,
    VIDEO_BITRATE,
    VIDEO_CODEC,
    VIDEO_RENDER_PREFER_GPU,
)
from modules.state_manager import StateManager


LOGGER = logging.getLogger(__name__)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
GPU_VIDEO_ENCODERS = {"h264_nvenc", "hevc_nvenc", "av1_nvenc"}


def _run_command(command: list[str], context: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr_tail = (completed.stderr or "")[-2000:]
        LOGGER.error("%s failed: %s", context, stderr_tail)
        raise RuntimeError(f"{context} failed with exit code {completed.returncode}")
    return completed


def _ffprobe_duration(media_path: str | Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(Path(media_path).expanduser().resolve()),
    ]
    completed = _run_command(command, "ffprobe media duration")
    return float((completed.stdout or "0").strip() or 0.0)


def _media_files(directory: Path, extensions: set[str]) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        file_path
        for file_path in directory.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in extensions
    )


@lru_cache(maxsize=1)
def _ffmpeg_available_encoders() -> set[str]:
    completed = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=False)
    encoders: set[str] = set()
    for line in (completed.stdout or "").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        if not parts[0] or parts[0][0] not in {"V", "A", "S", "D", "T", "."}:
            continue
        encoders.add(parts[1])
    return encoders


def _bitrate_to_kbps(value: str) -> int:
    """Parse an FFmpeg-style bitrate string (e.g. "5M", "4500k") to kbps."""
    raw = (value or "").strip().lower()
    match = re.match(r"^([0-9]*\.?[0-9]+)\s*([mk]?)", raw)
    if not match:
        return 5000
    number = float(match.group(1))
    unit = match.group(2)
    if unit == "m":
        return int(number * 1000)
    if unit == "k":
        return int(number)
    return int(number / 1000) if number > 10000 else int(number)


def _nvenc_rate_args() -> list[str]:
    """NVENC quality args with a hard bitrate cap so files don't balloon."""
    target_kbps = _bitrate_to_kbps(VIDEO_BITRATE)
    maxrate_kbps = int(target_kbps * 1.5)
    bufsize_kbps = target_kbps * 2
    return [
        "-rc", "vbr",
        "-cq", "26",
        "-b:v", f"{target_kbps}k",
        "-maxrate", f"{maxrate_kbps}k",
        "-bufsize", f"{bufsize_kbps}k",
    ]


def _x264_rate_args() -> list[str]:
    """libx264 quality args with a bitrate cap matching VIDEO_BITRATE."""
    target_kbps = _bitrate_to_kbps(VIDEO_BITRATE)
    maxrate_kbps = int(target_kbps * 1.5)
    bufsize_kbps = target_kbps * 2
    return [
        "-crf", "23",
        "-maxrate", f"{maxrate_kbps}k",
        "-bufsize", f"{bufsize_kbps}k",
    ]


@lru_cache(maxsize=1)
def _video_encoder_args() -> list[str]:
    available_encoders = _ffmpeg_available_encoders()
    requested_codec = VIDEO_CODEC.strip().lower()

    if VIDEO_RENDER_PREFER_GPU and "h264_nvenc" in available_encoders:
        LOGGER.info("Using FFmpeg h264_nvenc for GPU video rendering.")
        return ["-c:v", "h264_nvenc", "-preset", "p5", *_nvenc_rate_args()]

    if requested_codec in {"", "auto"}:
        requested_codec = "libx264"

    if requested_codec in GPU_VIDEO_ENCODERS:
        if requested_codec in available_encoders:
            return ["-c:v", requested_codec, "-preset", "p5", *_nvenc_rate_args()]
        LOGGER.warning("Requested GPU codec %s is unavailable; falling back to libx264.", requested_codec)
        return ["-c:v", "libx264", "-preset", "medium", *_x264_rate_args()]

    if requested_codec == "libx264":
        return ["-c:v", "libx264", "-preset", "medium", *_x264_rate_args()]

    if requested_codec in available_encoders:
        return ["-c:v", requested_codec]

    LOGGER.warning("Requested VIDEO_CODEC=%s is unavailable; falling back to libx264.", VIDEO_CODEC)
    return ["-c:v", "libx264", "-preset", "medium", *_x264_rate_args()]


def select_background_music() -> Path | None:
    """Return a background music file from the configured drag-and-drop folder."""
    if not BACKGROUND_MUSIC_ENABLED:
        return None

    if BACKGROUND_MUSIC_FILE.strip():
        configured_path = Path(BACKGROUND_MUSIC_FILE).expanduser().resolve()
        if configured_path.exists():
            return configured_path
        LOGGER.warning("Configured BACKGROUND_MUSIC_FILE does not exist: %s", configured_path)

    music_files = _media_files(MUSIC_DIR, AUDIO_EXTENSIONS)
    if not music_files:
        LOGGER.info("No background music files found in %s; rendering narration only.", MUSIC_DIR)
        return None
    return random.choice(music_files)


def _escape_filter_path(path: Path) -> str:
    return path.resolve().as_posix().replace(":", "\\:").replace("'", "\\'")


def _wrap_display_text(text: str, max_chars: int) -> str:
    wrapper = textwrap.TextWrapper(width=max_chars, break_long_words=False, break_on_hyphens=False)
    wrapped_lines = wrapper.wrap(text.strip())
    return "\n".join(wrapped_lines) if wrapped_lines else text.strip()


def _build_intro_drawtext_filters(lines: list[str]) -> str:
    filters: list[str] = [
        f"gblur=sigma={INTRO_BLUR_SIGMA}",
        (
            f"colorchannelmixer=rr={INTRO_DARKEN_MULTIPLIER}:gg={INTRO_DARKEN_MULTIPLIER}:"
            f"bb={INTRO_DARKEN_MULTIPLIER}"
        ),
    ]
    line_step = INTRO_TEXT_SIZE + INTRO_LINE_SPACING
    offset_base = ((len(lines) - 1) * line_step) / 2

    for index, line in enumerate(lines):
        line_file = TEMP_DIR / f"hook_text_line_{index + 1:02d}.txt"
        line_file.write_text(line, encoding="utf-8")
        line_y_offset = round((index * line_step) - offset_base, 2)
        filters.append(
            f"drawtext=fontfile='{_escape_filter_path(Path(INTRO_TEXT_FONTFILE))}':"
            f"textfile='{_escape_filter_path(line_file)}':fontcolor={INTRO_TEXT_COLOR}:"
            f"bordercolor={INTRO_TEXT_BORDER_COLOR}:borderw={INTRO_TEXT_BORDER_WIDTH}:"
            f"fontsize={INTRO_TEXT_SIZE}:box=1:boxcolor=black@0.22:boxborderw=24:"
            f"x=(w-text_w)/2:y=(h/2-text_h/2)+({line_y_offset})"
        )

    return ",".join(filters)


def _format_ass_timestamp(seconds: float) -> str:
    centiseconds = max(0, int(round(seconds * 100)))
    hours, rem = divmod(centiseconds, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _parse_ass_timestamp(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    secs, centiseconds = seconds.split(".")
    return int(hours) * 3600 + int(minutes) * 60 + int(secs) + int(centiseconds) / 100


def _offset_subtitles(subtitle_path: Path, offset_seconds: float) -> Path:
    shifted_path = TEMP_DIR / "subtitles_shifted.ass"
    shifted_lines = []
    for line in subtitle_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("Dialogue:"):
            shifted_lines.append(line)
            continue
        parts = line.split(",", 9)
        if len(parts) < 10:
            shifted_lines.append(line)
            continue
        parts[1] = _format_ass_timestamp(_parse_ass_timestamp(parts[1]) + offset_seconds)
        parts[2] = _format_ass_timestamp(_parse_ass_timestamp(parts[2]) + offset_seconds)
        shifted_lines.append(",".join(parts))
    shifted_path.write_text("\n".join(shifted_lines) + "\n", encoding="utf-8")
    return shifted_path


def _suppress_subtitles_before(subtitle_path: Path, start_seconds: float) -> Path:
    suppressed_path = TEMP_DIR / "subtitles_intro_suppressed.ass"
    output_lines = []
    for line in subtitle_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("Dialogue:"):
            output_lines.append(line)
            continue
        parts = line.split(",", 9)
        if len(parts) < 10:
            output_lines.append(line)
            continue
        start = _parse_ass_timestamp(parts[1])
        end = _parse_ass_timestamp(parts[2])
        if start < start_seconds:
            continue
        output_lines.append(",".join(parts))
    suppressed_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return suppressed_path


def _prepare_clip(footage_path: str | Path, start: float, duration: float) -> Path:
    """Extract and format a vertical base clip from source footage.

    Args:
        footage_path: Source Minecraft footage file.
        start: Start time in seconds.
        duration: Clip duration in seconds.

    Returns:
        Path to the prepared base clip.
    """
    output_path = TEMP_DIR / "base_clip.mp4"
    video_filter = (
        f"crop=in_h*9/16:in_h:(in_w-in_h*9/16)/2:0,"
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos,setsar=1"
    )
    encoder_args = _video_encoder_args()
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(Path(footage_path).expanduser().resolve()),
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-fflags",
        "+genpts",
        "-avoid_negative_ts",
        "make_zero",
        "-vf",
        video_filter,
        "-r",
        str(OUTPUT_FPS),
        "-an",
        *encoder_args,
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    _run_command(command, "FFmpeg base clip preparation")
    return output_path


def _make_intro_card(hook_text: str, base_clip: str | Path) -> Path | None:
    """Create a blurred intro card using the story hook text.

    Args:
        hook_text: Story hook to draw on screen.
        base_clip: Prepared base clip path.

    Returns:
        Path to the intro card clip, or None if creation fails.
    """
    intro_path = TEMP_DIR / "intro_card.mp4"
    wrapped_lines = _wrap_display_text(hook_text.strip(), INTRO_MAX_CHARS_PER_LINE).splitlines()
    video_filter = _build_intro_drawtext_filters([line for line in wrapped_lines if line.strip()])
    encoder_args = _video_encoder_args()
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(Path(base_clip).expanduser().resolve()),
        "-t",
        f"{INTRO_DURATION:.3f}",
        "-vf",
        video_filter,
        "-an",
        "-r",
        str(OUTPUT_FPS),
        *encoder_args,
        "-pix_fmt",
        "yuv420p",
        str(intro_path),
    ]

    try:
        _run_command(command, "FFmpeg intro card generation")
        return intro_path
    except RuntimeError as exc:
        LOGGER.warning("Intro card generation failed, continuing without intro: %s", exc)
        return None


def _trim_video_segment(video_path: str | Path, start: float, duration: float, output_name: str) -> Path | None:
    if duration <= 0.05:
        return None

    output_path = TEMP_DIR / output_name
    encoder_args = _video_encoder_args()
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(Path(video_path).expanduser().resolve()),
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-fflags",
        "+genpts",
        "-avoid_negative_ts",
        "make_zero",
        "-an",
        "-r",
        str(OUTPUT_FPS),
        *encoder_args,
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    _run_command(command, "FFmpeg timeline segment trim")
    return output_path


def _replace_timeline_intro(timeline: Path, intro_card: Path, intro_duration: float) -> Path:
    timeline_duration = _ffprobe_duration(timeline)
    if timeline_duration <= intro_duration + 0.05:
        return intro_card

    remainder = _trim_video_segment(
        timeline,
        intro_duration,
        timeline_duration - intro_duration,
        "timeline_after_intro.mp4",
    )
    if remainder is None:
        return intro_card
    return _concat_video_segments([intro_card, remainder])


def _scene_caption_drawtext(scene: dict[str, Any], index: int, start: float, end: float) -> str:
    """Build a drawtext filter that labels a scene image while it is on screen.

    Returns an empty string when the scene has no caption, so the caller can
    skip the extra filter stage entirely.
    """
    caption = str(scene.get("caption") or "").strip()
    if not caption:
        return ""

    wrapped_lines = [line for line in _wrap_display_text(caption, SCENE_CAPTION_MAX_CHARS_PER_LINE).splitlines() if line.strip()]
    if not wrapped_lines:
        return ""

    fade = min(IMAGE_FADE_SEC, max((end - start) / 2, 0.05))
    alpha = (
        f"if(lt(t,{start:.3f}),0,"
        f"if(lt(t,{start + fade:.3f}),(t-{start:.3f})/{fade:.3f},"
        f"if(lt(t,{max(start, end - fade):.3f}),1,"
        f"if(lt(t,{end:.3f}),({end:.3f}-t)/{fade:.3f},0))))"
    )

    line_step = SCENE_CAPTION_FONT_SIZE + 10
    draws: list[str] = []
    for line_index, line in enumerate(wrapped_lines):
        caption_file = TEMP_DIR / f"scene_caption_{index:02d}_{line_index + 1:02d}.txt"
        caption_file.write_text(line, encoding="utf-8")
        y_offset = SCENE_CAPTION_Y + line_index * line_step
        draws.append(
            f"drawtext=fontfile='{_escape_filter_path(Path(SCENE_CAPTION_FONTFILE))}':"
            f"textfile='{_escape_filter_path(caption_file)}':fontcolor={SCENE_CAPTION_TEXT_COLOR}:"
            f"fontsize={SCENE_CAPTION_FONT_SIZE}:borderw={SCENE_CAPTION_BORDER_WIDTH}:bordercolor=black:"
            f"box=1:boxcolor={SCENE_CAPTION_BOX_COLOR}:boxborderw=18:"
            f"x=(w-text_w)/2:y={y_offset}:alpha='{alpha}':"
            f"enable='between(t,{start:.3f},{end:.3f})'"
        )

    return f"[ov{index}]" + ",".join(draws)


def _composite_images(base_clip: str | Path, scenes_with_images: list[dict[str, Any]], total_duration: float) -> Path:
    """Overlay fetched context images onto the prepared base clip.

    Args:
        base_clip: Prepared base video clip path.
        scenes_with_images: Scene metadata with optional `image_path` values.
        total_duration: Total reel duration in seconds.

    Returns:
        Path to the composited clip.
    """
    active_scenes = [scene for scene in scenes_with_images if scene.get("image_path")]
    if not active_scenes:
        return Path(base_clip).expanduser().resolve()

    output_path = TEMP_DIR / "composited_clip.mp4"
    encoder_args = _video_encoder_args()
    command = ["ffmpeg", "-y", "-i", str(Path(base_clip).expanduser().resolve())]
    filter_parts: list[str] = []
    previous_label = "[0:v]"

    for input_index, scene in enumerate(active_scenes, start=1):
        command.extend(["-loop", "1", "-t", f"{total_duration:.3f}", "-i", str(Path(scene["image_path"]).resolve())])

        start = max(0.0, min(float(scene.get("fraction", 0.0)) * total_duration, max(total_duration - IMAGE_DISPLAY_SEC, 0.0)))
        end = min(total_duration, start + IMAGE_DISPLAY_SEC)
        if end <= start:
            continue

        fade_duration = min(IMAGE_FADE_SEC, max((end - start) / 2, 0.1))
        image_label = f"img{input_index}"
        base_label = f"base{input_index}"
        overlay_label = f"overlay{input_index}"
        output_label = f"out{input_index}"

        filter_parts.append(
            f"[{input_index}:v]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"format=rgba,colorchannelmixer=aa={IMAGE_OPACITY},"
            f"fade=t=in:st={start:.3f}:d={fade_duration:.3f}:alpha=1,"
            f"fade=t=out:st={max(start, end - fade_duration):.3f}:d={fade_duration:.3f}:alpha=1"
            f"[{image_label}]"
        )

        if IMAGE_BLUR_BG:
            filter_parts.append(
                f"{previous_label}gblur=sigma={IMAGE_BG_BLUR_SIGMA}:enable='between(t,{start:.3f},{end:.3f})'[{base_label}]"
            )
        else:
            filter_parts.append(f"{previous_label}null[{base_label}]")

        caption_filter = _scene_caption_drawtext(scene, input_index, start, end)
        if caption_filter:
            filter_parts.append(
                f"[{base_label}][{image_label}]overlay=0:0:enable='between(t,{start:.3f},{end:.3f})':"
                f"eof_action=pass[ov{input_index}],{caption_filter}[{overlay_label}]"
            )
        else:
            filter_parts.append(
                f"[{base_label}][{image_label}]overlay=0:0:enable='between(t,{start:.3f},{end:.3f})':"
                f"eof_action=pass[{overlay_label}]"
            )
        previous_label = f"[{overlay_label}]"

    command.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            previous_label,
            "-an",
            "-r",
            str(OUTPUT_FPS),
            *encoder_args,
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
    )
    _run_command(command, "FFmpeg image compositing")
    return output_path


def _concat_video_segments(video_paths: list[Path]) -> Path:
    concat_file = TEMP_DIR / "concat.txt"
    concat_file.write_text(
        "\n".join(f"file '{path.resolve()}'" for path in video_paths) + "\n",
        encoding="utf-8",
    )

    output_path = TEMP_DIR / "timeline_clip.mp4"
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-fflags",
        "+genpts",
        "-avoid_negative_ts",
        "make_zero",
        "-vsync",
        "cfr",
        "-c",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    _run_command(command, "FFmpeg video concat")
    return output_path


def assemble_video(
    base_clip: str | Path,
    audio_path: str | Path,
    subtitle_path: str | Path,
    scenes_with_images: list[dict[str, Any]],
    hook_text: str,
    background_music_path: str | Path | None,
    output_path: str | Path,
) -> Path:
    """Assemble the final reel video from prepared assets.

    Args:
        base_clip: Prepared footage clip path.
        audio_path: Narration audio path.
        subtitle_path: ASS subtitle path.
        scenes_with_images: Enriched scenes with optional image paths.
        hook_text: Hook text for the optional intro card.
        background_music_path: Optional background music file to loop under narration.
        output_path: Final video output path.

    Returns:
        Path to the rendered output video.
    """
    base_clip_path = Path(base_clip).expanduser().resolve()
    audio_path_obj = Path(audio_path).expanduser().resolve()
    subtitle_path_obj = Path(subtitle_path).expanduser().resolve()
    music_path_obj = Path(background_music_path).expanduser().resolve() if background_music_path else None
    output_path_obj = Path(output_path).expanduser().resolve()
    encoder_args = _video_encoder_args()

    timeline = _composite_images(base_clip_path, scenes_with_images, _ffprobe_duration(base_clip_path))
    intro_applied = False

    if SHOW_INTRO_CARD:
        intro_card = _make_intro_card(hook_text, base_clip_path)
        if intro_card is not None:
            timeline = _replace_timeline_intro(timeline, intro_card, INTRO_DURATION)
            intro_applied = True

    subtitle_input = subtitle_path_obj
    try:
        ass_filter_path = subtitle_input.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        ass_filter_path = _escape_filter_path(subtitle_input)
    ass_filter = f"ass={ass_filter_path}"
    timeline_duration = _ffprobe_duration(timeline)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(timeline),
        "-i",
        str(audio_path_obj),
    ]
    if music_path_obj is not None and music_path_obj.exists():
        command.extend(["-stream_loop", "-1", "-i", str(music_path_obj)])

    filter_parts = [f"[0:v]{ass_filter}[v]"]
    voice_filters: list[str] = []
    if AUDIO_MASTERING_ENABLED and AUDIO_MASTER_FILTER.strip():
        voice_filters.append(AUDIO_MASTER_FILTER.strip())
    if NARRATION_VOLUME != 1.0:
        voice_filters.append(f"volume={NARRATION_VOLUME:.3f}")
    voice_chain = ",".join(voice_filters) if voice_filters else "anull"
    filter_parts.append(f"[1:a]{voice_chain}[voice]")

    audio_output_label = "[voice]"
    if music_path_obj is not None and music_path_obj.exists():
        fade_duration = min(max(MUSIC_FADE_SEC, 0.0), max(timeline_duration / 2, 0.0))
        fade_out_start = max(timeline_duration - fade_duration, 0.0)
        music_filters = [
            f"atrim=0:{timeline_duration:.3f}",
            "asetpts=PTS-STARTPTS",
            f"volume={BACKGROUND_MUSIC_VOLUME:.3f}",
        ]
        if fade_duration > 0:
            music_filters.append(f"afade=t=in:st=0:d={fade_duration:.3f}")
            music_filters.append(f"afade=t=out:st={fade_out_start:.3f}:d={fade_duration:.3f}")
        filter_parts.append(f"[2:a]{','.join(music_filters)}[music]")
        filter_parts.append("[voice][music]amix=inputs=2:duration=longest:dropout_transition=0,alimiter=limit=0.97[a]")
        audio_output_label = "[a]"

    command.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[v]",
            "-map",
            audio_output_label,
            *encoder_args,
            "-r",
            str(OUTPUT_FPS),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            AUDIO_CODEC,
            "-b:a",
            AUDIO_BITRATE,
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path_obj),
        ]
    )
    _run_command(command, "FFmpeg final reel assembly")
    return output_path_obj


def select_footage_clip(state_manager: StateManager, audio_duration: float) -> tuple[Path, float, Path]:
    """Pick footage, reserve a free segment, and prepare the base clip.

    Args:
        state_manager: State manager instance tracking footage usage.
        audio_duration: Narration duration in seconds.

    Returns:
        A tuple of `(footage_path, start_time, prepared_clip_path)`.
    """
    clip_duration = max(audio_duration, 0.1)
    footage_files = _media_files(FOOTAGE_DIR, VIDEO_EXTENSIONS)
    if not footage_files:
        raise FileNotFoundError(f"No supported footage files found in {FOOTAGE_DIR}")

    forced = FORCE_FOOTAGE_FILE.strip()
    if forced:
        forced_path = Path(forced)
        if not forced_path.is_absolute():
            forced_path = FOOTAGE_DIR / forced
        forced_path = forced_path.expanduser().resolve()
        if forced_path.exists():
            footage_files = [forced_path]
            LOGGER.info("Using forced footage file: %s", forced_path.name)
        else:
            LOGGER.warning("FORCE_FOOTAGE_FILE not found, auto-selecting instead: %s", forced_path)

    ordered_files: list[Path] = []
    remaining = footage_files[:]
    while remaining:
        next_file = state_manager.least_used_footage(remaining)
        if next_file is None:
            break
        ordered_files.append(next_file)
        remaining.remove(next_file)

    for footage_path in ordered_files:
        total_duration = _ffprobe_duration(footage_path)
        start = state_manager.get_next_clip_start(footage_path, clip_duration, total_duration, MIN_CLIP_GAP)
        if start is None:
            continue
        clip_path = _prepare_clip(footage_path, start, clip_duration)
        state_manager.record_used_segment(footage_path, start, start + clip_duration)
        return footage_path, start, clip_path

    LOGGER.warning("All footage segments are exhausted; resetting footage history and picking a random slot.")
    state_manager.reset_footage()
    chosen_file = random.choice(footage_files)
    total_duration = _ffprobe_duration(chosen_file)
    max_start = max(total_duration - clip_duration, 0.0)
    start = round(random.uniform(0.0, max_start), 3) if max_start > 0 else 0.0
    clip_path = _prepare_clip(chosen_file, start, clip_duration)
    state_manager.record_used_segment(chosen_file, start, start + clip_duration)
    return chosen_file, start, clip_path
