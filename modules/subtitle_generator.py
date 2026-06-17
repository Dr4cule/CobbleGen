from __future__ import annotations

import logging
import math
import re
from pathlib import Path

from config import (
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    SUBTITLE_COLOR,
    SUBTITLE_FONT,
    SUBTITLE_FONT_SIZE,
    SUBTITLE_HIGHLIGHT_COL,
    SUBTITLE_MARGIN_H,
    SUBTITLE_MAX_CHARS_PER_LINE,
    SUBTITLE_MAX_LINES,
    SUBTITLE_OUTLINE_COLOR,
    SUBTITLE_OUTLINE_WIDTH,
    SUBTITLE_STYLE_NAME,
    SUBTITLE_V_MARGIN,
    TEMP_DIR,
    WORDS_PER_CAPTION_GROUP,
)


LOGGER = logging.getLogger(__name__)

COLOR_MAP = {
    "white": "FFFFFF",
    "black": "000000",
    "yellow": "FFFF00",
    "red": "FF0000",
    "green": "00FF00",
    "blue": "0000FF",
}


def _to_ass_colour(value: str) -> str:
    raw = value.strip().lower()
    hex_value = COLOR_MAP.get(raw, raw.lstrip("#"))
    hex_value = hex_value if len(hex_value) == 6 else "FFFFFF"
    rr = hex_value[0:2]
    gg = hex_value[2:4]
    bb = hex_value[4:6]
    return f"&H00{bb}{gg}{rr}"


def _format_ass_time(seconds: float) -> str:
    centiseconds = max(0, int(round(seconds * 100)))
    hours, rem = divmod(centiseconds, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _escape_ass_text(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace("{", "(")
        .replace("}", ")")
        .replace("\n", r"\N")
    )


def _group_word_timestamps(word_timestamps: list[dict[str, float | str]]) -> list[list[dict[str, float | str]]]:
    groups: list[list[dict[str, float | str]]] = []
    current_group: list[dict[str, float | str]] = []
    current_chars = 0
    max_words = max(1, WORDS_PER_CAPTION_GROUP)
    max_chars = max(8, SUBTITLE_MAX_CHARS_PER_LINE * max(1, SUBTITLE_MAX_LINES))

    for item in word_timestamps:
        word = str(item["word"]).strip()
        word_len = len(re.sub(r"[^\w']+", "", word)) or len(word)
        candidate_chars = current_chars + word_len + (1 if current_group else 0)
        if current_group and candidate_chars > max_chars:
            groups.append(current_group)
            current_group = []
            current_chars = 0
            candidate_chars = word_len

        current_group.append(item)
        current_chars = candidate_chars
        if len(current_group) >= max_words:
            groups.append(current_group)
            current_group = []
            current_chars = 0
        elif len(current_group) >= max(2, max_words - 1) and re.search(r"[.!?,;:]$", word):
            groups.append(current_group)
            current_group = []
            current_chars = 0

    if current_group:
        groups.append(current_group)
    return groups


def _karaoke_caption_text(group: list[dict[str, float | str]]) -> str:
    parts: list[str] = []
    line_chars = 0
    line_count = 1
    max_line_chars = max(8, SUBTITLE_MAX_CHARS_PER_LINE)
    max_lines = max(1, SUBTITLE_MAX_LINES)

    for item in group:
        word = _escape_ass_text(str(item["word"]))
        plain_word = re.sub(r"[^\w']+", "", str(item["word"]).strip()) or str(item["word"]).strip()
        duration_cs = max(1, math.ceil((float(item["end"]) - float(item["start"])) * 100))
        prefix = ""
        extra_chars = len(plain_word) + (1 if line_chars else 0)
        if line_chars and line_chars + extra_chars > max_line_chars and line_count < max_lines:
            prefix = r"\N"
            line_chars = 0
            line_count += 1
        elif parts:
            prefix = " "

        parts.append(f"{prefix}{{\\kf{duration_cs}}}{word}")
        line_chars += len(plain_word) + (1 if line_chars else 0)

    return "".join(parts)


def generate_ass_subtitles(word_timestamps: list[dict[str, float | str]]) -> Path:
    """Generate an ASS subtitle file with karaoke word highlighting.

    Args:
        word_timestamps: Word-level timing dictionaries with `word`, `start`, and `end`.

    Returns:
        Path to the generated `.ass` subtitle file.
    """
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    subtitle_path = TEMP_DIR / "subtitles.ass"
    groups = _group_word_timestamps(word_timestamps)

    header = "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {OUTPUT_WIDTH}",
            f"PlayResY: {OUTPUT_HEIGHT}",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            (
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
                "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
                "MarginR, MarginV, Encoding"
            ),
            (
                f"Style: {SUBTITLE_STYLE_NAME},{SUBTITLE_FONT},{SUBTITLE_FONT_SIZE},"
                f"{_to_ass_colour(SUBTITLE_COLOR)},{_to_ass_colour(SUBTITLE_HIGHLIGHT_COL)},"
                f"{_to_ass_colour(SUBTITLE_OUTLINE_COLOR)},&H00000000,1,0,0,0,100,100,0,0,1,"
                f"{SUBTITLE_OUTLINE_WIDTH},0,8,{SUBTITLE_MARGIN_H},{SUBTITLE_MARGIN_H},{SUBTITLE_V_MARGIN},1"
            ),
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
    )

    lines = [header]
    for group in groups:
        if not group:
            continue
        start = float(group[0]["start"])
        end = float(group[-1]["end"])
        text = _karaoke_caption_text(group)
        lines.append(
            "Dialogue: 0,"
            f"{_format_ass_time(start)},{_format_ass_time(end)},{SUBTITLE_STYLE_NAME},,"
            f"{SUBTITLE_MARGIN_H},{SUBTITLE_MARGIN_H},{SUBTITLE_V_MARGIN},,{text}"
        )

    subtitle_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Generated subtitle file at %s", subtitle_path)
    return subtitle_path
