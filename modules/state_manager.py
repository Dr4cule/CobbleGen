from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from config import STATE_FILE


LOGGER = logging.getLogger(__name__)


class StateManager:
    """Persist and query pipeline state in a JSON file."""

    def __init__(self, state_file: Path = STATE_FILE) -> None:
        """Initialise the state manager.

        Args:
            state_file: Path to the JSON state file on disk.

        Returns:
            None.
        """
        self.state_file = state_file.resolve()
        self._state: dict[str, Any] = {
            "processed_stories": [],
            "used_segments": {},
        }
        self._load()

    def _load(self) -> None:
        if not self.state_file.exists():
            self._save()
            return
        try:
            self._state = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Resetting invalid state file %s: %s", self.state_file, exc)
            self._state = {"processed_stories": [], "used_segments": {}}
            self._save()

    def _save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self._state, indent=2), encoding="utf-8")

    @staticmethod
    def _normalise_path(path: str | Path) -> str:
        return str(Path(path).expanduser().resolve())

    def _segments_for(self, footage_path: str | Path) -> list[dict[str, float]]:
        normalised = self._normalise_path(footage_path)
        segments = self._state.setdefault("used_segments", {}).setdefault(normalised, [])
        segments.sort(key=lambda segment: float(segment["start"]))
        return segments

    def is_processed(self, story_path: str | Path) -> bool:
        """Return whether a story file has already been processed.

        Args:
            story_path: Story text file path.

        Returns:
            True when the story path is already recorded as processed.
        """
        return self._normalise_path(story_path) in set(self._state.get("processed_stories", []))

    def mark_processed(self, story_path: str | Path) -> None:
        """Record a story file as processed and save state.

        Args:
            story_path: Story text file path.

        Returns:
            None.
        """
        normalised = self._normalise_path(story_path)
        processed = self._state.setdefault("processed_stories", [])
        if normalised not in processed:
            processed.append(normalised)
            self._save()

    def reset_processed_stories(self) -> None:
        """Clear the processed story history and save state.

        Args:
            None.

        Returns:
            None.
        """
        self._state["processed_stories"] = []
        self._save()

    def get_next_clip_start(
        self,
        footage_path: str | Path,
        clip_duration: float,
        total_duration: float,
        min_gap: float,
    ) -> float | None:
        """Find the first non-overlapping clip start for a footage file.

        Args:
            footage_path: Video file to scan.
            clip_duration: Desired clip duration in seconds.
            total_duration: Total footage length in seconds.
            min_gap: Minimum gap to enforce between used segments.

        Returns:
            The first free start time in seconds, or None if no slot exists.
        """
        if clip_duration <= 0 or total_duration <= 0 or clip_duration > total_duration:
            return None

        step = max(clip_duration + min_gap, 0.1)
        segments = self._segments_for(footage_path)
        start = 0.0

        while start + clip_duration <= total_duration + 1e-6:
            end = start + clip_duration
            has_overlap = any(
                start < float(segment["end"]) + min_gap and end > float(segment["start"]) - min_gap
                for segment in segments
            )
            if not has_overlap:
                return round(start, 3)
            start += step
        return None

    def record_used_segment(self, footage_path: str | Path, start: float, end: float) -> None:
        """Persist a used footage segment and save state.

        Args:
            footage_path: Video file whose segment was consumed.
            start: Segment start time in seconds.
            end: Segment end time in seconds.

        Returns:
            None.
        """
        segment = {"start": round(float(start), 3), "end": round(float(end), 3)}
        segments = self._segments_for(footage_path)
        segments.append(segment)
        segments.sort(key=lambda item: item["start"])
        self._save()

    def least_used_footage(self, footage_files: list[Path]) -> Path | None:
        """Return the footage file with the fewest recorded used segments.

        Args:
            footage_files: Candidate footage files.

        Returns:
            The least-used footage path, or None when the list is empty.
        """
        if not footage_files:
            return None
        return min(
            footage_files,
            key=lambda file_path: len(self._segments_for(file_path)),
        )

    def reset_footage(self, footage_path: str | Path | None = None) -> None:
        """Clear usage history for one footage file or all footage files.

        Args:
            footage_path: Optional single footage path to clear.

        Returns:
            None.
        """
        if footage_path is None:
            self._state["used_segments"] = {}
        else:
            self._state.setdefault("used_segments", {}).pop(self._normalise_path(footage_path), None)
        self._save()
