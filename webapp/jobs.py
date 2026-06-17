"""Background job manager that runs the reel pipeline as a subprocess.

Each job spawns ``python main.py --story ... --reprocess`` with per-job
environment overrides, then parses the ``@@REEL@@`` JSON progress events the
pipeline prints on stdout. State is kept in memory and exposed to the API.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Deque

BASE_DIR = Path(__file__).resolve().parent.parent
EVENT_PREFIX = "@@REEL@@ "
STEP_COUNT = 7


class Job:
    """A single reel-generation job and its live state."""

    def __init__(self, job_id: str, story_name: str, settings: dict[str, Any]) -> None:
        self.id = job_id
        self.story_name = story_name
        self.settings = settings
        self.status = "queued"  # queued | running | done | error | cancelled
        self.step = 0
        self.step_label = "Queued"
        self.title = ""
        self.hook = ""
        self.output_video: str | None = None
        self.metadata: str | None = None
        self.error: str | None = None
        self.created_at = time.time()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.logs: Deque[str] = deque(maxlen=400)
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        # Monotonic version bumped on every state change so SSE can detect updates.
        self.version = 0

    def _touch(self) -> None:
        self.version += 1

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "story_name": self.story_name,
                "status": self.status,
                "step": self.step,
                "step_count": STEP_COUNT,
                "step_label": self.step_label,
                "title": self.title,
                "hook": self.hook,
                "output_video": self.output_video,
                "metadata": self.metadata,
                "error": self.error,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "settings": self.settings,
                "version": self.version,
                "logs": list(self.logs)[-40:],
            }


class JobManager:
    """Owns the job queue and a single worker thread."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._queue: Deque[str] = deque()
        self._lock = threading.Lock()
        self._worker_started = False

    def _ensure_worker(self) -> None:
        if self._worker_started:
            return
        self._worker_started = True
        thread = threading.Thread(target=self._worker_loop, daemon=True)
        thread.start()

    def submit(self, story_name: str, story_path: Path, settings: dict[str, Any]) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(job_id, story_name, settings)
        job.settings["_story_path"] = str(story_path)
        with self._lock:
            self._jobs[job_id] = job
            self._order.append(job_id)
            self._queue.append(job_id)
        self._ensure_worker()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            ids = list(self._order)
        return [self._jobs[j].to_dict() for j in reversed(ids) if j in self._jobs]

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        with job._lock:
            if job.status == "queued":
                job.status = "cancelled"
                job.step_label = "Cancelled"
                job.finished_at = time.time()
                job._touch()
                return True
            process = job._process
        if job.status == "running" and process is not None:
            process.terminate()
            return True
        return False

    def _worker_loop(self) -> None:
        while True:
            job_id: str | None = None
            with self._lock:
                if self._queue:
                    job_id = self._queue.popleft()
            if job_id is None:
                time.sleep(0.3)
                continue
            job = self._jobs.get(job_id)
            if job is None or job.status == "cancelled":
                continue
            self._run_job(job)

    def _run_job(self, job: Job) -> None:
        story_path = job.settings.get("_story_path", "")
        env = os.environ.copy()
        for key, value in job.settings.items():
            if key.startswith("_") or key in {"logs"}:
                continue
            if value is None or value == "":
                continue
            env[key] = str(value)

        with job._lock:
            job.status = "running"
            job.started_at = time.time()
            job.step_label = "Starting"
            job._touch()

        command = [
            sys.executable,
            "-u",
            str(BASE_DIR / "main.py"),
            "--story",
            story_path,
            "--reprocess",
        ]
        try:
            process = subprocess.Popen(
                command,
                cwd=str(BASE_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:  # noqa: BLE001
            with job._lock:
                job.status = "error"
                job.error = f"Failed to launch pipeline: {exc}"
                job.finished_at = time.time()
                job._touch()
            return

        job._process = process
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if line.startswith(EVENT_PREFIX):
                self._handle_event(job, line[len(EVENT_PREFIX):])
            else:
                with job._lock:
                    job.logs.append(line)
                    job._touch()

        return_code = process.wait()
        with job._lock:
            job._process = None
            if job.status == "cancelled":
                job.finished_at = time.time()
            elif job.status != "error" and return_code == 0 and job.output_video:
                job.status = "done"
                job.step = STEP_COUNT
                job.step_label = "Complete"
                job.finished_at = time.time()
            elif job.status != "error":
                job.status = "error"
                job.error = job.error or f"Pipeline exited with code {return_code}"
                job.finished_at = time.time()
            job._touch()

    def _handle_event(self, job: Job, raw: str) -> None:
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            return
        kind = event.get("event")
        with job._lock:
            if kind == "step":
                job.step = int(event.get("step", job.step))
                job.step_label = str(event.get("label", job.step_label))
            elif kind == "meta":
                job.title = str(event.get("title") or job.title)
                job.hook = str(event.get("hook") or job.hook)
            elif kind == "done":
                job.output_video = str(event.get("output_video") or "")
                job.metadata = str(event.get("metadata") or "")
                job.title = str(event.get("title") or job.title)
            elif kind == "error":
                job.status = "error"
                job.error = str(event.get("message") or "Pipeline error")
            job._touch()


MANAGER = JobManager()
