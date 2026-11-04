# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""One background pipeline run at a time, observable without a dialog (MCP tools, scripts)."""
from __future__ import annotations

import threading
from typing import Callable, Optional

from . import pipeline


class PipelineRunner:
    """Starts ``pipeline.run`` on a worker thread and keeps the last status for pollers."""

    def __init__(self, run: Callable = pipeline.run):
        self._run = run
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        self._status: dict = self._idle_status()

    @staticmethod
    def _idle_status() -> dict:
        return {"active": False, "done": False, "cancelled": False, "error": "", "stage": "", "fraction": 0.0,
                "message": "", "project": "", "work_dir": "", "frames": 0, "registered": 0, "mode": "",
                "package_dir": ""}

    def start(self, request: pipeline.Request, hooks: pipeline.Hooks) -> dict:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return {"started": False, "error": "A video-to-splat run is already running.",
                        "project": self._status["project"]}
            self._cancel = threading.Event()
            self._status = self._idle_status()
            self._status.update(active=True, project=str(request.project), work_dir=str(request.work_dir))
            cancel = self._cancel

            def on_progress(stage: str, fraction: float, message: str, _tail: str) -> None:
                with self._lock:
                    self._status.update(stage=stage, fraction=fraction, message=message)

            def worker() -> None:
                outcome: dict = {}
                try:
                    state = self._run(request, hooks, on_progress=on_progress, should_cancel=cancel.is_set)
                    if state is not None:
                        outcome.update(frames=state.frames, registered=state.registered, mode=state.mode,
                                       package_dir=getattr(state, "package_dir", "") or "")
                except (pipeline.PipelineCancelled, pipeline.PipelineStopped):
                    outcome["cancelled"] = True
                except pipeline.PipelineError as exc:
                    outcome["error"] = str(exc)
                except Exception as exc:  # noqa: BLE001 - report, never kill the host
                    outcome["error"] = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    self._status.update(outcome, active=False, done=True)

            self._thread = threading.Thread(target=worker, daemon=True, name="VideoToSplatRunner")
            self._thread.start()
            return {"started": True, "project": str(request.project), "work_dir": str(request.work_dir)}

    def status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def cancel(self) -> dict:
        with self._lock:
            active = self._thread is not None and self._thread.is_alive()
            if active:
                self._cancel.set()
            return {"cancelling": active}

    def wait(self, timeout: Optional[float] = None) -> bool:
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()
