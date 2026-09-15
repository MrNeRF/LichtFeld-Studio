# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared transfer measurements and presentation for gallery UI surfaces."""
from __future__ import annotations

import math
import time
from collections import deque
from functools import partial

from .asset_format import format_size
from .gallery_messages import tr as gallery_tr

tr = partial(gallery_tr, prefix="gallery.transfer.")


def transfer_phase(job):
    status = job.get("status", "queued")
    if status != "running":
        return "interrupted" if status == "paused" and job.get("interrupted") else status
    if job.get("serverProcessing"):
        return "processing"
    if job.get("preparation") and not job.get("packaged"):
        return "preparing"
    return "downloading" if job.get("kind") == "download" else "uploading"


class TransferEstimate:
    def __init__(self):
        self._key = None
        self._samples = deque()
        self._last_advance = 0.0

    def sample(self, jobs, stage, *, now=None):
        if (stage not in ("uploading", "downloading") or not jobs
                or any(j.get("total", 0) <= 0 or (j.get("kind") == "download") != (stage == "downloading") for j in jobs)):
            self._key = None
            self._samples.clear()
            return None, None
        now = time.monotonic() if now is None else now
        key = tuple((j["id"], j.get("kind"), j["total"], j["status"]) for j in jobs)
        done = sum(min(j["total"], max(0, j.get("completed", 0))) for j in jobs)
        samples = self._samples
        if key != self._key or (samples and done < samples[-1][1]):
            samples.clear()
            self._key = key
            self._last_advance = now
        if not samples or done > samples[-1][1]:
            samples.append((now, done))
            self._last_advance = now
            # Keep a complete interval when updates arrive once per multipart part.
            while len(samples) > 2 and samples[1][0] < now - 10.0:
                samples.popleft()
        elapsed = now - samples[0][0]
        transferred = done - samples[0][1]
        if elapsed < 1.0 or transferred <= 0 or now - self._last_advance > 30.0:
            return None, None
        speed = transferred / elapsed
        return speed, math.ceil((sum(j["total"] for j in jobs) - done) / speed)


def transfer_metrics(speed, remaining):
    if remaining is None:
        return gallery_tr("gallery.status.estimating", prefix="")
    minutes, seconds = divmod(remaining, 60)
    hours, minutes = divmod(minutes, 60)
    duration = f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"
    return gallery_tr("gallery.status.metrics", prefix="", speed=format_size(speed), remaining=duration)


def transfer_rows(snapshot, history_limit=30):
    pending, history = [], []
    for job in snapshot.get("jobs", []):
        if job.get("retired"):
            continue
        status = job["status"]
        done, total = max(0, job.get("completed", 0)), max(0, job.get("total", 0))
        phase = transfer_phase(job)
        progress = 100 if status == "completed" else min(100, 100 * done / max(1, total))
        indeterminate = status == "running" and (phase in ("preparing", "processing") or total <= 0)
        phase_label = tr("phase.error") if job.get("needsAttention") else tr("phase." + phase)
        if status == "running" and not indeterminate:
            phase_label = gallery_tr("gallery.status.progress", prefix="", stage=phase_label, percent=int(progress))
        row = {"id": job["id"], "title": job.get("metadata", {}).get("title", "") or tr("title"),
               "direction": "↓" if job.get("kind") == "download" else "↑", "status": status,
               "bytes": (format_size(total) if status == "completed" else format_size(done) if status == "canceled"
                         else "" if job.get("batchQueued") else tr("bytes", done=format_size(done), total=format_size(total))),
               "phase": phase_label,
               "reason": job.get("message", "") if status in ("error", "conflict", "paused") else "",
               "detail": job.get("transferDetail", ""),
               "progress": progress, "progress_width": f"{35 if indeterminate else progress:.1f}%",
               "indeterminate": indeterminate,
               "can_pause": status == "running",
               "can_resume": status in ("paused", "error", "queued") and job.get("retryable") is not False
                             and not job.get("batchQueued") and not snapshot.get("busy"),
               "can_cancel": status not in ("completed", "canceled")}
        (history if status in ("completed", "canceled") else pending).append(row)
    pending.sort(key=lambda row: {"running": 0, "queued": 1}.get(row["status"], 2))
    if snapshot.get("phase", "idle") != "idle":
        progress = snapshot.get("preparationProgress", 0)
        pending.insert(0, {"id": "native", "title": snapshot.get("operationTitle") or tr("title"),
            "direction": "↓" if snapshot["phase"] == "applying" else "↑", "status": "running",
            "bytes": "", "phase": tr("phase." + snapshot["phase"]), "reason": "", "detail": "",
            "progress": progress, "progress_width": f"{progress or 35:.1f}%", "indeterminate": not progress,
            "can_pause": False, "can_resume": False, "can_cancel": True})
    if snapshot.get("batchQueued") and not any(j.get("batchQueued") for j in snapshot.get("jobs", [])):
        pending.append({"id": "batch-queue", "title": tr("batch", count=snapshot["batchQueued"]),
            "direction": "↑", "status": "queued", "bytes": "", "phase": tr("phase.queued"), "reason": "", "detail": "",
            "progress": 0, "progress_width": "0%", "indeterminate": False,
            "can_pause": False, "can_resume": False, "can_cancel": False})
    return pending + list(reversed(history))[:history_limit]
