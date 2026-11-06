# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Explicitly installed trainer publisher. No Blender or observer UI dependency."""
import importlib
import json
from _lfs_path_io import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from .capture import capture
from . import geometry
from .transport import MAX_BYTES, TERMINAL, SessionPublisher, read_text_shared

_stream = None


def ensure_numpy(lf):
    """Prepare numerical dependencies on a worker, never in a capture callback."""
    try:
        import numpy
    except ImportError:
        lf.packages.init()
        lf.packages.install("numpy>=2,<3")
        site = lf.packages.site_packages_dir()
        if site not in sys.path:
            sys.path.insert(0, site)
        importlib.invalidate_caches()
        import numpy
    return numpy


class TrainingPublisher:
    def __init__(self, lf, *, label="LichtFeld training", dataset="", limit=100000, degree=1, interval=2.0):
        if type(limit) is not int or limit < 1 or type(degree) is not int or not 0 <= degree <= 3:
            raise ValueError("Invalid preview point limit or SH degree")
        if type(interval) not in (int, float) or not 0.1 <= interval <= 3600:
            raise ValueError("Invalid preview interval")
        self.lf, self.limit, self.degree, self.interval = lf, limit, degree, interval
        self.metadata = {"label": label, "dataset": dataset, "source": "lichtfeld",
                         "backend": sys.executable, "state": "preparing",
                         "preview_limit": limit, "preview_sh_degree": degree}
        self.publisher = None
        self.writer = None
        self.dataset = dataset
        self.pending = None
        self.ready = False
        self.source_geometry = None
        self.closed = False
        self.last = 0.0
        self.retry_at = 0.0
        self.geometry = None
        self.hooks = lf.ScopedHandler()
        self.hooks.on_training_start(self.start)
        self.hooks.on_post_step(self.step)
        self.hooks.on_training_end(self.end)

    def start(self, hook):
        # Initialization may fail while the GUI remains open. Register only once
        # the trainer actually starts, rather than leaving a live phantom session.
        if self.closed or self.publisher is not None:
            return
        self.publisher = SessionPublisher(self.metadata, status_provider=self.status)
        self.writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="training-preview")
        self.pending = self.writer.submit(self.prepare)

    def status(self):
        # These AppContext fields copy CommandCenter state without reading a model
        # or dereferencing the trainer pointer from a heartbeat worker.
        context = self.lf.ui.context()
        return {"state": "paused" if context.is_paused else "training"} if context.is_training else {}

    def prepare(self):
        ensure_numpy(self.lf)
        if self.dataset:
            try:
                value = json.loads(read_text_shared(Path(self.dataset) / "view_geometry.json",
                                                    max_chars=geometry.MAX_BYTES))
            except FileNotFoundError:
                return None
            return geometry.validate(value)
        return None

    def step(self, hook):
        if self.closed or self.publisher is None:
            return
        # Only a short memory lock; heartbeat and writer own all metadata IO.
        self.publisher.update(flush=False, iteration=hook["iteration"],
                              iterations=hook["iterations"], count=hook["num_gaussians"],
                              state="paused" if hook["is_paused"] else "training")
        if time.monotonic() < self.retry_at:
            return
        try:
            if self.pending is not None:
                if not self.pending.done():
                    return
                result = self.pending.result()
                if not self.ready:
                    self.source_geometry = result
                self.pending = None
                self.ready = True
            if not self.ready:
                self.pending = self.writer.submit(self.prepare)
                return
            if not self.publisher.has_clients or time.monotonic() - self.last < self.interval:
                return
            if self.geometry is None:
                self.geometry = geometry.from_native_scene(self.lf.get_scene(), self.source_geometry)
            started = time.perf_counter()
            arrays = capture(self.lf, limit=self.limit, degree=self.degree, max_bytes=MAX_BYTES)
            elapsed = time.perf_counter() - started
            self.last = time.monotonic()
            self.pending = self.writer.submit(self._publish, arrays, hook["iteration"], elapsed)
        except Exception as error:
            self.pending = None
            self.publisher.update(flush=False, error=str(error))
            self.retry_at = time.monotonic() + 30

    def _publish(self, arrays, iteration, elapsed):
        if not self.publisher.record.get("geometry"):
            geometry.publish(self.publisher, self.geometry)
        self.publisher.publish(arrays, iteration, update_progress=False)
        self.publisher.update(error="", preview_capture_seconds=elapsed)

    def end(self, hook):
        if self.closed:
            return
        self.closed = True
        self.hooks.clear()
        terminal = hook.get("terminal_state", "unknown")
        if terminal not in TERMINAL:
            terminal = "failed"
        if self.writer is not None:
            # TrainingEnd runs on the trainer worker. Finish its pending CPU
            # publication before recording the terminal state or exiting headless.
            self.writer.shutdown(wait=True)
            self.publisher.finish(terminal, iteration=hook.get("iteration", 0))


def install(dataset="", **options):
    """Install one publisher for the current trainer, replacing the previous run."""
    import lichtfeld as lf
    global _stream
    previous = _stream
    options["dataset"] = dataset
    options.setdefault("label", Path(dataset).name or "LichtFeld training")
    stream = TrainingPublisher(lf, **options)
    if previous is not None:
        previous.end({"terminal_state": "superseded"})
        if previous.publisher is not None:
            previous.publisher.pool.close()
    _stream = stream
    return _stream
