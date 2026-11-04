# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded background readers. Never import or access bpy, widgets or scene APIs."""
from _lfs_path_io import Path
import queue
import threading
import time
from .transport import SessionClient, read_snapshot, validate_session, session_liveness
from .geometry import load as load_geometry

TERMINAL = {"completed", "stopped", "failed", "superseded"}
PREPARING_PREVIEW = "preparing preview"  # Protocol state, translated only by the presentation layer.


def load_record(root, record):
    return read_snapshot(record["shared"]), record["shared"]["transport"]


class Observer:
    """One reader per scene binding; at most one unconsumed frame, sticky errors."""
    def __init__(self, directory, prepare=None, bootstrap=None, reject_same_process=False, start=True):
        self.root = Path(directory)
        self.prepare = prepare or (lambda arrays: arrays)
        self.bootstrap = bootstrap
        self.reject_same_process = reject_same_process
        self.stop = threading.Event()
        self.frames = queue.Queue(maxsize=1)
        self.ack = queue.SimpleQueue()
        self.status = {"state": "connecting"}
        self.error = ""
        self.status_error = False
        self.pending = False
        self.last = None
        self.last_geometry_ref = None
        self.retry_at = 0
        self.failures = 0
        self.inflight = None
        self.geometry = None
        self.geometry_ref = None
        self.geometry_error = ""
        self.geometry_retry_at = 0
        self.thread = threading.Thread(target=self._run, daemon=True, name="training-reader")
        if start:
            self.thread.start()

    def take(self):
        try:
            return self.frames.get_nowait()
        except queue.Empty:
            return None

    def acknowledge(self, generation, error=None, *, geometry_ref=...):
        self.ack.put((generation, geometry_ref, error))

    def close(self):
        self.stop.set()  # Never join or wait on a UI thread.

    def _run(self):
        client = None
        try:
            while not self.stop.is_set():
                record = None
                reading_status = True
                try:
                    if self.bootstrap:
                        if time.monotonic() < self.retry_at:
                            self.stop.wait(.1)
                            continue
                        self.status = {"state": PREPARING_PREVIEW}
                        self.bootstrap()
                        self.bootstrap = None
                    while not self.ack.empty():
                        generation, geometry_ref, error = self.ack.get_nowait()
                        frame = self.inflight
                        if (frame is None or generation != frame["record"]["id"] or
                                geometry_ref is not ... and geometry_ref != frame["geometry_ref"]):
                            continue  # A stale acknowledgement cannot retire a newer frame.
                        if hasattr(frame["data"], "release"):
                            frame["data"].release()
                        self.inflight = None
                        self.pending = False
                        if error:
                            self.error = str(error)
                            self.failures += 1
                            self.retry_at = time.monotonic() + min(30, 2 ** min(self.failures, 5))
                        else:
                            self.last = generation
                            self.last_geometry_ref = frame["geometry_ref"]
                            self.error = ""
                            self.failures = 0
                    if client is None:
                        client = SessionClient(self.root, "External renderer")
                    status = validate_session(client.status(), self.root.name)
                    if self.reject_same_process:
                        import os
                        if status.get("pid") == os.getpid():
                            raise ValueError("Use the local training viewport for a trainer in this process")
                    liveness = session_liveness(status)
                    state = status.get("state", "connecting")
                    if liveness != "responsive" and state not in {"completed", "stopped", "failed", "superseded"}:
                        state = "disconnected"
                    job = {"id": "session:" + status["id"], "sources": []}
                    root = self.root
                    snapshot = status.get("snapshot")
                    record = {**snapshot, "shared": snapshot} if snapshot else None
                    self.status = {**status, "state": state, "liveness": liveness,
                                   "job": job, "root": str(root)}
                    reference = status.get("geometry")
                    if reference != self.geometry_ref and time.monotonic() >= self.geometry_retry_at:
                        try:
                            self.geometry = load_geometry(self.root, reference)
                            self.geometry_ref = reference
                            self.geometry_error = ""
                        except Exception as error:
                            # Camera metadata is optional. Keep the last valid
                            # transform and model stream usable during retries.
                            self.geometry_error = str(error)
                            self.geometry_retry_at = time.monotonic() + 2
                    self.status["geometry_error"] = self.geometry_error or status.get("geometry_error", "")
                    if self.status_error:
                        self.error = ""
                        self.status_error = False
                        self.failures = 0
                        self.retry_at = 0
                    reading_status = False
                    changed = record and (record["id"] != self.last or self.geometry_ref != self.last_geometry_ref)
                    if changed and not self.pending and time.monotonic() >= self.retry_at:
                        arrays, transport = load_record(root, record)
                        frame = {"record": record, "job": job, "transport": transport,
                                 "count": len(arrays["means"]), "geometry": self.geometry,
                                 "geometry_ref": self.geometry_ref,
                                 "model_count": status.get("model_count") if status.get("iteration") == record.get("iteration") else None,
                                 "data": self.prepare(arrays)}
                        if self.stop.is_set():
                            break
                        self.frames.put_nowait(frame)
                        self.inflight = frame
                        self.pending = True
                    if state in TERMINAL and not self.pending and not changed:
                        break  # The retained preview needs no more filesystem polling.
                except FileNotFoundError:
                    if self.status.get("state") in TERMINAL and not self.pending:
                        break  # The finished producer already retired its last mapping.
                    # A live producer's generation can expire before it is opened.
                except Exception as error:
                    self.status_error = reading_status
                    self.error = str(error)
                    self.failures += 1
                    self.retry_at = time.monotonic() + min(30, 2 ** min(self.failures, 5))
                self.stop.wait(0.1)
        finally:
            if self.inflight and hasattr(self.inflight["data"], "release"):
                self.inflight["data"].release()
            self.inflight = None
            while not self.frames.empty():
                self.frames.get_nowait()
            self.geometry = None
            if client:
                try:
                    client.close()
                except OSError:
                    pass
