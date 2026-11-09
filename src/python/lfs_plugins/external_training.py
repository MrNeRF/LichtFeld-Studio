# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only session consumer. File IO and CPU preparation stay on its worker."""

import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

import lichtfeld as lf
from lichtfeld_sessions import ensure_numpy, geometry
from lichtfeld_sessions.transport import (
    TERMINAL, SessionClient, read_snapshot, session_liveness, validate_session,
)
from .ui import RuntimeState

_reader_executor = None


def _session_finished(status):
    return status.get("state") in TERMINAL or session_liveness(status) == "exited"


def presentation_transform(value, np):
    """Compose source presentation once, before the renderer's data/display flip."""
    value = value or {}
    if value.get("source_frame") == "blender-world":
        blender_to_view = np.array(((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))
        return geometry.matrix(geometry.FLIP) @ blender_to_view @ geometry.matrix(value["data_to_source"])
    return np.eye(4)


def shutdown():
    """Called by the existing Python UI cleanup, after native uploads stop."""
    global _reader_executor
    active = ExternalTraining._active() if ExternalTraining._active is not None else None
    if active is not None:
        active._shutdown = True
        active.stop.set()
        if active._unsubscribe is not None:
            active._unsubscribe()
            active._unsubscribe = None
        active._pending = None
    executor, _reader_executor = _reader_executor, None
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)


class ExternalTraining:
    """One lease, one CPU reader and at most one pending native upload."""

    _active = None

    def __init__(self, record):
        global _reader_executor
        self.record = dict(record)
        self.status = dict(record)
        self.error = ""
        self.connection_error = ""
        self.geometry_error = ""
        self.shown = None
        self.generation = None
        self.geometry_revision = None
        self._geometry = None
        self._geometry_retry_at = 0.0
        self._splat_retry_at = 0.0
        self._failed_generation = None
        self.preparing_dependencies = True
        self._pending = None
        self._install_pending = False
        self._unsubscribe = None
        self._shutdown = False
        self._future = None
        self.stop = threading.Event()
        scene = lf.get_scene()
        self.uuid = scene.create_external_preview(record.get("label") or record["id"], lf.ui.tr("external_training.preview"))
        previous = ExternalTraining._active() if ExternalTraining._active is not None else None
        if previous is not None:
            previous.close(remove=True)
        ExternalTraining._active = weakref.ref(self)
        self._unsubscribe = RuntimeState.scene_generation.subscribe(self._check_target)
        try:
            if _reader_executor is None:
                _reader_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="external-training-reader")
            self._future = _reader_executor.submit(self._read)
        except RuntimeError:
            self.close(remove=True)
            raise

    @property
    def scene(self):
        # Scene.generation is a UI revision, incremented on ordinary mutations.
        # Resolve the canonical scene afresh; the protected UUID is our identity.
        return lf.get_scene()

    def close(self, *, remove=False):
        """UI thread; training belongs to the other process and is never stopped."""
        self.stop.set()
        if self._future is not None:
            self._future.cancel()
        self._retire()
        if remove:
            self.scene.remove_external_preview(self.uuid)

    def _retire(self):
        # A late callback from an old connection cannot cancel its replacement.
        if self.stop.is_set() and self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        if self._is_active():
            self.scene.cancel_external_preview_upload()

    def _is_active(self):
        return ExternalTraining._active is not None and ExternalTraining._active() is self

    def _check_target(self, _value=None):
        if not self.stop.is_set() and not self.scene.get_node_by_uuid(self.uuid):
            self.close()

    def _submit(self, update, kind, descriptor):
        if self.stop.is_set():
            return
        self._pending = (update, kind, descriptor)

        def upload():
            if self.stop.is_set():
                self._pending = None
                return
            if not self.scene.is_valid() or not self.scene.get_node_by_uuid(self.uuid):
                self.close()
                return
            try:
                if not self.scene.upload_external_preview(update):
                    if kind == "splat":
                        self._splat_retry_at = time.monotonic() + 2
                    self._pending = None  # Next poll retries; no queue of uploads.
            except Exception as error:
                self._failed(kind, error, descriptor)
                self._pending = None

        lf.ui.schedule_on_ui_thread(upload)

    def _install(self, update, kind, descriptor):
        if self.stop.is_set():
            return
        try:
            if not self.scene.apply_external_preview(self.uuid, update, lf.ui.tr("external_training.cameras")):
                self.close()
                return
            if kind == "splat":
                self.generation = descriptor["id"]
                self.shown = dict(descriptor)
                self.error = ""
                self._failed_generation = None
            else:
                self.geometry_revision, self._geometry = descriptor
                self.geometry_error = ""
            lf.ui.request_redraw()
        except Exception as error:
            self._failed(kind, error, descriptor)
        finally:
            self._pending = None
            self._install_pending = False

    def _failed(self, kind, error, descriptor=None):
        if kind == "cameras":
            self.geometry_error = str(error)
        else:
            self.error = str(error)
            self._splat_retry_at = time.monotonic() + 2
            self._failed_generation = descriptor.get("id") if descriptor else None

    def _read(self):
        if self.stop.is_set():
            return
        client = None
        try:
            while not self.stop.is_set():
                try:
                    np = ensure_numpy(lf, self.stop.is_set)
                    self.error = ""
                    break
                except Exception as error:
                    self.error = str(error)
                    self.stop.wait(2)
            self.preparing_dependencies = False
            if self.stop.is_set():
                return
            client = SessionClient(self.record["directory"], "LichtFeld Studio")
            while not self.stop.wait(0.2):
                snapshot = None
                try:
                    try:
                        status = validate_session(client.status(), self.record["id"])
                        if status["pid"] != self.record["pid"]:
                            raise ValueError("Training process identity changed")
                    except Exception as error:
                        self.connection_error = str(error)
                        if _session_finished(self.status):
                            self.stop.set()
                        continue
                    self.status, self.connection_error = status, ""
                    if self._pending is not None:
                        update, kind, descriptor = self._pending
                        if update.ready and not self._install_pending:
                            self._install_pending = True
                            lf.ui.schedule_on_ui_thread(lambda: self._install(update, kind, descriptor))
                        continue
                    reference = status.get("geometry")
                    revision = reference.get("revision") if isinstance(reference, dict) else None
                    if (reference and (revision is None or revision != self.geometry_revision)
                            and time.monotonic() >= self._geometry_retry_at):
                        self._geometry_retry_at = time.monotonic() + 2
                        try:
                            value = geometry.load(client.root, reference)
                            transform = presentation_transform(value, np)
                            rows = [list((transform @ geometry.matrix(camera["camera_to_data"])).reshape(-1)) +
                                    [camera[key] for key in ("fx", "fy", "cx", "cy", "width", "height")]
                                    for camera in value.get("cameras", [])]
                            tensor = lf.Tensor.from_numpy(np.asarray(rows, dtype=np.float32).reshape(-1, 22))
                            model_transform = transform @ geometry.matrix(value.get("model_to_data", geometry.IDENTITY))
                            update = lf.scene.prepare_external_cameras(tensor, np.asarray(model_transform, dtype=np.float32))
                            self._submit(update, "cameras", (reference["revision"], value))
                            continue
                        except Exception as error:
                            self.geometry_error = str(error)
                    snapshot = status.get("snapshot")
                    if snapshot and snapshot.get("id") != self.generation:
                        if snapshot.get("id") == self._failed_generation and _session_finished(status):
                            self.stop.set()  # A failed final upload must still release its lease.
                            continue
                        if time.monotonic() < self._splat_retry_at:
                            continue
                        arrays = read_snapshot(snapshot)
                        def tensor(value):
                            return lf.Tensor.from_numpy(np.ascontiguousarray(value, dtype=np.float32))
                        transform = presentation_transform(self._geometry, np) @ geometry.matrix(
                            (self._geometry or {}).get("model_to_data", geometry.IDENTITY))
                        update = lf.scene.prepare_external_splat(
                            tensor(arrays["means"]), tensor(arrays["sh"][:, :1]), tensor(arrays["sh"][:, 1:]),
                            tensor(arrays["scale"]), tensor(arrays["rotation"]), tensor(arrays["opacity"]),
                            snapshot["sh_degree"], np.asarray(transform, dtype=np.float32))
                        self._submit(update, "splat", dict(snapshot))
                        continue
                    if _session_finished(status):
                        self.stop.set()
                except Exception as error:
                    self._failed("splat", error, snapshot)
                    if _session_finished(self.status):
                        self.stop.set()
        except Exception as error:
            if not self.stop.is_set():
                self.error = str(error)
            self.stop.set()
        finally:
            self.preparing_dependencies = False
            if client is not None:
                try:
                    client.close()
                except OSError:
                    pass  # A producer may already have removed its lease directory.
            if not self._shutdown:
                try:
                    lf.ui.schedule_on_ui_thread(self._retire)
                    pending = self._pending
                    while (pending is not None and self._pending is pending and not pending[0].ready
                           and self._is_active()):
                        time.sleep(0.1)
                    lf.ui.schedule_on_ui_thread(self._retire)
                except RuntimeError:
                    pass  # The native owner drains uploads during application shutdown.
