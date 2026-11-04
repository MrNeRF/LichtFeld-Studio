# SPDX-License-Identifier: GPL-3.0-or-later
"""Worker preparation and constant-size UI installation of external models."""
import threading
import time
import queue
import uuid
import weakref
from _lfs_path_io import Path
from concurrent.futures import ThreadPoolExecutor

_retirement = ThreadPoolExecutor(max_workers=1, thread_name_prefix="viewer-retire")
from .producer import numerical_runtime, prepare_viewer_runtime
from .reader import Observer
from .geometry import IDENTITY, FLIP, viewer_data_transform, viewer_model_transform, visualizer_model_transform
from .presentation import _tr


class Viewer:
    def __init__(self, directory):
        import lichtfeld as lf
        self.name = "Attached trainer " + Path(directory).name
        self.session_info = {"id": Path(directory).name, "directory": str(Path(directory))}
        self.scene = lf.get_scene()
        self.stop = threading.Event()
        self.generation = None
        self.node_id = None
        self.node_uuid = None
        self.shown_iteration = 0
        self.display_arrays = None
        self.prepared_arrays = None
        self.retired_arrays = queue.SimpleQueue()
        self.status_pending = False
        self.last_status_at = 0
        self.copy_busy = False
        self.copy_error = ""
        self.ui_max_ms = 0.0
        self.display_geometry = None
        self.display_bounds = None
        self.auto_focus = len(self.scene.nodes) == 0
        self.has_focused = False
        self.prepared_cameras = None
        self.camera_lock = threading.Lock()
        self.camera_ref = None
        self.camera_count = 0
        self.cameras_visible = True
        self.camera_error = ""
        self.camera_epoch = 0
        self.reader = Observer(directory, prepare=self._prepare, bootstrap=prepare_viewer_runtime,
                               reject_same_process=True, start=False)
        self.reader.thread.start()
        def anchor():
            try:
                if self.stop.is_set():
                    return
                if not self.scene.is_valid():
                    raise RuntimeError("Project changed before connecting the viewer")
                current = lf.get_scene()
                self.node_id = current.create_external_viewer(self.name)
                self.node_uuid = str(current.get_node(self.name).uuid)
            except Exception as error:
                self.reader.error = str(error)
                self.close()
        lf.ui.schedule_on_ui_thread(anchor)
        lf.ui.request_redraw()
        self.client = type("Location", (), {"root": Path(directory)})()
        # Disconnected previews intentionally keep their arrays. Still release
        # them if their scene disappears, even with this panel closed.
        from ..ui import RuntimeState
        reference = weakref.ref(self)
        def scene_changed(_generation):
            owner = reference()
            if owner is not None and owner.stop.is_set() and owner.node_uuid and not owner.status_pending:
                owner.status_pending = True
                def check_retained():
                    try:
                        node = lf.get_scene().get_node(owner.name)
                        if node is None or str(node.uuid) != owner.node_uuid:
                            owner._update_status()
                    finally:
                        owner.status_pending = False
                lf.ui.schedule_on_ui_thread(check_retained)
                lf.ui.request_redraw()
        self._scene_watch = weakref.finalize(self, RuntimeState.scene_generation.subscribe(scene_changed))
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()

    @property
    def status(self):
        if self.reader.error:
            return self.reader.error
        if self.stop.is_set():
            return "Disconnected · retained read-only preview"
        status = self.reader.status
        return f"{status.get('state', 'connecting')} · step {status.get('iteration', 0):,}"

    def snapshot_in_scene(self):
        """UI-only, constant-size identity check; never reads files or model tensors."""
        if self.display_arrays is None or self.node_uuid is None:
            return False
        import lichtfeld as lf
        scene = lf.get_scene()
        node = scene.get_node(self.name) if scene else None
        return node is not None and str(node.uuid) == self.node_uuid

    def _prepare(self, arrays):
        geometry = self.reader.geometry
        prepared = self._native_prepare(arrays, geometry)
        np = numerical_runtime()
        low, high = arrays["means"].min(axis=0), arrays["means"].max(axis=0)
        corners = np.array([[x,y,z,1] for x in (low[0],high[0]) for y in (low[1],high[1]) for z in (low[2],high[2])])
        transform = visualizer_model_transform(geometry)
        corners = corners @ transform.T
        center = (corners[:,:3].min(axis=0)+corners[:,:3].max(axis=0))*.5
        radius = max(.01,float(np.linalg.norm(corners[:,:3]-center,axis=1).max()))
        self.prepared_bounds = (center.tolist(),radius)
        reference = self.reader.geometry_ref
        if geometry and reference != self.camera_ref:
            try:
                import lichtfeld as lf
                cameras = geometry.get("cameras", [])
                presentation = viewer_data_transform(geometry)
                rows = [list((presentation @ np.asarray(c["camera_to_data"])).reshape(-1)) +
                        [c[k] for k in ("fx","fy","cx","cy","width","height")] for c in cameras]
                payload = lf.scene.prepare_viewer_cameras(np.asarray(rows, dtype=np.float32).reshape(-1,22))
                with self.camera_lock:
                    stopped = self.stop.is_set()
                    if not stopped:
                        self.prepared_cameras = (reference, payload, len(cameras))
                if stopped:
                    payload.release()
            except Exception as error:
                self.camera_error = str(error)
        with self.camera_lock:
            if not self.stop.is_set():
                self.prepared_geometry = geometry
                self.prepared_arrays = arrays
        return prepared

    def _native_prepare(self, arrays, geometry=None):
        import lichtfeld as lf
        np = numerical_runtime()
        def contiguous(value):
            return np.ascontiguousarray(value, dtype=np.float32)
        return lf.scene.prepare_viewer_splat(contiguous(arrays["means"]), contiguous(arrays["sh"][:, :1]),
            contiguous(arrays["sh"][:, 1:]), contiguous(arrays["scale"]), contiguous(arrays["rotation"]),
            contiguous(arrays["opacity"]), int(round(arrays["sh"].shape[1] ** 0.5)) - 1,
            contiguous(viewer_model_transform(geometry)))

    def _watch(self):
        import lichtfeld as lf
        try:
            while not self.stop.wait(0.02):
                while not self.retired_arrays.empty():
                    self.retired_arrays.get_nowait()
                if self.node_uuid and not self.status_pending and time.monotonic() - self.last_status_at >= .5:
                    self.status_pending = True
                    self.last_status_at = time.monotonic()
                    lf.ui.schedule_on_ui_thread(self._update_status)
                    lf.ui.request_redraw()
                frame = self.reader.take()
                if frame is None:
                    continue
                def apply(frame=frame):
                    error = None
                    started = time.perf_counter()
                    try:
                        if self.stop.is_set():
                            return
                        current = lf.get_scene()
                        node = current.get_node(self.name)
                        if node is None or str(node.uuid) != self.node_uuid:
                            self.close()
                            raise RuntimeError("The viewer node was removed or the project changed. Reconnect this viewer.")
                        self.node_id = current.install_prepared_viewer(self.name, frame["data"])
                        self.node_uuid = str(current.get_node(self.name).uuid)
                        self.generation = frame["record"]["id"]
                        self.shown_iteration = frame["record"].get("iteration", 0)
                        if self.display_arrays is not None:
                            self.retired_arrays.put(self.display_arrays)
                        self.display_arrays = self.prepared_arrays
                        self.display_geometry = frame.get("geometry")
                        self.display_bounds = self.prepared_bounds
                        if self.auto_focus and not self.has_focused:
                            self.frame_preview()
                            self.has_focused = True
                        with self.camera_lock:
                            cameras, self.prepared_cameras = self.prepared_cameras, None
                        if cameras is not None:
                            self._install_cameras(cameras)
                        self._update_status()
                        lf.ui.request_redraw()
                    except Exception as failure:
                        error = str(failure)
                    finally:
                        self.ui_max_ms = max(self.ui_max_ms, (time.perf_counter() - started) * 1000)
                        self.reader.acknowledge(frame["record"]["id"], error)
                lf.ui.schedule_on_ui_thread(apply)
                lf.ui.request_redraw()
        except Exception as error:
            self.reader.error = str(error)
            self.close()
        finally:
            while not self.retired_arrays.empty():
                self.retired_arrays.get_nowait()
            self.prepared_arrays = None

    def _retire(self, payload):
        _retirement.submit(payload.release)

    def _install_cameras(self, prepared):
        import lichtfeld as lf
        reference, payload, count = prepared
        self.camera_epoch += 1
        epoch = self.camera_epoch
        self.camera_ref = reference
        self.camera_count = 0
        self.camera_error = ""
        index = 0
        removing = True
        def chunk():
            nonlocal index, removing
            try:
                current = lf.get_scene()
                node = current.get_node(self.name)
                if (self.stop.is_set() or epoch != self.camera_epoch or node is None or
                        str(node.uuid) != self.node_uuid):
                    self._retire(payload)
                    return
                deadline = time.perf_counter() + .002
                while time.perf_counter() < deadline:
                    current = lf.get_scene()
                    if removing:
                        removed, retired = current.remove_external_camera(self.node_uuid, self.name)
                        self._retire(retired)
                        if removed: continue
                        removing = False
                    if index >= count:
                        self._retire(payload)
                        lf.ui.request_redraw()
                        return
                    current.install_external_camera(self.name,self.node_uuid,payload,index)
                    index += 1
                    self.camera_count = index
                    group = current.get_node(self.name + " cameras")
                    if group is not None and group.visible != self.cameras_visible: group.visible = self.cameras_visible
                lf.ui.schedule_on_ui_thread(chunk)
                lf.ui.request_redraw()
            except Exception as error:
                self.camera_error = str(error)
                self.camera_ref = None  # Retry camera preparation on the next snapshot.
                self._retire(payload)
        lf.ui.schedule_on_ui_thread(chunk)
        settings = lf.get_render_settings()
        if settings and self.cameras_visible:
            settings.show_camera_frustums = True

    def toggle_cameras(self):
        import lichtfeld as lf
        self.cameras_visible = not self.cameras_visible
        group = lf.get_scene().get_node(self.name + " cameras")
        if group is not None: group.visible = self.cameras_visible
        settings = lf.get_render_settings()
        if settings and self.cameras_visible:
            settings.show_camera_frustums = True
        lf.ui.request_redraw()

    def frame_preview(self):
        if self.display_bounds is None: return
        import lichtfeld as lf
        center,radius = self.display_bounds
        up = (0,1,0)
        lf.set_camera(tuple(center[i]+radius*2.5*d for i,d in enumerate((1,-1,.7))),tuple(center),up)
        lf.ui.request_redraw()

    def _update_status(self):
        import lichtfeld as lf
        try:
            current = lf.get_scene()
            node = current.get_node(self.name)
            if node is None or str(node.uuid) != self.node_uuid:
                if not self.stop.is_set():
                    self.reader.error = _tr("external_training.project_changed")
                    self.close()
                self._retire_display()
                return
            status = self.reader.status
            current.update_external_viewer(self.name, "preview error" if self.reader.error else status.get("state", "connecting"),
                status.get("pid", 0), status.get("iteration", 0), status.get("iterations", 0),
                self.shown_iteration, not self.stop.is_set())
        finally:
            self.status_pending = False

    def create_local_copy(self):
        """Capture exactly the displayed snapshot; GPU work and retirement stay on a worker."""
        if self.copy_busy or self.display_arrays is None:
            return
        arrays = self.display_arrays
        geometry = self.display_geometry
        self.copy_busy = True
        self.copy_error = ""
        def work():
            import lichtfeld as lf
            prepared = None
            done = threading.Event()
            gate = threading.Lock()
            state = {"started": False, "cancelled": False}
            try:
                prepared = self._native_prepare(arrays, geometry)
                def install():
                    with gate:
                        if state["cancelled"]:
                            done.set()
                            return
                        state["started"] = True
                    try:
                        current = lf.get_scene()
                        node = current.get_node(self.name)
                        if node is None or str(node.uuid) != self.node_uuid:
                            raise RuntimeError(_tr("external_training.copy_project_changed"))
                        current.install_prepared_viewer("Local snapshot " + uuid.uuid4().hex[:8],
                                                        prepared, editable_copy=True)
                        lf.ui.request_redraw()
                    except Exception as error:
                        self.copy_error = str(error)
                    finally:
                        done.set()
                lf.ui.schedule_on_ui_thread(install)
                lf.ui.request_redraw()
                if not done.wait(30):  # Worker only. Never waits on the UI thread.
                    with gate:
                        state["cancelled"] = not state["started"]
                        started = state["started"]
                    if started:
                        done.wait()  # Worker keeps the native payload alive until installation ends.
                    else:
                        self.copy_error = _tr("external_training.copy_ui_timeout")
            except Exception as error:
                self.copy_error = str(error)
            finally:
                if prepared is not None:
                    prepared.release()
                self.copy_busy = False
        threading.Thread(target=work, daemon=True, name="viewer-local-copy").start()

    def discard(self):
        """Remove only our transient node; retire GPU allocations off the UI thread."""
        self.close()
        import lichtfeld as lf
        current = lf.get_scene()
        node = current.get_node(self.name)
        if node is not None and str(node.uuid) == self.node_uuid:
            retired = current.remove_external_viewer(self.name, self.node_uuid)
            self._retire(retired)
        self._retire_display()
        owner_uuid = self.node_uuid
        if not owner_uuid:
            return
        def remove_cameras():
            deadline = time.perf_counter() + .002
            while time.perf_counter() < deadline:
                removed, payload = lf.get_scene().remove_external_camera(owner_uuid, self.name)
                self._retire(payload)
                if not removed: return
            lf.ui.schedule_on_ui_thread(remove_cameras)
            lf.ui.request_redraw()
        lf.ui.schedule_on_ui_thread(remove_cameras)
        lf.ui.request_redraw()

    def _retire_display(self):
        """UI-only handoff; the worker drops large CPU allocations even after project replacement."""
        watch = getattr(self, '_scene_watch', None)
        if watch is not None:
            watch()
        with self.camera_lock:
            payload = [self.display_arrays, self.display_geometry, self.display_bounds,
                       self.prepared_arrays, getattr(self, 'prepared_geometry', None)]
            self.display_arrays = self.display_geometry = self.display_bounds = None
            self.prepared_arrays = self.prepared_geometry = None
        if any(value is not None for value in payload):
            # Only this list owns the transferred references. Clearing it on the
            # worker cannot race an array's last decref on the UI callback frame.
            _retirement.submit(payload.clear)

    def close(self, update_status=True):
        was_stopped = self.stop.is_set()
        self.stop.set()
        self.reader.close()
        with self.camera_lock:
            pending, self.prepared_cameras = self.prepared_cameras, None
        if pending is not None:
            self._retire(pending[1])
        import lichtfeld as lf
        if self.node_uuid and not was_stopped and update_status:
            lf.ui.schedule_on_ui_thread(self._update_status)
            lf.ui.request_redraw()
