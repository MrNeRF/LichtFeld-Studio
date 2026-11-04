# SPDX-License-Identifier: GPL-3.0-or-later
"""Opt-in snapshot demand on every trainer, independent of Blender."""
import os
from _lfs_path_io import Path
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from .transport import SessionPublisher, spatial_indices

_stream = None
_numerical_lock = threading.Lock()


def process_arguments():
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32")
        shell = ctypes.WinDLL("shell32")
        kernel.GetCommandLineW.restype = ctypes.c_wchar_p
        shell.CommandLineToArgvW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
        shell.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
        kernel.LocalFree.argtypes = [ctypes.c_void_p]
        count = ctypes.c_int()
        values = shell.CommandLineToArgvW(kernel.GetCommandLineW(), ctypes.byref(count))
        try:
            return [values[index] for index in range(count.value)]
        finally:
            kernel.LocalFree(values)
    try:
        return Path("/proc/self/cmdline").read_bytes().decode().strip("\0").split("\0")
    except OSError:
        return sys.argv


def numerical_runtime(wait=True):
    if not _numerical_lock.acquire(blocking=wait):
        raise NumericalRuntimeBusy("The preview numerical runtime is being prepared")
    try:
        return _load_numerical_runtime()
    finally:
        _numerical_lock.release()


class NumericalRuntimeUnavailable(RuntimeError):
    pass


class NumericalRuntimeBusy(NumericalRuntimeUnavailable):
    pass


def _load_numerical_runtime():
    try:
        import numpy as np
        return np
    except ImportError:
        import lichtfeld as lf
        import importlib
        site = lf.packages.site_packages_dir()
        if site and site not in sys.path:
            sys.path.insert(0, site)
        importlib.invalidate_caches()
        try:
            import numpy as np
            return np
        except ImportError as error:
            raise NumericalRuntimeUnavailable("The preview numerical runtime is not ready") from error


def prepare_viewer_runtime():
    """First-use setup on the reader worker; training hooks never install packages."""
    if threading.current_thread() is threading.main_thread():
        raise RuntimeError("Preview runtime preparation must run on a worker")
    with _numerical_lock:
        try:
            return _load_numerical_runtime()
        except NumericalRuntimeUnavailable:
            import lichtfeld as lf
            try:
                lf.packages.init()
                lf.packages.install("numpy>=2,<3")
                return _load_numerical_runtime()
            except Exception as error:
                raise RuntimeError("Preview setup could not finish: " + str(error) +
                                   ". The connection is kept and setup will retry automatically.") from error


def capture(model, limit=None, degree=None, runtime=None):
    import lichtfeld as lf
    np = runtime if runtime is not None else numerical_runtime(wait=False)
    # Do not allocate one host index per Gaussian just to publish a small preview.
    if not model.has_deleted_mask() and limit and model.num_points > limit:
        candidates = np.linspace(0, model.num_points - 1, min(model.num_points, limit * 4), dtype=np.int64)
        indices = None
    else:
        indices = np.arange(model.num_points, dtype=np.int64)
        if model.has_deleted_mask():
            indices = indices[~model.deleted.numpy().reshape(-1).astype(bool)]
        candidates = (indices[np.linspace(0, len(indices) - 1, min(len(indices), limit * 4), dtype=np.int64)]
                      if limit and len(indices) > limit else None)
    if candidates is not None:
        positions = model.means_raw.index_select(0, lf.Tensor.from_numpy(candidates).cuda()).cpu()
        positions.sync()
        indices = candidates[spatial_indices(positions.numpy(), limit)]
    cpu_selection = lf.Tensor.from_numpy(indices)
    selection = cpu_selection.cuda()
    def owned(tensor):
        tensor = tensor.index_select(0, selection).cpu()
        tensor.sync()
        return tensor.numpy()
    active = model.active_sh_degree
    if degree is not None:
        active = min(active, degree)
    dc = owned(model.sh0_raw).reshape(len(indices), 1, 3)
    sh = np.concatenate((dc, model.shN_rows(cpu_selection, active).numpy()), axis=1) if active else dc
    return {"means": owned(model.means_raw), "scale": owned(model.scaling_raw),
            "rotation": owned(model.rotation_raw), "opacity": owned(model.opacity_raw).reshape(-1, 1), "sh": sh}


def capture_view_geometry(runtime=None):
    """Called at a training safe boundary, never on the UI thread."""
    import lichtfeld as lf
    from .geometry import FLIP, IDENTITY, validate
    np = runtime if runtime is not None else numerical_runtime(wait=False)
    scene = lf.get_scene()
    model = scene.get_node(scene.training_model_node_name)
    value = {"schema": 1, "source_frame": "lichtfeld-visualizer", "data_to_source": FLIP,
             "model_to_data": np.asarray(model.world_transform).reshape(4,4).tolist() if model else IDENTITY,
             "cameras": []}
    for node in scene.get_active_cameras():
        if not node.has_camera:
            continue
        R = node.camera_R.cpu().numpy().reshape(3,3)
        T = node.camera_T.cpu().numpy().reshape(3)
        c2w = np.eye(4)
        c2w[:3,:3] = R.T
        c2w[:3,3] = -R.T @ T
        c2w = np.asarray(node.world_transform).reshape(4,4) @ c2w
        value["cameras"].append({"name": node.name, "camera_to_data": c2w.tolist(),
            "fx": node.camera_focal_x, "fy": node.camera_focal_y,
            "cx": node.camera_center_x, "cy": node.camera_center_y,
            "width": node.camera_width, "height": node.camera_height})
    return validate(value)


class AutoStream:
    def __init__(self):
        import lichtfeld as lf
        args = process_arguments()
        def argument(*names):
            return next((args[index + 1] for index, value in enumerate(args[:-1]) if value in names), "")
        self.metadata = {"source": "lichtfeld", "backend": sys.executable,
                         "dataset": argument("-d", "--data-path"), "output": argument("-o", "--output-path"),
                         "project": argument("--resume"),
                         "checkpoint_path": argument("--save-project-path")}
        try:
            params = lf.dataset_params()
            self.metadata.update(dataset=params.data_path, output=params.output_path,
                                 project=lf.project_path() or "",
                                 iterations=lf.optimization_params().iterations)
        except RuntimeError:
            pass  # Headless mode has no GUI TrainerManager.
        self.metadata["label"] = Path(self.metadata["project"] or self.metadata["dataset"]).name or "LichtFeld trainer"
        self.publisher = None
        self.last = 0.0
        self.last_status = 0.0
        self.interval = 2.0
        self.writer = None
        self.pending = None
        self.closed = False
        self.geometry_published = False
        self.hooks = lf.ScopedHandler()
        self.hooks.on_post_step(self.step)
        self.hooks.on_training_end(self.end)

    def status(self):
        """Thread-safe native control snapshot; remains live while steps are paused."""
        if self.closed:
            return {}
        import lichtfeld as lf
        context = lf.Context()
        if not context.is_paused and not context.is_training:
            return {}  # TrainingEnd owns the precise terminal outcome.
        return {"state": "paused" if context.is_paused else "active",
                "iteration": context.iteration, "iterations": context.max_iterations,
                "count": context.num_gaussians, "model_count": context.num_gaussians}

    def step(self, hook):
        if getattr(self, "closed", False):
            return
        if time.monotonic() < getattr(self, "retry_at", 0):
            return
        try:
            if self.publisher is None:
                self.publisher = SessionPublisher(self.metadata, status_provider=self.status)
            if time.monotonic() - self.last_status > 0.5:
                self.publisher.update(iteration=hook["iteration"], count=hook["num_gaussians"],
                                      iterations=hook.get("iterations", 0), model_count=hook["num_gaussians"])
                self.last_status = time.monotonic()
            pending = getattr(self, "pending", None)
            if pending is not None:
                if not pending.done():
                    return
                self.pending = None
                pending.result()  # Already done; reports optional publication failures.
            if time.monotonic() - self.last < self.interval or not self.publisher.clients():
                return
            import lichtfeld as lf
            try:
                runtime = numerical_runtime(wait=False)
            except NumericalRuntimeBusy:
                return  # Optional capture never waits for a viewer's package setup.
            if not getattr(self, "geometry_published", False) and time.monotonic() >= getattr(self, "geometry_retry_at", 0):
                try:
                    from .geometry import publish
                    publish(self.publisher, capture_view_geometry(runtime=runtime))
                    self.geometry_published = True
                    self.publisher.update(geometry_error="")
                except Exception as error:
                    self.publisher.update(geometry_error=str(error))
                    self.geometry_retry_at = time.monotonic() + 5
            started = time.perf_counter()
            arrays = capture(lf.get_scene().training_model(), 100000, 1, runtime=runtime)
            capture_seconds = time.perf_counter() - started
            self.last = time.monotonic()
            self.interval = max(2, capture_seconds / 0.02)
            if self.writer is None:
                self.writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="training-preview")
            self.pending = self.writer.submit(self._publish, arrays, hook["iteration"], capture_seconds)
        except Exception as error:
            try:
                if self.publisher:
                    self.publisher.update(error=str(error))
            except Exception:
                pass  # An optional observer must never stop the optimizer.
            self.last = time.monotonic()
            self.interval = 30.0
            self.retry_at = self.last + 30

    def _publish(self, arrays, iteration, capture_seconds):
        started = time.perf_counter()
        self.publisher.publish(arrays, iteration, update_progress=False)
        self.publisher.update(error="", preview_limit=100000, preview_sh_degree=1,
                              preview_capture_seconds=capture_seconds,
                              preview_publish_seconds=time.perf_counter() - started)

    def end(self, hook):
        if getattr(self, "closed", False):
            return
        self.closed = True
        if self.publisher is None:
            return
        def finish():
            try:
                values = {"iteration": hook["iteration"]}
                metadata = getattr(self, "metadata", {})
                output, checkpoint = metadata.get("output"), metadata.get("checkpoint_path")
                if output or checkpoint:
                    project = Path(checkpoint) if checkpoint else Path(output) / "project.licht"
                    if project.is_file():
                        values["saved_project"] = str(project)
                self.publisher.finish(hook.get("terminal_state", "completed"), **values)
            except Exception:
                pass  # Optional preview faults never change the real training outcome.
            finally:
                self.pending = None
                try:
                    self.publisher.pool.close()
                except Exception:
                    pass
        writer = getattr(self, "writer", None)
        if writer is not None:
            # Serial retirement after the last immutable publication; never UI join.
            try:
                writer.submit(finish)
            except RuntimeError:
                pending = getattr(self, "pending", None)
                if pending is not None and not pending.done():
                    pending.add_done_callback(lambda _: finish())
                else:
                    finish()
            finally:
                writer.shutdown(wait=False)
        else:
            finish()


def automatic_publication_enabled():
    """A host adapter can own publication without loading application-specific code."""
    owner = os.environ.get("LFS_TRAINING_PUBLISHER", "").strip().lower()
    if owner:
        if owner not in {"automatic", "external"}:
            raise ValueError("LFS_TRAINING_PUBLISHER must be automatic or external")
        return owner == "automatic"
    # Compatibility with the first distributed Blender extension. New adapters
    # should use the generic setting above; their job/control readers stay local.
    return not bool(os.environ.get("LFS_BLENDER_JOB"))


def install():
    global _stream
    if _stream:
        _stream.hooks.clear()
        last_iteration = _stream.publisher.record.get("iteration", 0) if _stream.publisher else 0
        _stream.end({"terminal_state": "superseded", "iteration": last_iteration})
    _stream = None
    if automatic_publication_enabled():
        _stream = AutoStream()
