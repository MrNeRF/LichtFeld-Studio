# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Concrete pipeline hooks that talk to the running LichtFeld app.

The pipeline runs on a worker thread; everything that touches the UI or the
scene is marshalled onto the UI thread and the worker blocks for the answer.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional, TypeVar

import lichtfeld as lf

from ..environment import value as env_value
from ..process_runner import run_cancellable_process
from . import finalize, masking, toolchain
from .pipeline import ExtractParams, Hooks, PipelineError
from .toolchain import Resolved

PLUGIN_NAME = "video_to_splat"
SETTING_COLMAP_PATH = "colmap_path"
TRAINING_START_TIMEOUT = 600.0  # trainer initialisation on a large dataset
TRAINING_DISPATCH_TIMEOUT = 15.0  # start_training must at least report 'starting' by then
# Keep the sharpest frame of each interval. A fixed edge-ratio cutoff empties low-texture
# footage: a white room scored right at the old threshold of 10 on a real 360 clip.
SHARPNESS_WINDOW_CANDIDATES = -1  # auto: ~2*sqrt(frames per interval), 5..20
T = TypeVar("T")


def tr(key: str) -> str:
    return lf.ui.tr(key)


def cache_dir() -> Path:
    resolved = env_value("LFS_RESOLVED_CACHE_DIR")
    if resolved:
        return Path(resolved)
    return Path.home() / ".lichtfeld" / "cache"


def configured_colmap_path() -> str:
    try:
        return str(lf.plugins.settings(PLUGIN_NAME).get(SETTING_COLMAP_PATH, "") or "")
    except Exception:  # noqa: BLE001 - settings unavailable in some test harnesses
        return ""


def ask_on_ui_thread(fn: Callable[[], T]) -> T:
    """Run fn on the UI thread and block the calling worker until it returns."""
    done = threading.Event()
    box: dict = {}

    def runner() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the worker
            box["error"] = exc
        finally:
            done.set()

    lf.ui.schedule_on_ui_thread(runner)
    done.wait()
    if "error" in box:
        raise box["error"]
    return box["value"]


def _dialog_answer(title: str, message: str, buttons: list[str], style: str = "info") -> str:
    """Show a confirm dialog from the worker thread; return the clicked label."""
    answered = threading.Event()
    chosen: dict = {}

    def show() -> None:
        def on_result(label: str) -> None:
            chosen["label"] = label
            answered.set()

        lf.ui.confirm_dialog(title, message, buttons, on_result, style)

    lf.ui.schedule_on_ui_thread(show)
    answered.wait()
    return chosen.get("label", "")


def confirm_download() -> bool:
    accept = tr("video_to_splat.download_accept")
    decline = tr("video_to_splat.download_decline")
    message = tr("video_to_splat.download_message").format(
        size_mb=toolchain.COLMAP_WINDOWS_SIZE_MB, url=toolchain.COLMAP_WINDOWS_URL, version=toolchain.COLMAP_VERSION)
    return _dialog_answer(tr("video_to_splat.download_title"), message, [accept, decline]) == accept


def confirm_mask_setup(missing: list[str]) -> bool:
    accept = tr("video_to_splat.mask_setup_accept")
    decline = tr("video_to_splat.mask_setup_decline")
    message = tr("video_to_splat.mask_setup_message").format(
        packages=", ".join(missing) or "-", size_mb=masking.MODEL_SIZE_MB, url=masking.MODEL_URL)
    return _dialog_answer(tr("video_to_splat.mask_setup_title"), message, [accept, decline]) == accept


def mask_people(images: list[Path], masks_dir: Path, on_progress: Callable[[int, int], None],
                should_cancel: Callable[[], bool], carry_state: bool) -> dict:
    """Pipeline hook: install what RVM needs on consent, fetch the pinned model, write the masks."""
    missing = masking.missing_packages()
    model = masking.cached_model_path(cache_dir())
    try:
        if missing or not model.is_file():
            if not confirm_mask_setup(missing):
                return {"outcome": "skipped", "reason": tr("video_to_splat.mask_setup_declined")}
            for package in missing:
                lf.packages.install(package)
            model = masking.ensure_model(cache_dir(), download=download_file)
        matte = masking.RvmMatter(model)
        masking.write_masks(images, masks_dir, matte, on_progress=on_progress, should_cancel=should_cancel,
                            carry_state=carry_state)
    except masking.MaskingCancelled:
        return {"outcome": "cancelled"}
    except Exception as exc:  # noqa: BLE001 - masking is optional; the run goes on without it
        return {"outcome": "skipped", "reason": str(exc)}
    return {"outcome": "completed", "masked": masking.count_masked_people(masks_dir)}


def confirm_weak(ratio: float, registered: int, total: int) -> bool:
    go = tr("video_to_splat.weak_train_anyway")
    stop = tr("video_to_splat.weak_stop")
    message = tr("video_to_splat.weak_message").format(
        percent=int(round(ratio * 100)), registered=registered, total=total)
    return _dialog_answer(tr("video_to_splat.weak_title"), message, [go, stop], "warning") == go


def download_file(url: str, dest: Path, on_progress: Callable[[str], None]) -> None:
    from ..http import urlopen  # same HTTP stack (CA handling) as the plugin installer

    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response, open(dest, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        received = 0
        last_report = 0.0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            received += len(chunk)
            now = time.monotonic()
            if now - last_report > 0.5:
                last_report = now
                if total:
                    on_progress(tr("video_to_splat.downloading_percent").format(percent=int(100 * received / total)))
                else:
                    on_progress(tr("video_to_splat.downloading_mb").format(mb=received // (1 << 20)))


def resolve_colmap(on_progress: Callable[[str], None]) -> Resolved:
    try:
        return toolchain.resolve_colmap(
            configured_path=configured_colmap_path(), cache_dir=cache_dir(),
            request_consent=confirm_download, download=download_file, on_progress=on_progress)
    except toolchain.ToolchainError as exc:
        raise PipelineError(str(exc)) from exc


def extract(params: ExtractParams, on_progress: Callable[[int, int, int], None],
            should_cancel: Callable[[], bool]) -> dict:
    return lf.video.extract_frames(
        video_path=str(params.video), output_dir=str(params.dataset / "images"), fps=params.fps,
        scale=params.scale, rotation=params.rotation, sharpness_enabled=True, sharpness_threshold=0.0,
        sharpness_window_mode=True, sharpness_window_candidates=SHARPNESS_WINDOW_CANDIDATES,
        on_progress=on_progress, should_cancel=should_cancel)


def run_process(cmd: list[str], on_output: Callable[[str], None], should_cancel: Callable[[], bool]):
    return run_cancellable_process(cmd, on_output=on_output, should_cancel=should_cancel)


def create_project_and_wait(project: Path, dataset: Path, images_folder: str, *, timeout: float = 3600.0,
                            poll_seconds: float = 0.25) -> None:
    """Create the .licht project, import the dataset folder, and block until the import settles."""

    def create() -> None:
        created = lf.project_create(str(project), discard_changes=True, stop_training=False, overwrite=True)
        if created is False and not lf.project_create_pending():
            raise PipelineError(tr("video_to_splat.project_create_failed"))
        lf.load_file(path=str(dataset), is_dataset=True, images_folder=images_folder, discard_changes=True)

    ask_on_ui_thread(create)
    deadline = time.monotonic() + timeout
    while True:
        import_state = ask_on_ui_thread(lf.ui.get_import_state)
        if not import_state.get("active", False):
            if import_state.get("success", True):
                return
            raise PipelineError(tr("video_to_splat.import_failed").format(error=import_state.get("error", "")))
        if time.monotonic() > deadline:
            raise PipelineError(tr("video_to_splat.import_timeout"))
        time.sleep(poll_seconds)


EXPORT_TIMEOUT = 1800.0
_pending_package: Optional[Path] = None   # the package to finalize when the current training ends
_training_end_registered = False
_finalize_threads: list[threading.Thread] = []


def splat_tensors() -> Optional[dict]:
    """The first splat node of the scene as numpy arrays (UI thread), or None without a scene."""

    def read() -> Optional[dict]:
        scene = lf.get_scene()
        if scene is None:
            return None
        for node in scene.get_nodes():
            data = node.splat_data()
            if data is None:
                continue
            return {"name": node.name, "means": data.get_means().cpu().numpy(),
                    "scales": data.get_scaling().cpu().numpy(), "rotations": data.get_rotation().cpu().numpy(),
                    "colors": data.get_colors_rgb().cpu().numpy(), "opacity": data.get_opacity().cpu().numpy(),
                    "sh_degree": int(getattr(data, "active_sh_degree", 0))}
        return None

    return ask_on_ui_thread(read)


def export_splat(code: int, path: Path, *, timeout: float = EXPORT_TIMEOUT, poll_seconds: float = 0.25) -> Optional[str]:
    """Export the scene's splat nodes through lf.export_scene (asynchronous) and wait for the result."""
    names = ask_on_ui_thread(lambda: [n.name for n in lf.get_scene().get_nodes() if n.splat_data() is not None]
                             if lf.get_scene() is not None else [])
    if not names:
        return tr("video_to_splat.export_no_splat")
    path.parent.mkdir(parents=True, exist_ok=True)
    ask_on_ui_thread(lambda: lf.export_scene(code, str(path), names, 3))
    deadline = time.monotonic() + timeout
    while True:
        state = ask_on_ui_thread(lf.ui.get_export_state)
        if not state.get("active", False):
            return str(state.get("error") or "") or None
        if time.monotonic() > deadline:
            return tr("video_to_splat.export_timeout")
        time.sleep(poll_seconds)


def finalize_hooks(colmap: Optional[Path] = None) -> finalize.FinalizeHooks:
    if colmap is None:
        try:
            colmap = resolve_colmap(lambda _m: None).path
        except PipelineError:
            colmap = None
    return finalize.FinalizeHooks(splat_tensors=splat_tensors, export_splat=export_splat, run=run_process,
                                  colmap=colmap)


def finalize_now(package: Path, on_progress: Callable[[str], None] = lambda _m: None,
                 should_cancel: Callable[[], bool] = lambda: False) -> finalize.FinalizeResult:
    tensors = splat_tensors()
    name = (tensors or {}).get("name") or "scene"
    return finalize.finalize_package(package, finalize_hooks(), on_progress=on_progress, should_cancel=should_cancel,
                                     name=str(name))


def _on_training_end(*_args, **_kwargs) -> None:
    """Trainer callback (lf.on_training_end passes one payload object): finish the package on a worker
    thread, never on the trainer's."""
    global _pending_package
    package, _pending_package = _pending_package, None
    if package is None:
        return
    thread = threading.Thread(target=lambda: finalize_now(package), daemon=True, name="VideoToSplatFinalize")
    _finalize_threads.append(thread)
    thread.start()


def schedule_finalize_on_training_end(package: Path) -> None:
    global _pending_package, _training_end_registered
    _pending_package = package
    if not _training_end_registered:
        lf.on_training_end(_on_training_end)
        _training_end_registered = True


def start_training(mode: str, masked: bool = False, package: Optional[Path] = None, *,
                   timeout: float = TRAINING_START_TIMEOUT, dispatch_timeout: float = TRAINING_DISPATCH_TIMEOUT,
                   poll_seconds: float = 0.25) -> None:
    """Start training on the UI thread and block until the trainer runs.

    Equirectangular cameras can only be rasterised with GUT; the trainer refuses them otherwise
    (messages.err_non_pinhole). People masks are applied in "ignore" mode so a walker is a hole in
    the loss, not a black blob. Both go here and not at project creation, which resets the
    parameters. start_training itself returns after dispatch and reports failures only through the
    trainer state, so a start that falls back to 'ready' is an error here."""

    def start() -> None:
        params = lf.optimization_params()
        if mode == "equirectangular":
            params.gut = True
        if masked:
            params.mask_mode = lf.MaskMode.IGNORE
        lf.start_training()

    if package is not None:
        schedule_finalize_on_training_end(Path(package))
    ask_on_ui_thread(start)
    began = time.monotonic()
    seen_starting = False
    while True:
        state = ask_on_ui_thread(lf.trainer_state)
        if state in ("running", "paused"):
            return
        if state == "starting":
            seen_starting = True
        elif seen_starting:
            raise PipelineError(tr("video_to_splat.training_failed"))
        elapsed = time.monotonic() - began
        if elapsed > timeout or (not seen_starting and elapsed > dispatch_timeout):
            raise PipelineError(tr("video_to_splat.training_failed"))
        time.sleep(poll_seconds)


def build_hooks() -> Hooks:
    return Hooks(
        resolve_colmap=resolve_colmap,
        probe=lf.video.probe,
        extract=extract,
        run=run_process,
        create_project=lambda project, dataset, images_folder: create_project_and_wait(project, dataset, images_folder),
        train=start_training,
        confirm_weak=confirm_weak,
        mask_people=mask_people,
    )
