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
from typing import Callable, TypeVar

import lichtfeld as lf

from ..environment import value as env_value
from ..process_runner import run_cancellable_process
from . import toolchain
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


def start_training(mode: str, *, timeout: float = TRAINING_START_TIMEOUT,
                   dispatch_timeout: float = TRAINING_DISPATCH_TIMEOUT, poll_seconds: float = 0.25) -> None:
    """Start training on the UI thread and block until the trainer runs.

    Equirectangular cameras can only be rasterised with GUT; the trainer refuses them otherwise
    (messages.err_non_pinhole). start_training itself returns after dispatch and reports failures
    only through the trainer state, so a start that falls back to 'ready' is an error here."""

    def start() -> None:
        if mode == "equirectangular":
            lf.optimization_params().gut = True
        lf.start_training()

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
    )
