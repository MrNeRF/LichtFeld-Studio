# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage orchestration for video -> frames -> COLMAP -> project -> training.

Everything that touches the app, the filesystem beyond the project folder, or a
subprocess is injected through Hooks so the orchestration is unit-testable.
Runs on a worker thread; callers marshal UI work themselves (see project_bridge).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from subprocess import CompletedProcess
from typing import Callable, Optional

from ..process_runner import ProcessCancelled
from . import probe as probe_mod
from . import sfm
from .text import _tr
from .toolchain import Resolved

STAGE_NAMES = ["toolchain", "extract", "features", "match", "map", "level", "create_project", "train"]
COLMAP_STAGES = ("features", "match", "map")
TOOL_STAGES = COLMAP_STAGES + ("level",)  # every stage that needs the COLMAP binary
STATE_FILE = "pipeline.json"
EXTRACTION_REPORT = "extraction_metadata.json"
DATASET_DIR = "dataset"
MIN_FRAMES = 20
MIN_REGISTERED = 20
MIN_REGISTERED_RATIO = 0.6
LOW_DISK_BYTES = 10 * 1024 ** 3
STATE_VERSION = 1
LOG_TAIL_LINES = 40

ProgressFn = Callable[[str, float, str, str], None]  # stage, fraction 0..1, message, log tail

__all__ = [
    "STAGE_NAMES", "STATE_FILE", "DATASET_DIR", "MIN_FRAMES", "MIN_REGISTERED", "MIN_REGISTERED_RATIO",
    "ExtractParams", "Hooks", "PipelineCancelled", "PipelineError", "PipelineStopped", "ProcessCancelled",
    "Request", "State", "run",
]


class PipelineError(Exception):
    """Unrecoverable failure; str(error) is user-facing."""


class PipelineCancelled(Exception):
    """The user cancelled."""


class PipelineStopped(Exception):
    """The user chose to stop at a decision point (weak reconstruction)."""


@dataclass(frozen=True)
class Request:
    """What to build. ``project`` is the .licht file the dialog targets; the frames,
    COLMAP database, sparse model and pipeline.json live in ``work_dir`` beside it
    (``<location>/<name>/``), because a .licht project is a single-file container."""

    video: Path
    project: Path
    mode_override: probe_mod.ModeOverride = "auto"
    preset: str = probe_mod.DEFAULT_PRESET
    use_gpu: bool = True

    @property
    def is_photo_set(self) -> bool:
        """``video`` may also be a folder of photos; then the extract stage copies them instead."""
        return self.video.is_dir()

    def fingerprint(self) -> str:
        facts = {"video": str(self.video), "mtime": _mtime(self.video), "mode": self.mode_override,
                 "preset": self.preset, "gpu": self.use_gpu}
        if self.is_photo_set:
            facts["photos"] = [p.name for p in probe_mod.list_photos(self.video)]
        raw = json.dumps(facts, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @property
    def work_dir(self) -> Path:
        return self.project.parent / self.project.stem

    @property
    def dataset(self) -> Path:
        return self.work_dir / DATASET_DIR


@dataclass(frozen=True)
class ExtractParams:
    video: Path
    dataset: Path
    fps: float
    scale: float
    rotation: int


@dataclass
class Hooks:
    """Side effects the orchestrator needs. project_bridge builds the real ones."""

    resolve_colmap: Callable[[Callable[[str], None]], Resolved]
    probe: Callable[[str], dict]
    extract: Callable[[ExtractParams, Callable[[int, int, int], None], Callable[[], bool]], dict]
    run: Callable[[list[str], Callable[[str], None], Callable[[], bool]], CompletedProcess]
    create_project: Callable[[Path, Path, str], None]
    train: Callable[[str], None]  # receives the camera mode
    confirm_weak: Callable[[float, int, int], bool]
    free_bytes: Callable[[Path], int] = lambda path: shutil.disk_usage(path).free


@dataclass
class State:
    fingerprint: str
    completed: list[dict] = field(default_factory=list)
    mode: str = ""
    frames: int = 0
    registered: int = 0
    model_dir: str = ""
    use_gpu: Optional[bool] = None  # set to False once a COLMAP stage had to fall back to the CPU
    version: int = STATE_VERSION

    def done(self) -> set[str]:
        return {entry["name"] for entry in self.completed}

    def mark(self, name: str, **facts) -> None:
        self.completed.append({"name": name, "at": time.time(), **facts})

    def save(self, work_dir: Path) -> None:
        work_dir.mkdir(parents=True, exist_ok=True)
        tmp = work_dir / (STATE_FILE + ".tmp")
        tmp.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")
        os.replace(tmp, work_dir / STATE_FILE)

    @classmethod
    def load(cls, work_dir: Path, fingerprint: str) -> "State":
        path = work_dir / STATE_FILE
        if not path.is_file():
            return cls(fingerprint=fingerprint)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls(fingerprint=fingerprint)
        if not isinstance(raw, dict) or raw.get("fingerprint") != fingerprint or raw.get("version") != STATE_VERSION:
            return cls(fingerprint=fingerprint)
        completed = [entry for entry in raw.get("completed", []) if isinstance(entry, dict) and "name" in entry]
        return cls(fingerprint=fingerprint, completed=completed, mode=str(raw.get("mode", "")),
                   frames=int(raw.get("frames", 0) or 0), registered=int(raw.get("registered", 0) or 0),
                   model_dir=str(raw.get("model_dir", "")),
                   use_gpu=raw.get("use_gpu") if isinstance(raw.get("use_gpu"), bool) else None)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _move_extraction_report(images: Path, dataset: Path) -> None:
    """The extractor writes its report next to the frames. COLMAP reads every file in the images folder,
    and its CPU feature extractor crashes on a non-image, so the report lives one level up."""
    report = images / EXTRACTION_REPORT
    if report.is_file():
        os.replace(report, dataset / EXTRACTION_REPORT)


def _remove_database(database: Path) -> None:
    """Delete database.db and the SQLite side files a crashed COLMAP leaves next to it."""
    for suffix in ("", "-wal", "-shm", "-journal"):
        try:
            os.remove(str(database) + suffix)
        except FileNotFoundError:
            pass


def _stage_shortfall(stage: str, database: Path) -> Optional[str]:
    """A user-facing message when a COLMAP stage exited 0 but left nothing behind, else None."""
    if stage == "features" and sfm.count_images_with_features(database) == 0:
        return _tr("video_to_splat.error.no_features",
                   "COLMAP found no features in the extracted frames. The frames may be too dark or uniform.")
    if stage == "match" and sfm.count_verified_pairs(database) == 0:
        return _tr("video_to_splat.error.no_matches",
                   "COLMAP found no matches between frames. Move slower or choose the High preset.")
    return None


def _import_photos(source: Path, images: Path, on_progress: Callable[[int, int], None]) -> int:
    """Bring a photo set into the dataset folder: hard links where the filesystem allows, copies otherwise."""
    photos = probe_mod.list_photos(source)
    for index, photo in enumerate(photos, start=1):
        target = images / photo.name
        try:
            os.link(photo, target)
        except OSError:
            shutil.copy2(photo, target)
        on_progress(index, len(photos))
    return len(photos)


def _count_frames(images: Path) -> int:
    if not images.is_dir():
        return 0
    return sum(1 for p in images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def run(request: Request, hooks: Hooks, *, on_progress: ProgressFn, should_cancel: Callable[[], bool]) -> State:
    """Run all stages, resuming from pipeline.json when the request fingerprint matches."""
    state = State.load(request.work_dir, request.fingerprint())
    done = state.done()
    preset = probe_mod.PRESETS[request.preset]
    images, database, sparse = sfm.dataset_paths(request.dataset)
    tail: list[str] = []

    def log(line: str) -> None:
        tail.append(line)
        del tail[:-LOG_TAIL_LINES]

    def report(stage: str, fraction: float, message: str) -> None:
        log_tail = "\n".join(tail)
        on_progress(stage, fraction, message, log_tail)

    def check_cancel() -> None:
        if should_cancel():
            raise PipelineCancelled()

    # toolchain: COLMAP is needed whenever a COLMAP stage still has to run, even on resume.
    colmap: Optional[Resolved] = None
    colmap_pending = any(stage not in done for stage in TOOL_STAGES)
    if "toolchain" not in done or colmap_pending:
        report("toolchain", 0.0, _tr("video_to_splat.progress.locating_colmap", "Locating COLMAP"))
        colmap = hooks.resolve_colmap(lambda msg: report("toolchain", 0.0, msg))
        if "toolchain" not in done:
            state.mark("toolchain", colmap=str(colmap.path), version=list(colmap.version))
            state.save(request.work_dir)
    check_cancel()

    info = probe_mod.probe_photo_folder(request.video) if request.is_photo_set else hooks.probe(str(request.video))
    mode = probe_mod.decide_mode(info, request.mode_override)
    state.mode = mode

    # extract
    if "extract" not in done:
        probe_root = request.work_dir if request.work_dir.exists() else request.project.parent
        free = hooks.free_bytes(probe_root)
        if free < LOW_DISK_BYTES:
            report("extract", 0.0, _tr("video_to_splat.progress.low_disk",
                                       "Only {free_gb} GB free on the destination; frames need several GB",
                                       free_gb=f"{free / 1024 ** 3:.1f}"))
        if images.exists():
            shutil.rmtree(images)
        images.mkdir(parents=True, exist_ok=True)
        if request.is_photo_set:
            report("extract", 0.0, _tr("video_to_splat.progress.copying_photos", "Copying photos"))

            def photo_progress(current: int, total: int) -> None:
                report("extract", current / total if total else 0.0,
                       _tr("video_to_splat.progress.copy_photo", "Photo {current}/{total}",
                           current=current, total=total))

            frames = _import_photos(request.video, images, photo_progress)
            if frames < MIN_FRAMES:
                raise PipelineError(_tr(
                    "video_to_splat.error.too_few_photos",
                    "Only {frames} photos were found; at least {minimum} are needed.",
                    frames=frames, minimum=MIN_FRAMES))
            state.frames = frames
            state.mark("extract", frames=frames)
            state.save(request.work_dir)
            check_cancel()
            return _run_colmap_and_train(request, hooks, state, done, preset, mode, colmap, images, database,
                                         sparse, tail, log, report, check_cancel, should_cancel)
        params = ExtractParams(video=request.video, dataset=request.dataset, fps=preset.fps,
                               scale=probe_mod.plan_frame_scale(int(info.get("width", 0) or 0), preset.max_width),
                               rotation=0)
        report("extract", 0.0, _tr("video_to_splat.progress.extracting", "Extracting frames"))

        def extract_progress(current: int, total: int, discarded: int) -> None:
            fraction = current / total if total else 0.0
            report("extract", fraction, _tr("video_to_splat.progress.extract_frame",
                                            "Frame {current}/{total}, {discarded} discarded as blurry",
                                            current=current, total=total, discarded=discarded))

        result = hooks.extract(params, extract_progress, should_cancel)
        outcome = result.get("outcome")
        if outcome == "cancelled":
            raise PipelineCancelled()
        if outcome != "completed":
            raise PipelineError(_tr("video_to_splat.error.extract_failed", "Frame extraction failed: {error}",
                                    error=result.get("error") or "unknown error"))
        _move_extraction_report(images, request.dataset)
        frames = _count_frames(images)
        if frames < MIN_FRAMES:
            raise PipelineError(_tr(
                "video_to_splat.error.too_few_frames",
                "Only {frames} usable frames were extracted; at least {minimum} are needed. "
                "Use a longer clip or raise the frame rate.",
                frames=frames, minimum=MIN_FRAMES))
        state.frames = frames
        state.mark("extract", frames=frames)
        state.save(request.work_dir)
    else:
        _move_extraction_report(images, request.dataset)
        state.frames = state.frames or _count_frames(images)
    check_cancel()
    return _run_colmap_and_train(request, hooks, state, done, preset, mode, colmap, images, database, sparse,
                                 tail, log, report, check_cancel, should_cancel)


def _run_colmap_and_train(request: Request, hooks: Hooks, state: State, done: set[str], preset, mode: str,
                          colmap: Optional[Resolved], images: Path, database: Path, sparse: Path, tail: list[str],
                          log: Callable[[str], None], report: Callable[[str, float, str], None],
                          check_cancel: Callable[[], None], should_cancel: Callable[[], bool]) -> State:
    """Stages after the frames exist: COLMAP, leveling, project creation, training."""
    colmap_pending = any(stage not in done for stage in TOOL_STAGES)

    # COLMAP stages. A stage that exits 0 but leaves the database empty is rerun on the CPU once: the official
    # CUDA build has no kernels for older GPUs and fails every image without a nonzero exit code.
    if colmap_pending:
        assert colmap is not None
        use_gpu = request.use_gpu if state.use_gpu is None else state.use_gpu
        for stage in COLMAP_STAGES:
            if stage in done:
                continue
            while True:
                cmd = sfm.plan_commands(colmap=colmap.path, dataset=request.dataset, mode=mode,
                                        overlap=preset.overlap, use_gpu=use_gpu)[COLMAP_STAGES.index(stage)]
                if stage == "features":
                    _remove_database(database)
                if stage == "match":
                    sfm.clear_matches(database)
                if stage == "map":
                    if sparse.exists():
                        shutil.rmtree(sparse)
                    sparse.mkdir(parents=True, exist_ok=True)
                tool = cmd[1]
                report(stage, 0.0, _tr("video_to_splat.progress.running_colmap", "Running COLMAP {tool}", tool=tool))

                def on_output(line: str, _stage: str = stage, _tool: str = tool) -> None:
                    log(line)
                    report(_stage, 0.0, _tr("video_to_splat.progress.colmap", "COLMAP {tool}", tool=_tool))

                try:
                    completed = hooks.run(cmd, on_output, should_cancel)
                except ProcessCancelled:
                    raise PipelineCancelled()
                if completed.returncode != 0:
                    summary = _tr("video_to_splat.error.colmap_failed", "COLMAP {tool} failed with exit code {code}.",
                                  tool=tool, code=completed.returncode)
                    raise PipelineError(summary + "\n" + "\n".join(tail[-10:]))
                shortfall = _stage_shortfall(stage, database)
                if shortfall is None:
                    break
                if not use_gpu:
                    raise PipelineError(shortfall + "\n" + "\n".join(tail[-10:]))
                use_gpu = False
                state.use_gpu = False
                state.save(request.work_dir)
                log(shortfall)
                report(stage, 0.0, _tr("video_to_splat.progress.cpu_fallback",
                                       "COLMAP {tool} found nothing on the GPU; retrying on the CPU", tool=tool))
            state.mark(stage, gpu=use_gpu)
            state.save(request.work_dir)
            check_cancel()

    # choose model, judge quality
    best = sfm.find_largest_model(sparse)
    if best is None:
        raise PipelineError(_tr("video_to_splat.error.no_model",
                                "COLMAP produced no reconstruction. The footage may lack overlap or texture."))
    model_dir, registered = best
    state.registered = registered
    state.model_dir = str(model_dir)
    ratio = registered / state.frames if state.frames else 0.0
    if "create_project" not in done and (ratio < MIN_REGISTERED_RATIO or registered < MIN_REGISTERED):
        if not hooks.confirm_weak(ratio, registered, state.frames):
            state.save(request.work_dir)
            raise PipelineStopped()

    # LichtFeld expects sparse/0; move the chosen model there when it is not already.
    canonical = sparse / "0"
    if model_dir != canonical:
        if canonical.exists():
            shutil.rmtree(canonical)
        shutil.move(str(model_dir), str(canonical))
        state.model_dir = str(canonical)
    state.save(request.work_dir)

    # Gravity-align the model so the floor is level in the viewer. Cosmetic: a failure keeps the model as is.
    if "level" not in done:
        assert colmap is not None
        report("level", 0.0, _tr("video_to_splat.progress.leveling", "Leveling the scene"))
        leveled = sparse / sfm.LEVELED_DIR
        if leveled.exists():
            shutil.rmtree(leveled)
        leveled.mkdir(parents=True, exist_ok=True)
        level_cmd = sfm.plan_level_command(colmap=colmap.path, dataset=request.dataset, mode=mode)

        def on_level_output(line: str) -> None:
            log(line)
            report("level", 0.0, _tr("video_to_splat.progress.colmap", "COLMAP {tool}", tool=level_cmd[1]))

        try:
            completed = hooks.run(level_cmd, on_level_output, should_cancel)
        except ProcessCancelled:
            shutil.rmtree(leveled, ignore_errors=True)
            raise PipelineCancelled()
        applied = completed.returncode == 0 and sfm.read_registered_image_count(leveled / "images.bin") == registered
        if applied:
            shutil.rmtree(canonical)
            shutil.move(str(leveled), str(canonical))
        else:
            log(_tr("video_to_splat.progress.level_skipped", "Leveling skipped; the scene keeps its orientation"))
            shutil.rmtree(leveled, ignore_errors=True)
        state.mark("level", applied=applied)
        state.save(request.work_dir)
        check_cancel()

    if "create_project" not in done:
        report("create_project", 0.0, _tr("video_to_splat.progress.creating_project", "Creating project"))
        hooks.create_project(request.project, request.dataset, sfm.IMAGES_DIR)
        state.mark("create_project")
        state.save(request.work_dir)
    check_cancel()

    if "train" not in done:
        report("train", 0.0, _tr("video_to_splat.progress.starting_training", "Starting training"))
        hooks.train(mode)
        state.mark("train")
        state.save(request.work_dir)
    report("train", 1.0, _tr("video_to_splat.progress.done", "Done: {registered}/{total} frames registered",
                             registered=registered, total=state.frames))
    return state
