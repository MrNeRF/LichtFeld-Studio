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
from .toolchain import Resolved

STAGE_NAMES = ["toolchain", "extract", "features", "match", "map", "create_project", "train"]
COLMAP_STAGES = ("features", "match", "map")
STATE_FILE = "pipeline.json"
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
    video: Path
    project: Path
    mode_override: probe_mod.ModeOverride = "auto"
    preset: str = probe_mod.DEFAULT_PRESET
    use_gpu: bool = True

    def fingerprint(self) -> str:
        raw = json.dumps(
            {"video": str(self.video), "mtime": _mtime(self.video), "mode": self.mode_override,
             "preset": self.preset, "gpu": self.use_gpu},
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @property
    def dataset(self) -> Path:
        return self.project / DATASET_DIR


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
    train: Callable[[], None]
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
    version: int = STATE_VERSION

    def done(self) -> set[str]:
        return {entry["name"] for entry in self.completed}

    def mark(self, name: str, **facts) -> None:
        self.completed.append({"name": name, "at": time.time(), **facts})

    def save(self, project: Path) -> None:
        project.mkdir(parents=True, exist_ok=True)
        tmp = project / (STATE_FILE + ".tmp")
        tmp.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")
        os.replace(tmp, project / STATE_FILE)

    @classmethod
    def load(cls, project: Path, fingerprint: str) -> "State":
        path = project / STATE_FILE
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
                   model_dir=str(raw.get("model_dir", "")))


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _count_frames(images: Path) -> int:
    if not images.is_dir():
        return 0
    return sum(1 for p in images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def run(request: Request, hooks: Hooks, *, on_progress: ProgressFn, should_cancel: Callable[[], bool]) -> State:
    """Run all stages, resuming from pipeline.json when the request fingerprint matches."""
    state = State.load(request.project, request.fingerprint())
    done = state.done()
    preset = probe_mod.PRESETS[request.preset]
    images, database, sparse = sfm.dataset_paths(request.dataset)
    tail: list[str] = []

    def log(line: str) -> None:
        tail.append(line)
        del tail[:-LOG_TAIL_LINES]

    def report(stage: str, fraction: float, message: str) -> None:
        on_progress(stage, fraction, message, "\n".join(tail))

    def check_cancel() -> None:
        if should_cancel():
            raise PipelineCancelled()

    # toolchain: COLMAP is needed whenever a COLMAP stage still has to run, even on resume.
    colmap: Optional[Resolved] = None
    colmap_pending = any(stage not in done for stage in COLMAP_STAGES)
    if "toolchain" not in done or colmap_pending:
        report("toolchain", 0.0, "Locating COLMAP")
        colmap = hooks.resolve_colmap(lambda msg: report("toolchain", 0.0, msg))
        if "toolchain" not in done:
            state.mark("toolchain", colmap=str(colmap.path), version=list(colmap.version))
            state.save(request.project)
    check_cancel()

    info = hooks.probe(str(request.video))
    mode = probe_mod.decide_mode(info, request.mode_override)
    state.mode = mode

    # extract
    if "extract" not in done:
        probe_root = request.project if request.project.exists() else request.project.parent
        free = hooks.free_bytes(probe_root)
        if free < LOW_DISK_BYTES:
            report("extract", 0.0, f"Only {free / 1024 ** 3:.1f} GB free on the destination; frames need several GB")
        if images.exists():
            shutil.rmtree(images)
        images.mkdir(parents=True, exist_ok=True)
        params = ExtractParams(video=request.video, dataset=request.dataset, fps=preset.fps,
                               scale=probe_mod.plan_frame_scale(int(info.get("width", 0) or 0), preset.max_width),
                               rotation=0)
        report("extract", 0.0, "Extracting frames")

        def extract_progress(current: int, total: int, discarded: int) -> None:
            fraction = current / total if total else 0.0
            report("extract", fraction, f"Frame {current}/{total}, {discarded} discarded as blurry")

        result = hooks.extract(params, extract_progress, should_cancel)
        outcome = result.get("outcome")
        if outcome == "cancelled":
            raise PipelineCancelled()
        if outcome != "completed":
            raise PipelineError(f"Frame extraction failed: {result.get('error') or 'unknown error'}")
        frames = _count_frames(images)
        if frames < MIN_FRAMES:
            raise PipelineError(
                f"Only {frames} usable frames were extracted; at least {MIN_FRAMES} are needed. "
                "Raise the frame rate or lower the sharpness threshold."
            )
        state.frames = frames
        state.mark("extract", frames=frames)
        state.save(request.project)
    else:
        state.frames = state.frames or _count_frames(images)
    check_cancel()

    # COLMAP stages
    if colmap_pending:
        assert colmap is not None
        commands = sfm.plan_commands(colmap=colmap.path, dataset=request.dataset, mode=mode,
                                     overlap=preset.overlap, use_gpu=request.use_gpu)
        for stage, cmd in zip(COLMAP_STAGES, commands):
            if stage in done:
                continue
            if stage == "features" and database.exists():
                database.unlink()
            if stage == "map":
                if sparse.exists():
                    shutil.rmtree(sparse)
                sparse.mkdir(parents=True, exist_ok=True)
            tool = cmd[1]
            report(stage, 0.0, f"Running COLMAP {tool}")

            def on_output(line: str, _stage: str = stage, _tool: str = tool) -> None:
                log(line)
                report(_stage, 0.0, f"COLMAP {_tool}")

            try:
                completed = hooks.run(cmd, on_output, should_cancel)
            except ProcessCancelled:
                raise PipelineCancelled()
            if completed.returncode != 0:
                raise PipelineError(
                    f"COLMAP {tool} failed with exit code {completed.returncode}.\n" + "\n".join(tail[-10:])
                )
            state.mark(stage)
            state.save(request.project)
            check_cancel()

    # choose model, judge quality
    best = sfm.find_largest_model(sparse)
    if best is None:
        raise PipelineError("COLMAP produced no reconstruction. The footage may lack overlap or texture.")
    model_dir, registered = best
    state.registered = registered
    state.model_dir = str(model_dir)
    ratio = registered / state.frames if state.frames else 0.0
    if "create_project" not in done and (ratio < MIN_REGISTERED_RATIO or registered < MIN_REGISTERED):
        if not hooks.confirm_weak(ratio, registered, state.frames):
            state.save(request.project)
            raise PipelineStopped()

    # LichtFeld expects sparse/0; move the chosen model there when it is not already.
    canonical = sparse / "0"
    if model_dir != canonical:
        if canonical.exists():
            shutil.rmtree(canonical)
        shutil.move(str(model_dir), str(canonical))
        state.model_dir = str(canonical)
    state.save(request.project)

    if "create_project" not in done:
        report("create_project", 0.0, "Creating project")
        hooks.create_project(request.project, request.dataset, sfm.IMAGES_DIR)
        state.mark("create_project")
        state.save(request.project)
    check_cancel()

    if "train" not in done:
        report("train", 0.0, "Starting training")
        hooks.train()
        state.mark("train")
        state.save(request.project)
    report("train", 1.0, f"Done: {registered}/{state.frames} frames registered")
    return state
