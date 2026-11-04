# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""MCP tools that drive the pipeline without the dialog: start, status, cancel.

Argument validation happens here so a caller gets a plain error dict instead of a
traceback; the pipeline itself still validates what only it can know."""
from __future__ import annotations

from pathlib import Path

import lichtfeld as lf

from . import probe, runner
from .pipeline import Request
from .project_bridge import build_hooks

_runner = runner.PipelineRunner()
TOOL_NAMES = ("video_to_splat_start", "video_to_splat_status", "video_to_splat_cancel")


def video_to_splat_start(source: str, project: str, mode: str = "auto", preset: str = "balanced",
                         use_gpu: bool = True, mask_people: bool = True, build_mesh: bool = False) -> dict:
    """Build a trained splat from a video file or a folder of photos. ``project`` is the .licht path to
    create (the extension is added when missing); ``mode`` is auto, on or off for 360 handling; ``preset``
    is fast, balanced or high; ``mask_people`` masks walkers out of training; ``build_mesh`` also runs
    COLMAP dense stereo, Poisson meshing and texturing into the package folder (slow, needs CUDA).
    Returns {started, project, work_dir} or {started: false, error}."""
    source_path = Path(source)
    if not source_path.exists():
        return {"started": False, "error": f"Source does not exist: {source_path}"}
    if mode not in ("auto", "on", "off"):
        return {"started": False, "error": f"mode must be auto, on or off, not '{mode}'"}
    if preset not in probe.PRESETS:
        return {"started": False, "error": f"preset must be one of {', '.join(probe.PRESETS)}, not '{preset}'"}
    if lf.is_training_active():
        return {"started": False, "error": lf.ui.tr("video_to_splat.training_busy")}
    project_path = Path(project)
    if project_path.suffix.lower() != ".licht":
        project_path = project_path.with_name(project_path.name + ".licht")
    request = Request(video=source_path, project=project_path, mode_override=mode, preset=preset,
                      use_gpu=bool(use_gpu), mask_people=bool(mask_people), build_mesh=bool(build_mesh))
    return _runner.start(request, build_hooks())


def video_to_splat_status() -> dict:
    """Progress of the current or last run: active, stage, fraction, message, done, error, cancelled,
    frames, registered, mode, project, work_dir, package_dir."""
    return _runner.status()


def video_to_splat_cancel() -> dict:
    """Ask the running pipeline to stop; COLMAP is killed and finished stages are kept for a resume."""
    return _runner.cancel()


def register_mcp_tools() -> None:
    lf.mcp.register_tool(video_to_splat_start, "video_to_splat_start", video_to_splat_start.__doc__ or "")
    lf.mcp.register_tool(video_to_splat_status, "video_to_splat_status", video_to_splat_status.__doc__ or "")
    lf.mcp.register_tool(video_to_splat_cancel, "video_to_splat_cancel", video_to_splat_cancel.__doc__ or "")


def unregister_mcp_tools() -> None:
    for name in TOOL_NAMES:
        lf.mcp.unregister_tool(name)
