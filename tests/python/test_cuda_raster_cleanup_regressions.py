# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Static guards for the removed CUDA Gaussian raster viewer path."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (PROJECT_ROOT / rel_path).read_text(encoding="utf-8")


def test_rendering_engine_no_longer_exposes_cuda_gaussian_viewer_api():
    header = _read("src/rendering/include/rendering/rendering.hpp")
    removed_names = [
        "GaussianGpuFrameResult",
        "GaussianImageResult",
        "DualGaussianImageResult",
        "HoveredGaussianQueryRequest",
        "ScreenPositionRenderRequest",
        "renderGaussiansGpuFrame",
        "renderGaussiansImage",
        "renderGaussiansImagePair",
        "queryHoveredGaussianId",
        "renderGaussianScreenPositions",
        "createRasterOnly",
        "initializeRasterOnly",
        "isRasterInitialized",
    ]
    for name in removed_names:
        assert name not in header


def test_gui_tools_do_not_call_removed_cuda_gaussian_raster_methods():
    checked_sources = [
        "src/rendering/raster_rendering_engine.cpp",
        "src/visualizer/rendering/rendering_manager_viewport.cpp",
        "src/visualizer/gui/async_task_manager.cpp",
        "src/visualizer/selection/selection_service.cpp",
    ]
    removed_calls = [
        "renderGaussiansGpuFrame",
        "renderGaussiansImage",
        "queryHoveredGaussianId",
        "renderGaussianScreenPositions",
    ]
    for rel_path in checked_sources:
        source = _read(rel_path)
        for call in removed_calls:
            assert call not in source

