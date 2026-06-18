# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression checks for the 3-point alignment viewport overlay."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (PROJECT_ROOT / rel_path).read_text(encoding="utf-8")


def test_alignment_overlay_uses_rendering_manager_overlay_not_rendering_engine():
    align_tool = _read("src/visualizer/tools/align_tool.cpp")
    get_overlay = align_tool[
        align_tool.index("getOverlayRenderer(const ToolContext& ctx)") :
        align_tool.index("struct PanelProjection")
    ]
    assert "getScreenOverlayRenderer()" in get_overlay
    assert "getRenderingEngineIfInitialized" not in get_overlay

    gui_manager = _read("src/visualizer/gui/gui_manager.cpp")
    assert "appendScreenOverlayCommandOverlays(params, rendering_manager->getScreenOverlayRenderer())" in gui_manager
    assert "overlay_renderer = rendering->getScreenOverlayRenderer()" in gui_manager


def test_alignment_projection_and_unprojection_respect_orthographic_settings():
    align_tool = _read("src/visualizer/tools/align_tool.cpp")
    assert "proj.orthographic = settings.orthographic" in align_tool
    assert "proj.ortho_scale = settings.ortho_scale" in align_tool
    assert "proj.orthographic,\n                proj.ortho_scale" in align_tool
    assert "world_radius * panel_proj.ortho_scale" in align_tool

    align_ops = _read("src/visualizer/operator/ops/align_ops.cpp")
    assert "render_settings.orthographic" in align_ops
    assert "render_settings.ortho_scale" in align_ops

    vksplat_header = _read("src/visualizer/rendering/vksplat_viewport_renderer.hpp")
    rendering_viewport = _read("src/visualizer/rendering/rendering_manager_viewport.cpp")
    assert "sampleDepthAtPixel" in vksplat_header
    assert "vksplat_viewport_renderer_->sampleDepthAtPixel" in rendering_viewport
