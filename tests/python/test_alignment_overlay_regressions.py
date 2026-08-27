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

    align_ops = _read("src/visualizer/operator/ops/align_ops.cpp")
    assert "kAlignMarkerWorldRadius * ortho_scale" in align_ops
    assert "render_settings.orthographic" in align_ops
    assert "render_settings.ortho_scale" in align_ops

    vksplat_header = _read("src/visualizer/rendering/vksplat_viewport_renderer.hpp")
    rendering_viewport = _read("src/visualizer/rendering/rendering_manager_viewport.cpp")
    assert "sampleDepthAtPixel" in vksplat_header
    assert "vksplat_viewport_renderer_->sampleDepthAtPixel" in rendering_viewport


def test_alignment_editable_points_and_axis_snap_are_wired():
    align_ops = _read("src/visualizer/operator/ops/align_ops.cpp")
    assert "kMarkerHitRadiusPx = 8.0" in align_ops
    assert "KEY_DELETE" in align_ops
    assert "snapAlignNormalToNodeAxes" in align_ops
    assert "takeAlignUiAction" in align_ops
    assert "drag_active_" in align_ops

    services = _read("src/visualizer/core/services.hpp")
    assert "align_selected_point_" in services
    assert "align_axis_snap_enabled_" in services
    assert "AlignUiAction" in services

    capabilities = _read("src/visualizer/gui_capabilities.cpp")
    assert "isAlignTransformTargetType" in capabilities
    editor = _read("src/visualizer/core/editor_context.cpp")
    assert "has_editable_align_selection_" in editor
    assert "isAlignTransformTargetType" in editor


def test_alignment_overlay_single_depth_sample_and_edge_labels():
    align_tool = _read("src/visualizer/tools/align_tool.cpp")
    assert align_tool.count("getDepthAtPixel(") == 1
    assert "drawEdgeLengthLabels" in align_tool
    assert '%.2f' in align_tool or '"%.2f"' in align_tool
    assert "forceGridOn" in align_tool
    assert "show_grid" in align_tool
    assert "SNAPPED" in align_tool or "align.snapped" in align_tool
    assert "SELECTED_OUTLINE" in align_tool or "palette.primary" in align_tool


def test_alignment_toolbar_rml_and_locale_keys():
    rml = _read("src/visualizer/gui/rmlui/resources/viewport_overlay.rml")
    assert 'id="primary-align-toolbar"' in rml
    assert 'id="secondary-align-toolbar"' in rml
    assert 'data-for="button : align_action_buttons"' in rml

    en = _read("src/visualizer/gui/resources/locales/en.json")
    for key in ('"snapped"', '"apply"', '"clear"', '"snap"', '"edge_to_axis"'):
        assert key in en
    keys = _read("src/visualizer/gui/string_keys.hpp")
    assert "SNAPPED" in keys
    assert 'align.snapped' in keys


def test_masked_gaussian_depth_pick_is_implemented():
    viewport = _read("src/visualizer/rendering/rendering_manager_viewport.cpp")
    assert "Masked Gaussian depth render skipped" not in viewport
    assert "renderDepthCaptureToPreviewSlotWithState" in viewport
    assert "readPreviewDepth" in viewport
    assert "renderMedianDepthAtPixel" in viewport
    # Gaussian branch must apply the node filter, not only the point-cloud path.
    fn = viewport[
        viewport.index("renderDepthAtPixelForNodeMask") :
        viewport.index("} // namespace lfs::vis")
    ]
    assert "node_visibility_mask" in fn
    assert "sampleDepthTensorAt" in fn

    align_ops = _read("src/visualizer/operator/ops/align_ops.cpp")
    align_ops_h = _read("src/visualizer/operator/ops/align_ops.hpp")
    assert "logged_masked_depth_fallback_" in align_ops or "logged_masked_depth_fallback_" in align_ops_h
    assert "logged_exact_depth_fallback_" not in align_ops and "logged_exact_depth_fallback_" not in align_ops_h
    assert "over_gui" in align_ops
    assert "renderMedianDepthAtPixel" in align_ops
    assert "falling back to full-scene exact depth" in align_ops or "falling back to interactive depth" in align_ops
    assert "getNodeVisibilityMask" in align_ops
    assert "getVisibleNodeIndex" in align_ops


def test_align_commit_picks_use_exact_depth_capture():
    """Interactive depth is the pick source of truth; masked capture is gated by agreement."""
    align_ops = _read("src/visualizer/operator/ops/align_ops.cpp")
    unproject = align_ops[
        align_ops.index("AlignPickPointOperator::unprojectScreenPoint") :
        align_ops.index("bool AlignPickPointOperator::applyAlignment")
    ]
    assert "renderMedianDepthAtPixel" in unproject
    assert "renderDepthAtPixelForNodeMask" in unproject
    assert "getDepthAtPixel" in unproject
    assert "kDepthAgreeTolerance" in unproject
    assert "panel_info->panel" in unproject
    # Point-cloud fallback must forward the panel (panel-local coords under split view).
    viewport = _read("src/visualizer/rendering/rendering_manager_viewport.cpp")
    assert "getDepthAtPixel(x, y, panel)" in viewport
    # Hover preview must keep the cheap interactive sample (exactly one call site).
    align_tool = _read("src/visualizer/tools/align_tool.cpp")
    assert align_tool.count("getDepthAtPixel(") == 1
    assert "renderMedianDepthAtPixel" not in align_tool


def test_align_edge_to_axis_is_wired():
    services = _read("src/visualizer/core/services.hpp")
    assert "align_edge_to_axis_enabled_" in services
    assert "getAlignEdgeToAxisEnabled" in services

    align_ops = _read("src/visualizer/operator/ops/align_ops.cpp")
    assert "alignEdgeToWorldXRotation" in align_ops
    assert "getAlignEdgeToAxisEnabled" in align_ops

    align_tool = _read("src/visualizer/tools/align_tool.cpp")
    assert "getAlignEdgeToAxisEnabled" in align_tool
    assert '"X"' in align_tool or "'X'" in align_tool

    py_ui = _read("src/python/lfs/py_ui.cpp")
    assert "get_align_edge_to_axis" in py_ui
    assert "set_align_edge_to_axis" in py_ui

    toolbar = _read("src/python/lfs_plugins/toolbar.py")
    assert "align_toggle_edge_to_axis" in toolbar
    assert "align.edge_to_axis" in toolbar

    locale_dir = PROJECT_ROOT / "src/visualizer/gui/resources/locales"
    for path in locale_dir.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert '"edge_to_axis"' in text, path.name
