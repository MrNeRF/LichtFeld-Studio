# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression checks for camera frustums under training image downscaling."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (PROJECT_ROOT / rel_path).read_text(encoding="utf-8")


def _function(source: str, start: str, end: str) -> str:
    return source[source.index(start) : source.index(end, source.index(start))]


def _assert_uses_calibrated_frustum(source: str) -> None:
    assert "camera_width()" in source
    assert "camera_height()" in source
    assert "FoVy()" in source
    assert "image_width()" not in source
    assert "image_height()" not in source
    assert "focal2fov" not in source


def test_displayed_frustum_ignores_training_image_resize():
    gui_manager = _read("src/visualizer/gui/gui_manager.cpp")
    frustum_model = _function(
        gui_manager,
        "cameraFrustumModelMatrix(",
        "void appendEquirectangularCameraFrustum(",
    )

    _assert_uses_calibrated_frustum(frustum_model)


def test_frustum_picking_uses_the_same_calibration_contract():
    raster_engine = _read("src/rendering/raster_rendering_engine.cpp")
    frustum_points = _function(
        raster_engine,
        "cameraFrustumWorldPoints(",
        "projectFrustumPoint(",
    )

    _assert_uses_calibrated_frustum(frustum_points)


def test_frustum_selection_bounds_ignore_training_image_resize():
    scene_manager = _read("src/visualizer/scene/scene_manager.cpp")
    camera_bounds = _function(
        scene_manager,
        "const bool is_equirect =",
        "glm::vec2 screen_min",
    )

    assert "camera_width()" in camera_bounds
    assert "camera_height()" in camera_bounds
    assert "FoVy()" in camera_bounds
    assert "image_width()" not in camera_bounds
    assert "image_height()" not in camera_bounds
    assert "focal2fov" not in camera_bounds


def test_live_pose_publication_invalidates_both_frustum_caches():
    gui = _read("src/visualizer/gui/gui_manager.cpp")
    assert "pose_generation_ == pose_generation" in gui
    assert "pose_sequence_ == pose_sequence" in gui
    assert "cache.pose_generation != pose_generation" in gui
    assert "cache.pose_sequence != pose_sequence" in gui
    assert "geometry_changed = camera_data_changed || !cache.valid" in gui
    assert "scene_transforms[i], poses.get()" in gui


def test_current_pose_is_wired_to_pick_focus_selection_and_scene_graph():
    picker = _read("src/visualizer/rendering/camera_interaction_service.cpp")
    renderer = _read("src/rendering/raster_rendering_engine.cpp")
    assert ".camera_world_to_camera = std::move(current_poses)" in picker
    assert "request.camera_world_to_camera[i]" in renderer
    for path in ("src/visualizer/input/input_controller.cpp",
                 "src/visualizer/scene/scene_manager.cpp",
                 "src/visualizer/selection/selection_service.cpp"):
        source = _read(path)
        assert "activeCameraPoses()" in source
        assert "cameraWorldToCamera(" in source
    graph = _read("src/visualizer/gui/rmlui/elements/scene_graph_element.cpp")
    assert "cameraPoseDisplacementLabel(pose)" in graph
    assert "row.camera_pose_label.empty() || renaming" in graph
    assert 'setCachedOptionalProperty(slot.type_icon, "image-color", row.camera_loss_icon_color)' in graph


def test_pose_markers_keep_numbers_in_tooltip_and_survive_cache_reuse():
    graph = _read("src/visualizer/gui/rmlui/elements/scene_graph_element.cpp")
    assert 'setCachedInnerRml(slot.pose_badge, std::string(pose_indicator.symbol))' in graph
    assert 'pose_badge->SetProperty("width", "12dp")' in graph
    assert 'row.camera_pose_state, row.camera_pose_label' in graph
    gui = _read("src/visualizer/gui/gui_manager.cpp")
    marker = gui.index("const auto indicator = cameraPoseIndicator(")
    reuse = gui.index("if (!geometry_changed && !loss_changed && !atlas_changed && cache.valid)")
    assert marker < reuse
    assert "params.shape_overlay_triangles, params, a, b, color" in gui
    assert "cache.data->frustum_instances.push_back({.model = cache.models[camera_index], .color = color})" in gui
