# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression checks for camera frustums under training image downscaling."""

from pathlib import Path
import json
import re
from string import Formatter


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_scene_graph_focus_uses_supported_rmlui_tab_index():
    source = (PROJECT_ROOT / "src/visualizer/gui/rmlui/elements/scene_graph_element.cpp").read_text(encoding="utf-8")
    focus = source.split("void SceneGraphElement::focusTree()", 1)[1].split("void SceneGraphElement::beginRename", 1)[0]
    assert 'SetProperty("tab-index", "auto")' in focus
    assert 'SetProperty("tab-index", "0")' not in source


def test_camera_pose_translations_cover_states_reasons_and_public_backend_name():
    sources = "\n".join((PROJECT_ROOT / path).read_text(encoding="utf-8") for path in (
        "src/core/parameters.cpp", "src/python/lfs/py_params.cpp",
        "src/python/lfs_plugins/training_panel.py", "src/visualizer/scene/camera_pose_view.hpp",
    ))
    keys = set(re.findall(r'training\.(pose\.[a-z_]+)', sources))
    keys.update("pose." + state for state in (
        "waiting", "ready", "updated", "rejected", "anchor", "evaluation",
        "frozen", "corrected", "unchanged", "not_refined",
    ))
    for path in (PROJECT_ROOT / "src/visualizer/gui/resources/locales").glob("*.json"):
        training = json.loads(path.read_text(encoding="utf-8"))["training"]
        for key in keys:
            assert training.get(key), (path.name, key)
            assert "fastgs" not in training[key].lower(), (path.name, key)
        assert "3DGS" in training["pose.backend"]
        assert "FastGS" not in training["tooltip.refine_camera_poses"]
        fields = [field for _, field, _, _ in Formatter().parse(training["pose.tooltip"]) if field is not None]
        assert sorted(fields) == ["0", "1", "2", "3", "4"], path.name


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
    assert "cameraPoseTooltip(*pose, *poses)" in graph
    assert "row.camera_pose_label.empty() || renaming" in graph
    assert 'setCachedOptionalProperty(slot.type_icon, "image-color", row.camera_loss_icon_color)' in graph


def test_pose_markers_keep_numbers_in_tooltip_and_survive_cache_reuse():
    graph = _read("src/visualizer/gui/rmlui/elements/scene_graph_element.cpp")
    assert 'setCachedInnerRml(slot.pose_badge, std::string(pose_indicator.symbol))' in graph
    assert 'pose_badge->SetProperty("width", "12dp")' in graph
    assert 'setCachedAttribute(slot.pose_badge, "title", row.camera_pose_label)' in graph
    gui = _read("src/visualizer/gui/gui_manager.cpp")
    marker = gui.index("const auto indicator = cameraPoseIndicator(")
    reuse = gui.index("if (!geometry_changed && !loss_changed && !atlas_changed && cache.valid)")
    assert marker < reuse
    assert "params.shape_overlay_triangles, params, a, b, color" in gui
    assert "cache.data->frustum_instances.push_back({.model = cache.models[camera_index], .color = color})" in gui


def test_pose_composition_keeps_shared_updates_outside_pose_visits():
    trainer = _read("src/training/trainer.cpp")
    pose = _function(trainer, "std::optional<FastGSCameraPoseOverride> refined_pose;", 'nvtxRangePush("rasterize")')
    assert "if (pose_session)" in pose
    assert "params_.optimization" in pose
    assert "compute_photometric_loss_with_mask(" in pose
    assert "corrected, target, mask, roi, output.alpha, opt, raw" in pose
    assert "bilateral_grid_->backward(grid_input, image_gradient, uid, false)" in pose
    assert "ppisp_->backward(isp_input, image_gradient, camera_id, uid, false)" in pose
    assert "optimizer_step" not in pose
    assert "zero_grad" not in pose
    assert "bg_image, opt.mip_filter" in pose
    assert "if (refined_pose && normal_prior_world_space_)" in trainer
    integration = _read("src/training/camera_pose/trainer_pose_integration.cpp")
    assert "resolved_camera_pose_stop_step()" in integration
    assert "pose_session_config_from_state(saved)" in integration
    assert "source_cameras = scene_->getActiveCameras()" in trainer
    assert "camera_pose_sources_ = source_cameras" in trainer
