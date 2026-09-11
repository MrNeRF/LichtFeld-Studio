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
