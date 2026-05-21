# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression checks for orthographic VkSplat viewport support."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (PROJECT_ROOT / rel_path).read_text(encoding="utf-8")


def test_vksplat_viewer_does_not_reject_orthographic_requests():
    source = _read("src/visualizer/rendering/vksplat_viewport_renderer.cpp")

    assert "not orthographic cameras" not in source
    assert "kVkSplatCameraModelOrthographic = 1u" in source
    assert "frame_view.orthographic ? kVkSplatCameraModelOrthographic" in source
    assert "frame_view.ortho_scale" in source


def test_vksplat_shaders_project_and_raytrace_orthographic_cameras():
    utils = _read("src/rendering/rasterizer/vksplat_fwd/shader/src/slang/utils.slang")

    assert "ORTHO = 1" in utils
    assert "project_point_ortho" in utils
    assert "compute_jacobian_ortho" in utils
    assert 'cam.model == int(CameraModel::ORTHO)' in utils
    assert "camera_origin + mul(view_to_world_rotation, float3(uv, 0.0f))" in utils
    assert "float3(0.0f, 0.0f, 1.0f)" in utils
