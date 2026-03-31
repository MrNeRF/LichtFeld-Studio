# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for Python gizmo context fallback coordinates."""

import pytest


def test_gizmo_context_projects_points_in_front_of_default_camera(lf):
    ctx = lf.GizmoContext()
    camera = ctx.camera_position
    forward = ctx.camera_forward

    in_front = tuple(camera[i] + forward[i] * 5.0 for i in range(3))
    behind = tuple(camera[i] - forward[i] * 5.0 for i in range(3))

    assert ctx.world_to_screen(in_front) == pytest.approx((400.0, 300.0))
    assert ctx.world_to_screen(behind) is None


def test_gizmo_context_center_ray_uses_negative_z_forward(lf):
    ctx = lf.GizmoContext()

    assert ctx.screen_to_world_ray((400.0, 300.0)) == pytest.approx((0.0, 0.0, -1.0))
