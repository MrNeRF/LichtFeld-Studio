# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""The camera MCP tool clamps renderable fields of view to the viewport range."""

import math

import pytest

from runtime_test_app import isolated_app
from test_render_on_demand_idle import _call, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _set_view(endpoint, args):
    result = _call(endpoint, "tools/call", {"name": "camera_set_view", "arguments": args})
    return result.get("structuredContent", {}), result.get("isError", False)


def test_camera_set_view_clamps_renderable_fov_and_rejects_impossible_values(tmp_path):
    with isolated_app(tmp_path) as endpoint:
        result, is_error = _set_view(endpoint, {
            "eye": [2, 3, 3], "target": [0, 0, 0], "fov_degrees": 45,
        })
        assert not is_error, result
        baseline = _tool(endpoint, "camera_get")["camera"]
        args = {"eye": [3, 3, 3], "target": [0, 0, 0]}

        # JSON has no NaN literal; an enormous valid JSON number converts to
        # non-finite float in the C++ handler and exercises the same guard.
        for fov in (-20, 0, 180, 1000, 1e100):
            result, is_error = _set_view(endpoint, {**args, "fov_degrees": fov})
            assert is_error and "error" in result, (fov, result)
            assert "fov_degrees" in result["error"]["message"]
            current = _tool(endpoint, "camera_get")["camera"]
            assert current["eye"] == pytest.approx(baseline["eye"])
            assert current["fov_degrees"] == pytest.approx(baseline["fov_degrees"])

        min_fov = math.degrees(2 * math.atan(24 / (2 * 200)))
        max_fov = math.degrees(2 * math.atan(24 / (2 * 10)))
        for fov in (0.5, 5, 179.9):
            result, is_error = _set_view(endpoint, {**args, "fov_degrees": fov})
            assert not is_error, (fov, result)
            effective_fov = min_fov if fov < min_fov else max_fov
            assert result["fov_degrees"] == pytest.approx(effective_fov, abs=0.02)
            assert result["fov_clamped"] is True
            assert "warning" in result and "clamp" in result["warning"].lower()
            current = _tool(endpoint, "camera_get")["camera"]
            assert current["fov_degrees"] == pytest.approx(effective_fov, abs=0.02)

        for fov in (min_fov, 7, 45, 100, max_fov):
            result, is_error = _set_view(endpoint, {**args, "fov_degrees": fov})
            assert not is_error, (fov, result)
            assert result["fov_degrees"] == pytest.approx(fov, abs=0.02)
            assert result["fov_clamped"] is False
            assert "warning" not in result
            current = _tool(endpoint, "camera_get")["camera"]
            assert current["fov_degrees"] == pytest.approx(fov, abs=0.02)
