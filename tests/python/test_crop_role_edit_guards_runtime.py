# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise crop role editing restrictions through the deployed application."""
import json

import pytest

from crop_role_runtime import pretraining_app, set_volume, volume
from test_render_on_demand_idle import _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.fixture
def crop_endpoint(tmp_path):
    with pretraining_app(tmp_path) as endpoint:
        for role in ("roi", "crop"):
            assert _tool(endpoint, "ui_tool_invoke", {"tool_id": "builtin." + role})["success"]
        yield endpoint


@pytest.mark.parametrize("role,shape", [("roi", "box"), ("roi", "polygon"),
                                         ("crop", "box"), ("crop", "polygon"), ("crop", "ellipsoid")])
@pytest.mark.parametrize("lock_owner", [False, True])
def test_locked_volume_rejects_mcp_and_python_edits(crop_endpoint, role, shape, lock_owner):
    endpoint = crop_endpoint
    assert _tool(endpoint, "crop_volume_set", {"owner": "PointCloud", "role": role, "shape": shape})["success"]
    before = volume(endpoint, role)
    target = "PointCloud" if lock_owner else before["name"]
    assert _tool(endpoint, "scene.set_node_locked", {"name": target, "locked": True})["success"]
    failures = []
    changes = {"box": {"min": [-2, -2, -2]}, "ellipsoid": {"radii": [2, 3, 4]},
               "polygon": {"vertices": [[-1, -1], [2, -1], [0, 2]]}}[shape]
    response = set_volume(endpoint, role, **changes)
    if not response.get("isError"):
        failures.append(("MCP accepted geometry", response))
    if shape == "polygon":
        python = _tool(endpoint, "editor_run", {"code": "import json,lichtfeld as lf\n"
            f"print(json.dumps(lf.ui.set_crop_polygon_outline({before['uuid']!r}, [[-2,-1],[2,-1],[0,3]])))",
            "wait_for_completion": True, "show_console": False, "timeout_ms": 30000})
        if json.loads(python["output"]["text"]) is not False:
            failures.append(("Python accepted outline", python))
    after = volume(endpoint, role)
    for field in ("uuid", "shape", "min", "max", "radii", "vertices", "half_depth", "translation", "rotation", "scale"):
        if field in before and before[field] != after[field]:
            failures.append((field, before[field], after[field]))
    assert not failures, failures


@pytest.mark.parametrize("role", ["roi", "crop"])
def test_invalid_full_and_partial_box_bounds_are_rejected_atomically(crop_endpoint, role):
    endpoint = crop_endpoint
    assert _tool(endpoint, "crop_volume_set", {"owner": "PointCloud", "role": role,
                 "min": [-1, -1, -1], "max": [1, 1, 1]})["success"]
    baseline = volume(endpoint, role)
    invalid = [{"min": [2, 2, 2], "max": [-2, -2, -2]}, {"min": [2, 0, 0]},
               {"max": [-2, 0, 0]}, {"min": [1, 0, 0]}]
    failures = []
    for changes in invalid:
        response = set_volume(endpoint, role, **changes)
        after = volume(endpoint, role)
        if not response.get("isError") or after != baseline:
            failures.append((changes, response, after))
            assert _tool(endpoint, "crop_volume_set", {"owner": "PointCloud", "role": role,
                         "min": baseline["min"], "max": baseline["max"]})["success"]
    assert _tool(endpoint, "crop_volume_set", {"owner": "PointCloud", "role": role, "shape": "polygon"})["success"]
    polygon_before = volume(endpoint, role)
    response = set_volume(endpoint, role, shape="box", min=[2, 2, 2], max=[-2, -2, -2])
    after_conversion = volume(endpoint, role)
    if not response.get("isError") or after_conversion != polygon_before:
        failures.append(("invalid conversion", response, after_conversion))
    assert not failures, failures


@pytest.mark.parametrize("hidden_role", ["roi", "crop"])
def test_selecting_other_volume_preserves_sibling_visibility(crop_endpoint, hidden_role):
    endpoint = crop_endpoint
    other_role = "crop" if hidden_role == "roi" else "roi"
    hidden = volume(endpoint, hidden_role)
    other = volume(endpoint, other_role)
    assert _tool(endpoint, "scene_select_node", {"name": hidden["name"]})["success"]
    assert _tool(endpoint, "scene.set_node_visibility", {"name": hidden["name"], "visible": False})["success"]
    assert volume(endpoint, hidden_role)["visible"] is False
    assert _tool(endpoint, "scene_select_node", {"name": other["name"]})["success"]
    assert volume(endpoint, hidden_role)["visible"] is False
    assert volume(endpoint, other_role)["visible"] is True
    for role in ("roi", "crop"):
        info = volume(endpoint, role)
        assert _tool(endpoint, "scene.set_node_visibility", {"name": info["name"], "visible": True})["success"]
    for name in (hidden["name"], other["name"], "PointCloud"):
        assert _tool(endpoint, "scene_select_node", {"name": name})["success"]
        assert volume(endpoint, "roi")["visible"] is True
        assert volume(endpoint, "crop")["visible"] is True
