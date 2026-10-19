# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Point clouds created through Python must satisfy the native color contract."""

import json
import time

import pytest

from test_api_camera_project_runtime import camera_runtime, _editor
from test_render_on_demand_idle import _call, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def test_point_cloud_rejects_int32_colors(camera_runtime, tmp_path):
    endpoint = camera_runtime
    snapshot = tmp_path / "validation.json"
    history_before = _tool(endpoint, "history_get", {})["undo_count"]
    _editor(endpoint, f'''
import lichtfeld as lf, json
scene = lf.get_scene()
points = lf.Tensor.zeros([3, 3], device="cpu", dtype="float32")
colors = lf.Tensor.zeros([3, 3], device="cpu", dtype="int32")
count = len(scene.get_nodes())
result = {{"rejected": False}}
try:
    scene.add_point_cloud("invalid colors", points, colors)
except ValueError as error:
    result = {{"rejected": True, "message": str(error)}}
result["unchanged"] = len(scene.get_nodes()) == count
open({str(snapshot)!r}, "w").write(json.dumps(result))
''')
    result = json.loads(snapshot.read_text())
    if not result["rejected"]:
        saved = _call(endpoint, "tools/call", {"name": "project_save_as", "arguments": {
            "path": str(tmp_path / "invalid.licht"),
        }})
        (tmp_path / "invalid-save.json").write_text(json.dumps(saved, indent=2))
    assert result["rejected"], "int32 colors were accepted at creation"
    assert result["message"] == "colors must have dtype uint8 or float32"
    assert result["unchanged"]
    assert _tool(endpoint, "history_get", {})["undo_count"] == history_before


@pytest.mark.parametrize("dtype", ["uint8", "float32"])
def test_point_cloud_colors_save_reopen_exact(camera_runtime, tmp_path, dtype):
    endpoint = camera_runtime
    _editor(endpoint, f'''
import lichtfeld as lf, json
scene = lf.get_scene()
parent = scene.add_group("cloud parent")
points = lf.Tensor.zeros([3, 3], device="cpu", dtype="float32")
points[1, 0] = 1.25
points[2, 1] = -2.5
colors = lf.Tensor.zeros([3, 3], device="cpu", dtype="float32")
colors[0, 0] = {255 if dtype == "uint8" else .25}
colors[1, 1] = {127 if dtype == "uint8" else .5}
colors[2, 2] = {63 if dtype == "uint8" else .75}
colors = colors.to({dtype!r})
scene.add_point_cloud("cloud", points, colors, parent)
def cloud_state():
    node = lf.get_scene().get_node("cloud")
    cloud = node.point_cloud() if node else None
    if cloud is None:
        return None
    return {{"uuid": node.uuid, "means": cloud.means.cpu().tolist(),
             "colors": cloud.colors.cpu().tolist(), "dtype": str(cloud.colors.dtype)}}
''')
    snapshot = tmp_path / "cloud.json"
    capture = f"open({str(snapshot)!r}, 'w').write(json.dumps(cloud_state()))"
    _editor(endpoint, capture)
    before = json.loads(snapshot.read_text())
    project = tmp_path / "cloud.licht"
    _tool(endpoint, "project_save_as", {"path": str(project)})
    _tool(endpoint, "project_open", {"path": str(project), "discard_changes": True})
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        _editor(endpoint, capture)
        after = json.loads(snapshot.read_text())
        if after is not None:
            break
        time.sleep(.05)
    assert after == before
    _tool(endpoint, "project_save_as", {"path": str(tmp_path / "reopened.licht")})
