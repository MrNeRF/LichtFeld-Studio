# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cameras created through the editor must survive native project persistence."""

import importlib.util
import json
import os
import urllib.request
from pathlib import Path

import pytest

from test_render_on_demand_idle import _call, _initialize, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.fixture
def camera_runtime(tmp_path, monkeypatch):
    executable = os.environ.get("LFS_EXECUTABLE")
    if not executable:
        pytest.skip("requires LFS_EXECUTABLE and a display")
    port = os.environ.get("LFS_MCP_PORT", "45841")
    endpoint = f"http://127.0.0.1:{port}/mcp"
    home = tmp_path / "home"
    home.mkdir()
    # urllib caches proxy configuration in its process-global opener.
    monkeypatch.setattr(urllib.request, "_opener", None)
    for name, value in {
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "XDG_DATA_HOME": str(home / ".local/share"),
        "HTTP_PROXY": "http://127.0.0.1:9",
        "HTTPS_PROXY": "http://127.0.0.1:9",
        "http_proxy": "http://127.0.0.1:9",
        "https_proxy": "http://127.0.0.1:9",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "UV_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "LFS_MCP_ENDPOINT": endpoint,
        "LFS_MCP_BRIDGE_LOG": str(tmp_path / "app.log"),
        "LFS_MCP_BRIDGE_LOCK": str(tmp_path / "bridge.lock"),
    }.items():
        monkeypatch.setenv(name, value)
    for name in ("PYTHONHOME", "PYTHONPATH"):
        monkeypatch.delenv(name, raising=False)
    path = Path(__file__).resolve().parents[2] / "scripts/lichtfeld_mcp_bridge.py"
    spec = importlib.util.spec_from_file_location("camera_test_bridge", path)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    assert not bridge.endpoint_ready(), "test port is already in use"
    monkeypatch.setattr(bridge, "pick_launch_command", lambda: [
        executable, "--no-splash", "--mcp-port", port,
    ])
    try:
        bridge.ensure_server_ready()
        _initialize(endpoint)
        tools = _call(endpoint, "tools/list")["tools"]
        (tmp_path / "tools.json").write_text(json.dumps(tools, indent=2))
        assert {"editor_run", "project_save_as", "project_open"} <= {t["name"] for t in tools}
        _call(endpoint, "resources/list")
        for uri in ("runtime/catalog", "runtime/state", "ui/state", "scene/state", "selection/current"):
            _call(endpoint, "resources/read", {"uri": "lichtfeld://" + uri})
        yield endpoint
    finally:
        bridge.cleanup_spawned_processes()


def _editor(endpoint, code):
    result = _tool(endpoint, "editor_run", {
        "code": code, "show_console": False, "timeout_ms": 10000,
    })
    if result.get("running"):
        result = _tool(endpoint, "editor_wait", {"timeout_ms": 10000})
    assert result.get("completed") and result.get("success"), result
    assert "Traceback" not in result.get("output", {}).get("text", ""), result
    return result.get("output", {}).get("text", "")


def test_api_camera_save_reopen_exact(camera_runtime, tmp_path):
    endpoint = camera_runtime
    _editor(endpoint, '''
import lichtfeld as lf, json
scene = lf.get_scene()
parent = scene.add_group("API cameras")
for index, translation in enumerate(([0., 0., 0.], [1.25, -2.5, 3.75])):
    r = lf.Tensor.eye(3, device="cpu", dtype="float32")
    t = lf.Tensor.zeros([3, 1], device="cpu", dtype="float32")
    for row, value in enumerate(translation):
        t[row, 0] = value
    scene.add_camera(f"API camera {index}", parent, r, t, 50., 55., 100, 120, uid=index)
def camera_state():
    return {n.name: {"R": n.camera_R.cpu().tolist(),
                     "T": n.camera_T.cpu().reshape([3]).tolist(),
                     "fx": n.camera_focal_x, "fy": n.camera_focal_y,
                     "width": n.camera_width, "height": n.camera_height,
                     "uid": n.camera_uid, "uuid": n.uuid}
            for n in lf.get_scene().get_nodes() if n.has_camera}
''')
    snapshot = tmp_path / "camera-state.json"
    capture = f"open({str(snapshot)!r}, 'w').write(json.dumps(camera_state(), sort_keys=True))"
    _editor(endpoint, capture)
    before = json.loads(snapshot.read_text())
    assert len(before) == 2
    assert before["API camera 1"]["T"] == [1.25, -2.5, 3.75]
    project = tmp_path / "api-cameras.licht"
    saved = _tool(endpoint, "project_save_as", {"path": str(project)})
    assert saved.get("success", True), saved
    assert project.is_file()
    opened = _tool(endpoint, "project_open", {"path": str(project), "discard_changes": True})
    assert opened.get("success", True), opened
    _editor(endpoint, capture)
    after = json.loads(snapshot.read_text())
    assert after == before
    # Re-saving exercises the restored distortion tensors too.
    saved = _tool(endpoint, "project_save_as", {"path": str(tmp_path / "reopened.licht")})
    assert saved.get("success", True), saved

    _editor(endpoint, '''
scene = lf.get_scene()
count = len(scene.get_nodes())
r = lf.Tensor.eye(3, device="cpu")
t = lf.Tensor.zeros([3], device="cpu")
cases = [
    (lf.Tensor(), t, 50., 50., 100, 100, "R"),
    (r, lf.Tensor(), 50., 50., 100, 100, "T"),
    (lf.Tensor.zeros([9], device="cpu"), t, 50., 50., 100, 100, "R"),
    (lf.Tensor.eye(3, device="cpu", dtype="int32"), t, 50., 50., 100, 100, "R"),
    (r, lf.Tensor.zeros([1, 3], device="cpu"), 50., 50., 100, 100, "T"),
    (r, lf.Tensor.zeros([3], device="cpu", dtype="int32"), 50., 50., 100, 100, "T"),
    (r, t, 0., 50., 100, 100, "focal"),
    (r, t, 50., -1., 100, 100, "focal"),
    (r, t, float("nan"), 50., 100, 100, "focal"),
    (r, t, 50., float("inf"), 100, 100, "focal"),
    (r, t, 50., 50., 0, 100, "width"),
    (r, t, 50., 50., 100, -1, "height"),
    (lf.Tensor.full([3, 3], float("nan"), device="cpu"), t, 50., 50., 100, 100, "R"),
    (r, lf.Tensor.full([3], float("inf"), device="cpu"), 50., 50., 100, 100, "T"),
]
for rotation, translation, fx, fy, width, height, field in cases:
    try:
        scene.add_camera("invalid", -1, rotation, translation, fx, fy, width, height)
    except ValueError as error:
        assert field in str(error), str(error)
    else:
        raise AssertionError(f"accepted invalid {field}")
    assert len(scene.get_nodes()) == count
print("rejected all invalid cameras without inserting nodes")
''')
