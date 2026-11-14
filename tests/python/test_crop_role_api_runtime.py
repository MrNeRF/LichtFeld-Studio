# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise role-specific Python and MCP crop control in the real application."""
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.integration]
from test_render_on_demand_idle import _call, _initialize, _tool


@pytest.mark.parametrize("order", [("roi", "crop"), ("crop", "roi")])
def test_role_discovery_activation_and_independent_manipulation(order):
    dataset = os.environ.get("LFS_REAL_CROP_DATASET")
    if not dataset:
        pytest.skip("Set LFS_REAL_CROP_DATASET to an existing COLMAP dataset")
    root = Path(__file__).resolve().parents[2]
    executable = Path(os.environ.get("LFS_EXECUTABLE", root / "build" / "LichtFeld-Studio.exe"))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}/mcp"
    app = subprocess.Popen([str(executable), "--no-download", "--mcp-port", str(port)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        deadline = time.monotonic() + 90
        while True:
            assert app.poll() is None
            try:
                _initialize(endpoint)
                break
            except OSError:
                assert time.monotonic() < deadline
                time.sleep(.1)
        out = str(Path(tempfile.mkdtemp(prefix="lfs-role-api-")).resolve())
        assert _tool(endpoint, "scene_load_dataset", {"path": dataset, "output_path": out, "min_track_length": 0})["success"]
        assert _tool(endpoint, "runtime_job_wait", {"job_id": "import.dataset", "until": "inactive", "timeout_ms": 90000})["success"]
        assert _tool(endpoint, "scene_select_node", {"name": "PointCloud"})["success"]
        catalog = {item["name"]: item for item in _call(endpoint, "tools/list", {})["tools"]}
        available = set(catalog)
        assert {"crop_volume_get", "crop_volume_set"} <= available
        for role in order:
            state = _tool(endpoint, "ui_tool_invoke", {"tool_id": "builtin." + role})
            assert state["success"] and state["registry_active_tool_id"] == "builtin." + role, state
            python = _tool(endpoint, "editor_run", {"code": "import json,lichtfeld as lf\n"
                f"print(json.dumps(lf.ui.activate_crop_tool({role!r})))",
                "wait_for_completion": True, "timeout_ms": 30000, "show_console": False})
            assert json.loads(python["output"]["text"]) is True
        rejected = _tool(endpoint, "editor_run", {"code": "import json,lichtfeld as lf\nprint(json.dumps(lf.ui.activate_crop_tool('invalid')))",
            "wait_for_completion": True, "timeout_ms": 30000, "show_console": False})
        assert json.loads(rejected["output"]["text"]) is False
        resource = _call(endpoint, "resources/read", {"uri": "lichtfeld://scene/nodes"})
        nodes = json.loads(resource["contents"][0]["text"])["nodes"]
        volumes = {n["crop_role"]: n for n in nodes if n.get("crop_role") in ("roi", "crop")}
        assert set(volumes) == {"roi", "crop"}, volumes
        assert volumes["roi"]["uuid"] != volumes["crop"]["uuid"]
        owner_uuid = next(n["uuid"] for n in nodes if n["name"] == "PointCloud")
        for role in ("roi", "crop"):
            assert _tool(endpoint, "crop_volume_get", {"owner_uuid": owner_uuid, "role": role})["volume"]["uuid"] == volumes[role]["uuid"]
        for role in ("roi", "crop"):
            info = _tool(endpoint, "crop_volume_get", {"owner": "PointCloud", "role": role})
            assert info["volume"]["uuid"] == volumes[role]["uuid"], info
        before = _tool(endpoint, "crop_volume_get", {"owner": "PointCloud", "role": "roi"})["volume"]
        no_op = _call(endpoint, "tools/call", {"name": "crop_volume_set", "arguments": {"owner": "PointCloud", "role": "roi"}})
        assert no_op.get("isError"), no_op
        edited = _tool(endpoint, "crop_volume_set", {"owner": "PointCloud", "role": "crop", "shape": "ellipsoid", "radii": [1.0, 2.0, 3.0], "translation": [1.0, 2.0, 3.0]})
        assert edited["volume"]["shape"] == "ellipsoid", edited
        assert edited["volume"]["radii"] == pytest.approx([1.0, 2.0, 3.0])
        assert edited["volume"]["translation"] == pytest.approx([1.0, 2.0, 3.0])
        after = _tool(endpoint, "crop_volume_get", {"owner": "PointCloud", "role": "roi"})["volume"]
        assert after == before
        python = _tool(endpoint, "editor_run", {"code": "import json,lichtfeld as lf\ns=lf.get_scene()\n"
            "print(json.dumps([(n.name, n.ellipsoid().role) for n in s.get_nodes() if n.ellipsoid()]))",
            "wait_for_completion": True, "timeout_ms": 30000, "show_console": False})
        assert json.loads(python["output"]["text"])[0][1] == "secondary"
        roi_poly = _tool(endpoint, "crop_volume_set", {"owner": "PointCloud", "role": "roi", "shape": "polygon",
            "vertices": [[-2, -1], [2, -1], [0, 2]], "half_depth": 3.0, "inverse": True})["volume"]
        assert roi_poly["uuid"] == before["uuid"] and roi_poly["shape"] == "polygon"
        assert roi_poly["half_depth"] == pytest.approx(3.0) and roi_poly["inverse"] is True
        rejected = _call(endpoint, "tools/call", {"name": "crop_volume_set", "arguments":
            {"owner": "PointCloud", "role": "roi", "shape": "ellipsoid"}})
        assert rejected.get("isError"), rejected
        assert _tool(endpoint, "crop_volume_get", {"owner": "PointCloud", "role": "roi"})["volume"] == roi_poly
        assert _tool(endpoint, "history_undo")["success"]
        assert _tool(endpoint, "crop_volume_get", {"owner": "PointCloud", "role": "roi"})["volume"] == before
        assert _tool(endpoint, "history_redo")["success"]
        assert _tool(endpoint, "crop_volume_get", {"owner": "PointCloud", "role": "roi"})["volume"] == roi_poly
        assert _tool(endpoint, "crop_volume_get", {"owner": "PointCloud", "role": "crop"})["volume"]["radii"] == pytest.approx([1.0, 2.0, 3.0])
        fitted_roi = _tool(endpoint, "crop_volume_action", {"owner_uuid": owner_uuid, "role": "roi", "action": "fit"})["volume"]
        assert fitted_roi["uuid"] == roi_poly["uuid"]
        reset_roi = _tool(endpoint, "crop_volume_action", {"owner_uuid": owner_uuid, "role": "roi", "action": "reset"})["volume"]
        assert reset_roi["uuid"] == roi_poly["uuid"]
        assert _tool(endpoint, "history_undo")["success"]
        assert _tool(endpoint, "history_undo")["success"]
        assert _tool(endpoint, "crop_volume_get", {"owner": "PointCloud", "role": "roi"})["volume"] == roi_poly
        deleted_roi = _tool(endpoint, "crop_volume_action", {"owner_uuid": owner_uuid, "role": "roi", "action": "delete"})
        assert deleted_roi["uuid"] == roi_poly["uuid"]
        assert _tool(endpoint, "crop_volume_get", {"owner": "PointCloud", "role": "crop"})["volume"]["uuid"] == volumes["crop"]["uuid"]
        assert _tool(endpoint, "history_undo")["success"]
        assert _tool(endpoint, "crop_volume_get", {"owner": "PointCloud", "role": "roi"})["volume"] == roi_poly
        refused = _call(endpoint, "tools/call", {"name": "crop_volume_action", "arguments":
            {"owner": "PointCloud", "role": "roi", "action": "apply"}})
        assert refused.get("isError"), refused
        for action in ("fit", "reset"):
            result = _tool(endpoint, "crop_volume_action", {"owner": "PointCloud", "role": "crop", "action": action})
            assert result["success"] and result["volume"]["uuid"] == volumes["crop"]["uuid"], result
        result = _tool(endpoint, "crop_volume_action", {"owner": "PointCloud", "role": "crop", "action": "delete"})
        assert result["success"] and result["uuid"] == volumes["crop"]["uuid"]
        assert _tool(endpoint, "history_undo")["success"]
        assert _tool(endpoint, "crop_volume_get", {"owner": "PointCloud", "role": "crop"})["volume"]["uuid"] == volumes["crop"]["uuid"]
        assert _tool(endpoint, "crop_volume_get", {"owner": "PointCloud", "role": "roi"})["volume"] == roi_poly
        result = _tool(endpoint, "crop_volume_action", {"owner": "PointCloud", "role": "crop", "action": "apply"})
        assert result["success"] and result["uuid"] == volumes["crop"]["uuid"], result
        assert _tool(endpoint, "crop_volume_get", {"owner": "PointCloud", "role": "roi"})["volume"] == roi_poly
        assert _tool(endpoint, "history_undo")["success"]
        assert _tool(endpoint, "crop_volume_get", {"owner": "PointCloud", "role": "crop"})["volume"]["uuid"] == volumes["crop"]["uuid"]
        python = _tool(endpoint, "editor_run", {"code": "import json,lichtfeld as lf\n"
            "ok=lf.ui.activate_crop_tool('crop')\nchanged=lf.ui.change_crop_tool_shape('box')\n"
            "print(json.dumps({'activated':ok,'changed':changed,'roles':[(n.uuid,n.crop_role) for n in lf.get_scene().get_nodes() if n.crop_role],'discoverable':all('crop_role' in dir(n) for n in lf.get_scene().get_nodes() if n.crop_role)}))",
            "wait_for_completion": True, "timeout_ms": 30000, "show_console": False})
        state = json.loads(python["output"]["text"])
        assert state["activated"] and state["changed"] and state["discoverable"], state
        assert catalog["crop_volume_action"]["annotations"]["destructiveHint"] is True
        assert {tuple(item) for item in state["roles"]} == {(volumes["roi"]["uuid"], "roi"), (volumes["crop"]["uuid"], "crop")}
        assert _tool(endpoint, "crop_volume_get", {"owner": "PointCloud", "role": "roi"})["volume"] == roi_poly
        python = _tool(endpoint, "editor_run", {"code": "import json,lichtfeld as lf\n"
            "roi=next(n for n in lf.get_scene().get_nodes() if n.crop_role == 'roi')\n"
            "print(json.dumps(lf.ui.set_crop_polygon_outline(roi.uuid, [[-1,-1],[2,-1],[0,3]])))",
            "wait_for_completion": True, "timeout_ms": 30000, "show_console": False})
        assert json.loads(python["output"]["text"]) is True
        invalid = _tool(endpoint, "editor_run", {"code": "import json,lichtfeld as lf\n"
            "roi=next(n for n in lf.get_scene().get_nodes() if n.crop_role == 'roi')\n"
            "print(json.dumps(lf.ui.set_crop_polygon_outline(roi.uuid, [[0,0],[1,1]])))",
            "wait_for_completion": True, "timeout_ms": 30000, "show_console": False})
        assert json.loads(invalid["output"]["text"]) is False
        assert _tool(endpoint, "history_undo")["success"]
        assert _tool(endpoint, "crop_volume_get", {"owner": "PointCloud", "role": "roi"})["volume"] == roi_poly
        saved = Path(out) / "role-api.licht"
        assert _tool(endpoint, "project_save_as", {"path": str(saved)}).get("success", True)
        assert _tool(endpoint, "project_open", {"path": str(saved), "discard_changes": True}).get("success", True)
        deadline = time.monotonic() + 30
        while True:
            response = _call(endpoint, "tools/call", {"name": "crop_volume_get", "arguments": {"owner": "PointCloud", "role": "roi"}})
            if not response.get("isError"):
                break
            assert time.monotonic() < deadline, response
            time.sleep(.1)
        for role, shape in (("roi", "polygon"), ("crop", "box")):
            restored = _tool(endpoint, "crop_volume_get", {"owner": "PointCloud", "role": role})["volume"]
            assert restored["uuid"] == volumes[role]["uuid"] and restored["shape"] == shape, restored
    finally:
        app.terminate()
        try:
            app.wait(timeout=15)
        except subprocess.TimeoutExpired:
            app.kill()
            app.wait(timeout=15)


@pytest.mark.parametrize("order", [("roi", "crop"), ("crop", "roi")])
@pytest.mark.parametrize("roi_shape", ["box", "polygon"])
@pytest.mark.parametrize("target_shape", ["box", "ellipsoid"])
def test_legacy_shape_command_changes_only_active_crop(order, roi_shape, target_shape):
    dataset = os.environ.get("LFS_REAL_CROP_DATASET")
    if not dataset:
        pytest.skip("Set LFS_REAL_CROP_DATASET to an existing COLMAP dataset")
    root = Path(__file__).resolve().parents[2]
    executable = Path(os.environ.get("LFS_EXECUTABLE", root / "build-tests-release" / "LichtFeld-Studio.exe"))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}/mcp"
    output = Path(tempfile.mkdtemp(prefix="lfs-legacy-role-")).resolve()
    app = subprocess.Popen([str(executable), "--no-download", "--mcp-port", str(port)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        deadline = time.monotonic() + 90
        while True:
            assert app.poll() is None
            try:
                _initialize(endpoint)
                break
            except OSError:
                assert time.monotonic() < deadline
                time.sleep(.1)
        assert _tool(endpoint, "scene_load_dataset", {"path": dataset, "output_path": str(output), "min_track_length": 0})["success"]
        assert _tool(endpoint, "runtime_job_wait", {"job_id": "import.dataset", "until": "inactive", "timeout_ms": 90000})["success"]
        assert _tool(endpoint, "scene_select_node", {"name": "PointCloud"})["success"]
        for role in order:
            assert _tool(endpoint, "ui_tool_invoke", {"tool_id": "builtin." + role})["success"]
        if roi_shape == "polygon":
            assert _tool(endpoint, "crop_volume_set", {"owner": "PointCloud", "role": "roi", "shape": "polygon"})["success"]
        if target_shape == "box":
            assert _tool(endpoint, "crop_volume_set", {"owner": "PointCloud", "role": "crop", "shape": "polygon"})["success"]
        roi = _tool(endpoint, "crop_volume_get", {"owner": "PointCloud", "role": "roi"})["volume"]
        crop = _tool(endpoint, "crop_volume_get", {"owner": "PointCloud", "role": "crop"})["volume"]
        assert _tool(endpoint, "ui_tool_invoke", {"tool_id": "builtin.crop"})["success"]
        command = "add_cropbox" if target_shape == "box" else "add_ellipsoid"
        result = _tool(endpoint, "editor_run", {"code": f"import lichtfeld as lf\nlf.ui.{command}('PointCloud')",
                       "wait_for_completion": True, "show_console": False, "timeout_ms": 30000})
        assert result["success"], result
        after_roi = _tool(endpoint, "crop_volume_get", {"owner": "PointCloud", "role": "roi"})["volume"]
        after_crop = _tool(endpoint, "crop_volume_get", {"owner": "PointCloud", "role": "crop"})["volume"]
        assert after_roi["uuid"] == roi["uuid"] and after_roi["shape"] == roi_shape
        for field in (("min", "max") if roi_shape == "box" else ("vertices", "half_depth")):
            assert after_roi[field] == roi[field]
        assert after_crop["uuid"] == crop["uuid"] and after_crop["shape"] == target_shape
        ui = _call(endpoint, "resources/read", {"uri": "lichtfeld://ui/state"})
        assert json.loads(ui["contents"][0]["text"])["registry_active_tool_id"] == "builtin.crop"
    finally:
        if app.poll() is None:
            app.terminate()
        try:
            app.wait(timeout=15)
        except subprocess.TimeoutExpired:
            app.kill()
            app.wait(timeout=15)
