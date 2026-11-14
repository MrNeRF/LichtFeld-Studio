# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Check ROI naming and compatibility against the running application."""
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


@pytest.mark.parametrize("order", [("builtin.roi", "builtin.crop"), ("builtin.crop", "builtin.roi")])
def test_roi_and_crop_names_and_legacy_activation(order):
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
                time.sleep(0.1)
        output = tempfile.mkdtemp(prefix="lfs-roi-naming-")
        assert _tool(endpoint, "scene_load_dataset", {"path": dataset, "output_path": output,
                                                       "min_track_length": 0})["success"]
        assert _tool(endpoint, "runtime_job_wait", {"job_id": "import.dataset", "until": "inactive",
                                                     "timeout_ms": 90000})["success"]
        assert _tool(endpoint, "scene_select_node", {"name": "PointCloud"})["success"]
        def run(code):
            result = _tool(endpoint, "editor_run", {"code": code, "wait_for_completion": True,
                                                     "timeout_ms": 30000, "show_console": False})
            output = result["output"]["text"]
            assert result["success"] and "Traceback" not in output, result
            return output
        toolbar = json.loads(run("import json\nfrom lfs_plugins.toolbar import _GizmoToolbarController\n"
                                 "s=_GizmoToolbarController().snapshot()\n"
                                 "print(json.dumps({'primary': s['crop_group_buttons'], 'other': s['gizmo_buttons']}))"))
        assert toolbar["primary"][0]["value"] == "builtin.roi", toolbar
        assert toolbar["primary"][0]["icon_src"].endswith("/roi.png"), toolbar
        crop_button = next(button for button in toolbar["other"] if button["value"] == "builtin.crop")
        assert crop_button["icon_src"].endswith("/cropbox.png"), toolbar
        tool_data = _call(endpoint, "resources/read", {"uri": "lichtfeld://ui/tools"})
        tools = {item["id"]: item for item in json.loads(tool_data["contents"][0]["text"])["tools"]}
        assert tools["builtin.roi"]["icon"] == "roi", tools["builtin.roi"]
        assert tools["builtin.crop"]["icon"] == "cropbox", tools["builtin.crop"]
        assert tools["builtin.cropbox"]["icon"] == "cropbox", tools["builtin.cropbox"]
        assert any(button["value"] == "builtin.crop" for button in toolbar["other"]), toolbar
        # A direct Python request must not create an unsupported ROI ellipsoid.
        assert run("import json, lichtfeld as lf\nlf.ui.set_active_operator('builtin.roi', 'translate')\n"
                   "print(json.dumps(lf.ui.new_crop_tool('ellipsoid')))").strip() == "null"
        resource = _call(endpoint, "resources/read", {"uri": "lichtfeld://scene/nodes"})
        assert not [n for n in json.loads(resource["contents"][0]["text"])["nodes"]
                    if n["type"] in ("crop_box", "ellipsoid", "crop_polygon")]
        run("import lichtfeld as lf\nlf.ui.add_ellipsoid('PointCloud')")
        resource = _call(endpoint, "resources/read", {"uri": "lichtfeld://scene/nodes"})
        assert not [n for n in json.loads(resource["contents"][0]["text"])["nodes"]
                    if n["type"] in ("crop_box", "ellipsoid", "crop_polygon")]
        run("import lichtfeld as lf\nlf.ui.clear_active_operator()")
        for tool in order:
            run("from lfs_plugins.toolbar import _GizmoToolbarController\n"
                f"_GizmoToolbarController().dispatch('tool', {tool!r})\n")
        result = _call(endpoint, "resources/read", {"uri": "lichtfeld://scene/nodes"})
        nodes = json.loads(result["contents"][0]["text"])["nodes"]
        volumes = [node for node in nodes if node["type"] == "crop_box"]
        assert len(volumes) == 2
        assert any("ROI" in n["name"] for n in volumes), volumes
        assert any("Crop" in n["name"] for n in volumes), volumes
        for tool in ("builtin.roi", "builtin.crop"):
            run("from lfs_plugins.toolbar import _GizmoToolbarController\n"
                f"_GizmoToolbarController().dispatch('tool', {tool!r})")
            state = json.loads(run("import json, lichtfeld as lf\n"
                "from lfs_plugins.toolbar import _GizmoToolbarController\n"
                "s=_GizmoToolbarController().snapshot()\n"
                "print(json.dumps({'shapes': [b['value'] for b in s['crop_object_buttons']], "
                "'changed': lf.ui.change_crop_tool_shape('ellipsoid')}))"))
            assert ("ellipsoid" in state["shapes"]) == (tool == "builtin.crop"), state
            assert state["changed"] == (tool == "builtin.crop"), state
            if tool == "builtin.crop":
                assert run("import lichtfeld as lf\nprint(lf.ui.change_crop_tool_shape('box'))").strip() == "True"
        result = run("import lichtfeld as lf\nlf.ui.set_active_operator('builtin.cropbox', 'translate')\n"
                     "print(lf.ui.get_active_tool())")
        assert result.strip() == "builtin.roi"
    finally:
        app.terminate()
        try:
            app.wait(timeout=15)
        except subprocess.TimeoutExpired:
            app.kill()
            app.wait(timeout=15)


def test_splat_scene_keeps_legacy_crop_identity():
    model = os.environ.get("LFS_REAL_SPLAT_MODEL")
    if not model:
        pytest.skip("Set LFS_REAL_SPLAT_MODEL to an existing Gaussian PLY")
    root = Path(__file__).resolve().parents[2]
    executable = Path(os.environ.get("LFS_EXECUTABLE", root / "build" / "LichtFeld-Studio.exe"))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}/mcp"
    app = subprocess.Popen([str(executable), "--no-download", "--mcp-port", str(port), "-v", model],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        deadline = time.monotonic() + 90
        while True:
            assert app.poll() is None
            try:
                _initialize(endpoint)
                nodes = _tool(endpoint, "scene_list_nodes")["nodes"]
                models = [node for node in nodes if node["type"] == "splat"]
                if models:
                    break
            except OSError:
                pass
            assert time.monotonic() < deadline
            time.sleep(0.1)
        assert _tool(endpoint, "scene_select_node", {"name": models[0]["name"]})["success"]
        result = _tool(endpoint, "editor_run", {
            "code": "import json, lichtfeld as lf\nfrom lfs_plugins.toolbar import _GizmoToolbarController\n"
                    "lf.ui.set_active_operator('builtin.cropbox', 'translate')\n"
                    "s=_GizmoToolbarController().snapshot()\n"
                    "print(json.dumps({'active': lf.ui.get_active_tool(), 'primary': s['crop_group_buttons'], 'other': s['gizmo_buttons']}))",
            "wait_for_completion": True, "timeout_ms": 30000, "show_console": False,
        })
        assert result["success"], result
        state = json.loads(result["output"]["text"])
        assert state["active"] == "builtin.cropbox", state
        assert state["primary"][0]["value"] == "builtin.cropbox", state
        assert all(button["value"] not in ("builtin.roi", "builtin.crop") for button in state["other"]), state
        result = _tool(endpoint, "editor_run", {
            "code": "import lichtfeld as lf\nlf.ui.new_crop_tool('ellipsoid')",
            "wait_for_completion": True, "timeout_ms": 30000, "show_console": False,
        })
        assert result["success"], result
        nodes = _tool(endpoint, "scene_list_nodes")["nodes"]
        assert any(node["type"] == "ellipsoid" for node in nodes), nodes
    finally:
        app.terminate()
        try:
            app.wait(timeout=15)
        except subprocess.TimeoutExpired:
            app.kill()
            app.wait(timeout=15)


def test_existing_roi_ellipsoid_survives_project_roundtrip_and_can_convert():
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
                time.sleep(0.1)
        directory = Path(tempfile.mkdtemp(prefix="lfs-legacy-roi-")).resolve()
        assert _tool(endpoint, "scene_load_dataset", {"path": dataset, "output_path": str(directory),
                                                       "min_track_length": 0})["success"]
        assert _tool(endpoint, "runtime_job_wait", {"job_id": "import.dataset", "until": "inactive",
                                                     "timeout_ms": 90000})["success"]
        assert _tool(endpoint, "scene_select_node", {"name": "PointCloud"})["success"]
        def run(code):
            result = _tool(endpoint, "editor_run", {"code": code, "wait_for_completion": True,
                                                     "timeout_ms": 30000, "show_console": False})
            output = result["output"]["text"]
            assert result["success"] and "Traceback" not in output, result
            return output
        def ellipsoids():
            result = _call(endpoint, "resources/read", {"uri": "lichtfeld://scene/nodes"})
            return [node for node in json.loads(result["contents"][0]["text"])["nodes"]
                    if node["type"] == "ellipsoid"]
        # The legacy generic scene command constructs an old primary ellipsoid.
        run("import lichtfeld as lf\nlf.ui.set_active_tool('builtin.select')\nlf.ui.add_ellipsoid('PointCloud')")
        before = ellipsoids()
        assert len(before) == 1, before
        project = directory / "legacy-roi.licht"
        saved = _tool(endpoint, "project_save_as", {"path": str(project)})
        assert saved.get("success", True), saved
        opened = _tool(endpoint, "project_open", {"path": str(project), "discard_changes": True})
        assert opened.get("success", True), opened
        deadline = time.monotonic() + 30
        while not ellipsoids() and time.monotonic() < deadline:
            time.sleep(.1)
        restored = ellipsoids()
        assert len(restored) == 1 and restored[0]["uuid"] == before[0]["uuid"], restored
        assert _tool(endpoint, "scene_select_node", {"name": restored[0]["name"]})["success"]
        run("import lichtfeld as lf\nfrom lfs_plugins.toolbar import _GizmoToolbarController\n"
            "lf.ui.clear_active_operator()\n_GizmoToolbarController().dispatch('tool', 'builtin.roi')")
        result = json.loads(run("import json, lichtfeld as lf\n"
                                "print(json.dumps({'shape':lf.ui.get_crop_tool_shape(), "
                                "'converted':lf.ui.change_crop_tool_shape('box')}))"))
        assert result == {"shape": "ellipsoid", "converted": True}, result
        nodes = _tool(endpoint, "scene_list_nodes")["nodes"]
        assert any(n["uuid"] == before[0]["uuid"] and n["type"] == "crop_box" for n in nodes)
    finally:
        app.terminate()
        try:
            app.wait(timeout=15)
        except subprocess.TimeoutExpired:
            app.kill()
            app.wait(timeout=15)
