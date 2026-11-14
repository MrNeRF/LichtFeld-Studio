# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise both crop toolbar workflows in the real pre-training application."""
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

from test_mcp_dataset_load_training import _dataset
from test_render_on_demand_idle import _call, _initialize, _tool


@pytest.mark.parametrize("order", [
    ("builtin.cropbox", "builtin.crop"),
    ("builtin.crop", "builtin.cropbox"),
])
def test_crop_toolbar_roles_keep_independent_shapes_and_outlines(order):
    root = Path(__file__).resolve().parents[2]
    build = Path(os.environ.get("LFS_TEST_BUILD_DIR", root / "build"))
    executable = Path(os.environ.get("LFS_EXECUTABLE", build / "LichtFeld-Studio.exe"))
    assert executable.is_file()
    dataset = _dataset(Path(tempfile.mkdtemp(prefix="lfs-crop-roles-")) / "dataset", 80)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}/mcp"
    app = subprocess.Popen(
        [str(executable), "--no-download", "--mcp-port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        deadline = time.monotonic() + 90
        while True:
            assert app.poll() is None, f"application exited: {app.returncode}"
            try:
                _initialize(endpoint)
                break
            except OSError:
                assert time.monotonic() < deadline, "MCP startup timed out"
                time.sleep(0.1)

        assert _tool(endpoint, "scene_load_dataset", {"path": dataset, "min_track_length": 0})["success"]
        wait = _tool(endpoint, "runtime_job_wait", {"job_id": "import.dataset", "until": "inactive", "timeout_ms": 20000})
        assert wait["success"]
        assert _tool(endpoint, "scene_select_node", {"name": "PointCloud"})["success"]

        def dispatch(kind, value):
            code = ("from lfs_plugins.toolbar import _GizmoToolbarController\n"
                    f"_GizmoToolbarController().dispatch({kind!r}, {value!r})\n"
                    "print('done')")
            result = _tool(endpoint, "editor_run", {"code": code, "wait_for_completion": True,
                                                     "timeout_ms": 15000, "show_console": False})
            assert result["success"], result

        def crops():
            result = _call(endpoint, "resources/read", {"uri": "lichtfeld://scene/nodes"})
            nodes = json.loads(result["contents"][0]["text"])["nodes"]
            return [node for node in nodes if node["type"] in ("crop_box", "ellipsoid", "crop_polygon")]

        dispatch("tool", order[0])
        assert len(crops()) == 1
        dispatch("crop_object", "ellipsoid" if order[0] == "builtin.crop" else "polygon")
        first = crops()[0]
        assert first["type"] == ("ellipsoid" if order[0] == "builtin.crop" else "crop_polygon")
        dispatch("tool", order[1])
        dispatch("crop_object", "ellipsoid" if order[1] == "builtin.crop" else "polygon")
        volumes = crops()
        assert len(volumes) == 2
        assert {node["type"] for node in volumes} == {"crop_polygon", "ellipsoid"}
        assert all(node["visible"] for node in volumes)
        identities = {node["uuid"]: node["type"] for node in volumes}
        for tool_id in order:
            dispatch("tool", tool_id)
            volumes = crops()
            assert {node["uuid"]: node["type"] for node in volumes} == identities
            assert all(node["visible"] for node in volumes)
        removed = next(node for node in volumes if node["uuid"] == first["uuid"])
        sibling = next(node for node in volumes if node["uuid"] != first["uuid"])
        result = _tool(endpoint, "editor_run", {
            "code": "import lichtfeld as lf\n" + f"lf.get_scene().remove_node({removed['name']!r})\nprint('done')",
            "wait_for_completion": True, "timeout_ms": 15000, "show_console": False,
        })
        assert result["success"], result
        remaining = crops()
        assert [(node["uuid"], node["type"]) for node in remaining] == [(sibling["uuid"], sibling["type"])]
    finally:
        app.terminate()
        try:
            app.wait(timeout=15)
        except subprocess.TimeoutExpired:
            app.kill()
            app.wait(timeout=15)


def test_existing_splat_crop_workflow_still_creates_new_instances():
    from test_mcp_ply_sh_degree import _model

    root = Path(__file__).resolve().parents[2]
    build = Path(os.environ.get("LFS_TEST_BUILD_DIR", root / "build"))
    executable = Path(os.environ.get("LFS_EXECUTABLE", build / "LichtFeld-Studio.exe"))
    directory = Path(tempfile.mkdtemp(prefix="lfs-crop-splat-"))
    model = directory / "model.ply"
    _model(model, 0)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}/mcp"
    app = subprocess.Popen(
        [str(executable), "--no-download", "--mcp-port", str(port), "-v", str(model)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        deadline = time.monotonic() + 90
        while True:
            assert app.poll() is None, f"application exited: {app.returncode}"
            try:
                _initialize(endpoint)
                nodes = _tool(endpoint, "scene_list_nodes")["nodes"]
                models = [node for node in nodes if node["type"] == "splat"]
                if models:
                    break
            except OSError:
                pass
            assert time.monotonic() < deadline, "splat did not load"
            time.sleep(0.1)
        assert _tool(endpoint, "scene_select_node", {"name": models[0]["name"]})["success"]
        for expected in (1, 2):
            result = _tool(endpoint, "editor_run", {
                "code": "import lichtfeld as lf\nlf.ui.new_crop_tool('box')\nprint('done')",
                "wait_for_completion": True, "timeout_ms": 15000, "show_console": False,
            })
            assert result["success"], result
            nodes = _tool(endpoint, "scene_list_nodes")["nodes"]
            boxes = [node for node in nodes if node["type"] == "crop_box"]
            assert len(boxes) == expected, boxes
    finally:
        app.terminate()
        try:
            app.wait(timeout=15)
        except subprocess.TimeoutExpired:
            app.kill()
            app.wait(timeout=15)


def test_crop_delete_undo_keeps_restored_size_and_active_role():
    """Reproduce a trimmed primary and reset secondary through the live toolbar."""
    root = Path(__file__).resolve().parents[2]
    executable = Path(os.environ.get("LFS_EXECUTABLE", root / "build" / "LichtFeld-Studio.exe"))
    dataset = _dataset(Path(tempfile.mkdtemp(prefix="lfs-crop-undo-")) / "dataset", 80)
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
        assert _tool(endpoint, "scene_load_dataset", {"path": dataset, "min_track_length": 0})["success"]
        assert _tool(endpoint, "runtime_job_wait", {"job_id": "import.dataset", "until": "inactive", "timeout_ms": 20000})["success"]
        assert _tool(endpoint, "scene_select_node", {"name": "PointCloud"})["success"]

        def dispatch(action, value=""):
            code = ("from lfs_plugins.toolbar import _GizmoToolbarController\n"
                    f"_GizmoToolbarController().dispatch({action!r}, {value!r})\nprint('done')")
            result = _tool(endpoint, "editor_run", {"code": code, "wait_for_completion": True,
                                                     "timeout_ms": 15000, "show_console": False})
            assert result["success"], result

        def resource(uri):
            return json.loads(_call(endpoint, "resources/read", {"uri": uri})["contents"][0]["text"])

        def crops():
            return [n for n in resource("lichtfeld://scene/nodes")["nodes"] if n["type"] == "crop_box"]

        def bounds(name):
            value = _tool(endpoint, "crop_box_get", {"node": name})
            return value

        dispatch("tool", "builtin.cropbox")
        primary = crops()[0]
        dispatch("tool", "builtin.crop")
        secondary = next(n for n in crops() if n["uuid"] != primary["uuid"])
        dispatch("tool", "builtin.cropbox")
        dispatch("crop_trim")
        primary_before = bounds(primary["name"])
        dispatch("tool", "builtin.crop")
        dispatch("crop_reset")
        secondary_before = bounds(secondary["name"])
        assert primary_before != secondary_before, (primary_before, secondary_before)
        dispatch("crop_delete")
        assert len(crops()) == 1
        dispatch("tool", "builtin.cropbox")
        assert _tool(endpoint, "history_undo")["success"]
        restored = next(n for n in crops() if n["uuid"] == secondary["uuid"])
        assert bounds(primary["name"]) == primary_before
        assert bounds(restored["name"]) == secondary_before
        selected = resource("lichtfeld://scene/selected_nodes")["nodes"]
        state = resource("lichtfeld://ui/state")
        assert selected[0]["uuid"] == secondary["uuid"]
        assert state["registry_active_tool_id"] == "builtin.crop", (selected, state)
        def scale_selected(name, value):
            dispatch("crop_transform", "scale")
            code = ("import lfs_plugins.toolbar as toolbar\n"
                    "toolbar._toolbar_controller._sync_tool_overlays_now()\n"
                    "ctrl = toolbar._toolbar_controller._transform_controls\n"
                    f"assert ctrl._selected == [{name!r}], ctrl._selected\n"
                    "assert ctrl._active_tool == 'builtin.scale', ctrl._active_tool\n"
                    f"ctrl._set_uniform_scale({str(value)!r})\n"
                    "ctrl._commit_active_edit()\nprint('scaled')")
            result = _tool(endpoint, "editor_run", {"code": code, "wait_for_completion": True,
                                                     "timeout_ms": 15000, "show_console": False})
            assert result["success"] and "scaled" in result["output"]["text"], result

        scale_selected(secondary["name"], 1.5)
        secondary_scaled = bounds(secondary["name"])
        assert secondary_scaled != secondary_before
        assert bounds(primary["name"]) == primary_before
        dispatch("tool", "builtin.cropbox")
        scale_selected(primary["name"], 0.75)
        primary_scaled = bounds(primary["name"])
        assert primary_scaled != primary_before
        assert bounds(secondary["name"]) == secondary_scaled

        # Repeat deletion in the other direction after both tools have been edited.
        dispatch("crop_delete")
        assert len(crops()) == 1
        dispatch("tool", "builtin.crop")
        assert _tool(endpoint, "history_undo")["success"]
        restored_primary = next(n for n in crops() if n["uuid"] == primary["uuid"])
        assert bounds(restored_primary["name"]) == primary_scaled
        assert bounds(secondary["name"]) == secondary_scaled
        assert resource("lichtfeld://scene/selected_nodes")["nodes"][0]["uuid"] == primary["uuid"]
        assert resource("lichtfeld://ui/state")["registry_active_tool_id"] == "builtin.roi"
    finally:
        app.terminate()
        try:
            app.wait(timeout=15)
        except subprocess.TimeoutExpired:
            app.kill()
            app.wait(timeout=15)

@pytest.mark.parametrize("order", [("builtin.cropbox", "builtin.crop"), ("builtin.crop", "builtin.cropbox")])
def test_roi_settings_follow_crop_role_and_close_when_switching(order):
    """The real viewport toolbar exposes ROI settings only for the primary role."""
    root = Path(__file__).resolve().parents[2]
    executable = Path(os.environ.get("LFS_EXECUTABLE", root / "build" / "LichtFeld-Studio.exe"))
    dataset = _dataset(Path(tempfile.mkdtemp(prefix="lfs-roi-gear-")) / "dataset", 80)
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
        assert _tool(endpoint, "scene_load_dataset", {"path": dataset, "min_track_length": 0})["success"]
        assert _tool(endpoint, "runtime_job_wait", {"job_id": "import.dataset", "until": "inactive", "timeout_ms": 20000})["success"]
        assert _tool(endpoint, "scene_select_node", {"name": "PointCloud"})["success"]

        def run(code):
            result = _tool(endpoint, "editor_run", {"code": code, "wait_for_completion": True,
                                                     "timeout_ms": 15000, "show_console": False})
            assert result["success"], result
            output = result["output"]["text"]
            assert "Traceback" not in output, output
            return output

        def dispatch(tool):
            run("from lfs_plugins.toolbar import _GizmoToolbarController\n"
                f"_GizmoToolbarController().dispatch('tool', {tool!r})\nprint('done')")

        def state():
            result = run("import json, lfs_plugins.toolbar as toolbar\n"
                         "c=toolbar._toolbar_controller\n"
                         "c._last_toolbar_signature=None\n"
                         "c._sync_toolbar_state()\n"
                         "print(json.dumps({'buttons': c._gizmo.snapshot()['crop_settings_buttons'], "
                         "'open': c._crop_roi_settings_open}))")
            return json.loads(result)

        def selected_name():
            value = _call(endpoint, "resources/read", {"uri": "lichtfeld://scene/selected_nodes"})
            return json.loads(value["contents"][0]["text"])["nodes"][0]["name"]

        dispatch(order[0])
        names = {order[0]: selected_name()}
        dispatch(order[1])
        names[order[1]] = selected_name()
        if order[1] != "builtin.cropbox":
            dispatch("builtin.cropbox")
        assert len(state()["buttons"]) == 1
        run("import lfs_plugins.toolbar as toolbar\n"
            "toolbar._toolbar_controller._on_toolbar_action(None, None, ['toggle_crop_roi_settings', ''])\n"
            "print('opened')")
        assert state()["open"] is True
        dispatch("builtin.crop")
        after = state()
        assert after["buttons"] == [], after
        assert after["open"] is False, after
        dispatch("builtin.cropbox")
        assert len(state()["buttons"]) == 1

        # Scene selection without an active crop tool must use the stored role.
        assert _tool(endpoint, "scene_select_node", {"name": names["builtin.crop"]})["success"]
        assert state()["buttons"] == []
        assert _tool(endpoint, "scene_select_node", {"name": names["builtin.cropbox"]})["success"]
        assert len(state()["buttons"]) == 1
        assert _tool(endpoint, "scene_select_node", {"name": "PointCloud"})["success"]
        assert len(state()["buttons"]) == 1

        dispatch("builtin.crop")
        run("from lfs_plugins.toolbar import _GizmoToolbarController\n"
            "_GizmoToolbarController().dispatch('crop_object', 'polygon')\nprint('done')")
        assert state()["buttons"] == []
        dispatch("builtin.cropbox")
        run("from lfs_plugins.toolbar import _GizmoToolbarController\n"
            "_GizmoToolbarController().dispatch('crop_object', 'polygon')\nprint('done')")
        assert len(state()["buttons"]) == 1
    finally:
        app.terminate()
        try:
            app.wait(timeout=15)
        except subprocess.TimeoutExpired:
            app.kill()
            app.wait(timeout=15)