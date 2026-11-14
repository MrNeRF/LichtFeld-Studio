# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise crop application against a real COLMAP dataset in the live application."""
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


@pytest.mark.parametrize("order", [("builtin.cropbox", "builtin.crop"), ("builtin.crop", "builtin.cropbox")])
@pytest.mark.parametrize("shape", ["box", "ellipsoid", "polygon"])
@pytest.mark.parametrize("method", ["toolbar", "api"])
def test_apply_crop_preserves_roi_and_undo_restores_points(order, shape, method):
    dataset = os.environ.get("LFS_REAL_CROP_DATASET")
    if not dataset:
        pytest.skip("Set LFS_REAL_CROP_DATASET to an existing COLMAP dataset")
    assert Path(dataset).is_dir()
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
        output = tempfile.mkdtemp(prefix="lfs-real-crop-output-")
        assert _tool(endpoint, "scene_load_dataset", {"path": dataset, "output_path": output,
                                                       "min_track_length": 0})["success"]
        result = _tool(endpoint, "runtime_job_wait", {"job_id": "import.dataset", "until": "inactive",
                                                       "timeout_ms": 90000})
        assert result["success"] and result["outcome"] == "completed", result
        assert _tool(endpoint, "scene_select_node", {"name": "PointCloud"})["success"]

        def run(code):
            result = _tool(endpoint, "editor_run", {"code": code, "wait_for_completion": True,
                                                     "timeout_ms": 30000, "show_console": False})
            output = result["output"]["text"]
            assert result["success"] and "Traceback" not in output, result
            return output

        def dispatch(kind, value=""):
            run("from lfs_plugins.toolbar import _GizmoToolbarController\n"
                f"_GizmoToolbarController().dispatch({kind!r}, {value!r})\nprint('done')")

        def nodes():
            resource = _call(endpoint, "resources/read", {"uri": "lichtfeld://scene/nodes"})
            return json.loads(resource["contents"][0]["text"])["nodes"]

        def selected():
            resource = _call(endpoint, "resources/read", {"uri": "lichtfeld://scene/selected_nodes"})
            return json.loads(resource["contents"][0]["text"])["nodes"][0]

        def roi_state(name):
            code = ("import json, lichtfeld as lf\n"
                    f"n=lf.get_scene().get_node({name!r})\n"
                    "v=n.cropbox() or n.crop_polygon() or n.ellipsoid()\n"
                    "print(json.dumps({'transform': lf.get_node_transform(n.name), "
                    "'enabled': bool(v.enabled), 'inverse': bool(v.inverse), "
                    "'vertices': list(v.vertices) if n.crop_polygon() else None, "
                    "'min': list(v.min) if n.cropbox() else None, "
                    "'max': list(v.max) if n.cropbox() else None}))")
            return json.loads(run(code))

        def point_digest():
            code = ("import hashlib, lichtfeld as lf\n"
                    "cloud=lf.get_scene().get_node('PointCloud').point_cloud()\n"
                    "h=hashlib.sha256()\n"
                    "h.update(cloud.means.cpu().numpy().tobytes())\n"
                    "h.update(cloud.colors.cpu().numpy().tobytes())\n"
                    "print(h.hexdigest())")
            return run(code).strip()
        names = {}
        uuids = {}
        for tool in order:
            dispatch("tool", tool)
            if shape == "polygon":
                dispatch("crop_object", "polygon")
            elif tool == "builtin.crop" and shape == "ellipsoid":
                dispatch("crop_object", "ellipsoid")
            node = selected()
            names[tool] = node["name"]
            uuids[tool] = node["uuid"]
            if tool == "builtin.cropbox":
                dispatch("crop_finish")
                assert any(n["uuid"] == uuids[tool] for n in nodes())
        if order[-1] != "builtin.crop":
            dispatch("tool", "builtin.crop")
        dispatch("crop_trim")
        before_roi = roi_state(names["builtin.cropbox"])
        before_count = next(n["gaussian_count"] for n in nodes() if n["name"] == "PointCloud")
        assert before_count > 0
        before_digest = point_digest()
        if method == "toolbar":
            actions = json.loads(run("import json\nfrom lfs_plugins.toolbar import _GizmoToolbarController\n"
                                     "print(json.dumps(_GizmoToolbarController().snapshot()['crop_action_buttons']))"))
            assert next(item for item in actions if item["action"] == "crop_finish")["tooltip_key"] == "common.apply"
            dispatch("crop_finish")
        else:
            run("import lichtfeld as lf\nlf.ui.apply_crop_tool()\nprint('applied')")
        after_nodes = nodes()
        after_count = next(n["gaussian_count"] for n in after_nodes if n["name"] == "PointCloud")
        assert 0 < after_count < before_count, (before_count, after_count, shape)
        after_digest = point_digest()
        assert after_digest != before_digest
        assert roi_state(names["builtin.cropbox"]) == before_roi
        assert all(n["uuid"] != uuids["builtin.crop"] for n in after_nodes)
        assert _tool(endpoint, "history_undo")["success"]
        undone = nodes()
        assert next(n["gaussian_count"] for n in undone if n["name"] == "PointCloud") == before_count
        assert point_digest() == before_digest
        assert any(n["uuid"] == uuids["builtin.crop"] for n in undone)
        assert roi_state(names["builtin.cropbox"]) == before_roi
        assert _tool(endpoint, "history_redo")["success"]
        redone = nodes()
        assert next(n["gaussian_count"] for n in redone if n["name"] == "PointCloud") == after_count
        assert point_digest() == after_digest
        assert roi_state(names["builtin.cropbox"]) == before_roi
    finally:
        app.terminate()
        try:
            app.wait(timeout=15)
        except subprocess.TimeoutExpired:
            app.kill()
            app.wait(timeout=15)
