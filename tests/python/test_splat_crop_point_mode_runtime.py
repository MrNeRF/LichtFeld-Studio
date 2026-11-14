# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Splat crop filtering in point-cloud display on a real model."""
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import pytest

from test_render_on_demand_idle import _initialize, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def test_splat_crop_filters_point_cloud_view_after_deselection():
    model = os.environ.get("LFS_REAL_SPLAT_MODEL")
    if not model:
        pytest.skip("Set LFS_REAL_SPLAT_MODEL to an existing Gaussian PLY")
    root = Path(__file__).resolve().parents[2]
    executable = Path(os.environ.get("LFS_EXECUTABLE", root / "build-tests-release" / "LichtFeld-Studio.exe"))
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
            time.sleep(.1)
        name = models[0]["name"]
        assert _tool(endpoint, "scene_select_node", {"name": name})["success"]
        assert _tool(endpoint, "render_settings_set", {"point_cloud_mode": True, "desaturate_cropping": False,
                     "show_grid": False, "show_coord_axes": False, "show_camera_frustums": False,
                     "background_color": [0, 0, 0]})["success"]
        focused = _tool(endpoint, "editor_run", {"code": "import lichtfeld as lf\nassert lf.focus_selection()",
                        "wait_for_completion": True, "show_console": False, "timeout_ms": 30000})
        assert focused["success"], focused
        added = _tool(endpoint, "crop_box.add", {"node": name})["crop_box"]
        box = added["node"]
        def set_enabled(enabled):
            return _tool(endpoint, "crop_box.set", {"node": box, "min": [-1, -1, -1], "max": [1, 1, 1],
                         "translation": [1000000, 1000000, 1000000], "enabled": enabled, "show": False})
        def brightness():
            time.sleep(.5)
            result = _tool(endpoint, "editor_run", {"code": "import json,lichtfeld as lf\n"
                "frame=lf.capture_viewport()\nprint(json.dumps(frame.image.sum_scalar() if frame else None))",
                "wait_for_completion": True, "show_console": False, "timeout_ms": 30000})
            assert result["success"], result
            return json.loads(result["output"]["text"])
        assert set_enabled(False)["success"]
        assert _tool(endpoint, "scene_select_node", {"name": name})["success"]
        before = brightness()
        assert set_enabled(True)["success"]
        assert _tool(endpoint, "scene_select_node", {"name": name})["success"]
        cropped = brightness()
        assert before is not None and cropped is not None and before - cropped > 1.0, (before, cropped)
    finally:
        if app.poll() is None:
            app.terminate()
        try:
            app.wait(timeout=15)
        except subprocess.TimeoutExpired:
            app.kill()
            app.wait(timeout=15)
