# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Training keeps the selected pre-training ROI as the same scene entry."""
import json
import time

import pytest

from crop_role_runtime import pretraining_app
from test_render_on_demand_idle import _call, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.mark.parametrize("shape", ["box", "polygon"])
def test_roi_identity_survives_start_and_pause(tmp_path, shape):
    with pretraining_app(tmp_path) as endpoint:
        try:
            assert _tool(endpoint, "ui_tool_invoke", {"tool_id": "builtin.roi"})["success"]
            if shape == "polygon":
                _tool(endpoint, "crop_volume_set", {"owner": "PointCloud", "role": "roi", "shape": "polygon"})
            before = _tool(endpoint, "crop_volume_get", {"owner": "PointCloud", "role": "roi"})["volume"]
            _tool(endpoint, "training_params_set", {"values": {"strategy": "mrnf", "iterations": 100000,
                                                                  "max_cap": 2000, "sh_degree": 0}})
            before_start = _tool(endpoint, "crop_volume_get", {"owner": "PointCloud", "role": "roi"})["volume"]
            assert before_start["visible"] == before["visible"], (before, before_start)
            result = _tool(endpoint, "training_start")
            assert result.get("success", True), result
            deadline = time.monotonic() + 120
            while True:
                state = _tool(endpoint, "training_get_state")
                if state.get("is_running") and state.get("iteration", 0) > 0:
                    break
                assert time.monotonic() < deadline, state
                time.sleep(.1)
            running_nodes = _tool(endpoint, "scene_list_nodes")["nodes"]
            running_roi = [node for node in running_nodes if node["uuid"] == before["uuid"]]
            assert len(running_roi) == 1 and running_roi[0].get("crop_role") == "roi", running_roi
            attempts = []
            if shape == "polygon":
                def attempt_outline():
                    result = _tool(endpoint, "editor_run", {"code": "import json,lichtfeld as lf\n"
                        f"print(json.dumps(lf.ui.set_crop_polygon_outline({before['uuid']!r}, [[-2,-1],[2,-1],[0,3]])))",
                        "wait_for_completion": True, "show_console": False, "timeout_ms": 30000})
                    return json.loads(result["output"]["text"])
                running_before = _tool(endpoint, "crop_volume_get", {"owner": "Model", "role": "roi"})["volume"]
                attempts.append(attempt_outline())
                running_after = _tool(endpoint, "crop_volume_get", {"owner": "Model", "role": "roi"})["volume"]
            assert _tool(endpoint, "runtime_job_control", {"job_id": "training.main", "action": "pause"})["success"]
            deadline = time.monotonic() + 60
            while True:
                state = _tool(endpoint, "training_get_state")
                if state.get("is_paused"):
                    break
                assert time.monotonic() < deadline, state
                time.sleep(.1)
            nodes = _tool(endpoint, "scene_list_nodes")["nodes"]
            restored = [n for n in nodes if n["uuid"] == before["uuid"]]
            assert len(restored) == 1, nodes
            assert restored[0]["name"] == before["name"] and restored[0]["crop_role"] == "roi", restored
            assert restored[0]["type"] == ("crop_box" if shape == "box" else "crop_polygon")
            assert restored[0]["visible"] == before["visible"]
            info = _tool(endpoint, "crop_volume_get", {"owner": "Model", "role": "roi"})["volume"]
            if shape == "polygon":
                attempts.append(attempt_outline())
                paused_after = _tool(endpoint, "crop_volume_get", {"owner": "Model", "role": "roi"})["volume"]
                assert attempts == [False, False], attempts
                assert running_after["vertices"] == running_before["vertices"]
                assert paused_after["vertices"] == info["vertices"]
            assert info["uuid"] == before["uuid"] and info["shape"] == shape
            for field in ("translation", "rotation", "scale"):
                assert info[field] == pytest.approx(before[field])
            if shape == "box":
                assert info["min"] == pytest.approx(before["min"])
                assert info["max"] == pytest.approx(before["max"])
            else:
                assert info["vertices"] == before["vertices"]
                assert info["half_depth"] == pytest.approx(before["half_depth"])
            python = _tool(endpoint, "editor_run", {"code": "import json, lichtfeld as lf\n"
                "print(json.dumps([(n.uuid, n.crop_role) for n in lf.get_scene().get_nodes() if n.crop_role]))",
                "wait_for_completion": True, "timeout_ms": 30000, "show_console": False})
            assert (before["uuid"], "roi") in {tuple(item) for item in json.loads(python["output"]["text"])}
            assert _tool(endpoint, "scene_select_node", {"name": before["name"]})["success"]
            tools = _call(endpoint, "resources/read", {"uri": "lichtfeld://ui/tools"})
            records = json.loads(tools["contents"][0]["text"])["tools"]
            assert not next(t for t in records if t["id"] == "builtin.translate")["poll"]
        finally:
            # Stop training before the app goes away.
            _tool(endpoint, "runtime_job_control", {"job_id": "training.main", "action": "cancel"})
            _tool(endpoint, "runtime_job_wait", {"job_id": "training.main", "until": "inactive", "timeout_ms": 90000})
