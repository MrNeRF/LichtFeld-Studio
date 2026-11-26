# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Real renderer regression for completed training cropped to its ROI."""
import json
import re
import time

import pytest

from crop_role_runtime import pretraining_app
from test_render_on_demand_idle import _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.mark.parametrize("strategy,crop", [("mrnf", True), ("mcmc", True), ("mrnf", False)])
@pytest.mark.parametrize("shape", ["box", "polygon"])
@pytest.mark.parametrize("phase", ["complete", "edit"])
def test_completed_training_edit_viewport_renders(tmp_path, strategy, crop, shape, phase):
    log = tmp_path / "app.log"
    with pretraining_app(tmp_path) as endpoint:
        assert _tool(endpoint, "ui_tool_invoke", {"tool_id": "builtin.roi"})["success"]
        if shape == "polygon":
            assert _tool(endpoint, "crop_volume_set", {"owner": "PointCloud", "role": "roi", "shape": "polygon"})["success"]
        _tool(endpoint, "crop_volume_action", {"owner": "PointCloud", "role": "roi", "action": "fit", "use_percentile": True})
        before = _tool(endpoint, "crop_volume_get", {"owner": "PointCloud", "role": "roi"})["volume"]
        params = _tool(endpoint, "training_params_set", {"values": {"strategy": strategy, "iterations": 10,
                         "max_cap": 2000, "sh_degree": 0, "crop_to_cropbox": crop}})
        assert params["success"], params
        assert before["shape"] == shape
        _tool(endpoint, "training_start")
        deadline = time.monotonic() + 120
        while True:
            job = _tool(endpoint, "runtime_job_describe", {"job_id": "training.main"})
            if job.get("status") in ("finished", "failed", "cancelled"):
                break
            assert time.monotonic() < deadline, job
            time.sleep(.1)
        assert job["status"] == "finished" and job["success"], job
        state = _tool(endpoint, "training_get_state")
        assert state["iteration"] == 10, state
        if crop:
            text = log.read_text(encoding="utf8", errors="replace")
            match = re.search(r"Cropped the trained model to its crop (?:box|polygon): (\d+) of (\d+) splats kept", text)
            assert match and 0 < int(match[1]) < int(match[2]), match
        if not crop:
            assert "Cropped the trained model to its crop" not in log.read_text(encoding="utf8", errors="replace")
        if phase == "complete":
            complete_nodes = _tool(endpoint, "scene_list_nodes")["nodes"]
            roi = [node for node in complete_nodes if node["uuid"] == before["uuid"]]
            assert len(roi) == 1 and roi[0].get("crop_role") == "roi", roi
            complete = _tool(endpoint, "crop_volume_get", {"owner": "Model", "role": "roi"})["volume"]
            assert complete["uuid"] == before["uuid"] and complete["shape"] == shape
            if crop:
                model = next(node for node in complete_nodes if node["name"] == "Model")
                assert model["gaussian_count"] == int(match[1]), (model, match[0])
            return
        count_result = _tool(endpoint, "editor_run", {"code": "import lichtfeld as lf\n"
                             "print(next(n for n in lf.get_scene().get_nodes() if n.name=='Model').splat_data().num_points)",
                             "wait_for_completion": True, "show_console": False, "timeout_ms": 30000})
        assert count_result["success"], count_result
        trained_count = int(match[1]) if crop else int(count_result["output"]["text"].strip())
        result = _tool(endpoint, "editor_run", {"code": "import lichtfeld as lf\nlf.switch_to_edit_mode()",
                        "wait_for_completion": True, "show_console": False, "timeout_ms": 30000})
        assert result["success"], result
        deadline = time.monotonic() + 30
        while True:
            nodes = _tool(endpoint, "scene_list_nodes")["nodes"]
            models = [node for node in nodes if node["name"] == "Trained Model" and node["type"] == "splat"]
            if models:
                break
            assert time.monotonic() < deadline, nodes
            time.sleep(.1)
        assert models[0]["gaussian_count"] == trained_count and trained_count > 0, models
        assert not any(node["uuid"] == before["uuid"] for node in nodes), nodes
        assert not any(node["type"] in ("crop_box", "crop_polygon", "ellipsoid") and
                       node.get("parent") == "Trained Model" for node in nodes), nodes
        _tool(endpoint, "render_settings_set", {"show_grid": False, "show_camera_frustums": False, "show_coord_axes": False, "background_color": [0, 0, 0]})
        focused = _tool(endpoint, "editor_run", {"code": "import lichtfeld as lf\nassert lf.focus_selection()",
                        "wait_for_completion": True, "show_console": False, "timeout_ms": 30000})
        assert focused["success"], focused
        time.sleep(1)
        text = log.read_text(encoding="utf8", errors="replace")
        assert "VkSplat refusing full input-copy fallback" not in text, "Renderer rejected model storage after crop"
        camera_result = _tool(endpoint, "editor_run", {"code": "import lichtfeld as lf, json\n"
            "c=lf.get_camera()\nprint(json.dumps({'eye':c.eye,'target':c.target,'up':c.up}))",
            "wait_for_completion": True, "show_console": False, "timeout_ms": 30000})
        camera = json.loads(camera_result["output"]["text"])
        brightness = []
        for distance in (1., .25, .1, .04):
            eye = [target + (position - target) * distance for position, target in zip(camera["eye"], camera["target"])]
            set_camera = _tool(endpoint, "editor_run", {"code": f"import lichtfeld as lf\nlf.set_camera({tuple(eye)!r}, {tuple(camera['target'])!r}, {tuple(camera['up'])!r})",
                "wait_for_completion": True, "show_console": False, "timeout_ms": 30000})
            assert set_camera["success"], set_camera
            captured = _tool(endpoint, "editor_run", {"code": "import lichtfeld as lf, json\n"
                "frame=lf.capture_viewport()\nprint(json.dumps({'max':frame.image.max_scalar()} if frame else None))",
                "wait_for_completion": True, "show_console": False, "timeout_ms": 30000})
            assert captured["success"], captured
            info = json.loads(captured["output"]["text"])
            assert info is not None, captured
            brightness.append(info["max"])
            if brightness[-1] > .03:
                break
        assert max(brightness) > .03, (brightness, camera)
        text = log.read_text(encoding="utf8", errors="replace")
        assert "VkSplat refusing full input-copy fallback" not in text, text[-8000:]
