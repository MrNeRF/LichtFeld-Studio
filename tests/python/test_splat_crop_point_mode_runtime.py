# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Splat crop filtering in point-cloud display."""
import json
import time

import pytest

from crop_role_runtime import grid_model
from runtime_test_app import isolated_app
from test_render_on_demand_idle import _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def test_splat_crop_filters_point_cloud_view_after_deselection(tmp_path):
    model = tmp_path / "grid.ply"
    grid_model(model)
    with isolated_app(tmp_path) as endpoint:
        assert _tool(endpoint, "scene_load_ply", {"path": str(model)})["success"]
        assert _tool(endpoint, "runtime_job_wait", {"job_id": "import.dataset", "until": "inactive",
                                                     "timeout_ms": 60000})["success"]
        name = next(node["name"] for node in _tool(endpoint, "scene_list_nodes")["nodes"] if node["type"] == "splat")
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
