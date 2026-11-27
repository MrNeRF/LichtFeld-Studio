# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Check ROI naming and compatibility against the running application."""
import json
import time

import pytest

from crop_role_runtime import crops, dispatch, grid_model, pretraining_app, run, selected, toolbar, volume
from runtime_test_app import isolated_app
from test_render_on_demand_idle import _call, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _pick_shape(endpoint, shape):
    """The toolbar's shape buttons: the tool shape, then a volume of that shape for the selection."""
    command = {"box": "add_cropbox", "ellipsoid": "add_ellipsoid"}[shape]
    run(endpoint, f"import lichtfeld as lf\nlf.ui.set_crop_tool_shape({shape!r})\n"
                  f"lf.ui.{command}(lf.get_selected_node_names()[0])\nprint('done')")


@pytest.mark.parametrize("order", [("builtin.roi", "builtin.crop"), ("builtin.crop", "builtin.roi")])
def test_roi_and_crop_names_and_legacy_activation(tmp_path, order):
    with pretraining_app(tmp_path) as endpoint:
        state = toolbar(endpoint)
        assert state["crop_group_buttons"][0]["value"] == "builtin.roi", state["crop_group_buttons"]
        assert state["crop_group_buttons"][0]["icon_src"].endswith("/roi.png"), state["crop_group_buttons"]
        crop_button = next(button for button in state["gizmo_buttons"] if button["value"] == "builtin.crop")
        assert crop_button["icon_src"].endswith("/cropbox.png"), crop_button
        tool_data = _call(endpoint, "resources/read", {"uri": "lichtfeld://ui/tools"})
        tools = {item["id"]: item for item in json.loads(tool_data["contents"][0]["text"])["tools"]}
        assert tools["builtin.roi"]["icon"] == "roi", tools["builtin.roi"]
        assert tools["builtin.crop"]["icon"] == "cropbox", tools["builtin.crop"]
        assert tools["builtin.cropbox"]["icon"] == "cropbox", tools["builtin.cropbox"]
        # Neither the ROI tool nor the generic command creates an ROI ellipsoid, which training would ignore.
        assert run(endpoint, "import lichtfeld as lf\nlf.ui.set_active_operator('builtin.roi', 'translate')\n"
                             "lf.ui.set_crop_tool_shape('ellipsoid')\nprint(lf.ui.get_crop_tool_shape())").strip() == "box"
        run(endpoint, "import lichtfeld as lf\nlf.ui.add_ellipsoid('PointCloud')")
        run(endpoint, "import lichtfeld as lf\nlf.ui.set_active_tool('builtin.select')\nlf.ui.add_ellipsoid('PointCloud')")
        assert not crops(endpoint)
        run(endpoint, "import lichtfeld as lf\nlf.ui.clear_active_operator()")
        for tool in order:
            dispatch(endpoint, "tool", tool)
        volumes = crops(endpoint)
        assert len(volumes) == 2 and all(n["type"] == "crop_box" for n in volumes), volumes
        assert any(n["name"].endswith("_ROI") for n in volumes), volumes
        assert any(n["name"].endswith("_Crop") for n in volumes), volumes
        for tool in ("builtin.roi", "builtin.crop"):
            role = "crop" if tool == "builtin.crop" else "roi"
            dispatch(endpoint, "tool", tool)
            shapes = [b["value"] for b in toolbar(endpoint)["crop_object_buttons"]]
            assert ("ellipsoid" in shapes) == (tool == "builtin.crop"), shapes
            _pick_shape(endpoint, "ellipsoid")
            assert volume(endpoint, role)["shape"] == ("ellipsoid" if role == "crop" else "box")
            if role == "crop":
                _pick_shape(endpoint, "box")
                assert volume(endpoint, role)["shape"] == "box"
        result = run(endpoint, "import lichtfeld as lf\nlf.ui.set_active_operator('builtin.cropbox', 'translate')\n"
                               "print(lf.ui.get_active_tool())")
        assert result.strip() == "builtin.roi"


def test_splat_scene_keeps_legacy_crop_identity(tmp_path):
    model = tmp_path / "grid.ply"
    grid_model(model)
    with isolated_app(tmp_path) as endpoint:
        assert _tool(endpoint, "scene_load_ply", {"path": str(model)})["success"]
        assert _tool(endpoint, "runtime_job_wait", {"job_id": "import.dataset", "until": "inactive",
                                                     "timeout_ms": 60000})["success"]
        name = next(node["name"] for node in _tool(endpoint, "scene_list_nodes")["nodes"] if node["type"] == "splat")
        assert _tool(endpoint, "scene_select_node", {"name": name})["success"]
        run(endpoint, "import lichtfeld as lf\nlf.ui.set_active_operator('builtin.cropbox', 'translate')")
        state = toolbar(endpoint)
        assert run(endpoint, "import lichtfeld as lf\nprint(lf.ui.get_active_tool())").strip() == "builtin.cropbox"
        assert state["crop_group_buttons"][0]["value"] == "builtin.cropbox", state["crop_group_buttons"]
        assert all(button["value"] not in ("builtin.roi", "builtin.crop") for button in state["gizmo_buttons"])
        _pick_shape(endpoint, "ellipsoid")
        volumes = crops(endpoint)
        assert [n["type"] for n in volumes] == ["ellipsoid"], volumes
        assert volumes[0]["name"] == name + "_ellipsoid", volumes


def test_crop_ellipsoid_and_roi_box_survive_project_roundtrip(tmp_path):
    with pretraining_app(tmp_path) as endpoint:
        dispatch(endpoint, "tool", "builtin.roi")
        dispatch(endpoint, "tool", "builtin.crop")
        _pick_shape(endpoint, "ellipsoid")
        before = {role: volume(endpoint, role) for role in ("roi", "crop")}
        assert before["roi"]["shape"] == "box" and before["crop"]["shape"] == "ellipsoid", before
        project = tmp_path / "roles.licht"
        saved = _tool(endpoint, "project_save_as", {"path": str(project)})
        assert saved.get("success", True), saved
        opened = _tool(endpoint, "project_open", {"path": str(project), "discard_changes": True})
        assert opened.get("success", True), opened
        deadline = time.monotonic() + 30
        while len(crops(endpoint)) < 2:
            assert time.monotonic() < deadline, crops(endpoint)
            time.sleep(.1)
        for role in ("roi", "crop"):
            restored = volume(endpoint, role)
            assert (restored["uuid"], restored["shape"]) == (before[role]["uuid"], before[role]["shape"]), restored
        # Reopened, each entry still opens its own tool.
        for role in ("roi", "crop"):
            assert _tool(endpoint, "scene_select_node", {"name": before[role]["name"]})["success"]
            ui = _call(endpoint, "resources/read", {"uri": "lichtfeld://ui/state"})
            assert json.loads(ui["contents"][0]["text"])["registry_active_tool_id"] == "builtin." + role
            assert selected(endpoint)["uuid"] == before[role]["uuid"]
