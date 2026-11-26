# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise both crop toolbar workflows in the real pre-training application."""
import json

import pytest

from crop_role_runtime import crops, dispatch, pretraining_app, run, selected
from test_render_on_demand_idle import _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _role(tool):
    return "crop" if tool == "builtin.crop" else "roi"


@pytest.mark.parametrize("order", [
    ("builtin.cropbox", "builtin.crop"),
    ("builtin.crop", "builtin.cropbox"),
])
def test_crop_toolbar_roles_keep_independent_shapes_and_outlines(tmp_path, order):
    with pretraining_app(tmp_path) as endpoint:
        def pick_shape(tool):
            # The Crop takes the ellipsoid button; the ROI becomes a polygon from its box footprint.
            if tool == "builtin.crop":
                dispatch(endpoint, "crop_object", "ellipsoid")
            else:
                assert _tool(endpoint, "crop_volume_set", {"owner": "PointCloud", "role": "roi",
                                                           "shape": "polygon"})["success"]

        dispatch(endpoint, "tool", order[0])
        assert len(crops(endpoint)) == 1
        pick_shape(order[0])
        first = crops(endpoint)[0]
        assert first["type"] == ("ellipsoid" if order[0] == "builtin.crop" else "crop_polygon")
        dispatch(endpoint, "tool", order[1])
        pick_shape(order[1])
        volumes = crops(endpoint)
        assert len(volumes) == 2
        assert {node["type"] for node in volumes} == {"crop_polygon", "ellipsoid"}
        assert all(node["visible"] for node in volumes)
        identities = {node["uuid"]: node["type"] for node in volumes}
        for tool_id in order:
            dispatch(endpoint, "tool", tool_id)
            volumes = crops(endpoint)
            assert {node["uuid"]: node["type"] for node in volumes} == identities
            assert all(node["visible"] for node in volumes)
        removed = next(node for node in volumes if node["uuid"] == first["uuid"])
        sibling = next(node for node in volumes if node["uuid"] != first["uuid"])
        run(endpoint, "import lichtfeld as lf\n" + f"lf.get_scene().remove_node({removed['name']!r})\nprint('done')")
        remaining = crops(endpoint)
        assert [(node["uuid"], node["type"]) for node in remaining] == [(sibling["uuid"], sibling["type"])]


def test_crop_delete_undo_keeps_restored_size_and_the_sibling(tmp_path):
    """Trim the ROI and reset the Crop through the live toolbar, then delete and undo each."""
    with pretraining_app(tmp_path) as endpoint:
        def boxes():
            return [n for n in crops(endpoint) if n["type"] == "crop_box"]

        def bounds(name):
            value = _tool(endpoint, "crop_box_get", {"node": name})
            return value["crop_box"]["min"], value["crop_box"]["max"]

        dispatch(endpoint, "tool", "builtin.cropbox")
        primary = boxes()[0]
        dispatch(endpoint, "tool", "builtin.crop")
        secondary = next(n for n in boxes() if n["uuid"] != primary["uuid"])
        dispatch(endpoint, "tool", "builtin.cropbox")
        dispatch(endpoint, "crop_trim")
        primary_before = bounds(primary["name"])
        dispatch(endpoint, "tool", "builtin.crop")
        dispatch(endpoint, "crop_reset")
        secondary_before = bounds(secondary["name"])
        assert primary_before != secondary_before, (primary_before, secondary_before)
        dispatch(endpoint, "crop_delete")
        assert [n["uuid"] for n in boxes()] == [primary["uuid"]]
        dispatch(endpoint, "tool", "builtin.cropbox")
        assert _tool(endpoint, "history_undo")["success"]
        assert {n["uuid"] for n in boxes()} == {primary["uuid"], secondary["uuid"]}
        assert bounds(primary["name"]) == primary_before
        assert bounds(secondary["name"]) == secondary_before

        # Delete the ROI the other way round; the Crop keeps its bounds.
        dispatch(endpoint, "tool", "builtin.cropbox")
        dispatch(endpoint, "crop_delete")
        assert [n["uuid"] for n in boxes()] == [secondary["uuid"]]
        assert bounds(secondary["name"]) == secondary_before
        assert _tool(endpoint, "history_undo")["success"]
        assert bounds(primary["name"]) == primary_before
        assert bounds(secondary["name"]) == secondary_before


@pytest.mark.parametrize("order", [("builtin.cropbox", "builtin.crop"), ("builtin.crop", "builtin.cropbox")])
def test_roi_settings_follow_crop_role_and_close_when_switching(tmp_path, order):
    """The real viewport toolbar exposes ROI settings only for the ROI."""
    with pretraining_app(tmp_path) as endpoint:
        def state():
            result = run(endpoint, "import json, lfs_plugins.toolbar as toolbar\n"
                                   "c=toolbar._toolbar_controller\n"
                                   "c._last_toolbar_signature=None\n"
                                   "c._sync_toolbar_state()\n"
                                   "print(json.dumps({'buttons': c._gizmo.snapshot()['crop_settings_buttons'], "
                                   "'open': c._crop_roi_settings_open}))")
            return json.loads(result)

        dispatch(endpoint, "tool", order[0])
        names = {order[0]: selected(endpoint)["name"]}
        dispatch(endpoint, "tool", order[1])
        names[order[1]] = selected(endpoint)["name"]
        if order[1] != "builtin.cropbox":
            dispatch(endpoint, "tool", "builtin.cropbox")
        assert len(state()["buttons"]) == 1
        run(endpoint, "import lfs_plugins.toolbar as toolbar\n"
                      "toolbar._toolbar_controller._on_toolbar_action(None, None, ['toggle_crop_roi_settings', ''])\n"
                      "print('opened')")
        assert state()["open"] is True
        dispatch(endpoint, "tool", "builtin.crop")
        after = state()
        assert after["buttons"] == [], after
        assert after["open"] is False, after
        dispatch(endpoint, "tool", "builtin.cropbox")
        assert len(state()["buttons"]) == 1

        # Scene selection without an active crop tool uses the stored role.
        assert _tool(endpoint, "scene_select_node", {"name": names["builtin.crop"]})["success"]
        assert state()["buttons"] == []
        assert _tool(endpoint, "scene_select_node", {"name": names["builtin.cropbox"]})["success"]
        assert len(state()["buttons"]) == 1
        assert _tool(endpoint, "scene_select_node", {"name": "PointCloud"})["success"]
        assert len(state()["buttons"]) == 1

        for tool in ("builtin.crop", "builtin.cropbox"):
            dispatch(endpoint, "tool", tool)
            assert _tool(endpoint, "crop_volume_set", {"owner": "PointCloud", "role": _role(tool),
                                                       "shape": "polygon"})["success"]
            assert (state()["buttons"] == []) == (tool == "builtin.crop")
