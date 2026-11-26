# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Applying the point cloud's Crop removes points, keeps the ROI and undoes in one step."""
import json

import pytest

from crop_role_runtime import dispatch as _dispatch
from crop_role_runtime import nodes as _nodes
from crop_role_runtime import pretraining_app
from crop_role_runtime import run as _run
from crop_role_runtime import selected as _selected
from test_render_on_demand_idle import _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.mark.parametrize("order", [("builtin.cropbox", "builtin.crop"), ("builtin.crop", "builtin.cropbox")])
@pytest.mark.parametrize("shape", ["box", "ellipsoid", "polygon"])
@pytest.mark.parametrize("method", ["toolbar", "api"])
def test_apply_crop_preserves_roi_and_undo_restores_points(tmp_path, order, shape, method):
    with pretraining_app(tmp_path) as endpoint:
        def run(code):
            return _run(endpoint, code)

        def dispatch(kind, value=""):
            _dispatch(endpoint, kind, value)

        def nodes():
            return _nodes(endpoint)

        def selected():
            return _selected(endpoint)

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
                    "h.update(repr(cloud.means.cpu().tolist()).encode())\n"
                    "h.update(repr(cloud.colors.cpu().tolist()).encode())\n"
                    "print(h.hexdigest())")
            return run(code).strip()
        names = {}
        uuids = {}
        for tool in order:
            dispatch("tool", tool)
            if shape == "polygon":
                role = "crop" if tool == "builtin.crop" else "roi"
                assert _tool(endpoint, "crop_volume_set", {"owner": "PointCloud", "role": role,
                                                           "shape": "polygon"})["success"]
            elif tool == "builtin.crop" and shape == "ellipsoid":
                dispatch("crop_object", "ellipsoid")
            node = selected()
            names[tool] = node["name"]
            uuids[tool] = node["uuid"]
            if tool == "builtin.cropbox":
                # The ROI's check commits it; it never removes points.
                dispatch("crop_apply")
                assert any(n["uuid"] == uuids[tool] for n in nodes())
        if order[-1] != "builtin.crop":
            dispatch("tool", "builtin.crop")
        dispatch("crop_trim")
        # The generated cloud is a flat grid whose trimmed bounds hold every point; a smaller Crop
        # leaves its outer ring outside.
        geometry = {"box": {"min": [-.5, -.5, -1.], "max": [.5, .5, 1.]},
                    "ellipsoid": {"radii": [.6, .6, 1.]},
                    "polygon": {"vertices": [[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]], "half_depth": 1.}}[shape]
        assert _tool(endpoint, "crop_volume_set", {"owner": "PointCloud", "role": "crop", "shape": shape,
                                                   "translation": [0., 0., 0.], **geometry})["success"]
        before_roi = roi_state(names["builtin.cropbox"])
        before_count = next(n["gaussian_count"] for n in nodes() if n["name"] == "PointCloud")
        assert before_count > 0
        before_digest = point_digest()
        if method == "toolbar":
            actions = json.loads(run("import json\nfrom lfs_plugins.toolbar import _GizmoToolbarController\n"
                                     "print(json.dumps(_GizmoToolbarController().snapshot()['crop_action_buttons']))"))
            assert next(item for item in actions if item["action"] == "crop_apply")["tooltip_key"] == "common.apply"
            dispatch("crop_apply")
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
