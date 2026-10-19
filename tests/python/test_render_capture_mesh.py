# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Live captures contain the same mesh geometry as the presented viewport."""

import base64
import io
import json
import os
import subprocess
import time
from unittest.mock import patch

import pytest

from runtime_test_app import isolated_app
from test_api_camera_project_runtime import _editor
from test_render_capture_environment import _splat_fixture
from test_render_capture_camera import _capture
from test_render_on_demand_idle import _call, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.mark.parametrize("backend", ["3dgs", "3dgut"])
@pytest.mark.parametrize("split", [False, True], ids=["single", "independent"])
def test_live_capture_includes_mesh_over_splats(tmp_path, split, backend):
    from PIL import Image, ImageChops, ImageGrab

    model = tmp_path / "background.ply"
    _splat_fixture(model)
    with isolated_app(tmp_path) as endpoint:
        _tool(endpoint, "scene_load_ply", {"path": str(model)})
        _tool(endpoint, "runtime_job_wait", {"job_id": "import.dataset", "timeout_ms": 60000})
        _editor(endpoint, '''
import lichtfeld as lf
from array import array
def mesh_tensor(rows, kind):
    values = array(kind, [value for row in rows for value in row])
    return lf.Tensor.from_numpy(memoryview(values).cast('B').cast(kind, (len(rows), len(rows[0]))))
vertices = [[-1., -1., 1.], [1., -1., 1.], [1., 1., 1.], [-1., 1., 1.]]
lf.get_scene().add_mesh('foreground', mesh_tensor(vertices, 'f'),
    mesh_tensor([[0, 1, 2], [0, 2, 3]], 'i'),
    mesh_tensor([[.02, 1., .02, 1.]] * 4, 'f'),
    mesh_tensor([[0., 0., 1.]] * 4, 'f'))
lf.set_node_visualizer_world_transform('foreground',
    [1.,0.,0.,0., 0.,1.,0.,0., 0.,0.,1.,0., 0.,0.,0.,1.], record_history=False)
settings = lf.get_render_settings()
settings.mesh_light_intensity = 0.
settings.mesh_ambient = 1.
settings.mesh_shadow_enabled = False
settings.mesh_backface_culling = False
''')
        nodes = _tool(endpoint, "scene_list_nodes")["nodes"]
        assert {node["type"] for node in nodes} == {"splat", "mesh"}, nodes
        _tool(endpoint, "ui_tool_invoke", {"tool_id": "builtin.translate"})
        _tool(endpoint, "scene_select_node", {"name": "foreground"})
        _tool(endpoint, "render_settings_set", {
            "show_grid": True, "show_coord_axes": True, "show_pivot": True,
            "render_scale": 1., "split_view_mode": 3 if split else 0,
            "raster_backend": backend,
        })
        _editor(endpoint, '''
for panel in ['left', 'right']:
    lf.set_camera((4.,3.,6.), (0.,0.,0.), (0.,1.,0.), panel=panel)
''')

        def capture(name):
            time.sleep(.3)
            for index in range(4):
                result, image = _capture(endpoint, {})
                assert image is not None, result
                image.save(tmp_path / (name + f'-{index}.png'))
                time.sleep(.3)
            return image

        visible = capture("visible")
        current_result = _call(endpoint, "resources/read", {"uri": "lichtfeld://render/current"})
        current_blob = next(item["blob"] for item in current_result["contents"]
                            if item.get("mimeType") == "image/png")
        current = Image.open(io.BytesIO(base64.b64decode(current_blob))).convert("RGB")
        current.save(tmp_path / "resource-current.png")
        assert max(high for low, high in ImageChops.difference(visible, current).getextrema()) <= 1
        window_result = _call(endpoint, "tools/call", {
            "name": "render_capture_window", "arguments": {},
        })
        content = next(item for item in window_result["content"] if item["type"] == "image")
        window = Image.open(io.BytesIO(base64.b64decode(content["data"]))).convert("RGB")
        window.save(tmp_path / "window.png")
        if "LFS_TEST_DISPLAY" in os.environ:
            with patch.dict(os.environ, {"DISPLAY": os.environ["LFS_TEST_DISPLAY"]}):
                screen = ImageGrab.grab().convert("RGB")
            screen.save(tmp_path / "x-screen.png")
            bounds_path = tmp_path / "viewport-bounds.json"
            _editor(endpoint, f"import json; open({str(bounds_path)!r}, 'w').write(json.dumps(lf.selection.get_viewport_bounds()))")
            vx, vy, vw, vh = map(round, json.loads(bounds_path.read_text()))
            display_env = dict(os.environ, DISPLAY=os.environ["LFS_TEST_DISPLAY"])
            window_id = subprocess.check_output(
                ["xdotool", "search", "--name", "LichtFeld Studio"],
                env=display_env, text=True).splitlines()[0]
            geometry = dict(line.split("=", 1) for line in subprocess.check_output(
                ["xdotool", "getwindowgeometry", "--shell", window_id],
                env=display_env, text=True).splitlines() if "=" in line)
            sx, sy = int(geometry["X"]) + vx, int(geometry["Y"]) + vy
            screen_viewport = screen.crop((sx, sy, sx + vw, sy + vh))
            assert screen_viewport.size == visible.size
            # The actual window has the active gizmo/axes; those red UI pixels
            # must have no matching red marks in the scene-only capture.
            ui_only_red = sum(
                sr > 90 and sr > 1.5 * sg and sr > 1.25 * sb and cr <= 1.2 * cg
                for (sr, sg, sb), (cr, cg, cb) in zip(screen_viewport.getdata(), visible.getdata()))
            assert ui_only_red > 25, ui_only_red
        assert sum(g > 2 * max(r, b) for r, g, b in window.getdata()) > 300
        _tool(endpoint, "render_settings_set", {
            "show_grid": False, "show_coord_axes": False, "show_pivot": False,
        })
        _tool(endpoint, "ui_tool_invoke", {"tool_id": "builtin.select"})
        clean = capture("visible-without-overlays")
        difference = ImageChops.difference(visible, clean)
        assert max(high for low, high in difference.getextrema()) <= 1, "Viewport UI leaked into the scene capture"
        _tool(endpoint, "scene_set_node_visibility", {"name": "foreground", "visible": False})
        hidden = capture("hidden")
        assert visible.size == hidden.size
        # Splat-only captures retain their exact pixels across repeated reads.
        assert hidden.tobytes() == capture("hidden-repeat").tobytes()
        centers = [visible.width // 4, 3 * visible.width // 4] if split else [visible.width // 2]
        for x in centers:
            rgb = visible.getpixel((x, visible.height // 2))
            background = hidden.getpixel((x, hidden.height // 2))
            assert rgb[1] > 2 * max(rgb[0], rgb[2]), (rgb, background)
            assert sum(abs(a - b) for a, b in zip(rgb, background)) > 80
