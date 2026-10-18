# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""GPU contracts for visible per-object selection contours."""

from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import struct
import time
import urllib.request

import pytest

from runtime_test_app import isolated_app

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.mark.parametrize("backend", [2, 3])
def test_selection_outline_accumulated_coverage(tmp_path, backend):
    endpoint = os.environ.get("LFS_OUTLINE_TEST_ENDPOINT")
    if not os.environ.get("LFS_EXECUTABLE"):
        pytest.skip("Set LFS_EXECUTABLE to a build with selection outlines")
    if os.name == "nt" and not endpoint:
        pytest.skip("Windows GPU test requires an owned LFS_OUTLINE_TEST_ENDPOINT")
    faint, below, stack, solid, halo = (tmp_path / (name + ".ply") for name in ("faint", "below", "stack", "solid", "halo"))
    _write_splat(faint, 0., (.5, .2, .1), opacity=.2)
    _write_splat(below, 0., (.5, .2, .1), opacity=.08, copies=8)
    _write_splat(stack, 0., (.5, .2, .1), opacity=.08, copies=16)
    _write_splat(solid, 0., (.2, .2, .1))
    _write_splat(halo, 1.5, (.3, .3, .1), opacity=.2)
    with nullcontext(endpoint) if endpoint else isolated_app(tmp_path) as endpoint:
        _rpc(endpoint, "initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                      "clientInfo": {"name": "outline-coverage-test", "version": "1"}})
        _rpc(endpoint, "tools/list", {})
        for resource in ("runtime/catalog", "runtime/state", "ui/state", "scene/state", "selection/current"):
            _rpc(endpoint, "resources/read", {"uri": "lichtfeld://" + resource})
        _editor(endpoint, f"""
lf.clear_scene()
s = lf.get_render_settings()
s.raster_backend = {backend}
s.selection_outline = True
s.desaturate_unselected = False
s.selection_color_center_marker = (1., 0., 0.)
s.show_grid = False
s.show_camera_frustums = False
s.background_color = (0., 0., 0.)
s.scene_upscaler = 'native'
s.render_scale = 1.
s.lod_enabled = False
lf.set_camera((0., 0., 6.), (0., 0., 0.))
""")

        def load(path):
            _editor(endpoint, f"lf.load_file({str(path)!r})")
            deadline = time.monotonic() + 30
            while True:
                nodes = json.loads(_rpc(endpoint, "resources/read", {"uri": "lichtfeld://scene/nodes"})["contents"][0]["text"])
                if any(node["name"] == path.stem for node in nodes["nodes"]):
                    return
                assert time.monotonic() < deadline, nodes
                time.sleep(.05)

        load(faint)
        _select(endpoint, "lf.select_node('faint')")
        assert not _mask(endpoint, tmp_path / "faint-mask.json")[1], "A single 20% floater produced an outline"
        load(below)
        _select(endpoint, "lf.select_node('below')")
        assert not _mask(endpoint, tmp_path / "below-mask.json")[1], "Coverage below 50% produced an outline"
        load(stack)
        _select(endpoint, "lf.select_node('stack')")
        assert _mask(endpoint, tmp_path / "stack-mask.json")[1], "Sub-10% splats did not accumulate into a visible silhouette"
        _select(endpoint, "lf.select_node('faint')")
        assert not _mask(endpoint, tmp_path / "faint-over-stack.json")[1], "Unselected opacity contributed to selected coverage"
        _editor(endpoint, "lf.set_node_visibility('faint', False)\nlf.set_node_visibility('below', False)\nlf.set_node_visibility('stack', False)")
        load(solid)
        _select(endpoint, "lf.select_node('solid')")
        baseline = _mask(endpoint, tmp_path / "solid-mask.json")[1]
        assert baseline
        load(halo)
        _select(endpoint, "lf.select_nodes(['solid', 'halo'])")
        assert _mask(endpoint, tmp_path / "solid-halo-mask.json")[1] == baseline, "A detached faint halo added a contour"


def _rpc(endpoint, method, params):
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    token_file = os.environ.get("LFS_OUTLINE_TEST_TOKEN_FILE")
    if token_file:
        headers["Authorization"] = "Bearer " + Path(token_file).read_text().strip()
    request = urllib.request.Request(endpoint, json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": method, "params": params,
    }).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read().decode()
    if data.startswith(("event:", "data:")):
        data = next(line[5:].strip() for line in data.splitlines() if line.startswith("data:"))
    result = json.loads(data)
    assert "error" not in result, result
    return result["result"]


def _editor(endpoint, code):
    result = _rpc(endpoint, "tools/call", {"name": "editor_run", "arguments": {
        "code": "import json, lichtfeld as lf\n" + code, "show_console": False,
        "wait_for_completion": True, "wait_for_output": True, "timeout_ms": 30000,
    }})
    assert not result.get("isError"), result
    value = result.get("structuredContent") or json.loads(result["content"][0]["text"])
    assert value["completed"] and value["success"], value
    assert "Traceback" not in value.get("output", {}).get("text", ""), value


def _select(endpoint, code):
    _editor(endpoint, code)
    _rpc(endpoint, "resources/read", {"uri": "lichtfeld://scene/selected_nodes"})
    _rpc(endpoint, "resources/read", {"uri": "lichtfeld://selection/current"})


def _write_splat(path, x, scales, dense=False, z=0., copies=1, opacity=0.95):
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    points = [(0., 0.)]
    if dense:
        points = [(i / 10., j / 10.) for i in range(-15, 16) for j in range(-15, 16)
                  if i * i + j * j <= 225]
    points *= copies
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n"
    header += "".join(f"property float {name}\n" for name in names) + "end_header\n"
    # No red component: the red UI contour can be measured without screenshots.
    dc = [(component - 0.5) / 0.28209479177387814 for component in (0., 0.3, 0.6)]
    local_scales = (0.08, 0.08, 0.05) if dense else scales
    data = b"".join(struct.pack("<14f", x + px * scales[0], py * scales[1], z,
                               *dc, math.log(opacity / (1. - opacity)),
                               *(math.log(scale) for scale in local_scales), 1, 0, 0, 0)
                    for px, py in points)
    path.write_bytes(header.encode() + data)


def _mask(endpoint, path):
    # MCP calls cross UI frames; warm up once before reading the final frame.
    _editor(endpoint, "warmup = lf.capture_viewport()\nassert warmup is not None\ndel warmup")
    _editor(endpoint, f"""
frame = lf.capture_viewport()
assert frame is not None
pixels = frame.image.cpu().tolist()
factor = 255 if isinstance(pixels[0][0][0], float) else 1
width, height = len(pixels[0]), len(pixels)
mask = [y * width + x for y, row in enumerate(pixels)
        for x, pixel in enumerate(row) if pixel[0] * factor > 5]
max_red = max(pixel[0] * factor for row in pixels for pixel in row)
open({str(path)!r}, 'w').write(json.dumps({{'width': width, 'height': height, 'mask': mask, 'max_red': max_red}}))
del frame, pixels, mask
""")
    value = json.loads(path.read_text())
    size_path = path.parent / "viewport-size.json"
    size = [value["width"], value["height"]]
    if size_path.exists():
        assert json.loads(size_path.read_text()) == size, "Viewport resized during the pixel comparison"
    else:
        size_path.write_text(json.dumps(size))
    return (value["width"], value["height"]), set(value["mask"])


@pytest.mark.parametrize("backend", [2, 3])
@pytest.mark.parametrize("roll", [0, 45, 90, 135])
@pytest.mark.parametrize("dense", [False, True], ids=["single-gaussian", "dense-model"])
def test_selection_outline_visible_object_boundaries(tmp_path, backend, roll, dense):
    endpoint = os.environ.get("LFS_OUTLINE_TEST_ENDPOINT")
    if not os.environ.get("LFS_EXECUTABLE"):
        pytest.skip("Set LFS_EXECUTABLE to a build with selection outlines")
    if os.name == "nt" and not endpoint:
        pytest.skip("Windows GPU test requires an owned LFS_OUTLINE_TEST_ENDPOINT")
    a, b = tmp_path / "a.ply", tmp_path / "b.ply"
    _write_splat(a, 0., (0.5, 0.2, 0.1), dense)
    _write_splat(b, 0.65, (0.3, 0.3, 0.1), dense)
    context = nullcontext(endpoint) if endpoint else isolated_app(tmp_path)
    with context as endpoint:
        _rpc(endpoint, "initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                      "clientInfo": {"name": "selection-outline-test", "version": "1"}})
        _rpc(endpoint, "tools/list", {})
        for resource in ("runtime/catalog", "runtime/state", "ui/state", "scene/state", "selection/current"):
            _rpc(endpoint, "resources/read", {"uri": "lichtfeld://" + resource})
        _editor(endpoint, f"lf.clear_scene()\nlf.load_file({str(a)!r})")
        for fixture, expected in ((None, 1), (b, 2)):
            if fixture:
                _editor(endpoint, f"lf.load_file({str(fixture)!r})")
            deadline = time.monotonic() + 30
            while True:
                nodes = _rpc(endpoint, "resources/read", {"uri": "lichtfeld://scene/nodes"})
                if json.loads(nodes["contents"][0]["text"])["count"] == expected:
                    break
                assert time.monotonic() < deadline, nodes
                time.sleep(0.1)
        angle = math.radians(roll)
        _editor(endpoint, f"""
s = lf.get_render_settings()
s.raster_backend = {backend}
s.selection_outline = True
s.desaturate_unselected = False
s.selection_color_center_marker = (1., 0., 0.)
s.show_grid = False
s.show_camera_frustums = False
s.background_color = (0., 0., 0.)
s.scene_upscaler = 'native'
s.render_scale = 1.
s.lod_enabled = False
frame = lf.capture_viewport()
assert frame is not None
rows = frame.image.cpu().tolist()
# Keep both fixtures in view even when docked panels leave a narrow viewport.
distance = 4. * max(1., len(rows) / (1.5 * len(rows[0])))
lf.set_camera((0., 0., distance), (0., 0., 0.), ({math.sin(angle)!r}, {math.cos(angle)!r}, 0.))
del frame, rows
""")
        _select(endpoint, "lf.select_node('a')")
        shape, mask_a = _mask(endpoint, tmp_path / "a.json")
        _select(endpoint, "lf.select_node('b')")
        shape_b, mask_b = _mask(endpoint, tmp_path / "b.json")
        _select(endpoint, "lf.select_nodes(['a', 'b'])")
        shape_both, both = _mask(endpoint, tmp_path / "both.json")
        assert shape == shape_b == shape_both
        # A singly selected object keeps its full silhouette behind unselected
        # objects. Selecting both suppresses hidden contours between them.
        assert len(mask_a) > 100 and len(mask_b) > 100 and len(both) > 100
        assert mask_a != mask_b, "Tree selection did not change the contoured object"
        # Overlapping selected objects can introduce depth-ID boundaries inside
        # both individual silhouettes, so set union/subtraction is not an oracle.
        _editor(endpoint, "s.desaturate_unselected = True")
        assert _mask(endpoint, tmp_path / "full-overlays.json")[1] == both
        _editor(endpoint, "s.desaturate_unselected = False")
        _select(endpoint, "lf.add_group('group')\nlf.reparent_node('a', 'group')\n"
                          "lf.reparent_node('b', 'group')\nlf.select_node('group')")
        assert _mask(endpoint, tmp_path / "group.json")[1] == both
        _select(endpoint, "lf.deselect_all()")
        assert not _mask(endpoint, tmp_path / "none.json")[1]
        # Change only tree selection between frames, without editor requests,
        # camera changes or settings writes that could force reprojection.
        cycle_path = tmp_path / "selection-cycle.json"
        _editor(endpoint, f"""
import os
outline_cycle_tick = 0
outline_cycle_masks = []
def outline_selection_cycle(dt):
    global outline_cycle_tick
    phase, tick = divmod(outline_cycle_tick, 6)
    if tick == 0:
        if phase < 2:
            lf.select_node(('a', 'b')[phase])
        else:
            lf.deselect_all()
    if tick == 5:
        frame = lf.capture_viewport()
        assert frame is not None
        pixels = frame.image.cpu().tolist()
        factor = 255 if isinstance(pixels[0][0][0], float) else 1
        width = len(pixels[0])
        outline_cycle_masks.append([y * width + x for y, row in enumerate(pixels)
                                    for x, pixel in enumerate(row) if pixel[0] * factor > 5])
        if phase == 2:
            with open({str(cycle_path) + '.tmp'!r}, 'w') as output:
                output.write(json.dumps(outline_cycle_masks))
            os.replace({str(cycle_path) + '.tmp'!r}, {str(cycle_path)!r})
            lf.stop_animation()
    outline_cycle_tick += 1
lf.on_frame(outline_selection_cycle, duration_s=20.)
""")
        deadline = time.monotonic() + 25
        while not cycle_path.exists():
            assert time.monotonic() < deadline, "Selection-only frame cycle did not complete"
            time.sleep(0.05)
        cycled = [set(mask) for mask in json.loads(cycle_path.read_text())]
        assert cycled == [mask_a, mask_b, set()], "A previous object's outline survived selection changes"
        _editor(endpoint, "s.selection_outline = False")
        _select(endpoint, "lf.select_node('a')")
        assert not _mask(endpoint, tmp_path / "disabled.json")[1]
        _editor(endpoint, "s.selection_outline = True")
        if roll == 0:
            _editor(endpoint, "lf.set_node_visibility('a', False)\nlf.set_node_visibility('b', False)")
            occluded = tmp_path / "occluded.ply"
            # Keep the selected silhouette well inside the opaque foreground.
            # Otherwise its faint tails can legitimately increase coverage at
            # the foreground's own semi-transparent outer boundary.
            _write_splat(occluded, 0., (.08, .08, .05))
            _editor(endpoint, f"lf.load_file({str(occluded)!r})")
            deadline = time.monotonic() + 30
            while True:
                nodes = _rpc(endpoint, "resources/read", {"uri": "lichtfeld://scene/nodes"})
                if any(node["name"] == "occluded" for node in json.loads(nodes["contents"][0]["text"])["nodes"]):
                    break
                assert time.monotonic() < deadline, nodes
                time.sleep(0.05)
            _select(endpoint, "lf.select_node('occluded')")
            _, visible = _mask(endpoint, tmp_path / "visible.json")
            front = tmp_path / "front.ply"
            _write_splat(front, 0., (2., 2., 0.1), z=-1., copies=4)
            _editor(endpoint, f"lf.load_file({str(front)!r})")
            deadline = time.monotonic() + 30
            while True:
                nodes = _rpc(endpoint, "resources/read", {"uri": "lichtfeld://scene/nodes"})
                if any(node["name"] == "front" for node in json.loads(nodes["contents"][0]["text"])["nodes"]):
                    break
                assert time.monotonic() < deadline, nodes
                time.sleep(0.05)
            _select(endpoint, "lf.select_node('occluded')")
            _, hidden = _mask(endpoint, tmp_path / "hidden.json")
            assert hidden == visible, "An unselected occluder cut the selected silhouette"
            visible_red = json.loads((tmp_path / "visible.json").read_text())["max_red"]
            hidden_red = json.loads((tmp_path / "hidden.json").read_text())["max_red"]
            assert 0.3 < hidden_red / visible_red < 0.4, "Occluded contour alpha differs from Blender's 0.35"
            _select(endpoint, "lf.select_nodes(['occluded', 'front'])")
            _, together = _mask(endpoint, tmp_path / "occluding-selected.json")
            _select(endpoint, "lf.select_node('front')")
            assert _mask(endpoint, tmp_path / "front-only.json")[1] == together
        (tmp_path / "evidence.json").write_text(json.dumps({
            "backend": backend, "roll": roll, "dense": dense,
            "single_a_pixels": len(mask_a), "single_b_pixels": len(mask_b), "both_pixels": len(both),
        }, indent=2))
