# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""GPU contracts for visible per-object selection contours."""

from contextlib import contextmanager, nullcontext
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
s.selection_outline_color = (1., 0., 0.)
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
s.selection_outline_color = (1., 0., 0.)
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


@pytest.mark.parametrize("backend", [2, 3])
def test_selection_outline_ignores_other_objects_faint_halo(tmp_path, backend):
    endpoint = os.environ.get("LFS_OUTLINE_TEST_ENDPOINT")
    if not os.environ.get("LFS_EXECUTABLE") or not endpoint:
        pytest.skip("Requires an owned GPU test application")
    _rpc(endpoint, "initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                 "clientInfo": {"name": "outline-crossing-test", "version": "1"}})
    _rpc(endpoint, "tools/list", {})
    for resource in ("runtime/catalog", "runtime/state", "ui/state", "scene/state", "selection/current"):
        _rpc(endpoint, "resources/read", {"uri": "lichtfeld://" + resource})
    left, halo, right = (tmp_path / (name + ".ply") for name in ("left", "halo", "right"))
    _write_splat(left, -1.2, (.2, .2, .1), z=-.4)
    _write_splat(halo, -.1, (.8, .8, .1), z=-.4, opacity=.003)
    _write_splat(right, .6, (.3, .3, .1))
    header, body = left.read_bytes().split(b"end_header\n", 1)
    left.write_bytes(header.replace(b"element vertex 1", b"element vertex 2") + b"end_header\n" +
                     body + halo.read_bytes().split(b"end_header\n", 1)[1])
    _editor(endpoint, f"""
lf.clear_scene()
s = lf.get_render_settings()
s.raster_backend = {backend}
s.selection_outline = True
s.desaturate_unselected = False
s.selection_color_center_marker = (1., 0., 0.)
s.selection_outline_color = (1., 0., 0.)
s.show_grid = False
s.show_camera_frustums = False
s.background_color = (0., 0., 0.)
s.scene_upscaler = 'native'
s.render_scale = 1.
s.lod_enabled = False
lf.set_camera((0., 0., 6.), (0., 0., 0.), (0., 1., 0.))
""")
    for path in (left, right):
        _editor(endpoint, f"lf.load_file({str(path)!r})")
        deadline = time.monotonic() + 30
        while True:
            nodes = json.loads(_rpc(endpoint, "resources/read", {"uri": "lichtfeld://scene/nodes"})["contents"][0]["text"])
            if any(node["name"] == path.stem for node in nodes["nodes"]):
                break
            assert time.monotonic() < deadline, nodes
            time.sleep(.05)
    _select(endpoint, "lf.select_node('right')")
    (width, _), baseline = _mask(endpoint, tmp_path / "right-only.json")
    assert baseline
    xmin, xmax = min(i % width for i in baseline), max(i % width for i in baseline)
    ymin, ymax = min(i // width for i in baseline), max(i // width for i in baseline)
    _select(endpoint, "lf.select_nodes(['left', 'right'])")
    together = _mask(endpoint, tmp_path / "both.json")[1]
    right_region = {i for i in together if xmin <= i % width <= xmax and ymin <= i // width <= ymax}
    # A genuine 0.3% contribution may move the coverage edge by a pixel;
    # it must not introduce an object-ID boundary through the opaque interior.
    edge_neighborhood = {i + dy * width + dx for i in baseline
                         for dy in range(-2, 3) for dx in range(-2, 3)}
    assert not right_region - edge_neighborhood, "Another selected object's faint halo drew an interior contour"



@pytest.mark.parametrize("backend", [2, 3])
@pytest.mark.parametrize("threshold", [.2, .5, .8])
def test_selection_outline_batch_pruning_preserves_sparse_and_hidden_selection(tmp_path, backend, threshold):
    endpoint = os.environ.get("LFS_OUTLINE_TEST_ENDPOINT")
    if not os.environ.get("LFS_EXECUTABLE") or not endpoint:
        pytest.skip("Requires an owned GPU test application")
    _rpc(endpoint, "initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                 "clientInfo": {"name": "outline-depth-bound-test", "version": "1"}})
    _rpc(endpoint, "tools/list", {})
    for resource in ("runtime/catalog", "runtime/state", "ui/state", "scene/state", "selection/current"):
        _rpc(endpoint, "resources/read", {"uri": "lichtfeld://" + resource})
    selected, behind, front, distant = (tmp_path / (name + ".ply")
                                         for name in ("selected", "behind", "front", "distant"))
    _write_splat(selected, 0., (.14, .14, .05))
    _write_splat(behind, 0., (3., 3., .05), z=2., copies=2048)
    _write_splat(front, 0., (3., 3., .05), z=-1., copies=8)
    _write_splat(distant, 0., (.3, .3, .05), z=1.)
    _editor(endpoint, f"""
lf.clear_scene()
s = lf.get_render_settings()
s.raster_backend = {backend}
s.selection_outline = True
s.selection_outline_color = (1., 0., 0.)
s.selection_outline_opacity = 1.
s.selection_outline_width = 2.
s.selection_outline_threshold = {threshold}
s.selection_outline_holes = True
s.desaturate_unselected = False
s.show_grid = False
s.show_camera_frustums = False
s.background_color = (0., 0., 0.)
s.scene_upscaler = 'native'
s.render_scale = 1.
s.lod_enabled = False
lf.set_camera((0., 0., 6.), (0., 0., 0.), (0., 1., 0.))
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

    load(selected)
    load(behind)  # Force the general multi-node mask, with a long irrelevant tail.
    _select(endpoint, "lf.select_node('selected')")
    baseline = _mask(endpoint, tmp_path / "background-tail.json")[1]
    assert baseline
    load(front)
    _select(endpoint, "lf.select_node('selected')")
    assert _mask(endpoint, tmp_path / "opaque-foreground.json")[1] == baseline, \
        "Scene transmittance prematurely terminated the selected-only silhouette"
    load(distant)
    _select(endpoint, "lf.select_node('distant')")
    far_mask = _mask(endpoint, tmp_path / "distant-selected.json")[1]
    assert far_mask and far_mask != baseline, "Selection mask was stale after changing objects"
    _select(endpoint, "lf.select_node('selected')")
    assert _mask(endpoint, tmp_path / "selected-again.json")[1] == baseline
    _editor(endpoint, "s.selection_outline_threshold = .5")


def test_selection_outline_mcp_roundtrip():
    endpoint = os.environ.get("LFS_OUTLINE_TEST_ENDPOINT")
    if not endpoint:
        pytest.skip("Requires an owned test application")
    _rpc(endpoint, "initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                 "clientInfo": {"name": "outline-mcp-test", "version": "1"}})
    tools = _rpc(endpoint, "tools/list", {})["tools"]
    for resource in ("runtime/catalog", "runtime/state", "ui/state", "scene/state", "selection/current"):
        _rpc(endpoint, "resources/read", {"uri": "lichtfeld://" + resource})
    setter = next(tool for tool in tools if tool["name"] == "render_settings_set")
    assert setter["inputSchema"]["properties"]["selection_outline"]["type"] == "boolean"

    def call(name, arguments):
        result = _rpc(endpoint, "tools/call", {"name": name, "arguments": arguments})
        assert not result.get("isError"), result
        return result.get("structuredContent") or json.loads(result["content"][0]["text"])

    original = call("render_settings_get", {})["settings"]["selection_outline"]
    try:
        for enabled in (False, True, False):
            call("render_settings_set", {"selection_outline": enabled})
            assert call("render_settings_get", {})["settings"]["selection_outline"] is enabled
    finally:
        call("render_settings_set", {"selection_outline": original})


@pytest.mark.parametrize("backend", [2, 3])
def test_selection_outline_single_node_respects_selection_depth_filter(tmp_path, backend):
    endpoint = os.environ.get("LFS_OUTLINE_TEST_ENDPOINT")
    if not endpoint:
        pytest.skip("Requires an owned GPU test application")
    _rpc(endpoint, "initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                 "clientInfo": {"name": "outline-filter-test", "version": "1"}})
    _rpc(endpoint, "tools/list", {})
    for resource in ("runtime/catalog", "runtime/state", "ui/state", "scene/state", "selection/current"):
        _rpc(endpoint, "resources/read", {"uri": "lichtfeld://" + resource})
    one, second = tmp_path / "one.ply", tmp_path / "second.ply"
    _write_splat(one, -.6, (.2, .2, .1))
    _write_splat(second, .6, (.2, .2, .1))
    header, body = one.read_bytes().split(b"end_header\n", 1)
    one.write_bytes(header.replace(b"element vertex 1", b"element vertex 2") + b"end_header\n" +
                    body + second.read_bytes().split(b"end_header\n", 1)[1])
    _editor(endpoint, f"lf.clear_scene()\nlf.load_file({str(one)!r})")
    deadline = time.monotonic() + 30
    while True:
        nodes = json.loads(_rpc(endpoint, "resources/read", {"uri": "lichtfeld://scene/nodes"})["contents"][0]["text"])
        if any(node["name"] == "one" for node in nodes["nodes"]):
            break
        assert time.monotonic() < deadline, nodes
        time.sleep(.05)
    _editor(endpoint, f"""
s = lf.get_render_settings()
s.raster_backend = {backend}
s.selection_outline = True
s.desaturate_unselected = False
s.selection_color_center_marker = (1., 0., 0.)
s.selection_outline_color = (1., 0., 0.)
s.show_grid = False
s.show_camera_frustums = False
s.background_color = (0., 0., 0.)
s.scene_upscaler = 'native'
s.render_scale = 1.
s.lod_enabled = False
s.hide_outside_depth_box = False
lf.set_camera((0., 0., 6.), (0., 0., 0.), (0., 1., 0.))
""")
    result = _rpc(endpoint, "tools/call", {"name": "render_settings_set", "arguments": {"depth_filter_enabled": False}})
    assert not result.get("isError"), result
    _select(endpoint, "lf.select_node('one')")
    baseline = _mask(endpoint, tmp_path / "all.json")[1]
    assert baseline
    result = _rpc(endpoint, "tools/call", {"name": "render_settings_set", "arguments": {
        "depth_filter_min": [0., -10., -10.], "depth_filter_max": [10., 10., 10.], "depth_filter_enabled": True}})
    assert not result.get("isError"), result
    filtered = _mask(endpoint, tmp_path / "filtered.json")[1]
    assert 0 < len(filtered) < len(baseline), "Rendered coverage bypassed the object selection filter"
    _rpc(endpoint, "tools/call", {"name": "render_settings_set", "arguments": {"depth_filter_enabled": False}})
    _editor(endpoint, "s.desaturate_unselected = False")


@pytest.mark.parametrize("backend", [2, 3])
@pytest.mark.parametrize("line_width", [2, 8])
@pytest.mark.parametrize("single_node", [True, False], ids=["single-node", "multi-node"])
@pytest.mark.parametrize("open_ring", [False, True], ids=["enclosed-hole", "open-cavity"])
def test_selection_outline_interior_contours(tmp_path, backend, open_ring, single_node, line_width):
    """Only a background component disconnected from the viewport is hidden."""
    with _appearance_scene(tmp_path, backend, ring=True, open_ring=open_ring, single_node=single_node) as endpoint:
        _rpc(endpoint, "tools/call", {"name": "render_settings_set", "arguments": {"selection_outline_holes": True, "selection_outline_width": line_width}})
        size, complete = _mask(endpoint, tmp_path / "with-holes.json")
        _rpc(endpoint, "tools/call", {"name": "render_settings_set", "arguments": {"selection_outline_holes": False}})
        _, external = _mask(endpoint, tmp_path / "without-holes.json")
        assert complete and external
        assert external <= complete, "Suppressing holes introduced new outline pixels"
        if open_ring:
            assert external == complete, "A cavity connected to the exterior was suppressed"
        else:
            assert len(external) < len(complete) * .8, "Enclosed hole retained its contour"
            # The removed contour must be inside the retained outer contour.
            width, _ = size
            center_x = sum(i % width for i in external) / len(external)
            center_y = sum(i // width for i in external) / len(external)
            mean_radius = lambda mask: sum(math.hypot(i % width - center_x, i // width - center_y) for i in mask) / len(mask)
            assert mean_radius(complete - external) < mean_radius(external)
        _rpc(endpoint, "tools/call", {"name": "render_settings_set", "arguments": {"selection_outline_holes": True}})
        assert _mask(endpoint, tmp_path / "restored-holes.json")[1] == complete


@pytest.mark.parametrize("backend", [2, 3])
@pytest.mark.parametrize("single_node", [True, False], ids=["single-node", "multi-node"])
def test_selection_outline_appearance(tmp_path, backend, single_node):
    with _appearance_scene(tmp_path, backend, single_node=single_node) as endpoint:
        def patch(**values):
            result = _rpc(endpoint, "tools/call", {"name": "render_settings_set", "arguments": values})
            assert not result.get("isError"), result
        size, solid = _mask(endpoint, tmp_path / "solid.json")
        assert solid
        patch(selection_outline_opacity=0)
        assert not _mask(endpoint, tmp_path / "transparent.json")[1]
        patch(selection_outline_opacity=.5)
        assert _mask(endpoint, tmp_path / "half-alpha.json")[1]
        full = json.loads((tmp_path / "solid.json").read_text())["max_red"]
        half = json.loads((tmp_path / "half-alpha.json").read_text())["max_red"]
        assert .35 * full < half < .7 * full
        patch(selection_outline_opacity=1, selection_outline_color=[0, 1, 0])
        assert not _mask(endpoint, tmp_path / "green.json")[1], "Dedicated outline color was ignored"
        patch(selection_outline_color=[1, 0, 0], selection_outline_width=1)
        thin = _mask(endpoint, tmp_path / "thin.json")[1]
        assert thin <= solid and thin
        assert json.loads((tmp_path / "thin.json").read_text())["max_red"] < .7 * full, "Thickness did not change line coverage"
        patch(selection_outline_width=2, selection_outline_threshold=.9)
        high = _mask(endpoint, tmp_path / "high-threshold.json")[1]
        patch(selection_outline_threshold=.1)
        low = _mask(endpoint, tmp_path / "low-threshold.json")[1]
        width, height = size
        # Higher coverage contracts the silhouette; it must not use the old 50% fast path.
        radius = lambda mask: max(math.hypot(i % width - width / 2, i // width - height / 2) for i in mask)
        assert high and low and radius(high) < radius(low)
        patch(selection_outline_threshold=.5)
        assert _mask(endpoint, tmp_path / "restored.json")[1] == solid


@contextmanager
def _appearance_scene(tmp_path, backend, *, ring=False, open_ring=False, single_node=True, scales=(.5, .3, .1)):
    endpoint = os.environ.get("LFS_OUTLINE_TEST_ENDPOINT")
    if not os.environ.get("LFS_EXECUTABLE"):
        pytest.skip("Set LFS_EXECUTABLE to a build with selection outlines")
    if os.name == "nt" and not endpoint:
        pytest.skip("Windows GPU test requires an owned LFS_OUTLINE_TEST_ENDPOINT")
    path = tmp_path / "appearance.ply"
    if ring:
        props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                 "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        points = [(math.cos(a) * .8, math.sin(a) * .8) for a in (i * math.tau / 128 for i in range(128))
                  if not open_ring or abs(math.atan2(math.sin(a), math.cos(a))) > .45]
        header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n"
        header += "".join(f"property float {name}\n" for name in props) + "end_header\n"
        dc = [(c - .5) / .28209479177387814 for c in (0, .3, .6)]
        data = b"".join(struct.pack("<14f", x, y, 0, *dc, math.log(19),
                                   math.log(.055), math.log(.055), math.log(.04), 1, 0, 0, 0)
                        for x, y in points)
        path.write_bytes(header.encode() + data)
    else:
        _write_splat(path, 0, scales)
    with nullcontext(endpoint) if endpoint else isolated_app(tmp_path) as endpoint:
        _rpc(endpoint, "initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                      "clientInfo": {"name": "outline-appearance-test", "version": "1"}})
        _rpc(endpoint, "tools/list", {})
        for resource in ("runtime/catalog", "runtime/state", "ui/state", "scene/state", "selection/current"):
            _rpc(endpoint, "resources/read", {"uri": "lichtfeld://" + resource})
        original = _rpc(endpoint, "tools/call", {"name": "render_settings_get", "arguments": {}})
        original = original.get("structuredContent") or json.loads(original["content"][0]["text"])
        _editor(endpoint, f"""
lf.clear_scene()
lf.load_file({str(path)!r})
s = lf.get_render_settings()
s.raster_backend = {backend}
s.selection_outline = True
s.selection_outline_color = (1., 0., 0.)
s.selection_outline_opacity = 1.
s.selection_outline_width = 2.
s.selection_outline_threshold = .5
s.selection_outline_holes = True
s.desaturate_unselected = False
s.show_grid = False
s.show_camera_frustums = False
s.background_color = (0., 0., 0.)
s.scene_upscaler = 'native'
s.render_scale = 1.
s.lod_enabled = False
lf.set_camera((0., 0., 6.), (0., 0., 0.))
""")
        if not single_node:
            other = tmp_path / "other.ply"
            _write_splat(other, 4, (.1, .1, .1), opacity=.1)
            _editor(endpoint, f"lf.load_file({str(other)!r})")
        deadline = time.monotonic() + 30
        while True:
            nodes = json.loads(_rpc(endpoint, "resources/read", {"uri": "lichtfeld://scene/nodes"})["contents"][0]["text"])
            expected = {path.stem} | ({"other"} if not single_node else set())
            if expected <= {node["name"] for node in nodes["nodes"]}:
                break
            assert time.monotonic() < deadline, nodes
            time.sleep(.05)
        _select(endpoint, "lf.select_node('appearance')")
        try:
            yield endpoint
        finally:
            keys = ["selection_outline_color", "selection_outline_opacity", "selection_outline_width",
                    "selection_outline_threshold", "selection_outline_holes"]
            _rpc(endpoint, "tools/call", {"name": "render_settings_set", "arguments": {k: original["settings"][k] for k in keys}})


def test_selection_outline_appearance_validation(tmp_path):
    with _appearance_scene(tmp_path, 2) as endpoint:
        result = _rpc(endpoint, "tools/call", {"name": "render_settings_get", "arguments": {}})
        value = result.get("structuredContent") or json.loads(result["content"][0]["text"])
        assert "selection_outline_style" not in value["settings"]
        _editor(endpoint, "assert not hasattr(lf.get_render_settings(), 'selection_outline_style')")
        for patch in ({"selection_outline_opacity": -1}, {"selection_outline_width": 13},
                      {"selection_outline_threshold": 0},
                      {"selection_outline_color": [1, 0]}, {"selection_outline_color": [0, -1, 0]}):
            result = _rpc(endpoint, "tools/call", {"name": "render_settings_set", "arguments": patch})
            assert result.get("isError"), result


@pytest.mark.parametrize("backend", [2, 3])
@pytest.mark.parametrize("single_node", [True, False], ids=["single-node", "multi-node"])
def test_selection_outline_extended_thickness(tmp_path, backend, single_node):
    with _appearance_scene(tmp_path, backend, single_node=single_node) as endpoint:
        def patch(**values):
            result = _rpc(endpoint, "tools/call", {"name": "render_settings_set", "arguments": values})
            assert not result.get("isError"), result
        baseline = _mask(endpoint, tmp_path / "width-2.json")[1]
        masks = {}
        previous = baseline
        for width in (2.01, 3, 6, 12):
            patch(selection_outline_width=width)
            current = _mask(endpoint, tmp_path / f"width-{width}.json")[1]
            assert previous <= current, "Increasing thickness removed contour pixels"
            masks[width] = current
            previous = current
        assert len(masks[2.01]) < len(baseline) * 1.2, "Thick-line path has a discontinuity at 2px"
        assert len(masks[6]) > len(baseline) * 1.5
        assert len(masks[12]) > len(baseline) * 3
        patch(selection_outline_opacity=0)
        assert not _mask(endpoint, tmp_path / "wide-transparent.json")[1]
        patch(selection_outline_width=2, selection_outline_opacity=1)
        assert _mask(endpoint, tmp_path / "thin-restored.json")[1] == baseline
