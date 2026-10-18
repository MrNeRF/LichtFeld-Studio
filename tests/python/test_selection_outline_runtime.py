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


def _write_splat(path, x, scales, dense=False):
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    points = [(0., 0.)]
    if dense:
        points = [(i / 10., j / 10.) for i in range(-15, 16) for j in range(-15, 16)
                  if i * i + j * j <= 225]
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\n"
    header += "".join(f"property float {name}\n" for name in names) + "end_header\n"
    # No red component: the red UI contour can be measured without screenshots.
    dc = [(component - 0.5) / 0.28209479177387814 for component in (0., 0.3, 0.6)]
    local_scales = (0.08, 0.08, 0.05) if dense else scales
    data = b"".join(struct.pack("<14f", x + px * scales[0], py * scales[1], 0,
                               *dc, math.log(0.95 / 0.05),
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
        for x, pixel in enumerate(row) if pixel[0] * factor > 85]
open({str(path)!r}, 'w').write(json.dumps({{'width': width, 'height': height, 'mask': mask}}))
del frame, pixels, mask
""")
    value = json.loads(path.read_text())
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
        # Single-object contours independently identify the shared visible seam.
        # This includes occlusion; hidden portions do not enter either mask.
        exterior = mask_a ^ mask_b
        interior = mask_a & mask_b
        assert len(exterior) > 100 and len(interior) > 30
        assert exterior <= both, "The exterior contour acquired gaps"
        assert both <= mask_a | mask_b, "A contour appeared outside visible object boundaries"
        assert interior <= both, "A visible object boundary acquired gaps"
        _editor(endpoint, "s.desaturate_unselected = True")
        assert _mask(endpoint, tmp_path / "full-overlays.json")[1] == both
        _editor(endpoint, "s.desaturate_unselected = False")
        _select(endpoint, "lf.add_group('group')\nlf.reparent_node('a', 'group')\n"
                          "lf.reparent_node('b', 'group')\nlf.select_node('group')")
        assert _mask(endpoint, tmp_path / "group.json")[1] == both
        _select(endpoint, "lf.deselect_all()")
        assert not _mask(endpoint, tmp_path / "none.json")[1]
        (tmp_path / "evidence.json").write_text(json.dumps({
            "backend": backend, "roll": roll, "dense": dense, "exterior_pixels": len(exterior),
            "interior_pixels": len(interior),
        }, indent=2))
