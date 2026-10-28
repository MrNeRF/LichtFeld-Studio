# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare GUT rendering of generated PLY splats and their quantized RAD pool.

Requires a GUI GPU runner. LFS_GUT_TEST_ENDPOINT optionally targets an owned
Windows test instance; LFS_GUT_TEST_TOKEN_FILE supplies its MCP bearer token.
Otherwise the existing isolated Linux/Xvfb runner is used.
"""

from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import time
import urllib.request

import pytest

from runtime_test_app import isolated_app

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _rpc(endpoint, method, params):
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    token_file = os.environ.get("LFS_GUT_TEST_TOKEN_FILE")
    if token_file:
        headers["Authorization"] = "Bearer " + Path(token_file).read_text().strip()
    request = urllib.request.Request(endpoint, json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": method, "params": params,
    }).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read().decode()
    if body.startswith(("event:", "data:")):
        body = next(line[5:].strip() for line in body.splitlines() if line.startswith("data:"))
    result = json.loads(body)
    assert "error" not in result, result
    return result["result"]


def _editor(endpoint, code):
    result = _rpc(endpoint, "tools/call", {"name": "editor_run", "arguments": {
        "code": "import json, lichtfeld as lf\n" + code,
        "show_console": False, "wait_for_completion": True, "wait_for_output": True,
        "timeout_ms": 30000,
    }})
    assert not result.get("isError"), result
    value = result.get("structuredContent") or json.loads(result["content"][0]["text"])
    assert value["completed"] and value["success"], value
    assert "Traceback" not in value.get("output", {}).get("text", ""), value


def _fixture(path, count):
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {count}\n"
    header += "".join(f"property float {name}\n" for name in names) + "end_header\n"
    with path.open("wb") as output:
        output.write(header.encode())
        for index in range(count):
            if count > 3:
                side = math.ceil(math.sqrt(count))
                x = (index % side - side / 2) * 0.015
                y = (index // side - side / 2) * 0.015
                scales = (0.003, 0.002, 0.0015)
            else:
                x = (index - (count - 1) / 2) * 0.9
                y = 0
                scales = (0.3, 0.12, 0.08)
            color = [(0.15, 0.5, 0.8), (0.8, 0.2, 0.1), (0.2, 0.8, 0.3)][index % 3]
            dc = [(value - 0.5) / 0.28209479177387814 for value in color]
            opacity = [0.95, 0.65, 0.35][index % 3]
            angle = (index % 3) * 0.4
            output.write(struct.pack("<14f", x, y, 0, *dc, math.log(opacity / (1 - opacity)),
                                     *(math.log(scale) for scale in scales),
                                     math.cos(angle / 2), 0, 0, math.sin(angle / 2)))


def _capture(endpoint, path):
    _editor(endpoint, f"""
frame = lf.capture_viewport()
assert frame is not None
pixels = frame.image.cpu().tolist()
factor = 255 if isinstance(pixels[0][0][0], float) else 1
payload = bytes(max(0, min(255, round(value * factor)))
                for row in pixels for pixel in row for value in pixel[:3])
open({str(path)!r}, 'wb').write(payload)
open({str(path.with_suffix('.json'))!r}, 'w').write(json.dumps([len(pixels[0]), len(pixels)]))
del frame, pixels, payload
""")
    return path.read_bytes(), json.loads(path.with_suffix(".json").read_text())


@pytest.mark.parametrize("count,overlays,profile", [
    (count, overlays, profile)
    for count in (1, 3) for overlays in (False, True) for profile in (0, 1)
] + [(4097, False, 0)])
def test_gut_rad_matches_source_splats(tmp_path, count, overlays, profile):
    executable = os.environ.get("LFS_EXECUTABLE")
    endpoint = os.environ.get("LFS_GUT_TEST_ENDPOINT")
    if not executable:
        pytest.skip("Set LFS_EXECUTABLE to a built GUI application")
    if os.name == "nt" and not endpoint:
        pytest.skip("Windows GPU test requires an owned LFS_GUT_TEST_ENDPOINT")
    source, rad = tmp_path / "source.ply", tmp_path / "source.rad"
    _fixture(source, count)
    with (tmp_path / "convert.log").open("w") as log:
        subprocess.run([executable, "convert", str(source), str(rad), "--lod-builder", "octree",
                        "--no-provenance"], check=True, timeout=120, stdout=log, stderr=subprocess.STDOUT)
    context = nullcontext(endpoint) if endpoint else isolated_app(tmp_path)
    with context as endpoint:
        _rpc(endpoint, "initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                      "clientInfo": {"name": "gut-rad-regression", "version": "1"}})
        _rpc(endpoint, "tools/list", {})
        for resource in ("runtime/catalog", "runtime/state", "ui/state", "scene/state", "selection/current"):
            _rpc(endpoint, "resources/read", {"uri": "lichtfeld://" + resource})
        references = []
        for label, fixture in (("ply", source), ("rad", rad)):
            _editor(endpoint, f"lf.clear_scene()\nlf.load_file({str(fixture)!r})")
            deadline = time.monotonic() + 30
            while True:
                state = _rpc(endpoint, "resources/read", {"uri": "lichtfeld://scene/nodes"})
                if json.loads(state["contents"][0]["text"])["count"]:
                    break
                assert time.monotonic() < deadline, state
                time.sleep(0.1)
            _editor(endpoint, """
s = lf.get_render_settings()
s.raster_backend = 3
s.splat_render_profile = 0
s.mip_filter = False
s.show_grid = False
s.show_camera_frustums = False
s.background_color = (0., 0., 0.)
s.desaturate_unselected = False
s.lod_enabled = True
s.lod_max_splats = 100000
s.lod_fade_frames = 0
s.scene_upscaler = "native"
s.render_scale = 1.
lf.deselect_all()
lf.set_camera((0., 0., 4.), (0., 0., 0.))
""")
            if overlays:
                _editor(endpoint, 'lf.select_node("source")\nlf.get_render_settings().desaturate_unselected = True')
                selected = _rpc(endpoint, "resources/read", {"uri": "lichtfeld://scene/selected_nodes"})
                assert '"source"' in selected["contents"][0]["text"], selected
            _rpc(endpoint, "resources/read", {"uri": "lichtfeld://selection/current"})
            if label == "rad":
                _editor(endpoint, f"lf.get_render_settings().splat_render_profile = {profile}")
                stats_path = tmp_path / "lod.json"
                while True:
                    _editor(endpoint, f"open({str(stats_path)!r}, 'w').write(json.dumps(lf.get_lod_stats()))")
                    stats = json.loads(stats_path.read_text())
                    if stats["active"] and stats["selected"] == count:
                        if count > 3:
                            assert stats["resident_chunks"] > 1, stats
                        break
                    assert time.monotonic() < deadline, stats
                    time.sleep(0.1)
            # Capture twice across frames so camera and asynchronous LOD changes settle.
            _capture(endpoint, tmp_path / (label + "-warmup.rgb"))
            references.append(_capture(endpoint, tmp_path / (label + ".rgb")))
        (ply_pixels, shape), (rad_pixels, rad_shape) = references
        assert shape == rad_shape and len(ply_pixels) == len(rad_pixels)
        assert max(ply_pixels) > 50, "Source fixture was not visible"
        mean_error = sum(abs(a - b) for a, b in zip(ply_pixels, rad_pixels)) / len(ply_pixels)
        differing = sum(abs(a - b) > 12 for a, b in zip(ply_pixels, rad_pixels)) / len(ply_pixels)
        (tmp_path / "comparison.json").write_text(json.dumps({"mean_error": mean_error, "differing": differing}))
        # RAD quantizes color, rotation, log-scale and opacity. Permit quantization,
        # while rejecting the screen-filling flat/tiled result of raw-f32 reads.
        assert mean_error < 1.5, mean_error
        assert differing < 0.015, differing
        if count == 3:
            # A coarse cut exercises high-opacity parent kernels, not only leaves.
            _editor(endpoint, "lf.get_render_settings().lod_max_splats = 1")
            deadline = time.monotonic() + 30
            while True:
                _editor(endpoint, f"open({str(stats_path)!r}, 'w').write(json.dumps(lf.get_lod_stats()))")
                stats = json.loads(stats_path.read_text())
                if stats["active"] and 0 < stats["selected"] < count and not stats["transition_active"]:
                    break
                assert time.monotonic() < deadline, stats
                time.sleep(0.1)
            _capture(endpoint, tmp_path / "coarse-gut-warmup.rgb")
            gut_pixels, gut_shape = _capture(endpoint, tmp_path / "coarse-gut.rgb")
            _editor(endpoint, "lf.get_render_settings().raster_backend = 2")
            _capture(endpoint, tmp_path / "coarse-3dgs-warmup.rgb")
            gs_pixels, gs_shape = _capture(endpoint, tmp_path / "coarse-3dgs.rgb")
            assert gut_shape == gs_shape
            coarse_error = sum(abs(a - b) for a, b in zip(gut_pixels, gs_pixels)) / len(gut_pixels)
            (tmp_path / "coarse-comparison.json").write_text(json.dumps({"mean_error": coarse_error}))
            # Ray evaluation and EWA differ slightly at this finite distance.
            assert coarse_error < 3.0, coarse_error
