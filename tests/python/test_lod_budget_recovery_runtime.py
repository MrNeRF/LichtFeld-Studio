# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""LOD budget recovery without camera input, using a generated RAD hierarchy."""

import base64
import io
import json
import math
import os
import struct
import subprocess
import time
from pathlib import Path

import pytest

from runtime_test_app import isolated_app
from test_render_on_demand_idle import _call, _initialize, _ledger, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

SCENE_SIDE = 256
FULL_BUDGET = SCENE_SIDE * SCENE_SIDE


@pytest.fixture
def lod_scene(tmp_path):
    directory = tmp_path / "lod-budget"
    directory.mkdir()
    model = directory / "surface.ply"
    properties = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
                  "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    # Enough detail to expose coarse-budget recovery, with a small runtime fixture.
    side = SCENE_SIDE
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {side * side}\n"
    header += "".join(f"property float {name}\n" for name in properties) + "end_header\n"
    with model.open("wb") as output:
        output.write(header.encode())
        for row in range(side):
            for column in range(side):
                x, y = (column - side / 2) * 0.01, (row - side / 2) * 0.01
                z = 0.1 * math.sin(x * 7) * math.cos(y * 9)
                scale = math.log(0.005 + 0.002 * (1 + math.sin(x * 13 + y * 17)))
                output.write(struct.pack("<17f", x, y, z, 0, 0, 0,
                                         math.sin(x * 31), math.cos(y * 37), math.sin((x + y) * 29),
                                         3, scale, scale, scale, 1, 0, 0, 0))
    root = Path(__file__).resolve().parents[2]
    build = Path(os.environ.get("LFS_TEST_BUILD_DIR", root / "build"))
    executable = Path(os.environ.get("LFS_EXECUTABLE", build / "LichtFeld-Studio"))
    hierarchy = directory / "surface.rad"
    with (directory / "convert.log").open("w") as log:
        subprocess.run([str(executable), "convert", str(model), str(hierarchy),
                        "--lod-builder", "octree", "--no-provenance"],
                       check=True, timeout=120, stdout=log, stderr=subprocess.STDOUT)
    with isolated_app(directory) as endpoint:
        yield endpoint, str(hierarchy)


def _editor(endpoint, code):
    result = _tool(endpoint, "editor_run", {
        "code": "import json, lichtfeld as lf\n" + code,
        "show_console": False,
        "wait_for_completion": True,
        "timeout_ms": 10000,
    })
    assert result["completed"] and result["success"], result
    return result["output"]["text"]


def _stats(endpoint, path):
    # Console output is line-wrapped for display; use a file for structured data.
    _editor(endpoint, f"open({str(path)!r}, 'w').write(json.dumps(lf.get_lod_stats()))")
    return json.loads(path.read_text())


def _capture(endpoint):
    image = pytest.importorskip("PIL.Image")
    result = _call(endpoint, "tools/call", {"name": "render_capture", "arguments": {}})
    assert not result.get("isError"), result
    block = next(block for block in result["content"] if block["type"] == "image")
    return image.open(io.BytesIO(base64.b64decode(block["data"]))).convert("RGB")


def _quiet(endpoint):
    deadline = time.monotonic() + 45
    previous = _ledger(endpoint)
    unchanged_since = time.monotonic()
    while time.monotonic() < deadline:
        time.sleep(0.2)
        current = _ledger(endpoint)
        if current["views_rendered"] != previous["views_rendered"]:
            unchanged_since = time.monotonic()
        if time.monotonic() - unchanged_since > 1.5:
            return current
        previous = current
    pytest.fail(f"LOD did not return to idle: {previous}")


@pytest.mark.parametrize("low_budget", [100, 1000])
def test_lod_budget_recovers_without_camera_input(lod_scene, tmp_path, low_budget):
    endpoint, fixture = lod_scene
    image_chops = pytest.importorskip("PIL.ImageChops")
    image_stat = pytest.importorskip("PIL.ImageStat")
    _initialize(endpoint)
    tools = {tool["name"] for tool in _call(endpoint, "tools/list")["tools"]}
    assert {"scene_load_ply", "camera_reset", "camera_get", "editor_run",
            "render_capture", "runtime_frame_ledger"} <= tools
    _tool(endpoint, "scene_load_ply", {"path": fixture})
    # Stay within the panel's full-quality limit so baseline and restoration
    # request the same budget even after the hierarchy statistics arrive.
    _editor(endpoint, f"lf.get_render_settings().lod_max_splats = {FULL_BUDGET}")
    # Import and the first shader compilation may outlive an idle frame window.
    deadline = time.monotonic() + 60
    while True:
        initial = _stats(endpoint, tmp_path / "initial.json")
        if initial["active"] and initial["selected"] > 10000:
            break
        assert time.monotonic() < deadline, initial
        time.sleep(0.2)
    _tool(endpoint, "camera_reset")
    _quiet(endpoint)
    baseline = _stats(endpoint, tmp_path / "baseline.json")
    assert baseline["active"] and baseline["selected"] > 10000, baseline
    camera = _tool(endpoint, "camera_get")["camera"]
    reference = _capture(endpoint)
    reference.save(tmp_path / "before.png")

    _editor(endpoint, f"lf.get_render_settings().lod_max_splats = {low_budget}")
    _quiet(endpoint)
    low = _stats(endpoint, tmp_path / "low.json")
    # The GPU selector permits 20% headroom for hierarchy transitions.
    assert 0 < low["selected"] <= low_budget * 1.2, low
    _editor(endpoint, f"lf.get_render_settings().lod_max_splats = {FULL_BUDGET}")
    _quiet(endpoint)
    restored = _stats(endpoint, tmp_path / "restored.json")
    recovered = _capture(endpoint)
    recovered.save(tmp_path / "restored.png")
    difference = sum(image_stat.Stat(image_chops.difference(reference, recovered)).mean) / 3
    (tmp_path / "stats.json").write_text(json.dumps({
        "baseline": baseline, "low": low, "restored": restored,
        "image_mean_absolute_difference": difference,
    }, indent=2))
    assert _tool(endpoint, "camera_get")["camera"] == camera
    assert restored["budget"] == baseline["budget"]
    assert restored["selected"] >= baseline["selected"] * 0.75, restored
    # The existing controller deadband permits nearby blended hierarchy cuts.
    # Require recovery within one intensity level on the 0-255 image scale.
    assert difference < 1.0, difference

    # Reading/capturing a settled cut and setting the same budget must not
    # leave a continuous render loop running.
    _editor(endpoint, f"lf.get_render_settings().lod_max_splats = {FULL_BUDGET}")
    settled = _quiet(endpoint)
    time.sleep(2)
    assert _ledger(endpoint)["views_rendered"] == settled["views_rendered"]
