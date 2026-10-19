# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Relative GPU regression checks using isolated, externally supplied scenes.

The runner owns the app, GPU isolation and fixture copies. A manifest supplies
setup calls and alternating transform calls for each supported case class.
"""

import base64
import io
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path

import pytest

from test_render_on_demand_idle import _call, _initialize, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]
CASE_CLASSES = ("single", "multiple", "subset", "group", "nested_group")


def _apply(endpoint, calls):
    for call in calls:
        result = _tool(endpoint, call["name"], call.get("arguments", {}))
        assert "error" not in result and result.get("success", True), result
        if call["name"] == "editor_run":
            assert result.get("completed") and not result.get("timed_out"), result
            assert "Traceback" not in result.get("output", {}).get("text", ""), result


def _raster_times(log, offset):
    with log.open() as stream:
        stream.seek(offset)
        text = stream.read()
    return [float(value) for value in re.findall(
        r"vksplat\.gpu\.RasterizeForward took ([0-9.]+)ms", text
    )]


def _world_matrices(endpoint, nodes):
    return {
        node["name"]: _tool(endpoint, "transform_get", {"node": node["name"]})["nodes"][0]["world"]["matrix"]
        for node in nodes
    }


@pytest.mark.parametrize("count", [2, 10, 50])
@pytest.mark.parametrize("operation", ["rotate", "scale"])
def test_individual_drag_keeps_resident_render_inputs(count, operation, tmp_path):
    manifest_path = os.environ.get("LFS_TRANSFORM_PERF_MANIFEST")
    endpoint = os.environ.get("LFS_TRANSFORM_PERF_ENDPOINT")
    display = os.environ.get("LFS_TRANSFORM_PERF_DISPLAY")
    if not manifest_path or not endpoint or not display:
        pytest.skip("requires isolated app, real-input fixture coordinates and X11 display")
    manifest = json.loads(Path(manifest_path).read_text())
    case = next(case for case in manifest["cases"]
                if case["class"] == "multiple" and case["splat_nodes"] == count)
    coordinates = case["drag_coordinates"][operation]
    _initialize(endpoint)
    _apply(endpoint, [{"name": "editor_run", "arguments": {
        "code": "import lichtfeld as lf; lf.deselect_all()",
        "wait_for_completion": True, "timeout_ms": 20000, "show_console": False
    }}])
    _apply(endpoint, case["setup"])
    _apply(endpoint, [{"name": "editor_run", "arguments": {
        "code": "import lichtfeld as lf; lf.ui.set_multi_transform_mode(1)",
        "wait_for_completion": True, "timeout_ms": 20000, "show_console": False
    }}])
    _tool(endpoint, "ui_tool_invoke", {"tool_id": "builtin." + operation})
    time.sleep(0.75)
    nodes = _tool(endpoint, "scene_list_nodes")["nodes"]
    before = _world_matrices(endpoint, nodes)
    camera = _tool(endpoint, "camera_get")["camera"]
    camera_args = {key: camera[key] for key in ("eye", "target", "up", "fov_degrees")}
    log = Path(manifest["app_log"])
    offset = log.stat().st_size
    for index in range(28):
        args = dict(camera_args)
        for key in ("eye", "target"):
            args[key] = [camera_args[key][0] + (0.001 if index % 2 else -0.001), *camera_args[key][1:]]
        _tool(endpoint, "camera_set_view", args)
    pan_samples = _raster_times(log, offset)
    assert len(pan_samples) >= 20
    _tool(endpoint, "camera_set_view", camera_args)
    before_frames = _tool(endpoint, "runtime_frame_ledger")["frames"]
    offset = log.stat().st_size
    env = dict(os.environ, DISPLAY=display)

    def mouse(*args):
        subprocess.run(["xdotool", *map(str, args)], env=env, check=True,
                       capture_output=True, timeout=10)

    x0, y0, x1, y1 = coordinates
    mouse("mousemove", x0, y0)
    time.sleep(0.15)
    mouse("mousedown", 1)
    time.sleep(0.1)
    try:
        for index in range(1, 26):
            mouse("mousemove", round(x0 + (x1 - x0) * index / 25),
                  round(y0 + (y1 - y0) * index / 25))
            # Fence synthetic X11 input with a frame at the unchanged camera;
            # otherwise an idle window can coalesce the entire input sequence.
            _tool(endpoint, "camera_set_view", camera_args)
            time.sleep(0.04)
        changed = _world_matrices(endpoint, nodes)
        assert sum(changed[name] != matrix for name, matrix in before.items()) == count
    finally:
        mouse("mouseup", 1)
    _tool(endpoint, "camera_set_view", camera_args)
    time.sleep(0.3)
    after_frames = _tool(endpoint, "runtime_frame_ledger")["frames"]
    samples = _raster_times(log, offset)
    assert len(samples) >= 12, samples
    settings_frames = (after_frames["view_renders_by_reason"]["SettingsChange"]
                       - before_frames["view_renders_by_reason"]["SettingsChange"])
    ratio = statistics.median(samples) / statistics.median(pan_samples)
    (tmp_path / "drag.json").write_text(json.dumps({
        "count": count, "operation": operation, "pan_ms": pan_samples,
        "drag_ms": samples, "ratio": ratio, "settings_frames": settings_frames,
        "before_frames": before_frames, "after_frames": after_frames
    }, indent=2))
    assert ratio <= 1.5 and settings_frames == 0


@pytest.mark.parametrize("case_class", CASE_CLASSES)
def test_transform_cost_matches_camera_movement(case_class, tmp_path):
    manifest_path = os.environ.get("LFS_TRANSFORM_PERF_MANIFEST")
    endpoint = os.environ.get("LFS_TRANSFORM_PERF_ENDPOINT")
    if not manifest_path or not endpoint:
        pytest.skip("requires an isolated running app and transform performance manifest")
    manifest = json.loads(Path(manifest_path).read_text())
    cases = [case for case in manifest["cases"] if case["class"] == case_class]
    assert cases, f"configured manifest is missing {case_class} coverage"
    log = Path(manifest["app_log"])
    _initialize(endpoint)
    tools = _call(endpoint, "tools/list")["tools"]
    known = {tool["name"] for tool in tools}
    for uri in ("runtime/catalog", "runtime/state", "ui/state", "scene/state", "selection/current"):
        _call(endpoint, "resources/read", {"uri": "lichtfeld://" + uri})
    results = []
    for case in cases:
        started_at = time.time()
        for call in case.get("setup", []) + case["updates"]:
            assert call["name"] in known, call
        _apply(endpoint, case.get("setup", []))
        _tool(endpoint, "render_settings_set", {"raster_backend": manifest.get("backend", "3dgs")})
        nodes = _tool(endpoint, "scene_list_nodes")["nodes"]
        if "gaussian_count" in case:
            assert sum(node.get("gaussian_count", 0) for node in nodes) == case["gaussian_count"], nodes
        if "splat_nodes" in case:
            assert sum(node["type"] == "splat" for node in nodes) == case["splat_nodes"], nodes
        before_transforms = _world_matrices(endpoint, nodes)
        _apply(endpoint, [case["updates"][0]])
        changed = {
            name for name, matrix in _world_matrices(endpoint, nodes).items()
            if matrix != before_transforms[name]
        }
        target = case["updates"][0].get("arguments", {}).get("node")
        if case_class == "multiple":
            assert len(changed) == case["splat_nodes"], (case["name"], changed)
        elif case_class in ("single", "subset"):
            assert changed == {target}, (case["name"], changed, target)
        else:
            assert target in changed, (case["name"], changed, target)
            assert any(node["type"] == "splat" and node["name"] in changed for node in nodes), (
                case["name"], changed
            )
        _apply(endpoint, [case["updates"][1]])
        assert _world_matrices(endpoint, nodes) == before_transforms, case["name"]
        untouched = {
            name: _tool(endpoint, "transform_get", {"node": name})["nodes"][0]["world"]["matrix"]
            for name in case.get("untouched", [])
        }
        camera = _tool(endpoint, "camera_get")["camera"]
        camera_args = {key: camera[key] for key in ("eye", "target", "up", "fov_degrees")}
        camera_updates = []
        for sign in (1, -1):
            args = dict(camera_args)
            for key in ("eye", "target"):
                args[key] = [camera_args[key][0] + sign * 0.001, *camera_args[key][1:]]
            camera_updates.append({"name": "camera_set_view", "arguments": args})
        repetitions = []
        try:
            for mode in ((0, 1) if case_class == "multiple" else (0,)):
                if case_class == "multiple":
                    _apply(endpoint, [{"name": "editor_run", "arguments": {
                        "code": f"import lichtfeld as lf; lf.ui.set_multi_transform_mode({mode})",
                        "wait_for_completion": True, "timeout_ms": 20000, "show_console": False
                    }}])
                for _ in range(3):
                    medians = []
                    viewport_fps = []
                    for updates in (camera_updates, case["updates"]):
                        for index in range(8):
                            _apply(endpoint, [updates[index % len(updates)]])
                        offset = log.stat().st_size
                        for index in range(20):
                            _apply(endpoint, [updates[index % len(updates)]])
                        _tool(endpoint, "camera_get")
                        samples = _raster_times(log, offset)
                        assert len(samples) >= 12, (case["name"], mode, samples)
                        medians.append(statistics.median(samples))
                        viewport_fps.append(_tool(endpoint, "runtime_frame_ledger")["frames"]["viewport_fps"])
                    repetitions.append({"mode": mode, "pan_ms": medians[0], "transform_ms": medians[1],
                                        "ratio": medians[1] / medians[0],
                                        "pan_fps": viewport_fps[0], "transform_fps": viewport_fps[1]})
                results.append({"name": case["name"], "mode": mode, "repetitions": repetitions[-3:],
                                "started_at": started_at, "finished_at": time.time()})
            for name, matrix in untouched.items():
                assert _tool(endpoint, "transform_get", {"node": name})["nodes"][0]["world"]["matrix"] == matrix
        finally:
            _tool(endpoint, "camera_set_view", camera_args)
            _apply(endpoint, case.get("restore", []))
            (tmp_path / "timings.json").write_text(json.dumps(results, indent=2))
    assert all(statistics.median(sample["ratio"] for sample in case["repetitions"]) <= 1.5
               for case in results), results


@pytest.mark.parametrize("backend", ["3dgs", "3dgut"])
def test_transform_images_match_reference(backend, tmp_path, numpy):
    manifest_path = os.environ.get("LFS_TRANSFORM_IMAGE_MANIFEST")
    endpoint = os.environ.get("LFS_TRANSFORM_PERF_ENDPOINT")
    if not manifest_path or not endpoint:
        pytest.skip("requires an isolated running app and reference image manifest")
    image = pytest.importorskip("PIL.Image")
    manifest = json.loads(Path(manifest_path).read_text())
    _initialize(endpoint)
    metrics = []
    for case in manifest["cases"]:
        _apply(endpoint, case["setup"])
        _tool(endpoint, "render_settings_set", {"raster_backend": backend})
        for state in ("identity", "transformed"):
            _apply(endpoint, case[state])
            result = _call(endpoint, "tools/call", {"name": "render_capture", "arguments": {}})
            assert not result.get("isError"), result
            images = [block for block in result["content"] if block["type"] == "image"]
            assert len(images) == 1, result
            actual = base64.b64decode(images[0]["data"])
            reference = Path(case["references"][backend][state]).read_bytes()
            (tmp_path / f'{case["name"]}-{state}.png').write_bytes(actual)
            actual_pixels = numpy.asarray(image.open(io.BytesIO(actual))).astype(numpy.int16)
            reference_pixels = numpy.asarray(image.open(io.BytesIO(reference))).astype(numpy.int16)
            assert actual_pixels.shape == reference_pixels.shape
            difference = numpy.abs(actual_pixels - reference_pixels)
            metrics.append({"case": case["name"], "backend": backend, "state": state,
                            "max_channel_delta": int(difference.max()),
                            "mean_channel_delta": float(difference.mean()),
                            "changed_pixels": int(numpy.count_nonzero(numpy.any(difference != 0, axis=2)))})
            (tmp_path / "pixel_metrics.json").write_text(json.dumps(metrics, indent=2))
            # Allow sparse Vulkan float-rounding differences while rejecting
            # visible or widespread changes. Images use 8-bit output channels.
            changed_fraction = metrics[-1]["changed_pixels"] / difference.shape[0] / difference.shape[1]
            assert difference.max() <= 3 and difference.mean() <= 0.0002 and changed_fraction <= 0.001, (
                case["name"], backend, state, difference.max(), difference.mean()
            )
