"""Source invalidation contract against an isolated, running GUI over MCP.

Set LFS_VIEW_INVALIDATION_ENDPOINT and LFS_VIEW_INVALIDATION_LOG. The test
creates its own synthetic splats; it never loads a user's project. An optional
LFS_VIEW_INVALIDATION_ACTIONS manifest adds dataset/GUI-specific operations.
"""
from __future__ import annotations

import json
import os
import struct
import time
from pathlib import Path

import pytest

from test_render_on_demand_idle import _call, _initialize, _ledger, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _python(endpoint, code):
    result = _tool(endpoint, "editor_run", {
        "code": "import lichtfeld as lf\n" + code,
        "show_console": False, "timeout_ms": 10000,
    })
    assert result["completed"] and not result["timed_out"], result
    assert "Traceback" not in result["output"]["text"], result
    return result


def _settle(endpoint, quiet_seconds=2):
    previous = _ledger(endpoint)["frames_presented"]
    deadline = time.monotonic() + 20
    stable = time.monotonic()
    while time.monotonic() < deadline:
        time.sleep(0.1)
        current = _ledger(endpoint)["frames_presented"]
        if current != previous:
            previous, stable = current, time.monotonic()
        # Import feedback and camera-image completion can post GUI-only work
        # after a one-second gap. Measure idle only after those have settled.
        if time.monotonic() - stable >= quiet_seconds:
            return
    pytest.fail("View did not return to idle")


def _check_action(endpoint, log, action, *, renders=True, quiet_seconds=2):
    _settle(endpoint, quiet_seconds)
    before = _ledger(endpoint)
    offset = log.stat().st_size
    result = action()
    assert not isinstance(result, dict) or not result.get("error"), result
    _settle(endpoint, quiet_seconds)
    after = _ledger(endpoint)
    with log.open("rb") as handle:
        handle.seek(offset)
        text = handle.read().decode(errors="replace")
    if after["frames_presented"] > before["frames_presented"]:
        assert "frame #" in text, "Run the GUI with perf logging so backup redraws cannot be missed"
    assert after["stale_detections"] == before["stale_detections"], text[-4000:]
    # Read the whole action interval: last_frame_details alone can miss a
    # backup render followed by a GUI-only frame.
    assert "view_inputs_changed" not in text, text[-4000:]
    assert "view_inputs_changed" not in after.get("last_frame_details", [])
    if renders:
        assert after["views_rendered"] > before["views_rendered"], "Changed input was never rendered"
    time.sleep(1)
    idle = _ledger(endpoint)
    assert idle["views_rendered"] == after["views_rendered"]
    assert idle["frames_presented"] == after["frames_presented"]


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    endpoint = os.environ.get("LFS_VIEW_INVALIDATION_ENDPOINT")
    log_path = os.environ.get("LFS_VIEW_INVALIDATION_LOG")
    if not endpoint or not log_path:
        pytest.skip("Requires an isolated GUI endpoint and its trace/perf log")
    log = Path(log_path)
    assert log.is_file()
    _initialize(endpoint)
    tools = {t["name"] for t in _call(endpoint, "tools/list")["tools"]}
    assert {"editor_run", "runtime_frame_ledger", "scene_load_ply"} <= tools
    path = tmp_path_factory.mktemp("redraw") / "synthetic.ply"
    fields = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
              "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    with path.open("wb") as handle:
        handle.write(("ply\nformat binary_little_endian 1.0\nelement vertex 400\n"
                      + "".join(f"property float {field}\n" for field in fields)
                      + "end_header\n").encode())
        for y in range(20):
            for x in range(20):
                handle.write(struct.pack("<17f", (x - 9.5) * .1, (y - 9.5) * .1, 0,
                                         0, 0, 0, 1, .2, .1, 3, -3, -3, -3, 1, 0, 0, 0))
    _python(endpoint, "lf.clear_scene()")
    _tool(endpoint, "scene_load_ply", {"path": str(path)})
    _python(endpoint, 'lf.set_camera((0, 0, 4), (0, 0, 0))\nlf.select_node("synthetic")')
    _settle(endpoint, quiet_seconds=3.2)
    yield endpoint, log, path
    _python(endpoint, "lf.clear_scene()")


@pytest.mark.parametrize("code", [
    "from lfs_plugins.ui.store import RuntimeState\nRuntimeState.scene_generation.value += 1",
    "from lfs_plugins.ui.store import RuntimeState\nRuntimeState.selection_generation.value += 1",
    "from lfs_plugins.ui.store import RuntimeState\nRuntimeState.render_settings_generation.value += 1",
    "lf.get_scene().invalidate_cache()",
    "lf.deselect_all()",
])
def test_input_publication_requests_render(runtime, code):
    endpoint, log, _ = runtime
    _python(endpoint, 'lf.select_node("synthetic")')
    try:
        _check_action(endpoint, log, lambda: _python(endpoint, code))
    finally:
        # Prevent a known-bad baseline fingerprint contaminating the next case.
        _python(endpoint, "lf.set_camera((0, 0, 4), (0, 0, 0))")


def test_editing_actions_request_render(runtime):
    endpoint, log, path = runtime
    actions = [
        ("selection_click", dict(x=300, y=334, radius=80, camera_index=-1)),
        ("selection_clear", {}),
        ("selection_rect", dict(x0=20, y0=20, x1=580, y1=640, camera_index=-1)),
        ("selection_brush", dict(x=300, y=334, radius=150, camera_index=-1)),
        ("selection_lasso", dict(points=[[10, 10], [590, 10], [590, 650], [10, 650]], camera_index=-1)),
        ("render_settings_set", {"background_color": [.1, .2, .3]}),
        ("gaussians_write", dict(node="synthetic", field="opacity_raw", indices=[0], values=[1])),
        ("transform_translate", dict(node="synthetic", value=[.1, 0, 0])),
        ("transform_rotate", dict(node="synthetic", value=[0, .1, 0])),
        ("transform_scale", dict(node="synthetic", value=[1.1, 1, 1])),
        ("transform_set", dict(node="synthetic", translation=[0, 0, 0])),
        ("operator_invoke", {"operator_id": "ed.undo"}),
        ("operator_invoke", {"operator_id": "ed.redo"}),
        ("scene_set_node_visibility", dict(name="synthetic", visible=False)),
        ("scene_set_node_visibility", dict(name="synthetic", visible=True)),
        ("operator_invoke", dict(operator_id="crop_box.add", arguments={"node": "synthetic"})),
        ("operator_invoke", dict(operator_id="crop_box.set", arguments={
            "node": "synthetic", "min": [-.5, -.5, -.5], "max": [.5, .5, .5],
            "enabled": True, "show": True, "use": True})),
    ]
    for tool, arguments in actions:
        _check_action(endpoint, log, lambda: _tool(endpoint, tool, arguments))
        if tool.startswith("selection_"):
            _tool(endpoint, "selection_get", {"max_indices": 10})
    _python(endpoint, 'lf.select_node("synthetic")')
    for code in [
        "lf.ui.select_all_gaussians()", "lf.ui.invert_gaussian_selection()",
        "lf.ui.deselect_all_gaussians()", "lf.get_scene().set_selection([0, 1, 2])",
        "lf.get_scene().clear_selection()", "lf.selection.by_opacity(.9, 1)",
        "lf.get_render_settings().show_grid = False", "lf.set_depth_view(True)",
        "lf.set_depth_view(False)", "lf.toggle_independent_split_view()",
        'lf.set_camera((1, 0, 4), (0, 0, 0), panel="right")',
        "lf.toggle_independent_split_view()",
    ]:
        _check_action(endpoint, log, lambda: _python(endpoint, code),
                      renders=code != "lf.ui.deselect_all_gaussians()")
        _tool(endpoint, "selection_get", {"max_indices": 10})
    _check_action(endpoint, log, lambda: _tool(endpoint, "scene_delete_node", {"name": "synthetic"}))
    # Import completion has a three-second, one-shot dismissal timer.
    _check_action(endpoint, log, lambda: _tool(endpoint, "scene_load_ply", {"path": str(path)}),
                  quiet_seconds=3.2)


def test_additional_runtime_actions(runtime):
    manifest = os.environ.get("LFS_VIEW_INVALIDATION_ACTIONS")
    if not manifest:
        pytest.skip("No additional GUI/dataset action manifest supplied")
    endpoint, log, _ = runtime
    for action in json.loads(Path(manifest).read_text()):
        _check_action(endpoint, log, lambda: _tool(endpoint, action["tool"], action.get("arguments", {})),
                      renders=action.get("renders", True))


def test_gui_selection_and_resize(runtime):
    import re
    import subprocess

    endpoint, log, _ = runtime
    display = os.environ.get("LFS_VIEW_INVALIDATION_DISPLAY")
    if not display:
        pytest.skip("Requires LFS_VIEW_INVALIDATION_DISPLAY for native GUI input")
    env = dict(os.environ, DISPLAY=display)

    def xdo(*args):
        return subprocess.check_output(["xdotool", *map(str, args)], env=env, text=True).strip()

    window = xdo("search", "--name", "LichtFeld Studio").splitlines()[0]
    xdo("windowmove", window, 0, 0, "windowsize", window, 1280, 720,
        "windowfocus", window, "key", "Escape")
    _python(endpoint, 'lf.select_node("synthetic")\nlf.set_camera((0, 0, 4), (0, 0, 0))')
    _settle(endpoint)
    output = _python(endpoint, 'print("BOUNDS", *map(int, lf.selection.get_viewport_bounds()))')["output"]["text"]
    match = re.search(r"BOUNDS\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", output)
    assert match, output
    x, y, width, height = map(int, match.groups())
    cx, cy = x + width // 2, y + height // 2
    for mode in ["centers", "rectangle", "lasso"]:
        _tool(endpoint, "ui_tool_invoke", {"tool_id": "builtin.select", "submode_id": mode})
        _check_action(endpoint, log, lambda: xdo(
            "mousemove", cx - 80, cy - 80, "sleep", .1, "mousedown", 1,
            "mousemove", cx + 80, cy - 80, "sleep", .1,
            "mousemove", cx + 80, cy + 80, "sleep", .1,
            "mousemove", cx - 80, cy + 80, "sleep", .1, "mouseup", 1), renders=False)
        _tool(endpoint, "selection_get", {"max_indices": 10})
    _tool(endpoint, "ui_tool_invoke", {"tool_id": "builtin.select", "submode_id": "centers"})
    for key in ["ctrl+a", "ctrl+i", "ctrl+a", "ctrl+d"]:
        _check_action(endpoint, log, lambda: xdo("mousemove", cx, cy, "key", key))
        _tool(endpoint, "selection_get", {"max_indices": 10})
    _check_action(endpoint, log, lambda: xdo("mousemove", cx, cy, "click", 1))
    _tool(endpoint, "selection_get", {"max_indices": 10})
    _tool(endpoint, "ui_tool_clear_active")
    xdo("mousemove", 20, 100)
    _check_action(endpoint, log, lambda: xdo("windowsize", window, 1400, 800))


def test_gt_camera_switch(runtime, tmp_path):
    import zlib

    endpoint, log, path = runtime
    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    sparse = dataset / "sparse" / "0"
    sparse.mkdir(parents=True)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    for index in range(2):
        pixels = (b"\0" + bytes([100 + index * 80, 70, 30]) * 640) * 480
        png = (b"\x89PNG\r\n\x1a\n"
               + chunk(b"IHDR", struct.pack(">2I5B", 640, 480, 8, 2, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))
        (dataset / "images" / f"{index}.png").write_bytes(png)
    (sparse / "cameras.txt").write_text("1 PINHOLE 640 480 500 500 320 240\n")
    (sparse / "images.txt").write_text(
        "1 1 0 0 0 0 0 4 1 0.png\n\n2 1 0 0 0 0.3 0 4 1 1.png\n\n")
    (sparse / "points3D.txt").write_text("".join(
        f"{y * 20 + x + 1} {(x - 9.5) * .1} {(y - 9.5) * .1} 0 200 100 50 0\n"
        for y in range(20) for x in range(20)))
    _tool(endpoint, "scene_load_dataset", {
        "path": str(dataset), "output_path": str(tmp_path / "output"),
        "min_track_length": 0, "max_iterations": 3,
    })
    job = _tool(endpoint, "runtime_job_wait", {
        "job_id": "import.dataset", "until": "inactive", "timeout_ms": 20000,
    })
    assert not job.get("timed_out") and job.get("status") != "failed", job
    _python(endpoint, f"loaded = lf.io.load({str(path)!r})\ns = loaded.splat_data\n"
            "lf.get_scene().add_splat('synthetic', s.means_raw, s.sh0_raw, s.shN_raw, "
            "s.scaling_raw, s.rotation_raw, s.opacity_raw, s.active_sh_degree, s.scene_scale)")
    _tool(endpoint, "render_settings_set", {"point_cloud_mode": False})
    _settle(endpoint, quiet_seconds=3.2)
    for code in ["lf.ui.go_to_camera_view(0)", "lf.ui.toggle_gt_comparison()",
                 "assert lf.ui.is_gt_comparison_active(); lf.ui.go_to_camera_view(1)",
                 "lf.ui.toggle_gt_comparison()"]:
        _check_action(endpoint, log, lambda: _python(endpoint, code))


def test_selection_by_description(runtime):
    endpoint, log, _ = runtime
    # This route needs an external vision provider. Do not replace its response
    # with a mock: that would not exercise the published selection mutation.
    def describe():
        response = _call(endpoint, "tools/call", {
            "name": "selection_by_description",
            "arguments": {"description": "the orange square", "camera_index": 0},
        })
        result = response.get("structuredContent", response)
        if "No API key found" in str(result.get("error", "")):
            pytest.skip("Selection by description requires a vision-provider API key")
        assert not response.get("isError"), response
        return result

    _check_action(endpoint, log, describe, renders=False)
    _tool(endpoint, "selection_get", {"max_indices": 10})
