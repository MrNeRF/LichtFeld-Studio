# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Node selection stays visually stable and does not sustain render demand."""

import base64
import io
import json
import os
import time

import pytest

from test_render_on_demand_idle import _call, _initialize, _ledger, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.fixture
def selection_scene():
    endpoint = os.environ.get("LFS_SELECTION_ENDPOINT")
    project = os.environ.get("LFS_SELECTION_PROJECT")
    if not endpoint or not project:
        pytest.skip("requires an isolated app and a single-model project")
    _initialize(endpoint)
    tools = _call(endpoint, "tools/list")["tools"]
    assert {"scene_select_node", "render_capture", "runtime_frame_ledger"} <= {
        tool["name"] for tool in tools
    }
    for uri in ("runtime/catalog", "runtime/state", "ui/state", "scene/state", "selection/current"):
        _call(endpoint, "resources/read", {"uri": "lichtfeld://" + uri})
    _tool(endpoint, "project_open", {"path": project, "discard_changes": True})
    setup = _tool(endpoint, "editor_run", {
        "code": "import lichtfeld as lf\nlf.switch_to_edit_mode()\n"
                "lf.ui.set_panel_enabled('lfs.asset_manager', False)\n"
                "lf.get_render_settings().desaturate_unselected = False",
        "wait_for_completion": True, "show_console": False, "timeout_ms": 10000,
    })
    assert setup.get("completed") and not setup.get("timed_out"), setup
    assert "Traceback" not in setup.get("output", {}).get("text", ""), setup
    nodes = _tool(endpoint, "scene_list_nodes")["nodes"]
    models = [node["name"] for node in nodes if node["type"] == "splat"]
    assert len(models) == 1
    _tool(endpoint, "scene_add_group", {"name": "SelectionControl"})
    _tool(endpoint, "render_settings_set", {"show_grid": False, "show_camera_frustums": False})
    yield endpoint, models[0]


def _capture(endpoint, path):
    from PIL import Image

    result = _call(endpoint, "tools/call", {"name": "render_capture", "arguments": {}})
    assert not result.get("isError"), result
    image = next(item for item in result["content"] if item["type"] == "image")
    data = base64.b64decode(image["data"])
    path.write_bytes(data)
    return Image.open(io.BytesIO(data)).convert("RGB")


@pytest.mark.parametrize("backend", ["3dgs", "3dgut"])
@pytest.mark.parametrize("tool", ["builtin.select", "builtin.translate"])
@pytest.mark.parametrize("models", [1, 2])
def test_node_selection_has_no_transient_frames(selection_scene, backend, tool, models, tmp_path):
    from PIL import ImageChops

    endpoint, model = selection_scene
    if models == 2:
        _tool(endpoint, "scene_duplicate_node", {"name": model})
    _tool(endpoint, "render_settings_set", {"raster_backend": backend})
    _tool(endpoint, "ui_tool_invoke", {"tool_id": tool})
    _tool(endpoint, "scene_select_node", {"name": "SelectionControl"})
    time.sleep(1)
    _ledger(endpoint, reset=True)
    started = time.monotonic()
    _tool(endpoint, "scene_select_node", {"name": model})
    active = _capture(endpoint, tmp_path / "active.png")
    elapsed = time.monotonic() - started
    time.sleep(0.7)
    ledger = _ledger(endpoint)
    settled = _capture(endpoint, tmp_path / "settled.png")
    difference = ImageChops.difference(active, settled)
    max_error = max(high for low, high in difference.getextrema())
    (tmp_path / "evidence.json").write_text(json.dumps({
        "capture_seconds": elapsed, "max_channel_error": max_error, "ledger": ledger,
    }, indent=2))
    assert elapsed < 0.5, "capture missed the former animation window"
    assert max_error <= 1, "node selection changed after its initial redraw"
    assert ledger["view_renders_by_reason"]["Overlay"] == 0, ledger
    assert not ledger["live_holders"], ledger
    assert not ledger["stale_detections"], ledger
