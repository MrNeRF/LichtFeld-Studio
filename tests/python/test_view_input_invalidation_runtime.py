# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run against a disposable viewer via LFS_TEST_MCP_ENDPOINT.

These integration tests change runtime state. Never point them at a user's viewer.
"""

import json
import os
from pathlib import Path
import time
import urllib.request

import pytest


@pytest.fixture(scope="module")
def viewer():
    endpoint = os.environ.get("LFS_TEST_MCP_ENDPOINT")
    if not endpoint:
        pytest.skip("Requires an explicitly configured disposable MCP viewer")

    def request(method, params=None):
        payload = {"jsonrpc": "2.0", "id": 1, "method": method}
        if params is not None:
            payload["params"] = params
        http = urllib.request.Request(
            endpoint, json.dumps(payload).encode(),
            {"Content-Type": "application/json", "Accept": "application/json"},
        )
        with urllib.request.urlopen(http, timeout=30) as response:
            result = json.load(response)
        assert "error" not in result, result
        return result["result"]

    request("initialize", {
        "protocolVersion": "2024-11-05", "capabilities": {},
        "clientInfo": {"name": "view-input-regression", "version": "1"},
    })
    tools = {tool["name"]: tool for tool in request("tools/list")["tools"]}
    request("resources/list")
    for uri in (
        "lichtfeld://runtime/catalog", "lichtfeld://runtime/state", "lichtfeld://ui/state",
        "lichtfeld://scene/state", "lichtfeld://selection/current",
    ):
        request("resources/read", {"uri": uri})

    def call(name, **arguments):
        assert name in tools
        result = request("tools/call", {"name": name, "arguments": arguments})
        assert not result.get("isError"), result
        data = result.get("structuredContent") or json.loads(result["content"][0]["text"])
        assert data.get("success", True), data
        return data

    dataset = Path(os.environ.get(
        "LFS_TEST_DATASET_PATH",
        Path(__file__).resolve().parents[2] / "build-windows-release" / "verify-2625-dataset",
    ))
    assert dataset.is_dir(), "Requires the local synthetic dataset fixture"
    call("scene_load_dataset", path=str(dataset), strategy="mrnf", max_iterations=3, min_track_length=0)
    result = call("runtime_job_wait", job_id="import.dataset", until="inactive", timeout_ms=20000)
    assert not result.get("timed_out"), result
    time.sleep(1)
    return call


def frames(viewer):
    return viewer("runtime_frame_ledger")["frames"]


@pytest.mark.parametrize("field", ["scene_generation", "selection_generation", "render_settings_generation"])
def test_changed_view_input_renders_once_then_returns_to_idle(viewer, field):
    before = frames(viewer)
    result = viewer("editor_run", code=(
        "from lfs_plugins.ui.store import RuntimeState\n"
        f"RuntimeState.{field}.value += 1\n"
    ), show_console=False, timeout_ms=10000)
    assert result["completed"] and "Traceback" not in result["output"]["text"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        after = frames(viewer)
        if after["views_rendered"] > before["views_rendered"]:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"Changing {field} left the cached view unrendered")
    time.sleep(0.3)
    settled = frames(viewer)
    time.sleep(0.3)
    idle = frames(viewer)
    assert settled["views_rendered"] == before["views_rendered"] + 1
    assert idle["views_rendered"] == settled["views_rendered"]
    assert idle["stale_detections"] == before["stale_detections"]


def test_gui_only_account_update_does_not_render_view(viewer):
    before = frames(viewer)
    result = viewer("editor_run", code=(
        "from lfs_plugins.ui.store import RuntimeState\n"
        "RuntimeState.account_state.value = dict(RuntimeState.account_state.value, tooltip='test update')\n"
    ), show_console=False, timeout_ms=10000)
    assert result["completed"] and "Traceback" not in result["output"]["text"]
    time.sleep(0.3)
    after = frames(viewer)
    assert after["views_rendered"] == before["views_rendered"]
    assert after["stale_detections"] == before["stale_detections"]


def test_camera_invalidation_remains_surgical(viewer):
    before = frames(viewer)
    viewer("camera_set_view", eye=[1, 1, 3], target=[0, 0, 0])
    time.sleep(0.3)
    after = frames(viewer)
    assert after["views_rendered"] > before["views_rendered"]
    assert after["view_renders_by_reason"]["SceneChange"] == before["view_renders_by_reason"]["SceneChange"]
    assert after["stale_detections"] == before["stale_detections"]
