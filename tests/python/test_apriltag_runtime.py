"""AprilTag bindings and job discovery against an isolated GUI.

Set LFS_APRILTAG_ENDPOINT to an otherwise unused GUI MCP endpoint.
These tests clear its scene and do not require an external dataset.
"""
import os

import pytest

from test_render_on_demand_idle import _call, _initialize, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.fixture
def endpoint():
    endpoint = os.environ.get("LFS_APRILTAG_ENDPOINT")
    if not endpoint:
        pytest.skip("Requires an isolated GUI at LFS_APRILTAG_ENDPOINT")
    _initialize(endpoint)
    return endpoint


def test_python_background_editor_snapshots_and_applies_on_viewer(endpoint):
    result = _tool(endpoint, "editor_run", {
        "code": """import lichtfeld as lf
lf.clear_scene()
detection = lf.detect_scene_apriltags()
assert detection['camera_count'] == 0, detection
result = lf.scale_scene_by_apriltags([(1, 2, 1.0)])
assert not result['applied'] and not result['success'], result
print('apriltag worker completed')
""",
        "show_console": False,
        "timeout_ms": 10000,
    })
    assert result["completed"] and not result["timed_out"], result
    text = result["output"]["text"]
    assert "Traceback" not in text, text
    assert "apriltag worker completed" in text


def test_mcp_scan_returns_registered_job_and_result(endpoint):
    tools = {tool["name"]: tool for tool in _call(endpoint, "tools/list")["tools"]}
    scan = tools["transform_detect_apriltags"]
    assert scan["_meta"]["app.lichtfeld/long_running"] is True
    result = _tool(endpoint, "transform_detect_apriltags")
    assert result["started"] and result["job_id"] == "apriltag.scan", result
    # This read must remain available while detection runs.
    state = _tool(endpoint, "runtime_job_describe", {"job_id": "apriltag.scan"})
    assert state["id"] == "apriltag.scan", state
    assert state["cancel_supported"], state
    state = _tool(endpoint, "runtime_job_wait", {
        "job_id": "apriltag.scan", "until": "inactive", "timeout_ms": 10000,
    })
    assert not state["timed_out"], state
    job = state
    assert job["outcome"] == "completed", state
    assert job["result"]["success"], job
    assert "triangulated_tags" in job["result"]
