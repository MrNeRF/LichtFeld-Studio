# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Restarting a terminal training run keeps its in-memory training input."""

import os
from pathlib import Path

import pytest

from test_mcp_dataset_load_training import _wait, endpoint
from test_render_on_demand_idle import _call, _tool


pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.fixture
def restart_dataset():
    path = Path(os.environ.get("LFS_RESTART_DATASET", ""))
    if not path.is_dir() or not (path / "images").is_dir():
        pytest.skip("set LFS_RESTART_DATASET to an approved COLMAP dataset")
    return path


def test_terminal_run_restarts_from_the_loaded_scene(endpoint, restart_dataset, tmp_path):
    """The deployed MCP operation returns a stopped run to Ready without reloading it."""
    def tool(name, **arguments):
        return _tool(endpoint, name, arguments)

    loaded = tool(
        "scene_load_dataset",
        path=str(restart_dataset),
        images_folder="images",
        min_track_length=2,
        output_path=str(tmp_path / "restart-output"),
    )
    assert loaded["success"]
    tool("runtime_job_wait", job_id="import.dataset", until="inactive", timeout_ms=90_000)
    tool("training_params_set", values={"strategy": "mrnf", "iterations": 1_000_000, "max_cap": 2_000})
    tool("training_start")
    _wait(lambda: tool("training_get_state")["iteration"] > 10)
    started_nodes = tool("scene_list_nodes")["nodes"]
    started_model = next(node for node in started_nodes if node["type"] == "SPLAT")
    tool("runtime_job_control", job_id="training.main", action="cancel")
    tool("runtime_job_wait", job_id="training.main", until="inactive", timeout_ms=90_000)
    before = tool("scene_list_nodes")["nodes"]

    result = _call(endpoint, "tools/call", {"name": "training_restart", "arguments": {}})
    assert not result.get("isError"), result
    assert result["structuredContent"] == {
        "success": True,
        "operation": "training.restart",
        "iteration": 0,
    }
    assert tool("training_get_state")["iteration"] == 0
    after = tool("scene_list_nodes")["nodes"]
    assert [(node["name"], node["type"]) for node in after] != [
        (node["name"], node["type"]) for node in before
    ]
    restarted_point_cloud = next(node for node in after if node["type"] == "POINTCLOUD")
    assert restarted_point_cloud["gaussian_count"] == started_model["gaussian_count"]
