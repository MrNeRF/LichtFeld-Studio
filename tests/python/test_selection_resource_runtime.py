"""Selection resource counts must follow the current scene mask."""

from __future__ import annotations

import json
import os
import struct
import time
from pathlib import Path

import pytest

from test_render_on_demand_idle import _call, _initialize, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _write_splats(path: Path, count: int) -> None:
    fields = [
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    with path.open("wb") as handle:
        header = (
            f"ply\nformat binary_little_endian 1.0\nelement vertex {count}\n"
            + "".join(f"property float {field}\n" for field in fields)
            + "end_header\n"
        )
        handle.write(header.encode())
        for index in range(count):
            handle.write(struct.pack(
                "<17f", index * 0.1, 0, 0, 0, 0, 0, 1, 0.2, 0.1, 3,
                -3, -3, -3, 1, 0, 0, 0,
            ))


def _resource(endpoint: str) -> dict:
    result = _call(endpoint, "resources/read", {"uri": "lichtfeld://selection/current"})
    return json.loads(result["contents"][0]["text"])


def test_selection_resource_clears_count_after_scene_replacement(tmp_path):
    endpoint = os.environ.get("LFS_TEST_MCP_ENDPOINT")
    if not endpoint:
        pytest.skip("requires an isolated GUI MCP endpoint")
    _initialize(endpoint)

    initial = tmp_path / "initial.ply"
    replacements = [tmp_path / "replacement-a.ply", tmp_path / "replacement-b.ply"]
    _write_splats(initial, 8)
    for index, path in enumerate(replacements, start=1):
        _write_splats(path, index + 3)

    _tool(endpoint, "scene_load_ply", {"path": str(initial)})
    _tool(endpoint, "runtime_job_wait", {
        "job_id": "import.dataset", "until": "inactive", "timeout_ms": 10000,
    })
    _tool(endpoint, "editor_run", {
        "code": "import lichtfeld as lf\nlf.get_scene().set_selection([0, 1, 2, 3])",
        "show_console": False, "wait_for_completion": True, "timeout_ms": 5000,
    })
    assert _tool(endpoint, "selection_get", {"max_indices": 10})["selected_count"] == 4
    assert _resource(endpoint)["selected_count"] == 4

    for path in replacements:
        _tool(endpoint, "scene_load_ply", {"path": str(path)})
        _tool(endpoint, "runtime_job_wait", {
            "job_id": "import.dataset", "until": "inactive", "timeout_ms": 10000,
        })
        assert _tool(endpoint, "selection_get", {"max_indices": 10})["selected_count"] == 0
        state = _resource(endpoint)
        assert state == {
            "indices": [], "selected_count": 0, "success": True, "truncated": False,
        }


def test_selection_resource_clears_count_after_training_changes_model_size(tmp_path):
    endpoint = os.environ.get("LFS_TEST_MCP_ENDPOINT")
    dataset = os.environ.get("LFS_SELECTION_DENSIFY_DATASET")
    if not endpoint or not dataset:
        pytest.skip("requires an isolated GUI MCP endpoint and a training dataset")
    _initialize(endpoint)

    _tool(endpoint, "scene_load_dataset", {
        "path": dataset,
        "images_folder": os.environ.get("LFS_SELECTION_DENSIFY_IMAGES", "images_8"),
        "max_iterations": 300,
        "output_path": str(tmp_path / "training-output"),
    })
    _tool(endpoint, "runtime_job_wait", {
        "job_id": "import.dataset", "until": "inactive", "timeout_ms": 150000,
    })
    _tool(endpoint, "training_params_set", {"values": {
        "iterations": 300,
        "max_cap": 500000,
        "refine_every": 1,
        "start_refine": 1,
        "stop_refine": 300,
    }})
    _tool(endpoint, "training_start")
    _tool(endpoint, "runtime_job_control", {
        "job_id": "training.main", "action": "pause", "include_output": False,
    })
    initial = _tool(endpoint, "training_get_state", {})
    assert initial["is_running"]

    _tool(endpoint, "editor_run", {
        "code": "import lichtfeld as lf\nlf.get_scene().set_selection([0, 1, 2, 3])",
        "show_console": False,
        "wait_for_completion": True,
        "timeout_ms": 5000,
    })
    assert _tool(endpoint, "selection_get", {"max_indices": 10})["selected_count"] == 4
    assert _resource(endpoint)["selected_count"] == 4

    _tool(endpoint, "runtime_job_control", {
        "job_id": "training.main", "action": "resume", "include_output": False,
    })
    deadline = time.monotonic() + 240
    changed = False
    while time.monotonic() < deadline:
        state = _tool(endpoint, "training_get_state", {})
        selected = _tool(endpoint, "selection_get", {"max_indices": 10})["selected_count"]
        if state["num_gaussians"] != initial["num_gaussians"] and selected == 0:
            changed = True
            break
        time.sleep(0.1)

    assert changed, "training did not change the model size and clear selection before timeout"
    assert _resource(endpoint) == {
        "indices": [], "selected_count": 0, "success": True, "truncated": False,
    }
    _tool(endpoint, "runtime_job_control", {
        "job_id": "training.main", "action": "pause", "include_output": False,
    })
