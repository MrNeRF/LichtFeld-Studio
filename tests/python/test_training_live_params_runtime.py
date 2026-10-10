"""The active run must keep its initialization settings when next-run settings change."""

import struct
import time
import zlib

import pytest

from test_render_on_demand_idle import _tool
from runtime_test_app import isolated_app

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _dataset_fixture(root):
    """Four small RGB images, calibrated cameras and a visible sparse grid."""
    images = root / "images"
    sparse = root / "sparse" / "0"
    images.mkdir(parents=True)
    sparse.mkdir(parents=True)
    (sparse / "cameras.txt").write_text("1 PINHOLE 96 96 80 80 48 48\n")
    poses = []
    for index in range(4):
        name = f"view_{index}.png"
        poses.append(f"{index + 1} 1 0 0 0 {(index - 1.5) * 0.1} 0 0 1 {name}\n\n")
        # PNG encoding uses only the standard library, including CRCs for each chunk.
        def chunk(kind, data):
            return (struct.pack(">I", len(data)) + kind + data
                    + struct.pack(">I", zlib.crc32(kind + data)))
        pixels = b"".join(b"\0" + bytes(component for x in range(96)
                           for component in (40 + x * 2, 40 + y * 2, 100)) for y in range(96))
        png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">2I5B", 96, 96, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")
        (images / name).write_bytes(png)
    (sparse / "images.txt").write_text("".join(poses))
    (sparse / "points3D.txt").write_text("".join(
        f"{y * 5 + x + 1} {(x - 2) * 0.2} {(y - 2) * 0.2} 3 160 120 100 0 1 0 2 0\n"
        for y in range(5) for x in range(5)))
    return root


@pytest.fixture
def training_endpoint(tmp_path):
    # Each parametrized case owns a fresh app and cannot inherit another run.
    with isolated_app(tmp_path) as endpoint:
        yield endpoint


@pytest.mark.parametrize("paused", [False, True])
@pytest.mark.parametrize("writer", ["mcp", "python"])
def test_non_live_updates_are_deferred(tmp_path, training_endpoint, paused, writer):
    endpoint = training_endpoint
    dataset = _dataset_fixture(tmp_path / "dataset")

    def call(name, args=None):
        return _tool(endpoint, name, args)
    loaded = call("scene_load_dataset", {
        "path": str(dataset), "images_folder": "images",
        "output_path": str(tmp_path / "output"),
    })
    assert loaded["success"], loaded
    loaded = call("runtime_job_wait", {
        "job_id": "import.dataset", "until": "inactive", "timeout_ms": 120000,
    })
    assert loaded["success"], loaded
    initial = call("training_params_set", {"values": {
        "strategy": "mrnf", "iterations": 3000, "max_cap": 1000,
        "sparsify_steps": 2000, "enable_sparsity": False,
    }})
    assert initial["success"]
    assert call("training_start")["success"]
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        state = call("training_get_state")
        if state["iteration"] >= 500:
            break
        time.sleep(0.05)
    else:
        pytest.fail("training did not reach the update iteration")
    assert state["iteration"] < 3000, "run finished before the update could be tested"
    if paused:
        call("runtime_job_control", {"job_id": "training.main", "action": "pause"})
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if call("training_get_state")["is_paused"]:
                break
            call("runtime_job_wait", {
                "job_id": "training.main", "until": "changed", "timeout_ms": 100,
            })
        else:
            pytest.fail("training did not pause")
    changes = {"iterations": 1500, "enable_sparsity": True, "gut": True,
               "max_cap": 2000, "means_lr": 0.0001}
    if writer == "mcp":
        result = call("training_params_set", {"values": changes})
    else:
        update = call("editor_run", {
            "code": "import lichtfeld as lf\nopt = lf.optimization_params()\n"
                    "opt.iterations = 1500\nopt.gut = True\nopt.max_cap = 2000\n"
                    "opt.set('enable_sparsity', True)\nopt.means_lr = 0.0001",
            "show_console": False, "wait_for_completion": True, "timeout_ms": 10000,
        })
        assert update["success"], update
        result = call("training_params_get")
    assert result.get("success", True)
    editable = {p["id"]: p["value"] for p in result["properties"]}
    for name, value in changes.items():
        assert editable[name] == pytest.approx(value)
    # Readbacks describe the next run; the current run must retain its schedule.
    if paused:
        call("runtime_job_control", {"job_id": "training.main", "action": "resume"})
    call("runtime_job_wait", {
        "job_id": "training.main", "until": "inactive", "timeout_ms": 120000,
    })
    state = call("training_get_state")
    assert state["iteration"] == 3000, state
    assert state["max_iterations"] == 3000, state
    assert state["num_gaussians"] <= 1000, state
    if writer == "mcp":
        assert sorted(result["deferred_parameters"]) == [
            "enable_sparsity", "gut", "iterations", "max_cap",
        ]
