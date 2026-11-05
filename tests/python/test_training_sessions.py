# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared protocol contracts independent of Blender, Python ABI and CUDA."""
import threading
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from lichtfeld_sessions import geometry, publisher, transport
from lichtfeld_sessions.capture import PreviewCapacityError, capture


def arrays(count=3):
    return {"means": np.zeros((count, 3), np.float32), "scale": np.zeros((count, 3), np.float32),
            "rotation": np.tile([1, 0, 0, 0], (count, 1)).astype(np.float32),
            "opacity": np.zeros((count, 1), np.float32), "sh": np.zeros((count, 1, 3), np.float32)}


def test_immutable_generations_integrity_and_owned_reader_storage(tmp_path):
    pool = transport.SnapshotPool(tmp_path)
    try:
        original = arrays()
        first = pool.write(original, 10)
        original["means"][:] = 9
        second = pool.write(original, 11)
        assert first["id"] != second["id"]
        assert np.all(transport.read_snapshot(first)["means"] == 0)
        installed = transport.read_snapshot(second)
        with pytest.raises(ValueError, match="integrity"):
            transport.read_snapshot({**second, "sha256": "0" * 64})
        pool.close()
        assert np.all(installed["means"] == 9)
    finally:
        pool.close()


def test_live_discovery_rejects_terminal_and_reused_pid(tmp_path, monkeypatch):
    monkeypatch.setenv("LFS_TRAINING_SESSIONS_DIR", str(tmp_path))
    session = transport.SessionPublisher({"label": "test"})
    one = transport.SessionClient(session.root, "one")
    two = transport.SessionClient(session.root, "two")
    try:
        session._clients_at = 0
        assert len(session.clients()) == 2
        assert [record["id"] for record in transport.discover()] == [session.id]
        monkeypatch.setattr(transport, "process_identity", lambda pid: "reused")
        assert transport.discover() == []
        monkeypatch.undo()
        monkeypatch.setenv("LFS_TRAINING_SESSIONS_DIR", str(tmp_path))
        session.finish("stopped")
        assert transport.discover() == []
    finally:
        one.close()
        two.close()
        session.finish("stopped")
        session.thread.join(3)
        session.pool.close()


def test_slow_metadata_io_does_not_hold_the_training_status_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("LFS_TRAINING_SESSIONS_DIR", str(tmp_path))
    session = transport.SessionPublisher({})
    entered, release, updated = threading.Event(), threading.Event(), threading.Event()
    original = transport.atomic_json

    def slow(*args):
        entered.set()
        assert release.wait(3)
        original(*args)

    monkeypatch.setattr(transport, "atomic_json", slow)
    writer = threading.Thread(target=session.heartbeat)
    update = threading.Thread(target=lambda: (session.update(flush=False, iteration=77), updated.set()))
    try:
        writer.start()
        assert entered.wait(2)
        update.start()
        assert updated.wait(2), "Training status waited for filesystem IO"
    finally:
        release.set()
        writer.join(3)
        update.join(3)
        monkeypatch.setattr(transport, "atomic_json", original)
        session.finish("stopped")
        session.thread.join(3)


class Tensor:
    def __init__(self, value): self.value = np.asarray(value)
    def cpu(self): return self
    def cuda(self): return self
    def sync(self): pass
    def numpy(self): return self.value.copy()
    def index_select(self, axis, rows): return Tensor(self.value[rows.value])


@pytest.mark.parametrize("degree", [0, 1, 3])
def test_capture_filters_deleted_rows_and_gathers_only_requested_sh(degree):
    model = SimpleNamespace(num_points=8, active_sh_degree=3,
        has_deleted_mask=lambda: True, deleted=Tensor([0, 1, 0, 1, 0, 1, 0, 1]),
        means_raw=Tensor(np.arange(24).reshape(8, 3)), scaling_raw=Tensor(np.zeros((8, 3))),
        rotation_raw=Tensor(np.tile([1, 0, 0, 0], (8, 1))), opacity_raw=Tensor(np.zeros((8, 1))),
        sh0_raw=Tensor(np.zeros((8, 1, 3))),
        shN_rows=Mock(side_effect=lambda rows, band: Tensor(np.zeros((len(rows.value), (band + 1)**2 - 1, 3)))))
    lf = SimpleNamespace(get_scene=lambda: SimpleNamespace(training_model=lambda: model),
                         Tensor=SimpleNamespace(from_numpy=Tensor))
    result = capture(lf, limit=2, degree=degree)
    assert len(result["means"]) == 2
    assert np.all(result["means"][:, 0] % 6 == 0)
    assert result["sh"].shape == (2, (degree + 1)**2, 3)
    if degree:
        assert len(model.shN_rows.call_args.args[0].value) == 2
        assert model.shN_rows.call_args.args[1] == degree
    else:
        model.shN_rows.assert_not_called()


def test_capture_capacity_rejected_before_attribute_access():
    model = SimpleNamespace(num_points=10000000, active_sh_degree=3, has_deleted_mask=lambda: False)
    lf = SimpleNamespace(get_scene=lambda: SimpleNamespace(training_model=lambda: model))
    with pytest.raises(PreviewCapacityError):
        capture(lf, max_bytes=transport.MAX_BYTES)


def test_native_camera_uses_actual_principal_point_and_model_frame():
    world = np.eye(4); world[:3, 3] = [1, 2, 3]
    model = np.eye(4); model[:3, 3] = [2, -3, 4]
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    translation = np.array([3, 4, 5])
    camera = SimpleNamespace(has_camera=True, name="Camera 1", world_transform=world,
        camera_R=Tensor(rotation), camera_T=Tensor(translation),
        camera_focal_x=50., camera_focal_y=60., camera_center_x=17., camera_center_y=22.,
        camera_width=64, camera_height=48)
    scene = SimpleNamespace(training_model_node_name="model", get_node=lambda name: SimpleNamespace(world_transform=model),
                            get_active_cameras=lambda: [SimpleNamespace(has_camera=False), camera])
    result = geometry.from_native_scene(scene)
    assert result["cameras"][0]["cx"] == 17.
    assert result["cameras"][0]["cy"] == 22.
    expected = np.eye(4); expected[:3, :3] = rotation.T; expected[:3, 3] = -rotation.T @ translation
    np.testing.assert_array_equal(result["cameras"][0]["camera_to_data"], world @ expected)
    np.testing.assert_array_equal(result["model_to_data"], model)
    assert result["cameras"][0]["fy"] == 60.
    assert result["cameras"][0]["height"] == 48
    source = {"source_frame": "blender-world", "data_to_source": geometry.BLENDER_TO_VISUALIZER}
    result = geometry.from_native_scene(scene, source)
    assert result["source_frame"] == "blender-world"
    np.testing.assert_array_equal(result["data_to_source"], source["data_to_source"])
    assert result["cameras"][0]["cx"] == 17.
    camera.camera_focal_x = float("nan")
    with pytest.raises(ValueError, match="intrinsics"):
        geometry.from_native_scene(scene)


def test_dependency_preparation_never_blocks_capture_or_terminal_hook(tmp_path, monkeypatch):
    monkeypatch.setenv("LFS_TRAINING_SESSIONS_DIR", str(tmp_path))
    entered, release = threading.Event(), threading.Event()

    def prepare(lf):
        entered.set()
        assert release.wait(3)

    monkeypatch.setattr(publisher, "ensure_numpy", prepare)
    lf = SimpleNamespace(ScopedHandler=Mock)
    stream = publisher.TrainingPublisher(lf, degree=0)
    hook = {"iteration": 1, "iterations": 20, "num_gaussians": 5, "is_paused": False}
    try:
        assert entered.wait(2)
        pending = stream.pending
        stream.step(hook)
        stream.step(hook)
        assert stream.pending is pending  # No unbounded queue of capture work.
        stream.end({**hook, "terminal_state": "stopped"})
        assert not release.is_set()
    finally:
        release.set()
        stream.end(hook)
        stream.writer.shutdown(wait=True)
        stream.publisher.thread.join(3)
    assert transport.read_json(stream.publisher.root / "session.json")["state"] == "stopped"


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize("enabled", [False, True])
def test_headless_publisher_requires_explicit_opt_in(tmp_path, gpu_available, enabled):
    if not gpu_available:
        pytest.skip("CUDA GPU required for training")
    from test_mcp_dataset_load_training import _dataset
    root = Path(__file__).resolve().parents[2]
    build = Path(os.environ.get("LFS_TEST_BUILD_DIR", os.environ.get("LFS_BUILD_DIR", root / "build")))
    executable = build / ("LichtFeld-Studio.exe" if os.name == "nt" else "LichtFeld-Studio")
    sessions = tmp_path / "sessions"
    environment = {**os.environ, "LFS_HOME": str(tmp_path / "runtime"),
                   "LFS_TRAINING_SESSIONS_DIR": str(sessions),
                   "PYTHONPATH": str(Path(np.__file__).resolve().parents[1])}
    environment.pop("PYTHONHOME", None)
    environment.pop("LFS_TRAINING_PUBLISHER", None)
    if enabled:
        environment["LFS_TRAINING_PUBLISHER"] = "automatic"
    dataset = _dataset(tmp_path / "dataset", 60)
    command = [str(executable), "--headless", "-d", dataset, "-o", str(tmp_path / "output"),
               "--iter", "8", "--max-cap", "1000", "--sh-degree", "1", "--strategy", "mcmc",
               "--min-track-length", "0", "--centralize", "off", "--no-download"]
    log = tmp_path / "trainer.log"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT,
            timeout=60, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    text = log.read_text(encoding="utf-8")
    assert result.returncode == 0, text[-8000:]
    records = [transport.read_json(path) for path in sessions.glob("*/session.json")]
    if enabled:
        assert len(records) == 1, text[-8000:]
        assert records[0]["state"] == "completed"
        assert records[0]["iteration"] == 8
        assert records[0]["source"] == "lichtfeld"
        assert records[0]["dataset"] == dataset
    else:
        assert not records
        assert "Successfully pre-imported lichtfeld module" not in text
