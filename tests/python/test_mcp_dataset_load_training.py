# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dataset replacement must not stop an unconfirmed training run."""

import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import time
import zlib

import pytest

from test_render_on_demand_idle import _call, _initialize, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _wait(predicate, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if result := predicate():
            return result
        time.sleep(0.05)
    raise AssertionError("training state did not settle")


def _dataset(root, color):
    images = root / "images"
    sparse = root / "sparse" / "0"
    images.mkdir(parents=True)
    sparse.mkdir(parents=True)

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))

    for index in range(2):
        pixels = b"".join(b"\0" + bytes(
            component for x in range(64)
            for component in ((color + x + index * 20) % 256, y * 3, 70)
        ) for y in range(64))
        png = (b"\x89PNG\r\n\x1a\n"
               + chunk(b"IHDR", struct.pack(">2I5B", 64, 64, 8, 2, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))
        (images / f"{index}.png").write_bytes(png)
    (sparse / "cameras.txt").write_text("1 PINHOLE 64 64 50 50 32 32\n")
    (sparse / "images.txt").write_text(
        "1 1 0 0 0 0 0 4 1 0.png\n\n2 1 0 0 0 0.3 0 4 1 1.png\n\n")
    (sparse / "points3D.txt").write_text("".join(
        f"{y * 20 + x + 1} {(x - 9.5) * .1} {(y - 9.5) * .1} 0 200 100 50 0\n"
        for y in range(20) for x in range(20)))
    return str(root)


@pytest.fixture
def endpoint(tmp_path):
    configured = os.environ.get("LFS_TEST_MCP_ENDPOINT")
    if configured:
        _initialize(configured)
        yield configured
        return

    root = Path(__file__).resolve().parents[2]
    build = Path(os.environ.get("LFS_TEST_BUILD_DIR", root / "build"))
    executable = Path(os.environ.get(
        "LFS_EXECUTABLE", build / ("LichtFeld-Studio.exe" if os.name == "nt" else "LichtFeld-Studio")))
    assert executable.is_file(), f"Build the GUI executable first: {executable}"
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", int(os.environ.get("LFS_MCP_PORT", "0"))))
        port = sock.getsockname()[1]
    address = f"http://127.0.0.1:{port}/mcp"
    home = tmp_path / "home"
    for directory in (home, home / ".config", home / ".cache", home / ".local/share", home / "tmp"):
        directory.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HOME=str(home), XDG_CONFIG_HOME=str(home / ".config"),
               XDG_CACHE_HOME=str(home / ".cache"), XDG_DATA_HOME=str(home / ".local/share"),
               TMPDIR=str(home / "tmp"), NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
    for name in ("PYTHONHOME", "PYTHONPATH"):
        env.pop(name, None)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env[name] = "http://127.0.0.1:9"
    with (tmp_path / "app.log").open("w") as log:
        app = subprocess.Popen([str(executable), "--no-splash", "--no-download", "--mcp-port", str(port)],
                               env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            def ready():
                assert app.poll() is None, f"GUI exited; see {tmp_path / 'app.log'}"
                try:
                    _initialize(address)
                    return True
                except OSError:
                    return False
            _wait(ready)
            discovery = {method: _call(address, method) for method in ("tools/list", "resources/list")}
            for name in ("runtime/catalog", "runtime/state", "ui/state", "scene/state", "selection/current"):
                discovery[name] = _call(address, "resources/read", {"uri": "lichtfeld://" + name})
            (tmp_path / "discovery.json").write_text(json.dumps(discovery))
            yield address
        finally:
            app.terminate()
            try:
                app.wait(timeout=20)
            except subprocess.TimeoutExpired:
                app.kill()
                app.wait(timeout=20)


@pytest.mark.parametrize("paused", [False, True], ids=["running", "paused"])
def test_dataset_load_preserves_active_training(tmp_path, paused, endpoint):
    dataset = _dataset(tmp_path / "source", 60)
    replacement = _dataset(tmp_path / "replacement-source", 150)

    def tool(name, **arguments):
        return _tool(endpoint, name, arguments)

    def load(path, output):
        return tool("scene_load_dataset", path=path, images_folder="images", min_track_length=0,
                    output_path=str(output))

    def state():
        return tool("training_get_state")

    # Idle admission and normal dataset loading must keep working.
    assert load(dataset, tmp_path / "initial")["success"]
    tool("runtime_job_wait", job_id="import.dataset", until="inactive", timeout_ms=90000)
    tool("training_params_set", values={"strategy": "mrnf", "iterations": 1000000,
                                       "max_cap": 2000})
    tool("training_start")
    try:
        _wait(lambda: state()["iteration"] > 10)
        if paused:
            tool("runtime_job_control", job_id="training.main", action="pause")
            _wait(lambda: (snapshot := state())["is_paused"] and not snapshot["is_running"])
        before = state()
        params = tool("training_params_get")
        modal = tool("ui_modal_get")
        nodes = [(n["name"], n["type"])
                 for n in tool("scene_list_nodes")["nodes"]]
        response = _call(endpoint, "tools/call", {
            "name": "scene_load_dataset", "arguments": {
                "path": replacement, "images_folder": "images", "min_track_length": 0,
                "output_path": str(tmp_path / "replacement"),
                "strategy": "mcmc", "max_iterations": 12345,
            },
        })
        assert response["isError"], response
        time.sleep(0.2)  # Allow any wrongly requested asynchronous stop to complete.
        after = state()
        assert after["is_running"] or after["is_paused"], "rejected dataset load stopped the training run"
        assert after["is_paused"] == paused
        assert tool("training_params_get") == params, "rejected load changed parameters"
        assert [(n["name"], n["type"])
                for n in tool("scene_list_nodes")["nodes"]] == nodes
        assert tool("ui_modal_get") == modal
        if paused:
            assert after["iteration"] == before["iteration"]
            assert after["num_gaussians"] == before["num_gaussians"]
            tool("runtime_job_control", job_id="training.main", action="resume")
        _wait(lambda: state()["iteration"] > before["iteration"])
    finally:
        if tool("runtime_job_describe", job_id="training.main")["actions"]["cancel"]:
            tool("runtime_job_control", job_id="training.main", action="cancel")
        tool("runtime_job_wait", job_id="training.main", until="inactive", timeout_ms=90000)

    # Explicit stop still permits replacement with the requested settings.
    assert load(replacement, tmp_path / "after-stop")["success"]
    tool("runtime_job_wait", job_id="import.dataset", until="inactive", timeout_ms=90000)
    assert not state()["is_running"]
