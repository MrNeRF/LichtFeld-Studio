"""A saved point-cloud dataset must open clean in a fresh application."""

import os
import struct
import subprocess
import time
import zlib
from contextlib import contextmanager
from pathlib import Path

import pytest

from test_render_on_demand_idle import _call, _initialize, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _png_chunk(kind, data):
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def _write_image(path, color):
    pixels = b"".join(b"\0" + bytes(color) * 16 for _ in range(16))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 16, 16, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(pixels))
        + _png_chunk(b"IEND", b"")
    )


def _write_dataset(path):
    import json

    images = path / "images_8"
    images.mkdir(parents=True)
    _write_image(images / "frame-0.png", (255, 0, 0))
    _write_image(images / "frame-1.png", (0, 255, 0))
    frames = []
    for index in range(2):
        frames.append({
            "file_path": f"images_8/frame-{index}.png",
            "transform_matrix": [
                [1, 0, 0, index * 0.2],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
        })
    (path / "transforms.json").write_text(json.dumps({
        "camera_model": "PINHOLE", "fl_x": 8, "fl_y": 8,
        "cx": 8, "cy": 8, "w": 16, "h": 16,
        "aabb_scale": 16, "frames": frames,
    }))
    (path / "pointcloud.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 4\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
        "0 0 0 255 0 0\n1 0 0 0 255 0\n0 1 0 0 0 255\n0 0 1 255 255 255\n"
    )


@contextmanager
def _app(executable, home, port):
    home.mkdir(parents=True)
    env = os.environ.copy()
    env.update({
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        "XDG_DATA_HOME": str(home / ".local/share"),
        "HTTP_PROXY": "http://127.0.0.1:9",
        "HTTPS_PROXY": "http://127.0.0.1:9",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "UV_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
    })
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    endpoint = f"http://127.0.0.1:{port}/mcp"
    with (home / "app.log").open("w") as log:
        app = subprocess.Popen(
            [executable, "--no-download", "--no-splash", "--mcp-port", str(port)],
            env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                assert app.poll() is None, "app exited during startup"
                try:
                    _initialize(endpoint)
                    break
                except OSError:
                    time.sleep(0.25)
            else:
                pytest.fail("MCP did not start")
            assert {"project_open", "project_save_as", "project_get_info", "scene_load_dataset"} <= {
                item["name"] for item in _call(endpoint, "tools/list")["tools"]
            }
            _call(endpoint, "resources/list")
            for name in ("runtime/catalog", "runtime/state", "ui/state", "scene/state", "selection/current"):
                _call(endpoint, "resources/read", {"uri": "lichtfeld://" + name})
            yield endpoint
        finally:
            app.terminate()
            try:
                app.wait(timeout=15)
            except subprocess.TimeoutExpired:
                app.kill()
                app.wait(timeout=15)


@pytest.fixture(scope="module")
def projects(tmp_path_factory):
    executable = os.environ.get("LFS_EXECUTABLE")
    if not executable or not os.environ.get("DISPLAY"):
        pytest.skip("requires LFS_EXECUTABLE and a display")
    root = tmp_path_factory.mktemp("point-cloud-first-open")
    dataset = root / "dataset"
    _write_dataset(dataset)
    control = root / "control.licht"
    normalized_point_cloud = root / "normalized-point-cloud.licht"
    point_cloud = root / "point-cloud.licht"
    port = int(os.environ.get("LFS_MCP_PORT", "45811"))
    with _app(executable, root / "creator", port) as endpoint:
        created = _tool(endpoint, "editor_run", {
            "code": """
import lichtfeld as lf
scene = lf.get_scene()
for index in range(2):
    means = lf.Tensor.zeros([2, 3], device="cpu", dtype="float32")
    means[1, 0] = float(index + 1)
    sh0 = lf.Tensor.zeros([2, 1, 3], device="cpu", dtype="float32")
    shn = lf.Tensor.zeros([2, 0, 3], device="cpu", dtype="float32")
    scaling = lf.Tensor.zeros([2, 3], device="cpu", dtype="float32")
    rotation = lf.Tensor.zeros([2, 4], device="cpu", dtype="float32")
    rotation[0, 0] = 1.0
    rotation[1, 0] = 1.0
    opacity = lf.Tensor.zeros([2, 1], device="cpu", dtype="float32")
    scene.add_splat(f"control-{index}", means, sh0, shn, scaling,
                    rotation, opacity, sh_degree=0)
""",
            "show_console": False, "wait_for_completion": True,
            "timeout_ms": 30000,
        })
        assert created.get("success"), created
        saved = _tool(endpoint, "project_save_as", {"path": str(control)})
        assert saved.get("success"), saved
        loaded = _tool(endpoint, "scene_load_dataset", {
            "path": str(dataset), "images_folder": "images_8",
            "output_path": str(root / "output"),
        })
        assert loaded.get("success", True), loaded
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            nodes = _tool(endpoint, "scene_list_nodes")["nodes"]
            if sum(node["type"] == "camera" for node in nodes) == 2 and any(
                node["type"] == "pointcloud" and node["gaussian_count"] == 4 for node in nodes
            ):
                break
            time.sleep(0.1)
        else:
            pytest.fail(f"dataset import did not produce two cameras and four points: {nodes}")
        saved = _tool(endpoint, "project_save_as", {"path": str(normalized_point_cloud)})
        assert saved.get("success"), saved
        saved = _tool(endpoint, "editor_run", {
            "code": f"""
import lichtfeld as lf
params = lf.optimization_params()
params.set("init_opacity", 0.5)
params.set("lambda_dssim", 0.2)
assert lf.project_save_as({str(point_cloud)!r}, wait=True)
""",
            "show_console": False, "wait_for_completion": True,
            "timeout_ms": 30000,
        })
        assert saved.get("success"), saved
    assert control.is_file() and point_cloud.is_file()
    assert normalized_point_cloud.is_file()
    return executable, root, port, control, normalized_point_cloud, point_cloud


def _wait_for_hydration(endpoint):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        info = _tool(endpoint, "project_get_info")
        if info["hydration_state"] == "complete":
            return info
        time.sleep(0.1)
    pytest.fail(f"project hydration did not complete: {info}")


def test_first_open_of_saved_point_cloud_is_clean(projects):
    executable, root, port, control, _, point_cloud = projects
    saved_bytes = point_cloud.read_bytes()
    with _app(executable, root / "point-opener", port) as endpoint:
        opened = _tool(endpoint, "project_open", {"path": str(point_cloud)})
        assert opened.get("success"), opened
        info = _wait_for_hydration(endpoint)
        assert any(payload["chapter"] == "PCLD" for payload in info["payloads"])
        assert not info["dirty"], info
        assert point_cloud.read_bytes() == saved_bytes
        switched = _tool(endpoint, "project_open", {"path": str(control)})
        assert switched.get("success"), switched


def test_first_open_of_normalized_point_cloud_stays_clean(projects):
    executable, root, port, control, normalized_point_cloud, _ = projects
    with _app(executable, root / "normalized-opener", port) as endpoint:
        opened = _tool(endpoint, "project_open", {"path": str(normalized_point_cloud)})
        assert opened.get("success"), opened
        info = _wait_for_hydration(endpoint)
        assert any(payload["chapter"] == "PCLD" for payload in info["payloads"])
        assert not info["dirty"], info
        switched = _tool(endpoint, "project_open", {"path": str(control)})
        assert switched.get("success"), switched


def test_first_open_of_saved_splat_project_stays_clean(projects):
    executable, root, port, control, _, _ = projects
    with _app(executable, root / "control-opener", port) as endpoint:
        opened = _tool(endpoint, "project_open", {"path": str(control)})
        assert opened.get("success"), opened
        info = _wait_for_hydration(endpoint)
        assert not info["dirty"], info
