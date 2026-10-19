# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression coverage for setting scene transforms from CUDA tensors."""

import os
import subprocess
import time

import pytest

from test_render_on_demand_idle import _call, _initialize, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def test_set_node_transform_from_cuda_tensor_keeps_editor_alive(tmp_path):
    executable = os.environ.get("LFS_EXECUTABLE")
    if not executable:
        pytest.skip("requires LFS_EXECUTABLE and an isolated X display")

    port = os.environ.get("LFS_MCP_PORT", "45707")
    endpoint = f"http://127.0.0.1:{port}/mcp"
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, HOME=str(home))

    with (tmp_path / "app.log").open("w") as log:
        app = subprocess.Popen(
            [executable, "--no-splash", "--mcp-port", port],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                assert app.poll() is None, "app exited during startup"
                try:
                    _initialize(endpoint)
                    break
                except (OSError, AssertionError):
                    time.sleep(0.25)
            else:
                pytest.fail("MCP did not start")

            _call(endpoint, "tools/list")
            for uri in (
                "lichtfeld://runtime/catalog",
                "lichtfeld://runtime/state",
                "lichtfeld://ui/state",
                "lichtfeld://scene/state",
                "lichtfeld://selection/current",
            ):
                _call(endpoint, "resources/read", {"uri": uri})

            setup = _tool(
                endpoint,
                "editor_run",
                {
                    "code": (
                        "import lichtfeld as lf\n"
                        "scene = lf.get_scene()\n"
                        "node_id = scene.add_group('cuda_transform_group')\n"
                        "transform = lf.Tensor.eye(4, device='cuda')\n"
                        "print('CUDA_TRANSFORM_READY', node_id)"
                    ),
                    "timeout_ms": 8000,
                    "output_max_chars": 1000,
                },
            )
            assert setup["completed"] and setup["success"], setup

            cpu_copy = _tool(
                endpoint,
                "editor_run",
                {
                    "code": "cpu_transform = transform.cpu(); print('CUDA_COPY_READY')",
                    "timeout_ms": 8000,
                    "output_max_chars": 1000,
                },
            )
            assert cpu_copy["completed"] and cpu_copy["success"], cpu_copy

            cpu_set = _tool(
                endpoint,
                "editor_run",
                {
                    "code": "scene.set_node_transform('cuda_transform_group', cpu_transform); print('CPU_TRANSFORM_SET')",
                    "timeout_ms": 8000,
                    "output_max_chars": 1000,
                },
            )
            assert cpu_set["completed"] and cpu_set["success"], cpu_set

            ndarray_set = _tool(
                endpoint,
                "editor_run",
                {
                    "code": (
                        "scene.set_node_transform('cuda_transform_group', cpu_transform.numpy())\n"
                        "print('CPU_NDARRAY_TRANSFORM_SET')"
                    ),
                    "timeout_ms": 8000,
                    "output_max_chars": 1000,
                },
            )
            assert ndarray_set["completed"] and ndarray_set["success"], ndarray_set

            result = _tool(
                endpoint,
                "editor_run",
                {
                    "code": (
                        "scene.set_node_transform('cuda_transform_group', transform)\n"
                        "print('CUDA_TRANSFORM_SET')"
                    ),
                    "timeout_ms": 8000,
                    "output_max_chars": 1000,
                },
            )
            assert result["completed"] and result["success"], result
            assert "CUDA_TRANSFORM_SET" in result["output"]["text"], result
            assert app.poll() is None, "app exited after setting the CUDA transform"
        finally:
            app.terminate()
            try:
                app.wait(timeout=10)
            except subprocess.TimeoutExpired:
                app.kill()
                app.wait(timeout=10)


@pytest.mark.parametrize(
    ("api", "code"),
    [
        (
            "PySceneNode.set_local_transform",
            "scene = lf.get_scene(); scene.add_group('node_input_probe'); "
            "node = scene.get_node('node_input_probe'); tensor = lf.Tensor.eye(4, device='cuda'); "
            "node.set_local_transform(tensor)",
        ),
        (
            "Tensor.from_numpy",
            "tensor = lf.Tensor.eye(4, device='cuda'); lf.Tensor.from_numpy(tensor)",
        ),
        (
            "TriMesh.add_faces",
            "mesh = lf.mesh.TriMesh(); faces = lf.Tensor.zeros((1,3), dtype='int32', device='cuda'); "
            "mesh.add_faces(faces)",
        ),
    ],
)
def test_cuda_tensors_are_rejected_by_cpu_ndarray_inputs(tmp_path, api, code):
    executable = os.environ.get("LFS_EXECUTABLE")
    if not executable:
        pytest.skip("requires LFS_EXECUTABLE and an isolated X display")

    port = os.environ.get("LFS_MCP_PORT", "45709")
    endpoint = f"http://127.0.0.1:{port}/mcp"
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, HOME=str(home))

    with (tmp_path / "app.log").open("w") as log:
        app = subprocess.Popen(
            [executable, "--no-splash", "--mcp-port", port],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                assert app.poll() is None, "app exited during startup"
                try:
                    _initialize(endpoint)
                    break
                except (OSError, AssertionError):
                    time.sleep(0.25)
            else:
                pytest.fail("MCP did not start")

            result = _tool(
                endpoint,
                "editor_run",
                {
                    "code": (
                        "import lichtfeld as lf\n"
                        "try:\n"
                        + "    " + code.replace("; ", "\n    ") + "\n"
                        "except TypeError:\n"
                        f"    print('TYPE_ERROR:{api}')\n"
                        "else:\n"
                        f"    print('ACCEPTED:{api}')"
                    ),
                    "timeout_ms": 8000,
                    "output_max_chars": 1000,
                },
            )
            assert result["completed"] and result["success"], result
            assert f"TYPE_ERROR:{api}" in result["output"]["text"], result
            assert app.poll() is None, f"app exited while checking {api}"
        finally:
            app.terminate()
            try:
                app.wait(timeout=10)
            except subprocess.TimeoutExpired:
                app.kill()
                app.wait(timeout=10)
