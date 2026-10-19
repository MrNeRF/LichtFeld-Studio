# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep already-open project nodes when importing during asynchronous hydration."""

import os
import subprocess
import time
from pathlib import Path

import pytest

from test_render_on_demand_idle import _call, _initialize, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def test_mesh_import_during_hydration_preserves_project_splats(tmp_path):
    executable = os.environ.get("LFS_EXECUTABLE")
    project = os.environ.get("LFS_PROJECT_HYDRATION_SOURCE")
    mesh = os.environ.get("LFS_PROJECT_HYDRATION_MESH")
    if not executable or not project or not mesh:
        pytest.skip("requires LFS_EXECUTABLE and project/mesh fixtures")

    port = os.environ.get("LFS_MCP_PORT", "45732")
    endpoint = f"http://127.0.0.1:{port}/mcp"
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, HOME=str(home))
    env.pop("PYTHONPATH", None)
    python_home = os.environ.get("LFS_PYTHON_HOME")
    if not python_home:
        cache = Path(executable).resolve().parent / "CMakeCache.txt"
        if cache.is_file():
            for line in cache.read_text().splitlines():
                if line.startswith(("Python_EXECUTABLE:", "Python3_EXECUTABLE:")):
                    python_executable = Path(line.split("=", 1)[1])
                    if python_executable.exists():
                        python_home = str(python_executable.parents[2])
                    break
    if python_home:
        env["PYTHONHOME"] = python_home

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

            tool_names = {tool["name"] for tool in _call(endpoint, "tools/list")["tools"]}
            assert {"project_open", "project_get_info", "scene_list_nodes", "editor_run", "editor_wait"} <= tool_names
            resource_uris = {resource["uri"] for resource in _call(endpoint, "resources/list")["resources"]}
            for uri in (
                "lichtfeld://runtime/catalog",
                "lichtfeld://runtime/state",
                "lichtfeld://ui/state",
                "lichtfeld://scene/state",
                "lichtfeld://selection/current",
            ):
                assert uri in resource_uris
                _call(endpoint, "resources/read", {"uri": uri})

            opened = _tool(endpoint, "project_open", {"path": project, "discard_changes": True})
            assert opened.get("success", True), opened
            initial = _tool(endpoint, "project_get_info")
            assert initial.get("hydration_state") == "hydrating", initial
            assert any(
                payload.get("chapter") == "SPLT" and payload.get("hydration_state") == "unloaded"
                for payload in initial.get("payloads", [])
            ), initial

            imported = _tool(
                endpoint,
                "editor_run",
                {
                    "code": (
                        "import lichtfeld as lf, threading\n"
                        f"threading.Thread(target=lambda: lf.load_file({mesh!r}, "
                        "is_dataset=False, discard_changes=True, replace=False), "
                        "daemon=True).start()"
                    ),
                    "show_console": False,
                    "timeout_ms": 2000,
                    "output_max_chars": 2000,
                },
            )
            assert imported.get("success", True), imported

            deadline = time.monotonic() + 90
            info = initial
            scene = {}
            while time.monotonic() < deadline:
                info = _tool(endpoint, "project_get_info")
                scene = _tool(endpoint, "scene_list_nodes", {"include_hidden": True, "include_auxiliary": False})
                nodes = scene.get("nodes", [])
                if info.get("hydration_state") != "hydrating" and any(
                    node.get("type") == "mesh" for node in nodes
                ):
                    break
                time.sleep(0.1)
            assert info.get("hydration_state") != "hydrating", info

            nodes = scene.get("nodes", [])
            splats = [node for node in nodes if node.get("type") == "splat"]
            meshes = [node for node in nodes if node.get("type") == "mesh"]
            assert len(splats) >= 2 and meshes, scene
            assert app.poll() is None, "app exited while importing during hydration"
        finally:
            app.terminate()
            try:
                app.wait(timeout=10)
            except subprocess.TimeoutExpired:
                app.kill()
                app.wait(timeout=10)
