# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression coverage for SystemExit in the integrated Python editor."""

import os
import subprocess
import time

import pytest

from test_render_on_demand_idle import _call, _initialize, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def test_sys_exit_finishes_editor_script_and_allows_recovery(tmp_path):
    executable = os.environ.get("LFS_EXECUTABLE")
    if not executable:
        pytest.skip("requires LFS_EXECUTABLE and an isolated X display")

    port = os.environ.get("LFS_MCP_PORT", "45706")
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

            tool_names = {
                tool["name"] for tool in _call(endpoint, "tools/list")["tools"]
            }
            assert {"editor_run", "editor_is_running"} <= tool_names
            resource_uris = {
                resource["uri"]
                for resource in _call(endpoint, "resources/list")["resources"]
            }
            for uri in (
                "lichtfeld://runtime/catalog",
                "lichtfeld://runtime/state",
                "lichtfeld://ui/state",
                "lichtfeld://scene/state",
                "lichtfeld://selection/current",
            ):
                assert uri in resource_uris
                _call(endpoint, "resources/read", {"uri": uri})

            exited = _tool(
                endpoint,
                "editor_run",
                {
                    "code": "import sys; sys.exit(7)",
                    "show_console": False,
                    "timeout_ms": 3000,
                    "output_max_chars": 1000,
                },
            )
            assert not exited["timed_out"] and not exited["running"], exited

            status = _tool(endpoint, "editor_is_running")
            assert not status["running"], status

            recovered = _tool(
                endpoint,
                "editor_run",
                {
                    "code": "print('editor recovered')",
                    "show_console": False,
                    "timeout_ms": 5000,
                    "output_max_chars": 1000,
                },
            )
            assert recovered["completed"] and recovered["success"], recovered
            assert "editor recovered" in recovered["output"]["text"], recovered
        finally:
            app.terminate()
            try:
                app.wait(timeout=10)
            except subprocess.TimeoutExpired:
                app.kill()
                app.wait(timeout=10)
