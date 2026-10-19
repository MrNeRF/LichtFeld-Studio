# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reject unsupported render scales at the MCP settings boundary."""

import os
import subprocess
import time
from pathlib import Path

import pytest

from test_render_on_demand_idle import _call, _initialize, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def test_render_settings_rejects_negative_scale_without_changing_state(tmp_path):
    executable = os.environ.get("LFS_EXECUTABLE")
    if not executable:
        pytest.skip("requires LFS_EXECUTABLE")

    port = os.environ.get("LFS_MCP_PORT", "45761")
    endpoint = f"http://127.0.0.1:{port}/mcp"
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, HOME=str(home))
    env.pop("PYTHONPATH", None)
    env.setdefault("DISPLAY", ":92")

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

            tools = {
                tool["name"] for tool in _call(endpoint, "tools/list")["tools"]
            }
            assert {"render_settings_get", "render_settings_set"} <= tools, tools

            current = _tool(endpoint, "render_settings_get")["settings"]
            original_scale = current["render_scale"]
            result = _call(
                endpoint,
                "tools/call",
                {
                    "name": "render_settings_set",
                    "arguments": {"render_scale": -1.0},
                },
            )
            assert result.get("isError") is True, result
            assert any(
                "render_scale" in block.get("text", "")
                for block in result.get("content", [])
            ), result
            assert any(
                "between 0.25 and 1.0" in block.get("text", "")
                for block in result.get("content", [])
            ), result

            current = _tool(endpoint, "render_settings_get")["settings"]
            assert current["render_scale"] == original_scale
            assert app.poll() is None, "app exited after invalid render scale"
        finally:
            app.terminate()
            try:
                app.wait(timeout=10)
            except subprocess.TimeoutExpired:
                app.kill()
                app.wait(timeout=10)
