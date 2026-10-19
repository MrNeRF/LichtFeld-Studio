# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reject null for optional boolean MCP tool arguments before handlers run."""

import os
import subprocess
import time

import pytest

from test_render_on_demand_idle import _call, _initialize

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def test_ui_tool_list_null_include_poll_is_invalid_argument(tmp_path):
    executable = os.environ.get("LFS_EXECUTABLE")
    if not executable:
        pytest.skip("requires LFS_EXECUTABLE")

    port = os.environ.get("LFS_MCP_PORT", "45767")
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

            listed = _call(endpoint, "tools/list")["tools"]
            tool = next(item for item in listed if item["name"] == "ui_tool_list")
            include_poll = tool["inputSchema"]["properties"]["include_poll"]
            assert include_poll["type"] == "boolean"
            assert "include_poll" not in tool["inputSchema"].get("required", [])

            response = _call(
                endpoint,
                "tools/call",
                {
                    "name": "ui_tool_list",
                    "arguments": {"include_poll": None},
                },
            )
            assert response.get("isError") is True, response
            structured = response.get("structuredContent", {})
            assert structured.get("error", {}).get("code") == "InvalidArgument", structured
            assert (
                structured.get("error", {}).get("details", {}).get("parameter")
                == "include_poll"
            ), structured

            followup = _call(
                endpoint,
                "tools/call",
                {"name": "ui_tool_list", "arguments": {"include_poll": False}},
            )
            assert followup.get("isError") is False, followup
            assert followup.get("structuredContent", {}).get("success") is True
            assert app.poll() is None, "app exited after invalid UI tool arguments"
        finally:
            app.terminate()
            try:
                app.wait(timeout=10)
            except subprocess.TimeoutExpired:
                app.kill()
                app.wait(timeout=10)
