# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression coverage for unregistering a Python panel after poll fails."""

import os
import subprocess
import time

import pytest

from test_render_on_demand_idle import _call, _initialize, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def test_unregister_after_panel_poll_exception_keeps_editor_alive(tmp_path):
    executable = os.environ.get("LFS_EXECUTABLE")
    if not executable:
        pytest.skip("requires LFS_EXECUTABLE and an isolated X display")

    endpoint = f"http://127.0.0.1:{os.environ.get('LFS_MCP_PORT', '45704')}/mcp"
    port = endpoint.split(":")[-1].split("/")[0]
    env = dict(os.environ, HOME=str(tmp_path / "home"))
    (tmp_path / "home").mkdir()

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

            registered = _tool(
                endpoint,
                "editor_run",
                {
                    "code": (
                        "import time, lichtfeld as lf\n"
                        "poll_calls = 0\n"
                        "class PollFaultPanel(lf.ui.Panel):\n"
                        "    id = 'py_fault_panel_poll_regression'\n"
                        "    label = 'Fault poll regression'\n"
                        "    @classmethod\n"
                        "    def poll(cls, ctx):\n"
                        "        global poll_calls\n"
                        "        poll_calls += 1\n"
                        "        raise RuntimeError('panel poll failure')\n"
                        "lf.register_class(PollFaultPanel)\n"
                        "lf.ui.set_panel_enabled(PollFaultPanel.id, True)\n"
                        "lf.ui.request_redraw()\n"
                        "time.sleep(0.6)\n"
                        "assert poll_calls > 0, 'panel poll callback did not run'\n"
                    ),
                    "wait_for_completion": True,
                    "wait_for_output": True,
                    "timeout_ms": 7000,
                    "output_max_chars": 2000,
                },
            )
            assert registered["completed"] and registered["success"], registered

            removed = _tool(
                endpoint,
                "editor_run",
                {
                    "code": "lf.unregister_class(PollFaultPanel)",
                    "wait_for_completion": True,
                    "wait_for_output": True,
                    "timeout_ms": 7000,
                    "output_max_chars": 2000,
                },
            )
            assert removed["completed"] and removed["success"], removed

            time.sleep(1.0)
            assert app.poll() is None, "app exited after unregistering the failed panel"
            recovered = _tool(
                endpoint,
                "editor_run",
                {
                    "code": "print('editor recovered')",
                    "wait_for_completion": True,
                    "wait_for_output": True,
                    "timeout_ms": 7000,
                    "output_max_chars": 2000,
                },
            )
            assert recovered["completed"] and recovered["success"], recovered
        finally:
            app.terminate()
            try:
                app.wait(timeout=10)
            except subprocess.TimeoutExpired:
                app.kill()
                app.wait(timeout=10)
