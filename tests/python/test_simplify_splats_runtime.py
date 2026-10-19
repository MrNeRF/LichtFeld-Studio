# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression coverage for starting splat simplification in the editor."""

import os
import subprocess
import time

import pytest

from test_render_on_demand_idle import _call, _initialize, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def test_simplify_splats_returns_and_editor_remains_responsive(tmp_path):
    executable = os.environ.get("LFS_EXECUTABLE")
    if not executable:
        pytest.skip("requires LFS_EXECUTABLE and an isolated X display")

    port = os.environ.get("LFS_MCP_PORT", "45708")
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

            started = _tool(
                endpoint,
                "editor_run",
                {
                    "code": (
                        "import lichtfeld as lf\n"
                        "scene = lf.get_scene()\n"
                        "count = 2\n"
                        "means = lf.Tensor.randn((count, 3), device='cuda')\n"
                        "sh0 = lf.Tensor.zeros((count, 1, 3), device='cuda')\n"
                        "shn = lf.Tensor.zeros((count, 0, 3), device='cuda')\n"
                        "scaling = lf.Tensor.full((count, 3), -2., device='cuda')\n"
                        "rotation = lf.Tensor.zeros((count, 4), device='cuda')\n"
                        "rotation[:, 0] = 1.\n"
                        "opacity = lf.Tensor.zeros((count, 1), device='cuda')\n"
                        "scene.add_splat('simplify_probe', means, sh0, shn, "
                        "scaling, rotation, opacity)\n"
                        "lf.simplify_splats('simplify_probe', ratio=.5, lod_base=2., "
                        "opacity_prune_threshold=0.)\n"
                        "print('SIMPLIFY_RETURNED')"
                    ),
                    "timeout_ms": 4000,
                    "output_max_chars": 1000,
                },
            )
            assert started["completed"] and started["success"], started
            assert "SIMPLIFY_RETURNED" in started["output"]["text"], started
            assert app.poll() is None, "app exited while starting simplification"

            recovered = _tool(
                endpoint,
                "editor_run",
                {
                    "code": "print('EDITOR_RESPONSIVE')",
                    "timeout_ms": 5000,
                    "output_max_chars": 1000,
                },
            )
            assert recovered["completed"] and recovered["success"], recovered
            assert "EDITOR_RESPONSIVE" in recovered["output"]["text"], recovered
        finally:
            app.terminate()
            try:
                app.wait(timeout=10)
            except subprocess.TimeoutExpired:
                app.kill()
                app.wait(timeout=10)
