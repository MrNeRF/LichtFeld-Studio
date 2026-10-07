# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Optimization parameter bounds are enforced by direct and generic setters."""

import os
import subprocess
import time
from pathlib import Path

import pytest

from test_render_on_demand_idle import _initialize, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def test_optimization_setters_validate_contract_without_mutating_on_error(tmp_path):
    executable = os.environ.get("LFS_EXECUTABLE")
    if not executable:
        pytest.skip("requires LFS_EXECUTABLE")

    port = os.environ.get("LFS_MCP_PORT", "45776")
    endpoint = f"http://127.0.0.1:{port}/mcp"
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, HOME=str(home))
    env.pop("PYTHONPATH", None)
    env.setdefault("DISPLAY", ":92")
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
    else:
        env.pop("PYTHONHOME", None)

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

            result = _tool(endpoint, "editor_run", {
                "code": (
                    "import lichtfeld as lf\n"
                    "opt = lf.optimization_params()\n"
                    "def reject_unchanged(name, value, setter):\n"
                    "    original = opt.get(name)\n"
                    "    try:\n"
                    "        setter(value)\n"
                    "    except ValueError as error:\n"
                    "        assert name in str(error), str(error)\n"
                    "    else:\n"
                    "        raise AssertionError(f'accepted invalid {name}={value!r}')\n"
                    "    assert opt.get(name) == original, f'{name} changed after rejected assignment'\n"
                    "reject_unchanged('means_lr', -1.0, lambda value: setattr(opt, 'means_lr', value))\n"
                    "reject_unchanged('means_lr', -1.0, lambda value: opt.set('means_lr', value))\n"
                    "reject_unchanged('iterations', 0, lambda value: setattr(opt, 'iterations', value))\n"
                    "reject_unchanged('iterations', 0, lambda value: opt.set('iterations', value))\n"
                    "reject_unchanged('gradient_loss_weight', float('nan'), lambda value: opt.set('gradient_loss_weight', value))\n"
                    "if opt.strategy == 'mrnf':\n"
                    "    opt.shs_lr = -1.0\n"
                    "    assert opt.shs_lr == -1.0\n"
                    "old_interval = opt.get('sh_degree_interval')\n"
                    "old_iterations = opt.iterations\n"
                    "opt.set('sh_degree_interval', 50)\n"
                    "opt.iterations = 2_000_000\n"
                    "assert opt.get('sh_degree_interval') == 50 and opt.iterations == 2_000_000\n"
                    "opt.set('sh_degree_interval', old_interval)\n"
                    "opt.iterations = old_iterations\n"
                    "print('REJECTED')\n"
                ),
                "show_console": False,
                "timeout_ms": 10000,
                "output_max_chars": 2500,
            })
            assert result.get("completed") and not result.get("timed_out"), result
            assert "REJECTED" in result.get("output", {}).get("text", ""), result
            assert app.poll() is None, "app exited after invalid optimization parameters"
        finally:
            app.terminate()
            try:
                app.wait(timeout=10)
            except subprocess.TimeoutExpired:
                app.kill()
                app.wait(timeout=10)
