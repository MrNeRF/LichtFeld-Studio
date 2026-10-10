# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""An owned GUI process for generated-fixture GPU integration tests."""

import os
import select
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from test_render_on_demand_idle import _call, _initialize


def _stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


@contextmanager
def isolated_app(directory):
    root = Path(__file__).resolve().parents[2]
    build = Path(os.environ.get("LFS_TEST_BUILD_DIR", root / "build"))
    executable = Path(os.environ.get("LFS_EXECUTABLE", build / "LichtFeld-Studio"))
    assert executable.is_file(), f"Build the GUI executable first: {executable}"
    home = directory / "home"
    home.mkdir()
    env = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH"):
        env.pop(key, None)
    env.update(HOME=str(home), XDG_CONFIG_HOME=str(home / "config"),
               XDG_CACHE_HOME=str(home / "cache"), XDG_DATA_HOME=str(home / "data"),
               LFS_ASSET_MANAGER_DIR=str(home / "catalog"),
               LFS_ASSET_MANAGER_ASSETS_DIR=str(home / "assets"),
               NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env[key] = "http://127.0.0.1:9"
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = str(reservation.getsockname()[1])
    port = os.environ.get("LFS_TEST_MCP_PORT", port)
    endpoint = f"http://127.0.0.1:{port}/mcp"
    with (directory / "xvfb.log").open("w") as display_log:
        read_fd, write_fd = os.pipe()
        display = subprocess.Popen(
            ["Xvfb", *([os.environ["LFS_TEST_DISPLAY"]] if "LFS_TEST_DISPLAY" in os.environ else []),
             "-displayfd", str(write_fd), "-screen", "0", "1280x960x24", "-nolisten", "tcp"],
            pass_fds=(write_fd,), stdout=display_log, stderr=subprocess.STDOUT,
        )
        os.close(write_fd)
        try:
            assert select.select([read_fd], [], [], 15)[0], "Xvfb did not start"
            number = os.read(read_fd, 32).decode().strip()
            assert number.isdigit(), f"Xvfb failed; see {directory / 'xvfb.log'}"
            env["DISPLAY"] = f":{number}"
            with (directory / "app.log").open("w") as log:
                app = subprocess.Popen(
                    [str(executable), "--no-splash", "--no-download", "--mcp-port", port],
                    env=env, stdout=log, stderr=subprocess.STDOUT,
                )
                try:
                    deadline = time.monotonic() + 90
                    while time.monotonic() < deadline:
                        assert app.poll() is None, f"App exited; see {directory / 'app.log'}"
                        try:
                            _initialize(endpoint)
                            break
                        except (OSError, AssertionError):
                            time.sleep(0.1)
                    else:
                        raise AssertionError(f"MCP did not start; see {directory / 'app.log'}")
                    _call(endpoint, "tools/list")
                    _call(endpoint, "resources/list")
                    for uri in ("runtime/catalog", "runtime/state", "ui/state", "scene/state", "selection/current"):
                        _call(endpoint, "resources/read", {"uri": f"lichtfeld://{uri}"})
                    yield endpoint
                finally:
                    _stop(app)
        finally:
            os.close(read_fd)
            _stop(display)
