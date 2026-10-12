# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Direct scene exports preserve requested paths and report misleading names."""

import json
import os
from pathlib import Path
import socket
import subprocess
import time

import pytest

from test_render_on_demand_idle import _call, _initialize

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _mcp(endpoint, name, arguments):
    result = _call(endpoint, "tools/call", {"name": name, "arguments": arguments})
    return result.get("structuredContent", result)


def _editor(endpoint, code):
    result = _mcp(endpoint, "editor_run", {"code": code, "wait_for_completion": True,
                                            "timeout_ms": 120000, "output_max_chars": 4000})
    assert result.get("success") and result.get("completed"), result
    return result["output"]["text"]


def _state(endpoint, capture):
    _editor(endpoint, f"import json, lichtfeld as lf; open({str(capture)!r}, 'w').write(json.dumps(lf.ui.get_export_state()))")
    return json.loads(capture.read_text())


def _python_export(endpoint, format_id, path):
    _editor(endpoint, f"import lichtfeld as lf; lf.export_scene({format_id}, {str(path)!r}, ['path_probe'], 0)")
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        state = _state(endpoint, path.parent / "export-state.json")
        if state["path"] == str(path) and not state["active"] and state["outcome"] != "idle":
            return state
        time.sleep(0.2)
    pytest.fail(f"export did not settle: {path.name}")


def test_direct_scene_export_path_contract(tmp_path: Path):
    executable = os.environ.get("LFS_EXECUTABLE")
    if not executable:
        pytest.skip("requires an app binary and an isolated X display")

    port = _port()
    endpoint = f"http://127.0.0.1:{port}/mcp"
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, HOME=str(home), XDG_CONFIG_HOME=str(home / "config"),
               XDG_CACHE_HOME=str(home / "cache"), XDG_DATA_HOME=str(home / "data"),
               SDL_VIDEODRIVER="x11", HTTP_PROXY="http://127.0.0.1:9",
               HTTPS_PROXY="http://127.0.0.1:9", NO_PROXY="127.0.0.1,localhost")
    env.pop("WAYLAND_DISPLAY", None)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    with (tmp_path / "app.log").open("w") as log:
        app = subprocess.Popen([executable, "--no-splash", "--no-download", "--mcp-port", str(port)],
                               env=env, stdout=log, stderr=subprocess.STDOUT)
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

            names = {tool["name"] for tool in _call(endpoint, "tools/list")["tools"]}
            def tool_name(fmt):
                return next(name for name in (f"scene_export_{fmt}", f"scene.export_{fmt}") if name in names)

            _editor(endpoint, """
import lichtfeld as lf
count = 256
means = lf.Tensor.randn((count, 3), device='cuda')
sh0 = lf.Tensor.zeros((count, 1, 3), device='cuda')
shn = lf.Tensor.zeros((count, 0, 3), device='cuda')
scaling = lf.Tensor.full((count, 3), -2., device='cuda')
rotation = lf.Tensor.zeros((count, 4), device='cuda')
rotation[:, 0] = 1.
opacity = lf.Tensor.full((count, 1), 2., device='cuda')
lf.get_scene().add_splat('path_probe', means, sh0, shn, scaling, rotation, opacity, sh_degree=0)
""")

            for format_id, fmt, wrong in ((0, "ply", "spz"), (1, "sog", "ply"), (2, "spz", "sog")):
                for suffix, expected in (("", "warn"), (".bak", "warn"),
                                         (".glb", "warn"), (".html", "warn"),
                                         (f".{wrong}", "reject"), (f".{fmt}", "match")):
                    for route in ("python", "mcp"):
                        path = tmp_path / f"{route}-{fmt}-{expected}{suffix}"
                        if route == "python":
                            result = _python_export(endpoint, format_id, path)
                            succeeded = result["outcome"] == "completed"
                        else:
                            result = _mcp(endpoint, tool_name(fmt), {"path": str(path), "node": "path_probe"})
                            succeeded = result.get("success") is True

                        if expected == "reject":
                            assert not succeeded and not path.exists(), (path, result)
                            error = result["error"]
                            message = error["message"] if isinstance(error, dict) else error
                            assert f".{fmt}" in message, result
                        else:
                            assert succeeded and path.is_file(), (path, result)
                            assert result["path"] == str(path), result
                            if expected == "warn":
                                assert fmt.upper() in result["warning"], result
                            else:
                                assert not result.get("warning"), result
        finally:
            app.terminate()
            try:
                app.wait(timeout=15)
            except subprocess.TimeoutExpired:
                app.kill()
                app.wait(timeout=5)
