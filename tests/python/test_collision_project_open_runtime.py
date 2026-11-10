# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise deferred Python project opens in a separate, disposable app instance.

Requires LFS_EXECUTABLE, LFS_TEST_COLLISION_PROJECT_OPEN_EXE and a display.
The native fixture writer is built from test_collision_project_open.cpp.
Run directly with Python or collect through pytest.
"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
import urllib.request
from unittest.mock import patch


class CollisionProjectOpenRuntimeTest(unittest.TestCase):
    def test_optional_routes_report_deferred_until_collision_installation(self):
        executable = os.environ.get("LFS_EXECUTABLE")
        fixture_writer = os.environ.get("LFS_TEST_COLLISION_PROJECT_OPEN_EXE")
        if not executable or not fixture_writer:
            self.skipTest("requires app and native collision fixture writer executables")
        with tempfile.TemporaryDirectory(prefix="lfs-collision-open-") as directory:
            directory = Path(directory)
            project = directory / "collision.licht"
            subprocess.run(
                [fixture_writer, "--gtest_filter=CollisionProjectOpenFixture.*"],
                env={**os.environ, "LFS_TEST_COLLISION_OPEN_FIXTURE": str(project)},
                check=True, timeout=30,
            )
            home = directory / "home"
            (home / "config").mkdir(parents=True)
            (home / "config/preferences.json").write_text(
                json.dumps({"file_association": "declined"})
            )
            port = os.environ.get("LFS_COLLISION_TEST_PORT", "45843")
            endpoint = f"http://127.0.0.1:{port}/mcp"
            environment = {
                "HOME": str(home), "LFS_HOME": str(home),
                "XDG_CONFIG_HOME": str(home / ".config"),
                "XDG_CACHE_HOME": str(home / ".cache"),
                "XDG_DATA_HOME": str(home / ".local/share"),
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
                "UV_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
                "LFS_MCP_ENDPOINT": endpoint,
                "LFS_MCP_BRIDGE_LOG": str(directory / "app.log"),
                "LFS_MCP_BRIDGE_LOCK": str(directory / "bridge.lock"),
            }
            with patch.dict(os.environ, environment):
                os.environ.pop("PYTHONHOME", None)
                os.environ.pop("PYTHONPATH", None)
                bridge_path = Path(__file__).resolve().parents[2] / "scripts/lichtfeld_mcp_bridge.py"
                spec = importlib.util.spec_from_file_location("collision_open_bridge", bridge_path)
                bridge = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(bridge)
                self.assertFalse(bridge.endpoint_ready(), "test port already in use")
                bridge.pick_launch_command = lambda: [executable, "--no-splash", "--mcp-port", port]

                def call(method, params=None):
                    request = urllib.request.Request(
                        endpoint,
                        data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": method,
                                         "params": params or {}}).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(request, timeout=30) as response:
                        result = json.loads(response.read())
                    self.assertNotIn("error", result)
                    return result.get("result", {})

                def tool(name, arguments=None):
                    result = call("tools/call", {"name": name, "arguments": arguments or {}})
                    self.assertFalse(result.get("isError"), result)
                    return result["structuredContent"]

                def editor(code):
                    result = tool("editor_run", {"code": code, "show_console": False,
                                                 "timeout_ms": 10000})
                    if result.get("running"):
                        result = tool("editor_wait", {"timeout_ms": 10000})
                    self.assertTrue(result.get("completed") and result.get("success"), result)
                    self.assertNotIn("Traceback", result.get("output", {}).get("text", ""))

                try:
                    bridge.ensure_server_ready()
                    call("initialize", {"protocolVersion": "2025-03-26", "capabilities": {},
                                        "clientInfo": {"name": "collision-open-test", "version": "1"}})
                    call("notifications/initialized")
                    tools = call("tools/list")["tools"]
                    self.assertTrue({"editor_run", "project_get_info"} <= {t["name"] for t in tools})
                    call("resources/list")
                    for uri in ("runtime/catalog", "runtime/state", "ui/state", "scene/state", "selection/current"):
                        call("resources/read", {"uri": "lichtfeld://" + uri})
                    for option in ("keep_asset_manager_open", "stop_training"):
                        editor(f"""
import lichtfeld as lf
lf.get_scene().add_group("Previous scene marker")
outcome = lf.project_open({str(project)!r}, discard_changes=True, {option}=True)
assert outcome == lf.ProjectOpenOutcome.DEFERRED, repr(outcome)
""")
                        deadline = time.monotonic() + 20
                        while time.monotonic() < deadline:
                            info = tool("project_get_info")
                            if info.get("hydration_state") == "complete":
                                break
                            time.sleep(0.05)
                        else:
                            self.fail(f"project did not finish opening: {info}")
                        editor("""
names = [node.name for node in lf.get_scene().get_nodes()]
assert "Deferred collision" in names, names
assert "Previous scene marker" not in names, names
""")
                finally:
                    bridge.cleanup_spawned_processes()


if __name__ == "__main__":
    unittest.main()
