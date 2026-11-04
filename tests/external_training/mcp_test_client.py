# SPDX-License-Identifier: GPL-3.0-or-later
"""Small JSON-RPC client for renderer integration fixtures, not product code."""
import json
import os
import urllib.request
import sys
from pathlib import Path


class MCP:
    def __init__(self, endpoint=None):
        self.endpoint = endpoint or os.environ.get("LFS_TEST_MCP_ENDPOINT", "http://127.0.0.1:45689/mcp")
        self.serial = 0

    def rpc(self, method, params=None, timeout=60):
        self.serial += 1
        request = urllib.request.Request(self.endpoint,
            data=json.dumps({"jsonrpc": "2.0", "id": self.serial, "method": method,
                             "params": params or {}}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read())
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result.get("result")

    def resource(self, uri):
        return self.rpc("resources/read", {"uri": uri})

    def call(self, name, arguments=None):
        return self.rpc("tools/call", {"name": name, "arguments": arguments or {}})


if __name__ == "__main__":
    client = MCP()
    initialized = client.rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                             "clientInfo": {"name": "training-session-fixture", "version": "0.1"}})
    if len(sys.argv) > 1 and sys.argv[1] == "--dismiss-defaults":
        modal = client.call("ui_modal_get").get("structuredContent", {})
        if modal.get("title") == "Set as Default Application":
            print(client.call("ui_modal_press", {"label": "Don't Ask Again"}))
        raise SystemExit()
    if len(sys.argv) > 1 and sys.argv[1] == "--discard-test-exit":
        modal = client.call("ui_modal_get").get("structuredContent", {})
        assert modal.get("title") == "Exit Application?", modal
        print(client.call("ui_modal_press", {"label": "Discard"}))
        raise SystemExit()
    if len(sys.argv) > 1:
        print(json.dumps(client.call("editor_run", {"code": Path(sys.argv[1]).read_text(encoding="utf-8"),
                         "wait_for_completion": True, "timeout_ms": 2000, "output_max_chars": 4000})))
        if Path(sys.argv[1]).name != "renderer_exit.py":
            print(json.dumps(client.call("editor_get_output", {"max_chars": 5000, "tail": True})))
        raise SystemExit()
    print(json.dumps(initialized))
    tools = client.rpc("tools/list")
    print("TOOLS", json.dumps([tool for tool in tools["tools"] if tool["name"].startswith("editor_")]))
    print("RESOURCES", json.dumps(client.rpc("resources/list")))
    for uri in ("lichtfeld://runtime/catalog", "lichtfeld://runtime/state", "lichtfeld://ui/state",
                "lichtfeld://scene/state", "lichtfeld://selection/current"):
        value = client.resource(uri)
        if uri.endswith("catalog"):
            value = str(value)[:2000]
        print(uri, json.dumps(value))
