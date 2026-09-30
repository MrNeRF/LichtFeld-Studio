#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Verify visible editor views in an already running MCP-enabled GUI with a loaded scene.

Split the editor into at least two 3D areas first. The test orbits each area through
real mouse input, checks camera/target isolation, and saves composited captures.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request


class Mcp:
    def __init__(self, port):
        self.url = f"http://localhost:{port}/mcp"
        self.session = None
        self.sequence = 0
        self.rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                "clientInfo": {"name": "editor-view-verification", "version": "1"}})
        self.rpc("notifications/initialized", {}, notification=True)
        self.tools = {tool["name"]: tool for tool in self.rpc("tools/list", {})["tools"]}
        self.rpc("resources/list", {})
        for uri in ("runtime/catalog", "runtime/state", "ui/state", "scene/state", "selection/current"):
            self.rpc("resources/read", {"uri": f"lichtfeld://{uri}"})

    def rpc(self, method, params, notification=False):
        self.sequence += 1
        body = {"jsonrpc": "2.0", "method": method, "params": params}
        if not notification:
            body["id"] = self.sequence
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.session:
            headers["Mcp-Session-Id"] = self.session
        request = urllib.request.Request(self.url, json.dumps(body).encode(), headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            self.session = response.headers.get("Mcp-Session-Id", self.session)
            raw = response.read().decode()
        if notification or not raw:
            return None
        if raw.lstrip().startswith("data:") or "\ndata:" in raw:
            raw = next(line[5:].strip() for line in raw.splitlines() if line.startswith("data:"))
        reply = json.loads(raw)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["result"]

    def call(self, name, args=None):
        assert name in self.tools, f"Missing MCP tool: {name}"
        result = self.rpc("tools/call", {"name": name, "arguments": args or {}})
        if result.get("isError"):
            raise RuntimeError(result)
        if "structuredContent" in result:
            return result["structuredContent"]
        for item in result.get("content", []):
            if item["type"] == "image":
                return {"data": item["data"]}
            if item["type"] == "text":
                value = json.loads(item["text"])
                if "error" in value:
                    raise RuntimeError(value["error"])
                return value
        raise RuntimeError("Missing MCP result")

    def views(self):
        return {view["id"]: view for view in self.call("render_view_states")["views"]}

    def settle(self):
        previous = None
        stable = 0
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            current = self.views()
            signature = [(v["id"], v["generation"], v["rotation"], v["position"]) for v in current.values()]
            stable = stable + 1 if signature == previous else 0
            if stable >= 4:
                return current
            previous = signature
            time.sleep(0.15)
        raise AssertionError("View frames did not settle")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=45697)
    parser.add_argument("--display", default=":97")
    parser.add_argument("--window", help="X11 window ID; otherwise discover the visible app window")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    env = dict(os.environ, DISPLAY=args.display)
    window = args.window
    if not window:
        windows = subprocess.check_output(["xdotool", "search", "--onlyvisible", "--name", "LichtFeld"],
                                          env=env, text=True).splitlines()
        assert len(windows) == 1, "Use --window to select one app window"
        window = windows[0]
    args.output.mkdir(parents=True, exist_ok=True)
    mcp = Mcp(args.port)
    before = mcp.settle()
    assert len(before) >= 2, "Split into two or more visible 3D areas before running this test"
    targets = [view["target"] for view in before.values()]
    assert all(targets) and len(set(targets)) == len(targets), "Views must own distinct valid targets"
    assert all(min(view["image_size"]) > 0 for view in before.values()), "Every view must publish a frame"
    capture = lambda name: (args.output / name).write_bytes(base64.b64decode(mcp.call("render_capture_window")["data"]))
    capture(f"{len(before)}-views-before.png")
    evidence = []
    for index, view_id in enumerate(before):
        old = mcp.settle()
        view = old[view_id]
        x, y, width, height = view["rect"]
        x, y = round(x + width * .45), round(y + height * .55)
        dx, dy = 40 + index * 17, 12 + index * 11
        subprocess.run(["xdotool", "mousemove", "--sync", "--window", window, str(x), str(y), "mousedown", "2",
                        "sleep", ".1", "mousemove", "--sync", "--window", window, str(x + dx), str(y + dy),
                        "sleep", ".1", "mouseup", "2"], env=env, check=True)
        time.sleep(.3)
        new = mcp.settle()
        assert old.keys() == new.keys(), "Orbit must preserve the area layout"
        pose = lambda v: (v["rotation"], v["position"])
        assert pose(old[view_id]) != pose(new[view_id]), "The dragged camera must move"
        assert old[view_id]["generation"] != new[view_id]["generation"], "The dragged view must render"
        for other in old.keys() - {view_id}:
            assert pose(old[other]) == pose(new[other]), "Orbit changed another camera"
            assert old[other]["generation"] == new[other]["generation"], "Orbit rerendered an unchanged view"
            assert old[other]["target"] == new[other]["target"]
        evidence.append({"moved_view": view_id, "before": list(old.values()), "after": list(new.values())})
    final = mcp.settle()
    assert len({tuple(v["rotation"]) for v in final.values()}) == len(final), "Cameras must differ"
    capture(f"{len(final)}-views-independent.png")
    (args.output / f"{len(final)}-views-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"PASS: {len(final)} cameras render independently into {len(set(targets))} targets")


if __name__ == "__main__":
    main()
