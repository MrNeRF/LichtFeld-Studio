#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise a running native viewport workspace through its discovered MCP API.

This replaces the current scene with a generated fixture and writes a temporary
project under --output. Run in a disposable app session. It does not launch the
app. Pillow is required for comparing actual presented pixels.
"""

import argparse
import base64
import json
import math
import struct
from pathlib import Path
import time
import urllib.request


class Probe:
    def __init__(self, endpoint, output):
        self.endpoint = endpoint
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.tools = {}

    def rpc(self, method, params=None):
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": method,
                             "params": params or {}}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=50) as response:
            payload = json.load(response)
        if "error" in payload:
            raise RuntimeError(payload["error"])
        return payload["result"]

    def discover(self):
        self.rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                "clientInfo": {"name": "workspace-probe", "version": "1"}})
        manifest = self.save("tools", self.rpc("tools/list"))
        self.tools = {tool["name"]: tool for tool in manifest["tools"]}
        self.save("resources", self.rpc("resources/list"))
        for resource in ("runtime/catalog", "runtime/state", "ui/state", "scene/state",
                         "selection/current"):
            self.save(resource.replace("/", "-"), self.rpc(
                "resources/read", {"uri": "lichtfeld://" + resource}))
        modal = self.call("ui_modal_get")
        if modal.get("open"):
            raise RuntimeError("Resolve the app's existing modal before running: " + str(modal))

    def raw_call(self, name, **arguments):
        tool = self.tools[name]
        schema = tool["inputSchema"]
        missing = set(schema.get("required", [])) - arguments.keys()
        unknown = arguments.keys() - schema.get("properties", {}).keys()
        if missing or unknown:
            raise RuntimeError(f"Discovered schema mismatch for {name}: {missing=}, {unknown=}")
        result = self.rpc("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            raise RuntimeError(result)
        return result

    def call(self, name, **arguments):
        result = self.raw_call(name, **arguments)
        if "structuredContent" in result:
            data = result["structuredContent"]
        else:
            data = json.loads(next(block["text"] for block in result["content"]
                                   if block["type"] == "text"))
        if isinstance(data, dict) and (data.get("error") or data.get("success") is False):
            raise RuntimeError(data)
        return data

    def save(self, name, data):
        (self.output / (name + ".json")).write_text(json.dumps(data, indent=2))
        return data

    def capture(self, name):
        from PIL import Image
        result = self.raw_call("render_capture_window")
        block = next(block for block in result["content"] if block["type"] == "image")
        path = self.output / (name + ".png")
        path.write_bytes(base64.b64decode(block["data"]))
        # Ignore alpha: the difference of two opaque RGBA images has zero alpha.
        return Image.open(path).convert("RGB")

    def state(self):
        return self.call("workspace_get")

    def wait(self, predicate, name, timeout=30):
        deadline = time.monotonic() + timeout
        while True:
            state = self.state()
            if predicate(state):
                return self.save(name, state)
            if time.monotonic() >= deadline:
                self.save(name + "-timeout", state)
                raise RuntimeError("Workspace did not reach " + name)
            time.sleep(.1)

    def ready(self, count, name):
        def published(state):
            visible = [view for view in state["views"] if view["visible"]]
            return len(visible) == count and all(
                view["published"] and view["color_has_image"] and view["depth_has_image"]
                and view["matches_viewport_extent"]
                and view["render_size"] == view["framebuffer_size"]
                for view in visible)
        self.wait(published, name)

        # A published frame can still be followed by GPU LOD page arrivals.
        # Those arrivals republish an unchanged camera pane while the shared
        # page cache settles.  Do not compare presented pixels until every
        # visible pane's color/depth generations have held steady.
        deadline = time.monotonic() + 30
        previous = None
        stable_polls = 0
        last_state = None
        while time.monotonic() < deadline:
            state = self.state()
            last_state = state
            if not published(state):
                previous = None
                stable_polls = 0
            else:
                key = tuple(sorted(
                    (view["id"], view["color_external_generation"],
                     view["depth_external_generation"])
                    for view in state["views"] if view["visible"]))
                if key == previous:
                    stable_polls += 1
                else:
                    previous = key
                    stable_polls = 1
                if stable_polls >= 8:
                    return self.save(name, state)
            time.sleep(.1)
        self.save(name + "-unstable-timeout", last_state)
        raise RuntimeError("Workspace generations did not stabilize for " + name)

    def load(self, path):
        self.call("scene_load_ply", path=str(path))
        job = self.call("runtime_job_wait", job_id="import.dataset", until="inactive",
                        timeout_ms=30000)
        if job.get("active") or job.get("success") is not True:
            raise RuntimeError(job)


def write_fixture(path):
    properties = ("x y z f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 "
                  "rot_0 rot_1 rot_2 rot_3").split()
    rows = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                rows.append([x * .35 - .7, y * .3 - .6, z * .4 - .8,
                             (x / 4 - .5) / .28209479,
                             (y / 4 - .5) / .28209479,
                             (z / 4 - .5) / .28209479,
                             math.log(19), *([math.log(.06)] * 3), 1, 0, 0, 0])
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(rows)}"]
    header += [f"property float {prop}" for prop in properties]
    path.write_bytes(("\n".join(header + ["end_header"]) + "\n").encode() +
                     b"".join(struct.pack("<14f", *row) for row in rows))


def durable(state):
    fields = ("id", "eye", "target", "up", "orthographic", "ortho_scale",
              "focal_length_mm", "near_plane", "far_plane", "grid_plane")
    return [{key: view[key] for key in fields} for view in state["views"]]


def verify(probe):
    from PIL import ImageChops
    probe.discover()
    fixture = probe.output / "workspace-fixture.ply"
    write_fixture(fixture)
    probe.call("render_settings_set", scene_upscaler="native", scene_upscaler_preset="native",
               render_scale=1.0, point_cloud_mode=False, show_grid=False,
               environment_mode=0, background_color=[0, 0, 0])
    single = probe.call("workspace_set_layout", layout="single")
    primary = single["views"][0]["id"]
    probe.call("workspace_set_editor", view_id=primary, editor_id="viewport")
    probe.call("workspace_set_camera", view_id=primary, eye=[3, 2, 5],
               target=[0, 0, 0], up=[0, 1, 0])
    probe.call("workspace_set_projection", view_id=primary, orthographic=False,
               focal_length_mm=35)
    probe.load(fixture)
    probe.ready(1, "single")
    probe.call("workspace_set_layout", layout="quad")
    before = probe.ready(4, "quad")
    ids = [view["id"] for view in before["views"]]
    probe.call("workspace_focus", view_id=ids[0])
    before = probe.ready(4, "before-camera")
    assert before["focused"] == ids[0]
    assert before["active_viewport"] == ids[0], before
    image_before = probe.capture("before-camera")
    source = before["views"][0]
    probe.call("workspace_set_camera", view_id=source["id"],
               eye=[a + b for a, b in zip(source["eye"], [2, 0, 0])],
               target=[a + b for a, b in zip(source["target"], [2, 0, 0])], up=[0, 1, 0])
    probe.wait(lambda state: state["views"][0]["color_external_generation"] >
               source["color_external_generation"], "camera-published")
    after = probe.ready(4, "after-camera")
    image_after = probe.capture("after-camera")
    assert durable(before)[1:] == durable(after)[1:], "Moving a camera changed another view"
    pixels = []
    for view in before["views"]:
        x, y, width, height = view["rect"]
        scale_x = view["framebuffer_size"][0] / width
        scale_y = view["framebuffer_size"][1] / height
        box = (round((x + width / 5) * scale_x), round((y + height / 4) * scale_y),
               round((x + width * 4 / 5) * scale_x), round((y + height * 3 / 4) * scale_y))
        changed = ImageChops.difference(image_before.crop(box), image_after.crop(box)).getbbox() is not None
        pixels.append({"view": view["id"], "changed": changed})
        assert changed == (view["id"] == source["id"]), pixels
    probe.save("camera-pixels", pixels)
    probe.call("workspace_maximize", view_id=ids[2])
    maximized = probe.ready(1, "maximized")
    assert [view["id"] for view in maximized["views"] if view["visible"]] == [ids[2]]
    probe.capture("maximized")
    probe.call("workspace_maximize")
    restored = probe.ready(4, "restored")
    assert durable(after) == durable(restored), "Maximize/restore changed camera state"
    probe.call("workspace_resize", split_id=restored["splitters"][-1]["id"], ratio=.62)
    resized = probe.ready(4, "resized")
    assert any(abs(split["ratio"] - .62) < .001 for split in resized["splitters"])
    probe.call("workspace_close", view_id=ids[-1])
    probe.ready(3, "closed")
    probe.call("workspace_split", view_id=ids[2], axis="vertical")
    reopened = probe.ready(4, "reopened")
    new_ids = set(view["id"] for view in reopened["views"]) - set(ids)
    assert len(new_ids) == 1 and min(new_ids) > max(ids), "Closed view ID was recycled"
    project = probe.output / "workspace-roundtrip.licht"
    probe.call("project_save_as", path=str(project))
    saved = probe.save("saved", probe.state())
    probe.call("workspace_set_layout", layout="single")
    probe.call("project_open", path=str(project), discard_changes=True)
    loaded = probe.ready(4, "loaded")
    assert durable(saved) == durable(loaded), "Project round trip changed view state"
    assert saved["splitters"] == loaded["splitters"], "Project round trip changed layout"
    probe.capture("loaded")
    probe.save("result", {"passed": True, "checks": ["four published color/depth outputs",
               "independent camera state and presented pixels", "maximize and restore",
               "splitter resize", "close and split identity", "native project round trip"]})
    print("Native workspace checks passed. Artifacts:", probe.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:45677/mcp")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    verify(Probe(args.endpoint, args.output.resolve()))
