#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise PR 2623 in a disposable, visible viewer started by the MCP bridge.

Loads the supplied generated fixture and changes that viewer's comparison/layout.
Uses synchronous captures and runtime jobs; no timing sleeps. Apple manual checks
and memory measurements are described in docs/reviews/pr2623-validation.md.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path


class Check:
    def __init__(self, endpoint, output):
        self.endpoint, self.output, self.records = endpoint, output, []

    def rpc(self, method, params=None):
        payload = {"jsonrpc": "2.0", "id": len(self.records) + 1, "method": method}
        if params is not None:
            payload["params"] = params
        request = urllib.request.Request(self.endpoint, json.dumps(payload).encode(),
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.load(response)
        self.records.append({"method": method, "params": params, "response": result})
        if "error" in result:
            raise RuntimeError(result["error"])
        return result.get("result", {})

    def tool(self, name, **arguments):
        result = self.rpc("tools/call", {"name": name, "arguments": arguments})
        data = result.get("structuredContent")
        if data is None:
            data = json.loads(next(v["text"] for v in result["content"] if v["type"] == "text"))
        if result.get("isError") or "error" in data or data.get("success") is False:
            raise RuntimeError(data)
        return data

    def python(self, code):
        result = self.tool("editor_run", code=code, show_console=False,
                           wait_for_completion=True, timeout_ms=15000)
        output = result["output"]["text"]
        if not result["completed"] or "Traceback" in output or "AssertionError" in output:
            raise RuntimeError(result)
        return output

    def identity(self, uid, native):
        deadline = time.monotonic() + 45
        while True:
            result = self.python("a=lf.capture_viewport().image\n"
                "samples=[a[int(a.shape[0]*y),int(a.shape[1]*x)].tolist() for y in (.4,.5,.6) for x in (.2,.25,.3)]\n"
                "rgb=[sorted(p[c] for p in samples)[4] for c in range(3)]\n"
                "print('PIXEL',lf.ui.get_current_camera_id(),lf.ui.is_gt_comparison_actual_size_active(),*[round(v,6) for v in rgb])")
            parts = result.split()
            if (len(parts) == 6 and parts[0] == "PIXEL" and int(parts[1]) == uid
                    and (parts[2] == "True") == native
                    and abs(float(parts[3]) - (224 if uid == 0 else 24) / 255) < .003):
                print("Presented", uid, "native", native, "RGB", parts[3:])
                return
            if time.monotonic() >= deadline:
                raise AssertionError((uid, native, result))

    def run(self, fixture):
        self.rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                 "clientInfo": {"name": "pr2623-check", "version": "1"}})
        self.rpc("tools/list")
        self.rpc("resources/list")
        for uri in ("runtime/catalog", "runtime/state", "ui/state", "scene/state", "selection/current"):
            self.rpc("resources/read", {"uri": "lichtfeld://" + uri})
        modal = self.tool("ui_modal_get")
        if modal.get("open"):
            raise RuntimeError("Dismiss the startup modal in the disposable viewer before testing: " + modal.get("title", ""))
        self.tool("scene_load_dataset", path=str(fixture.resolve()))
        self.tool("runtime_job_wait", job_id="import.dataset", until="inactive", timeout_ms=30000)
        self.python("import lichtfeld as lf\n"
                    "if not lf.ui.is_gt_comparison_active(): lf.ui.toggle_gt_comparison()")
        for native in (False, True):
            self.python("s=lf.get_render_settings()\ns.gt_comparison_actual_size=" + str(native))
            for uid in (0, 1):
                self.tool("camera_go_to_dataset_camera", uid=uid)
                self.identity(uid, native)
            # Exercise rapid source replacement; deterministic in-flight completion
            # ordering is covered by RenderingManagerGTComparisonReviewTest.
            for uid in (0, 1, 0, 1):
                self.tool("camera_go_to_dataset_camera", uid=uid)
            self.identity(1, native)

        views = self.tool("render_view_states")["views"]
        view = next(v for v in views if v["active"])
        owner = view["id"]
        x, y, w, h = view["rect"]
        if w < 350 or h < 300:
            raise RuntimeError("Restore the disposable viewer to a usable visible size")
        self.python("_before=lf.capture_viewport().image")
        self.tool("ui_pointer", action="drag", button="right",
                  **{"from": [x+w*.30, y+h*.45], "to": [x+w*.30+30, y+h*.45+20], "steps": 8})
        self.python("_after=lf.capture_viewport().image\n"
            "aa=_after[100:140,80:140].tolist()\nbb=_before[80:120,50:110].tolist()\n"
            "err=max(abs(x-y) for ra,rb in zip(aa,bb) for pa,pb in zip(ra,rb) for x,y in zip(pa,pb))\n"
            "print('PAN_ERROR',err)\nassert err < 1e-6")
        print("Physical 30x20 drag: exact shifted pixels")

        second = self.tool("screen_split", area=owner, direction="vertical", factor=.5)["area"]
        assert self.tool("view_get_settings", view=second)["settings"]["split_view_mode"] == 0
        self.python("_owner_before=lf.capture_viewport().image[100:180,40:100].tolist()")
        self.tool("view_set_settings", view=second,
                  settings={"orthographic": True, "focal_length_mm": 63.0, "show_grid": False})
        self.python("_owner_after=lf.capture_viewport().image[100:180,40:100].tolist()\n"
            "assert _owner_before == _owner_after\nassert lf.ui.is_gt_comparison_actual_size_active()")
        print("Unrelated view settings: owner pixels preserved")
        self.tool("screen_close", area=second)

        for field in ("orthographic", "equirectangular"):
            self.python("s=lf.get_render_settings()\ns.gt_comparison_actual_size=True")
            self.tool("view_set_settings", view=owner, settings={field: True})
            self.python("assert not lf.get_render_settings().gt_comparison_actual_size")
            self.tool("view_set_settings", view=owner, settings={field: False})
            self.python("assert not lf.get_render_settings().gt_comparison_actual_size")
        self.python("s=lf.get_render_settings()\ns.gt_comparison_actual_size=True")
        self.tool("view_command", view=owner, command="projection")
        self.python("assert not lf.get_render_settings().gt_comparison_actual_size")
        self.tool("view_command", view=owner, command="projection")
        self.python("assert not lf.get_render_settings().gt_comparison_actual_size")
        print("Projection normalization: clears 1:1 and stays in Fit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:45677/mcp")
    parser.add_argument("--fixture", type=Path, required=True,
                        help="Generated fixture larger than the physical viewport, so pan is not clamped")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check = Check(args.endpoint, args.output)
    status = "failed"
    try:
        check.run(args.fixture)
        status = "passed"
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"status": status, "records": check.records}, indent=2), encoding="utf-8")
