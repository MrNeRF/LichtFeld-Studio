#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Verify displayed tensor uploads in a running local Studio MCP session.

Select CUDA or Vulkan in Studio's tensor preferences and restart before running.
The current scene stays loaded. The script creates a temporary floating panel,
checks small HWC images, then performs 96 displayed updates with 23 resizes.
Results and presented captures are written to --output. Run once per backend.
"""

import argparse
import base64
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcp-url", default="http://127.0.0.1:45677/mcp")
    parser.add_argument("--backend", choices=("cuda", "vulkan"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-pixel", nargs=2, type=int, metavar=("X", "Y"),
                        help="Check a point inside the displayed image (requires Pillow)")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    def rpc(method, params):
        request = urllib.request.Request(
            args.mcp_url,
            data=json.dumps({"jsonrpc": "2.0", "id": 1, "method": method,
                             "params": params}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.load(response)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["result"]

    def call(name, arguments=None, raw=False):
        result = rpc("tools/call", {"name": name, "arguments": arguments or {}})
        if result.get("isError"):
            raise RuntimeError(result)
        if raw:
            return result
        for item in result.get("content", []):
            if item["type"] == "text":
                value = json.loads(item["text"])
                if value.get("error") or value.get("success") is False:
                    raise RuntimeError(value)
                return value
        raise RuntimeError(result)

    rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "ui-tensor-verification", "version": "1"}})
    available = {t["name"] for t in rpc("tools/list", {})["tools"]}
    required = {"editor_run", "camera_get", "render_reconstruction_sample_frame", "render_capture"}
    if not required <= available:
        raise RuntimeError(f"Missing MCP tools: {sorted(required - available)}")
    for uri in ("runtime/catalog", "runtime/state", "ui/state", "scene/state", "selection/current"):
        rpc("resources/read", {"uri": f"lichtfeld://{uri}"})
    if {"runtime_job_describe", "runtime_job_control"} <= available:
        job = call("runtime_job_describe", {"job_id": "import.dataset"})
        if not job.get("active") and job.get("actions", {}).get("dismiss"):
            call("runtime_job_control", {"job_id": "import.dataset", "action": "dismiss"})

    camera = call("camera_get")["camera"]
    view = {k: camera[k] for k in ("eye", "target", "up", "fov_degrees")}
    marker = output / "callback.json"
    setup = '''import lichtfeld as lf
class TensorVerificationPanel(lf.ui.Panel):
    id = 'test.tensor_upload_verification'
    label = 'Tensor upload verification'
    space = lf.ui.PanelSpace.FLOATING
    size = (300, 270)
    def draw(self, ui):
        ui.image(lf._upload_verify_texture.id, (192, 192))
        ui.label('Tensor upload verification')
def setup():
    lf._upload_verify_texture = lf.ui.DynamicTexture(lf.Tensor.full([64,64,3], .5, device='cuda'))
    lf._upload_verify_panel = TensorVerificationPanel
    lf.register_class(TensorVerificationPanel)
lf.ui.schedule_on_ui_thread(setup)
'''
    results = []
    try:
        call("editor_run", {"code": setup, "show_console": False})
        call("render_reconstruction_sample_frame", view)
        shapes = [(1, 1920, 3), (3, 9, 4), (4, 7, 3), (3, 3, 3)]
        shapes += [(64, 64, 3) if (i // 4) % 2 == 0 else (71, 93, 3) for i in range(96)]
        for index, (height, width, channels) in enumerate(shapes):
            value = .125 if index % 2 == 0 else .875
            code = f'''import lichtfeld as lf
import pathlib, json, traceback, numpy as np
def update():
    result = {{'index': {index}, 'success': False}}
    try:
        data = np.full(({height},{width},{channels}), {value}, dtype=np.float32)
        if {index} < 4:
            data[...,0] = .1; data[...,1] = .5; data[...,2] = .9
        if {channels} == 4:
            data[...,3] = 1
        if {index} >= 4 and {index} % 2 == 0:
            storage = np.full(({height},{width}+1,{channels}), .3, dtype=np.float32)
            storage[:,:{width},:] = data
            source = lf.Tensor.from_numpy(storage).gpu()[:,:{width},:]
            assert not source.is_contiguous
        else:
            source = lf.Tensor.from_numpy(data).gpu()
        assert source.backend == {args.backend!r}, source.backend
        previous_id = lf._upload_verify_texture.id
        lf._upload_verify_texture.update(source)
        assert lf._upload_verify_texture.valid
        assert lf._upload_verify_texture.id == previous_id and previous_id != 0
        assert lf._upload_verify_texture.width == {width}
        assert lf._upload_verify_texture.height == {height}
        assert np.array_equal(source.cpu().numpy(), data), 'Upload modified its source'
        result.update(success=True, backend=source.backend, shape=list(data.shape))
    except Exception:
        result['error'] = traceback.format_exc()
    path = pathlib.Path({str(marker)!r})
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result))
    temporary.replace(path)
lf.ui.schedule_on_ui_thread(update)
'''
            call("editor_run", {"code": code, "show_console": False})
            call("render_reconstruction_sample_frame", view)
            result = json.loads(marker.read_text())
            if not result.get("success") or result["index"] != index:
                raise RuntimeError(result)
            results.append(result)
            if index in (3, 4, 51, 99):
                capture = call("render_capture", {"presented": True}, raw=True)
                images = [c for c in capture.get("content", []) if c["type"] == "image"]
                if not images:
                    raise RuntimeError(capture)
                path = output / f"frame-{index}.png"
                path.write_bytes(base64.b64decode(images[0]["data"]))
                if args.sample_pixel:
                    from PIL import Image
                    actual = Image.open(path).convert("RGB").getpixel(tuple(args.sample_pixel))
                    expected = (26, 128, 230) if index < 4 else ((32,) * 3 if index % 2 == 0 else (223,) * 3)
                    if any(abs(a - b) > 1 for a, b in zip(actual, expected)):
                        raise RuntimeError(f"Frame {index}: pixel {actual}, expected {expected}")
    finally:
        (output / "results.json").write_text(json.dumps(results, indent=2))
        cleanup = '''import lichtfeld as lf
def cleanup():
    if hasattr(lf, '_upload_verify_panel'):
        lf.unregister_class(lf._upload_verify_panel)
        del lf._upload_verify_panel
    if hasattr(lf, '_upload_verify_texture'):
        lf._upload_verify_texture.destroy()
        assert not lf._upload_verify_texture.valid and lf._upload_verify_texture.id == 0
        del lf._upload_verify_texture
lf.ui.schedule_on_ui_thread(cleanup)
'''
        call("editor_run", {"code": cleanup, "show_console": False})
        call("render_reconstruction_sample_frame", view)
    print(f"Passed: {len(results)} displayed uploads on {args.backend}; captures in {output}")


if __name__ == "__main__":
    main()
