#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render and independently decode asymmetric camera-path fixtures.

Requires a CUDA-enabled LichtFeld build and an FFmpeg executable. Examples:
  python scripts/verify_video_orientation.py --executable build/LichtFeld-Studio
  python scripts/verify_video_orientation.py --mcp-url http://localhost:45677/mcp

The executable mode regresses the CLI camera conversion. MCP mode checks the
independent GUI export reference; it does not exercise that CLI conversion.
The MCP mode requires an empty test app: it loads fixtures and changes the
camera, timeline, and render settings. Use a separate --mcp-port app instance.
Artifacts include MP4s, decoded PNGs, GUI window captures, and results.json.
"""

import argparse
import base64
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time
import urllib.request


WIDTH, HEIGHT = 320, 240
FOCAL_MM = 12 * math.sqrt(3)  # 60-degree vertical field of view.
PATCHES = ((-1, 1, (1, 0, 0)), (1, 1, (0, 1, 0)),
           (-1, -1, (0, 0, 1)), (1, -1, (1, 1, 0)))
# Each fixture has identical screen-space landmarks under a different pose.
S = math.sqrt(0.5)
POSES = (
    ("level", (0, 0, 5), (1, 0, 0, 0), lambda x, y, z: (x, y, z)),
    ("roll90", (0, 0, 5), (S, 0, 0, S), lambda x, y, z: (-y, x, z)),
    ("look_down", (0, 5, 0), (S, -S, 0, 0), lambda x, y, z: (x, z, -y)),
    ("look_up", (0, -5, 0), (S, S, 0, 0), lambda x, y, z: (x, -z, y)),
)


class Mcp:
    def __init__(self, url):
        self.url = url
        self.rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                               "clientInfo": {"name": "video-orientation-test", "version": "1"}})
        self.rpc("notifications/initialized", {})
        self.tools = {t["name"]: t for t in self.rpc("tools/list", {})["tools"]}
        self.rpc("resources/list", {})
        for uri in ("runtime/catalog", "runtime/state", "ui/state", "scene/state", "selection/current"):
            self.resource(uri)
        if self.resource("scene/nodes")["count"] or self.call("sequencer_get")["has_keyframes"]:
            raise RuntimeError("Use an empty test app; refusing to replace an existing scene or timeline")

    def rpc(self, method, params):
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        if not method.startswith("notifications/"):
            payload["id"] = 1
        request = urllib.request.Request(self.url, json.dumps(payload).encode(),
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
        result = json.loads(body) if body else {}
        if "error" in result:
            raise RuntimeError(result["error"])
        return result.get("result", {})

    def resource(self, uri):
        return json.loads(self.rpc("resources/read", {"uri": "lichtfeld://" + uri})["contents"][0]["text"])

    def raw_call(self, name, **arguments):
        if name not in self.tools:
            raise RuntimeError(f"Tool unavailable: {name}")
        result = self.rpc("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            raise RuntimeError(result)
        return result

    def call(self, name, **arguments):
        result = self.raw_call(name, **arguments)
        return result.get("structuredContent") or json.loads(result["content"][0]["text"])

    def wait(self, job_id):
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            result = self.call("runtime_job_wait", job_id=job_id, until="inactive", timeout_ms=10000)
            if not result["active"]:
                if result.get("outcome") != "completed":
                    raise RuntimeError(result)
                return
        raise TimeoutError(job_id)


def fixture(root, pose):
    name, eye, quaternion, rotate = pose
    rows = []
    for x, y, color in PATCHES:
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                offset = rotate(x + dx * .075, y + dy * .075, -5)
                world = [eye[i] + offset[i] for i in range(3)]
                raw = (world[0], -world[1], -world[2])
                rows.append((*raw, 0, 0, 0, *((c - .5) / .28209479177387814 for c in color),
                             5, *([math.log(.08)] * 3), 1, 0, 0, 0))
    properties = "x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3".split()
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(rows)}\n" +
              "".join(f"property float {p}\n" for p in properties) + "end_header\n")
    ply = root / f"{name}.ply"
    ply.write_bytes(header.encode() + b"".join(struct.pack("<17f", *row) for row in rows))
    path = root / f"{name}.json"
    path.write_text(json.dumps({"version": 4, "clip_duration": .1, "keyframes": [
        {"time": t, "position": eye, "rotation": quaternion,
         "focal_length_mm": FOCAL_MM, "easing": 0} for t in (0, .1)]}), encoding="utf-8")
    return ply, path


def verify_pixels(raw):
    frame_bytes = WIDTH * HEIGHT * 3
    frame_count, remainder = divmod(len(raw), frame_bytes)
    if remainder or frame_count < 1:
        raise AssertionError(f"Expected complete decoded frames, got {len(raw) / frame_bytes}")
    maximum_error = 0
    focal = HEIGHT * math.sqrt(3) / 2
    for frame in range(frame_count):
        pixels = raw[frame * frame_bytes:(frame + 1) * frame_bytes]
        for x, y, color in PATCHES:
            cx, cy = round(WIDTH / 2 + focal * x / 5), round(HEIGHT / 2 - focal * y / 5)
            for channel, target in enumerate(color):
                values = [pixels[(py * WIDTH + px) * 3 + channel]
                          for py in range(cy - 2, cy + 3) for px in range(cx - 2, cx + 3)]
                error = abs(sum(values) / len(values) - target * 255)
                maximum_error = max(maximum_error, error)
                if error > 25:
                    raise AssertionError(f"Frame {frame}, corner {(x, y)}, channel {channel}: "
                                         f"error {error:.2f}/255 (missing, flipped, or mirrored patch)")
    return {"frames": frame_count, "maximum_patch_error_u8": maximum_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--executable", type=Path)
    mode.add_argument("--mcp-url")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.ffmpeg:
        parser.error("Install FFmpeg or provide --ffmpeg /path/to/ffmpeg")
    root = (args.output or Path(tempfile.mkdtemp(prefix="lfs-video-orientation-"))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    mcp = Mcp(args.mcp_url) if args.mcp_url else None
    results = {}
    for pose in POSES:
        name = pose[0]
        ply, path = fixture(root, pose)
        video = root / f"{name}.mp4"
        try:
            if mcp:
                mcp.call("scene_load_ply", path=str(ply))
                mcp.wait("import.dataset")
                mcp.call("runtime_job_control", job_id="import.dataset", action="dismiss")
                mcp.call("render_settings_set", raster_backend="3dgs", environment_mode=0,
                         point_cloud_mode=False, background_color=[0, 0, 0], show_grid=False)
                timeline = mcp.call("sequencer_load_path", path=str(path))
                mcp.call("sequencer_go_to_keyframe", keyframe_id=timeline["keyframes"][0]["id"])
                mcp.call("editor_run", show_console=False, code=("import lichtfeld as lf\n" +
                         f"lf.ui.export_video({WIDTH},{HEIGHT},24,18,{str(video)!r})"))
                mcp.wait("export.video")
                capture = mcp.raw_call("render_capture_window")
                block = next(c for c in capture["content"] if c["type"] == "image")
                (root / f"{name}-window.png").write_bytes(base64.b64decode(block["data"]))
            else:
                with (root / f"{name}.log").open("w") as log:
                    subprocess.run([str(args.executable.resolve()), "--render-camera-path", str(path),
                                    "--render-load", str(ply), "--render-output", str(video),
                                    "--render-width", str(WIDTH), "--render-height", str(HEIGHT),
                                    "--render-fps", "24"], stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=120)
            subprocess.run([args.ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(video), "-frames:v", "1",
                            str(root / f"{name}-decoded.png")], check=True, timeout=30)
            raw = subprocess.check_output([args.ffmpeg, "-nostdin", "-v", "error", "-i", str(video),
                                           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], timeout=30)
            results[name] = {"passed": True, **verify_pixels(raw)}
        except (AssertionError, RuntimeError, subprocess.SubprocessError) as error:
            results[name] = {"passed": False, "error": str(error)}
        print(name, results[name], flush=True)
    (root / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Artifacts: {root}")
    return 0 if all(r["passed"] for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
