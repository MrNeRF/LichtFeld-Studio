# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Native event/idle regression. Uses the same isolated GPU runner as idle E2E."""

import json
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest

from test_render_on_demand_idle import _call, _initialize, _ledger, _tool

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def test_fps_resize_scale_and_background_updates(tmp_path):
    executable = os.environ.get("LFS_EXECUTABLE")
    if not executable:
        pytest.skip("requires LFS_EXECUTABLE and an isolated X display")
    image_grab = pytest.importorskip("PIL.ImageGrab")
    endpoint = f"http://127.0.0.1:{int(os.environ.get('LFS_MCP_PORT', '45696'))}/mcp"
    display = os.environ.get("DISPLAY", ":97")
    port = endpoint.split(":")[-1].split("/")[0]
    home = tmp_path / "home"
    home.mkdir()
    env = dict(os.environ, HOME=str(home), DISPLAY=display)
    fixture = tmp_path / "points.ply"
    fixture.write_text(
        "ply\nformat ascii 1.0\nelement vertex 400\nproperty float x\n"
        "property float y\nproperty float z\nproperty uchar red\n"
        "property uchar green\nproperty uchar blue\nend_header\n"
        + "".join(f"{x / 10} {y / 10} 0 80 180 220\n" for x in range(-10, 10) for y in range(-10, 10))
    )
    stamps = []

    def capture(name):
        stamps.append({"name": name, "time_ns": time.time_ns(), "frames": _ledger(endpoint)})
        image_grab.grab(xdisplay=display).save(tmp_path / f"{name}.png")
        (tmp_path / "timestamps.json").write_text(json.dumps(stamps, indent=2))

    def quiet():
        deadline = time.monotonic() + 15
        previous = _ledger(endpoint)
        stable_since = time.monotonic()
        while time.monotonic() < deadline:
            time.sleep(0.15)
            current = _ledger(endpoint)
            if current["frames_presented"] != previous["frames_presented"]:
                stable_since = time.monotonic()
            if current["ui_fps"] == current["viewport_fps"] == 0 and time.monotonic() - stable_since > 1.2:
                return current
            previous = current
        pytest.fail(f"did not become idle: {previous}")

    def editor(code):
        result = _tool(endpoint, "editor_run", {"code": code, "show_console": False})
        assert result["completed"], result

    def xd(*args):
        subprocess.run(["xdotool", *map(str, args)], env=env, check=True)

    with (tmp_path / "app.log").open("w") as log:
        app = subprocess.Popen([executable, "--no-splash", "--mcp-port", port], env=env, stdout=log, stderr=subprocess.STDOUT)
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
            names = {t["name"] for t in _call(endpoint, "tools/list")["tools"]}
            assert {"runtime_frame_ledger", "scene_load_ply", "editor_run"} <= names
            _tool(endpoint, "scene_load_ply", {"path": str(fixture)})
            deadline = time.monotonic() + 30
            while _ledger(endpoint)["views_rendered"] == 0:
                assert time.monotonic() < deadline, "scene never rendered"
                time.sleep(0.1)
            quiet()
            window = subprocess.check_output(["xdotool", "search", "--pid", str(app.pid)], env=env, text=True).splitlines()[0]
            xd("windowmove", window, 0, 0)
            xd("windowsize", window, 1600, 1000)
            xd("windowfocus", window)
            xd("mousemove", 0, 1300)
            quiet()
            # A Python callback must reveal the HUD without any unrelated input.
            before = _ledger(endpoint)
            editor("import lichtfeld as lf\nlf.ui.toggle_vram_hud()")
            after = quiet()
            assert after["frames_presented"] > before["frames_presented"]
            assert after["views_rendered"] == before["views_rendered"]
            capture("hud-idle")
            time.sleep(1.2)
            assert _ledger(endpoint)["frames_presented"] == after["frames_presented"]
            editor("lf.ui.toggle_vram_hud()")
            quiet()
            # Hover samples UI presents only; no camera or scene input changes.
            for i in range(40):
                xd("mousemove", 20 + i % 30, 120)
                time.sleep(0.015)
            active = _ledger(endpoint)
            assert active["ui_fps"] > 0 and active["viewport_fps"] == 0
            capture("ui-only")
            stopped = quiet()
            time.sleep(1.2)
            assert _ledger(endpoint)["frames_presented"] == stopped["frames_presented"]
            assert stopped["last_frame_reasons"] == ["FpsIdle"]
            capture("idle")
            editor("lf.ui.message_dialog('Redraw check', 'Dismiss without mouse input')")
            quiet()
            before = _ledger(endpoint)
            _tool(endpoint, "ui_modal_press", {"label": "OK"})
            after = quiet()
            assert after["frames_presented"] > before["frames_presented"]
            assert after["views_rendered"] == before["views_rendered"]
            capture("modal-dismissed")
            editor("lf.ui.input_dialog('Caret check', 'Input', 'text')")
            time.sleep(0.2)
            before = _ledger(endpoint)
            time.sleep(1.5)
            after = _ledger(endpoint)
            assert after["frames_presented"] > before["frames_presented"]
            assert after["views_rendered"] == before["views_rendered"]
            _tool(endpoint, "ui_modal_press", {"label": "OK"})
            quiet()
            # A focused text caret has a real Rml deadline; blur removes it.
            xd("mousemove", 80, 48)
            xd("click", 1)
            xd("type", "caret")
            time.sleep(0.2)
            before = _ledger(endpoint)
            time.sleep(1.5)
            after = _ledger(endpoint)
            assert after["frames_presented"] > before["frames_presented"]
            assert after["views_rendered"] == before["views_rendered"]
            xd("key", "ctrl+a", "BackSpace", "Tab")
            stopped = quiet()
            time.sleep(1.2)
            assert _ledger(endpoint)["frames_presented"] == stopped["frames_presented"]
            for scale in (2.0, 1.5, 1.0):
                before = _ledger(endpoint)
                editor(f"lf.ui.set_ui_scale({scale})")
                after = quiet()
                assert after["views_rendered"] > before["views_rendered"]
                assert after["stale_detections"] == 0
                capture(f"scale-{scale}")
            for width, height in ((800, 500), (1800, 1100), (640, 400), (1600, 1000)):
                before = _ledger(endpoint)
                xd("windowsize", window, width, height)
                after = quiet()
                assert after["views_rendered"] > before["views_rendered"]
                assert after["stale_detections"] == 0
                capture(f"resize-{width}")
        finally:
            if app.poll() is None:
                proc = Path(f"/proc/{app.pid}")
                assert (proc / "exe").resolve() == Path(executable).resolve()
                assert f"DISPLAY={display}".encode() in (proc / "environ").read_bytes().split(b"\0")
                os.kill(app.pid, signal.SIGTERM)
                app.wait(timeout=15)
