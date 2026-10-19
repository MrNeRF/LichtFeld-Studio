"""Real-GUI idle contract test, driven by a public-fixture MCP manifest.

The manifest supplies only generic scene setup tool calls and viewport coordinates;
fixture paths are kept in the GPU runner configuration, not in the repository.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.integration]
ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MODES = {
    "empty",
    "colmap_pointcloud",
    "ply_pointcloud",
    "splat",
    "mesh",
    "dataset_cameras",
    "depth_view",
    "gt_compare",
    "selection_crop",
    "python_console",
    "sequencer",
}


def _gpu_utilization() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return max(int(line.strip()) for line in result.stdout.splitlines() if line.strip())


def _post(endpoint: str, payload: dict) -> dict:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def _call(endpoint: str, method: str, params: dict | None = None) -> dict:
    response = _post(endpoint, {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}})
    assert "error" not in response, response
    return response.get("result", {})


def _initialize(endpoint: str) -> None:
    _call(
        endpoint,
        "initialize",
        {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "render-on-demand-idle-test", "version": "1"},
        },
    )
    _post(endpoint, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})


def _tool(endpoint: str, name: str, arguments: dict | None = None) -> dict:
    result = _call(endpoint, "tools/call", {"name": name, "arguments": arguments or {}})
    if result.get("isError"):
        raise AssertionError(result)
    return result["structuredContent"]

def _ledger(endpoint: str, reset: bool = False) -> dict:
    payload = _tool(endpoint, "runtime_frame_ledger", {"reset": reset})
    return payload.get("frames", payload)


def _save_screenshot(mode: str, directory: str | None, fallback: Path) -> None:
    try:
        from PIL import ImageGrab
    except ImportError:
        return
    output = Path(directory) if directory else fallback
    output.mkdir(parents=True, exist_ok=True)
    ImageGrab.grab(xdisplay=os.environ.get("DISPLAY", ":92")).save(output / f"{mode}.png")


def test_every_render_mode_is_quiet_when_idle(tmp_path: Path) -> None:
    manifest_path = os.environ.get("LFS_RENDER_IDLE_MANIFEST")
    executable = os.environ.get("LFS_EXECUTABLE")
    if not manifest_path or not executable:
        pytest.skip("GPU idle test requires LFS_RENDER_IDLE_MANIFEST and LFS_EXECUTABLE")

    manifest = json.loads(Path(manifest_path).read_text())
    modes = manifest.get("modes", [])
    names = {mode.get("name") for mode in modes}
    assert names == EXPECTED_MODES, f"manifest mode set differs: missing={EXPECTED_MODES - names}, extra={names - EXPECTED_MODES}"

    endpoint_port = int(os.environ.get("LFS_MCP_PORT", "45692"))
    endpoint = f"http://127.0.0.1:{endpoint_port}/mcp"
    idle_seconds = float(os.environ.get("LFS_RENDER_IDLE_SECONDS", "60"))
    home = tmp_path / "home"
    home.mkdir()
    app_env = os.environ.copy()
    app_env.update({"HOME": str(home), "DISPLAY": os.environ.get("DISPLAY", ":92")})

    floor_samples = []
    for _ in range(10):
        floor_samples.append(_gpu_utilization())
        time.sleep(1)
    floor = max(floor_samples)
    app_log = (tmp_path / "app.log").open("w")
    app = subprocess.Popen(
        [executable, "--no-splash", "--mcp-port", str(endpoint_port)],
        env=app_env,
        stdout=app_log,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if app.poll() is not None:
                raise AssertionError(f"app exited during startup; see {tmp_path / 'app.log'}")
            try:
                _initialize(endpoint)
                break
            except (OSError, AssertionError):
                time.sleep(0.5)
        else:
            raise AssertionError("MCP endpoint did not become ready")

        tools = _call(endpoint, "tools/list").get("tools", [])
        tool_names = {item.get("name") for item in tools}
        required = {"runtime_frame_ledger"}
        for mode in modes:
            for operation in mode.get("setup", []):
                required.add(operation["tool"])
        assert required <= tool_names, f"manifest references undiscovered tools: {required - tool_names}"

        for mode in modes:
            for operation in mode.get("setup", []):
                _tool(endpoint, operation["tool"], operation.get("arguments", {}))
            settle_deadline = time.monotonic() + 20
            last_count = _ledger(endpoint).get("frames_presented", 0)
            stable_since = time.monotonic()
            while time.monotonic() < settle_deadline and time.monotonic() - stable_since < 2:
                time.sleep(0.2)
                current = _ledger(endpoint).get("frames_presented", 0)
                if current != last_count:
                    last_count, stable_since = current, time.monotonic()
            _ledger(endpoint, reset=True)
            utilization = []
            end = time.monotonic() + idle_seconds
            while time.monotonic() < end:
                time.sleep(1)
                utilization.append(_gpu_utilization())
            after = _ledger(endpoint)
            assert after["ui_fps"] == 0, mode["name"]
            assert after["viewport_fps"] == 0, mode["name"]
            assert after.get("views_rendered", 0) == 0, mode["name"]
            assert after.get("frames_presented", 0) == 0, mode["name"]
            assert after.get("frames_without_reason", 0) == 0, mode["name"]
            assert after.get("stale_detections", 0) == 0, mode["name"]
            assert after.get("wakes_without_frame", 0) <= 1, mode["name"]
            assert after.get("live_holders", []) == [], mode["name"]
            assert max(utilization, default=floor) <= floor + 3, mode["name"]

            _save_screenshot(mode["name"], os.environ.get("LFS_RENDER_IDLE_SHOTS"), tmp_path / "shots")
            navigation = mode.get("navigation", manifest.get("navigation"))
            assert navigation, f"manifest needs navigation coordinates for {mode['name']}"
            _ledger(endpoint, reset=True)
            navigation_started = time.monotonic()
            subprocess.run(
                ["xdotool", "mousemove", str(navigation["x0"]), str(navigation["y0"]),
                 "mousedown", "1", "mousemove", str(navigation["x1"]), str(navigation["y1"]),
                 "mouseup", "1"],
                check=True,
            )
            navigation_latency_limit = float(os.environ.get("LFS_NAVIGATION_LATENCY_S", "0.5"))
            navigation_deadline = navigation_started + navigation_latency_limit
            navigated = _ledger(endpoint)
            while navigated.get("views_rendered", 0) == 0 and time.monotonic() < navigation_deadline:
                time.sleep(0.05)
                navigated = _ledger(endpoint)
            assert 1 <= navigated.get("views_rendered", 0) <= 10, mode["name"]
            assert time.monotonic() - navigation_started <= navigation_latency_limit
            assert navigated.get("frames_presented", 0) <= navigated.get("views_rendered", 0) + 1
            deadline = time.monotonic() + 5
            quiet = _ledger(endpoint)
            while quiet["ui_fps"] != 0 or quiet["viewport_fps"] != 0:
                assert time.monotonic() < deadline, quiet
                time.sleep(0.1)
                quiet = _ledger(endpoint)
            assert quiet["views_rendered"] == navigated["views_rendered"]
            assert 0 <= quiet["frames_presented"] - navigated["frames_presented"] <= 1
            if quiet["frames_presented"] != navigated["frames_presented"]:
                assert quiet["last_frame_reasons"] == ["FpsIdle"]
    finally:
        if app.poll() is None:
            # Stop only the process launched above, after confirming its executable
            # and display from /proc. No process-name matching is used.
            proc = Path(f"/proc/{app.pid}")
            if (proc / "exe").resolve() == Path(executable).resolve():
                environment = (proc / "environ").read_bytes().split(b"\0")
                assert f"DISPLAY={app_env['DISPLAY']}".encode() in environment
                os.kill(app.pid, signal.SIGTERM)
                try:
                    app.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.kill(app.pid, signal.SIGKILL)
                    app.wait(timeout=5)
        app_log.close()
