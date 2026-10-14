# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Opt-in Windows integration checks against a dedicated, already running GUI.

Set LFS_NATIVE_WINDOWS_TEST_ENDPOINT and LFS_NATIVE_WINDOWS_TEST_PROFILE to an
isolated MCP instance and its exact LFS_HOME. The profile check prevents resets
against the user's ordinary settings. No dataset, plugin installation or export
is performed. Standard Python test runs skip this module.
"""

import ctypes
from ctypes import wintypes
import json
import base64
import struct
import zlib
import os
from pathlib import Path
import sys
import time
import urllib.request

import pytest


pytestmark = pytest.mark.skipif(
    sys.platform != "win32" or not os.environ.get("LFS_NATIVE_WINDOWS_TEST_ENDPOINT"),
    reason="requires an explicitly selected isolated Windows GUI instance",
)


class NativeWindowsClient:
    def __init__(self):
        self.endpoint = os.environ["LFS_NATIVE_WINDOWS_TEST_ENDPOINT"]
        self.profile = Path(os.environ["LFS_NATIVE_WINDOWS_TEST_PROFILE"]).resolve()
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        self.user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        self.user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        self.user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        self.user32.IsWindowVisible.argtypes = [wintypes.HWND]
        self.user32.IsWindow.argtypes = [wintypes.HWND]
        self.user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        self.user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        self.user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        self.user32.SendMessageW.restype = ctypes.c_ssize_t
        self.user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                                          ctypes.c_int, ctypes.c_int, wintypes.UINT]
        self.user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
        self.user32.SetForegroundWindow.argtypes = [wintypes.HWND]
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.original_cursor = wintypes.POINT()
        assert self.user32.GetCursorPos(ctypes.byref(self.original_cursor))
        self.rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                "clientInfo": {"name": "native-window-tests", "version": "1"}})
        for uri in ("lichtfeld://runtime/catalog", "lichtfeld://runtime/state", "lichtfeld://ui/state",
                    "lichtfeld://scene/state", "lichtfeld://selection/current"):
            self.rpc("resources/read", {"uri": uri})
        manifest = self.rpc("tools/list")["tools"]
        self.tools = {item["name"]: item for item in manifest}
        for name in ("editor_run", "ui_panel_describe", "ui_panel_update", "ui_modal_get", "ui_modal_press"):
            assert name in self.tools
        identity = self.python("import os\nprint('NATIVE_WINDOWS_TEST:' + __import__('json').dumps({'pid':os.getpid(),'home':os.environ.get('LFS_HOME','')}))")
        assert Path(identity["home"]).resolve() == self.profile, "refusing to modify a different profile"
        assert self.profile != (Path.home() / ".lichtfeld").resolve(), "refusing to reset the normal profile"
        self.pid = identity["pid"]
        modal = self.call("ui_modal_get")
        if modal.get("open") and modal.get("title") == "Set as Default Application":
            self.call("ui_modal_press", label="Not Now")

    def rpc(self, method, params=None):
        request = urllib.request.Request(self.endpoint, data=json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": method, "params": params,
        }).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
        assert "error" not in result, result
        return result["result"]

    def call(self, name, **arguments):
        if name == "ui_pointer" and name not in self.tools:
            self._used_native_pointer = True
            assert arguments.get("button", "left") == "left"
            # Master has no synthetic-pointer MCP tool. Use real mouse input:
            # posted messages do not update the physical held-button state used
            # by drag handoff. Coordinates target this isolated app's main HWND.
            main = next(window for window in self.windows()
                        if window["title"].startswith("LichtFeld Studio"))
            point = wintypes.POINT(int(arguments["x"]), int(arguments["y"]))
            assert self.user32.ClientToScreen(main["hwnd"], ctypes.byref(point))
            action = arguments["action"]
            class MouseInput(ctypes.Structure):
                _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                            ("mouseData", wintypes.DWORD), ("flags", wintypes.DWORD),
                            ("time", wintypes.DWORD), ("extra", ctypes.c_size_t)]
            class InputValue(ctypes.Union):
                _fields_ = [("mouse", MouseInput)]
            class Input(ctypes.Structure):
                _fields_ = [("type", wintypes.DWORD), ("value", InputValue)]
            self.user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(Input), ctypes.c_int]
            if action == "down":
                foreground = self.user32.GetForegroundWindow()
                assert foreground, "Master pointer tests require an unlocked interactive Windows desktop"
                foreground_pid = wintypes.DWORD()
                self.user32.GetWindowThreadProcessId(foreground, ctypes.byref(foreground_pid))
                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
                kernel.OpenProcess.restype = wintypes.HANDLE
                kernel.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                              wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
                kernel.CloseHandle.argtypes = [wintypes.HANDLE]
                process = kernel.OpenProcess(0x1000, False, foreground_pid.value)
                if process:
                    try:
                        executable = ctypes.create_unicode_buffer(1024)
                        length = wintypes.DWORD(len(executable))
                        if kernel.QueryFullProcessImageNameW(process, 0, executable, ctypes.byref(length)):
                            assert Path(executable.value).name.lower() not in {"lockapp.exe", "logonui.exe"}, \
                                "Master pointer tests require an unlocked Windows session; no mouse input was sent"
                    finally:
                        kernel.CloseHandle(process)
                self.user32.SetForegroundWindow(main["hwnd"])
                time.sleep(.1)
                if self.user32.GetForegroundWindow() != main["hwnd"]:
                    # Windows can deny a background process's activation request.
                    # Activate with a normal click in the main window's empty corner
                    # before issuing the title press used by the drag assertion.
                    corner = wintypes.POINT(4, 4)
                    assert self.user32.ClientToScreen(main["hwnd"], ctypes.byref(corner))
                    assert self.user32.SetCursorPos(corner.x, corner.y)
                    activation = (Input * 2)(Input(0, InputValue(mouse=MouseInput(flags=2))),
                                             Input(0, InputValue(mouse=MouseInput(flags=4))))
                    assert self.user32.SendInput(2, activation, ctypes.sizeof(Input)) == 2
                    self.until(lambda: self.user32.GetForegroundWindow() == main["hwnd"])
                    time.sleep(.1)
            assert self.user32.SetCursorPos(point.x, point.y)
            if action == "down":
                flags = 0x0002  # MOUSEEVENTF_LEFTDOWN
            elif action == "up":
                flags = 0x0004  # MOUSEEVENTF_LEFTUP
            else:
                assert action == "move", action
                return {"success": True}
            packet = Input(0, InputValue(mouse=MouseInput(flags=flags)))
            assert self.user32.SendInput(1, ctypes.byref(packet), ctypes.sizeof(Input)) == 1
            self._pointer_buttons = 1 if action == "down" else 0
            return {"success": True}
        result = self.rpc("tools/call", {"name": name, "arguments": arguments})
        assert not result.get("isError"), result
        if "structuredContent" in result:
            return result["structuredContent"]
        return json.loads(result["content"][0]["text"])

    def python(self, code):
        result = self.call("editor_run", code=code, show_console=False, timeout_ms=10000)
        assert result["completed"] and result["success"], result
        text = result["output"]["text"]
        marker = "NATIVE_WINDOWS_TEST:"
        assert marker in text, text
        # The integrated terminal inserts physical wraps into long output lines.
        payload = text[text.rfind(marker) + len(marker):].replace("\r", "").replace("\n", "")
        return json.JSONDecoder().raw_decode(payload)[0]

    def windows(self):
        windows = []
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def inspect(hwnd, _):
            pid = wintypes.DWORD()
            self.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value == self.pid and self.user32.IsWindowVisible(hwnd):
                title = ctypes.create_unicode_buffer(self.user32.GetWindowTextLengthW(hwnd) + 1)
                self.user32.GetWindowTextW(hwnd, title, len(title))
                rect = wintypes.RECT()
                self.user32.GetWindowRect(hwnd, ctypes.byref(rect))
                windows.append({"hwnd": hwnd, "title": title.value,
                                "rect": (rect.left, rect.top, rect.right, rect.bottom)})
            return True

        callback = callback_type(inspect)
        self.user32.EnumWindows(callback, 0)
        return windows

    def monitors(self):
        bounds = []
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC,
                                          ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)
        def inspect(_, __, rect, ___):
            value = rect.contents
            bounds.append((value.left, value.top, value.right, value.bottom))
            return True
        callback = callback_type(inspect)
        self.user32.EnumDisplayMonitors(None, None, callback, 0)
        return bounds

    def until(self, check):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            result = check()
            if result:
                return result
            time.sleep(0.05)
        raise AssertionError("native window state did not settle within ten seconds")

    def open(self, panel_id):
        panel = self.call("ui_panel_update", panel_id=panel_id, enabled=True)["panel"]
        if panel.get("desktop_window"):
            return self.until(lambda: next((window for window in self.windows() if window["title"] == panel["label"]), None))
        existing = next((window for window in self.windows() if window["title"] == panel["label"]), None)
        if existing:
            return existing
        # Reset actions can reopen embedded panels at overlapping defaults. An
        # enabled target is not necessarily on top; don't drag a different
        # panel's title or an underlying control using the target's bounds.
        # Keep existing desktop hosts open for the multi-window assertions.
        for other_id in ("lfs.preferences", "lfs.plugin_marketplace", "native.video_extractor"):
            if other_id == panel_id:
                continue
            other = self.call("ui_panel_describe", panel_id=other_id)["panel"]
            if other["enabled"] and not other.get("desktop_window"):
                self.call("ui_panel_update", panel_id=other_id, enabled=False)
        state = self.until(lambda: next((window for window in self.windows() if window["title"] == panel["label"]), None)
                           or self.call("ui_panel_describe", panel_id=panel_id)["panel"].get("floating_bounds"))
        if "hwnd" in state:
            return state
        main = next(window for window in self.windows() if window["title"].startswith("LichtFeld Studio"))
        rect = wintypes.RECT()
        assert self.user32.GetClientRect(main["hwnd"], ctypes.byref(rect))
        start_x, start_y = int(state["x"] + 80), int(state["y"] + 20)
        self.call("ui_pointer", action="down", button="left", x=start_x, y=start_y)
        time.sleep(.1)
        self.call("ui_pointer", action="move", x=rect.right + 100, y=start_y)
        try:
            self.until(lambda: any(window["title"] == panel["label"] for window in self.windows()))
        finally:
            self.call("ui_pointer", action="up", button="left", x=rect.right + 100, y=start_y)
        return self.until(lambda: next((window for window in self.windows()
                                      if window["title"] == panel["label"]), None))

    def panel_elements(self, panel_id, identifiers):
        return self.python(
            "import lichtfeld as lf,json\n"
            f"doc=lf.ui.rml.get_document({panel_id!r})\n"
            f"ids={identifiers!r}\n"
            "result={}\n"
            "for name in ids:\n"
            " e=doc.get_element_by_id(name)\n"
            " result[name]=[e.absolute_left,e.absolute_top,e.absolute_width,e.absolute_height]\n"
            "print('NATIVE_WINDOWS_TEST:'+json.dumps(result))"
        )


@pytest.fixture(scope="module")
def native_client():
    client = NativeWindowsClient()
    try:
        yield client
    finally:
        if getattr(client, "_pointer_buttons", 0):
            client.call("ui_pointer", action="up", button="left", x=0, y=0)
        if getattr(client, "_used_native_pointer", False):
            client.user32.SetCursorPos(client.original_cursor.x, client.original_cursor.y)


def test_opening_embedded_panels_does_not_create_desktop_hosts(native_client):
    assert len(native_client.windows()) == 1
    for panel_id in ("lfs.preferences", "lfs.plugin_marketplace", "native.video_extractor"):
        native_client.call("ui_panel_update", panel_id=panel_id, enabled=True)
        native_client.until(lambda: native_client.call("ui_panel_describe", panel_id=panel_id)["panel"].get("floating_bounds"))
        assert len(native_client.windows()) == 1
        native_client.call("ui_panel_update", panel_id=panel_id, enabled=False)


def test_preferences_and_plugins_detach_to_distinct_desktop_windows(native_client):
    preferences = native_client.open("lfs.preferences")
    plugins = native_client.open("lfs.plugin_marketplace")
    assert preferences["hwnd"] != plugins["hwnd"]
    assert len(native_client.windows()) >= 3


def test_preferences_preserves_rml_chrome_without_a_native_container(native_client):
    preferences = native_client.open("lfs.preferences")
    client = wintypes.RECT()
    assert native_client.user32.GetClientRect(preferences["hwnd"], ctypes.byref(client))
    outer_width = preferences["rect"][2] - preferences["rect"][0]
    outer_height = preferences["rect"][3] - preferences["rect"][1]
    # SDL retains Win32 resize style bits but removes all non-client chrome.
    assert outer_width == client.right and outer_height == client.bottom
    elements = native_client.panel_elements("lfs.preferences", ["window-frame", "title-bar", "close-btn"])
    left, top, width, height = elements["window-frame"]
    assert abs(left) <= 2 and abs(top) <= 2, "Rml frame is inset inside a visible container"
    assert abs(width - client.right) <= 2 and abs(height - client.bottom) <= 2
    assert elements["title-bar"][3] > 20, "original Rml title bar is hidden"
    assert elements["close-btn"][2] > 0, "original close control is hidden"


def test_original_rml_close_button_closes_only_its_panel(native_client):
    preferences = native_client.open("lfs.preferences")
    x, y, width, height = native_client.panel_elements("lfs.preferences", ["close-btn"])["close-btn"]
    coordinates = (int(y + height / 2) << 16) | int(x + width / 2)
    for message, buttons in ((0x0200, 0), (0x0201, 1), (0x0202, 0)):
        assert native_client.user32.PostMessageW(preferences["hwnd"], message, buttons, coordinates)
    native_client.until(lambda: not native_client.call("ui_panel_describe", panel_id="lfs.preferences")["panel"]["enabled"])
    native_client.until(lambda: not native_client.user32.IsWindow(preferences["hwnd"]))
    assert native_client.open("lfs.preferences")["title"] == "Preferences"


def test_title_drag_and_resize_use_the_original_rml_window(native_client):
    preferences = native_client.open("lfs.preferences")
    hwnd = preferences["hwnd"]
    left, top, right, bottom = preferences["rect"]
    assert native_client.user32.SetWindowPos(hwnd, 0, left, top, 800, 500, 0x0014)

    def resized():
        elements = native_client.panel_elements("lfs.preferences", ["window-frame", "title-bar"])
        return elements if abs(elements["window-frame"][2] - 800) <= 2 and abs(elements["window-frame"][3] - 500) <= 2 else None

    elements = native_client.until(resized)
    rect = wintypes.RECT()
    assert native_client.user32.GetWindowRect(hwnd, ctypes.byref(rect))
    title_y = int(elements["title-bar"][1] + elements["title-bar"][3] / 2)

    def hit(x, y):
        return native_client.user32.SendMessageW(hwnd, 0x0084, 0, ((y & 0xffff) << 16) | (x & 0xffff))

    assert hit(rect.left + 80, rect.top + title_y) == 2  # HTCAPTION: Rml title drag
    assert hit(rect.right - 2, rect.bottom - 2) == 17  # HTBOTTOMRIGHT: original resize edge


def test_native_close_preserves_application_and_reopens_the_same_panel(native_client):
    preferences = native_client.open("lfs.preferences")
    assert native_client.user32.PostMessageW(preferences["hwnd"], 0x0010, 0, 0)  # WM_CLOSE
    native_client.until(lambda: not native_client.call("ui_panel_describe", panel_id="lfs.preferences")["panel"]["enabled"])
    native_client.until(lambda: not native_client.user32.IsWindow(preferences["hwnd"]))
    reopened = native_client.open("lfs.preferences")
    assert reopened["title"] == preferences["title"]
    assert reopened["rect"][2] - reopened["rect"][0] == preferences["rect"][2] - preferences["rect"][0]
    assert reopened["rect"][3] - reopened["rect"][1] == preferences["rect"][3] - preferences["rect"][1]


def test_reopening_an_inaccessible_window_recovers_its_rml_title(native_client):
    preferences = native_client.open("lfs.preferences")
    assert native_client.user32.SetWindowPos(preferences["hwnd"], 0, -30000, -30000, 0, 0, 0x0015)
    native_client.call("ui_panel_update", panel_id="lfs.preferences", enabled=False)
    native_client.until(lambda: not native_client.user32.IsWindow(preferences["hwnd"]))
    reopened = native_client.open("lfs.preferences")
    def accessible():
        panel = next(window for window in native_client.windows() if window["hwnd"] == reopened["hwnd"])
        left, top, right, bottom = panel["rect"]
        return any(min(right, mr) - max(left, ml) >= 96 and min(top + 30, mb) - max(top, mt) >= 24
                   for ml, mt, mr, mb in native_client.monitors())
    native_client.until(accessible)
    assert reopened["title"] == preferences["title"]


def test_window_geometry_persists_and_reset_keeps_preferences_open(native_client):
    preferences = native_client.open("lfs.preferences")
    assert native_client.user32.SetWindowPos(preferences["hwnd"], 0, 40, 50, 760, 540, 0x0014)
    path = native_client.profile / "config/preferences.json"

    def moved():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (PermissionError, json.JSONDecodeError):
            return False  # The UI may be saving the geometry in this frame.
        rect = state.get("floating_window_geometry", {}).get("lfs.preferences", [])
        return len(rect) == 4 and rect[2] < 850

    native_client.until(moved)
    moved_rect = next(window["rect"] for window in native_client.windows()
                      if window["hwnd"] == preferences["hwnd"])
    result = native_client.python("import lichtfeld as lf, json\nerror=lf.ui.reset_window_state()\nprint('NATIVE_WINDOWS_TEST:'+json.dumps({'error':error,'native':lf.ui.get_native_floating_windows()}))")
    assert not result["error"]
    assert result["native"] is True
    reset = native_client.until(lambda: next((window for window in native_client.windows()
                                            if window["title"] == preferences["title"]
                                            and window["rect"] != moved_rect), None))
    assert reset["hwnd"] == preferences["hwnd"]
    assert native_client.call("ui_panel_describe", panel_id="lfs.preferences")["panel"]["enabled"]


def test_confirmation_stays_a_popup_and_preserves_callback(native_client):
    native_client.open("lfs.preferences")
    windows_before = {window["hwnd"] for window in native_client.windows()}
    result = native_client.python("import lichtfeld as lf, builtins, json\nbuiltins._native_window_confirmation=None\nlf.ui.confirm_dialog('Native windows smoke', 'Confirm callback test', ['Cancel','OK'], lambda button:setattr(builtins,'_native_window_confirmation',button))\nprint('NATIVE_WINDOWS_TEST:'+json.dumps({'queued':True}))")
    assert result["queued"]
    native_client.until(lambda: native_client.call("ui_modal_get").get("title") == "Native windows smoke")
    assert {window["hwnd"] for window in native_client.windows()} == windows_before
    native_client.call("ui_modal_press", label="OK")
    result = native_client.python("import builtins,json\nprint('NATIVE_WINDOWS_TEST:'+json.dumps({'button':builtins._native_window_confirmation}))")
    assert result["button"] == "OK"


def test_video_extractor_uses_a_separate_window(native_client):
    extractor = native_client.open("native.video_extractor")
    preferences = native_client.open("lfs.preferences")
    assert extractor["hwnd"] != preferences["hwnd"]


@pytest.mark.parametrize("panel_id", [None, "lfs.preferences", "native.video_extractor"])
def test_native_window_capture_contains_rendered_controls(native_client, panel_id):
    assert_rendered_capture(native_client, panel_id)


def test_native_window_capture_reports_unavailable_panel(native_client):
    response = native_client.rpc("tools/call", {
        "name": "render_capture_window", "arguments": {"panel_id": "test.nonexistent-panel"},
    })
    assert response["isError"]
    result = json.loads(response["content"][0]["text"])
    assert "unavailable" in result["error"]["message"]
    assert "success" not in result


def assert_rendered_capture(native_client, panel_id):
    if panel_id:
        native_client.open(panel_id)
    capture = native_client.call("render_capture_window", **({"panel_id": panel_id} if panel_id else {}))
    assert capture["success"]
    png = base64.b64decode(capture["data"])
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    offset = 8
    compressed = bytearray()
    while offset < len(png):
        length = struct.unpack_from(">I", png, offset)[0]
        if png[offset + 4:offset + 8] == b"IDAT":
            compressed.extend(png[offset + 8:offset + 8 + length])
        offset += length + 12
    assert len(set(zlib.decompress(compressed))) > 32, "native capture is an empty clear frame"
    filename = "preferences-rendered.png" if panel_id == "lfs.preferences" else f"capture-{panel_id or 'main'}.png"
    (native_client.profile.parent / filename).write_bytes(png)


def test_moving_geometry_is_buffered_then_flushed_at_rest_and_close(native_client):
    preferences = native_client.open("lfs.preferences")
    path = native_client.profile / "config/preferences.json"
    time.sleep(.5)
    before = path.stat().st_mtime_ns
    for step in range(30):
        assert native_client.user32.SetWindowPos(preferences["hwnd"], 0, 100 + step, 100, 0, 0, 0x0015)
        time.sleep(.02)
        assert path.stat().st_mtime_ns == before, "preferences were rewritten while the panel was moving"
    native_client.until(lambda: path.stat().st_mtime_ns != before)
    saved = json.loads(path.read_text(encoding="utf-8"))["floating_window_geometry"]["lfs.preferences"]
    assert saved[:2] == [129, 100]
    assert native_client.user32.SetWindowPos(preferences["hwnd"], 0, 160, 120, 0, 0, 0x0015)
    native_client.call("ui_panel_update", panel_id="lfs.preferences", enabled=False)
    native_client.until(lambda: not native_client.user32.IsWindow(preferences["hwnd"]))
    saved = json.loads(path.read_text(encoding="utf-8"))["floating_window_geometry"]["lfs.preferences"]
    assert saved[:2] == [160, 120], "close did not flush the final rectangle"
    reopened = native_client.open("lfs.preferences")
    assert reopened["rect"][:2] == (160, 120)


@pytest.mark.parametrize("panel_id", ["lfs.preferences", "lfs.plugin_marketplace", "native.video_extractor"])
def test_closing_external_panels_destroys_their_desktop_hosts(native_client, panel_id):
    window = native_client.open(panel_id)
    native_client.call("ui_panel_update", panel_id=panel_id, enabled=False)
    native_client.until(lambda: not native_client.user32.IsWindow(window["hwnd"]))
    assert not native_client.call("ui_panel_describe", panel_id=panel_id)["panel"]["desktop_window"]
    reopened = native_client.open(panel_id)
    assert reopened["title"] == window["title"]


def test_repeated_close_reopen_keeps_rendering_valid(native_client):
    for cycle in range(5):
        window = native_client.open("lfs.preferences")
        assert_rendered_capture(native_client, "lfs.preferences")
        native_client.call("ui_panel_update", panel_id="lfs.preferences", enabled=False)
        native_client.until(lambda: not native_client.user32.IsWindow(window["hwnd"]))


@pytest.mark.parametrize("panel_id", ["lfs.preferences", "lfs.plugin_marketplace", "native.video_extractor"])
@pytest.mark.parametrize("reset_action", ["reset_window_state", "reset_layout"])
def test_reset_discards_closed_external_panel_reopen_position(native_client, panel_id, reset_action):
    native = native_client.open(panel_id)
    native_client.call("ui_panel_update", panel_id=panel_id, enabled=False)
    native_client.until(lambda: not native_client.user32.IsWindow(native["hwnd"]))
    result = native_client.python(f"import lichtfeld as lf,json\nerror=lf.ui.{reset_action}()\nprint('NATIVE_WINDOWS_TEST:'+json.dumps({{'error':error}}))")
    assert not result["error"]
    native_client.call("ui_panel_update", panel_id=panel_id, enabled=True)
    native_client.until(lambda: native_client.call("ui_panel_describe", panel_id=panel_id)["panel"].get("floating_bounds"))
    time.sleep(.2)
    assert not any(window["title"] == native["title"] for window in native_client.windows())
