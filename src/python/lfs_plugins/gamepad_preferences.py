# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Controller diagnostics, deliberately independent of camera ownership."""

import time
import lichtfeld as lf


class GamepadTestSection:
    def __init__(self):
        self.state = {}
        self.handle = None
        self._records = None
        self._pressed = []
        self._pressed_until = 0.0
        self.navigation = {}
        self.navigation_error = ""

    def bind(self, model):
        model.bind_record_list("gamepad_test_devices")
        model.bind("gamepad_test_id", lambda: str(self.state.get("selected", 0)), self.select)
        model.bind_func("gamepad_test_status", self.status)
        model.bind_func("gamepad_test_axes", self.axes)
        model.bind_func("gamepad_test_buttons", self.buttons)
        model.bind("gamepad_navigation_enabled", lambda: self.navigation.get("enabled", False),
                   lambda value: self.configure(enabled=bool(value)))
        model.bind("gamepad_navigation_id", lambda: str(self.navigation.get("selected", 0)),
                   lambda value: self.configure(selected=int(value)))
        for name, default in (("dead_zone", 0.15), ("look_speed", 1.5), ("move_speed", 1.0)):
            model.bind("gamepad_" + name, lambda n=name, d=default: self.navigation.get(n, d),
                       lambda value, n=name: self.configure(**{n: float(value)}))
        model.bind_func("gamepad_navigation_status", self.navigation_status)
        model.bind_func("gamepad_navigation_preset", lambda: lf.ui.tr("gamepad.preset." + self.navigation.get("mode", "orbit")))
        self.handle = model.get_handle()
        self.sync()

    def close(self):
        self.deactivate()
        self.handle = None
        self._records = None
        self._pressed = []

    def deactivate(self):
        if self.state.get("selected"):
            select = getattr(lf.ui, "set_gamepad_test_device", None)
            if select:
                select(0)
            self.state = {}
            self._pressed = []

    def select(self, value):
        try:
            device_id = int(value)
        except (TypeError, ValueError):
            return
        # IDs are connection-specific. Unknown/disconnected IDs are rejected natively.
        if 0 <= device_id <= 0xFFFFFFFF:
            select = getattr(lf.ui, "set_gamepad_test_device", None)
            if select:
                select(device_id)
        self._pressed = []
        self.sync()

    def configure(self, **changes):
        settings = {"enabled": False, "selected": 0, "dead_zone": 0.15, "look_speed": 1.5, "move_speed": 1.0}
        settings.update({key: self.navigation[key] for key in settings if key in self.navigation})
        settings.update(changes)
        setter = getattr(lf.ui, "set_gamepad_navigation", None)
        self.navigation_error = ""
        try:
            if setter is None or not setter(settings["enabled"], settings["selected"], settings["dead_zone"],
                                             settings["look_speed"], settings["move_speed"]):
                self.navigation_error = lf.ui.tr("gamepad.unavailable")
        except (ValueError, RuntimeError) as error:
            self.navigation_error = str(error)
        self.sync()
        if self.handle is not None:
            self.handle.dirty("gamepad_navigation_status")

    def navigation_status(self):
        return self.navigation_error or lf.ui.tr("gamepad.navigation." + self.navigation.get("status", "disabled"))

    def sync(self):
        if self.handle is None:
            return
        state = getattr(lf.ui, "get_gamepad_test_state", lambda: {})()
        changed = state != self.state
        navigation = getattr(lf.ui, "get_gamepad_navigation_state", lambda: {})()
        changed |= navigation != self.navigation
        self.navigation = navigation
        if (state.get("selected"), state.get("generation")) != (
            self.state.get("selected"), self.state.get("generation")
        ):
            self._pressed = []
        if state.get("pressed"):
            self._pressed = list(state["pressed"])
            self._pressed_until = time.monotonic() + 0.25
        elif self._pressed and time.monotonic() >= self._pressed_until:
            self._pressed = []
            changed = True
        if self._pressed:
            redraw = getattr(lf.ui, "request_redraw", None)
            if redraw:
                redraw(delay=max(0.0, self._pressed_until - time.monotonic()))
        self.state = state
        records = [{"id": "0", "label": lf.ui.tr("gamepad.none")}]
        for device in state.get("devices", ()):
            label = f"{device['name']} · #{device['id']}"
            if not device.get("mapped"):
                label += " — " + lf.ui.tr("gamepad.unmapped")
            if device.get("error"):
                label += " — " + device["error"]
            records.append({"id": str(device["id"]), "label": label})
        if records != self._records:
            self._records = records
            self.handle.update_record_list("gamepad_test_devices", records)
            changed = True
        if changed:
            for name in ("gamepad_test_id", "gamepad_test_status", "gamepad_test_axes", "gamepad_test_buttons"):
                self.handle.dirty(name)
            for name in ("navigation_enabled", "navigation_id", "navigation_status", "navigation_preset", "dead_zone", "look_speed", "move_speed"):
                self.handle.dirty("gamepad_" + name)

    def status(self):
        if not self.state.get("available"):
            return self.state.get("error") or lf.ui.tr("gamepad.unavailable")
        if not self.state.get("devices"):
            return lf.ui.tr("gamepad.empty")
        return ""

    def axes(self):
        if not self.state.get("selected"):
            return "—"
        axes = self.state.get("axes", ())
        present = self.state.get("has_axes", (True,) * 6)
        return "   ".join(
            f"{name} {value:+.2f}" if available else f"{name} —"
            for name, value, available in zip(("LX", "LY", "RX", "RY", "LT", "RT"), axes, present)
        )

    def buttons(self):
        return ", ".join(dict.fromkeys([*self.state.get("buttons", ()), *self._pressed])) or "—"
