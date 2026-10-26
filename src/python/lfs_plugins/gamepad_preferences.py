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
        self._axis_records = None
        self._button_records = None
        self._pressed = []
        self._pressed_until = 0.0
        self.navigation = {}
        self.navigation_error = ""
        self._selection_intent = set()

    def bind(self, model):
        model.bind_record_list("gamepad_test_devices")
        model.bind_record_list("gamepad_test_axis_rows")
        model.bind_record_list("gamepad_test_button_rows")
        model.bind_func("gamepad_test_buttons_empty", lambda: not self.button_names())
        model.bind("gamepad_test_id", lambda: str(self.state.get("selected", 0)),
                   lambda value: self._select_from_input("test", value))
        model.bind_func("gamepad_test_status", self.status)
        model.bind_func("gamepad_test_axes", self.axes)
        model.bind_func("gamepad_test_buttons", self.buttons)
        model.bind("gamepad_navigation_enabled", lambda: self.navigation.get("enabled", False),
                   lambda value: self.configure(enabled=bool(value)))
        model.bind("gamepad_navigation_id", lambda: str(self.navigation.get("selected", 0)),
                   lambda value: self._select_from_input("navigation", value))
        for name, default in (("dead_zone", 0.15), ("look_speed", 1.5), ("move_speed", 1.0)):
            model.bind("gamepad_" + name, lambda n=name, d=default: self.navigation.get(n, d),
                       lambda value, n=name: self.configure(**{n: float(value)}))
        model.bind_func("gamepad_navigation_status", self.navigation_status)
        model.bind_func("gamepad_navigation_show_status", self.show_navigation_status)
        model.bind_func("gamepad_navigation_preset", lambda: lf.ui.tr("gamepad.preset." + self.navigation.get("mode", "orbit")))
        self.handle = model.get_handle()
        self.sync()

    def mount(self, document):
        self._selection_intent.clear()
        if not callable(getattr(document, "get_element_by_id", None)):
            return
        for channel in ("test", "navigation"):
            element = document.get_element_by_id("gamepad-" + channel + "-device")
            if element is not None:
                for event in ("mousedown", "keydown"):
                    element.add_event_listener(event, lambda _event, c=channel: self._selection_intent.add(c))
                element.add_event_listener("blur", lambda _event, c=channel: self._selection_intent.discard(c))

    def _select_from_input(self, channel, value):
        # RmlUI also writes select values while constructing/rebuilding options.
        # Only pointer/keyboard interaction may change the native device choice.
        if channel not in self._selection_intent:
            if self.handle is not None:
                self.handle.dirty("gamepad_test_id" if channel == "test" else "gamepad_navigation_id")
            return
        self._selection_intent.discard(channel)
        if channel == "test":
            self.select(value)
        else:
            try:
                identifier = int(value)
            except (TypeError, ValueError):
                return
            if 0 <= identifier <= 0xFFFFFFFF:
                self.configure(selected=identifier)

    def close(self):
        self._selection_intent.clear()
        self.deactivate()
        self.handle = None
        self._records = None
        self._axis_records = None
        self._button_records = None
        self._pressed = []

    def deactivate(self):
        self._selection_intent.clear()
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
            self.handle.dirty("gamepad_navigation_show_status")

    def navigation_status(self):
        if self.navigation_error or self.state.get("error"):
            return self.navigation_error or self.state["error"]
        if self.state and not self.state.get("available"):
            return lf.ui.tr("gamepad.unavailable")
        return lf.ui.tr("gamepad.navigation." + self.navigation.get("status", "disabled"))

    def show_navigation_status(self):
        return bool(self.navigation_error or self.state.get("error") or self.navigation.get("enabled") or
                    (self.state and not self.state.get("available")))

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
        axis_rows = self.axis_rows()
        if axis_rows != self._axis_records:
            self._axis_records = axis_rows
            self.handle.update_record_list("gamepad_test_axis_rows", axis_rows)
        button_rows = [{"label": name} for name in self.button_names()]
        if button_rows != self._button_records:
            self._button_records = button_rows
            self.handle.update_record_list("gamepad_test_button_rows", button_rows)
            self.handle.dirty("gamepad_test_buttons_empty")
        if changed:
            for name in ("gamepad_test_id", "gamepad_test_status", "gamepad_test_axes", "gamepad_test_buttons"):
                self.handle.dirty(name)
            for name in ("navigation_enabled", "navigation_id", "navigation_status", "navigation_show_status", "navigation_preset", "dead_zone", "look_speed", "move_speed"):
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

    def axis_rows(self):
        values = self.state.get("axes", ())
        present = self.state.get("has_axes", (True,) * 6)
        rows = []
        for index, label in enumerate(("LX", "LY", "RX", "RY", "LT", "RT")):
            available = bool(self.state.get("selected")) and index < len(values) and index < len(present) and present[index]
            value = max(-1.0, min(1.0, values[index])) if available else 0.0
            rows.append({"label": label, "value": f"{value:+.2f}" if available else "—",
                         "magnitude": abs(value), "available": available,
                         "group_start": index % 2 == 0,
                         "group": lf.ui.tr(("gamepad.left_stick", "gamepad.right_stick", "gamepad.triggers")[index // 2])})
        return rows

    def buttons(self):
        return ", ".join(self.button_names()) or "—"

    def button_names(self):
        return list(dict.fromkeys([*self.state.get("buttons", ()), *self._pressed]))
