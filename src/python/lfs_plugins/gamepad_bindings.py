# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Editing SDL semantic bindings; native validation remains authoritative."""
import copy
import json
import time
import lichtfeld as lf

EMPTY = {"positive": "none", "negative": "none", "inverted": False}


def effective(profile, mode, action):
    bindings = profile["bindings"]
    return bindings.get("modes", {}).get(mode, {}).get(action, bindings["common"][action])


def conflicts(profile, catalog):
    result = []
    for mode in catalog["modes"]:
        used = {}
        for action in catalog["actions"]:
            if mode not in action["modes"]:
                continue
            binding = effective(profile, mode, action["id"])
            for control in (binding["positive"], binding["negative"]):
                if control == "none":
                    continue
                if control in used:
                    result.append((mode, used[control], action["id"]))
                used[control] = action["id"]
    return result


class GamepadBindingsSection:
    def __init__(self, owner, model):
        self.owner = owner
        self.handle = model.get_handle()
        self.scope = "common"
        self.capture = None
        self.message = ""
        self._rows = None
        self.expanded_action = None
        self._scope_records = None
        self.catalog = json.loads(lf.ui.get_gamepad_binding_catalog_json())
        model.bind_record_list("gamepad_binding_rows")
        model.bind_record_list("gamepad_binding_scopes")
        model.bind("gamepad_binding_scope", lambda: self.scope, self.set_scope)
        model.bind_func("gamepad_binding_message", lambda: self.message)
        model.bind_func("gamepad_binding_capturing", lambda: self.capture is not None)
        model.bind_func("gamepad_binding_can_skip", lambda: bool(self.capture and self.capture["phase"] == "negative"))
        model.bind_event("gamepad_binding_edit", lambda _h, _e, args: self.edit(*map(str, args)))
        model.bind_event("gamepad_binding_cancel", lambda *_: self.cancel())
        model.bind_event("gamepad_binding_skip", lambda *_: self.finish_capture())
        model.bind_event("gamepad_binding_reset", lambda *_: self.reset())


    @staticmethod
    def tr(key):
        return lf.ui.tr(f"gamepad.binding.{key}")

    def profile(self):
        return json.loads(lf.ui.get_gamepad_profile_json())

    def set_scope(self, value):
        if value != self.scope and value in ("common", *self.catalog["modes"]):
            self.cancel()
            self.expanded_action = None
            self.scope = value
            self.sync()

    def cancel(self):
        if self.capture:
            lf.ui.set_gamepad_test_device(0)
        self.capture = None
        self.message = ""
        self.handle.dirty("gamepad_binding_capturing")
        self.handle.dirty("gamepad_binding_can_skip")
        self.handle.dirty("gamepad_binding_message")

    def save(self, profile):
        try:
            if not lf.ui.set_gamepad_profile_json(json.dumps(profile)):
                raise RuntimeError("Unavailable viewer")
            self.message = ""
        except (ValueError, RuntimeError):
            self.message = self.tr("save_error")
        self._rows = None
        self.sync()

    def change(self, action, binding):
        profile = self.profile()
        target = profile["bindings"]["common"] if self.scope == "common" else profile["bindings"].setdefault("modes", {}).setdefault(self.scope, {})
        if binding is None:
            target.pop(action, None)
        else:
            target[action] = binding
        clashes = conflicts(profile, self.catalog)
        if not clashes:
            self.save(profile)
            return
        label = self.tr("replace")
        detail = "\n".join(self.tr(mode) + ": " + self.tr(a if b == action else b) for mode, a, b in clashes)
        def answer(button):
            if button != label:
                return
            for mode, a, b in clashes:
                other = a if b == action else b
                profile["bindings"].setdefault("modes", {}).setdefault(mode, {})[other] = dict(EMPTY)
            self.save(profile)
        lf.ui.confirm_dialog(self.tr("conflict"), detail, [lf.ui.tr("common.cancel"), label], answer)

    def edit(self, action, operation):
        if action not in [a["id"] for a in self.catalog["actions"]]:
            return
        self.cancel()
        if operation == "options":
            self.expanded_action = None if self.expanded_action == action else action
            self.sync()
        elif operation == "clear":
            self.change(action, dict(EMPTY))
        elif operation == "inherit" and self.scope != "common":
            self.change(action, None)
        elif operation == "invert":
            binding = copy.deepcopy(self.profile()["bindings"]["common"][action] if self.scope == "common" else effective(self.profile(), self.scope, action))
            binding["inverted"] = not binding["inverted"]
            self.change(action, binding)
        elif operation == "assign":
            identifier = self.owner.navigation.get("selected", 0)
            if not identifier or not lf.ui.set_gamepad_test_device(identifier):
                self.message = self.tr("select_device")
            else:
                profile = self.profile()
                current = profile["bindings"]["common"][action] if self.scope == "common" else effective(profile, self.scope, action)
                self.capture = {"action": action, "id": identifier, "generation": None, "neutral": None,
                                "binding": dict(EMPTY), "phase": "positive", "axis_inverted": current["inverted"]}
                self.message = self.tr("release")
            self.sync()

    def finish_capture(self):
        if not self.capture or self.capture["phase"] != "negative":
            return
        action, binding = self.capture["action"], self.capture["binding"]
        self.cancel()
        self.change(action, binding)

    def sample(self):
        capture = self.capture
        if not capture:
            return
        state = lf.ui.get_gamepad_test_state()
        if state.get("selected") != capture["id"] or (capture["generation"] is not None and state.get("generation") != capture["generation"]):
            self.cancel()
            return
        capture["generation"] = state.get("generation")
        axes = state.get("axes", [])
        buttons = state.get("raw_buttons", [])
        pressed = state.get("raw_pressed", [False] * len(buttons))
        neutral = all(abs(x) < .2 for x in axes[:4]) and all(x < .2 for x in axes[4:]) and not any(buttons) and not any(pressed)
        now = time.monotonic()
        if capture["neutral"] is None:
            if neutral:
                capture["neutral"] = now
            self.message = self.tr("release")
        elif now - capture["neutral"] < .12:
            if not neutral:
                capture["neutral"] = None
        else:
            action = next(a for a in self.catalog["actions"] if a["id"] == capture["action"])
            self.message = self.tr("capture_" + (capture["phase"] if action["continuous"] else "button"))
            controls = [c for c in self.catalog["controls"] if c["id"] != "none"]
            for control in controls:
                if control["axis"]:
                    index = control["index"]
                    if index >= len(axes) or abs(axes[index]) < .65:
                        continue
                    if control["signed"] and (not action["continuous"] or capture["phase"] == "negative"):
                        continue
                elif control["index"] >= len(buttons) or not (buttons[control["index"]] or (control["index"] < len(pressed) and pressed[control["index"]])):
                    continue
                if control["id"] == capture["binding"]["positive"]:
                    continue
                capture["binding"][capture["phase"]] = control["id"]
                # Changing a stick assignment must preserve the user's inversion option.
                if control["signed"]:
                    capture["binding"]["inverted"] = capture["axis_inverted"]
                if capture["phase"] == "positive" and action["continuous"] and not control["signed"]:
                    capture["phase"] = "negative"
                    capture["neutral"] = None
                else:
                    captured_action, binding = capture["action"], capture["binding"]
                    self.cancel()
                    self.change(captured_action, binding)
                break
        lf.ui.request_redraw(delay=.12)

    def reset(self):
        scope = self.scope
        label = self.tr("reset")
        def answer(button):
            if button != label:
                return
            self.cancel()
            profile = self.profile()
            defaults = json.loads(lf.ui.get_gamepad_default_profile_json())
            if scope == "common":
                profile["bindings"] = defaults["bindings"]
            else:
                profile["bindings"].setdefault("modes", {})[scope] = {
                    a["id"]: copy.deepcopy(effective(defaults, scope, a["id"]))
                    for a in self.catalog["actions"] if scope in a["modes"]}
            self.save(profile)
        lf.ui.confirm_dialog(label, self.tr("reset_all" if scope == "common" else "reset_mode"),
                             [lf.ui.tr("common.cancel"), label], answer)

    def sync(self):
        scopes = [{"id": scope, "label": self.tr(scope)} for scope in ["common", *self.catalog["modes"]]]
        if scopes != self._scope_records:
            self._scope_records = scopes
            self.handle.update_record_list("gamepad_binding_scopes", scopes)
        self.sample()
        profile = self.profile()
        rows = []
        for action in self.catalog["actions"]:
            name = action["id"]
            if self.scope == "common" and len(action["modes"]) == 1:
                continue
            if self.scope != "common" and self.scope not in action["modes"]:
                continue
            binding = profile["bindings"]["common"][name] if self.scope == "common" else effective(profile, self.scope, name)
            inherited = self.scope != "common" and name not in profile["bindings"].get("modes", {}).get(self.scope, {})
            controls = [self.tr(binding[key]) for key in ("positive", "negative") if binding[key] != "none"]
            rows.append({"id": name, "label": self.tr(name), "value": " − ".join(controls) or self.tr("none"),
                         "inherited": inherited, "mode": self.scope != "common", "continuous": action["continuous"] and bool(controls),
                         "inverted": binding["inverted"], "expanded": self.expanded_action == name,
                         "overridden": self.scope != "common" and not inherited,
                         "assigned": bool(controls)})
        if rows != self._rows:
            self._rows = rows
            self.handle.update_record_list("gamepad_binding_rows", rows)
        for key in ("scope", "message", "capturing", "can_skip"):
            self.handle.dirty("gamepad_binding_" + key)
