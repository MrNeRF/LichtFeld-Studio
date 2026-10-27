# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Named controller layouts; native validation and atomic persistence are authoritative."""
import copy
import json
import uuid

import lichtfeld as lf


FORMAT = "lichtfeld.gamepad-profile"
MAX_IMPORT_BYTES = 256 * 1024


def profile_export(profile):
    settings = copy.deepcopy(profile["settings"])
    settings.pop("enabled", None)
    return {"format": FORMAT, "schema_version": 1, "name": profile["name"], "settings": settings}


def profile_import(value):
    if not isinstance(value, dict):
        raise ValueError("Expected a controller profile")
    if value.get("format") == FORMAT:
        if set(value) != {"format", "schema_version", "name", "settings"} or type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("Unsupported controller profile format")
        if not isinstance(value["name"], str) or not isinstance(value["settings"], dict) or "enabled" in value["settings"]:
            raise ValueError("Invalid controller profile")
        return copy.deepcopy(value)
    # Previously exported flat controller settings are accepted without their
    # global enablement. Keyboard profiles and whole libraries are not imports.
    if not set(value) <= {"schema_version", "enabled", "dead_zone", "look_speed", "move_speed", "bindings"} or not isinstance(value.get("bindings"), dict):
        raise ValueError("Not a controller profile")
    settings = copy.deepcopy(value)
    settings.pop("enabled", None)
    return {"format": FORMAT, "schema_version": 1, "name": "", "settings": settings}


class GamepadProfilesSection:
    def __init__(self, owner, model):
        self.owner = owner
        self.handle = model.get_handle()
        self.library = {}
        self.message = ""
        self.expanded = False
        self._state_key = None
        self._records = None
        self._selection_intent = False
        model.bind_record_list("gamepad_profiles")
        model.bind("gamepad_profile_id", lambda: self.library.get("active", ""), self.select)
        model.bind_func("gamepad_profile_message", lambda: self.message)
        model.bind_func("gamepad_profile_manage", lambda: self.expanded)
        model.bind_func("gamepad_profile_manage_title", lambda: self.tr("manage"))
        model.bind_func("gamepad_profile_can_delete", lambda: len(self.library.get("profiles", ())) > 1)
        model.bind_func("gamepad_profile_has_device", lambda: self.valid_device())
        model.bind("gamepad_profile_remember", self.remembered, self.remember)
        model.bind_event("gamepad_profile_toggle", lambda *_: self.toggle())
        for operation in ("new", "duplicate", "rename", "delete", "import", "export"):
            model.bind_event("gamepad_profile_" + operation, lambda *_args, op=operation: self.action(op))

    @staticmethod
    def tr(key):
        return lf.ui.tr(f"gamepad.profile.{key}")

    def mount(self, document):
        self._selection_intent = False
        element = document.get_element_by_id("gamepad-profile-select")
        if element is not None:
            for event in ("mousedown", "keydown"):
                element.add_event_listener(event, lambda _event: self._intent(True))
            element.add_event_listener("blur", lambda _event: self._intent(False))

    def _intent(self, value):
        self._selection_intent = value

    def toggle(self):
        self.expanded = not self.expanded
        self.handle.dirty("gamepad_profile_manage")

    def snapshot(self):
        raw = lf.ui.get_gamepad_profiles_json()
        return raw, json.loads(raw)

    def sync(self, force=False):
        key = (self.owner.navigation.get("profile_revision"), self.owner.navigation.get("selected"),
               self.owner.navigation.get("device_key"), self.tr("personal"))
        if force or key != self._state_key:
            try:
                _, self.library = self.snapshot()
                self._state_key = key
            except (ValueError, RuntimeError):
                self.message = self.tr("load_error")
                self.handle.dirty("gamepad_profile_message")
                return
            records = [{"id": p["id"], "label": self.label(p)} for p in self.library["profiles"]]
            if records != self._records:
                self._records = records
                self.handle.update_record_list("gamepad_profiles", records)
            for field in ("id", "can_delete", "has_device", "remember", "manage_title"):
                self.handle.dirty("gamepad_profile_" + field)

    def label(self, profile):
        return self.tr("personal") if profile["id"] == "personal" and profile["name"] == "Personal" else profile["name"]

    def save(self, library, expected):
        self.owner.deactivate()
        try:
            if not lf.ui.set_gamepad_profiles_json(json.dumps(library), expected):
                raise RuntimeError("Unavailable viewer")
            self.message = ""
            self.owner.sync()
            self.sync(force=True)
            return True
        except (ValueError, RuntimeError):
            self.message = self.tr("save_error")
            self.sync(force=True)
            return False
        finally:
            self.handle.dirty("gamepad_profile_message")

    def select(self, identifier):
        if not self._selection_intent:
            self.handle.dirty("gamepad_profile_id")
            return
        self._selection_intent = False
        try:
            expected, library = self.snapshot()
            if identifier != library["active"] and any(p["id"] == identifier for p in library["profiles"]):
                library["active"] = identifier
                self.save(library, expected)
        except (ValueError, RuntimeError):
            self.error("load_error")

    def valid_device(self):
        key = self.owner.navigation.get("device_key", "")
        return bool(self.owner.navigation.get("selected") and len(key) == 32 and
                    all(c in "0123456789abcdef" for c in key) and key != "0" * 32)

    def remembered(self):
        key = self.owner.navigation.get("device_key", "")
        return self.valid_device() and self.library.get("device_profiles", {}).get(key) == self.library.get("active")

    def remember(self, enabled):
        if not self.valid_device() or bool(enabled) == bool(self.remembered()):
            return
        try:
            expected, library = self.snapshot()
            key = self.owner.navigation["device_key"]
            if enabled:
                library["device_profiles"][key] = library["active"]
            elif library["device_profiles"].get(key) == library["active"]:
                del library["device_profiles"][key]
            self.save(library, expected)
        except (ValueError, RuntimeError):
            self.error("save_error")

    def error(self, key):
        self.message = self.tr(key)
        self.handle.dirty("gamepad_profile_message")

    def action(self, operation):
        self.owner.deactivate()
        try:
            expected, library = self.snapshot()
            current = next(p for p in library["profiles"] if p["id"] == library["active"])
            if operation == "delete":
                if len(library["profiles"]) <= 1:
                    return
                label = self.tr("delete")
                def answer(button):
                    if button != label:
                        return
                    library["profiles"] = [p for p in library["profiles"] if p["id"] != current["id"]]
                    library["active"] = library["profiles"][0]["id"]
                    library["device_profiles"] = {k: v for k, v in library["device_profiles"].items() if v != current["id"]}
                    self.save(library, expected)
                lf.ui.confirm_dialog(label, self.tr("delete_confirm").format(name=self.label(current)),
                                     [lf.ui.tr("common.cancel"), label], answer)
                return
            if operation == "export":
                self.export(current)
                return
            if operation == "import":
                from _lfs_path_io import Path
                path = lf.ui.open_json_file_dialog()
                if not path:
                    return
                with Path(path).open("rb") as source:
                    data = source.read(MAX_IMPORT_BYTES + 1)
                if len(data) > MAX_IMPORT_BYTES:
                    raise ValueError("Controller profile is too large")
                imported = profile_import(json.loads(data.decode("utf-8-sig")))
                settings = imported["settings"]
                default_name = imported["name"] or self.tr("imported")
            elif operation == "new":
                settings = json.loads(lf.ui.get_gamepad_default_profile_json())
                settings.pop("enabled", None)
                default_name = self.tr("new_name")
            elif operation in ("duplicate", "rename"):
                settings = copy.deepcopy(current["settings"])
                default_name = self.label(current) if operation == "rename" else self.tr("copy_name").format(name=self.label(current))
            else:
                return
            if operation != "rename":
                base = default_name
                index = 2
                while any(p["name"] == default_name for p in library["profiles"]):
                    default_name = f"{base} {index}"
                    index += 1
            def named(name):
                if name is None:
                    return
                name = name.strip()
                if operation == "rename":
                    current["name"] = name
                else:
                    identifier = uuid.uuid4().hex
                    library["profiles"].append({"id": identifier, "name": name, "settings": settings})
                    library["active"] = identifier
                self.save(library, expected)
            lf.ui.input_dialog(self.tr(operation), self.tr("name"), default_name, named)
        except (OSError, ValueError, RuntimeError, UnicodeError, StopIteration):
            self.error("import_error" if operation == "import" else "load_error")

    def export(self, profile):
        from _lfs_path_io import Path
        from .persisted_settings import write_object
        path, confirmed = lf.ui.save_json_file_dialog_with_confirmation("joypad-profile.json")
        if not path:
            return
        target = Path(path)
        payload = profile_export(profile)
        def write():
            try:
                write_object(target, payload)
                self.message = ""
                self.handle.dirty("gamepad_profile_message")
            except (OSError, ValueError, UnicodeError):
                self.error("export_error")
        if target.exists() and not confirmed:
            label = self.tr("replace")
            lf.ui.confirm_dialog(self.tr("export"), self.tr("replace_confirm"),
                                 [lf.ui.tr("common.cancel"), label], lambda button: write() if button == label else None)
        else:
            write()
