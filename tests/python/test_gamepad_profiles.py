# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def profiles(monkeypatch):
    source = Path(__file__).resolve().parents[2] / "src/python"
    monkeypatch.syspath_prepend(str(source))
    package = ModuleType("_gamepad_profiles_test")
    package.__path__ = [str(source / "lfs_plugins")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    settings = {"schema_version": 1, "dead_zone": .15, "look_speed": 1.5, "move_speed": 1.,
                "bindings": {"common": {}, "modes": {}}}
    library = {"schema_version": 1, "active": "personal", "profiles": [
        {"id": "personal", "name": "Personal", "settings": copy.deepcopy(settings)}], "device_profiles": {}}
    prompts, confirmations, saves, dirty, records = [], [], [], [], {}
    navigation = {"enabled": True, "selected": 7, "device_key": "a" * 32, "profile_revision": 1}
    def save(raw, expected):
        if expected != json.dumps(library):
            raise RuntimeError("Concurrent edit")
        value = json.loads(raw)
        names = [p["name"] for p in value["profiles"]]
        if any(not n for n in names) or len(set(names)) != len(names):
            raise ValueError("Invalid names")
        library.clear()
        library.update(value)
        saves.append(copy.deepcopy(value))
        navigation["profile_revision"] += 1
        return True
    ui = SimpleNamespace(
        tr=lambda key: {"gamepad.profile.copy_name": "Copy of {name}",
                        "gamepad.profile.delete_confirm": "Delete {name}?"}.get(key, key),
        get_gamepad_profiles_json=lambda: json.dumps(library), set_gamepad_profiles_json=save,
        get_gamepad_default_profile_json=lambda: json.dumps({**settings, "enabled": False}),
        input_dialog=lambda *args: prompts.append(args), confirm_dialog=lambda *args: confirmations.append(args),
        open_json_file_dialog=lambda: "", save_json_file_dialog_with_confirmation=lambda *_: ("", False))
    monkeypatch.setitem(sys.modules, "lichtfeld", SimpleNamespace(ui=ui))
    name = package.__name__ + ".gamepad_profiles"
    spec = importlib.util.spec_from_file_location(name, source / "lfs_plugins/gamepad_profiles.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    handle = SimpleNamespace(dirty=dirty.append, update_record_list=lambda name, rows: records.update({name: rows}))
    model = SimpleNamespace(get_handle=lambda: handle, bind_record_list=lambda *_: None,
                            bind=lambda *_: None, bind_func=lambda *_: None, bind_event=lambda *_: None)
    owner = SimpleNamespace(navigation=navigation, deactivate=lambda: None, sync=lambda: None)
    subject = module.GamepadProfilesSection(owner, model)
    subject.sync()
    return SimpleNamespace(subject=subject, module=module, library=library, navigation=navigation, ui=ui,
                           prompts=prompts, confirmations=confirmations, saves=saves, dirty=dirty, records=records)


def named(profiles, operation, name):
    profiles.subject.action(operation)
    profiles.prompts[-1][-1](name)


def test_create_duplicate_rename_and_selection_do_not_change_enablement(profiles):
    p = profiles
    named(p, "new", "Flight")
    flight = p.library["active"]
    assert flight != "personal"
    assert p.library["profiles"][-1]["name"] == "Flight"
    assert "enabled" not in p.library["profiles"][-1]["settings"]
    p.library["profiles"][-1]["settings"]["move_speed"] = 2.5
    named(p, "duplicate", "Precision")
    assert p.library["active"] != flight
    assert p.library["profiles"][-1]["settings"]["move_speed"] == 2.5
    identifier = p.library["active"]
    named(p, "rename", "Precise")
    assert p.library["active"] == identifier
    assert p.library["profiles"][-1]["name"] == "Precise"
    p.subject._intent(True)
    p.subject.select(flight)
    assert p.library["active"] == flight
    assert p.navigation["enabled"] is True


def test_select_rebuild_and_cancelled_name_prompt_do_not_change_library(profiles):
    p = profiles
    named(p, "new", "Other")
    before = copy.deepcopy(p.library)
    p.subject.select("personal")
    assert p.library == before
    named(p, "rename", None)
    assert p.library == before
    assert p.dirty[-1] == "gamepad_profile_id"


def test_delete_requires_confirmation_and_removes_only_its_associations(profiles):
    p = profiles
    named(p, "new", "Other")
    p.library["device_profiles"] = {"a" * 32: p.library["active"], "b" * 32: "personal"}
    before = copy.deepcopy(p.library)
    p.subject.action("delete")
    p.confirmations[-1][-1]("common.cancel")
    assert p.library == before
    p.subject.action("delete")
    p.confirmations[-1][-1]("gamepad.profile.delete")
    assert p.library["active"] == "personal"
    assert p.library["device_profiles"] == {"b" * 32: "personal"}
    count = len(p.confirmations)
    p.subject.action("delete")
    assert len(p.confirmations) == count


def test_rename_preserves_model_link_and_disconnected_device_cannot_be_linked(profiles):
    p = profiles
    p.subject.remember(True)
    assert p.library["device_profiles"]["a" * 32] == "personal"
    named(p, "rename", "Custom")
    assert p.library["device_profiles"]["a" * 32] == "personal"
    p.subject.remember(False)
    assert not p.library["device_profiles"]
    p.navigation["selected"] = 0
    p.subject.remember(True)
    assert not p.library["device_profiles"]


def test_delayed_modal_cannot_overwrite_a_newer_profile_edit(profiles):
    p = profiles
    p.subject.action("rename")
    p.library["profiles"][0]["settings"]["move_speed"] = 3.0
    p.prompts[-1][-1]("Stale")
    assert p.library["profiles"][0]["name"] == "Personal"
    assert p.library["profiles"][0]["settings"]["move_speed"] == 3.0
    assert p.subject.message == "gamepad.profile.save_error"


def test_remembered_model_follows_explicit_profile_selection_and_can_always_be_unlinked(profiles):
    p = profiles
    p.subject.remember(True)
    named(p, 'new', 'Other')
    p.subject._intent(True)
    p.subject.select('personal')
    p.subject.sync(force=True)
    assert p.subject.remembered()
    assert p.library['device_profiles']['a' * 32] == 'personal'
    # An external profile switch can leave a different profile linked. The UI
    # must still expose and allow clearing that model association.
    p.library['active'] = p.library['profiles'][-1]['id']
    p.subject.sync(force=True)
    assert p.subject.remembered()
    p.subject.remember(False)
    assert not p.library['device_profiles']


def test_profile_name_prompt_does_not_reassign_a_different_connected_model(profiles):
    p = profiles
    p.subject.remember(True)
    p.library['device_profiles']['b' * 32] = 'personal'
    p.subject.action('new')
    p.navigation['device_key'] = 'b' * 32
    p.prompts[-1][-1]('Other')
    assert p.library['device_profiles'] == {'a' * 32: 'personal', 'b' * 32: 'personal'}


def test_duplicate_name_is_not_published(profiles):
    p = profiles
    named(p, "new", "Personal")
    assert len(p.library["profiles"]) == 1
    assert not p.saves


def test_export_import_roundtrip_excludes_connection_enablement_and_links(profiles):
    p = profiles
    profile = p.library["profiles"][0]
    profile["settings"]["enabled"] = True
    exported = p.module.profile_export(profile)
    assert set(exported) == {"format", "schema_version", "name", "settings"}
    assert "enabled" not in exported["settings"]
    assert p.module.profile_import(exported) == exported
    legacy = copy.deepcopy(profile["settings"])
    assert "enabled" not in p.module.profile_import(legacy)["settings"]


@pytest.mark.parametrize("bad", [[], {}, {"format": "keymap", "bindings": []},
    {"format": "lichtfeld.gamepad-profile", "schema_version": 2, "name": "Future", "settings": {}},
    {"format": "lichtfeld.gamepad-profile", "schema_version": True, "name": "Bad", "settings": {}},
    {"format": "lichtfeld.gamepad-profile", "schema_version": 1, "name": "Bad", "settings": {"enabled": True}}])
def test_unrelated_future_and_invalid_imports_are_rejected(profiles, bad):
    with pytest.raises(ValueError):
        profiles.module.profile_import(bad)


def test_import_publishes_a_new_profile_and_preserves_existing_profiles(profiles, tmp_path):
    p = profiles
    path = tmp_path / "controller.json"
    value = p.module.profile_export(p.library["profiles"][0])
    path.write_text(json.dumps(value), encoding="utf-8")
    p.ui.open_json_file_dialog = lambda: str(path)
    p.subject.action("import")
    p.prompts[-1][-1]("Imported")
    assert len(p.library["profiles"]) == 2
    assert p.library["profiles"][0]["name"] == "Personal"
    assert p.library["profiles"][1]["settings"] == value["settings"]


def test_export_does_not_overwrite_without_confirmation(profiles, tmp_path):
    p = profiles
    path = tmp_path / "existing.json"
    path.write_text("keep", encoding="utf-8")
    p.ui.save_json_file_dialog_with_confirmation = lambda *_: (str(path), False)
    p.subject.action("export")
    p.confirmations[-1][-1]("common.cancel")
    assert path.read_text() == "keep"
    p.subject.action("export")
    p.confirmations[-1][-1]("gamepad.profile.replace")
    assert json.loads(path.read_text(encoding="utf-8"))["format"] == p.module.FORMAT
    assert not list(tmp_path.glob("*.tmp"))


def test_oversized_import_preserves_library(profiles, tmp_path):
    p = profiles
    path = tmp_path / "huge.json"
    path.write_bytes(b" " * (p.module.MAX_IMPORT_BYTES + 1))
    p.ui.open_json_file_dialog = lambda: str(path)
    p.subject.action("import")
    assert not p.prompts and not p.saves
    assert p.subject.message == "gamepad.profile.import_error"


def test_all_dynamic_profile_labels_exist_in_every_locale():
    root = Path(__file__).resolve().parents[2] / "src/visualizer/gui/resources/locales"
    keys = "manage new duplicate rename personal name new_name copy_name imported remember remember_help help delete_confirm load_error save_error import_error export_error replace_confirm reset_scope reset_all_scope changed_error active delete import export replace".split()
    for path in root.glob("*.json"):
        values = json.loads(path.read_text(encoding="utf-8"))
        for key in keys:
            assert values.get("gamepad.profile." + key), (path.name, key)
