# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
from types import SimpleNamespace
from unittest.mock import MagicMock
import sys
import pytest
from test_depth_view_scrub import attach_fields, Event


@pytest.fixture
def controls(monkeypatch):
    monkeypatch.setitem(sys.modules, "lichtfeld", SimpleNamespace())
    from lfs_plugins import collision_rendering as module
    lf = MagicMock()
    lf.ui.tr.side_effect = lambda key: key
    lf.get_collision_state.return_value = {"walk_settings": (.5, .1, .2, 4.)}
    lf.get_collision_state.side_effect = lambda: dict(lf.get_collision_state.return_value)
    lf.set_walk_settings.side_effect = lambda *settings: lf.get_collision_state.return_value.update(walk_settings=settings)
    monkeypatch.setattr(module, "lf", lf)
    monkeypatch.setitem(sys.modules, "lichtfeld", lf)
    return module.CollisionRendering()


def test_completed_scrub_edits_recover_and_walk_merges_live_settings(controls):
    from lfs_plugins import collision_rendering as module
    controls.update()
    scrub, prop = controls.scrubs, "collision_step"
    keys = attach_fields(scrub)
    field = scrub._fields[prop]
    field.input_el.blur = lambda: scrub._on_input_blur(prop, None)
    enter = Event(key_identifier=keys.KI_RETURN)
    scrub._on_field_keydown(prop, enter)
    for draft in ("0", "0.", "0.35"):
        field.input_el.set_attribute("value", draft)
        scrub._on_input_change(prop, Event())
        assert controls.values["step"] == .2
    module.lf.get_collision_state.return_value = {"walk_settings": (.6, .15, .25, 5.)}
    controls.update()
    assert field.input_el.get_attribute("value") == "0.35"
    scrub._on_input_change(prop, Event(linebreak=True))
    assert not field.editing
    module.lf.set_walk_settings.assert_called_once_with(.6, .15, .35, 5.)
    blocked = RuntimeError("The edited Walk capsule does not fit")
    blocked.message_key = "collision.walk_blocked"
    module.lf.set_walk_settings.side_effect = blocked
    scrub._on_field_keydown(prop, enter)
    field.input_el.set_attribute("value", ".9")
    scrub._on_input_change(prop, Event(linebreak=True))
    assert controls.values["step"] == .35 and controls.error == "collision.walk_blocked"
    scrub._on_field_keydown(prop, enter)
    field.input_el.set_attribute("value", "nan")
    scrub._on_input_change(prop, Event(linebreak=True))
    assert controls.error == "preferences.viewport_defaults_invalid" and not field.editing
    scrub._on_field_keydown(prop, enter)
    assert field.editing


@pytest.mark.parametrize("approved,choice", [(False, "common.cancel"), (False, "export.export"),
                                          (True, None), (False, None)])
def test_export_picker_and_pair_confirmation_latch_target(controls, tmp_path, approved, choice):
    from lfs_plugins import collision_rendering as module
    path = tmp_path / "collision.voxel.json"
    path.with_suffix(".bin").write_bytes(b"old")
    controls.state = {"target_uuid": "first", "target_name": "Floor"}
    def pick(*args, **kwargs):
        assert args == ("collision.voxel.json",) and not kwargs
        controls.state["target_uuid"] = "second"
        return (str(path) if approved or choice else "", approved)
    module.lf.ui.save_json_file_dialog_with_confirmation.side_effect = pick
    controls.export()
    if choice:
        prompt = module.lf.ui.confirm_dialog.call_args.args
        assert prompt[1] == f"collision.replace_pair\n{path}\n{path.with_suffix('.bin')}"
        prompt[-1](choice)
    else:
        module.lf.ui.confirm_dialog.assert_not_called()
    expected = [("first", str(path))] if approved or choice == "export.export" else []
    assert [call.args for call in module.lf.export_collision.call_args_list] == expected
    assert path.with_suffix(".bin").read_bytes() == b"old" and not path.exists()
    module.lf.export_collision.side_effect = RuntimeError("destination denied")
    module.lf.ui.save_json_file_dialog_with_confirmation.side_effect = lambda *a: (str(path), True)
    controls.run(controls.export)
    assert controls.error == "destination denied"


def test_bindings_generation_resolution_and_idle_updates(controls):
    from lfs_plugins import collision_rendering as module
    model, bindings, events = MagicMock(), {}, {}
    model.bind_func.side_effect = bindings.__setitem__
    model.bind_event.side_effect = events.__setitem__
    controls.bind(model)
    assert not bindings["collision_can_generate"]()
    for changes in ({"source_name": "Selected"}, {"active": True, "progress": .4, "stage": "Sorting collision records"},
                    {"active": False, "error": "/tmp/output.bin: denied"}):
        module.lf.get_collision_state.return_value.update(changes)
        assert controls.update()
        assert controls.update() is False
        assert bindings["collision_can_generate"]() == (not controls.state.get("active", False))
    controls.state.update(target_uuid="internal-id", target_name="Floor")
    assert bindings["collision_target"]() == "Floor"
    controls.values.update(size=.005, opacity=.2, area=3., automatic=True)
    events["collision_generate"]()
    module.lf.generate_collision.assert_called_once_with(.005, .2, True, True, 3.)
    controls.state.update(resolved_voxel_size=.01234, generation_settings=(.00500000007, .200000001, True, True, 3.))
    assert controls.resolved_size() == "0.01234"
    for changes in ({"cleanup": False}, {"cleanup": True, "area": 1}, {"area": 3}):
        controls.values.update(changes)
        if changes == {"area": 3}:
            controls.state["resolved_voxel_size"] = 0
        assert controls.resolved_size() == "collision.size_pending"
    for stage, key in (("Preparing collision", "gallery.transfer.phase.preparing"),
                       ("Publishing voxel metadata", "export.export"), ("Collision ready", "status.ready")):
        controls.state["stage"] = stage
        assert bindings["collision_status"]().startswith(key)


def test_structured_errors_translate_only_explicit_keys(controls):
    error = RuntimeError("Readable English diagnostic, not a translation lookup")
    error.message_key = "collision.output_limit"
    error.message_details = "/tmp/collision.walk_no_support/64000001.bin"
    controls.show_error(error)
    assert controls.error == f"{error.message_key}\n{error.message_details}"
    assert controls.localized_error(RuntimeError(error.message_details)) == error.message_details
