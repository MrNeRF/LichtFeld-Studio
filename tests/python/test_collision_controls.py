# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from xml.etree import ElementTree as ET
import sys
import pytest
from test_depth_view_scrub import attach_fields, Event
from test_localization_contracts import _flatten

ROOT = Path(__file__).resolve().parents[2]


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
    assert tuple(controls.values[k] for k in module.WALK_FIELDS) == (.6, .15, .35, 5.)
    module.lf.set_walk_settings.side_effect = RuntimeError("The edited Walk capsule does not fit")
    scrub._on_field_keydown(prop, enter)
    field.input_el.set_attribute("value", ".9")
    scrub._on_input_change(prop, Event(linebreak=True))
    assert controls.values["step"] == .35 and controls.error == "collision.walk_blocked"
    scrub._on_field_keydown(prop, enter)
    field.input_el.set_attribute("value", "nan")
    scrub._on_input_change(prop, Event(linebreak=True))
    assert controls.error and not field.editing
    scrub._on_field_keydown(prop, enter)
    assert field.editing


@pytest.mark.parametrize("approved,choice", [(False, "common.cancel"), (False, "collision.export"),
                                          (True, None), (False, None)])
def test_export_picker_and_pair_confirmation_latch_target(controls, tmp_path, approved, choice):
    from lfs_plugins import collision_rendering as module
    path = tmp_path / "collision.voxel.json"
    path.with_suffix(".bin").write_bytes(b"old")
    controls.state = {"target_uuid": "first", "target_name": "Floor"}
    def pick(*args, **kwargs):
        assert kwargs == {"with_overwrite_confirmation": True}
        controls.state["target_uuid"] = "second"
        return (str(path) if approved or choice else "", approved)
    module.lf.ui.save_json_file_dialog.side_effect = pick
    controls.export()
    if choice:
        prompt = module.lf.ui.confirm_dialog.call_args.args
        assert prompt[1] == f"collision.replace_pair\n{path}\n{path.with_suffix('.bin')}"
        prompt[-1](choice)
    else:
        module.lf.ui.confirm_dialog.assert_not_called()
    expected = [("first", str(path))] if approved or choice == "collision.export" else []
    assert [call.args for call in module.lf.export_collision.call_args_list] == expected
    assert path.with_suffix(".bin").read_bytes() == b"old" and not path.exists()
    module.lf.export_collision.side_effect = RuntimeError("destination denied")
    module.lf.ui.save_json_file_dialog.side_effect = lambda *a, **kw: (str(path), True)
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
    controls.popup = True
    assert bindings["collision_error"]() == "/tmp/output.bin: denied"
    assert controls.localized_error("No Walk support below the camera") == "collision.walk_no_support"


def test_generator_and_rendering_surfaces_and_translations(controls):
    from lfs_plugins import collision_rendering as module
    resources = ROOT / "src/visualizer/gui/rmlui/resources"
    generator, rendering = (ET.parse(resources / name).getroot() for name in ("collision.rml", "rendering.rml"))
    assert generator.find(".//button[@data-event-click='collision_generate']").get("data-attrif-disabled") == "!collision_can_generate"
    assert generator.find(".//*[@id='collision-size-fields']").get("data-if") == "!collision_automatic"
    assert rendering.find(".//*[@id='hdr-collision']/..").get("data-if") == "collision_has_collision"
    assert rendering.find(".//*[@id='collision-walk-fields']") is not None
    assert rendering.find(".//*[@data-value='collision_preview_opacity']") is not None
    assert rendering.find(".//*[@data-event-click='collision_generate']") is None
    for root in (generator, rendering):
        assert root.find(".//*[@data-if='collision_has_error']/span").text == "@tr:status.error"
    controls.mount(MagicMock())
    labels = {element.text for root in (generator, rendering)
              for element in root.findall(".//span[@class='setting-row__label-col']")}
    assert {"@tr:main_panel.voxel_size", "@tr:ui.speed"} <= labels
    keys = {element.get("data-tooltip") for root in (generator, rendering)
            for element in root.iter() if element.get("data-tooltip", "").startswith("tooltip.collision.")}
    assert {"tooltip.collision." + key for key in ("voxel_size", "opacity", "area")} <= keys
    keys.update(("tooltip.collision.auto", "tooltip.collision.cleanup", "collision.replace_pair", "collision.walk_enter_failed"))
    keys.update((*module.ERRORS.values(), *module.STAGES.values(), "status.error", "ui.speed"))
    locales = sorted((ROOT / "src/visualizer/gui/resources/locales").glob("*.json"))
    assert len(locales) == 10
    for locale in locales:
        translations = dict(_flatten(json.loads(locale.read_text())))
        assert all(translations.get(key) for key in keys), locale.name


@pytest.mark.parametrize("resource,expected", [
    ("collision.rml", {"size", "opacity", "area"}),
    ("rendering.rml", {"height", "radius", "step", "speed"}),
])
def test_collision_controls_mount_declared_fields(controls, resource, expected):
    root = ET.parse(ROOT / "src/visualizer/gui/rmlui/resources" / resource).getroot()

    def element():
        node = MagicMock()
        node.attrs = {}
        node.get_attribute.side_effect = lambda name, default="": node.attrs.get(name, default)
        node.set_attribute.side_effect = node.attrs.__setitem__
        node.append_child.side_effect = lambda _: element()
        return node

    doc = element()
    doc.create_element.side_effect = lambda _: element()
    ranges = []
    for field in root.findall(".//input[@type='range'][@class='setting-slider']"):
        node = element()
        node.attrs.update(field.attrib)
        ranges.append(node)
    doc.query_selector_all.return_value = ranges
    controls.mount(doc)
    assert set(controls.scrubs._fields) == {f"collision_{key}" for key in expected}
    for prop, state in controls.scrubs._fields.items():
        state.display.set_text.assert_called_with(
            controls.scrubs._format_value(state.spec, controls.values[prop.removeprefix("collision_")]))
