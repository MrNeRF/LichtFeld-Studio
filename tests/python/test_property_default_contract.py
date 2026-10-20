# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Enum metadata exposes a numeric value and an explicit control identifier."""

import pytest


def _assert_enum_default(info):
    item = next(item for item in info["items"] if item["value"] == info["default"])
    assert info["default_identifier"] == item["identifier"]


@pytest.mark.parametrize("group", ["render_settings", "optimization"])
def test_declared_enum_defaults_resolve_by_value(lf, group):
    properties = lf.ui.property_group_info(group)["properties"]
    enums = [p for p in properties if p["type"] == "enum"]
    assert enums
    for info in enums:
        assert "default" in info and "default_identifier" in info, info["id"]
        _assert_enum_default(info)


def test_sparse_rendering_defaults(lf):
    properties = {p["id"]: p for p in lf.ui.property_group_info("render_settings")["properties"]}
    assert properties["raster_backend"]["default"] == 2
    assert properties["raster_backend"]["default_identifier"] == "3dgs"
    assert properties["mesh_shadow_resolution"]["default"] == 2048
    assert properties["mesh_shadow_resolution"]["default_identifier"] == "2048"


def test_session_enum_defaults_use_same_contract(lf):
    params = lf.optimization_params()
    properties = lf.ui.property_group_info("optimization")["properties"]
    for prop in properties:
        if prop["type"] == "enum":
            _assert_enum_default(params.prop_info(prop["id"]))


def test_python_enum_default_identifier(lf):
    from lfs_plugins.props import EnumProperty

    class Example:
        @classmethod
        def _get_property_descriptors(cls):
            return {"mode": EnumProperty(items=[("first", "First", ""), ("second", "Second", "")], default="second")}

    group = "tests.enum_default_contract"
    lf.ui.register_property_group(group, "Contract", Example)
    try:
        info = lf.ui.property_group_info(group)["properties"][0]
        _assert_enum_default(info)
        assert info["default"] == 1
        assert info["default_identifier"] == "second"
    finally:
        lf.ui.unregister_property_group(group)


def test_non_enum_defaults_preserve_falsey_values(lf):
    from lfs_plugins.props import BoolProperty, FloatProperty, IntProperty, StringProperty

    class Example:
        @classmethod
        def _get_property_descriptors(cls):
            return {"enabled": BoolProperty(default=False), "count": IntProperty(default=0),
                    "amount": FloatProperty(default=0.0), "name": StringProperty(default="")}

    group = "tests.scalar_default_contract"
    lf.ui.register_property_group(group, "Contract", Example)
    try:
        properties = {p["id"]: p for p in lf.ui.property_group_info(group)["properties"]}
        for name, value in {"enabled": False, "count": 0, "amount": 0.0, "name": ""}.items():
            assert properties[name]["default"] == value
            assert type(properties[name]["default"]) is type(value)
            assert "default_identifier" not in properties[name]
    finally:
        lf.ui.unregister_property_group(group)


def test_invalid_python_enum_default_retains_ordinary_plugin_behavior(lf):
    from lfs_plugins.props import BoolProperty, EnumProperty, PropertyGroup

    class Example(PropertyGroup):
        mode = EnumProperty(items=[("only", "Only", "")], default="missing")
        enabled = BoolProperty(default=False)

    group = "tests.invalid_enum_default_contract"
    lf.ui.register_property_group(group, "Contract", Example)
    try:
        properties = {p["id"]: p for p in lf.ui.property_group_info(group)["properties"]}
        assert set(properties) == {"mode", "enabled"}
        assert "default" not in properties["mode"]
        assert "default_identifier" not in properties["mode"]
        instance = Example()
        assert instance.mode == "missing"  # Existing descriptor behavior is preserved.
        instance.mode = "only"
        assert instance.mode == "only"
        Example.mode.default = "only"
        lf.ui.register_property_group(group, "Contract", Example)
        fixed = {p["id"]: p for p in lf.ui.property_group_info(group)["properties"]}
        assert fixed["mode"]["default_identifier"] == "only"
        assert instance.mode == "only"  # Reloading metadata does not reset values.
    finally:
        lf.ui.unregister_property_group(group)


@pytest.mark.parametrize("kind, kwargs, edited", [
    ("EnumProperty", {"items": [("a", "A", ""), ("a", "Again", "")], "default": "a"}, "a"),
    ("IntProperty", {"default": "legacy"}, 7),
    ("FloatProperty", {"default": 2.0, "min": 0.0, "max": 1.0}, 0.75),
])
def test_imprecise_defaults_preserve_properties_and_callbacks(lf, kind, kwargs, edited):
    from lfs_plugins import props
    changes = []
    descriptor = getattr(props, kind)(**kwargs, update=lambda obj, _ctx: changes.append(obj.value))
    example = type("LegacySettings", (props.PropertyGroup,), {"value": descriptor})
    group = "tests.legacy_default_contract"
    lf.ui.register_property_group(group, "Contract", example)
    try:
        info = lf.ui.property_group_info(group)["properties"][0]
        assert info["id"] == "value"
        assert "default" not in info
        assert "default_identifier" not in info
        instance = example()
        assert instance.value == kwargs["default"]
        instance.value = edited
        assert instance.value == edited
        assert changes == [edited]
    finally:
        lf.ui.unregister_property_group(group)


def test_real_rendering_descriptors_use_same_default(lf):
    settings = lf.get_render_settings()
    if settings is None:
        pytest.skip("Requires a GUI rendering context")
    descriptors = settings.get_all_properties()
    for info in lf.ui.property_group_info("render_settings")["properties"]:
        if info["type"] == "enum":
            assert descriptors[info["id"]].default == info["default_identifier"]
            assert settings.prop_info(info["id"])["default_identifier"] == info["default_identifier"]
