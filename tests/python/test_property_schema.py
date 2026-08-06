# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for shared registry metadata from operator properties."""

import sys
from pathlib import Path

import pytest


@pytest.fixture
def lf():
    """Import lichtfeld module."""
    project_root = Path(__file__).parent.parent.parent
    build_python = project_root / "build" / "src" / "python"
    if str(build_python) not in sys.path:
        sys.path.insert(0, str(build_python))

    try:
        import lichtfeld

        return lichtfeld
    except ImportError as e:
        pytest.skip(f"lichtfeld module not available: {e}")


@pytest.fixture
def lfs_types():
    """Import lfs_plugins.types module."""
    project_root = Path(__file__).parent.parent.parent
    build_python = project_root / "build" / "src" / "python"
    if str(build_python) not in sys.path:
        sys.path.insert(0, str(build_python))

    try:
        from lfs_plugins import types

        return types
    except ImportError as e:
        pytest.skip(f"lfs_plugins.types module not available: {e}")


class TestOperatorPropertyExtraction:
    """Tests for property metadata extraction from Python operators."""

    def test_float_property_basic(self, lf, lfs_types):
        """FloatProperty should be recognized."""
        received = {}

        class FloatOp(lfs_types.Operator):
            lf_label = "Float Op"
            amount: float = 1.0

            def execute(self, context):
                received["amount"] = self.amount
                return {"FINISHED"}

        lf.register_class(FloatOp)
        try:
            lf.ops.invoke(FloatOp._class_id())
            assert received["amount"] == 1.0
        finally:
            lf.unregister_class(FloatOp)

    def test_int_property_basic(self, lf, lfs_types):
        """IntProperty should be recognized."""
        received = {}

        class IntOp(lfs_types.Operator):
            lf_label = "Int Op"
            count: int = 10

            def execute(self, context):
                received["count"] = self.count
                return {"FINISHED"}

        lf.register_class(IntOp)
        try:
            lf.ops.invoke(IntOp._class_id())
            assert received["count"] == 10
        finally:
            lf.unregister_class(IntOp)

    def test_string_property_basic(self, lf, lfs_types):
        """StringProperty should be recognized."""
        received = {}

        class StringOp(lfs_types.Operator):
            lf_label = "String Op"
            name: str = "default"

            def execute(self, context):
                received["name"] = self.name
                return {"FINISHED"}

        lf.register_class(StringOp)
        try:
            lf.ops.invoke(StringOp._class_id())
            assert received["name"] == "default"
        finally:
            lf.unregister_class(StringOp)

    def test_bool_property_basic(self, lf, lfs_types):
        """BoolProperty should be recognized."""
        received = {}

        class BoolOp(lfs_types.Operator):
            lf_label = "Bool Op"
            enabled: bool = False

            def execute(self, context):
                received["enabled"] = self.enabled
                return {"FINISHED"}

        lf.register_class(BoolOp)
        try:
            lf.ops.invoke(BoolOp._class_id())
            assert received["enabled"] is False
        finally:
            lf.unregister_class(BoolOp)


class TestOperatorPropertyConstraints:
    """Tests for operator properties with constraints (min/max/step)."""

    def test_override_float_with_value(self, lf, lfs_types):
        """Float properties should accept override values."""
        received = {}

        class ConstrainedFloat(lfs_types.Operator):
            lf_label = "Constrained Float"
            value: float = 0.5

            def execute(self, context):
                received["value"] = self.value
                return {"FINISHED"}

        lf.register_class(ConstrainedFloat)
        try:
            # Override with explicit value
            lf.ops.invoke(ConstrainedFloat._class_id(), value=0.75)
            assert abs(received["value"] - 0.75) < 0.001
        finally:
            lf.unregister_class(ConstrainedFloat)

    def test_override_int_with_value(self, lf, lfs_types):
        """Int properties should accept override values."""
        received = {}

        class ConstrainedInt(lfs_types.Operator):
            lf_label = "Constrained Int"
            count: int = 5

            def execute(self, context):
                received["count"] = self.count
                return {"FINISHED"}

        lf.register_class(ConstrainedInt)
        try:
            lf.ops.invoke(ConstrainedInt._class_id(), count=42)
            assert received["count"] == 42
        finally:
            lf.unregister_class(ConstrainedInt)


class TestOperatorPropertyComplexTypes:
    """Tests for complex property types (vectors, enums)."""

    def test_list_property_passthrough(self, lf, lfs_types):
        """Lists should pass through as properties."""
        received = {}

        class ListOp(lfs_types.Operator):
            lf_label = "List Op"

            def execute(self, context):
                values = getattr(self, "values", None)
                if values is not None:
                    received["sum"] = sum(values)
                return {"FINISHED"}

        lf.register_class(ListOp)
        try:
            lf.ops.invoke(ListOp._class_id(), values=[1, 2, 3, 4])
            assert received["sum"] == 10
        finally:
            lf.unregister_class(ListOp)

    def test_dict_property_passthrough(self, lf, lfs_types):
        """Dicts should pass through as properties."""
        received = {}

        class DictOp(lfs_types.Operator):
            lf_label = "Dict Op"

            def execute(self, context):
                config = getattr(self, "config", None)
                if config is not None:
                    received["key_count"] = len(config)
                return {"FINISHED"}

        lf.register_class(DictOp)
        try:
            lf.ops.invoke(DictOp._class_id(), config={"a": 1, "b": 2})
            assert received["key_count"] == 2
        finally:
            lf.unregister_class(DictOp)


class TestPropertyRegistryConversion:
    """Tests for the shared Python descriptor-to-property metadata converter."""

    def test_failed_operator_registration_removes_property_group(self, lf, lfs_types):
        from lfs_plugins import props

        class BrokenOperator(lfs_types.Operator):
            amount = props.FloatProperty(default=1.0)

            def __init__(self):
                raise RuntimeError("intentional registration failure")

        group_id = "operator." + BrokenOperator._class_id()
        lf.register_class(BrokenOperator)
        assert lf.ui.property_group_info(group_id) == {}

    def test_operator_arg_flags_vector_types_and_optional_bounds(self, lf, lfs_types):
        from lfs_plugins import props

        class Item(props.PropertyGroup):
            value = props.StringProperty(default="item")

        class UnknownProperty(props.Property):
            pass

        class BaseOperator(lfs_types.Operator):
            inherited = props.FloatProperty(default=0.25)
            inherited_text = props.StringProperty(default="base")
            inherited_vec3 = props.FloatVectorProperty(
                default=(0.25, 0.5, 0.75), size=3
            )

        class DescriptorOperator(BaseOperator):
            label = "Descriptor Operator"
            unconstrained = props.FloatProperty(default=0.5)
            constrained = props.FloatProperty(default=1.0, min=-2.0, max=3.0)
            enabled = props.BoolProperty(default=True)
            text = props.StringProperty(default="hello")
            color = props.FloatVectorProperty(
                default=(0.1, 0.2, 0.3), size=3, subtype=props.PropSubtype.COLOR
            )
            vec2 = props.FloatVectorProperty(default=(1.0, 2.0), size=2)
            vec3 = props.FloatVectorProperty(default=(1.0, 2.0, 3.0), size=3)
            vec4 = props.FloatVectorProperty(default=(1.0, 2.0, 3.0, 4.0), size=4)
            vec5 = props.FloatVectorProperty(
                default=(1.0, 2.0, 3.0, 4.0, 5.0), size=5
            )
            indices = props.IntVectorProperty(default=(1, 2, 3), size=3)
            mode = props.EnumProperty(
                items=[("REPLACE", "Replace", ""), ("ADD", "Add", "")],
                default="ADD",
            )
            invalid_enum = props.EnumProperty(items=[("ONLY", "Only")])
            invalid_enum_fields = props.EnumProperty(items=[("ONLY", "Only", 7)])
            tensor = props.TensorProperty(shape=(-1, 3))
            collection = props.CollectionProperty(type=Item)
            unknown = UnknownProperty(default=2.0)

            def execute(self, context):
                return {"FINISHED"}

        group_id = "operator." + DescriptorOperator._class_id()
        lf.register_class(DescriptorOperator)
        try:
            info = lf.ui.property_group_info(group_id)
            assert info["id"] == group_id
            metas = {meta["id"]: meta for meta in info["properties"]}

            expected_args = {
                "unconstrained",
                "constrained",
                "enabled",
                "text",
                "color",
                "vec2",
                "vec3",
                "vec4",
                "vec5",
                "indices",
                "mode",
                "tensor",
            }
            assert {
                prop_id for prop_id, meta in metas.items() if meta["operator_arg"]
            } == expected_args
            assert set(metas) == expected_args | {
                "inherited",
                "inherited_text",
                "inherited_vec3",
                "invalid_enum",
                "invalid_enum_fields",
                "collection",
                "unknown",
            }

            assert metas["vec2"]["type"] == "vec2"
            assert metas["vec3"]["type"] == "vec3"
            assert metas["vec4"]["type"] == "vec4"
            assert metas["vec5"]["type"] == "float_vector"
            assert metas["vec5"]["vector_size"] == 5
            assert metas["indices"]["type"] == "int_vector"
            assert metas["indices"]["vector_size"] == 3
            assert metas["color"]["type"] == "color3"

            assert "min" not in metas["unconstrained"]
            assert "max" not in metas["unconstrained"]
            assert metas["constrained"]["min"] == -2.0
            assert metas["constrained"]["max"] == 3.0
            assert metas["color"]["min"] == 0.0
            assert metas["color"]["max"] == 1.0

            assert all("default" not in metas[prop_id] for prop_id in expected_args)
            assert metas["inherited"]["default"] == 0.25
            assert metas["inherited_text"]["default"] == "base"
            assert list(metas["inherited_vec3"]["default"]) == [0.25, 0.5, 0.75]
            assert metas["unknown"]["default"] == 2.0
            assert not metas["inherited"]["operator_arg"]
            assert not metas["inherited_text"]["operator_arg"]
            assert not metas["inherited_vec3"]["operator_arg"]
            assert not metas["invalid_enum"]["operator_arg"]
            assert not metas["invalid_enum_fields"]["operator_arg"]
            assert not metas["collection"]["operator_arg"]
            assert not metas["unknown"]["operator_arg"]
        finally:
            lf.unregister_class(DescriptorOperator)

        assert lf.ui.property_group_info(group_id) == {}


class TestOperatorPropertyEdgeCases:
    """Tests for edge cases in operator property handling."""

    def test_none_property_value(self, lf, lfs_types):
        """None values should be handled correctly."""
        received = {}

        class NoneOp(lfs_types.Operator):
            lf_label = "None Op"

            def execute(self, context):
                val = getattr(self, "optional", "not_set")
                received["value"] = val
                return {"FINISHED"}

        lf.register_class(NoneOp)
        try:
            # Without the property
            lf.ops.invoke(NoneOp._class_id())
            assert received["value"] == "not_set"
        finally:
            lf.unregister_class(NoneOp)

    def test_empty_string_property(self, lf, lfs_types):
        """Empty strings should be handled correctly."""
        received = {}

        class EmptyStringOp(lfs_types.Operator):
            lf_label = "Empty String"
            text: str = "default"

            def execute(self, context):
                received["text"] = self.text
                received["empty"] = self.text == ""
                return {"FINISHED"}

        lf.register_class(EmptyStringOp)
        try:
            lf.ops.invoke(EmptyStringOp._class_id(), text="")
            assert received["empty"]
        finally:
            lf.unregister_class(EmptyStringOp)

    def test_special_characters_in_property(self, lf, lfs_types):
        """Special characters in string properties should work."""
        received = {}

        class SpecialCharsOp(lfs_types.Operator):
            lf_label = "Special Chars"
            text: str = ""

            def execute(self, context):
                received["text"] = self.text
                return {"FINISHED"}

        lf.register_class(SpecialCharsOp)
        try:
            special = "Hello\nWorld\t\"'\\"
            lf.ops.invoke(SpecialCharsOp._class_id(), text=special)
            assert received["text"] == special
        finally:
            lf.unregister_class(SpecialCharsOp)

    def test_zero_values(self, lf, lfs_types):
        """Zero values should be handled correctly (not treated as missing)."""
        received = {}

        class ZeroOp(lfs_types.Operator):
            lf_label = "Zero"
            int_val: int = 100
            float_val: float = 1.0

            def execute(self, context):
                received["int"] = self.int_val
                received["float"] = self.float_val
                return {"FINISHED"}

        lf.register_class(ZeroOp)
        try:
            lf.ops.invoke(ZeroOp._class_id(), int_val=0, float_val=0.0)
            assert received["int"] == 0
            assert received["float"] == 0.0
        finally:
            lf.unregister_class(ZeroOp)
