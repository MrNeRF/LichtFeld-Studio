# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for OperatorProperties typed API."""

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


@pytest.fixture
def numpy():
    """Import numpy."""
    try:
        import numpy as np

        return np
    except ImportError:
        pytest.skip("numpy not available")


class TestOperatorPropertiesTyped:
    """Tests for typed get/set operations on operator properties."""

    def test_property_type_coercion_int(self, lf, lfs_types):
        """Int properties should be coerced correctly."""
        received = {}

        class IntTypeOp(lfs_types.Operator):
            lf_label = "Int Type"
            int_val: int = 42

            def execute(self, context):
                received["type"] = type(self.int_val).__name__
                received["value"] = self.int_val
                return {"FINISHED"}

        lf.register_class(IntTypeOp)
        try:
            lf.ops.invoke(IntTypeOp._class_id())
            assert received["type"] == "int"
            assert received["value"] == 42
        finally:
            lf.unregister_class(IntTypeOp)

    def test_property_type_coercion_float(self, lf, lfs_types):
        """Float properties should be coerced correctly."""
        received = {}

        class FloatTypeOp(lfs_types.Operator):
            lf_label = "Float Type"
            float_val: float = 3.14

            def execute(self, context):
                received["type"] = type(self.float_val).__name__
                received["value"] = self.float_val
                return {"FINISHED"}

        lf.register_class(FloatTypeOp)
        try:
            lf.ops.invoke(FloatTypeOp._class_id())
            assert received["type"] == "float"
            assert abs(received["value"] - 3.14) < 0.001
        finally:
            lf.unregister_class(FloatTypeOp)

    def test_property_default_values(self, lf, lfs_types):
        """Properties should have default values accessible."""
        received = {}

        class DefaultOp(lfs_types.Operator):
            lf_label = "Default"
            value: int = 100

            def execute(self, context):
                received["value"] = self.value
                return {"FINISHED"}

        lf.register_class(DefaultOp)
        try:
            lf.ops.invoke(DefaultOp._class_id())
            assert received["value"] == 100
        finally:
            lf.unregister_class(DefaultOp)

    def test_property_override_via_kwargs(self, lf, lfs_types):
        """Properties should be overridable via kwargs."""
        received = {}

        class OverrideOp(lfs_types.Operator):
            lf_label = "Override"
            value: int = 100

            def execute(self, context):
                received["value"] = self.value
                return {"FINISHED"}

        lf.register_class(OverrideOp)
        try:
            lf.ops.invoke(OverrideOp._class_id(), value=999)
            assert received["value"] == 999
        finally:
            lf.unregister_class(OverrideOp)


class TestOperatorPropertiesIsolation:
    """Tests for property isolation between operator calls."""

    def test_keyword_overrides_do_not_persist(self, lf, lfs_types):
        """Invocation kwargs should apply only to the call that supplied them."""
        received = []

        class FreshStartOp(lfs_types.Operator):
            lf_label = "Fresh Start"

            def execute(self, context):
                received.append(getattr(self, "value", "default"))
                return {"FINISHED"}

        lf.register_class(FreshStartOp)
        try:
            lf.ops.invoke(FreshStartOp._class_id(), value="override")
            lf.ops.invoke(FreshStartOp._class_id())

            assert received == ["override", "default"]
        finally:
            lf.unregister_class(FreshStartOp)

    def test_keyword_overrides_do_not_persist_after_failure(self, lf, lfs_types):
        received = []

        class FailingOverrideOp(lfs_types.Operator):
            lf_label = "Failing Override"

            def execute(self, context):
                value = getattr(self, "value", "default")
                received.append(value)
                if value == "fail":
                    raise RuntimeError("intentional failure")
                return {"FINISHED"}

        lf.register_class(FailingOverrideOp)
        try:
            lf.ops.invoke(FailingOverrideOp._class_id(), value="fail")
            lf.ops.invoke(FailingOverrideOp._class_id())

            assert received == ["fail", "default"]
        finally:
            lf.unregister_class(FailingOverrideOp)

    def test_multiple_operators_independent(self, lf, lfs_types):
        """Different operators should have independent property storage."""
        results = {}

        class Op1(lfs_types.Operator):
            lf_label = "Op1"
            shared_name: str = "op1"

            def execute(self, context):
                results["op1"] = self.shared_name
                return {"FINISHED"}

        class Op2(lfs_types.Operator):
            lf_label = "Op2"
            shared_name: str = "op2"

            def execute(self, context):
                results["op2"] = self.shared_name
                return {"FINISHED"}

        lf.register_class(Op1)
        lf.register_class(Op2)
        try:
            lf.ops.invoke(Op1._class_id())
            lf.ops.invoke(Op2._class_id())

            assert results["op1"] == "op1"
            assert results["op2"] == "op2"
        finally:
            lf.unregister_class(Op1)
            lf.unregister_class(Op2)


class TestOperatorPropertiesWithTensors:
    """Tests for tensor properties in operators."""

    def test_tensor_property_passthrough(self, lf, lfs_types, numpy):
        """Tensors should pass through correctly."""
        received = {}

        class TensorOp(lfs_types.Operator):
            lf_label = "Tensor"

            def execute(self, context):
                t = getattr(self, "input_tensor", None)
                if t is not None:
                    received["shape"] = t.shape
                    received["sum"] = float(t.sum().item())
                return {"FINISHED"}

        arr = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32)
        tensor = lf.Tensor.from_numpy(arr)

        lf.register_class(TensorOp)
        try:
            lf.ops.invoke(TensorOp._class_id(), input_tensor=tensor)
            assert received["shape"] == (3,)
            assert received["sum"] == 6.0
        finally:
            lf.unregister_class(TensorOp)

    def test_tensor_modification_in_operator(self, lf, lfs_types, numpy):
        """Operators should be able to modify tensor passed via kwargs."""

        class ModifyTensorOp(lfs_types.Operator):
            lf_label = "Modify Tensor"

            def execute(self, context):
                t = getattr(self, "tensor", None)
                if t is not None:
                    t *= 2.0
                return {"FINISHED"}

        lf.register_class(ModifyTensorOp)
        try:
            t = lf.Tensor.ones([10], dtype="float32", device="cpu")
            assert t.sum().item() == 10.0

            lf.ops.invoke(ModifyTensorOp._class_id(), tensor=t)
            assert t.sum().item() == 20.0
        finally:
            lf.unregister_class(ModifyTensorOp)
