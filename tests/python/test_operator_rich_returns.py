# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for operator rich returns and data passing.

These tests verify that operators can return structured data via
OperatorReturnValue and that the data is accessible from Python.
"""

import sys
from pathlib import Path

import pytest


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


class TestOperatorReturnValueBinding:
    """Tests for OperatorReturnValue Python binding."""

    def test_return_value_has_status_attribute(self, lf, lfs_types):
        """OperatorReturnValue should have a status attribute."""

        class SimpleOp(lfs_types.Operator):
            lf_label = "Simple Test Op"

            def execute(self, context):
                return {"FINISHED"}

        lf.register_class(SimpleOp)
        try:
            result = lf.ops.invoke(SimpleOp._class_id())
            assert hasattr(result, "status")
            assert result.status == "FINISHED"
        finally:
            lf.unregister_class(SimpleOp)

    def test_return_value_finished_property(self, lf, lfs_types):
        """OperatorReturnValue should have finished property."""

        class FinishedOp(lfs_types.Operator):
            lf_label = "Finished Op"

            def execute(self, context):
                return {"FINISHED"}

        lf.register_class(FinishedOp)
        try:
            result = lf.ops.invoke(FinishedOp._class_id())
            assert result.finished is True
            assert result.cancelled is False
        finally:
            lf.unregister_class(FinishedOp)

    def test_return_value_cancelled_property(self, lf, lfs_types):
        """OperatorReturnValue should have cancelled property."""

        class CancelledOp(lfs_types.Operator):
            lf_label = "Cancelled Op"

            def execute(self, context):
                return {"CANCELLED"}

        lf.register_class(CancelledOp)
        try:
            result = lf.ops.invoke(CancelledOp._class_id())
            assert result.cancelled is True
            assert result.finished is False
        finally:
            lf.unregister_class(CancelledOp)

    def test_return_value_bool_conversion(self, lf, lfs_types):
        """OperatorReturnValue should be truthy when finished."""

        class TruthyOp(lfs_types.Operator):
            lf_label = "Truthy Op"

            def execute(self, context):
                return {"FINISHED"}

        class FalsyOp(lfs_types.Operator):
            lf_label = "Falsy Op"

            def execute(self, context):
                return {"CANCELLED"}

        lf.register_class(TruthyOp)
        lf.register_class(FalsyOp)
        try:
            truthy_result = lf.ops.invoke(TruthyOp._class_id())
            falsy_result = lf.ops.invoke(FalsyOp._class_id())

            # __bool__ should return finished status
            assert bool(truthy_result) is True
            assert bool(falsy_result) is False

            # Can use in if statements
            if truthy_result:
                pass  # Expected
            else:
                pytest.fail("Expected truthy result to be truthy")

            if falsy_result:
                pytest.fail("Expected falsy result to be falsy")
        finally:
            lf.unregister_class(TruthyOp)
            lf.unregister_class(FalsyOp)


class TestRichReturnsWithData:
    """Tests for operators returning data in dict format."""

    def test_return_dict_with_status_and_data(self, lf, lfs_types):
        """Operator can return dict with status and extra data."""

        class DataOp(lfs_types.Operator):
            lf_label = "Data Op"

            def execute(self, context):
                return {"status": "FINISHED", "count": 42, "name": "test"}

        lf.register_class(DataOp)
        try:
            result = lf.ops.invoke(DataOp._class_id())
            assert result.finished is True
            assert result.status == "FINISHED"
            assert result.data["count"] == 42
            assert result.data["name"] == "test"
        finally:
            lf.unregister_class(DataOp)

    def test_return_data_accessible_via_getattr(self, lf, lfs_types):
        """Returned data should be accessible via attribute access."""

        class AttrOp(lfs_types.Operator):
            lf_label = "Attr Op"

            def execute(self, context):
                return {"status": "FINISHED", "result_value": 123, "message": "ok"}

        lf.register_class(AttrOp)
        try:
            result = lf.ops.invoke(AttrOp._class_id())
            # Should be able to access data as attributes
            assert result.result_value == 123
            assert result.message == "ok"
        finally:
            lf.unregister_class(AttrOp)

    def test_return_tensor_in_data(self, lf, lfs_types, numpy):
        """Operator can return tensors in the data dict."""

        class TensorOp(lfs_types.Operator):
            lf_label = "Tensor Op"

            def execute(self, context):
                import lichtfeld

                t = lichtfeld.Tensor.zeros([10, 3], dtype="float32", device="cpu")
                return {"status": "FINISHED", "tensor": t, "shape": (10, 3)}

        lf.register_class(TensorOp)
        try:
            result = lf.ops.invoke(TensorOp._class_id())
            assert result.finished
            assert result.tensor is not None
            assert tuple(result.tensor.shape) == (10, 3)
            assert result.shape == (10, 3)
        finally:
            lf.unregister_class(TensorOp)

    def test_return_multiple_tensors(self, lf, lfs_types, numpy):
        """Operator can return multiple tensors."""

        class MultiTensorOp(lfs_types.Operator):
            lf_label = "Multi Tensor Op"

            def execute(self, context):
                import lichtfeld

                positions = lichtfeld.Tensor.zeros(
                    [100, 3], dtype="float32", device="cpu"
                )
                indices = lichtfeld.Tensor.zeros([50], dtype="int32", device="cpu")
                return {
                    "status": "FINISHED",
                    "positions": positions,
                    "indices": indices,
                    "count": 50,
                }

        lf.register_class(MultiTensorOp)
        try:
            result = lf.ops.invoke(MultiTensorOp._class_id())
            assert result.finished
            assert tuple(result.positions.shape) == (100, 3)
            assert tuple(result.indices.shape) == (50,)
            assert result.count == 50
        finally:
            lf.unregister_class(MultiTensorOp)

    def test_return_list_and_dict_in_data(self, lf, lfs_types):
        """Operator can return complex types like lists and dicts."""

        class ComplexOp(lfs_types.Operator):
            lf_label = "Complex Op"

            def execute(self, context):
                return {
                    "status": "FINISHED",
                    "items": [1, 2, 3, 4, 5],
                    "mapping": {"a": 1, "b": 2},
                    "nested": {"list": [1, 2], "value": 42},
                }

        lf.register_class(ComplexOp)
        try:
            result = lf.ops.invoke(ComplexOp._class_id())
            assert result.finished
            assert result.items == [1, 2, 3, 4, 5]
            assert result.mapping == {"a": 1, "b": 2}
            assert result.nested["list"] == [1, 2]
            assert result.nested["value"] == 42
        finally:
            lf.unregister_class(ComplexOp)

    def test_getattr_raises_for_nonexistent_key(self, lf, lfs_types):
        """Accessing non-existent data key should raise AttributeError."""

        class MinimalOp(lfs_types.Operator):
            lf_label = "Minimal Op"

            def execute(self, context):
                return {"status": "FINISHED", "exists": True}

        lf.register_class(MinimalOp)
        try:
            result = lf.ops.invoke(MinimalOp._class_id())
            assert result.exists is True

            with pytest.raises(AttributeError):
                _ = result.nonexistent_key
        finally:
            lf.unregister_class(MinimalOp)


class TestOperatorDataPassing:
    """Tests for passing data to operators via kwargs."""

    def test_pass_simple_kwargs(self, lf, lfs_types):
        """Operator should receive kwargs as attributes."""
        received = {}

        class KwargsOp(lfs_types.Operator):
            lf_label = "Kwargs Op"

            def execute(self, context):
                received["x"] = getattr(self, "x", None)
                received["y"] = getattr(self, "y", None)
                return {"FINISHED"}

        lf.register_class(KwargsOp)
        try:
            lf.ops.invoke(KwargsOp._class_id(), x=10, y=20)
            assert received["x"] == 10
            assert received["y"] == 20
        finally:
            lf.unregister_class(KwargsOp)

    def test_pass_tensor_as_kwarg(self, lf, lfs_types, numpy):
        """Operator should be able to receive tensor via kwargs."""
        received = {}

        class TensorKwargOp(lfs_types.Operator):
            lf_label = "Tensor Kwarg Op"

            def execute(self, context):
                received["tensor"] = getattr(self, "input_tensor", None)
                return {"FINISHED"}

        lf.register_class(TensorKwargOp)
        try:
            input_t = lf.Tensor.ones([5, 3], dtype="float32", device="cpu")
            lf.ops.invoke(TensorKwargOp._class_id(), input_tensor=input_t)

            assert received["tensor"] is not None
            assert tuple(received["tensor"].shape) == (5, 3)
        finally:
            lf.unregister_class(TensorKwargOp)

    def test_roundtrip_tensor_kwarg_and_return(self, lf, lfs_types, numpy):
        """Tensor passed as kwarg can be processed and returned."""

        class ProcessOp(lfs_types.Operator):
            lf_label = "Process Op"

            def execute(self, context):
                input_t = getattr(self, "data", None)
                if input_t is None:
                    return {"CANCELLED"}

                # Process: double the values
                output = input_t * 2.0
                return {"status": "FINISHED", "result": output}

        lf.register_class(ProcessOp)
        try:
            input_data = lf.Tensor.ones([10], dtype="float32", device="cpu")
            result = lf.ops.invoke(ProcessOp._class_id(), data=input_data)

            assert result.finished
            result_np = result.result.numpy()
            assert all(v == 2.0 for v in result_np)
        finally:
            lf.unregister_class(ProcessOp)


class TestLegacyReturnFormat:
    """Tests for backward compatibility with set return format."""

    def test_set_return_still_works(self, lf, lfs_types):
        """Old-style set return {'FINISHED'} should still work."""

        class LegacyOp(lfs_types.Operator):
            lf_label = "Legacy Op"

            def execute(self, context):
                return {"FINISHED"}

        lf.register_class(LegacyOp)
        try:
            result = lf.ops.invoke(LegacyOp._class_id())
            assert result.finished
            assert result.status == "FINISHED"
            assert len(result.data) == 0
        finally:
            lf.unregister_class(LegacyOp)

    def test_mixed_operators(self, lf, lfs_types):
        """Mix of legacy and new-style operators should work."""

        class LegacyStyleOp(lfs_types.Operator):
            lf_label = "Legacy Style"

            def execute(self, context):
                return {"FINISHED"}

        class NewStyleOp(lfs_types.Operator):
            lf_label = "New Style"

            def execute(self, context):
                return {"status": "FINISHED", "data_key": 123}

        lf.register_class(LegacyStyleOp)
        lf.register_class(NewStyleOp)
        try:
            legacy_result = lf.ops.invoke(LegacyStyleOp._class_id())
            new_result = lf.ops.invoke(NewStyleOp._class_id())

            assert legacy_result.finished
            assert new_result.finished
            assert new_result.data_key == 123
        finally:
            lf.unregister_class(LegacyStyleOp)
            lf.unregister_class(NewStyleOp)


class TestReturnStatusValues:
    """Tests for all possible return status values."""

    def test_finished_status(self, lf, lfs_types):
        """Test FINISHED status."""

        class FinishedStatusOp(lfs_types.Operator):
            lf_label = "Finished Status"

            def execute(self, context):
                return {"status": "FINISHED"}

        lf.register_class(FinishedStatusOp)
        try:
            result = lf.ops.invoke(FinishedStatusOp._class_id())
            assert result.status == "FINISHED"
            assert result.finished
            assert not result.cancelled
        finally:
            lf.unregister_class(FinishedStatusOp)

    def test_cancelled_status(self, lf, lfs_types):
        """Test CANCELLED status."""

        class CancelledStatusOp(lfs_types.Operator):
            lf_label = "Cancelled Status"

            def execute(self, context):
                return {"status": "CANCELLED"}

        lf.register_class(CancelledStatusOp)
        try:
            result = lf.ops.invoke(CancelledStatusOp._class_id())
            assert result.status == "CANCELLED"
            assert result.cancelled
            assert not result.finished
        finally:
            lf.unregister_class(CancelledStatusOp)


class TestDataCleanup:
    """Tests for data cleanup between operator invocations."""

    def test_data_not_persisted_between_calls(self, lf, lfs_types):
        """Data from one invocation should not leak to next."""
        call_count = [0]

        class CountingOp(lfs_types.Operator):
            lf_label = "Counting Op"

            def execute(self, context):
                call_count[0] += 1
                return {"status": "FINISHED", "call": call_count[0]}

        lf.register_class(CountingOp)
        try:
            result1 = lf.ops.invoke(CountingOp._class_id())
            result2 = lf.ops.invoke(CountingOp._class_id())

            assert result1.call == 1
            assert result2.call == 2

            # Each result should have its own data
            assert result1.data["call"] == 1
            assert result2.data["call"] == 2
        finally:
            lf.unregister_class(CountingOp)
