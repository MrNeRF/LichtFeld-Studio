# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for operator hot-reload safety and lambda capture lifetime."""

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


class TestOperatorHotReload:
    """Tests for operator lifecycle and hot-reload behavior."""

    def test_operator_survives_class_redefine(self, lf, lfs_types):
        """Redefining an operator class should update the registration."""
        call_log = []

        class TestOp(lfs_types.Operator):
            lf_label = "Hot Reload V1"

            def execute(self, context):
                call_log.append("v1")
                return {"FINISHED"}

        lf.register_class(TestOp)
        try:
            lf.ops.invoke(TestOp._class_id())
            assert call_log == ["v1"]
        finally:
            lf.unregister_class(TestOp)

        # Redefine with same structure
        class TestOp2(lfs_types.Operator):  # noqa: F811
            lf_label = "Hot Reload V2"

            def execute(self, context):
                call_log.append("v2")
                return {"FINISHED"}

        # Re-register new version
        lf.register_class(TestOp2)
        try:
            lf.ops.invoke(TestOp2._class_id())
            assert call_log == ["v1", "v2"]
        finally:
            lf.unregister_class(TestOp2)

    def test_reregister_after_unregister(self, lf, lfs_types):
        """Re-registering after unregister should work."""
        call_log = []

        class ReregisterOp(lfs_types.Operator):
            lf_label = "Reregister"

            def execute(self, context):
                call_log.append("called")
                return {"FINISHED"}

        lf.register_class(ReregisterOp)
        lf.ops.invoke(ReregisterOp._class_id())

        lf.unregister_class(ReregisterOp)

        # Re-register
        lf.register_class(ReregisterOp)
        lf.ops.invoke(ReregisterOp._class_id())

        assert call_log == ["called", "called"]
        lf.unregister_class(ReregisterOp)


class TestLambdaCaptureLifetime:
    """Tests for lambda capture safety in operator callbacks."""

    def test_captured_closure_survives(self, lf, lfs_types):
        """Closures captured in operators should survive."""
        captured_value = {"count": 0}

        class ClosureOp(lfs_types.Operator):
            lf_label = "Closure"

            def execute(self, context):
                captured_value["count"] += 1
                return {"FINISHED"}

        lf.register_class(ClosureOp)
        try:
            lf.ops.invoke(ClosureOp._class_id())
            lf.ops.invoke(ClosureOp._class_id())

            assert captured_value["count"] == 2
        finally:
            lf.unregister_class(ClosureOp)

    def test_module_level_state_preserved(self, lf, lfs_types):
        """Module-level state should be accessible across calls."""
        import types

        test_module = types.ModuleType("test_state_module")
        test_module.state = {"initialized": False, "call_count": 0}

        class StateOp(lfs_types.Operator):
            lf_label = "State"

            def execute(self, context):
                if not test_module.state["initialized"]:
                    test_module.state["initialized"] = True
                test_module.state["call_count"] += 1
                return {"FINISHED"}

        lf.register_class(StateOp)
        try:
            lf.ops.invoke(StateOp._class_id())
            assert test_module.state["initialized"]
            assert test_module.state["call_count"] == 1

            lf.ops.invoke(StateOp._class_id())
            assert test_module.state["call_count"] == 2
        finally:
            lf.unregister_class(StateOp)


class TestOperatorInstanceLifecycle:
    """Tests for operator instance creation and cleanup."""

    def test_multiple_invokes_work(self, lf, lfs_types):
        """Multiple invocations should work correctly."""
        instance_ids = []

        class InstanceOp(lfs_types.Operator):
            lf_label = "Instance"

            def execute(self, context):
                instance_ids.append(id(self))
                return {"FINISHED"}

        lf.register_class(InstanceOp)
        try:
            lf.ops.invoke(InstanceOp._class_id())
            lf.ops.invoke(InstanceOp._class_id())
            lf.ops.invoke(InstanceOp._class_id())

            # Should have been called 3 times
            assert len(instance_ids) == 3
        finally:
            lf.unregister_class(InstanceOp)

    def test_exception_in_execute_doesnt_crash(self, lf, lfs_types):
        """Exceptions should be handled gracefully."""
        call_count = [0]

        class ExceptionOp(lfs_types.Operator):
            lf_label = "Exception"

            def execute(self, context):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise ValueError("Intentional test error")
                return {"FINISHED"}

        lf.register_class(ExceptionOp)
        try:
            # First call raises
            result1 = lf.ops.invoke(ExceptionOp._class_id())
            assert not result1.finished

            # Second call should work fine
            result2 = lf.ops.invoke(ExceptionOp._class_id())
            assert result2.finished
        finally:
            lf.unregister_class(ExceptionOp)
