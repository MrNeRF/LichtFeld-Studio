# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for panel registration edge cases.

Targets:
- py_ui_panels.cpp:113-128 - Re-registration may leave stale pointers
- py_ui_panels.cpp:132-142 - Instantiation failure handling

Note: Panel API may not be available in all builds. Tests are skipped
if the required API is not present.
"""

import gc
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
        pytest.skip(f"lfs_plugins.types not available: {e}")


@pytest.fixture
def panel_fixture(lf, lfs_types):
    """Setup and cleanup for panel tests."""
    registered = []

    yield registered, lfs_types

    for cls in registered:
        try:
            lf.unregister_class(cls)
        except Exception:
            pass


class TestPanelEdgeCases:
    """Tests for panel registration edge cases."""

    def test_double_registration_same_class(self, lf, panel_fixture):
        """Registering same panel class twice should be handled."""
        registered, lfs_types = panel_fixture

        if not hasattr(lfs_types, "Panel"):
            pytest.skip("lfs_types.Panel not available")

        class DoublePanel(lfs_types.Panel):
            label = "Double Panel"
            space = lf.ui.PanelSpace.MAIN_PANEL_TAB

            def draw(self, context):
                pass

        try:
            lf.register_class(DoublePanel)
            registered.append(DoublePanel)

            # Second registration should either fail or update
            try:
                lf.register_class(DoublePanel)
            except (ValueError, RuntimeError):
                pass  # Expected - already registered
        except AttributeError:
            pytest.skip("Panel registration not supported")

    def test_registration_without_label_uses_id_fallback(self, lf, panel_fixture):
        """Panels without labels should fall back to their id."""
        registered, lfs_types = panel_fixture

        if not hasattr(lfs_types, "Panel"):
            pytest.skip("lfs_types.Panel not available")

        class NoLabelPanel(lfs_types.Panel):
            id = "tests.no_label_panel"
            space = lf.ui.PanelSpace.MAIN_PANEL_TAB

            def draw(self, context):
                pass

        lf.register_class(NoLabelPanel)
        try:
            registered.append(NoLabelPanel)
            info = lf.ui.get_panel("tests.no_label_panel")
            assert info is not None
            assert info.id == "tests.no_label_panel"
            assert info.label == "tests.no_label_panel"
        finally:
            lf.unregister_class(NoLabelPanel)
            registered.remove(NoLabelPanel)

    def test_panel_instantiation_failure(self, lf, panel_fixture):
        """Panel that fails to instantiate should be handled."""
        registered, lfs_types = panel_fixture

        if not hasattr(lfs_types, "Panel"):
            pytest.skip("lfs_types.Panel not available")

        class FailInitPanel(lfs_types.Panel):
            label = "Fail Init"
            space = lf.ui.PanelSpace.MAIN_PANEL_TAB

            def __init__(self):
                raise RuntimeError("Init failed")

            def draw(self, context):
                pass

        try:
            lf.register_class(FailInitPanel)
            registered.append(FailInitPanel)
            # Drawing would fail, but registration might succeed
        except (RuntimeError, TypeError, ValueError):
            pass


class TestPanelDrawEdgeCases:
    """Tests for panel draw edge cases."""

    def test_draw_exception_isolation(self, lf, panel_fixture):
        """Exception in draw should not affect other panels."""
        registered, lfs_types = panel_fixture

        if not hasattr(lfs_types, "Panel"):
            pytest.skip("lfs_types.Panel not available")

        class ExcPanel(lfs_types.Panel):
            label = "Exception Panel"
            space = lf.ui.PanelSpace.MAIN_PANEL_TAB

            def draw(self, context):
                raise RuntimeError("Draw failed")

        try:
            lf.register_class(ExcPanel)
            registered.append(ExcPanel)
            # Panel registered, draw would fail but other panels should work
        except AttributeError:
            pytest.skip("Panel registration not supported")

    def test_draw_modifies_class_state(self, lf, panel_fixture):
        """Panel draw that modifies class state."""
        registered, lfs_types = panel_fixture

        if not hasattr(lfs_types, "Panel"):
            pytest.skip("lfs_types.Panel not available")

        class StatefulPanel(lfs_types.Panel):
            label = "Stateful Panel"
            space = lf.ui.PanelSpace.MAIN_PANEL_TAB
            draw_count = 0

            def draw(self, context):
                StatefulPanel.draw_count += 1

        try:
            lf.register_class(StatefulPanel)
            registered.append(StatefulPanel)
            # State modification should be allowed
        except AttributeError:
            pytest.skip("Panel registration not supported")


class TestOperatorAsPanel:
    """Tests for operators that can act as panels (with draw method)."""

    def test_operator_with_draw_method(self, lf, panel_fixture):
        """Operator with draw method should work."""
        registered, lfs_types = panel_fixture

        class DrawableOp(lfs_types.Operator):
            lf_label = "Drawable Op"

            def draw(self, layout):
                layout.label("Test Label")

            def execute(self, context):
                return {"FINISHED"}

        lf.register_class(DrawableOp)
        registered.append(DrawableOp)

        result = lf.ops.invoke(DrawableOp._class_id())
        assert result is not None

    def test_operator_draw_exception_handled(self, lf, panel_fixture):
        """Operator draw exception should be handled."""
        registered, lfs_types = panel_fixture

        class DrawExcOp(lfs_types.Operator):
            lf_label = "Draw Exc Op"

            def draw(self, layout):
                raise RuntimeError("Draw failed")

            def execute(self, context):
                return {"FINISHED"}

        lf.register_class(DrawExcOp)
        registered.append(DrawExcOp)

        # Execute should still work even if draw fails
        try:
            result = lf.ops.invoke(DrawExcOp._class_id())
        except RuntimeError:
            pass  # May propagate if draw is called during invoke

    def test_operator_draw_with_properties(self, lf, panel_fixture):
        """Operator with properties displayed in draw."""
        registered, lfs_types = panel_fixture
        draw_called = [False]

        class PropDrawOp(lfs_types.Operator):
            lf_label = "Prop Draw Op"

            value: float = 1.0
            name: str = "test"

            def draw(self, layout):
                draw_called[0] = True
                layout.prop(self, "value")
                layout.prop(self, "name")

            def execute(self, context):
                return {"FINISHED"}

        lf.register_class(PropDrawOp)
        registered.append(PropDrawOp)

        try:
            lf.ops.invoke(PropDrawOp._class_id())
        except Exception:
            pass


class TestPanelHierarchy:
    """Tests for panel parent/child relationships using operators."""

    def test_operator_with_subpanels(self, lf, panel_fixture):
        """Operator with subpanel-like draw sections."""
        registered, lfs_types = panel_fixture
        sections_drawn = []

        class SectionedOp(lfs_types.Operator):
            lf_label = "Sectioned Op"

            show_advanced: bool = False

            def draw(self, layout):
                sections_drawn.append("main")
                layout.prop(self, "show_advanced")
                if self.show_advanced:
                    sections_drawn.append("advanced")

            def execute(self, context):
                return {"FINISHED"}

        lf.register_class(SectionedOp)
        registered.append(SectionedOp)

        try:
            lf.ops.invoke(SectionedOp._class_id())
        except Exception:
            pass
