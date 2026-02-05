# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the sub-layout composition API."""

import pytest


class TestSubLayoutClass:
    """Tests for SubLayout type availability and protocol."""

    def test_sublayout_exists(self, lf):
        assert hasattr(lf.ui, "SubLayout")

    def test_uilayout_returns_sublayout(self, lf):
        layout = lf.ui.UILayout()
        sub = layout.row()
        assert type(sub).__name__ == "SubLayout"

    def test_column_returns_sublayout(self, lf):
        layout = lf.ui.UILayout()
        sub = layout.column()
        assert type(sub).__name__ == "SubLayout"

    def test_split_returns_sublayout(self, lf):
        layout = lf.ui.UILayout()
        sub = layout.split(0.3)
        assert type(sub).__name__ == "SubLayout"

    def test_box_returns_sublayout(self, lf):
        layout = lf.ui.UILayout()
        sub = layout.box()
        assert type(sub).__name__ == "SubLayout"

    def test_grid_flow_returns_sublayout(self, lf):
        layout = lf.ui.UILayout()
        sub = layout.grid_flow(columns=3)
        assert type(sub).__name__ == "SubLayout"


class TestSubLayoutContextManager:
    """Tests for context manager protocol."""

    def test_row_context_manager(self, lf):
        layout = lf.ui.UILayout()
        sub = layout.row()
        assert hasattr(sub, "__enter__")
        assert hasattr(sub, "__exit__")

    def test_column_context_manager(self, lf):
        layout = lf.ui.UILayout()
        sub = layout.column()
        assert hasattr(sub, "__enter__")
        assert hasattr(sub, "__exit__")

    def test_split_context_manager(self, lf):
        layout = lf.ui.UILayout()
        sub = layout.split()
        assert hasattr(sub, "__enter__")
        assert hasattr(sub, "__exit__")

    def test_box_context_manager(self, lf):
        layout = lf.ui.UILayout()
        sub = layout.box()
        assert hasattr(sub, "__enter__")
        assert hasattr(sub, "__exit__")

    def test_grid_flow_context_manager(self, lf):
        layout = lf.ui.UILayout()
        sub = layout.grid_flow()
        assert hasattr(sub, "__enter__")
        assert hasattr(sub, "__exit__")


class TestSubLayoutStateProperties:
    """Tests for state properties."""

    def test_enabled_default(self, lf):
        sub = lf.ui.UILayout().row()
        assert sub.enabled is True

    def test_enabled_set(self, lf):
        sub = lf.ui.UILayout().row()
        sub.enabled = False
        assert sub.enabled is False

    def test_active_default(self, lf):
        sub = lf.ui.UILayout().row()
        assert sub.active is True

    def test_active_set(self, lf):
        sub = lf.ui.UILayout().row()
        sub.active = False
        assert sub.active is False

    def test_alert_default(self, lf):
        sub = lf.ui.UILayout().row()
        assert sub.alert is False

    def test_alert_set(self, lf):
        sub = lf.ui.UILayout().row()
        sub.alert = True
        assert sub.alert is True


class TestSubLayoutDrawingMethods:
    """Tests for drawing method availability."""

    EXPECTED_METHODS = [
        "label", "button", "button_styled", "prop", "checkbox",
        "slider_float", "slider_int", "drag_float", "drag_int",
        "input_text", "combo", "separator", "spacing", "heading",
        "collapsing_header", "tree_node", "tree_pop", "progress_bar",
        "text_colored", "text_wrapped", "prop_enum",
    ]

    def test_has_drawing_methods(self, lf):
        sub = lf.ui.UILayout().row()
        for method_name in self.EXPECTED_METHODS:
            assert hasattr(sub, method_name), f"SubLayout missing method: {method_name}"

    def test_has_sublayout_creation(self, lf):
        sub = lf.ui.UILayout().row()
        for method_name in ["row", "column", "split", "box", "grid_flow"]:
            assert hasattr(sub, method_name), f"SubLayout missing: {method_name}"


class TestSubLayoutNesting:
    """Tests for sub-layout nesting."""

    def test_row_in_column(self, lf):
        layout = lf.ui.UILayout()
        col = layout.column()
        row = col.row()
        assert type(row).__name__ == "SubLayout"

    def test_box_in_split(self, lf):
        layout = lf.ui.UILayout()
        sp = layout.split(0.3)
        bx = sp.box()
        assert type(bx).__name__ == "SubLayout"

    def test_deep_nesting(self, lf):
        layout = lf.ui.UILayout()
        col = layout.column()
        row = col.row()
        bx = row.box()
        sp = bx.split(0.5)
        assert type(sp).__name__ == "SubLayout"


class TestSubLayoutGetattr:
    """Tests for __getattr__ delegation."""

    def test_getattr_delegates_to_parent(self, lf):
        sub = lf.ui.UILayout().row()
        assert hasattr(sub, "same_line")
        assert hasattr(sub, "begin_group")
        assert hasattr(sub, "end_group")

    def test_getattr_raises_for_unknown(self, lf):
        sub = lf.ui.UILayout().row()
        with pytest.raises(AttributeError):
            sub.nonexistent_method_xyz

    def test_getattr_returns_callable(self, lf):
        sub = lf.ui.UILayout().row()
        method = sub.same_line
        assert callable(method)


class TestUILayoutPropEnum:
    """Tests for prop_enum on UILayout."""

    def test_prop_enum_exists(self, lf):
        layout = lf.ui.UILayout()
        assert hasattr(layout, "prop_enum")
        assert callable(layout.prop_enum)


class TestEmptyContainers:
    """Tests for empty container edge cases."""

    def test_empty_row_sublayout(self, lf):
        sub = lf.ui.UILayout().row()
        assert sub.enabled is True

    def test_empty_column_sublayout(self, lf):
        sub = lf.ui.UILayout().column()
        assert sub.active is True


class TestNoOldAPI:
    """Verify old dead API is removed."""

    def test_no_layout_context_class(self, lf):
        assert not hasattr(lf.ui, "LayoutContext")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
