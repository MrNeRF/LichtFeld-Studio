# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Focused contracts for the sub-layout composition API."""

import pytest


CONTAINER_FACTORIES = {
    "row": lambda layout: layout.row(),
    "column": lambda layout: layout.column(),
    "split": lambda layout: layout.split(0.3),
    "box": lambda layout: layout.box(),
    "grid_flow": lambda layout: layout.grid_flow(columns=3),
}

DRAWING_METHODS = (
    "label",
    "button",
    "button_styled",
    "prop",
    "checkbox",
    "slider_float",
    "slider_int",
    "drag_float",
    "drag_int",
    "input_text",
    "combo",
    "separator",
    "spacing",
    "heading",
    "collapsing_header",
    "tree_node",
    "tree_pop",
    "progress_bar",
    "text_colored",
    "text_wrapped",
    "prop_enum",
)

EXPLICIT_METHODS = (
    "begin_table",
    "table_setup_column",
    "end_table",
    "table_next_row",
    "table_next_column",
    "input_int_formatted",
    "input_float",
    "input_int",
    "radio_button",
    "small_button",
    "selectable",
    "color_edit3",
    "text_disabled",
    "image",
    "image_button",
    "input_text_with_hint",
    "input_text_enter",
    "listbox",
    "same_line",
    "begin_disabled",
    "end_disabled",
    "push_item_width",
    "pop_item_width",
    "set_tooltip",
    "is_item_hovered",
    "is_item_clicked",
    "push_id",
    "pop_id",
    "begin_child",
    "end_child",
    "begin_context_menu",
    "end_context_menu",
    "menu_item",
    "begin_menu",
    "end_menu",
    "get_content_region_avail",
    "table_headers_row",
)


@pytest.mark.parametrize("factory_name", CONTAINER_FACTORIES)
def test_layout_containers_return_context_managed_sublayouts(lf, factory_name):
    sublayout = CONTAINER_FACTORIES[factory_name](lf.ui.UILayout())

    assert isinstance(sublayout, lf.ui.SubLayout)
    assert hasattr(sublayout, "__enter__")
    assert hasattr(sublayout, "__exit__")


@pytest.mark.parametrize(
    ("attribute", "default", "updated"),
    (("enabled", True, False), ("active", True, False), ("alert", False, True)),
)
def test_sublayout_state_round_trips(lf, attribute, default, updated):
    sublayout = lf.ui.UILayout().row()

    assert getattr(sublayout, attribute) is default
    setattr(sublayout, attribute, updated)
    assert getattr(sublayout, attribute) is updated


def test_sublayout_exposes_complete_drawing_surface(lf):
    sublayout = lf.ui.UILayout().row()

    for name in (*DRAWING_METHODS, *EXPLICIT_METHODS, *CONTAINER_FACTORIES):
        assert callable(getattr(sublayout, name)), f"SubLayout missing callable: {name}"

    assert callable(lf.ui.UILayout().prop_enum)


def test_sublayouts_nest_to_arbitrary_depth(lf):
    nested = lf.ui.UILayout().column().row().box().split(0.5)

    assert isinstance(nested, lf.ui.SubLayout)


def test_sublayout_delegates_parent_methods_and_rejects_unknown_names(lf):
    sublayout = lf.ui.UILayout().row()

    assert callable(sublayout.begin_group)
    assert callable(sublayout.end_group)
    with pytest.raises(AttributeError):
        _ = sublayout.nonexistent_method_xyz


def test_removed_layout_context_is_not_exposed(lf):
    assert not hasattr(lf.ui, "LayoutContext")
