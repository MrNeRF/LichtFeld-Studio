/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_ui.hpp"
#include "rml_im_mode_layout.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace lfs::python {

    void register_rml_im_mode_layout(nb::module_& m) {
        nb::class_<RmlImModeLayout>(m, "RmlUILayout")
            // Text
            .def("label", &RmlImModeLayout::label, nb::arg("text"))
            .def("label_centered", &RmlImModeLayout::label_centered, nb::arg("text"))
            .def("heading", &RmlImModeLayout::heading, nb::arg("text"))
            .def("text_colored", &RmlImModeLayout::text_colored, nb::arg("text"), nb::arg("color"))
            .def("text_colored_centered", &RmlImModeLayout::text_colored_centered, nb::arg("text"), nb::arg("color"))
            .def("text_selectable", &RmlImModeLayout::text_selectable, nb::arg("text"), nb::arg("height") = 0.0f)
            .def("text_wrapped", &RmlImModeLayout::text_wrapped, nb::arg("text"))
            .def("text_disabled", &RmlImModeLayout::text_disabled, nb::arg("text"))
            .def("bullet_text", &RmlImModeLayout::bullet_text, nb::arg("text"))
            // Buttons
            .def("button", &RmlImModeLayout::button, nb::arg("label"), nb::arg("size") = std::make_tuple(0.0f, 0.0f))
            .def("button_callback", &RmlImModeLayout::button_callback, nb::arg("label"),
                 nb::arg("callback") = nb::none(), nb::arg("size") = std::make_tuple(0.0f, 0.0f))
            .def("small_button", &RmlImModeLayout::small_button, nb::arg("label"))
            .def("checkbox", &RmlImModeLayout::checkbox, nb::arg("label"), nb::arg("value"))
            .def("radio_button", &RmlImModeLayout::radio_button, nb::arg("label"), nb::arg("current"), nb::arg("value"))
            // Sliders
            .def("slider_float", &RmlImModeLayout::slider_float, nb::arg("label"), nb::arg("value"), nb::arg("min"), nb::arg("max"))
            .def("slider_int", &RmlImModeLayout::slider_int, nb::arg("label"), nb::arg("value"), nb::arg("min"), nb::arg("max"))
            .def("slider_float2", &RmlImModeLayout::slider_float2, nb::arg("label"), nb::arg("value"), nb::arg("min"), nb::arg("max"))
            .def("slider_float3", &RmlImModeLayout::slider_float3, nb::arg("label"), nb::arg("value"), nb::arg("min"), nb::arg("max"))
            // Drags
            .def("drag_float", &RmlImModeLayout::drag_float, nb::arg("label"), nb::arg("value"),
                 nb::arg("speed") = 1.0f, nb::arg("min") = 0.0f, nb::arg("max") = 0.0f)
            .def("drag_int", &RmlImModeLayout::drag_int, nb::arg("label"), nb::arg("value"),
                 nb::arg("speed") = 1.0f, nb::arg("min") = 0, nb::arg("max") = 0)
            // Input
            .def("input_text", &RmlImModeLayout::input_text, nb::arg("label"), nb::arg("value"))
            .def("input_text_with_hint", &RmlImModeLayout::input_text_with_hint,
                 nb::arg("label"), nb::arg("hint"), nb::arg("value"))
            .def("input_text_enter", &RmlImModeLayout::input_text_enter, nb::arg("label"), nb::arg("value"))
            .def("input_float", &RmlImModeLayout::input_float, nb::arg("label"), nb::arg("value"),
                 nb::arg("step") = 0.0f, nb::arg("step_fast") = 0.0f, nb::arg("format") = "%.3f")
            .def("input_int", &RmlImModeLayout::input_int, nb::arg("label"), nb::arg("value"),
                 nb::arg("step") = 1, nb::arg("step_fast") = 100)
            .def("input_int_formatted", &RmlImModeLayout::input_int_formatted, nb::arg("label"), nb::arg("value"),
                 nb::arg("step") = 0, nb::arg("step_fast") = 0)
            .def("stepper_float", &RmlImModeLayout::stepper_float, nb::arg("label"), nb::arg("value"),
                 nb::arg("steps") = std::vector<float>{1.0f, 0.1f, 0.01f})
            .def("path_input", &RmlImModeLayout::path_input, nb::arg("label"), nb::arg("value"),
                 nb::arg("folder_mode") = true, nb::arg("dialog_title") = "",
                 "Draw a path input, returns (changed, path). dialog_title is accepted for compatibility and currently ignored.")
            // Color
            .def("color_edit3", &RmlImModeLayout::color_edit3, nb::arg("label"), nb::arg("color"))
            .def("color_edit4", &RmlImModeLayout::color_edit4, nb::arg("label"), nb::arg("color"))
            .def("color_picker3", &RmlImModeLayout::color_picker3, nb::arg("label"), nb::arg("color"))
            .def("color_button", &RmlImModeLayout::color_button, nb::arg("label"), nb::arg("color"),
                 nb::arg("size") = std::make_tuple(0.0f, 0.0f))
            // Selection
            .def("combo", &RmlImModeLayout::combo, nb::arg("label"), nb::arg("current_idx"), nb::arg("items"))
            .def("listbox", &RmlImModeLayout::listbox, nb::arg("label"), nb::arg("current_idx"),
                 nb::arg("items"), nb::arg("height_items") = -1)
            // Layout
            .def("separator", &RmlImModeLayout::separator)
            .def("spacing", &RmlImModeLayout::spacing)
            .def("same_line", &RmlImModeLayout::same_line, nb::arg("offset") = 0.0f, nb::arg("spacing") = -1.0f)
            .def("new_line", &RmlImModeLayout::new_line)
            .def("indent", &RmlImModeLayout::indent, nb::arg("width") = 0.0f)
            .def("unindent", &RmlImModeLayout::unindent, nb::arg("width") = 0.0f)
            .def("set_next_item_width", &RmlImModeLayout::set_next_item_width, nb::arg("width"))
            // Grouping
            .def("begin_group", &RmlImModeLayout::begin_group)
            .def("end_group", &RmlImModeLayout::end_group)
            .def("collapsing_header", &RmlImModeLayout::collapsing_header, nb::arg("label"), nb::arg("default_open") = false)
            .def("tree_node", &RmlImModeLayout::tree_node, nb::arg("label"))
            .def("tree_node_ex", &RmlImModeLayout::tree_node_ex, nb::arg("label"), nb::arg("flags") = "")
            .def("set_next_item_open", &RmlImModeLayout::set_next_item_open, nb::arg("is_open"))
            .def("tree_pop", &RmlImModeLayout::tree_pop)
            // Tables
            .def("begin_table", &RmlImModeLayout::begin_table, nb::arg("id"), nb::arg("columns"))
            .def("table_setup_column", &RmlImModeLayout::table_setup_column, nb::arg("label"), nb::arg("width") = 0.0f)
            .def("end_table", &RmlImModeLayout::end_table)
            .def("table_next_row", &RmlImModeLayout::table_next_row)
            .def("table_next_column", &RmlImModeLayout::table_next_column)
            .def("table_set_column_index", &RmlImModeLayout::table_set_column_index, nb::arg("column"))
            .def("table_headers_row", &RmlImModeLayout::table_headers_row)
            .def("table_set_bg_color", &RmlImModeLayout::table_set_bg_color, nb::arg("target"), nb::arg("color"))
            // Styled buttons
            .def("button_styled", &RmlImModeLayout::button_styled, nb::arg("label"), nb::arg("style"),
                 nb::arg("size") = std::make_tuple(0.0f, 0.0f))
            // Item width
            .def("push_item_width", &RmlImModeLayout::push_item_width, nb::arg("width"))
            .def("pop_item_width", &RmlImModeLayout::pop_item_width)
            // Plots
            .def("plot_lines", &RmlImModeLayout::plot_lines, nb::arg("label"), nb::arg("values"),
                 nb::arg("scale_min") = 0.0f, nb::arg("scale_max") = 0.0f,
                 nb::arg("size") = std::make_tuple(0.0f, 0.0f))
            // Selectable
            .def("selectable", &RmlImModeLayout::selectable, nb::arg("label"),
                 nb::arg("selected") = false, nb::arg("height") = 0.0f)
            // Context menus
            .def("begin_context_menu", &RmlImModeLayout::begin_context_menu, nb::arg("id") = "")
            .def("end_context_menu", &RmlImModeLayout::end_context_menu)
            .def("begin_popup", &RmlImModeLayout::begin_popup, nb::arg("id"))
            .def("open_popup", &RmlImModeLayout::open_popup, nb::arg("id"))
            .def("end_popup", &RmlImModeLayout::end_popup)
            .def("menu_item", &RmlImModeLayout::menu_item, nb::arg("label"), nb::arg("enabled") = true, nb::arg("selected") = false)
            .def("begin_menu", &RmlImModeLayout::begin_menu, nb::arg("label"))
            .def("end_menu", &RmlImModeLayout::end_menu)
            // Focus control
            .def("set_keyboard_focus_here", &RmlImModeLayout::set_keyboard_focus_here)
            .def("is_window_focused", &RmlImModeLayout::is_window_focused)
            .def("is_window_hovered", &RmlImModeLayout::is_window_hovered)
            .def("capture_keyboard_from_app", &RmlImModeLayout::capture_keyboard_from_app, nb::arg("capture") = true)
            .def("capture_mouse_from_app", &RmlImModeLayout::capture_mouse_from_app, nb::arg("capture") = true)
            // Scrolling
            .def("set_scroll_here_y", &RmlImModeLayout::set_scroll_here_y, nb::arg("center_y_ratio") = 0.5f)
            // Cursor / content
            .def("get_cursor_screen_pos", &RmlImModeLayout::get_cursor_screen_pos)
            .def("get_mouse_pos", &RmlImModeLayout::get_mouse_pos)
            .def("get_window_pos", &RmlImModeLayout::get_window_pos)
            .def("get_window_width", &RmlImModeLayout::get_window_width)
            .def("get_text_line_height", &RmlImModeLayout::get_text_line_height)
            // Modal popups
            .def("begin_popup_modal", &RmlImModeLayout::begin_popup_modal, nb::arg("title"))
            .def("end_popup_modal", &RmlImModeLayout::end_popup_modal)
            .def("close_current_popup", &RmlImModeLayout::close_current_popup)
            .def("set_next_window_pos_center", &RmlImModeLayout::set_next_window_pos_center)
            .def("set_next_window_pos_viewport_center", &RmlImModeLayout::set_next_window_pos_viewport_center, nb::arg("always") = false)
            .def("set_next_window_focus", &RmlImModeLayout::set_next_window_focus)
            .def("push_modal_style", &RmlImModeLayout::push_modal_style)
            .def("pop_modal_style", &RmlImModeLayout::pop_modal_style)
            // Content region
            .def("get_content_region_avail", &RmlImModeLayout::get_content_region_avail)
            .def("get_cursor_pos", &RmlImModeLayout::get_cursor_pos)
            .def("set_cursor_pos_x", &RmlImModeLayout::set_cursor_pos_x, nb::arg("x"))
            .def("calc_text_size", &RmlImModeLayout::calc_text_size, nb::arg("text"))
            // Disabled state
            .def("begin_disabled", &RmlImModeLayout::begin_disabled, nb::arg("disabled") = true)
            .def("end_disabled", &RmlImModeLayout::end_disabled)
            // Images
            .def("image", &RmlImModeLayout::image, nb::arg("texture_id"), nb::arg("size"),
                 nb::arg("tint") = nb::none())
            .def("image_uv", &RmlImModeLayout::image_uv, nb::arg("texture_id"), nb::arg("size"),
                 nb::arg("uv0"), nb::arg("uv1"), nb::arg("tint") = nb::none())
            .def("image_button", &RmlImModeLayout::image_button, nb::arg("id"), nb::arg("texture_id"),
                 nb::arg("size"), nb::arg("tint") = nb::none())
            .def("toolbar_button", &RmlImModeLayout::toolbar_button, nb::arg("id"), nb::arg("texture_id"),
                 nb::arg("size"), nb::arg("selected") = false, nb::arg("disabled") = false,
                 nb::arg("tooltip") = "")
            .def("invisible_button", &RmlImModeLayout::invisible_button, nb::arg("id"), nb::arg("size"))
            .def("set_cursor_pos", &RmlImModeLayout::set_cursor_pos, nb::arg("pos"))
            // Child windows
            .def("begin_child", &RmlImModeLayout::begin_child, nb::arg("id"), nb::arg("size"), nb::arg("border") = false)
            .def("end_child", &RmlImModeLayout::end_child)
            // Menu bar
            .def("begin_menu_bar", &RmlImModeLayout::begin_menu_bar)
            .def("end_menu_bar", &RmlImModeLayout::end_menu_bar)
            .def("menu_item_toggle", &RmlImModeLayout::menu_item_toggle, nb::arg("label"), nb::arg("shortcut"), nb::arg("selected"))
            .def("menu_item_shortcut", &RmlImModeLayout::menu_item_shortcut, nb::arg("label"), nb::arg("shortcut"), nb::arg("enabled") = true)
            .def("push_id", &RmlImModeLayout::push_id, nb::arg("id"))
            .def("push_id_int", &RmlImModeLayout::push_id_int, nb::arg("id"))
            .def("pop_id", &RmlImModeLayout::pop_id)
            // Window
            .def("begin_window", &RmlImModeLayout::begin_window, nb::arg("title"), nb::arg("flags") = 0)
            .def("begin_window_closable", &RmlImModeLayout::begin_window_closable, nb::arg("title"), nb::arg("flags") = 0)
            .def("end_window", &RmlImModeLayout::end_window)
            .def("push_window_style", &RmlImModeLayout::push_window_style)
            .def("pop_window_style", &RmlImModeLayout::pop_window_style)
            // Window positioning
            .def("set_next_window_pos", &RmlImModeLayout::set_next_window_pos, nb::arg("pos"), nb::arg("first_use") = false)
            .def("set_next_window_size", &RmlImModeLayout::set_next_window_size, nb::arg("size"), nb::arg("first_use") = false)
            .def("set_next_window_pos_centered", &RmlImModeLayout::set_next_window_pos_centered, nb::arg("first_use") = false)
            .def("set_next_window_bg_alpha", &RmlImModeLayout::set_next_window_bg_alpha, nb::arg("alpha"))
            .def("get_viewport_pos", &RmlImModeLayout::get_viewport_pos)
            .def("get_viewport_size", &RmlImModeLayout::get_viewport_size)
            .def("get_dpi_scale", &RmlImModeLayout::get_dpi_scale)
            .def("set_mouse_cursor_hand", &RmlImModeLayout::set_mouse_cursor_hand)
            // Style control
            .def("push_style_var", &RmlImModeLayout::push_style_var_float, nb::arg("var"), nb::arg("value"))
            .def("push_style_var_vec2", &RmlImModeLayout::push_style_var_vec2, nb::arg("var"), nb::arg("value"))
            .def("pop_style_var", &RmlImModeLayout::pop_style_var, nb::arg("count") = 1)
            .def("push_style_color", &RmlImModeLayout::push_style_color, nb::arg("col"), nb::arg("color"))
            .def("pop_style_color", &RmlImModeLayout::pop_style_color, nb::arg("count") = 1)
            // Property binding
            .def("prop", &RmlImModeLayout::prop, nb::arg("data"), nb::arg("prop_id"), nb::arg("text") = nb::none())
            .def("row", &RmlImModeLayout::row)
            .def("column", &RmlImModeLayout::column)
            .def("split", &RmlImModeLayout::split, nb::arg("factor") = 0.5f)
            .def("box", &RmlImModeLayout::box)
            .def("grid_flow", &RmlImModeLayout::grid_flow, nb::arg("columns") = 0, nb::arg("even_columns") = true, nb::arg("even_rows") = true)
            .def("prop_enum", &RmlImModeLayout::prop_enum, nb::arg("data"), nb::arg("prop_id"), nb::arg("value"), nb::arg("text") = "")
            .def("operator_", &RmlImModeLayout::operator_, nb::arg("operator_id"), nb::arg("text") = "", nb::arg("icon") = "")
            .def("prop_search", &RmlImModeLayout::prop_search, nb::arg("data"), nb::arg("prop_id"), nb::arg("search_data"), nb::arg("search_prop"), nb::arg("text") = "")
            .def("template_list", &RmlImModeLayout::template_list, nb::arg("list_type_id"), nb::arg("list_id"), nb::arg("data"), nb::arg("prop_id"), nb::arg("active_data"), nb::arg("active_prop"), nb::arg("rows") = 5)
            .def("menu", &RmlImModeLayout::menu, nb::arg("menu_id"), nb::arg("text") = "", nb::arg("icon") = "")
            .def("popover", &RmlImModeLayout::popover, nb::arg("panel_id"), nb::arg("text") = "", nb::arg("icon") = "")
            // Drawing
            .def("draw_circle", &RmlImModeLayout::draw_circle, nb::arg("x"), nb::arg("y"), nb::arg("radius"), nb::arg("color"), nb::arg("segments") = 32, nb::arg("thickness") = 1.0f)
            .def("draw_circle_filled", &RmlImModeLayout::draw_circle_filled, nb::arg("x"), nb::arg("y"), nb::arg("radius"), nb::arg("color"), nb::arg("segments") = 32)
            .def("draw_rect", &RmlImModeLayout::draw_rect, nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"), nb::arg("thickness") = 1.0f)
            .def("draw_rect_filled", &RmlImModeLayout::draw_rect_filled, nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"), nb::arg("background") = false)
            .def("draw_rect_rounded", &RmlImModeLayout::draw_rect_rounded, nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"), nb::arg("rounding"), nb::arg("thickness") = 1.0f, nb::arg("background") = false)
            .def("draw_rect_rounded_filled", &RmlImModeLayout::draw_rect_rounded_filled, nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"), nb::arg("rounding"), nb::arg("background") = false)
            .def("draw_triangle_filled", &RmlImModeLayout::draw_triangle_filled, nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("x2"), nb::arg("y2"), nb::arg("color"), nb::arg("background") = false)
            .def("draw_line", &RmlImModeLayout::draw_line, nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"), nb::arg("thickness") = 1.0f)
            .def("draw_polyline", &RmlImModeLayout::draw_polyline, nb::arg("points"), nb::arg("color"), nb::arg("closed") = false, nb::arg("thickness") = 1.0f)
            .def("draw_poly_filled", &RmlImModeLayout::draw_poly_filled, nb::arg("points"), nb::arg("color"))
            .def("draw_text", &RmlImModeLayout::draw_text, nb::arg("x"), nb::arg("y"), nb::arg("text"), nb::arg("color"), nb::arg("background") = false)
            .def("draw_window_rect_filled", &RmlImModeLayout::draw_window_rect_filled, nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"))
            .def("draw_window_rect", &RmlImModeLayout::draw_window_rect, nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"), nb::arg("thickness") = 1.0f)
            .def("draw_window_rect_rounded", &RmlImModeLayout::draw_window_rect_rounded, nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"), nb::arg("rounding"), nb::arg("thickness") = 1.0f)
            .def("draw_window_rect_rounded_filled", &RmlImModeLayout::draw_window_rect_rounded_filled, nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"), nb::arg("rounding"))
            .def("draw_window_line", &RmlImModeLayout::draw_window_line, nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("color"), nb::arg("thickness") = 1.0f)
            .def("draw_window_text", &RmlImModeLayout::draw_window_text, nb::arg("x"), nb::arg("y"), nb::arg("text"), nb::arg("color"))
            .def("draw_window_triangle_filled", &RmlImModeLayout::draw_window_triangle_filled, nb::arg("x0"), nb::arg("y0"), nb::arg("x1"), nb::arg("y1"), nb::arg("x2"), nb::arg("y2"), nb::arg("color"))
            .def("crf_curve_preview", &RmlImModeLayout::crf_curve_preview, nb::arg("label"), nb::arg("gamma"), nb::arg("toe"), nb::arg("shoulder"), nb::arg("gamma_r") = 0.0f, nb::arg("gamma_g") = 0.0f, nb::arg("gamma_b") = 0.0f)
            .def("chromaticity_diagram", &RmlImModeLayout::chromaticity_diagram, nb::arg("label"), nb::arg("red_x"), nb::arg("red_y"), nb::arg("green_x"), nb::arg("green_y"), nb::arg("blue_x"), nb::arg("blue_y"), nb::arg("neutral_x"), nb::arg("neutral_y"), nb::arg("range") = 0.5f)
            // Progress
            .def("progress_bar", &RmlImModeLayout::progress_bar, nb::arg("fraction"), nb::arg("overlay") = "", nb::arg("width") = 0.0f, nb::arg("height") = 0.0f)
            .def("set_tooltip", &RmlImModeLayout::set_tooltip, nb::arg("text"))
            .def("is_item_hovered", &RmlImModeLayout::is_item_hovered)
            .def("is_item_clicked", &RmlImModeLayout::is_item_clicked, nb::arg("button") = 0)
            .def("is_item_active", &RmlImModeLayout::is_item_active)
            .def("is_mouse_double_clicked", &RmlImModeLayout::is_mouse_double_clicked, nb::arg("button") = 0)
            .def("is_mouse_dragging", &RmlImModeLayout::is_mouse_dragging, nb::arg("button") = 0)
            .def("get_mouse_wheel", &RmlImModeLayout::get_mouse_wheel)
            .def("get_mouse_delta", &RmlImModeLayout::get_mouse_delta)
            // Drag-drop
            .def("begin_drag_drop_source", &RmlImModeLayout::begin_drag_drop_source)
            .def("set_drag_drop_payload", &RmlImModeLayout::set_drag_drop_payload, nb::arg("type"), nb::arg("data"))
            .def("end_drag_drop_source", &RmlImModeLayout::end_drag_drop_source)
            .def("begin_drag_drop_target", &RmlImModeLayout::begin_drag_drop_target)
            .def("accept_drag_drop_payload", &RmlImModeLayout::accept_drag_drop_payload, nb::arg("type"))
            .def("end_drag_drop_target", &RmlImModeLayout::end_drag_drop_target);

        nb::class_<RmlSubLayout>(m, "RmlSubLayout")
            .def("__enter__", &RmlSubLayout::enter, nb::rv_policy::reference)
            .def("__exit__", [](RmlSubLayout& s, nb::args) { s.exit(); return false; })
            .def_rw("enabled", &RmlSubLayout::enabled)
            .def_rw("active", &RmlSubLayout::active)
            .def_rw("alert", &RmlSubLayout::alert)
            .def_rw("scale_x", &RmlSubLayout::scale_x)
            .def_rw("scale_y", &RmlSubLayout::scale_y)
            .def("row", &RmlSubLayout::row)
            .def("column", &RmlSubLayout::column)
            .def("split", &RmlSubLayout::split, nb::arg("factor") = 0.5f)
            .def("box", &RmlSubLayout::box)
            .def("grid_flow", &RmlSubLayout::grid_flow, nb::arg("columns") = 0, nb::arg("even_columns") = true, nb::arg("even_rows") = true)
            .def("label", &RmlSubLayout::label, nb::arg("text"))
            .def("label_centered", &RmlSubLayout::label_centered, nb::arg("text"))
            .def("heading", &RmlSubLayout::heading, nb::arg("text"))
            .def("text_colored", &RmlSubLayout::text_colored, nb::arg("text"), nb::arg("color"))
            .def("text_colored_centered", &RmlSubLayout::text_colored_centered, nb::arg("text"), nb::arg("color"))
            .def("text_selectable", &RmlSubLayout::text_selectable, nb::arg("text"), nb::arg("height") = 0.0f)
            .def("text_wrapped", &RmlSubLayout::text_wrapped, nb::arg("text"))
            .def("text_disabled", &RmlSubLayout::text_disabled, nb::arg("text"))
            .def("bullet_text", &RmlSubLayout::bullet_text, nb::arg("text"))
            .def("button", &RmlSubLayout::button, nb::arg("label"), nb::arg("size") = std::make_tuple(0.0f, 0.0f))
            .def("button_callback", &RmlSubLayout::button_callback, nb::arg("label"), nb::arg("callback") = nb::none(), nb::arg("size") = std::make_tuple(0.0f, 0.0f))
            .def("small_button", &RmlSubLayout::small_button, nb::arg("label"))
            .def("button_styled", &RmlSubLayout::button_styled, nb::arg("label"), nb::arg("style"), nb::arg("size") = std::make_tuple(0.0f, 0.0f))
            .def("checkbox", &RmlSubLayout::checkbox, nb::arg("label"), nb::arg("value"))
            .def("radio_button", &RmlSubLayout::radio_button, nb::arg("label"), nb::arg("current"), nb::arg("value"))
            .def("slider_float", &RmlSubLayout::slider_float, nb::arg("label"), nb::arg("value"), nb::arg("min"), nb::arg("max"))
            .def("slider_int", &RmlSubLayout::slider_int, nb::arg("label"), nb::arg("value"), nb::arg("min"), nb::arg("max"))
            .def("drag_float", &RmlSubLayout::drag_float, nb::arg("label"), nb::arg("value"), nb::arg("speed") = 1.0f, nb::arg("min") = 0.0f, nb::arg("max") = 0.0f)
            .def("drag_int", &RmlSubLayout::drag_int, nb::arg("label"), nb::arg("value"), nb::arg("speed") = 1.0f, nb::arg("min") = 0, nb::arg("max") = 0)
            .def("input_text", &RmlSubLayout::input_text, nb::arg("label"), nb::arg("value"))
            .def("input_text_with_hint", &RmlSubLayout::input_text_with_hint, nb::arg("label"), nb::arg("hint"), nb::arg("value"))
            .def("input_text_enter", &RmlSubLayout::input_text_enter, nb::arg("label"), nb::arg("value"))
            .def("input_float", &RmlSubLayout::input_float, nb::arg("label"), nb::arg("value"), nb::arg("step") = 0.0f, nb::arg("step_fast") = 0.0f, nb::arg("format") = "%.3f")
            .def("input_int", &RmlSubLayout::input_int, nb::arg("label"), nb::arg("value"), nb::arg("step") = 1, nb::arg("step_fast") = 100)
            .def("input_int_formatted", &RmlSubLayout::input_int_formatted, nb::arg("label"), nb::arg("value"), nb::arg("step") = 0, nb::arg("step_fast") = 0)
            .def("stepper_float", &RmlSubLayout::stepper_float, nb::arg("label"), nb::arg("value"), nb::arg("steps") = std::vector<float>{1.0f, 0.1f, 0.01f})
            .def("color_edit3", &RmlSubLayout::color_edit3, nb::arg("label"), nb::arg("color"))
            .def("combo", &RmlSubLayout::combo, nb::arg("label"), nb::arg("current_idx"), nb::arg("items"))
            .def("listbox", &RmlSubLayout::listbox, nb::arg("label"), nb::arg("current_idx"), nb::arg("items"), nb::arg("height_items") = -1)
            .def("selectable", &RmlSubLayout::selectable, nb::arg("label"), nb::arg("selected") = false, nb::arg("height") = 0.0f)
            .def("separator", &RmlSubLayout::separator)
            .def("spacing", &RmlSubLayout::spacing)
            .def("same_line", &RmlSubLayout::same_line, nb::arg("offset") = 0.0f, nb::arg("spacing") = -1.0f)
            .def("new_line", &RmlSubLayout::new_line)
            .def("collapsing_header", &RmlSubLayout::collapsing_header, nb::arg("label"), nb::arg("default_open") = false)
            .def("tree_node", &RmlSubLayout::tree_node, nb::arg("label"))
            .def("tree_pop", &RmlSubLayout::tree_pop)
            .def("begin_table", &RmlSubLayout::begin_table, nb::arg("id"), nb::arg("columns"))
            .def("table_setup_column", &RmlSubLayout::table_setup_column, nb::arg("label"), nb::arg("width") = 0.0f)
            .def("table_next_row", &RmlSubLayout::table_next_row)
            .def("table_next_column", &RmlSubLayout::table_next_column)
            .def("table_headers_row", &RmlSubLayout::table_headers_row)
            .def("end_table", &RmlSubLayout::end_table)
            .def("progress_bar", &RmlSubLayout::progress_bar, nb::arg("fraction"), nb::arg("overlay") = "", nb::arg("width") = 0.0f, nb::arg("height") = 0.0f)
            .def("push_item_width", &RmlSubLayout::push_item_width, nb::arg("width"))
            .def("pop_item_width", &RmlSubLayout::pop_item_width)
            .def("set_tooltip", &RmlSubLayout::set_tooltip, nb::arg("text"))
            .def("is_item_hovered", &RmlSubLayout::is_item_hovered)
            .def("is_item_clicked", &RmlSubLayout::is_item_clicked, nb::arg("button") = 0)
            .def("begin_disabled", &RmlSubLayout::begin_disabled, nb::arg("disabled") = true)
            .def("end_disabled", &RmlSubLayout::end_disabled)
            .def("push_id", &RmlSubLayout::push_id, nb::arg("id"))
            .def("pop_id", &RmlSubLayout::pop_id)
            .def("begin_child", &RmlSubLayout::begin_child, nb::arg("id"), nb::arg("size"), nb::arg("border") = false)
            .def("end_child", &RmlSubLayout::end_child)
            .def("image", &RmlSubLayout::image, nb::arg("texture_id"), nb::arg("size"), nb::arg("tint") = nb::none())
            .def("image_button", &RmlSubLayout::image_button, nb::arg("id"), nb::arg("texture_id"), nb::arg("size"), nb::arg("tint") = nb::none())
            .def("begin_context_menu", &RmlSubLayout::begin_context_menu, nb::arg("id") = "")
            .def("end_context_menu", &RmlSubLayout::end_context_menu)
            .def("menu_item", &RmlSubLayout::menu_item, nb::arg("label"), nb::arg("enabled") = true, nb::arg("selected") = false)
            .def("begin_menu", &RmlSubLayout::begin_menu, nb::arg("label"))
            .def("end_menu", &RmlSubLayout::end_menu)
            .def("get_content_region_avail", &RmlSubLayout::get_content_region_avail)
            .def("prop", &RmlSubLayout::prop, nb::arg("data"), nb::arg("prop_id"), nb::arg("text") = nb::none())
            .def("prop_enum", &RmlSubLayout::prop_enum, nb::arg("data"), nb::arg("prop_id"), nb::arg("value"), nb::arg("text") = "")
            .def("operator_", &RmlSubLayout::operator_, nb::arg("operator_id"), nb::arg("text") = "", nb::arg("icon") = "");
    }

} // namespace lfs::python
