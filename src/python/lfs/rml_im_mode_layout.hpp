/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/EventListener.h>

#include <nanobind/nanobind.h>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nb = nanobind;

namespace lfs::python {

    enum class SlotType : uint8_t {
        Label,
        Heading,
        TextColored,
        TextDisabled,
        TextWrapped,
        BulletText,
        Button,
        ButtonStyled,
        SmallButton,
        Checkbox,
        RadioButton,
        SliderFloat,
        SliderInt,
        DragFloat,
        DragInt,
        InputText,
        InputFloat,
        InputInt,
        Combo,
        Selectable,
        CollapsHeader,
        Separator,
        Spacing,
        ProgressBar,
        DisabledGroup,
        Line,
    };

    struct SlotEventState {
        bool clicked = false;
        bool changed = false;
        bool bool_value = false;
        int int_value = 0;
        float float_value = 0.0f;
        std::string string_value;
        bool open = true;
    };

    struct Slot {
        SlotType type;
        std::string key;
        Rml::Element* element = nullptr;
        SlotEventState events;
    };

    struct ContainerLevel {
        Rml::Element* parent = nullptr;
        std::deque<Slot> slots;
        int cursor = 0;
    };

    struct MouseState {
        float pos_x = 0.0f;
        float pos_y = 0.0f;
        float delta_x = 0.0f;
        float delta_y = 0.0f;
        float wheel = 0.0f;
        bool double_clicked = false;
        bool dragging = false;
    };

    struct TableState {
        int num_columns = 0;
        int current_column = -1;
        std::vector<float> column_widths;
        Rml::Element* table_element = nullptr;
        Rml::Element* current_row = nullptr;
        Rml::Element* current_cell = nullptr;
    };

    class SlotEventListener : public Rml::EventListener {
    public:
        explicit SlotEventListener(SlotEventState* state) : state_(state) {}
        void ProcessEvent(Rml::Event& event) override;
        void OnDetach(Rml::Element*) override { delete this; }

    private:
        SlotEventState* state_;
    };

    class RmlSubLayout;

    class RmlImModeLayout {
        friend class RmlSubLayout;

    public:
        RmlImModeLayout() = default;
        RmlImModeLayout(const RmlImModeLayout&) = delete;
        RmlImModeLayout& operator=(const RmlImModeLayout&) = delete;
        RmlImModeLayout(RmlImModeLayout&&) = default;
        RmlImModeLayout& operator=(RmlImModeLayout&&) = default;

        void begin_frame(Rml::ElementDocument* doc, const MouseState& mouse = {});
        void end_frame();

        // --- Text ---
        void label(const std::string& text);
        void label_centered(const std::string& text);
        void heading(const std::string& text);
        void text_colored(const std::string& text, nb::object color);
        void text_colored_centered(const std::string& text, nb::object color);
        void text_selectable(const std::string& text, float height = 0.0f);
        void text_wrapped(const std::string& text);
        void text_disabled(const std::string& text);
        void bullet_text(const std::string& text);

        // --- Buttons ---
        bool button(const std::string& label, std::tuple<float, float> size = {0, 0});
        bool button_callback(const std::string& label, nb::object callback,
                             std::tuple<float, float> size = {0, 0});
        bool small_button(const std::string& label);
        bool button_styled(const std::string& label, const std::string& style,
                           std::tuple<float, float> size = {0, 0});
        std::tuple<bool, bool> checkbox(const std::string& label, bool value);
        std::tuple<bool, int> radio_button(const std::string& label, int current, int value);

        // --- Sliders ---
        std::tuple<bool, float> slider_float(const std::string& label, float value, float min, float max);
        std::tuple<bool, int> slider_int(const std::string& label, int value, int min, int max);
        std::tuple<bool, std::tuple<float, float>> slider_float2(const std::string& label,
                                                                 std::tuple<float, float> value, float min, float max);
        std::tuple<bool, std::tuple<float, float, float>> slider_float3(const std::string& label,
                                                                        std::tuple<float, float, float> value, float min, float max);

        // --- Drags ---
        std::tuple<bool, float> drag_float(const std::string& label, float value,
                                           float speed = 1.0f, float min = 0.0f, float max = 0.0f);
        std::tuple<bool, int> drag_int(const std::string& label, int value,
                                       float speed = 1.0f, int min = 0, int max = 0);

        // --- Input ---
        std::tuple<bool, std::string> input_text(const std::string& label, const std::string& value);
        std::tuple<bool, std::string> input_text_with_hint(const std::string& label,
                                                           const std::string& hint, const std::string& value);
        std::tuple<bool, std::string> input_text_enter(const std::string& label, const std::string& value);
        std::tuple<bool, float> input_float(const std::string& label, float value,
                                            float step = 0.0f, float step_fast = 0.0f,
                                            const std::string& format = "%.3f");
        std::tuple<bool, int> input_int(const std::string& label, int value,
                                        int step = 1, int step_fast = 100);
        std::tuple<bool, int> input_int_formatted(const std::string& label, int value,
                                                  int step = 0, int step_fast = 0);
        std::tuple<bool, float> stepper_float(const std::string& label, float value,
                                              const std::vector<float>& steps = {1.0f, 0.1f, 0.01f});
        std::tuple<bool, std::string> path_input(const std::string& label, const std::string& value,
                                                 bool folder_mode = true,
                                                 const std::string& dialog_title = "");

        // --- Color ---
        std::tuple<bool, std::tuple<float, float, float>> color_edit3(const std::string& label,
                                                                      std::tuple<float, float, float> color);
        std::tuple<bool, std::tuple<float, float, float, float>> color_edit4(const std::string& label,
                                                                             std::tuple<float, float, float, float> color);
        std::tuple<bool, std::tuple<float, float, float>> color_picker3(const std::string& label,
                                                                        std::tuple<float, float, float> color);
        bool color_button(const std::string& label, nb::object color,
                          std::tuple<float, float> size = {0, 0});

        // --- Selection ---
        std::tuple<bool, int> combo(const std::string& label, int current_idx,
                                    const std::vector<std::string>& items);
        std::tuple<bool, int> listbox(const std::string& label, int current_idx,
                                      const std::vector<std::string>& items, int height_items = -1);
        bool selectable(const std::string& label, bool selected = false, float height = 0.0f);

        // --- Layout ---
        void separator();
        void spacing();
        void same_line(float offset = 0.0f, float spacing = -1.0f);
        void new_line();
        void indent(float width = 0.0f);
        void unindent(float width = 0.0f);
        void set_next_item_width(float width);
        void begin_group();
        void end_group();

        // --- Grouping ---
        bool collapsing_header(const std::string& label, bool default_open = false);
        bool tree_node(const std::string& label);
        bool tree_node_ex(const std::string& label, const std::string& flags = "");
        void set_next_item_open(bool is_open);
        void tree_pop();

        // --- Tables ---
        bool begin_table(const std::string& id, int columns);
        void table_setup_column(const std::string& label, float width = 0.0f);
        void end_table();
        void table_next_row();
        void table_next_column();
        bool table_set_column_index(int column);
        void table_headers_row();
        void table_set_bg_color(int target, nb::object color);

        // --- Disabled state ---
        void begin_disabled(bool disabled = true);
        void end_disabled();

        // --- Progress ---
        void progress_bar(float fraction, const std::string& overlay = "",
                          float width = 0.0f, float height = 0.0f);

        // --- ID stack ---
        void push_id(const std::string& id);
        void push_id_int(int id);
        void pop_id();

        // --- Style ---
        void push_style_var_float(const std::string& var, float value);
        void push_style_var_vec2(const std::string& var, std::tuple<float, float> value);
        void pop_style_var(int count = 1);
        void push_style_color(const std::string& col, nb::object color);
        void pop_style_color(int count = 1);
        void push_item_width(float width);
        void pop_item_width();

        // --- Tooltip ---
        void set_tooltip(const std::string& text);

        // --- Item state ---
        bool is_item_hovered();
        bool is_item_clicked(int button = 0);
        bool is_item_active();

        // --- Mouse ---
        bool is_mouse_double_clicked(int button = 0);
        bool is_mouse_dragging(int button = 0);
        float get_mouse_wheel();
        std::tuple<float, float> get_mouse_delta();
        std::tuple<float, float> get_mouse_pos() const;
        void set_mouse_cursor_hand();

        // --- Window queries ---
        std::tuple<float, float> get_content_region_avail();
        float get_window_width() const;
        float get_text_line_height() const;
        std::tuple<float, float> get_cursor_pos();
        std::tuple<float, float> get_cursor_screen_pos() const;
        void set_cursor_pos(std::tuple<float, float> pos);
        void set_cursor_pos_x(float x);
        std::tuple<float, float> calc_text_size(const std::string& text);
        std::tuple<float, float> get_window_pos() const;

        // --- Viewport ---
        std::tuple<float, float> get_viewport_pos();
        std::tuple<float, float> get_viewport_size();
        float get_dpi_scale();

        // --- Window management (no-op for panel context) ---
        bool begin_window(const std::string& title, int flags = 0);
        std::tuple<bool, bool> begin_window_closable(const std::string& title, int flags = 0);
        void end_window();
        void push_window_style();
        void pop_window_style();
        void set_next_window_pos(std::tuple<float, float> pos, bool first_use = false);
        void set_next_window_size(std::tuple<float, float> size, bool first_use = false);
        void set_next_window_pos_centered(bool first_use = false);
        void set_next_window_bg_alpha(float alpha);
        void set_next_window_pos_center();
        void set_next_window_pos_viewport_center(bool always = false);
        void set_next_window_focus();

        // --- Focus ---
        void set_keyboard_focus_here();
        bool is_window_focused() const;
        bool is_window_hovered() const;
        void capture_keyboard_from_app(bool capture = true);
        void capture_mouse_from_app(bool capture = true);

        // --- Scrolling ---
        void set_scroll_here_y(float center_y_ratio = 0.5f);
        bool begin_child(const std::string& id, std::tuple<float, float> size, bool border = false);
        void end_child();

        // --- Menu bar ---
        bool begin_menu_bar();
        void end_menu_bar();
        bool begin_menu(const std::string& label);
        void end_menu();
        bool menu_item(const std::string& label, bool enabled = true, bool selected = false);
        std::tuple<bool, bool> menu_item_toggle(const std::string& label, const std::string& shortcut, bool selected);
        bool menu_item_shortcut(const std::string& label, const std::string& shortcut, bool enabled = true);

        // --- Popups (no-op) ---
        bool begin_popup(const std::string& id);
        void open_popup(const std::string& id);
        void end_popup();
        bool begin_context_menu(const std::string& id = "");
        void end_context_menu();
        bool begin_popup_modal(const std::string& title);
        void end_popup_modal();
        void close_current_popup();
        void push_modal_style();
        void pop_modal_style();

        // --- Images (no-op Phase 3) ---
        void image(uint64_t texture_id, std::tuple<float, float> size, nb::object tint = nb::none());
        void image_uv(uint64_t texture_id, std::tuple<float, float> size,
                      std::tuple<float, float> uv0, std::tuple<float, float> uv1,
                      nb::object tint = nb::none());
        bool image_button(const std::string& id, uint64_t texture_id,
                          std::tuple<float, float> size, nb::object tint = nb::none());
        bool toolbar_button(const std::string& id, uint64_t texture_id,
                            std::tuple<float, float> size, bool selected = false,
                            bool disabled = false, const std::string& tooltip = "");
        bool invisible_button(const std::string& id, std::tuple<float, float> size);

        // --- Drag-drop (no-op) ---
        bool begin_drag_drop_source();
        void set_drag_drop_payload(const std::string& type, const std::string& data);
        void end_drag_drop_source();
        bool begin_drag_drop_target();
        std::optional<std::string> accept_drag_drop_payload(const std::string& type);
        void end_drag_drop_target();

        // --- Drawing primitives (no-op for non-overlay) ---
        void draw_circle(float x, float y, float radius, nb::object color, int segments = 32, float thickness = 1.0f);
        void draw_circle_filled(float x, float y, float radius, nb::object color, int segments = 32);
        void draw_rect(float x0, float y0, float x1, float y1, nb::object color, float thickness = 1.0f);
        void draw_rect_filled(float x0, float y0, float x1, float y1, nb::object color, bool background = false);
        void draw_rect_rounded(float x0, float y0, float x1, float y1, nb::object color, float rounding, float thickness = 1.0f, bool background = false);
        void draw_rect_rounded_filled(float x0, float y0, float x1, float y1, nb::object color, float rounding, bool background = false);
        void draw_triangle_filled(float x0, float y0, float x1, float y1, float x2, float y2, nb::object color, bool background = false);
        void draw_line(float x0, float y0, float x1, float y1, nb::object color, float thickness = 1.0f);
        void draw_polyline(nb::object points, nb::object color, bool closed = false, float thickness = 1.0f);
        void draw_poly_filled(nb::object points, nb::object color);
        void draw_text(float x, float y, const std::string& text, nb::object color, bool background = false);
        void draw_window_rect_filled(float x0, float y0, float x1, float y1, nb::object color);
        void draw_window_rect(float x0, float y0, float x1, float y1, nb::object color, float thickness = 1.0f);
        void draw_window_rect_rounded(float x0, float y0, float x1, float y1, nb::object color, float rounding, float thickness = 1.0f);
        void draw_window_rect_rounded_filled(float x0, float y0, float x1, float y1, nb::object color, float rounding);
        void draw_window_line(float x0, float y0, float x1, float y1, nb::object color, float thickness = 1.0f);
        void draw_window_text(float x, float y, const std::string& text, nb::object color);
        void draw_window_triangle_filled(float x0, float y0, float x1, float y1, float x2, float y2, nb::object color);

        // --- Specialized (no-op) ---
        void crf_curve_preview(const std::string& label, float gamma, float toe, float shoulder,
                               float gamma_r = 0.0f, float gamma_g = 0.0f, float gamma_b = 0.0f);
        std::tuple<bool, std::vector<float>> chromaticity_diagram(const std::string& label,
                                                                  float red_x, float red_y,
                                                                  float green_x, float green_y,
                                                                  float blue_x, float blue_y,
                                                                  float neutral_x, float neutral_y,
                                                                  float range = 0.5f);

        // --- Plots (no-op) ---
        void plot_lines(const std::string& label, nb::object values,
                        float scale_min = 0.0f, float scale_max = 0.0f,
                        std::tuple<float, float> size = {0, 0});

        // --- Sub-layouts (simplified) ---
        nb::object row();
        nb::object column();
        nb::object split(float factor = 0.5f);
        nb::object box();
        nb::object grid_flow(int columns = 0, bool even_columns = true, bool even_rows = true);

        // --- Property binding (no-op) ---
        std::tuple<bool, nb::object> prop(nb::object data, const std::string& prop_id,
                                          std::optional<std::string> text = std::nullopt);
        bool prop_enum(nb::object data, const std::string& prop_id,
                       const std::string& value, const std::string& text = "");
        nb::object operator_(const std::string& operator_id, const std::string& text = "",
                             const std::string& icon = "");
        std::tuple<bool, int> prop_search(nb::object data, const std::string& prop_id,
                                          nb::object search_data, const std::string& search_prop,
                                          const std::string& text = "");
        std::tuple<int, int> template_list(const std::string& list_type_id, const std::string& list_id,
                                           nb::object data, const std::string& prop_id,
                                           nb::object active_data, const std::string& active_prop,
                                           int rows = 5);
        void menu(const std::string& menu_id, const std::string& text = "",
                  const std::string& icon = "");
        void popover(const std::string& panel_id, const std::string& text = "",
                     const std::string& icon = "");

        bool is_active() const { return doc_ != nullptr; }

        void release_elements();

    private:
        Slot& ensure_slot(SlotType type, const std::string& key);
        Rml::Element* create_element(SlotType type, const std::string& key);
        Rml::Element* ensure_line_container();
        void finish_current_line();
        void prune_excess_slots(ContainerLevel& level);
        std::string build_id(const std::string& key) const;
        std::string build_slot_id(const char* prefix, const std::string* label = nullptr) const;
        std::string color_to_css(nb::object color) const;
        static std::string stable_label_token(const std::string& label);

        void warn_unsupported(const char* method);

        Rml::ElementDocument* doc_ = nullptr;
        Rml::Element* root_ = nullptr;
        std::vector<ContainerLevel> containers_;
        Rml::Element* current_line_ = nullptr;
        bool next_same_line_ = false;
        int indent_level_ = 0;
        bool disabled_ = false;
        int disabled_depth_ = 0;
        std::vector<std::string> id_stack_;
        bool force_next_open_ = false;
        struct ChildSlotCache {
            Rml::Element* container = nullptr;
            std::deque<Slot> slots;
        };

        std::vector<Rml::ElementPtr> removed_elements_;
        std::unordered_map<std::string, ChildSlotCache> child_slots_;
        std::vector<std::string> child_key_stack_;
        std::unordered_set<std::string> warned_methods_;

        Rml::Element* last_element_ = nullptr;
        bool last_clicked_ = false;

        Rml::Element* tooltip_el_ = nullptr;
        bool tooltip_shown_ = false;
        Rml::Element* tooltip_hover_el_ = nullptr;
        std::string tooltip_text_;
        std::chrono::steady_clock::time_point tooltip_hover_started_at_{};
        bool tooltip_candidate_seen_ = false;

        std::unordered_map<std::string, bool> popup_open_;
        std::string active_popup_id_;
        Rml::Element* popup_backdrop_ = nullptr;
        Rml::Element* popup_dialog_ = nullptr;

        MouseState mouse_;
        std::optional<TableState> table_;
        float cached_line_height_ = 18.0f;

        std::vector<float> item_width_stack_;
        void apply_item_width(Rml::Element* el);
    };

    enum class RmlLayoutDirection : uint8_t { Row,
                                              Column };

    class RmlSubLayout {
    public:
        RmlSubLayout(RmlImModeLayout* parent, RmlLayoutDirection dir);
        RmlSubLayout& enter();
        void exit();

        bool enabled = true;
        bool active = true;
        bool alert = false;
        float scale_x = 1.0f;
        float scale_y = 1.0f;

        void label(const std::string& t) { parent_->label(t); }
        void label_centered(const std::string& t) { parent_->label_centered(t); }
        void heading(const std::string& t) { parent_->heading(t); }
        void text_colored(const std::string& t, nb::object c) { parent_->text_colored(t, c); }
        void text_colored_centered(const std::string& t, nb::object c) { parent_->text_colored_centered(t, c); }
        void text_selectable(const std::string& t, float h = 0.0f) { parent_->text_selectable(t, h); }
        void text_wrapped(const std::string& t) { parent_->text_wrapped(t); }
        void text_disabled(const std::string& t) { parent_->text_disabled(t); }
        void bullet_text(const std::string& t) { parent_->bullet_text(t); }

        bool button(const std::string& l, std::tuple<float, float> s = {0, 0}) { return parent_->button(l, s); }
        bool button_callback(const std::string& l, nb::object cb, std::tuple<float, float> s = {0, 0}) { return parent_->button_callback(l, cb, s); }
        bool small_button(const std::string& l) { return parent_->small_button(l); }
        bool button_styled(const std::string& l, const std::string& st, std::tuple<float, float> s = {0, 0}) { return parent_->button_styled(l, st, s); }
        std::tuple<bool, bool> checkbox(const std::string& l, bool v) { return parent_->checkbox(l, v); }
        std::tuple<bool, int> radio_button(const std::string& l, int c, int v) { return parent_->radio_button(l, c, v); }

        std::tuple<bool, float> slider_float(const std::string& l, float v, float mn, float mx) { return parent_->slider_float(l, v, mn, mx); }
        std::tuple<bool, int> slider_int(const std::string& l, int v, int mn, int mx) { return parent_->slider_int(l, v, mn, mx); }
        std::tuple<bool, float> drag_float(const std::string& l, float v, float sp = 1.0f, float mn = 0.0f, float mx = 0.0f) { return parent_->drag_float(l, v, sp, mn, mx); }
        std::tuple<bool, int> drag_int(const std::string& l, int v, float sp = 1.0f, int mn = 0, int mx = 0) { return parent_->drag_int(l, v, sp, mn, mx); }

        std::tuple<bool, std::string> input_text(const std::string& l, const std::string& v) { return parent_->input_text(l, v); }
        std::tuple<bool, std::string> input_text_with_hint(const std::string& l, const std::string& h, const std::string& v) { return parent_->input_text_with_hint(l, h, v); }
        std::tuple<bool, std::string> input_text_enter(const std::string& l, const std::string& v) { return parent_->input_text_enter(l, v); }
        std::tuple<bool, float> input_float(const std::string& l, float v, float st = 0.0f, float sf = 0.0f, const std::string& fmt = "%.3f") { return parent_->input_float(l, v, st, sf, fmt); }
        std::tuple<bool, int> input_int(const std::string& l, int v, int st = 1, int sf = 100) { return parent_->input_int(l, v, st, sf); }
        std::tuple<bool, int> input_int_formatted(const std::string& l, int v, int st = 0, int sf = 0) { return parent_->input_int_formatted(l, v, st, sf); }
        std::tuple<bool, float> stepper_float(const std::string& l, float v, const std::vector<float>& st = {1.0f, 0.1f, 0.01f}) { return parent_->stepper_float(l, v, st); }

        std::tuple<bool, std::tuple<float, float, float>> color_edit3(const std::string& l, std::tuple<float, float, float> c) { return parent_->color_edit3(l, c); }
        std::tuple<bool, int> combo(const std::string& l, int idx, const std::vector<std::string>& items) { return parent_->combo(l, idx, items); }
        std::tuple<bool, int> listbox(const std::string& l, int idx, const std::vector<std::string>& items, int h = -1) { return parent_->listbox(l, idx, items, h); }
        bool selectable(const std::string& l, bool sel = false, float h = 0.0f) { return parent_->selectable(l, sel, h); }

        void separator() { parent_->separator(); }
        void spacing() { parent_->spacing(); }
        void same_line(float off = 0.0f, float sp = -1.0f) { parent_->same_line(off, sp); }
        void new_line() { parent_->new_line(); }
        bool collapsing_header(const std::string& l, bool def_open = false) { return parent_->collapsing_header(l, def_open); }
        bool tree_node(const std::string& l) { return parent_->tree_node(l); }
        void tree_pop() { parent_->tree_pop(); }

        bool begin_table(const std::string& id, int cols) { return parent_->begin_table(id, cols); }
        void table_setup_column(const std::string& l, float w = 0.0f) { parent_->table_setup_column(l, w); }
        void table_next_row() { parent_->table_next_row(); }
        void table_next_column() { parent_->table_next_column(); }
        void table_headers_row() { parent_->table_headers_row(); }
        void end_table() { parent_->end_table(); }

        void progress_bar(float f, const std::string& o = "", float w = 0.0f, float h = 0.0f) { parent_->progress_bar(f, o, w, h); }

        void push_item_width(float w) { parent_->push_item_width(w); }
        void pop_item_width() { parent_->pop_item_width(); }

        void set_tooltip(const std::string& t) { parent_->set_tooltip(t); }
        bool is_item_hovered() { return parent_->is_item_hovered(); }
        bool is_item_clicked(int b = 0) { return parent_->is_item_clicked(b); }

        void begin_disabled(bool d = true) { parent_->begin_disabled(d); }
        void end_disabled() { parent_->end_disabled(); }

        void push_id(const std::string& id) { parent_->push_id(id); }
        void pop_id() { parent_->pop_id(); }

        bool begin_child(const std::string& id, std::tuple<float, float> sz, bool border = false) { return parent_->begin_child(id, sz, border); }
        void end_child() { parent_->end_child(); }

        void image(uint64_t tex, std::tuple<float, float> sz, nb::object tint = nb::none()) { parent_->image(tex, sz, tint); }
        bool image_button(const std::string& id, uint64_t tex, std::tuple<float, float> sz, nb::object tint = nb::none()) { return parent_->image_button(id, tex, sz, tint); }

        bool begin_context_menu(const std::string& id = "") { return parent_->begin_context_menu(id); }
        void end_context_menu() { parent_->end_context_menu(); }
        bool menu_item(const std::string& l, bool en = true, bool sel = false) { return parent_->menu_item(l, en, sel); }
        bool begin_menu(const std::string& l) { return parent_->begin_menu(l); }
        void end_menu() { parent_->end_menu(); }

        std::tuple<float, float> get_content_region_avail() { return parent_->get_content_region_avail(); }

        std::tuple<bool, nb::object> prop(nb::object data, const std::string& prop_id, std::optional<std::string> text = std::nullopt) { return parent_->prop(data, prop_id, text); }
        bool prop_enum(nb::object data, const std::string& prop_id, const std::string& value, const std::string& text = "") { return parent_->prop_enum(data, prop_id, value, text); }
        nb::object operator_(const std::string& op_id, const std::string& text = "", const std::string& icon = "") { return parent_->operator_(op_id, text, icon); }

        RmlSubLayout row();
        RmlSubLayout column();
        RmlSubLayout split(float factor = 0.5f);
        RmlSubLayout box();
        RmlSubLayout grid_flow(int columns = 0, bool even_columns = true, bool even_rows = true);

    private:
        RmlImModeLayout* parent_;
        RmlLayoutDirection direction_;
        bool entered_ = false;
    };

} // namespace lfs::python
