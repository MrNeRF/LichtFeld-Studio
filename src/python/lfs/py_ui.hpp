/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "python/python_runtime.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cfloat>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;

namespace lfs::vis::gui {
    struct UIContext;
}

namespace lfs::vis::op {
    struct ModalEvent;
}

namespace lfs::python {

    class PyOperatorReturnValue {
    public:
        std::string status;
        nb::dict data;

        PyOperatorReturnValue(std::string s, nb::dict d)
            : status(std::move(s)),
              data(std::move(d)) {}

        [[nodiscard]] bool finished() const { return status == "FINISHED"; }
        [[nodiscard]] bool cancelled() const { return status == "CANCELLED"; }
        [[nodiscard]] bool running_modal() const { return status == "RUNNING_MODAL"; }
        [[nodiscard]] bool pass_through() const { return status == "PASS_THROUGH"; }

        nb::object getattr(const std::string& key) const {
            auto py_key = nb::str(key.c_str());
            if (data.contains(py_key)) {
                return data[py_key];
            }
            if (key == "data") {
                return data;
            }
            throw nb::attribute_error(("OperatorReturnValue has no attribute '" + key + "'").c_str());
        }

        nb::dict get_all_data() const { return data; }
    };

    void register_operator_return_value(nb::module_& m);

    enum class LayoutType { Root,
                            Row,
                            Column,
                            Split,
                            Box,
                            GridFlow };

    struct LayoutState {
        bool enabled = true;
        bool active = true;
        bool alert = false;
    };

    struct LayoutContext {
        LayoutType type = LayoutType::Root;
        int child_index = 0;
        float split_factor = 0.5f;
        bool is_first_child = true;
        float available_width = 0.0f;
        float cursor_start_x = 0.0f;
        int grid_columns = 0;
        int grid_actual_columns = 0;
        bool table_active = false;
        LayoutState state;
    };

    class PyUILayout;

    class PySubLayout {
    public:
        PySubLayout(PyUILayout* parent, LayoutType type, float split_factor = 0.5f,
                    int grid_columns = 0);
        PySubLayout(PySubLayout* parent_sub, LayoutType type, float split_factor = 0.5f,
                    int grid_columns = 0);
        ~PySubLayout();

        PySubLayout& enter();
        void exit();

        bool get_enabled() const { return own_state_.enabled; }
        void set_enabled(bool v) { own_state_.enabled = v; }
        bool get_active() const { return own_state_.active; }
        void set_active(bool v) { own_state_.active = v; }
        bool get_alert() const { return own_state_.alert; }
        void set_alert(bool v) { own_state_.alert = v; }

        PySubLayout row();
        PySubLayout column();
        PySubLayout split(float factor = 0.5f);
        PySubLayout box();
        PySubLayout grid_flow(int columns = 0, bool even_columns = true, bool even_rows = true);

        void label(const std::string& text);
        bool button(const std::string& label, std::tuple<float, float> size = {0, 0});
        bool button_styled(const std::string& label, const std::string& style,
                           std::tuple<float, float> size = {0, 0});
        std::tuple<bool, nb::object> prop(nb::object data, const std::string& prop_id,
                                          std::optional<std::string> text = std::nullopt);
        std::tuple<bool, bool> checkbox(const std::string& label, bool value);
        std::tuple<bool, float> slider_float(const std::string& label, float v, float min, float max);
        std::tuple<bool, int> slider_int(const std::string& label, int v, int min, int max);
        std::tuple<bool, float> drag_float(const std::string& label, float v,
                                           float speed = 1.0f, float min = 0.0f, float max = 0.0f);
        std::tuple<bool, int> drag_int(const std::string& label, int v,
                                       float speed = 1.0f, int min = 0, int max = 0);
        std::tuple<bool, std::string> input_text(const std::string& label, const std::string& value);
        std::tuple<bool, int> combo(const std::string& label, int idx,
                                    const std::vector<std::string>& items);
        void separator();
        void spacing();
        void heading(const std::string& text);
        bool collapsing_header(const std::string& label, bool default_open = false);
        bool tree_node(const std::string& label);
        void tree_pop();
        void progress_bar(float fraction, const std::string& overlay = "", float width = 0.0f);
        void text_colored(const std::string& text, std::tuple<float, float, float, float> color);
        void text_wrapped(const std::string& text);

        bool prop_enum(nb::object data, const std::string& prop_id,
                       const std::string& value, const std::string& text = "");

        PyUILayout* parent() const { return parent_; }
        void advance_child();
        void apply_state();
        void pop_per_item_state();

    private:
        LayoutState effective_state() const;

        PyUILayout* parent_;
        LayoutState own_state_;
        LayoutState inherited_state_;
        LayoutType type_;
        float split_factor_;
        int grid_columns_ = 0;
        bool entered_ = false;
        int color_push_count_ = 0;
        bool disabled_pushed_ = false;
    };

    // PanelSpace enum is defined in py_panel_registry.hpp

    class PyEvent {
    public:
        std::string type;  // 'MOUSEMOVE', 'LEFTMOUSE', 'RIGHTMOUSE', 'MIDDLEMOUSE', 'KEY_A', etc.
        std::string value; // 'PRESS', 'RELEASE', 'CLICK', 'NOTHING'

        // Mouse position
        double mouse_x = 0.0;
        double mouse_y = 0.0;
        double mouse_region_x = 0.0;
        double mouse_region_y = 0.0;

        // Mouse delta (for drag operations)
        double delta_x = 0.0;
        double delta_y = 0.0;

        // Scroll (for WHEELUPMOUSE, WHEELDOWNMOUSE)
        double scroll_x = 0.0;
        double scroll_y = 0.0;

        // Modifier keys
        bool shift = false;
        bool ctrl = false;
        bool alt = false;

        // Tablet support (defaults to 1.0 for mouse)
        float pressure = 1.0f;

        // GUI state - true if mouse is over ImGui window
        bool over_gui = false;

        // Raw key code for KEY events
        int key_code = 0;
    };

    // Convert C++ ModalEvent to Python-friendly PyEvent
    // (ModalEvent is defined in visualizer/operator/operator_context.hpp)
    PyEvent convert_modal_event(const lfs::vis::op::ModalEvent& event);

    // Python operator instance registry - for property setting during invocation
    nb::object get_python_operator_instance(const std::string& id);

    // Forward declarations for scene types (defined in py_scene.hpp)
    class PyScene;
    class PySceneNode;

    // Unified application context - single object for all Python code
    // Replaces scattered lf.has_scene(), lf.get_trainer_manager(), etc. calls
    class PyAppContext {
    public:
        // Scene state
        bool has_scene() const;
        uint64_t scene_generation() const;

        // Training state
        bool has_trainer() const;
        bool is_training() const;
        bool is_paused() const;
        int iteration() const;
        int max_iterations() const;
        float loss() const;

        // Selection state
        bool has_selection() const;
        size_t num_gaussians() const;
        int selection_submode() const;
        int pivot_mode() const;
        int transform_space() const;

        // Viewport bounds
        std::tuple<float, float, float, float> viewport_bounds() const;
        bool viewport_valid() const;

        nb::object scene() const;
        nb::object selected_objects() const;
        nb::object active_object() const;

        // Returns selected gaussians as a boolean mask tensor
        nb::object selected_gaussians() const;
    };

    // Get the current application context
    PyAppContext get_app_context();

    // UI layout object passed to draw() - wraps ImGui calls
    class PyUILayout {
    public:
        explicit PyUILayout(int initial_menu_depth = 0) : menu_depth_(initial_menu_depth) {}

        // Text
        void label(const std::string& text);
        void label_centered(const std::string& text);
        void heading(const std::string& text);
        void text_colored(const std::string& text, std::tuple<float, float, float, float> color);
        void text_colored_centered(const std::string& text, std::tuple<float, float, float, float> color);
        void text_selectable(const std::string& text, float height = 0);
        void text_wrapped(const std::string& text);
        void text_disabled(const std::string& text);
        void bullet_text(const std::string& text);

        // Buttons
        bool button(const std::string& label, std::tuple<float, float> size = {0, 0});
        bool button_callback(const std::string& label, nb::object callback,
                             std::tuple<float, float> size = {0, 0});
        bool small_button(const std::string& label);
        std::tuple<bool, bool> checkbox(const std::string& label, bool value);
        std::tuple<bool, int> radio_button(const std::string& label, int current, int value);

        // Sliders
        std::tuple<bool, float> slider_float(const std::string& label, float value, float min, float max);
        std::tuple<bool, int> slider_int(const std::string& label, int value, int min, int max);
        std::tuple<bool, std::tuple<float, float>> slider_float2(const std::string& label,
                                                                 std::tuple<float, float> value,
                                                                 float min, float max);
        std::tuple<bool, std::tuple<float, float, float>> slider_float3(const std::string& label,
                                                                        std::tuple<float, float, float> value,
                                                                        float min, float max);

        // Drags
        std::tuple<bool, float> drag_float(const std::string& label, float value,
                                           float speed = 1.0f, float min = 0.0f, float max = 0.0f);
        std::tuple<bool, int> drag_int(const std::string& label, int value,
                                       float speed = 1.0f, int min = 0, int max = 0);

        // Input
        std::tuple<bool, std::string> input_text(const std::string& label, const std::string& value);
        std::tuple<bool, std::string> input_text_with_hint(const std::string& label, const std::string& hint,
                                                           const std::string& value);
        std::tuple<bool, float> input_float(const std::string& label, float value, float step = 0.0f,
                                            float step_fast = 0.0f, const std::string& format = "%.3f");
        std::tuple<bool, int> input_int(const std::string& label, int value, int step = 1, int step_fast = 100);
        std::tuple<bool, int> input_int_formatted(const std::string& label, int value, int step = 0, int step_fast = 0);
        std::tuple<bool, std::string> path_input(const std::string& label, const std::string& value,
                                                 bool folder_mode = true, const std::string& dialog_title = "");

        // Color
        std::tuple<bool, std::tuple<float, float, float>> color_edit3(const std::string& label,
                                                                      std::tuple<float, float, float> color);
        std::tuple<bool, std::tuple<float, float, float, float>> color_edit4(const std::string& label,
                                                                             std::tuple<float, float, float, float> color);
        bool color_button(const std::string& label, std::tuple<float, float, float, float> color,
                          std::tuple<float, float> size = {0, 0});

        // Selection
        std::tuple<bool, int> combo(const std::string& label, int current_idx,
                                    const std::vector<std::string>& items);
        std::tuple<bool, int> listbox(const std::string& label, int current_idx,
                                      const std::vector<std::string>& items, int height_items = -1);

        // Layout
        void separator();
        void spacing();
        void same_line(float offset = 0.0f, float spacing = -1.0f);
        void new_line();
        void indent(float width = 0.0f);
        void unindent(float width = 0.0f);
        void set_next_item_width(float width);

        // Grouping
        void begin_group();
        void end_group();
        bool collapsing_header(const std::string& label, bool default_open = false);
        bool tree_node(const std::string& label);
        bool tree_node_ex(const std::string& label, const std::string& flags);
        void tree_pop();

        // Tables
        bool begin_table(const std::string& id, int columns);
        void table_setup_column(const std::string& label, float width = 0.0f);
        void end_table();
        void table_next_row();
        void table_next_column();
        bool table_set_column_index(int column);
        void table_headers_row();
        void table_set_bg_color(int target, std::tuple<float, float, float, float> color);

        // Styled buttons
        bool button_styled(const std::string& label, const std::string& style,
                           std::tuple<float, float> size = {0, 0});

        // Item width stack
        void push_item_width(float width);
        void pop_item_width();

        // Plots
        void plot_lines(const std::string& label, const std::vector<float>& values,
                        float scale_min = FLT_MAX, float scale_max = FLT_MAX,
                        std::tuple<float, float> size = {0, 0});

        // Selectable
        bool selectable(const std::string& label, bool selected = false, float height = 0.0f);

        // Context menus
        bool begin_context_menu(const std::string& id = "");
        void end_context_menu();
        bool begin_popup(const std::string& id);
        void open_popup(const std::string& id);
        void end_popup();
        bool menu_item(const std::string& label, bool enabled = true);
        bool begin_menu(const std::string& label);
        void end_menu();
        std::tuple<bool, std::string> input_text_enter(const std::string& label, const std::string& value);
        void set_keyboard_focus_here();
        bool is_window_focused() const;
        bool is_window_hovered() const;
        void capture_keyboard_from_app(bool capture = true);
        void capture_mouse_from_app(bool capture = true);
        void set_scroll_here_y(float center_y_ratio = 0.5f);
        std::tuple<float, float> get_cursor_screen_pos() const;
        std::tuple<float, float> get_window_pos() const;
        float get_window_width() const;
        float get_text_line_height() const;

        // Modal popups
        bool begin_popup_modal(const std::string& title);
        void end_popup_modal();
        void close_current_popup();
        void set_next_window_pos_center();
        void set_next_window_pos_viewport_center(bool always = false);
        void set_next_window_focus();
        void push_modal_style();
        void pop_modal_style();

        // Cursor and content region
        std::tuple<float, float> get_content_region_avail();
        std::tuple<float, float> get_cursor_pos();
        void set_cursor_pos_x(float x);
        std::tuple<float, float> calc_text_size(const std::string& text);

        // Disabled state
        void begin_disabled(bool disabled = true);
        void end_disabled();

        // Images (texture_id is OpenGL texture handle as uint64)
        void image(uint64_t texture_id, std::tuple<float, float> size,
                   std::tuple<float, float, float, float> tint = {1, 1, 1, 1});
        void image_uv(uint64_t texture_id, std::tuple<float, float> size,
                      std::tuple<float, float> uv0, std::tuple<float, float> uv1,
                      std::tuple<float, float, float, float> tint = {1, 1, 1, 1});
        bool image_button(const std::string& id, uint64_t texture_id, std::tuple<float, float> size,
                          std::tuple<float, float, float, float> tint = {1, 1, 1, 1});

        // Toolbar button (icon with selection state, themed for toolbar use)
        bool toolbar_button(const std::string& id, uint64_t texture_id, std::tuple<float, float> size,
                            bool selected = false, bool disabled = false, const std::string& tooltip = "");

        // Drag-drop
        bool begin_drag_drop_source();
        void set_drag_drop_payload(const std::string& type, const std::string& data);
        void end_drag_drop_source();
        bool begin_drag_drop_target();
        std::optional<std::string> accept_drag_drop_payload(const std::string& type);
        void end_drag_drop_target();

        // Misc
        void progress_bar(float fraction, const std::string& overlay = "", float width = 0.0f);
        void set_tooltip(const std::string& text);
        bool is_item_hovered();
        bool is_item_clicked(int button = 0);
        bool is_item_active();
        bool is_mouse_double_clicked(int button = 0);
        bool is_mouse_dragging(int button = 0);
        float get_mouse_wheel();
        std::tuple<float, float> get_mouse_delta();
        bool invisible_button(const std::string& id, std::tuple<float, float> size);
        void set_cursor_pos(std::tuple<float, float> pos);

        // Child windows
        bool begin_child(const std::string& id, std::tuple<float, float> size, bool border = false);
        void end_child();

        // Menu bar
        bool begin_menu_bar();
        void end_menu_bar();
        bool menu_item_toggle(const std::string& label, const std::string& shortcut, bool selected);
        bool menu_item_shortcut(const std::string& label, const std::string& shortcut, bool enabled = true);

        void push_id(const std::string& id);
        void push_id_int(int id);
        void pop_id();

        // Window control (for floating panels)
        // flags: combination of WindowFlags values
        bool begin_window(const std::string& title, int flags = 0);
        std::tuple<bool, bool> begin_window_closable(const std::string& title, int flags = 0);
        void end_window();
        void push_window_style();
        void pop_window_style();

        // Window positioning (for viewport overlays)
        void set_next_window_pos(std::tuple<float, float> pos, bool first_use = false);
        void set_next_window_size(std::tuple<float, float> size, bool first_use = false);
        void set_next_window_pos_centered(bool first_use = false);
        void set_next_window_bg_alpha(float alpha);
        std::tuple<float, float> get_viewport_pos();
        std::tuple<float, float> get_viewport_size();
        float get_dpi_scale();

        // Cursor
        void set_mouse_cursor_hand();

        // Style control
        void push_style_var_float(const std::string& var, float value);
        void push_style_var_vec2(const std::string& var, std::tuple<float, float> value);
        void pop_style_var(int count = 1);
        void push_style_color(const std::string& col, std::tuple<float, float, float, float> color);
        void pop_style_color(int count = 1);

        // RNA-style property widget (auto-generates from metadata)
        // Returns (changed, new_value) - draws widget based on property type
        std::tuple<bool, nb::object> prop(nb::object data,
                                          const std::string& prop_id,
                                          std::optional<std::string> text = std::nullopt);

        PySubLayout row();
        PySubLayout column();
        PySubLayout split(float factor = 0.5f);
        PySubLayout box();
        PySubLayout grid_flow(int columns = 0, bool even_columns = true, bool even_rows = true);

        bool prop_enum(nb::object data, const std::string& prop_id,
                       const std::string& value, const std::string& text = "");

        int next_box_id() { return box_id_counter_++; }
        int next_grid_id() { return grid_id_counter_++; }
        void reset_frame_state() {
            box_id_counter_ = 0;
            grid_id_counter_ = 0;
        }

        nb::object operator_(const std::string& operator_id, const std::string& text = "",
                             const std::string& icon = "");

        // Searchable dropdown for selecting from a collection
        std::tuple<bool, int> prop_search(nb::object data, const std::string& prop_id,
                                          nb::object search_data, const std::string& search_prop,
                                          const std::string& text = "");

        // UIList template for drawing custom lists
        // Returns (active_index, list_length)
        std::tuple<int, int> template_list(const std::string& list_type_id, const std::string& list_id,
                                           nb::object data, const std::string& prop_id,
                                           nb::object active_data, const std::string& active_prop,
                                           int rows = 5);

        // Inline menu reference
        void menu(const std::string& menu_id, const std::string& text = "", const std::string& icon = "");

        // Panel popover
        void popover(const std::string& panel_id, const std::string& text = "", const std::string& icon = "");

        // Drawing functions for viewport overlays
        void draw_circle(float x, float y, float radius,
                         std::tuple<float, float, float, float> color,
                         int segments = 32, float thickness = 1.0f);
        void draw_circle_filled(float x, float y, float radius,
                                std::tuple<float, float, float, float> color,
                                int segments = 32);
        void draw_rect(float x0, float y0, float x1, float y1,
                       std::tuple<float, float, float, float> color,
                       float thickness = 1.0f);
        void draw_rect_filled(float x0, float y0, float x1, float y1,
                              std::tuple<float, float, float, float> color, bool background = false);
        void draw_rect_rounded(float x0, float y0, float x1, float y1,
                               std::tuple<float, float, float, float> color,
                               float rounding, float thickness = 1.0f, bool background = false);
        void draw_rect_rounded_filled(float x0, float y0, float x1, float y1,
                                      std::tuple<float, float, float, float> color,
                                      float rounding, bool background = false);
        void draw_triangle_filled(float x0, float y0, float x1, float y1, float x2, float y2,
                                  std::tuple<float, float, float, float> color, bool background = false);
        void draw_line(float x0, float y0, float x1, float y1,
                       std::tuple<float, float, float, float> color,
                       float thickness = 1.0f);
        void draw_polyline(const std::vector<std::tuple<float, float>>& points,
                           std::tuple<float, float, float, float> color,
                           bool closed = false, float thickness = 1.0f);
        void draw_poly_filled(const std::vector<std::tuple<float, float>>& points,
                              std::tuple<float, float, float, float> color);
        void draw_text(float x, float y, const std::string& text,
                       std::tuple<float, float, float, float> color, bool background = false);

        // Window-scoped drawing (respects z-order)
        void draw_window_rect_filled(float x0, float y0, float x1, float y1,
                                     std::tuple<float, float, float, float> color);
        void draw_window_rect(float x0, float y0, float x1, float y1,
                              std::tuple<float, float, float, float> color, float thickness = 1.0f);
        void draw_window_rect_rounded(float x0, float y0, float x1, float y1,
                                      std::tuple<float, float, float, float> color,
                                      float rounding, float thickness = 1.0f);
        void draw_window_rect_rounded_filled(float x0, float y0, float x1, float y1,
                                             std::tuple<float, float, float, float> color,
                                             float rounding);
        void draw_window_line(float x0, float y0, float x1, float y1,
                              std::tuple<float, float, float, float> color, float thickness = 1.0f);
        void draw_window_text(float x, float y, const std::string& text,
                              std::tuple<float, float, float, float> color);
        void draw_window_triangle_filled(float x0, float y0, float x1, float y1, float x2, float y2,
                                         std::tuple<float, float, float, float> color);

        void crf_curve_preview(const std::string& label, float gamma, float toe, float shoulder,
                               float gamma_r = 0.0f, float gamma_g = 0.0f, float gamma_b = 0.0f);

        std::tuple<bool, std::vector<float>> chromaticity_diagram(
            const std::string& label,
            float red_x, float red_y, float green_x, float green_y,
            float blue_x, float blue_y, float neutral_x, float neutral_y,
            float range = 0.5f);

    private:
        int menu_depth_;
        int box_id_counter_ = 0;
        int grid_id_counter_ = 0;
    };

    enum class PanelOption : uint32_t {
        DEFAULT_CLOSED = 1 << 0,
        HIDE_HEADER = 1 << 1,
    };

    // Panel poll dependencies (mirrors operator PollDependency)
    // Must be declared before PyPanelInfo which uses it
    enum class PanelPollDependency : uint8_t {
        NONE = 0,
        SELECTION = 1 << 0,
        TRAINING = 1 << 1,
        SCENE = 1 << 2,
        ALL = SELECTION | TRAINING | SCENE
    };

    inline PanelPollDependency operator|(PanelPollDependency a, PanelPollDependency b) {
        return static_cast<PanelPollDependency>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
    }

    inline PanelPollDependency operator&(PanelPollDependency a, PanelPollDependency b) {
        return static_cast<PanelPollDependency>(static_cast<uint8_t>(a) & static_cast<uint8_t>(b));
    }

    // Panel info stored in registry
    struct PyPanelInfo {
        nb::object panel_class;
        nb::object panel_instance;
        std::string label;
        std::string idname;
        PanelSpace space = PanelSpace::Floating;
        int order = 100;
        bool enabled = true;

        std::string category;  // Tab name for grouping panels
        std::string parent_id; // Parent panel ID for nesting
        uint32_t options = 0;  // Bitmask of PanelOption values
        PanelPollDependency poll_deps = PanelPollDependency::ALL;

        // Error recovery tracking
        int consecutive_errors = 0;
        bool error_disabled = false;
        static constexpr int MAX_CONSECUTIVE_ERRORS = 3;

        bool has_option(PanelOption opt) const {
            return (options & static_cast<uint32_t>(opt)) != 0;
        }
    };

    struct PanelPollCacheEntry {
        bool result;
        uint64_t scene_generation;
        bool has_selection;
        bool is_training;
        PanelPollDependency deps;
    };

    // Singleton registry for Python panels
    class PyPanelRegistry {
    public:
        static PyPanelRegistry& instance();

        // Initialize property registry subscriptions for UI invalidation
        void init();

        void register_panel(nb::object panel_class);
        void unregister_panel(nb::object panel_class);
        void unregister_all();

        void draw_panels(PanelSpace space);
        void draw_single_panel(const std::string& label);
        bool has_panels(PanelSpace space) const;

        std::vector<std::string> get_panel_names(PanelSpace space) const;
        void set_panel_enabled(const std::string& label, bool enabled);
        bool is_panel_enabled(const std::string& label) const;

        // Main panel tab API - returns panels in order
        std::vector<PyPanelInfo*> get_main_panel_tabs();

        // Panel attribute modification API
        PyPanelInfo* get_panel(const std::string& idname);
        bool set_panel_label(const std::string& idname, const std::string& new_label);
        bool set_panel_order(const std::string& idname, int new_order);
        bool set_panel_space(const std::string& idname, PanelSpace new_space);

        void invalidate_poll_cache();

    private:
        PyPanelRegistry() = default;
        ~PyPanelRegistry() = default;
        PyPanelRegistry(const PyPanelRegistry&) = delete;
        PyPanelRegistry& operator=(const PyPanelRegistry&) = delete;

        bool check_poll_cached(const std::string& idname, nb::object panel_class, PanelPollDependency deps);

        mutable std::mutex mutex_;
        std::vector<PyPanelInfo> panels_;
        mutable std::unordered_map<std::string, PanelPollCacheEntry> poll_cache_;
    };

    // Theme palette wrapper (read-only)
    struct PyThemePalette {
        std::tuple<float, float, float, float> background;
        std::tuple<float, float, float, float> surface;
        std::tuple<float, float, float, float> surface_bright;
        std::tuple<float, float, float, float> primary;
        std::tuple<float, float, float, float> primary_dim;
        std::tuple<float, float, float, float> secondary;
        std::tuple<float, float, float, float> text;
        std::tuple<float, float, float, float> text_dim;
        std::tuple<float, float, float, float> border;
        std::tuple<float, float, float, float> success;
        std::tuple<float, float, float, float> warning;
        std::tuple<float, float, float, float> error;
        std::tuple<float, float, float, float> info;
        std::tuple<float, float, float, float> toolbar_background;
        std::tuple<float, float, float, float> row_even;
        std::tuple<float, float, float, float> row_odd;
        std::tuple<float, float, float, float> overlay_border;
        std::tuple<float, float, float, float> overlay_icon;
        std::tuple<float, float, float, float> overlay_text;
        std::tuple<float, float, float, float> overlay_text_dim;
    };

    // Theme sizes wrapper (read-only)
    struct PyThemeSizes {
        float window_rounding;
        float frame_rounding;
        float popup_rounding;
        float scrollbar_rounding;
        float tab_rounding;
        float border_size;
        std::tuple<float, float> window_padding;
        std::tuple<float, float> frame_padding;
        std::tuple<float, float> item_spacing;
        float toolbar_button_size;
        float toolbar_padding;
        float toolbar_spacing;
    };

    // Theme wrapper (read-only)
    struct PyTheme {
        std::string name;
        PyThemePalette palette;
        PyThemeSizes sizes;
    };

    // Get current theme
    PyTheme get_current_theme();

    // Hook position enum (mirrors ui_hooks.hpp but with Python bindings)
    enum class PyHookPosition {
        Prepend, // Run before native content
        Append   // Run after native content
    };

    // UI Hook registry for Python callbacks
    class PyUIHookRegistry {
    public:
        static PyUIHookRegistry& instance();

        // Register a Python callback for a hook point
        void add_hook(const std::string& panel,
                      const std::string& section,
                      nb::object callback,
                      PyHookPosition position = PyHookPosition::Append);

        // Remove a specific hook
        void remove_hook(const std::string& panel,
                         const std::string& section,
                         nb::object callback);

        // Clear hooks for a panel/section
        void clear_hooks(const std::string& panel, const std::string& section = "");

        // Clear all hooks
        void clear_all();

        // Invoke hooks - called from C++ panels
        void invoke(const std::string& panel,
                    const std::string& section,
                    PyHookPosition position);

        // Check if hooks exist
        bool has_hooks(const std::string& panel, const std::string& section) const;

        // Get all registered hook points
        std::vector<std::string> get_hook_points() const;

    private:
        PyUIHookRegistry() = default;
        ~PyUIHookRegistry() = default;
        PyUIHookRegistry(const PyUIHookRegistry&) = delete;
        PyUIHookRegistry& operator=(const PyUIHookRegistry&) = delete;

        struct HookEntry {
            nb::object callback;
            PyHookPosition position;
        };

        mutable std::mutex mutex_;
        std::unordered_map<std::string, std::vector<HookEntry>> hooks_;
    };

    // MenuLocation enum defined in python/python_runtime.hpp

    struct PyMenuClassInfo {
        std::string idname;
        std::string label;
        MenuLocation location;
        int order;
        nb::object menu_class;
        nb::object menu_instance;
    };

    class PyMenuRegistry {
    public:
        static PyMenuRegistry& instance();

        void unregister_all();

        void draw_menu_items(MenuLocation location);
        bool has_items(MenuLocation location) const;

        bool has_menu_bar_entries() const;
        std::vector<PyMenuClassInfo*> get_menu_bar_entries();
        void draw_menu_bar_entry(const std::string& idname);

        void sync_from_python();

    private:
        PyMenuRegistry() = default;
        ~PyMenuRegistry() = default;
        PyMenuRegistry(const PyMenuRegistry&) = delete;
        PyMenuRegistry& operator=(const PyMenuRegistry&) = delete;

        void ensure_synced();

        mutable std::mutex mutex_;
        std::vector<PyMenuClassInfo> menu_classes_;
        bool synced_from_python_ = false;
    };

    class PyOperatorProperties {
    public:
        explicit PyOperatorProperties(const std::string& operator_id);

        void set_property(const std::string& name, nb::object value);
        [[nodiscard]] nb::object get_property(const std::string& name) const;
        [[nodiscard]] nb::dict get_properties() const;
        [[nodiscard]] const std::string& get_operator_id() const { return operator_id_; }

    private:
        std::string operator_id_;
        nb::dict properties_;
    };

    // Modal dialog types
    enum class ModalDialogType { Confirm,
                                 Input,
                                 Message };

    // Message style for visual appearance
    enum class MessageStyle { Info,
                              Warning,
                              Error };

    // Modal dialog info
    struct PyModalDialog {
        std::string id;
        std::string title;
        std::string message;
        std::vector<std::string> buttons;
        nb::object callback;
        ModalDialogType type;
        MessageStyle style = MessageStyle::Info;
        std::string input_value;
        bool is_open = true;
        bool needs_open = true;
    };

    // Singleton registry for modal dialogs
    class PyModalRegistry {
    public:
        static PyModalRegistry& instance();

        // Show dialogs
        void show_confirm(const std::string& title, const std::string& message,
                          const std::vector<std::string>& buttons, nb::object callback);
        void show_input(const std::string& title, const std::string& message,
                        const std::string& default_value, nb::object callback);
        void show_message(const std::string& title, const std::string& message,
                          MessageStyle style = MessageStyle::Info,
                          nb::object callback = nb::none());

        // Draw pending modals (called from C++ render loop)
        void draw_modals();

        // Check if any modals are open
        bool has_open_modals() const;

    private:
        PyModalRegistry() = default;
        ~PyModalRegistry() = default;
        PyModalRegistry(const PyModalRegistry&) = delete;
        PyModalRegistry& operator=(const PyModalRegistry&) = delete;

        void draw_confirm_dialog(PyModalDialog& modal, float scale);
        void draw_input_dialog(PyModalDialog& modal, float scale);
        void draw_message_dialog(PyModalDialog& modal, float scale);

        mutable std::mutex mutex_;
        std::vector<PyModalDialog> modals_;
        uint32_t next_id_ = 0;
    };

    // Register UI classes with nanobind module
    void register_ui(nb::module_& m);

    // Register unified class API on root module (lf.register_class, lf.unregister_class)
    void register_class_api(nb::module_& m);

    // Sub-registration functions (called by register_ui)
    void register_ui_context(nb::module_& m);
    void register_ui_theme(nb::module_& m);
    void register_ui_operators(nb::module_& m);
    void register_ui_modals(nb::module_& m);
    void register_ui_hooks(nb::module_& m);
    void register_ui_menus(nb::module_& m);
    void register_ui_panels(nb::module_& m);
    void register_keymap(nb::module_& m);

} // namespace lfs::python
