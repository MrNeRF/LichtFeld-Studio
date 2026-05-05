/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/line_renderer.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <glm/glm.hpp>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;

namespace lfs::python {

    enum class GizmoEventType { Press,
                                Release,
                                Move,
                                Drag };

    struct PyGizmoEvent {
        GizmoEventType type = GizmoEventType::Move;
        int button = 0;
        float mouse_x = 0.0f;
        float mouse_y = 0.0f;
        float delta_x = 0.0f;
        float delta_y = 0.0f;
        bool shift = false;
        bool ctrl = false;
        bool alt = false;
    };

    enum class GizmoResult { PassThrough,
                             Running,
                             Finished,
                             Cancelled };

    enum class TransformGizmoOperation {
        Translate,
        Rotate,
        Scale
    };

    enum class TransformGizmoSpace {
        Local,
        World
    };

    class PyGizmoContext {
    public:
        struct DrawCommand {
            enum Type { LINE_2D,
                        CIRCLE_2D,
                        RECT_2D,
                        FILLED_RECT_2D,
                        FILLED_CIRCLE_2D,
                        LINE_3D };
            Type type;
            float x1, y1, z1;
            float x2, y2, z2;
            float r, g, b, a;
            float thickness;
            float radius;
        };

        [[nodiscard]] bool has_selection() const;
        [[nodiscard]] std::tuple<float, float, float> selection_center() const;
        [[nodiscard]] std::tuple<float, float> selection_center_screen() const;
        [[nodiscard]] std::tuple<float, float, float> camera_position() const;
        [[nodiscard]] std::tuple<float, float, float> camera_forward() const;
        [[nodiscard]] std::optional<std::tuple<float, float>> world_to_screen(std::tuple<float, float, float> pos) const;
        [[nodiscard]] std::optional<std::tuple<float, float, float>> screen_to_world_ray(std::tuple<float, float> pos) const;

        void draw_line_2d(std::tuple<float, float> start, std::tuple<float, float> end,
                          std::tuple<float, float, float, float> color, float thickness = 1.0f);
        void draw_circle_2d(std::tuple<float, float> center, float radius,
                            std::tuple<float, float, float, float> color, float thickness = 1.0f);
        void draw_rect_2d(std::tuple<float, float> min, std::tuple<float, float> max,
                          std::tuple<float, float, float, float> color, float thickness = 1.0f);
        void draw_filled_rect_2d(std::tuple<float, float> min, std::tuple<float, float> max,
                                 std::tuple<float, float, float, float> color);
        void draw_filled_circle_2d(std::tuple<float, float> center, float radius,
                                   std::tuple<float, float, float, float> color);
        void draw_line_3d(std::tuple<float, float, float> start, std::tuple<float, float, float> end,
                          std::tuple<float, float, float, float> color, float thickness = 1.0f);

        [[nodiscard]] const std::vector<DrawCommand>& get_draw_commands() const { return draw_commands_; }
        void clear_draw_commands() { draw_commands_.clear(); }

    private:
        mutable std::vector<DrawCommand> draw_commands_;
    };

    struct PyGizmoInfo {
        std::string id;
        nb::object gizmo_class;
        nb::object gizmo_instance;
        bool has_poll = false;
        bool has_draw = false;
        bool has_handle_mouse = false;
    };

    class PyGizmoRegistry {
    public:
        static PyGizmoRegistry& instance();

        void register_gizmo(nb::object gizmo_class);
        void unregister_gizmo(const std::string& id);
        void unregister_all();

        [[nodiscard]] bool poll(const std::string& id);
        void draw_all(PyGizmoContext& ctx);
        [[nodiscard]] GizmoResult handle_mouse(const std::string& id, PyGizmoContext& ctx, const PyGizmoEvent& event);

        [[nodiscard]] std::vector<std::string> get_gizmo_ids() const;
        [[nodiscard]] bool has_gizmos() const;

    private:
        PyGizmoRegistry() = default;
        PyGizmoRegistry(const PyGizmoRegistry&) = delete;
        PyGizmoRegistry& operator=(const PyGizmoRegistry&) = delete;

        PyGizmoInfo* ensure_instance(PyGizmoInfo& gizmo);

        mutable std::mutex mutex_;
        std::unordered_map<std::string, PyGizmoInfo> gizmos_;
    };

    struct PyTransformGizmoState {
        enum class TargetKind {
            None,
            Callback,
            Node
        };

        int instance_id = 0;
        std::string id;
        TransformGizmoOperation operation = TransformGizmoOperation::Translate;
        TransformGizmoSpace space = TransformGizmoSpace::Local;
        glm::mat4 matrix{1.0f};

        bool attached = false;
        bool visible = true;
        bool enabled = true;
        bool input_enabled = true;
        bool active = false;
        bool hovered = false;
        bool changed = false;
        bool snap = false;
        float translate_snap = 0.1f;
        float rotate_snap_degrees = 5.0f;
        float scale_snap_ratio = 0.1f;

        TargetKind target_kind = TargetKind::None;
        nb::object target_getter;
        nb::object target_setter;
        std::string target_node_name;
        bool target_node_visualizer_world = true;

        nb::object on_begin;
        nb::object on_change;
        nb::object on_end;
    };

    class PyTransformGizmo {
    public:
        PyTransformGizmo(const std::string& operation = "translate",
                         const std::vector<float>& matrix = {},
                         const std::string& id = "");

        [[nodiscard]] const std::string& id() const { return state_->id; }
        [[nodiscard]] std::string operation() const;
        void set_operation(const std::string& operation);

        [[nodiscard]] std::string space() const;
        void set_space(const std::string& space);

        [[nodiscard]] std::vector<float> matrix() const;
        void set_matrix(const std::vector<float>& matrix);
        [[nodiscard]] std::vector<float> translation() const;
        void set_translation(const std::vector<float>& translation);

        [[nodiscard]] bool attached() const { return state_->attached; }
        [[nodiscard]] bool visible() const { return state_->visible; }
        void set_visible(bool visible) { state_->visible = visible; }
        [[nodiscard]] bool enabled() const { return state_->enabled; }
        void set_enabled(bool enabled) { state_->enabled = enabled; }
        [[nodiscard]] bool input_enabled() const { return state_->input_enabled; }
        void set_input_enabled(bool enabled) { state_->input_enabled = enabled; }
        [[nodiscard]] bool active() const { return state_->active; }
        [[nodiscard]] bool hovered() const { return state_->hovered; }
        [[nodiscard]] bool changed() const { return state_->changed; }

        [[nodiscard]] bool snap() const { return state_->snap; }
        void set_snap(bool snap) { state_->snap = snap; }
        [[nodiscard]] float translate_snap() const { return state_->translate_snap; }
        void set_translate_snap(float units) { state_->translate_snap = units; }
        [[nodiscard]] float rotate_snap_degrees() const { return state_->rotate_snap_degrees; }
        void set_rotate_snap_degrees(float degrees) { state_->rotate_snap_degrees = degrees; }
        [[nodiscard]] float scale_snap_ratio() const { return state_->scale_snap_ratio; }
        void set_scale_snap_ratio(float ratio) { state_->scale_snap_ratio = ratio; }

        void attach();
        void attach_to_callbacks(nb::object getter, nb::object setter);
        void attach_to_node(const std::string& node_name, bool visualizer_world = true);
        void detach();

        void set_on_begin(nb::object callback);
        void set_on_change(nb::object callback);
        void set_on_end(nb::object callback);

        void draw_frame(const glm::mat4& view,
                        const glm::mat4& projection,
                        const glm::vec2& viewport_pos,
                        const glm::vec2& viewport_size,
                        lfs::vis::gui::NativeOverlayDrawList* draw_list);

    private:
        explicit PyTransformGizmo(std::shared_ptr<PyTransformGizmoState> state);

        void sync_from_target_if_idle();
        void apply_to_target();
        void call_lifecycle_callback(const nb::object& callback);
        [[nodiscard]] glm::mat3 orientation_for_operation() const;

        std::shared_ptr<PyTransformGizmoState> state_;

        friend class PyTransformGizmoRegistry;
    };

    class PyTransformGizmoRegistry {
    public:
        static PyTransformGizmoRegistry& instance();

        void attach(std::shared_ptr<PyTransformGizmoState> state);
        void detach(int instance_id);
        void clear_all();

        [[nodiscard]] bool has_attached() const;
        [[nodiscard]] std::vector<std::string> ids() const;

        void draw_all(const glm::mat4& view,
                      const glm::mat4& projection,
                      const glm::vec2& viewport_pos,
                      const glm::vec2& viewport_size,
                      lfs::vis::gui::NativeOverlayDrawList* draw_list);

    private:
        mutable std::mutex mutex_;
        std::unordered_map<int, std::shared_ptr<PyTransformGizmoState>> gizmos_;
    };

    void register_gizmos(nb::module_& m);

} // namespace lfs::python
