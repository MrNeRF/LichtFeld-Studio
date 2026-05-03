/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_gizmo.hpp"
#include "core/logger.hpp"
#include "gui/gui_focus_state.hpp"
#include "gui/gui_manager.hpp"
#include "gui/rotation_gizmo.hpp"
#include "gui/scale_gizmo.hpp"
#include "gui/translation_gizmo.hpp"
#include "python/python_runtime.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "visualizer/scene_coordinate_utils.hpp"
#include "visualizer_impl.hpp"

#include <nanobind/stl/shared_ptr.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <stdexcept>

namespace lfs::python {

    namespace {
        constexpr float DEFAULT_VIEWPORT_WIDTH = 800.0f;
        constexpr float DEFAULT_VIEWPORT_HEIGHT = 600.0f;
        constexpr float DEFAULT_CAMERA_Z = 5.0f;
        constexpr float PROJECTION_SCALE = 100.0f;
        constexpr float MATRIX_EPSILON = 1e-6f;

        std::atomic<int> g_next_transform_gizmo_id{100000};

        [[nodiscard]] TransformGizmoOperation parse_transform_gizmo_operation(const std::string& operation) {
            if (operation == "translate" || operation == "translation" || operation == "move")
                return TransformGizmoOperation::Translate;
            if (operation == "rotate" || operation == "rotation")
                return TransformGizmoOperation::Rotate;
            if (operation == "scale")
                return TransformGizmoOperation::Scale;
            throw std::invalid_argument("Unknown transform gizmo operation: " + operation);
        }

        [[nodiscard]] const char* transform_gizmo_operation_name(const TransformGizmoOperation operation) {
            switch (operation) {
            case TransformGizmoOperation::Translate: return "translate";
            case TransformGizmoOperation::Rotate: return "rotate";
            case TransformGizmoOperation::Scale: return "scale";
            }
            return "translate";
        }

        [[nodiscard]] TransformGizmoSpace parse_transform_gizmo_space(const std::string& space) {
            if (space == "local" || space == "Local")
                return TransformGizmoSpace::Local;
            if (space == "world" || space == "World")
                return TransformGizmoSpace::World;
            throw std::invalid_argument("Unknown transform gizmo space: " + space);
        }

        [[nodiscard]] const char* transform_gizmo_space_name(const TransformGizmoSpace space) {
            switch (space) {
            case TransformGizmoSpace::Local: return "local";
            case TransformGizmoSpace::World: return "world";
            }
            return "local";
        }

        [[nodiscard]] std::vector<float> matrix_to_vector(const glm::mat4& matrix) {
            return std::vector<float>(&matrix[0][0], &matrix[0][0] + 16);
        }

        [[nodiscard]] glm::mat4 matrix_from_vector(const std::vector<float>& values) {
            if (values.empty())
                return glm::mat4(1.0f);
            if (values.size() != 16)
                throw std::invalid_argument("Transform gizmo matrix must contain 16 floats");

            glm::mat4 matrix(1.0f);
            std::memcpy(&matrix[0][0], values.data(), 16 * sizeof(float));
            return matrix;
        }

        [[nodiscard]] glm::vec3 vector3_from_vector(const std::vector<float>& values, const char* name) {
            if (values.size() != 3)
                throw std::invalid_argument(std::string(name) + " must contain 3 floats");
            return {values[0], values[1], values[2]};
        }

        [[nodiscard]] glm::vec3 safe_normalize(const glm::vec3& value, const glm::vec3& fallback) {
            const float len2 = glm::dot(value, value);
            if (len2 <= MATRIX_EPSILON * MATRIX_EPSILON || !std::isfinite(len2))
                return fallback;
            return value / std::sqrt(len2);
        }

        [[nodiscard]] glm::mat3 extract_rotation(const glm::mat4& matrix) {
            return glm::mat3(
                safe_normalize(glm::vec3(matrix[0]), {1.0f, 0.0f, 0.0f}),
                safe_normalize(glm::vec3(matrix[1]), {0.0f, 1.0f, 0.0f}),
                safe_normalize(glm::vec3(matrix[2]), {0.0f, 0.0f, 1.0f}));
        }

        void mark_scene_transform_changed() {
            if (auto* sm = get_scene_manager()) {
                sm->getScene().notifyMutation(core::Scene::MutationType::MODEL_CHANGED);
            }

            auto* gm = get_gui_manager();
            auto* viewer = gm ? gm->getViewer() : nullptr;
            auto* rm = viewer ? viewer->getRenderingManager() : nullptr;
            if (rm) {
                rm->markDirty(vis::DirtyFlag::SPLATS | vis::DirtyFlag::MESH | vis::DirtyFlag::OVERLAY);
            }
        }

        [[nodiscard]] bool callable_or_none(const nb::object& object) {
            return !object.is_valid() || object.is_none() || PyCallable_Check(object.ptr());
        }
    } // namespace

    bool PyGizmoContext::has_selection() const { return false; }

    std::tuple<float, float, float> PyGizmoContext::selection_center() const { return {0.0f, 0.0f, 0.0f}; }

    std::tuple<float, float, float> PyGizmoContext::camera_position() const { return {0.0f, 0.0f, DEFAULT_CAMERA_Z}; }

    std::tuple<float, float, float> PyGizmoContext::camera_forward() const { return {0.0f, 0.0f, -1.0f}; }

    std::tuple<float, float> PyGizmoContext::selection_center_screen() const {
        if (const auto screen = world_to_screen(selection_center()))
            return *screen;
        return {0.0f, 0.0f};
    }

    std::optional<std::tuple<float, float>> PyGizmoContext::world_to_screen(std::tuple<float, float, float> pos) const {
        const auto [wx, wy, wz] = pos;

        // Match the documented visualizer-world convention with a default camera
        // at +Z looking along -Z.
        const float view_z = wz - DEFAULT_CAMERA_Z;
        if (view_z >= -1e-6f)
            return std::nullopt;
        const float depth = -view_z;
        const float sx = DEFAULT_VIEWPORT_WIDTH / 2.0f + wx * PROJECTION_SCALE / depth;
        const float sy = DEFAULT_VIEWPORT_HEIGHT / 2.0f - wy * PROJECTION_SCALE / depth;
        return std::make_tuple(sx, sy);
    }

    std::optional<std::tuple<float, float, float>> PyGizmoContext::screen_to_world_ray(std::tuple<float, float> pos) const {
        const auto [sx, sy] = pos;
        const float dx = (sx - DEFAULT_VIEWPORT_WIDTH / 2.0f) / (DEFAULT_VIEWPORT_WIDTH / 2.0f);
        const float dy = -(sy - DEFAULT_VIEWPORT_HEIGHT / 2.0f) / (DEFAULT_VIEWPORT_HEIGHT / 2.0f);
        const float len = std::sqrt(dx * dx + dy * dy + 1.0f);
        return std::make_tuple(dx / len, dy / len, -1.0f / len);
    }

    void PyGizmoContext::draw_line_2d(std::tuple<float, float> start, std::tuple<float, float> end,
                                      std::tuple<float, float, float, float> color, float thickness) {
        draw_commands_.push_back({DrawCommand::LINE_2D,
                                  std::get<0>(start), std::get<1>(start), 0.0f,
                                  std::get<0>(end), std::get<1>(end), 0.0f,
                                  std::get<0>(color), std::get<1>(color), std::get<2>(color), std::get<3>(color),
                                  thickness, 0.0f});
    }

    void PyGizmoContext::draw_circle_2d(std::tuple<float, float> center, float radius,
                                        std::tuple<float, float, float, float> color, float thickness) {
        draw_commands_.push_back({DrawCommand::CIRCLE_2D,
                                  std::get<0>(center), std::get<1>(center), 0.0f,
                                  0.0f, 0.0f, 0.0f,
                                  std::get<0>(color), std::get<1>(color), std::get<2>(color), std::get<3>(color),
                                  thickness, radius});
    }

    void PyGizmoContext::draw_rect_2d(std::tuple<float, float> min, std::tuple<float, float> max,
                                      std::tuple<float, float, float, float> color, float thickness) {
        draw_commands_.push_back({DrawCommand::RECT_2D,
                                  std::get<0>(min), std::get<1>(min), 0.0f,
                                  std::get<0>(max), std::get<1>(max), 0.0f,
                                  std::get<0>(color), std::get<1>(color), std::get<2>(color), std::get<3>(color),
                                  thickness, 0.0f});
    }

    void PyGizmoContext::draw_filled_rect_2d(std::tuple<float, float> min, std::tuple<float, float> max,
                                             std::tuple<float, float, float, float> color) {
        draw_commands_.push_back({DrawCommand::FILLED_RECT_2D,
                                  std::get<0>(min), std::get<1>(min), 0.0f,
                                  std::get<0>(max), std::get<1>(max), 0.0f,
                                  std::get<0>(color), std::get<1>(color), std::get<2>(color), std::get<3>(color),
                                  0.0f, 0.0f});
    }

    void PyGizmoContext::draw_filled_circle_2d(std::tuple<float, float> center, float radius,
                                               std::tuple<float, float, float, float> color) {
        draw_commands_.push_back({DrawCommand::FILLED_CIRCLE_2D,
                                  std::get<0>(center), std::get<1>(center), 0.0f,
                                  0.0f, 0.0f, 0.0f,
                                  std::get<0>(color), std::get<1>(color), std::get<2>(color), std::get<3>(color),
                                  0.0f, radius});
    }

    void PyGizmoContext::draw_line_3d(std::tuple<float, float, float> start, std::tuple<float, float, float> end,
                                      std::tuple<float, float, float, float> color, float thickness) {
        draw_commands_.push_back({DrawCommand::LINE_3D,
                                  std::get<0>(start), std::get<1>(start), std::get<2>(start),
                                  std::get<0>(end), std::get<1>(end), std::get<2>(end),
                                  std::get<0>(color), std::get<1>(color), std::get<2>(color), std::get<3>(color),
                                  thickness, 0.0f});
    }

    PyGizmoRegistry& PyGizmoRegistry::instance() {
        static PyGizmoRegistry inst;
        return inst;
    }

    void PyGizmoRegistry::register_gizmo(nb::object gizmo_class) {
        std::lock_guard lock(mutex_);

        if (!nb::hasattr(gizmo_class, "gizmo_id")) {
            LOG_ERROR("Gizmo class missing gizmo_id");
            return;
        }

        const auto id = nb::cast<std::string>(gizmo_class.attr("gizmo_id"));
        gizmos_[id] = {
            id,
            gizmo_class,
            nb::object(),
            nb::hasattr(gizmo_class, "poll"),
            nb::hasattr(gizmo_class, "draw"),
            nb::hasattr(gizmo_class, "handle_mouse")};
    }

    void PyGizmoRegistry::unregister_gizmo(const std::string& id) {
        std::lock_guard lock(mutex_);
        gizmos_.erase(id);
    }

    void PyGizmoRegistry::unregister_all() {
        std::lock_guard lock(mutex_);
        gizmos_.clear();
    }

    PyGizmoInfo* PyGizmoRegistry::ensure_instance(PyGizmoInfo& gizmo) {
        if (!gizmo.gizmo_instance.is_valid() || gizmo.gizmo_instance.is_none()) {
            nb::gil_scoped_acquire gil;
            try {
                gizmo.gizmo_instance = gizmo.gizmo_class();
            } catch (const std::exception& e) {
                LOG_ERROR("Failed to instantiate gizmo {}: {}", gizmo.id, e.what());
                return nullptr;
            }
        }
        return &gizmo;
    }

    bool PyGizmoRegistry::poll(const std::string& id) {
        PyGizmoInfo* gizmo;
        {
            std::lock_guard lock(mutex_);
            auto it = gizmos_.find(id);
            if (it == gizmos_.end())
                return false;
            gizmo = &it->second;
        }

        if (!gizmo->has_poll)
            return true;

        nb::gil_scoped_acquire gil;
        try {
            PyGizmoContext ctx;
            return nb::cast<bool>(gizmo->gizmo_class.attr("poll")(ctx));
        } catch (const std::exception& e) {
            LOG_ERROR("Gizmo '{}' poll: {}", id, e.what());
            return false;
        }
    }

    void PyGizmoRegistry::draw_all(PyGizmoContext& ctx) {
        std::vector<PyGizmoInfo> gizmos_copy;
        {
            std::lock_guard lock(mutex_);
            gizmos_copy.reserve(gizmos_.size());
            for (auto& [_, gizmo] : gizmos_)
                gizmos_copy.push_back(gizmo);
        }

        nb::gil_scoped_acquire gil;
        for (auto& gizmo : gizmos_copy) {
            if (!gizmo.has_draw)
                continue;

            auto* inst = ensure_instance(gizmo);
            if (!inst)
                continue;

            if (gizmo.has_poll) {
                try {
                    if (!nb::cast<bool>(gizmo.gizmo_class.attr("poll")(ctx)))
                        continue;
                } catch (const std::exception& e) {
                    LOG_ERROR("Gizmo '{}' poll: {}", gizmo.id, e.what());
                    continue;
                }
            }

            try {
                gizmo.gizmo_instance.attr("draw")(ctx);
            } catch (const std::exception& e) {
                LOG_ERROR("Gizmo '{}' draw: {}", gizmo.id, e.what());
            }
        }
    }

    GizmoResult PyGizmoRegistry::handle_mouse(const std::string& id, PyGizmoContext& ctx, const PyGizmoEvent& event) {
        PyGizmoInfo* gizmo;
        {
            std::lock_guard lock(mutex_);
            auto it = gizmos_.find(id);
            if (it == gizmos_.end())
                return GizmoResult::PassThrough;
            gizmo = ensure_instance(it->second);
        }

        if (!gizmo || !gizmo->has_handle_mouse)
            return GizmoResult::PassThrough;

        nb::gil_scoped_acquire gil;
        try {
            nb::dict evt;
            switch (event.type) {
            case GizmoEventType::Press: evt["type"] = "PRESS"; break;
            case GizmoEventType::Release: evt["type"] = "RELEASE"; break;
            case GizmoEventType::Move: evt["type"] = "MOVE"; break;
            case GizmoEventType::Drag: evt["type"] = "DRAG"; break;
            }
            evt["button"] = event.button;
            evt["x"] = event.mouse_x;
            evt["y"] = event.mouse_y;
            evt["delta_x"] = event.delta_x;
            evt["delta_y"] = event.delta_y;
            evt["shift"] = event.shift;
            evt["ctrl"] = event.ctrl;
            evt["alt"] = event.alt;

            const nb::object result = gizmo->gizmo_instance.attr("handle_mouse")(ctx, evt);
            if (nb::isinstance<nb::dict>(result)) {
                const auto d = nb::cast<nb::dict>(result);
                if (d.contains("RUNNING_MODAL"))
                    return GizmoResult::Running;
                if (d.contains("FINISHED"))
                    return GizmoResult::Finished;
                if (d.contains("CANCELLED"))
                    return GizmoResult::Cancelled;
            } else if (nb::isinstance<nb::str>(result)) {
                const auto s = nb::cast<std::string>(result);
                if (s == "RUNNING_MODAL")
                    return GizmoResult::Running;
                if (s == "FINISHED")
                    return GizmoResult::Finished;
                if (s == "CANCELLED")
                    return GizmoResult::Cancelled;
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Gizmo '{}' handle_mouse: {}", id, e.what());
        }

        return GizmoResult::PassThrough;
    }

    std::vector<std::string> PyGizmoRegistry::get_gizmo_ids() const {
        std::lock_guard lock(mutex_);
        std::vector<std::string> ids;
        ids.reserve(gizmos_.size());
        for (const auto& [id, _] : gizmos_)
            ids.push_back(id);
        return ids;
    }

    bool PyGizmoRegistry::has_gizmos() const {
        std::lock_guard lock(mutex_);
        return !gizmos_.empty();
    }

    PyTransformGizmo::PyTransformGizmo(const std::string& operation,
                                       const std::vector<float>& matrix,
                                       const std::string& id)
        : state_(std::make_shared<PyTransformGizmoState>()) {
        state_->instance_id = g_next_transform_gizmo_id.fetch_add(1, std::memory_order_relaxed);
        state_->id = id.empty() ? "transform_gizmo_" + std::to_string(state_->instance_id) : id;
        state_->operation = parse_transform_gizmo_operation(operation);
        state_->matrix = matrix_from_vector(matrix);
        state_->target_getter = nb::none();
        state_->target_setter = nb::none();
        state_->on_begin = nb::none();
        state_->on_change = nb::none();
        state_->on_end = nb::none();
    }

    PyTransformGizmo::PyTransformGizmo(std::shared_ptr<PyTransformGizmoState> state)
        : state_(std::move(state)) {
    }

    std::string PyTransformGizmo::operation() const {
        return transform_gizmo_operation_name(state_->operation);
    }

    void PyTransformGizmo::set_operation(const std::string& operation) {
        state_->operation = parse_transform_gizmo_operation(operation);
    }

    std::string PyTransformGizmo::space() const {
        return transform_gizmo_space_name(state_->space);
    }

    void PyTransformGizmo::set_space(const std::string& space) {
        state_->space = parse_transform_gizmo_space(space);
    }

    std::vector<float> PyTransformGizmo::matrix() const {
        return matrix_to_vector(state_->matrix);
    }

    void PyTransformGizmo::set_matrix(const std::vector<float>& matrix) {
        state_->matrix = matrix_from_vector(matrix);
    }

    std::vector<float> PyTransformGizmo::translation() const {
        const glm::vec3 t(state_->matrix[3]);
        return {t.x, t.y, t.z};
    }

    void PyTransformGizmo::set_translation(const std::vector<float>& translation) {
        const glm::vec3 t = vector3_from_vector(translation, "translation");
        state_->matrix[3] = glm::vec4(t, 1.0f);
    }

    void PyTransformGizmo::attach() {
        state_->target_kind = PyTransformGizmoState::TargetKind::None;
        state_->target_getter = nb::none();
        state_->target_setter = nb::none();
        state_->target_node_name.clear();
        state_->target_node_visualizer_world = true;
        PyTransformGizmoRegistry::instance().attach(state_);
        state_->attached = true;
    }

    void PyTransformGizmo::attach_to_callbacks(nb::object getter, nb::object setter) {
        if (!callable_or_none(getter))
            throw std::invalid_argument("TransformGizmo getter must be callable or None");
        if (!callable_or_none(setter))
            throw std::invalid_argument("TransformGizmo setter must be callable or None");

        state_->target_kind = PyTransformGizmoState::TargetKind::Callback;
        state_->target_getter = std::move(getter);
        state_->target_setter = std::move(setter);
        state_->target_node_name.clear();
        state_->target_node_visualizer_world = true;
        PyTransformGizmoRegistry::instance().attach(state_);
        state_->attached = true;
    }

    void PyTransformGizmo::attach_to_node(const std::string& node_name, bool visualizer_world) {
        state_->target_kind = PyTransformGizmoState::TargetKind::Node;
        state_->target_getter = nb::none();
        state_->target_setter = nb::none();
        state_->target_node_name = node_name;
        state_->target_node_visualizer_world = visualizer_world;
        PyTransformGizmoRegistry::instance().attach(state_);
        state_->attached = true;
    }

    void PyTransformGizmo::detach() {
        PyTransformGizmoRegistry::instance().detach(state_->instance_id);
        state_->attached = false;
        state_->active = false;
        state_->hovered = false;
        state_->changed = false;
    }

    void PyTransformGizmo::set_on_begin(nb::object callback) {
        if (!callable_or_none(callback))
            throw std::invalid_argument("TransformGizmo on_begin must be callable or None");
        state_->on_begin = std::move(callback);
    }

    void PyTransformGizmo::set_on_change(nb::object callback) {
        if (!callable_or_none(callback))
            throw std::invalid_argument("TransformGizmo on_change must be callable or None");
        state_->on_change = std::move(callback);
    }

    void PyTransformGizmo::set_on_end(nb::object callback) {
        if (!callable_or_none(callback))
            throw std::invalid_argument("TransformGizmo on_end must be callable or None");
        state_->on_end = std::move(callback);
    }

    glm::mat3 PyTransformGizmo::orientation_for_operation() const {
        if (state_->space == TransformGizmoSpace::World)
            return glm::mat3(1.0f);
        return extract_rotation(state_->matrix);
    }

    void PyTransformGizmo::sync_from_target_if_idle() {
        if (state_->active)
            return;

        if (state_->target_kind == PyTransformGizmoState::TargetKind::Callback) {
            if (!state_->target_getter.is_valid() || state_->target_getter.is_none())
                return;

            nb::gil_scoped_acquire gil;
            try {
                nb::object result = state_->target_getter();
                if (result.is_none())
                    return;
                state_->matrix = matrix_from_vector(nb::cast<std::vector<float>>(result));
            } catch (const std::exception& e) {
                LOG_ERROR("TransformGizmo '{}' target getter: {}", state_->id, e.what());
            }
            return;
        }

        if (state_->target_kind == PyTransformGizmoState::TargetKind::Node) {
            auto* sm = get_scene_manager();
            if (!sm)
                return;

            if (state_->target_node_visualizer_world) {
                const auto transform = vis::scene_coords::nodeVisualizerWorldTransform(sm->getScene(), state_->target_node_name);
                if (transform)
                    state_->matrix = *transform;
            } else if (sm->getScene().getNode(state_->target_node_name)) {
                state_->matrix = sm->getNodeTransform(state_->target_node_name);
            }
        }
    }

    void PyTransformGizmo::apply_to_target() {
        if (state_->target_kind == PyTransformGizmoState::TargetKind::Callback) {
            if (state_->target_setter.is_valid() && !state_->target_setter.is_none()) {
                nb::gil_scoped_acquire gil;
                try {
                    state_->target_setter(matrix_to_vector(state_->matrix));
                } catch (const std::exception& e) {
                    LOG_ERROR("TransformGizmo '{}' target setter: {}", state_->id, e.what());
                }
            }
        } else if (state_->target_kind == PyTransformGizmoState::TargetKind::Node) {
            auto* sm = get_scene_manager();
            if (sm && sm->getScene().getNode(state_->target_node_name)) {
                if (state_->target_node_visualizer_world) {
                    const auto local_transform =
                        vis::scene_coords::nodeLocalTransformFromVisualizerWorld(
                            sm->getScene(), state_->target_node_name, state_->matrix);
                    if (local_transform)
                        sm->setNodeTransform(state_->target_node_name, *local_transform);
                } else {
                    sm->setNodeTransform(state_->target_node_name, state_->matrix);
                }
                mark_scene_transform_changed();
            }
        }

        call_lifecycle_callback(state_->on_change);
    }

    void PyTransformGizmo::call_lifecycle_callback(const nb::object& callback) {
        if (!callback.is_valid() || callback.is_none())
            return;

        nb::gil_scoped_acquire gil;
        try {
            callback(nb::cast(PyTransformGizmo(state_)));
        } catch (const std::exception& e) {
            LOG_ERROR("TransformGizmo '{}' callback: {}", state_->id, e.what());
        }
    }

    void PyTransformGizmo::draw_frame(const glm::mat4& view,
                                      const glm::mat4& projection,
                                      const glm::vec2& viewport_pos,
                                      const glm::vec2& viewport_size,
                                      ImDrawList* draw_list) {
        state_->changed = false;
        state_->hovered = false;

        if (!state_->visible || !state_->enabled || !draw_list || viewport_size.x <= 1.0f || viewport_size.y <= 1.0f) {
            if (state_->active) {
                state_->active = false;
                call_lifecycle_callback(state_->on_end);
            }
            return;
        }

        sync_from_target_if_idle();

        const bool was_active = state_->active;
        bool active_now = false;
        bool changed_now = false;
        bool hovered_now = false;

        if (state_->operation == TransformGizmoOperation::Translate) {
            vis::gui::TranslationGizmoConfig config;
            config.id = state_->instance_id;
            config.viewport_pos = viewport_pos;
            config.viewport_size = viewport_size;
            config.view = view;
            config.projection = projection;
            config.pivot_world = glm::vec3(state_->matrix[3]);
            config.orientation_world = orientation_for_operation();
            config.draw_list = draw_list;
            config.input_enabled = state_->input_enabled;
            config.snap = state_->snap;
            config.snap_units = state_->translate_snap;

            const auto result = vis::gui::drawTranslationGizmo(config);
            active_now = result.active;
            hovered_now = result.hovered;
            changed_now = result.changed;
            if (result.changed) {
                state_->matrix[3] += glm::vec4(result.delta_translation, 0.0f);
            }
        } else if (state_->operation == TransformGizmoOperation::Rotate) {
            vis::gui::RotationGizmoConfig config;
            config.id = state_->instance_id;
            config.viewport_pos = viewport_pos;
            config.viewport_size = viewport_size;
            config.view = view;
            config.projection = projection;
            config.pivot_world = glm::vec3(state_->matrix[3]);
            config.orientation_world = orientation_for_operation();
            config.draw_list = draw_list;
            config.input_enabled = state_->input_enabled;
            config.snap = state_->snap;
            config.snap_degrees = state_->rotate_snap_degrees;

            const auto result = vis::gui::drawRotationGizmo(config);
            active_now = result.active;
            hovered_now = result.hovered;
            changed_now = result.changed;
            if (result.changed) {
                const glm::vec3 pivot(state_->matrix[3]);
                state_->matrix = glm::translate(glm::mat4(1.0f), pivot) *
                                 glm::mat4(result.delta_rotation) *
                                 glm::translate(glm::mat4(1.0f), -pivot) *
                                 state_->matrix;
            }
        } else {
            vis::gui::ScaleGizmoConfig config;
            config.id = state_->instance_id;
            config.viewport_pos = viewport_pos;
            config.viewport_size = viewport_size;
            config.view = view;
            config.projection = projection;
            config.pivot_world = glm::vec3(state_->matrix[3]);
            config.orientation_world = orientation_for_operation();
            config.draw_list = draw_list;
            config.input_enabled = state_->input_enabled;
            config.snap = state_->snap;
            config.snap_ratio = state_->scale_snap_ratio;

            const auto result = vis::gui::drawScaleGizmo(config);
            active_now = result.active;
            hovered_now = result.hovered;
            changed_now = result.changed;
            if (result.changed) {
                if (state_->space == TransformGizmoSpace::World) {
                    const glm::vec3 pivot(state_->matrix[3]);
                    state_->matrix = glm::translate(glm::mat4(1.0f), pivot) *
                                     glm::scale(glm::mat4(1.0f), result.delta_scale) *
                                     glm::translate(glm::mat4(1.0f), -pivot) *
                                     state_->matrix;
                } else {
                    state_->matrix[0] *= result.delta_scale.x;
                    state_->matrix[1] *= result.delta_scale.y;
                    state_->matrix[2] *= result.delta_scale.z;
                }
            }
        }

        if ((hovered_now || active_now) && state_->input_enabled) {
            vis::gui::guiFocusState().want_capture_mouse = true;
        }

        if (active_now && !was_active) {
            state_->active = true;
            call_lifecycle_callback(state_->on_begin);
        }

        state_->hovered = hovered_now;
        state_->changed = changed_now;

        if (changed_now) {
            apply_to_target();
        }

        if (!active_now && was_active) {
            state_->active = false;
            call_lifecycle_callback(state_->on_end);
        } else {
            state_->active = active_now;
        }
    }

    PyTransformGizmoRegistry& PyTransformGizmoRegistry::instance() {
        static PyTransformGizmoRegistry registry;
        return registry;
    }

    void PyTransformGizmoRegistry::attach(std::shared_ptr<PyTransformGizmoState> state) {
        if (!state)
            return;
        std::lock_guard lock(mutex_);
        gizmos_[state->instance_id] = std::move(state);
    }

    void PyTransformGizmoRegistry::detach(int instance_id) {
        std::lock_guard lock(mutex_);
        gizmos_.erase(instance_id);
    }

    void PyTransformGizmoRegistry::clear_all() {
        std::vector<std::shared_ptr<PyTransformGizmoState>> states;
        {
            std::lock_guard lock(mutex_);
            states.reserve(gizmos_.size());
            for (auto& [_, state] : gizmos_)
                states.push_back(state);
            gizmos_.clear();
        }
        for (auto& state : states) {
            state->attached = false;
            state->active = false;
            state->hovered = false;
            state->changed = false;
        }
    }

    bool PyTransformGizmoRegistry::has_attached() const {
        std::lock_guard lock(mutex_);
        return !gizmos_.empty();
    }

    std::vector<std::string> PyTransformGizmoRegistry::ids() const {
        std::lock_guard lock(mutex_);
        std::vector<std::string> result;
        result.reserve(gizmos_.size());
        for (const auto& [_, state] : gizmos_)
            result.push_back(state->id);
        return result;
    }

    void PyTransformGizmoRegistry::draw_all(const glm::mat4& view,
                                            const glm::mat4& projection,
                                            const glm::vec2& viewport_pos,
                                            const glm::vec2& viewport_size,
                                            ImDrawList* draw_list) {
        std::vector<std::shared_ptr<PyTransformGizmoState>> states;
        {
            std::lock_guard lock(mutex_);
            states.reserve(gizmos_.size());
            for (auto& [_, state] : gizmos_)
                states.push_back(state);
        }

        for (const auto& state : states) {
            PyTransformGizmo(state).draw_frame(view, projection, viewport_pos, viewport_size, draw_list);
        }
    }

    void register_gizmos(nb::module_& m) {
        nb::enum_<GizmoEventType>(m, "GizmoEventType")
            .value("PRESS", GizmoEventType::Press)
            .value("RELEASE", GizmoEventType::Release)
            .value("MOVE", GizmoEventType::Move)
            .value("DRAG", GizmoEventType::Drag);

        nb::enum_<GizmoResult>(m, "GizmoResult")
            .value("PASS_THROUGH", GizmoResult::PassThrough)
            .value("RUNNING_MODAL", GizmoResult::Running)
            .value("FINISHED", GizmoResult::Finished)
            .value("CANCELLED", GizmoResult::Cancelled);

        nb::enum_<TransformGizmoOperation>(m, "TransformGizmoOperation")
            .value("TRANSLATE", TransformGizmoOperation::Translate)
            .value("ROTATE", TransformGizmoOperation::Rotate)
            .value("SCALE", TransformGizmoOperation::Scale);

        nb::enum_<TransformGizmoSpace>(m, "TransformGizmoSpace")
            .value("LOCAL", TransformGizmoSpace::Local)
            .value("WORLD", TransformGizmoSpace::World);

        nb::class_<PyGizmoContext>(m, "GizmoContext")
            .def(nb::init<>())
            .def_prop_ro("has_selection", &PyGizmoContext::has_selection, "Whether any gaussians are selected")
            .def_prop_ro("selection_center", &PyGizmoContext::selection_center, "Selection center in visualizer-world space (x, y, z)")
            .def_prop_ro("selection_center_screen", &PyGizmoContext::selection_center_screen, "Selection center in screen space (x, y)")
            .def_prop_ro("camera_position", &PyGizmoContext::camera_position, "Camera position in visualizer-world space (x, y, z)")
            .def_prop_ro("camera_forward", &PyGizmoContext::camera_forward, "Camera forward direction (x, y, z)")
            .def("world_to_screen", &PyGizmoContext::world_to_screen, nb::arg("pos"), "Project visualizer-world position to screen coordinates")
            .def("screen_to_world_ray", &PyGizmoContext::screen_to_world_ray, nb::arg("pos"), "Get visualizer-world ray direction from screen point")
            .def("draw_line", &PyGizmoContext::draw_line_2d, nb::arg("start"), nb::arg("end"),
                 nb::arg("color"), nb::arg("thickness") = 1.0f, "Draw a 2D line")
            .def("draw_circle", &PyGizmoContext::draw_circle_2d, nb::arg("center"), nb::arg("radius"),
                 nb::arg("color"), nb::arg("thickness") = 1.0f, "Draw a 2D circle outline")
            .def("draw_rect", &PyGizmoContext::draw_rect_2d, nb::arg("min"), nb::arg("max"),
                 nb::arg("color"), nb::arg("thickness") = 1.0f, "Draw a 2D rectangle outline")
            .def("draw_filled_rect", &PyGizmoContext::draw_filled_rect_2d, nb::arg("min"), nb::arg("max"),
                 nb::arg("color"), "Draw a filled 2D rectangle")
            .def("draw_filled_circle", &PyGizmoContext::draw_filled_circle_2d, nb::arg("center"),
                 nb::arg("radius"), nb::arg("color"), "Draw a filled 2D circle")
            .def("draw_line_3d", &PyGizmoContext::draw_line_3d, nb::arg("start"), nb::arg("end"),
                 nb::arg("color"), nb::arg("thickness") = 1.0f, "Draw a 3D line");

        nb::class_<PyTransformGizmo>(m, "TransformGizmo")
            .def(nb::init<const std::string&, const std::vector<float>&, const std::string&>(),
                 nb::arg("operation") = "translate",
                 nb::arg("matrix") = std::vector<float>{},
                 nb::arg("id") = "",
                 "Create a reusable native TRS viewport gizmo")
            .def_prop_ro("id", &PyTransformGizmo::id, "Stable gizmo id")
            .def_prop_rw("operation", &PyTransformGizmo::operation, &PyTransformGizmo::set_operation,
                         "Gizmo operation: 'translate', 'rotate', or 'scale'")
            .def_prop_rw("space", &PyTransformGizmo::space, &PyTransformGizmo::set_space,
                         "Axis space: 'local' or 'world'")
            .def_prop_rw("matrix", &PyTransformGizmo::matrix, &PyTransformGizmo::set_matrix,
                         "4x4 transform matrix as 16 column-major floats")
            .def_prop_rw("translation", &PyTransformGizmo::translation, &PyTransformGizmo::set_translation,
                         "Translation component as (x, y, z)")
            .def_prop_ro("attached", &PyTransformGizmo::attached, "Whether the gizmo is registered for viewport drawing")
            .def_prop_rw("visible", &PyTransformGizmo::visible, &PyTransformGizmo::set_visible,
                         "Whether the gizmo is drawn")
            .def_prop_rw("enabled", &PyTransformGizmo::enabled, &PyTransformGizmo::set_enabled,
                         "Whether the gizmo updates and handles lifecycle state")
            .def_prop_rw("input_enabled", &PyTransformGizmo::input_enabled, &PyTransformGizmo::set_input_enabled,
                         "Whether the gizmo accepts mouse input")
            .def_prop_ro("active", &PyTransformGizmo::active, "Whether the gizmo is currently being dragged")
            .def_prop_ro("hovered", &PyTransformGizmo::hovered, "Whether a gizmo handle was hovered last frame")
            .def_prop_ro("changed", &PyTransformGizmo::changed, "Whether the gizmo changed its matrix last frame")
            .def_prop_rw("snap", &PyTransformGizmo::snap, &PyTransformGizmo::set_snap,
                         "Enable operation snapping")
            .def_prop_rw("translate_snap", &PyTransformGizmo::translate_snap, &PyTransformGizmo::set_translate_snap,
                         "Translate snap step in world units")
            .def_prop_rw("rotate_snap_degrees", &PyTransformGizmo::rotate_snap_degrees,
                         &PyTransformGizmo::set_rotate_snap_degrees,
                         "Rotation snap step in degrees")
            .def_prop_rw("scale_snap_ratio", &PyTransformGizmo::scale_snap_ratio,
                         &PyTransformGizmo::set_scale_snap_ratio,
                         "Scale snap step as a ratio")
            .def("attach", &PyTransformGizmo::attach,
                 "Attach the gizmo to the viewport overlay without an automatic target")
            .def("attach_to_callbacks", &PyTransformGizmo::attach_to_callbacks,
                 nb::arg("getter"), nb::arg("setter"),
                 "Attach to arbitrary Python get/set transform callbacks")
            .def("attach_to_node", &PyTransformGizmo::attach_to_node,
                 nb::arg("node_name"), nb::arg("visualizer_world") = true,
                 "Attach to a scene node transform")
            .def("detach", &PyTransformGizmo::detach,
                 "Detach the gizmo from the viewport overlay")
            .def("set_on_begin", &PyTransformGizmo::set_on_begin, nb::arg("callback"),
                 "Set a callback called with this gizmo when dragging begins")
            .def("set_on_change", &PyTransformGizmo::set_on_change, nb::arg("callback"),
                 "Set a callback called with this gizmo after its matrix changes")
            .def("set_on_end", &PyTransformGizmo::set_on_end, nb::arg("callback"),
                 "Set a callback called with this gizmo when dragging ends");

        m.attr("TRSGizmo") = m.attr("TransformGizmo");

        m.def(
            "TranslationGizmo",
            [](const std::vector<float>& matrix, const std::string& id) {
                return PyTransformGizmo("translate", matrix, id);
            },
            nb::arg("matrix") = std::vector<float>{}, nb::arg("id") = "",
            "Create a TransformGizmo configured for translation");
        m.def(
            "RotationGizmo",
            [](const std::vector<float>& matrix, const std::string& id) {
                return PyTransformGizmo("rotate", matrix, id);
            },
            nb::arg("matrix") = std::vector<float>{}, nb::arg("id") = "",
            "Create a TransformGizmo configured for rotation");
        m.def(
            "ScaleGizmo",
            [](const std::vector<float>& matrix, const std::string& id) {
                return PyTransformGizmo("scale", matrix, id);
            },
            nb::arg("matrix") = std::vector<float>{}, nb::arg("id") = "",
            "Create a TransformGizmo configured for scale");
        m.def(
            "clear_transform_gizmos", []() { PyTransformGizmoRegistry::instance().clear_all(); },
            "Detach all native transform gizmos");
        m.def(
            "get_transform_gizmo_ids", []() { return PyTransformGizmoRegistry::instance().ids(); },
            "Get ids of attached native transform gizmos");
        m.def(
            "has_transform_gizmos", []() { return PyTransformGizmoRegistry::instance().has_attached(); },
            "Check whether native transform gizmos are attached");

        m.def(
            "register_gizmo", [](nb::object cls) { PyGizmoRegistry::instance().register_gizmo(cls); },
            nb::arg("gizmo_class"),
            "Register a gizmo class for viewport overlay drawing");
        m.def(
            "unregister_gizmo", [](const std::string& id) { PyGizmoRegistry::instance().unregister_gizmo(id); },
            nb::arg("id"),
            "Unregister a gizmo by ID");
        m.def(
            "unregister_all_gizmos", []() { PyGizmoRegistry::instance().unregister_all(); },
            "Unregister all gizmos");
        m.def(
            "get_gizmo_ids", []() { return PyGizmoRegistry::instance().get_gizmo_ids(); },
            "Get all registered gizmo IDs");
        m.def(
            "has_gizmos", []() { return PyGizmoRegistry::instance().has_gizmos(); },
            "Check if any gizmos are registered");
    }

} // namespace lfs::python
