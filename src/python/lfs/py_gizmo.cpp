/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_gizmo.hpp"
#include "core/logger.hpp"

#include <cmath>

namespace lfs::python {

    namespace {
        constexpr float DEFAULT_VIEWPORT_WIDTH = 800.0f;
        constexpr float DEFAULT_VIEWPORT_HEIGHT = 600.0f;
        constexpr float DEFAULT_CAMERA_Z = 5.0f;
        constexpr float PROJECTION_SCALE = 100.0f;
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
