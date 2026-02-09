/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_viewport.hpp"
#include "core/logger.hpp"
#include "python/python_runtime.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>

namespace lfs::python {

    namespace {
        constexpr float DEFAULT_VIEWPORT_WIDTH = 800.0f;
        constexpr float DEFAULT_VIEWPORT_HEIGHT = 600.0f;
        constexpr float DEFAULT_CAMERA_Z = 5.0f;
        constexpr float PROJECTION_SCALE = 100.0f;

        struct Color {
            float r, g, b, a;
        };

        Color parse_color(const nb::object& obj) {
            const auto len = nb::len(obj);
            assert(len == 3 || len == 4);
            if (len == 3) {
                auto c = nb::cast<std::tuple<float, float, float>>(obj);
                return {std::get<0>(c), std::get<1>(c), std::get<2>(c), 1.0f};
            }
            if (len == 4) {
                auto c = nb::cast<std::tuple<float, float, float, float>>(obj);
                return {std::get<0>(c), std::get<1>(c), std::get<2>(c), std::get<3>(c)};
            }
            throw nb::type_error("color must be an RGB (r, g, b) or RGBA (r, g, b, a) tuple");
        }

        DrawHandlerTiming parse_timing(const nb::object& obj) {
            if (nb::isinstance<DrawHandlerTiming>(obj))
                return nb::cast<DrawHandlerTiming>(obj);

            auto timing_str = nb::cast<std::string>(obj);
            if (timing_str == "PRE_VIEW")
                return DrawHandlerTiming::PreView;
            if (timing_str == "POST_UI")
                return DrawHandlerTiming::PostUI;
            return DrawHandlerTiming::PostView;
        }
    } // namespace

    std::optional<std::tuple<float, float>> PyViewportDrawContext::world_to_screen(std::tuple<float, float, float> pos) const {
        const auto [wx, wy, wz] = pos;

        if (has_camera_state_) {
            const glm::vec4 clip = proj_matrix_ * view_matrix_ * glm::vec4(wx, wy, wz, 1.0f);
            if (clip.w <= 0.0f)
                return std::nullopt;
            const glm::vec3 ndc = glm::vec3(clip) / clip.w;
            const float sx = viewport_pos_.x + (ndc.x * 0.5f + 0.5f) * viewport_size_.x;
            const float sy = viewport_pos_.y + (1.0f - (ndc.y * 0.5f + 0.5f)) * viewport_size_.y;
            return std::make_tuple(sx, sy);
        }

        if (wz <= 0.0f)
            return std::nullopt;
        const float sx = DEFAULT_VIEWPORT_WIDTH / 2.0f + wx * PROJECTION_SCALE / wz;
        const float sy = DEFAULT_VIEWPORT_HEIGHT / 2.0f - wy * PROJECTION_SCALE / wz;
        return std::make_tuple(sx, sy);
    }

    std::tuple<float, float, float> PyViewportDrawContext::screen_to_world_ray(std::tuple<float, float> screen_pos) const {
        const auto [sx, sy] = screen_pos;

        if (has_camera_state_) {
            const float ndc_x = ((sx - viewport_pos_.x) / viewport_size_.x) * 2.0f - 1.0f;
            const float ndc_y = 1.0f - ((sy - viewport_pos_.y) / viewport_size_.y) * 2.0f;
            const glm::mat4 inv_proj = glm::inverse(proj_matrix_);
            const glm::mat4 inv_view = glm::inverse(view_matrix_);
            const glm::vec4 clip_near(ndc_x, ndc_y, -1.0f, 1.0f);
            glm::vec4 eye = inv_proj * clip_near;
            eye /= eye.w;
            eye = glm::vec4(eye.x, eye.y, eye.z, 0.0f);
            const glm::vec3 world_dir = glm::normalize(glm::vec3(inv_view * eye));
            return {world_dir.x, world_dir.y, world_dir.z};
        }

        const float dx = (sx - DEFAULT_VIEWPORT_WIDTH / 2.0f) / (DEFAULT_VIEWPORT_WIDTH / 2.0f);
        const float dy = -(sy - DEFAULT_VIEWPORT_HEIGHT / 2.0f) / (DEFAULT_VIEWPORT_HEIGHT / 2.0f);
        const float len = std::sqrt(dx * dx + dy * dy + 1.0f);
        return {dx / len, dy / len, -1.0f / len};
    }

    std::tuple<float, float, float> PyViewportDrawContext::camera_position() const {
        if (has_camera_state_)
            return {camera_pos_.x, camera_pos_.y, camera_pos_.z};
        return {0.0f, 0.0f, DEFAULT_CAMERA_Z};
    }

    std::tuple<float, float, float> PyViewportDrawContext::camera_forward() const {
        if (has_camera_state_)
            return {camera_fwd_.x, camera_fwd_.y, camera_fwd_.z};
        return {0.0f, 0.0f, -1.0f};
    }

    std::tuple<float, float> PyViewportDrawContext::viewport_size() const {
        if (has_camera_state_)
            return {viewport_size_.x, viewport_size_.y};
        return {DEFAULT_VIEWPORT_WIDTH, DEFAULT_VIEWPORT_HEIGHT};
    }

    void PyViewportDrawContext::draw_line_2d(std::tuple<float, float> start, std::tuple<float, float> end,
                                             nb::object color, float thickness) {
        const auto c = parse_color(color);
        draw_commands_.push_back({DrawCommand::LINE_2D,
                                  std::get<0>(start),
                                  std::get<1>(start),
                                  0.0f,
                                  std::get<0>(end),
                                  std::get<1>(end),
                                  0.0f,
                                  c.r,
                                  c.g,
                                  c.b,
                                  c.a,
                                  thickness,
                                  0.0f,
                                  0.0f,
                                  {}});
    }

    void PyViewportDrawContext::draw_circle_2d(std::tuple<float, float> center, float radius,
                                               nb::object color, float thickness) {
        const auto c = parse_color(color);
        draw_commands_.push_back({DrawCommand::CIRCLE_2D,
                                  std::get<0>(center),
                                  std::get<1>(center),
                                  0.0f,
                                  0.0f,
                                  0.0f,
                                  0.0f,
                                  c.r,
                                  c.g,
                                  c.b,
                                  c.a,
                                  thickness,
                                  radius,
                                  0.0f,
                                  {}});
    }

    void PyViewportDrawContext::draw_rect_2d(std::tuple<float, float> min, std::tuple<float, float> max,
                                             nb::object color, float thickness) {
        const auto c = parse_color(color);
        draw_commands_.push_back({DrawCommand::RECT_2D,
                                  std::get<0>(min),
                                  std::get<1>(min),
                                  0.0f,
                                  std::get<0>(max),
                                  std::get<1>(max),
                                  0.0f,
                                  c.r,
                                  c.g,
                                  c.b,
                                  c.a,
                                  thickness,
                                  0.0f,
                                  0.0f,
                                  {}});
    }

    void PyViewportDrawContext::draw_filled_rect_2d(std::tuple<float, float> min, std::tuple<float, float> max,
                                                    nb::object color) {
        const auto c = parse_color(color);
        draw_commands_.push_back({DrawCommand::FILLED_RECT_2D,
                                  std::get<0>(min),
                                  std::get<1>(min),
                                  0.0f,
                                  std::get<0>(max),
                                  std::get<1>(max),
                                  0.0f,
                                  c.r,
                                  c.g,
                                  c.b,
                                  c.a,
                                  0.0f,
                                  0.0f,
                                  0.0f,
                                  {}});
    }

    void PyViewportDrawContext::draw_filled_circle_2d(std::tuple<float, float> center, float radius,
                                                      nb::object color) {
        const auto c = parse_color(color);
        draw_commands_.push_back({DrawCommand::FILLED_CIRCLE_2D,
                                  std::get<0>(center),
                                  std::get<1>(center),
                                  0.0f,
                                  0.0f,
                                  0.0f,
                                  0.0f,
                                  c.r,
                                  c.g,
                                  c.b,
                                  c.a,
                                  0.0f,
                                  radius,
                                  0.0f,
                                  {}});
    }

    void PyViewportDrawContext::draw_text_2d(std::tuple<float, float> pos, const std::string& text,
                                             nb::object color, float font_size) {
        const auto c = parse_color(color);
        draw_commands_.push_back({DrawCommand::TEXT_2D,
                                  std::get<0>(pos), std::get<1>(pos), 0.0f,
                                  0.0f, 0.0f, 0.0f,
                                  c.r, c.g, c.b, c.a,
                                  0.0f, 0.0f, font_size, text});
    }

    void PyViewportDrawContext::draw_line_3d(std::tuple<float, float, float> start, std::tuple<float, float, float> end,
                                             nb::object color, float thickness) {
        const auto c = parse_color(color);
        draw_commands_.push_back({DrawCommand::LINE_3D,
                                  std::get<0>(start),
                                  std::get<1>(start),
                                  std::get<2>(start),
                                  std::get<0>(end),
                                  std::get<1>(end),
                                  std::get<2>(end),
                                  c.r,
                                  c.g,
                                  c.b,
                                  c.a,
                                  thickness,
                                  0.0f,
                                  0.0f,
                                  {}});
    }

    void PyViewportDrawContext::draw_point_3d(std::tuple<float, float, float> pos,
                                              nb::object color, float size) {
        const auto c = parse_color(color);
        draw_commands_.push_back({DrawCommand::POINT_3D,
                                  std::get<0>(pos),
                                  std::get<1>(pos),
                                  std::get<2>(pos),
                                  0.0f,
                                  0.0f,
                                  0.0f,
                                  c.r,
                                  c.g,
                                  c.b,
                                  c.a,
                                  0.0f,
                                  size,
                                  0.0f,
                                  {}});
    }

    void PyViewportDrawContext::draw_text_3d(std::tuple<float, float, float> pos, const std::string& text,
                                             nb::object color, float font_size) {
        const auto c = parse_color(color);
        draw_commands_.push_back({DrawCommand::TEXT_3D,
                                  std::get<0>(pos), std::get<1>(pos), std::get<2>(pos),
                                  0.0f, 0.0f, 0.0f,
                                  c.r, c.g, c.b, c.a,
                                  0.0f, 0.0f, font_size, text});
    }

    void PyViewportDrawContext::set_camera_state(const glm::mat4& view, const glm::mat4& proj,
                                                 const glm::vec2& viewport_pos, const glm::vec2& viewport_size,
                                                 const glm::vec3& camera_pos, const glm::vec3& camera_fwd) {
        view_matrix_ = view;
        proj_matrix_ = proj;
        viewport_pos_ = viewport_pos;
        viewport_size_ = viewport_size;
        camera_pos_ = camera_pos;
        camera_fwd_ = camera_fwd;
        has_camera_state_ = true;
    }

    PyViewportDrawRegistry& PyViewportDrawRegistry::instance() {
        static PyViewportDrawRegistry registry;
        return registry;
    }

    void PyViewportDrawRegistry::add_handler(const std::string& id, nb::object callback, DrawHandlerTiming timing) {
        std::lock_guard lock(mutex_);
        handlers_.erase(
            std::remove_if(handlers_.begin(), handlers_.end(),
                           [&id](const PyDrawHandlerInfo& h) { return h.id == id; }),
            handlers_.end());
        handlers_.push_back({id, std::move(callback), timing});
    }

    void PyViewportDrawRegistry::remove_handler(const std::string& id) {
        std::lock_guard lock(mutex_);
        handlers_.erase(
            std::remove_if(handlers_.begin(), handlers_.end(),
                           [&id](const PyDrawHandlerInfo& h) { return h.id == id; }),
            handlers_.end());
    }

    void PyViewportDrawRegistry::clear_all() {
        std::lock_guard lock(mutex_);
        handlers_.clear();
    }

    void PyViewportDrawRegistry::invoke_handlers(DrawHandlerTiming timing, PyViewportDrawContext& ctx) {
        std::vector<nb::object> callbacks;
        {
            std::lock_guard lock(mutex_);
            for (const auto& handler : handlers_) {
                if (handler.timing == timing) {
                    callbacks.push_back(handler.callback);
                }
            }
        }

        if (callbacks.empty())
            return;

        nb::gil_scoped_acquire gil;
        nb::object py_ctx = nb::cast(ctx, nb::rv_policy::reference);
        for (const auto& cb : callbacks) {
            try {
                cb(py_ctx);
            } catch (const std::exception& e) {
                LOG_ERROR("Viewport draw handler error: {}", e.what());
            }
        }
    }

    std::vector<std::string> PyViewportDrawRegistry::get_handler_ids() const {
        std::lock_guard lock(mutex_);
        std::vector<std::string> ids;
        ids.reserve(handlers_.size());
        for (const auto& h : handlers_) {
            ids.push_back(h.id);
        }
        return ids;
    }

    bool PyViewportDrawRegistry::has_handlers() const {
        std::lock_guard lock(mutex_);
        return !handlers_.empty();
    }

    bool PyViewportDrawRegistry::has_handler(const std::string& id) const {
        std::lock_guard lock(mutex_);
        return std::any_of(handlers_.begin(), handlers_.end(),
                           [&id](const PyDrawHandlerInfo& h) { return h.id == id; });
    }

    void register_viewport(nb::module_& m) {
        nb::enum_<DrawHandlerTiming>(m, "DrawHandlerTiming")
            .value("PRE_VIEW", DrawHandlerTiming::PreView)
            .value("POST_VIEW", DrawHandlerTiming::PostView)
            .value("POST_UI", DrawHandlerTiming::PostUI);

        nb::class_<PyViewportDrawContext>(m, "ViewportDrawContext")
            .def("world_to_screen", &PyViewportDrawContext::world_to_screen, nb::arg("pos"),
                 "Project a (x, y, z) world position to (sx, sy) screen coordinates")
            .def("screen_to_world_ray", &PyViewportDrawContext::screen_to_world_ray, nb::arg("screen_pos"),
                 "Convert (sx, sy) screen position to a normalized world-space ray direction")
            .def_prop_ro("camera_position", &PyViewportDrawContext::camera_position,
                         "Camera position as (x, y, z)")
            .def_prop_ro("camera_forward", &PyViewportDrawContext::camera_forward,
                         "Camera forward direction as (x, y, z)")
            .def_prop_ro("viewport_size", &PyViewportDrawContext::viewport_size,
                         "Viewport dimensions as (width, height)")
            .def("draw_line_2d", &PyViewportDrawContext::draw_line_2d, nb::arg("start"), nb::arg("end"),
                 nb::arg("color"), nb::arg("thickness") = 1.0f,
                 "Draw a 2D line from start to end in screen coordinates")
            .def("draw_circle_2d", &PyViewportDrawContext::draw_circle_2d, nb::arg("center"), nb::arg("radius"),
                 nb::arg("color"), nb::arg("thickness") = 1.0f,
                 "Draw a 2D circle outline in screen coordinates")
            .def("draw_rect_2d", &PyViewportDrawContext::draw_rect_2d, nb::arg("min"), nb::arg("max"),
                 nb::arg("color"), nb::arg("thickness") = 1.0f,
                 "Draw a 2D rectangle outline in screen coordinates")
            .def("draw_filled_rect_2d", &PyViewportDrawContext::draw_filled_rect_2d, nb::arg("min"), nb::arg("max"),
                 nb::arg("color"),
                 "Draw a filled 2D rectangle in screen coordinates")
            .def("draw_filled_circle_2d", &PyViewportDrawContext::draw_filled_circle_2d, nb::arg("center"),
                 nb::arg("radius"), nb::arg("color"),
                 "Draw a filled 2D circle in screen coordinates")
            .def("draw_text_2d", &PyViewportDrawContext::draw_text_2d, nb::arg("pos"), nb::arg("text"),
                 nb::arg("color"), nb::arg("font_size") = 0.0f,
                 "Draw text at a 2D screen position (font_size=0 uses default)")
            .def("draw_line_3d", &PyViewportDrawContext::draw_line_3d, nb::arg("start"), nb::arg("end"),
                 nb::arg("color"), nb::arg("thickness") = 1.0f,
                 "Draw a 3D line between two world-space points")
            .def("draw_point_3d", &PyViewportDrawContext::draw_point_3d, nb::arg("pos"),
                 nb::arg("color"), nb::arg("size") = 4.0f,
                 "Draw a point at a 3D world-space position")
            .def("draw_text_3d", &PyViewportDrawContext::draw_text_3d, nb::arg("pos"), nb::arg("text"),
                 nb::arg("color"), nb::arg("font_size") = 0.0f,
                 "Draw text at a 3D world-space position (font_size=0 uses default)");

        m.def(
            "draw_handler",
            [](const std::string& timing_str) {
                DrawHandlerTiming timing = DrawHandlerTiming::PostView;
                if (timing_str == "PRE_VIEW")
                    timing = DrawHandlerTiming::PreView;
                else if (timing_str == "POST_UI")
                    timing = DrawHandlerTiming::PostUI;

                static std::atomic<int> handler_counter{0};
                return nb::cpp_function([timing](nb::object func) {
                    std::string id = "draw_handler_" + std::to_string(++handler_counter);
                    PyViewportDrawRegistry::instance().add_handler(id, func, timing);
                    return func;
                });
            },
            nb::arg("timing") = "POST_VIEW",
            "Decorator to register a viewport draw handler (PRE_VIEW, POST_VIEW, POST_UI)");

        m.def(
            "add_draw_handler",
            [](const std::string& id, nb::object callback, nb::object timing_obj) {
                PyViewportDrawRegistry::instance().add_handler(id, callback, parse_timing(timing_obj));
            },
            nb::arg("id"), nb::arg("callback"), nb::arg("timing") = "POST_VIEW",
            "Add a viewport draw handler with explicit id");

        m.def(
            "remove_draw_handler", [](const std::string& id) -> bool {
                auto& reg = PyViewportDrawRegistry::instance();
                if (!reg.has_handler(id))
                    return false;
                reg.remove_handler(id);
                return true;
            },
            nb::arg("id"), "Remove a viewport draw handler (returns false if not found)");

        m.def(
            "clear_draw_handlers", []() {
                PyViewportDrawRegistry::instance().clear_all();
            },
            "Clear all viewport draw handlers");

        m.def(
            "get_draw_handler_ids", []() {
                return PyViewportDrawRegistry::instance().get_handler_ids();
            },
            "Get list of registered draw handler ids");

        m.def(
            "has_draw_handlers", []() {
                return PyViewportDrawRegistry::instance().has_handlers();
            },
            "Check if any draw handlers are registered");

        m.def(
            "has_draw_handler", [](const std::string& id) {
                return PyViewportDrawRegistry::instance().has_handler(id);
            },
            nb::arg("id"), "Check if a specific draw handler exists");
    }

} // namespace lfs::python
