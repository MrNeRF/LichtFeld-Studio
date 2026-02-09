/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <glm/glm.hpp>

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;

namespace lfs::python {

    enum class DrawHandlerTiming { PreView,
                                   PostView,
                                   PostUI };

    class PyViewportDrawContext {
    public:
        struct DrawCommand {
            enum Type { LINE_2D,
                        CIRCLE_2D,
                        RECT_2D,
                        FILLED_RECT_2D,
                        FILLED_CIRCLE_2D,
                        TEXT_2D,
                        LINE_3D,
                        POINT_3D,
                        TEXT_3D };
            Type type;
            float x1, y1, z1;
            float x2, y2, z2;
            float r, g, b, a;
            float thickness;
            float radius;
            float font_size;
            std::string text;
        };

        [[nodiscard]] std::optional<std::tuple<float, float>> world_to_screen(std::tuple<float, float, float> pos) const;
        [[nodiscard]] std::tuple<float, float, float> screen_to_world_ray(std::tuple<float, float> screen_pos) const;
        [[nodiscard]] std::tuple<float, float, float> camera_position() const;
        [[nodiscard]] std::tuple<float, float, float> camera_forward() const;
        [[nodiscard]] std::tuple<float, float> viewport_size() const;

        void draw_line_2d(std::tuple<float, float> start, std::tuple<float, float> end,
                          nb::object color, float thickness = 1.0f);
        void draw_circle_2d(std::tuple<float, float> center, float radius,
                            nb::object color, float thickness = 1.0f);
        void draw_rect_2d(std::tuple<float, float> min, std::tuple<float, float> max,
                          nb::object color, float thickness = 1.0f);
        void draw_filled_rect_2d(std::tuple<float, float> min, std::tuple<float, float> max,
                                 nb::object color);
        void draw_filled_circle_2d(std::tuple<float, float> center, float radius,
                                   nb::object color);
        void draw_text_2d(std::tuple<float, float> pos, const std::string& text,
                          nb::object color, float font_size = 0.0f);
        void draw_line_3d(std::tuple<float, float, float> start, std::tuple<float, float, float> end,
                          nb::object color, float thickness = 1.0f);
        void draw_point_3d(std::tuple<float, float, float> pos,
                           nb::object color, float size = 4.0f);
        void draw_text_3d(std::tuple<float, float, float> pos, const std::string& text,
                          nb::object color, float font_size = 0.0f);

        void set_camera_state(const glm::mat4& view, const glm::mat4& proj,
                              const glm::vec2& viewport_pos, const glm::vec2& viewport_size,
                              const glm::vec3& camera_pos, const glm::vec3& camera_fwd);

        [[nodiscard]] const std::vector<DrawCommand>& get_draw_commands() const { return draw_commands_; }
        void clear_draw_commands() { draw_commands_.clear(); }

    private:
        mutable std::vector<DrawCommand> draw_commands_;

        glm::mat4 view_matrix_{1.0f};
        glm::mat4 proj_matrix_{1.0f};
        glm::vec2 viewport_pos_{0.0f};
        glm::vec2 viewport_size_{800.0f, 600.0f};
        glm::vec3 camera_pos_{0.0f, 0.0f, 5.0f};
        glm::vec3 camera_fwd_{0.0f, 0.0f, -1.0f};
        bool has_camera_state_ = false;
    };

    struct PyDrawHandlerInfo {
        std::string id;
        nb::object callback;
        DrawHandlerTiming timing;
    };

    class PyViewportDrawRegistry {
    public:
        static PyViewportDrawRegistry& instance();

        void add_handler(const std::string& id, nb::object callback, DrawHandlerTiming timing);
        void remove_handler(const std::string& id);
        void clear_all();
        void invoke_handlers(DrawHandlerTiming timing, PyViewportDrawContext& ctx);

        [[nodiscard]] std::vector<std::string> get_handler_ids() const;
        [[nodiscard]] bool has_handlers() const;
        [[nodiscard]] bool has_handler(const std::string& id) const;

    private:
        PyViewportDrawRegistry() = default;
        PyViewportDrawRegistry(const PyViewportDrawRegistry&) = delete;
        PyViewportDrawRegistry& operator=(const PyViewportDrawRegistry&) = delete;

        mutable std::mutex mutex_;
        std::vector<PyDrawHandlerInfo> handlers_;
    };

    void register_viewport(nb::module_& m);

} // namespace lfs::python
