/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

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
                        POINT_3D };
            Type type;
            float x1, y1, z1;
            float x2, y2, z2;
            float r, g, b, a;
            float thickness;
            float radius;
            std::string text;
        };

        [[nodiscard]] std::optional<std::tuple<float, float>> world_to_screen(std::tuple<float, float, float> pos) const;
        [[nodiscard]] std::tuple<float, float, float> screen_to_world_ray(std::tuple<float, float> screen_pos) const;
        [[nodiscard]] std::tuple<float, float, float> camera_position() const;
        [[nodiscard]] std::tuple<float, float, float> camera_forward() const;
        [[nodiscard]] std::tuple<float, float> viewport_size() const;

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
        void draw_text_2d(std::tuple<float, float> pos, const std::string& text,
                          std::tuple<float, float, float, float> color);
        void draw_line_3d(std::tuple<float, float, float> start, std::tuple<float, float, float> end,
                          std::tuple<float, float, float, float> color, float thickness = 1.0f);
        void draw_point_3d(std::tuple<float, float, float> pos,
                           std::tuple<float, float, float, float> color, float size = 4.0f);

        [[nodiscard]] const std::vector<DrawCommand>& get_draw_commands() const { return draw_commands_; }
        void clear_draw_commands() { draw_commands_.clear(); }

    private:
        mutable std::vector<DrawCommand> draw_commands_;
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
