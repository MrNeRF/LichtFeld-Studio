/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "py_prop.hpp"
#include "py_tensor.hpp"
#include "visualizer/ipc/view_context.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <optional>
#include <tuple>

namespace nb = nanobind;

namespace lfs::core {
    class Scene;
}

namespace lfs::python {

    struct PyViewInfo {
        PyTensor rotation;
        PyTensor translation;
        int width;
        int height;
        float fov_x;
        float fov_y;
    };

    struct PyViewportRender {
        PyTensor image;
        std::optional<PyTensor> screen_positions;
    };

    [[nodiscard]] std::optional<PyViewportRender> get_viewport_render();

    [[nodiscard]] std::optional<PyViewportRender> capture_viewport();

    [[nodiscard]] std::optional<PyTensor> render_view(const PyTensor& rotation, const PyTensor& translation, int width,
                                                      int height, float fov_degrees = 60.0f,
                                                      const PyTensor* bg_color = nullptr);

    [[nodiscard]] std::optional<PyTensor> compute_screen_positions(const PyTensor& rotation, const PyTensor& translation,
                                                                   int width, int height, float fov_degrees = 60.0f);

    [[nodiscard]] std::optional<PyViewInfo> get_current_view();

    [[nodiscard]] std::tuple<PyTensor, PyTensor> look_at(
        std::tuple<float, float, float> eye, std::tuple<float, float, float> target,
        std::tuple<float, float, float> up = {0.0f, 1.0f, 0.0f});

    [[nodiscard]] std::optional<PyTensor> render_at(
        std::tuple<float, float, float> eye, std::tuple<float, float, float> target, int width, int height,
        float fov_degrees = 60.0f, std::tuple<float, float, float> up = {0.0f, 1.0f, 0.0f},
        const PyTensor* bg_color = nullptr);

    class PyRenderSettings {
    public:
        explicit PyRenderSettings(vis::RenderSettingsProxy settings);

        [[nodiscard]] const std::string& property_group() const { return prop_.group_id(); }
        [[nodiscard]] nb::object get(const std::string& name) const { return prop_.getattr(name); }
        void set(const std::string& name, nb::object value);
        [[nodiscard]] nb::dict prop_info(const std::string& name) const {
            return prop_.prop_info(name);
        }
        [[nodiscard]] nb::object prop_getattr(const std::string& name) const {
            return prop_.getattr(name);
        }
        void prop_setattr(const std::string& name, nb::object value);
        [[nodiscard]] bool has_prop(const std::string& name) const {
            return core::prop::PropertyRegistry::instance().get_property(prop_.group_id(), name).has_value();
        }
        [[nodiscard]] nb::list python_dir() const {
            nb::list result;
            result.append("prop_info");
            result.append("get_all_properties");
            const nb::list props = prop_.dir();
            for (size_t i = 0; i < nb::len(props); ++i) {
                result.append(props[i]);
            }
            return result;
        }

        [[nodiscard]] nb::dict get_all_properties() const;

        PyRenderSettings(PyRenderSettings&& other) noexcept
            : settings_(std::move(other.settings_)),
              prop_(&settings_, "render_settings") {}

        PyRenderSettings& operator=(PyRenderSettings&& other) noexcept {
            if (this != &other) {
                settings_ = std::move(other.settings_);
                prop_ = PyProp<vis::RenderSettingsProxy>(&settings_, "render_settings");
            }
            return *this;
        }

        PyRenderSettings(const PyRenderSettings&) = delete;
        PyRenderSettings& operator=(const PyRenderSettings&) = delete;

    private:
        vis::RenderSettingsProxy settings_;
        PyProp<vis::RenderSettingsProxy> prop_;
    };

    void register_render_settings_properties();
    [[nodiscard]] std::optional<PyRenderSettings> get_render_settings();

    void register_rendering(nb::module_& m);

    void set_render_scene_context(core::Scene* scene);
    [[nodiscard]] core::Scene* get_render_scene();

} // namespace lfs::python
