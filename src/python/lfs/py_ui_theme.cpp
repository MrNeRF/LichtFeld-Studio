/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_ui.hpp"
#include "visualizer/theme/theme.hpp"

#include <imgui.h>

namespace lfs::python {

    namespace {
        std::tuple<float, float, float, float> imvec4_to_tuple(const ImVec4& c) {
            return {c.x, c.y, c.z, c.w};
        }
    } // namespace

    PyTheme get_current_theme() {
        const auto& t = lfs::vis::theme();

        PyTheme py_theme;
        py_theme.name = t.name;

        py_theme.palette.background = imvec4_to_tuple(t.palette.background);
        py_theme.palette.surface = imvec4_to_tuple(t.palette.surface);
        py_theme.palette.surface_bright = imvec4_to_tuple(t.palette.surface_bright);
        py_theme.palette.primary = imvec4_to_tuple(t.palette.primary);
        py_theme.palette.primary_dim = imvec4_to_tuple(t.palette.primary_dim);
        py_theme.palette.secondary = imvec4_to_tuple(t.palette.secondary);
        py_theme.palette.text = imvec4_to_tuple(t.palette.text);
        py_theme.palette.text_dim = imvec4_to_tuple(t.palette.text_dim);
        py_theme.palette.border = imvec4_to_tuple(t.palette.border);
        py_theme.palette.success = imvec4_to_tuple(t.palette.success);
        py_theme.palette.warning = imvec4_to_tuple(t.palette.warning);
        py_theme.palette.error = imvec4_to_tuple(t.palette.error);
        py_theme.palette.info = imvec4_to_tuple(t.palette.info);
        py_theme.palette.toolbar_background = imvec4_to_tuple(t.toolbar_background());
        py_theme.palette.row_even = imvec4_to_tuple(t.palette.row_even);
        py_theme.palette.row_odd = imvec4_to_tuple(t.palette.row_odd);
        py_theme.palette.overlay_border = imvec4_to_tuple(t.overlay.border);
        py_theme.palette.overlay_icon = imvec4_to_tuple(t.overlay.icon);
        py_theme.palette.overlay_text = imvec4_to_tuple(t.overlay.text);
        py_theme.palette.overlay_text_dim = imvec4_to_tuple(t.overlay.text_dim);

        py_theme.sizes.window_rounding = t.sizes.window_rounding;
        py_theme.sizes.frame_rounding = t.sizes.frame_rounding;
        py_theme.sizes.popup_rounding = t.sizes.popup_rounding;
        py_theme.sizes.scrollbar_rounding = t.sizes.scrollbar_rounding;
        py_theme.sizes.tab_rounding = t.sizes.tab_rounding;
        py_theme.sizes.border_size = t.sizes.border_size;
        py_theme.sizes.window_padding = {t.sizes.window_padding.x, t.sizes.window_padding.y};
        py_theme.sizes.frame_padding = {t.sizes.frame_padding.x, t.sizes.frame_padding.y};
        py_theme.sizes.item_spacing = {t.sizes.item_spacing.x, t.sizes.item_spacing.y};
        py_theme.sizes.toolbar_button_size = t.sizes.toolbar_button_size;
        py_theme.sizes.toolbar_padding = t.sizes.toolbar_padding;
        py_theme.sizes.toolbar_spacing = t.sizes.toolbar_spacing;

        py_theme.vignette.enabled = t.vignette.enabled;
        py_theme.vignette.intensity = t.vignette.intensity;
        py_theme.vignette.radius = t.vignette.radius;
        py_theme.vignette.softness = t.vignette.softness;

        return py_theme;
    }

    void register_ui_theme(nb::module_& m) {
        nb::class_<PyThemePalette>(m, "ThemePalette")
            .def_ro("background", &PyThemePalette::background)
            .def_ro("surface", &PyThemePalette::surface)
            .def_ro("surface_bright", &PyThemePalette::surface_bright)
            .def_ro("primary", &PyThemePalette::primary)
            .def_ro("primary_dim", &PyThemePalette::primary_dim)
            .def_ro("secondary", &PyThemePalette::secondary)
            .def_ro("text", &PyThemePalette::text)
            .def_ro("text_dim", &PyThemePalette::text_dim)
            .def_ro("border", &PyThemePalette::border)
            .def_ro("success", &PyThemePalette::success)
            .def_ro("warning", &PyThemePalette::warning)
            .def_ro("error", &PyThemePalette::error)
            .def_ro("info", &PyThemePalette::info)
            .def_ro("toolbar_background", &PyThemePalette::toolbar_background)
            .def_ro("row_even", &PyThemePalette::row_even)
            .def_ro("row_odd", &PyThemePalette::row_odd)
            .def_ro("overlay_border", &PyThemePalette::overlay_border)
            .def_ro("overlay_icon", &PyThemePalette::overlay_icon)
            .def_ro("overlay_text", &PyThemePalette::overlay_text)
            .def_ro("overlay_text_dim", &PyThemePalette::overlay_text_dim);

        nb::class_<PyThemeSizes>(m, "ThemeSizes")
            .def_ro("window_rounding", &PyThemeSizes::window_rounding)
            .def_ro("frame_rounding", &PyThemeSizes::frame_rounding)
            .def_ro("popup_rounding", &PyThemeSizes::popup_rounding)
            .def_ro("scrollbar_rounding", &PyThemeSizes::scrollbar_rounding)
            .def_ro("tab_rounding", &PyThemeSizes::tab_rounding)
            .def_ro("border_size", &PyThemeSizes::border_size)
            .def_ro("window_padding", &PyThemeSizes::window_padding)
            .def_ro("frame_padding", &PyThemeSizes::frame_padding)
            .def_ro("item_spacing", &PyThemeSizes::item_spacing)
            .def_ro("toolbar_button_size", &PyThemeSizes::toolbar_button_size)
            .def_ro("toolbar_padding", &PyThemeSizes::toolbar_padding)
            .def_ro("toolbar_spacing", &PyThemeSizes::toolbar_spacing);

        nb::class_<PyThemeVignette>(m, "ThemeVignette")
            .def_ro("enabled", &PyThemeVignette::enabled)
            .def_ro("intensity", &PyThemeVignette::intensity)
            .def_ro("radius", &PyThemeVignette::radius)
            .def_ro("softness", &PyThemeVignette::softness);

        nb::class_<PyTheme>(m, "Theme")
            .def_ro("name", &PyTheme::name)
            .def_ro("palette", &PyTheme::palette)
            .def_ro("sizes", &PyTheme::sizes)
            .def_ro("vignette", &PyTheme::vignette);

        m.def("theme", &get_current_theme, "Get the current theme");
        m.def("set_theme_vignette_enabled", &lfs::vis::setThemeVignetteEnabled, "Set theme vignette enabled");
        m.def("set_theme_vignette_intensity", &lfs::vis::setThemeVignetteIntensity, "Set theme vignette intensity");
        m.def("set_theme_vignette_style", &lfs::vis::setThemeVignetteStyle, "Set vignette intensity, radius, and softness");
    }

} // namespace lfs::python
