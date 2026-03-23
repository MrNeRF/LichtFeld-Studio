/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <concepts>
#include <core/export.hpp>
#include <cstddef>
#include <functional>
#include <string>

namespace lfs::vis {
    struct Theme;
}

namespace Rml {
    class ElementDocument;
} // namespace Rml

namespace lfs::vis::gui {

    struct RmlColor {
        float r = 0.0f, g = 0.0f, b = 0.0f, a = 1.0f;

        RmlColor() = default;
        RmlColor(float r_, float g_, float b_, float a_) : r(r_),
                                                           g(g_),
                                                           b(b_),
                                                           a(a_) {}

        template <typename T>
            requires requires(const T& v) {
                { v.x } -> std::convertible_to<float>;
                { v.y } -> std::convertible_to<float>;
                { v.z } -> std::convertible_to<float>;
                { v.w } -> std::convertible_to<float>;
            }
        RmlColor(const T& v) : r(v.x),
                               g(v.y),
                               b(v.z),
                               a(v.w) {}
    };

} // namespace lfs::vis::gui

namespace lfs::vis::gui::rml_theme {

    LFS_VIS_API std::string colorToRml(const RmlColor& c);
    LFS_VIS_API std::string colorToRmlAlpha(const RmlColor& c, float alpha);
    LFS_VIS_API std::string loadBaseRCSS(const std::string& asset_name);
    LFS_VIS_API const std::string& getComponentsRCSS();
    LFS_VIS_API std::string generateComponentsThemeRCSS(const Theme& t);
    LFS_VIS_API std::string generateSpriteSheetRCSS();
    LFS_VIS_API std::size_t currentThemeSignature();
    LFS_VIS_API void applyTheme(Rml::ElementDocument* doc, const std::string& base_rcss,
                                const std::string& panel_theme_media);
    LFS_VIS_API std::string darkenColorToRml(const RmlColor& c, float amount);
    LFS_VIS_API std::string layeredShadow(const Theme& t, int elevation);

    using ThemeGenerator = std::function<std::string(const Theme&)>;
    LFS_VIS_API std::string generateAllThemeMedia(const ThemeGenerator& gen);
    LFS_VIS_API const std::string& getComponentsThemeMedia();
    LFS_VIS_API void invalidateThemeMediaCache();

} // namespace lfs::vis::gui::rml_theme
