/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <core/export.hpp>
#include <string>
#include <imgui.h>

namespace Rml {
    class ElementDocument;
} // namespace Rml

namespace lfs::vis::gui::rml_theme {

    LFS_VIS_API std::string colorToRml(const ImVec4& c);
    LFS_VIS_API std::string colorToRmlAlpha(const ImVec4& c, float alpha);
    LFS_VIS_API std::string loadBaseRCSS(const std::string& asset_name);
    LFS_VIS_API const std::string& getComponentsRCSS();
    LFS_VIS_API std::string generateComponentsThemeRCSS();
    LFS_VIS_API std::string generateSpriteSheetRCSS();
    LFS_VIS_API void applyTheme(Rml::ElementDocument* doc, const std::string& base_rcss,
                                const std::string& theme_rcss);
    LFS_VIS_API std::string darkenColorToRml(const ImVec4& c, float amount);

} // namespace lfs::vis::gui::rml_theme
