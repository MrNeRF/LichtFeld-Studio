/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/rml_theme.hpp"
#include "core/logger.hpp"
#include "internal/resource_paths.hpp"

#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/Factory.h>
#include <cassert>
#include <format>
#include <fstream>

namespace lfs::vis::gui::rml_theme {

    std::string colorToRml(const ImVec4& c) {
        const auto r = static_cast<int>(c.x * 255.0f);
        const auto g = static_cast<int>(c.y * 255.0f);
        const auto b = static_cast<int>(c.z * 255.0f);
        const auto a = static_cast<int>(c.w * 255.0f);
        return std::format("rgba({},{},{},{})", r, g, b, a);
    }

    std::string colorToRmlAlpha(const ImVec4& c, float alpha) {
        const auto r = static_cast<int>(c.x * 255.0f);
        const auto g = static_cast<int>(c.y * 255.0f);
        const auto b = static_cast<int>(c.z * 255.0f);
        const auto a = static_cast<int>(alpha * 255.0f);
        return std::format("rgba({},{},{},{})", r, g, b, a);
    }

    std::string loadBaseRCSS(const std::string& asset_name) {
        try {
            auto rcss_path = lfs::vis::getAssetPath(asset_name);
            std::ifstream f(rcss_path);
            if (f) {
                return {std::istreambuf_iterator<char>(f),
                        std::istreambuf_iterator<char>()};
            }
            LOG_ERROR("RmlTheme: failed to open RCSS at {}", rcss_path.string());
        } catch (const std::exception& e) {
            LOG_ERROR("RmlTheme: RCSS not found: {}", e.what());
        }
        return {};
    }

    void applyTheme(Rml::ElementDocument* doc, const std::string& base_rcss,
                    const std::string& theme_rcss) {
        assert(doc);
        const std::string combined = base_rcss + "\n" + theme_rcss;
        auto sheet = Rml::Factory::InstanceStyleSheetString(combined);
        if (sheet)
            doc->SetStyleSheetContainer(std::move(sheet));
    }

} // namespace lfs::vis::gui::rml_theme
