/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/rml_theme.hpp"
#include "core/logger.hpp"
#include "internal/resource_paths.hpp"
#include "theme/theme.hpp"

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

    const std::string& getComponentsRCSS() {
        static std::string cached = loadBaseRCSS("rmlui/components.rcss");
        return cached;
    }

    std::string generateComponentsThemeRCSS() {
        const auto& p = lfs::vis::theme().palette;
        const auto text = colorToRml(p.text);
        const auto text_dim = colorToRml(p.text_dim);
        const auto surface = colorToRml(p.surface);
        const auto surface_bright = colorToRml(p.surface_bright);
        const auto primary = colorToRml(p.primary);
        const auto border = colorToRml(p.border);

        return std::format(
            "input[type=\"checkbox\"] {{ border-color: {5}; }}\n"
            "input[type=\"checkbox\"]:checked {{ background-color: {4}; border-color: {4}; }}\n"
            "input[type=\"range\"] slidertrack {{ background-color: {2}; border-color: {5}; }}\n"
            "input[type=\"range\"] sliderbar {{ background-color: {4}; }}\n"
            "input[type=\"text\"] {{ color: {0}; background-color: {2}; border-color: {5}; }}\n"
            "input[type=\"text\"]:focus {{ border-color: {4}; }}\n"
            "select {{ color: {0}; background-color: {2}; border-color: {5}; }}\n"
            "select:hover {{ border-color: {4}; }}\n"
            "selectbox {{ background-color: {2}; border-color: {5}; }}\n"
            "selectbox option:hover {{ background-color: {4}; }}\n"
            "progress {{ background-color: {2}; border-color: {5}; }}\n"
            "progress fill {{ background-color: {4}; }}\n"
            ".progress__text {{ color: {0}; }}\n"
            ".setting-label {{ color: {0}; }}\n"
            ".prop-label {{ color: {0}; }}\n"
            ".slider-value {{ color: {1}; }}\n"
            ".section-header {{ color: {0}; }}\n"
            ".section-header:hover {{ background-color: {3}; }}\n"
            ".section-arrow {{ color: {1}; }}\n"
            ".separator {{ background-color: {5}; }}\n"
            ".text-disabled {{ color: {1}; }}\n"
            ".empty-message {{ color: {1}; }}\n"
            ".color-swatch {{ border-color: {5}; }}\n"
            ".color-comp {{ color: {1}; background-color: {2}; border-color: {5}; }}\n"
            ".color-hex {{ color: {0}; background-color: {2}; border-color: {5}; }}\n"
            ".color-hex:focus {{ border-color: {4}; }}\n"
            ".context-menu {{ background-color: {2}; border-color: {5}; }}\n"
            ".context-menu-item {{ color: {0}; }}\n"
            ".context-menu-item:hover {{ background-color: {4}; }}\n"
            ".context-menu-separator {{ background-color: {5}; }}\n"
            ".btn {{ color: {0}; background-color: {3}; border-color: {5}; }}\n"
            ".btn:hover {{ background-color: {5}; }}\n"
            ".btn:active {{ background-color: {2}; }}\n"
            ".btn--secondary {{ background-color: transparent; border-color: {5}; color: {0}; }}\n"
            ".btn--secondary:hover {{ background-color: {2}; }}\n"
            ".icon-btn.selected {{ background-color: {4}; }}\n",
            text, text_dim, surface, surface_bright, primary, border);
    }

    void applyTheme(Rml::ElementDocument* doc, const std::string& base_rcss,
                    const std::string& theme_rcss) {
        assert(doc);
        const std::string combined = getComponentsRCSS() + "\n" + base_rcss + "\n" + generateComponentsThemeRCSS() + "\n" + theme_rcss;
        auto sheet = Rml::Factory::InstanceStyleSheetString(combined);
        if (sheet)
            doc->SetStyleSheetContainer(std::move(sheet));
    }

} // namespace lfs::vis::gui::rml_theme
