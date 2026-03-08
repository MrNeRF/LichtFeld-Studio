/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/rml_theme.hpp"
#include "core/logger.hpp"
#include "internal/resource_paths.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/Factory.h>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <format>
#include <fstream>
#include <functional>
#include <mutex>

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

    namespace {
        ImVec4 blend(const ImVec4& base, const ImVec4& accent, float factor) {
            return {base.x + (accent.x - base.x) * factor,
                    base.y + (accent.y - base.y) * factor,
                    base.z + (accent.z - base.z) * factor, 1.0f};
        }

        template <typename T>
        void hashCombine(std::size_t& seed, const T& value) {
            seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }

        void hashColor(std::size_t& seed, const ImVec4& color) {
            hashCombine(seed, color.x);
            hashCombine(seed, color.y);
            hashCombine(seed, color.z);
            hashCombine(seed, color.w);
        }

        void hashVec2(std::size_t& seed, const ImVec2& value) {
            hashCombine(seed, value.x);
            hashCombine(seed, value.y);
        }

        void hashFonts(std::size_t& seed, const ThemeFonts& fonts) {
            hashCombine(seed, fonts.regular_path);
            hashCombine(seed, fonts.bold_path);
            hashCombine(seed, fonts.base_size);
            hashCombine(seed, fonts.small_size);
            hashCombine(seed, fonts.large_size);
            hashCombine(seed, fonts.heading_size);
            hashCombine(seed, fonts.section_size);
        }

        void hashMenu(std::size_t& seed, const ThemeMenu& menu) {
            hashCombine(seed, menu.bg_lighten);
            hashCombine(seed, menu.hover_lighten);
            hashCombine(seed, menu.active_alpha);
            hashCombine(seed, menu.popup_lighten);
            hashCombine(seed, menu.popup_rounding);
            hashCombine(seed, menu.popup_border_size);
            hashCombine(seed, menu.border_alpha);
            hashCombine(seed, menu.bottom_border_darken);
            hashVec2(seed, menu.frame_padding);
            hashVec2(seed, menu.item_spacing);
            hashVec2(seed, menu.popup_padding);
        }

        void hashContextMenu(std::size_t& seed, const ThemeContextMenu& context_menu) {
            hashCombine(seed, context_menu.rounding);
            hashCombine(seed, context_menu.header_alpha);
            hashCombine(seed, context_menu.header_hover_alpha);
            hashCombine(seed, context_menu.header_active_alpha);
            hashVec2(seed, context_menu.padding);
            hashVec2(seed, context_menu.item_spacing);
        }

        void hashViewport(std::size_t& seed, const ThemeViewport& viewport) {
            hashCombine(seed, viewport.corner_radius);
            hashCombine(seed, viewport.border_size);
            hashCombine(seed, viewport.border_alpha);
            hashCombine(seed, viewport.border_darken);
        }

        void hashShadows(std::size_t& seed, const ThemeShadows& shadows) {
            hashCombine(seed, shadows.enabled);
            hashVec2(seed, shadows.offset);
            hashCombine(seed, shadows.blur);
            hashCombine(seed, shadows.alpha);
        }

        void hashVignette(std::size_t& seed, const ThemeVignette& vignette) {
            hashCombine(seed, vignette.enabled);
            hashCombine(seed, vignette.intensity);
            hashCombine(seed, vignette.radius);
            hashCombine(seed, vignette.softness);
        }

        void hashOverlay(std::size_t& seed, const ThemeOverlay& overlay) {
            hashColor(seed, overlay.background);
            hashColor(seed, overlay.text);
            hashColor(seed, overlay.text_dim);
            hashColor(seed, overlay.border);
            hashColor(seed, overlay.icon);
            hashColor(seed, overlay.highlight);
            hashColor(seed, overlay.selection);
            hashColor(seed, overlay.selection_flash);
        }

        std::string shadowToRml(const Theme& t, float blur_scale = 1.0f, float alpha_scale = 1.0f) {
            const float alpha = std::clamp(t.shadows.alpha * alpha_scale, 0.0f, 1.0f);
            return std::format(
                "{} {:.1f}dp {:.1f}dp {:.1f}dp",
                colorToRmlAlpha({0.0f, 0.0f, 0.0f, 1.0f}, alpha),
                t.shadows.offset.x,
                t.shadows.offset.y,
                std::max(0.0f, t.shadows.blur * blur_scale));
        }

    } // namespace

    std::string generateComponentsThemeRCSS(const Theme& t) {
        const auto& p = t.palette;
        const auto text = colorToRml(p.text);
        const auto text_dim = colorToRml(p.text_dim);
        const auto surface = colorToRml(p.surface);
        const auto surface_bright = colorToRml(p.surface_bright);
        const auto primary = colorToRml(p.primary);
        const auto primary_dim = colorToRml(p.primary_dim);
        const auto background = colorToRml(p.background);
        const auto border = colorToRml(p.border);
        const auto primary_select = colorToRmlAlpha(p.primary, 0.18f);

        std::string check_path;
        try {
            check_path = lfs::vis::getAssetPath("icon/check.png").string();
        } catch (...) {}

        std::string arrow_path;
        try {
            arrow_path = lfs::vis::getAssetPath("icon/dropdown-arrow.png").string();
        } catch (...) {}

        const float tn = t.button.tint_normal;
        const float th = t.button.tint_hover;
        const float ta = t.button.tint_active;

        const auto btn_primary = colorToRml(blend(p.surface, p.primary, tn));
        const auto btn_primary_h = colorToRml(blend(p.surface, p.primary, th));
        const auto btn_primary_a = colorToRml(blend(p.surface, p.primary, ta));
        const auto btn_success = colorToRml(blend(p.surface, p.success, tn));
        const auto btn_success_h = colorToRml(blend(p.surface, p.success, th));
        const auto btn_success_a = colorToRml(blend(p.surface, p.success, ta));
        const auto btn_warning = colorToRml(blend(p.surface, p.warning, tn));
        const auto btn_warning_h = colorToRml(blend(p.surface, p.warning, th));
        const auto btn_warning_a = colorToRml(blend(p.surface, p.warning, ta));
        const auto btn_error = colorToRml(blend(p.surface, p.error, tn));
        const auto btn_error_h = colorToRml(blend(p.surface, p.error, th));
        const auto btn_error_a = colorToRml(blend(p.surface, p.error, ta));

        const auto success = colorToRml(p.success);
        const auto warning = colorToRml(p.warning);
        const auto error = colorToRml(p.error);
        const auto info = colorToRml(p.info);
        const auto header_decor = std::format("decorator: vertical-gradient({} {}); background-color: transparent",
                                              colorToRmlAlpha(p.primary, 0.31f), colorToRmlAlpha(p.primary, 0.16f));
        const auto header_hover_decor = std::format("decorator: vertical-gradient({} {}); background-color: transparent",
                                                    colorToRmlAlpha(p.primary, 0.55f), colorToRmlAlpha(p.primary, 0.40f));
        const auto prog_fill_decor = std::format("decorator: horizontal-gradient({} {}); background-color: transparent",
                                                 colorToRml(p.primary), colorToRml(blend(p.primary, ImVec4(1, 1, 1, 1), 0.08f)));
        const int rounding = static_cast<int>(t.sizes.frame_rounding);
        const int row_pad_y = static_cast<int>(t.sizes.item_spacing.y * 0.5f);
        const int indent = static_cast<int>(t.sizes.indent_spacing);
        const int inner_gap = static_cast<int>(t.sizes.item_inner_spacing.x);
        const int fp_x = static_cast<int>(t.sizes.frame_padding.x);
        const int fp_y = static_cast<int>(t.sizes.frame_padding.y);

        const auto check_decorator =
            check_path.empty()
                ? std::string{}
                : std::format("input[type=\"checkbox\"]:checked {{ decorator: image({}); }}\n", check_path);

        const auto arrow_decorator =
            arrow_path.empty()
                ? std::string{}
                : std::format("selectarrow {{ decorator: image({}); image-color: {}; }}\n", arrow_path, text_dim);

        const auto error_col = colorToRml(p.error);

        return std::format(
                   "#window-frame {{ background-color: {0}; border-color: {1}; }}\n"
                   "#title-bar {{ background-color: {2}; }}\n"
                   "#title-text {{ color: {3}; }}\n"
                   "#close-btn {{ color: {4}; }}\n"
                   "#close-btn:hover {{ color: {5}; }}\n"
                   ".panel-title {{ color: {6}; }}\n"
                   ".description {{ color: {3}; }}\n"
                   ".info-key {{ color: {4}; }}\n"
                   ".info-val {{ color: {3}; }}\n"
                   ".link-label {{ color: {3}; }}\n"
                   ".link-url {{ color: {6}; }}\n"
                   ".link-url:hover {{ text-decoration: underline; }}\n"
                   ".footer-text {{ color: {4}; }}\n"
                   ".footer-sep {{ color: {1}; }}\n"
                   ".card-body {{ background-color: {0}; border-color: {1}; }}\n"
                   ".video-card:hover .card-body {{ border-color: {6}; background-color: {2}; }}\n"
                   ".play-icon {{ color: {6}; }}\n"
                   ".card-title {{ color: {4}; }}\n",
                   surface, border, surface_bright, text, text_dim,
                   error_col, primary) +
                   check_decorator +
                   arrow_decorator +
                   std::format(
                       "input[type=\"checkbox\"] {{ border-color: {5}; }}\n"
                       "input[type=\"checkbox\"]:checked {{ background-color: {4}; border-color: {4}; }}\n"
                       "input[type=\"range\"] slidertrack {{ background-color: {5}; border-width: 0; }}\n"
                       "input[type=\"range\"] sliderprogress {{ background-color: {4}; }}\n"
                       "input[type=\"range\"] sliderbar {{ background-color: {4}; }}\n"
                       "input[type=\"text\"] {{ color: {0}; background-color: {2}; border-color: {5}; }}\n"
                       "input[type=\"text\"]:focus {{ border-color: {4}; }}\n"
                       "select {{ color: {0}; background-color: {2}; border-color: {5}; }}\n"
                       "select:hover {{ border-color: {4}; }}\n"
                       "selectbox {{ background-color: {2}; border-color: {5}; }}\n"
                       "selectbox option:hover {{ background-color: {4}; }}\n"
                       "progress {{ background-color: {2}; border-color: {5}; }}\n"
                       "progress fill {{ {9}; }}\n"
                       ".progress__text {{ color: {0}; }}\n"
                       ".setting-label {{ color: {0}; }}\n"
                       ".prop-label {{ color: {0}; }}\n"
                       ".slider-value {{ color: {1}; }}\n"
                       ".section-header {{ color: {0}; {6}; }}\n"
                       ".section-header:hover {{ {7}; }}\n"
                       ".section-arrow {{ color: {1}; }}\n"
                       ".separator {{ background-color: {5}; }}\n"
                       ".text-disabled {{ color: {1}; }}\n"
                       ".section-label {{ color: {1}; }}\n"
                       ".empty-message {{ color: {1}; }}\n"
                       ".color-swatch {{ border-color: {5}; }}\n"
                       ".color-comp {{ color: {1}; background-color: {2}; border-color: {5}; }}\n"
                       ".color-hex {{ color: {0}; background-color: {2}; border-color: {5}; }}\n"
                       ".color-hex:focus {{ border-color: {4}; }}\n"
                       ".context-menu {{ background-color: {2}; border-color: {5}; }}\n"
                       ".context-menu-item {{ color: {0}; }}\n"
                       ".context-menu-item:hover {{ background-color: {4}; }}\n"
                       ".context-menu-separator {{ background-color: {5}; }}\n"
                       ".btn {{ color: {0}; background-color: {3}; border-color: {5}; border-radius: {8}dp; }}\n"
                       ".btn:hover {{ background-color: {5}; }}\n"
                       ".btn:active {{ background-color: {2}; }}\n"
                       ".btn--secondary {{ background-color: transparent; border-color: {5}; color: {0}; }}\n"
                       ".btn--secondary:hover {{ background-color: {2}; }}\n"
                       ".icon-btn.selected {{ background-color: {4}; }}\n",
                       text, text_dim, surface, surface_bright, primary, border,
                       header_decor, header_hover_decor, rounding, prog_fill_decor) +
                   std::format(
                       ".btn--primary {{ background-color: {0}; border-color: {0}; color: {6}; }}\n"
                       ".btn--primary:hover {{ background-color: {1}; border-color: {1}; }}\n"
                       ".btn--primary:active {{ background-color: {2}; border-color: {2}; }}\n"
                       ".btn--success {{ background-color: {3}; border-color: {3}; color: {6}; }}\n"
                       ".btn--success:hover {{ background-color: {4}; border-color: {4}; }}\n"
                       ".btn--success:active {{ background-color: {5}; border-color: {5}; }}\n",
                       btn_primary, btn_primary_h, btn_primary_a,
                       btn_success, btn_success_h, btn_success_a, text) +
                   std::format(
                       ".btn--warning {{ background-color: {0}; border-color: {0}; color: {6}; }}\n"
                       ".btn--warning:hover {{ background-color: {1}; border-color: {1}; }}\n"
                       ".btn--warning:active {{ background-color: {2}; border-color: {2}; }}\n"
                       ".btn--error {{ background-color: {3}; border-color: {3}; color: {6}; }}\n"
                       ".btn--error:hover {{ background-color: {4}; border-color: {4}; }}\n"
                       ".btn--error:active {{ background-color: {5}; border-color: {5}; }}\n",
                       btn_warning, btn_warning_h, btn_warning_a,
                       btn_error, btn_error_h, btn_error_a, text) +
                   std::format(
                       ".status-success {{ color: {0}; }}\n"
                       ".status-error {{ color: {1}; }}\n"
                       ".status-muted {{ color: {2}; }}\n"
                       ".status-info {{ color: {3}; }}\n",
                       success, error, text_dim, info) +
                   std::format(
                       ".setting-row {{ padding: {0}dp 0; }}\n"
                       ".indent {{ margin-left: {1}dp; }}\n"
                       ".prop-label {{ margin-right: {2}dp; }}\n"
                       "input[type=\"text\"] {{ padding: {4}dp {3}dp; }}\n"
                       "select {{ padding: {4}dp {3}dp; }}\n"
                       ".btn--full {{ padding: {4}dp {3}dp; }}\n",
                       row_pad_y, indent, inner_gap, fp_x, fp_y) +
                   std::format(
                       ".num-step-btn {{ color: {0}; background-color: {1}; border-color: {2}; }}\n"
                       ".num-step-btn:hover {{ background-color: {3}; border-color: {4}; }}\n",
                       text_dim, surface, border, surface_bright, primary) +
                   std::format(
                       ".bg-deep {{ background-color: {0}; }}\n"
                       "#filmstrip {{ background-color: {0}; border-color: {2}; }}\n"
                       ".thumb-item:hover {{ border-color: {2}; }}\n"
                       ".thumb-item.selected {{ border-color: {7}; }}\n"
                       ".section-label-ip {{ color: {8}; }}\n"
                       ".sidebar-header-label-ip {{ color: {4}; }}\n"
                       ".sidebar-header-ip {{ border-color: {2}; }}\n"
                       ".sidebar-section-ip {{ border-color: {2}; }}\n"
                       ".meta-key {{ color: {4}; }}\n"
                       ".meta-val-accent {{ color: {7}; }}\n"
                       ".meta-val-secondary {{ color: {4}; }}\n"
                       "#image-container {{ background-color: {0}; }}\n"
                       ".nav-arrow {{ color: {4}; border-color: {2}; }}\n"
                       ".nav-arrow:hover {{ background-color: {3}; color: {5}; border-color: {8}; }}\n"
                       "#sidebar {{ background-color: {1}; border-color: {2}; }}\n"
                       ".hk-key {{ background-color: {3}; color: {5}; }}\n"
                       ".hk-label {{ color: {4}; }}\n"
                       "#status-bar {{ background-color: {1}; border-color: {2}; }}\n"
                       ".status-item {{ color: {4}; }}\n"
                       ".status-counter {{ color: {4}; }}\n"
                       "#no-image-text {{ color: {4}; }}\n"
                       ".btn-copy-icon {{ image-color: {4}; }}\n"
                       ".btn-copy:hover .btn-copy-icon {{ image-color: {5}; }}\n"
                       ".btn-copy:hover {{ background-color: {3}; }}\n",
                       background, surface, border, surface_bright, text_dim,
                       text, primary_select, primary, primary_dim) +
                   [&]() -> std::string {
            if (!t.shadows.enabled)
                return {};
            return std::format(
                "#window-frame {{ box-shadow: {}; }}\n"
                ".context-menu {{ box-shadow: {}; }}\n"
                "selectbox {{ box-shadow: {}; }}\n"
                ".modal-dialog {{ box-shadow: {}; }}\n"
                ".color-picker-popup {{ box-shadow: {}; }}\n"
                ".confirm-dialog {{ box-shadow: {}; }}\n",
                shadowToRml(t, 1.0f),
                shadowToRml(t, 0.85f),
                shadowToRml(t, 0.75f),
                shadowToRml(t, 1.35f),
                shadowToRml(t, 0.9f),
                shadowToRml(t, 0.9f));
        }();
    }

    std::string generateAllThemeMedia(const ThemeGenerator& gen) {
        std::string result;
        visitThemePresets([&](const std::string_view theme_id, const Theme& theme) {
            auto rules = gen(theme);
            if (!rules.empty())
                result += std::format("@media (theme: {}) {{\n{}}}\n", theme_id, rules);
        });
        return result;
    }

    namespace {
        std::string components_theme_media_cache;
        bool components_theme_media_valid = false;
        std::mutex cache_mutex;
    } // namespace

    const std::string& getComponentsThemeMedia() {
        std::lock_guard lock(cache_mutex);
        if (!components_theme_media_valid) {
            components_theme_media_cache = generateAllThemeMedia(&generateComponentsThemeRCSS);
            components_theme_media_valid = true;
        }
        return components_theme_media_cache;
    }

    void invalidateThemeMediaCache() {
        std::lock_guard lock(cache_mutex);
        components_theme_media_valid = false;
    }

    std::string generateSpriteSheetRCSS() {
        std::string result;
        try {
            const auto atlas = lfs::vis::getAssetPath("icon/scene/scene-sprites.png").string();
            result = std::format(
                "@spritesheet scene-icons {{\n"
                "    src: {};\n"
                "    resolution: 1x;\n"
                "    icon-camera:           0px  0px 24px 24px;\n"
                "    icon-cropbox:          24px 0px 24px 24px;\n"
                "    icon-dataset:          48px 0px 24px 24px;\n"
                "    icon-ellipsoid:        72px 0px 24px 24px;\n"
                "    icon-grip:             96px 0px 24px 24px;\n"
                "    icon-group:            120px 0px 24px 24px;\n"
                "    icon-hidden:           0px  24px 24px 24px;\n"
                "    icon-locked:           24px 24px 24px 24px;\n"
                "    icon-mask:             48px 24px 24px 24px;\n"
                "    icon-mesh:             72px 24px 24px 24px;\n"
                "    icon-pointcloud:       96px 24px 24px 24px;\n"
                "    icon-search:           120px 24px 24px 24px;\n"
                "    icon-selection-group:  0px  48px 24px 24px;\n"
                "    icon-splat:            24px 48px 24px 24px;\n"
                "    icon-trash:            48px 48px 24px 24px;\n"
                "    icon-unlocked:         72px 48px 24px 24px;\n"
                "    icon-visible:          96px 48px 24px 24px;\n"
                "}}\n\n",
                atlas);
        } catch (...) {}
        return result;
    }

    const std::string& getSpriteSheetRCSS() {
        static std::string cached = generateSpriteSheetRCSS();
        return cached;
    }

    std::string darkenColorToRml(const ImVec4& c, float amount) {
        return colorToRml({c.x - amount, c.y - amount, c.z - amount, c.w});
    }

    std::size_t currentThemeSignature() {
        const auto& t = lfs::vis::theme();
        const auto& p = t.palette;
        const auto& s = t.sizes;
        const auto& f = t.fonts;
        const auto& m = t.menu;
        const auto& c = t.context_menu;
        const auto& v = t.viewport;
        const auto& sh = t.shadows;
        const auto& vg = t.vignette;
        const auto& b = t.button;
        const auto& o = t.overlay;

        std::size_t seed = 0;
        hashCombine(seed, t.name);

        hashColor(seed, p.background);
        hashColor(seed, p.surface);
        hashColor(seed, p.surface_bright);
        hashColor(seed, p.primary);
        hashColor(seed, p.primary_dim);
        hashColor(seed, p.secondary);
        hashColor(seed, p.text);
        hashColor(seed, p.text_dim);
        hashColor(seed, p.border);
        hashColor(seed, p.success);
        hashColor(seed, p.warning);
        hashColor(seed, p.error);
        hashColor(seed, p.info);
        hashColor(seed, p.row_even);
        hashColor(seed, p.row_odd);

        hashCombine(seed, s.window_rounding);
        hashCombine(seed, s.frame_rounding);
        hashCombine(seed, s.popup_rounding);
        hashCombine(seed, s.scrollbar_rounding);
        hashCombine(seed, s.grab_rounding);
        hashCombine(seed, s.tab_rounding);
        hashCombine(seed, s.border_size);
        hashCombine(seed, s.child_border_size);
        hashCombine(seed, s.popup_border_size);
        hashVec2(seed, s.window_padding);
        hashVec2(seed, s.frame_padding);
        hashVec2(seed, s.item_spacing);
        hashVec2(seed, s.item_inner_spacing);
        hashCombine(seed, s.indent_spacing);
        hashCombine(seed, s.scrollbar_size);
        hashCombine(seed, s.grab_min_size);
        hashCombine(seed, s.toolbar_button_size);
        hashCombine(seed, s.toolbar_padding);
        hashCombine(seed, s.toolbar_spacing);

        hashFonts(seed, f);
        hashMenu(seed, m);
        hashContextMenu(seed, c);
        hashViewport(seed, v);
        hashShadows(seed, sh);
        hashVignette(seed, vg);
        hashCombine(seed, b.tint_normal);
        hashCombine(seed, b.tint_hover);
        hashCombine(seed, b.tint_active);
        hashOverlay(seed, o);
        return seed;
    }

    void applyTheme(Rml::ElementDocument* doc, const std::string& base_rcss,
                    const std::string& panel_theme_media) {
        assert(doc);
        const std::string combined = getSpriteSheetRCSS() + getComponentsRCSS() + "\n" +
                                     base_rcss + "\n" + getComponentsThemeMedia() + "\n" +
                                     panel_theme_media;
        auto sheet = Rml::Factory::InstanceStyleSheetString(combined);
        if (sheet)
            doc->SetStyleSheetContainer(std::move(sheet));
    }

} // namespace lfs::vis::gui::rml_theme
