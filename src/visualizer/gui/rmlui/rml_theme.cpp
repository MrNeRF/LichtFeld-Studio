/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/rmlui/rml_theme.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "gui/rmlui/rml_path_utils.hpp"
#include "internal/resource_paths.hpp"
#include "preferences.hpp"
#include "theme/theme.hpp"

#include <RmlUi/Core/ElementDocument.h>
#include <RmlUi/Core/Factory.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <format>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace lfs::vis::gui::rml_theme {

    std::string colorToRml(const RmlColor& c) {
        const auto r = static_cast<int>(c.r * 255.0f);
        const auto g = static_cast<int>(c.g * 255.0f);
        const auto b = static_cast<int>(c.b * 255.0f);
        const auto a = static_cast<int>(c.a * 255.0f);
        return std::format("rgba({},{},{},{})", r, g, b, a);
    }

    std::string colorToRmlAlpha(const RmlColor& c, float alpha) {
        const auto r = static_cast<int>(c.r * 255.0f);
        const auto g = static_cast<int>(c.g * 255.0f);
        const auto b = static_cast<int>(c.b * 255.0f);
        const auto a = static_cast<int>(alpha * 255.0f);
        return std::format("rgba({},{},{},{})", r, g, b, a);
    }

    std::string pathToRmlImageSource(const std::filesystem::path& path) {
        const std::string normalized = rml_paths::normalizeFilesystemPath(path);

#ifdef _WIN32
        return rml_paths::filesystemPathToFileUri(path);
#else
        return rml_paths::percentEncode(normalized);
#endif
    }

    namespace {
        std::mutex base_rcss_cache_mutex;
        std::unordered_map<std::string, std::string> base_rcss_cache;

        std::string components_rcss_cache;
        bool components_rcss_valid = false;
        std::mutex components_rcss_mutex;
        std::atomic_bool frosted_glass_available{true};
        std::atomic_size_t viewport_chrome_runtime_revision{0};

        std::string rcssCacheKey(const std::filesystem::path& path) {
            return lfs::core::path_to_utf8(path.lexically_normal());
        }
    } // namespace

    std::string loadBaseRCSS(const std::string& asset_name) {
        try {
            const auto requested_path = std::filesystem::path(asset_name);
            const auto rcss_path = requested_path.is_absolute()
                                       ? requested_path
                                       : lfs::vis::getAssetPath(asset_name);
            const std::string cache_key = rcssCacheKey(rcss_path);

            {
                std::lock_guard lock(base_rcss_cache_mutex);
                if (auto it = base_rcss_cache.find(cache_key); it != base_rcss_cache.end())
                    return it->second;
            }

            std::ifstream f(rcss_path);
            if (f) {
                std::string contents{std::istreambuf_iterator<char>(f),
                                     std::istreambuf_iterator<char>()};
                std::lock_guard lock(base_rcss_cache_mutex);
                auto inserted = base_rcss_cache.emplace(cache_key, std::move(contents));
                return inserted.first->second;
            }
            LOG_ERROR("RmlTheme: failed to open RCSS at {}", rcss_path.string());
        } catch (const std::exception& e) {
            LOG_ERROR("RmlTheme: RCSS not found: {}", e.what());
        }
        return {};
    }

    const std::string& getComponentsRCSS() {
        std::lock_guard lock(components_rcss_mutex);
        if (!components_rcss_valid) {
            components_rcss_cache = loadBaseRCSS("rmlui/components.rcss");
            components_rcss_valid = true;
        }
        return components_rcss_cache;
    }

    void invalidateBaseRcssCache() {
        std::scoped_lock lock(components_rcss_mutex, base_rcss_cache_mutex);
        components_rcss_cache.clear();
        components_rcss_valid = false;
        base_rcss_cache.clear();
    }

    std::string effectiveViewportChromeStyle() {
        std::string style = loadViewportChromeStylePreference();
        if (style == "frosted" &&
            !frosted_glass_available.load(std::memory_order_relaxed)) {
            style = "translucent";
        }
        return style;
    }

    bool setFrostedGlassAvailable(const bool available) {
        if (frosted_glass_available.exchange(available, std::memory_order_relaxed) == available)
            return false;

        viewport_chrome_runtime_revision.fetch_add(1, std::memory_order_relaxed);
        invalidateThemeMediaCache();
        return true;
    }

    namespace {
        ThemeColor blend(const ThemeColor& base, const ThemeColor& accent, float factor) {
            return {base.x + (accent.x - base.x) * factor,
                    base.y + (accent.y - base.y) * factor,
                    base.z + (accent.z - base.z) * factor, 1.0f};
        }

        float linearColorChannel(const float channel) {
            return channel <= 0.04045f
                       ? channel / 12.92f
                       : std::pow((channel + 0.055f) / 1.055f, 2.4f);
        }

        float relativeLuminance(const ThemeColor& color) {
            return 0.2126f * linearColorChannel(color.x) +
                   0.7152f * linearColorChannel(color.y) +
                   0.0722f * linearColorChannel(color.z);
        }

        float contrastRatio(const ThemeColor& lhs, const ThemeColor& rhs) {
            const float lhs_luminance = relativeLuminance(lhs);
            const float rhs_luminance = relativeLuminance(rhs);
            const float lighter = std::max(lhs_luminance, rhs_luminance);
            const float darker = std::min(lhs_luminance, rhs_luminance);
            return (lighter + 0.05f) / (darker + 0.05f);
        }

        ThemeColor readableIconColor(const ThemeColor& background,
                                     const ThemeColor& preferred) {
            constexpr float MIN_ICON_CONTRAST = 4.5f;
            const ThemeColor preferred_opaque{preferred.x, preferred.y, preferred.z, 1.0f};
            if (contrastRatio(background, preferred_opaque) >= MIN_ICON_CONTRAST)
                return preferred_opaque;

            constexpr ThemeColor BLACK{0.0f, 0.0f, 0.0f, 1.0f};
            constexpr ThemeColor WHITE{1.0f, 1.0f, 1.0f, 1.0f};
            return contrastRatio(background, BLACK) >= contrastRatio(background, WHITE)
                       ? BLACK
                       : WHITE;
        }

        ThemeColor readableIconColor(const ThemeGradient& gradient,
                                     const ThemeColor& preferred) {
            const auto minimum_contrast = [&](const ThemeColor& foreground) {
                return std::min(contrastRatio(gradient.start, foreground),
                                contrastRatio(gradient.end, foreground));
            };

            constexpr float MIN_ICON_CONTRAST = 4.5f;
            const ThemeColor preferred_opaque{preferred.x, preferred.y, preferred.z, 1.0f};
            if (minimum_contrast(preferred_opaque) >= MIN_ICON_CONTRAST)
                return preferred_opaque;

            constexpr ThemeColor BLACK{0.0f, 0.0f, 0.0f, 1.0f};
            constexpr ThemeColor WHITE{1.0f, 1.0f, 1.0f, 1.0f};
            return minimum_contrast(BLACK) >= minimum_contrast(WHITE) ? BLACK : WHITE;
        }

        ThemeColor readableIconColor(const ThemeGradient& translucent_gradient,
                                     const float opacity,
                                     const ThemeColor& preferred) {
            constexpr ThemeColor BLACK{0.0f, 0.0f, 0.0f, 1.0f};
            constexpr ThemeColor WHITE{1.0f, 1.0f, 1.0f, 1.0f};
            const std::array backgrounds{
                blend(BLACK, translucent_gradient.start, opacity),
                blend(BLACK, translucent_gradient.end, opacity),
                blend(WHITE, translucent_gradient.start, opacity),
                blend(WHITE, translucent_gradient.end, opacity)};
            const auto minimum_contrast = [&](const ThemeColor& foreground) {
                float minimum = std::numeric_limits<float>::max();
                for (const auto& background : backgrounds)
                    minimum = std::min(minimum, contrastRatio(background, foreground));
                return minimum;
            };

            constexpr float MIN_ICON_CONTRAST = 4.5f;
            const ThemeColor preferred_opaque{preferred.x, preferred.y, preferred.z, 1.0f};
            if (minimum_contrast(preferred_opaque) >= MIN_ICON_CONTRAST)
                return preferred_opaque;

            return minimum_contrast(BLACK) >= minimum_contrast(WHITE) ? BLACK : WHITE;
        }

        template <typename T>
        void hashCombine(std::size_t& seed, const T& value) {
            seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }

        void hashColor(std::size_t& seed, const ThemeColor& color) {
            hashCombine(seed, color.x);
            hashCombine(seed, color.y);
            hashCombine(seed, color.z);
            hashCombine(seed, color.w);
        }

        void hashVec2(std::size_t& seed, const ThemeVec2& value) {
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

        void hashGradient(std::size_t& seed, const std::optional<ThemeGradient>& gradient) {
            hashCombine(seed, gradient.has_value());
            if (!gradient)
                return;
            hashColor(seed, gradient->start);
            hashColor(seed, gradient->end);
        }

        void hashGradients(std::size_t& seed, const ThemeGradients& gradients) {
            hashGradient(seed, gradients.window_body);
            hashGradient(seed, gradients.panel_body);
            hashGradient(seed, gradients.window_title);
            hashGradient(seed, gradients.section_header);
            hashGradient(seed, gradients.section_header_hover);
            hashGradient(seed, gradients.progress);
            hashGradient(seed, gradients.scrubber_track);
            hashGradient(seed, gradients.scrubber_fill);
            hashGradient(seed, gradients.histogram_header);
            hashGradient(seed, gradients.histogram_fill);
            hashGradient(seed, gradients.histogram_selection);
        }

    } // namespace

    namespace {
        struct ElevationParams {
            float tight_y, tight_blur, tight_alpha;
            float ambient_y, ambient_blur_scale, ambient_alpha;
        };

        constexpr ElevationParams ELEVATION_LEVELS[] = {
            {1.0f, 2.0f, 0.40f, 3.0f, 0.5f, 0.20f},
            {1.0f, 3.0f, 0.35f, 5.0f, 1.0f, 0.18f},
            {2.0f, 4.0f, 0.32f, 8.0f, 1.3f, 0.16f},
            {3.0f, 6.0f, 0.30f, 14.0f, 2.0f, 0.15f},
        };

        const ElevationParams& elevationParams(const int elevation) {
            return ELEVATION_LEVELS[std::clamp(elevation - 1, 0, 3)];
        }
    } // namespace

    std::string layeredShadow(const Theme& t, const int elevation) {
        const float a = t.shadows.alpha;
        const float blur = t.shadows.blur;
        const auto& lv = elevationParams(elevation);
        const float ta = std::clamp(a * lv.tight_alpha, 0.0f, 1.0f);
        const float aa = std::clamp(a * lv.ambient_alpha, 0.0f, 1.0f);

        return std::format("{} 0dp {:.1f}dp {:.1f}dp, {} 0dp {:.1f}dp {:.1f}dp",
                           colorToRmlAlpha({0, 0, 0, 1}, ta), lv.tight_y, lv.tight_blur,
                           colorToRmlAlpha({0, 0, 0, 1}, aa), lv.ambient_y,
                           std::max(0.0f, blur * lv.ambient_blur_scale));
    }

    float layeredShadowPadding(const Theme& t, const int elevation) {
        if (!t.shadows.enabled)
            return 0.0f;

        const auto& lv = elevationParams(elevation);
        const float ambient_blur =
            std::max(0.0f, t.shadows.blur * lv.ambient_blur_scale);
        return std::ceil(std::max(std::abs(lv.tight_y) + lv.tight_blur,
                                  std::abs(lv.ambient_y) + ambient_blur));
    }

    namespace {

        std::string trimCopy(std::string_view s) {
            while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())))
                s.remove_prefix(1);
            while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
                s.remove_suffix(1);
            return std::string{s};
        }

        std::vector<std::string> splitArgs(std::string_view args) {
            std::vector<std::string> out;
            std::string current;
            int depth = 0;
            for (const char ch : args) {
                if (ch == '(') {
                    ++depth;
                    current.push_back(ch);
                } else if (ch == ')') {
                    depth = std::max(0, depth - 1);
                    current.push_back(ch);
                } else if (ch == ',' && depth == 0) {
                    out.push_back(trimCopy(current));
                    current.clear();
                } else {
                    current.push_back(ch);
                }
            }
            if (!current.empty() || !args.empty())
                out.push_back(trimCopy(current));
            return out;
        }

        std::optional<ThemeColor> paletteColor(const Theme& t, const std::string& name) {
            const auto& p = t.palette;
            if (name == "background" || name == "palette.background")
                return p.background;
            if (name == "surface" || name == "palette.surface")
                return p.surface;
            if (name == "surface_bright" || name == "palette.surface_bright")
                return p.surface_bright;
            if (name == "primary" || name == "palette.primary")
                return p.primary;
            if (name == "primary_dim" || name == "palette.primary_dim")
                return p.primary_dim;
            if (name == "secondary" || name == "palette.secondary")
                return p.secondary;
            if (name == "text" || name == "palette.text")
                return p.text;
            if (name == "text_dim" || name == "palette.text_dim")
                return p.text_dim;
            if (name == "border" || name == "palette.border")
                return p.border;
            if (name == "success" || name == "palette.success")
                return p.success;
            if (name == "warning" || name == "palette.warning")
                return p.warning;
            if (name == "error" || name == "palette.error")
                return p.error;
            if (name == "info" || name == "palette.info")
                return p.info;
            if (name == "row_even" || name == "palette.row_even")
                return p.row_even;
            if (name == "row_odd" || name == "palette.row_odd")
                return p.row_odd;
            if (name == "white")
                return ThemeColor{1, 1, 1, 1};
            if (name == "black")
                return ThemeColor{0, 0, 0, 1};
            if (name == "menu.background")
                return t.menu_background();
            if (name == "menu.hover")
                return t.menu_hover();
            if (name == "menu.active")
                return t.menu_active();
            if (name == "menu.popup_background")
                return t.menu_popup_background();
            if (name == "menu.border")
                return t.menu_border();
            if (name == "toolbar.background")
                return t.toolbar_background();
            if (name == "toolbar.sub_background")
                return t.subtoolbar_background();
            return std::nullopt;
        }

        float numericThemeValue(const Theme& t, const std::string& name) {
            if (name == "button.tint_normal")
                return t.button.tint_normal;
            if (name == "button.tint_hover")
                return t.button.tint_hover;
            if (name == "button.tint_active")
                return t.button.tint_active;
            if (name == "size.window_rounding")
                return t.sizes.window_rounding;
            if (name == "size.frame_rounding")
                return t.sizes.frame_rounding;
            if (name == "size.popup_rounding")
                return t.sizes.popup_rounding;
            if (name == "size.scrollbar_rounding")
                return t.sizes.scrollbar_rounding;
            if (name == "size.scrollbar_size")
                return t.sizes.scrollbar_size;
            if (name == "size.grab_min_size")
                return t.sizes.grab_min_size;
            if (name == "size.indent_spacing")
                return t.sizes.indent_spacing;
            if (name == "size.item_inner_spacing_x")
                return t.sizes.item_inner_spacing.x;
            if (name == "size.item_spacing_y_half")
                return t.sizes.item_spacing.y * 0.5f;
            if (name == "size.frame_padding_x")
                return t.sizes.frame_padding.x;
            if (name == "size.frame_padding_y")
                return t.sizes.frame_padding.y;
            return std::stof(name);
        }

        std::string numericThemeValueToRml(const Theme& t, const std::string& name) {
            const float value = numericThemeValue(t, name);
            if (std::abs(value - std::round(value)) < 0.001f)
                return std::format("{:.0f}", value);
            return std::format("{:.2f}", value);
        }

        std::optional<ThemeColor> resolveColorExpression(const Theme& t, const std::string& expr) {
            if (auto c = paletteColor(t, expr))
                return c;

            const auto open = expr.find('(');
            const auto close = expr.rfind(')');
            if (open == std::string::npos || close == std::string::npos || close <= open)
                return std::nullopt;

            const auto fn = trimCopy(std::string_view(expr).substr(0, open));
            const auto args = splitArgs(std::string_view(expr).substr(open + 1, close - open - 1));
            if (fn == "blend" && args.size() == 3) {
                const auto a = resolveColorExpression(t, args[0]);
                const auto b = resolveColorExpression(t, args[1]);
                if (a && b)
                    return blend(*a, *b, numericThemeValue(t, args[2]));
            } else if (fn == "lighten" && args.size() == 2) {
                const auto c = resolveColorExpression(t, args[0]);
                if (c)
                    return lighten(*c, numericThemeValue(t, args[1]));
            } else if (fn == "darken" && args.size() == 2) {
                const auto c = resolveColorExpression(t, args[0]);
                if (c)
                    return darken(*c, numericThemeValue(t, args[1]));
            }
            return std::nullopt;
        }

        std::string assetToken(std::string_view asset_name) {
            try {
                return pathToRmlImageSource(lfs::vis::getAssetPath(std::string(asset_name)));
            } catch (const std::exception& e) {
                LOG_ERROR("RmlTheme: asset token '{}' failed: {}", asset_name, e.what());
                return {};
            }
        }

        std::unordered_map<std::string, std::string> namedThemeTokens(const Theme& t) {
            const auto& p = t.palette;
            const bool is_light = t.isLightTheme();
            const std::string viewport_chrome_style = effectiveViewportChromeStyle();
            const bool viewport_chrome_solid = viewport_chrome_style == "solid";
            const bool viewport_chrome_frosted = viewport_chrome_style == "frosted";
            const auto text = colorToRml(p.text);
            const auto text_dim = colorToRml(p.text_dim);
            const auto surface = colorToRml(p.surface);
            const auto surface_bright = colorToRml(p.surface_bright);
            const auto primary = colorToRml(p.primary);
            const auto border = colorToRml(p.border);
            const auto background = colorToRml(p.background);

            ThemeColor menu_toolbar_bg = t.menu_background();
            menu_toolbar_bg.w = 1.0f;
            const ThemeColor menu_toolbar_selected_bg =
                blend(menu_toolbar_bg, p.primary, std::clamp(t.menu.active_alpha, 0.24f, 0.55f));
            const ThemeColor viewport_glass_base =
                blend(p.surface, p.background, is_light ? 0.04f : 0.10f);

            const auto window_surface =
                colorToRml(is_light ? lighten(p.surface, 0.015f) : lighten(p.surface, 0.02f));
            const auto window_body_decor = t.gradients.window_body
                                               ? std::format(
                                                     "decorator: vertical-gradient({} {}); background-color: {}",
                                                     colorToRml(t.gradients.window_body->start),
                                                     colorToRml(t.gradients.window_body->end), surface)
                                               : std::format("background-color: {}", window_surface);
            const auto panel_body_decor = t.gradients.panel_body
                                               ? std::format(
                                                     "decorator: vertical-gradient({} {}); background-color: {}",
                                                     colorToRml(t.gradients.panel_body->start),
                                                     colorToRml(t.gradients.panel_body->end), surface)
                                               : std::format("background-color: {}", surface);
            const bool enhanced_panel_chrome = t.gradients.panel_body.has_value();
            const auto panel_host_body_decor = enhanced_panel_chrome
                                                   ? "decorator: none; background-color: transparent"
                                                   : panel_body_decor;
            const int enhanced_tab_rounding =
                std::max(4, static_cast<int>(std::round(t.sizes.tab_rounding)));
            const int enhanced_header_rounding =
                std::max(4, static_cast<int>(std::round(t.sizes.frame_rounding)));
            const auto right_panel_tab_shape = enhanced_panel_chrome
                                                   ? std::format(
                                                         "margin: 3dp 2dp; padding: 0 8dp; min-height: 22dp; "
                                                         "border-bottom-width: 1dp; border-radius: {}dp",
                                                         enhanced_tab_rounding)
                                                   : "margin: 0; padding: 0 14dp; min-height: 28dp; "
                                                     "border-top-left-radius: 4dp; border-top-right-radius: 4dp";
            const auto right_panel_tab_label_shape =
                enhanced_panel_chrome ? "line-height: 20dp" : "line-height: 28dp";
            const auto right_panel_tab_hover_decor = enhanced_panel_chrome
                                                         ? std::format(
                                                               "decorator: vertical-gradient({} {}); background-color: transparent",
                                                               colorToRml(blend(p.surface_bright, p.primary, 0.12f)),
                                                               colorToRml(blend(p.surface, p.primary, 0.07f)))
                                                         : std::format(
                                                               "background-color: {}",
                                                               colorToRmlAlpha(p.surface_bright, 0.5f));
            const auto right_panel_tab_active_decor = enhanced_panel_chrome
                                                          ? std::format(
                                                                "decorator: vertical-gradient({} {}); background-color: transparent",
                                                                colorToRml(blend(p.surface_bright, p.primary, 0.24f)),
                                                                colorToRml(blend(p.surface, p.primary, 0.15f)))
                                                          : std::format(
                                                                "background-color: {}",
                                                                colorToRmlAlpha(p.surface_bright, 0.4f));
            const auto title_surface_col = is_light ? darken(p.surface, 0.02f) : lighten(p.surface, 0.045f);
            const ThemeGradient title_gradient = t.gradients.window_title.value_or(
                ThemeGradient{lighten(title_surface_col, 0.12f), title_surface_col});
            const auto title_decor = std::format("decorator: vertical-gradient({} {}); background-color: transparent",
                                                 colorToRml(title_gradient.start),
                                                 colorToRml(title_gradient.end));
            const auto optional_vertical_gradient = [](const std::optional<ThemeGradient>& gradient) {
                if (!gradient)
                    return std::string{"decorator: none; background-color: transparent"};
                return std::format(
                    "decorator: vertical-gradient({} {}); background-color: transparent",
                    colorToRml(gradient->start), colorToRml(gradient->end));
            };
            const auto header_decor = optional_vertical_gradient(t.gradients.section_header);
            const auto header_hover_decor =
                optional_vertical_gradient(t.gradients.section_header_hover);
            const auto header_shape = t.gradients.section_header
                                          ? std::format(
                                                "margin: 4dp 0 2dp 0; border-width: 1dp; "
                                                "border-radius: {}dp; border-color: {}",
                                                enhanced_header_rounding,
                                                colorToRmlAlpha(p.border, 0.58f))
                                          : "margin: 0 -6dp; border-width: 0; "
                                            "border-top-width: 4dp; border-radius: 0";
            const ThemeGradient progress_gradient = t.gradients.progress.value_or(
                ThemeGradient{p.primary, blend(p.primary, ThemeColor{1, 1, 1, 1}, 0.08f)});
            const auto progress_fill_decor = std::format("decorator: horizontal-gradient({} {}); background-color: transparent",
                                                         colorToRml(progress_gradient.start),
                                                         colorToRml(progress_gradient.end));
            const auto translucent_vertical_decor = [](const ThemeGradient& gradient,
                                                        const float alpha) {
                return std::format(
                    "decorator: vertical-gradient({} {}); background-color: transparent",
                    colorToRmlAlpha(gradient.start, alpha),
                    colorToRmlAlpha(gradient.end, alpha));
            };
            const ThemeGradient viewport_panel_gradient = t.gradients.panel_body.value_or(
                ThemeGradient{viewport_glass_base, viewport_glass_base});
            const ThemeGradient viewport_glass_gradient{
                blend(viewport_panel_gradient.start, p.primary, is_light ? 0.13f : 0.16f),
                blend(viewport_panel_gradient.end, p.secondary, is_light ? 0.16f : 0.20f)};
            const float viewport_toolbar_glass_opacity = viewport_chrome_solid
                                                             ? 1.0f
                                                         : viewport_chrome_frosted
                                                             ? (is_light ? 0.50f : 0.44f)
                                                             : (is_light ? 0.70f : 0.68f);
            const ThemeColor viewport_toolbar_icon =
                readableIconColor(viewport_glass_gradient, viewport_toolbar_glass_opacity, p.text);
            const auto viewport_toolbar_glass_decor = translucent_vertical_decor(
                viewport_glass_gradient, viewport_toolbar_glass_opacity);
            const ThemeColor viewport_toolbar_selected_bg =
                blend(viewport_glass_base, p.primary, 0.56f);
            const ThemeGradient viewport_toolbar_selected_gradient{
                blend(viewport_glass_gradient.start, progress_gradient.start, 0.56f),
                blend(viewport_glass_gradient.end, progress_gradient.end, 0.56f)};
            constexpr float VIEWPORT_TOOLBAR_SELECTED_OPACITY = 0.82f;
            const auto viewport_toolbar_selected_decor = t.gradients.progress
                                                             ? translucent_vertical_decor(
                                                                   viewport_toolbar_selected_gradient,
                                                                   VIEWPORT_TOOLBAR_SELECTED_OPACITY)
                                                             : std::format(
                                                                   "decorator: none; background-color: {}",
                                                                   colorToRmlAlpha(
                                                                       viewport_toolbar_selected_bg,
                                                                       VIEWPORT_TOOLBAR_SELECTED_OPACITY));
            const ThemeColor viewport_toolbar_selected_icon = t.gradients.progress
                                                                  ? readableIconColor(
                                                                        viewport_toolbar_selected_gradient,
                                                                        VIEWPORT_TOOLBAR_SELECTED_OPACITY,
                                                                        p.text)
                                                                  : readableIconColor(
                                                                        viewport_toolbar_selected_bg, p.text);
            const auto viewport_toolbar_shadow = t.shadows.enabled
                                                     ? std::format(
                                                           "0dp 3dp 12dp {}",
                                                           colorToRmlAlpha(
                                                               ThemeColor{0.0f, 0.0f, 0.0f, 1.0f},
                                                               is_light ? 0.18f : 0.30f))
                                                     : std::string{"none"};

            const ThemeGradient viewport_gizmo_gradient = viewport_glass_gradient;
            const ThemeGradient viewport_gizmo_hover_gradient{
                blend(viewport_gizmo_gradient.start, p.primary, 0.16f),
                blend(viewport_gizmo_gradient.end, p.primary, 0.16f)};
            const ThemeGradient viewport_gizmo_selected_gradient = progress_gradient;
            const float VIEWPORT_GIZMO_OPACITY = viewport_chrome_solid
                                                     ? 1.0f
                                                 : viewport_chrome_frosted
                                                     ? (is_light ? 0.48f : 0.42f)
                                                     : 0.64f;
            constexpr float VIEWPORT_GIZMO_HOVER_OPACITY = 0.76f;
            constexpr float VIEWPORT_GIZMO_SELECTED_OPACITY = 0.82f;
            constexpr float VIEWPORT_GIZMO_SELECTED_HOVER_OPACITY = 0.88f;
            const ThemeColor viewport_gizmo_selected_icon =
                readableIconColor(viewport_gizmo_selected_gradient,
                                  VIEWPORT_GIZMO_SELECTED_OPACITY, p.text);
            const ThemeGradient viewport_gizmo_selected_hover_gradient{
                blend(viewport_gizmo_selected_gradient.start,
                      viewport_gizmo_selected_icon, 0.10f),
                blend(viewport_gizmo_selected_gradient.end,
                      viewport_gizmo_selected_icon, 0.10f)};
            const auto viewport_gizmo_decor = translucent_vertical_decor(
                viewport_gizmo_gradient, VIEWPORT_GIZMO_OPACITY);
            const auto viewport_gizmo_hover_decor = t.gradients.panel_body
                                                        ? translucent_vertical_decor(
                                                              viewport_gizmo_hover_gradient,
                                                              VIEWPORT_GIZMO_HOVER_OPACITY)
                                                        : std::format(
                                                              "decorator: none; background-color: {}",
                                                              colorToRmlAlpha(
                                                                  viewport_gizmo_hover_gradient.start,
                                                                  VIEWPORT_GIZMO_HOVER_OPACITY));
            const auto viewport_gizmo_selected_decor = t.gradients.progress
                                                           ? translucent_vertical_decor(
                                                                 viewport_gizmo_selected_gradient,
                                                                 VIEWPORT_GIZMO_SELECTED_OPACITY)
                                                           : std::format(
                                                                 "decorator: none; background-color: {}",
                                                                 colorToRmlAlpha(
                                                                     viewport_gizmo_selected_gradient.start,
                                                                     VIEWPORT_GIZMO_SELECTED_OPACITY));
            const auto viewport_gizmo_selected_hover_decor = t.gradients.progress
                                                                 ? translucent_vertical_decor(
                                                                       viewport_gizmo_selected_hover_gradient,
                                                                       VIEWPORT_GIZMO_SELECTED_HOVER_OPACITY)
                                                                 : std::format(
                                                                       "decorator: none; background-color: {}",
                                                                       colorToRmlAlpha(
                                                                           viewport_gizmo_selected_hover_gradient.start,
                                                                           VIEWPORT_GIZMO_SELECTED_HOVER_OPACITY));
            const auto menu_chrome_decor = t.gradients.window_title
                                               ? std::format(
                                                     "decorator: vertical-gradient({} {}); background-color: {}",
                                                     colorToRml(t.gradients.window_title->start),
                                                     colorToRml(t.gradients.window_title->end),
                                                     colorToRml(menu_toolbar_bg))
                                               : std::format("decorator: none; background-color: {}",
                                                             colorToRml(menu_toolbar_bg));
            const auto status_chrome_decor = t.gradients.panel_body
                                                 ? std::format(
                                                       "decorator: vertical-gradient({} {}); background-color: {}",
                                                       colorToRml(t.gradients.panel_body->start),
                                                       colorToRml(t.gradients.panel_body->end), surface)
                                                 : std::format("decorator: none; background-color: {}",
                                                               colorToRml(t.menu_background()));
            const auto right_panel_chrome_decor = t.gradients.section_header
                                                      ? std::format(
                                                            "decorator: vertical-gradient({} {}); background-color: {}",
                                                            colorToRml(t.gradients.section_header->start),
                                                            colorToRml(t.gradients.section_header->end), surface)
                                                      : "decorator: none; background-color: transparent";
            const auto right_panel_edge_shape = t.gradients.progress
                                                    ? std::format(
                                                          "display: block; position: absolute; left: 0; top: 0; "
                                                          "width: 1dp; height: 100%; z-index: 5; pointer-events: none; "
                                                          "decorator: vertical-gradient({} {}); background-color: {}",
                                                          colorToRml(blend(p.border, progress_gradient.start, 0.42f)),
                                                          colorToRml(blend(p.border, progress_gradient.end, 0.42f)),
                                                          colorToRmlAlpha(p.border, 0.62f))
                                                    : "display: none";
            const auto right_panel_separator_decor = t.gradients.progress
                                                         ? std::format(
                                                               "decorator: horizontal-gradient({} {}); background-color: {}",
                                                               colorToRml(blend(p.border, progress_gradient.start, 0.36f)),
                                                               colorToRml(blend(p.border, progress_gradient.end, 0.36f)),
                                                               colorToRmlAlpha(p.border, 0.55f))
                                                         : std::format("decorator: none; background-color: {}",
                                                                       colorToRmlAlpha(p.border, 0.4f));
            const auto toolbar_decor = t.gradients.panel_body
                                           ? std::format(
                                                 "decorator: vertical-gradient({} {}); background-color: {}",
                                                 colorToRml(t.gradients.panel_body->start),
                                                 colorToRml(t.gradients.panel_body->end), surface)
                                           : std::format("decorator: none; background-color: {}",
                                                         colorToRml(t.toolbar_background()));
            const auto selected_control_decor = t.gradients.progress
                                                    ? progress_fill_decor
                                                    : std::format("decorator: none; background-color: {}",
                                                                  colorToRml(menu_toolbar_selected_bg));
            const ThemeColor selected_control_icon = t.gradients.progress
                                                         ? readableIconColor(progress_gradient, p.text)
                                                         : readableIconColor(menu_toolbar_selected_bg, p.text);
            const ThemeGradient scrub_track_gradient = t.gradients.scrubber_track.value_or(
                ThemeGradient{is_light
                                  ? blend(p.surface, p.surface_bright, 0.10f)
                                  : blend(p.surface, p.surface_bright, 0.45f),
                              is_light
                                  ? blend(p.surface, p.background, 0.08f)
                                  : blend(p.surface, p.background, 0.22f)});
            const auto scrub_bg_decor =
                std::format("decorator: vertical-gradient({} {}); background-color: {}",
                            colorToRml(scrub_track_gradient.start),
                            colorToRml(scrub_track_gradient.end), surface);
            const ThemeGradient scrub_fill_gradient =
                t.gradients.scrubber_fill.value_or(
                    ThemeGradient{
                        withAlpha(blend(p.primary_dim, p.primary, 0.20f),
                                  is_light ? 0.36f : 0.42f),
                        withAlpha(p.primary, is_light ? 0.52f : 0.60f)});
            const auto scrub_fill_decor =
                std::format("decorator: horizontal-gradient({} {}); background-color: transparent",
                            colorToRml(scrub_fill_gradient.start),
                            colorToRml(scrub_fill_gradient.end));
            const ThemeGradient histogram_header_gradient =
                t.gradients.histogram_header.value_or(
                    ThemeGradient{
                        blend(p.surface_bright, p.primary, is_light ? 0.04f : 0.10f),
                        blend(p.surface, p.primary_dim, is_light ? 0.02f : 0.05f)});
            const auto histogram_hero_decor =
                std::format("decorator: vertical-gradient({} {}); background-color: transparent",
                            colorToRml(histogram_header_gradient.start),
                            colorToRml(histogram_header_gradient.end));
            const ThemeGradient histogram_fill_gradient =
                t.gradients.histogram_fill.value_or(
                    ThemeGradient{
                        blend(p.primary_dim, p.primary, 0.25f),
                        blend(p.primary, ThemeColor{1, 1, 1, 1},
                              is_light ? 0.04f : 0.10f)});
            const auto histogram_fill_decor =
                std::format("decorator: vertical-gradient({} {}); background-color: {}",
                            colorToRml(histogram_fill_gradient.start),
                            colorToRml(histogram_fill_gradient.end),
                            primary);
            const ThemeGradient histogram_selection_gradient =
                t.gradients.histogram_selection.value_or(
                    ThemeGradient{
                        blend(p.warning, p.primary, 0.18f),
                        blend(p.warning, ThemeColor{1, 1, 1, 1},
                              is_light ? 0.06f : 0.12f)});
            const auto histogram_fill_selected_decor =
                std::format("decorator: vertical-gradient({} {}); background-color: {}",
                            colorToRml(histogram_selection_gradient.start),
                            colorToRml(histogram_selection_gradient.end),
                            colorToRml(p.warning));

            const int window_rounding = std::max(4, static_cast<int>(t.sizes.window_rounding));
            const int scrollbar_rounding = std::max(1, static_cast<int>(std::max(
                                                           t.sizes.scrollbar_rounding,
                                                           t.sizes.scrollbar_size * 0.5f)));

            const ThemeColor startup_base_color = blend(p.surface, p.text, is_light ? 0.04f : 0.10f);
            const ThemeColor startup_border_color = blend(p.border, p.text, is_light ? 0.28f : 0.38f);
            std::string startup_box_shadow;
            if (t.shadows.enabled) {
                startup_box_shadow = std::format("box-shadow: {}, {} 0dp 0dp 0dp 1dp inset;",
                                                 layeredShadow(t, 4),
                                                 colorToRmlAlpha(RmlColor{1, 1, 1, 1}, is_light ? 0.08f : 0.05f));
            }
            const std::string no_shadow = "none";
            const std::string layered_shadow_3 =
                t.shadows.enabled ? layeredShadow(t, 3) : no_shadow;
            const std::string layered_shadow_4 =
                t.shadows.enabled ? layeredShadow(t, 4) : no_shadow;
            const float window_shadow_padding = layeredShadowPadding(t, 4);

            std::unordered_map<std::string, std::string> tokens{
                {"window.surface", window_surface},
                {"window.body_decor", window_body_decor},
                {"panel.body_decor", panel_body_decor},
                {"panel.host_body_decor", panel_host_body_decor},
                {"window.title_decor", title_decor},
                {"window.rounding", std::format("{}", window_rounding)},
                {"window.shadow_padding", std::format("{:.1f}", window_shadow_padding)},
                {"components.header_decor", header_decor},
                {"components.header_hover_decor", header_hover_decor},
                {"components.header_shape", header_shape},
                {"components.header_state_border",
                 t.gradients.section_header ? colorToRmlAlpha(p.border, 0.72f)
                                            : colorToRml(t.menu_background())},
                {"components.header_text", t.gradients.section_header ? text : primary},
                {"components.progress_fill_decor", progress_fill_decor},
                {"components.scrub_bg_decor", scrub_bg_decor},
                {"components.scrub_fill_decor", scrub_fill_decor},
                {"components.border_soft", colorToRmlAlpha(p.border, 0.3f)},
                {"components.border_med", colorToRmlAlpha(p.border, 0.5f)},
                {"components.text_hi", colorToRmlAlpha(p.text, 0.9f)},
                {"menu.toolbar_icon", colorToRml(readableIconColor(menu_toolbar_bg, p.text))},
                {"menu.toolbar_selected_decor", selected_control_decor},
                {"menu.toolbar_selected_icon", colorToRml(selected_control_icon)},
                {"chrome.menu_decor", menu_chrome_decor},
                {"chrome.status_decor", status_chrome_decor},
                {"chrome.right_panel_decor", right_panel_chrome_decor},
                {"chrome.right_panel_edge_shape", right_panel_edge_shape},
                {"chrome.right_panel_separator_decor", right_panel_separator_decor},
                {"chrome.toolbar_decor", toolbar_decor},
                {"controls.selected_decor", selected_control_decor},
                {"controls.selected_icon", colorToRml(selected_control_icon)},
                {"menu.bottom_border", darkenColorToRml(p.surface, t.menu.bottom_border_darken)},
                {"components.scroll_track", colorToRmlAlpha(p.background, 0.5f)},
                {"components.scroll_thumb", colorToRmlAlpha(p.text_dim, 0.63f)},
                {"components.scroll_hover", colorToRmlAlpha(p.primary, 0.78f)},
                {"components.scrollbar_rounding", std::format("{}", scrollbar_rounding)},
                {"components.title_surface", colorToRml(title_surface_col)},
                {"layered_shadow.3", layered_shadow_3},
                {"layered_shadow.4", layered_shadow_4},
                {"asset.icon.check", assetToken("icon/check.png")},
                {"asset.icon.dropdown_arrow", assetToken("icon/dropdown-arrow.png")},
                {"panel.row_hover", colorToRmlAlpha(p.primary, 0.12f)},
                {"panel.row_selected", colorToRmlAlpha(p.primary, 0.28f)},
                {"panel.row_selected_hover", colorToRmlAlpha(p.primary, 0.38f)},
                {"panel.tab_inactive_bg", colorToRmlAlpha(p.surface_bright, 0.55f)},
                {"panel.tab_hover_bg", colorToRmlAlpha(p.primary, 0.09f)},
                {"panel.tab_hover_border", colorToRmlAlpha(p.primary, 0.43f)},
                {"panel.tab_active_bg", colorToRmlAlpha(p.primary, 0.11f)},
                {"panel.tab_active_border", colorToRmlAlpha(p.primary, 0.52f)},
                {"panel.chip_accent_bg", colorToRmlAlpha(p.primary, 0.28f)},
                {"panel.chip_accent_border", colorToRmlAlpha(p.primary, 0.59f)},
                {"panel.primary_bg_soft", colorToRmlAlpha(p.primary, 0.16f)},
                {"panel.histogram_hero_decor", histogram_hero_decor},
                {"panel.histogram_surface", colorToRmlAlpha(p.surface, 0.94f)},
                {"panel.histogram_chart_bg", colorToRmlAlpha(blend(p.background, p.surface, 0.22f), is_light ? 0.97f : 0.95f)},
                {"panel.histogram_grid", colorToRmlAlpha(p.border, is_light ? 0.16f : 0.34f)},
                {"panel.histogram_toolbar_item", colorToRmlAlpha(p.surface_bright, is_light ? 0.85f : 0.72f)},
                {"panel.histogram_toolbar_select", colorToRmlAlpha(blend(p.surface, p.background, 0.28f), is_light ? 0.95f : 0.92f)},
                {"panel.histogram_toolbar_divider", colorToRmlAlpha(p.border, is_light ? 0.55f : 0.85f)},
                {"panel.histogram_footer_chip", colorToRmlAlpha(p.surface_bright, is_light ? 0.82f : 0.68f)},
                {"panel.histogram_fill_decor", histogram_fill_decor},
                {"panel.histogram_fill_selected_decor", histogram_fill_selected_decor},
                {"panel.histogram_selection_fill", colorToRmlAlpha(p.warning, is_light ? 0.14f : 0.18f)},
                {"panel.histogram_history_icon_disabled", colorToRmlAlpha(p.text_dim, is_light ? 0.68f : 0.52f)},
                {"panel.primary_border_soft", colorToRmlAlpha(p.primary, 0.33f)},
                {"panel.primary_border_faint", colorToRmlAlpha(p.primary, 0.13f)},
                {"panel.primary_accent", colorToRmlAlpha(p.primary, 0.22f)},
                {"right_panel.tab_hover", colorToRmlAlpha(p.surface_bright, 0.5f)},
                {"right_panel.tab_shape", right_panel_tab_shape},
                {"right_panel.tab_label_shape", right_panel_tab_label_shape},
                {"right_panel.tab_hover_decor", right_panel_tab_hover_decor},
                {"right_panel.tab_active_decor", right_panel_tab_active_decor},
                {"right_panel.tab_border", colorToRmlAlpha(p.border, enhanced_panel_chrome ? 0.46f : 0.42f)},
                {"right_panel.tab_bottom_border", colorToRmlAlpha(p.border, enhanced_panel_chrome ? 0.58f : 0.62f)},
                {"right_panel.tab_active_border",
                 enhanced_panel_chrome ? colorToRmlAlpha(p.primary, 0.78f)
                                       : colorToRmlAlpha(p.border, 0.42f)},
                {"right_panel.tab_active_bottom_border",
                 enhanced_panel_chrome ? colorToRmlAlpha(p.primary, 0.95f) : primary},
                {"right_panel.tab_nav_bg", colorToRmlAlpha(p.surface_bright, 0.2f)},
                {"right_panel.tab_nav_hover", colorToRmlAlpha(p.surface_bright, 0.55f)},
                {"right_panel.tab_nav_disabled", colorToRmlAlpha(p.text_dim, 0.35f)},
                {"right_panel.splitter_bg", colorToRmlAlpha(p.border, 0.4f)},
                {"right_panel.splitter_hover", colorToRmlAlpha(p.info, 0.6f)},
                {"right_panel.splitter_active", colorToRmlAlpha(p.info, 0.8f)},
                {"right_panel.border", colorToRmlAlpha(p.border, 0.6f)},
                {"right_panel.resize_hover", colorToRmlAlpha(p.info, 0.3f)},
                {"right_panel.resize_active", colorToRmlAlpha(p.info, 0.5f)},
                {"modal.surface", colorToRmlAlpha(p.surface, 0.98f)},
                {"modal.border", colorToRmlAlpha(p.border, 0.4f)},
                {"modal.backdrop", colorToRmlAlpha(is_light ? ThemeColor{0.12f, 0.14f, 0.18f, 1.0f} : p.background,
                                                   is_light ? 0.18f : 0.44f)},
                {"overlay.surface", colorToRmlAlpha(p.surface, 0.95f)},
                {"overlay.border", colorToRmlAlpha(p.border, 0.4f)},
                {"viewport.icon_dim", colorToRml(viewport_toolbar_icon)},
                {"viewport.icon_disabled", colorToRmlAlpha(viewport_toolbar_icon, 0.70f)},
                {"viewport.toolbar_text", colorToRml(viewport_toolbar_icon)},
                {"viewport.toolbar_glass_decor", viewport_toolbar_glass_decor},
                {"viewport.toolbar_border", colorToRmlAlpha(
                                                blend(blend(p.border, p.surface_bright, 0.30f), p.primary, 0.18f),
                                                is_light ? 0.72f : 0.62f)},
                {"viewport.toolbar_shadow", viewport_toolbar_shadow},
                {"viewport.toolbar_selected_decor", viewport_toolbar_selected_decor},
                {"viewport.toolbar_selected_icon", colorToRml(viewport_toolbar_selected_icon)},
                {"viewport.gizmo_decor", viewport_gizmo_decor},
                {"viewport.gizmo_icon",
                 colorToRml(readableIconColor(
                     viewport_gizmo_gradient, VIEWPORT_GIZMO_OPACITY, p.text))},
                {"viewport.gizmo_hover_decor", viewport_gizmo_hover_decor},
                {"viewport.gizmo_hover_icon",
                 colorToRml(readableIconColor(
                     viewport_gizmo_hover_gradient, VIEWPORT_GIZMO_HOVER_OPACITY, p.text))},
                {"viewport.gizmo_disabled_icon",
                 colorToRmlAlpha(readableIconColor(
                                     viewport_gizmo_gradient, VIEWPORT_GIZMO_OPACITY, p.text_dim),
                                 0.62f)},
                {"viewport.gizmo_selected_decor", viewport_gizmo_selected_decor},
                {"viewport.gizmo_selected_icon", colorToRml(viewport_gizmo_selected_icon)},
                {"viewport.gizmo_selected_hover_decor", viewport_gizmo_selected_hover_decor},
                {"viewport.gizmo_selected_hover_icon",
                 colorToRml(readableIconColor(viewport_gizmo_selected_hover_gradient,
                                              VIEWPORT_GIZMO_SELECTED_HOVER_OPACITY,
                                              p.text))},
                {"viewport.selected_hover", colorToRml(ThemeColor{
                                                std::min(1.0f, p.primary.x + 0.1f),
                                                std::min(1.0f, p.primary.y + 0.1f),
                                                std::min(1.0f, p.primary.z + 0.1f),
                                                p.primary.w})},
                {"viewport.hover_bg", colorToRmlAlpha(p.surface_bright, 0.3f)},
                {"viewport.status_backdrop", colorToRmlAlpha(p.background, 0.55f)},
                {"viewport.panel_bg", colorToRmlAlpha(p.surface, 0.97f)},
                {"viewport.panel_border", colorToRmlAlpha(p.border, 0.45f)},
                {"viewport.metrics_bg", colorToRmlAlpha(p.background, 0.92f)},
                {"viewport.metrics_border", colorToRmlAlpha(p.primary, 0.55f)},
                {"statusbar.progress_bg", colorToRmlAlpha(p.surface_bright, 0.5f)},
                {"statusbar.progress_fill_decor",
                 t.gradients.progress ? progress_fill_decor
                                      : std::format("background-color: {}", primary)},
                {"statusbar.progress_marker", colorToRmlAlpha(p.error, 0.78f)},
                {"statusbar.progress_marker_active", colorToRmlAlpha(p.warning, 0.92f)},
                {"statusbar.progress_marker_past", colorToRmlAlpha(p.text_dim, 0.58f)},
                {"sequencer_overlay.surface", colorToRmlAlpha(p.surface, 0.95f)},
                {"sequencer_overlay.border", colorToRmlAlpha(p.border, 0.4f)},
                {"sequencer_overlay.primary_border", colorToRmlAlpha(p.primary, 0.6f)},
                {"startup.overlay_bg", colorToRmlAlpha(startup_base_color, is_light ? 0.82f : 0.86f)},
                {"startup.overlay_border", colorToRmlAlpha(startup_border_color, is_light ? 0.40f : 0.50f)},
                {"startup.box_shadow", startup_box_shadow},
                {"startup.primary", colorToRmlAlpha(p.primary, is_light ? 0.78f : 0.62f)},
                {"startup.select_bg", colorToRmlAlpha(p.background, is_light ? 0.90f : 0.78f)},
                {"startup.selectbox_bg", colorToRmlAlpha(p.surface, is_light ? 0.95f : 0.90f)},
            };
            return tokens;
        }

        std::string resolveThemeExpression(const Theme& t,
                                           const std::unordered_map<std::string, std::string>& tokens,
                                           const std::string& expression) {
            const auto expr = trimCopy(expression);
            if (const auto it = tokens.find(expr); it != tokens.end())
                return it->second;

            const auto open = expr.find('(');
            const auto close = expr.rfind(')');
            if (open != std::string::npos && close != std::string::npos && close > open) {
                const auto fn = trimCopy(std::string_view(expr).substr(0, open));
                const auto args = splitArgs(std::string_view(expr).substr(open + 1, close - open - 1));
                if (fn == "alpha" && args.size() == 2) {
                    if (auto c = resolveColorExpression(t, args[0]))
                        return colorToRmlAlpha(*c, numericThemeValue(t, args[1]));
                }
                if (fn == "num" && args.size() == 1)
                    return numericThemeValueToRml(t, args[0]);
                if (auto c = resolveColorExpression(t, expr))
                    return colorToRml(*c);
            }

            if (auto c = resolveColorExpression(t, expr))
                return colorToRml(*c);

            try {
                return numericThemeValueToRml(t, expr);
            } catch (const std::exception&) {
                LOG_ERROR("RmlTheme: unknown theme token '{}'", expr);
                return {};
            }
        }

        std::string expandThemeTemplate(const std::string& theme_template, const Theme& t) {
            const auto tokens = namedThemeTokens(t);
            std::string out;
            out.reserve(theme_template.size());
            std::size_t pos = 0;
            while (pos < theme_template.size()) {
                const auto open = theme_template.find("@{", pos);
                if (open == std::string::npos) {
                    out.append(theme_template, pos, std::string::npos);
                    break;
                }
                out.append(theme_template, pos, open - pos);
                const auto close = theme_template.find('}', open + 2);
                if (close == std::string::npos) {
                    out.append(theme_template, open, std::string::npos);
                    break;
                }
                out += resolveThemeExpression(t, tokens, theme_template.substr(open + 2, close - open - 2));
                pos = close + 1;
            }
            return out;
        }

    } // namespace

    std::string generateThemeMediaFromTemplate(const std::string& theme_template) {
        if (theme_template.empty())
            return {};

        std::string result;
        visitThemePresets([&](const std::string_view theme_id, const Theme& theme) {
            auto rules = expandThemeTemplate(theme_template, theme);
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
            components_theme_media_cache =
                generateThemeMediaFromTemplate(loadBaseRCSS("rmlui/components.theme.rcss"));
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
            const auto atlas =
                pathToRmlImageSource(lfs::vis::getAssetPath("icon/scene/scene-sprites.png"));
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

    std::string darkenColorToRml(const RmlColor& c, float amount) {
        return colorToRml({c.r - amount, c.g - amount, c.b - amount, c.a});
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
        hashGradients(seed, t.gradients);
        hashCombine(seed, themePresentationRevision());
        hashCombine(seed, viewport_chrome_runtime_revision.load(std::memory_order_relaxed));
        return seed;
    }

    void applyTheme(Rml::ElementDocument* doc, const std::string& base_rcss,
                    const std::string& panel_theme_template) {
        assert(doc);
        const std::string combined = getSpriteSheetRCSS() + getComponentsRCSS() + "\n" +
                                     getComponentsThemeMedia() + "\n" + base_rcss + "\n" +
                                     generateThemeMediaFromTemplate(panel_theme_template);
        auto sheet = Rml::Factory::InstanceStyleSheetString(combined);
        if (sheet)
            doc->SetStyleSheetContainer(std::move(sheet));
    }

} // namespace lfs::vis::gui::rml_theme
