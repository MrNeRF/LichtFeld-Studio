/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include <functional>
#include <string>
#include <string_view>

namespace lfs::vis {

    struct LFS_VIS_API ThemeVec2 {
        float x = 0.0f;
        float y = 0.0f;

        constexpr ThemeVec2() = default;
        constexpr ThemeVec2(float x_, float y_) : x(x_),
                                                  y(y_) {}
    };

    struct LFS_VIS_API ThemeColor {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        float w = 1.0f;

        constexpr ThemeColor() = default;
        constexpr ThemeColor(float x_, float y_, float z_, float w_ = 1.0f)
            : x(x_),
              y(y_),
              z(z_),
              w(w_) {}
    };

    // Base color palette
    struct ThemePalette {
        ThemeColor background;
        ThemeColor surface;
        ThemeColor surface_bright;
        ThemeColor primary;
        ThemeColor primary_dim;
        ThemeColor secondary;
        ThemeColor text;
        ThemeColor text_dim;
        ThemeColor border;
        ThemeColor success;
        ThemeColor warning;
        ThemeColor error;
        ThemeColor info;
        ThemeColor row_even;
        ThemeColor row_odd;
    };

    // Size configuration
    struct ThemeSizes {
        float window_rounding = 6.0f;
        float frame_rounding = 2.0f;
        float popup_rounding = 4.0f;
        float scrollbar_rounding = 6.0f;
        float grab_rounding = 2.0f;
        float tab_rounding = 4.0f;
        float border_size = 0.0f;
        float child_border_size = 1.0f;
        float popup_border_size = 1.0f;
        ThemeVec2 window_padding = {6.0f, 6.0f};
        ThemeVec2 frame_padding = {4.0f, 3.0f};
        ThemeVec2 item_spacing = {8.0f, 4.0f};
        ThemeVec2 item_inner_spacing = {4.0f, 4.0f};
        float indent_spacing = 21.0f;
        float scrollbar_size = 12.0f;
        float grab_min_size = 16.0f;
        float toolbar_button_size = 24.0f;
        float toolbar_padding = 6.0f;
        float toolbar_spacing = 4.0f;
    };

    // Font configuration
    struct ThemeFonts {
        std::string regular_path = "Inter-Regular.ttf";
        std::string bold_path = "Inter-SemiBold.ttf";
        float base_size = 12.0f;
        float small_size = 12.0f;
        float large_size = 16.0f;
        float heading_size = 18.0f;
        float section_size = 13.0f;
    };

    struct ThemeMenu {
        float bg_lighten = 0.04f;
        float hover_lighten = 0.08f;
        float active_alpha = 0.35f;
        float popup_lighten = 0.02f;
        float popup_rounding = 6.0f;
        float popup_border_size = 1.0f;
        float border_alpha = 0.6f;
        float bottom_border_darken = 0.08f;
        ThemeVec2 frame_padding = {12.0f, 8.0f};
        ThemeVec2 item_spacing = {12.0f, 6.0f};
        ThemeVec2 popup_padding = {8.0f, 8.0f};
    };

    struct ThemeContextMenu {
        float rounding = 6.0f;
        float header_alpha = 0.4f;
        float header_hover_alpha = 0.6f;
        float header_active_alpha = 0.8f;
        ThemeVec2 padding = {14.0f, 10.0f};
        ThemeVec2 item_spacing = {10.0f, 8.0f};
    };

    struct ThemeViewport {
        float corner_radius = 8.0f;
        float border_size = 2.0f;
        float border_alpha = 0.4f;
        float border_darken = 0.15f;
    };

    struct ThemeShadows {
        bool enabled = true;
        ThemeVec2 offset = {4.0f, 4.0f};
        float blur = 12.0f;
        float alpha = 0.35f;
    };

    struct ThemeVignette {
        bool enabled = true;
        float intensity = 0.3f;
        float radius = 0.7f;
        float softness = 0.5f;
    };

    struct ThemeButton {
        float tint_normal = 0.15f;
        float tint_hover = 0.25f;
        float tint_active = 0.35f;
    };

    struct ThemeOverlay {
        ThemeColor background = {0.20f, 0.20f, 0.22f, 1.0f};
        ThemeColor text = {1.0f, 1.0f, 1.0f, 1.0f};
        ThemeColor text_dim = {0.7f, 0.7f, 0.7f, 1.0f};
        ThemeColor border = {0.4f, 0.55f, 0.7f, 1.0f};
        ThemeColor icon = {0.47f, 0.63f, 0.78f, 1.0f};
        ThemeColor highlight = {0.31f, 0.47f, 0.7f, 1.0f};
        ThemeColor selection = {0.23f, 0.39f, 0.63f, 1.0f};
        ThemeColor selection_flash = {0.55f, 0.7f, 0.94f, 1.0f};
    };

    // Complete theme
    struct LFS_VIS_API Theme {
        std::string name;
        ThemePalette palette;
        ThemeSizes sizes;
        ThemeFonts fonts;
        ThemeMenu menu;
        ThemeContextMenu context_menu;
        ThemeViewport viewport;
        ThemeShadows shadows;
        ThemeVignette vignette;
        ThemeButton button;
        ThemeOverlay overlay;

        // Toolbar
        [[nodiscard]] ThemeColor toolbar_background() const;
        [[nodiscard]] ThemeColor subtoolbar_background() const;

        // Menu bar
        [[nodiscard]] ThemeColor menu_background() const;
        [[nodiscard]] ThemeColor menu_hover() const;
        [[nodiscard]] ThemeColor menu_active() const;
        [[nodiscard]] ThemeColor menu_popup_background() const;
        [[nodiscard]] ThemeColor menu_border() const;

        // Modal dialog helpers retained for plugin API compatibility.
        void pushModalStyle() const;
        static void popModalStyle();

        [[nodiscard]] bool isLightTheme() const {
            constexpr float BRIGHTNESS_THRESHOLD = 0.5f;
            const float brightness = (palette.background.x + palette.background.y + palette.background.z) / 3.0f;
            return brightness > BRIGHTNESS_THRESHOLD;
        }
    };

    // DPI scale for theme sizing
    LFS_VIS_API void setThemeDpiScale(float scale);
    [[nodiscard]] LFS_VIS_API float getThemeDpiScale();

    [[nodiscard]] LFS_VIS_API const Theme& theme();
    LFS_VIS_API void setTheme(const Theme& t);

    using ThemeChangeCallback = std::function<void(const std::string& theme_id)>;
    using ThemePresetVisitor = std::function<void(std::string_view theme_id, const Theme& theme)>;
    struct LFS_VIS_API ThemePresetInfo {
        std::string id;
        std::string name;
        std::string label_key;
        std::string mode;
        int order = 0;
    };
    using ThemePresetInfoVisitor = std::function<void(const ThemePresetInfo& info)>;
    LFS_VIS_API void setThemeChangeCallback(ThemeChangeCallback cb);
    [[nodiscard]] LFS_VIS_API const std::string& currentThemeId();
    LFS_VIS_API void visitThemePresets(const ThemePresetVisitor& visitor);
    LFS_VIS_API void visitThemePresetInfos(const ThemePresetInfoVisitor& visitor);

    // Presets (loaded from JSON files with hot-reload support)
    [[nodiscard]] LFS_VIS_API const Theme& darkTheme();
    LFS_VIS_API bool setThemeByName(const std::string& name);
    LFS_VIS_API bool checkThemeFileChanges(); // Call periodically to hot-reload; returns true when any preset changed

    // Runtime vignette control (does not persist to theme file)
    LFS_VIS_API void setThemeVignetteEnabled(bool enabled);
    LFS_VIS_API void setThemeVignetteIntensity(float intensity);
    LFS_VIS_API void setThemeVignetteStyle(float intensity, float radius, float softness);

    // Persistence
    LFS_VIS_API bool saveTheme(const Theme& t, const std::string& path);
    LFS_VIS_API bool loadTheme(Theme& t, const std::string& path);

    // Theme preference (for splash screen)
    LFS_VIS_API void saveThemePreferenceName(const std::string& theme_name);
    [[nodiscard]] LFS_VIS_API std::string loadThemePreferenceName(); // Returns a canonical theme id

    // UI scale preference. The API represents automatic OS scaling as 0.0;
    // preferences.json serializes that selection as the explicit string "auto".
    LFS_VIS_API void saveUiScalePreference(float scale);
    [[nodiscard]] LFS_VIS_API float loadUiScalePreference();

    // Color utilities
    [[nodiscard]] LFS_VIS_API ThemeColor lighten(const ThemeColor& color, float amount);
    [[nodiscard]] LFS_VIS_API ThemeColor darken(const ThemeColor& color, float amount);
    [[nodiscard]] LFS_VIS_API ThemeColor withAlpha(const ThemeColor& color, float alpha);

} // namespace lfs::vis
