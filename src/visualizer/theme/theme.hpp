/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <string>
#include <imgui.h>

namespace lfs::vis {

    // Base color palette
    struct ThemePalette {
        ImVec4 background;
        ImVec4 surface;
        ImVec4 surface_bright;
        ImVec4 primary;
        ImVec4 primary_dim;
        ImVec4 secondary;
        ImVec4 text;
        ImVec4 text_dim;
        ImVec4 border;
        ImVec4 success;
        ImVec4 warning;
        ImVec4 error;
        ImVec4 info;
        ImVec4 row_even;
        ImVec4 row_odd;
    };

    // Size configuration
    struct ThemeSizes {
        float window_rounding = 6.0f;
        float frame_rounding = 2.0f;
        float popup_rounding = 4.0f;
        float scrollbar_rounding = 4.0f;
        float grab_rounding = 2.0f;
        float tab_rounding = 4.0f;
        float border_size = 0.0f;
        float child_border_size = 1.0f;
        float popup_border_size = 1.0f;
        ImVec2 window_padding = {6.0f, 6.0f};
        ImVec2 frame_padding = {4.0f, 3.0f};
        ImVec2 item_spacing = {8.0f, 4.0f};
        ImVec2 item_inner_spacing = {4.0f, 4.0f};
        float indent_spacing = 21.0f;
        float scrollbar_size = 14.0f;
        float grab_min_size = 10.0f;
        float toolbar_button_size = 24.0f;
        float toolbar_padding = 6.0f;
        float toolbar_spacing = 4.0f;
    };

    // Font configuration
    struct ThemeFonts {
        std::string regular_path = "Inter-Regular.ttf";
        std::string bold_path = "Inter-SemiBold.ttf";
        float base_size = 14.0f;
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
        ImVec2 frame_padding = {12.0f, 8.0f};
        ImVec2 item_spacing = {12.0f, 6.0f};
        ImVec2 popup_padding = {8.0f, 8.0f};
    };

    struct ThemeContextMenu {
        float rounding = 6.0f;
        float header_alpha = 0.4f;
        float header_hover_alpha = 0.6f;
        float header_active_alpha = 0.8f;
        ImVec2 padding = {12.0f, 8.0f};
        ImVec2 item_spacing = {8.0f, 6.0f};
    };

    struct ThemeViewport {
        float corner_radius = 8.0f;
        float border_size = 2.0f;
        float border_alpha = 0.4f;
        float border_darken = 0.15f;
    };

    struct ThemeShadows {
        bool enabled = true;
        ImVec2 offset = {4.0f, 4.0f};
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
        ImVec4 background = {0.20f, 0.20f, 0.22f, 1.0f};
        ImVec4 text = {1.0f, 1.0f, 1.0f, 1.0f};
        ImVec4 text_dim = {0.7f, 0.7f, 0.7f, 1.0f};
        ImVec4 border = {0.4f, 0.55f, 0.7f, 1.0f};
        ImVec4 icon = {0.47f, 0.63f, 0.78f, 1.0f};
        ImVec4 highlight = {0.31f, 0.47f, 0.7f, 1.0f};
        ImVec4 selection = {0.23f, 0.39f, 0.63f, 1.0f};
        ImVec4 selection_flash = {0.55f, 0.7f, 0.94f, 1.0f};
    };

    // Complete theme
    struct Theme {
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

        // ImU32 accessors for ImDrawList
        [[nodiscard]] ImU32 primary_u32() const;
        [[nodiscard]] ImU32 error_u32() const;
        [[nodiscard]] ImU32 success_u32() const;
        [[nodiscard]] ImU32 warning_u32() const;
        [[nodiscard]] ImU32 text_u32() const;
        [[nodiscard]] ImU32 text_dim_u32() const;
        [[nodiscard]] ImU32 border_u32() const;
        [[nodiscard]] ImU32 surface_u32() const;

        // Selection colors
        [[nodiscard]] ImU32 selection_fill_u32() const;
        [[nodiscard]] ImU32 selection_border_u32() const;
        [[nodiscard]] ImU32 selection_line_u32() const;

        // Polygon colors
        [[nodiscard]] ImU32 polygon_vertex_u32() const;
        [[nodiscard]] ImU32 polygon_vertex_hover_u32() const;
        [[nodiscard]] ImU32 polygon_close_hint_u32() const;

        // Overlay colors
        [[nodiscard]] ImU32 overlay_background_u32() const;
        [[nodiscard]] ImU32 overlay_text_u32() const;
        [[nodiscard]] ImU32 overlay_shadow_u32() const;
        [[nodiscard]] ImU32 overlay_hint_u32() const;
        [[nodiscard]] ImU32 overlay_border_u32() const;
        [[nodiscard]] ImU32 overlay_icon_u32() const;
        [[nodiscard]] ImU32 overlay_highlight_u32() const;
        [[nodiscard]] ImU32 overlay_selection_u32() const;
        [[nodiscard]] ImU32 overlay_selection_flash_u32() const;

        // Progress bar colors
        [[nodiscard]] ImU32 progress_bar_bg_u32() const;
        [[nodiscard]] ImU32 progress_bar_fill_u32() const;
        [[nodiscard]] ImU32 progress_marker_u32() const;

        // Button states
        [[nodiscard]] ImVec4 button_normal() const;
        [[nodiscard]] ImVec4 button_hovered() const;
        [[nodiscard]] ImVec4 button_active() const;
        [[nodiscard]] ImVec4 button_selected() const;
        [[nodiscard]] ImVec4 button_selected_hovered() const;

        // Toolbar
        [[nodiscard]] ImVec4 toolbar_background() const;
        [[nodiscard]] ImVec4 subtoolbar_background() const;

        // Menu bar
        [[nodiscard]] ImVec4 menu_background() const;
        [[nodiscard]] ImVec4 menu_hover() const;
        [[nodiscard]] ImVec4 menu_active() const;
        [[nodiscard]] ImVec4 menu_popup_background() const;
        [[nodiscard]] ImVec4 menu_border() const;
        [[nodiscard]] ImU32 menu_bottom_border_u32() const;

        // Viewport
        [[nodiscard]] ImU32 viewport_border_u32() const;

        // Scene graph row colors
        [[nodiscard]] ImU32 row_even_u32() const;
        [[nodiscard]] ImU32 row_odd_u32() const;

        // Context menu helpers (pushes 6 colors, 3 style vars)
        void pushContextMenuStyle() const;
        static void popContextMenuStyle();

        [[nodiscard]] bool isLightTheme() const {
            constexpr float BRIGHTNESS_THRESHOLD = 0.5f;
            const float brightness = (palette.background.x + palette.background.y + palette.background.z) / 3.0f;
            return brightness > BRIGHTNESS_THRESHOLD;
        }

        [[nodiscard]] float frameDarkenAmount() const {
            constexpr float LIGHT_DARKEN = 0.15f;
            constexpr float DARK_DARKEN = 0.05f;
            return isLightTheme() ? LIGHT_DARKEN : DARK_DARKEN;
        }
    };

    // Global access
    [[nodiscard]] const Theme& theme();
    void setTheme(const Theme& t);
    void applyThemeToImGui();

    // Presets (loaded from JSON files with hot-reload support)
    [[nodiscard]] const Theme& darkTheme();
    [[nodiscard]] const Theme& lightTheme();
    void checkThemeFileChanges(); // Call periodically to hot-reload

    // Persistence
    bool saveTheme(const Theme& t, const std::string& path);
    bool loadTheme(Theme& t, const std::string& path);

    // Theme preference (for splash screen)
    void saveThemePreference(bool is_dark);
    [[nodiscard]] bool loadThemePreference(); // Returns true if dark theme

    // Color utilities
    [[nodiscard]] ImVec4 lighten(const ImVec4& color, float amount);
    [[nodiscard]] ImVec4 darken(const ImVec4& color, float amount);
    [[nodiscard]] ImVec4 withAlpha(const ImVec4& color, float alpha);
    [[nodiscard]] ImU32 toU32(const ImVec4& color);
    [[nodiscard]] ImU32 toU32WithAlpha(const ImVec4& color, float alpha);

} // namespace lfs::vis
