/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "theme.hpp"
#include "core/logger.hpp"
#include "internal/resource_paths.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

namespace lfs::vis {

    namespace {

        // Alpha constants for derived colors
        constexpr float SELECTION_FILL_ALPHA = 0.15f;
        constexpr float SELECTION_BORDER_ALPHA = 0.85f;
        constexpr float SELECTION_LINE_ALPHA = 0.40f;
        constexpr float POLYGON_CLOSE_ALPHA = 0.78f;
        constexpr float PROGRESS_FILL_ALPHA = 0.78f;
        constexpr float PROGRESS_MARKER_ALPHA = 0.78f;
        constexpr float TOOLBAR_BG_ALPHA = 0.9f;
        constexpr float SUBTOOLBAR_BG_ALPHA = 0.95f;

        // Overlay constants
        constexpr ImU32 OVERLAY_BG = IM_COL32(50, 50, 50, 180);
        constexpr ImU32 OVERLAY_TEXT = IM_COL32(255, 255, 255, 255);
        constexpr ImU32 OVERLAY_SHADOW = IM_COL32(0, 0, 0, 180);
        constexpr ImU32 OVERLAY_HINT = IM_COL32(180, 180, 180, 200);
        constexpr ImU32 PROGRESS_BG = IM_COL32(50, 50, 50, 180);

        // Theme state
        Theme g_current_theme;
        Theme g_dark_theme;
        Theme g_light_theme;
        bool g_initialized = false;
        bool g_themes_loaded = false;

        // Hot-reload state
        std::filesystem::path g_dark_path;
        std::filesystem::path g_light_path;
        std::filesystem::file_time_type g_dark_mtime;
        std::filesystem::file_time_type g_light_mtime;

        void ensureInitialized() {
            if (!g_initialized) {
                g_current_theme = darkTheme();
                g_initialized = true;
            }
        }

    } // namespace

    using json = nlohmann::json;

    namespace {

        json colorToJson(const ImVec4& c) {
            return json::array({c.x, c.y, c.z, c.w});
        }

        ImVec4 colorFromJson(const json& j) {
            if (j.is_array() && j.size() >= 4) {
                return {j[0].get<float>(), j[1].get<float>(), j[2].get<float>(), j[3].get<float>()};
            }
            return {0.0f, 0.0f, 0.0f, 1.0f};
        }

        json vec2ToJson(const ImVec2& v) {
            return json::array({v.x, v.y});
        }

        ImVec2 vec2FromJson(const json& j) {
            if (j.is_array() && j.size() >= 2) {
                return {j[0].get<float>(), j[1].get<float>()};
            }
            return {0.0f, 0.0f};
        }

    } // namespace

    // Color utilities
    ImVec4 lighten(const ImVec4& color, const float amount) {
        return {
            std::min(1.0f, color.x + amount),
            std::min(1.0f, color.y + amount),
            std::min(1.0f, color.z + amount),
            color.w};
    }

    ImVec4 darken(const ImVec4& color, const float amount) {
        return {
            std::max(0.0f, color.x - amount),
            std::max(0.0f, color.y - amount),
            std::max(0.0f, color.z - amount),
            color.w};
    }

    ImVec4 withAlpha(const ImVec4& color, const float alpha) {
        return {color.x, color.y, color.z, alpha};
    }

    ImU32 toU32(const ImVec4& color) {
        return IM_COL32(
            static_cast<int>(color.x * 255.0f),
            static_cast<int>(color.y * 255.0f),
            static_cast<int>(color.z * 255.0f),
            static_cast<int>(color.w * 255.0f));
    }

    ImU32 toU32WithAlpha(const ImVec4& color, const float alpha) {
        return IM_COL32(
            static_cast<int>(color.x * 255.0f),
            static_cast<int>(color.y * 255.0f),
            static_cast<int>(color.z * 255.0f),
            static_cast<int>(alpha * 255.0f));
    }

    // Theme computed colors
    ImU32 Theme::primary_u32() const { return toU32(palette.primary); }
    ImU32 Theme::error_u32() const { return toU32(palette.error); }
    ImU32 Theme::success_u32() const { return toU32(palette.success); }
    ImU32 Theme::warning_u32() const { return toU32(palette.warning); }
    ImU32 Theme::text_u32() const { return toU32(palette.text); }
    ImU32 Theme::text_dim_u32() const { return toU32(palette.text_dim); }
    ImU32 Theme::border_u32() const { return toU32(palette.border); }
    ImU32 Theme::surface_u32() const { return toU32(palette.surface); }

    ImU32 Theme::selection_fill_u32() const { return toU32WithAlpha(palette.primary, SELECTION_FILL_ALPHA); }
    ImU32 Theme::selection_border_u32() const { return toU32WithAlpha(palette.primary, SELECTION_BORDER_ALPHA); }
    ImU32 Theme::selection_line_u32() const { return toU32WithAlpha(palette.primary, SELECTION_LINE_ALPHA); }

    ImU32 Theme::polygon_vertex_u32() const { return toU32(palette.warning); }
    ImU32 Theme::polygon_vertex_hover_u32() const { return toU32(lighten(palette.warning, 0.2f)); }
    ImU32 Theme::polygon_close_hint_u32() const { return toU32WithAlpha(palette.success, POLYGON_CLOSE_ALPHA); }

    ImU32 Theme::overlay_background_u32() const { return OVERLAY_BG; }
    ImU32 Theme::overlay_text_u32() const { return OVERLAY_TEXT; }
    ImU32 Theme::overlay_shadow_u32() const { return OVERLAY_SHADOW; }
    ImU32 Theme::overlay_hint_u32() const { return OVERLAY_HINT; }

    ImU32 Theme::progress_bar_bg_u32() const { return PROGRESS_BG; }
    ImU32 Theme::progress_bar_fill_u32() const { return toU32WithAlpha(palette.warning, PROGRESS_FILL_ALPHA); }
    ImU32 Theme::progress_marker_u32() const { return toU32WithAlpha(palette.error, PROGRESS_MARKER_ALPHA); }

    ImVec4 Theme::button_normal() const { return palette.surface; }
    ImVec4 Theme::button_hovered() const { return palette.surface_bright; }
    ImVec4 Theme::button_active() const { return darken(palette.surface_bright, 0.05f); }
    ImVec4 Theme::button_selected() const { return palette.primary; }
    ImVec4 Theme::button_selected_hovered() const { return lighten(palette.primary, 0.1f); }

    ImVec4 Theme::toolbar_background() const { return withAlpha(palette.surface, TOOLBAR_BG_ALPHA); }
    ImVec4 Theme::subtoolbar_background() const { return withAlpha(darken(palette.surface, 0.03f), SUBTOOLBAR_BG_ALPHA); }

    ImVec4 Theme::menu_background() const { return lighten(palette.surface, menu.bg_lighten); }
    ImVec4 Theme::menu_hover() const { return lighten(palette.surface_bright, menu.hover_lighten); }
    ImVec4 Theme::menu_active() const { return withAlpha(palette.primary, menu.active_alpha); }
    ImVec4 Theme::menu_popup_background() const { return lighten(palette.surface, menu.popup_lighten); }
    ImVec4 Theme::menu_border() const { return withAlpha(palette.border, menu.border_alpha); }
    ImU32 Theme::menu_bottom_border_u32() const { return toU32(darken(palette.surface, menu.bottom_border_darken)); }

    ImU32 Theme::viewport_border_u32() const { return toU32WithAlpha(darken(palette.background, viewport.border_darken), viewport.border_alpha); }

    ImU32 Theme::row_even_u32() const { return toU32(palette.row_even); }
    ImU32 Theme::row_odd_u32() const { return toU32(palette.row_odd); }

    void Theme::pushContextMenuStyle() const {
        ImGui::PushStyleColor(ImGuiCol_PopupBg, palette.surface);
        ImGui::PushStyleColor(ImGuiCol_Border, palette.border);
        ImGui::PushStyleColor(ImGuiCol_Header, withAlpha(palette.primary, context_menu.header_alpha));
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, withAlpha(palette.primary, context_menu.header_hover_alpha));
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, withAlpha(palette.primary, context_menu.header_active_alpha));
        ImGui::PushStyleColor(ImGuiCol_Text, palette.text);
        ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, context_menu.rounding);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, context_menu.padding);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, context_menu.item_spacing);
    }

    void Theme::popContextMenuStyle() {
        ImGui::PopStyleVar(3);
        ImGui::PopStyleColor(6);
    }

    // Global access
    const Theme& theme() {
        ensureInitialized();
        return g_current_theme;
    }

    void setTheme(const Theme& t) {
        g_current_theme = t;
        g_initialized = true;
        applyThemeToImGui();
    }

    void applyThemeToImGui() {
        ensureInitialized();
        ImGuiStyle& style = ImGui::GetStyle();
        const auto& p = g_current_theme.palette;
        const auto& s = g_current_theme.sizes;

        style.WindowRounding = s.window_rounding;
        style.FrameRounding = s.frame_rounding;
        style.PopupRounding = s.popup_rounding;
        style.ScrollbarRounding = s.scrollbar_rounding;
        style.GrabRounding = s.grab_rounding;
        style.TabRounding = s.tab_rounding;
        style.WindowBorderSize = s.border_size;
        style.ChildBorderSize = s.child_border_size;
        style.PopupBorderSize = s.popup_border_size;
        style.WindowPadding = s.window_padding;
        style.FramePadding = s.frame_padding;
        style.ItemSpacing = s.item_spacing;
        style.ItemInnerSpacing = s.item_inner_spacing;
        style.IndentSpacing = s.indent_spacing;
        style.ScrollbarSize = s.scrollbar_size;
        style.GrabMinSize = s.grab_min_size;
        style.WindowTitleAlign = ImVec2(0.5f, 0.5f);

        ImVec4* const colors = style.Colors;
        colors[ImGuiCol_Text] = p.text;
        colors[ImGuiCol_TextDisabled] = p.text_dim;
        colors[ImGuiCol_WindowBg] = p.background;
        colors[ImGuiCol_ChildBg] = p.background;
        colors[ImGuiCol_PopupBg] = p.surface;
        colors[ImGuiCol_Border] = p.border;
        colors[ImGuiCol_BorderShadow] = ImVec4(0, 0, 0, 0);
        colors[ImGuiCol_FrameBg] = darken(p.surface, 0.05f);
        colors[ImGuiCol_FrameBgHovered] = p.surface_bright;
        colors[ImGuiCol_FrameBgActive] = p.primary_dim;
        colors[ImGuiCol_TitleBg] = p.surface;
        colors[ImGuiCol_TitleBgActive] = p.surface;
        colors[ImGuiCol_TitleBgCollapsed] = p.surface;
        colors[ImGuiCol_MenuBarBg] = p.surface;
        colors[ImGuiCol_ScrollbarBg] = darken(p.background, 0.05f);
        colors[ImGuiCol_ScrollbarGrab] = p.surface_bright;
        colors[ImGuiCol_ScrollbarGrabHovered] = lighten(p.surface_bright, 0.1f);
        colors[ImGuiCol_ScrollbarGrabActive] = p.primary;
        colors[ImGuiCol_CheckMark] = p.primary;
        colors[ImGuiCol_SliderGrab] = p.primary;
        colors[ImGuiCol_SliderGrabActive] = lighten(p.primary, 0.1f);
        colors[ImGuiCol_Button] = p.surface;
        colors[ImGuiCol_ButtonHovered] = p.surface_bright;
        colors[ImGuiCol_ButtonActive] = p.primary_dim;
        colors[ImGuiCol_Header] = withAlpha(p.primary, 0.25f);
        colors[ImGuiCol_HeaderHovered] = withAlpha(p.primary, 0.5f);
        colors[ImGuiCol_HeaderActive] = withAlpha(p.primary, 0.7f);
        colors[ImGuiCol_Separator] = p.border;
        colors[ImGuiCol_SeparatorHovered] = p.primary;
        colors[ImGuiCol_SeparatorActive] = p.primary;
        colors[ImGuiCol_ResizeGrip] = withAlpha(p.primary, 0.2f);
        colors[ImGuiCol_ResizeGripHovered] = withAlpha(p.primary, 0.6f);
        colors[ImGuiCol_ResizeGripActive] = p.primary;
        colors[ImGuiCol_Tab] = p.surface;
        colors[ImGuiCol_TabHovered] = p.surface_bright;
        colors[ImGuiCol_TabActive] = p.primary_dim;
        colors[ImGuiCol_TabUnfocused] = p.surface;
        colors[ImGuiCol_TabUnfocusedActive] = p.surface_bright;
        colors[ImGuiCol_PlotLines] = p.primary;
        colors[ImGuiCol_PlotLinesHovered] = lighten(p.primary, 0.2f);
        colors[ImGuiCol_PlotHistogram] = p.primary;
        colors[ImGuiCol_PlotHistogramHovered] = lighten(p.primary, 0.2f);
        colors[ImGuiCol_TableHeaderBg] = p.surface;
        colors[ImGuiCol_TableBorderStrong] = p.border;
        colors[ImGuiCol_TableBorderLight] = withAlpha(p.border, 0.5f);
        colors[ImGuiCol_TableRowBg] = ImVec4(0, 0, 0, 0);
        colors[ImGuiCol_TableRowBgAlt] = withAlpha(p.surface, 0.3f);
        colors[ImGuiCol_TextSelectedBg] = withAlpha(p.primary, 0.35f);
        colors[ImGuiCol_DragDropTarget] = p.primary;
        colors[ImGuiCol_NavHighlight] = p.primary;
        colors[ImGuiCol_NavWindowingHighlight] = withAlpha(p.primary, 0.7f);
        colors[ImGuiCol_NavWindowingDimBg] = withAlpha(p.background, 0.2f);
        colors[ImGuiCol_ModalWindowDimBg] = withAlpha(p.background, 0.35f);
    }

    namespace {

        const Theme DEFAULT_DARK = {
            .name = "Dark",
            .palette = {
                .background = {0.11f, 0.11f, 0.12f, 1.0f},
                .surface = {0.15f, 0.15f, 0.17f, 1.0f},
                .surface_bright = {0.22f, 0.22f, 0.25f, 1.0f},
                .primary = {0.26f, 0.59f, 0.98f, 1.0f},
                .primary_dim = {0.2f, 0.45f, 0.75f, 1.0f},
                .secondary = {0.6f, 0.4f, 0.8f, 1.0f},
                .text = {0.95f, 0.95f, 0.95f, 1.0f},
                .text_dim = {0.6f, 0.6f, 0.6f, 1.0f},
                .border = {0.3f, 0.3f, 0.35f, 1.0f},
                .success = {0.2f, 0.8f, 0.2f, 1.0f},
                .warning = {1.0f, 0.6f, 0.2f, 1.0f},
                .error = {0.9f, 0.3f, 0.3f, 1.0f},
                .info = {0.26f, 0.59f, 0.98f, 1.0f},
                .row_even = {1.0f, 1.0f, 1.0f, 0.04f},
                .row_odd = {0.0f, 0.0f, 0.0f, 0.15f},
            },
            .sizes = {},
            .fonts = {},
        };

        const Theme DEFAULT_LIGHT = {
            .name = "Light",
            .palette = {
                .background = {0.82f, 0.82f, 0.84f, 1.0f},
                .surface = {0.88f, 0.88f, 0.90f, 1.0f},
                .surface_bright = {0.92f, 0.92f, 0.94f, 1.0f},
                .primary = {0.2f, 0.5f, 0.9f, 1.0f},
                .primary_dim = {0.3f, 0.55f, 0.85f, 1.0f},
                .secondary = {0.5f, 0.3f, 0.7f, 1.0f},
                .text = {0.1f, 0.1f, 0.12f, 1.0f},
                .text_dim = {0.4f, 0.4f, 0.45f, 1.0f},
                .border = {0.68f, 0.68f, 0.72f, 1.0f},
                .success = {0.15f, 0.6f, 0.15f, 1.0f},
                .warning = {0.85f, 0.5f, 0.1f, 1.0f},
                .error = {0.8f, 0.2f, 0.2f, 1.0f},
                .info = {0.15f, 0.5f, 0.85f, 1.0f},
                .row_even = {0.0f, 0.0f, 0.0f, 0.04f},
                .row_odd = {0.0f, 0.0f, 0.0f, 0.10f},
            },
            .sizes = {},
            .fonts = {},
        };

        void loadThemesFromFiles() {
            // Load dark theme
            g_dark_theme = DEFAULT_DARK;
            try {
                g_dark_path = getAssetPath("themes/dark.json");
                if (loadTheme(g_dark_theme, g_dark_path.string())) {
                    g_dark_mtime = std::filesystem::last_write_time(g_dark_path);
                    LOG_INFO("Loaded dark theme from {}", g_dark_path.string());
                }
            } catch (...) {
                g_dark_path.clear();
            }

            // Load light theme
            g_light_theme = DEFAULT_LIGHT;
            try {
                g_light_path = getAssetPath("themes/light.json");
                if (loadTheme(g_light_theme, g_light_path.string())) {
                    g_light_mtime = std::filesystem::last_write_time(g_light_path);
                    LOG_INFO("Loaded light theme from {}", g_light_path.string());
                }
            } catch (...) {
                g_light_path.clear();
            }

            g_themes_loaded = true;
        }

        void ensureThemesLoaded() {
            if (!g_themes_loaded) {
                loadThemesFromFiles();
            }
        }

    } // namespace

    const Theme& darkTheme() {
        ensureThemesLoaded();
        return g_dark_theme;
    }

    const Theme& lightTheme() {
        ensureThemesLoaded();
        return g_light_theme;
    }

    void checkThemeFileChanges() {
        if (!g_themes_loaded)
            return;

        bool current_is_dark = (g_current_theme.name == "Dark");
        bool reloaded = false;

        // Check dark theme
        if (!g_dark_path.empty() && std::filesystem::exists(g_dark_path)) {
            const auto mtime = std::filesystem::last_write_time(g_dark_path);
            if (mtime != g_dark_mtime) {
                Theme t = DEFAULT_DARK;
                if (loadTheme(t, g_dark_path.string())) {
                    g_dark_theme = t;
                    g_dark_mtime = mtime;
                    LOG_INFO("Hot-reloaded dark theme");
                    if (current_is_dark)
                        reloaded = true;
                }
            }
        }

        // Check light theme
        if (!g_light_path.empty() && std::filesystem::exists(g_light_path)) {
            const auto mtime = std::filesystem::last_write_time(g_light_path);
            if (mtime != g_light_mtime) {
                Theme t = DEFAULT_LIGHT;
                if (loadTheme(t, g_light_path.string())) {
                    g_light_theme = t;
                    g_light_mtime = mtime;
                    LOG_INFO("Hot-reloaded light theme");
                    if (!current_is_dark)
                        reloaded = true;
                }
            }
        }

        // Re-apply current theme if it was reloaded
        if (reloaded) {
            setTheme(current_is_dark ? g_dark_theme : g_light_theme);
        }
    }

    bool saveTheme(const Theme& t, const std::string& path) {
        try {
            json j;
            j["name"] = t.name;

            auto& palette = j["palette"];
            palette["background"] = colorToJson(t.palette.background);
            palette["surface"] = colorToJson(t.palette.surface);
            palette["surface_bright"] = colorToJson(t.palette.surface_bright);
            palette["primary"] = colorToJson(t.palette.primary);
            palette["primary_dim"] = colorToJson(t.palette.primary_dim);
            palette["secondary"] = colorToJson(t.palette.secondary);
            palette["text"] = colorToJson(t.palette.text);
            palette["text_dim"] = colorToJson(t.palette.text_dim);
            palette["border"] = colorToJson(t.palette.border);
            palette["success"] = colorToJson(t.palette.success);
            palette["warning"] = colorToJson(t.palette.warning);
            palette["error"] = colorToJson(t.palette.error);
            palette["info"] = colorToJson(t.palette.info);
            palette["row_even"] = colorToJson(t.palette.row_even);
            palette["row_odd"] = colorToJson(t.palette.row_odd);

            auto& sizes = j["sizes"];
            sizes["window_rounding"] = t.sizes.window_rounding;
            sizes["frame_rounding"] = t.sizes.frame_rounding;
            sizes["popup_rounding"] = t.sizes.popup_rounding;
            sizes["scrollbar_rounding"] = t.sizes.scrollbar_rounding;
            sizes["grab_rounding"] = t.sizes.grab_rounding;
            sizes["tab_rounding"] = t.sizes.tab_rounding;
            sizes["border_size"] = t.sizes.border_size;
            sizes["child_border_size"] = t.sizes.child_border_size;
            sizes["popup_border_size"] = t.sizes.popup_border_size;
            sizes["window_padding"] = vec2ToJson(t.sizes.window_padding);
            sizes["frame_padding"] = vec2ToJson(t.sizes.frame_padding);
            sizes["item_spacing"] = vec2ToJson(t.sizes.item_spacing);
            sizes["item_inner_spacing"] = vec2ToJson(t.sizes.item_inner_spacing);
            sizes["indent_spacing"] = t.sizes.indent_spacing;
            sizes["scrollbar_size"] = t.sizes.scrollbar_size;
            sizes["grab_min_size"] = t.sizes.grab_min_size;
            sizes["toolbar_button_size"] = t.sizes.toolbar_button_size;
            sizes["toolbar_padding"] = t.sizes.toolbar_padding;
            sizes["toolbar_spacing"] = t.sizes.toolbar_spacing;

            auto& fonts = j["fonts"];
            fonts["regular_path"] = t.fonts.regular_path;
            fonts["bold_path"] = t.fonts.bold_path;
            fonts["base_size"] = t.fonts.base_size;
            fonts["small_size"] = t.fonts.small_size;
            fonts["large_size"] = t.fonts.large_size;
            fonts["heading_size"] = t.fonts.heading_size;
            fonts["section_size"] = t.fonts.section_size;

            auto& menu = j["menu"];
            menu["bg_lighten"] = t.menu.bg_lighten;
            menu["hover_lighten"] = t.menu.hover_lighten;
            menu["active_alpha"] = t.menu.active_alpha;
            menu["popup_lighten"] = t.menu.popup_lighten;
            menu["popup_rounding"] = t.menu.popup_rounding;
            menu["popup_border_size"] = t.menu.popup_border_size;
            menu["border_alpha"] = t.menu.border_alpha;
            menu["bottom_border_darken"] = t.menu.bottom_border_darken;
            menu["frame_padding"] = vec2ToJson(t.menu.frame_padding);
            menu["item_spacing"] = vec2ToJson(t.menu.item_spacing);
            menu["popup_padding"] = vec2ToJson(t.menu.popup_padding);

            auto& ctx = j["context_menu"];
            ctx["rounding"] = t.context_menu.rounding;
            ctx["header_alpha"] = t.context_menu.header_alpha;
            ctx["header_hover_alpha"] = t.context_menu.header_hover_alpha;
            ctx["header_active_alpha"] = t.context_menu.header_active_alpha;
            ctx["padding"] = vec2ToJson(t.context_menu.padding);
            ctx["item_spacing"] = vec2ToJson(t.context_menu.item_spacing);

            auto& viewport = j["viewport"];
            viewport["corner_radius"] = t.viewport.corner_radius;
            viewport["border_size"] = t.viewport.border_size;
            viewport["border_alpha"] = t.viewport.border_alpha;
            viewport["border_darken"] = t.viewport.border_darken;

            auto& shadows = j["shadows"];
            shadows["enabled"] = t.shadows.enabled;
            shadows["offset"] = vec2ToJson(t.shadows.offset);
            shadows["blur"] = t.shadows.blur;
            shadows["alpha"] = t.shadows.alpha;

            auto& vignette = j["vignette"];
            vignette["enabled"] = t.vignette.enabled;
            vignette["intensity"] = t.vignette.intensity;
            vignette["radius"] = t.vignette.radius;
            vignette["softness"] = t.vignette.softness;

            std::ofstream file(path);
            if (!file.is_open())
                return false;
            file << j.dump(2);
            return true;
        } catch (...) {
            return false;
        }
    }

    bool loadTheme(Theme& t, const std::string& path) {
        try {
            std::ifstream file(path);
            if (!file.is_open())
                return false;

            json j;
            file >> j;

            t.name = j.value("name", "Custom");

            if (j.contains("palette")) {
                const auto& p = j["palette"];
                if (p.contains("background"))
                    t.palette.background = colorFromJson(p["background"]);
                if (p.contains("surface"))
                    t.palette.surface = colorFromJson(p["surface"]);
                if (p.contains("surface_bright"))
                    t.palette.surface_bright = colorFromJson(p["surface_bright"]);
                if (p.contains("primary"))
                    t.palette.primary = colorFromJson(p["primary"]);
                if (p.contains("primary_dim"))
                    t.palette.primary_dim = colorFromJson(p["primary_dim"]);
                if (p.contains("secondary"))
                    t.palette.secondary = colorFromJson(p["secondary"]);
                if (p.contains("text"))
                    t.palette.text = colorFromJson(p["text"]);
                if (p.contains("text_dim"))
                    t.palette.text_dim = colorFromJson(p["text_dim"]);
                if (p.contains("border"))
                    t.palette.border = colorFromJson(p["border"]);
                if (p.contains("success"))
                    t.palette.success = colorFromJson(p["success"]);
                if (p.contains("warning"))
                    t.palette.warning = colorFromJson(p["warning"]);
                if (p.contains("error"))
                    t.palette.error = colorFromJson(p["error"]);
                if (p.contains("info"))
                    t.palette.info = colorFromJson(p["info"]);
                if (p.contains("row_even"))
                    t.palette.row_even = colorFromJson(p["row_even"]);
                if (p.contains("row_odd"))
                    t.palette.row_odd = colorFromJson(p["row_odd"]);
            }

            if (j.contains("sizes")) {
                const auto& s = j["sizes"];
                t.sizes.window_rounding = s.value("window_rounding", t.sizes.window_rounding);
                t.sizes.frame_rounding = s.value("frame_rounding", t.sizes.frame_rounding);
                t.sizes.popup_rounding = s.value("popup_rounding", t.sizes.popup_rounding);
                t.sizes.scrollbar_rounding = s.value("scrollbar_rounding", t.sizes.scrollbar_rounding);
                t.sizes.grab_rounding = s.value("grab_rounding", t.sizes.grab_rounding);
                t.sizes.tab_rounding = s.value("tab_rounding", t.sizes.tab_rounding);
                t.sizes.border_size = s.value("border_size", t.sizes.border_size);
                t.sizes.child_border_size = s.value("child_border_size", t.sizes.child_border_size);
                t.sizes.popup_border_size = s.value("popup_border_size", t.sizes.popup_border_size);
                if (s.contains("window_padding"))
                    t.sizes.window_padding = vec2FromJson(s["window_padding"]);
                if (s.contains("frame_padding"))
                    t.sizes.frame_padding = vec2FromJson(s["frame_padding"]);
                if (s.contains("item_spacing"))
                    t.sizes.item_spacing = vec2FromJson(s["item_spacing"]);
                if (s.contains("item_inner_spacing"))
                    t.sizes.item_inner_spacing = vec2FromJson(s["item_inner_spacing"]);
                t.sizes.indent_spacing = s.value("indent_spacing", t.sizes.indent_spacing);
                t.sizes.scrollbar_size = s.value("scrollbar_size", t.sizes.scrollbar_size);
                t.sizes.grab_min_size = s.value("grab_min_size", t.sizes.grab_min_size);
                t.sizes.toolbar_button_size = s.value("toolbar_button_size", t.sizes.toolbar_button_size);
                t.sizes.toolbar_padding = s.value("toolbar_padding", t.sizes.toolbar_padding);
                t.sizes.toolbar_spacing = s.value("toolbar_spacing", t.sizes.toolbar_spacing);
            }

            if (j.contains("fonts")) {
                const auto& f = j["fonts"];
                t.fonts.regular_path = f.value("regular_path", t.fonts.regular_path);
                t.fonts.bold_path = f.value("bold_path", t.fonts.bold_path);
                t.fonts.base_size = f.value("base_size", t.fonts.base_size);
                t.fonts.small_size = f.value("small_size", t.fonts.small_size);
                t.fonts.large_size = f.value("large_size", t.fonts.large_size);
                t.fonts.heading_size = f.value("heading_size", t.fonts.heading_size);
                t.fonts.section_size = f.value("section_size", t.fonts.section_size);
            }

            if (j.contains("menu")) {
                const auto& m = j["menu"];
                t.menu.bg_lighten = m.value("bg_lighten", t.menu.bg_lighten);
                t.menu.hover_lighten = m.value("hover_lighten", t.menu.hover_lighten);
                t.menu.active_alpha = m.value("active_alpha", t.menu.active_alpha);
                t.menu.popup_lighten = m.value("popup_lighten", t.menu.popup_lighten);
                t.menu.popup_rounding = m.value("popup_rounding", t.menu.popup_rounding);
                t.menu.popup_border_size = m.value("popup_border_size", t.menu.popup_border_size);
                t.menu.border_alpha = m.value("border_alpha", t.menu.border_alpha);
                t.menu.bottom_border_darken = m.value("bottom_border_darken", t.menu.bottom_border_darken);
                if (m.contains("frame_padding"))
                    t.menu.frame_padding = vec2FromJson(m["frame_padding"]);
                if (m.contains("item_spacing"))
                    t.menu.item_spacing = vec2FromJson(m["item_spacing"]);
                if (m.contains("popup_padding"))
                    t.menu.popup_padding = vec2FromJson(m["popup_padding"]);
            }

            if (j.contains("context_menu")) {
                const auto& ctx = j["context_menu"];
                t.context_menu.rounding = ctx.value("rounding", t.context_menu.rounding);
                t.context_menu.header_alpha = ctx.value("header_alpha", t.context_menu.header_alpha);
                t.context_menu.header_hover_alpha = ctx.value("header_hover_alpha", t.context_menu.header_hover_alpha);
                t.context_menu.header_active_alpha = ctx.value("header_active_alpha", t.context_menu.header_active_alpha);
                if (ctx.contains("padding"))
                    t.context_menu.padding = vec2FromJson(ctx["padding"]);
                if (ctx.contains("item_spacing"))
                    t.context_menu.item_spacing = vec2FromJson(ctx["item_spacing"]);
            }

            if (j.contains("viewport")) {
                const auto& v = j["viewport"];
                t.viewport.corner_radius = v.value("corner_radius", t.viewport.corner_radius);
                t.viewport.border_size = v.value("border_size", t.viewport.border_size);
                t.viewport.border_alpha = v.value("border_alpha", t.viewport.border_alpha);
                t.viewport.border_darken = v.value("border_darken", t.viewport.border_darken);
            }

            if (j.contains("shadows")) {
                const auto& sh = j["shadows"];
                t.shadows.enabled = sh.value("enabled", t.shadows.enabled);
                if (sh.contains("offset"))
                    t.shadows.offset = vec2FromJson(sh["offset"]);
                t.shadows.blur = sh.value("blur", t.shadows.blur);
                t.shadows.alpha = sh.value("alpha", t.shadows.alpha);
            }

            if (j.contains("vignette")) {
                const auto& v = j["vignette"];
                t.vignette.enabled = v.value("enabled", t.vignette.enabled);
                t.vignette.intensity = v.value("intensity", t.vignette.intensity);
                t.vignette.radius = v.value("radius", t.vignette.radius);
                t.vignette.softness = v.value("softness", t.vignette.softness);
            }

            if (j.contains("button")) {
                const auto& b = j["button"];
                t.button.tint_normal = b.value("tint_normal", t.button.tint_normal);
                t.button.tint_hover = b.value("tint_hover", t.button.tint_hover);
                t.button.tint_active = b.value("tint_active", t.button.tint_active);
            }

            return true;
        } catch (...) {
            return false;
        }
    }

} // namespace lfs::vis
