/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "theme.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "internal/resource_paths.hpp"
#include <algorithm>
#include <cctype>
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

        // Overlay alpha constants
        constexpr float OVERLAY_BG_ALPHA = 0.9f;
        constexpr float OVERLAY_HINT_ALPHA = 0.78f;
        constexpr float OVERLAY_HIGHLIGHT_ALPHA = 0.7f;
        constexpr float OVERLAY_SELECTION_ALPHA = 0.78f;
        constexpr float OVERLAY_SELECTION_FLASH_ALPHA = 0.9f;

        // Theme state
        Theme g_current_theme;
        std::string g_current_theme_id = "dark";
        Theme g_dark_theme;
        Theme g_light_theme;
        Theme g_gruvbox_theme;
        Theme g_catppuccin_mocha_theme;
        Theme g_catppuccin_latte_theme;
        Theme g_nord_theme;
        float g_dpi_scale = 1.0f;
        bool g_initialized = false;
        bool g_themes_loaded = false;

        // Hot-reload state
        std::filesystem::path g_dark_path;
        std::filesystem::path g_light_path;
        std::filesystem::path g_gruvbox_path;
        std::filesystem::path g_catppuccin_mocha_path;
        std::filesystem::path g_catppuccin_latte_path;
        std::filesystem::path g_nord_path;
        std::filesystem::file_time_type g_dark_mtime;
        std::filesystem::file_time_type g_light_mtime;
        std::filesystem::file_time_type g_gruvbox_mtime;
        std::filesystem::file_time_type g_catppuccin_mocha_mtime;
        std::filesystem::file_time_type g_catppuccin_latte_mtime;
        std::filesystem::file_time_type g_nord_mtime;

        void ensureThemesLoaded();
        void applyCurrentTheme(const Theme& theme, std::string_view theme_id);
        bool activateThemePreset(std::string_view theme_id);

        void ensureInitialized() {
            if (!g_initialized) {
                ensureThemesLoaded();
                g_current_theme = darkTheme();
                g_current_theme_id = "dark";
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

        std::string normalizeThemeIdImpl(std::string name) {
            std::transform(
                name.begin(),
                name.end(),
                name.begin(),
                [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });

            std::replace(name.begin(), name.end(), '-', '_');
            std::replace(name.begin(), name.end(), ' ', '_');

            if (name == "gruvbox_dark") {
                return "gruvbox";
            }
            if (name == "catppuccin" || name == "catppuccin_dark") {
                return "catppuccin_mocha";
            }
            if (name == "catppuccin_light") {
                return "catppuccin_latte";
            }
            if (name == "nord_dark") {
                return "nord";
            }
            return name;
        }

        bool useLightPopupBackground(const Theme& t) {
            // Use palette luminance instead of theme name so popup behavior stays
            // correct even if theme names vary or are edited in JSON files.
            constexpr float LIGHT_POPUP_BG_THRESHOLD = 0.72f;
            const float brightness =
                (t.palette.background.x + t.palette.background.y + t.palette.background.z) / 3.0f;
            return brightness >= LIGHT_POPUP_BG_THRESHOLD;
        }

        ImVec4 mix(const ImVec4& a, const ImVec4& b, const float factor) {
            return {
                a.x + (b.x - a.x) * factor,
                a.y + (b.y - a.y) * factor,
                a.z + (b.z - a.z) * factor,
                a.w + (b.w - a.w) * factor};
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

    ImU32 Theme::overlay_background_u32() const { return toU32WithAlpha(overlay.background, OVERLAY_BG_ALPHA); }
    ImU32 Theme::overlay_text_u32() const { return toU32(overlay.text); }
    ImU32 Theme::overlay_shadow_u32() const { return IM_COL32(0, 0, 0, 180); }
    ImU32 Theme::overlay_hint_u32() const { return toU32WithAlpha(overlay.text_dim, OVERLAY_HINT_ALPHA); }
    ImU32 Theme::overlay_border_u32() const { return toU32(overlay.border); }
    ImU32 Theme::overlay_icon_u32() const { return toU32(overlay.icon); }
    ImU32 Theme::overlay_highlight_u32() const { return toU32WithAlpha(overlay.highlight, OVERLAY_HIGHLIGHT_ALPHA); }
    ImU32 Theme::overlay_selection_u32() const { return toU32WithAlpha(overlay.selection, OVERLAY_SELECTION_ALPHA); }
    ImU32 Theme::overlay_selection_flash_u32() const { return toU32WithAlpha(overlay.selection_flash, OVERLAY_SELECTION_FLASH_ALPHA); }

    ImU32 Theme::progress_bar_bg_u32() const { return toU32WithAlpha(overlay.background, OVERLAY_BG_ALPHA); }
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
        const ImVec4 popup_bg = useLightPopupBackground(*this) ? palette.background : palette.surface;
        ImGui::PushStyleColor(ImGuiCol_PopupBg, popup_bg);
        ImGui::PushStyleColor(ImGuiCol_Border, palette.border);
        ImGui::PushStyleColor(ImGuiCol_Header, withAlpha(palette.primary, context_menu.header_alpha));
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, withAlpha(palette.primary, context_menu.header_hover_alpha));
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, withAlpha(palette.primary, context_menu.header_active_alpha));
        ImGui::PushStyleColor(ImGuiCol_Text, palette.text);
        const float dpi = g_dpi_scale;
        ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, context_menu.rounding * dpi);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(context_menu.padding.x * dpi, context_menu.padding.y * dpi));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(context_menu.item_spacing.x * dpi, context_menu.item_spacing.y * dpi));
    }

    void Theme::popContextMenuStyle() {
        ImGui::PopStyleVar(3);
        ImGui::PopStyleColor(6);
    }

    void Theme::pushModalStyle() const {
        constexpr float MODAL_BG_ALPHA = 0.98f;
        constexpr float MODAL_BORDER_SIZE = 2.0f;
        constexpr float MODAL_PADDING_X = 20.0f;
        constexpr float MODAL_PADDING_Y = 15.0f;
        constexpr float TITLE_DARKEN = 0.1f;
        constexpr float TITLE_ACTIVE_DARKEN = 0.05f;
        const bool is_light_popup = useLightPopupBackground(*this);
        const ImVec4 modal_surface = is_light_popup ? palette.background : palette.surface;
        const ImVec4 popup_bg{modal_surface.x, modal_surface.y, modal_surface.z, MODAL_BG_ALPHA};
        const ImVec4 title_bg = darken(modal_surface, is_light_popup ? 0.0f : TITLE_DARKEN);
        const ImVec4 title_bg_active = darken(modal_surface, is_light_popup ? 0.0f : TITLE_ACTIVE_DARKEN);

        ImGui::PushStyleColor(ImGuiCol_WindowBg, popup_bg);
        ImGui::PushStyleColor(ImGuiCol_PopupBg, popup_bg);
        ImGui::PushStyleColor(ImGuiCol_TitleBg, title_bg);
        ImGui::PushStyleColor(ImGuiCol_TitleBgActive, title_bg_active);
        ImGui::PushStyleColor(ImGuiCol_Border, palette.border);
        ImGui::PushStyleColor(ImGuiCol_Text, palette.text);
        const float dpi = g_dpi_scale;
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, MODAL_BORDER_SIZE * dpi);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, sizes.popup_rounding * dpi);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(MODAL_PADDING_X * dpi, MODAL_PADDING_Y * dpi));
    }

    void Theme::popModalStyle() {
        ImGui::PopStyleVar(3);
        ImGui::PopStyleColor(6);
    }

    void setThemeDpiScale(const float scale) { g_dpi_scale = scale; }
    float getThemeDpiScale() { return g_dpi_scale; }

    // Global access
    const Theme& theme() {
        ensureInitialized();
        return g_current_theme;
    }

    namespace {
        ThemeChangeCallback g_theme_change_cb;
    }

    void setThemeChangeCallback(ThemeChangeCallback cb) { g_theme_change_cb = std::move(cb); }

    const std::string& currentThemeId() {
        ensureInitialized();
        return g_current_theme_id;
    }

    std::string normalizeThemeId(std::string name) {
        return normalizeThemeIdImpl(std::move(name));
    }

    void setTheme(const Theme& t) {
        applyCurrentTheme(t, normalizeThemeIdImpl(t.name));
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

        const bool is_light = g_current_theme.isLightTheme();
        const bool is_light_popup = useLightPopupBackground(g_current_theme);
        const ImVec4 window_bg = p.surface;
        const ImVec4 child_bg = is_light ? darken(p.surface, 0.015f) : darken(p.surface, 0.025f);
        const ImVec4 popup_bg = is_light_popup ? lighten(p.surface, 0.02f) : lighten(p.surface, 0.01f);
        const ImVec4 title_bg = is_light ? darken(p.surface, 0.02f) : lighten(p.surface, 0.035f);
        const ImVec4 title_bg_active = is_light ? mix(p.surface, p.surface_bright, 0.55f)
                                                : mix(p.surface, p.surface_bright, 0.75f);
        const ImVec4 button_bg = is_light ? darken(p.surface, 0.01f) : darken(p.surface, 0.015f);
        const ImVec4 tab_bg = is_light ? darken(p.surface, 0.01f) : darken(p.surface, 0.008f);
        const ImVec4 tab_active_bg = is_light ? mix(p.surface_bright, p.primary_dim, 0.10f)
                                              : mix(p.surface_bright, p.primary_dim, 0.18f);
        ImVec4* const colors = style.Colors;
        colors[ImGuiCol_Text] = p.text;
        colors[ImGuiCol_TextDisabled] = p.text_dim;
        colors[ImGuiCol_WindowBg] = window_bg;
        colors[ImGuiCol_ChildBg] = child_bg;
        colors[ImGuiCol_PopupBg] = popup_bg;
        colors[ImGuiCol_Border] = p.border;
        colors[ImGuiCol_BorderShadow] = ImVec4(0, 0, 0, 0);

        const float frame_darken = g_current_theme.frameDarkenAmount();
        colors[ImGuiCol_FrameBg] = darken(window_bg, frame_darken);
        colors[ImGuiCol_FrameBgHovered] = is_light ? darken(p.surface, 0.08f) : p.surface_bright;
        colors[ImGuiCol_FrameBgActive] = p.primary_dim;
        colors[ImGuiCol_TitleBg] = title_bg;
        colors[ImGuiCol_TitleBgActive] = title_bg_active;
        colors[ImGuiCol_TitleBgCollapsed] = title_bg;
        colors[ImGuiCol_MenuBarBg] = title_bg;
        colors[ImGuiCol_ScrollbarBg] = darken(p.background, 0.05f);
        colors[ImGuiCol_ScrollbarGrab] = p.surface_bright;
        colors[ImGuiCol_ScrollbarGrabHovered] = lighten(p.surface_bright, 0.1f);
        colors[ImGuiCol_ScrollbarGrabActive] = p.primary;
        colors[ImGuiCol_CheckMark] = p.primary;
        colors[ImGuiCol_SliderGrab] = p.primary;
        colors[ImGuiCol_SliderGrabActive] = lighten(p.primary, 0.1f);
        colors[ImGuiCol_Button] = button_bg;
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
        colors[ImGuiCol_Tab] = tab_bg;
        colors[ImGuiCol_TabHovered] = p.surface_bright;
        colors[ImGuiCol_TabActive] = tab_active_bg;
        colors[ImGuiCol_TabUnfocused] = tab_bg;
        colors[ImGuiCol_TabUnfocusedActive] = tab_active_bg;
        colors[ImGuiCol_PlotLines] = p.primary;
        colors[ImGuiCol_PlotLinesHovered] = lighten(p.primary, 0.2f);
        colors[ImGuiCol_PlotHistogram] = p.primary;
        colors[ImGuiCol_PlotHistogramHovered] = lighten(p.primary, 0.2f);
        colors[ImGuiCol_TableHeaderBg] = title_bg;
        colors[ImGuiCol_TableBorderStrong] = p.border;
        colors[ImGuiCol_TableBorderLight] = withAlpha(p.border, 0.65f);
        colors[ImGuiCol_TableRowBg] = ImVec4(0, 0, 0, 0);
        colors[ImGuiCol_TableRowBgAlt] = withAlpha(p.surface_bright, is_light ? 0.16f : 0.14f);
        colors[ImGuiCol_TextSelectedBg] = withAlpha(p.primary, 0.35f);
        colors[ImGuiCol_DragDropTarget] = p.primary;
        colors[ImGuiCol_NavHighlight] = p.primary;
        colors[ImGuiCol_NavWindowingHighlight] = withAlpha(p.primary, 0.7f);
        colors[ImGuiCol_NavWindowingDimBg] = withAlpha(p.background, 0.2f);
        colors[ImGuiCol_ModalWindowDimBg] = withAlpha(p.background, 0.35f);

        style.FrameBorderSize = is_light ? 1.0f : 0.0f;

        style.ScaleAllSizes(g_dpi_scale);
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
            .menu = {},
            .context_menu = {},
            .viewport = {},
            .shadows = {},
            .vignette = {},
            .button = {},
            .overlay = {},
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
            .menu = {},
            .context_menu = {},
            .viewport = {},
            .shadows = {},
            .vignette = {},
            .button = {},
            .overlay = {
                .background = {0.95f, 0.95f, 0.96f, 1.0f},
                .text = {0.1f, 0.1f, 0.12f, 1.0f},
                .text_dim = {0.4f, 0.4f, 0.45f, 1.0f},
                .border = {0.5f, 0.55f, 0.6f, 1.0f},
                .icon = {0.3f, 0.4f, 0.5f, 1.0f},
                .highlight = {0.7f, 0.8f, 0.9f, 1.0f},
                .selection = {0.5f, 0.65f, 0.85f, 1.0f},
                .selection_flash = {0.65f, 0.78f, 0.95f, 1.0f},
            },
        };

        const Theme DEFAULT_GRUVBOX = {
            .name = "Gruvbox",
            .palette = {
                .background = {0.157f, 0.157f, 0.157f, 1.0f},     // #282828
                .surface = {0.235f, 0.220f, 0.212f, 1.0f},        // #3c3836
                .surface_bright = {0.314f, 0.286f, 0.271f, 1.0f}, // #504945
                .primary = {0.514f, 0.647f, 0.596f, 1.0f},        // #83a598
                .primary_dim = {0.271f, 0.522f, 0.533f, 1.0f},    // #458588
                .secondary = {0.827f, 0.525f, 0.608f, 1.0f},      // #d3869b
                .text = {0.922f, 0.859f, 0.698f, 1.0f},           // #ebdbb2
                .text_dim = {0.573f, 0.514f, 0.455f, 1.0f},       // #928374
                .border = {0.400f, 0.361f, 0.329f, 1.0f},         // #665c54
                .success = {0.722f, 0.733f, 0.149f, 1.0f},        // #b8bb26
                .warning = {0.980f, 0.741f, 0.184f, 1.0f},        // #fabd2f
                .error = {0.984f, 0.286f, 0.204f, 1.0f},          // #fb4934
                .info = {0.557f, 0.753f, 0.486f, 1.0f},           // #8ec07c
                .row_even = {1.0f, 1.0f, 1.0f, 0.035f},
                .row_odd = {0.0f, 0.0f, 0.0f, 0.14f},
            },
            .sizes = {},
            .fonts = {},
            .menu = {},
            .context_menu = {},
            .viewport = {},
            .shadows = {},
            .vignette = {},
            .button = {},
            .overlay = {
                .background = {0.235f, 0.220f, 0.212f, 1.0f},
                .text = {0.922f, 0.859f, 0.698f, 1.0f},
                .text_dim = {0.573f, 0.514f, 0.455f, 1.0f},
                .border = {0.514f, 0.459f, 0.424f, 1.0f}, // #82756a
                .icon = {0.514f, 0.647f, 0.596f, 1.0f},
                .highlight = {0.400f, 0.467f, 0.431f, 1.0f},
                .selection = {0.271f, 0.522f, 0.533f, 1.0f},
                .selection_flash = {0.557f, 0.753f, 0.486f, 1.0f},
            },
        };

        const Theme DEFAULT_CATPPUCCIN_MOCHA = {
            .name = "Catppuccin Mocha",
            .palette = {
                .background = {0.118f, 0.118f, 0.180f, 1.0f},
                .surface = {0.188f, 0.196f, 0.259f, 1.0f},
                .surface_bright = {0.271f, 0.278f, 0.353f, 1.0f},
                .primary = {0.537f, 0.706f, 0.980f, 1.0f},
                .primary_dim = {0.455f, 0.780f, 0.925f, 1.0f},
                .secondary = {0.796f, 0.651f, 0.969f, 1.0f},
                .text = {0.804f, 0.839f, 0.957f, 1.0f},
                .text_dim = {0.651f, 0.678f, 0.784f, 1.0f},
                .border = {0.345f, 0.353f, 0.443f, 1.0f},
                .success = {0.651f, 0.890f, 0.631f, 1.0f},
                .warning = {0.976f, 0.886f, 0.686f, 1.0f},
                .error = {0.953f, 0.545f, 0.659f, 1.0f},
                .info = {0.537f, 0.706f, 0.980f, 1.0f},
                .row_even = {1.0f, 1.0f, 1.0f, 0.035f},
                .row_odd = {0.0f, 0.0f, 0.0f, 0.13f},
            },
            .sizes = {},
            .fonts = {},
            .menu = {},
            .context_menu = {},
            .viewport = {},
            .shadows = {},
            .vignette = {},
            .button = {},
            .overlay = {
                .background = {0.188f, 0.196f, 0.259f, 1.0f},
                .text = {0.804f, 0.839f, 0.957f, 1.0f},
                .text_dim = {0.651f, 0.678f, 0.784f, 1.0f},
                .border = {0.455f, 0.780f, 0.925f, 1.0f},
                .icon = {0.537f, 0.706f, 0.980f, 1.0f},
                .highlight = {0.455f, 0.502f, 0.624f, 1.0f},
                .selection = {0.345f, 0.482f, 0.757f, 1.0f},
                .selection_flash = {0.651f, 0.890f, 0.631f, 1.0f},
            },
        };

        const Theme DEFAULT_CATPPUCCIN_LATTE = {
            .name = "Catppuccin Latte",
            .palette = {
                .background = {0.937f, 0.945f, 0.961f, 1.0f},
                .surface = {0.902f, 0.914f, 0.937f, 1.0f},
                .surface_bright = {0.863f, 0.878f, 0.910f, 1.0f},
                .primary = {0.118f, 0.400f, 0.961f, 1.0f},
                .primary_dim = {0.125f, 0.624f, 0.710f, 1.0f},
                .secondary = {0.533f, 0.224f, 0.937f, 1.0f},
                .text = {0.298f, 0.310f, 0.412f, 1.0f},
                .text_dim = {0.424f, 0.435f, 0.522f, 1.0f},
                .border = {0.675f, 0.690f, 0.741f, 1.0f},
                .success = {0.251f, 0.627f, 0.169f, 1.0f},
                .warning = {0.875f, 0.557f, 0.114f, 1.0f},
                .error = {0.824f, 0.059f, 0.224f, 1.0f},
                .info = {0.118f, 0.400f, 0.961f, 1.0f},
                .row_even = {0.0f, 0.0f, 0.0f, 0.03f},
                .row_odd = {0.0f, 0.0f, 0.0f, 0.08f},
            },
            .sizes = {},
            .fonts = {},
            .menu = {},
            .context_menu = {},
            .viewport = {},
            .shadows = {},
            .vignette = {},
            .button = {},
            .overlay = {
                .background = {0.937f, 0.945f, 0.961f, 1.0f},
                .text = {0.298f, 0.310f, 0.412f, 1.0f},
                .text_dim = {0.424f, 0.435f, 0.522f, 1.0f},
                .border = {0.675f, 0.690f, 0.741f, 1.0f},
                .icon = {0.125f, 0.624f, 0.710f, 1.0f},
                .highlight = {0.804f, 0.839f, 0.957f, 1.0f},
                .selection = {0.627f, 0.729f, 0.949f, 1.0f},
                .selection_flash = {0.745f, 0.816f, 0.969f, 1.0f},
            },
        };

        const Theme DEFAULT_NORD = {
            .name = "Nord",
            .palette = {
                .background = {0.180f, 0.204f, 0.251f, 1.0f},
                .surface = {0.231f, 0.259f, 0.322f, 1.0f},
                .surface_bright = {0.263f, 0.298f, 0.369f, 1.0f},
                .primary = {0.533f, 0.753f, 0.816f, 1.0f},
                .primary_dim = {0.369f, 0.506f, 0.675f, 1.0f},
                .secondary = {0.706f, 0.557f, 0.678f, 1.0f},
                .text = {0.925f, 0.937f, 0.957f, 1.0f},
                .text_dim = {0.722f, 0.753f, 0.816f, 1.0f},
                .border = {0.298f, 0.333f, 0.420f, 1.0f},
                .success = {0.639f, 0.745f, 0.549f, 1.0f},
                .warning = {0.922f, 0.796f, 0.545f, 1.0f},
                .error = {0.749f, 0.380f, 0.416f, 1.0f},
                .info = {0.561f, 0.737f, 0.733f, 1.0f},
                .row_even = {1.0f, 1.0f, 1.0f, 0.032f},
                .row_odd = {0.0f, 0.0f, 0.0f, 0.12f},
            },
            .sizes = {},
            .fonts = {},
            .menu = {},
            .context_menu = {},
            .viewport = {},
            .shadows = {},
            .vignette = {},
            .button = {},
            .overlay = {
                .background = {0.231f, 0.259f, 0.322f, 1.0f},
                .text = {0.925f, 0.937f, 0.957f, 1.0f},
                .text_dim = {0.722f, 0.753f, 0.816f, 1.0f},
                .border = {0.533f, 0.753f, 0.816f, 1.0f},
                .icon = {0.561f, 0.737f, 0.733f, 1.0f},
                .highlight = {0.369f, 0.427f, 0.518f, 1.0f},
                .selection = {0.369f, 0.506f, 0.675f, 1.0f},
                .selection_flash = {0.639f, 0.745f, 0.549f, 1.0f},
            },
        };

        struct ThemePresetRecord {
            const char* id;
            const char* asset_name;
            Theme* theme;
            const Theme* defaults;
            std::filesystem::path* path;
            std::filesystem::file_time_type* mtime;
        };

        ThemePresetRecord THEME_PRESETS[] = {
            {"dark", "themes/dark.json", &g_dark_theme, &DEFAULT_DARK, &g_dark_path, &g_dark_mtime},
            {"light", "themes/light.json", &g_light_theme, &DEFAULT_LIGHT, &g_light_path, &g_light_mtime},
            {"gruvbox", "themes/gruvbox.json", &g_gruvbox_theme, &DEFAULT_GRUVBOX, &g_gruvbox_path, &g_gruvbox_mtime},
            {"catppuccin_mocha", "themes/catppuccin_mocha.json", &g_catppuccin_mocha_theme, &DEFAULT_CATPPUCCIN_MOCHA, &g_catppuccin_mocha_path, &g_catppuccin_mocha_mtime},
            {"catppuccin_latte", "themes/catppuccin_latte.json", &g_catppuccin_latte_theme, &DEFAULT_CATPPUCCIN_LATTE, &g_catppuccin_latte_path, &g_catppuccin_latte_mtime},
            {"nord", "themes/nord.json", &g_nord_theme, &DEFAULT_NORD, &g_nord_path, &g_nord_mtime},
        };

        ThemePresetRecord* findThemePreset(std::string_view theme_id) {
            const auto normalized = normalizeThemeIdImpl(std::string(theme_id));
            for (auto& preset : THEME_PRESETS) {
                if (preset.id == normalized)
                    return &preset;
            }
            return nullptr;
        }

        bool isKnownThemePresetId(std::string_view theme_id) {
            return findThemePreset(theme_id) != nullptr;
        }

        void loadThemePreset(ThemePresetRecord& preset) {
            *preset.theme = *preset.defaults;
            preset.path->clear();

            try {
                *preset.path = getAssetPath(preset.asset_name);
                if (!loadTheme(*preset.theme, lfs::core::path_to_utf8(*preset.path)))
                    return;

                *preset.mtime = std::filesystem::last_write_time(*preset.path);
                LOG_INFO("Loaded {} theme from {}", preset.id, lfs::core::path_to_utf8(*preset.path));
            } catch (...) {
                preset.path->clear();
            }
        }

        bool hotReloadThemePreset(ThemePresetRecord& preset) {
            if (preset.path->empty() || !std::filesystem::exists(*preset.path))
                return false;

            const auto mtime = std::filesystem::last_write_time(*preset.path);
            if (mtime == *preset.mtime)
                return false;

            Theme reloaded = *preset.defaults;
            if (!loadTheme(reloaded, lfs::core::path_to_utf8(*preset.path)))
                return false;

            *preset.theme = std::move(reloaded);
            *preset.mtime = mtime;
            LOG_INFO("Hot-reloaded {} theme", preset.id);
            return true;
        }

        void loadThemesFromFiles() {
            for (auto& preset : THEME_PRESETS) {
                loadThemePreset(preset);
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

    const Theme& gruvboxTheme() {
        ensureThemesLoaded();
        return g_gruvbox_theme;
    }

    const Theme& catppuccinMochaTheme() {
        ensureThemesLoaded();
        return g_catppuccin_mocha_theme;
    }

    const Theme& catppuccinLatteTheme() {
        ensureThemesLoaded();
        return g_catppuccin_latte_theme;
    }

    const Theme& nordTheme() {
        ensureThemesLoaded();
        return g_nord_theme;
    }

    void visitThemePresets(const ThemePresetVisitor& visitor) {
        ensureThemesLoaded();
        for (const auto& preset : THEME_PRESETS) {
            visitor(preset.id, *preset.theme);
        }
    }

    namespace {
        void applyCurrentTheme(const Theme& theme, std::string_view theme_id) {
            g_current_theme = theme;
            g_current_theme_id = std::string(theme_id);
            g_initialized = true;
            applyThemeToImGui();
            if (g_theme_change_cb)
                g_theme_change_cb(g_current_theme_id);
        }

        bool activateThemePreset(std::string_view theme_id) {
            ensureThemesLoaded();

            const auto* preset = findThemePreset(theme_id);
            if (!preset)
                return false;

            applyCurrentTheme(*preset->theme, preset->id);
            return true;
        }
    } // namespace

    bool setThemeByName(const std::string& name) {
        return activateThemePreset(name);
    }

    bool checkThemeFileChanges() {
        if (!g_themes_loaded)
            return false;

        const std::string active_theme_id = g_current_theme_id;
        bool any_reloaded = false;
        bool active_theme_reloaded = false;

        for (auto& preset : THEME_PRESETS) {
            if (!hotReloadThemePreset(preset))
                continue;

            any_reloaded = true;
            if (active_theme_id == preset.id)
                active_theme_reloaded = true;
        }

        if (active_theme_reloaded && !activateThemePreset(active_theme_id)) {
            activateThemePreset("dark");
        }

        return any_reloaded;
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

            auto& button = j["button"];
            button["tint_normal"] = t.button.tint_normal;
            button["tint_hover"] = t.button.tint_hover;
            button["tint_active"] = t.button.tint_active;

            auto& overlay = j["overlay"];
            overlay["background"] = colorToJson(t.overlay.background);
            overlay["text"] = colorToJson(t.overlay.text);
            overlay["text_dim"] = colorToJson(t.overlay.text_dim);
            overlay["border"] = colorToJson(t.overlay.border);
            overlay["icon"] = colorToJson(t.overlay.icon);
            overlay["highlight"] = colorToJson(t.overlay.highlight);
            overlay["selection"] = colorToJson(t.overlay.selection);
            overlay["selection_flash"] = colorToJson(t.overlay.selection_flash);

            std::ofstream file;
            if (!lfs::core::open_file_for_write(lfs::core::utf8_to_path(path), file))
                return false;
            file << j.dump(2);
            return true;
        } catch (...) {
            return false;
        }
    }

    bool loadTheme(Theme& t, const std::string& path) {
        try {
            std::ifstream file;
            if (!lfs::core::open_file_for_read(lfs::core::utf8_to_path(path), file))
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

            if (j.contains("overlay")) {
                const auto& o = j["overlay"];
                if (o.contains("background"))
                    t.overlay.background = colorFromJson(o["background"]);
                if (o.contains("text"))
                    t.overlay.text = colorFromJson(o["text"]);
                if (o.contains("text_dim"))
                    t.overlay.text_dim = colorFromJson(o["text_dim"]);
                if (o.contains("border"))
                    t.overlay.border = colorFromJson(o["border"]);
                if (o.contains("icon"))
                    t.overlay.icon = colorFromJson(o["icon"]);
                if (o.contains("highlight"))
                    t.overlay.highlight = colorFromJson(o["highlight"]);
                if (o.contains("selection"))
                    t.overlay.selection = colorFromJson(o["selection"]);
                if (o.contains("selection_flash"))
                    t.overlay.selection_flash = colorFromJson(o["selection_flash"]);
            }

            return true;
        } catch (...) {
            return false;
        }
    }

    namespace {
        std::filesystem::path getThemeConfigDir() {
            std::filesystem::path config_dir;
#ifdef _WIN32
            const char* path = std::getenv("APPDATA");
            if (path) {
                config_dir = std::filesystem::path(path) / "LichtFeldStudio";
            } else {
                config_dir = std::filesystem::current_path() / "config";
            }
#else
            const char* xdg = std::getenv("XDG_CONFIG_HOME");
            if (xdg) {
                config_dir = std::filesystem::path(xdg) / "LichtFeldStudio";
            } else {
                const char* home = std::getenv("HOME");
                if (home) {
                    config_dir = std::filesystem::path(home) / ".config" / "LichtFeldStudio";
                } else {
                    config_dir = std::filesystem::current_path() / "config";
                }
            }
#endif
            return config_dir;
        }
    } // namespace

    void saveThemePreferenceName(const std::string& theme_name) {
        try {
            const auto config_dir = getThemeConfigDir();
            std::filesystem::create_directories(config_dir);
            const auto pref_path = config_dir / "theme_preference";
            std::ofstream file(pref_path);
            if (file) {
                const std::string normalized = normalizeThemeIdImpl(theme_name);
                if (isKnownThemePresetId(normalized)) {
                    file << normalized;
                } else {
                    file << "dark";
                }
            }
        } catch (...) {
            // Silently ignore - not critical
        }
    }

    std::string loadThemePreferenceName() {
        try {
            const auto config_dir = getThemeConfigDir();
            const auto pref_path = config_dir / "theme_preference";
            if (std::filesystem::exists(pref_path)) {
                std::ifstream file(pref_path);
                std::string pref;
                if (file >> pref) {
                    const std::string normalized = normalizeThemeIdImpl(pref);
                    if (isKnownThemePresetId(normalized)) {
                        return normalized;
                    }
                }
            }
        } catch (...) {
            // Silently ignore - not critical
        }
        return "dark";
    }

    void saveThemePreference(const bool is_dark) {
        saveThemePreferenceName(is_dark ? "dark" : "light");
    }

    bool loadThemePreference() {
        return loadThemePreferenceName() != "light";
    }

    void saveUiScalePreference(float scale) {
        try {
            const auto config_dir = getThemeConfigDir();
            std::filesystem::create_directories(config_dir);
            const auto pref_path = config_dir / "ui_scale";
            std::ofstream file(pref_path);
            if (file) {
                file << scale;
            }
        } catch (const std::exception& e) {
            LOG_WARN("Failed to save UI scale preference: {}", e.what());
        }
    }

    float loadUiScalePreference() {
        try {
            const auto config_dir = getThemeConfigDir();
            const auto pref_path = config_dir / "ui_scale";
            if (std::filesystem::exists(pref_path)) {
                std::ifstream file(pref_path);
                float scale = 0.0f;
                if (file >> scale)
                    return scale;
            }
        } catch (const std::exception& e) {
            LOG_WARN("Failed to load UI scale preference: {}", e.what());
        }
        return 0.0f;
    }

} // namespace lfs::vis
