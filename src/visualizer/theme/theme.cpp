/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "theme.hpp"
#include "core/config_paths.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "internal/resource_paths.hpp"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <future>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <utility>
#include <vector>

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
        float g_dpi_scale = 1.0f;
        bool g_initialized = false;
        bool g_themes_loaded = false;

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

        json colorToJson(const ThemeColor& c) {
            return json::array({c.x, c.y, c.z, c.w});
        }

        ThemeColor colorFromJson(const json& j) {
            if (j.is_array() && j.size() >= 4) {
                return {j[0].get<float>(), j[1].get<float>(), j[2].get<float>(), j[3].get<float>()};
            }
            return {0.0f, 0.0f, 0.0f, 1.0f};
        }

        json vec2ToJson(const ThemeVec2& v) {
            return json::array({v.x, v.y});
        }

        ThemeVec2 vec2FromJson(const json& j) {
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

    } // namespace

    // Color utilities
    ThemeColor lighten(const ThemeColor& color, const float amount) {
        return {
            std::min(1.0f, color.x + amount),
            std::min(1.0f, color.y + amount),
            std::min(1.0f, color.z + amount),
            color.w};
    }

    ThemeColor darken(const ThemeColor& color, const float amount) {
        return {
            std::max(0.0f, color.x - amount),
            std::max(0.0f, color.y - amount),
            std::max(0.0f, color.z - amount),
            color.w};
    }

    ThemeColor withAlpha(const ThemeColor& color, const float alpha) {
        return {color.x, color.y, color.z, alpha};
    }

    // Theme computed colors
    ThemeColor Theme::toolbar_background() const { return withAlpha(palette.surface, TOOLBAR_BG_ALPHA); }
    ThemeColor Theme::subtoolbar_background() const { return withAlpha(darken(palette.surface, 0.03f), SUBTOOLBAR_BG_ALPHA); }

    ThemeColor Theme::menu_background() const { return lighten(palette.surface, menu.bg_lighten); }
    ThemeColor Theme::menu_hover() const { return lighten(palette.surface_bright, menu.hover_lighten); }
    ThemeColor Theme::menu_active() const { return withAlpha(palette.primary, menu.active_alpha); }
    ThemeColor Theme::menu_popup_background() const { return lighten(palette.surface, menu.popup_lighten); }
    ThemeColor Theme::menu_border() const { return withAlpha(palette.border, menu.border_alpha); }
    void Theme::pushModalStyle() const {}

    void Theme::popModalStyle() {}

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

    void setTheme(const Theme& t) {
        applyCurrentTheme(t, normalizeThemeIdImpl(t.name));
    }

    namespace {
        void applyThemePreservingCurrentId(const Theme& t) {
            ensureInitialized();

            // Keep runtime style tweaks attached to the active preset ID so
            // RML theme activation and preset hot-reload continue to target
            // the selected preset even if the theme JSON name is customized.
            const std::string active_theme_id = g_current_theme_id;
            applyCurrentTheme(t, active_theme_id);
        }
    } // namespace

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

        constexpr std::string_view THEMES_MANIFEST_ASSET_NAME = "themes/manifest.json";
        constexpr std::string_view THEMES_ASSET_PREFIX = "themes/";

        struct ThemeDefaultRecord {
            std::string_view id;
            const Theme* theme;
        };

        const ThemeDefaultRecord THEME_DEFAULTS[] = {
            {"dark", &DEFAULT_DARK},
            {"light", &DEFAULT_LIGHT},
            {"gruvbox", &DEFAULT_GRUVBOX},
            {"catppuccin_mocha", &DEFAULT_CATPPUCCIN_MOCHA},
            {"catppuccin_latte", &DEFAULT_CATPPUCCIN_LATTE},
            {"nord", &DEFAULT_NORD},
        };

        struct ThemePresetRecord {
            ThemePresetRecord(
                std::string preset_id,
                std::string preset_asset_name,
                const Theme* preset_defaults,
                ThemePresetInfo preset_info)
                : id(std::move(preset_id)),
                  asset_name(std::move(preset_asset_name)),
                  defaults(preset_defaults),
                  theme(*preset_defaults),
                  info(std::move(preset_info)) {}

            std::string id;
            std::string asset_name;
            const Theme* defaults;
            Theme theme;
            ThemePresetInfo info;
            std::filesystem::path path;
            std::filesystem::file_time_type mtime{};
        };

        std::vector<ThemePresetRecord> g_theme_presets;
        std::filesystem::path g_theme_manifest_path;
        std::filesystem::file_time_type g_theme_manifest_mtime{};

        ThemePresetRecord* findThemePreset(std::string_view theme_id) {
            const auto normalized = normalizeThemeIdImpl(std::string(theme_id));
            for (auto& preset : g_theme_presets) {
                if (normalized == preset.id)
                    return &preset;
            }
            return nullptr;
        }

        const Theme* findThemeDefaults(std::string_view theme_id) {
            const auto normalized = normalizeThemeIdImpl(std::string(theme_id));
            for (const auto& defaults : THEME_DEFAULTS) {
                if (normalized == defaults.id)
                    return defaults.theme;
            }
            return nullptr;
        }

        bool isKnownThemePresetId(std::string_view theme_id) {
            ensureThemesLoaded();
            return findThemePreset(theme_id) != nullptr;
        }

        void syncThemePresetName(ThemePresetRecord& preset) {
            preset.info.name = preset.theme.name.empty() ? preset.defaults->name : preset.theme.name;
        }

        bool isSafeThemeRelativeFile(const std::string& file_name) {
            if (file_name.empty())
                return false;
            if (file_name.find(':') != std::string::npos)
                return false;

            const std::filesystem::path relative_path = lfs::core::utf8_to_path(file_name);
            if (relative_path.is_absolute() || relative_path.has_root_name() || relative_path.has_root_directory())
                return false;

            for (const auto& part : relative_path) {
                if (part == "..")
                    return false;
            }
            return true;
        }

        ThemePresetRecord makeFallbackDarkPreset() {
            ThemePresetInfo info{
                .id = "dark",
                .name = DEFAULT_DARK.name,
                .label_key = "menu.view.theme.dark",
                .mode = "dark",
                .order = 10,
            };
            return ThemePresetRecord("dark", "themes/dark.json", &DEFAULT_DARK, std::move(info));
        }

        std::optional<ThemePresetRecord> parseThemeManifestEntry(
            const json& entry,
            const std::size_t entry_index,
            std::set<std::string>& ids) {
            if (!entry.is_object()) {
                LOG_WARN("Ignoring theme manifest entry {}: expected object", entry_index);
                return std::nullopt;
            }

            const std::string raw_id = entry.value("id", "");
            const std::string id = normalizeThemeIdImpl(raw_id);
            if (raw_id.empty() || id != raw_id) {
                LOG_WARN("Ignoring theme manifest entry {}: invalid id '{}'", entry_index, raw_id);
                return std::nullopt;
            }

            if (!ids.insert(id).second) {
                LOG_WARN("Ignoring duplicate theme id '{}' in manifest", id);
                return std::nullopt;
            }

            const std::string file = entry.value("file", "");
            if (!isSafeThemeRelativeFile(file)) {
                LOG_WARN("Ignoring theme '{}' in manifest: unsafe or empty file '{}'", id, file);
                return std::nullopt;
            }

            std::string fallback_id = normalizeThemeIdImpl(entry.value("fallback", id));
            const Theme* defaults = findThemeDefaults(fallback_id);
            if (!defaults) {
                LOG_WARN("Theme '{}' references unknown fallback '{}'; using dark", id, fallback_id);
                fallback_id = "dark";
                defaults = &DEFAULT_DARK;
            }

            ThemePresetInfo info{
                .id = id,
                .name = defaults->name,
                .label_key = entry.value("label_key", "menu.view.theme." + id),
                .mode = entry.value(
                    "mode",
                    std::string(defaults->isLightTheme() ? "light" : "dark")),
                .order = entry.value("order", static_cast<int>((entry_index + 1) * 10)),
            };

            if (info.mode != "dark" && info.mode != "light") {
                LOG_WARN("Theme '{}' has invalid mode '{}'; deriving mode from fallback", id, info.mode);
                info.mode = defaults->isLightTheme() ? "light" : "dark";
            }

            return ThemePresetRecord(
                id,
                std::string(THEMES_ASSET_PREFIX) + file,
                defaults,
                std::move(info));
        }

        std::vector<ThemePresetRecord> loadThemeCatalogFromManifest() {
            std::vector<ThemePresetRecord> presets;

            try {
                g_theme_manifest_path = getAssetPath(std::string(THEMES_MANIFEST_ASSET_NAME));
                g_theme_manifest_mtime = std::filesystem::last_write_time(g_theme_manifest_path);

                std::ifstream file;
                if (!lfs::core::open_file_for_read(g_theme_manifest_path, file))
                    throw std::runtime_error("could not open manifest");

                json manifest;
                file >> manifest;

                const int schema_version = manifest.value("schema_version", 0);
                if (schema_version != 1)
                    throw std::runtime_error("unsupported schema_version " + std::to_string(schema_version));

                const auto themes_it = manifest.find("themes");
                if (themes_it == manifest.end() || !themes_it->is_array())
                    throw std::runtime_error("themes must be an array");

                std::set<std::string> ids;
                for (std::size_t i = 0; i < themes_it->size(); ++i) {
                    if (auto preset = parseThemeManifestEntry((*themes_it)[i], i, ids)) {
                        presets.push_back(std::move(*preset));
                    }
                }

                std::stable_sort(
                    presets.begin(),
                    presets.end(),
                    [](const ThemePresetRecord& a, const ThemePresetRecord& b) {
                        return a.info.order < b.info.order;
                    });

                if (presets.empty())
                    throw std::runtime_error("manifest did not define any valid themes");
            } catch (const std::exception& e) {
                LOG_WARN("Failed to load theme manifest: {}; falling back to built-in dark theme", e.what());
                presets.push_back(makeFallbackDarkPreset());
            }

            return presets;
        }

        void loadThemePreset(ThemePresetRecord& preset) {
            preset.theme = *preset.defaults;
            preset.path.clear();
            syncThemePresetName(preset);

            try {
                preset.path = getAssetPath(preset.asset_name);
                if (!loadTheme(preset.theme, lfs::core::path_to_utf8(preset.path)))
                    return;

                syncThemePresetName(preset);
                preset.mtime = std::filesystem::last_write_time(preset.path);
                LOG_INFO("Loaded {} theme from {}", preset.id, lfs::core::path_to_utf8(preset.path));
            } catch (...) {
                preset.path.clear();
            }
        }

        bool hotReloadThemePreset(ThemePresetRecord& preset) {
            if (preset.path.empty() || !std::filesystem::exists(preset.path))
                return false;

            const auto mtime = std::filesystem::last_write_time(preset.path);
            if (mtime == preset.mtime)
                return false;

            Theme reloaded = *preset.defaults;
            if (!loadTheme(reloaded, lfs::core::path_to_utf8(preset.path)))
                return false;

            preset.theme = std::move(reloaded);
            syncThemePresetName(preset);
            preset.mtime = mtime;
            LOG_INFO("Hot-reloaded {} theme", preset.id);
            return true;
        }

        void loadThemesFromFiles() {
            g_theme_presets = loadThemeCatalogFromManifest();

            std::vector<std::future<void>> jobs;
            jobs.reserve(g_theme_presets.size());
            for (auto& preset : g_theme_presets) {
                jobs.emplace_back(std::async(std::launch::async, [&preset]() {
                    loadThemePreset(preset);
                }));
            }
            for (auto& job : jobs) {
                job.get();
            }

            g_themes_loaded = true;
        }

        void ensureThemesLoaded() {
            if (!g_themes_loaded) {
                loadThemesFromFiles();
            }
        }

    } // namespace

    namespace {
        const Theme& themePreset(std::string_view theme_id) {
            ensureThemesLoaded();
            const auto* preset = findThemePreset(theme_id);
            return preset ? preset->theme : g_theme_presets.front().theme;
        }
    } // namespace

    const Theme& darkTheme() {
        return themePreset("dark");
    }

    void visitThemePresets(const ThemePresetVisitor& visitor) {
        ensureThemesLoaded();
        for (const auto& preset : g_theme_presets) {
            visitor(preset.id, preset.theme);
        }
    }

    void visitThemePresetInfos(const ThemePresetInfoVisitor& visitor) {
        ensureThemesLoaded();
        for (const auto& preset : g_theme_presets) {
            visitor(preset.info);
        }
    }

    namespace {
        void applyCurrentTheme(const Theme& theme, std::string_view theme_id) {
            g_current_theme = theme;
            g_current_theme_id = std::string(theme_id);
            g_initialized = true;
            if (g_theme_change_cb)
                g_theme_change_cb(g_current_theme_id);
        }

        bool activateThemePreset(std::string_view theme_id) {
            ensureThemesLoaded();

            const auto* preset = findThemePreset(theme_id);
            if (!preset)
                return false;

            applyCurrentTheme(preset->theme, preset->id);
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

        try {
            if (!g_theme_manifest_path.empty() && std::filesystem::exists(g_theme_manifest_path)) {
                const auto manifest_mtime = std::filesystem::last_write_time(g_theme_manifest_path);
                if (manifest_mtime != g_theme_manifest_mtime) {
                    LOG_INFO("Hot-reloading theme manifest");
                    loadThemesFromFiles();
                    if (!activateThemePreset(active_theme_id))
                        activateThemePreset("dark");
                    return true;
                }
            }
        } catch (...) {
            LOG_WARN("Failed to check theme manifest for hot reload");
        }

        for (auto& preset : g_theme_presets) {
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
            return lfs::core::user_config_dir();
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

    void setThemeVignetteEnabled(bool enabled) {
        Theme t = theme();
        t.vignette.enabled = enabled;
        applyThemePreservingCurrentId(t);
    }

    void setThemeVignetteIntensity(float intensity) {
        Theme t = theme();
        t.vignette.intensity = std::clamp(intensity, 0.0f, 1.0f);
        applyThemePreservingCurrentId(t);
    }

    void setThemeVignetteStyle(float intensity, float radius, float softness) {
        Theme t = theme();
        t.vignette.intensity = std::clamp(intensity, 0.0f, 1.0f);
        t.vignette.radius = std::clamp(radius, 0.0f, 1.0f);
        t.vignette.softness = std::clamp(softness, 0.0f, 1.0f);
        applyThemePreservingCurrentId(t);
    }

} // namespace lfs::vis
