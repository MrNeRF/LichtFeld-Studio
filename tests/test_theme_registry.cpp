/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/error.hpp"
#include "core/path_utils.hpp"
#include "core/user_paths.hpp"
#include "visualizer/internal/resource_paths.hpp"
#include "visualizer/preferences.hpp"
#include "visualizer/theme/theme.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace {

    class ScopedLfsHome {
    public:
        explicit ScopedLfsHome(const std::filesystem::path& path) {
            if (const char* previous = std::getenv("LFS_HOME"))
                previous_ = previous;
#ifdef _WIN32
            (void)_putenv_s("LFS_HOME", path.string().c_str());
#else
            (void)setenv("LFS_HOME", path.string().c_str(), 1);
#endif
        }

        ~ScopedLfsHome() {
#ifdef _WIN32
            (void)_putenv_s("LFS_HOME", previous_ ? previous_->c_str() : "");
#else
            if (previous_)
                (void)setenv("LFS_HOME", previous_->c_str(), 1);
            else
                (void)unsetenv("LFS_HOME");
#endif
        }

    private:
        std::optional<std::string> previous_;
    };

    class ScopedEnvVar {
    public:
        ScopedEnvVar(const char* name, const std::string& value) : name_(name) {
            if (const char* previous = std::getenv(name_))
                previous_ = previous;
#ifdef _WIN32
            (void)_putenv_s(name_, value.c_str());
#else
            (void)setenv(name_, value.c_str(), 1);
#endif
        }

        ~ScopedEnvVar() {
#ifdef _WIN32
            (void)_putenv_s(name_, previous_ ? previous_->c_str() : "");
#else
            if (previous_)
                (void)setenv(name_, previous_->c_str(), 1);
            else
                (void)unsetenv(name_);
#endif
        }

        ScopedEnvVar(const ScopedEnvVar&) = delete;
        ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

    private:
        const char* name_;
        std::optional<std::string> previous_;
    };

    class ScopedSafeMode {
    public:
        ScopedSafeMode() {
            if (const char* previous = std::getenv("LFS_SAFE_MODE"))
                previous_ = previous;
#ifdef _WIN32
            (void)_putenv_s("LFS_SAFE_MODE", "1");
#else
            (void)setenv("LFS_SAFE_MODE", "1", 1);
#endif
        }

        ~ScopedSafeMode() {
#ifdef _WIN32
            (void)_putenv_s("LFS_SAFE_MODE", previous_ ? previous_->c_str() : "");
#else
            if (previous_)
                (void)setenv("LFS_SAFE_MODE", previous_->c_str(), 1);
            else
                (void)unsetenv("LFS_SAFE_MODE");
#endif
        }

    private:
        std::optional<std::string> previous_;
    };

    std::vector<lfs::vis::ThemePresetInfo> themePresetInfos() {
        std::vector<lfs::vis::ThemePresetInfo> infos;
        lfs::vis::visitThemePresetInfos([&infos](const lfs::vis::ThemePresetInfo& info) {
            infos.push_back(info);
        });
        return infos;
    }

} // namespace

TEST(ThemeRegistry, VersionTwoManifestOwnsThemeFamilies) {
    const auto manifest_path = lfs::vis::getAssetPath("themes/manifest.json");

    std::ifstream manifest_file;
    ASSERT_TRUE(lfs::core::open_file_for_read(manifest_path, manifest_file));

    nlohmann::json manifest;
    manifest_file >> manifest;

    ASSERT_EQ(manifest.value("schema_version", 0), 2);
    ASSERT_TRUE(manifest.contains("families"));
    ASSERT_TRUE(manifest["families"].is_array());

    const auto infos = themePresetInfos();
    std::map<std::string, lfs::vis::ThemePresetInfo> info_by_id;
    for (const auto& info : infos) {
        info_by_id.emplace(info.id, info);
    }

    std::size_t variant_count = 0;
    for (const auto& family_entry : manifest["families"]) {
        ASSERT_TRUE(family_entry.is_object());
        ASSERT_TRUE(family_entry.contains("file"));
        ASSERT_TRUE(family_entry.contains("order"));

        const std::string family_file = family_entry["file"].get<std::string>();
        const auto theme_path = lfs::vis::getAssetPath("themes/" + family_file);

        std::ifstream theme_stream;
        ASSERT_TRUE(lfs::core::open_file_for_read(theme_path, theme_stream)) << family_file;

        nlohmann::json family;
        theme_stream >> family;
        ASSERT_EQ(family.value("schema_version", 0), 2) << family_file;
        ASSERT_TRUE(family.contains("id")) << family_file;
        ASSERT_TRUE(family.contains("name")) << family_file;
        ASSERT_TRUE(family.contains("variants")) << family_file;
        ASSERT_TRUE(family["variants"].is_object()) << family_file;
        EXPECT_FALSE(family.contains("fonts")) << family_file;
        if (const auto shared = family.find("shared"); shared != family.end())
            EXPECT_TRUE(shared->is_object()) << family_file;

        const std::string family_id = family["id"].get<std::string>();
        const std::string family_name = family["name"].get<std::string>();
        const int family_order = family_entry["order"].get<int>();
        for (const std::string mode : {"dark", "light"}) {
            const auto variant = family["variants"].find(mode);
            if (variant == family["variants"].end())
                continue;

            ASSERT_TRUE(variant->is_object()) << family_file << ':' << mode;
            ASSERT_TRUE(variant->contains("id")) << family_file << ':' << mode;
            ASSERT_TRUE(variant->contains("name")) << family_file << ':' << mode;
            ASSERT_TRUE(variant->contains("label_key")) << family_file << ':' << mode;
            ASSERT_TRUE(variant->contains("fallback")) << family_file << ':' << mode;
            if (variant->contains("fonts"))
                EXPECT_TRUE((*variant)["fonts"].is_object()) << family_file << ':' << mode;
            if (const auto gradients = variant->find("gradients"); gradients != variant->end()) {
                ASSERT_TRUE(gradients->is_object()) << family_file << ':' << mode;
                for (const auto& [name, gradient] : gradients->items()) {
                    if (name.starts_with('_'))
                        continue;
                    ASSERT_TRUE(gradient.is_object()) << family_file << ':' << mode << ':' << name;
                    ASSERT_TRUE(gradient.contains("start")) << name;
                    ASSERT_TRUE(gradient.contains("end")) << name;
                    EXPECT_TRUE(gradient["start"].is_array()) << name;
                    EXPECT_EQ(gradient["start"].size(), 4u) << name;
                    EXPECT_TRUE(gradient["end"].is_array()) << name;
                    EXPECT_EQ(gradient["end"].size(), 4u) << name;
                }
            }

            const std::string id = (*variant)["id"].get<std::string>();
            ASSERT_TRUE(info_by_id.contains(id)) << id;
            const auto& info = info_by_id.at(id);
            EXPECT_EQ(info.label_key, (*variant)["label_key"].get<std::string>()) << id;
            EXPECT_EQ(info.mode, mode) << id;
            EXPECT_EQ(info.family_id, family_id) << id;
            EXPECT_EQ(info.family_name, family_name) << id;
            EXPECT_EQ(info.variant_name, (*variant)["name"].get<std::string>()) << id;
            EXPECT_EQ(
                info.order,
                family_order + variant->value("order", mode == "light" ? 1 : 0))
                << id;
            ++variant_count;
        }
    }
    EXPECT_EQ(variant_count, info_by_id.size());
}

TEST(ThemeRegistry, CatalogIsStableAndSelfDescribing) {
    const auto infos = themePresetInfos();

    ASSERT_EQ(infos.size(), 8u);

    int previous_order = 0;
    std::set<std::string> ids;
    for (const auto& info : infos) {
        EXPECT_FALSE(info.id.empty());
        EXPECT_FALSE(info.name.empty()) << info.id;
        EXPECT_FALSE(info.label_key.empty()) << info.id;
        EXPECT_TRUE(info.mode == "dark" || info.mode == "light") << info.id;
        EXPECT_FALSE(info.family_id.empty()) << info.id;
        EXPECT_FALSE(info.family_name.empty()) << info.id;
        EXPECT_FALSE(info.variant_name.empty()) << info.id;
        EXPECT_GT(info.order, previous_order) << info.id;
        EXPECT_TRUE(ids.insert(info.id).second) << info.id;
        previous_order = info.order;
    }

    EXPECT_TRUE(ids.contains("dark"));
    EXPECT_TRUE(ids.contains("light"));
    EXPECT_TRUE(ids.contains("signal_night"));
    EXPECT_TRUE(ids.contains("signal_day"));
    EXPECT_TRUE(ids.contains("gruvbox"));
    EXPECT_TRUE(ids.contains("catppuccin_mocha"));
    EXPECT_TRUE(ids.contains("catppuccin_latte"));
    EXPECT_TRUE(ids.contains("nord"));
}

TEST(ThemeRegistry, CurrentThemeUsesStablePresetId) {
    const std::string original_theme = lfs::vis::currentThemeId();

    ASSERT_TRUE(lfs::vis::setThemeByName("Catppuccin Mocha"));
    EXPECT_EQ(lfs::vis::currentThemeId(), "catppuccin_mocha");
    EXPECT_EQ(lfs::vis::theme().name, "Catppuccin Mocha");

    ASSERT_TRUE(lfs::vis::setThemeByName("catppuccin-latte"));
    EXPECT_EQ(lfs::vis::currentThemeId(), "catppuccin_latte");
    EXPECT_EQ(lfs::vis::theme().name, "Catppuccin Latte");

    if (!original_theme.empty()) {
        EXPECT_TRUE(lfs::vis::setThemeByName(original_theme));
    }
}

TEST(ThemeRegistry, OptionalGradientsLoadAndLegacyThemesKeepDerivedFallbacks) {
    const std::string original_theme = lfs::vis::currentThemeId();

    ASSERT_TRUE(lfs::vis::setThemeByName("dark"));
    EXPECT_FALSE(lfs::vis::theme().gradients.window_title.has_value());
    EXPECT_FALSE(lfs::vis::theme().gradients.progress.has_value());

    ASSERT_TRUE(lfs::vis::setThemeByName("signal_night"));
    ASSERT_TRUE(lfs::vis::theme().gradients.window_body.has_value());
    ASSERT_TRUE(lfs::vis::theme().gradients.panel_body.has_value());
    ASSERT_TRUE(lfs::vis::theme().gradients.window_title.has_value());
    ASSERT_TRUE(lfs::vis::theme().gradients.progress.has_value());
    EXPECT_FLOAT_EQ(lfs::vis::theme().gradients.progress->start.x, 0.224f);
    EXPECT_FLOAT_EQ(lfs::vis::theme().gradients.progress->end.z, 1.0f);

    if (!original_theme.empty())
        EXPECT_TRUE(lfs::vis::setThemeByName(original_theme));
}

TEST(ThemeRegistry, InvalidOptionalGradientDoesNotRejectTheme) {
    const auto path =
        std::filesystem::temp_directory_path() / "lfs_invalid_optional_gradient.json";
    {
        std::ofstream file(path);
        file << R"({
            "name": "Invalid Gradient Theme",
            "palette": {"primary": [0.1, 0.2, 0.3, 1.0]},
            "gradients": {"progress": {"start": [2.0, 0.0, 0.0, 1.0], "end": [0.0, 1.0]}}
        })";
    }

    lfs::vis::Theme loaded = lfs::vis::darkTheme();
    ASSERT_TRUE(lfs::vis::loadTheme(loaded, lfs::core::path_to_utf8(path)));
    EXPECT_FLOAT_EQ(loaded.palette.primary.x, 0.1f);
    EXPECT_FALSE(loaded.gradients.progress.has_value());

    std::error_code error;
    std::filesystem::remove(path, error);
}

TEST(ThemeRegistry, OptionalGradientsRoundTripThroughLegacyThemeSave) {
    const auto path =
        std::filesystem::temp_directory_path() / "lfs_optional_gradient_roundtrip.json";

    lfs::vis::Theme source = lfs::vis::darkTheme();
    source.gradients.progress = lfs::vis::ThemeGradient{
        {0.1f, 0.2f, 0.3f, 0.4f},
        {0.5f, 0.6f, 0.7f, 0.8f}};
    ASSERT_TRUE(lfs::vis::saveTheme(source, lfs::core::path_to_utf8(path)));

    lfs::vis::Theme loaded = lfs::vis::darkTheme();
    ASSERT_TRUE(lfs::vis::loadTheme(loaded, lfs::core::path_to_utf8(path)));
    ASSERT_TRUE(loaded.gradients.progress.has_value());
    EXPECT_FLOAT_EQ(loaded.gradients.progress->start.w, 0.4f);
    EXPECT_FLOAT_EQ(loaded.gradients.progress->end.z, 0.7f);

    std::error_code error;
    std::filesystem::remove(path, error);
}

TEST(ThemeRegistry, LegacyStandaloneThemeJsonRemainsLoadable) {
    const auto path =
        std::filesystem::temp_directory_path() / "lfs_legacy_theme_v1.json";
    {
        std::ofstream file(path);
        file << R"({
            "name": "Legacy Theme",
            "palette": {"primary": [0.1, 0.2, 0.3, 1.0]},
            "fonts": {"large_size": 19.0},
            "button": {"tint_normal": 0.42}
        })";
    }

    lfs::vis::Theme loaded = lfs::vis::darkTheme();
    ASSERT_TRUE(lfs::vis::loadTheme(loaded, lfs::core::path_to_utf8(path)));
    EXPECT_EQ(loaded.name, "Legacy Theme");
    EXPECT_FLOAT_EQ(loaded.palette.primary.x, 0.1f);
    EXPECT_FLOAT_EQ(loaded.palette.primary.y, 0.2f);
    EXPECT_FLOAT_EQ(loaded.palette.primary.z, 0.3f);
    EXPECT_FLOAT_EQ(loaded.fonts.large_size, 19.0f);
    EXPECT_FLOAT_EQ(loaded.button.tint_normal, 0.42f);

    std::error_code error;
    std::filesystem::remove(path, error);
}

TEST(ThemePreferencesContract, InvalidValuesFallBackToBuiltInDefaults) {
    const auto root = std::filesystem::temp_directory_path() / "lfs_theme_preferences_invalid";
    std::error_code error;
    std::filesystem::remove_all(root, error);
    const ScopedLfsHome home(root);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths.has_value()) << lfs::format_for_developer(paths.error());
    ASSERT_TRUE(paths->ensureDirectories().has_value());
    std::ofstream(paths->preferencesFile())
        << R"({"theme":"not-a-theme","ui_scale":999,"language":42})";

    EXPECT_EQ(lfs::vis::loadThemePreferenceName(), "dark");
    EXPECT_FLOAT_EQ(lfs::vis::loadUiScalePreference(), 0.0f);
    EXPECT_TRUE(lfs::vis::loadLanguagePreference().empty());
    std::filesystem::remove_all(root, error);
}

TEST(ThemePreferencesContract, FamilyAutoSelectionResolvesWithoutLosingFamily) {
    const auto root = std::filesystem::temp_directory_path() / "lfs_theme_preferences_family_auto";
    std::error_code error;
    std::filesystem::remove_all(root, error);
    const ScopedLfsHome home(root);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths.has_value()) << lfs::format_for_developer(paths.error());
    ASSERT_TRUE(paths->ensureDirectories().has_value());
    std::ofstream(paths->preferencesFile()) << R"({"theme":"catppuccin:auto"})";

    const std::string resolved = lfs::vis::loadThemePreferenceName();
    EXPECT_TRUE(resolved == "catppuccin_mocha" || resolved == "catppuccin_latte");
    EXPECT_EQ(lfs::vis::currentThemeFamilyId(), "catppuccin");
    EXPECT_EQ(lfs::vis::currentThemeSelectionMode(),
              lfs::vis::supportsSystemThemePreference() ? "auto" : "dark");

    std::filesystem::remove_all(root, error);
}

TEST(ThemePreferencesContract, LegacyCatppuccinAliasStillSelectsMocha) {
    const auto root = std::filesystem::temp_directory_path() / "lfs_theme_preferences_legacy_catppuccin";
    std::error_code error;
    std::filesystem::remove_all(root, error);
    const ScopedLfsHome home(root);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths.has_value()) << lfs::format_for_developer(paths.error());
    ASSERT_TRUE(paths->ensureDirectories().has_value());
    std::ofstream(paths->preferencesFile()) << R"({"theme":"catppuccin"})";

    EXPECT_EQ(lfs::vis::loadThemePreferenceName(), "catppuccin_mocha");
    EXPECT_EQ(lfs::vis::currentThemeFamilyId(), "catppuccin");
    EXPECT_EQ(lfs::vis::currentThemeSelectionMode(), "dark");

    std::filesystem::remove_all(root, error);
}

TEST(ThemePreferencesContract, MalformedJsonFallsBackToBuiltInDefaults) {
    const auto root = std::filesystem::temp_directory_path() / "lfs_theme_preferences_malformed";
    std::error_code error;
    std::filesystem::remove_all(root, error);
    const ScopedLfsHome home(root);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths.has_value()) << lfs::format_for_developer(paths.error());
    ASSERT_TRUE(paths->ensureDirectories().has_value());
    std::ofstream(paths->preferencesFile()) << "{broken";

    EXPECT_EQ(lfs::vis::loadThemePreferenceName(), "dark");
    EXPECT_FLOAT_EQ(lfs::vis::loadUiScalePreference(), 0.0f);
    EXPECT_TRUE(lfs::vis::loadLanguagePreference().empty());
    EXPECT_FALSE(std::filesystem::exists(paths->preferencesFile()));
    std::size_t backup_count = 0;
    std::filesystem::path backup_path;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(paths->backupDir())) {
        if (entry.is_regular_file() && entry.path().filename() == "preferences.json") {
            backup_path = entry.path();
            ++backup_count;
        }
    }
    EXPECT_EQ(backup_count, 1u);
    ASSERT_FALSE(backup_path.empty());
    std::ifstream backup(backup_path, std::ios::binary);
    const std::string backup_contents((std::istreambuf_iterator<char>(backup)), {});
    EXPECT_EQ(backup_contents, "{broken");
    lfs::vis::saveLanguagePreference("de");
    EXPECT_EQ(lfs::vis::loadLanguagePreference(), "de");
    EXPECT_TRUE(std::filesystem::is_regular_file(paths->preferencesFile()));
    std::filesystem::remove_all(root, error);
}

TEST(ThemePreferencesContract, SceneReconstructionRoundTripsAndValidatesPerBackendPreset) {
    const auto root = std::filesystem::temp_directory_path() / "lfs_scene_reconstruction_preferences";
    std::error_code error;
    std::filesystem::remove_all(root, error);
    const ScopedLfsHome home(root);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths.has_value()) << lfs::format_for_developer(paths.error());
    ASSERT_TRUE(paths->ensureDirectories().has_value());

    EXPECT_EQ(lfs::vis::loadSceneUpscalerPreference(), "native");
    EXPECT_EQ(lfs::vis::loadSceneUpscalerPresetPreference("native"), "native");

    lfs::vis::saveSceneUpscalerPreference("spatial", "performance");
    EXPECT_EQ(lfs::vis::loadSceneUpscalerPreference(), "spatial");
    EXPECT_EQ(lfs::vis::loadSceneUpscalerPresetPreference("spatial"), "performance");

    lfs::vis::saveSceneUpscalerPreference("temporal", "balanced");
    EXPECT_EQ(lfs::vis::loadSceneUpscalerPreference(), "temporal");
    EXPECT_EQ(lfs::vis::loadSceneUpscalerPresetPreference("temporal"), "balanced");
    EXPECT_EQ(lfs::vis::loadSceneUpscalerPresetPreference("spatial"), "performance");

    lfs::vis::saveSceneUpscalerPreference("unknown", "unknown");
    EXPECT_EQ(lfs::vis::loadSceneUpscalerPreference(), "native");
    EXPECT_EQ(lfs::vis::loadSceneUpscalerPresetPreference("native"), "native");

    lfs::vis::saveSceneUpscalerPreference("spatial", "performance");
    lfs::vis::clearSceneUpscalerPreference();
    EXPECT_EQ(lfs::vis::loadSceneUpscalerPreference(), "native");
    EXPECT_EQ(lfs::vis::loadSceneUpscalerPresetPreference("native"), "native");
    EXPECT_EQ(lfs::vis::loadSceneUpscalerPresetPreference("spatial"), "quality");

    std::filesystem::remove_all(root, error);
}

TEST(ThemePreferencesContract, McpPreferencesRoundTripAndValidateInput) {
    const auto root = std::filesystem::temp_directory_path() / "lfs_mcp_preferences";
    std::error_code error;
    std::filesystem::remove_all(root, error);
    const ScopedLfsHome home(root);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths.has_value()) << lfs::format_for_developer(paths.error());
    ASSERT_TRUE(paths->ensureDirectories().has_value());

    const auto defaults = lfs::vis::loadMcpPreferences();
    EXPECT_TRUE(defaults.enabled);
    EXPECT_FALSE(defaults.expose_network);
    EXPECT_EQ(defaults.port, 45677);
    EXPECT_FALSE(defaults.request_logging);

    lfs::vis::saveMcpPreferences({
        .enabled = false,
        .expose_network = true,
        .port = 50123,
        .request_logging = true,
    });
    const auto saved = lfs::vis::loadMcpPreferences();
    EXPECT_FALSE(saved.enabled);
    EXPECT_TRUE(saved.expose_network);
    EXPECT_EQ(saved.port, 50123);
    EXPECT_TRUE(saved.request_logging);

    const auto invalid_root = std::filesystem::temp_directory_path() /
                              "lfs_mcp_preferences_invalid";
    std::filesystem::remove_all(invalid_root, error);
    {
        const ScopedLfsHome invalid_home(invalid_root);
        const auto invalid_paths = lfs::core::UserPaths::resolve();
        ASSERT_TRUE(invalid_paths.has_value())
            << lfs::format_for_developer(invalid_paths.error());
        ASSERT_TRUE(invalid_paths->ensureDirectories().has_value());
        std::ofstream(invalid_paths->preferencesFile())
            << R"({"mcp":{"enabled":"yes","expose_network":7,"port":70000,"request_logging":[]}})";

        const auto invalid = lfs::vis::loadMcpPreferences();
        EXPECT_TRUE(invalid.enabled);
        EXPECT_FALSE(invalid.expose_network);
        EXPECT_EQ(invalid.port, 45677);
        EXPECT_FALSE(invalid.request_logging);
    }
    std::filesystem::remove_all(invalid_root, error);
    std::filesystem::remove_all(root, error);
}

TEST(ThemePreferencesContract, WorkingDirectoryRoundTripAndRejectsUnwritable) {
    const auto root = std::filesystem::temp_directory_path() / "lfs_working_directory_preferences";
    std::error_code error;
    std::filesystem::remove_all(root, error);
    const ScopedLfsHome home(root);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths.has_value()) << lfs::format_for_developer(paths.error());
    ASSERT_TRUE(paths->ensureDirectories().has_value());

    EXPECT_TRUE(lfs::vis::workingDirectoryPreferenceRaw().empty());
    EXPECT_EQ(
        lfs::vis::loadWorkingDirectoryPreference().lexically_normal(),
        paths->rootDir().lexically_normal());
    EXPECT_EQ(
        lfs::vis::defaultWorkingDirectory().lexically_normal(),
        paths->rootDir().lexically_normal());
    EXPECT_EQ(
        lfs::vis::tempProjectDirectoryPreference().lexically_normal(),
        (paths->rootDir() / "tmp").lexically_normal());

    const auto custom = root / "custom-working";
    auto set = lfs::vis::setWorkingDirectoryPreference(custom);
    ASSERT_TRUE(set) << lfs::format_for_developer(set.error());
    EXPECT_EQ(
        lfs::vis::workingDirectoryPreferenceRaw().lexically_normal(),
        custom.lexically_normal());
    EXPECT_EQ(
        lfs::vis::loadWorkingDirectoryPreference().lexically_normal(),
        custom.lexically_normal());
    EXPECT_TRUE(std::filesystem::is_directory(custom));

    const auto as_file = root / "not-a-directory";
    {
        std::ofstream(as_file) << "file";
    }
    const auto before = lfs::vis::workingDirectoryPreferenceRaw();
    auto rejected = lfs::vis::setWorkingDirectoryPreference(as_file);
    ASSERT_FALSE(rejected);
    EXPECT_EQ(
        lfs::vis::workingDirectoryPreferenceRaw().lexically_normal(),
        before.lexically_normal());

    lfs::vis::clearWorkingDirectoryPreference();
    EXPECT_TRUE(lfs::vis::workingDirectoryPreferenceRaw().empty());
    std::filesystem::remove_all(root, error);
}

TEST(ThemePreferencesContract, AssetManagerDirectoryRoundTripAndDefaultsUnderLfsHome) {
    const auto root =
        std::filesystem::temp_directory_path() / "lfs_asset_manager_directory_preferences";
    std::error_code error;
    std::filesystem::remove_all(root, error);
    const ScopedLfsHome home(root);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths.has_value()) << lfs::format_for_developer(paths.error());
    ASSERT_TRUE(paths->ensureDirectories().has_value());

    EXPECT_TRUE(lfs::vis::assetManagerDirectoryPreferenceRaw().empty());
    EXPECT_EQ(
        lfs::vis::defaultAssetManagerDirectory().lexically_normal(),
        (paths->rootDir() / "assets").lexically_normal());
    EXPECT_EQ(
        lfs::vis::loadAssetManagerDirectoryPreference().lexically_normal(),
        (paths->rootDir() / "assets").lexically_normal());

    const auto custom = root / "custom-assets";
    auto set = lfs::vis::setAssetManagerDirectoryPreference(custom);
    ASSERT_TRUE(set) << lfs::format_for_developer(set.error());
    EXPECT_TRUE(std::filesystem::is_directory(custom));
    EXPECT_EQ(
        lfs::vis::assetManagerDirectoryPreferenceRaw().lexically_normal(),
        custom.lexically_normal());
    EXPECT_EQ(
        lfs::vis::loadAssetManagerDirectoryPreference().lexically_normal(),
        custom.lexically_normal());

    const auto as_file = root / "not-an-asset-directory";
    std::ofstream(as_file) << "file";
    const auto before = lfs::vis::assetManagerDirectoryPreferenceRaw();
    const auto rejected = lfs::vis::setAssetManagerDirectoryPreference(as_file);
    ASSERT_FALSE(rejected);
    EXPECT_EQ(
        lfs::vis::assetManagerDirectoryPreferenceRaw().lexically_normal(),
        before.lexically_normal());

    lfs::vis::clearAssetManagerDirectoryPreference();
    EXPECT_TRUE(lfs::vis::assetManagerDirectoryPreferenceRaw().empty());
    std::filesystem::remove_all(root, error);
}

TEST(ThemePreferencesContract, WorkingDirectoryRejectsRelativeAndExpandsHome) {
    const auto root =
        std::filesystem::temp_directory_path() / "lfs_working_directory_absolute";
    std::error_code error;
    std::filesystem::remove_all(root, error);
    const ScopedLfsHome lfs_home(root);
    const auto fake_home = root / "fake-home";
    const ScopedEnvVar home("HOME", fake_home.string());
#ifdef _WIN32
    const ScopedEnvVar userprofile("USERPROFILE", fake_home.string());
#endif
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths.has_value()) << lfs::format_for_developer(paths.error());
    ASSERT_TRUE(paths->ensureDirectories().has_value());

    const auto relative = std::filesystem::path("lfs_wd_rel_must_not_exist");
    const auto cwd_target = std::filesystem::current_path() / relative;
    std::filesystem::remove_all(cwd_target, error);
    const auto rejected = lfs::vis::setWorkingDirectoryPreference(relative);
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code(), lfs::ErrorCode::InvalidArgument);
    EXPECT_EQ(
        rejected.error().user_message(),
        "The working folder path must be absolute.");
    EXPECT_TRUE(lfs::vis::workingDirectoryPreferenceRaw().empty());
    EXPECT_FALSE(std::filesystem::exists(cwd_target));
    EXPECT_FALSE(std::filesystem::exists(fake_home / relative));

    const auto tilde = std::filesystem::path("~") / "custom-working";
    const auto expanded = (fake_home / "custom-working").lexically_normal();
    const auto set = lfs::vis::setWorkingDirectoryPreference(tilde);
    ASSERT_TRUE(set) << lfs::format_for_developer(set.error());
    EXPECT_TRUE(std::filesystem::is_directory(expanded));
    EXPECT_EQ(
        lfs::vis::workingDirectoryPreferenceRaw().lexically_normal(),
        expanded);
    EXPECT_EQ(
        lfs::vis::loadWorkingDirectoryPreference().lexically_normal(),
        expanded);

    lfs::vis::clearWorkingDirectoryPreference();
    std::filesystem::remove_all(root, error);
}

TEST(ThemePreferencesContract, AssetManagerDirectoryRejectsRelativeAndExpandsHome) {
    const auto root =
        std::filesystem::temp_directory_path() / "lfs_asset_manager_directory_absolute";
    std::error_code error;
    std::filesystem::remove_all(root, error);
    const ScopedLfsHome lfs_home(root);
    const auto fake_home = root / "fake-home";
    const ScopedEnvVar home("HOME", fake_home.string());
#ifdef _WIN32
    const ScopedEnvVar userprofile("USERPROFILE", fake_home.string());
#endif
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths.has_value()) << lfs::format_for_developer(paths.error());
    ASSERT_TRUE(paths->ensureDirectories().has_value());

    const auto relative = std::filesystem::path("lfs_am_rel_must_not_exist");
    const auto cwd_target = std::filesystem::current_path() / relative;
    std::filesystem::remove_all(cwd_target, error);
    const auto rejected = lfs::vis::setAssetManagerDirectoryPreference(relative);
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code(), lfs::ErrorCode::InvalidArgument);
    EXPECT_EQ(
        rejected.error().user_message(),
        "The Asset Manager folder path must be absolute.");
    EXPECT_TRUE(lfs::vis::assetManagerDirectoryPreferenceRaw().empty());
    EXPECT_FALSE(std::filesystem::exists(cwd_target));
    EXPECT_FALSE(std::filesystem::exists(fake_home / relative));

    const auto tilde = std::filesystem::path("~") / "custom-assets";
    const auto expanded = (fake_home / "custom-assets").lexically_normal();
    const auto set = lfs::vis::setAssetManagerDirectoryPreference(tilde);
    ASSERT_TRUE(set) << lfs::format_for_developer(set.error());
    EXPECT_TRUE(std::filesystem::is_directory(expanded));
    EXPECT_EQ(
        lfs::vis::assetManagerDirectoryPreferenceRaw().lexically_normal(),
        expanded);
    EXPECT_EQ(
        lfs::vis::loadAssetManagerDirectoryPreference().lexically_normal(),
        expanded);

    lfs::vis::clearAssetManagerDirectoryPreference();
    std::filesystem::remove_all(root, error);
}

TEST(ThemePreferencesContract, SafeModeNeitherReadsNorWritesPreferences) {
    const auto root = std::filesystem::temp_directory_path() / "lfs_theme_preferences_safe_mode";
    std::error_code error;
    std::filesystem::remove_all(root, error);
    const ScopedLfsHome home(root);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths.has_value()) << lfs::format_for_developer(paths.error());
    ASSERT_TRUE(paths->ensureDirectories().has_value());
    const std::string original = R"({"theme":"light","ui_scale":1.5,"language":"it","mcp":{"enabled":false,"expose_network":true,"port":50000,"request_logging":true}})";
    std::ofstream(paths->preferencesFile()) << original;

    {
        const ScopedSafeMode safe_mode;
        EXPECT_EQ(lfs::vis::loadThemePreferenceName(), "dark");
        EXPECT_FLOAT_EQ(lfs::vis::loadUiScalePreference(), 0.0f);
        EXPECT_EQ(lfs::vis::loadSceneUpscalerPreference(), "native");
        EXPECT_EQ(lfs::vis::loadSceneUpscalerPresetPreference("native"), "native");
        EXPECT_TRUE(lfs::vis::loadLanguagePreference().empty());
        const auto mcp = lfs::vis::loadMcpPreferences();
        EXPECT_TRUE(mcp.enabled);
        EXPECT_FALSE(mcp.expose_network);
        EXPECT_EQ(mcp.port, 45677);
        EXPECT_FALSE(mcp.request_logging);
        lfs::vis::saveThemePreferenceName("gruvbox");
        lfs::vis::saveUiScalePreference(2.0f);
        lfs::vis::saveSceneUpscalerPreference("spatial", "performance");
        lfs::vis::saveLanguagePreference("fr");
        lfs::vis::saveMcpPreferences({
            .enabled = true,
            .expose_network = false,
            .port = 45677,
            .request_logging = false,
        });
    }

    std::ifstream file(paths->preferencesFile());
    const std::string persisted((std::istreambuf_iterator<char>(file)), {});
    EXPECT_EQ(persisted, original);
    file.close();
    std::filesystem::remove_all(root, error);
}
