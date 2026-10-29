/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/path_utils.hpp"
#include "core/user_paths.hpp"
#include "gui/layout_state.hpp"
#include "input/gamepad_navigation.hpp"
#include "input/gamepad_profiles.hpp"
#include "project/session_state.hpp"
#include "rendering/rendering_types.hpp"
#include "visualizer/preferences.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {
    using json = nlohmann::json;

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

    std::filesystem::path makeHome(const char* name) {
        const auto home = std::filesystem::temp_directory_path() / name;
        std::error_code error;
        std::filesystem::remove_all(home, error);
        std::filesystem::create_directories(home);
        return home;
    }

    void writePreferences(const lfs::core::UserPaths& paths, const json& values) {
        std::ofstream output(paths.preferencesFile());
        ASSERT_TRUE(output);
        output << values.dump(2) << '\n';
    }

    json readPreferences(const lfs::core::UserPaths& paths) {
        std::ifstream input(paths.preferencesFile());
        return json::parse(input);
    }

    bool hasBackup(const std::filesystem::path& root, const std::filesystem::path& original,
                   const std::string& contents) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(root))
            if (entry.is_regular_file() && entry.path().filename() == original.filename()) {
                std::ifstream input(entry.path());
                if (std::string(std::istreambuf_iterator<char>(input), {}) == contents)
                    return true;
            }
        return false;
    }
} // namespace

TEST(PreferencesMigration, ClearingAfterAnotherInstanceRemovesTheParentDoesNotWriteNulls) {
    const auto home = makeHome("lfs_sidebar_missing_parent");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    auto& preferences = lfs::vis::UserPreferences::instance();
    preferences.setSidebarDefaults(510, 0.35f, "lfs.rendering");
    writePreferences(*paths, {{"schema_version", 1}});
    preferences.clearSidebarDefaults();
    EXPECT_EQ(readPreferences(*paths)["sidebar_defaults"], json::object());
    preferences.setSidebarDefaults(610, 0.45f, "lfs.training");
    EXPECT_EQ(readPreferences(*paths)["sidebar_defaults"]["right_panel_width"], 610);
    for (const auto& entry : std::filesystem::recursive_directory_iterator(paths->backupDir()))
        EXPECT_NE(entry.path().filename(), paths->preferencesFile().filename());
}

TEST(PreferencesMigration, SidebarSaveClearSaveDoesNotBackupHealthyPreferences) {
    const auto home = makeHome("lfs_sidebar_clear_no_backup");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    writePreferences(*paths, {{"schema_version", 1}, {"sidebar_defaults", {{"future_child", "keep"}}}});
    auto& preferences = lfs::vis::UserPreferences::instance();
    preferences.setSidebarDefaults(510, 0.35f, "lfs.rendering");
    const auto before = readPreferences(*paths).dump(2) + "\n";
    preferences.clearSidebarDefaults();
    EXPECT_EQ(preferences.sidebarDefaults(), "{}");
    EXPECT_EQ(readPreferences(*paths)["sidebar_defaults"], json({{"future_child", "keep"}}));
    preferences.setSidebarDefaults(610, 0.45f, "lfs.training");
    EXPECT_EQ(readPreferences(*paths)["sidebar_defaults"]["future_child"], "keep");
    EXPECT_FALSE(hasBackup(paths->backupDir(), paths->preferencesFile(), before));
    for (const auto& entry : std::filesystem::recursive_directory_iterator(paths->backupDir()))
        EXPECT_NE(entry.path().filename(), paths->preferencesFile().filename());
}

TEST(PreferencesMigration, AssetDirectoryWinsOverWorkingDirectoryAndErasesLegacyKeys) {
    const auto home = makeHome("lfs_preferences_migration_asset_wins");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    const auto asset = home / "assets";
    const auto working = home / "working";
    writePreferences(*paths, {
                                 {"asset_manager_directory", asset.string()},
                                 {"working_directory", working.string()},
                             });

    EXPECT_EQ(lfs::vis::loadProjectLocationPreference(), asset);
    const auto migrated = readPreferences(*paths);
    EXPECT_EQ(migrated.at("project_location"), asset.string());
    EXPECT_EQ(migrated.at("legacy_working_directory"), working.string());
    EXPECT_FALSE(migrated.contains("asset_manager_directory"));
    EXPECT_FALSE(migrated.contains("working_directory"));
}

TEST(PreferencesMigration, FutureSchemaAndUnknownFieldsSurviveSettingChange) {
    const auto home = makeHome("lfs_preferences_future_schema");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    writePreferences(*paths, {{"schema_version", 999},
                              {"asset_manager_directory", (home / "assets").string()},
                              {"future_feature", {{"enabled", true}}}});
    auto& preferences = lfs::vis::UserPreferences::instance();
    EXPECT_EQ(lfs::vis::loadProjectLocationPreference(), home / "assets");
    preferences.setThemeName("dark");
    const auto saved = readPreferences(*paths);
    EXPECT_EQ(saved.at("schema_version"), 999);
    EXPECT_EQ(saved.at("asset_manager_directory"), (home / "assets").string());
    EXPECT_EQ(saved.at("future_feature"), json({{"enabled", true}}));
}

TEST(PreferencesMigration, WrongTypedThemeFallsBackWithoutCrashing) {
    const auto home = makeHome("lfs_preferences_theme_number");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    writePreferences(*paths, {{"schema_version", 1}, {"theme", 42}});
    EXPECT_EQ(lfs::vis::UserPreferences::instance().themeName(), "");
}

TEST(PreferencesMigration, LegacyLayoutLoadsGeometryAndSequencerPerField) {
    const auto home = makeHome("lfs_preferences_legacy_layout");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    std::ofstream(paths->configDir() / "layout.json") << R"({
        "windows": {"python_console": true, "system_console": "bad"},
        "right_panel_width": 540, "show_sequencer": true})";
    lfs::vis::gui::LayoutState state;
    state.load();
    EXPECT_TRUE(state.window_visibility.at("python_console"));
    EXPECT_FALSE(state.window_visibility.contains("system_console"));
}

TEST(PreferencesMigration, UiPreferencesKeepFutureFieldsAndBackUpDamagedFile) {
    const auto home = makeHome("lfs_preferences_ui_future");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    const auto path = paths->uiPreferencesFile();
    const auto old_layout = paths->layoutFile();
    const std::string damaged_layout = R"({"right_panel_width":540,"show_sequencer":)";
    std::ofstream(old_layout) << damaged_layout;
    std::ofstream(path) << R"({"schema_version":999,"future_panel":{"user_choice":"keep_this"},"perf_hud":{"visible":true,"future_setting":7},"vram_hud":{"collapsed":["current_path",{"future_path":"keep_this"},42]}})";
    lfs::vis::gui::LayoutState state;
    state.load();
    EXPECT_TRUE(state.perf_hud_visible);
    EXPECT_TRUE(hasBackup(paths->backupDir(), old_layout, damaged_layout));
    EXPECT_TRUE(lfs::vis::gui::LayoutState::takeLegacyLayoutBackupNotice());
    bool backed_up = false;
    const auto save_result = state.saveUserPreferencesChecked(&backed_up);
    ASSERT_TRUE(save_result) << lfs::format_for_developer(save_result.error());
    EXPECT_FALSE(backed_up);
    std::ifstream saved_file(path);
    const auto saved = json::parse(saved_file);
    EXPECT_EQ(saved.at("schema_version"), 999);
    EXPECT_EQ(saved.at("future_panel").at("user_choice"), "keep_this");
    EXPECT_EQ(saved.at("perf_hud").at("future_setting"), 7);
    EXPECT_EQ(saved.at("vram_hud").at("collapsed"),
              json::array({"current_path", {{"future_path", "keep_this"}}, 42}));
    saved_file.close();
    std::ofstream(path, std::ios::trunc) << "{damaged";
    ASSERT_TRUE(state.saveUserPreferencesChecked(&backed_up));
    EXPECT_TRUE(backed_up);
    EXPECT_TRUE(hasBackup(paths->backupDir(), path, "{damaged"));
    const std::string wrong_root = R"(["bad-root",1])";
    std::ofstream(path, std::ios::trunc) << wrong_root;
    ASSERT_TRUE(state.saveUserPreferencesChecked(&backed_up));
    EXPECT_TRUE(backed_up);
    EXPECT_TRUE(hasBackup(paths->backupDir(), path, wrong_root));
    EXPECT_TRUE(json::parse(std::ifstream(path)).at("perf_hud").at("visible"));
}

TEST(PreferencesMigration, ExistingProjectLocationPreservesLegacyWorkingDirectory) {
    const auto home = makeHome("lfs_preferences_migration_existing_project");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    const auto project = home / "projects";
    const auto working = home / "working";
    writePreferences(*paths, {
                                 {"project_location", project.string()},
                                 {"working_directory", working.string()},
                             });

    EXPECT_EQ(lfs::vis::loadProjectLocationPreference(), project);
    const auto migrated = readPreferences(*paths);
    EXPECT_EQ(migrated.at("legacy_working_directory"), working.string());
    EXPECT_FALSE(migrated.contains("working_directory"));
}

TEST(PreferencesMigration, WorkingDirectoryIsUsedWhenAssetDirectoryIsEmpty) {
    const auto home = makeHome("lfs_preferences_migration_working");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    const auto working = home / "working";
    writePreferences(*paths, {
                                 {"asset_manager_directory", ""},
                                 {"working_directory", working.string()},
                             });
    EXPECT_EQ(lfs::vis::loadProjectLocationPreference(), working);
}

TEST(PreferencesMigration, WorkingDirectoryEqualToRootFallsBackToDefault) {
    const auto home = makeHome("lfs_preferences_migration_root");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    writePreferences(*paths, {{"working_directory", paths->rootDir().string()}});

    EXPECT_EQ(lfs::vis::loadProjectLocationPreference(), paths->rootDir() / "projects");
}

TEST(PreferencesMigration, EmptyWorkingDirectoryDoesNotWriteLegacyTombstone) {
    const auto home = makeHome("lfs_preferences_migration_empty_working");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    writePreferences(*paths, {{"working_directory", ""}});

    EXPECT_EQ(lfs::vis::loadProjectLocationPreference(), paths->rootDir() / "projects");
    EXPECT_FALSE(readPreferences(*paths).contains("legacy_working_directory"));
}

TEST(PreferencesMigration, ExistingLegacyWorkingDirectoryIsNotOverwritten) {
    const auto home = makeHome("lfs_preferences_migration_existing_tombstone");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    const auto preserved = home / "preserved-working";
    const auto current = home / "current-working";
    writePreferences(*paths, {
                                 {"legacy_working_directory", preserved.string()},
                                 {"working_directory", current.string()},
                             });

    EXPECT_EQ(lfs::vis::loadProjectLocationPreference(), current);
    EXPECT_EQ(readPreferences(*paths).at("legacy_working_directory"), preserved.string());
}

TEST(PreferencesMigration, ExistingAssetsCatalogIsPinnedWhenLegacyLocationsUnset) {
    const auto home = makeHome("lfs_preferences_migration_assets_pin");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    const auto assets = paths->rootDir() / "assets";
    std::filesystem::create_directories(assets / "nested");
    std::ofstream(assets / "nested" / "catalog.licht") << "catalog";
    writePreferences(*paths, json::object());

    EXPECT_EQ(lfs::vis::loadProjectLocationPreference(), assets);
}

TEST(PreferencesMigration, FreshPreferencesUseProjectsDefaultAndSetClearRoundTrip) {
    const auto home = makeHome("lfs_preferences_migration_fresh");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());

    EXPECT_EQ(lfs::vis::loadProjectLocationPreference(), paths->rootDir() / "projects");
    const auto custom = home / "custom";
    ASSERT_TRUE(lfs::vis::setProjectLocationPreference(custom));
    EXPECT_EQ(lfs::vis::projectLocationPreferenceRaw(), custom);
    lfs::vis::clearProjectLocationPreference();
    EXPECT_TRUE(lfs::vis::projectLocationPreferenceRaw().empty());
    EXPECT_EQ(lfs::vis::loadProjectLocationPreference(), paths->rootDir() / "projects");
}

TEST(PreferencesMigration, InvalidProjectLocationIsRejectedWithoutPersisting) {
    const auto home = makeHome("lfs_preferences_migration_invalid");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    const auto file = home / "not-a-directory";
    std::ofstream(file) << "file";

    ASSERT_TRUE(lfs::vis::setProjectLocationPreference(home / "valid"));
    const auto before = readPreferences(*paths);
    EXPECT_FALSE(lfs::vis::setProjectLocationPreference(file));
    EXPECT_EQ(readPreferences(*paths), before);
}

TEST(PreferencesMigration, NativeFloatingWindowsPersistAndGeometryResetPreservesOtherPreferences) {
    const auto home = makeHome("lfs_preferences_native_windows");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    auto& preferences = lfs::vis::UserPreferences::instance();
    EXPECT_FALSE(preferences.nativeFloatingWindows());
    preferences.setNativeFloatingWindows(true);
    EXPECT_TRUE(preferences.nativeFloatingWindows());
    preferences.setNativeFloatingWindows(false);
    preferences.setThemeName("dark");
    preferences.setFloatingWindowGeometry(R"({"preferences":[-900,50,800,600]})");
    EXPECT_FALSE(preferences.nativeFloatingWindows());
    EXPECT_FALSE(readPreferences(*paths).at("native_floating_windows").get<bool>());
    EXPECT_EQ(json::parse(preferences.floatingWindowGeometry()).at("preferences")[0], -900);
    preferences.setFloatingWindowGeometry("invalid");
    EXPECT_TRUE(json::parse(preferences.floatingWindowGeometry()).contains("preferences"));
    preferences.setFloatingWindowGeometry(R"({"video_extractor":[100,50,800,600]})");
    EXPECT_FALSE(readPreferences(*paths).at("floating_window_geometry").contains("preferences"));
    preferences.setFloatingWindowGeometry("{}");
    EXPECT_EQ(preferences.floatingWindowGeometry(), "{}");
    EXPECT_TRUE(readPreferences(*paths).at("floating_window_geometry").empty());
    EXPECT_EQ(preferences.themeName(), "dark");
    EXPECT_FALSE(preferences.nativeFloatingWindows());
}

TEST(PreferencesMigration, CameraSpeedPreferencesPersistAndClamp) {
    const auto home = makeHome("lfs_preferences_camera_speed");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());

    auto& preferences = lfs::vis::UserPreferences::instance();
    EXPECT_FLOAT_EQ(preferences.zoomSpeed(), 11.0f);
    EXPECT_FLOAT_EQ(preferences.navigationSpeed(), 10.0f);

    preferences.setZoomSpeed(42.0f);
    preferences.setNavigationSpeed(73.0f);
    EXPECT_FLOAT_EQ(preferences.zoomSpeed(), 42.0f);
    EXPECT_FLOAT_EQ(preferences.navigationSpeed(), 73.0f);
    EXPECT_FLOAT_EQ(readPreferences(*paths).at("zoom_speed"), 42.0f);
    EXPECT_FLOAT_EQ(readPreferences(*paths).at("navigation_speed"), 73.0f);

    preferences.setZoomSpeed(0.0f);
    preferences.setNavigationSpeed(101.0f);
    EXPECT_FLOAT_EQ(preferences.zoomSpeed(), 1.0f);
    EXPECT_FLOAT_EQ(preferences.navigationSpeed(), 100.0f);
}

TEST(PreferencesMigration, TensorBackendOptionsPersistWithoutChangingTheRunningBackend) {
    const auto home = makeHome("lfs_preferences_tensor_backend");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    auto& preferences = lfs::vis::UserPreferences::instance();
    const auto active_backend = lfs::core::default_gpu_backend();
    const lfs::vis::TensorPreferenceState selected{
        .backend = lfs::core::GpuBackend::Vulkan,
        .options = {.vulkan_device = "0", .rendering_vulkan_device = "different-renderer", .cuda_device = "independent-cuda-gpu", .vulkan_validation = 2, .force_fp32_half = true, .force_no_atomic_float = true},
    };
    preferences.setTensorBackend(selected);
    const auto saved = readPreferences(*paths).at("tensor_backend");
    EXPECT_EQ(saved.at("backend"), "vulkan");
    EXPECT_EQ(saved.at("vulkan_device"), "0");
    EXPECT_EQ(saved.at("rendering_vulkan_device"), "different-renderer");
    EXPECT_EQ(saved.at("cuda_device"), "independent-cuda-gpu");
    EXPECT_EQ(preferences.tensorBackend().options.cuda_device, "independent-cuda-gpu");
    EXPECT_EQ(preferences.tensorBackend().options.rendering_vulkan_device, "different-renderer");
    EXPECT_EQ(saved.at("vulkan_validation"), 2);
    EXPECT_TRUE(saved.at("force_fp32_half").get<bool>());
    EXPECT_TRUE(saved.at("force_no_atomic_float").get<bool>());
    EXPECT_EQ(preferences.tensorBackend().backend, lfs::core::GpuBackend::Vulkan);
    EXPECT_EQ(lfs::core::default_gpu_backend(), active_backend);
}

TEST(PreferencesMigration, TensorBackendDefaultsToAutomaticAndRoundTrips) {
    const auto home = makeHome("lfs_preferences_tensor_auto");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    auto& preferences = lfs::vis::UserPreferences::instance();
    EXPECT_FALSE(preferences.tensorBackend().backend.has_value());
    preferences.setTensorBackend({});
    EXPECT_EQ(readPreferences(*paths).at("tensor_backend").at("backend"), "auto");
    EXPECT_FALSE(preferences.tensorBackend().backend.has_value());
    preferences.setTensorBackend({.backend = lfs::core::GpuBackend::Vulkan});
    EXPECT_EQ(preferences.tensorBackend().backend, lfs::core::GpuBackend::Vulkan);
}

TEST(PreferencesMigration, LegacyMacVulkanPreferenceBecomesAutomatic) {
    const auto home = makeHome("lfs_preferences_tensor_legacy");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    writePreferences(*paths, {{"schema_version", 1}, {"tensor_backend", {{"backend", "vulkan"}}}});
    const auto legacy = lfs::vis::UserPreferences::instance().tensorBackend();
#ifdef __APPLE__
    // Before schema 2 the panel saved the default, so a Mac never chose Vulkan.
    EXPECT_FALSE(legacy.backend.has_value());
    EXPECT_EQ(readPreferences(*paths).at("tensor_backend").at("backend"), "auto");
#else
    EXPECT_EQ(legacy.backend, lfs::core::GpuBackend::Vulkan);
#endif

    // A choice saved under schema 2 is explicit.
    const auto current = makeHome("lfs_preferences_tensor_current");
    const ScopedLfsHome scoped_current(current);
    const auto current_paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(current_paths);
    ASSERT_TRUE(current_paths->ensureDirectories());
    writePreferences(*current_paths, {{"schema_version", 2}, {"tensor_backend", {{"backend", "vulkan"}}}});
    EXPECT_EQ(lfs::vis::UserPreferences::instance().tensorBackend().backend, lfs::core::GpuBackend::Vulkan);
}

TEST(PreferencesMigration, MalformedTensorPreferencesKeepSafeDefaults) {
    const auto home = makeHome("lfs_preferences_tensor_invalid");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    writePreferences(*paths, {{"tensor_backend", {
                                                     {"backend", "unknown"},
                                                     {"vulkan_device", 42},
                                                     {"vulkan_validation", -8},
                                                     {"force_fp32_half", "yes"},
                                                     {"force_no_atomic_float", 1},
                                                 }}});
    const auto state = lfs::vis::UserPreferences::instance().tensorBackend();
    EXPECT_EQ(state.backend, lfs::vis::TensorPreferenceState{}.backend);
    EXPECT_TRUE(state.options.vulkan_device.empty());
    EXPECT_EQ(state.options.vulkan_validation, 0);
    EXPECT_FALSE(state.options.force_fp32_half);
    EXPECT_FALSE(state.options.force_no_atomic_float);
}

TEST(PreferencesMigration, ProjectManagerPreferencesUseCanonicalStoreAndResetInIsolation) {
    const auto home = makeHome("lfs_preferences_project_manager");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    writePreferences(*paths, {{"theme", "light"}});

    auto& preferences = lfs::vis::UserPreferences::instance();
    EXPECT_EQ(preferences.projectManagerDefaultView(), "remember");
    EXPECT_TRUE(preferences.openProjectManagerAtStartup());
    EXPECT_TRUE(preferences.rememberProjectManagerState());
    EXPECT_EQ(json::parse(preferences.projectManagerState()), json::object());

    preferences.setProjectManagerDefaultView("gallery");
    preferences.setOpenProjectManagerAtStartup(false);
    preferences.setRememberProjectManagerState(false);
    preferences.setProjectManagerState(R"({"view_mode":"list","navigator_width":312.5})");

    EXPECT_EQ(preferences.projectManagerDefaultView(), "gallery");
    EXPECT_FALSE(preferences.openProjectManagerAtStartup());
    EXPECT_FALSE(preferences.rememberProjectManagerState());
    EXPECT_EQ(json::parse(preferences.projectManagerState()).at("view_mode"), "list");
    const auto persisted = readPreferences(*paths);
    ASSERT_TRUE(persisted.at("project_manager").is_object());
    EXPECT_EQ(persisted.at("project_manager").at("default_view"), "gallery");
    EXPECT_EQ(persisted.at("project_manager").at("open_at_startup"), false);
    EXPECT_EQ(persisted.at("project_manager").at("remember_state"), false);
    EXPECT_EQ(persisted.at("theme"), "light");

    preferences.resetProjectManagerPreferences();

    const auto reset = readPreferences(*paths);
    EXPECT_FALSE(reset.contains("project_manager"));
    EXPECT_EQ(reset.at("theme"), "light");
    EXPECT_EQ(preferences.projectManagerDefaultView(), "remember");
    EXPECT_TRUE(preferences.openProjectManagerAtStartup());
    EXPECT_TRUE(preferences.rememberProjectManagerState());

    preferences.setLanguage("fr");
    preferences.clearLanguage();
    EXPECT_FALSE(readPreferences(*paths).contains("language"));
}

TEST(PreferencesMigration, InvalidProjectManagerPreferencesFallBackWithoutContamination) {
    const auto home = makeHome("lfs_preferences_project_manager_invalid");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    writePreferences(*paths, {
                                 {"theme", "dark"},
                                 {"project_manager", {
                                                         {"default_view", "tiles"},
                                                         {"open_at_startup", "yes"},
                                                         {"remember_state", "yes"},
                                                         {"state", json::array({1, 2, 3})},
                                                     }},
                             });

    auto& preferences = lfs::vis::UserPreferences::instance();
    EXPECT_EQ(preferences.projectManagerDefaultView(), "remember");
    EXPECT_TRUE(preferences.openProjectManagerAtStartup());
    EXPECT_TRUE(preferences.rememberProjectManagerState());
    EXPECT_EQ(json::parse(preferences.projectManagerState()), json::object());
    EXPECT_THROW(preferences.setProjectManagerDefaultView("tiles"), std::invalid_argument);
    EXPECT_THROW(preferences.setProjectManagerState("[]"), std::invalid_argument);
    EXPECT_EQ(readPreferences(*paths).at("theme"), "dark");
}

TEST(PreferencesMigration, StartupIgnoresBackendMissingFromBuildAndPreservesSavedChoice) {
    const auto home = makeHome("lfs_preferences_tensor_missing");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    const auto missing = std::find(lfs::core::kCompiledGpuBackends.begin(),
                                   lfs::core::kCompiledGpuBackends.end(), lfs::core::GpuBackend::Vulkan) ==
                         lfs::core::kCompiledGpuBackends.end();
    if (!missing)
        GTEST_SKIP() << "Vulkan is compiled in this configuration";
    writePreferences(*paths, {{"schema_version", 2}, {"tensor_backend", {{"backend", "vulkan"}, {"vulkan_device", "0"}, {"vulkan_validation", 2}}}});
    auto& preferences = lfs::vis::UserPreferences::instance();
    ASSERT_EQ(preferences.tensorBackend().backend, lfs::core::GpuBackend::Vulkan);
    const auto sanitized = preferences.sanitizeTensorBackend();
    EXPECT_FALSE(sanitized.backend);
    EXPECT_EQ(sanitized.options.vulkan_device, "0");
    EXPECT_EQ(sanitized.options.vulkan_validation, 2);
    EXPECT_EQ(preferences.tensorBackend().backend, lfs::core::GpuBackend::Vulkan);
    EXPECT_EQ(readPreferences(*paths).at("tensor_backend").at("backend"), "vulkan");
    EXPECT_EQ(readPreferences(*paths).at("tensor_backend").at("vulkan_device"), "0");
    EXPECT_EQ(readPreferences(*paths).at("tensor_backend").at("vulkan_validation"), 2);
}

TEST(PreferencesMigration, MarketplacePreferencesPersistAndResetIndependently) {
    const auto home = makeHome("lfs_preferences_marketplace");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    writePreferences(*paths, {{"schema_version", 999}, {"theme", "light"}, {"future_field", 42}});
    auto& preferences = lfs::vis::UserPreferences::instance();
    EXPECT_EQ(preferences.marketplaceView(), "list");
    EXPECT_FALSE(preferences.marketplaceGitCheckout());
    preferences.setMarketplaceView("card");
    preferences.setMarketplaceGitCheckout(true);
    EXPECT_EQ(preferences.marketplaceView(), "card");
    EXPECT_TRUE(preferences.marketplaceGitCheckout());
    EXPECT_EQ(readPreferences(*paths).at("marketplace_view"), "card");
    EXPECT_EQ(readPreferences(*paths).at("marketplace_git_checkout"), true);
    EXPECT_THROW(preferences.setMarketplaceView("tiles"), std::invalid_argument);
    preferences.resetMarketplacePreferences();
    EXPECT_EQ(preferences.marketplaceView(), "list");
    EXPECT_FALSE(preferences.marketplaceGitCheckout());
    const auto saved = readPreferences(*paths);
    EXPECT_FALSE(saved.contains("marketplace_view"));
    EXPECT_FALSE(saved.contains("marketplace_git_checkout"));
    EXPECT_EQ(saved.at("theme"), "light");
    EXPECT_EQ(saved.at("future_field"), 42);
    EXPECT_EQ(saved.at("schema_version"), 999);
}

TEST(PreferencesMigration, InvalidMarketplacePreferencesUseDefaults) {
    const auto home = makeHome("lfs_preferences_marketplace_invalid");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    writePreferences(*paths, {{"marketplace_view", "tiles"}, {"marketplace_git_checkout", "yes"}});
    auto& preferences = lfs::vis::UserPreferences::instance();
    EXPECT_EQ(preferences.marketplaceView(), "list");
    EXPECT_FALSE(preferences.marketplaceGitCheckout());
}

TEST(PreferencesMigration, RenderingDefaultsPreserveFutureFieldsAndDoNotChangeProjectValues) {
    const auto home = makeHome("lfs_rendering_defaults_future");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    writePreferences(*paths, {{"schema_version", 999}, {"theme", "dark"}, {"rendering_defaults", {{"show_grid", false}, {"future_limit", 42}}}});
    auto& preferences = lfs::vis::UserPreferences::instance();
    lfs::vis::RenderSettings project;
    project.grid_opacity = 0.8f;
    project.focal_length_mm = 85.0f;
    ASSERT_TRUE(preferences.setRenderingDefaults(R"({"grid_opacity":0.25})"));
    EXPECT_FLOAT_EQ(project.grid_opacity, 0.8f);
    EXPECT_TRUE(project.show_grid);
    const auto saved = readPreferences(*paths);
    EXPECT_EQ(saved["schema_version"], 999);
    EXPECT_EQ(saved["rendering_defaults"]["future_limit"], 42);
    EXPECT_EQ(saved["rendering_defaults"]["show_grid"], false);
    EXPECT_EQ(saved["theme"], "dark");
    ASSERT_TRUE(lfs::vis::applyRenderingDefaults(project, preferences.renderingDefaults()));
    EXPECT_FLOAT_EQ(project.grid_opacity, 0.25f);
    EXPECT_FALSE(project.show_grid);
    EXPECT_FLOAT_EQ(project.focal_length_mm, 85.0f);
}

TEST(PreferencesMigration, RenderingDefaultsRejectInvalidPatchesAtomically) {
    lfs::vis::RenderSettings settings;
    const auto before = lfs::vis::renderingDefaultsFromSettings(settings);
    for (const auto* value : {R"({"show_grid":false,"axes_size":-1})",
                              R"({"lod_page_pool_splats":1.5})", R"({"lod_pool_vram_fraction":1})",
                              R"({"selection_color_preview":[0,2,0]})", R"({"show_grid":"false"})",
                              R"({"axes_size":"nan"})", R"([])"}) {
        EXPECT_FALSE(lfs::vis::applyRenderingDefaults(settings, value));
        EXPECT_EQ(lfs::vis::renderingDefaultsFromSettings(settings), before);
    }
}

TEST(PreferencesMigration, DamagedRenderingDefaultDoesNotDiscardValidFields) {
    const auto home = makeHome("lfs_rendering_defaults_damaged");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    writePreferences(*paths, {{"rendering_defaults", {{"show_grid", false}, {"axes_size", "bad"}, {"selection_color_preview", {0, 2, 0}}, {"lod_pool_vram_fraction", 0.3}}}});
    const auto values = json::parse(lfs::vis::UserPreferences::instance().renderingDefaults());
    EXPECT_EQ(values["show_grid"], false);
    EXPECT_EQ(values["axes_size"], lfs::vis::RenderSettings{}.axes_size);
    EXPECT_NEAR(values["lod_pool_vram_fraction"].get<float>(), 0.3f, 0.00001f);
    EXPECT_EQ(values["selection_color_preview"],
              json::parse(lfs::vis::renderingDefaultsFromSettings(lfs::vis::RenderSettings{}))["selection_color_preview"]);
}

TEST(PreferencesMigration, SavedProjectViewOverridesNewUserDefaults) {
    lfs::vis::RenderSettings original;
    original.grid_opacity = 0.9f;
    original.selection_color_preview = {0.2f, 0.3f, 0.4f};
    original.lod_page_pool_splats = 123456;
    const auto saved_view = lfs::vis::project::renderSettingsToProjectJson(original);
    lfs::vis::RenderSettings restored;
    ASSERT_TRUE(lfs::vis::applyRenderingDefaults(restored, R"({"grid_opacity":0.1,"lod_page_pool_splats":999})"));
    // The existing VIEW restoration remains authoritative over startup defaults.
    const auto decoded = lfs::vis::project::renderSettingsFromProjectJson(saved_view, restored);
    ASSERT_TRUE(decoded);
    EXPECT_EQ(lfs::vis::renderingDefaultsFromSettings(*decoded),
              lfs::vis::renderingDefaultsFromSettings(original));
}

TEST(PreferencesMigration, ViewportVignetteIsGlobalAndValidated) {
    const auto home = makeHome("lfs_viewport_vignette_preferences");
    const ScopedLfsHome scoped_home(home);
    auto& preferences = lfs::vis::UserPreferences::instance();
    EXPECT_EQ(json::parse(preferences.viewportVignette()), json::object());
    ASSERT_TRUE(preferences.setViewportVignette(R"({"enabled":true,"radius":0.4,"intensity":0.2,"softness":0.6})"));
    EXPECT_EQ(json::parse(preferences.viewportVignette())["enabled"], true);
    EXPECT_FALSE(preferences.setViewportVignette(R"({"enabled":false,"radius":2})"));
    EXPECT_EQ(json::parse(preferences.viewportVignette())["enabled"], true);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    EXPECT_EQ(readPreferences(*paths)["viewport_vignette"]["radius"], 0.4);
}

TEST(PreferencesMigration, ZoomTowardCursorDefaultsOffAndPersists) {
    const auto home = makeHome("lfs_preferences_zoom_toward_cursor");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    writePreferences(*paths, {{"schema_version", 1}});

    EXPECT_FALSE(lfs::vis::loadZoomTowardCursorPreference());

    lfs::vis::saveZoomTowardCursorPreference(true);
    EXPECT_TRUE(lfs::vis::loadZoomTowardCursorPreference());
    EXPECT_EQ(readPreferences(*paths).at("zoom_toward_cursor"), true);

    lfs::vis::saveZoomTowardCursorPreference(false);
    EXPECT_FALSE(lfs::vis::loadZoomTowardCursorPreference());
    EXPECT_EQ(readPreferences(*paths).at("zoom_toward_cursor"), false);
}

TEST(PreferencesMigration, NonBooleanZoomTowardCursorFallsBackToOff) {
    // Preferences are read once per path, so the value must be on disk before the first read.
    const auto home = makeHome("lfs_preferences_zoom_toward_cursor_invalid");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    writePreferences(*paths, {{"schema_version", 1}, {"zoom_toward_cursor", "yes"}});

    EXPECT_FALSE(lfs::vis::loadZoomTowardCursorPreference());
    EXPECT_EQ(readPreferences(*paths).at("zoom_toward_cursor"), "yes");
}

TEST(PreferencesMigration, CameraBrowseRepeatKeepsValuesSavedUnderOldNames) {
    const auto home = makeHome("lfs_camera_browse_repeat_rename");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    ASSERT_TRUE(paths->ensureDirectories());
    writePreferences(*paths, {{"schema_version", 1},
                              {"gt_comparison_key_repeat", false},
                              {"gt_comparison_repeat_rate", 9.0}});
    auto& preferences = lfs::vis::UserPreferences::instance();
    EXPECT_FALSE(preferences.cameraBrowseRepeat());
    EXPECT_FLOAT_EQ(preferences.cameraBrowseRepeatRate(), 9.0f);
    preferences.setCameraBrowseRepeatRate(11.0f);
    const auto saved = readPreferences(*paths);
    EXPECT_EQ(saved["camera_browse_repeat"], false);
    EXPECT_EQ(saved["camera_browse_repeat_rate"], 11.0);
    EXPECT_FALSE(saved.contains("gt_comparison_key_repeat"));
    EXPECT_FALSE(saved.contains("gt_comparison_repeat_rate"));
}

TEST(PreferencesMigration, CameraBrowseRepeatRateIsClampedAndPersistedOnRequest) {
    const auto home = makeHome("lfs_camera_browse_repeat_rate");
    const ScopedLfsHome scoped_home(home);
    auto& preferences = lfs::vis::UserPreferences::instance();
    EXPECT_TRUE(preferences.cameraBrowseRepeat());
    EXPECT_FLOAT_EQ(preferences.cameraBrowseRepeatRate(), 5.0f);
    preferences.setCameraBrowseRepeatRate(200.0f);
    EXPECT_FLOAT_EQ(preferences.cameraBrowseRepeatRate(), 20.0f);
    preferences.setCameraBrowseRepeatRate(-2.0f);
    EXPECT_FLOAT_EQ(preferences.cameraBrowseRepeatRate(), 1.0f);
    preferences.setCameraBrowseRepeatRate(std::numeric_limits<float>::quiet_NaN());
    EXPECT_FLOAT_EQ(preferences.cameraBrowseRepeatRate(), 5.0f);
    preferences.setCameraBrowseRepeatRate(7.0f);
    preferences.setCameraBrowseRepeatRate(12.0f, false);
    EXPECT_FLOAT_EQ(preferences.cameraBrowseRepeatRate(), 12.0f);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    EXPECT_EQ(readPreferences(*paths)["camera_browse_repeat_rate"], 7.0);
}

TEST(PreferencesMigration, GamepadEnablementAndAssignmentsSurviveReloadWithoutConnectionIds) {
    const auto home = makeHome("lfs_gamepad_preferences_reload");
    const auto other = makeHome("lfs_gamepad_preferences_reload_other");
    const ScopedLfsHome scoped_home(home);
    auto& preferences = lfs::vis::UserPreferences::instance();
    lfs::vis::input::GamepadNavigationSettings settings;
    settings.enabled = true;
    settings.move_speed = 2.3f;
    settings.bindings.modes[0][0] = lfs::vis::input::GamepadBinding{};
    ASSERT_TRUE(preferences.setGamepadNavigation(settings.json().dump()));
    settings.bindings.modes[0][0].reset();
    ASSERT_TRUE(preferences.setGamepadNavigation(settings.json().dump()));
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    const auto saved = readPreferences(*paths)["gamepad_navigation"];
    EXPECT_TRUE(saved["enabled"].get<bool>());
    EXPECT_FALSE(saved.contains("selected"));
    EXPECT_FALSE(saved.contains("revision"));
    EXPECT_FALSE(saved["bindings"]["modes"].contains("orbit"));
    {
        const ScopedLfsHome switch_home(other);
        EXPECT_EQ(json::parse(preferences.gamepadNavigation()), json::object());
    }
    const auto restored = lfs::vis::input::GamepadNavigationSettings::fromJson(json::parse(preferences.gamepadNavigation()));
    EXPECT_TRUE(restored.enabled);
    EXPECT_FLOAT_EQ(restored.move_speed, 2.3f);
    EXPECT_FALSE(restored.bindings.modes[0][0]);
    settings.enabled = false;
    ASSERT_TRUE(preferences.setGamepadNavigation(settings.json().dump()));
    {
        const ScopedLfsHome switch_home(other);
        (void)preferences.gamepadNavigation();
    }
    EXPECT_FALSE(lfs::vis::input::GamepadNavigationSettings::fromJson(json::parse(preferences.gamepadNavigation())).enabled);
}

TEST(PreferencesMigration, InvalidGamepadProfileDoesNotOverwriteSavedPreferences) {
    const auto home = makeHome("lfs_gamepad_invalid_preferences");
    const ScopedLfsHome scoped_home(home);
    auto& preferences = lfs::vis::UserPreferences::instance();
    lfs::vis::input::GamepadNavigationSettings settings;
    ASSERT_TRUE(preferences.setGamepadNavigation(settings.json().dump()));
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    const auto before = readPreferences(*paths);
    auto invalid = settings.json();
    invalid["enabled"] = "true";
    EXPECT_FALSE(preferences.setGamepadNavigation(invalid.dump()));
    EXPECT_EQ(readPreferences(*paths), before);
    settings.bindings.common[8] = settings.bindings.common[10];
    EXPECT_FALSE(preferences.setGamepadNavigation(settings.json().dump()));
    EXPECT_EQ(readPreferences(*paths), before);
}

TEST(PreferencesMigration, NamedGamepadProfilesMigrateAndSwitchAtomicallyWithoutChangingEnablement) {
    const auto home = makeHome("lfs_named_gamepad_profiles");
    const ScopedLfsHome scoped_home(home);
    const auto paths = lfs::core::UserPaths::resolve();
    ASSERT_TRUE(paths);
    lfs::vis::input::GamepadNavigationSettings legacy{true};
    legacy.move_speed = 2.4f;
    ASSERT_TRUE(paths->ensureDirectories());
    writePreferences(*paths, {{"gamepad_navigation", legacy.json()}, {"unrelated", 42}});
    auto& preferences = lfs::vis::UserPreferences::instance();
    auto library = json::parse(preferences.gamepadProfiles());
    EXPECT_EQ(library["profiles"][0]["name"], "Personal");
    EXPECT_FALSE(library["profiles"][0]["settings"].contains("enabled"));
    EXPECT_FLOAT_EQ(library["profiles"][0]["settings"]["move_speed"].get<float>(), 2.4f);
    auto second = library["profiles"][0];
    second["id"] = "second";
    second["name"] = "Second";
    second["settings"]["move_speed"] = 0.7;
    library["profiles"].push_back(second);
    library["active"] = "second";
    library["device_profiles"][std::string(32, 'b')] = "second";
    ASSERT_TRUE(preferences.setGamepadProfiles(library.dump(), true));
    const auto saved = readPreferences(*paths);
    EXPECT_EQ(saved["gamepad_profiles"]["active"], "second");
    EXPECT_TRUE(saved["gamepad_navigation"]["enabled"].get<bool>());
    EXPECT_EQ(saved["unrelated"], 42);
    EXPECT_FLOAT_EQ(json::parse(preferences.gamepadNavigation())["move_speed"].get<float>(), 0.7f);
    // Ordinary slider/binding writes update only the selected layout.
    auto settings = lfs::vis::input::GamepadNavigationSettings::fromJson(json::parse(preferences.gamepadNavigation()));
    settings.look_speed = 2.2f;
    ASSERT_TRUE(preferences.setGamepadNavigation(settings.json().dump()));
    const auto updated = json::parse(preferences.gamepadProfiles());
    EXPECT_EQ(updated["profiles"][0], library["profiles"][0]);
    EXPECT_EQ(updated["device_profiles"], library["device_profiles"]);
    const auto other_home = makeHome("lfs_named_gamepad_profiles_other");
    {
        const ScopedLfsHome other(other_home);
        (void)preferences.gamepadProfiles();
    }
    EXPECT_EQ(json::parse(preferences.gamepadProfiles()), updated);
}

TEST(PreferencesMigration, StaleGamepadProfileEditsCannotOverwriteConcurrentChanges) {
    const ScopedLfsHome scoped_home(makeHome("lfs_named_gamepad_stale"));
    auto& preferences = lfs::vis::UserPreferences::instance();
    const auto expected = preferences.gamepadProfiles();
    auto library = json::parse(expected);
    ASSERT_TRUE(preferences.setGamepadProfiles(library.dump(), false, expected));
    const auto old = preferences.gamepadProfiles();
    lfs::vis::input::GamepadNavigationSettings changed;
    changed.move_speed = 3.0f;
    ASSERT_TRUE(preferences.setGamepadNavigation(changed.json().dump()));
    const auto paths = lfs::core::UserPaths::resolve();
    const auto before = readPreferences(*paths);
    library["profiles"][0]["name"] = "Stale rename";
    EXPECT_FALSE(preferences.setGamepadProfiles(library.dump(), false, old));
    EXPECT_EQ(readPreferences(*paths), before);
}

TEST(PreferencesMigration, FutureGamepadLibraryIsNeverReplacedByEditingOrReset) {
    const ScopedLfsHome scoped_home(makeHome("lfs_named_gamepad_future"));
    const auto paths = lfs::core::UserPaths::resolve();
    auto library = lfs::vis::input::GamepadProfiles::migrate({}).value;
    auto future = library;
    future["schema_version"] = 2;
    ASSERT_TRUE(paths->ensureDirectories());
    writePreferences(*paths, {{"schema_version", 1}, {"project_location", ""}, {"gamepad_profiles", future}});
    auto& preferences = lfs::vis::UserPreferences::instance();
    const auto before = readPreferences(*paths);
    EXPECT_FALSE(preferences.setGamepadProfiles(library.dump(), false));
    EXPECT_FALSE(preferences.setGamepadNavigation(lfs::vis::input::GamepadNavigationSettings{}.json().dump()));
    EXPECT_EQ(readPreferences(*paths), before);
}

TEST(PreferencesMigration, SafeModeCannotModifyOrResetSavedGamepadProfiles) {
    const ScopedLfsHome scoped_home(makeHome("lfs_named_gamepad_safe_mode"));
    auto& preferences = lfs::vis::UserPreferences::instance();
    const auto library = preferences.gamepadProfiles();
    ASSERT_TRUE(preferences.setGamepadProfiles(library, true));
    const auto paths = lfs::core::UserPaths::resolve();
    const auto before = readPreferences(*paths);
    struct SafeMode {
        std::optional<std::string> previous;
        SafeMode() {
            if (const auto* value = std::getenv("LFS_SAFE_MODE"))
                previous = value;
#ifdef _WIN32
            (void)_putenv_s("LFS_SAFE_MODE", "1");
#else
            (void)setenv("LFS_SAFE_MODE", "1", 1);
#endif
        }
        ~SafeMode() {
#ifdef _WIN32
            (void)_putenv_s("LFS_SAFE_MODE", previous ? previous->c_str() : "");
#else
            if (previous)
                (void)setenv("LFS_SAFE_MODE", previous->c_str(), 1);
            else
                (void)unsetenv("LFS_SAFE_MODE");
#endif
        }
    };
    {
        const SafeMode safe_mode;
        EXPECT_FALSE(preferences.setGamepadProfiles(library, false));
        EXPECT_FALSE(preferences.setGamepadNavigation(lfs::vis::input::GamepadNavigationSettings{}.json().dump()));
        EXPECT_EQ(readPreferences(*paths), before);
    }
    EXPECT_EQ(json::parse(preferences.gamepadProfiles()), json::parse(library));
    EXPECT_TRUE(json::parse(preferences.gamepadNavigation())["enabled"].get<bool>());
}
