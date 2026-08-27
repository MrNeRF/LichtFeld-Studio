/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "core/legacy_settings_migration.hpp"
#include "core/user_paths.hpp"

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <ostream>
#include <string>
#include <thread>
#include <vector>

namespace lfs {
    std::ostream& operator<<(std::ostream& stream, const Error& error) {
        return stream << format_for_developer(error);
    }
} // namespace lfs

namespace {

    namespace fs = std::filesystem;

    class ScopedEnvironmentVariable {
    public:
        ScopedEnvironmentVariable(const char* name, const std::optional<std::string>& value)
            : name_(name) {
            if (const char* previous = std::getenv(name))
                previous_ = previous;
            set(value);
        }

        ~ScopedEnvironmentVariable() { set(previous_); }

    private:
        void set(const std::optional<std::string>& value) const {
#ifdef _WIN32
            (void)_putenv_s(name_.c_str(), value ? value->c_str() : "");
#else
            if (value)
                (void)setenv(name_.c_str(), value->c_str(), 1);
            else
                (void)unsetenv(name_.c_str());
#endif
        }

        std::string name_;
        std::optional<std::string> previous_;
    };

    class UserPathsContractTest : public ::testing::Test {
    protected:
        void SetUp() override {
            const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
            root_ = fs::temp_directory_path() / ("lfs_user_paths_" + std::to_string(nonce));
            std::error_code error;
            fs::remove_all(root_, error);
        }

        void TearDown() override {
            std::error_code error;
            fs::remove_all(root_, error);
        }

        lfs::Result<lfs::core::UserPaths> resolvePaths() const {
            const lfs::core::UserPathOptions options{
                .explicit_root = root_ / "current",
            };
            return lfs::core::UserPaths::resolve(options);
        }

        fs::path portableExecutableDir() const { return root_ / "portable-app"; }

        fs::path root_;
    };

    TEST_F(UserPathsContractTest, ExplicitRootUsesOnlyTheUnifiedStorageTree) {
        const auto resolved = resolvePaths();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        const auto& paths = *resolved;
        ASSERT_TRUE(paths.ensureDirectories().has_value());

        EXPECT_TRUE(paths.usesUnifiedRoot());
        EXPECT_TRUE(fs::is_directory(paths.configDir()));
        EXPECT_TRUE(fs::is_directory(paths.dataDir()));
        EXPECT_TRUE(fs::is_directory(paths.cacheDir()));
        EXPECT_TRUE(fs::is_directory(paths.logDir()));
        EXPECT_FALSE(fs::exists(paths.mcpLogDir()));
        EXPECT_TRUE(fs::is_directory(paths.pluginDir()));
        EXPECT_TRUE(fs::is_directory(paths.venvDir()));
        EXPECT_EQ(paths.uiPreferencesFile(), paths.configDir() / "ui_preferences.json");
        EXPECT_EQ(paths.projectLifecycleFile(), paths.configDir() / "project_lifecycle.json");
        EXPECT_EQ(paths.rootDir(), paths.configDir().parent_path());
        EXPECT_EQ(paths.recoveryDir(), paths.configDir().parent_path() / "recovery");
        EXPECT_FALSE(fs::exists(paths.recoveryDir()));
        EXPECT_FALSE(fs::exists(root_ / "LichtFeldStudio"));
    }

    TEST_F(UserPathsContractTest, ResetPreferencesWritesDefaults) {
        const auto resolved = resolvePaths();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        const auto& paths = *resolved;
        ASSERT_TRUE(paths.ensureDirectories().has_value());
        auto reset = paths.resetPreferences();
        ASSERT_TRUE(reset.has_value()) << reset.error();
        EXPECT_TRUE(fs::is_regular_file(paths.preferencesFile()));

        std::ifstream preferences(paths.preferencesFile(), std::ios::binary);
        const std::string contents((std::istreambuf_iterator<char>(preferences)), {});
        EXPECT_NE(contents.find("\"theme\": \"dark\""), std::string::npos);
        EXPECT_NE(contents.find("\"ui_scale\": \"auto\""), std::string::npos);
        EXPECT_NE(contents.find("\"language\": \"en\""), std::string::npos);
        const auto json = nlohmann::json::parse(contents);
        ASSERT_TRUE(json.at("mcp").is_object());
        EXPECT_TRUE(json.at("mcp").at("enabled").get<bool>());
        EXPECT_FALSE(json.at("mcp").at("expose_network").get<bool>());
        EXPECT_EQ(json.at("mcp").at("port").get<int>(), 45677);
        EXPECT_FALSE(json.at("mcp").at("request_logging").get<bool>());
        ASSERT_TRUE(json.contains("working_directory"));
        EXPECT_TRUE(json.at("working_directory").is_string());
        EXPECT_EQ(json.at("working_directory").get<std::string>(), "");
        ASSERT_TRUE(json.contains("asset_manager_directory"));
        EXPECT_TRUE(json.at("asset_manager_directory").is_string());
        EXPECT_EQ(json.at("asset_manager_directory").get<std::string>(), "");
    }

    TEST_F(UserPathsContractTest, ResetPreferencesBacksUpExistingFile) {
        const auto resolved = resolvePaths();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        const auto& paths = *resolved;
        ASSERT_TRUE(paths.ensureDirectories().has_value());

        {
            std::ofstream preferences(paths.preferencesFile(), std::ios::binary);
            preferences << R"({"theme":"light","ui_scale":"150"})";
        }

        const auto reset = paths.resetPreferences();
        ASSERT_TRUE(reset.has_value()) << reset.error();
        ASSERT_TRUE(reset->has_value());
        EXPECT_TRUE(fs::is_regular_file(**reset));
        EXPECT_TRUE(fs::is_regular_file(paths.preferencesFile()));
        std::ifstream backup(**reset, std::ios::binary);
        const std::string backup_contents((std::istreambuf_iterator<char>(backup)), {});
        EXPECT_EQ(backup_contents, R"({"theme":"light","ui_scale":"150"})");
    }

    TEST_F(UserPathsContractTest, McpLogsAreLazyAppendOnlyAndConfinedToTheirDirectory) {
        const auto resolved = resolvePaths();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        const auto& paths = *resolved;
        ASSERT_TRUE(paths.ensureDirectories().has_value());
        EXPECT_FALSE(fs::exists(paths.mcpLogDir()));

        ASSERT_TRUE(paths.appendMcpLogLine("20260817-120000-mcp.jsonl", "first").has_value());
        ASSERT_TRUE(fs::is_directory(paths.mcpLogDir()));
        ASSERT_TRUE(paths.appendMcpLogLine("20260817-120000-mcp.jsonl", "second").has_value());

        std::ifstream log(paths.mcpLogDir() / "20260817-120000-mcp.jsonl", std::ios::binary);
        const std::string contents((std::istreambuf_iterator<char>(log)), {});
        EXPECT_EQ(contents, "first\nsecond\n");
        EXPECT_FALSE(paths.appendMcpLogLine("../outside.jsonl", "invalid").has_value());
        EXPECT_FALSE(fs::exists(paths.mcpLogDir().parent_path() / "outside.jsonl"));
    }

    TEST_F(UserPathsContractTest, LegacySettingsMigrationIsOneShotAndNonDestructive) {
        const ScopedEnvironmentVariable lfs_home("LFS_HOME", std::nullopt);
        const ScopedEnvironmentVariable home(
#ifdef _WIN32
            "USERPROFILE",
#else
            "HOME",
#endif
            (root_ / "home").string());
#ifdef _WIN32
        const ScopedEnvironmentVariable legacy_base("APPDATA", (root_ / "legacy-base").string());
        const fs::path legacy = root_ / "legacy-base" / "LichtFeldStudio";
#else
        const ScopedEnvironmentVariable legacy_base("XDG_CONFIG_HOME", (root_ / "legacy-base").string());
        const fs::path legacy = root_ / "legacy-base" / "LichtFeldStudio";
#endif
        ASSERT_TRUE(fs::create_directories(legacy / "input_profiles"));
        std::ofstream(legacy / "theme_preference") << "gruvbox";
        std::ofstream(legacy / "language_preference") << "it";
        std::ofstream(legacy / "ui_scale") << "1.5";
        std::ofstream(legacy / "layout.json") << R"({"legacy":true})";
        std::ofstream(legacy / "input_profiles" / "Custom.json") << R"({"name":"Custom"})";

        const auto paths = lfs::core::UserPaths::resolve();
        ASSERT_TRUE(paths.has_value()) << paths.error();
        ASSERT_TRUE(lfs::core::migrateLegacySettings(*paths).has_value());
        EXPECT_TRUE(fs::is_regular_file(paths->layoutFile()));
        EXPECT_TRUE(fs::is_regular_file(paths->keymapDir() / "Custom.json"));
        EXPECT_TRUE(fs::is_regular_file(legacy / "layout.json"));
        std::ifstream preferences(paths->preferencesFile());
        const auto values = nlohmann::json::parse(preferences);
        EXPECT_EQ(values["theme"], "gruvbox");
        EXPECT_EQ(values["language"], "it");
        EXPECT_FLOAT_EQ(values["ui_scale"].get<float>(), 1.5f);

        std::ofstream(paths->layoutFile(), std::ios::trunc) << R"({"current":true})";
        ASSERT_TRUE(lfs::core::migrateLegacySettings(*paths).has_value());
        std::ifstream layout(paths->layoutFile());
        const std::string layout_contents((std::istreambuf_iterator<char>(layout)), {});
        EXPECT_EQ(layout_contents, R"({"current":true})");
    }

    TEST_F(UserPathsContractTest, PortableRootUsesExecutableDirectory) {
        const lfs::core::UserPathOptions options{
            .executable_dir = portableExecutableDir(),
            .portable = true,
        };
        const auto resolved = lfs::core::UserPaths::resolve(options);
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        EXPECT_TRUE(resolved->usesUnifiedRoot());
        EXPECT_EQ(resolved->configDir(), portableExecutableDir() / ".lichtfeld" / "config");
        EXPECT_EQ(resolved->dataDir(), portableExecutableDir() / ".lichtfeld" / "data");
    }

    TEST_F(UserPathsContractTest, PortableRootRequiresExecutableDirectory) {
        const lfs::core::UserPathOptions options{.portable = true};
        const auto resolved = lfs::core::UserPaths::resolve(options);
        ASSERT_FALSE(resolved.has_value());
        EXPECT_NE(lfs::format_for_developer(resolved.error()).find("executable directory"),
                  std::string::npos);
    }

    TEST_F(UserPathsContractTest, ExplicitRootTakesPrecedenceOverEnvironmentRoot) {
        const ScopedEnvironmentVariable lfs_home("LFS_HOME", (root_ / "environment").string());
        const auto resolved = resolvePaths();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        EXPECT_EQ(resolved->configDir(), root_ / "current" / "config");
    }

    TEST_F(UserPathsContractTest, EnvironmentRootUsesUnifiedStorageTree) {
        const ScopedEnvironmentVariable lfs_home("LFS_HOME", (root_ / "environment").string());
        const auto resolved = lfs::core::UserPaths::resolve();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        EXPECT_TRUE(resolved->usesUnifiedRoot());
        EXPECT_EQ(resolved->configDir(), root_ / "environment" / "config");
        EXPECT_EQ(resolved->pluginDir(), root_ / "environment" / "plugins");
    }

    TEST_F(UserPathsContractTest, AtomicPreferenceWriteCreatesValidJsonAndNoTemporaryFiles) {
        const auto resolved = resolvePaths();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        const auto& paths = *resolved;
        ASSERT_TRUE(paths.ensureDirectories().has_value());

        const auto write = paths.writePreferencesAtomically(R"({"theme":"light","ui_scale":"auto"})");
        ASSERT_TRUE(write.has_value()) << write.error();
        ASSERT_TRUE(fs::is_regular_file(paths.preferencesFile()));
        std::ifstream file(paths.preferencesFile());
        const auto json = nlohmann::json::parse(file);
        EXPECT_EQ(json.at("theme"), "light");
        EXPECT_EQ(json.at("ui_scale"), "auto");

        for (const auto& entry : fs::directory_iterator(paths.configDir()))
            EXPECT_EQ(entry.path().filename().string().find("preferences.json.tmp-"), std::string::npos);
    }

    TEST_F(UserPathsContractTest, AtomicPreferenceWriteReplacesPreviousContents) {
        const auto resolved = resolvePaths();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        const auto& paths = *resolved;

        ASSERT_TRUE(paths.writePreferencesAtomically(R"({"theme":"dark"})").has_value());
        ASSERT_TRUE(paths.writePreferencesAtomically(R"({"theme":"light"})").has_value());

        std::ifstream file(paths.preferencesFile());
        const auto json = nlohmann::json::parse(file);
        EXPECT_EQ(json.at("theme"), "light");
        EXPECT_EQ(json.size(), 1U);
    }

    TEST_F(UserPathsContractTest, ConcurrentAtomicWritesDoNotShareTemporaryFiles) {
        const auto resolved = resolvePaths();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        const auto& paths = *resolved;

        std::array<std::optional<std::string>, 8> errors;
        std::vector<std::thread> writers;
        for (int index = 0; index < 8; ++index) {
            writers.emplace_back([&paths, &errors, index] {
                const auto result = paths.writePreferencesAtomically(
                    nlohmann::json{{"writer", index}}.dump());
                if (!result)
                    errors[static_cast<std::size_t>(index)] =
                        lfs::format_for_developer(result.error());
            });
        }
        for (auto& writer : writers)
            writer.join();
        for (const auto& error : errors)
            EXPECT_FALSE(error.has_value()) << error.value_or("");

        std::ifstream file(paths.preferencesFile());
        const auto json = nlohmann::json::parse(file);
        EXPECT_GE(json.at("writer").get<int>(), 0);
        EXPECT_LT(json.at("writer").get<int>(), 8);
        for (const auto& entry : fs::directory_iterator(paths.configDir()))
            EXPECT_EQ(entry.path().filename().string().find("preferences.json.tmp-"), std::string::npos);
    }

    TEST_F(UserPathsContractTest, FailedAtomicReplacementPreservesDestinationAndCleansTemporaryFile) {
        const auto resolved = resolvePaths();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        const auto& paths = *resolved;
        ASSERT_TRUE(fs::create_directories(paths.preferencesFile()));

        const auto result = paths.writePreferencesAtomically(R"({"theme":"light"})");
        EXPECT_FALSE(result.has_value());
        EXPECT_TRUE(fs::is_directory(paths.preferencesFile()));
        for (const auto& entry : fs::directory_iterator(paths.configDir()))
            EXPECT_EQ(entry.path().filename().string().find("preferences.json.tmp-"), std::string::npos);
    }

    TEST_F(UserPathsContractTest, AtomicWindowWriteCreatesParentDirectory) {
        const auto resolved = resolvePaths();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        const auto& paths = *resolved;
        ASSERT_TRUE(paths.writeWindowStateAtomically(R"({"width":1280,"height":720})").has_value());
        EXPECT_TRUE(fs::is_regular_file(paths.windowStateFile()));
    }

    TEST_F(UserPathsContractTest, AtomicUiPreferenceWriteUsesResolvedConfigTree) {
        const auto resolved = resolvePaths();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        ASSERT_TRUE(resolved->writeUiPreferencesAtomically(
                                R"({"perf_hud":{"visible":false}})")
                        .has_value());
        EXPECT_TRUE(fs::is_regular_file(resolved->uiPreferencesFile()));
        std::ifstream file(resolved->uiPreferencesFile());
        EXPECT_FALSE(nlohmann::json::parse(file)["perf_hud"]["visible"].get<bool>());
    }

    TEST_F(UserPathsContractTest, ResetLayoutBacksUpExistingFile) {
        const auto resolved = resolvePaths();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        const auto& paths = *resolved;
        ASSERT_TRUE(paths.ensureDirectories().has_value());

        {
            std::ofstream layout(paths.layoutFile(), std::ios::binary);
            layout << R"({"right_panel_width":420})";
        }

        const auto reset = paths.resetLayout();
        ASSERT_TRUE(reset.has_value()) << reset.error();
        ASSERT_TRUE(reset->has_value());
        EXPECT_TRUE(fs::is_regular_file(**reset));
        EXPECT_FALSE(fs::exists(paths.layoutFile()));
        std::ifstream backup(**reset);
        EXPECT_EQ(std::string((std::istreambuf_iterator<char>(backup)), {}),
                  R"({"right_panel_width":420})");
    }

    TEST_F(UserPathsContractTest, ResetWindowStateBacksUpExistingFile) {
        const auto resolved = resolvePaths();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        const auto& paths = *resolved;
        ASSERT_TRUE(paths.ensureDirectories().has_value());

        {
            std::ofstream window(paths.windowStateFile(), std::ios::binary);
            window << R"({"x":10,"y":20,"width":1280,"height":720,"maximized":false})";
        }

        const auto reset = paths.resetWindowState();
        ASSERT_TRUE(reset.has_value()) << reset.error();
        ASSERT_TRUE(reset->has_value());
        EXPECT_TRUE(fs::is_regular_file(**reset));
        EXPECT_FALSE(fs::exists(paths.windowStateFile()));
        std::ifstream backup(**reset);
        EXPECT_EQ(std::string((std::istreambuf_iterator<char>(backup)), {}),
                  R"({"x":10,"y":20,"width":1280,"height":720,"maximized":false})");
    }

    TEST_F(UserPathsContractTest, ResetUiAndLifecycleSettingsBackUpResolvedFiles) {
        const auto resolved = resolvePaths();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        const auto& paths = *resolved;
        ASSERT_TRUE(paths.writeUiPreferencesAtomically(R"({"perf_hud":{}})").has_value());
        ASSERT_TRUE(lfs::core::writeTextFileAtomically(
                        paths.projectLifecycleFile(), R"({"version":2})")
                        .has_value());

        const auto ui_reset = paths.resetUiPreferences();
        ASSERT_TRUE(ui_reset && ui_reset->has_value());
        const auto lifecycle_reset = paths.resetProjectLifecycle();
        ASSERT_TRUE(lifecycle_reset && lifecycle_reset->has_value());
        EXPECT_FALSE(fs::exists(paths.uiPreferencesFile()));
        EXPECT_FALSE(fs::exists(paths.projectLifecycleFile()));
        EXPECT_TRUE(fs::is_regular_file(**ui_reset));
        EXPECT_TRUE(fs::is_regular_file(**lifecycle_reset));
    }

    TEST_F(UserPathsContractTest, ResetWithoutExistingFilesCreatesNoBackup) {
        const auto resolved = resolvePaths();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        const auto& paths = *resolved;
        ASSERT_TRUE(paths.ensureDirectories().has_value());

        const auto preferences_reset = paths.resetPreferences();
        ASSERT_TRUE(preferences_reset.has_value()) << preferences_reset.error();
        EXPECT_FALSE(preferences_reset->has_value());

        const auto layout_reset = paths.resetLayout();
        ASSERT_TRUE(layout_reset.has_value()) << layout_reset.error();
        EXPECT_FALSE(layout_reset->has_value());

        const auto window_reset = paths.resetWindowState();
        ASSERT_TRUE(window_reset.has_value()) << window_reset.error();
        EXPECT_FALSE(window_reset->has_value());

        const auto ui_reset = paths.resetUiPreferences();
        ASSERT_TRUE(ui_reset.has_value()) << ui_reset.error();
        EXPECT_FALSE(ui_reset->has_value());

        const auto lifecycle_reset = paths.resetProjectLifecycle();
        ASSERT_TRUE(lifecycle_reset.has_value()) << lifecycle_reset.error();
        EXPECT_FALSE(lifecycle_reset->has_value());
    }

    TEST_F(UserPathsContractTest, ResetRejectsNonRegularSettingsPath) {
        const auto resolved = resolvePaths();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        const auto& paths = *resolved;
        ASSERT_TRUE(fs::create_directories(paths.layoutFile()));

        const auto reset = paths.resetLayout();
        EXPECT_FALSE(reset.has_value());
        EXPECT_TRUE(fs::is_directory(paths.layoutFile()));
    }

    TEST_F(UserPathsContractTest, EnsureDirectoriesRejectsFileCollision) {
        const auto resolved = resolvePaths();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        ASSERT_TRUE(fs::create_directories(resolved->configDir().parent_path()));
        {
            std::ofstream collision(resolved->configDir());
            collision << "not a directory";
        }
        EXPECT_FALSE(resolved->ensureDirectories().has_value());
    }

    TEST_F(UserPathsContractTest, ConsecutiveResetsUseDistinctBackupDirectories) {
        const auto resolved = resolvePaths();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        const auto& paths = *resolved;
        ASSERT_TRUE(paths.ensureDirectories().has_value());

        std::ofstream(paths.layoutFile()) << "first";
        const auto first = paths.resetLayout();
        ASSERT_TRUE(first && first->has_value());
        std::ofstream(paths.layoutFile()) << "second";
        const auto second = paths.resetLayout();
        ASSERT_TRUE(second && second->has_value());
        EXPECT_NE((*first)->parent_path(), (*second)->parent_path());
    }

#ifdef _WIN32
    TEST_F(UserPathsContractTest, WindowsDefaultUsesProfileDotLichtfeld) {
        const ScopedEnvironmentVariable lfs_home("LFS_HOME", std::nullopt);
        const ScopedEnvironmentVariable profile("USERPROFILE", root_.string());
        const auto resolved = lfs::core::UserPaths::resolve();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        EXPECT_EQ(resolved->configDir(), root_ / ".lichtfeld" / "config");
    }
#else
    TEST_F(UserPathsContractTest, LinuxDefaultUsesHomeDotLichtfeldAndIgnoresXdg) {
        const ScopedEnvironmentVariable lfs_home("LFS_HOME", std::nullopt);
        const ScopedEnvironmentVariable home("HOME", (root_ / "home").string());
        const ScopedEnvironmentVariable config("XDG_CONFIG_HOME", (root_ / "xdg-config").string());
        const ScopedEnvironmentVariable data("XDG_DATA_HOME", (root_ / "xdg-data").string());
        const ScopedEnvironmentVariable cache("XDG_CACHE_HOME", (root_ / "xdg-cache").string());
        const ScopedEnvironmentVariable state("XDG_STATE_HOME", (root_ / "xdg-state").string());
        const auto resolved = lfs::core::UserPaths::resolve();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        EXPECT_EQ(resolved->configDir(), root_ / "home" / ".lichtfeld" / "config");
        EXPECT_EQ(resolved->dataDir(), root_ / "home" / ".lichtfeld" / "data");
        EXPECT_EQ(resolved->cacheDir(), root_ / "home" / ".lichtfeld" / "cache");
        EXPECT_EQ(resolved->logDir(), root_ / "home" / ".lichtfeld" / "logs");
        EXPECT_EQ(resolved->pluginDir(), root_ / "home" / ".lichtfeld" / "plugins");
        EXPECT_EQ(resolved->venvDir(), root_ / "home" / ".lichtfeld" / "venv");
    }

    TEST_F(UserPathsContractTest, LinuxDefaultFallsBackToHomeForEmptyXdgVariables) {
        const ScopedEnvironmentVariable lfs_home("LFS_HOME", std::nullopt);
        const ScopedEnvironmentVariable home("HOME", (root_ / "home").string());
        const ScopedEnvironmentVariable config("XDG_CONFIG_HOME", "");
        const ScopedEnvironmentVariable data("XDG_DATA_HOME", "");
        const ScopedEnvironmentVariable cache("XDG_CACHE_HOME", "");
        const ScopedEnvironmentVariable state("XDG_STATE_HOME", "");
        const auto resolved = lfs::core::UserPaths::resolve();
        ASSERT_TRUE(resolved.has_value()) << resolved.error();
        EXPECT_EQ(resolved->configDir(), root_ / "home" / ".lichtfeld" / "config");
        EXPECT_EQ(resolved->logDir(), root_ / "home" / ".lichtfeld" / "logs");
    }
#endif

} // namespace
