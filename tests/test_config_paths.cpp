/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/config_paths.hpp"
#include "core/event_bridge/localization_manager.hpp"

#include <gtest/gtest.h>

#include <visualizer/gui/layout_state.hpp>
#include <visualizer/input/input_bindings.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

#ifndef _WIN32
    class ScopedXdgConfigHome {
    public:
        explicit ScopedXdgConfigHome(const fs::path& dir) {
            if (const char* previous = std::getenv("XDG_CONFIG_HOME")) {
                previous_ = previous;
            }
            fs::create_directories(dir);
            setenv("XDG_CONFIG_HOME", dir.string().c_str(), 1);
        }

        ~ScopedXdgConfigHome() {
            if (previous_) {
                setenv("XDG_CONFIG_HOME", previous_->c_str(), 1);
            } else {
                unsetenv("XDG_CONFIG_HOME");
            }
        }

        ScopedXdgConfigHome(const ScopedXdgConfigHome&) = delete;
        ScopedXdgConfigHome& operator=(const ScopedXdgConfigHome&) = delete;

    private:
        std::optional<std::string> previous_;
    };

    fs::path makeTempConfigRoot(const char* tag) {
        const auto root = fs::temp_directory_path() /
                          (std::string("lfs_config_paths_") + tag + "_" + std::to_string(::getpid()));
        std::error_code ec;
        fs::remove_all(root, ec);
        fs::create_directories(root);
        return root;
    }
#endif

    TEST(UserConfigDir, ReturnsNonEmptyStablePath) {
        const auto first = lfs::core::user_config_dir();
        const auto second = lfs::core::user_config_dir();

        EXPECT_FALSE(first.empty());
        EXPECT_EQ(first, second);
    }

#ifndef _WIN32
    TEST(UserConfigDir, HonorsXdgConfigHomeOverride) {
        const auto root = makeTempConfigRoot("xdg");
        {
            const ScopedXdgConfigHome xdg(root);
            EXPECT_EQ(lfs::core::user_config_dir(), root / "LichtFeldStudio");
        }
        std::error_code ec;
        fs::remove_all(root, ec);
    }

    TEST(UserConfigDir, InputProfilesUseSharedConfigRoot) {
        const auto root = makeTempConfigRoot("input_profiles");
        {
            const ScopedXdgConfigHome xdg(root);
            EXPECT_EQ(lfs::vis::input::InputBindings::getConfigDir(),
                      root / "LichtFeldStudio" / "input_profiles");
        }
        std::error_code ec;
        fs::remove_all(root, ec);
    }

    TEST(LayoutStatePersistence, ImportsLegacyLayoutWithoutWritingIt) {
        const auto root =
            makeTempConfigRoot("layout_import");
        {
            const ScopedXdgConfigHome xdg(root);
            const auto legacy =
                root / "LichtFeldStudio" /
                "layout.json";
            fs::create_directories(
                legacy.parent_path());
            std::ofstream output(legacy);
            output
                << R"({"left_dock_width":417.0,"bottom_dock_height":222.0,"scene_panel_ratio":0.61,"show_sequencer":true})";
            output.close();

            lfs::vis::gui::LayoutState loaded;
            loaded.load();
            EXPECT_FLOAT_EQ(loaded.left_dock_width, 417.0f);
            EXPECT_FLOAT_EQ(loaded.bottom_dock_height, 222.0f);
            EXPECT_FLOAT_EQ(loaded.scene_panel_ratio, 0.61f);
            EXPECT_TRUE(loaded.show_sequencer);
            EXPECT_EQ(
                fs::file_size(legacy),
                std::string_view(
                    R"({"left_dock_width":417.0,"bottom_dock_height":222.0,"scene_panel_ratio":0.61,"show_sequencer":true})")
                    .size());
        }
        std::error_code ec;
        fs::remove_all(root, ec);
    }

    TEST(LayoutStatePersistence, WritesOnlyUserGlobalUiPreferences) {
        const auto root =
            makeTempConfigRoot("ui_preferences");
        {
            const ScopedXdgConfigHome xdg(root);
            lfs::vis::gui::LayoutState saved;
            saved.left_dock_width = 417.0f;
            saved.file_association = "declined";
            saved.vram_hud_x = 23.0f;
            saved.saveUserPreferences();

            const auto config =
                root / "LichtFeldStudio";
            EXPECT_FALSE(
                fs::exists(
                    config / "layout.json"));
            ASSERT_TRUE(
                fs::exists(
                    config /
                    "ui_preferences.json"));

            lfs::vis::gui::LayoutState loaded;
            loaded.load();
            EXPECT_EQ(
                loaded.file_association,
                "declined");
            EXPECT_FLOAT_EQ(
                loaded.vram_hud_x, 23.0f);
            EXPECT_FLOAT_EQ(
                loaded.left_dock_width,
                320.0f);
        }
        std::error_code ec;
        fs::remove_all(root, ec);
    }

    TEST(LocalizationPersistence, SavesReloadsAndFallsBackFromUnavailableLanguage) {
        const auto root = makeTempConfigRoot("language");
        const auto locales = root / "locales";
        fs::create_directories(locales);
        {
            std::ofstream english(locales / "en.json");
            english << R"({"_language_name":"English","test":{"value":"English value"}})";
            std::ofstream german(locales / "de.json");
            german << R"({"_language_name":"Deutsch","test":{"value":"Deutscher Wert"}})";
        }

        {
            const ScopedXdgConfigHome xdg(root / "config");
            auto& manager = lfs::event::LocalizationManager::getInstance();
            ASSERT_TRUE(manager.initialize(locales.string()));
            EXPECT_TRUE(manager.hasKey("test.value"));
            EXPECT_FALSE(manager.hasKey("session.licht"));
            ASSERT_TRUE(manager.setLanguage("de"));
            EXPECT_EQ(manager.getCurrentLanguage(), "de");

            const auto preference = root / "config" / "LichtFeldStudio" / "language_preference";
            ASSERT_TRUE(fs::exists(preference));
            {
                std::ifstream saved(preference);
                std::string language;
                ASSERT_TRUE(saved >> language);
                EXPECT_EQ(language, "de");
            }

            ASSERT_TRUE(manager.initialize(locales.string()));
            EXPECT_EQ(manager.getCurrentLanguage(), "de");

            ASSERT_TRUE(fs::remove(locales / "de.json"));
            ASSERT_TRUE(manager.initialize(locales.string()));
            EXPECT_EQ(manager.getCurrentLanguage(), "en");
            {
                std::ifstream saved(preference);
                std::string language;
                ASSERT_TRUE(saved >> language);
                EXPECT_EQ(language, "en");
            }
        }

        std::error_code ec;
        fs::remove_all(root, ec);
    }

    TEST(LocalizationPersistence, LocaleProjectStringsAreTranslated) {
        const fs::path locale_root =
            fs::current_path() / "src/visualizer/gui/resources/locales";
        ASSERT_TRUE(fs::exists(locale_root));
        for (const std::string_view language :
             {"de", "es", "fr", "it", "ja", "ko", "nl", "pl", "zh"}) {
            std::ifstream input(locale_root / (std::string(language) + ".json"));
            ASSERT_TRUE(input);
            const std::string contents{
                std::istreambuf_iterator<char>(input),
                std::istreambuf_iterator<char>()};
            EXPECT_EQ(contents.find("\"opening_project\": \"Opening project...\""),
                      std::string::npos)
                << language;
            EXPECT_EQ(contents.find("\"project_missing\": \"Project file not found: {name}\""),
                      std::string::npos)
                << language;
            EXPECT_EQ(contents.find("\"blocked_during_training\": \"Stop training before loading a new scene.\""),
                      std::string::npos)
                << language;
        }
    }
#endif

} // namespace
