/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/localization_manager.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

    class PluginLocalizationTest : public ::testing::Test {
    protected:
        void SetUp() override {
            static std::atomic_uint64_t sequence{0};
            test_directory_ = std::filesystem::temp_directory_path() /
                              ("lfs_plugin_localization_" +
                               std::to_string(
                                   std::chrono::steady_clock::now().time_since_epoch().count()) +
                               "_" +
                               std::to_string(sequence.fetch_add(1)));
            locale_directory_ = test_directory_ / "locales";
            std::filesystem::create_directories(locale_directory_);
            {
                std::ofstream index(test_directory_ / "locale_index.json");
                index << R"({"languages":[{"code":"en","name":"English"},{"code":"it","name":"Italiano"}]})";
                ASSERT_TRUE(index.good());
            }
            writeLocale("en", R"({"_language_name":"English","core":{"label":"Core"}})");
            writeLocale("it", R"({"_language_name":"Italiano","core":{"label":"Nucleo"}})");

            auto& localization = lfs::event::LocalizationManager::getInstance();
            localization.reset();
            ASSERT_TRUE(localization.initialize(locale_directory_.string()));
        }

        void TearDown() override {
            lfs::event::LocalizationManager::getInstance().reset();
            std::error_code error;
            std::filesystem::remove_all(test_directory_, error);
        }

        void writeLocale(const std::string& language, const std::string& contents) const {
            std::ofstream stream(locale_directory_ / (language + ".json"));
            stream << contents;
        }

        std::filesystem::path test_directory_;
        std::filesystem::path locale_directory_;
    };

} // namespace

TEST_F(PluginLocalizationTest, UsesActiveLanguageThenEnglishFallback) {
    auto& localization = lfs::event::LocalizationManager::getInstance();
    using TranslationMap = lfs::event::LocalizationManager::TranslationMap;

    const auto english = localization.registerPluginCatalog(
        "example-plugin", "en", TranslationMap{{"panel.title", "Title"}});
    const auto italian = localization.registerPluginCatalog(
        "example-plugin", "it", TranslationMap{{"panel.title", "Titolo"}});
    ASSERT_NE(english, 0);
    ASSERT_NE(italian, 0);

    ASSERT_TRUE(localization.setLanguage("it"));
    EXPECT_STREQ(localization.get("plugins.example-plugin.panel.title"), "Titolo");

    EXPECT_TRUE(localization.unregisterPluginCatalog(italian));
    EXPECT_STREQ(localization.get("plugins.example-plugin.panel.title"), "Title");
    EXPECT_STREQ(localization.getEnglishFallback("plugins.example-plugin.panel.title"),
                 "Title");

    EXPECT_TRUE(localization.unregisterPluginCatalog(english));
    EXPECT_STREQ(localization.get("plugins.example-plugin.panel.title"),
                 "plugins.example-plugin.panel.title");
}

TEST_F(PluginLocalizationTest, IsolatesOwnersAndRejectsCollisions) {
    auto& localization = lfs::event::LocalizationManager::getInstance();
    using TranslationMap = lfs::event::LocalizationManager::TranslationMap;

    const auto first = localization.registerPluginCatalog(
        "first-plugin", "en", TranslationMap{{"action.run", "First"}});
    const auto second = localization.registerPluginCatalog(
        "second-plugin", "en", TranslationMap{{"action.run", "Second"}});
    ASSERT_NE(first, 0);
    ASSERT_NE(second, 0);

    std::string error;
    EXPECT_EQ(localization.registerPluginCatalog(
                  "first-plugin", "en", TranslationMap{{"action.run", "Duplicate"}}, &error),
              0);
    EXPECT_FALSE(error.empty());
    EXPECT_STREQ(localization.get("plugins.first-plugin.action.run"), "First");
    EXPECT_STREQ(localization.get("plugins.second-plugin.action.run"), "Second");

    EXPECT_EQ(localization.unregisterPluginCatalogs("first-plugin"), 1);
    EXPECT_FALSE(localization.hasKey("plugins.first-plugin.action.run"));
    EXPECT_TRUE(localization.hasKey("plugins.second-plugin.action.run"));
}

TEST_F(PluginLocalizationTest, PreservesLegacyOverridePrecedence) {
    auto& localization = lfs::event::LocalizationManager::getInstance();
    using TranslationMap = lfs::event::LocalizationManager::TranslationMap;

    ASSERT_NE(localization.registerPluginCatalog(
                  "example-plugin", "en", TranslationMap{{"panel.title", "Catalog"}}),
              0);
    localization.setOverride("plugins.example-plugin.panel.title", "Override");
    EXPECT_STREQ(localization.get("plugins.example-plugin.panel.title"), "Override");

    localization.clearOverride("plugins.example-plugin.panel.title");
    EXPECT_STREQ(localization.get("plugins.example-plugin.panel.title"), "Catalog");
}

TEST_F(PluginLocalizationTest, RejectsInvalidCatalogBoundariesWithoutMutation) {
    auto& localization = lfs::event::LocalizationManager::getInstance();
    using TranslationMap = lfs::event::LocalizationManager::TranslationMap;

    EXPECT_EQ(localization.registerPluginCatalog(
                  "Example_Plugin", "en", TranslationMap{{"panel.title", "Title"}}),
              0);
    EXPECT_EQ(localization.registerPluginCatalog(
                  "example-plugin", "EN", TranslationMap{{"panel.title", "Title"}}),
              0);
    EXPECT_EQ(localization.registerPluginCatalog(
                  "example-plugin", "en", TranslationMap{{"plugins.core.title", "Title"}}),
              0);
    EXPECT_EQ(localization.registerPluginCatalog(
                  "example-plugin", "en", TranslationMap{{"panel.title", ""}}),
              0);
    EXPECT_EQ(localization.registerPluginCatalog(
                  "example-plugin", "en", TranslationMap{{"panel.title", std::string("A\0B", 3)}}),
              0);
    EXPECT_FALSE(localization.hasKey("plugins.example-plugin.panel.title"));
}

TEST_F(PluginLocalizationTest, NativeMutationsPublishGenerationButNoOpsDoNot) {
    auto& localization = lfs::event::LocalizationManager::getInstance();
    using TranslationMap = lfs::event::LocalizationManager::TranslationMap;

    const auto initial = localization.getCurrentLanguageGeneration();
    const auto english = localization.registerPluginCatalog(
        "example-plugin", "en", TranslationMap{{"panel.title", "Title"}});
    ASSERT_NE(english, 0);
    const auto registered = localization.getCurrentLanguageGeneration();
    EXPECT_GT(registered, initial);
    EXPECT_EQ(localization.registerPluginCatalog(
                  "example-plugin", "en", TranslationMap{{"panel.title", "Duplicate"}}),
              0);
    EXPECT_FALSE(localization.unregisterPluginCatalog(0));
    EXPECT_EQ(localization.unregisterPluginCatalogs("missing-plugin"), 0);
    EXPECT_EQ(localization.getCurrentLanguageGeneration(), registered);

    ASSERT_TRUE(localization.unregisterPluginCatalog(english));
    const auto removed = localization.getCurrentLanguageGeneration();
    EXPECT_GT(removed, registered);
    EXPECT_FALSE(localization.unregisterPluginCatalog(english));
    EXPECT_EQ(localization.getCurrentLanguageGeneration(), removed);

    ASSERT_NE(localization.registerPluginCatalog(
                  "example-plugin", "en", TranslationMap{{"panel.title", "Title"}}),
              0);
    ASSERT_NE(localization.registerPluginCatalog(
                  "example-plugin", "it", TranslationMap{{"panel.title", "Titolo"}}),
              0);
    const auto before_bulk_remove = localization.getCurrentLanguageGeneration();
    EXPECT_EQ(localization.unregisterPluginCatalogs("example-plugin"), 2);
    EXPECT_GT(localization.getCurrentLanguageGeneration(), before_bulk_remove);
}

TEST_F(PluginLocalizationTest, ResetInvalidatesCatalogsWithoutReusingTokens) {
    auto& localization = lfs::event::LocalizationManager::getInstance();
    using TranslationMap = lfs::event::LocalizationManager::TranslationMap;

    const auto stale_token = localization.registerPluginCatalog(
        "example-plugin", "en", TranslationMap{{"panel.title", "Before reset"}});
    ASSERT_NE(stale_token, 0);

    localization.reset();
    ASSERT_TRUE(localization.initialize(locale_directory_.string()));
    const auto current_token = localization.registerPluginCatalog(
        "example-plugin", "en", TranslationMap{{"panel.title", "After reset"}});
    ASSERT_NE(current_token, 0);
    EXPECT_NE(current_token, stale_token);

    EXPECT_FALSE(localization.unregisterPluginCatalog(stale_token));
    EXPECT_STREQ(localization.get("plugins.example-plugin.panel.title"), "After reset");
}
