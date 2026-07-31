/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "project/project_lifecycle.hpp"

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

namespace {

    namespace fs = std::filesystem;
    using lfs::core::Uuid;
    using namespace lfs::vis::project;

    class TemporaryDirectory {
    public:
        TemporaryDirectory()
            : path(fs::temp_directory_path() /
                   ("lfs-p6-lifecycle-" +
                    lfs::core::generate_uuid_v4().to_string())) {
            fs::create_directories(path);
        }

        ~TemporaryDirectory() {
            std::error_code ignored;
            fs::remove_all(path, ignored);
        }

        fs::path path;
    };

    TEST(ProjectLifecycleSettingsTest,
         MissingSettingsUseVisionDefaults) {
        TemporaryDirectory temporary;
        auto loaded = loadProjectLifecycleSettings(
            temporary.path / "missing.json");
        ASSERT_TRUE(loaded)
            << lfs::format_for_developer(loaded.error());
        EXPECT_TRUE(loaded->reopen_last_project);
        EXPECT_TRUE(loaded->auto_save_on_close);
        EXPECT_EQ(
            loaded->autosave_interval_seconds,
            5u * 60u);
        EXPECT_EQ(
            loaded->autosave_dirty_epoch_threshold,
            20u);
        EXPECT_EQ(
            loaded->compaction_idle_seconds,
            30u);
        EXPECT_TRUE(loaded->mru.empty());
    }

    TEST(ProjectLifecycleSettingsTest,
         RoundTripPersistsUuidFirstMruAndRebindsMovedPath) {
        TemporaryDirectory temporary;
        const auto settings_path =
            temporary.path / "project_lifecycle.json";
        const Uuid first_uuid =
            lfs::core::generate_uuid_v4();
        const Uuid second_uuid =
            lfs::core::generate_uuid_v4();

        ProjectLifecycleSettings settings;
        settings.reopen_last_project = false;
        settings.auto_save_on_close = false;
        settings.autosave_interval_seconds = 17;
        settings.autosave_dirty_epoch_threshold = 9;
        settings.compaction_idle_seconds = 41;
        rememberProject(
            settings, first_uuid,
            temporary.path / "old" / "first.licht");
        rememberProject(
            settings, second_uuid,
            temporary.path / "second.licht");
        rememberProject(
            settings, first_uuid,
            temporary.path / "moved" / "first.licht");

        ASSERT_EQ(settings.mru.size(), 2u);
        EXPECT_EQ(settings.mru[0].project_uuid, first_uuid);
        EXPECT_EQ(
            settings.mru[0].last_known_path,
            temporary.path / "moved" / "first.licht");
        EXPECT_EQ(settings.mru[1].project_uuid, second_uuid);

        auto saved =
            saveProjectLifecycleSettings(
                settings_path, settings);
        ASSERT_TRUE(saved)
            << lfs::format_for_developer(saved.error());
        auto loaded =
            loadProjectLifecycleSettings(settings_path);
        ASSERT_TRUE(loaded)
            << lfs::format_for_developer(loaded.error());
        EXPECT_EQ(*loaded, settings);
    }

    TEST(ProjectLifecycleSettingsTest,
         InvalidSettingsFailClosedInsteadOfChangingDefaults) {
        TemporaryDirectory temporary;
        const auto path =
            temporary.path / "project_lifecycle.json";
        {
            std::ofstream stream(path);
            stream << R"({"version":2,"reopen_last_project":false})";
        }
        auto loaded = loadProjectLifecycleSettings(path);
        ASSERT_FALSE(loaded);
        EXPECT_EQ(
            loaded.error().code(),
            lfs::ErrorCode::DataLoss);
    }

} // namespace
