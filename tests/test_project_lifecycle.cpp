/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/services.hpp"
#include "licht_test_support.hpp"
#include "operation/undo_history.hpp"
#include "project/project_lifecycle.hpp"
#include "visualizer/visualizer.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

namespace {

    namespace fs = std::filesystem;
    using lfs::core::Uuid;
    using lfs::test::licht::TemporaryDirectory;
    using namespace lfs::vis::project;

    void createEmptyProjectFile(const fs::path& path) {
        std::error_code error;
        fs::create_directories(path.parent_path(), error);
        ASSERT_FALSE(error) << error.message();
        std::ofstream stream(path, std::ios::binary);
        ASSERT_TRUE(stream) << path;
    }

    TEST(ProjectLifecycleSettingsTest,
         MissingSettingsUseVisionDefaults) {
        TemporaryDirectory temporary;
        auto loaded = loadProjectLifecycleSettings(
            temporary.path / "missing.json");
        ASSERT_TRUE(loaded)
            << lfs::format_for_developer(loaded.error());
        EXPECT_TRUE(loaded->reopen_last_project);
        EXPECT_FALSE(loaded->auto_save_on_close);
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
        const auto old_path =
            temporary.path / "old" / "first.licht";
        const auto second_path =
            temporary.path / "second.licht";
        const auto moved_path =
            temporary.path / "moved" / "first.licht";
        createEmptyProjectFile(old_path);
        createEmptyProjectFile(second_path);
        createEmptyProjectFile(moved_path);

        ProjectLifecycleSettings settings;
        settings.reopen_last_project = false;
        settings.auto_save_on_close = false;
        settings.autosave_interval_seconds = 17;
        settings.autosave_dirty_epoch_threshold = 9;
        settings.compaction_idle_seconds = 41;
        rememberProject(
            settings, first_uuid, old_path);
        rememberProject(
            settings, second_uuid, second_path);
        rememberProject(
            settings, first_uuid, moved_path);

        ASSERT_EQ(settings.mru.size(), 2u);
        EXPECT_EQ(settings.mru[0].project_uuid, first_uuid);
        EXPECT_EQ(
            settings.mru[0].last_known_path,
            resolveProjectMruPath(moved_path));
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
         RememberDeduplicatesResolvedPathAcrossProjectUuids) {
        TemporaryDirectory temporary;
        const auto project_path =
            temporary.path / "dir" / "project.licht";
        createEmptyProjectFile(project_path);
        const Uuid first_uuid =
            lfs::core::generate_uuid_v4();
        const Uuid second_uuid =
            lfs::core::generate_uuid_v4();

        ProjectLifecycleSettings settings;
        rememberProject(settings, first_uuid, project_path);
        rememberProject(
            settings, second_uuid,
            temporary.path / "dir" / ".." / "dir" /
                "project.licht");

        ASSERT_EQ(settings.mru.size(), 1u);
        EXPECT_EQ(settings.mru.front().project_uuid, second_uuid);
        EXPECT_EQ(
            settings.mru.front().last_known_path,
            resolveProjectMruPath(project_path));
    }

    TEST(ProjectLifecycleSettingsTest,
         RememberPrunesMissingSiblingEntries) {
        TemporaryDirectory temporary;
        const auto keep_path =
            temporary.path / "keep.licht";
        const auto gone_path =
            temporary.path / "gone.licht";
        const auto new_path =
            temporary.path / "new.licht";
        createEmptyProjectFile(keep_path);
        createEmptyProjectFile(gone_path);
        createEmptyProjectFile(new_path);
        const Uuid keep_uuid =
            lfs::core::generate_uuid_v4();
        const Uuid gone_uuid =
            lfs::core::generate_uuid_v4();
        const Uuid new_uuid =
            lfs::core::generate_uuid_v4();

        ProjectLifecycleSettings settings;
        rememberProject(settings, keep_uuid, keep_path);
        rememberProject(settings, gone_uuid, gone_path);
        std::error_code error;
        ASSERT_TRUE(fs::remove(gone_path, error));
        ASSERT_FALSE(error) << error.message();
        rememberProject(settings, new_uuid, new_path);

        ASSERT_EQ(settings.mru.size(), 2u);
        EXPECT_EQ(settings.mru[0].project_uuid, new_uuid);
        EXPECT_EQ(settings.mru[1].project_uuid, keep_uuid);
    }

    TEST(ProjectLifecycleSettingsTest,
         PruneMissingMruEntriesKeepsExistingEntry) {
        TemporaryDirectory temporary;
        const auto existing_path =
            temporary.path / "existing.licht";
        const auto missing_path =
            temporary.path / "missing.licht";
        createEmptyProjectFile(existing_path);
        const Uuid existing_uuid =
            lfs::core::generate_uuid_v4();

        ProjectLifecycleSettings settings;
        settings.mru = {
            {
                .project_uuid =
                    lfs::core::generate_uuid_v4(),
                .last_known_path = missing_path,
            },
            {
                .project_uuid = existing_uuid,
                .last_known_path = existing_path,
            },
        };

        pruneMissingMruEntries(settings);

        ASSERT_EQ(settings.mru.size(), 1u);
        EXPECT_EQ(
            settings.mru.front().project_uuid,
            existing_uuid);
        EXPECT_EQ(
            settings.mru.front().last_known_path,
            existing_path);
    }

    TEST(ProjectLifecycleSettingsTest,
         LoadKeepsMissingMruUntilExplicitPrune) {
        TemporaryDirectory temporary;
        const auto settings_path =
            temporary.path / "project_lifecycle.json";
        const auto missing_path =
            temporary.path / "missing.licht";
        ProjectLifecycleSettings settings;
        settings.mru.push_back({
            .project_uuid =
                lfs::core::generate_uuid_v4(),
            .last_known_path = missing_path,
        });

        ASSERT_TRUE(saveProjectLifecycleSettings(
            settings_path, settings));
        auto loaded =
            loadProjectLifecycleSettings(settings_path);
        ASSERT_TRUE(loaded)
            << lfs::format_for_developer(loaded.error());
        ASSERT_EQ(loaded->mru.size(), 1u);
        EXPECT_EQ(
            loaded->mru.front().last_known_path,
            missing_path);

        pruneMissingMruEntries(*loaded);
        EXPECT_TRUE(loaded->mru.empty());
    }

    TEST(ProjectLifecycleSettingsTest,
         InvalidSettingsFailClosedInsteadOfChangingDefaults) {
        TemporaryDirectory temporary;
        const auto path =
            temporary.path / "project_lifecycle.json";
        {
            std::ofstream stream(path);
            stream << R"({"version":99,"reopen_last_project":false})";
        }
        auto loaded = loadProjectLifecycleSettings(path);
        ASSERT_FALSE(loaded);
        EXPECT_EQ(
            loaded.error().code(),
            lfs::ErrorCode::DataLoss);
    }

    TEST(ProjectLifecycleSettingsTest,
         Version1MigratesAutoSaveOnCloseToFalse) {
        TemporaryDirectory temporary;
        const auto path =
            temporary.path / "project_lifecycle.json";
        {
            std::ofstream stream(path);
            stream << R"({
              "version": 1,
              "reopen_last_project": true,
              "auto_save_on_close": true,
              "autosave_interval_seconds": 300,
              "autosave_dirty_epoch_threshold": 20,
              "compaction_idle_seconds": 30,
              "mru": []
            })";
        }
        auto loaded = loadProjectLifecycleSettings(path);
        ASSERT_TRUE(loaded)
            << lfs::format_for_developer(loaded.error());
        EXPECT_FALSE(loaded->auto_save_on_close);

        ASSERT_TRUE(saveProjectLifecycleSettings(
            path, *loaded))
            << "persist after migration";
        {
            std::ifstream stream(path);
            const auto json =
                nlohmann::json::parse(stream);
            EXPECT_EQ(json.value("version", 0), 2);
            EXPECT_FALSE(
                json.value("auto_save_on_close", true));
        }
        auto reloaded =
            loadProjectLifecycleSettings(path);
        ASSERT_TRUE(reloaded)
            << lfs::format_for_developer(
                   reloaded.error());
        EXPECT_FALSE(reloaded->auto_save_on_close);
    }

    TEST(ProjectLifecycleSettingsTest,
         Version2PreservesExplicitAutoSaveOnCloseTrue) {
        TemporaryDirectory temporary;
        const auto path =
            temporary.path / "project_lifecycle.json";
        {
            std::ofstream stream(path);
            stream << R"({
              "version": 2,
              "reopen_last_project": true,
              "auto_save_on_close": true,
              "autosave_interval_seconds": 300,
              "autosave_dirty_epoch_threshold": 20,
              "compaction_idle_seconds": 30,
              "mru": []
            })";
        }
        auto loaded = loadProjectLifecycleSettings(path);
        ASSERT_TRUE(loaded)
            << lfs::format_for_developer(loaded.error());
        EXPECT_TRUE(loaded->auto_save_on_close);
    }

    TEST(ProjectLifecycleSettingsTest,
         RememberProjectSkipsUnpublishedWriteTemp) {
        // Would fail if rememberProject still inserted
        // *.project-write.*.tmp.licht into MRU.
        TemporaryDirectory temporary;
        const auto temp_path =
            temporary.path /
            "project.project-write.1.2.3.tmp.licht";
        createEmptyProjectFile(temp_path);
        ProjectLifecycleSettings settings;
        rememberProject(
            settings, lfs::core::generate_uuid_v4(),
            temp_path);
        EXPECT_TRUE(settings.mru.empty());
    }

    TEST(ProjectLifecycleSettingsTest,
         PruneMissingMruEntriesDropsUnpublishedPath) {
        // Would fail if prune only checked existence and
        // left unpublished temps in MRU.
        TemporaryDirectory temporary;
        const auto temp_path =
            temporary.path /
            "project.project-write.1.2.3.tmp.licht";
        createEmptyProjectFile(temp_path);
        ProjectLifecycleSettings settings;
        settings.mru.push_back({
            .project_uuid = lfs::core::generate_uuid_v4(),
            .last_known_path = temp_path,
        });
        pruneMissingMruEntries(settings);
        EXPECT_TRUE(settings.mru.empty());
    }

    TEST(ProjectLifecycleSettingsTest,
         ClearRecentProjectsEmptiesMenuInfoAndPersists) {
        lfs::event::EventBridge::instance().clear_all();
        lfs::core::event::bus().clear_all();
        lfs::vis::services().clear();
        lfs::vis::op::undoHistory().clear();

        TemporaryDirectory temporary;
        const auto settings_path =
            temporary.path / "project_lifecycle.json";
        const auto project_path =
            temporary.path / "recent.licht";
        createEmptyProjectFile(project_path);

        ProjectLifecycleSettings settings;
        rememberProject(
            settings, lfs::core::generate_uuid_v4(),
            project_path);
        ASSERT_FALSE(settings.mru.empty());
        ASSERT_TRUE(saveProjectLifecycleSettings(
            settings_path, settings))
            << "seed MRU settings";

        lfs::vis::ViewerOptions options;
        options.show_startup_overlay = false;
        options.project_lifecycle_settings_path =
            settings_path;

        {
            auto viewer =
                lfs::vis::Visualizer::create(options);
            ASSERT_NE(viewer, nullptr);
            auto info = viewer->projectGetMenuInfo();
            ASSERT_TRUE(info)
                << lfs::format_for_developer(
                       info.error());
            ASSERT_EQ(info->recent_projects.size(), 1u);
            EXPECT_EQ(
                info->recent_projects.front()
                    .last_known_path,
                resolveProjectMruPath(project_path));

            auto cleared =
                viewer->projectClearRecentFiles();
            ASSERT_TRUE(cleared)
                << lfs::format_for_developer(
                       cleared.error());

            info = viewer->projectGetMenuInfo();
            ASSERT_TRUE(info)
                << lfs::format_for_developer(
                       info.error());
            EXPECT_TRUE(info->recent_projects.empty());
        }

        auto reloaded =
            loadProjectLifecycleSettings(settings_path);
        ASSERT_TRUE(reloaded)
            << lfs::format_for_developer(
                   reloaded.error());
        EXPECT_TRUE(reloaded->mru.empty());

        lfs::vis::op::undoHistory().clear();
        lfs::vis::services().clear();
        lfs::core::event::bus().clear_all();
        lfs::event::EventBridge::instance().clear_all();
    }

} // namespace
