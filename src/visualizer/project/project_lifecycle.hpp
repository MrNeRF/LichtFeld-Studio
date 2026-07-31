/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "core/error.hpp"
#include "core/uuid.hpp"
#include "io/project_document.hpp"
#include "visualizer/visualizer.hpp"

#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace lfs::vis {
    class VisualizerImpl;
}

namespace lfs::vis::project {

    struct ProjectMruEntry {
        lfs::core::Uuid project_uuid;
        std::filesystem::path last_known_path;

        friend bool operator==(const ProjectMruEntry&,
                               const ProjectMruEntry&) = default;
    };

    struct ProjectLifecycleSettings {
        bool reopen_last_project = true;
        bool auto_save_on_close = true;
        std::vector<ProjectMruEntry> mru;

        friend bool operator==(const ProjectLifecycleSettings&,
                               const ProjectLifecycleSettings&) = default;
    };

    [[nodiscard]] LFS_VIS_API lfs::Result<ProjectLifecycleSettings>
    loadProjectLifecycleSettings(const std::filesystem::path& path);
    [[nodiscard]] LFS_VIS_API lfs::Result<void>
    saveProjectLifecycleSettings(
        const std::filesystem::path& path,
        const ProjectLifecycleSettings& settings);
    LFS_VIS_API void rememberProject(
        ProjectLifecycleSettings& settings,
        const lfs::core::Uuid& project_uuid,
        const std::filesystem::path& path);

    class ProjectLifecycle {
    public:
        enum class CloseSaveStatus {
            NotDirty,
            NeedsPrompt,
            Saving,
            Succeeded,
            Failed,
        };

        explicit ProjectLifecycle(
            VisualizerImpl& viewer,
            std::optional<std::filesystem::path>
                settings_path = std::nullopt);
        ~ProjectLifecycle();

        ProjectLifecycle(const ProjectLifecycle&) = delete;
        ProjectLifecycle& operator=(const ProjectLifecycle&) = delete;

        [[nodiscard]] lfs::Result<void>
        open(
            const std::filesystem::path& path,
            ProjectSwitchDisposition disposition =
                ProjectSwitchDisposition::RequireClean);
        [[nodiscard]] lfs::Result<void>
        save(bool regenerate_preview);
        [[nodiscard]] lfs::Result<void>
        saveAs(const std::filesystem::path& path,
               bool regenerate_preview);
        [[nodiscard]] lfs::Result<void>
        newProject(
            ProjectSwitchDisposition disposition =
                ProjectSwitchDisposition::RequireClean);
        [[nodiscard]] lfs::Result<ProjectInfo> info();
        [[nodiscard]] lfs::Result<void>
        preflightSwitch(
            ProjectSwitchDisposition disposition);

        void openStartupProject(
            const std::optional<std::filesystem::path>& explicit_path);
        void markSceneMutation(std::uint32_t mutation_flags);
        [[nodiscard]] bool hasDirtyProject();
        [[nodiscard]] bool autoSaveOnClose() const noexcept {
            return settings_.auto_save_on_close;
        }
        [[nodiscard]] lfs::Result<void>
        setReopenLastProject(bool enabled);
        [[nodiscard]] lfs::Result<void>
        setAutoSaveOnClose(bool enabled);
        [[nodiscard]] bool containsEmbeddedSecrets() const;
        [[nodiscard]] CloseSaveStatus
        beginOrPollCloseSave();
        void resetCloseSaveAttempt();
        [[nodiscard]] std::string
        closeSaveError() const;

    private:
        enum class Hydration {
            Empty,
            ShellReady,
            Hydrating,
            Complete,
            Failed,
        };

        enum class CloseSaveState {
            Idle,
            Saving,
            Succeeded,
            Failed,
        };

        [[nodiscard]] lfs::Result<void>
        synchronizeDocumentFromViewer();
        [[nodiscard]] lfs::Result<void>
        adoptCompletedTrainingSnapshot();
        [[nodiscard]] lfs::Result<std::vector<std::byte>>
        capturePreviewPng() const;
        [[nodiscard]] lfs::Result<void>
        launchHydration(
            std::shared_ptr<lfs::io::project::ProjectDocument> document,
            std::uint64_t epoch,
            std::uint64_t selection_mutation_serial,
            std::vector<lfs::core::Uuid> selected_node_uuids);
        [[nodiscard]] lfs::Result<void>
        persistSettings();
        void markHydrationFailed(
            std::uint64_t epoch,
            const std::string& detail);
        [[nodiscard]] static std::string
        hydrationName(Hydration state);

        VisualizerImpl& viewer_;
        std::shared_ptr<lfs::io::project::ProjectDocument> document_;
        ProjectLifecycleSettings settings_;
        std::filesystem::path settings_path_;
        std::atomic<std::uint64_t> epoch_{0};
        std::atomic<std::uint64_t> scene_mutation_serial_{0};
        std::atomic<std::uint64_t>
            selection_mutation_serial_{0};
        std::atomic<Hydration> hydration_{Hydration::Empty};
        std::atomic<bool> scene_dirty_{false};
        std::atomic<bool> payload_dirty_{false};
        std::uint64_t
            adopted_training_snapshot_count_ = 0;
        mutable std::mutex thread_mutex_;
        std::vector<std::jthread> hydration_threads_;
        std::jthread close_save_thread_;
        std::atomic<CloseSaveState>
            close_save_state_{CloseSaveState::Idle};
        mutable std::mutex close_save_mutex_;
        std::string close_save_error_;
        std::string hydration_error_;
    };

} // namespace lfs::vis::project
