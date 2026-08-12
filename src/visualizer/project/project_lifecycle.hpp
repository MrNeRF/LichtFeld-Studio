/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "core/error.hpp"
#include "core/uuid.hpp"
#include "io/project_document.hpp"
#include "io/project_recovery.hpp"
#include "visualizer/core/job_registry.hpp"
#include "visualizer/visualizer.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace lfs::vis {
    class VisualizerImpl;
    class VisualizerImplResetTest_AutosaveStartsAfterFirstSaveAsWithoutReopen_Test;
    class VisualizerImplResetTest_AutosaveSkipsWhileManualProjectWriteJobIsRunning_Test;
    class VisualizerImplResetTest_RecoveryDeclineKeepsSidecarSuppressesRepeatAndExplicitSaveDeletesIt_Test;
    class VisualizerImplResetTest_RecoveredProjectSwitchDeletesTempOnlyAfterReplacement_Test;
    class VisualizerImplResetTest_FailedNewProjectKeepsRecoveredSessionTemp_Test;
    class VisualizerImplResetTest_RecoveredCloseDeletesTempAfterDocumentTeardown_Test;
    class VisualizerImplResetTest_ProjectWriteSettlementCompletesBeforeNextDocumentWrite_Test;
    class VisualizerImplResetTest_TrainingSnapshotCleanupTerminalizesProjectWrite_Test;
    class VisualizerImplResetTest_TrainingSnapshotPrepareFailureTerminalizesProjectWrite_Test;
    class VisualizerImplResetTest_TrainingSnapshotSupersedeTerminalizesOldAndCompletesNew_Test;
    class VisualizerImplResetTest_TrainingSnapshotCancelTerminalizesBeforeSettlement_Test;
    class VisualizerImplResetTest_FailedAutosaveSettlementAppliesBackoffBeforeRetry_Test;
    class VisualizerImplResetTest_PendingCloseSuppressesBackgroundAutosave_Test;
    class VisualizerImplResetTest_StoppingTrainerBlocksIdleCompactionAndAutosave_Test;
    class VisualizerImplResetTest_SessionSoftDirtyDoesNotPromptOrArmAutosave_Test;
    class VisualizerImplResetTest_SceneEditStillPromptsAndArmsAutosave_Test;
    class VisualizerImplResetTest_ParametersUnchangedRoundTripStaysClean_Test;
    class VisualizerImplResetTest_ParametersValueChangeIsHardDirty_Test;
    class VisualizerImplResetTest_BaselineIdleCheckpointTrainerClosesWithoutTrainingPrompt_Test;
    class VisualizerImplResetTest_ProgressedPausedTrainerStillBlocksCleanClose_Test;
} // namespace lfs::vis

namespace lfs::vis::project {

    struct ProjectMruEntry {
        lfs::core::Uuid project_uuid;
        std::filesystem::path last_known_path;

        friend bool operator==(const ProjectMruEntry&,
                               const ProjectMruEntry&) = default;
    };

    struct ProjectLifecycleSettings {
        bool reopen_last_project = true;
        bool auto_save_on_close = false;
        std::uint64_t autosave_interval_seconds = 5 * 60;
        std::uint64_t autosave_dirty_epoch_threshold = 20;
        std::uint64_t compaction_idle_seconds = 30;
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

        [[nodiscard]] lfs::Result<ProjectOpenOutcome>
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
        compact();
        [[nodiscard]] lfs::Result<void>
        newProject(
            ProjectSwitchDisposition disposition =
                ProjectSwitchDisposition::RequireClean);
        [[nodiscard]] bool isDirty();
        [[nodiscard]] bool hasSourcePath() const;
        [[nodiscard]] lfs::Result<ProjectInfo> info();
        [[nodiscard]] ProjectMenuInfo menuInfo() const;
        [[nodiscard]] lfs::Result<void>
        preflightSwitch(
            ProjectSwitchDisposition disposition);

        void openStartupProject(
            const std::optional<std::filesystem::path>& explicit_path);
        void markSceneMutation(std::uint32_t mutation_flags);
        void updateMaintenance();
        void noteProjectFrameRendered(double render_ms);
        [[nodiscard]] bool hasDirtyProject();
        [[nodiscard]] lfs::Result<void>
        setReopenLastProject(bool enabled);
        [[nodiscard]] lfs::Result<void>
        setAutoSaveOnClose(bool enabled);
        [[nodiscard]] lfs::Result<void>
        setAutosaveIntervalSeconds(
            std::uint64_t seconds);
        [[nodiscard]] bool containsEmbeddedSecrets() const;
        [[nodiscard]] CloseSaveStatus
        beginOrPollCloseSave();
        void resetCloseSaveAttempt();
        [[nodiscard]] std::string
        closeSaveError() const;
        void markApplicationClosePending();
        [[nodiscard]] bool
        isApplicationClosePending() const;
        void setSuppressTrainingAdoption(bool suppress);
        [[nodiscard]] bool
        suppressTrainingAdoption() const;

    private:
        friend class lfs::vis::VisualizerImplResetTest_AutosaveStartsAfterFirstSaveAsWithoutReopen_Test;
        friend class lfs::vis::VisualizerImplResetTest_AutosaveSkipsWhileManualProjectWriteJobIsRunning_Test;
        friend class lfs::vis::VisualizerImplResetTest_RecoveryDeclineKeepsSidecarSuppressesRepeatAndExplicitSaveDeletesIt_Test;
        friend class lfs::vis::VisualizerImplResetTest_RecoveredProjectSwitchDeletesTempOnlyAfterReplacement_Test;
        friend class lfs::vis::VisualizerImplResetTest_FailedNewProjectKeepsRecoveredSessionTemp_Test;
        friend class lfs::vis::VisualizerImplResetTest_RecoveredCloseDeletesTempAfterDocumentTeardown_Test;
        friend class lfs::vis::VisualizerImplResetTest_ProjectWriteSettlementCompletesBeforeNextDocumentWrite_Test;
        friend class lfs::vis::VisualizerImplResetTest_TrainingSnapshotCleanupTerminalizesProjectWrite_Test;
        friend class lfs::vis::VisualizerImplResetTest_TrainingSnapshotPrepareFailureTerminalizesProjectWrite_Test;
        friend class lfs::vis::VisualizerImplResetTest_TrainingSnapshotSupersedeTerminalizesOldAndCompletesNew_Test;
        friend class lfs::vis::VisualizerImplResetTest_TrainingSnapshotCancelTerminalizesBeforeSettlement_Test;
        friend class lfs::vis::VisualizerImplResetTest_FailedAutosaveSettlementAppliesBackoffBeforeRetry_Test;
        friend class lfs::vis::VisualizerImplResetTest_PendingCloseSuppressesBackgroundAutosave_Test;
        friend class lfs::vis::VisualizerImplResetTest_StoppingTrainerBlocksIdleCompactionAndAutosave_Test;
        friend class lfs::vis::VisualizerImplResetTest_SessionSoftDirtyDoesNotPromptOrArmAutosave_Test;
        friend class lfs::vis::VisualizerImplResetTest_SceneEditStillPromptsAndArmsAutosave_Test;
        friend class lfs::vis::VisualizerImplResetTest_ParametersUnchangedRoundTripStaysClean_Test;
        friend class lfs::vis::VisualizerImplResetTest_ParametersValueChangeIsHardDirty_Test;
        friend class lfs::vis::VisualizerImplResetTest_BaselineIdleCheckpointTrainerClosesWithoutTrainingPrompt_Test;
        friend class lfs::vis::VisualizerImplResetTest_ProgressedPausedTrainerStillBlocksCleanClose_Test;
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

        enum class ProjectWritePurpose {
            None,
            Autosave,
            ExplicitSave,
            SaveAs,
            CloseSave,
            Compaction,
            TrainingAutosave,
            TrainingExplicitSave,
        };

        struct DeclinedRecoveryIdentity {
            std::filesystem::path sidecar_path;
            std::uint64_t autosave_sequence = 0;
            lfs::core::Uuid snapshot_uuid;

            friend bool operator==(
                const DeclinedRecoveryIdentity&,
                const DeclinedRecoveryIdentity&) =
                default;
        };

        [[nodiscard]] lfs::Result<void>
        synchronizeDocumentFromViewer();
        [[nodiscard]] lfs::Result<void>
        openMaster(
            const std::filesystem::path& path,
            ProjectSwitchDisposition disposition);
        [[nodiscard]] lfs::Result<void>
        openRecovered(
            const std::filesystem::path& master_path,
            const std::filesystem::path& sidecar_path,
            ProjectSwitchDisposition disposition);
        [[nodiscard]] lfs::Result<void>
        startAutosave();
        [[nodiscard]] lfs::Result<void>
        startDocumentWrite(
            ProjectWritePurpose purpose,
            std::shared_ptr<
                lfs::io::project::ProjectDocument>
                document,
            std::filesystem::path destination,
            lfs::io::project::
                ProjectDocumentSaveOptions options,
            std::optional<
                lfs::io::project::
                    ProjectDocumentAutosaveOptions>
                autosave = std::nullopt);
        [[nodiscard]] lfs::Result<void>
        startCompaction(bool automatic);
        [[nodiscard]] lfs::Result<void>
        startTrainingWrite(
            ProjectWritePurpose purpose,
            std::uint64_t request_id,
            std::filesystem::path master_path,
            std::uint64_t dirty_epoch,
            std::uint64_t scene_serial);
        void queueProjectWriteSettlement(
            JobHandle handle);
        void settleProjectWrite();
        void refreshStorageStats();
        void resetMaintenanceClocks();
        void clearAutosaveFailureBackoff();
        void scheduleAutosaveFailureBackoff();
        [[nodiscard]] bool
        isBackgroundAutosaveSuppressed() const;
        [[nodiscard]] bool
        isTrainingWriteWindowOpen() const;
        void cancelBackgroundAutosaveIfRunning();
        void cleanupRecoverySession();
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
        void stopHydrationThreads();
        void markHydrationFailed(
            std::uint64_t epoch,
            const std::string& detail);
        [[nodiscard]] static std::string
        hydrationName(Hydration state);

        VisualizerImpl& viewer_;
        std::shared_ptr<lfs::io::project::ProjectDocument> document_;
        ProjectLifecycleSettings settings_;
        mutable std::mutex settings_mutex_;
        std::filesystem::path settings_path_;
        std::atomic<std::uint64_t> epoch_{0};
        std::atomic<std::uint64_t> scene_mutation_serial_{0};
        std::atomic<std::uint64_t>
            selection_mutation_serial_{0};
        std::atomic<Hydration> hydration_{Hydration::Empty};
        std::atomic<bool> scene_dirty_{false};
        std::atomic<bool> payload_dirty_{false};
        std::chrono::steady_clock::time_point
            last_autosave_at_;
        std::chrono::steady_clock::time_point
            last_mutation_at_;
        std::chrono::steady_clock::time_point
            next_storage_check_at_;
        std::chrono::steady_clock::time_point
            autosave_retry_not_before_{};
        std::chrono::steady_clock::time_point
            project_open_started_at_{};
        std::chrono::steady_clock::time_point
            hydration_committed_at_{};
        bool project_first_render_pending_ = false;
        std::uint64_t
            autosave_failure_backoff_seconds_ = 0;
        std::uint64_t
            last_autosaved_dirty_epoch_ = 0;
        std::uint64_t
            last_autosaved_scene_serial_ = 0;
        std::uint64_t autosave_sequence_ = 0;
        bool application_close_pending_ = false;
        bool suppress_training_adoption_ = false;
        std::uint64_t
            project_write_autosave_sequence_ = 0;
        std::optional<JobHandle>
            project_write_job_;
        std::optional<JobHandle>
            project_open_job_;
        ProjectWritePurpose
            project_write_purpose_ =
                ProjectWritePurpose::None;
        std::jthread project_write_thread_;
        std::uint64_t
            project_write_dirty_epoch_ = 0;
        std::uint64_t
            project_write_scene_serial_ = 0;
        std::filesystem::path
            project_write_destination_;
        bool project_write_automatic_ = false;
        std::string last_project_write_error_;
        lfs::io::project::ProjectStorageStats
            storage_stats_;
        bool compaction_suggested_ = false;
        bool compaction_suggestion_reported_ =
            false;
        std::optional<std::filesystem::path>
            recovered_master_path_;
        std::optional<std::filesystem::path>
            recovery_session_path_;
        std::optional<
            lfs::io::project::RecoverySession>
            recovery_session_;
        bool recovery_prompt_pending_ = false;
        std::optional<DeclinedRecoveryIdentity>
            declined_recovery_;
        mutable std::mutex
            document_access_mutex_;
        std::optional<ProjectInfo>
            cached_project_info_;
        std::uint64_t
            adopted_training_snapshot_count_ = 0;
        mutable std::mutex thread_mutex_;
        std::vector<std::jthread> hydration_threads_;
        std::atomic<CloseSaveState>
            close_save_state_{CloseSaveState::Idle};
        mutable std::mutex close_save_mutex_;
        std::string close_save_error_;
        std::string hydration_error_;
    };

} // namespace lfs::vis::project
