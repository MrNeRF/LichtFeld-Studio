/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "core/uuid.hpp"
#include "io/project_document.hpp"
#include "io/project_recovery.hpp"
#include "visualizer/core/job_registry.hpp"
#include "visualizer/visualizer.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
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
    class VisualizerImplResetTest_NewProjectClearsRecoveryPromptPendingSoNextOpenProceeds_Test;
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
    class VisualizerImplResetTest_CloseSaveRoutesTrainingSnapshotToLiveDocument_Test;
    class VisualizerImplResetTest_TrainerOwnedSaveTargetsLiveDocumentPath_Test;
    class VisualizerImplResetTest_StartTrainingPreparesProjectAndGrantsSaves_Test;
    class VisualizerImplResetTest_StartConflictSeesDiskCheckpointAfterTrainerReplacement_Test;
    class VisualizerImplResetTest_SaveAsRoutesThroughFinishedTrainer_Test;
    class VisualizerImplResetTest_SaveWhilePausedTrainingRoutesThroughLiveTrainer_Test;
    class VisualizerImplResetTest_SaveWhilePausedNoWorkerTrainerCompletes_Test;
    class VisualizerImplResetTest_SaveWhileStoppingStillBlocksUntilSnapshotPublished_Test;
    class VisualizerImplResetTest_SaveAsWhilePausedTrainingRoutesThroughLiveTrainer_Test;
    class VisualizerImplResetTest_SaveAsRoutesThroughFailedTerminalSnapshotAftermath_Test;
    class VisualizerImplResetTest_InfoSurvivesFailedTerminalSnapshotAftermath_Test;
    class VisualizerImplResetTest_AdoptCompletedTrainingSnapshotSkipsOpenWhenCountersEqual_Test;
    class VisualizerImplResetTest_AdoptedStepBoundaryPublishRebasesAutosaveBase_Test;
    class VisualizerImplResetTest_TrainingAutosaveIsLightOnlyAndRecoversSpecifiedCkpt_Test;
    class VisualizerImplResetTest_TrainingAutosaveWithoutSpecifiedCkptStillWritesLightChapters_Test;
    class VisualizerImplResetTest_CancelExitDuringCloseSaveDoesNotClose_Test;
    class VisualizerImplResetTest_ProjectGetInfoSucceedsDuringCloseSave_Test;
    class VisualizerImplResetTest_ProjectGetInfoSucceedsWithUnboundPausedTrainer_Test;
    class VisualizerImplResetTest_FailedSaveAsAndExitResetsCloseLatches_Test;
    class VisualizerImplResetTest_ForceExitWhileSavingDoesNotWaitForSettlement_Test;
    class VisualizerImplResetTest_OpenAndNewProjectClearSuppressAdoption_Test;
    class VisualizerImplResetTest_InfoDoesNotMutateDocumentUntilExplicitSync_Test;
    class VisualizerImplResetTest_BoundCheckpointIterationCacheSkipsHeaderWhenWarm_Test;
    class VisualizerImplResetTest_ForceExitDiscardDeletesAutosaveSidecarOnTeardown_Test;
    class VisualizerImplResetTest_EmergencyForceExitKeepsAutosaveSidecarOnTeardown_Test;
    class VisualizerImplResetTest_DiscardSwitchDeletesOldProjectAutosaveSidecar_Test;
    class VisualizerImplResetTest_DiscardSamePathReopenSkipsRecoveryPromptAndDeletesSidecar_Test;
    class VisualizerImplResetTest_DirtyRequireCleanSwitchKeepsAutosaveSidecar_Test;
    class VisualizerImplResetTest_NewProjectDiscardDeletesAutosaveSidecar_Test;
    class VisualizerImplResetTest_StartupOffersRecoveryAfterUncleanShutdown_Test;
    class VisualizerImplResetTest_StartupWithCleanLastSessionLeavesBlankSession_Test;
    class VisualizerImplResetTest_UntitledDirtySessionAutosavesToScratch_Test;
    class VisualizerImplResetTest_BlankUntitledSessionUpdateMaintenanceWritesNoScratch_Test;
    class VisualizerImplResetTest_DirtyUntitledSessionUpdateMaintenanceWritesScratch_Test;
    class VisualizerImplResetTest_DirtyUntitledSessionUpdateMaintenanceWaitsForAutosaveQuietPeriod_Test;
    class VisualizerImplResetTest_SaveAsMigratesScratchAutosaveToSidecar_Test;
    class VisualizerImplResetTest_RecoveryDismissalPersistsAndNewerCandidateIsOffered_Test;
    class VisualizerImplResetTest_RecoverThenCleanQuitDoesNotReoffer_Test;
    class VisualizerImplResetTest_RecoverThenDiscardExitRemovesMasterSidecar_Test;
    class VisualizerImplResetTest_RecoverThenCrashStillOffersRecovery_Test;
    class VisualizerImplResetTest_StartupOffersScratchRecoveryAsUntitled_Test;
    class VisualizerImplResetTest_StartupSweepsEmptyScratchAndDoesNotOffer_Test;
} // namespace lfs::vis

namespace lfs::vis::project {

    struct ProjectMruEntry {
        lfs::core::Uuid project_uuid;
        std::filesystem::path last_known_path;

        friend bool operator==(const ProjectMruEntry&,
                               const ProjectMruEntry&) = default;
    };

    struct DismissedRecoveryEntry {
        std::filesystem::path sidecar_path;
        std::uint64_t autosave_sequence = 0;
        lfs::core::Uuid commit_uuid;

        friend bool operator==(const DismissedRecoveryEntry&,
                               const DismissedRecoveryEntry&) = default;
    };

    struct ProjectLifecycleSettings {
        bool reopen_last_project = true;
        bool auto_save_on_close = false;
        std::uint64_t autosave_interval_seconds = 5 * 60;
        std::uint64_t autosave_dirty_epoch_threshold = 20;
        std::uint64_t autosave_quiet_seconds = 2;
        std::uint64_t compaction_idle_seconds = 30;
        std::vector<ProjectMruEntry> mru;
        std::vector<DismissedRecoveryEntry> dismissed_recovery;

        friend bool operator==(const ProjectLifecycleSettings&,
                               const ProjectLifecycleSettings&) = default;
    };

    [[nodiscard]] LFS_VIS_API lfs::Result<ProjectLifecycleSettings>
    loadProjectLifecycleSettings(const std::filesystem::path& path);
    [[nodiscard]] LFS_VIS_API lfs::Result<void>
    saveProjectLifecycleSettings(
        const std::filesystem::path& path,
        const ProjectLifecycleSettings& settings);
    [[nodiscard]] LFS_VIS_API std::filesystem::path
    resolveProjectMruPath(const std::filesystem::path& path);
    LFS_VIS_API void pruneMissingMruEntries(
        ProjectLifecycleSettings& settings);
    LFS_VIS_API void rememberProject(
        ProjectLifecycleSettings& settings,
        const lfs::core::Uuid& project_uuid,
        const std::filesystem::path& path);

    class LFS_VIS_API ProjectLifecycle {
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
               bool regenerate_preview,
               bool allow_existing_destination_replacement = false);
        [[nodiscard]] lfs::Result<void>
        compact();
        [[nodiscard]] lfs::Result<void>
        newProject(
            ProjectSwitchDisposition disposition =
                ProjectSwitchDisposition::RequireClean);
        [[nodiscard]] bool isDirty();
        [[nodiscard]] bool hasSourcePath() const;
        [[nodiscard]] lfs::Result<ProjectInfo> info();
        [[nodiscard]] ProjectWritePoll pollWrite();
        void joinPendingWrite();
        [[nodiscard]] ProjectMenuInfo menuInfo() const;
        [[nodiscard]] lfs::Result<void>
        preflightSwitch(
            ProjectSwitchDisposition disposition,
            bool allow_active_training = false);

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
        clearRecentProjects();
        [[nodiscard]] lfs::Result<void>
        removeRecentProject(
            const std::filesystem::path& path);
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
        void markCloseDiscardRequested();
        [[nodiscard]] bool
        isApplicationClosePending() const;
        void setSuppressTrainingAdoption(bool suppress);
        void bindTrainerSnapshotTarget(
            std::optional<std::filesystem::path> destination =
                std::nullopt,
            bool allow_existing_destination_replacement = false);
        [[nodiscard]] lfs::Result<void>
        prepareTrainingStartProject();
        // Returns the blocking conflict a fresh training start would overwrite, if any:
        // the bound checkpoint iteration of the open project (in memory or on the
        // titled master), or -1 when an untitled session's default destination file
        // already exists or an existing master is unreadable (contents unknown).
        [[nodiscard]] std::optional<int>
        trainingStartOverwriteConflict();
        [[nodiscard]] lfs::Result<void>
        prepareForEditModeTransition();

        [[nodiscard]] std::optional<std::filesystem::path>
        pendingDatasetRelocationPath() const;
        bool relocateProjectDataset(
            const std::filesystem::path& new_root,
            std::string* error_message = nullptr);

    private:
        friend class lfs::vis::VisualizerImplResetTest_AutosaveStartsAfterFirstSaveAsWithoutReopen_Test;
        friend class lfs::vis::VisualizerImplResetTest_AutosaveSkipsWhileManualProjectWriteJobIsRunning_Test;
        friend class lfs::vis::VisualizerImplResetTest_RecoveryDeclineKeepsSidecarSuppressesRepeatAndExplicitSaveDeletesIt_Test;
        friend class lfs::vis::VisualizerImplResetTest_NewProjectClearsRecoveryPromptPendingSoNextOpenProceeds_Test;
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
        friend class lfs::vis::VisualizerImplResetTest_CloseSaveRoutesTrainingSnapshotToLiveDocument_Test;
        friend class lfs::vis::VisualizerImplResetTest_TrainerOwnedSaveTargetsLiveDocumentPath_Test;
        friend class lfs::vis::VisualizerImplResetTest_StartTrainingPreparesProjectAndGrantsSaves_Test;
        friend class lfs::vis::VisualizerImplResetTest_StartConflictSeesDiskCheckpointAfterTrainerReplacement_Test;
        friend class lfs::vis::VisualizerImplResetTest_SaveAsRoutesThroughFinishedTrainer_Test;
        friend class lfs::vis::VisualizerImplResetTest_SaveWhilePausedTrainingRoutesThroughLiveTrainer_Test;
        friend class lfs::vis::VisualizerImplResetTest_SaveWhilePausedNoWorkerTrainerCompletes_Test;
        friend class lfs::vis::VisualizerImplResetTest_SaveWhileStoppingStillBlocksUntilSnapshotPublished_Test;
        friend class lfs::vis::VisualizerImplResetTest_SaveAsWhilePausedTrainingRoutesThroughLiveTrainer_Test;
        friend class lfs::vis::VisualizerImplResetTest_SaveAsRoutesThroughFailedTerminalSnapshotAftermath_Test;
        friend class lfs::vis::VisualizerImplResetTest_InfoSurvivesFailedTerminalSnapshotAftermath_Test;
        friend class lfs::vis::VisualizerImplResetTest_AdoptCompletedTrainingSnapshotSkipsOpenWhenCountersEqual_Test;
        friend class lfs::vis::VisualizerImplResetTest_AdoptedStepBoundaryPublishRebasesAutosaveBase_Test;
        friend class lfs::vis::VisualizerImplResetTest_TrainingAutosaveIsLightOnlyAndRecoversSpecifiedCkpt_Test;
        friend class lfs::vis::VisualizerImplResetTest_TrainingAutosaveWithoutSpecifiedCkptStillWritesLightChapters_Test;
        friend class lfs::vis::VisualizerImplResetTest_CancelExitDuringCloseSaveDoesNotClose_Test;
        friend class lfs::vis::VisualizerImplResetTest_ProjectGetInfoSucceedsDuringCloseSave_Test;
        friend class lfs::vis::VisualizerImplResetTest_ProjectGetInfoSucceedsWithUnboundPausedTrainer_Test;
        friend class lfs::vis::VisualizerImplResetTest_FailedSaveAsAndExitResetsCloseLatches_Test;
        friend class lfs::vis::VisualizerImplResetTest_ForceExitWhileSavingDoesNotWaitForSettlement_Test;
        friend class lfs::vis::VisualizerImplResetTest_OpenAndNewProjectClearSuppressAdoption_Test;
        friend class lfs::vis::VisualizerImplResetTest_InfoDoesNotMutateDocumentUntilExplicitSync_Test;
        friend class lfs::vis::VisualizerImplResetTest_BoundCheckpointIterationCacheSkipsHeaderWhenWarm_Test;
        friend class lfs::vis::VisualizerImplResetTest_ForceExitDiscardDeletesAutosaveSidecarOnTeardown_Test;
        friend class lfs::vis::VisualizerImplResetTest_EmergencyForceExitKeepsAutosaveSidecarOnTeardown_Test;
        friend class lfs::vis::VisualizerImplResetTest_DiscardSwitchDeletesOldProjectAutosaveSidecar_Test;
        friend class lfs::vis::VisualizerImplResetTest_DiscardSamePathReopenSkipsRecoveryPromptAndDeletesSidecar_Test;
        friend class lfs::vis::VisualizerImplResetTest_DirtyRequireCleanSwitchKeepsAutosaveSidecar_Test;
        friend class lfs::vis::VisualizerImplResetTest_NewProjectDiscardDeletesAutosaveSidecar_Test;
        friend class lfs::vis::VisualizerImplResetTest_StartupOffersRecoveryAfterUncleanShutdown_Test;
        friend class lfs::vis::VisualizerImplResetTest_StartupWithCleanLastSessionLeavesBlankSession_Test;
        friend class lfs::vis::VisualizerImplResetTest_UntitledDirtySessionAutosavesToScratch_Test;
        friend class lfs::vis::VisualizerImplResetTest_BlankUntitledSessionUpdateMaintenanceWritesNoScratch_Test;
        friend class lfs::vis::VisualizerImplResetTest_DirtyUntitledSessionUpdateMaintenanceWritesScratch_Test;
        friend class lfs::vis::VisualizerImplResetTest_DirtyUntitledSessionUpdateMaintenanceWaitsForAutosaveQuietPeriod_Test;
        friend class lfs::vis::VisualizerImplResetTest_SaveAsMigratesScratchAutosaveToSidecar_Test;
        friend class lfs::vis::VisualizerImplResetTest_RecoveryDismissalPersistsAndNewerCandidateIsOffered_Test;
        friend class lfs::vis::VisualizerImplResetTest_RecoverThenCleanQuitDoesNotReoffer_Test;
        friend class lfs::vis::VisualizerImplResetTest_RecoverThenDiscardExitRemovesMasterSidecar_Test;
        friend class lfs::vis::VisualizerImplResetTest_RecoverThenCrashStillOffersRecovery_Test;
        friend class lfs::vis::VisualizerImplResetTest_StartupOffersScratchRecoveryAsUntitled_Test;
        friend class lfs::vis::VisualizerImplResetTest_StartupSweepsEmptyScratchAndDoesNotOffer_Test;
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
            TrainingCloseSave,
        };

        using DeclinedRecoveryIdentity = DismissedRecoveryEntry;

        struct RecoveryCandidate {
            std::filesystem::path master_path;
            std::filesystem::path selected_path;
            std::uint64_t autosave_sequence = 0;
            lfs::core::Uuid commit_uuid;
            lfs::core::Uuid snapshot_uuid;
            std::uint64_t wallclock_unix_ns = 0;
            bool untitled_scratch = false;
        };

        enum class DocumentSyncMode {
            Default,
            LightTrainingAutosave,
        };

        void offerStartupCrashRecovery();
        [[nodiscard]] std::optional<RecoveryCandidate>
        selectStartupRecoveryCandidate();
        void enqueueRecoveryPrompt(
            RecoveryCandidate candidate,
            ProjectSwitchDisposition disposition,
            std::uint64_t previous_autosave_sequence);
        void handleRecoverySkip(
            const RecoveryCandidate& candidate);
        [[nodiscard]] bool isRecoveryDismissed(
            const DeclinedRecoveryIdentity& identity) const;
        void persistRecoveryDismissal(
            const DeclinedRecoveryIdentity& identity);
        [[nodiscard]] lfs::Result<void>
        openScratchRecovered(
            const std::filesystem::path& scratch_path,
            ProjectSwitchDisposition disposition);
        [[nodiscard]] std::filesystem::path
        scratchAutosaveDirectory() const;
        void removeScratchAutosave();
        [[nodiscard]] bool isBlankUntitledSession() const;
        [[nodiscard]] lfs::Result<void>
        ensureScratchAutosaveBinding();
        [[nodiscard]] lfs::Result<void>
        lockScratchAutosave();
        [[nodiscard]] lfs::Result<void>
        synchronizeDocumentFromViewer();
        [[nodiscard]] lfs::Result<void>
        synchronizeDocumentFromViewer(DocumentSyncMode mode);
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
        [[nodiscard]] lfs::Result<void>
        startLiveTrainingSnapshotWrite(
            ProjectWritePurpose purpose,
            bool regenerate_preview);
        [[nodiscard]] bool
        isTrainingCheckpointStale() const;
        [[nodiscard]] bool
        canFlushFinishedTrainerSnapshot() const;
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
        [[nodiscard]] lfs::Result<void>
        waitOutBackgroundAutosaveForExplicitSave();
        void cleanupRecoverySession();
        void removeDiscardedAutosaveArtifacts(
            const std::filesystem::path& master);
        [[nodiscard]] lfs::Result<void>
        adoptCompletedTrainingSnapshot(
            bool allow_during_application_close = false);
        [[nodiscard]] lfs::Result<void>
        adoptSettledTrainerPublishOntoCurrentMaster();
        [[nodiscard]] lfs::Result<std::vector<std::byte>>
        capturePreviewPng() const;
        [[nodiscard]] lfs::Result<void>
        launchHydration(
            std::shared_ptr<lfs::io::project::ProjectDocument> document,
            std::uint64_t epoch,
            std::uint64_t selection_mutation_serial,
            std::vector<lfs::core::Uuid> selected_node_uuids,
            std::uint64_t restore_ticket);
        [[nodiscard]] lfs::Result<void>
        persistSettings();
        void stopHydrationThreads();
        void markHydrationFailed(
            std::uint64_t epoch,
            const std::string& detail);
        [[nodiscard]] static std::string
        hydrationName(Hydration state);

        struct PendingDatasetRelocation {
            std::filesystem::path missing_path;
            std::uint64_t open_epoch = 0;
            std::function<void(const std::filesystem::path&)>
                retry;
        };

        void tryInstallTrainerFromHydratedProject(
            SceneManager& scene_manager,
            lfs::io::project::ProjectDocument& document,
            const lfs::io::project::ProjectDocumentHydrationReport&
                report);
        void beginPendingDatasetRelocation(
            std::filesystem::path missing_path,
            std::function<void(const std::filesystem::path&)>
                retry);
        void armMissingDatasetReload(
            std::filesystem::path missing_path,
            std::filesystem::path output_path);
        void armMissingCheckpointDataset(
            std::filesystem::path missing_path,
            lfs::core::Uuid checkpoint_uuid,
            const lfs::core::param::TrainingParameters&
                ckpt_params,
            int expected_iteration);
        void cancelPendingDatasetRelocation(
            std::uint64_t epoch);
        void enqueueMissingDatasetDialog(
            std::uint64_t epoch);
        void enqueueInvalidDatasetDialog(
            std::uint64_t epoch,
            const std::filesystem::path& chosen_path,
            const std::string& detail);
        void handleLocateDatasetPicker(
            std::uint64_t epoch);
        [[nodiscard]] bool isDatasetRelocationCurrent(
            std::uint64_t epoch) const;

        VisualizerImpl& viewer_;
        std::shared_ptr<lfs::io::project::ProjectDocument> document_;
        ProjectLifecycleSettings settings_;
        mutable std::mutex settings_mutex_;
        std::filesystem::path settings_path_;
        std::filesystem::path recovery_directory_;
        bool settings_persistence_enabled_ = true;
        std::atomic<std::uint64_t> epoch_{0};
        std::atomic<std::uint64_t> scene_mutation_serial_{0};
        std::uint64_t active_restore_ticket_ = 0;
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
        bool close_discard_requested_ = false;
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
        std::uint64_t
            project_write_parameter_serial_ = 0;
        std::filesystem::path
            project_write_destination_;
        bool project_write_automatic_ = false;
        std::string last_project_write_error_;
        std::optional<lfs::ErrorCode> last_project_write_error_code_;
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
        std::uint64_t recovery_prompt_generation_ = 0;
        std::optional<RecoveryCandidate>
            pending_recovery_candidate_;
        std::optional<DeclinedRecoveryIdentity>
            declined_recovery_;
        std::optional<std::filesystem::path>
            scratch_autosave_path_;
        std::optional<
            lfs::io::project::WriterLockLease>
            scratch_lock_;
        mutable std::mutex
            document_access_mutex_;
        std::optional<ProjectInfo>
            cached_project_info_;
        std::uint64_t
            adopted_training_snapshot_count_ = 0;
        std::string
            last_unadoptable_training_snapshot_warning_;
        mutable std::optional<int>
            cached_bound_checkpoint_iteration_;
        mutable std::mutex thread_mutex_;
        std::vector<std::jthread> hydration_threads_;
        std::atomic<CloseSaveState>
            close_save_state_{CloseSaveState::Idle};
        mutable std::mutex close_save_mutex_;
        std::string close_save_error_;
        std::string hydration_error_;
        std::optional<PendingDatasetRelocation>
            pending_dataset_relocation_;
    };

} // namespace lfs::vis::project
