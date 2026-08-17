/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/editor_context.hpp"
#include "core/error_codes.hpp"
#include "core/frame_state_machine.hpp"
#include "core/job_registry.hpp"
#include "core/main_loop.hpp"
#include "core/parameter_manager.hpp"
#include "core/parameters.hpp"
#include "gui/gui_manager.hpp"
#include "input/input_controller.hpp"
#include "internal/viewport.hpp"
#include "project/project_lifecycle.hpp"
#include "project/session_state.hpp"
#include "rendering/rendering.hpp"
#include "rendering/rendering_manager.hpp"
#include "scene/scene_manager.hpp"
#include "tools/tool_base.hpp"
#include "training/training_manager.hpp"
#include "visualizer/visualizer.hpp"
#include "window/window_manager.hpp"
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

struct SDL_Window;

namespace lfs::python {
    struct SequencerUIStateData;
} // namespace lfs::python

namespace lfs::vis {
    class SceneManager;
} // namespace lfs::vis

namespace lfs::vis {
    class DataLoadingService;

    namespace tools {
        class AlignTool;
        class SelectionTool;
    } // namespace tools

    class LFS_VIS_API VisualizerImpl : public Visualizer {
        friend class gui::GuiManager;

    public:
        explicit VisualizerImpl(const ViewerOptions& options);
        ~VisualizerImpl() override;

        void run() override;
        void setParameters(const lfs::core::param::TrainingParameters& params) override;
        std::expected<void, std::string> loadPLY(const std::filesystem::path& path) override;
        std::expected<void, std::string> addSplatFile(const std::filesystem::path& path) override;
        std::expected<void, std::string> loadDataset(const std::filesystem::path& path) override;
        std::expected<void, std::string> loadCheckpointForTraining(const std::filesystem::path& path) override;
        void consolidateModels() override;
        [[nodiscard]] std::expected<void, std::string> clearScene() override;
        core::Scene& getScene() override { return scene_manager_->getScene(); }
        [[nodiscard]] const core::Scene& getScene() const {
            return scene_manager_->getScene();
        }
        bool postWork(WorkItem work) override;
        bool postRenderWork(WorkItem work);
        [[nodiscard]] bool isOnViewerThread() const override {
            return std::this_thread::get_id() == viewer_thread_id_;
        }
        [[nodiscard]] bool acceptsPostedWork() const override;
        [[nodiscard]] bool isProcessingRenderWork() const {
            assert(isOnViewerThread());
            return processing_render_work_;
        }
        void setShutdownRequestedCallback(std::function<void()> callback) override;
        std::expected<void, std::string> startTraining() override;
        lfs::Result<void>
        projectSave(bool regenerate_preview = true) override;
        lfs::Result<void>
        projectSaveAs(const std::filesystem::path& path,
                      bool regenerate_preview = true) override;
        lfs::Result<void> projectSaveAsExplicit(
            const std::filesystem::path& path,
            bool regenerate_preview = true);
        lfs::Result<ProjectOpenOutcome>
        projectOpen(
            const std::filesystem::path& path,
            ProjectSwitchDisposition disposition =
                ProjectSwitchDisposition::RequireClean) override;
        lfs::Result<void>
        projectCompact() override;
        lfs::Result<bool>
        projectIsDirty() override;
        lfs::Result<bool>
        projectHasPath() override;
        lfs::Result<ProjectInfo>
        projectGetInfo() override;
        lfs::Result<ProjectWritePoll>
        projectPollWrite() override;
        lfs::Result<ProjectMenuInfo>
        projectGetMenuInfo() override;
        lfs::Result<void>
        projectClearRecentFiles() override;
        [[nodiscard]] bool projectContainsEmbeddedSecrets() const;

        // Getters for GUI (delegating to state manager)
        lfs::training::Trainer* getTrainer() const { return trainer_manager_->getTrainer(); }

        // Component access
        TrainerManager* getTrainerManager() { return trainer_manager_.get(); }
        const TrainerManager* getTrainerManager() const {
            return trainer_manager_.get();
        }
        SceneManager* getSceneManager() override { return scene_manager_.get(); }
        SDL_Window* getWindow() const { return window_manager_->getWindow(); }
        WindowManager* getWindowManager() { return window_manager_.get(); }
        const WindowManager* getWindowManager() const {
            return window_manager_.get();
        }
        RenderingManager* getRenderingManager() override { return rendering_manager_.get(); }
        const RenderingManager* getRenderingManager() const {
            return rendering_manager_.get();
        }
        gui::GuiManager* getGuiManager() { return gui_manager_.get(); }
        const gui::GuiManager* getGuiManager() const {
            return gui_manager_.get();
        }
        [[nodiscard]] JobRegistry& jobs() noexcept {
            return job_registry_;
        }
        [[nodiscard]] const JobRegistry& jobs() const noexcept {
            return job_registry_;
        }
        const Viewport& getViewport() const { return viewport_; }
        Viewport& getViewport() { return viewport_; }
        [[nodiscard]] lfs::Result<
            lfs::io::project::ProjectSessionChapters>
        captureProjectSession(
            lfs::io::project::ReferencesChapter*
                references = nullptr,
            const std::filesystem::path& project_root =
                {}) const;
        [[nodiscard]] project::GuiSessionRestoreTicket
        stagePreparedProjectSessionRestore(
            project::PreparedGuiSessionRestore prepared);
        [[nodiscard]] bool
        isProjectSessionRestorePending() const noexcept {
            return gui_session_restore_.hasPending() ||
                   pending_project_tools_restore_.has_value();
        }
        void tryApplyProjectSessionTools(
            project::GuiSessionRestoreTicket ticket);
        void noteHydrationTerminalForRestoreTicket(
            project::GuiSessionRestoreTicket ticket);
        void noteGuiSessionRestoreOwnerReady(
            std::uint64_t panels_registration_revision);
        void deactivateProjectTools();
        void bindTrainerProjectSnapshotTarget();
        // FPS monitoring
        [[nodiscard]] float getAverageFPS() const {
            return rendering_manager_ ? rendering_manager_->getAverageFPS() : 0.0f;
        }

        // Antialiasing state
        bool isAntiAliasingEnabled() const {
            return rendering_manager_ ? rendering_manager_->getSettings().antialiasing : false;
        }

        tools::AlignTool* getAlignTool() {
            return align_tool_.get();
        }

        const tools::AlignTool* getAlignTool() const {
            return align_tool_.get();
        }

        tools::SelectionTool* getSelectionTool() {
            return selection_tool_.get();
        }

        const tools::SelectionTool* getSelectionTool() const {
            return selection_tool_.get();
        }

        InputController* getInputController() {
            return input_controller_.get();
        }
        const InputController* getInputController() const {
            return input_controller_.get();
        }

        DataLoadingService* getDataLoader() {
            return data_loader_.get();
        }
        ParameterManager* getParameterManager() {
            return parameter_manager_.get();
        }
        const ParameterManager* getParameterManager() const {
            return parameter_manager_.get();
        }

        EditorContext& getEditorContext() { return editor_context_; }
        const EditorContext& getEditorContext() const { return editor_context_; }

        // Undo/Redo
        void undo();
        void redo();

        // GUI manager
        std::unique_ptr<gui::GuiManager> gui_manager_;
        JobRegistry job_registry_;
        friend class gui::GuiManager;
        friend class project::ProjectLifecycle;
        friend class VisualizerImplResetTest_ResetTrainingPreservesExplicitInitPath_Test;
        friend class VisualizerImplResetTest_DirtyProjectSwitchRequiresExplicitDiscardAuthorization_Test;
        friend class VisualizerImplResetTest_NewProjectDirtyGateRunsBelowEveryCommandEntry_Test;
        friend class VisualizerImplResetTest_FileExitWithDefaultSettingsNeedsPrompt_Test;
        friend class VisualizerImplResetTest_FileExitRoutesThroughCloseSaveWhenAutoSaveOnCloseEnabled_Test;
        friend class VisualizerImplResetTest_CloseSavePendingActionSkipsPreviewRegen_Test;
        friend class VisualizerImplResetTest_SaveAsAndExitContinuesAfterProjectWriteCompletes_Test;
        friend class VisualizerImplResetTest_SaveAsAndExitClearsSelectionDirtyBaseline_Test;
        friend class VisualizerImplResetTest_SaveAsAndExitClearsParameterDirtyBaseline_Test;
        friend class VisualizerImplResetTest_DialogSaveAsReplacesExistingFirstSave_Test;
        friend class VisualizerImplResetTest_McpExplicitSaveAsReplacesExistingFirstSave_Test;
        friend class VisualizerImplResetTest_McpImplicitSaveAsReportsTypedFailure_Test;
        friend class VisualizerImplResetTest_CancelExitAndNextWindowAttemptRecoverFromFailedCloseSave_Test;
        friend class VisualizerImplResetTest_RecoveryDeclineKeepsSidecarSuppressesRepeatAndExplicitSaveDeletesIt_Test;
        friend class VisualizerImplResetTest_NewProjectClearsRecoveryPromptPendingSoNextOpenProceeds_Test;
        friend class VisualizerImplResetTest_RecoveredPublishUsesRecoveredCommitKind_Test;
        friend class VisualizerImplResetTest_AutosaveStartsAfterFirstSaveAsWithoutReopen_Test;
        friend class VisualizerImplResetTest_AutosaveSkipsWhileManualProjectWriteJobIsRunning_Test;
        friend class VisualizerImplResetTest_RecoveredProjectSwitchDeletesTempOnlyAfterReplacement_Test;
        friend class VisualizerImplResetTest_FailedNewProjectKeepsRecoveredSessionTemp_Test;
        friend class VisualizerImplResetTest_RecoveredCloseDeletesTempAfterDocumentTeardown_Test;
        friend class VisualizerImplResetTest_ProjectWriteSettlementCompletesBeforeNextDocumentWrite_Test;
        friend class VisualizerImplResetTest_TrainingSnapshotCleanupTerminalizesProjectWrite_Test;
        friend class VisualizerImplResetTest_TrainingSnapshotPrepareFailureTerminalizesProjectWrite_Test;
        friend class VisualizerImplResetTest_TrainingSnapshotSupersedeTerminalizesOldAndCompletesNew_Test;
        friend class VisualizerImplResetTest_TrainingSnapshotCancelTerminalizesBeforeSettlement_Test;
        friend class VisualizerImplResetTest_FailedAutosaveSettlementAppliesBackoffBeforeRetry_Test;
        friend class VisualizerImplResetTest_PendingCloseSuppressesBackgroundAutosave_Test;
        friend class VisualizerImplResetTest_StoppingTrainerBlocksIdleCompactionAndAutosave_Test;
        friend class VisualizerImplResetTest_SessionSoftDirtyDoesNotPromptOrArmAutosave_Test;
        friend class VisualizerImplResetTest_SceneEditStillPromptsAndArmsAutosave_Test;
        friend class VisualizerImplResetTest_ParametersUnchangedRoundTripStaysClean_Test;
        friend class VisualizerImplResetTest_ParametersValueChangeIsHardDirty_Test;
        friend class VisualizerImplResetTest_BaselineIdleCheckpointTrainerClosesWithoutTrainingPrompt_Test;
        friend class VisualizerImplResetTest_ProgressedPausedTrainerStillBlocksCleanClose_Test;
        friend class VisualizerImplResetTest_CloseSaveRoutesTrainingSnapshotToLiveDocument_Test;
        friend class VisualizerImplResetTest_TrainerOwnedSaveTargetsLiveDocumentPath_Test;
        friend class VisualizerImplResetTest_SaveAsRoutesThroughFinishedTrainer_Test;
        friend class VisualizerImplResetTest_SaveWhilePausedTrainingRoutesThroughLiveTrainer_Test;
        friend class VisualizerImplResetTest_SaveWhileStoppingStillBlocksUntilSnapshotPublished_Test;
        friend class VisualizerImplResetTest_SaveAsWhilePausedTrainingRoutesThroughLiveTrainer_Test;
        friend class VisualizerImplResetTest_SaveAsRoutesThroughFailedTerminalSnapshotAftermath_Test;
        friend class VisualizerImplResetTest_InfoSurvivesFailedTerminalSnapshotAftermath_Test;
        friend class VisualizerImplResetTest_AdoptCompletedTrainingSnapshotSkipsOpenWhenCountersEqual_Test;
        friend class VisualizerImplResetTest_TrainingAutosaveIsLightOnlyAndRecoversSpecifiedCkpt_Test;
        friend class VisualizerImplResetTest_TrainingAutosaveWithoutSpecifiedCkptStillWritesLightChapters_Test;
        friend class VisualizerImplResetTest_CancelExitDuringCloseSaveDoesNotClose_Test;
        friend class VisualizerImplResetTest_ProjectGetInfoSucceedsDuringCloseSave_Test;
        friend class VisualizerImplResetTest_ProjectGetInfoSucceedsWithUnboundPausedTrainer_Test;
        friend class VisualizerImplResetTest_FailedSaveAsAndExitResetsCloseLatches_Test;
        friend class VisualizerImplResetTest_ForceExitWhileSavingDoesNotWaitForSettlement_Test;
        friend class VisualizerImplResetTest_ForceExitWhileStoppingArmsWatcher_Test;
        friend class VisualizerImplResetTest_OpenAndNewProjectClearSuppressAdoption_Test;
        friend class VisualizerImplResetTest_ExitConfirmationPendingOwnedByGuiManager_Test;
        friend class VisualizerImplResetTest_StopSaveAndExitBindsUntitledDestinationBeforeStop_Test;
        friend class VisualizerImplResetTest_InfoDoesNotMutateDocumentUntilExplicitSync_Test;
        friend class VisualizerImplResetTest_BoundCheckpointIterationCacheSkipsHeaderWhenWarm_Test;
        friend class VisualizerImplResetTest_SelectedGaussiansAndSelectionToolSurviveSaveAndReopen_Test;
        friend class VisualizerImplResetTest_DatasetProjectWithoutCheckpointReloadsTrainer_Test;
        friend class VisualizerImplResetTest_DatasetProjectWithoutReferenceIsNotRecoveredFromContainingDirectory_Test;
        friend class VisualizerImplResetTest_NonDatasetProjectInsideDatasetRootIsNotReimported_Test;
        friend class VisualizerImplResetTest_OpeningAnotherProjectCancelsPendingDatasetRestoreImport_Test;
        friend class VisualizerImplResetTest_NonDatasetSaveDoesNotBindStaleDatasetPath_Test;
        friend class VisualizerImplResetTest_TrainingCheckpointReopenRestoresPausedResumableState_Test;
        friend class VisualizerImplResetTest_EditModeSaveDropsFormerTrainingCheckpoint_Test;
        friend class VisualizerImplResetTest_ReopenedTwoSplatProjectBuildsExternalCombinedModel_Test;
        friend class VisualizerImplResetTest_ForceExitDiscardDeletesAutosaveSidecarOnTeardown_Test;
        friend class VisualizerImplResetTest_EmergencyForceExitKeepsAutosaveSidecarOnTeardown_Test;
        friend class VisualizerImplResetTest_DiscardSwitchDeletesOldProjectAutosaveSidecar_Test;
        friend class VisualizerImplResetTest_DiscardSamePathReopenSkipsRecoveryPromptAndDeletesSidecar_Test;
        friend class VisualizerImplResetTest_DirtyRequireCleanSwitchKeepsAutosaveSidecar_Test;
        friend class VisualizerImplResetTest_NewProjectDiscardDeletesAutosaveSidecar_Test;
        friend class VisualizerImplResetTest_StartupOffersRecoveryAfterUncleanShutdown_Test;
        friend class VisualizerImplResetTest_StartupWithCleanLastSessionLeavesBlankSession_Test;

        // Allow ToolContext to access GUI manager for logging
        friend class ToolContext;

    private:
        lfs::Result<void> projectSaveAsFromDialog(
            const std::filesystem::path& path,
            bool regenerate_preview);
        void abandonSaveAndExitAttempt();
        void armStopSaveAndExit(
            std::optional<std::filesystem::path>
                untitled_destination = std::nullopt);
        void completeSaveAsAndExit(
            const std::filesystem::path& path);
        void armForceExitCompletionWatcher();
        void stopForceExitCompletionWatcher();

        // Main loop callbacks
        bool initialize();
        void update();
        void render();
        void shutdown();
        bool allowclose();
        void wakeMainLoop() const;

        // Frame exception boundary. Contains an OOM or other error escaping a
        // frame so the loop never aborts: OOM triggers one render-safe pressure
        // episode, other errors escalate under a rate limit.
        void handleFrameException(std::exception_ptr eptr) noexcept;
        void onFrameCompleted() noexcept;
        // Executes the FrameStateMachine's publish effects on the UI thread
        // (pressure toast, OOM-paused modal, renderer-dead modal + OS dialog).
        void applyFrameStateEffects(const FrameStateMachine::Effects& effects) noexcept;
        // Internal-terminal modal carrying the caught fault's code and developer
        // detail; renderer-dead modal (+ AMB-P3-1 OS dialog) for a lost/stalled
        // renderer that can no longer present its own obituary.
        void publishRendererInternalModal(lfs::ErrorCode code, std::string detail) noexcept;
        void publishRendererInternalModal(const lfs::Error& error) noexcept;
        std::vector<lfs::ErrorAction> rendererInternalActions();
        void publishRendererDeadModal(RendererTerminalState cause) noexcept;
        void logRateLimitedFrameError(const std::exception& e) noexcept;

        // Event system
        void setupEventHandlers();
        void setupComponentConnections();
        void handleTrainingCompleted(const lfs::core::events::state::TrainingCompleted& event);
        void handleLoadConfigFile(const std::filesystem::path& path);
        void handleNewProject(
            ProjectSwitchDisposition disposition);
        void performNewProject(
            ProjectSwitchDisposition disposition);
        void schedulePendingTrainingAction();
        void performPendingTrainingAction();
        void requestApplicationClose();
        void performReset();
        void resetProjectState(bool reset_panel_registry = true);
        void tryApplyProjectSessionRestore();

        // Tool initialization
        void initializeTools();

        // Subsystem wiring
        void setupPythonBridge();
        void setupViewContextBridge();
        void beginShutdown(std::string_view reason = "Viewer is shutting down");
        void processRenderWorkQueue();
        [[nodiscard]] bool hasPendingRenderWork();
        [[nodiscard]] bool inputFrameRequestsRender() const;

        struct FrameDemand {
            bool viewport_export_locked = false;
            bool scene_dirty = false;
            bool continuous_input = false;
            bool python_animation = false;
            bool python_overlay = false;
            bool python_redraw = false;
            bool gui_animation = false;
            bool input_event = false;
            bool posted_work = false;
            bool render_work = false;
            bool store_dirty = false;
            bool swapchain_resize_pending = false;
            bool swapchain_resize_ready = false;
            bool window_resize_paint_pending = false;
            bool viewport_resize_deferring = false;
            bool viewport_resize_settle_ready = false;

            [[nodiscard]] bool shouldRenderFrame() const {
                return viewport_export_locked || scene_dirty || continuous_input ||
                       python_animation || python_overlay || python_redraw ||
                       gui_animation || input_event || posted_work || render_work ||
                       store_dirty || swapchain_resize_ready || window_resize_paint_pending ||
                       viewport_resize_settle_ready;
            }

            [[nodiscard]] bool needsContinuousLoop() const {
                const bool resize_deferral_throttles_animation =
                    viewport_resize_deferring ||
                    (swapchain_resize_pending && !swapchain_resize_ready);
                return scene_dirty || continuous_input || python_animation ||
                       python_overlay || python_redraw ||
                       (gui_animation && !resize_deferral_throttles_animation) ||
                       render_work || viewport_export_locked || store_dirty ||
                       swapchain_resize_ready || window_resize_paint_pending ||
                       viewport_resize_settle_ready;
            }
        };

        [[nodiscard]] FrameDemand collectFrameDemand(bool viewport_export_locked,
                                                     bool drained_store_dirty = false,
                                                     bool consume_python_redraw = true);
        void waitForNextEvent(bool is_training);

        class CallbackCleanup {
            std::vector<std::function<void()>> cleanups_;

        public:
            void add(std::function<void()> fn) { cleanups_.push_back(std::move(fn)); }
            void clear() {
                for (auto it = cleanups_.rbegin(); it != cleanups_.rend(); ++it)
                    (*it)();
                cleanups_.clear();
            }
            ~CallbackCleanup() { clear(); }
            CallbackCleanup() = default;
            CallbackCleanup(const CallbackCleanup&) = delete;
            CallbackCleanup& operator=(const CallbackCleanup&) = delete;
        };

        // Options
        ViewerOptions options_;

        // Core components
        Viewport viewport_;
        std::unique_ptr<WindowManager> window_manager_;
        std::unique_ptr<InputController> input_controller_;
        std::unique_ptr<RenderingManager> rendering_manager_;
        std::unique_ptr<SceneManager> scene_manager_;
        std::shared_ptr<TrainerManager> trainer_manager_;
        std::unique_ptr<DataLoadingService> data_loader_;
        std::unique_ptr<ParameterManager> parameter_manager_;
        std::unique_ptr<MainLoop> main_loop_;

        // Frame exception boundary state (viewer thread only).
        FrameStateMachine frame_state_;
        uint64_t suppressed_frame_errors_ = 0;
        std::chrono::steady_clock::time_point last_frame_error_log_{};

        // Tools
        std::shared_ptr<tools::AlignTool> align_tool_;
        std::shared_ptr<tools::SelectionTool> selection_tool_;
        std::unique_ptr<ToolContext> tool_context_;

        // Centralized editor state
        EditorContext editor_context_;

        mutable std::mutex work_queue_mutex_;
        std::vector<WorkItem> work_queue_;
        std::vector<WorkItem> render_work_queue_;
        std::thread::id viewer_thread_id_;
        bool accepting_work_ = true;
        bool shutdown_started_ = false;
        bool processing_render_work_ = false;

        std::mutex shutdown_callback_mutex_;
        std::function<void()> shutdown_requested_callback_;

        CallbackCleanup callback_cleanup_;

        // State tracking
        bool fully_initialized_ = false;
        bool window_initialized_ = false;
        bool gui_initialized_ = false;
        bool tools_initialized_ = false;
        bool view_context_bridge_initialized_ = false;
        bool pending_auto_train_ = false;
        enum class PendingTrainingAction : std::uint8_t {
            None,
            Reset,
            NewProject,
            CloseSave,
            CloseDiscard,
        };
        PendingTrainingAction pending_training_action_ = PendingTrainingAction::None;
        bool pending_training_action_posted_ = false;
        ProjectSwitchDisposition
            pending_new_project_disposition_ =
                ProjectSwitchDisposition::RequireClean;
        int pending_training_completion_refresh_frames_ = 0;
        bool gui_frame_rendered_ = false;
        bool startup_plugin_preload_started_ = false;
        bool startup_project_open_attempted_ = false;
        bool close_save_notice_posted_ = false;
        std::optional<std::filesystem::path>
            pending_close_save_path_;
        std::jthread force_exit_completion_watcher_;
        std::atomic<bool> force_exit_wait_expired_{false};
        bool force_exit_watcher_armed_ = false;
        std::chrono::milliseconds
            force_exit_completion_timeout_{30000};
        bool gui_panels_ready_emitted_ = false;
        std::uint64_t startup_plugin_load_status_revision_ = 0;
        bool plugin_preload_timing_active_ = false;
        std::chrono::nanoseconds plugin_preload_max_update_stall_{};
        bool update_work_processed_ = false;
        std::chrono::high_resolution_clock::time_point last_frame_time_ = std::chrono::high_resolution_clock::now();
        float live_scene_clip_time_ = 0.0f;
        bool sequencer_ui_initialized_ = false;
        std::unique_ptr<python::SequencerUIStateData> sequencer_ui_state_;
        std::vector<std::filesystem::path> pending_view_paths_;
        std::filesystem::path pending_dataset_path_;
        project::GuiSessionRestoreCoordinator
            gui_session_restore_;
        std::optional<project::PreparedGuiSessionRestore>
            pending_project_tools_restore_;
        std::optional<project::GuiSessionRestoreTicket>
            hydration_terminal_restore_ticket_;
        lfs::io::project::ProjectSessionChapters
            retained_project_session_;
        std::vector<
            project::CameraBookmarkProjectState>
            camera_bookmarks_;
        std::unique_ptr<project::ProjectLifecycle>
            project_lifecycle_;
    };

} // namespace lfs::vis
