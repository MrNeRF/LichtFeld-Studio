/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/error_latch.hpp"
#include "core/export.hpp"
#include "core/parameters.hpp"
#include "core/splat_exportable_storage.hpp"
#include "io/session_chapters.hpp"
#include "training/trainer.hpp"
#include "training_state.hpp"
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <thread>
#include <vector>

namespace lfs::core {
    class Scene;
}

namespace lfs::vis {

    // Forward declarations
    class VisualizerImpl;
    class VisualizerImplResetTest_ForceExitWhileStoppingArmsWatcher_Test;
    class VisualizerImplResetTest_SaveWhilePausedTrainingRoutesThroughLiveTrainer_Test;
    class VisualizerImplResetTest_SaveWhileStoppingStillBlocksUntilSnapshotPublished_Test;
    class VisualizerImplResetTest_SaveAsWhilePausedTrainingRoutesThroughLiveTrainer_Test;

    class LFS_VIS_API TrainerManager {
    public:
        // Legacy State enum for backwards compatibility
        // Use TrainingState from training_state.hpp for new code
        using State = TrainingState;

        TrainerManager();
        ~TrainerManager();

        // Delete copy operations
        TrainerManager(const TrainerManager&) = delete;
        TrainerManager& operator=(const TrainerManager&) = delete;

        // Allow move operations
        TrainerManager(TrainerManager&&) = default;
        TrainerManager& operator=(TrainerManager&&) = default;

        // Setup and teardown
        void setTrainer(std::unique_ptr<lfs::training::Trainer> trainer);
        void setTrainerFromCheckpoint(std::unique_ptr<lfs::training::Trainer> trainer, int checkpoint_iteration);
        [[nodiscard]] bool clearTrainer();
        bool hasTrainer() const;

        // Link to viewer for notifications
        void setViewer(VisualizerImpl* viewer) { viewer_ = viewer; }

        // Link to scene for data access (Scene-based trainer mode)
        void setScene(core::Scene* scene) { scene_ = scene; }
        [[nodiscard]] const core::Scene* getScene() const { return scene_; }

        // Training control
        bool startTraining();
        void pauseTraining();
        void resumeTraining();
        void stopTraining();
        bool requestSaveProject();
        // Suppress the completion notification modal for the next TrainingCompleted
        // dispatch (stop initiated by New Project / app close / reset, issue #1604).
        void suppressCompletionNotification() { suppress_completion_notification_.store(true, std::memory_order_relaxed); }

        // Temporary pause for short synchronization-sensitive operations; does not change UI state.
        struct TemporaryPauseResult {
            bool synchronized = false;
            bool resume_required = false;
        };

        void pauseTrainingTemporary();
        [[nodiscard]] TemporaryPauseResult pauseTrainingTemporaryAndWait(std::chrono::milliseconds timeout);
        void resumeTrainingTemporary();

        // State machine access
        [[nodiscard]] const TrainingStateMachine& getStateMachine() const { return state_machine_; }
        [[nodiscard]] bool canPerform(TrainingAction action) const { return state_machine_.canPerform(action); }
        [[nodiscard]] std::string_view getActionBlockedReason(TrainingAction action) const {
            return state_machine_.getActionBlockedReason(action);
        }

        // State queries (delegate to state machine)
        [[nodiscard]] TrainingState getState() const { return state_machine_.getState(); }
        [[nodiscard]] bool isRunning() const { return state_machine_.isInState(TrainingState::Running); }
        [[nodiscard]] bool isPaused() const { return state_machine_.isInState(TrainingState::Paused); }
        [[nodiscard]] bool isFinished() const { return state_machine_.isInState(TrainingState::Finished); }
        [[nodiscard]] bool isTrainingActive() const { return state_machine_.isActive(); }
        [[nodiscard]] bool canStart() const { return canPerform(TrainingAction::Start); }
        [[nodiscard]] bool canPause() const { return canPerform(TrainingAction::Pause); }
        [[nodiscard]] bool canResume() const { return canPerform(TrainingAction::Resume); }
        [[nodiscard]] bool canStop() const { return canPerform(TrainingAction::Stop); }
        [[nodiscard]] bool canReset() const { return canPerform(TrainingAction::Reset); }
        [[nodiscard]] bool isCompletionPending() const {
            return completion_pending_.load(std::memory_order_acquire);
        }
        [[nodiscard]] bool isPublishingFinalSnapshot() const {
            return isCompletionPending() && !isTrainingActive();
        }
        [[nodiscard]] bool isPausedAtCheckpointBaseline() const;
        [[nodiscard]] std::optional<int> checkpointBaselineIteration() const {
            return checkpoint_baseline_iteration_;
        }

        // Progress information - directly query trainer
        int getCurrentIteration() const;
        float getCurrentLoss() const;
        int getTotalIterations() const;
        int getNumSplats() const;
        int getMaxGaussians() const;
        std::vector<size_t> getSaveSteps() const;
        void setSaveSteps(std::vector<size_t> save_steps);
        const char* getStrategyType() const;
        bool isGutEnabled() const;

        // Time tracking
        float getElapsedSeconds() const;
        float getEstimatedRemainingSeconds() const;

        // Loss buffer management (this needs to be stored)
        std::deque<float> getLossBuffer() const;
        void updateLoss(float loss);

        // PSNR buffer management
        struct EvaluationMetricsSnapshot {
            int iteration = 0;
            float psnr = 0.0f;
            float ssim = 0.0f;
        };

        std::deque<float> getPSNRBuffer() const;
        void updatePSNR(float psnr);
        void updateEvaluationMetrics(int iteration, float psnr, float ssim);
        std::optional<EvaluationMetricsSnapshot> getLastEvaluationMetrics() const;
        void clearEvaluationMetrics();
        [[nodiscard]] lfs::io::project::MetricsChapter
        captureProjectMetrics() const;
        void restoreProjectMetrics(
            const lfs::io::project::MetricsChapter& metrics);
        void clearRestoredProjectMetrics();

        // Access to trainer (for rendering, etc.)
        lfs::training::Trainer* getTrainer() { return trainer_.get(); }
        const lfs::training::Trainer* getTrainer() const { return trainer_.get(); }

        // Splat exportable storage — populated when training starts with a viewer
        // active. The viewer's vksplat renderer imports the same physical block
        // for zero-copy interop. nullptr if running headless or fallback.
        const lfs::core::SplatExportableStorage* splatExportableStorage() const {
            return splat_storage_.has_value() ? &*splat_storage_ : nullptr;
        }

        // Wait for training to complete (blocking)
        [[nodiscard]] bool waitForCompletion();

        // Get last error message
        const std::string& getLastError() const { return last_error_; }

        // Most recent training failure as a typed error. During an
        // active run this may briefly differ from the legacy getLastError()
        // string, which keeps its exact current lifecycle for the deprecation
        // window.
        [[nodiscard]] std::optional<lfs::Error> lastTrainingError() const {
            return last_training_error_.get();
        }

        // Camera access
        std::shared_ptr<const lfs::core::Camera> getCamById(int camId) const;
        std::vector<std::shared_ptr<lfs::core::Camera>> getAllCamList() const;
        std::expected<lfs::training::Trainer::CameraMetricsSnapshot, std::string> computeCameraMetricsForCameraId(
            int camera_id,
            bool include_ssim,
            const lfs::training::Trainer::CameraMetricsAppearanceConfig& appearance) const;

        // Pending parameters (editable in Ready state, applied on start)
        lfs::core::param::OptimizationParameters& getEditableOptParams() { return pending_opt_params_; }
        const lfs::core::param::OptimizationParameters& getEditableOptParams() const { return pending_opt_params_; }
        lfs::core::param::DatasetConfig& getEditableDatasetParams() { return pending_dataset_params_; }
        const lfs::core::param::DatasetConfig& getEditableDatasetParams() const { return pending_dataset_params_; }
        void applyPendingParams();

    private:
        struct TrainingCompletionData {
            int iteration = 0;
            float final_loss = 0.0f;
            float elapsed_seconds = 0.0f;
            bool success = false;
            bool user_stopped = false;
            bool resource_exhausted = false;
            FinishReason reason = FinishReason::None;
            std::optional<std::string> error;
            std::optional<lfs::Error> typed_error;
        };

        friend class VisualizerImplResetTest_ForceExitWhileStoppingArmsWatcher_Test;
        friend class VisualizerImplResetTest_SaveWhilePausedTrainingRoutesThroughLiveTrainer_Test;
        friend class VisualizerImplResetTest_SaveWhileStoppingStillBlocksUntilSnapshotPublished_Test;
        friend class VisualizerImplResetTest_SaveAsWhilePausedTrainingRoutesThroughLiveTrainer_Test;

        // Training thread function
        void trainingThreadFunc(std::stop_token stop_token);
        void launchTrainingThread();
        void completionReaperLoop(std::stop_token stop_token);
        void finishTrainingThreadJoin();
        void dispatchTrainingCompleted(TrainingCompletionData completion);

        // State management
        void handleTrainingComplete(bool success, const std::string& error = "",
                                    bool resource_exhausted = false,
                                    const std::optional<lfs::Error>& typed_error = std::nullopt);
        void setupEventHandlers();
        void setupStateMachineCallbacks();

        [[nodiscard]] lfs::core::SplatTensorAllocator createTrainingSplatTensorAllocator(
            const lfs::core::param::TrainingParameters& params,
            std::size_t min_capacity = 0);

        // Install densify-time grow/rebind hook on the training model.
        void installExportableCapacityEnsure(lfs::core::SplatData& model);
        // Install densify barrier on the live trainer (no-op
        // when there is no exportable block / no trainer).
        void installExportableDensifyBarrier();
        // Body of the capacity-ensure hook (member so rebind cannot destroy the
        // running std::function target mid-call).
        [[nodiscard]] bool growExportableForDensify(std::size_t needed_rows);

        // Drop Vulkan import of the exportable block (cuda-only views). Nested.
        [[nodiscard]] bool beginExportableDensifyBarrier();
        // Re-import exportable block into Vulkan after densify/commit. Nested.
        [[nodiscard]] bool endExportableDensifyBarrier();
        // Shared drop / re-import helpers used by grow and the densify barrier.
        [[nodiscard]] bool rebindExportableCudaOnly();
        [[nodiscard]] bool rebindExportableVulkanInterop();

        // Member variables
        std::unique_ptr<lfs::training::Trainer> trainer_;
        std::unique_ptr<std::jthread> training_thread_;
        std::optional<std::stop_source> training_stop_source_;
        std::mutex training_thread_mutex_;
        std::condition_variable training_thread_cv_;
        std::jthread completion_reaper_;
        VisualizerImpl* viewer_ = nullptr;
        core::Scene* scene_ = nullptr;
        std::optional<lfs::core::SplatExportableStorage> splat_storage_;
        // Nesting depth for densify-window Vulkan exclusion.
        int exportable_densify_barrier_depth_ = 0;

        // State machine (single source of truth for state)
        TrainingStateMachine state_machine_;
        std::string last_error_;
        core::ErrorLatch last_training_error_;
        mutable std::mutex state_mutex_;
        mutable std::mutex trainer_lifetime_mutex_;

        // Synchronization
        std::condition_variable completion_cv_;
        std::mutex completion_mutex_;
        bool training_joined_ = true;
        std::optional<TrainingCompletionData> pending_completion_;
        std::atomic<bool> completion_pending_{false};
        std::atomic<bool> suppress_completion_notification_{false};

        static constexpr int COMPLETION_TIMEOUT_SEC = 30;
        static constexpr int MAX_LOSS_POINTS = 200;
        std::deque<float> loss_buffer_;
        mutable std::mutex loss_buffer_mutex_;
        static constexpr int MAX_PSNR_POINTS = 200;
        std::deque<float> psnr_buffer_;
        mutable std::mutex psnr_buffer_mutex_;
        std::mutex temporary_pause_mutex_;
        std::uint32_t temporary_pause_depth_ = 0;
        bool temporary_pause_initially_paused_ = false;
        bool temporary_pause_resume_in_flight_ = false;
        std::optional<EvaluationMetricsSnapshot> last_eval_metrics_;
        std::vector<EvaluationMetricsSnapshot>
            evaluation_history_;
        mutable std::mutex eval_metrics_mutex_;

        // Training time tracking
        std::chrono::steady_clock::time_point training_start_time_;
        std::chrono::steady_clock::duration accumulated_training_time_{0};

        lfs::core::param::OptimizationParameters pending_opt_params_;
        lfs::core::param::DatasetConfig pending_dataset_params_;
        std::optional<int> checkpoint_baseline_iteration_;
        std::optional<std::chrono::steady_clock::duration>
            restored_accumulated_training_time_;
        std::optional<lfs::io::project::TrainingFinishReason>
            restored_finish_reason_;
        bool restored_finish_published_ = false;

        [[nodiscard]] FinishReason resolvedRestoredFinishReason() const;
        void applyRestoredCheckpointPresentation();
        void publishRestoredTrainingStore();
    };

} // namespace lfs::vis
