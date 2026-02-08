/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/export.hpp"
#include "core/parameters.hpp"
#include "training/trainer.hpp"
#include "training_state.hpp"
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <stop_token>
#include <thread>

namespace lfs::vis {

    // Forward declarations
    class VisualizerImpl;
    class Scene;

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
        void clearTrainer();
        bool hasTrainer() const;

        // Link to viewer for notifications
        void setViewer(VisualizerImpl* viewer) { viewer_ = viewer; }

        // Link to scene for data access (Scene-based trainer mode)
        void setScene(Scene* scene) { scene_ = scene; }
        [[nodiscard]] const Scene* getScene() const { return scene_; }

        // Training control
        bool startTraining();
        void pauseTraining();
        void resumeTraining();
        void stopTraining();
        void requestSaveCheckpoint();

        // Temporary pause (for camera movement - doesn't change UI state)
        void pauseTrainingTemporary();
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

        // Progress information - directly query trainer
        int getCurrentIteration() const;
        float getCurrentLoss() const;
        int getTotalIterations() const;
        int getNumSplats() const;
        int getMaxGaussians() const;
        const char* getStrategyType() const;
        bool isGutEnabled() const;

        // Time tracking
        float getElapsedSeconds() const;
        float getEstimatedRemainingSeconds() const;

        // Loss buffer management (this needs to be stored)
        std::deque<float> getLossBuffer() const;
        void updateLoss(float loss);

        // Access to trainer (for rendering, etc.)
        lfs::training::Trainer* getTrainer() { return trainer_.get(); }
        const lfs::training::Trainer* getTrainer() const { return trainer_.get(); }

        // Wait for training to complete (blocking)
        void waitForCompletion();

        // Get last error message
        const std::string& getLastError() const { return last_error_; }

        // Camera access
        std::shared_ptr<const lfs::core::Camera> getCamById(int camId) const;
        std::vector<std::shared_ptr<lfs::core::Camera>> getCamList() const;
        std::vector<std::shared_ptr<lfs::core::Camera>> getAllCamList() const;

        // Pending parameters (editable in Ready state, applied on start)
        lfs::core::param::OptimizationParameters& getEditableOptParams() { return pending_opt_params_; }
        const lfs::core::param::OptimizationParameters& getEditableOptParams() const { return pending_opt_params_; }
        lfs::core::param::DatasetConfig& getEditableDatasetParams() { return pending_dataset_params_; }
        const lfs::core::param::DatasetConfig& getEditableDatasetParams() const { return pending_dataset_params_; }
        void applyPendingParams();

    private:
        // Training thread function
        void trainingThreadFunc(std::stop_token stop_token);

        // State management
        void handleTrainingComplete(bool success, const std::string& error = "");
        void setupEventHandlers();
        void setupStateMachineCallbacks();

        // Resource cleanup (called by state machine)
        void cleanupTrainingResources(const TrainingResources& resources);
        void updateResourceTracking();

        // Member variables
        std::unique_ptr<lfs::training::Trainer> trainer_;
        std::unique_ptr<std::jthread> training_thread_;
        VisualizerImpl* viewer_ = nullptr;
        Scene* scene_ = nullptr;

        // State machine (single source of truth for state)
        TrainingStateMachine state_machine_;
        std::string last_error_;
        mutable std::mutex state_mutex_;

        // Synchronization
        std::condition_variable completion_cv_;
        std::mutex completion_mutex_;
        bool training_complete_ = false;

        static constexpr int COMPLETION_TIMEOUT_SEC = 30;
        static constexpr int MAX_LOSS_POINTS = 200;
        std::deque<float> loss_buffer_;
        mutable std::mutex loss_buffer_mutex_;

        // Training time tracking
        std::chrono::steady_clock::time_point training_start_time_;
        std::chrono::steady_clock::duration accumulated_training_time_{0};

        lfs::core::param::OptimizationParameters pending_opt_params_;
        lfs::core::param::DatasetConfig pending_dataset_params_;
    };

} // namespace lfs::vis