/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training/training_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/parameter_manager.hpp"
#include "core/services.hpp"
#include "scene/scene.hpp"
#include "training/training_setup.hpp"
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>

namespace lfs::vis {

    using namespace lfs::core::events;

    TrainerManager::TrainerManager() {
        setupEventHandlers();
        setupStateMachineCallbacks();
        LOG_DEBUG("TrainerManager created");
    }

    void TrainerManager::setupStateMachineCallbacks() {
        state_machine_.setCleanupCallback([this](const TrainingResources& resources) {
            cleanupTrainingResources(resources);
        });

        state_machine_.setStateChangeCallback([this](TrainingState old_state, TrainingState new_state) {
            // Emit events on state changes
            if (new_state == TrainingState::Idle) {
                loss_buffer_.clear();
                last_error_.clear();
            }
        });
    }

    void TrainerManager::cleanupTrainingResources(const TrainingResources& /*resources*/) {
        LOG_DEBUG("Cleaning up training resources");

        if (training_thread_ && training_thread_->joinable()) {
            training_thread_->request_stop();
            auto timeout = std::chrono::milliseconds(500);
            {
                std::unique_lock<std::mutex> lock(completion_mutex_);
                if (completion_cv_.wait_for(lock, timeout, [this] { return training_complete_; })) {
                    lock.unlock();
                    training_thread_->join();
                } else {
                    lock.unlock();
                    LOG_WARN("Thread didn't respond to stop request, detaching");
                    training_thread_->detach();
                }
            }
            training_thread_.reset();
        }

        if (trainer_) {
            trainer_->shutdown();
            trainer_.reset();
        }
    }

    void TrainerManager::updateResourceTracking() {
        TrainingResources resources;
        resources.has_trainer = trainer_ != nullptr;
        resources.has_training_thread = training_thread_ != nullptr && training_thread_->joinable();
        resources.has_scene_data = scene_ != nullptr;
        resources.has_gpu_tensors = trainer_ && trainer_->isInitialized();
        if (scene_) {
            resources.training_node_name = scene_->getTrainingModelNodeName();
        }
        state_machine_.setResources(resources);
    }

    TrainerManager::~TrainerManager() {
        // Ensure training is stopped before destruction
        if (training_thread_ && training_thread_->joinable()) {
            LOG_INFO("Stopping training thread during destruction...");
            stopTraining();
            waitForCompletion();
        }
    }

    void TrainerManager::setTrainer(std::unique_ptr<lfs::training::Trainer> trainer) {
        LOG_TIMER_TRACE("TrainerManager::setTrainer");

        clearTrainer();

        if (trainer) {
            const auto& params = trainer->getParams();
            pending_opt_params_ = params.optimization;
            pending_dataset_params_ = params.dataset;

            trainer_ = std::move(trainer);
            updateResourceTracking();

            if (!state_machine_.transitionTo(TrainingState::Ready)) {
                LOG_WARN("Failed to transition to Ready");
            }

            internal::TrainerReady{}.emit();
            LOG_DEBUG("Trainer ready");
        }
    }

    void TrainerManager::setTrainerFromCheckpoint(std::unique_ptr<lfs::training::Trainer> trainer, int checkpoint_iteration) {
        LOG_TIMER_TRACE("TrainerManager::setTrainerFromCheckpoint");

        clearTrainer();

        if (trainer) {
            const auto& params = trainer->getParams();
            pending_opt_params_ = params.optimization;
            pending_dataset_params_ = params.dataset;

            trainer_ = std::move(trainer);
            updateResourceTracking();
            // Checkpoint load goes to Paused

            if (!state_machine_.transitionTo(TrainingState::Paused)) {
                LOG_WARN("Failed to transition to Paused");
            }

            state::TrainingPaused{.iteration = checkpoint_iteration}.emit();
            LOG_DEBUG("Trainer paused from checkpoint (iteration {})", checkpoint_iteration);
        }
    }

    bool TrainerManager::hasTrainer() const {
        return trainer_ != nullptr;
    }

    void TrainerManager::clearTrainer() {
        LOG_DEBUG("Clearing trainer");

        // Stop any ongoing training first
        const auto state = getState();
        if (state == TrainingState::Running || state == TrainingState::Paused) {
            LOG_INFO("Stopping active training before clearing");
            // If paused, resume first so thread can process stop request
            if (state == TrainingState::Paused && trainer_) {
                trainer_->request_resume();
            }
            stopTraining();
            waitForCompletion();
        } else if (state == TrainingState::Stopping) {
            // Already stopping, just wait for completion
            LOG_INFO("Waiting for training to finish stopping");
            waitForCompletion();
        }

        // Destroy trainer - destructor handles cleanup
        trainer_.reset();

        // Transition to Idle
        updateResourceTracking();

        if (!state_machine_.transitionTo(TrainingState::Idle)) {
            LOG_WARN("Failed to transition to Idle");
        }

        LOG_INFO("Trainer cleared");
    }

    bool TrainerManager::startTraining() {
        LOG_TIMER("TrainerManager::startTraining");

        if (!canStart()) {
            LOG_WARN("Cannot start: {}", getActionBlockedReason(TrainingAction::Start));
            return false;
        }

        if (!trainer_) {
            LOG_ERROR("Cannot start training - no trainer available");
            return false;
        }

        applyPendingParams();

        if (trainer_->isInitialized()) {
            LOG_DEBUG("Resuming from iteration {}", trainer_->get_current_iteration());
        } else {
            const auto& params = trainer_->getParams();

            if (scene_) {
                if (auto result = lfs::training::initializeTrainingModel(params, *scene_); !result) {
                    LOG_ERROR("Failed to initialize model: {}", result.error());
                    last_error_ = result.error();

                    if (!state_machine_.transitionToFinished(FinishReason::Error)) {
                        LOG_WARN("Failed to transition to Finished(Error)");
                    }
                    return false;
                }
            }

            if (auto result = trainer_->initialize(params); !result) {
                LOG_ERROR("Failed to initialize trainer: {}", result.error());
                last_error_ = result.error();

                if (!state_machine_.transitionToFinished(FinishReason::Error)) {
                    LOG_WARN("Failed to transition to Finished(Error)");
                }
                return false;
            }
        }

        {
            std::lock_guard<std::mutex> lock(completion_mutex_);
            training_complete_ = false;
        }

        updateResourceTracking();

        if (!state_machine_.transitionTo(TrainingState::Running)) {
            LOG_WARN("Failed to transition to Running");
        }

        training_start_time_ = std::chrono::steady_clock::now();
        accumulated_training_time_ = std::chrono::steady_clock::duration{0};

        state::TrainingStarted{.total_iterations = getTotalIterations()}.emit();

        training_thread_ = std::make_unique<std::jthread>(
            [this](std::stop_token stop_token) {
                trainingThreadFunc(stop_token);
            });

        LOG_INFO("Training started - {} iterations planned", getTotalIterations());
        return true;
    }

    void TrainerManager::pauseTraining() {
        if (!canPause()) {
            LOG_TRACE("Cannot pause: {}", getActionBlockedReason(TrainingAction::Pause));
            return;
        }

        if (trainer_) {
            trainer_->request_pause();
            accumulated_training_time_ += std::chrono::steady_clock::now() - training_start_time_;

            if (!state_machine_.transitionTo(TrainingState::Paused)) {
                LOG_WARN("Failed to transition to Paused");
            }

            state::TrainingPaused{.iteration = getCurrentIteration()}.emit();
            LOG_INFO("Training paused at iteration {}", getCurrentIteration());
        }
    }

    void TrainerManager::resumeTraining() {
        if (!canResume()) {
            LOG_TRACE("Cannot resume: {}", getActionBlockedReason(TrainingAction::Resume));
            return;
        }
        if (!trainer_)
            return;

        const int iter = getCurrentIteration();
        const bool need_thread = !training_thread_ || !training_thread_->joinable();

        if (need_thread) {
            // Checkpoint resume: no thread exists yet
            training_complete_ = false;
            accumulated_training_time_ = std::chrono::steady_clock::duration{0};
            training_thread_ = std::make_unique<std::jthread>(
                [this](std::stop_token st) { trainingThreadFunc(st); });
        } else {
            trainer_->request_resume();
        }

        training_start_time_ = std::chrono::steady_clock::now();
        updateResourceTracking();

        if (!state_machine_.transitionTo(TrainingState::Running)) {
            LOG_WARN("Failed to transition to Running");
        }

        state::TrainingResumed{.iteration = iter}.emit();
        LOG_INFO("Training resumed at iteration {}", iter);
    }

    void TrainerManager::pauseTrainingTemporary() {
        if (!isRunning())
            return;

        if (trainer_) {
            trainer_->request_pause();
            LOG_TRACE("Training temporarily paused at iteration {}", getCurrentIteration());
        }
    }

    void TrainerManager::resumeTrainingTemporary() {
        if (!isRunning())
            return;

        if (trainer_) {
            trainer_->request_resume();
            LOG_TRACE("Training resumed from temporary pause at iteration {}", getCurrentIteration());
        }
    }

    void TrainerManager::stopTraining() {
        if (!canStop()) {
            LOG_TRACE("Cannot stop: {}", getActionBlockedReason(TrainingAction::Stop));
            return;
        }

        LOG_DEBUG("Requesting training stop");
        updateResourceTracking();

        if (!state_machine_.transitionTo(TrainingState::Stopping)) {
            LOG_WARN("Failed to transition to Stopping");
        }

        if (trainer_) {
            trainer_->request_stop();
        }

        if (training_thread_ && training_thread_->joinable()) {
            training_thread_->request_stop();
        }

        state::TrainingStopped{.iteration = getCurrentIteration(), .user_requested = true}.emit();
        LOG_INFO("Training stop requested at iteration {}", getCurrentIteration());
    }

    void TrainerManager::requestSaveCheckpoint() {
        if (trainer_ && isTrainingActive()) {
            trainer_->request_save();
            LOG_INFO("Checkpoint save requested at iteration {}", getCurrentIteration());
        } else {
            LOG_WARN("Cannot save checkpoint - training not active");
        }
    }

    bool TrainerManager::resetTraining() {
        LOG_INFO("Resetting training to initial state");

        if (!trainer_) {
            LOG_WARN("No trainer to reset");
            return false;
        }

        // Stop if active
        if (isTrainingActive()) {
            stopTraining();
            waitForCompletion();
        }

        if (trainer_->isInitialized()) {
            LOG_DEBUG("Releasing GPU memory");
            trainer_.reset();
            cudaDeviceSynchronize();

            if (!scene_) {
                LOG_ERROR("Cannot reset: no scene");

                if (!state_machine_.transitionToFinished(FinishReason::Error)) {
                    LOG_WARN("Failed to transition to Finished(Error)");
                }
                return false;
            }
            trainer_ = std::make_unique<lfs::training::Trainer>(*scene_);
        }

        loss_buffer_.clear();

        if (!state_machine_.transitionTo(TrainingState::Ready)) {
            LOG_WARN("Failed to transition to Ready");
        }

        LOG_DEBUG("Training reset complete");
        return true;
    }

    void TrainerManager::waitForCompletion() {
        if (!training_thread_ || !training_thread_->joinable()) {
            return;
        }

        LOG_DEBUG("Waiting for training thread to complete...");

        std::unique_lock<std::mutex> lock(completion_mutex_);
        completion_cv_.wait(lock, [this] { return training_complete_; });

        training_thread_->join();
        training_thread_.reset();

        LOG_DEBUG("Training thread joined successfully");
    }

    int TrainerManager::getCurrentIteration() const {
        return trainer_ ? trainer_->get_current_iteration() : 0;
    }

    float TrainerManager::getCurrentLoss() const {
        return trainer_ ? trainer_->get_current_loss() : 0.0f;
    }

    int TrainerManager::getTotalIterations() const {
        if (!trainer_)
            return 0;
        return static_cast<int>(trainer_->getParams().optimization.iterations);
    }

    int TrainerManager::getNumSplats() const {
        if (!trainer_)
            return 0;
        // Strategy may not be created yet if using Scene-based constructor
        // In that case, try to get size from scene
        if (scene_) {
            const auto* model = scene_->getTrainingModel();
            if (model) {
                return static_cast<int>(model->size());
            }
        }
        // Fall back to strategy if trainer is initialized
        if (trainer_->isInitialized()) {
            return static_cast<int>(trainer_->get_strategy().get_model().size());
        }
        return 0;
    }

    int TrainerManager::getMaxGaussians() const {
        if (!trainer_)
            return 0;
        return trainer_->getParams().optimization.max_cap;
    }

    const char* TrainerManager::getStrategyType() const {
        if (!trainer_ || !trainer_->isInitialized())
            return "unknown";
        return trainer_->get_strategy().strategy_type();
    }

    bool TrainerManager::isGutEnabled() const {
        if (!trainer_)
            return false;
        return trainer_->getParams().optimization.gut;
    }

    float TrainerManager::getElapsedSeconds() const {
        const auto state = getState();
        if (state == TrainingState::Running) {
            const auto current = std::chrono::steady_clock::now() - training_start_time_;
            return std::chrono::duration<float>(accumulated_training_time_ + current).count();
        }
        if (state == TrainingState::Paused || state == TrainingState::Finished) {
            return std::chrono::duration<float>(accumulated_training_time_).count();
        }
        return 0.0f;
    }

    float TrainerManager::getEstimatedRemainingSeconds() const {
        const float elapsed = getElapsedSeconds();
        const int current_iter = getCurrentIteration();
        const int total_iter = getTotalIterations();

        if (current_iter <= 0 || elapsed <= 0.0f || total_iter <= current_iter)
            return 0.0f;

        const float secs_per_iter = elapsed / static_cast<float>(current_iter);
        return secs_per_iter * static_cast<float>(total_iter - current_iter);
    }

    void TrainerManager::updateLoss(float loss) {
        std::lock_guard<std::mutex> lock(loss_buffer_mutex_);
        loss_buffer_.push_back(loss);
        while (loss_buffer_.size() > static_cast<size_t>(MAX_LOSS_POINTS)) {
            loss_buffer_.pop_front();
        }
        LOG_TRACE("Loss updated: {:.6f} (buffer size: {})", loss, loss_buffer_.size());
    }

    std::deque<float> TrainerManager::getLossBuffer() const {
        std::lock_guard<std::mutex> lock(loss_buffer_mutex_);
        return loss_buffer_;
    }

    void TrainerManager::trainingThreadFunc(std::stop_token stop_token) {
        LOG_INFO("Training thread started");
        LOG_TIMER("Training execution");

        try {
            LOG_DEBUG("Starting trainer->train() with stop token");
            auto train_result = trainer_->train(stop_token);

            if (!train_result) {
                LOG_ERROR("Training failed: {}", train_result.error());
                handleTrainingComplete(false, train_result.error());
            } else {
                LOG_INFO("Training completed successfully");
                handleTrainingComplete(true);
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Exception in training thread: {}", e.what());
            handleTrainingComplete(false, std::format("Exception in training: {}", e.what()));
        } catch (...) {
            LOG_CRITICAL("Unknown exception in training thread");
            handleTrainingComplete(false, "Unknown exception in training");
        }

        LOG_INFO("Training thread finished");
    }

    void TrainerManager::handleTrainingComplete(const bool success, const std::string& error) {
        if (!error.empty()) {
            last_error_ = error;
            LOG_ERROR("Training error: {}", error);
        }

        const float elapsed = getElapsedSeconds();
        const int final_iter = getCurrentIteration();
        const float final_loss = getCurrentLoss();
        const bool user_stopped = (getState() == TrainingState::Stopping);

        updateResourceTracking();
        if (!user_stopped) {
            if (!state_machine_.transitionTo(TrainingState::Stopping)) {
                LOG_WARN("Failed to transition to Stopping");
            }
        }

        const FinishReason reason = !success       ? FinishReason::Error
                                    : user_stopped ? FinishReason::UserStopped
                                                   : FinishReason::Completed;

        if (!state_machine_.transitionToFinished(reason)) {
            LOG_WARN("Failed to transition to Finished");
        }

        LOG_INFO("Training finished: iter={}, loss={:.6f}, time={:.1f}s",
                 final_iter, final_loss, elapsed);

        // Skip popup for user-stopped (user knows they stopped it)
        if (!user_stopped) {
            state::TrainingCompleted{
                .iteration = final_iter,
                .final_loss = final_loss,
                .elapsed_seconds = elapsed,
                .success = success,
                .error = error.empty() ? std::nullopt : std::optional(error)}
                .emit();
        }

        {
            std::lock_guard lock(completion_mutex_);
            training_complete_ = true;
        }
        completion_cv_.notify_all();
    }

    void TrainerManager::setupEventHandlers() {
        using namespace lfs::core::events;

        // Training control commands
        cmd::StartTraining::when([this](const auto&) {
            startTraining();
        });

        cmd::PauseTraining::when([this](const auto&) {
            pauseTraining();
        });

        cmd::ResumeTraining::when([this](const auto&) {
            resumeTraining();
        });

        cmd::StopTraining::when([this](const auto&) {
            stopTraining();
        });

        cmd::SaveCheckpoint::when([this](const auto&) {
            requestSaveCheckpoint();
        });

        // Listen for training progress events - update loss buffer
        state::TrainingProgress::when([this](const auto& event) {
            updateLoss(event.loss);
        });
    }

    std::shared_ptr<const lfs::core::Camera> TrainerManager::getCamById(int camId) const {
        // Get camera from Scene (Scene owns all training data)
        if (scene_) {
            return scene_->getCameraByUid(camId);
        }
        LOG_ERROR("getCamById called but scene is not set");
        return nullptr;
    }

    std::vector<std::shared_ptr<const lfs::core::Camera>> TrainerManager::getCamList() const {
        // Get cameras from Scene (Scene owns all training data)
        if (scene_) {
            return scene_->getAllCameras();
        }
        LOG_ERROR("getCamList called but scene is not set");
        return {};
    }

    void TrainerManager::applyPendingParams() {
        if (!trainer_)
            return;

        auto params = trainer_->getParams();
        params.dataset = pending_dataset_params_;

        // Use ParameterManager in GUI mode, fallback to pending_opt_params_ for headless
        if (auto* const param_mgr = services().paramsOrNull()) {
            params.optimization = param_mgr->getActiveParams();
            LOG_DEBUG("Applied params: strategy={}, iter={}, max_cap={}",
                      params.optimization.strategy, params.optimization.iterations, params.optimization.max_cap);
        } else {
            params.optimization = pending_opt_params_;
        }
        trainer_->setParams(params);
    }

} // namespace lfs::vis