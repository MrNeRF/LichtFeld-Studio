/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training/training_manager.hpp"
#include "core/events.hpp"
#include "core/logger.hpp"
#include "core/parameter_manager.hpp"
#include "core/scene.hpp"
#include "core/services.hpp"
#include "core/tensor.hpp"
#include "python/python_runtime.hpp"
#include "rendering/vulkan_external_tensor.hpp"
#include "training/training_setup.hpp"
#include "visualizer/visualizer_impl.hpp"
#include "window/vulkan_context.hpp"
#include "window/window_manager.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>
#include <format>
#include <stdexcept>
#include <thread>
#include <utility>

namespace lfs::vis {

    using namespace lfs::core::events;

    namespace {
        [[nodiscard]] std::vector<size_t> normalize_save_steps(std::vector<size_t> steps) {
            steps.erase(std::remove(steps.begin(), steps.end(), 0), steps.end());
            std::sort(steps.begin(), steps.end());
            steps.erase(std::unique(steps.begin(), steps.end()), steps.end());
            return steps;
        }

        void apply_save_steps(lfs::core::param::OptimizationParameters& params,
                              const std::vector<size_t>& steps) {
            params.save_steps = steps;
            if (params.enable_eval)
                params.eval_steps = steps;
        }

        [[nodiscard]] lfs::core::SplatTensorAllocator makeVulkanTrainingTensorAllocator(VisualizerImpl* viewer) {
            if (!viewer || !viewer->getWindowManager()) {
                return {};
            }
            auto* const context = viewer->getWindowManager()->getVulkanContext();
            if (!context || !context->externalMemoryInteropEnabled()) {
                return {};
            }

            return [context](lfs::core::TensorShape shape,
                             const size_t capacity,
                             const lfs::core::DataType dtype,
                             const std::string_view name) -> lfs::core::Tensor {
                const std::string debug_name{name};
                auto tensor = makeVulkanExternalTensor(
                    *context,
                    std::move(shape),
                    dtype,
                    capacity,
                    debug_name.c_str());
                if (!tensor) {
                    throw lfs::core::TensorError(std::format(
                        "Vulkan-external training tensor allocation failed for '{}': {}",
                        debug_name,
                        tensor.error()));
                }
                tensor->set_name(debug_name);
                return std::move(*tensor);
            };
        }
    } // namespace

    TrainerManager::TrainerManager() {
        setupEventHandlers();
        setupStateMachineCallbacks();
        completion_reaper_ = std::jthread([this](const std::stop_token stop_token) {
            completionReaperLoop(stop_token);
        });
        LOG_DEBUG("TrainerManager created");
    }

    lfs::core::SplatTensorAllocator TrainerManager::createTrainingSplatTensorAllocator(
        const lfs::core::param::TrainingParameters& params,
        const std::size_t min_capacity) {
        splat_storage_.reset();
        lfs::core::SplatTensorAllocator tensor_allocator;

        const std::size_t configured_capacity =
            params.optimization.max_cap > 0
                ? static_cast<std::size_t>(params.optimization.max_cap)
                : 0;
        const std::size_t exportable_capacity = std::max(configured_capacity, min_capacity);
        const int sh_degree = params.optimization.sh_degree;

        VulkanContext* vk_ctx = nullptr;
        if (viewer_ && viewer_->getWindowManager()) {
            vk_ctx = viewer_->getWindowManager()->getVulkanContext();
        }
        const bool vulkan_interop_available =
            vk_ctx && vk_ctx->externalMemoryInteropEnabled();

        if (vulkan_interop_available && exportable_capacity > 0) {
            auto storage_result =
                lfs::core::SplatExportableStorage::create(exportable_capacity, sh_degree);
            if (storage_result) {
                splat_storage_ = std::move(*storage_result);
                auto interop_alloc_result =
                    makeSplatExportableInteropAllocator(*vk_ctx, *splat_storage_);
                if (interop_alloc_result) {
                    tensor_allocator = std::move(*interop_alloc_result);
                    LOG_INFO("Training tensors share one CUDA-exportable VMM block "
                             "imported into Vulkan (capacity={}, sh_degree={}, "
                             "block={} MiB) — zero-copy viewer interop",
                             exportable_capacity,
                             sh_degree,
                             splat_storage_->block->size >> 20);
                } else {
                    LOG_WARN("Exportable-interop allocator failed ({}); dropping storage "
                             "and falling back to legacy Vulkan-external allocator",
                             interop_alloc_result.error());
                    splat_storage_.reset();
                }
            } else {
                LOG_WARN("SplatExportableStorage creation failed ({}); falling back to "
                         "legacy Vulkan-external allocator",
                         storage_result.error());
            }
        }

        if (!tensor_allocator) {
            tensor_allocator = makeVulkanTrainingTensorAllocator(viewer_);
            if (tensor_allocator) {
                LOG_INFO("Training model tensors will use Vulkan-external CUDA storage");
            }
        }

        return tensor_allocator;
    }

    void TrainerManager::setupStateMachineCallbacks() {
        state_machine_.setStateChangeCallback([this](TrainingState, TrainingState new_state) {
            // Emit events on state changes
            if (new_state == TrainingState::Idle) {
                {
                    std::lock_guard<std::mutex> lock(loss_buffer_mutex_);
                    loss_buffer_.clear();
                }
                clearEvaluationMetrics();
                last_error_.clear();
            }
        });
    }

    TrainerManager::~TrainerManager() {
        if (isCompletionPending()) {
            LOG_INFO("Stopping training thread during destruction...");
            if (canStop()) {
                stopTraining();
            } else if (trainer_) {
                trainer_->request_stop();
            }
            if (!waitForCompletion()) {
                LOG_WARN("Training worker exceeded the shutdown completion timeout");
            }
        }
        completion_reaper_.request_stop();
        training_thread_cv_.notify_all();
        if (completion_reaper_.joinable()) {
            completion_reaper_.join();
        }
    }

    void TrainerManager::setTrainer(std::unique_ptr<lfs::training::Trainer> trainer) {
        LOG_TIMER_TRACE("TrainerManager::setTrainer");

        if (!clearTrainer()) {
            LOG_ERROR("Cannot install trainer while the previous training worker is still stopping");
            return;
        }

        if (trainer) {
            const auto& params = trainer->getParams();
            pending_opt_params_ = params.optimization;
            pending_dataset_params_ = params.dataset;

            std::lock_guard<std::mutex> lock(trainer_lifetime_mutex_);
            trainer_ = std::move(trainer);
            if (!state_machine_.transitionTo(TrainingState::Ready)) {
                LOG_WARN("Failed to transition to Ready");
            }

            internal::TrainerReady{}.emit();
        }
    }

    void TrainerManager::setTrainerFromCheckpoint(std::unique_ptr<lfs::training::Trainer> trainer, int checkpoint_iteration) {
        LOG_TIMER_TRACE("TrainerManager::setTrainerFromCheckpoint");

        if (!clearTrainer()) {
            LOG_ERROR("Cannot install checkpoint trainer while the previous training worker is still stopping");
            return;
        }

        if (trainer) {
            const auto& params = trainer->getParams();
            pending_opt_params_ = params.optimization;
            pending_dataset_params_ = params.dataset;

            std::lock_guard<std::mutex> lock(trainer_lifetime_mutex_);
            trainer_ = std::move(trainer);
            internal::TrainerReady{}.emit();

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

    bool TrainerManager::clearTrainer() {
        LOG_DEBUG("Clearing trainer");

        const auto state = getState();
        if (state == TrainingState::Running || state == TrainingState::Paused) {
            LOG_INFO("Stopping active training before clearing");
            if (state == TrainingState::Paused && trainer_) {
                trainer_->request_resume();
            }
            stopTraining();
        }

        if (isCompletionPending()) {
            if (viewer_ && viewer_->isOnViewerThread()) {
                LOG_ERROR("Trainer clear deferred until the training completion event");
                return false;
            }
            LOG_INFO("Waiting for training thread before clearing trainer");
            if (!waitForCompletion()) {
                LOG_ERROR("Trainer clear deferred: training worker did not reach its terminal state");
                return false;
            }
        }

        {
            std::lock_guard<std::mutex> lock(trainer_lifetime_mutex_);
            trainer_.reset();
            // Model tensors retain their own shared ownership while edit/view mode
            // still uses the exportable block. The manager must not remain the final
            // owner after scene teardown.
            splat_storage_.reset();
        }
        // Trainer::shutdown() trims before Tensor-valued members are destroyed.
        // Trim again after destruction so those returned blocks do not survive clear.
        lfs::core::Tensor::trim_memory_pool();

        if (getState() != TrainingState::Idle && !state_machine_.transitionTo(TrainingState::Idle)) {
            LOG_WARN("Failed to transition to Idle");
        }

        python::update_training_state(false, "idle");
        python::update_trainer_loaded(false, 0);
        LOG_INFO("Trainer cleared");
        return true;
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

        clearEvaluationMetrics();
        applyPendingParams();

        if (auto error = trainer_->getParams().validate(); !error.empty()) {
            LOG_ERROR("Cannot start training: {}", error);
            last_error_ = error;
            state::TrainingCompleted{
                .iteration = 0,
                .final_loss = 0.0f,
                .elapsed_seconds = 0.0f,
                .success = false,
                .user_stopped = false,
                .error = last_error_}
                .emit();
            if (!state_machine_.transitionToFinished(FinishReason::Error)) {
                LOG_WARN("Failed to transition to Finished(Error)");
            }
            return false;
        }

        if (trainer_->isInitialized()) {
            const auto& params = trainer_->getParams();
            auto* model = scene_ ? scene_->getTrainingModel() : nullptr;
            const std::size_t model_size = model ? static_cast<std::size_t>(model->size()) : 0;
            auto tensor_allocator = scene_ ? createTrainingSplatTensorAllocator(params, model_size)
                                           : lfs::core::SplatTensorAllocator{};
            const bool force_reallocation = splat_storage_.has_value();
            if (scene_ && tensor_allocator) {
                trainer_->setSplatTensorAllocator(tensor_allocator);
                if (model) {
                    if (auto result = lfs::training::migrateTrainingModelToAllocator(
                            params, *model, tensor_allocator, force_reallocation);
                        !result) {
                        LOG_ERROR("Failed to migrate initialized training model: {}", result.error());
                        last_error_ = result.error();
                        state::TrainingCompleted{
                            .iteration = getCurrentIteration(),
                            .final_loss = 0.0f,
                            .elapsed_seconds = 0.0f,
                            .success = false,
                            .user_stopped = false,
                            .error = last_error_}
                            .emit();
                        if (!state_machine_.transitionToFinished(FinishReason::Error)) {
                            LOG_WARN("Failed to transition to Finished(Error)");
                        }
                        return false;
                    }
                }
            }
            LOG_DEBUG("Resuming from iteration {}", trainer_->get_current_iteration());
        } else {
            const auto& params = trainer_->getParams();

            if (scene_) {
                auto tensor_allocator = createTrainingSplatTensorAllocator(params);
                trainer_->setSplatTensorAllocator(tensor_allocator);
                if (auto result = lfs::training::initializeTrainingModel(
                        params, *scene_, std::move(tensor_allocator));
                    !result) {
                    LOG_ERROR("Failed to initialize model: {}", result.error());
                    last_error_ = result.error();

                    std::string error_msg = result.error();
                    if (auto pos = error_msg.find("CUDA out of memory"); pos != std::string::npos) {
                        error_msg = error_msg.substr(pos);
                    }
                    state::TrainingCompleted{
                        .iteration = 0,
                        .final_loss = 0.0f,
                        .elapsed_seconds = 0.0f,
                        .success = false,
                        .user_stopped = false,
                        .error = error_msg}
                        .emit();

                    if (!state_machine_.transitionToFinished(FinishReason::Error)) {
                        LOG_WARN("Failed to transition to Finished(Error)");
                    }
                    return false;
                }
                lfs::core::Tensor::log_storage_memory("After training model initialization");
            }

            if (auto result = trainer_->initialize(params); !result) {
                LOG_ERROR("Failed to initialize trainer: {}", result.error());
                last_error_ = result.error();

                std::string error_msg = result.error();
                if (auto pos = error_msg.find("CUDA out of memory"); pos != std::string::npos) {
                    error_msg = error_msg.substr(pos);
                }
                state::TrainingCompleted{
                    .iteration = 0,
                    .final_loss = 0.0f,
                    .elapsed_seconds = 0.0f,
                    .success = false,
                    .user_stopped = false,
                    .error = error_msg}
                    .emit();

                if (!state_machine_.transitionToFinished(FinishReason::Error)) {
                    LOG_WARN("Failed to transition to Finished(Error)");
                }
                return false;
            }
            lfs::core::Tensor::log_storage_memory("After trainer initialization");

            // Match headless mode: release init-time cached pool allocations before the
            // first training batch spins up image decoders and render workspaces.
            lfs::core::Tensor::trim_memory_pool();
        }

        if (viewer_) {
            auto* const rendering_manager = viewer_->getRenderingManager();
            auto* const window_manager = viewer_->getWindowManager();
            auto* const vulkan_context = window_manager ? window_manager->getVulkanContext() : nullptr;
            auto* const model = scene_ ? scene_->getTrainingModel() : nullptr;
            if (rendering_manager && vulkan_context && model) {
                glm::ivec2 prime_size = rendering_manager->getRenderedSize();
                if (prime_size.x <= 0 || prime_size.y <= 0) {
                    prime_size = window_manager ? window_manager->getWindowSize() : glm::ivec2{1280, 720};
                }
                if (auto ok = rendering_manager->ensureVksplatTrainingSharedScratchReady(
                        *vulkan_context,
                        *model,
                        prime_size);
                    !ok) {
                    LOG_WARN("VkSplat training shared-scratch pre-start prime skipped: {}", ok.error());
                }
            }
        }

        if (!state_machine_.transitionTo(TrainingState::Running)) {
            LOG_WARN("Failed to transition to Running");
        }

        training_start_time_ = std::chrono::steady_clock::now();
        accumulated_training_time_ = std::chrono::steady_clock::duration{0};

        state::TrainingStarted{.total_iterations = getTotalIterations()}.emit();

        launchTrainingThread();

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
        const bool need_thread = !isCompletionPending();

        if (need_thread) {
            // Checkpoint resume: no thread exists yet
            accumulated_training_time_ = std::chrono::steady_clock::duration{0};
            launchTrainingThread();
        } else {
            trainer_->request_resume();
        }

        training_start_time_ = std::chrono::steady_clock::now();
        if (!state_machine_.transitionTo(TrainingState::Running)) {
            LOG_WARN("Failed to transition to Running");
        }

        state::TrainingResumed{.iteration = iter}.emit();
        LOG_INFO("Training resumed at iteration {}", iter);
    }

    void TrainerManager::pauseTrainingTemporary() {
        if (!isRunning() || !trainer_) {
            return;
        }

        const int iteration = getCurrentIteration();
        const bool was_paused = trainer_->is_paused();
        {
            std::lock_guard lock(temporary_pause_mutex_);
            if (temporary_pause_depth_ == 0) {
                temporary_pause_initially_paused_ = was_paused && !temporary_pause_resume_in_flight_;
                temporary_pause_resume_in_flight_ = false;
            }
            ++temporary_pause_depth_;
        }

        trainer_->request_pause();
        LOG_TRACE("Training temporary pause requested at iteration {}", iteration);
    }

    bool TrainerManager::pauseTrainingTemporaryIfActive() {
        if (!isRunning() || !trainer_ || trainer_->is_paused()) {
            return false;
        }

        pauseTrainingTemporary();
        return true;
    }

    TrainerManager::TemporaryPauseResult
    TrainerManager::pauseTrainingTemporaryAndWait(const std::chrono::milliseconds timeout) {
        if (!isRunning() || !trainer_) {
            return {};
        }

        const int start_iteration = getCurrentIteration();
        const bool was_paused = trainer_->is_paused();
        {
            std::lock_guard lock(temporary_pause_mutex_);
            if (temporary_pause_depth_ == 0) {
                temporary_pause_initially_paused_ = was_paused && !temporary_pause_resume_in_flight_;
                temporary_pause_resume_in_flight_ = false;
            }
            ++temporary_pause_depth_;
        }

        const auto release_failed_lease = [&]() -> bool {
            bool resume_training = false;
            bool initially_paused = false;
            const bool can_resume = isRunning() && trainer_ != nullptr;
            {
                std::lock_guard lock(temporary_pause_mutex_);
                if (temporary_pause_depth_ == 0) {
                    LOG_WARN("Temporary training pause lease release underflow");
                    return false;
                }
                initially_paused = temporary_pause_initially_paused_;
                --temporary_pause_depth_;
                resume_training = temporary_pause_depth_ == 0 && !initially_paused;
                if (temporary_pause_depth_ == 0) {
                    temporary_pause_initially_paused_ = false;
                    temporary_pause_resume_in_flight_ = resume_training && can_resume;
                }
            }
            return resume_training;
        };

        trainer_->request_pause();

        const auto pause_start = std::chrono::steady_clock::now();
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (isRunning() && !trainer_->is_paused()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                LOG_WARN("Timed out waiting for temporary training pause: start_iteration={}, current_iteration={}, waited_ms={}, was_paused={}",
                         start_iteration,
                         getCurrentIteration(),
                         timeout.count(),
                         was_paused);
                const bool resume_training = release_failed_lease();
                if (resume_training && isRunning() && trainer_) {
                    trainer_->request_resume();
                }
                return {};
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }

        if (!isRunning()) {
            (void)release_failed_lease();
            return {};
        }

        const auto paused_at = std::chrono::steady_clock::now();
        const double pause_wait_ms = std::chrono::duration<double, std::milli>(paused_at - pause_start).count();
        const auto sync_start = std::chrono::steady_clock::now();
        const cudaError_t sync_status = cudaDeviceSynchronize();
        const auto sync_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sync_start)
                .count();
        if (sync_status != cudaSuccess) {
            LOG_WARN("CUDA sync after temporary training pause failed: error={}, sync_ms={:.1f}, start_iteration={}, current_iteration={}",
                     cudaGetErrorString(sync_status),
                     sync_ms,
                     start_iteration,
                     getCurrentIteration());
            const bool resume_training = release_failed_lease();
            if (resume_training && isRunning() && trainer_) {
                trainer_->request_resume();
            }
            return {};
        }

        LOG_DEBUG("Training temporarily paused and synchronized: start_iteration={}, current_iteration={}, pause_wait_ms={:.1f}, sync_ms={:.1f}",
                  start_iteration,
                  getCurrentIteration(),
                  pause_wait_ms,
                  sync_ms);
        return {
            .synchronized = true,
            .resume_required = true,
        };
    }

    void TrainerManager::resumeTrainingTemporary() {
        const bool running = isRunning();
        const int iteration = getCurrentIteration();
        const bool trainer_present = trainer_ != nullptr;
        bool resume_training = false;
        bool root_initially_paused = false;
        {
            std::lock_guard lock(temporary_pause_mutex_);
            if (temporary_pause_depth_ == 0) {
                LOG_WARN("Temporary training resume ignored without active lease: iteration={}, running={}, trainer_present={}",
                         iteration,
                         running,
                         trainer_present);
                return;
            }
            root_initially_paused = temporary_pause_initially_paused_;
            --temporary_pause_depth_;
            resume_training = temporary_pause_depth_ == 0 && !root_initially_paused;
            if (temporary_pause_depth_ == 0) {
                temporary_pause_initially_paused_ = false;
                temporary_pause_resume_in_flight_ = resume_training && running && trainer_present;
            }
        }

        if (resume_training && running && trainer_) {
            trainer_->request_resume();
            LOG_TRACE("Training resumed from temporary pause at iteration {}", iteration);
        }
    }

    void TrainerManager::stopTraining() {
        if (!canStop()) {
            LOG_TRACE("Cannot stop: {}", getActionBlockedReason(TrainingAction::Stop));
            return;
        }

        LOG_DEBUG("Requesting training stop");
        if (!state_machine_.transitionTo(TrainingState::Stopping)) {
            LOG_WARN("Failed to transition to Stopping");
        }

        if (trainer_) {
            trainer_->request_stop();
        }

        const bool has_thread = isCompletionPending();
        std::optional<std::stop_source> stop_source;
        if (has_thread) {
            std::lock_guard lock(training_thread_mutex_);
            stop_source = training_stop_source_;
        }
        if (stop_source) {
            stop_source->request_stop();
        }

        state::TrainingStopped{.iteration = getCurrentIteration(), .user_requested = true}.emit();
        LOG_INFO("Training stop requested at iteration {}", getCurrentIteration());

        if (!has_thread) {
            handleTrainingComplete(true);
            finishTrainingThreadJoin();
        }
    }

    void TrainerManager::requestSaveCheckpoint() {
        if (trainer_ && isTrainingActive()) {
            trainer_->request_save();
            LOG_INFO("Checkpoint save requested at iteration {}", getCurrentIteration());
        } else {
            LOG_WARN("Cannot save checkpoint - training not active");
        }
    }

    bool TrainerManager::waitForCompletion() {
        std::unique_lock<std::mutex> lock(completion_mutex_);
        if (viewer_ && viewer_->isOnViewerThread() && !training_joined_) {
            LOG_ERROR("Refusing to block the viewer thread on training completion");
            return false;
        }
        if (!completion_cv_.wait_for(lock, std::chrono::seconds(COMPLETION_TIMEOUT_SEC),
                                     [this] { return training_joined_; })) {
            LOG_ERROR("Training thread join timed out ({}s)", COMPLETION_TIMEOUT_SEC);
            return false;
        }
        return true;
    }

    void TrainerManager::launchTrainingThread() {
        {
            std::lock_guard lock(completion_mutex_);
            training_joined_ = false;
            pending_completion_.reset();
        }
        completion_pending_.store(true, std::memory_order_release);

        auto worker = std::make_unique<std::jthread>(
            [this](const std::stop_token stop_token) {
                trainingThreadFunc(stop_token);
            });
        {
            std::lock_guard lock(training_thread_mutex_);
            training_stop_source_ = worker->get_stop_source();
            training_thread_ = std::move(worker);
        }
        training_thread_cv_.notify_one();
    }

    void TrainerManager::completionReaperLoop(const std::stop_token stop_token) {
        while (true) {
            std::unique_ptr<std::jthread> worker;
            {
                std::unique_lock lock(training_thread_mutex_);
                training_thread_cv_.wait(lock, [this, stop_token] {
                    return stop_token.stop_requested() || training_thread_ != nullptr;
                });
                if (!training_thread_) {
                    if (stop_token.stop_requested()) {
                        return;
                    }
                    continue;
                }
                worker = std::move(training_thread_);
            }

            if (worker->joinable()) {
                worker->join();
            }
            finishTrainingThreadJoin();
        }
    }

    void TrainerManager::finishTrainingThreadJoin() {
        std::optional<TrainingCompletionData> completion;
        {
            std::lock_guard lock(completion_mutex_);
            training_joined_ = true;
            completion = std::move(pending_completion_);
            pending_completion_.reset();
        }
        completion_cv_.notify_all();

        if (!completion) {
            completion_pending_.store(false, std::memory_order_release);
            LOG_ERROR("Training worker exited without terminal completion data");
            return;
        }
        dispatchTrainingCompleted(std::move(*completion));
    }

    void TrainerManager::dispatchTrainingCompleted(TrainingCompletionData completion) {
        auto emit_completion = [this, completion = std::move(completion)]() mutable {
            if (!state_machine_.transitionToFinished(completion.reason)) {
                LOG_WARN("Failed to transition to Finished");
            }
            LOG_INFO("Training finished: iter={}, loss={:.6f}, time={:.1f}s",
                     completion.iteration, completion.final_loss, completion.elapsed_seconds);
            completion_pending_.store(false, std::memory_order_release);
            state::TrainingCompleted{
                .iteration = completion.iteration,
                .final_loss = completion.final_loss,
                .elapsed_seconds = completion.elapsed_seconds,
                .success = completion.success,
                .user_stopped = completion.user_stopped,
                .error = std::move(completion.error)}
                .emit();
        };

        if (viewer_) {
            if (!viewer_->postWork({
                    .run = std::move(emit_completion),
                    .cancel = [this] {
                        completion_pending_.store(false, std::memory_order_release);
                    },
                })) {
                completion_pending_.store(false, std::memory_order_release);
                LOG_WARN("Training completion event dropped during viewer shutdown");
            }
            return;
        }
        emit_completion();
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
        return trainer_->get_total_iterations();
    }

    int TrainerManager::getNumSplats() const {
        if (!trainer_)
            return 0;

        // Prefer scene metadata so UI polling does not dereference the live
        // training model while topology-changing refinement is in progress.
        if (scene_) {
            return static_cast<int>(scene_->getTrainingModelGaussianCount());
        }

        // Legacy fallback for non-scene-backed trainers.
        if (trainer_->isInitialized()) {
            const std::shared_lock lock(trainer_->getRenderMutex());
            return static_cast<int>(trainer_->get_strategy().get_model().size());
        }
        return 0;
    }

    int TrainerManager::getMaxGaussians() const {
        if (!trainer_)
            return 0;
        return trainer_->getParams().optimization.max_cap;
    }

    std::vector<size_t> TrainerManager::getSaveSteps() const {
        if (auto* const param_mgr = services().paramsOrNull(); param_mgr && param_mgr->isLoaded())
            return param_mgr->copyActiveParams().save_steps;
        if (trainer_)
            return trainer_->getParams().optimization.save_steps;
        return pending_opt_params_.save_steps;
    }

    bool TrainerManager::canEditSaveSteps() const {
        return !trainer_ ||
               !trainer_->isInitialized() ||
               !trainer_->getParams().resume_checkpoint.has_value();
    }

    bool TrainerManager::setSaveSteps(std::vector<size_t> save_steps) {
        if (!canEditSaveSteps())
            return false;

        save_steps = normalize_save_steps(std::move(save_steps));
        apply_save_steps(pending_opt_params_, save_steps);

        bool updated_active_params = false;
        if (auto* const param_mgr = services().paramsOrNull()) {
            if (const auto loaded = param_mgr->ensureLoaded(); loaded) {
                param_mgr->modifyActiveParams([&save_steps](auto& params) {
                    apply_save_steps(params, save_steps);
                });
                updated_active_params = true;
            } else {
                LOG_WARN("Could not update save steps: {}", loaded.error());
            }
        }

        if (!updated_active_params && trainer_) {
            auto params = trainer_->getParams();
            apply_save_steps(params.optimization, save_steps);
            trainer_->setParams(params);
        }

        return true;
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

    void TrainerManager::updatePSNR(float psnr) {
        std::lock_guard<std::mutex> lock(psnr_buffer_mutex_);
        psnr_buffer_.push_back(psnr);
        while (psnr_buffer_.size() > static_cast<size_t>(MAX_PSNR_POINTS)) {
            psnr_buffer_.pop_front();
        }
    }

    std::deque<float> TrainerManager::getPSNRBuffer() const {
        std::lock_guard<std::mutex> lock(psnr_buffer_mutex_);
        return psnr_buffer_;
    }

    void TrainerManager::updateEvaluationMetrics(int iteration, float psnr, float ssim) {
        updatePSNR(psnr);
        setLastPSNR(psnr);
        std::lock_guard<std::mutex> lock(eval_metrics_mutex_);
        last_eval_metrics_ = EvaluationMetricsSnapshot{
            .iteration = iteration,
            .psnr = psnr,
            .ssim = ssim};
    }

    std::optional<TrainerManager::EvaluationMetricsSnapshot> TrainerManager::getLastEvaluationMetrics() const {
        std::lock_guard<std::mutex> lock(eval_metrics_mutex_);
        return last_eval_metrics_;
    }

    void TrainerManager::clearEvaluationMetrics() {
        {
            std::lock_guard<std::mutex> lock(psnr_buffer_mutex_);
            psnr_buffer_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(eval_metrics_mutex_);
            last_eval_metrics_.reset();
        }
        last_psnr_.store(0.0f);
    }

    void TrainerManager::trainingThreadFunc(std::stop_token stop_token) {
        LOG_INFO("Training thread started");
        LOG_TIMER("Training execution");

        try {
            trainer_->setOnIterationStart([this] {
                if (auto* pm = services().paramsOrNull(); pm && pm->consumeDirty()) {
                    applyPendingParams();
                }
            });

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

        if (!user_stopped) {
            if (!state_machine_.transitionTo(TrainingState::Stopping)) {
                LOG_WARN("Failed to transition to Stopping");
            }
        }

        const FinishReason reason = !success       ? FinishReason::Error
                                    : user_stopped ? FinishReason::UserStopped
                                                   : FinishReason::Completed;

        {
            std::lock_guard lock(completion_mutex_);
            pending_completion_ = TrainingCompletionData{
                .iteration = final_iter,
                .final_loss = final_loss,
                .elapsed_seconds = elapsed,
                .success = success,
                .user_stopped = user_stopped,
                .reason = reason,
                .error = error.empty() ? std::nullopt : std::optional(error)};
        }
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

        // Listen for evaluation completed events - update PSNR buffer
        state::EvaluationCompleted::when([this](const auto& event) {
            updateEvaluationMetrics(event.iteration, event.psnr, event.ssim);
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

    std::vector<std::shared_ptr<lfs::core::Camera>> TrainerManager::getCamList() const {
        if (scene_) {
            return scene_->getActiveCameras();
        }
        LOG_ERROR("getCamList called but scene is not set");
        return {};
    }

    std::vector<std::shared_ptr<lfs::core::Camera>> TrainerManager::getAllCamList() const {
        if (scene_) {
            return scene_->getAllCameras();
        }
        return {};
    }

    std::expected<lfs::training::Trainer::CameraMetricsSnapshot, std::string>
    TrainerManager::computeCameraMetricsForCameraId(
        const int camera_id,
        const bool include_ssim,
        const lfs::training::Trainer::CameraMetricsAppearanceConfig& appearance) const {
        std::lock_guard<std::mutex> lock(trainer_lifetime_mutex_);

        if (!trainer_) {
            return std::unexpected("trainer unavailable");
        }
        if (!scene_) {
            return std::unexpected("scene unavailable");
        }

        const auto cam = scene_->getCameraByUid(camera_id);
        if (!cam) {
            return std::unexpected(std::format("camera {} not found", camera_id));
        }

        return trainer_->computeCameraMetrics(*cam, include_ssim, appearance);
    }

    void TrainerManager::applyPendingParams() {
        if (!trainer_)
            return;

        if (trainer_->isInitialized() && trainer_->getParams().resume_checkpoint.has_value()) {
            if (auto* const param_mgr = services().paramsOrNull()) {
                param_mgr->importTrainingParams(trainer_->getParams());
            }
            LOG_DEBUG("Ignoring parameter updates for checkpoint-backed trainer");
            return;
        }

        auto params = trainer_->getParams();
        params.dataset = pending_dataset_params_;

        // Use ParameterManager in GUI mode, fallback to pending_opt_params_ for headless
        if (auto* const param_mgr = services().paramsOrNull()) {
            params.optimization = param_mgr->copyActiveParams();
            LOG_DEBUG("Applied params: strategy={}, iter={}, max_cap={}",
                      params.optimization.strategy, params.optimization.iterations, params.optimization.max_cap);
        } else {
            params.optimization = pending_opt_params_;
        }
        trainer_->setParams(params);
    }

} // namespace lfs::vis
