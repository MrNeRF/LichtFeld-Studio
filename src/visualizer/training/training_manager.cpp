/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "training/training_manager.hpp"
#include "core/error.hpp"
#include "core/error_envelope.hpp"
#include "core/error_reporter.hpp"
#include "core/events.hpp"
#include "core/guarded_task.hpp"
#include "core/logger.hpp"
#include "core/parameter_manager.hpp"
#include "core/reactive/store.hpp"
#include "core/scene.hpp"
#include "core/services.hpp"
#include "core/shareable_allocation_limit.hpp"
#include "core/tensor.hpp"
#include "core/tensor/internal/tensor_ops.hpp"
#include "python/python_runtime.hpp"
#include "rendering/vulkan_external_tensor.hpp"
#include "training/control/command_api.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include "training/rasterization/gsplat/Ops.h"
#include "training/rasterization/gsplat_rasterizer.hpp"
#include "training/training_setup.hpp"
#include "visualizer/app_store.hpp"
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

        [[nodiscard]] lfs::io::project::TrainingFinishReason
        toIoFinishReason(const FinishReason reason) {
            switch (reason) {
            case FinishReason::Completed:
                return lfs::io::project::TrainingFinishReason::Completed;
            case FinishReason::UserStopped:
                return lfs::io::project::TrainingFinishReason::UserStopped;
            case FinishReason::Error:
                return lfs::io::project::TrainingFinishReason::Error;
            case FinishReason::None:
                break;
            }
            return lfs::io::project::TrainingFinishReason::None;
        }

        [[nodiscard]] FinishReason
        fromIoFinishReason(
            const lfs::io::project::TrainingFinishReason reason) {
            switch (reason) {
            case lfs::io::project::TrainingFinishReason::Completed:
                return FinishReason::Completed;
            case lfs::io::project::TrainingFinishReason::UserStopped:
                return FinishReason::UserStopped;
            case lfs::io::project::TrainingFinishReason::Error:
                return FinishReason::Error;
            case lfs::io::project::TrainingFinishReason::None:
                break;
            }
            return FinishReason::None;
        }

        template <typename Fn>
        class ScopeExit final {
        public:
            explicit ScopeExit(Fn fn) : fn_(std::move(fn)) {}
            ScopeExit(const ScopeExit&) = delete;
            ScopeExit& operator=(const ScopeExit&) = delete;

            ~ScopeExit() noexcept {
                if (active_) {
                    fn_();
                }
            }

            void release() noexcept { active_ = false; }

        private:
            Fn fn_;
            bool active_ = true;
        };

        void release_training_thread_local_cuda_caches() noexcept {
            (void)lfs::training::release_fast_rasterizer_thread_local_caches();
            (void)lfs::training::release_gsplat_rasterizer_thread_local_caches();
            (void)gsplat_lfs::release_intersect_thread_local_cache();
            (void)lfs::core::tensor_ops::release_nan_check_thread_buffers();
            // sort workspaces — explicit release before thread join so
            // high-water VRAM is not held until TLS dtor races CUDA teardown.
            lfs::training::release_fastgs_sort_workspace_buffers();
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
                if (keepFloatShNInPooledCuda(debug_name, dtype)) {
                    auto pooled = lfs::core::Tensor::zeros_direct(
                        std::move(shape), capacity, lfs::core::Device::CUDA, dtype);
                    pooled.set_name(debug_name);
                    return pooled;
                }
                auto tensor = makeVulkanExternalTensor(
                    *context,
                    std::move(shape),
                    dtype,
                    capacity,
                    debug_name.c_str());
                if (!tensor) {
                    const auto message = std::format(
                        "Vulkan-external training tensor allocation failed for '{}': {}",
                        debug_name,
                        tensor.error());
                    if (lfs::core::is_shareable_allocation_limit_message(tensor.error())) {
                        throw lfs::core::ShareableAllocationLimitError(message);
                    }
                    throw lfs::core::TensorError(message);
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
        splat_interop_allocator_ = {};
        splat_interop_parent_.reset();
        splat_storage_.reset();
        lfs::core::SplatTensorAllocator tensor_allocator;

        const std::size_t configured_capacity =
            params.optimization.max_cap > 0
                ? static_cast<std::size_t>(params.optimization.max_cap)
                : 0;
        const int sh_degree = params.optimization.sh_degree;

        // size the exportable block to live N (+ 1.5× headroom), not
        // max_cap. Virtual-reserve max_cap so densify can grow in place.
        std::size_t live_estimate = min_capacity;
        if (live_estimate == 0 && scene_) {
            if (const auto* model = scene_->getTrainingModel()) {
                live_estimate = static_cast<std::size_t>(model->size());
            } else if (const auto pc = scene_->getInitialPointCloud()) {
                live_estimate = static_cast<std::size_t>(pc->size());
            } else {
                for (const auto* node : scene_->getNodes()) {
                    if (node && node->type == lfs::core::NodeType::POINTCLOUD && node->point_cloud) {
                        live_estimate = static_cast<std::size_t>(node->point_cloud->size());
                        break;
                    }
                }
            }
        }
        if (live_estimate == 0 && params.optimization.random) {
            live_estimate = static_cast<std::size_t>(
                std::max(params.optimization.init_num_pts, 1));
        }
        if (live_estimate == 0) {
            live_estimate = 1;
        }

        const std::size_t exportable_capacity =
            lfs::core::SplatExportableStorage::growthCapacity(live_estimate, configured_capacity);
        std::size_t reserve_capacity = configured_capacity;
        if (reserve_capacity == 0) {
            std::size_t free_mem = 0;
            std::size_t total_mem = 0;
            if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess && total_mem > 0) {
                const std::size_t per_splat =
                    lfs::core::SplatExportableStorage::layoutBytesPerSplat(sh_degree);
                reserve_capacity = total_mem / std::max<std::size_t>(per_splat, 1);
                reserve_capacity = std::min(reserve_capacity, std::size_t{0x7fffffff});
            }
            reserve_capacity = std::max(reserve_capacity, exportable_capacity);
            LOG_INFO("Exportable splat reserve derived from device memory: reserve_capacity={} "
                     "(exportable_capacity={}, sh_degree={})",
                     reserve_capacity,
                     exportable_capacity,
                     sh_degree);
        }

        VulkanContext* vk_ctx = nullptr;
        if (viewer_ && viewer_->getWindowManager()) {
            vk_ctx = viewer_->getWindowManager()->getVulkanContext();
        }
        const bool vulkan_interop_available =
            vk_ctx && vk_ctx->externalMemoryInteropEnabled();

        if (vulkan_interop_available && exportable_capacity > 0) {
            auto storage_result = lfs::core::SplatExportableStorage::create(
                exportable_capacity, sh_degree, /*device=*/0, reserve_capacity);
            if (storage_result) {
                splat_storage_ = std::move(*storage_result);
                auto interop_alloc_result = makeSplatExportableInteropAllocator(
                    *vk_ctx, *splat_storage_, &splat_interop_parent_);
                if (interop_alloc_result) {
                    splat_interop_allocator_ = std::move(*interop_alloc_result);
                    tensor_allocator = splat_interop_allocator_;
                    LOG_INFO("Training tensors share one CUDA-exportable VMM block "
                             "imported into Vulkan (live≈{}, capacity={}, reserve={}, "
                             "sh_degree={}, committed={} MiB reserved={} MiB chunks={}) "
                             "— zero-copy viewer interop during live-N growth",
                             live_estimate,
                             exportable_capacity,
                             reserve_capacity,
                             sh_degree,
                             splat_storage_->block->committed_bytes >> 20,
                             splat_storage_->block->reserved_bytes >> 20,
                             splat_storage_->block->chunks.size());
                } else {
                    LOG_WARN("Exportable-interop allocator failed ({}); dropping storage "
                             "and falling back to legacy Vulkan-external allocator",
                             interop_alloc_result.error());
                    splat_interop_parent_.reset();
                    splat_storage_.reset();
                }
            } else if (lfs::core::is_shareable_allocation_limit_message(storage_result.error())) {
                LOG_WARN("SplatExportableStorage creation exceeded the shareable allocation "
                         "limit ({}); falling back to legacy Vulkan-external allocator",
                         storage_result.error());
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

    void TrainerManager::installExportableCapacityEnsure(lfs::core::SplatData& model) {
        if (!splat_storage_ || !splat_storage_->valid()) {
            return;
        }
        // Thin trampoline only: rebindSplatData assigns into the live SplatData and
        // would destroy a capturing std::function mid-call. The real work lives in
        // growExportableForDensify (member function, immune to that).
        model.set_capacity_ensure([this](std::size_t needed_rows) -> bool {
            return growExportableForDensify(needed_rows);
        });
    }

    void TrainerManager::installExportableDensifyBarrier() {
        if (!trainer_) {
            return;
        }
        if (!splat_storage_ || !splat_storage_->valid()) {
            trainer_->setExportableDensifyBarrier({}, {});
            return;
        }
        trainer_->setExportableDensifyBarrier(
            [this]() -> bool { return beginExportableDensifyBarrier(); },
            [this]() -> bool { return endExportableDensifyBarrier(); });
    }

    bool TrainerManager::beginExportableDensifyBarrier() {
        if (!splat_storage_ || !splat_storage_->valid()) {
            return false;
        }
        if (exportable_densify_barrier_depth_ > 0) {
            ++exportable_densify_barrier_depth_;
            return true;
        }
        // Device-sync under render_mutex exclusive + waitForModelReaders.
        // Full cuda-only↔Vulkan rebind is reserved for capacity grow (physical
        // remap). Generation-checked bind handles protect FastGS and Adam
        // readers from stale pointers during densification.
        if (const cudaError_t err = cudaDeviceSynchronize(); err != cudaSuccess) {
            LOG_ERROR("cudaDeviceSynchronize before densify exportable barrier failed: {} ({})",
                      cudaGetErrorName(err),
                      cudaGetErrorString(err));
            return false;
        }
        exportable_densify_barrier_depth_ = 1;
        return true;
    }

    bool TrainerManager::endExportableDensifyBarrier() {
        if (exportable_densify_barrier_depth_ <= 0) {
            return true;
        }
        --exportable_densify_barrier_depth_;
        if (exportable_densify_barrier_depth_ > 0) {
            return true;
        }
        if (const cudaError_t err = cudaDeviceSynchronize(); err != cudaSuccess) {
            LOG_ERROR("cudaDeviceSynchronize after densify exportable barrier failed: {} ({})",
                      cudaGetErrorName(err),
                      cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool TrainerManager::growExportableForDensify(std::size_t needed_rows) {
        if (!splat_storage_ || !splat_storage_->valid()) {
            return false;
        }
        if (splat_storage_->capacity() >= needed_rows) {
            return true;
        }
        const std::size_t want = lfs::core::SplatExportableStorage::growthCapacity(
            needed_rows, splat_storage_->reservedCapacity());

        auto* model_ptr = scene_ ? scene_->getTrainingModel() : nullptr;
        if (!model_ptr) {
            return false;
        }

        const std::size_t old_capacity = splat_storage_->capacity();
        const auto old_bytes = splat_storage_->region_bytes;
        const std::uint64_t old_generation = splat_storage_->generation();

        if (const cudaError_t err = cudaDeviceSynchronize(); err != cudaSuccess) {
            LOG_ERROR("cudaDeviceSynchronize before exportable grow failed: {} ({})",
                      cudaGetErrorName(err),
                      cudaGetErrorString(err));
            return false;
        }

        auto grew = splat_storage_->grow(want);
        if (!grew) {
            LOG_ERROR("Exportable splat grow failed (need={}): {}", needed_rows, grew.error());
            return false;
        }
        if (splat_storage_->capacity() < needed_rows) {
            LOG_ERROR("Exportable splat grow left capacity {} < needed {}",
                      splat_storage_->capacity(),
                      needed_rows);
            splat_storage_->restoreCapacity(old_capacity, old_bytes, old_generation);
            return false;
        }

        if (splat_interop_parent_) {
            if (!splat_interop_parent_->bindNewExportableChunks(*splat_storage_->block)) {
                LOG_ERROR("Exportable Vulkan bindNewChunks after grow failed; restoring capacity");
                splat_storage_->restoreCapacity(old_capacity, old_bytes, old_generation);
                return false;
            }
        }

        model_ptr = scene_ ? scene_->getTrainingModel() : nullptr;
        if (!model_ptr) {
            splat_storage_->restoreCapacity(old_capacity, old_bytes, old_generation);
            return false;
        }

        const auto alloc = splat_interop_allocator_ ? splat_interop_allocator_
                                                    : splat_storage_->make_allocator();
        if (auto ok = splat_storage_->rebindSplatData(*model_ptr, alloc); !ok) {
            LOG_ERROR("Exportable rebind after grow failed: {}", ok.error());
            splat_storage_->restoreCapacity(old_capacity, old_bytes, old_generation);
            return false;
        }
        installExportableCapacityEnsure(*model_ptr);
        if (trainer_) {
            trainer_->setSplatTensorAllocator(alloc);
        }
        LOG_INFO("Exportable splat storage grew for densify: capacity={} committed={} MiB "
                 "gen={} chunks={} (appended/bound, no re-import)",
                 splat_storage_->capacity(),
                 splat_storage_->block->committed_bytes >> 20,
                 splat_storage_->generation(),
                 splat_storage_->block->chunks.size());
        model_ptr = scene_ ? scene_->getTrainingModel() : nullptr;
        return model_ptr && model_ptr->means_raw().capacity() >= needed_rows;
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
                last_training_error_.clear();
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
        {
            // Publish the stop request under the predicate mutex so it cannot
            // race between the reaper's predicate check and its wait.
            std::lock_guard lock(training_thread_mutex_);
            completion_reaper_.request_stop();
        }
        training_thread_cv_.notify_all();
        if (completion_reaper_.joinable()) {
            completion_reaper_.join();
        }
        if (trainer_) {
            lfs::training::CommandCenter::instance().reset_snapshot();
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
            // A new training run has no resumable elapsed-time authority.
            clearRestoredProjectMetrics();
            accumulated_training_time_ =
                std::chrono::steady_clock::duration{0};
            checkpoint_baseline_iteration_.reset();

            std::lock_guard<std::mutex> lock(trainer_lifetime_mutex_);
            trainer_ = std::move(trainer);
            // One-lock: Scene live-model readers (cache rebuild, status) share the
            // trainer step-boundary mutex with densify commit/trim and preview draw.
            if (scene_ && trainer_) {
                scene_->setLiveModelMutex(&trainer_->getRenderMutex());
            }
            if (!state_machine_.transitionTo(TrainingState::Ready)) {
                LOG_WARN("Failed to transition to Ready");
            }

            internal::TrainerReady{}.emit();
        }
        if (viewer_) {
            viewer_->bindTrainerProjectSnapshotTarget();
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
            // METR may already have been applied (panels-ready before
            // hydration). Keep that elapsed time; otherwise start at zero
            // until restoreProjectMetrics runs.
            accumulated_training_time_ =
                restored_accumulated_training_time_.value_or(
                    std::chrono::steady_clock::duration{0});
            checkpoint_baseline_iteration_ = checkpoint_iteration;

            {
                std::lock_guard<std::mutex> lock(trainer_lifetime_mutex_);
                trainer_ = std::move(trainer);
                if (scene_ && trainer_) {
                    scene_->setLiveModelMutex(&trainer_->getRenderMutex());
                }
            }
            internal::TrainerReady{}.emit();

            const FinishReason finish_reason =
                resolvedRestoredFinishReason();
            if (finish_reason != FinishReason::None) {
                if (!state_machine_.transitionToFinished(finish_reason)) {
                    LOG_WARN("Failed to transition restored trainer to Finished");
                }
            } else {
                if (!state_machine_.transitionTo(TrainingState::Paused)) {
                    LOG_WARN("Failed to transition to Paused");
                }

                state::TrainingPaused{.iteration = checkpoint_iteration}.emit();
                LOG_DEBUG("Trainer paused from checkpoint (iteration {})", checkpoint_iteration);
            }
            applyRestoredCheckpointPresentation();
        }
        if (viewer_) {
            viewer_->bindTrainerProjectSnapshotTarget();
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
            suppressCompletionNotification();
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

        // Pause events and no-thread stops do not run TrainingEnd's clear_snapshot.
        lfs::training::CommandCenter::instance().reset_snapshot();

        {
            std::lock_guard<std::mutex> lock(trainer_lifetime_mutex_);
            if (scene_) {
                scene_->setLiveModelMutex(nullptr);
            }
            trainer_.reset();
            // Model tensors retain their own shared ownership while edit/view mode
            // still uses the exportable block. The manager must not remain the final
            // owner after scene teardown.
            splat_interop_allocator_ = {};
            splat_interop_parent_.reset();
            splat_storage_.reset();
        }
        checkpoint_baseline_iteration_.reset();
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

    bool TrainerManager::hasLiveTrainingThread() const {
        // stopTraining's no-thread branch uses this same flag: the reaper
        // steals training_thread_ immediately, so joinable() is not the
        // live-worker signal.
        return isCompletionPending();
    }

    bool TrainerManager::isPausedAtCheckpointBaseline() const {
        if (!trainer_ || !checkpoint_baseline_iteration_) {
            return false;
        }
        if (isCompletionPending()) {
            return false;
        }
        if (!isPaused()) {
            return false;
        }
        return getCurrentIteration() == *checkpoint_baseline_iteration_;
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
            lfs::Error typed = lfs::make_legacy_error(error, lfs::LegacyErrorContext{
                                                                 .code = lfs::ErrorCode::InvalidArgument,
                                                                 .domain = lfs::ErrorDomain::Training,
                                                                 .operation = "training.start",
                                                                 .source = LFS_SOURCE_SITE_CURRENT(),
                                                                 .operation_id = lfs::OperationId::generate(),
                                                             });
            state::TrainingCompleted{
                .iteration = 0,
                .final_loss = 0.0f,
                .elapsed_seconds = 0.0f,
                .success = false,
                .user_stopped = false,
                .error = last_error_,
                .error_info = core::to_wire_error(typed)}
                .emit();
            last_training_error_.set(std::move(typed));
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
                        lfs::Error typed = lfs::make_legacy_error(result.error(), lfs::LegacyErrorContext{
                                                                                      .code = lfs::ErrorCode::FailedPrecondition,
                                                                                      .domain = lfs::ErrorDomain::Training,
                                                                                      .operation = "training.start",
                                                                                      .source = LFS_SOURCE_SITE_CURRENT(),
                                                                                      .operation_id = lfs::OperationId::generate(),
                                                                                  });
                        state::TrainingCompleted{
                            .iteration = getCurrentIteration(),
                            .final_loss = 0.0f,
                            .elapsed_seconds = 0.0f,
                            .success = false,
                            .user_stopped = false,
                            .error = last_error_,
                            .error_info = core::to_wire_error(typed)}
                            .emit();
                        last_training_error_.set(std::move(typed));
                        if (!state_machine_.transitionToFinished(FinishReason::Error)) {
                            LOG_WARN("Failed to transition to Finished(Error)");
                        }
                        return false;
                    }
                    installExportableCapacityEnsure(*model);
                    installExportableDensifyBarrier();
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

                    lfs::Error typed = lfs::make_legacy_error(last_error_, lfs::LegacyErrorContext{
                                                                               .code = lfs::ErrorCode::FailedPrecondition,
                                                                               .domain = lfs::ErrorDomain::Training,
                                                                               .operation = "training.start",
                                                                               .source = LFS_SOURCE_SITE_CURRENT(),
                                                                               .operation_id = lfs::OperationId::generate(),
                                                                           });

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
                        .error = error_msg,
                        .error_info = core::to_wire_error(typed)}
                        .emit();
                    last_training_error_.set(std::move(typed));

                    if (!state_machine_.transitionToFinished(FinishReason::Error)) {
                        LOG_WARN("Failed to transition to Finished(Error)");
                    }
                    return false;
                }
                lfs::core::Tensor::log_storage_memory("After training model initialization");
                if (auto* model = scene_->getTrainingModel()) {
                    installExportableCapacityEnsure(*model);
                    installExportableDensifyBarrier();
                }
            }

            if (auto result = trainer_->initialize(params); !result) {
                LOG_ERROR("Failed to initialize trainer: {}", result.error());
                last_error_ = result.error();

                lfs::Error typed = lfs::make_legacy_error(last_error_, lfs::LegacyErrorContext{
                                                                           .code = lfs::ErrorCode::FailedPrecondition,
                                                                           .domain = lfs::ErrorDomain::Training,
                                                                           .operation = "training.start",
                                                                           .source = LFS_SOURCE_SITE_CURRENT(),
                                                                           .operation_id = lfs::OperationId::generate(),
                                                                       });

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
                    .error = error_msg,
                    .error_info = core::to_wire_error(typed)}
                    .emit();
                last_training_error_.set(std::move(typed));

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
        suppress_completion_notification_.store(false, std::memory_order_relaxed);

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

    bool TrainerManager::requestSaveProject() {
        if (viewer_) {
            const bool dispatched = viewer_->postWork({
                .run = [viewer = viewer_] {
                    if (auto saved = viewer->projectSave(true);
                        !saved) {
                        LOG_ERROR(
                            "Project save failed: {}",
                            lfs::format_for_developer(
                                saved.error()));
                    }
                },
                .cancel = {},
            });
            if (!dispatched) {
                LOG_WARN("Project save request dropped during viewer shutdown");
            }
            return dispatched;
        }

        if (trainer_ && isTrainingActive() &&
            trainer_->bound_project_path()) {
            static_cast<void>(
                trainer_
                    ->request_project_save());
            LOG_INFO("Project save requested at iteration {}", getCurrentIteration());
            return true;
        }
        LOG_WARN("Cannot save project snapshot - training not active or no project destination is bound");
        return false;
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
        suppress_completion_notification_.store(false, std::memory_order_relaxed);
        {
            std::lock_guard lock(completion_mutex_);
            training_joined_ = false;
            pending_completion_.reset();
        }
        completion_pending_.store(true, std::memory_order_release);

        last_training_error_.clear();
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
                .error = std::move(completion.error),
                .resource_exhausted = completion.resource_exhausted,
                .error_info = completion.typed_error
                                  ? std::optional(core::to_wire_error(*completion.typed_error))
                                  : std::nullopt,
                .suppress_notification = suppress_completion_notification_.exchange(false, std::memory_order_relaxed)}
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

    void TrainerManager::setSaveSteps(std::vector<size_t> save_steps) {
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
        return std::chrono::duration<float>(
                   accumulated_training_time_)
            .count();
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
        std::lock_guard<std::mutex> lock(eval_metrics_mutex_);
        last_eval_metrics_ = EvaluationMetricsSnapshot{
            .iteration = iteration,
            .psnr = psnr,
            .ssim = ssim};
        const auto position = std::lower_bound(
            evaluation_history_.begin(),
            evaluation_history_.end(), iteration,
            [](const EvaluationMetricsSnapshot& sample,
               const int target_iteration) {
                return sample.iteration <
                       target_iteration;
            });
        if (position != evaluation_history_.end() &&
            position->iteration == iteration) {
            *position = *last_eval_metrics_;
        } else {
            evaluation_history_.insert(
                position, *last_eval_metrics_);
        }
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
            evaluation_history_.clear();
        }
    }

    lfs::io::project::MetricsChapter
    TrainerManager::captureProjectMetrics() const {
        using lfs::io::project::LastEvaluationMetrics;
        using lfs::io::project::MetricHistorySample;

        lfs::io::project::MetricsChapter result;
        const auto loss =
            lfs::training::CommandCenter::instance()
                .loss_history();
        result.loss_history.reserve(loss.size());
        for (const auto& sample : loss) {
            result.loss_history.push_back(
                MetricHistorySample{
                    .iteration = sample.iteration,
                    .value = sample.loss,
                });
        }
        {
            std::lock_guard<std::mutex> lock(
                eval_metrics_mutex_);
            result.psnr_history.reserve(
                evaluation_history_.size());
            for (const auto& sample :
                 evaluation_history_) {
                result.psnr_history.push_back(
                    MetricHistorySample{
                        .iteration =
                            sample.iteration,
                        .value = sample.psnr,
                    });
            }
            if (last_eval_metrics_) {
                result.last_evaluation =
                    LastEvaluationMetrics{
                        .iteration =
                            last_eval_metrics_
                                ->iteration,
                        .psnr =
                            last_eval_metrics_->psnr,
                        .ssim =
                            last_eval_metrics_->ssim,
                    };
            }
        }
        result.accumulated_training_seconds =
            getElapsedSeconds();
        result.finish_reason =
            toIoFinishReason(state_machine_.getFinishReason());
        return result;
    }

    void TrainerManager::restoreProjectMetrics(
        const lfs::io::project::MetricsChapter&
            metrics) {
        std::vector<
            lfs::training::LossHistoryPoint>
            loss;
        loss.reserve(metrics.loss_history.size());
        for (const auto& sample :
             metrics.loss_history) {
            loss.push_back({
                .iteration = sample.iteration,
                .loss = sample.value,
            });
        }
        lfs::training::CommandCenter::instance()
            .replace_loss_history(std::move(loss));

        {
            std::lock_guard<std::mutex> lock(
                loss_buffer_mutex_);
            loss_buffer_.clear();
            const std::size_t begin =
                metrics.loss_history.size() >
                        static_cast<std::size_t>(
                            MAX_LOSS_POINTS)
                    ? metrics.loss_history.size() -
                          MAX_LOSS_POINTS
                    : 0;
            for (std::size_t index = begin;
                 index < metrics.loss_history.size();
                 ++index) {
                loss_buffer_.push_back(
                    metrics.loss_history[index]
                        .value);
            }
        }
        {
            std::scoped_lock lock(
                psnr_buffer_mutex_,
                eval_metrics_mutex_);
            psnr_buffer_.clear();
            evaluation_history_.clear();
            evaluation_history_.reserve(
                metrics.psnr_history.size());
            const std::size_t begin =
                metrics.psnr_history.size() >
                        static_cast<std::size_t>(
                            MAX_PSNR_POINTS)
                    ? metrics.psnr_history.size() -
                          MAX_PSNR_POINTS
                    : 0;
            for (std::size_t index = 0;
                 index < metrics.psnr_history.size();
                 ++index) {
                const auto& sample =
                    metrics.psnr_history[index];
                evaluation_history_.push_back({
                    .iteration = sample.iteration,
                    .psnr = sample.value,
                    .ssim = 0.0f,
                });
                if (index >= begin)
                    psnr_buffer_.push_back(
                        sample.value);
            }
            if (metrics.last_evaluation) {
                last_eval_metrics_ = {
                    .iteration =
                        metrics.last_evaluation
                            ->iteration,
                    .psnr =
                        metrics.last_evaluation
                            ->psnr,
                    .ssim =
                        metrics.last_evaluation
                            ->ssim,
                };
                if (!evaluation_history_.empty() &&
                    evaluation_history_.back()
                            .iteration ==
                        last_eval_metrics_
                            ->iteration) {
                    evaluation_history_.back()
                        .ssim =
                        last_eval_metrics_->ssim;
                }
            } else {
                last_eval_metrics_.reset();
            }
        }
        accumulated_training_time_ =
            std::chrono::duration_cast<
                std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(
                    metrics
                        .accumulated_training_seconds));
        restored_accumulated_training_time_ =
            accumulated_training_time_;
        restored_finish_reason_ = metrics.finish_reason;
        restored_finish_published_ = false;
        if (trainer_) {
            applyRestoredCheckpointPresentation();
        }
    }

    void TrainerManager::clearRestoredProjectMetrics() {
        restored_accumulated_training_time_.reset();
        restored_finish_reason_.reset();
        restored_finish_published_ = false;
    }

    FinishReason TrainerManager::resolvedRestoredFinishReason() const {
        // UserStopped and Error are saved pauses: resume unless the run
        // already hit total. An error terminal save is a valid safe-point
        // snapshot; the persisted Error value is provenance, not a restore
        // directive. Only Completed still restores as Finished.
        if (restored_finish_reason_ &&
            *restored_finish_reason_ !=
                lfs::io::project::TrainingFinishReason::None &&
            *restored_finish_reason_ !=
                lfs::io::project::TrainingFinishReason::UserStopped &&
            *restored_finish_reason_ !=
                lfs::io::project::TrainingFinishReason::Error) {
            return fromIoFinishReason(*restored_finish_reason_);
        }
        int iteration = getCurrentIteration();
        if (checkpoint_baseline_iteration_ &&
            *checkpoint_baseline_iteration_ > iteration) {
            iteration = *checkpoint_baseline_iteration_;
        }
        const int total = getTotalIterations();
        if (total > 0 && iteration >= total) {
            return FinishReason::Completed;
        }
        return FinishReason::None;
    }

    void TrainerManager::applyRestoredCheckpointPresentation() {
        if (!trainer_) {
            return;
        }
        if (restored_accumulated_training_time_) {
            accumulated_training_time_ =
                *restored_accumulated_training_time_;
        }
        {
            std::lock_guard<std::mutex> lock(loss_buffer_mutex_);
            if (!loss_buffer_.empty()) {
                trainer_->restore_current_loss(loss_buffer_.back());
            }
        }
        const FinishReason finish_reason =
            resolvedRestoredFinishReason();
        if (finish_reason == FinishReason::None &&
            restored_finish_reason_ &&
            *restored_finish_reason_ ==
                lfs::io::project::TrainingFinishReason::
                    Error) {
            int iteration = getCurrentIteration();
            if (checkpoint_baseline_iteration_ &&
                *checkpoint_baseline_iteration_ >
                    iteration) {
                iteration =
                    *checkpoint_baseline_iteration_;
            }
            LOG_INFO(
                "Previous training run ended in an error; restoring as paused at iteration {}",
                iteration);
        }
        if (finish_reason != FinishReason::None &&
            getState() != TrainingState::Finished) {
            if (!state_machine_.transitionToFinished(finish_reason)) {
                LOG_WARN(
                    "Failed to install restored finish state {}",
                    static_cast<int>(finish_reason));
            }
        }
        if (finish_reason != FinishReason::None &&
            getState() == TrainingState::Finished &&
            !restored_finish_published_) {
            restored_finish_published_ = true;
            suppress_completion_notification_.store(
                true, std::memory_order_relaxed);
            state::TrainingCompleted{
                .iteration = getCurrentIteration(),
                .final_loss = getCurrentLoss(),
                .elapsed_seconds = getElapsedSeconds(),
                .success = finish_reason != FinishReason::Error,
                .user_stopped =
                    finish_reason == FinishReason::UserStopped,
                .error = std::nullopt,
                .resource_exhausted = false,
                .error_info = std::nullopt,
                .suppress_notification = true}
                .emit();
        }
        publishRestoredTrainingStore();
    }

    void TrainerManager::publishRestoredTrainingStore() {
        int iteration = getCurrentIteration();
        if (checkpoint_baseline_iteration_ &&
            *checkpoint_baseline_iteration_ > iteration) {
            iteration = *checkpoint_baseline_iteration_;
        }
        const int total_iterations = getTotalIterations();
        const float loss = getCurrentLoss();
        const int num_gaussians = getNumSplats();

        auto& store = app_store();
        lfs::core::reactive::BatchUpdate batch(store.store());
        store.iteration.set(iteration);
        store.total_iterations.set(total_iterations);
        store.loss.set(loss);
        store.num_gaussians.set(
            static_cast<std::int64_t>(num_gaussians));
        if (const auto last = getLastEvaluationMetrics()) {
            store.eval_psnr.set(last->psnr);
            store.eval_ssim.set(last->ssim);
        }

        if (!trainer_) {
            return;
        }
        lfs::training::CommandCenter::instance().update_snapshot(
            lfs::training::HookContext{
                .iteration = iteration,
                .loss = loss,
                .num_gaussians = static_cast<std::size_t>(
                    std::max(0, num_gaussians)),
                .trainer = trainer_.get()},
            total_iterations,
            isPaused(),
            isRunning(),
            false,
            lfs::training::TrainingPhase::Idle);
    }

    void TrainerManager::trainingThreadFunc(std::stop_token stop_token) {
        LOG_INFO("Training thread started");
        LOG_TIMER("Training execution");

        trainer_->setOnIterationStart([this] {
            if (auto* pm = services().paramsOrNull(); pm && pm->consumeDirty()) {
                applyPendingParams();
            }
        });

        lfs::core::run_guarded<void>(
            lfs::core::TaskContext{
                .name = "training-worker",
                .domain = lfs::ErrorDomain::Training,
                .operation_id = lfs::OperationId::generate(),
                .site = LFS_SOURCE_SITE_CURRENT(),
            },
            [this, stop_token]() -> lfs::Result<void> {
                LOG_DEBUG("Starting trainer->train() with stop token");
                return trainer_->train(stop_token);
            },
            [this](lfs::Result<void>&& result) {
                if (result) {
                    LOG_INFO("Training {}",
                             trainer_->has_stopped() ? "stopped by user" : "completed successfully");
                    handleTrainingComplete(true);
                } else {
                    const auto& error = result.error();
                    const std::string message =
                        error.user_message().empty() ? std::string(error.detail())
                                                     : std::string(error.user_message());
                    LOG_ERROR("Training failed: {}", message);
                    lfs::core::ErrorReporter::get().report(error, lfs::core::ReportChannel::OwnerLog);
                    handleTrainingComplete(
                        false, message,
                        error.code() == lfs::ErrorCode::ResourceExhausted, error);
                }
            });

        release_training_thread_local_cuda_caches();

        LOG_INFO("Training thread finished");
    }

    void TrainerManager::handleTrainingComplete(const bool success, const std::string& error,
                                                const bool resource_exhausted,
                                                const std::optional<lfs::Error>& typed_error) {
        if (!error.empty()) {
            last_error_ = error;
            LOG_ERROR("Training error: {}", error);
        }
        if (typed_error) {
            last_training_error_.set(*typed_error);
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
                .resource_exhausted = resource_exhausted,
                .reason = reason,
                .error = error.empty() ? std::nullopt : std::optional(error),
                .typed_error = typed_error};
        }
    }

    void TrainerManager::setupEventHandlers() {
        using namespace lfs::core::events;

        lfs::training::CommandCenter::instance().bind_state_events();

        // Training control commands
        cmd::StartTraining::when([this](const auto&) {
            if (viewer_) {
                if (auto result = viewer_->startTraining();
                    !result) {
                    LOG_ERROR(
                        "Failed to start training: {}",
                        result.error());
                }
                return;
            }
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
                auto params = trainer_->getParams();
                params.optimization.save_steps = param_mgr->copyActiveParams().save_steps;
                trainer_->setParams(params);
                param_mgr->importTrainingParams(params);
            }
            LOG_DEBUG("Ignoring parameter updates for checkpoint-backed trainer (save steps kept)");
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
