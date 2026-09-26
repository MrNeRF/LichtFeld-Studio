/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/training_manager.hpp"
#include "core/error.hpp"
#include "core/error_envelope.hpp"
#include "core/error_reporter.hpp"
#include "core/event_bridge/command_api.hpp"
#include "core/events.hpp"
#include "core/gpu_device_runtime.hpp"
#include "core/guarded_task.hpp"
#include "core/logger.hpp"
#include "core/parameter_manager.hpp"
#include "core/path_utils.hpp"
#include "core/reactive/store.hpp"
#include "core/scene.hpp"
#include "core/services.hpp"
#include "core/shareable_allocation_limit.hpp"
#include "core/tensor.hpp"
#include "core/tensor/backend/cuda/kernels/tensor_ops.hpp"
#include "core/tensor/backend/cuda/runtime/size_bucketed_pool.hpp"
#include "core/tensor_backend.hpp"
#include "python/gil.hpp"
#include "python/python_runtime.hpp"
#include "rendering/vulkan_external_tensor.hpp"
#include "training/rasterization/fast_rasterizer.hpp"
#include "training/rasterization/gsplat/Ops.h"
#include "training/rasterization/gsplat_rasterizer.hpp"
#include "training/trainer.hpp"
#include "training/training_setup.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer/post_work_utils.hpp"
#include "visualizer/visualizer_impl.hpp"
#include "window/vulkan_context.hpp"
#include "window/window_manager.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <expected>
#include <filesystem>
#include <format>
#include <functional>
#include <memory>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include "project/training_session_internal.hpp"

namespace lfs::vis {
    using namespace training_session_detail;

    using namespace lfs::core::events;

    namespace {
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
        }

        [[nodiscard]] std::uint64_t thread_id_for_logging(const std::thread::id id) noexcept {
            return static_cast<std::uint64_t>(std::hash<std::thread::id>{}(id));
        }

        void join_thread_if_not_current(std::jthread& thread, const std::string_view name) {
            if (!thread.joinable()) {
                return;
            }
            const auto target_id = thread_id_for_logging(thread.get_id());
            const auto caller_id = thread_id_for_logging(std::this_thread::get_id());
            LOG_DEBUG("Joining {} thread: target={}, caller={}", name, target_id, caller_id);
            if (thread.get_id() == std::this_thread::get_id()) {
                // A worker must never join itself. Detach before destruction so
                // std::jthread does not retry the self-join from its destructor.
                LOG_ERROR("Skipping self-join of {} thread", name);
                thread.detach();
                return;
            }
            thread.join();
        }

        [[nodiscard]] lfs::core::SplatTensorAllocator makeVulkanTrainingTensorAllocator(VisualizerImpl* viewer) {
            if (!viewer || !viewer->getWindowManager()) {
                return {};
            }
            auto* const context = viewer->getWindowManager()->getVulkanContext();
            if (!context || !context->externalMemoryInteropEnabled()) {
                return {};
            }

            return [context, viewer](lfs::core::TensorShape shape,
                                     const size_t capacity,
                                     const lfs::core::DataType dtype,
                                     const std::string_view name) -> lfs::core::Tensor {
                auto allocate = [context,
                                 shape = std::move(shape),
                                 capacity,
                                 dtype,
                                 debug_name = std::string{name}]() mutable -> lfs::core::Tensor {
                    const lfs::core::GpuBackendScope backend(lfs::core::GpuBackend::CUDA);
                    if (keepFloatShNInPooledCuda(debug_name, dtype)) {
                        auto pooled = lfs::core::Tensor::zeros_direct(
                            std::move(shape), capacity, lfs::core::Device::GPU, dtype);
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

                if (viewer && !viewer->isOnViewerThread()) {
                    return post_work_and_wait(
                        [viewer](Visualizer::WorkItem work) {
                            return viewer->postWork(std::move(work));
                        },
                        std::move(allocate),
                        []() -> lfs::core::Tensor {
                            throw lfs::core::TensorError(
                                "Vulkan-external training tensor allocation cancelled during viewer shutdown");
                        });
                }
                return allocate();
            };
        }

        struct TrainingSceneInitializationRollback {
            using OwnerRunner = std::function<void(std::function<void()>)>;

            explicit TrainingSceneInitializationRollback(lfs::core::Scene& scene,
                                                         OwnerRunner run_on_owner = {})
                : scene(&scene),
                  initial_point_cloud(scene.getInitialPointCloud()),
                  point_cloud_modified(scene.isPointCloudModified()),
                  run_on_owner(std::move(run_on_owner)) {
                const auto* const model_node =
                    scene.getNodeByUuid(scene.getTrainingModelNodeUuid());
                if (model_node && model_node->model) {
                    had_model = true;
                    model_name = model_node->name;
                    original_model = std::make_unique<lfs::core::SplatData>(
                        model_node->model->clone());
                }

                for (const auto& camera : scene.getAllCameras()) {
                    if (camera) {
                        camera_splits.emplace_back(camera, camera->split());
                    }
                }

                if (!had_model) {
                    for (const auto* node : scene.getNodes()) {
                        if (!node || node->type != lfs::core::NodeType::POINTCLOUD ||
                            !node->point_cloud) {
                            continue;
                        }
                        point_cloud = makeRestoreNodeDesc(*node);
                        const auto cropbox_id = scene.getCropBoxForSplat(node->id);
                        if (const auto* cropbox_node = scene.getNodeById(cropbox_id);
                            cropbox_node && cropbox_node->cropbox) {
                            cropbox = makeRestoreNodeDesc(*cropbox_node);
                        }
                        break;
                    }
                }
            }

            TrainingSceneInitializationRollback(const TrainingSceneInitializationRollback&) = delete;
            TrainingSceneInitializationRollback& operator=(const TrainingSceneInitializationRollback&) = delete;

            ~TrainingSceneInitializationRollback() {
                if (committed) {
                    return;
                }
                auto body = [this] { restore(); };
                if (run_on_owner) {
                    try {
                        run_on_owner(std::move(body));
                    } catch (const std::exception& error) {
                        LOG_ERROR("Failed to dispatch training initialization scene rollback: {}",
                                  error.what());
                    } catch (...) {
                        LOG_ERROR("Failed to dispatch training initialization scene rollback");
                    }
                } else {
                    restore();
                }
            }

            void commit() noexcept { committed = true; }

            void noteGraphWillMutate() noexcept { graph_mutated = true; }

        private:
            static lfs::core::Scene::RestoreNodeDesc makeRestoreNodeDesc(
                const lfs::core::SceneNode& node) {
                lfs::core::Scene::RestoreNodeDesc result{};
                result.uuid = node.uuid;
                result.type = node.type;
                result.name = node.name;
                result.parent = node.parent_id;
                result.gaussian_count = node.gaussian_count.load(std::memory_order_acquire);
                result.local_transform = node.transform();
                result.visible = node.visible.get();
                result.locked = node.locked.get();
                result.training_enabled = node.training_enabled;
                result.payload_diverged = node.payload_diverged;
                result.payload_hydration = node.payload_hydration;
                result.georef_pose = node.georef_pose;
                result.point_cloud = node.point_cloud;
                if (node.cropbox) {
                    result.cropbox = std::make_unique<lfs::core::CropBoxData>(*node.cropbox);
                }
                return result;
            }

            void restore() noexcept {
                try {
                    for (const auto& [camera, split] : camera_splits) {
                        if (camera) {
                            camera->set_split(split);
                        }
                    }

                    if (had_model) {
                        scene->replaceNodeModel(model_name, std::move(original_model));
                    } else if (graph_mutated) {
                        const auto model_id = scene->getTrainingModelNodeId();
                        if (model_id != lfs::core::NULL_NODE) {
                            scene->removeNodeById(model_id, false);
                        }
                        if (point_cloud) {
                            auto restored_point_cloud = std::move(*point_cloud);
                            const auto point_cloud_id =
                                scene->restoreNodeWithUuid(std::move(restored_point_cloud));
                            if (point_cloud_id != lfs::core::NULL_NODE && cropbox) {
                                auto restored_cropbox = std::move(*cropbox);
                                restored_cropbox.parent = point_cloud_id;
                                static_cast<void>(scene->restoreNodeWithUuid(std::move(restored_cropbox)));
                            }
                        }
                        scene->setTrainingModelNode(lfs::core::Uuid{});
                    }
                    scene->setInitialPointCloud(initial_point_cloud);
                    scene->setPointCloudModified(point_cloud_modified);
                } catch (const std::exception& error) {
                    LOG_ERROR("Failed to roll back training initialization scene changes: {}",
                              error.what());
                } catch (...) {
                    LOG_ERROR("Failed to roll back training initialization scene changes");
                }
            }

            lfs::core::Scene* scene = nullptr;
            bool had_model = false;
            std::string model_name;
            std::unique_ptr<lfs::core::SplatData> original_model;
            std::optional<lfs::core::Scene::RestoreNodeDesc> point_cloud;
            std::optional<lfs::core::Scene::RestoreNodeDesc> cropbox;
            std::shared_ptr<lfs::core::PointCloud> initial_point_cloud;
            bool point_cloud_modified = false;
            std::vector<std::pair<std::shared_ptr<lfs::core::Camera>, lfs::core::CameraSplit>> camera_splits;
            OwnerRunner run_on_owner;
            bool committed = false;
            bool graph_mutated = false;
        };
    } // namespace

    TrainerManager::TrainerManager() {
        setupEventHandlers();
        setupStateMachineCallbacks();
        completion_reaper_ = std::jthread([this](const std::stop_token stop_token) {
            completionReaperLoop(stop_token);
        });
        LOG_DEBUG("TrainerManager created");
    }

    lfs::Result<lfs::core::SplatTensorAllocator>
    TrainerManager::createTrainingSplatTensorAllocator(
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
            const auto memory = lfs::core::gpu_backend_memory_info(lfs::core::GpuBackend::CUDA);
            const std::size_t total_mem = memory.total_bytes;
            if (total_mem > 0) {
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
                auto make_interop_allocator = [this, vk_ctx] {
                    return makeSplatExportableInteropAllocator(
                        *vk_ctx, *splat_storage_, &splat_interop_parent_);
                };
                auto interop_alloc_result = viewer_ && !viewer_->isOnViewerThread()
                                                ? post_work_and_wait(
                                                      [this](Visualizer::WorkItem work) {
                                                          return viewer_->postWork(std::move(work));
                                                      },
                                                      make_interop_allocator,
                                                      []() -> lfs::Result<lfs::core::SplatTensorAllocator> {
                                                          return lfs::Result<lfs::core::SplatTensorAllocator>(
                                                              lfs::make_error(lfs::ErrorInit{
                                                                  .code = lfs::ErrorCode::Cancelled,
                                                                  .domain = lfs::ErrorDomain::Vulkan,
                                                                  .user_message =
                                                                      "Vulkan interop import cancelled during viewer shutdown",
                                                                  .detection = LFS_SOURCE_SITE_CURRENT(),
                                                              }));
                                                      })
                                                : make_interop_allocator();
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
                             lfs::format_for_developer(interop_alloc_result.error()));
                    if (interop_alloc_result.error().code() == lfs::ErrorCode::Cancelled) {
                        splat_interop_parent_.reset();
                        splat_storage_.reset();
                        return lfs::Result<lfs::core::SplatTensorAllocator>(
                            std::move(interop_alloc_result.error()));
                    }
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

        return lfs::Result<lfs::core::SplatTensorAllocator>(std::move(tensor_allocator));
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
        try {
            lfs::core::gpu_device_barrier(lfs::core::GpuBackend::CUDA);
        } catch (const std::exception& error) {
            LOG_ERROR("Device barrier before densify exportable barrier failed: {}", error.what());
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
        try {
            lfs::core::gpu_device_barrier(lfs::core::GpuBackend::CUDA);
        } catch (const std::exception& error) {
            LOG_ERROR("Device barrier after densify exportable barrier failed: {}", error.what());
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

        try {
            lfs::core::gpu_device_barrier(lfs::core::GpuBackend::CUDA);
        } catch (const std::exception& error) {
            LOG_ERROR("Device barrier before exportable grow failed: {}", error.what());
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
            const auto bind_new_chunks = [this] {
                return splat_interop_parent_->bindNewExportableChunks(*splat_storage_->block);
            };
            const bool bound = viewer_ && !viewer_->isOnViewerThread()
                                   ? post_work_and_wait(
                                         [this](Visualizer::WorkItem work) {
                                             return viewer_->postWork(std::move(work));
                                         },
                                         bind_new_chunks,
                                         [] { return false; })
                                   : bind_new_chunks();
            if (!bound) {
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
            const bool training_cache_active =
                new_state == TrainingState::Starting ||
                new_state == TrainingState::Running ||
                new_state == TrainingState::Paused ||
                new_state == TrainingState::Stopping;
            lfs::core::SizeBucketedPool::instance().set_training_active(training_cache_active);

            if (new_state == TrainingState::Starting) {
                auto& store = app_store();
                lfs::core::reactive::BatchUpdate batch(store.store());
                store.trainer_loaded.set(true);
                store.training_running.set(false);
                store.training_state.set("starting");
                store.total_iterations.set(getTotalIterations());
                python::update_trainer_loaded(true, getTotalIterations());
                python::update_training_state(false, "starting");
            }

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
        if (g_last_stored_session_publish.owner == this) {
            g_last_stored_session_publish = {};
        }
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
        join_thread_if_not_current(completion_reaper_, "completion reaper");
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
            clearStoredSessionPresentation();
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
            clearStoredSessionPresentation();
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

    bool TrainerManager::canPerform(const TrainingAction action) const {
        if (action == TrainingAction::Stop && !trainer_ && viewer_ &&
            getState() == TrainingState::Paused) {
            const auto session = viewer_->projectTrainingSessionState();
            return session.available && !session.hydrated && !session.restoring;
        }
        return state_machine_.canPerform(action);
    }

    bool TrainerManager::clearTrainer() {
        LOG_DEBUG("Clearing trainer");

        const auto state = state_machine_.getState();
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
        {
            std::lock_guard lock(initialization_mutex_);
            start_params_candidate_.reset();
        }
        // Trainer::shutdown() trims before Tensor-valued members are destroyed.
        // Trim again after destruction so those returned blocks do not survive clear.
        lfs::core::Tensor::trim_memory_pool();

        if (state_machine_.getState() != TrainingState::Idle && !state_machine_.transitionTo(TrainingState::Idle)) {
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

        // Parameter validation is deliberately synchronous: callers get an
        // immediate rejection without starting a worker or touching the scene.
        auto start_params = pendingParamsCandidate();
        if (auto error = start_params.validate(); !error.empty()) {
            static_cast<void>(
                rejectStart(std::move(error), lfs::ErrorCode::InvalidArgument));
            return false;
        }

        if (scene_ && !scene_->hasTrainingData()) {
            static_cast<void>(
                rejectStart("Scene has no cameras", lfs::ErrorCode::FailedPrecondition));
            return false;
        }

        if (!lfs::core::gpu_backend_available(lfs::core::GpuBackend::CUDA)) {
            static_cast<void>(rejectStart("Training requires an available CUDA device", lfs::ErrorCode::FailedPrecondition));
            return false;
        }

        {
            std::lock_guard lock(initialization_mutex_);
            start_params_candidate_ = std::move(start_params);
        }
        if (!state_machine_.transitionTo(TrainingState::Starting)) {
            std::lock_guard lock(initialization_mutex_);
            start_params_candidate_.reset();
            LOG_WARN("Failed to transition to Starting");
            return false;
        }
        launchTrainingThread();

        LOG_INFO("Training initialization started - {} iterations planned", getTotalIterations());
        return true;
    }

    lfs::Status
    TrainerManager::preflightStartParameters() {
        if (!trainer_) {
            return lfs::Status::failure(training_initialization_error(
                "No trainer available"));
        }
        auto error = pendingParamsCandidate().validate();
        if (!error.empty()) {
            return lfs::Status::failure(
                rejectStart(std::move(error), lfs::ErrorCode::InvalidArgument));
        }
        if (scene_ && !scene_->hasTrainingData()) {
            return lfs::Status::failure(
                rejectStart("Scene has no cameras", lfs::ErrorCode::FailedPrecondition));
        }
        return {};
    }

    lfs::Result<void> TrainerManager::waitForInitialization() {
        if (viewer_ && viewer_->isOnViewerThread()) {
            return lfs::Result<void>::failure(training_initialization_error(
                "Cannot wait for training initialization on the viewer thread"));
        }

        std::unique_lock lock(initialization_mutex_);
        initialization_cv_.wait(lock, [this] { return initialization_complete_; });
        if (initialization_error_) {
            return lfs::Result<void>::failure(*initialization_error_);
        }
        return {};
    }

    void TrainerManager::runOnSceneOwnerThread(std::function<void()> run,
                                               std::function<void()> cancel) {
        const bool use_queue = static_cast<bool>(test_scene_owner_poster_) ||
                               (viewer_ && !viewer_->isOnViewerThread());
        if (!use_queue) {
            run();
            return;
        }
        post_work_and_wait(
            [this](Visualizer::WorkItem work) {
                if (test_scene_owner_poster_) {
                    return test_scene_owner_poster_(std::move(work.run), std::move(work.cancel));
                }
                return viewer_->postWork(std::move(work));
            },
            std::move(run),
            std::move(cancel));
    }

    lfs::Result<void>
    TrainerManager::initializeTrainingOnWorker(const std::stop_token stop_token) {
        if (!trainer_) {
            return lfs::Result<void>::failure(training_initialization_error(
                "Trainer disappeared during training initialization"));
        }

        // Applying parameters can call Trainer::apply_param_side_effects(),
        // which takes render_mutex_ exclusively. Keep it off the caller thread
        // so startTraining() can acknowledge Starting while that mutex is used
        // to gate initialization.
        if (auto applied = applyPendingParams(); !applied) {
            return lfs::Result<void>::failure(
                std::move(applied).error().with_context(
                    "apply pending training parameters",
                    LFS_SOURCE_SITE_CURRENT()));
        }

        if (evaluation_weights_preparer_ && trainer_->getParams().optimization.enable_eval)
            trainer_->set_lpips_weights_path(evaluation_weights_preparer_(!trainer_->getParams().no_download));

        std::optional<TrainingSceneInitializationRollback> scene_rollback;
        std::optional<lfs::training::TrainingModelGraphCapture> graph_capture;
        if (scene_) {
            if (stop_token.stop_requested()) {
                return lfs::Result<void>::failure(training_initialization_error(
                    "Training stop requested before scene snapshot"));
            }
            std::string snapshot_cancel;
            runOnSceneOwnerThread(
                [&] {
                    scene_rollback.emplace(*scene_, [this](std::function<void()> work) {
                        runOnSceneOwnerThread(std::move(work), [] {
                            LOG_WARN("Training scene rollback skipped because the owner thread cancelled the work");
                        });
                    });
                    graph_capture = lfs::training::captureTrainingModelGraph(*scene_);
                    for (const auto& camera : scene_->getAllCameras()) {
                        if (camera) {
                            camera->to_backend(lfs::core::GpuBackend::CUDA);
                        }
                    }
                },
                [&] {
                    snapshot_cancel =
                        "Training scene snapshot cancelled during viewer shutdown";
                });
            if (!snapshot_cancel.empty()) {
                return lfs::Result<void>::failure(training_initialization_error(
                    std::move(snapshot_cancel)));
            }
        }

        const auto& params = trainer_->getParams();
        if (!trainer_->isInitialized() && params.init_path && !params.init_path->empty()) {
            std::error_code path_error;
            const auto init_path = lfs::core::utf8_to_path(*params.init_path);
            if (!std::filesystem::exists(init_path, path_error)) {
                const auto reason = path_error
                                        ? std::format("Cannot access training initialization file '{}': {}",
                                                      *params.init_path,
                                                      path_error.message())
                                        : std::format("Training initialization file '{}' does not exist",
                                                      *params.init_path);
                return lfs::Result<void>::failure(training_initialization_error(reason));
            }
        }

        if (trainer_->isInitialized()) {
            if (scene_) {
                auto* const model = graph_capture->training_model;
                const std::size_t model_size = model ? static_cast<std::size_t>(model->size()) : 0;
                const bool force_reallocation = splat_storage_.has_value();
                auto tensor_allocator_result = createTrainingSplatTensorAllocator(params, model_size);
                if (!tensor_allocator_result) {
                    return lfs::Result<void>::failure(std::move(tensor_allocator_result.error()));
                }
                auto tensor_allocator = std::move(*tensor_allocator_result);
                if (tensor_allocator) {
                    trainer_->setSplatTensorAllocator(tensor_allocator);
                    if (model) {
                        std::unique_lock scene_lock(trainer_->getRenderMutex());
                        if (auto result = lfs::training::migrateTrainingModelToAllocator(
                                params, *model, tensor_allocator, force_reallocation);
                            !result) {
                            return lfs::Result<void>::failure(
                                training_initialization_error(result.error()));
                        }
                        installExportableCapacityEnsure(*model);
                    }
                }
            }
            if (scene_) {
                installExportableDensifyBarrier();
            }
            LOG_DEBUG("Resuming from iteration {}", trainer_->get_current_iteration());
        } else {
            std::optional<lfs::training::TrainingModelGraphInstall> pending_install;
            if (scene_) {
                const std::size_t estimate =
                    graph_capture->training_model
                        ? static_cast<std::size_t>(graph_capture->training_model->size())
                    : graph_capture->point_cloud
                        ? static_cast<std::size_t>(graph_capture->point_cloud->size())
                        : 0;
                auto tensor_allocator_result =
                    createTrainingSplatTensorAllocator(params, estimate);
                if (!tensor_allocator_result) {
                    return lfs::Result<void>::failure(std::move(tensor_allocator_result.error()));
                }
                auto tensor_allocator = std::move(*tensor_allocator_result);
                trainer_->setSplatTensorAllocator(tensor_allocator);
                std::unique_lock scene_lock(trainer_->getRenderMutex());
                auto prepared = lfs::training::prepareTrainingModel(
                    params, *scene_, std::move(tensor_allocator),
                    graph_capture ? &*graph_capture : nullptr);
                if (!prepared) {
                    return lfs::Result<void>::failure(
                        training_initialization_error(prepared.error()));
                }
                if (*prepared) {
                    pending_install = std::move(**prepared);
                }
                lfs::core::Tensor::log_storage_memory("After training model initialization");
            }

            if (scene_) {
                if (stop_token.stop_requested()) {
                    return lfs::Result<void>::failure(training_initialization_error(
                        "Training stop requested before scene-graph install"));
                }
                // Graph writes must run on the scene-owner thread. Do not hold
                // render_mutex across this wait: the owner thread may take it shared.
                std::expected<void, std::string> install_result;
                runOnSceneOwnerThread(
                    [&] {
                        if (pending_install) {
                            if (scene_rollback) {
                                scene_rollback->noteGraphWillMutate();
                            }
                            install_result = lfs::training::installTrainingModel(
                                *scene_, std::move(*pending_install));
                        } else if (auto* model = graph_capture->training_model) {
                            scene_->syncTrainingModelTopology(static_cast<size_t>(model->size()));
                            scene_->notifyMutation(lfs::core::Scene::MutationType::MODEL_CHANGED);
                        }
                        if (install_result) {
                            if (auto* model = scene_->getTrainingModel()) {
                                installExportableCapacityEnsure(*model);
                            }
                        }
                    },
                    [&] {
                        install_result = std::unexpected(
                            "Training scene-graph install cancelled during viewer shutdown");
                    });
                if (!install_result) {
                    return lfs::Result<void>::failure(
                        training_initialization_error(install_result.error()));
                }
            }

            if (auto result = trainer_->initialize(params); !result) {
                return lfs::Result<void>::failure(
                    training_initialization_error(result.error()));
            }
            lfs::core::Tensor::log_storage_memory("After trainer initialization");
            lfs::core::Tensor::trim_memory_pool();
            if (scene_) {
                installExportableDensifyBarrier();
            }
        }

        if (stop_token.stop_requested()) {
            LOG_DEBUG("Training stop requested during initialization");
        }
        if (scene_rollback) {
            scene_rollback->commit();
        }

        // TrainingStarted is synchronously delivered to its subscribers. Keep
        // the completion handoff on the viewer thread and hold the training
        // worker behind it so renderer/GUI state is ready before train().
        const int total_iterations = getTotalIterations();
        const auto open_initialization_gate = [this](const bool failed) {
            {
                std::lock_guard lock(initialization_gate_mutex_);
                initialization_main_step_failed_ = failed;
                initialization_gate_open_ = true;
            }
            initialization_gate_cv_.notify_all();
        };
        const auto main_thread_step = [this, stop_token, total_iterations, open_initialization_gate] {
            bool failed = false;
            try {
                if (viewer_) {
                    auto* const rendering_manager = viewer_->getRenderingManager();
                    auto* const window_manager = viewer_->getWindowManager();
                    auto* const vulkan_context = window_manager ? window_manager->getVulkanContext() : nullptr;
                    std::shared_lock scene_lock(trainer_->getRenderMutex(), std::defer_lock);
                    if (scene_) {
                        scene_lock.lock();
                    }
                    auto* const model = scene_ ? scene_->getTrainingModel() : nullptr;
                    if (rendering_manager && vulkan_context && model) {
                        glm::ivec2 prime_size = rendering_manager->getRenderedSize();
                        if (prime_size.x <= 0 || prime_size.y <= 0) {
                            prime_size = window_manager ? window_manager->getWindowSize() : glm::ivec2{1280, 720};
                        }
                        if (auto ok = rendering_manager->ensureVksplatTrainingSharedScratchReady(
                                *vulkan_context, *model, prime_size);
                            !ok) {
                            LOG_WARN("VkSplat training shared-scratch pre-start prime skipped: {}", ok.error());
                        }
                    }
                }

                if (state_machine_.getState() == TrainingState::Starting &&
                    !stop_token.stop_requested() &&
                    !trainer_->has_stopped()) {
                    if (!state_machine_.transitionTo(TrainingState::Running)) {
                        LOG_WARN("Failed to transition to Running");
                    }
                    training_start_time_ = std::chrono::steady_clock::now();
                    accumulated_training_time_ = std::chrono::steady_clock::duration{0};
                    state::TrainingStarted{.total_iterations = total_iterations}.emit();
                    if (initialization_pause_requested_.load(std::memory_order_acquire)) {
                        if (!state_machine_.transitionTo(TrainingState::Paused)) {
                            LOG_WARN("Failed to transition to Paused after initialization pause request");
                        }
                        state::TrainingPaused{.iteration = getCurrentIteration()}.emit();
                    }
                }
            } catch (const std::exception& error) {
                LOG_ERROR("Training main-thread initialization handoff failed: {}", error.what());
                failed = true;
            } catch (...) {
                LOG_ERROR("Training main-thread initialization handoff failed");
                failed = true;
            }
            open_initialization_gate(failed);
        };

        const auto cancel_main_thread_step = [open_initialization_gate] {
            open_initialization_gate(true);
        };
        if (viewer_) {
            if (!viewer_->postWork({
                    .run = main_thread_step,
                    .cancel = cancel_main_thread_step,
                })) {
                cancel_main_thread_step();
            }
        } else {
            main_thread_step();
        }

        {
            std::unique_lock lock(initialization_gate_mutex_);
            initialization_gate_cv_.wait(lock, [this] { return initialization_gate_open_; });
            if (initialization_main_step_failed_) {
                return lfs::Result<void>::failure(training_initialization_error(
                    "Training main-thread initialization handoff was cancelled"));
            }
        }
        return {};
    }

    void TrainerManager::pauseTraining() {
        if (!canPause()) {
            LOG_TRACE("Cannot pause: {}", getActionBlockedReason(TrainingAction::Pause));
            return;
        }

        if (trainer_) {
            trainer_->request_pause();
            if (getState() == TrainingState::Starting) {
                initialization_pause_requested_.store(true, std::memory_order_release);
                LOG_INFO("Training pause requested during initialization");
                return;
            }
            accumulated_training_time_ += std::chrono::steady_clock::now() - training_start_time_;

            if (!state_machine_.transitionTo(TrainingState::Paused)) {
                LOG_WARN("Failed to transition to Paused");
            }

            state::TrainingPaused{.iteration = getCurrentIteration()}.emit();
            LOG_INFO("Training paused at iteration {}", getCurrentIteration());
        }
    }

    lfs::Status TrainerManager::resumeTraining() {
        if (!trainer_ && viewer_) {
            const auto session =
                viewer_->projectTrainingSessionState();
            if (session.available && !session.hydrated) {
                if (auto restored =
                        viewer_->startTraining();
                    !restored) {
                    return lfs::Status::failure(training_initialization_error(restored.error()));
                }
                return {};
            }
        }
        if (!canResume()) {
            return lfs::Status::failure(rejectStart(
                std::string(getActionBlockedReason(TrainingAction::Resume)),
                lfs::ErrorCode::FailedPrecondition));
        }
        if (!trainer_)
            return lfs::Status::failure(rejectStart(
                "No trainer available", lfs::ErrorCode::FailedPrecondition));

        if (auto preflight = preflightStartParameters(); !preflight)
            return preflight;

        const int iter = getCurrentIteration();
        const bool need_thread = !isCompletionPending();

        last_error_.clear();
        last_training_error_.clear();
        {
            std::lock_guard lock(initialization_mutex_);
            initialization_error_.reset();
            initialization_complete_ = true;
        }
        initialization_cv_.notify_all();

        training_start_time_ = std::chrono::steady_clock::now();
        if (!state_machine_.transitionTo(TrainingState::Running)) {
            return lfs::Status::failure(rejectStart(
                "Cannot transition to running", lfs::ErrorCode::FailedPrecondition));
        }
        if (!need_thread) {
            trainer_->request_resume();
        }
        if (need_thread) {
            // Checkpoint resume: publish Running before the worker begins its
            // off-thread storage preparation, avoiding a state race.
            launchTrainingThread();
        }

        state::TrainingResumed{.iteration = iter}.emit();
        LOG_INFO("Training resumed at iteration {}", iter);
        return {};
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

        // A reopened project presents its checkpoint as paused before a live
        // trainer exists. Stop that stored session without allocating an
        // optimizer or executing a training step just to reach Edit Mode.
        if (!trainer_ && stored_session_presentation_active_) {
            if (!state_machine_.transitionTo(TrainingState::Paused)) {
                return;
            }
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
            std::lock_guard lock(training_thread_mutex_);
            training_stop_source_.reset();
            initialization_thread_done_ = false;
        }
        {
            std::lock_guard lock(completion_mutex_);
            training_joined_ = false;
            pending_completion_.reset();
        }
        {
            std::lock_guard lock(initialization_mutex_);
            initialization_complete_ = false;
            initialization_error_.reset();
        }
        {
            std::lock_guard lock(initialization_gate_mutex_);
            initialization_gate_open_ = false;
            initialization_main_step_failed_ = false;
        }
        initialization_pause_requested_.store(false, std::memory_order_release);
        completion_pending_.store(true, std::memory_order_release);

        last_error_.clear();
        last_training_error_.clear();
        auto worker = std::make_unique<std::jthread>(
            [this](const std::stop_token stop_token) {
                trainingInitializationThreadFunc(stop_token);
            });
        {
            std::lock_guard lock(training_thread_mutex_);
            training_stop_source_ = worker->get_stop_source();
            initialization_thread_ = std::move(worker);
        }
        training_thread_cv_.notify_one();
    }

    void TrainerManager::completionReaperLoop(const std::stop_token stop_token) {
        while (true) {
            std::unique_ptr<std::jthread> initialization_worker;
            std::unique_ptr<std::jthread> training_worker;
            {
                std::unique_lock lock(training_thread_mutex_);
                training_thread_cv_.wait(lock, [this, stop_token] {
                    return stop_token.stop_requested() ||
                           (initialization_thread_ && initialization_thread_done_) ||
                           training_thread_ != nullptr;
                });
                if (initialization_thread_ &&
                    (initialization_thread_done_ || stop_token.stop_requested())) {
                    initialization_worker = std::move(initialization_thread_);
                }
                training_worker = std::move(training_thread_);

                if (!initialization_worker && !training_worker && stop_token.stop_requested()) {
                    return;
                }
            }

            if (initialization_worker) {
                join_thread_if_not_current(*initialization_worker, "training initialization");
            }
            if (training_worker) {
                join_thread_if_not_current(*training_worker, "training execution");
            }

            // Initialization failures do not launch a training worker, but
            // still publish through the same completion/reaper path.
            bool completion_ready = false;
            {
                std::lock_guard lock(completion_mutex_);
                completion_ready = pending_completion_.has_value();
            }
            if (training_worker || completion_ready) {
                finishTrainingThreadJoin();
            }
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
        if (trainer_) {
            return trainer_->get_current_iteration();
        }
        if (stored_session_presentation_active_) {
            return stored_session_presentation_iteration_;
        }
        return 0;
    }

    float TrainerManager::getCurrentLoss() const {
        return trainer_ ? trainer_->get_current_loss() : 0.0f;
    }

    int TrainerManager::getTotalIterations() const {
        if (trainer_) {
            return trainer_->get_total_iterations();
        }
        if (stored_session_presentation_active_) {
            return stored_session_presentation_max_iterations_;
        }
        return 0;
    }

    int TrainerManager::getNumSplats() const {
        if (!trainer_) {
            if (stored_session_presentation_active_) {
                const core::Scene* scene = scene_;
                if (!scene && viewer_) {
                    scene = &viewer_->getScene();
                }
                if (scene) {
                    return static_cast<int>(
                        scene->getTrainingModelGaussianCount());
                }
            }
            return 0;
        }

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
            if (auto updated = trainer_->setParams(params); !updated) {
                LOG_ERROR(
                    "Could not update save steps: {}",
                    lfs::format_for_developer(updated.error()));
            }
        }
    }

    const char* TrainerManager::getStrategyType() const {
        if (trainer_ && trainer_->isInitialized()) {
            return trainer_->get_strategy().strategy_type();
        }
        if (stored_session_presentation_active_ &&
            !stored_session_presentation_strategy_.empty()) {
            return stored_session_presentation_strategy_.c_str();
        }
        return "unknown";
    }

    bool TrainerManager::isGutEnabled() const {
        if (!trainer_)
            return false;
        return trainer_->getParams().optimization.gut;
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
            store.eval_lpips.set(last->lpips);
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

    void TrainerManager::trainingInitializationThreadFunc(std::stop_token stop_token) {
        const lfs::core::GpuBackendScope backend(lfs::core::GpuBackend::CUDA);
        LOG_INFO("Training initialization thread started");
        lfs::Result<void> initialization_result;
        try {
            initialization_result = initializeTrainingOnWorker(stop_token);
        } catch (const std::exception& error) {
            // LFS-CENSUS-OK(empty-catch): normalize initialization exceptions into typed training errors.
            initialization_result = lfs::Result<void>::failure(training_initialization_error(
                std::format("Failed to initialize training: {}", error.what())));
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): normalize unknown initialization exceptions into typed training errors.
            initialization_result = lfs::Result<void>::failure(training_initialization_error(
                "Failed to initialize training: unknown error"));
        }

        if (!initialization_result) {
            const lfs::Error typed = initialization_result.error();
            const std::string error_message = typed.user_message().empty()
                                                  ? std::string(typed.detail())
                                                  : std::string(typed.user_message());
            const bool user_stopped = stop_token.stop_requested();
            if (!user_stopped) {
                LOG_ERROR("Training initialization failed: {}", error_message);
                last_error_ = error_message;
                last_training_error_.set(typed);
            }

            // The initialization helper rolls back all scene changes before
            // this existing completion/error path is made observable. Keep
            // this worker-side path limited to completion data: the reaper
            // owns the joins, and joining initialization_thread_ here would
            // self-join.
            {
                std::lock_guard lock(completion_mutex_);
                pending_completion_ = TrainingCompletionData{
                    .iteration = 0,
                    .final_loss = 0.0f,
                    .elapsed_seconds = 0.0f,
                    .success = user_stopped,
                    .user_stopped = user_stopped,
                    .resource_exhausted = false,
                    .reason = user_stopped ? FinishReason::UserStopped : FinishReason::Error,
                    .error = user_stopped ? std::string{} : error_message,
                    .typed_error = user_stopped ? std::nullopt : std::optional{typed}};
            }
            {
                std::lock_guard lock(initialization_mutex_);
                initialization_complete_ = true;
                initialization_error_ = user_stopped ? std::nullopt : std::optional{typed};
            }
            initialization_cv_.notify_all();
            release_training_thread_local_cuda_caches();
            {
                std::lock_guard lock(training_thread_mutex_);
                initialization_thread_done_ = true;
            }
            training_thread_cv_.notify_one();
            LOG_INFO("Training initialization thread finished");
            return;
        }

        // The initialization thread must not execute train(). A new jthread
        // gives the training loop a clean CUDA/tensor thread-local context.
        auto training_worker = std::make_unique<std::jthread>(
            [this](const std::stop_token training_stop_token) {
                trainingThreadFunc(training_stop_token);
            });
        {
            std::lock_guard lock(training_thread_mutex_);
            training_stop_source_ = training_worker->get_stop_source();
            if (stop_token.stop_requested()) {
                training_stop_source_->request_stop();
            }
            training_thread_ = std::move(training_worker);
            initialization_thread_done_ = true;
        }
        {
            std::lock_guard lock(initialization_mutex_);
            initialization_complete_ = true;
            initialization_error_.reset();
        }
        initialization_cv_.notify_all();
        release_training_thread_local_cuda_caches();
        training_thread_cv_.notify_one();

        LOG_INFO("Training initialization thread finished");
    }

    void TrainerManager::trainingThreadFunc(std::stop_token stop_token) {
        const lfs::core::GpuBackendScope backend(lfs::core::GpuBackend::CUDA);
        {
            std::unique_lock lock(initialization_gate_mutex_);
            initialization_gate_cv_.wait(lock, [this] { return initialization_gate_open_; });
            if (initialization_main_step_failed_) {
                LOG_ERROR("Training execution blocked because the main-thread initialization handoff failed");
                return;
            }
        }
        LOG_INFO("Training thread started");
        LOG_TIMER("Training execution");

        trainer_->setOnIterationStart([this] {
            if (auto* pm = services().paramsOrNull(); pm && pm->consumeDirty()) {
                if (auto applied = applyPendingParams(); !applied) {
                    auto typed = std::move(applied).error();
                    last_error_ = typed.user_message().empty()
                                      ? std::string(typed.detail())
                                      : std::string(typed.user_message());
                    last_training_error_.set(typed);
                    pm->importTrainingParams(trainer_->getParams());
                    LOG_ERROR("Rejected training parameter update: {}", last_error_);
                }
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
        if (success) {
            last_error_.clear();
            last_training_error_.clear();
        }
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
            updateEvaluationMetrics(event.iteration, event.psnr, event.ssim,
                                    event.lpips);
        });
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

    lfs::core::param::TrainingParameters TrainerManager::pendingParamsCandidate() const {
        auto params = trainer_->getParams();
        if (trainer_->isInitialized() && params.resume_checkpoint.has_value()) {
            if (auto* const param_mgr = services().paramsOrNull()) {
                params.optimization.save_steps = param_mgr->copyActiveParams().save_steps;
            }
            return params;
        }

        params.dataset = pending_dataset_params_;
        if (auto* const param_mgr = services().paramsOrNull()) {
            params.optimization = param_mgr->copyActiveParams();
        } else {
            params.optimization = pending_opt_params_;
        }
        return params;
    }

    lfs::Status TrainerManager::applyPendingParams() {
        if (!trainer_)
            return {};

        const auto previous_params = trainer_->getParams();
        std::optional<lfs::core::param::TrainingParameters> frozen_start_params;
        {
            std::lock_guard lock(initialization_mutex_);
            frozen_start_params = std::move(start_params_candidate_);
            start_params_candidate_.reset();
        }
        auto params = frozen_start_params
                          ? std::move(*frozen_start_params)
                          : pendingParamsCandidate();
        if (auto error = params.validate(); !error.empty()) {
            return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::InvalidArgument,
                .domain = lfs::ErrorDomain::Training,
                .user_message = error,
                .detail = "Rejected invalid pending training parameters: " + error,
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        }
        if (trainer_->isInitialized() && previous_params.resume_checkpoint.has_value()) {
            if (auto updated = trainer_->setParams(params); !updated) {
                return lfs::Status::failure(std::move(updated).error());
            }
            if (auto* const param_mgr = services().paramsOrNull()) {
                param_mgr->importTrainingParams(params);
            }
            LOG_DEBUG("Ignoring parameter updates for checkpoint-backed trainer (save steps kept)");
            return {};
        }

        LOG_DEBUG("Applied params: strategy={}, iter={}, max_cap={}",
                  params.optimization.strategy, params.optimization.iterations, params.optimization.max_cap);

        const bool evaluation_split_changed =
            previous_params.optimization.holds_out_eval_images() != params.optimization.holds_out_eval_images() ||
            previous_params.dataset.test_every != params.dataset.test_every;
        if (!trainer_->isInitialized() && scene_ && evaluation_split_changed) {
            refreshCameraEvaluationSplit(
                *scene_,
                params.optimization.holds_out_eval_images(),
                params.dataset.test_every);
        }
        return trainer_->setParams(params);
    }

} // namespace lfs::vis
