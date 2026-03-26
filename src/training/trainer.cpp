/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "trainer.hpp"
#include "components/bilateral_grid.hpp"
#include "components/ppisp.hpp"
#include "components/ppisp_controller_pool.hpp"
#include "components/ppisp_file.hpp"
#include "components/sparsity_optimizer.hpp"
#include "control/command_api.hpp"
#include "control/control_boundary.hpp"
#include "core/checkpoint_format.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/events.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/scene.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor/internal/gpu_slab_allocator.hpp"
#include "core/tensor/internal/size_bucketed_pool.hpp"
#include "io/cache_image_loader.hpp"
#include "io/exporter.hpp"
#include "io/filesystem_utils.hpp"
#include "lfs/kernels/ssim.cuh"
#include "losses/losses.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "python/runner.hpp"
#include "rasterization/fast_rasterizer.hpp"
#include "rasterization/gsplat_rasterizer.hpp"
#include "strategies/adc.hpp"
#include "strategies/mcmc.hpp"
#include "strategies/strategy_factory.hpp"
#include "training/kernels/grad_alpha.hpp"
#include "training/kernels/lfs_kernels.hpp"

#include <filesystem>
#include <fstream>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <expected>
#include <memory>
#include <numeric>
#include <nvtx3/nvToolsExt.h>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace lfs::training {

    namespace {
        constexpr double BYTES_PER_MIB = 1024.0 * 1024.0;
        constexpr double BYTES_PER_GIB = 1024.0 * 1024.0 * 1024.0;

        [[nodiscard]] bool env_flag_enabled(const char* name) {
            const char* raw = std::getenv(name);
            if (!raw) {
                return false;
            }

            const std::string_view value(raw);
            if (value.empty()) {
                return false;
            }

            return value != "0" && value != "false" && value != "FALSE" &&
                   value != "off" && value != "OFF" &&
                   value != "no" && value != "NO";
        }

        [[nodiscard]] double bytes_to_mib(const size_t bytes) {
            return static_cast<double>(bytes) / BYTES_PER_MIB;
        }

        [[nodiscard]] double bytes_to_gib(const size_t bytes) {
            return static_cast<double>(bytes) / BYTES_PER_GIB;
        }

        [[nodiscard]] size_t tensor_reserved_bytes(const lfs::core::Tensor& tensor) {
            if (!tensor.is_valid()) {
                return 0;
            }

            if (tensor.capacity() == 0 || tensor.ndim() == 0) {
                return tensor.bytes();
            }

            size_t row_elems = 1;
            if (tensor.ndim() > 1) {
                for (size_t dim = 1; dim < tensor.ndim(); ++dim) {
                    row_elems *= tensor.shape()[dim];
                }
            }

            return tensor.capacity() * row_elems * lfs::core::dtype_size(tensor.dtype());
        }

        template <typename Entries>
        void add_tensor_entry(Entries& entries, std::string label, const lfs::core::Tensor& tensor) {
            const size_t bytes = tensor_reserved_bytes(tensor);
            if (bytes > 0) {
                entries.emplace_back(std::move(label), bytes);
            }
        }

        template <typename Entries>
        [[nodiscard]] size_t sum_entry_bytes(const Entries& entries) {
            size_t total = 0;
            for (const auto& [_, bytes] : entries) {
                total += bytes;
            }
            return total;
        }

        template <typename Entries>
        void log_entry_bytes(std::string_view label, Entries entries) {
            std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.second > rhs.second;
            });

            for (const auto& [name, bytes] : entries) {
                LOG_INFO("[MEM] {} {} = {:.2f} MiB", label, name, bytes_to_mib(bytes));
            }

            const size_t total = sum_entry_bytes(entries);
            LOG_INFO("[MEM] {} total = {:.2f} MiB ({:.2f} GiB)",
                     label, bytes_to_mib(total), bytes_to_gib(total));
        }

        [[nodiscard]] size_t photometric_workspace_bytes(const losses::PhotometricLoss& photometric_loss) {
            std::vector<std::pair<std::string, size_t>> entries;

            const auto& fused = photometric_loss.fused_workspace();
            add_tensor_entry(entries, "fused.ssim_map", fused.ssim_map);
            add_tensor_entry(entries, "fused.dm_dmu1", fused.dm_dmu1);
            add_tensor_entry(entries, "fused.dm_dsigma1_sq", fused.dm_dsigma1_sq);
            add_tensor_entry(entries, "fused.dm_dsigma12", fused.dm_dsigma12);
            add_tensor_entry(entries, "fused.grad_img", fused.grad_img);
            add_tensor_entry(entries, "fused.reduction_temp", fused.reduction_temp);
            add_tensor_entry(entries, "fused.reduction_result", fused.reduction_result);

            const auto& ssim = photometric_loss.ssim_workspace();
            add_tensor_entry(entries, "ssim.ssim_map", ssim.ssim_map);
            add_tensor_entry(entries, "ssim.dm_dmu1", ssim.dm_dmu1);
            add_tensor_entry(entries, "ssim.dm_dsigma1_sq", ssim.dm_dsigma1_sq);
            add_tensor_entry(entries, "ssim.dm_dsigma12", ssim.dm_dsigma12);
            add_tensor_entry(entries, "ssim.dL_dmap", ssim.dL_dmap);
            add_tensor_entry(entries, "ssim.dL_dimg1", ssim.dL_dimg1);
            add_tensor_entry(entries, "ssim.reduction_temp", ssim.reduction_temp);
            add_tensor_entry(entries, "ssim.reduction_result", ssim.reduction_result);

            return sum_entry_bytes(entries);
        }

        [[nodiscard]] size_t masked_fused_workspace_bytes(const kernels::MaskedFusedL1SSIMWorkspace& workspace) {
            std::vector<std::pair<std::string, size_t>> entries;
            add_tensor_entry(entries, "masked.ssim_map", workspace.ssim_map);
            add_tensor_entry(entries, "masked.dm_dmu1", workspace.dm_dmu1);
            add_tensor_entry(entries, "masked.dm_dsigma1_sq", workspace.dm_dsigma1_sq);
            add_tensor_entry(entries, "masked.dm_dsigma12", workspace.dm_dsigma12);
            add_tensor_entry(entries, "masked.grad_img", workspace.grad_img);
            add_tensor_entry(entries, "masked.masked_loss", workspace.masked_loss);
            add_tensor_entry(entries, "masked.mask_sum", workspace.mask_sum);
            return sum_entry_bytes(entries);
        }

        [[nodiscard]] size_t ssim_map_workspace_bytes(const kernels::SSIMMapWorkspace& workspace) {
            return tensor_reserved_bytes(workspace.ssim_map);
        }

        [[nodiscard]] size_t bg_image_cache_bytes(const std::unordered_map<uint64_t, lfs::core::Tensor>& cache) {
            size_t total = 0;
            for (const auto& [_, tensor] : cache) {
                total += tensor_reserved_bytes(tensor);
            }
            return total;
        }

        [[nodiscard]] size_t splat_tensor_bytes(const lfs::core::SplatData& splat) {
            std::vector<std::pair<std::string, size_t>> entries;
            add_tensor_entry(entries, "means", splat.means());
            add_tensor_entry(entries, "sh0", splat.sh0());
            add_tensor_entry(entries, "shN", splat.shN());
            add_tensor_entry(entries, "scaling", splat.scaling_raw());
            add_tensor_entry(entries, "rotation", splat.rotation_raw());
            add_tensor_entry(entries, "opacity", splat.opacity_raw());
            add_tensor_entry(entries, "deleted", splat.deleted());
            add_tensor_entry(entries, "densification_info", splat._densification_info);
            return sum_entry_bytes(entries);
        }

        [[nodiscard]] size_t optimizer_state_bytes(const AdamOptimizer& optimizer) {
            size_t total = 0;
            for (const auto type : AdamOptimizer::all_param_types()) {
                const auto* state = optimizer.get_state(type);
                if (!state) {
                    continue;
                }
                total += tensor_reserved_bytes(state->grad);
                total += tensor_reserved_bytes(state->exp_avg);
                total += tensor_reserved_bytes(state->exp_avg_sq);
            }
            return total;
        }

        struct VramSnapshot {
            size_t gpu_used = 0;
            size_t gpu_total = 0;
            size_t mempool_used = 0;
            size_t mempool_reserved = 0;
            size_t bucket_cached = 0;
            size_t slab_reserved = 0;
            std::optional<lfs::core::RasterizerMemoryArena::MemoryInfo> arena;
        };

        [[nodiscard]] VramSnapshot capture_vram_snapshot(const bool synchronize) {
            if (synchronize) {
                cudaDeviceSynchronize();
            }

            VramSnapshot snapshot;
            size_t gpu_free = 0;
            cudaMemGetInfo(&gpu_free, &snapshot.gpu_total);
            snapshot.gpu_used = snapshot.gpu_total - gpu_free;

#if CUDART_VERSION >= 12080
            int device = 0;
            if (cudaGetDevice(&device) == cudaSuccess) {
                cudaMemPool_t pool;
                if (cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess) {
                    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &snapshot.mempool_used);
                    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &snapshot.mempool_reserved);
                }
            }
#endif

            snapshot.bucket_cached =
                lfs::core::SizeBucketedPool::instance().stats().bytes_cached.load(std::memory_order_relaxed);
            snapshot.slab_reserved =
                lfs::core::GPUSlabAllocator::instance().stats().total_slab_memory;

            if (auto* arena = lfs::core::GlobalArenaManager::instance().try_get_arena()) {
                snapshot.arena = arena->get_memory_info();
            }

            return snapshot;
        }

        void log_vram_snapshot(std::string_view label, const VramSnapshot& snapshot) {
            LOG_INFO("[MEM] {} gpu_used={:.2f} MiB ({:.2f} GiB) / total={:.2f} GiB",
                     label,
                     bytes_to_mib(snapshot.gpu_used),
                     bytes_to_gib(snapshot.gpu_used),
                     bytes_to_gib(snapshot.gpu_total));
            LOG_INFO("[MEM] {} mempool_used={:.2f} MiB, mempool_reserved={:.2f} MiB, slab_reserved={:.2f} MiB, bucket_cached={:.2f} MiB",
                     label,
                     bytes_to_mib(snapshot.mempool_used),
                     bytes_to_mib(snapshot.mempool_reserved),
                     bytes_to_mib(snapshot.slab_reserved),
                     bytes_to_mib(snapshot.bucket_cached));

            if (snapshot.arena) {
                LOG_INFO("[MEM] {} arena_capacity={:.2f} MiB, arena_current={:.2f} MiB, arena_peak={:.2f} MiB, arena_reallocs={}",
                         label,
                         bytes_to_mib(snapshot.arena->arena_capacity),
                         bytes_to_mib(snapshot.arena->current_usage),
                         bytes_to_mib(snapshot.arena->peak_usage),
                         snapshot.arena->num_reallocations);
            } else {
                LOG_INFO("[MEM] {} arena=not_created", label);
            }
        }

        void syncTrainingSceneTopology(lfs::core::Scene* const scene,
                                       const lfs::core::SplatData& model) {
            if (!scene) {
                return;
            }
            scene->syncTrainingModelTopology(static_cast<size_t>(model.size()));
        }

        template <typename Fn>
        class ScopeGuard {
        public:
            explicit ScopeGuard(Fn fn)
                : fn_(std::move(fn)) {}

            ScopeGuard(const ScopeGuard&) = delete;
            ScopeGuard& operator=(const ScopeGuard&) = delete;

            ScopeGuard(ScopeGuard&& other) noexcept
                : fn_(std::move(other.fn_)),
                  active_(other.active_) {
                other.active_ = false;
            }

            ScopeGuard& operator=(ScopeGuard&&) = delete;

            ~ScopeGuard() {
                if (active_) {
                    fn_();
                }
            }

            void release() noexcept { active_ = false; }

        private:
            Fn fn_;
            bool active_ = true;
        };

        template <typename Fn>
        ScopeGuard<Fn> makeScopeGuard(Fn fn) {
            return ScopeGuard<Fn>(std::move(fn));
        }

        struct PipelineMemoryEstimate {
            size_t max_image_bytes = 0;
            bool masks_possible = false;
        };

        [[nodiscard]] PipelineMemoryEstimate estimatePipelineMemory(
            const std::shared_ptr<CameraDataset>& dataset) {
            PipelineMemoryEstimate estimate;
            if (!dataset) {
                return estimate;
            }

            for (const auto& cam : dataset->get_cameras()) {
                if (!cam) {
                    continue;
                }

                const size_t width = static_cast<size_t>(std::max(cam->image_width(), 0));
                const size_t height = static_cast<size_t>(std::max(cam->image_height(), 0));
                if (width > 0 && height > 0) {
                    estimate.max_image_bytes = std::max(
                        estimate.max_image_bytes,
                        width * height * size_t{3} * sizeof(float));
                }

                estimate.masks_possible = estimate.masks_possible || cam->has_mask() || cam->has_alpha();
            }

            return estimate;
        }

        [[nodiscard]] lfs::io::PipelinedLoaderConfig tunePipelinedLoaderConfig(
            lfs::io::PipelinedLoaderConfig config,
            const std::shared_ptr<CameraDataset>& dataset) {
            const auto estimate = estimatePipelineMemory(dataset);
            if (estimate.max_image_bytes == 0) {
                config.decoder_pool_size = std::min(config.decoder_pool_size, config.jpeg_batch_size);
                return config;
            }

            size_t free_bytes = 0;
            size_t total_bytes = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess || free_bytes == 0 || total_bytes == 0) {
                config.decoder_pool_size = std::min(config.decoder_pool_size, config.jpeg_batch_size);
                return config;
            }

            const size_t per_ready_image_bytes = estimate.max_image_bytes +
                                                 (estimate.masks_possible ? estimate.max_image_bytes / 3 : 0);
            constexpr size_t MIN_PIPELINE_BUDGET_BYTES = 256ULL * 1024 * 1024;
            constexpr size_t MAX_PIPELINE_BUDGET_BYTES = 512ULL * 1024 * 1024;
            const size_t target_pipeline_budget =
                std::clamp(free_bytes / 32, MIN_PIPELINE_BUDGET_BYTES, MAX_PIPELINE_BUDGET_BYTES);

            const size_t recommended_prefetch = std::clamp(
                target_pipeline_budget / std::max<size_t>(per_ready_image_bytes, 1),
                size_t{2},
                config.prefetch_count);

            if (recommended_prefetch < config.prefetch_count) {
                LOG_INFO(
                    "Reducing image pipeline depth {} -> {} (largest image {:.1f} MB, free VRAM {:.1f} GB)",
                    config.prefetch_count,
                    recommended_prefetch,
                    estimate.max_image_bytes / (1024.0 * 1024.0),
                    free_bytes / (1024.0 * 1024.0 * 1024.0));
                config.prefetch_count = recommended_prefetch;
            }

            config.jpeg_batch_size = std::min(
                config.jpeg_batch_size,
                std::max<size_t>(1, config.prefetch_count));
            config.output_queue_size = std::min(
                config.output_queue_size,
                std::max<size_t>(1, config.prefetch_count / 2));
            config.decoder_pool_size = std::min(
                std::max<size_t>(1, config.decoder_pool_size),
                std::max<size_t>(1, config.jpeg_batch_size));

            return config;
        }

        PPISPRenderOverrides toRenderOverrides(const PPISPViewportOverrides& ov) {
            PPISPRenderOverrides r;
            r.exposure_offset = ov.exposure_offset;
            r.vignette_enabled = ov.vignette_enabled;
            r.vignette_strength = ov.vignette_strength;
            r.wb_temperature = ov.wb_temperature;
            r.wb_tint = ov.wb_tint;
            r.color_red_x = ov.color_red_x;
            r.color_red_y = ov.color_red_y;
            r.color_green_x = ov.color_green_x;
            r.color_green_y = ov.color_green_y;
            r.color_blue_x = ov.color_blue_x;
            r.color_blue_y = ov.color_blue_y;
            r.gamma_multiplier = ov.gamma_multiplier;
            r.gamma_red = ov.gamma_red;
            r.gamma_green = ov.gamma_green;
            r.gamma_blue = ov.gamma_blue;
            r.crf_toe = ov.crf_toe;
            r.crf_shoulder = ov.crf_shoulder;
            return r;
        }
    } // namespace

    // Tile configuration for memory-efficient training
    enum class TileMode {
        One = 1, // 1 tile  - 1x1 - Render full image (no tiling)
        Two = 2, // 2 tiles - 2x1 - Two horizontal tiles
        Four = 4 // 4 tiles - 2x2 - Four tiles in a grid
    };

    void Trainer::cleanup() {
        LOG_DEBUG("Cleaning up trainer for re-initialization");

        // Stop any ongoing operations
        stop_requested_ = true;

        // Sync callback stream to avoid race conditions
        if (callback_stream_) {
            cudaStreamSynchronize(callback_stream_);
        }
        callback_busy_ = false;

        // Reset all components
        progress_.reset();
        bilateral_grid_.reset();
        ppisp_.reset();
        ppisp_controller_pool_.reset();
        sparsity_optimizer_.reset();
        evaluator_.reset();

        // Clear datasets (will be recreated)
        train_dataset_.reset();
        val_dataset_.reset();

        // Reset flags
        pause_requested_ = false;
        save_requested_ = false;
        stop_requested_ = false;
        is_paused_ = false;
        is_running_ = false;
        training_complete_ = false;
        ready_to_start_ = false;
        current_iteration_ = 0;
        current_loss_ = 0.0f;
        train_dataset_size_ = 0;
        total_cameras_count_ = 0;

        LOG_DEBUG("Trainer cleanup complete");
    }

    std::expected<void, std::string> Trainer::initialize_bilateral_grid() {
        if (!params_.optimization.use_bilateral_grid) {
            return {};
        }

        try {
            BilateralGrid::Config config;
            config.lr = params_.optimization.bilateral_grid_lr;

            // BilateralGrid is indexed with cam->uid() in the training loop. Those UIDs stay
            // in the original camera space even when train/val splits are enabled, so the grid
            // must be sized for the full camera set rather than only the training subset.
            bilateral_grid_ = std::make_unique<BilateralGrid>(
                static_cast<int>(total_cameras_count_),
                params_.optimization.bilateral_grid_X,
                params_.optimization.bilateral_grid_Y,
                params_.optimization.bilateral_grid_W,
                params_.optimization.iterations,
                config);

            LOG_INFO("Bilateral grid initialized: {}x{}x{} for {} camera slots ({} train images)",
                     params_.optimization.bilateral_grid_X,
                     params_.optimization.bilateral_grid_Y,
                     params_.optimization.bilateral_grid_W,
                     total_cameras_count_,
                     train_dataset_size_);

            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to init bilateral grid: {}", e.what()));
        }
    }

    std::expected<void, std::string> Trainer::initialize_ppisp() {
        if (!params_.optimization.use_ppisp) {
            return {};
        }

        try {
            PPISPConfig config;
            config.lr = params_.optimization.ppisp_lr;
            config.warmup_steps = params_.optimization.ppisp_warmup_steps;

            ppisp_ = std::make_unique<PPISP>(params_.optimization.iterations, config);
            for (const auto& cam : train_dataset_->get_cameras()) {
                if (cam) {
                    ppisp_->register_frame(cam->uid(), cam->camera_id());
                }
            }
            ppisp_->finalize();

            LOG_INFO("PPISP initialized: {} cameras (physical), {} frames, lr={:.2e}, warmup={}",
                     ppisp_->num_cameras(), ppisp_->num_frames(), params_.optimization.ppisp_lr, config.warmup_steps);

            if (auto result = apply_ppisp_sidecar_if_configured(); !result) {
                return result;
            }

            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to init PPISP: {}", e.what()));
        }
    }

    std::expected<PPISPFileMetadata, std::string> Trainer::build_ppisp_sidecar_metadata() const {
        if (!ppisp_ || !ppisp_->isFinalized()) {
            return std::unexpected("Cannot build PPISP sidecar metadata before PPISP is initialized");
        }
        if (!train_dataset_) {
            return std::unexpected("Cannot build PPISP sidecar metadata without an active training dataset");
        }

        PPISPFileMetadata metadata;
        metadata.dataset_path_utf8 = lfs::core::path_to_utf8(params_.dataset.data_path);
        metadata.images_folder = params_.dataset.images;
        metadata.camera_ids = ppisp_->ordered_camera_ids();

        for (const auto& cam : train_dataset_->get_cameras()) {
            if (!cam) {
                continue;
            }
            metadata.frame_image_names.push_back(cam->image_name());
            metadata.frame_camera_ids.push_back(cam->camera_id());
        }

        if (static_cast<int>(metadata.frame_image_names.size()) != ppisp_->num_frames() ||
            static_cast<int>(metadata.frame_camera_ids.size()) != ppisp_->num_frames()) {
            return std::unexpected(std::format(
                "PPISP metadata frame mismatch: metadata has {} names / {} camera ids but PPISP has {} frames",
                metadata.frame_image_names.size(),
                metadata.frame_camera_ids.size(),
                ppisp_->num_frames()));
        }
        if (static_cast<int>(metadata.camera_ids.size()) != ppisp_->num_cameras()) {
            return std::unexpected(std::format(
                "PPISP metadata camera mismatch: metadata has {} camera ids but PPISP has {} cameras",
                metadata.camera_ids.size(),
                ppisp_->num_cameras()));
        }

        return metadata;
    }

    std::expected<Trainer::PPISPSidecarMappings, std::string> Trainer::build_ppisp_sidecar_mappings(
        const PPISP& loaded_ppisp,
        const PPISPFileMetadata& metadata,
        const std::filesystem::path& sidecar_path) const {

        if (!ppisp_ || !ppisp_->isFinalized()) {
            return std::unexpected("Cannot apply PPISP sidecar before PPISP initialization is complete");
        }
        if (!train_dataset_) {
            return std::unexpected("Cannot apply PPISP sidecar without an active training dataset");
        }
        if (metadata.empty()) {
            return std::unexpected(std::format(
                "Frozen PPISP sidecar '{}' has no dataset metadata. Older sidecars cannot be verified against the current dataset; resave the source model with sidecar metadata first.",
                lfs::core::path_to_utf8(sidecar_path)));
        }
        if (static_cast<int>(metadata.frame_image_names.size()) != loaded_ppisp.num_frames() ||
            static_cast<int>(metadata.frame_camera_ids.size()) != loaded_ppisp.num_frames()) {
            return std::unexpected(std::format(
                "PPISP sidecar metadata frame count mismatch: metadata has {} names / {} camera ids but sidecar has {} frames",
                metadata.frame_image_names.size(),
                metadata.frame_camera_ids.size(),
                loaded_ppisp.num_frames()));
        }
        if (static_cast<int>(metadata.camera_ids.size()) != loaded_ppisp.num_cameras()) {
            return std::unexpected(std::format(
                "PPISP sidecar metadata camera count mismatch: metadata has {} camera ids but sidecar has {} cameras",
                metadata.camera_ids.size(),
                loaded_ppisp.num_cameras()));
        }

        const auto current_dataset_path = lfs::core::path_to_utf8(params_.dataset.data_path);
        if (!metadata.dataset_path_utf8.empty() && metadata.dataset_path_utf8 != current_dataset_path) {
            LOG_INFO("Frozen PPISP sidecar dataset path differs from current dataset path: '{}' vs '{}'",
                     metadata.dataset_path_utf8, current_dataset_path);
        }
        if (!metadata.images_folder.empty() && metadata.images_folder != params_.dataset.images) {
            LOG_INFO("Frozen PPISP sidecar images folder differs from current training config: '{}' vs '{}'",
                     metadata.images_folder, params_.dataset.images);
        }

        auto make_frame_key = [](std::string_view image_name, int camera_id) {
            return std::format("{}\n{}", image_name, camera_id);
        };

        std::unordered_map<std::string, int> source_frame_index_by_key;
        source_frame_index_by_key.reserve(metadata.frame_image_names.size());
        for (size_t i = 0; i < metadata.frame_image_names.size(); ++i) {
            auto [_, inserted] = source_frame_index_by_key.emplace(
                make_frame_key(metadata.frame_image_names[i], metadata.frame_camera_ids[i]),
                static_cast<int>(i));
            if (!inserted) {
                return std::unexpected(std::format(
                    "PPISP sidecar metadata contains duplicate frame key for image '{}' and camera {}",
                    metadata.frame_image_names[i],
                    metadata.frame_camera_ids[i]));
            }
        }

        PPISPSidecarMappings mappings;
        mappings.frame_mapping.reserve(static_cast<size_t>(ppisp_->num_frames()));
        std::unordered_set<std::string> seen_target_frames;
        seen_target_frames.reserve(static_cast<size_t>(ppisp_->num_frames()));
        for (const auto& cam : train_dataset_->get_cameras()) {
            if (!cam) {
                continue;
            }
            const auto key = make_frame_key(cam->image_name(), cam->camera_id());
            if (!seen_target_frames.insert(key).second) {
                return std::unexpected(std::format(
                    "Current training dataset contains duplicate frame key for image '{}' and camera {}",
                    cam->image_name(),
                    cam->camera_id()));
            }
            const auto it = source_frame_index_by_key.find(key);
            if (it == source_frame_index_by_key.end()) {
                return std::unexpected(std::format(
                    "Frozen PPISP sidecar is missing frame '{}' for camera {}",
                    cam->image_name(),
                    cam->camera_id()));
            }
            mappings.frame_mapping.push_back(it->second);
        }
        if (seen_target_frames.size() != source_frame_index_by_key.size()) {
            return std::unexpected(std::format(
                "Frozen PPISP sidecar dataset mismatch: sidecar has {} frame keys but current training dataset has {}",
                source_frame_index_by_key.size(),
                seen_target_frames.size()));
        }

        std::unordered_map<int, int> source_camera_index_by_id;
        source_camera_index_by_id.reserve(metadata.camera_ids.size());
        for (size_t i = 0; i < metadata.camera_ids.size(); ++i) {
            auto [_, inserted] = source_camera_index_by_id.emplace(metadata.camera_ids[i], static_cast<int>(i));
            if (!inserted) {
                return std::unexpected(std::format(
                    "PPISP sidecar metadata contains duplicate camera id {}",
                    metadata.camera_ids[i]));
            }
        }

        const auto target_camera_ids = ppisp_->ordered_camera_ids();
        mappings.camera_mapping.reserve(target_camera_ids.size());
        for (const int camera_id : target_camera_ids) {
            const auto it = source_camera_index_by_id.find(camera_id);
            if (it == source_camera_index_by_id.end()) {
                return std::unexpected(std::format(
                    "Frozen PPISP sidecar is missing camera id {} required by the current dataset",
                    camera_id));
            }
            mappings.camera_mapping.push_back(it->second);
        }
        if (target_camera_ids.size() != source_camera_index_by_id.size()) {
            return std::unexpected(std::format(
                "Frozen PPISP sidecar dataset mismatch: sidecar has {} camera ids but current training dataset uses {}",
                source_camera_index_by_id.size(),
                target_camera_ids.size()));
        }

        return mappings;
    }

    std::expected<void, std::string> Trainer::apply_ppisp_sidecar_if_configured() {
        if (!should_apply_ppisp_sidecar_on_init()) {
            return {};
        }
        if (!ppisp_ || !ppisp_->isFinalized()) {
            return std::unexpected("Cannot apply PPISP sidecar before PPISP initialization is complete");
        }

        PPISP loaded_ppisp(1);
        PPISPFileMetadata metadata;
        const auto sidecar_path = params_.optimization.ppisp_sidecar_path;

        if (auto result = load_ppisp_file(sidecar_path, loaded_ppisp, nullptr, &metadata); !result) {
            return std::unexpected(std::format(
                "Failed to load frozen PPISP sidecar '{}': {}",
                lfs::core::path_to_utf8(sidecar_path),
                result.error()));
        }

        auto mappings_result = build_ppisp_sidecar_mappings(loaded_ppisp, metadata, sidecar_path);
        if (!mappings_result) {
            return std::unexpected(mappings_result.error());
        }
        auto& mappings = *mappings_result;

        if (static_cast<int>(mappings.frame_mapping.size()) != ppisp_->num_frames()) {
            return std::unexpected(std::format(
                "Frozen PPISP sidecar frame mapping size mismatch: {} mappings for {} target frames",
                mappings.frame_mapping.size(),
                ppisp_->num_frames()));
        }
        if (static_cast<int>(mappings.camera_mapping.size()) != ppisp_->num_cameras()) {
            return std::unexpected(std::format(
                "Frozen PPISP sidecar camera mapping size mismatch: {} mappings for {} target cameras",
                mappings.camera_mapping.size(),
                ppisp_->num_cameras()));
        }

        if (auto result = ppisp_->copy_inference_weights_from(
                loaded_ppisp, mappings.frame_mapping, mappings.camera_mapping);
            !result) {
            return std::unexpected(std::format(
                "Failed to import frozen PPISP weights from '{}': {}",
                lfs::core::path_to_utf8(sidecar_path),
                result.error()));
        }

        LOG_INFO("Loaded frozen PPISP sidecar '{}' ({} cameras, {} frames{})",
                 lfs::core::path_to_utf8(sidecar_path),
                 loaded_ppisp.num_cameras(),
                 loaded_ppisp.num_frames(),
                 ", metadata-mapped");
        return {};
    }

    std::expected<void, std::string> Trainer::initialize_ppisp_controller() {
        if (!params_.optimization.ppisp_use_controller || !params_.optimization.use_ppisp) {
            return {};
        }

        if (!ppisp_) {
            return std::unexpected("PPISP must be initialized before controller");
        }

        try {
            const bool import_frozen_sidecar_controller = should_apply_ppisp_sidecar_on_init();
            const auto sidecar_path = params_.optimization.ppisp_sidecar_path;
            PPISPFileHeader sidecar_header{};
            if (import_frozen_sidecar_controller) {
                std::ifstream file;
                if (!lfs::core::open_file_for_read(sidecar_path, std::ios::binary, file)) {
                    return std::unexpected("Failed to open frozen PPISP sidecar: " +
                                           lfs::core::path_to_utf8(sidecar_path));
                }
                file.read(reinterpret_cast<char*>(&sidecar_header), sizeof(sidecar_header));
                if (!file) {
                    return std::unexpected("Failed to read frozen PPISP sidecar header: " +
                                           lfs::core::path_to_utf8(sidecar_path));
                }
                if (sidecar_header.magic != PPISP_FILE_MAGIC) {
                    return std::unexpected("Invalid frozen PPISP sidecar: wrong magic number");
                }
                if (sidecar_header.version > PPISP_FILE_VERSION) {
                    return std::unexpected("Unsupported frozen PPISP sidecar version: " +
                                           std::to_string(sidecar_header.version));
                }
                if (!has_flag(sidecar_header.flags, PPISPFileFlags::HAS_CONTROLLER)) {
                    LOG_INFO("Frozen PPISP sidecar '{}' has no controller pool; controller inference will remain disabled",
                             lfs::core::path_to_utf8(sidecar_path));
                    return {};
                }
            }

            PPISPControllerPool::Config config;
            config.lr = params_.optimization.ppisp_controller_lr;

            const int activation_step = params_.optimization.resolved_ppisp_controller_activation_step();
            if (params_.optimization.ppisp_controller_activation_step < 0) {
                params_.optimization.ppisp_controller_activation_step = activation_step;
            }
            int distillation_iters = static_cast<int>(params_.optimization.iterations) - activation_step;
            int num_cameras = ppisp_->num_cameras();

            ppisp_controller_pool_ = std::make_unique<PPISPControllerPool>(num_cameras, distillation_iters, config);

            size_t max_h = 0, max_w = 0;
            for (const auto& cam : train_dataset_->get_cameras()) {
                if (cam) {
                    max_h = std::max(max_h, static_cast<size_t>(cam->image_height()));
                    max_w = std::max(max_w, static_cast<size_t>(cam->image_width()));
                }
            }
            ppisp_controller_pool_->allocate_buffers(max_h, max_w);

            LOG_INFO("PPISP controller pool initialized: num_cameras={}, activation_step={}, lr={:.2e}, max_image={}x{}",
                     num_cameras, activation_step,
                     params_.optimization.ppisp_controller_lr, static_cast<int>(max_h), static_cast<int>(max_w));

            if (import_frozen_sidecar_controller) {
                PPISP loaded_ppisp(1);
                auto loaded_controller = std::make_unique<PPISPControllerPool>(
                    static_cast<int>(sidecar_header.num_cameras),
                    1);
                PPISPFileMetadata metadata;
                if (auto result = load_ppisp_file(sidecar_path, loaded_ppisp, loaded_controller.get(), &metadata); !result) {
                    return std::unexpected(std::format(
                        "Failed to load frozen PPISP controller sidecar '{}': {}",
                        lfs::core::path_to_utf8(sidecar_path),
                        result.error()));
                }
                auto mappings_result = build_ppisp_sidecar_mappings(loaded_ppisp, metadata, sidecar_path);
                if (!mappings_result) {
                    return std::unexpected(mappings_result.error());
                }
                if (const auto error = ppisp_controller_pool_->copy_inference_weights_from(
                        *loaded_controller, mappings_result->camera_mapping);
                    !error.empty()) {
                    return std::unexpected(std::format(
                        "Failed to import frozen PPISP controller weights from '{}': {}",
                        lfs::core::path_to_utf8(sidecar_path),
                        error));
                }
                LOG_INFO("Loaded frozen PPISP controller from '{}' ({} cameras)",
                         lfs::core::path_to_utf8(sidecar_path),
                         loaded_controller->num_cameras());
            }

            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to init PPISP controller pool: {}", e.what()));
        }
    }

    // Compute photometric loss AND gradient manually
    std::expected<std::pair<lfs::core::Tensor, lfs::core::Tensor>, std::string> Trainer::compute_photometric_loss_with_gradient(
        const lfs::core::Tensor& rendered,
        const lfs::core::Tensor& gt_image,
        const lfs::core::param::OptimizationParameters& opt_params) {
        lfs::training::losses::PhotometricLoss::Params params{.lambda_dssim = opt_params.lambda_dssim};
        auto result = photometric_loss_.forward(rendered, gt_image, params);
        if (!result) {
            return std::unexpected(result.error());
        }
        auto [loss_tensor, ctx] = *result;
        return std::make_pair(loss_tensor, ctx.grad_image);
    }

    std::expected<void, std::string> Trainer::validate_masks() {
        const auto& opt = params_.optimization;
        if (opt.mask_mode == lfs::core::param::MaskMode::None) {
            return {};
        }

        size_t alpha_count = 0;
        size_t masks_found = 0;
        for (const auto& cam : train_dataset_->get_cameras()) {
            if (cam && cam->has_alpha())
                ++alpha_count;
            if (cam && cam->has_mask())
                ++masks_found;
        }

        if (opt.use_alpha_as_mask && alpha_count > 0) {
            LOG_INFO("Using alpha channel as mask source ({}/{} cameras){}",
                     alpha_count, train_dataset_->get_cameras().size(),
                     opt.invert_masks ? " (inverted)" : "");
            return {};
        }

        if (masks_found == 0) {
            const auto path_str = lfs::core::path_to_utf8(params_.dataset.data_path);
            if (opt.use_alpha_as_mask) {
                return std::unexpected(std::format(
                    "Mask mode enabled with use_alpha_as_mask but no images have alpha and no mask files found in {}/masks/",
                    path_str));
            }
            return std::unexpected(std::format(
                "Mask mode enabled but no masks found in {}/masks/",
                path_str));
        }

        LOG_INFO("Found {} masks{}", masks_found, opt.invert_masks ? " (inverted)" : "");
        return {};
    }

    std::expected<Trainer::MaskLossResult, std::string> Trainer::compute_photometric_loss_with_mask(
        const lfs::core::Tensor& rendered,
        const lfs::core::Tensor& gt_image,
        const lfs::core::Tensor& mask,
        const lfs::core::Tensor& alpha,
        const lfs::core::param::OptimizationParameters& opt_params) {

        using namespace lfs::core;
        constexpr float EPSILON = 1e-8f;
        constexpr float ALPHA_CONSISTENCY_WEIGHT = 10.0f;

        const auto mode = opt_params.mask_mode;
        const Tensor mask_2d = mask.ndim() == 3 ? mask.squeeze(0) : mask;

        Tensor loss, grad, grad_alpha;

        if (mode == param::MaskMode::Segment || mode == param::MaskMode::Ignore) {
            if (opt_params.lambda_dssim > 0.0f) {
                // Use FUSED masked L1+SSIM kernel
                auto [loss_tensor, ctx] = lfs::training::kernels::masked_fused_l1_ssim_forward(
                    rendered, gt_image, mask_2d, opt_params.lambda_dssim, masked_fused_workspace_);

                grad = lfs::training::kernels::masked_fused_l1_ssim_backward(ctx, masked_fused_workspace_);
                loss = loss_tensor;

                // Squeeze gradient to match input dimensions (loss is scalar, no adjustment needed)
                if (grad.ndim() == 4 && rendered.ndim() == 3) {
                    grad = grad.squeeze(0);
                }
            } else {
                // Pure L1 with mask (no SSIM)
                const Tensor mask_3d = mask_2d.unsqueeze(0);
                const Tensor mask_sum = mask_2d.sum() * static_cast<float>(rendered.shape()[0]) + EPSILON;
                const Tensor diff = rendered - gt_image;
                const Tensor masked_l1 = (diff.abs() * mask_3d).sum() / mask_sum;
                const Tensor sign_diff = diff.sign();
                grad = sign_diff * mask_3d / mask_sum;
                loss = masked_l1;
            }

            // Segment: opacity penalty for background
            if (mode == param::MaskMode::Segment && alpha.is_valid()) {
                const Tensor alpha_2d = alpha.ndim() == 3 ? alpha.squeeze(0) : alpha;
                const Tensor bg_mask = Tensor::full(mask_2d.shape(), 1.0f, mask_2d.device()) - mask_2d;
                const Tensor penalty_weights = bg_mask.pow(opt_params.mask_opacity_penalty_power);
                const Tensor penalty = (alpha_2d * penalty_weights).mean() * opt_params.mask_opacity_penalty_weight;

                const float inv_pixels = opt_params.mask_opacity_penalty_weight / static_cast<float>(alpha_2d.numel());
                grad_alpha = penalty_weights * inv_pixels;
                loss = loss + penalty;
            }

        } else if (mode == param::MaskMode::AlphaConsistent) {
            // Standard photometric loss
            const lfs::training::losses::PhotometricLoss::Params params{.lambda_dssim = opt_params.lambda_dssim};
            auto result = photometric_loss_.forward(rendered, gt_image, params);
            if (!result) {
                return std::unexpected(result.error());
            }
            auto [photo_loss, ctx] = *result;
            loss = photo_loss;
            grad = ctx.grad_image;

            // Alpha should match mask
            if (alpha.is_valid()) {
                const Tensor alpha_2d = alpha.ndim() == 3 ? alpha.squeeze(0) : alpha;
                const Tensor alpha_loss = (alpha_2d - mask_2d).abs().mean() * ALPHA_CONSISTENCY_WEIGHT;
                loss = loss + alpha_loss;
                grad_alpha = (alpha_2d - mask_2d).sign() * (ALPHA_CONSISTENCY_WEIGHT / static_cast<float>(alpha_2d.numel()));
            }
        } else {
            auto fallback = compute_photometric_loss_with_gradient(rendered, gt_image, opt_params);
            if (!fallback) {
                return std::unexpected(fallback.error());
            }
            return MaskLossResult{.loss = fallback->first, .grad_image = fallback->second, .grad_alpha = {}};
        }

        return MaskLossResult{.loss = loss, .grad_image = grad, .grad_alpha = grad_alpha};
    }

    // Returns GPU tensor for loss - NO SYNC!
    std::expected<lfs::core::Tensor, std::string> Trainer::compute_scale_reg_loss(
        lfs::core::SplatData& splatData,
        AdamOptimizer& optimizer,
        const lfs::core::param::OptimizationParameters& opt_params) {
        lfs::training::losses::ScaleRegularization::Params params{.weight = opt_params.scale_reg};
        return lfs::training::losses::ScaleRegularization::forward(splatData.scaling_raw(), optimizer.get_grad(ParamType::Scaling), params);
    }

    // Returns GPU tensor for loss - NO SYNC!
    std::expected<lfs::core::Tensor, std::string> Trainer::compute_opacity_reg_loss(
        lfs::core::SplatData& splatData,
        AdamOptimizer& optimizer,
        const lfs::core::param::OptimizationParameters& opt_params) {
        lfs::training::losses::OpacityRegularization::Params params{.weight = opt_params.opacity_reg};
        return lfs::training::losses::OpacityRegularization::forward(splatData.opacity_raw(), optimizer.get_grad(ParamType::Opacity), params);
    }

    std::expected<std::pair<lfs::core::Tensor, SparsityLossContext>, std::string>
    Trainer::compute_sparsity_loss_forward(const int iter, const lfs::core::SplatData& splat_data) {
        if (!sparsity_optimizer_ || !sparsity_optimizer_->should_apply_loss(iter)) {
            auto zero = lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            return std::make_pair(std::move(zero), SparsityLossContext{});
        }

        if (!sparsity_optimizer_->is_initialized()) {
            if (auto result = sparsity_optimizer_->initialize(splat_data.opacity_raw()); !result) {
                return std::unexpected(result.error());
            }
            LOG_DEBUG("Sparsity optimizer initialized at iteration {}", iter);
        }

        return sparsity_optimizer_->compute_loss_forward(splat_data.opacity_raw());
    }

    std::expected<void, std::string> Trainer::handle_sparsity_update(const int iter, lfs::core::SplatData& splat_data) {
        if (!sparsity_optimizer_ || !sparsity_optimizer_->should_update(iter)) {
            return {};
        }
        return sparsity_optimizer_->update_state(splat_data.opacity_raw());
    }

    std::expected<void, std::string> Trainer::apply_sparsity_pruning(const int iter, lfs::core::SplatData& splat_data) {
        if (!sparsity_optimizer_ || !sparsity_optimizer_->should_prune(iter)) {
            return {};
        }

        auto mask_result = sparsity_optimizer_->get_prune_mask(splat_data.opacity_raw());
        if (!mask_result) {
            return std::unexpected(mask_result.error());
        }

        const int n_before = static_cast<int>(splat_data.size());
        strategy_->remove_gaussians(*mask_result);
        const int n_after = static_cast<int>(splat_data.size());

        LOG_INFO("Sparsity pruning: {} -> {} Gaussians ({}% reduction)",
                 n_before, n_after, static_cast<int>(100.0f * (n_before - n_after) / n_before));

        sparsity_optimizer_.reset();
        return {};
    }

    Trainer::Trainer(std::shared_ptr<CameraDataset> dataset,
                     std::unique_ptr<IStrategy> strategy,
                     std::optional<std::tuple<std::vector<std::string>, std::vector<std::string>>> provided_splits)
        : base_dataset_(std::move(dataset)),
          strategy_(std::move(strategy)),
          provided_splits_(std::move(provided_splits)) {
        // Check CUDA availability
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0) {
            throw std::runtime_error("CUDA is not available – aborting.");
        }

        cudaStreamCreateWithFlags(&callback_stream_, cudaStreamNonBlocking);

        LOG_DEBUG("Trainer constructed with {} cameras", base_dataset_->get_cameras().size());
    }

    Trainer::Trainer(lfs::core::Scene& scene)
        : scene_(&scene) {
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0) {
            throw std::runtime_error("CUDA is not available – aborting.");
        }

        cudaStreamCreateWithFlags(&callback_stream_, cudaStreamNonBlocking);

        if (!scene.hasTrainingData()) {
            throw std::runtime_error("Scene has no cameras");
        }

        LOG_DEBUG("Trainer constructed from Scene with {} cameras", scene.getAllCameras().size());
    }

    std::expected<void, std::string> Trainer::initialize(const lfs::core::param::TrainingParameters& params) {
        // Thread-safe initialization using mutex
        std::lock_guard<std::mutex> lock(init_mutex_);

        // Check again after acquiring lock (double-checked locking pattern)
        if (initialized_.load()) {
            LOG_INFO("Re-initializing trainer with new parameters");
            // Clean up existing state for re-initialization
            cleanup();
        }

        LOG_INFO("Initializing trainer with {} iterations", params.optimization.iterations);

        try {
            params_ = params;
            memory_breakdown_enabled_ = env_flag_enabled("LFS_MEM_BREAKDOWN");
            memory_breakdown_logged_init_ = false;
            memory_breakdown_logged_train_setup_ = false;
            memory_breakdown_logged_first_batch_ = false;
            memory_breakdown_logged_first_raster_ = false;
            memory_breakdown_logged_first_step_ = false;

            // Create DatasetConfig for lfs::training::CameraDataset
            lfs::training::DatasetConfig dataset_config;
            dataset_config.resize_factor = params.dataset.resize_factor;
            dataset_config.max_width = params.dataset.max_width;
            dataset_config.test_every = params.dataset.test_every;

            // Get source cameras from Scene nodes or base_dataset_
            std::vector<std::shared_ptr<lfs::core::Camera>> source_cameras;
            if (scene_) {
                source_cameras = scene_->getActiveCameras();
                if (source_cameras.empty()) {
                    return std::unexpected("Scene has no active cameras enabled for training");
                }
            } else if (base_dataset_) {
                source_cameras = base_dataset_->get_cameras();
            } else {
                return std::unexpected("No camera source available");
            }

            total_cameras_count_ = source_cameras.size();

            // Handle dataset split based on evaluation flag
            if (params.optimization.enable_eval) {
                // Create train/val split
                train_dataset_ = std::make_shared<CameraDataset>(
                    source_cameras, dataset_config, CameraDataset::Split::TRAIN,
                    provided_splits_ ? std::make_optional(std::get<0>(*provided_splits_)) : std::nullopt);
                val_dataset_ = std::make_shared<CameraDataset>(
                    source_cameras, dataset_config, CameraDataset::Split::VAL,
                    provided_splits_ ? std::make_optional(std::get<1>(*provided_splits_)) : std::nullopt);

                LOG_INFO("Created train/val split: {} train, {} val images",
                         train_dataset_->size(),
                         val_dataset_->size());
            } else {
                // Use all images for training
                train_dataset_ = std::make_shared<CameraDataset>(
                    source_cameras, dataset_config, CameraDataset::Split::ALL);
                val_dataset_ = nullptr;

                LOG_INFO("Using all {} images for training (no evaluation)",
                         train_dataset_->size());
            }

            train_dataset_size_ = train_dataset_->size();

            // If using Scene mode and no strategy yet, create one
            if (scene_ && !strategy_) {
                auto* model = scene_->getTrainingModel();
                if (!model) {
                    return std::unexpected("Scene has no training model set");
                }

                auto result = StrategyFactory::instance().create(params.optimization.strategy, *model);
                if (!result) {
                    return std::unexpected(result.error());
                }
                strategy_ = std::move(*result);
                LOG_DEBUG("Created {} strategy from Scene model", params.optimization.strategy);
            }

            auto& splat = strategy_->get_model();

            int max_cap = params.optimization.max_cap;
            if (max_cap < splat.size()) {
                LOG_WARN("Max cap is less than to {} initial splats {}. Choosing randomly {} splats", max_cap, splat.size(), max_cap);
                lfs::core::random_choose(splat, max_cap);
            }

            // Re-initialize strategy with new parameters
            strategy_->set_training_dataset(train_dataset_);
            strategy_->initialize(params.optimization);
            LOG_DEBUG("Strategy initialized");

            // Initialize bilateral grid if enabled
            if (auto result = initialize_bilateral_grid(); !result) {
                return std::unexpected(result.error());
            }

            // Initialize PPISP if enabled
            if (auto result = initialize_ppisp(); !result) {
                return std::unexpected(result.error());
            }

            // Initialize PPISP controller if enabled
            if (auto result = initialize_ppisp_controller(); !result) {
                return std::unexpected(result.error());
            }

            // Validate masks if mask mode is enabled
            if (auto result = validate_masks(); !result) {
                return std::unexpected(result.error());
            }

            // Apply undistortion to camera intrinsics (params already precomputed at load time)
            if (params.optimization.undistort) {
                int prepared = 0;
                for (auto& cam : train_dataset_->get_cameras()) {
                    if (cam && cam->has_distortion()) {
                        cam->prepare_undistortion();
                        ++prepared;
                    }
                }
                if (val_dataset_) {
                    for (auto& cam : val_dataset_->get_cameras()) {
                        if (cam && cam->has_distortion()) {
                            cam->prepare_undistortion();
                        }
                    }
                }
                if (prepared > 0) {
                    LOG_INFO("Prepared undistortion for {} cameras", prepared);
                }
            }

            // Initialize sparsity optimizer
            if (params.optimization.enable_sparsity) {
                constexpr int UPDATE_INTERVAL = 50;
                const int sparsify_steps = params.optimization.sparsify_steps;
                const int stored_iters = static_cast<int>(params.optimization.iterations);

                // Checkpoint already has total iterations; fresh start needs sparsify_steps added
                const bool is_resume = params.resume_checkpoint.has_value();
                const int base_iters = is_resume ? (stored_iters - sparsify_steps) : stored_iters;

                if (!is_resume) {
                    params_.optimization.iterations = static_cast<size_t>(base_iters + sparsify_steps);
                }

                const ADMMSparsityOptimizer::Config config{
                    .sparsify_steps = sparsify_steps,
                    .init_rho = params.optimization.init_rho,
                    .prune_ratio = params.optimization.prune_ratio,
                    .update_every = UPDATE_INTERVAL,
                    .start_iteration = base_iters};

                sparsity_optimizer_ = SparsityOptimizerFactory::create("admm", config);
                if (sparsity_optimizer_) {
                    LOG_INFO("Sparsity: base={}, steps={}, prune={:.0f}%",
                             base_iters, sparsify_steps, params.optimization.prune_ratio * 100);
                }
            }

            // Initialize background color tensor from params
            {
                const auto& bg_color = params.optimization.bg_color;
                background_ = lfs::core::Tensor::empty({3}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
                auto* bg_ptr = background_.ptr<float>();
                bg_ptr[0] = bg_color[0];
                bg_ptr[1] = bg_color[1];
                bg_ptr[2] = bg_color[2];
                background_ = background_.to(lfs::core::Device::CUDA);
                LOG_INFO("Background color set to RGB({:.2f}, {:.2f}, {:.2f})", bg_color[0], bg_color[1], bg_color[2]);
            }

            // Initialize image cache loader before any code path that calls getInstance()
            auto& cache_loader = lfs::io::CacheLoader::getInstance(
                params_.dataset.loading_params.use_cpu_memory,
                params_.dataset.loading_params.use_fs_cache);
            cache_loader.update_cache_params(
                params_.dataset.loading_params.use_cpu_memory,
                params_.dataset.loading_params.use_fs_cache,
                train_dataset_size_,
                params_.dataset.loading_params.min_cpu_free_GB,
                params_.dataset.loading_params.min_cpu_free_memory_ratio,
                params_.dataset.loading_params.print_cache_status,
                params_.dataset.loading_params.print_status_freq_num);

            // Load background image if specified
            if (params.optimization.bg_mode == lfs::core::param::BackgroundMode::Image &&
                !params.optimization.bg_image_path.empty() &&
                std::filesystem::exists(params.optimization.bg_image_path)) {
                try {
                    auto& loader = lfs::io::CacheLoader::getInstance();
                    lfs::io::LoadParams load_params{
                        .resize_factor = 1,
                        .max_width = 0, // No max width limit
                        .cuda_stream = nullptr};
                    bg_image_base_ = loader.load_cached_image(params.optimization.bg_image_path, load_params);
                    if (bg_image_base_.device() != lfs::core::Device::CUDA) {
                        bg_image_base_ = bg_image_base_.to(lfs::core::Device::CUDA);
                    }
                    if (bg_image_base_.shape()[0] != 3) {
                        LOG_WARN("Background image has {} channels, expected 3 (RGB)", bg_image_base_.shape()[0]);
                        bg_image_base_ = {};
                        params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                    } else {
                        LOG_INFO("Background image: {} [{}x{}]",
                                 lfs::core::path_to_utf8(params.optimization.bg_image_path),
                                 bg_image_base_.shape()[2], bg_image_base_.shape()[1]);
                    }
                } catch (const std::exception& e) {
                    LOG_WARN("Failed to load background image: {}", e.what());
                    params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                }
            }

            // Create progress bar based on headless flag
            if (params.optimization.headless) {
                progress_ = std::make_unique<TrainingProgress>(
                    params_.optimization.iterations, // This now includes sparsity steps if enabled
                    /*update_frequency=*/100);
                LOG_DEBUG("Progress bar initialized for {} total iterations", params_.optimization.iterations);
            }

            // Initialize the evaluator - it handles all metrics internally
            evaluator_ = std::make_unique<lfs::training::MetricsEvaluator>(params_);
            LOG_DEBUG("Metrics evaluator initialized");

            // Resume from checkpoint if provided
            if (params_.resume_checkpoint.has_value()) {
                auto resume_result = load_checkpoint(*params_.resume_checkpoint);
                if (!resume_result) {
                    return std::unexpected(std::format("Failed to resume from checkpoint: {}", resume_result.error()));
                }
                LOG_INFO("Resumed training from checkpoint at iteration {}", *resume_result);

                // Reload bg_image if checkpoint restored different settings
                if (params_.optimization.bg_mode == lfs::core::param::BackgroundMode::Image &&
                    !params_.optimization.bg_image_path.empty() &&
                    std::filesystem::exists(params_.optimization.bg_image_path) &&
                    !bg_image_base_.is_valid()) {
                    try {
                        auto& loader = lfs::io::CacheLoader::getInstance();
                        lfs::io::LoadParams load_params{.resize_factor = 1, .max_width = 0, .cuda_stream = nullptr};
                        bg_image_base_ = loader.load_cached_image(params_.optimization.bg_image_path, load_params);
                        if (bg_image_base_.device() != lfs::core::Device::CUDA) {
                            bg_image_base_ = bg_image_base_.to(lfs::core::Device::CUDA);
                        }
                        if (bg_image_base_.shape()[0] != 3) {
                            LOG_WARN("Background image has {} channels, expected 3", bg_image_base_.shape()[0]);
                            bg_image_base_ = {};
                            params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                        } else {
                            LOG_INFO("Background image from checkpoint: {} [{}x{}]",
                                     lfs::core::path_to_utf8(params_.optimization.bg_image_path),
                                     bg_image_base_.shape()[2], bg_image_base_.shape()[1]);
                        }
                    } catch (const std::exception& e) {
                        LOG_WARN("Failed to load background image from checkpoint: {}", e.what());
                        params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                    }
                }
            }

            // Print configuration
            LOG_INFO("Visualization: {}", params.optimization.headless ? "disabled" : "enabled");
            LOG_INFO("Strategy: {}", params.optimization.strategy);
            if (params.optimization.mask_mode != lfs::core::param::MaskMode::None) {
                static constexpr const char* MASK_MODE_NAMES[] = {"none", "segment", "ignore", "alpha_consistent"};
                LOG_INFO("Mask mode: {}", MASK_MODE_NAMES[static_cast<int>(params.optimization.mask_mode)]);
            }
            if (current_iteration_ > 0) {
                LOG_INFO("Starting from iteration: {}", current_iteration_.load());
            }

            // Expose initial snapshot for Python control (iteration 0)
            {
                lfs::training::HookContext ctx{
                    .iteration = current_iteration_.load(),
                    .loss = current_loss_.load(),
                    .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
                    .is_refining = strategy_ ? strategy_->is_refining(current_iteration_.load()) : false,
                    .trainer = this};
                lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);
                lfs::training::CommandCenter::instance().update_snapshot(
                    ctx, params_.optimization.iterations, is_paused_.load(), is_running_.load(), stop_requested_.load(),
                    lfs::training::TrainingPhase::SafeControl);
            }

            // Execute configured Python scripts to register iteration callbacks
            if (!python_scripts_.empty()) {
                auto py_result = lfs::python::run_scripts(python_scripts_);
                if (!py_result) {
                    return std::unexpected(std::format("Failed to run Python scripts: {}", py_result.error()));
                }
            }

            initialized_ = true;

            if (memory_breakdown_enabled_ && !memory_breakdown_logged_init_) {
                const auto snapshot = capture_vram_snapshot(true);
                log_vram_snapshot("after_trainer_initialize", snapshot);

                std::vector<std::pair<std::string, size_t>> direct_entries;
                const auto& initialized_splat = strategy_->get_model();
                add_tensor_entry(direct_entries, "splat.means", initialized_splat.means());
                add_tensor_entry(direct_entries, "splat.sh0", initialized_splat.sh0());
                add_tensor_entry(direct_entries, "splat.shN", initialized_splat.shN());
                add_tensor_entry(direct_entries, "splat.scaling", initialized_splat.scaling_raw());
                add_tensor_entry(direct_entries, "splat.rotation", initialized_splat.rotation_raw());
                add_tensor_entry(direct_entries, "splat.opacity", initialized_splat.opacity_raw());
                add_tensor_entry(direct_entries, "splat.deleted", initialized_splat.deleted());
                add_tensor_entry(direct_entries, "splat.densification_info", initialized_splat._densification_info);
                log_entry_bytes("direct_splat", direct_entries);

                std::vector<std::pair<std::string, size_t>> optimizer_entries;
                for (const auto type : AdamOptimizer::all_param_types()) {
                    const auto* state = strategy_->get_optimizer().get_state(type);
                    if (!state) {
                        continue;
                    }
                    const std::string prefix = std::string("optimizer.") +
                                               (type == ParamType::Means      ? "means"
                                                : type == ParamType::Sh0      ? "sh0"
                                                : type == ParamType::ShN      ? "shN"
                                                : type == ParamType::Scaling  ? "scaling"
                                                : type == ParamType::Rotation ? "rotation"
                                                                              : "opacity");
                    add_tensor_entry(optimizer_entries, prefix + ".grad", state->grad);
                    add_tensor_entry(optimizer_entries, prefix + ".exp_avg", state->exp_avg);
                    add_tensor_entry(optimizer_entries, prefix + ".exp_avg_sq", state->exp_avg_sq);
                }
                log_entry_bytes("direct_optimizer", optimizer_entries);

                std::vector<std::pair<std::string, size_t>> trainer_entries;
                add_tensor_entry(trainer_entries, "trainer.background", background_);
                add_tensor_entry(trainer_entries, "trainer.bg_mix_buffer", bg_mix_buffer_);
                add_tensor_entry(trainer_entries, "trainer.bg_image_base", bg_image_base_);
                add_tensor_entry(trainer_entries, "trainer.random_bg_buffer", random_bg_buffer_);
                add_tensor_entry(trainer_entries, "trainer.loss_accumulator", loss_accumulator_);
                add_tensor_entry(trainer_entries, "trainer.densification_error_map", densification_error_map_);
                add_tensor_entry(trainer_entries, "trainer.pipelined_mask", pipelined_mask_);
                if (const size_t cache_bytes = bg_image_cache_bytes(bg_image_cache_); cache_bytes > 0) {
                    trainer_entries.emplace_back("trainer.bg_image_cache", cache_bytes);
                }
                if (const size_t bytes = photometric_workspace_bytes(photometric_loss_); bytes > 0) {
                    trainer_entries.emplace_back("trainer.photometric_workspaces", bytes);
                }
                if (const size_t bytes = masked_fused_workspace_bytes(masked_fused_workspace_); bytes > 0) {
                    trainer_entries.emplace_back("trainer.masked_fused_workspace", bytes);
                }
                if (const size_t bytes = ssim_map_workspace_bytes(densification_ssim_workspace_); bytes > 0) {
                    trainer_entries.emplace_back("trainer.densification_ssim_workspace", bytes);
                }
                log_entry_bytes("pool_backed_trainer", trainer_entries);

                LOG_INFO("[MEM] model logical_size={} capacity={} direct_model_plus_optimizer={:.2f} MiB",
                         strategy_->get_model().size(),
                         strategy_->get_model().means().capacity(),
                         bytes_to_mib(splat_tensor_bytes(strategy_->get_model()) +
                                      optimizer_state_bytes(strategy_->get_optimizer())));
                memory_breakdown_logged_init_ = true;
            }

            LOG_INFO("Trainer initialization complete");
            return {};
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to initialize trainer: {}", e.what()));
        }
    }

    Trainer::~Trainer() {
        shutdown();
    }

    std::shared_ptr<lfs::io::PipelinedImageLoader> Trainer::getActiveImageLoader() const {
        std::lock_guard<std::mutex> lock(active_image_loader_mutex_);
        return active_image_loader_;
    }

    Trainer::GTLoadConfigSnapshot Trainer::getGTLoadConfigSnapshot() const {
        std::lock_guard<std::mutex> lock(gt_load_config_mutex_);
        return gt_load_config_snapshot_;
    }

    void Trainer::updateGTLoadConfigSnapshot() {
        GTLoadConfigSnapshot snapshot;
        if (train_dataset_) {
            snapshot.resize_factor = std::max(1, train_dataset_->get_resize_factor());
            snapshot.max_width = train_dataset_->get_max_width();

            for (const auto& cam : train_dataset_->get_cameras()) {
                if (cam && cam->is_undistort_prepared()) {
                    snapshot.undistort = true;
                    break;
                }
            }
        }

        std::lock_guard<std::mutex> lock(gt_load_config_mutex_);
        gt_load_config_snapshot_ = snapshot;
    }

    void Trainer::setActiveImageLoader(std::shared_ptr<lfs::io::PipelinedImageLoader> loader) {
        std::lock_guard<std::mutex> lock(active_image_loader_mutex_);
        active_image_loader_ = std::move(loader);
    }

    void Trainer::clearActiveImageLoader() {
        if (strategy_) {
            strategy_->set_image_loader(nullptr);
        }
        setActiveImageLoader(nullptr);
    }

    void Trainer::shutdown() {
        if (shutdown_complete_.exchange(true)) {
            return;
        }

        LOG_DEBUG("Trainer shutdown");
        stop_requested_ = true;

        lfs::core::image_io::wait_for_pending_saves();

        if (callback_stream_) {
            cudaStreamSynchronize(callback_stream_);
            cudaStreamDestroy(callback_stream_);
            callback_stream_ = nullptr;
        }
        callback_busy_ = false;

        cudaDeviceSynchronize();

        clearActiveImageLoader();
        strategy_.reset();
        bilateral_grid_.reset();
        ppisp_.reset();
        ppisp_controller_pool_.reset();
        sparsity_optimizer_.reset();
        evaluator_.reset();
        progress_.reset();
        train_dataset_.reset();
        val_dataset_.reset();

        // Release GPU memory pools back to system
        lfs::core::Tensor::trim_memory_pool();
        lfs::core::GlobalArenaManager::instance().get_arena().full_reset();
        cudaDeviceSynchronize();
        LOG_DEBUG("GPU memory released");

        initialized_ = false;
        is_running_ = false;
        training_complete_ = false;
    }

    void Trainer::setParams(const lfs::core::param::TrainingParameters& params) {
        // Check if background image path changed and needs to be (re)loaded
        const bool bg_image_path_changed =
            params.optimization.bg_image_path != params_.optimization.bg_image_path;
        const bool bg_mode_is_image =
            params.optimization.bg_mode == lfs::core::param::BackgroundMode::Image;

        // Update params first
        params_ = params;

        // Load/reload background image if needed
        if (bg_mode_is_image && bg_image_path_changed &&
            !params.optimization.bg_image_path.empty() &&
            std::filesystem::exists(params.optimization.bg_image_path)) {
            try {
                auto& loader = lfs::io::CacheLoader::getInstance();
                lfs::io::LoadParams load_params{
                    .resize_factor = 1,
                    .max_width = 0,
                    .cuda_stream = nullptr};
                bg_image_base_ = loader.load_cached_image(params.optimization.bg_image_path, load_params);
                if (bg_image_base_.device() != lfs::core::Device::CUDA) {
                    bg_image_base_ = bg_image_base_.to(lfs::core::Device::CUDA);
                }
                bg_image_cache_.clear();
                if (bg_image_base_.shape()[0] != 3) {
                    LOG_WARN("Background image has {} channels, expected 3 (RGB)", bg_image_base_.shape()[0]);
                    bg_image_base_ = {};
                    params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
                } else {
                    LOG_INFO("Background image: {} [{}x{}]",
                             lfs::core::path_to_utf8(params.optimization.bg_image_path),
                             bg_image_base_.shape()[2], bg_image_base_.shape()[1]);
                }
            } catch (const std::exception& e) {
                LOG_WARN("Failed to load background image: {}", e.what());
                params_.optimization.bg_mode = lfs::core::param::BackgroundMode::SolidColor;
            }
        }

        if (!bg_mode_is_image && (bg_image_base_.is_valid() || !bg_image_cache_.empty())) {
            bg_image_cache_.clear();
            bg_image_base_ = {};
        }

        // Update background color tensor if changed
        const auto& bg_color = params.optimization.bg_color;
        if (background_.is_valid()) {
            auto bg_cpu = lfs::core::Tensor::empty({3}, lfs::core::Device::CPU, lfs::core::DataType::Float32);
            auto* bg_ptr = bg_cpu.ptr<float>();
            bg_ptr[0] = bg_color[0];
            bg_ptr[1] = bg_color[1];
            bg_ptr[2] = bg_color[2];
            background_ = bg_cpu.to(lfs::core::Device::CUDA);
        }
    }

    void Trainer::handle_control_requests(int iter, std::stop_token stop_token) {
        // Check stop token first
        if (stop_token.stop_requested()) {
            stop_requested_ = true;
            return;
        }

        // Handle pause/resume
        if (pause_requested_.load() && !is_paused_.load()) {
            is_paused_ = true;
            if (progress_) {
                progress_->pause();
            }
            LOG_INFO("Training paused at iteration {}", iter);
            LOG_DEBUG("Click 'Resume Training' to continue.");
        } else if (!pause_requested_.load() && is_paused_.load()) {
            is_paused_ = false;
            if (progress_) {
                progress_->resume(iter, current_loss_.load(), static_cast<int>(strategy_->get_model().size()));
            }
            LOG_INFO("Training resumed at iteration {}", iter);
        }

        if (save_requested_.exchange(false)) {
            LOG_INFO("Saving checkpoint and PLY at iteration {}...", iter);
            save_ply(params_.dataset.output_path, iter, /*join=*/false);
            auto result = save_checkpoint(iter);
            if (result) {
                const auto checkpoint_path = lfs::training::checkpoint_output_path(params_.dataset.output_path);
                LOG_INFO("Checkpoint and PLY saved to {} (checkpoint: {})",
                         lfs::core::path_to_utf8(params_.dataset.output_path),
                         lfs::core::path_to_utf8(checkpoint_path));
            } else {
                LOG_ERROR("Failed to save checkpoint: {}", result.error());
            }
        }

        // Handle stop request - this permanently stops training
        if (stop_requested_.load()) {
            LOG_INFO("Stopping training permanently at iteration {}...", iter);
            LOG_DEBUG("Saving final model...");
            save_ply(params_.dataset.output_path, iter, /*join=*/true);
            is_running_ = false;
        }
    }

    inline float inv_weight_piecewise(int step, int max_steps) {
        // Phases by fraction of training
        const float phase = std::max(0.f, std::min(1.f, step / float(std::max(1, max_steps))));

        const float limit_hi = 1.0f / 4.0f;  // start limit
        const float limit_mid = 2.0f / 4.0f; // middle limit
        const float limit_lo = 3.0f / 4.0f;  // final limit

        const float weight_hi = 1.0f;  // start weight
        const float weight_mid = 0.5f; // middle weight
        const float weight_lo = 0.0f;  // final weight

        if (phase < limit_hi) {
            return weight_hi; // hold until bypasses the start limit
        } else if (phase < limit_mid) {
            const float t = (phase - limit_hi) / (limit_mid - limit_hi);
            return weight_hi + (weight_mid - weight_hi) * t; // decay to mid value
        } else {
            const float t = (phase - limit_mid) / (limit_lo - limit_mid);
            return weight_mid + (weight_lo - weight_mid) * t; // decay to final value
        }
    }

    namespace {
        constexpr float TWO_PI = static_cast<float>(M_PI * 2.0);
        constexpr float PHASE_OFFSET_G = TWO_PI / 3.0f;
        constexpr float PHASE_OFFSET_B = TWO_PI * 2.0f / 3.0f;
        constexpr float CLAMP_EPS = 1e-4f;
        constexpr int BG_PERIOD_R = 37;
        constexpr int BG_PERIOD_G = 41;
        constexpr int BG_PERIOD_B = 43;
    } // anonymous namespace

    lfs::core::Tensor& Trainer::background_for_step(int iter) {
        if (!params_.optimization.bg_modulation) {
            return background_;
        }

        const float w = inv_weight_piecewise(iter, params_.optimization.iterations);
        if (w <= 0.0f) {
            return background_;
        }

        // Sine-based RGB with prime periods for color diversity
        const float pr = TWO_PI * static_cast<float>(iter % BG_PERIOD_R) / BG_PERIOD_R;
        const float pg = TWO_PI * static_cast<float>(iter % BG_PERIOD_G) / BG_PERIOD_G;
        const float pb = TWO_PI * static_cast<float>(iter % BG_PERIOD_B) / BG_PERIOD_B;

        const float result[3] = {
            std::clamp(0.5f * (1.0f + std::sin(pr)) * w, CLAMP_EPS, 1.0f - CLAMP_EPS),
            std::clamp(0.5f * (1.0f + std::sin(pg + PHASE_OFFSET_G)) * w, CLAMP_EPS, 1.0f - CLAMP_EPS),
            std::clamp(0.5f * (1.0f + std::sin(pb + PHASE_OFFSET_B)) * w, CLAMP_EPS, 1.0f - CLAMP_EPS)};

        if (bg_mix_buffer_.is_empty()) {
            bg_mix_buffer_ = lfs::core::Tensor::empty({3}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        }

        cudaMemcpyAsync(bg_mix_buffer_.ptr<float>(), result, sizeof(result), cudaMemcpyHostToDevice, bg_mix_buffer_.stream());
        return bg_mix_buffer_;
    }

    lfs::core::Tensor Trainer::get_background_image_for_camera(int width, int height) {
        // Return empty tensor if no background image is loaded
        if (!bg_image_base_.is_valid() || bg_image_base_.is_empty()) {
            return lfs::core::Tensor();
        }

        // Check cache first - key is (height << 32) | width
        const uint64_t cache_key = (static_cast<uint64_t>(height) << 32) | static_cast<uint64_t>(width);
        auto it = bg_image_cache_.find(cache_key);
        if (it != bg_image_cache_.end()) {
            return it->second;
        }

        // Resize background image to match camera dimensions
        const int src_h = static_cast<int>(bg_image_base_.shape()[1]);
        const int src_w = static_cast<int>(bg_image_base_.shape()[2]);
        const int channels = static_cast<int>(bg_image_base_.shape()[0]);

        // If dimensions match, use the original
        if (src_w == width && src_h == height) {
            bg_image_cache_[cache_key] = bg_image_base_;
            return bg_image_base_;
        }

        // Create resized tensor
        auto resized = lfs::core::Tensor::empty(
            {static_cast<size_t>(channels), static_cast<size_t>(height), static_cast<size_t>(width)},
            lfs::core::Device::CUDA,
            lfs::core::DataType::Float32);

        // Use bilinear resize kernel
        kernels::launch_bilinear_resize_chw(
            bg_image_base_.ptr<float>(),
            resized.ptr<float>(),
            channels,
            src_h, src_w,
            height, width,
            resized.stream());

        // Cache the resized image
        bg_image_cache_[cache_key] = resized;
        LOG_DEBUG("Background image resized: {}x{} -> {}x{}", src_w, src_h, width, height);

        return resized;
    }

    lfs::core::Tensor Trainer::get_random_background_for_camera(int width, int height, int iteration) {
        const size_t required_size = 3 * static_cast<size_t>(height) * static_cast<size_t>(width);

        if (!random_bg_buffer_.is_valid() || random_bg_buffer_.numel() != required_size) {
            random_bg_buffer_ = lfs::core::Tensor::empty(
                {3, static_cast<size_t>(height), static_cast<size_t>(width)},
                lfs::core::Device::CUDA,
                lfs::core::DataType::Float32);
        }

        kernels::launch_random_background(
            random_bg_buffer_.ptr<float>(),
            height, width,
            static_cast<uint64_t>(iteration),
            random_bg_buffer_.stream());

        return random_bg_buffer_;
    }

    std::expected<Trainer::StepResult, std::string> Trainer::train_step(
        int iter,
        lfs::core::Camera* cam,
        lfs::core::Tensor gt_image,
        RenderMode render_mode,
        std::stop_token stop_token) {
        try {
            if (params_.optimization.gut) {
                if (cam->camera_model_type() == core::CameraModelType::ORTHO) {
                    return std::unexpected("Training on cameras with ortho model is not supported yet.");
                }
            } else if (!params_.optimization.undistort || !cam->is_undistort_prepared()) {
                if (cam->radial_distortion().numel() != 0 ||
                    cam->tangential_distortion().numel() != 0) {
                    return std::unexpected("Distorted images detected. Use --gut or --undistort to train on cameras with distortion.");
                }
                if (cam->camera_model_type() != core::CameraModelType::PINHOLE) {
                    return std::unexpected("Use --gut or --undistort to train on cameras with non-pinhole model.");
                }
            }

            current_iteration_ = iter;

            // Check control requests at the beginning
            handle_control_requests(iter, stop_token);

            if (on_iteration_start_)
                on_iteration_start_();

            // Python hook: iteration start (safe, pre-forward)
            {
                lfs::training::HookContext ctx{
                    .iteration = iter,
                    .loss = current_loss_.load(),
                    .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
                    .is_refining = strategy_ ? strategy_->is_refining(iter) : false,
                    .trainer = this};
                lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::IterationStart);
                lfs::training::CommandCenter::instance().update_snapshot(
                    ctx, params_.optimization.iterations, is_paused_.load(), is_running_.load(), stop_requested_.load(),
                    lfs::training::TrainingPhase::IterationStart);
                lfs::training::ControlBoundary::instance().notify(lfs::training::ControlHook::IterationStart, ctx);
                auto view = lfs::training::CommandCenter::instance().snapshot();
                lfs::training::CommandCenter::instance().drain_enqueued(view);
            }

            // Training step entering forward/backward/optimizer region (commands blocked)
            lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::Forward);

            // If stop requested, return Stop
            if (stop_requested_.load() || stop_token.stop_requested()) {
                return StepResult::Stop;
            }

            // If paused, wait
            while (is_paused_.load() && !stop_requested_.load() && !stop_token.stop_requested()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                handle_control_requests(iter, stop_token);
            }

            // Check stop again after potential pause
            if (stop_requested_.load() || stop_token.stop_requested()) {
                return StepResult::Stop;
            }

            nvtxRangePush("background_for_step");
            lfs::core::Tensor& bg = background_for_step(iter);
            nvtxRangePop();

            lfs::core::Tensor bg_image;
            if (params_.optimization.bg_mode == lfs::core::param::BackgroundMode::Image) {
                bg_image = get_background_image_for_camera(cam->image_width(), cam->image_height());
            } else if (params_.optimization.bg_mode == lfs::core::param::BackgroundMode::Random) {
                bg_image = get_random_background_for_camera(cam->image_width(), cam->image_height(), iter);
            }

            // Configurable tile-based training to reduce peak memory
            const int full_width = cam->image_width();
            const int full_height = cam->image_height();

            // Read tile mode from parameters (1=1 tile, 2=2 tiles, 4=4 tiles)
            const TileMode tile_mode = static_cast<TileMode>(params_.optimization.tile_mode);

            // Determine tile configuration
            int tile_rows = 1, tile_cols = 1;
            switch (tile_mode) {
            case TileMode::One:
                tile_rows = 1;
                tile_cols = 1;
                break;
            case TileMode::Two:
                tile_rows = 2;
                tile_cols = 1;
                break;
            case TileMode::Four:
                tile_rows = 2;
                tile_cols = 2;
                break;
            }

            const int tile_width = full_width / tile_cols;
            const int tile_height = full_height / tile_rows;
            const int num_tiles = tile_rows * tile_cols;

            if (!loss_accumulator_.is_valid()) {
                loss_accumulator_ = core::Tensor::zeros({1}, core::Device::CUDA);
            } else {
                loss_accumulator_.zero_();
            }
            auto& loss_tensor_gpu = loss_accumulator_;
            RenderOutput r_output;
            int tiles_processed = 0;

            // Determine controller phase before tile loop (does not depend on tile results)
            const bool known_ppisp_camera = ppisp_ && ppisp_->is_known_camera(cam->camera_id());
            const int ppisp_cam_idx = known_ppisp_camera ? ppisp_->camera_index(cam->camera_id()) : -1;
            const int ppisp_activation_step = params_.optimization.resolved_ppisp_controller_activation_step();
            const bool ppisp_frozen = is_ppisp_frozen();
            const bool in_controller_phase = ppisp_controller_pool_ && known_ppisp_camera &&
                                             params_.optimization.ppisp_use_controller &&
                                             !ppisp_frozen &&
                                             params_.optimization.ppisp_freeze_gaussians_on_distill &&
                                             iter >= ppisp_activation_step &&
                                             ppisp_cam_idx >= 0 &&
                                             ppisp_cam_idx < ppisp_controller_pool_->num_cameras();
            const bool use_pixel_error_densification =
                (params_.optimization.strategy == "mcmc") ||
                (params_.optimization.strategy == "igs+") ||
                (params_.optimization.strategy == "lfs" &&
                 params_.optimization.use_error_map);
            const bool use_ssim_error = use_pixel_error_densification;
            DensificationType densification_type = DensificationType::None;
            if (params_.optimization.strategy == "mcmc")
                densification_type = DensificationType::MCMC;
            else if (params_.optimization.strategy == "lfs")
                densification_type = DensificationType::LFS;

            // Loop over tiles (row-major order)
            for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
                const int tile_row = tile_idx / tile_cols;
                const int tile_col = tile_idx % tile_cols;
                const int tile_x_offset = tile_col * tile_width;
                const int tile_y_offset = tile_row * tile_height;

                nvtxRangePush(std::format("tile_{}x{}", tile_row, tile_col).c_str());

                // Extract GT image tile
                lfs::core::Tensor gt_tile;
                if (num_tiles == 1) {
                    // No tiling - use full image
                    gt_tile = gt_image;
                } else if (gt_image.shape()[0] == 3) {
                    // CHW layout: gt_image is [3, H, W]
                    // Slice both height and width dimensions
                    auto tile_h = gt_image.slice(1, tile_y_offset, tile_y_offset + tile_height);
                    gt_tile = tile_h.slice(2, tile_x_offset, tile_x_offset + tile_width);
                } else {
                    // HWC layout: gt_image is [H, W, 3]
                    auto tile_h = gt_image.slice(0, tile_y_offset, tile_y_offset + tile_height);
                    gt_tile = tile_h.slice(1, tile_x_offset, tile_x_offset + tile_width);
                }

                // Extract background image tile (if using background image)
                lfs::core::Tensor bg_tile;
                if (bg_image.is_valid() && !bg_image.is_empty()) {
                    if (num_tiles == 1) {
                        // No tiling - use full image
                        bg_tile = bg_image;
                    } else {
                        // CHW layout: bg_image is [3, H, W]
                        // Slice both height and width dimensions
                        auto tile_h = bg_image.slice(1, tile_y_offset, tile_y_offset + tile_height);
                        bg_tile = tile_h.slice(2, tile_x_offset, tile_x_offset + tile_width);
                    }
                }

                // Render the tile
                nvtxRangePush("rasterize_forward");

                // Storage for render output (used by both paths)
                RenderOutput output;
                std::optional<FastRasterizeContext> fast_ctx;
                std::optional<GsplatRasterizeContext> gsplat_ctx;

                if (params_.optimization.gut) {
                    const int tw = (num_tiles > 1) ? tile_width : 0;
                    const int th = (num_tiles > 1) ? tile_height : 0;
                    auto rasterize_result = gsplat_rasterize_forward(
                        *cam, strategy_->get_model(), bg,
                        tile_x_offset, tile_y_offset, tw, th,
                        1.0f, false, GsplatRenderMode::RGB, true, bg_tile);

                    if (!rasterize_result) {
                        nvtxRangePop(); // rasterize_forward
                        nvtxRangePop(); // tile
                        return std::unexpected(rasterize_result.error());
                    }

                    output = std::move(rasterize_result->first);
                    gsplat_ctx.emplace(std::move(rasterize_result->second));
                } else {
                    // Standard mode: use fast rasterizer with tiling support
                    auto rasterize_result = fast_rasterize_forward(
                        *cam, strategy_->get_model(), bg,
                        tile_x_offset, tile_y_offset,
                        (num_tiles > 1) ? tile_width : 0, // 0 means full image
                        (num_tiles > 1) ? tile_height : 0,
                        params_.optimization.mip_filter, bg_tile);

                    // Check for OOM error
                    if (!rasterize_result) {
                        const std::string& error = rasterize_result.error();
                        if (error.find("OUT_OF_MEMORY") != std::string::npos) {
                            nvtxRangePop(); // rasterize_forward
                            nvtxRangePop(); // tile

                            // Handle OOM by switching tile mode
                            if (tile_mode == TileMode::Four) {
                                // Already at maximum tiling - can't tile further, return error
                                LOG_ERROR("OUT OF MEMORY at maximum tile mode (2x2). Cannot continue training.");
                                LOG_ERROR("Arena error: {}", error);
                                return std::unexpected(error);
                            } else {
                                // Upgrade to next tile mode
                                TileMode new_mode = (tile_mode == TileMode::One) ? TileMode::Two : TileMode::Four;
                                LOG_WARN("OUT OF MEMORY detected. Switching tile mode from {} to {}",
                                         static_cast<int>(tile_mode), static_cast<int>(new_mode));
                                LOG_WARN("Arena error: {}", error);
                                params_.optimization.tile_mode = static_cast<int>(new_mode);

                                // Retry this step with new tile mode
                                return std::unexpected("OOM_RETRY"); // Signal to retry the step
                            }
                        } else {
                            // Non-OOM error - propagate
                            nvtxRangePop();
                            nvtxRangePop();
                            return std::unexpected(error);
                        }
                    }

                    output = std::move(rasterize_result->first);
                    fast_ctx.emplace(std::move(rasterize_result->second));

                    if (fast_ctx->forward_ctx.n_visible_primitives == 0) {
                        auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
                        arena.end_frame(fast_ctx->forward_ctx.frame_id);
                        nvtxRangePop();
                        nvtxRangePop();
                        continue;
                    }
                }

                r_output = output; // Save last tile for densification
                nvtxRangePop();

                if (in_controller_phase) {
                    // Controller phase: forward through ISP with controller params, photometric loss,
                    // backward only through controller (base params frozen)
                    nvtxRangePush("controller_phase");
                    auto cleanup_controller_tile_context = [&]() {
                        auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
                        if (fast_ctx) {
                            arena.end_frame(fast_ctx->forward_ctx.frame_id);
                        } else if (gsplat_ctx) {
                            if (gsplat_ctx->isect_ids_ptr != nullptr) {
                                cudaFree(gsplat_ctx->isect_ids_ptr);
                                gsplat_ctx->isect_ids_ptr = nullptr;
                            }
                            if (gsplat_ctx->flatten_ids_ptr != nullptr) {
                                cudaFree(gsplat_ctx->flatten_ids_ptr);
                                gsplat_ctx->flatten_ids_ptr = nullptr;
                            }
                            arena.end_frame(gsplat_ctx->frame_id);
                        }
                    };

                    lfs::core::Tensor corrected_image = output.image;
                    if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                        corrected_image = bilateral_grid_->apply(output.image, cam->uid());
                    }
                    auto ppisp_input = corrected_image;

                    auto pred = ppisp_controller_pool_->predict(ppisp_cam_idx, corrected_image.unsqueeze(0), 1.0f);
                    corrected_image = ppisp_->apply_with_controller_params(corrected_image, pred, ppisp_cam_idx);

                    // Photometric loss
                    nvtxRangePush("compute_photometric_loss");
                    lfs::core::Tensor tile_loss;
                    lfs::core::Tensor tile_grad;

                    const bool use_mask = params_.optimization.mask_mode != lfs::core::param::MaskMode::None &&
                                          (cam->has_mask() || (params_.optimization.use_alpha_as_mask && cam->has_alpha()));
                    if (use_mask) {
                        lfs::core::Tensor mask;
                        if (pipelined_mask_.is_valid() && pipelined_mask_.numel() > 0) {
                            mask = pipelined_mask_;
                        } else {
                            mask = cam->load_and_get_mask(
                                params_.dataset.resize_factor,
                                params_.dataset.max_width,
                                params_.optimization.invert_masks,
                                params_.optimization.mask_threshold);
                        }

                        lfs::core::Tensor mask_tile = mask;
                        if (num_tiles > 1 && mask.ndim() == 2) {
                            auto tile_h = mask.slice(0, tile_y_offset, tile_y_offset + tile_height);
                            mask_tile = tile_h.slice(1, tile_x_offset, tile_x_offset + tile_width);
                        }

                        auto result = compute_photometric_loss_with_mask(
                            corrected_image, gt_tile, mask_tile, output.alpha, params_.optimization);
                        if (!result) {
                            cleanup_controller_tile_context();
                            nvtxRangePop();
                            nvtxRangePop();
                            nvtxRangePop();
                            return std::unexpected(result.error());
                        }
                        tile_loss = result->loss;
                        tile_grad = result->grad_image;
                    } else {
                        auto result = compute_photometric_loss_with_gradient(
                            corrected_image, gt_tile, params_.optimization);
                        if (!result) {
                            cleanup_controller_tile_context();
                            nvtxRangePop();
                            nvtxRangePop();
                            nvtxRangePop();
                            return std::unexpected(result.error());
                        }
                        tile_loss = result->first;
                        tile_grad = result->second;
                    }

                    loss_tensor_gpu = loss_tensor_gpu + tile_loss;
                    tiles_processed++;
                    nvtxRangePop(); // compute_photometric_loss

                    // ISP backward for controller params
                    auto ctrl_grad = ppisp_->backward_with_controller_params(ppisp_input, tile_grad, pred, ppisp_cam_idx);
                    ppisp_controller_pool_->backward(ppisp_cam_idx, ctrl_grad);

                    // End arena frame explicitly (normally done inside rasterize_backward which we skip)
                    cleanup_controller_tile_context();

                    nvtxRangePop(); // controller_phase
                } else {
                    // Normal phase: full forward + backward through all components
                    lfs::core::Tensor corrected_image = output.image;
                    if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                        nvtxRangePush("bilateral_grid_forward");
                        corrected_image = bilateral_grid_->apply(output.image, cam->uid());
                        nvtxRangePop();
                    }

                    if (ppisp_ && params_.optimization.use_ppisp) {
                        nvtxRangePush("ppisp_forward");
                        corrected_image = ppisp_->apply(corrected_image, cam->camera_id(), cam->uid());
                        nvtxRangePop();
                    }

                    // Final tonemapping: clamp to [0, 1] for loss computation.
                    // This is redundant when PPISP is active (CRF already clamps), but ensures
                    // valid output range for bilateral grids and raw rasterizer output.
                    corrected_image.clamp_(0.0f, 1.0f);

                    nvtxRangePush("compute_photometric_loss");
                    lfs::core::Tensor tile_loss;
                    lfs::core::Tensor tile_grad;
                    lfs::core::Tensor tile_grad_alpha;
                    lfs::core::Tensor tile_error_map;
                    lfs::core::Tensor mask_tile;

                    // 1) Compute photometric loss (populates ssim_map in workspace)
                    const bool use_mask = params_.optimization.mask_mode != lfs::core::param::MaskMode::None &&
                                          (cam->has_mask() || (params_.optimization.use_alpha_as_mask && cam->has_alpha()));
                    const bool used_masked_fused =
                        use_mask &&
                        (params_.optimization.mask_mode == lfs::core::param::MaskMode::Segment ||
                         params_.optimization.mask_mode == lfs::core::param::MaskMode::Ignore) &&
                        params_.optimization.lambda_dssim > 0.0f;
                    if (use_mask) {
                        lfs::core::Tensor mask;
                        if (pipelined_mask_.is_valid() && pipelined_mask_.numel() > 0) {
                            mask = pipelined_mask_;
                        } else {
                            mask = cam->load_and_get_mask(
                                params_.dataset.resize_factor,
                                params_.dataset.max_width,
                                params_.optimization.invert_masks,
                                params_.optimization.mask_threshold);
                        }

                        mask_tile = mask;
                        if (num_tiles > 1 && mask.ndim() == 2) {
                            auto tile_h = mask.slice(0, tile_y_offset, tile_y_offset + tile_height);
                            mask_tile = tile_h.slice(1, tile_x_offset, tile_x_offset + tile_width);
                        }

                        auto result = compute_photometric_loss_with_mask(
                            corrected_image, gt_tile, mask_tile, output.alpha, params_.optimization);
                        if (!result) {
                            nvtxRangePop();
                            nvtxRangePop();
                            return std::unexpected(result.error());
                        }
                        tile_loss = result->loss;
                        tile_grad = result->grad_image;
                        tile_grad_alpha = result->grad_alpha;
                    } else {
                        auto result = compute_photometric_loss_with_gradient(
                            corrected_image, gt_tile, params_.optimization);
                        if (!result) {
                            nvtxRangePop();
                            nvtxRangePop();
                            return std::unexpected(result.error());
                        }
                        tile_loss = result->first;
                        tile_grad = result->second;
                    }

                    // 2) Extract error map from workspace's ssim_map
                    if (use_pixel_error_densification) {
                        if (use_ssim_error && params_.optimization.lambda_dssim > 0.0f) {
                            lfs::core::Tensor ssim_map;
                            if (used_masked_fused) {
                                ssim_map = masked_fused_workspace_.ssim_map;
                            } else if (params_.optimization.lambda_dssim < 1.0f) {
                                ssim_map = photometric_loss_.fused_workspace().ssim_map;
                            } else {
                                ssim_map = photometric_loss_.ssim_workspace().ssim_map;
                            }
                            {
                                const size_t H = ssim_map.shape()[2];
                                const size_t W = ssim_map.shape()[3];
                                if (!densification_error_map_.is_valid() ||
                                    densification_error_map_.shape()[0] != H ||
                                    densification_error_map_.shape()[1] != W) {
                                    densification_error_map_ = core::Tensor::empty(
                                        {static_cast<size_t>(H), static_cast<size_t>(W)},
                                        core::Device::CUDA);
                                }
                                lfs::training::kernels::launch_ssim_to_error_map(ssim_map, densification_error_map_);
                                tile_error_map = densification_error_map_;
                            }
                        } else if (use_ssim_error) {
                            // lambda_dssim == 0 but error-priority densification still needs SSIM error
                            lfs::core::Tensor pred_chw = corrected_image;
                            lfs::core::Tensor gt_chw = gt_tile;
                            if (pred_chw.ndim() == 3 && pred_chw.shape()[2] == 3 &&
                                gt_chw.ndim() == 3 && gt_chw.shape()[2] == 3) {
                                pred_chw = pred_chw.permute({2, 0, 1}).contiguous();
                                gt_chw = gt_chw.permute({2, 0, 1}).contiguous();
                            }
                            lfs::training::kernels::ssim_error_map_forward(
                                pred_chw, gt_chw, densification_ssim_workspace_, densification_error_map_);
                            tile_error_map = densification_error_map_;
                        } else {
                            const lfs::core::Tensor abs_diff = (corrected_image - gt_tile).abs();
                            if (abs_diff.ndim() == 3 && abs_diff.shape()[0] == 3) {
                                tile_error_map = abs_diff.mean({0}, false);
                            } else if (abs_diff.ndim() == 3 && abs_diff.shape()[2] == 3) {
                                tile_error_map = abs_diff.mean({2}, false);
                            } else {
                                tile_error_map = abs_diff;
                            }
                            tile_error_map = tile_error_map.contiguous();
                        }

                        if (use_mask &&
                            (params_.optimization.mask_mode == lfs::core::param::MaskMode::Segment ||
                             params_.optimization.mask_mode == lfs::core::param::MaskMode::Ignore)) {
                            tile_error_map.mul_(mask_tile);
                        }
                    }

                    if (tile_error_map.is_valid() && params_.optimization.strategy == "lfs") {
                        const float map_mean = tile_error_map.mean().item();
                        if (map_mean > 1e-6f)
                            tile_error_map.div_(map_mean);
                    }

                    if (memory_breakdown_enabled_ && !memory_breakdown_logged_first_raster_ && fast_ctx) {
                        const auto snapshot = capture_vram_snapshot(true);
                        log_vram_snapshot("during_first_fastgs_tile", snapshot);

                        const size_t n_primitives = static_cast<size_t>(strategy_->get_model().size());
                        const size_t grad_mean2d_helper_bytes = n_primitives * 2 * sizeof(float);
                        const size_t grad_conic_helper_bytes = n_primitives * 3 * sizeof(float);

                        std::vector<std::pair<std::string, size_t>> fastgs_entries;
                        if (fast_ctx->forward_ctx.per_primitive_buffers_size > 0) {
                            fastgs_entries.emplace_back("fastgs.per_primitive_buffers",
                                                        fast_ctx->forward_ctx.per_primitive_buffers_size);
                        }
                        if (fast_ctx->forward_ctx.per_tile_buffers_size > 0) {
                            fastgs_entries.emplace_back("fastgs.per_tile_buffers",
                                                        fast_ctx->forward_ctx.per_tile_buffers_size);
                        }
                        if (fast_ctx->forward_ctx.per_instance_buffers_size > 0) {
                            fastgs_entries.emplace_back("fastgs.per_instance_buffers",
                                                        fast_ctx->forward_ctx.per_instance_buffers_size);
                        }
                        if (fast_ctx->forward_ctx.per_bucket_buffers_size > 0) {
                            fastgs_entries.emplace_back("fastgs.per_bucket_buffers",
                                                        fast_ctx->forward_ctx.per_bucket_buffers_size);
                        }
                        fastgs_entries.emplace_back("fastgs.grad_mean2d_helper", grad_mean2d_helper_bytes);
                        fastgs_entries.emplace_back("fastgs.grad_conic_helper", grad_conic_helper_bytes);
                        fastgs_entries.emplace_back("render.output_image", tensor_reserved_bytes(output.image));
                        fastgs_entries.emplace_back("render.output_alpha", tensor_reserved_bytes(output.alpha));
                        fastgs_entries.emplace_back("train.gt_tile", tensor_reserved_bytes(gt_tile));
                        if (bg_image.is_valid() && !bg_image.is_empty()) {
                            fastgs_entries.emplace_back("train.background_image", tensor_reserved_bytes(bg_image));
                        }
                        if (tile_error_map.is_valid()) {
                            fastgs_entries.emplace_back("train.tile_error_map", tensor_reserved_bytes(tile_error_map));
                        }
                        log_entry_bytes("first_fastgs_tile_live", fastgs_entries);

                        std::vector<std::pair<std::string, size_t>> trainer_entries;
                        if (const size_t bytes = photometric_workspace_bytes(photometric_loss_); bytes > 0) {
                            trainer_entries.emplace_back("trainer.photometric_workspaces", bytes);
                        }
                        if (const size_t bytes = masked_fused_workspace_bytes(masked_fused_workspace_); bytes > 0) {
                            trainer_entries.emplace_back("trainer.masked_fused_workspace", bytes);
                        }
                        if (const size_t bytes = ssim_map_workspace_bytes(densification_ssim_workspace_); bytes > 0) {
                            trainer_entries.emplace_back("trainer.densification_ssim_workspace", bytes);
                        }
                        add_tensor_entry(trainer_entries, "trainer.densification_error_map", densification_error_map_);
                        add_tensor_entry(trainer_entries, "trainer.loss_accumulator", loss_accumulator_);
                        add_tensor_entry(trainer_entries, "trainer.pipelined_mask", pipelined_mask_);
                        log_entry_bytes("trainer_pool_backed_live", trainer_entries);

                        if (auto loader = getActiveImageLoader()) {
                            const auto stats = loader->get_stats();
                            LOG_INFO("[MEM] first_fastgs_tile loader output_queue={}, pending_pairs={}, hot_queue={}, cold_queue={}, prefetch_queue={}",
                                     stats.output_queue_size,
                                     stats.pending_pairs_count,
                                     stats.hot_queue_size,
                                     stats.cold_queue_size,
                                     stats.prefetch_queue_size);
                        }

                        LOG_INFO("[MEM] fastgs counts visible_primitives={}, instances={}, buckets={}, image={}x{}, tile_mode={}, num_tiles={}",
                                 fast_ctx->forward_ctx.n_visible_primitives,
                                 fast_ctx->forward_ctx.n_instances,
                                 fast_ctx->forward_ctx.n_buckets,
                                 output.width,
                                 output.height,
                                 params_.optimization.tile_mode,
                                 num_tiles);
                        memory_breakdown_logged_first_raster_ = true;
                    }

                    loss_tensor_gpu = loss_tensor_gpu + tile_loss;
                    tiles_processed++;
                    nvtxRangePop();

                    lfs::core::Tensor raster_grad = tile_grad;
                    if (ppisp_ && params_.optimization.use_ppisp) {
                        nvtxRangePush("ppisp_backward");
                        lfs::core::Tensor ppisp_input = output.image;
                        if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                            ppisp_input = bilateral_grid_->apply(output.image, cam->uid());
                        }
                        raster_grad = ppisp_->backward(ppisp_input, raster_grad, cam->camera_id(), cam->uid());
                        if (ppisp_frozen) {
                            ppisp_->zero_grad();
                        }
                        nvtxRangePop();
                    }

                    if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                        nvtxRangePush("bilateral_grid_backward");
                        raster_grad = bilateral_grid_->backward(output.image, raster_grad, cam->uid());
                        nvtxRangePop();
                    }

                    nvtxRangePush("rasterize_backward");
                    if (gsplat_ctx) {
                        auto grad_alpha = tile_grad_alpha.is_valid()
                                              ? tile_grad_alpha
                                              : lfs::core::Tensor::zeros_like(output.alpha);
                        gsplat_rasterize_backward(*gsplat_ctx, raster_grad, grad_alpha,
                                                  strategy_->get_model(), strategy_->get_optimizer(),
                                                  use_pixel_error_densification ? tile_error_map : lfs::core::Tensor{});
                    } else {
                        fast_rasterize_backward(*fast_ctx, raster_grad, strategy_->get_model(),
                                                strategy_->get_optimizer(), tile_grad_alpha,
                                                use_pixel_error_densification ? tile_error_map : lfs::core::Tensor{},
                                                densification_type);
                    }
                    nvtxRangePop();
                }

                nvtxRangePop(); // End tile
            }

            if (tiles_processed > 1)
                loss_tensor_gpu = loss_tensor_gpu / static_cast<float>(tiles_processed);

            if (tiles_processed == 0) {
                LOG_DEBUG("Skipping iteration {} - no visible primitives", iter);
                return iter < params_.optimization.iterations && !stop_requested_.load() && !stop_token.stop_requested()
                           ? StepResult::Continue
                           : StepResult::Stop;
            }

            if (in_controller_phase) {
                // Controller phase: only update controller weights
                nvtxRangePush("controller_optimizer_step");
                ppisp_controller_pool_->optimizer_step(ppisp_cam_idx);
                ppisp_controller_pool_->zero_grad();
                ppisp_controller_pool_->scheduler_step(ppisp_cam_idx);
                nvtxRangePop();
            } else {
                // Normal phase: regularization losses + optimizer steps for all components

                if (params_.optimization.scale_reg > 0.0f) {
                    nvtxRangePush("compute_scale_reg_loss");
                    auto scale_loss_result = compute_scale_reg_loss(strategy_->get_model(), strategy_->get_optimizer(), params_.optimization);
                    if (!scale_loss_result) {
                        return std::unexpected(scale_loss_result.error());
                    }
                    loss_tensor_gpu = loss_tensor_gpu + *scale_loss_result;
                    nvtxRangePop();
                }

                if (params_.optimization.opacity_reg > 0.0f) {
                    nvtxRangePush("compute_opacity_reg_loss");
                    auto opacity_loss_result = compute_opacity_reg_loss(strategy_->get_model(), strategy_->get_optimizer(), params_.optimization);
                    if (!opacity_loss_result) {
                        return std::unexpected(opacity_loss_result.error());
                    }
                    loss_tensor_gpu = loss_tensor_gpu + *opacity_loss_result;
                    nvtxRangePop();
                }

                if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                    nvtxRangePush("bilateral_grid_tv_and_step");
                    const float tv_weight = params_.optimization.tv_loss_weight;

                    loss_tensor_gpu = loss_tensor_gpu + bilateral_grid_->tv_loss_gpu() * tv_weight;
                    bilateral_grid_->tv_backward(tv_weight);
                    bilateral_grid_->optimizer_step();
                    bilateral_grid_->zero_grad();
                    bilateral_grid_->scheduler_step();

                    nvtxRangePop();
                }

                if (ppisp_ && params_.optimization.use_ppisp && !ppisp_frozen) {
                    nvtxRangePush("ppisp_reg_and_step");

                    loss_tensor_gpu = loss_tensor_gpu + ppisp_->reg_loss_gpu();
                    ppisp_->reg_backward();
                    ppisp_->optimizer_step();
                    ppisp_->zero_grad();
                    ppisp_->scheduler_step();

                    nvtxRangePop();
                }
            }

            // Sparsity loss - ALL ON GPU, no CPU sync here
            lfs::core::Tensor sparsity_loss_gpu;
            if (sparsity_optimizer_ && sparsity_optimizer_->should_apply_loss(iter)) {
                nvtxRangePush("sparsity_loss");
                auto sparsity_result = compute_sparsity_loss_forward(iter, strategy_->get_model());
                if (!sparsity_result) {
                    nvtxRangePop();
                    return std::unexpected(sparsity_result.error());
                }
                auto& [loss_tensor, ctx] = *sparsity_result;
                sparsity_loss_gpu = std::move(loss_tensor);

                if (ctx.n > 0) {
                    if (auto result = sparsity_optimizer_->compute_loss_backward(
                            ctx, 1.0f, strategy_->get_optimizer().get_grad(ParamType::Opacity));
                        !result) {
                        nvtxRangePop();
                        return std::unexpected(result.error());
                    }
                }
                nvtxRangePop();
            }

            // Sparsification phase logging (once per phase transition)
            if (params_.optimization.enable_sparsity) {
                const int base_iterations = params_.optimization.iterations - params_.optimization.sparsify_steps;
                if (iter == base_iterations + 1) {
                    LOG_INFO("Entering sparsification: {} Gaussians, target prune={}%",
                             strategy_->get_model().size(), params_.optimization.prune_ratio * 100);
                }
            }

            // Sync loss to CPU only at intervals - single sync point
            constexpr int LOSS_SYNC_INTERVAL = 10;
            float loss_value = 0.0f;
            if (iter % LOSS_SYNC_INTERVAL == 0 || iter == 1) {
                // Accumulate on GPU then sync once
                auto total_loss = sparsity_loss_gpu.numel() > 0
                                      ? (loss_tensor_gpu + sparsity_loss_gpu)
                                      : loss_tensor_gpu;
                loss_value = total_loss.item<float>();

                if (std::isnan(loss_value) || std::isinf(loss_value)) {
                    return std::unexpected(std::format("NaN/Inf loss at iteration {}", iter));
                }

                current_loss_ = loss_value;
                if (progress_) {
                    progress_->update(iter, loss_value,
                                      static_cast<int>(strategy_->get_model().size()),
                                      strategy_->is_refining(iter));
                }
                lfs::core::events::state::TrainingProgress{
                    .iteration = iter,
                    .loss = loss_value,
                    .num_gaussians = static_cast<int>(strategy_->get_model().size()),
                    .is_refining = strategy_->is_refining(iter)}
                    .emit();
            }

            const bool in_sparsification = params_.optimization.enable_sparsity &&
                                           iter > (params_.optimization.iterations - params_.optimization.sparsify_steps);

            if (!in_sparsification) {
                strategy_->pre_step(iter, r_output);
            }

            {
                DeferredEvents deferred;
                {
                    std::unique_lock<std::shared_mutex> lock(render_mutex_);
                    auto& model = strategy_->get_model();
                    const size_t model_size_before = static_cast<size_t>(model.size());

                    // Python hook: pre-optimizer-step (post-backward, pre-step)
                    {
                        lfs::training::HookContext ctx{
                            .iteration = iter,
                            .loss = current_loss_.load(),
                            .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
                            .is_refining = strategy_ ? strategy_->is_refining(iter) : false,
                            .trainer = this};
                        lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::OptimizerStep);
                        lfs::training::CommandCenter::instance().update_snapshot(
                            ctx, params_.optimization.iterations, is_paused_.load(), is_running_.load(), stop_requested_.load(),
                            lfs::training::TrainingPhase::OptimizerStep);
                        lfs::training::ControlBoundary::instance().notify(lfs::training::ControlHook::PreOptimizerStep, ctx);
                    }

                    if (!in_sparsification) {
                        strategy_->post_backward(iter, r_output);
                    }

                    // Skip strategy step if we're in controller distillation phase and freeze is enabled
                    const int ppisp_activation_step = params_.optimization.resolved_ppisp_controller_activation_step();
                    const bool freeze_gaussians = ppisp_controller_pool_ &&
                                                  params_.optimization.ppisp_use_controller &&
                                                  params_.optimization.ppisp_freeze_gaussians_on_distill &&
                                                  iter >= ppisp_activation_step;
                    if (!freeze_gaussians) {
                        strategy_->step(iter);
                    }

                    if (auto result = handle_sparsity_update(iter, model); !result) {
                        LOG_ERROR("Sparsity update: {}", result.error());
                    }
                    if (auto result = apply_sparsity_pruning(iter, model); !result) {
                        LOG_ERROR("Sparsity pruning: {}", result.error());
                    }

                    if (static_cast<size_t>(model.size()) != model_size_before) {
                        syncTrainingSceneTopology(scene_, model);
                    }
                }

                // Clean evaluation - let the evaluator handle everything
                if (evaluator_->is_enabled() && evaluator_->should_evaluate(iter)) {
                    evaluator_->print_evaluation_header(iter);
                    auto metrics = evaluator_->evaluate(iter,
                                                        strategy_->get_model(),
                                                        val_dataset_,
                                                        background_);
                    LOG_INFO("{}", metrics.to_string());
                }

                // Save checkpoint (not PLY) at specified steps
                for (size_t save_step : params_.optimization.save_steps) {
                    if (iter == static_cast<int>(save_step) && iter != params_.optimization.iterations) {
                        auto result = save_checkpoint(iter);
                        if (!result) {
                            LOG_WARN("Failed to save checkpoint at iteration {}: {}", iter, result.error());
                        }
                    }
                }

                if (!params_.dataset.timelapse_images.empty() && iter % params_.dataset.timelapse_every == 0) {
                    for (const auto& img_name : params_.dataset.timelapse_images) {
                        auto train_cam = train_dataset_->get_camera_by_filename(img_name);
                        auto val_cam = val_dataset_ ? val_dataset_->get_camera_by_filename(img_name) : std::nullopt;
                        if (train_cam.has_value() || val_cam.has_value()) {
                            lfs::core::Camera* cam_to_use = train_cam.has_value() ? train_cam.value() : val_cam.value();

                            // Image size isn't correct until the image has been loaded once
                            // If we use the camera before it's loaded, it will render images at the non-scaled size
                            if ((cam_to_use->camera_height() == cam_to_use->image_height() && params_.dataset.resize_factor != 1) ||
                                cam_to_use->image_height() > params_.dataset.max_width ||
                                cam_to_use->image_width() > params_.dataset.max_width) {
                                cam_to_use->load_image_size(params_.dataset.resize_factor, params_.dataset.max_width);
                            }

                            RenderOutput rendered_timelapse_output;
                            if (params_.optimization.gut) {
                                rendered_timelapse_output = gsplat_rasterize(*cam_to_use, strategy_->get_model(), background_,
                                                                             1.0f, false, GsplatRenderMode::RGB, true);
                            } else {
                                rendered_timelapse_output = fast_rasterize(*cam_to_use, strategy_->get_model(), background_);
                            }

                            // Get folder name to save in by stripping file extension
                            std::string folder_name = lfs::io::strip_extension(img_name);

                            auto output_path = params_.dataset.output_path / "timelapse" / folder_name;
                            std::filesystem::create_directories(output_path);

                            lfs::core::image_io::save_image_async(output_path / std::format("{:06d}.jpg", iter),
                                                                  rendered_timelapse_output.image);
                        } else {
                            LOG_WARN("Timelapse image '{}' not found in dataset.", img_name);
                        }
                    }
                }
            }

            // Python hook: post-step (after optimizer and side-effects)
            {
                lfs::training::HookContext ctx{
                    .iteration = iter,
                    .loss = current_loss_.load(),
                    .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
                    .is_refining = strategy_ ? strategy_->is_refining(iter) : false,
                    .trainer = this};
                lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);
                lfs::training::CommandCenter::instance().update_snapshot(
                    ctx, params_.optimization.iterations, is_paused_.load(), is_running_.load(), stop_requested_.load(),
                    lfs::training::TrainingPhase::SafeControl);
                lfs::training::ControlBoundary::instance().notify(lfs::training::ControlHook::PostStep, ctx);
            }

            if (memory_breakdown_enabled_ && !memory_breakdown_logged_first_step_) {
                const auto snapshot = capture_vram_snapshot(true);
                log_vram_snapshot("after_first_train_step", snapshot);
                LOG_INFO("[MEM] after_first_train_step model logical_size={} capacity={} direct_model_plus_optimizer={:.2f} MiB",
                         strategy_->get_model().size(),
                         strategy_->get_model().means().capacity(),
                         bytes_to_mib(splat_tensor_bytes(strategy_->get_model()) +
                                      optimizer_state_bytes(strategy_->get_optimizer())));
                memory_breakdown_logged_first_step_ = true;
            }

            if (memory_breakdown_enabled_ && iter > 1 && (iter % 1000) == 0) {
                const auto label = std::format("after_step_{}", iter);
                const auto snapshot = capture_vram_snapshot(true);
                log_vram_snapshot(label, snapshot);
                LOG_INFO("[MEM] {} model logical_size={} capacity={} direct_model_plus_optimizer={:.2f} MiB",
                         label,
                         strategy_->get_model().size(),
                         strategy_->get_model().means().capacity(),
                         bytes_to_mib(splat_tensor_bytes(strategy_->get_model()) +
                                      optimizer_state_bytes(strategy_->get_optimizer())));
            }

            // Return Continue if we should continue training
            if (iter < params_.optimization.iterations && !stop_requested_.load() && !stop_token.stop_requested()) {
                return StepResult::Continue;
            } else {
                return StepResult::Stop;
            }
        } catch (const std::exception& e) {
            return std::unexpected(std::format("Training step failed: {}", e.what()));
        }
    }

    std::expected<void, std::string> Trainer::train(std::stop_token stop_token) {
        // Check if initialized
        if (!initialized_.load()) {
            return std::unexpected("Trainer not initialized. Call initialize() before train()");
        }

        is_running_ = false;
        training_complete_ = false;
        ready_to_start_ = false; // Reset the flag
        lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);

        ready_to_start_ = true; // Skip GUI wait for now

        is_running_ = true; // Now we can start
        LOG_INFO("Starting training loop");
        auto& cache_loader = lfs::io::CacheLoader::getInstance();
        cache_loader.reset_cache();
        cache_loader.update_cache_params(params_.dataset.loading_params.use_cpu_memory,
                                         params_.dataset.loading_params.use_fs_cache,
                                         train_dataset_size_,
                                         params_.dataset.loading_params.min_cpu_free_GB,
                                         params_.dataset.loading_params.min_cpu_free_memory_ratio,
                                         params_.dataset.loading_params.print_cache_status,
                                         params_.dataset.loading_params.print_status_freq_num);

        // Notify Python control layer that training is starting
        {
            lfs::training::HookContext ctx{
                .iteration = 0,
                .loss = current_loss_.load(),
                .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
                .is_refining = strategy_ ? strategy_->is_refining(0) : false,
                .trainer = this};
            lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);
            lfs::training::CommandCenter::instance().update_snapshot(
                ctx, params_.optimization.iterations, is_paused_.load(), is_running_.load(), stop_requested_.load(),
                lfs::training::TrainingPhase::SafeControl);
            lfs::training::ControlBoundary::instance().notify(lfs::training::ControlHook::TrainingStart, ctx);
        }

        try {
            // Start from current_iteration_ (allows resume from checkpoint)
            int iter = current_iteration_.load() > 0 ? current_iteration_.load() + 1 : 1;
            const RenderMode render_mode = RenderMode::RGB;

            if (progress_) {
                progress_->update(iter, current_loss_.load(),
                                  static_cast<int>(strategy_->get_model().size()),
                                  strategy_->is_refining(iter));
            }

            // Conservative prefetch to avoid VRAM exhaustion
            lfs::io::PipelinedLoaderConfig pipelined_config;
            pipelined_config.jpeg_batch_size = 8;
            pipelined_config.prefetch_count = 8;
            pipelined_config.output_queue_size = 4;
            pipelined_config.io_threads = 2;

            // Non-JPEG images (PNG, WebP) need CPU decoding - use more threads until cache warms
            constexpr float NON_JPEG_THRESHOLD = 0.1f;
            constexpr size_t MIN_COLD_THREADS = 4;
            constexpr size_t COLD_PREFETCH_COUNT = 16;
            const float non_jpeg_ratio = train_dataset_->get_non_jpeg_ratio();
            if (non_jpeg_ratio > NON_JPEG_THRESHOLD) {
                const size_t cold_threads = std::max(MIN_COLD_THREADS,
                                                     static_cast<size_t>(std::thread::hardware_concurrency() / 2));
                pipelined_config.cold_process_threads = cold_threads;
                pipelined_config.prefetch_count = COLD_PREFETCH_COUNT;
                LOG_INFO("{:.0f}% non-JPEG images, using {} cold threads", non_jpeg_ratio * 100.0f, cold_threads);
            }

            pipelined_config = tunePipelinedLoaderConfig(pipelined_config, train_dataset_);

            const bool alpha_available = scene_ && scene_->imagesHaveAlpha();
            PipelinedMaskConfig mask_pipeline_config;
            if (params_.optimization.mask_mode != lfs::core::param::MaskMode::None) {
                mask_pipeline_config.invert_masks = params_.optimization.invert_masks;
                mask_pipeline_config.mask_threshold = params_.optimization.mask_threshold;
                if (params_.optimization.use_alpha_as_mask && alpha_available) {
                    mask_pipeline_config.use_alpha_as_mask = true;
                    mask_pipeline_config.load_masks = true;
                    LOG_INFO("Alpha-as-mask enabled (invert={}, threshold={})",
                             mask_pipeline_config.invert_masks, mask_pipeline_config.mask_threshold);
                } else {
                    mask_pipeline_config.load_masks = true;
                    LOG_INFO("Mask file loading enabled (invert={}, threshold={})",
                             mask_pipeline_config.invert_masks, mask_pipeline_config.mask_threshold);
                }
            }

            auto train_dataloader = create_infinite_pipelined_dataloader(
                train_dataset_, pipelined_config, mask_pipeline_config);
            auto active_image_loader_guard = makeScopeGuard([this]() {
                clearActiveImageLoader();
            });
            updateGTLoadConfigSnapshot();
            setActiveImageLoader(train_dataloader->get_loader_shared());
            strategy_->set_image_loader(train_dataloader->get_loader());

            if (memory_breakdown_enabled_ && !memory_breakdown_logged_train_setup_) {
                const auto snapshot = capture_vram_snapshot(true);
                log_vram_snapshot("after_dataloader_setup", snapshot);
                const auto stats = train_dataloader->get_stats();
                LOG_INFO("[MEM] dataloader setup in_flight={}, output_queue={}, pending_pairs={}, hot_queue={}, cold_queue={}, prefetch_queue={}",
                         train_dataloader->get_loader_shared()->in_flight_count(),
                         stats.output_queue_size,
                         stats.pending_pairs_count,
                         stats.hot_queue_size,
                         stats.cold_queue_size,
                         stats.prefetch_queue_size);
                memory_breakdown_logged_train_setup_ = true;
            }

            LOG_DEBUG("Starting training iterations");
            while (iter <= params_.optimization.iterations) {
                lfs::core::Tensor::set_memory_pool_iteration(iter);

                if (stop_token.stop_requested() || stop_requested_.load())
                    break;
                if (callback_busy_.load(std::memory_order_acquire)) {
                    const cudaError_t callback_status = cudaStreamQuery(callback_stream_);
                    if (callback_status == cudaSuccess) {
                        callback_busy_.store(false, std::memory_order_release);
                    } else if (callback_status != cudaErrorNotReady) {
                        LOG_WARN("Callback stream query failed: {}", cudaGetErrorString(callback_status));
                        callback_busy_.store(false, std::memory_order_release);
                    }
                }

                lfs::core::Camera* cam = nullptr;
                lfs::core::Tensor gt_image;
                auto example_opt = train_dataloader->next();
                if (!example_opt) {
                    LOG_ERROR("DataLoader returned nullopt unexpectedly");
                    break;
                }
                auto& example = *example_opt;
                cam = example.data.camera;
                gt_image = std::move(example.data.image);

                // Store pipelined mask for use in train_step
                pipelined_mask_ = example.mask.has_value() ? std::move(*example.mask) : lfs::core::Tensor();

                if (memory_breakdown_enabled_ && !memory_breakdown_logged_first_batch_) {
                    const auto snapshot = capture_vram_snapshot(true);
                    log_vram_snapshot("after_first_batch_fetch", snapshot);
                    const auto stats = train_dataloader->get_stats();
                    const size_t channels = gt_image.ndim() > 2 ? gt_image.shape()[0] : 1;
                    const size_t height = gt_image.ndim() > 2 ? gt_image.shape()[1] : gt_image.shape()[0];
                    const size_t width = gt_image.ndim() > 2 ? gt_image.shape()[2] : gt_image.shape()[1];
                    LOG_INFO("[MEM] first_batch gt_image={}x{}x{} = {:.2f} MiB, mask = {:.2f} MiB, output_queue={}, pending_pairs={}, in_flight={}",
                             channels,
                             height,
                             width,
                             bytes_to_mib(tensor_reserved_bytes(gt_image)),
                             bytes_to_mib(tensor_reserved_bytes(pipelined_mask_)),
                             stats.output_queue_size,
                             stats.pending_pairs_count,
                             train_dataloader->get_loader_shared()->in_flight_count());
                    memory_breakdown_logged_first_batch_ = true;
                }

                auto step_result = train_step(iter, cam, gt_image, render_mode, stop_token);
                if (!step_result) {
                    // Check if this is an OOM_RETRY signal
                    if (step_result.error() == "OOM_RETRY") {
                        cudaDeviceSynchronize();
                        cudaGetLastError();

                        lfs::core::GlobalArenaManager::instance().get_arena().full_reset();
                        lfs::core::Tensor::trim_memory_pool();

                        cudaDeviceSynchronize();
                        cudaGetLastError();

                        LOG_INFO("OOM recovery: retrying iteration {}", iter);
                        step_result = train_step(iter, cam, gt_image, render_mode, stop_token);
                        if (!step_result) {
                            return std::unexpected(step_result.error());
                        }
                    } else {
                        return std::unexpected(step_result.error());
                    }
                }

                // Transition to safe control phase and execute deferred Python callbacks
                lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);
                lfs::training::ControlBoundary::instance().drain_callbacks();

                if (*step_result == StepResult::Stop) {
                    break;
                }

                // Launch callback for async progress update (except first iteration)
                if (iter > 1 && callback_ && !callback_busy_.load(std::memory_order_acquire)) {
                    callback_busy_.store(true, std::memory_order_release);
                    auto err = cudaLaunchHostFunc(
                        callback_stream_,
                        [](void* self) {
                            auto* trainer = static_cast<Trainer*>(self);
                            if (trainer->callback_) {
                                trainer->callback_();
                            }
                            trainer->callback_busy_.store(false, std::memory_order_release);
                        },
                        this);
                    if (err != cudaSuccess) {
                        LOG_WARN("Failed to launch callback: {}", cudaGetErrorString(err));
                        callback_busy_.store(false, std::memory_order_release);
                    }
                }

                ++iter;
            }

            clearActiveImageLoader();
            active_image_loader_guard.release();

            // Ensure callback is finished before final save
            if (callback_busy_.load()) {
                cudaStreamSynchronize(callback_stream_);
            }

            // Final save if not already saved by stop request
            if (!stop_requested_.load() && !stop_token.stop_requested()) {
                auto final_path = params_.dataset.output_path;
                save_ply(final_path, params_.optimization.iterations, /*join=*/true);
            }

            if (progress_) {
                progress_->complete();
            }
            evaluator_->save_report();
            if (progress_) {
                progress_->print_final_summary(static_cast<int>(strategy_->get_model().size()));
            }

            is_running_ = false;
            training_complete_ = true;

            cache_loader.clear_cpu_cache();
            lfs::core::image_io::wait_for_pending_saves();

            // Notify training end
            {
                lfs::training::HookContext ctx{
                    .iteration = current_iteration_.load(),
                    .loss = current_loss_.load(),
                    .num_gaussians = strategy_ ? strategy_->get_model().size() : 0,
                    .is_refining = strategy_ ? strategy_->is_refining(current_iteration_.load()) : false,
                    .trainer = this};
                lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::SafeControl);
                lfs::training::CommandCenter::instance().update_snapshot(
                    ctx, params_.optimization.iterations, is_paused_.load(), is_running_.load(), stop_requested_.load(),
                    lfs::training::TrainingPhase::SafeControl);
                lfs::training::ControlBoundary::instance().notify(lfs::training::ControlHook::TrainingEnd, ctx);
            }

            lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::Idle);

            LOG_INFO("Training completed successfully");
            return {};
        } catch (const std::exception& e) {
            is_running_ = false;
            cache_loader.clear_cpu_cache();
            lfs::core::image_io::wait_for_pending_saves();
            lfs::training::CommandCenter::instance().set_phase(lfs::training::TrainingPhase::Idle);

            return std::unexpected(std::format("Training failed: {}", e.what()));
        }
    }

    void Trainer::save_ply(const std::filesystem::path& save_path, const int iter_num, const bool join_threads) {
        const lfs::io::PlySaveOptions ply_options{
            .output_path = save_path / ("splat_" + std::to_string(iter_num) + ".ply"),
            .binary = true,
            .async = !join_threads};

        const auto ply_result = lfs::io::save_ply(strategy_->get_model(), ply_options);
        if (!ply_result) {
            if (ply_result.error().code == lfs::io::ErrorCode::INSUFFICIENT_DISK_SPACE) {
                lfs::core::events::state::DiskSpaceSaveFailed{
                    .iteration = iter_num,
                    .path = ply_options.output_path,
                    .error = ply_result.error().message,
                    .required_bytes = ply_result.error().required_bytes,
                    .available_bytes = ply_result.error().available_bytes,
                    .is_disk_space_error = true,
                    .is_checkpoint = false}
                    .emit();
            }
            LOG_WARN("Failed to save PLY: {}", ply_result.error().message);
            return; // Don't save checkpoint if PLY failed
        }

        PPISPControllerPool* controller_to_save = controller_pool_for_save(iter_num);

        // Save checkpoint alongside PLY for training resumption
        auto ckpt_result = lfs::training::save_checkpoint(save_path, iter_num, *strategy_, params_,
                                                          bilateral_grid_.get(), ppisp_.get(), controller_to_save);
        if (!ckpt_result) {
            LOG_WARN("Failed to save checkpoint: {}", ckpt_result.error());
        }

        if (ppisp_) {
            const auto ppisp_path = get_ppisp_companion_path(ply_options.output_path);
            std::optional<PPISPFileMetadata> metadata;
            if (auto metadata_result = build_ppisp_sidecar_metadata(); metadata_result) {
                metadata = std::move(*metadata_result);
            } else {
                LOG_WARN("Failed to build PPISP sidecar metadata for '{}': {}. Saving sidecar without metadata.",
                         lfs::core::path_to_utf8(ppisp_path), metadata_result.error());
            }
            const auto ppisp_result = save_ppisp_file(ppisp_path, *ppisp_, controller_to_save,
                                                      metadata ? &*metadata : nullptr);
            if (!ppisp_result) {
                LOG_WARN("Failed to save PPISP file: {}", ppisp_result.error());
            }
        }

        LOG_DEBUG("PLY save initiated: {} (sync={})", lfs::core::path_to_utf8(save_path), join_threads);
    }

    std::expected<void, std::string> Trainer::save_checkpoint(int iteration) {
        if (!strategy_) {
            return std::unexpected("Cannot save checkpoint: no strategy initialized");
        }

        PPISPControllerPool* controller_to_save = controller_pool_for_save(iteration);

        return lfs::training::save_checkpoint(params_.dataset.output_path, iteration, *strategy_, params_,
                                              bilateral_grid_.get(), ppisp_.get(), controller_to_save);
    }

    std::expected<void, std::string> Trainer::save_checkpoint_to(const std::filesystem::path& output_path,
                                                                 int iteration) {
        if (!strategy_) {
            return std::unexpected("Cannot save checkpoint: no strategy initialized");
        }

        PPISPControllerPool* controller_to_save = controller_pool_for_save(iteration);

        return lfs::training::save_checkpoint(output_path, iteration, *strategy_, params_,
                                              bilateral_grid_.get(), ppisp_.get(), controller_to_save);
    }

    lfs::core::Tensor Trainer::applyPPISPForViewport(const lfs::core::Tensor& rgb, const int camera_uid,
                                                     const PPISPViewportOverrides& overrides,
                                                     const bool use_controller) const {
        if (!ppisp_ || !params_.optimization.use_ppisp || rgb.shape().rank() != 3) {
            return rgb;
        }

        const bool is_chw = (rgb.shape()[0] == 3);
        const auto rgb_chw = is_chw ? rgb : rgb.permute({2, 0, 1}).contiguous();
        const bool is_training_camera = ppisp_->is_known_frame(camera_uid);
        const bool has_controller = ppisp_controller_pool_ && params_.optimization.ppisp_use_controller;

        lfs::core::Tensor result;

        if (use_controller && has_controller) {
            constexpr int CONTROLLER_IDX = 0;
            const auto controller_params = ppisp_controller_pool_->predict(CONTROLLER_IDX, rgb_chw.unsqueeze(0), 1.0f);
            result = overrides.isIdentity()
                         ? ppisp_->apply_with_controller_params(rgb_chw, controller_params, CONTROLLER_IDX)
                         : ppisp_->apply_with_controller_params_and_overrides(rgb_chw, controller_params, CONTROLLER_IDX,
                                                                              toRenderOverrides(overrides));
        } else if (is_training_camera) {
            const int camera_id = ppisp_->camera_for_frame(camera_uid);
            result = overrides.isIdentity() ? ppisp_->apply(rgb_chw, camera_id, camera_uid)
                                            : ppisp_->apply_with_overrides(rgb_chw, camera_id, camera_uid,
                                                                           toRenderOverrides(overrides));
        } else {
            const int fallback_camera = ppisp_->any_camera_id();
            const int fallback_frame = ppisp_->any_frame_uid();
            result = overrides.isIdentity() ? ppisp_->apply(rgb_chw, fallback_camera, fallback_frame)
                                            : ppisp_->apply_with_overrides(rgb_chw, fallback_camera, fallback_frame,
                                                                           toRenderOverrides(overrides));
        }

        return is_chw ? result : result.permute({1, 2, 0}).contiguous();
    }

    PPISPControllerPool* Trainer::controller_pool_for_save(const int iteration) const {
        if (!ppisp_controller_pool_) {
            return nullptr;
        }
        if (is_ppisp_frozen()) {
            return ppisp_controller_pool_.get();
        }
        return iteration >= params_.optimization.resolved_ppisp_controller_activation_step()
                   ? ppisp_controller_pool_.get()
                   : nullptr;
    }

    void Trainer::save_final_ply_and_checkpoint(const int iteration) {
        save_ply(params_.dataset.output_path, iteration, /*join=*/true);
    }

    std::expected<int, std::string> Trainer::load_checkpoint(const std::filesystem::path& checkpoint_path) {
        if (!strategy_) {
            return std::unexpected("Cannot load checkpoint: no strategy initialized");
        }

        // Create bilateral grid before loading if needed (checkpoint may contain grid state)
        if (params_.optimization.use_bilateral_grid && !bilateral_grid_) {
            if (auto init_result = initialize_bilateral_grid(); !init_result) {
                LOG_WARN("Failed to init bilateral grid for resume: {}", init_result.error());
            }
        }

        // Create PPISP before loading if needed
        if (params_.optimization.use_ppisp && !ppisp_) {
            if (auto init_result = initialize_ppisp(); !init_result) {
                LOG_WARN("Failed to init PPISP for resume: {}", init_result.error());
            }
        }

        // Create PPISP controller pool before loading if needed
        if (params_.optimization.ppisp_use_controller && !ppisp_controller_pool_) {
            bool should_initialize_controller = true;
            if (is_ppisp_frozen()) {
                const auto checkpoint_header = lfs::core::load_checkpoint_header(checkpoint_path);
                if (!checkpoint_header) {
                    LOG_WARN("Failed to inspect checkpoint header for PPISP controller state: {}",
                             checkpoint_header.error());
                    should_initialize_controller = false;
                } else {
                    should_initialize_controller =
                        lfs::core::has_flag(checkpoint_header->flags, lfs::core::CheckpointFlags::HAS_PPISP_CONTROLLER);
                    if (!should_initialize_controller) {
                        LOG_INFO("Checkpoint has no PPISP controller pool; frozen controller state remains disabled");
                    }
                }
            }
            if (should_initialize_controller) {
                if (auto init_result = initialize_ppisp_controller(); !init_result) {
                    LOG_WARN("Failed to init PPISP controller pool for resume: {}", init_result.error());
                }
            }
        }

        auto result = lfs::training::load_checkpoint(
            checkpoint_path, *strategy_, params_, bilateral_grid_.get(), ppisp_.get(),
            ppisp_controller_pool_.get());
        if (!result) {
            return result;
        }
        current_iteration_ = *result;

        LOG_INFO("Restored training state from checkpoint at iteration {}", *result);
        return result;
    }

} // namespace lfs::training
