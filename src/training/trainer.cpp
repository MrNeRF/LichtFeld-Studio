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
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/gpu_slab_allocator.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "core/tensor/internal/size_bucketed_pool.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "io/cache_image_loader.hpp"
#include "io/cuda/image_format_kernels.cuh"
#include "io/exporter.hpp"
#include "io/filesystem_utils.hpp"
#include "kernels/image_kernels.hpp"
#include "lfs/kernels/ssim.cuh"
#include "losses/losses.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "python/runner.hpp"
#include "rasterization/fast_rasterizer.hpp"
#include "rasterization/gsplat_rasterizer.hpp"
#include "strategies/mcmc.hpp"
#include "strategies/strategy_factory.hpp"
#include "strategies/strategy_utils.hpp"
#include "training/kernels/camera_loss_heatmap.cuh"
#include "training/kernels/depth_loss.hpp"
#include "training/kernels/grad_alpha.hpp"
#include "training/kernels/mrnf_kernels.hpp"
#include "training/training_setup.hpp"

#include <filesystem>
#include <fstream>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <expected>
#include <format>
#include <limits>
#include <memory>
#include <numeric>
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCudaRt.h>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace lfs::training {

    namespace {
        constexpr double BYTES_PER_MIB = 1024.0 * 1024.0;
        constexpr double BYTES_PER_GIB = 1024.0 * 1024.0 * 1024.0;
        constexpr float CAMERA_LOSS_EMA_ALPHA = 0.2f;
        constexpr int CAMERA_LOSS_PUBLISH_INTERVAL = 16;

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

        [[nodiscard]] std::unique_ptr<lfs::core::SplatData> make_ply_export_model(
            const lfs::core::SplatData& model,
            const bool exclude_frozen_ranges) {
            if (!exclude_frozen_ranges || !model.has_frozen_ranges()) {
                return nullptr;
            }

            const size_t count = model.size();
            if (count == 0) {
                return nullptr;
            }

            std::vector<bool> keep(count, true);
            size_t excluded_count = 0;
            for (const auto& range : model.frozen_ranges()) {
                if (range.count == 0 || range.start >= count) {
                    continue;
                }
                const size_t remaining = count - range.start;
                const size_t end = range.start + std::min(range.count, remaining);
                for (size_t idx = range.start; idx < end; ++idx) {
                    if (keep[idx]) {
                        keep[idx] = false;
                        ++excluded_count;
                    }
                }
            }

            if (excluded_count == 0) {
                return nullptr;
            }
            if (excluded_count == count) {
                LOG_WARN("Skipping frozen-add-splat export exclusion because it would remove all {} Gaussians",
                         count);
                return nullptr;
            }

            auto keep_mask = lfs::core::Tensor::from_vector(
                keep,
                lfs::core::TensorShape({count}),
                model.means_raw().device());
            auto filtered = std::make_unique<lfs::core::SplatData>(
                lfs::core::extract_by_mask(model, keep_mask));
            if (!filtered->means_raw().is_valid() || filtered->size() == 0) {
                LOG_WARN("Failed to build frozen-add-splat filtered export model; exporting full model");
                return nullptr;
            }

            LOG_INFO("Excluding {} frozen added Gaussian{} from PLY export ({} -> {})",
                     excluded_count,
                     excluded_count == 1 ? "" : "s",
                     count,
                     filtered->size());
            return filtered;
        }

        [[nodiscard]] int env_int_or_default(const char* name, const int fallback) {
            const char* raw = std::getenv(name);
            if (!raw || raw[0] == '\0') {
                return fallback;
            }

            char* end = nullptr;
            const long parsed = std::strtol(raw, &end, 10);
            if (end == raw || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
                return fallback;
            }
            return static_cast<int>(parsed);
        }

        [[nodiscard]] kernels::DepthLossMode depth_loss_mode_from_name(const std::string_view mode) {
            if (mode == "adaptive-warped-l1") {
                return kernels::DepthLossMode::AdaptiveWarpedL1;
            }
            if (mode == "pearson") {
                return kernels::DepthLossMode::PearsonAbs;
            }
            LOG_WARN("Unknown depth loss mode '{}'; using adaptive-warped-l1", mode);
            return kernels::DepthLossMode::AdaptiveWarpedL1;
        }

        [[nodiscard]] const char* depth_loss_mode_name(const kernels::DepthLossMode mode) {
            switch (mode) {
            case kernels::DepthLossMode::AdaptiveWarpedL1:
                return "adaptive-warped-l1";
            case kernels::DepthLossMode::PearsonAbs:
            default:
                return "pearson";
            }
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

        struct LoadedCameraMetricsInputs {
            lfs::core::Tensor gt_image;
            lfs::core::Tensor mask;
        };

        [[nodiscard]] lfs::io::LoadParams make_metrics_load_params(
            const Trainer::GTLoadConfigSnapshot& gt_config,
            const lfs::core::Camera& camera,
            const bool apply_undistort,
            const bool output_uint8 = false) {
            lfs::io::LoadParams params;
            params.resize_factor = gt_config.resize_factor;
            params.max_width = gt_config.max_width;
            params.output_uint8 = output_uint8;
            if (apply_undistort && camera.is_undistort_prepared()) {
                params.undistort = &camera.undistort_params();
            }
            return params;
        }

        [[nodiscard]] lfs::core::Tensor normalize_mask_tensor(lfs::core::Tensor mask) {
            if (!mask.is_valid()) {
                return {};
            }

            if (mask.device() != lfs::core::Device::CUDA) {
                mask = mask.to(lfs::core::Device::CUDA);
            }

            if (mask.ndim() == 3 && mask.shape()[0] >= 3) {
                const auto r = mask.slice(0, 0, 1).squeeze(0);
                const auto g = mask.slice(0, 1, 2).squeeze(0);
                const auto b = mask.slice(0, 2, 3).squeeze(0);
                mask = ((r + g + b) / 3.0f).contiguous();
            } else if (mask.ndim() == 3 && mask.shape()[0] == 1) {
                mask = mask.squeeze(0).contiguous();
            }

            return mask;
        }

        std::expected<lfs::core::Tensor, std::string> load_external_mask_for_metrics(
            const lfs::core::Camera& camera,
            const Trainer::GTLoadConfigSnapshot& gt_config,
            const lfs::core::param::OptimizationParameters& opt_params,
            lfs::io::PipelinedImageLoader& image_loader) {
            try {
                auto mask = image_loader.load_image_immediate(
                    camera.mask_path(),
                    make_metrics_load_params(gt_config, camera, false, false));
                mask = normalize_mask_tensor(std::move(mask));
                if (!mask.is_valid()) {
                    return std::unexpected("failed to decode mask");
                }

                const size_t H = mask.shape()[0];
                const size_t W = mask.shape()[1];
                float* const mask_ptr = mask.ptr<float>();
                if (opt_params.invert_masks) {
                    lfs::io::cuda::launch_mask_invert(mask_ptr, H, W, nullptr);
                }
                if (opt_params.mask_threshold > 0.0f) {
                    lfs::io::cuda::launch_mask_threshold(mask_ptr, H, W, opt_params.mask_threshold, nullptr);
                }

                if (camera.is_undistort_prepared()) {
                    const auto scaled = lfs::core::scale_undistort_params(
                        camera.undistort_params(),
                        static_cast<int>(W),
                        static_cast<int>(H));
                    mask = lfs::core::undistort_mask(mask, scaled, nullptr);
                }

                return mask.ge(0.5f).to(lfs::core::DataType::UInt8).contiguous();
            } catch (const std::exception& e) {
                return std::unexpected(e.what());
            }
        }

        std::expected<LoadedCameraMetricsInputs, std::string> load_alpha_masked_metrics_inputs(
            const lfs::core::Camera& camera,
            const Trainer::GTLoadConfigSnapshot& gt_config,
            const lfs::core::param::OptimizationParameters& opt_params) {
            try {
                auto [img_data, width, height, channels] = lfs::core::load_image_with_alpha(
                    camera.image_path(), gt_config.resize_factor, gt_config.max_width);

                if (!img_data || channels != 4) {
                    if (img_data) {
                        lfs::core::free_image(img_data);
                    }
                    return std::unexpected("failed to decode RGBA image");
                }

                const auto H = static_cast<size_t>(height);
                const auto W = static_cast<size_t>(width);

                auto cpu_tensor = lfs::core::Tensor::from_blob(
                    img_data, lfs::core::TensorShape({H, W, 4}),
                    lfs::core::Device::CPU, lfs::core::DataType::UInt8);
                auto gpu_uint8 = cpu_tensor.to(lfs::core::Device::CUDA);
                lfs::core::free_image(img_data);

                auto rgb = lfs::core::Tensor::zeros(
                    lfs::core::TensorShape({3, H, W}),
                    lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
                auto mask = lfs::core::Tensor::zeros(
                    lfs::core::TensorShape({H, W}),
                    lfs::core::Device::CUDA, lfs::core::DataType::Float32);

                lfs::io::cuda::launch_uint8_rgba_split_to_uint8_rgb_and_float32_alpha(
                    gpu_uint8.ptr<uint8_t>(), rgb.ptr<uint8_t>(), mask.ptr<float>(), H, W, nullptr);

                if (opt_params.invert_masks) {
                    lfs::io::cuda::launch_mask_invert(mask.ptr<float>(), H, W, nullptr);
                }
                if (opt_params.mask_threshold > 0.0f) {
                    lfs::io::cuda::launch_mask_threshold(mask.ptr<float>(), H, W, opt_params.mask_threshold, nullptr);
                }

                if (camera.is_undistort_prepared()) {
                    const auto scaled = lfs::core::scale_undistort_params(
                        camera.undistort_params(),
                        static_cast<int>(W),
                        static_cast<int>(H));
                    auto rgb_float = rgb.to(lfs::core::DataType::Float32) / 255.0f;
                    rgb_float = lfs::core::undistort_image(rgb_float, scaled, nullptr);
                    auto rgb_uint8 = lfs::core::Tensor::empty(
                        rgb_float.shape(), lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
                    lfs::io::cuda::launch_float32_chw_to_uint8_chw(
                        rgb_float.ptr<float>(),
                        rgb_uint8.ptr<uint8_t>(),
                        rgb_float.shape()[1],
                        rgb_float.shape()[2],
                        rgb_float.shape()[0],
                        nullptr);
                    rgb = std::move(rgb_uint8);
                    mask = lfs::core::undistort_mask(mask, scaled, nullptr);
                }

                mask = mask.ge(0.5f).to(lfs::core::DataType::UInt8).contiguous();
                return LoadedCameraMetricsInputs{.gt_image = std::move(rgb), .mask = std::move(mask)};
            } catch (const std::exception& e) {
                return std::unexpected(e.what());
            }
        }

        std::expected<LoadedCameraMetricsInputs, std::string> load_camera_metrics_inputs(
            const lfs::core::Camera& camera,
            const Trainer::GTLoadConfigSnapshot& gt_config,
            const lfs::core::param::OptimizationParameters& opt_params,
            const std::shared_ptr<lfs::io::PipelinedImageLoader>& image_loader) {
            std::optional<lfs::io::PipelinedImageLoader> fallback_loader;
            lfs::io::PipelinedImageLoader* loader = image_loader.get();
            if (!loader) {
                fallback_loader.emplace();
                loader = &*fallback_loader;
            }

            const auto mask_mode = opt_params.mask_mode;
            const bool use_masking =
                mask_mode == lfs::core::param::MaskMode::Segment ||
                mask_mode == lfs::core::param::MaskMode::Ignore ||
                mask_mode == lfs::core::param::MaskMode::SegmentAndIgnore;

            // Sidecar mask file wins when present; alpha-as-mask is only used as fallback
            // (some datasets ship RGBA images with a degenerate constant alpha alongside
            // real per-pixel masks in masks/, and we must not let the alpha channel mask
            // them out).
            if (use_masking && !camera.has_mask() && opt_params.use_alpha_as_mask && camera.has_alpha()) {
                return load_alpha_masked_metrics_inputs(camera, gt_config, opt_params);
            }

            LoadedCameraMetricsInputs inputs;

            try {
                inputs.gt_image = loader->load_image_immediate(
                    camera.image_path(),
                    make_metrics_load_params(gt_config, camera, true, true));
            } catch (const std::exception& e) {
                return std::unexpected(e.what());
            }

            if (!inputs.gt_image.is_valid()) {
                return std::unexpected("failed to load ground-truth image");
            }

            if (use_masking && camera.has_mask()) {
                auto mask = load_external_mask_for_metrics(camera, gt_config, opt_params, *loader);
                if (!mask) {
                    return std::unexpected(mask.error());
                }
                inputs.mask = std::move(*mask);
            }

            return inputs;
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

        [[nodiscard]] size_t decoupled_fused_workspace_bytes(const kernels::DecoupledFusedL1SSIMWorkspace& workspace) {
            std::vector<std::pair<std::string, size_t>> entries;
            add_tensor_entry(entries, "decoupled.ssim_map", workspace.ssim_map);
            add_tensor_entry(entries, "decoupled.app_dm_dmu1", workspace.app_dm_dmu1);
            add_tensor_entry(entries, "decoupled.raw_dm_dmu1", workspace.raw_dm_dmu1);
            add_tensor_entry(entries, "decoupled.raw_dm_dsigma1_sq", workspace.raw_dm_dsigma1_sq);
            add_tensor_entry(entries, "decoupled.raw_dm_dsigma12", workspace.raw_dm_dsigma12);
            add_tensor_entry(entries, "decoupled.zero_terms", workspace.zero_terms);
            add_tensor_entry(entries, "decoupled.grad_corrected", workspace.grad_corrected);
            add_tensor_entry(entries, "decoupled.grad_raw", workspace.grad_raw);
            add_tensor_entry(entries, "decoupled.reduction_temp", workspace.reduction_temp);
            add_tensor_entry(entries, "decoupled.reduction_result", workspace.reduction_result);
            return sum_entry_bytes(entries);
        }

        [[nodiscard]] size_t masked_decoupled_fused_workspace_bytes(const kernels::MaskedDecoupledFusedL1SSIMWorkspace& workspace) {
            std::vector<std::pair<std::string, size_t>> entries;
            add_tensor_entry(entries, "masked_decoupled.ssim_map", workspace.ssim_map);
            add_tensor_entry(entries, "masked_decoupled.app_dm_dmu1", workspace.app_dm_dmu1);
            add_tensor_entry(entries, "masked_decoupled.raw_dm_dmu1", workspace.raw_dm_dmu1);
            add_tensor_entry(entries, "masked_decoupled.raw_dm_dsigma1_sq", workspace.raw_dm_dsigma1_sq);
            add_tensor_entry(entries, "masked_decoupled.raw_dm_dsigma12", workspace.raw_dm_dsigma12);
            add_tensor_entry(entries, "masked_decoupled.zero_terms", workspace.zero_terms);
            add_tensor_entry(entries, "masked_decoupled.grad_corrected", workspace.grad_corrected);
            add_tensor_entry(entries, "masked_decoupled.grad_raw", workspace.grad_raw);
            add_tensor_entry(entries, "masked_decoupled.reduction_temp", workspace.reduction_temp);
            add_tensor_entry(entries, "masked_decoupled.masked_loss", workspace.masked_loss);
            add_tensor_entry(entries, "masked_decoupled.mask_sum", workspace.mask_sum);
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

        [[nodiscard]] bool live_vram_profiler_enabled() {
            return lfs::diagnostics::VramProfiler::instance().enabled();
        }

        void record_vram_current(std::string_view scope,
                                 std::string_view label,
                                 const size_t bytes,
                                 const bool publish_zero = false,
                                 const lfs::diagnostics::VramAllocationMethod method =
                                     lfs::diagnostics::VramAllocationMethod::External) {
            if (bytes == 0 && !publish_zero) {
                return;
            }
            lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(scope, label, bytes, method);
        }

        void record_vram_tensor(std::string_view scope,
                                std::string_view label,
                                const lfs::core::Tensor& tensor) {
            const auto method =
                tensor.device() == lfs::core::Device::CUDA && !tensor.is_external_storage()
                    ? lfs::diagnostics::VramAllocationMethod::Direct
                    : lfs::diagnostics::VramAllocationMethod::External;
            record_vram_current(scope, label, tensor_reserved_bytes(tensor), false, method);
        }

        void record_vram_entries(std::string_view scope,
                                 const std::vector<std::pair<std::string, size_t>>& entries) {
            for (const auto& [name, bytes] : entries) {
                record_vram_current(scope, name, bytes);
            }
        }

        void record_pipeline_vram_breakdown(const std::shared_ptr<lfs::io::PipelinedImageLoader>& loader) {
            if (!loader) {
                record_vram_current("io.pipeline", "output_queue.images", 0, true);
                record_vram_current("io.pipeline", "output_queue.masks", 0, true);
                record_vram_current("io.pipeline", "pending.images", 0, true);
                record_vram_current("io.pipeline", "pending.masks", 0, true);
                return;
            }

            const auto stats = loader->get_gpu_memory_stats();
            record_vram_current("io.pipeline", "output_queue.images", stats.output_image_bytes, true);
            record_vram_current("io.pipeline", "output_queue.masks", stats.output_mask_bytes, true);
            record_vram_current("io.pipeline", "pending.images", stats.pending_image_bytes, true);
            record_vram_current("io.pipeline", "pending.masks", stats.pending_mask_bytes, true);
        }

        void record_splat_vram_breakdown(const lfs::core::SplatData& splat) {
            std::vector<std::pair<std::string, size_t>> entries;
            add_tensor_entry(entries, "means", splat.means());
            add_tensor_entry(entries, "sh0", splat.sh0());
            add_tensor_entry(entries, "shN", splat.shN());
            add_tensor_entry(entries, "scaling", splat.scaling_raw());
            add_tensor_entry(entries, "rotation", splat.rotation_raw());
            add_tensor_entry(entries, "opacity", splat.opacity_raw());
            add_tensor_entry(entries, "deleted", splat.deleted());
            add_tensor_entry(entries, "densification_info", splat._densification_info);
            record_vram_entries("model.gaussians", entries);
        }

        void record_optimizer_vram_breakdown(const AdamOptimizer& optimizer) {
            for (const auto type : AdamOptimizer::all_param_types()) {
                const auto* state = optimizer.get_state(type);
                if (!state) {
                    continue;
                }
                const std::string prefix =
                    type == ParamType::Means      ? "means"
                    : type == ParamType::Sh0      ? "sh0"
                    : type == ParamType::ShN      ? "shN"
                    : type == ParamType::Scaling  ? "scaling"
                    : type == ParamType::Rotation ? "rotation"
                                                  : "opacity";
                record_vram_tensor("optimizer.adam", prefix + ".grad", state->grad);
                record_vram_tensor("optimizer.adam", prefix + ".exp_avg", state->exp_avg);
                record_vram_tensor("optimizer.adam", prefix + ".exp_avg_sq", state->exp_avg_sq);
            }
        }

        void record_fastgs_vram_breakdown(const FastRasterizeContext& ctx,
                                          const RenderOutput& output,
                                          const lfs::core::Tensor& gt_tile,
                                          const lfs::core::Tensor& bg_tile,
                                          const lfs::core::Tensor& tile_error_map,
                                          const bool run_gaussian_backward,
                                          const std::size_t num_primitives) {
            constexpr std::string_view scope = "rasterizer.fastgs";
            record_vram_current(scope, "forward.per_primitive_buffers", ctx.forward_ctx.per_primitive_buffers_size);
            record_vram_current(scope, "forward.per_tile_buffers", ctx.forward_ctx.per_tile_buffers_size);
            record_vram_current(scope, "forward.sorted_indices_live", ctx.forward_ctx.sorted_primitive_indices_size);
            // Sort scratch is released after forward, and sort_total includes sorted_indices_live.
            // Clear these legacy live rows so the HUD does not count transient/duplicate bytes
            // as retained process VRAM.
            record_vram_current(scope, "forward.sort_scratch_transient", 0, true);
            record_vram_current(scope, "forward.sort_total_transient", 0, true);
            record_vram_current(scope, "backward.grad_mean2d_helper", num_primitives * 2 * sizeof(float));
            record_vram_current(scope, "backward.grad_conic_helper", num_primitives * 3 * sizeof(float));
            record_vram_current(scope,
                                "backward.fused_grad_opacity_helper",
                                run_gaussian_backward && ctx.forward_ctx.grad_opacity_helper
                                    ? num_primitives * sizeof(float)
                                    : 0,
                                true);
            record_vram_current(scope,
                                "backward.fused_grad_color_helper",
                                run_gaussian_backward && ctx.forward_ctx.grad_color_helper
                                    ? num_primitives * 3 * sizeof(float)
                                    : 0,
                                true);
            record_vram_tensor(scope, "output.image", output.image);
            record_vram_tensor(scope, "output.alpha", output.alpha);
            record_vram_tensor(scope, "saved.bg_color", ctx.bg_color);
            record_vram_tensor("train.inputs", "gt_tile", gt_tile);
            record_vram_tensor("train.inputs", "background_tile", bg_tile);
            record_vram_tensor("train.losses", "densification_error_map.live", tile_error_map);
        }

        void record_gsplat_vram_breakdown(const GsplatRasterizeContext& ctx,
                                          const RenderOutput& output,
                                          const lfs::core::Tensor& gt_tile,
                                          const lfs::core::Tensor& bg_tile,
                                          const lfs::core::Tensor& tile_error_map) {
            constexpr std::string_view scope = "rasterizer.gsplat";
            if (auto* arena = lfs::core::GlobalArenaManager::instance().try_get_arena()) {
                std::size_t frame_bytes = 0;
                for (const auto& buffer : arena->get_frame_buffers(ctx.frame_id)) {
                    frame_bytes += buffer.size;
                }
                record_vram_current(scope, "arena.frame_buffers", frame_bytes);
                const auto info = arena->get_memory_info();
                record_vram_current(scope, "arena.capacity", info.arena_capacity);
                record_vram_current(scope, "arena.current_usage", info.current_usage);
                record_vram_current(scope, "arena.peak_usage", info.peak_usage);
            }
            record_vram_current(scope, "forward.isect_ids", static_cast<std::size_t>(ctx.n_isects) * sizeof(std::int64_t));
            record_vram_current(scope, "forward.flatten_ids", static_cast<std::size_t>(ctx.n_isects) * sizeof(std::int32_t));
            record_vram_tensor(scope, "output.image", output.image);
            record_vram_tensor(scope, "output.alpha", output.alpha);
            record_vram_tensor(scope, "camera.K_tensor", ctx.K_tensor);
            record_vram_tensor(scope, "camera.radial_cuda", ctx.radial_cuda);
            record_vram_tensor(scope, "camera.tangential_cuda", ctx.tangential_cuda);
            record_vram_tensor(scope, "camera.thin_prism_cuda", ctx.thin_prism_cuda);
            record_vram_tensor("train.inputs", "gt_tile", gt_tile);
            record_vram_tensor("train.inputs", "background_tile", bg_tile);
            record_vram_tensor("train.losses", "densification_error_map.live", tile_error_map);
        }

        void syncTrainingSceneTopology(lfs::core::Scene* const scene,
                                       const lfs::core::SplatData& model) {
            if (!scene) {
                return;
            }
            scene->syncTrainingModelTopology(static_cast<size_t>(model.size()));
        }

        [[nodiscard]] std::array<float, 3> lerp_color(const std::array<float, 3>& a,
                                                      const std::array<float, 3>& b,
                                                      const float t) {
            return {
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t};
        }

        [[nodiscard]] std::array<float, 3> camera_loss_color(const float t) {
            constexpr std::array<float, 3> BEST{0.10f, 0.84f, 0.24f};
            constexpr std::array<float, 3> WORST{0.90f, 0.16f, 0.16f};
            return lerp_color(BEST, WORST, std::clamp(t, 0.0f, 1.0f));
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
                        width * height * size_t{3} * sizeof(uint8_t));
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

            constexpr float NON_JPEG_THRESHOLD = 0.1f;
            constexpr size_t JPEG_HOT_OUTPUT_QUEUE_SIZE = 2;
            constexpr size_t JPEG_HOT_DECODER_POOL_SIZE = 2;
            const float non_jpeg_ratio = dataset ? dataset->get_non_jpeg_ratio() : 0.0f;
            if (non_jpeg_ratio <= NON_JPEG_THRESHOLD) {
                if (config.output_queue_size > JPEG_HOT_OUTPUT_QUEUE_SIZE) {
                    LOG_INFO(
                        "Reducing JPEG image ready queue {} -> {} (hot path keeps compressed prefetch)",
                        config.output_queue_size,
                        JPEG_HOT_OUTPUT_QUEUE_SIZE);
                    config.output_queue_size = JPEG_HOT_OUTPUT_QUEUE_SIZE;
                }
                if (config.decoder_pool_size > JPEG_HOT_DECODER_POOL_SIZE) {
                    LOG_INFO(
                        "Reducing nvImageCodec decoder pool {} -> {} for JPEG hot path",
                        config.decoder_pool_size,
                        JPEG_HOT_DECODER_POOL_SIZE);
                    config.decoder_pool_size = JPEG_HOT_DECODER_POOL_SIZE;
                }
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
        setCameraLossHeatmap(nullptr);

        LOG_DEBUG("Trainer cleanup complete");
    }

    int Trainer::get_regular_iterations() const {
        return static_cast<int>(params_.optimization.iterations);
    }

    int Trainer::get_active_sparsify_steps() const {
        return params_.optimization.enable_sparsity
                   ? std::max(0, params_.optimization.sparsify_steps)
                   : 0;
    }

    int Trainer::get_sparsity_boundary_iteration() const {
        return get_regular_iterations();
    }

    int Trainer::get_total_iterations() const {
        return get_regular_iterations() + get_active_sparsify_steps();
    }

    lfs::core::param::OptimizationParameters Trainer::get_runtime_optimization_params() const {
        auto runtime_params = params_.optimization;
        runtime_params.iterations = static_cast<size_t>(std::max(0, get_total_iterations()));
        return runtime_params;
    }

    void Trainer::sync_strategy_optimization_params() {
        if (strategy_) {
            strategy_->set_optimization_params(get_runtime_optimization_params());
        }
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
                get_total_iterations(),
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
            const float reg_weight = params_.optimization.ppisp_reg_weight;
            config.exposure_mean *= reg_weight;
            config.vig_center *= reg_weight;
            config.vig_channel *= reg_weight;
            config.vig_non_pos *= reg_weight;
            config.color_mean *= reg_weight;
            config.crf_channel *= reg_weight;

            ppisp_ = std::make_unique<PPISP>(get_total_iterations(), config);
            for (const auto& cam : train_dataset_->get_cameras()) {
                if (cam) {
                    ppisp_->register_frame(cam->uid(), cam->camera_id());
                }
            }
            ppisp_->finalize();

            LOG_INFO("PPISP initialized: {} cameras (physical), {} frames, lr={:.2e}, warmup={}, reg_weight={:.2e}",
                     ppisp_->num_cameras(), ppisp_->num_frames(), params_.optimization.ppisp_lr,
                     config.warmup_steps, reg_weight);

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

            const int activation_step = params_.optimization.resolved_ppisp_controller_activation_step(get_total_iterations());
            if (params_.optimization.ppisp_controller_activation_step < 0) {
                params_.optimization.ppisp_controller_activation_step = activation_step;
            }
            int distillation_iters = get_total_iterations() - activation_step;
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
    std::expected<Trainer::PhotometricLossResult, std::string> Trainer::compute_photometric_loss_with_gradient(
        const lfs::core::Tensor& corrected,
        const lfs::core::Tensor& gt_image,
        const lfs::core::param::OptimizationParameters& opt_params,
        const lfs::core::Tensor& raw_rendered) {
        const bool use_decoupled_appearance_loss =
            raw_rendered.is_valid() &&
            raw_rendered.numel() > 0 &&
            opt_params.lambda_dssim > 0.0f;

        if (use_decoupled_appearance_loss) {
            auto [loss_tensor, ctx] = lfs::training::kernels::decoupled_fused_l1_ssim_forward(
                corrected, raw_rendered, gt_image, opt_params.lambda_dssim, decoupled_fused_workspace_,
                /*apply_valid_padding=*/true);
            auto grads = lfs::training::kernels::decoupled_fused_l1_ssim_backward(ctx, decoupled_fused_workspace_);

            if (corrected.ndim() == 3) {
                grads.grad_corrected = grads.grad_corrected.squeeze(0);
                grads.grad_raw = grads.grad_raw.squeeze(0);
            }

            return PhotometricLossResult{
                .loss = loss_tensor,
                .grad_corrected = grads.grad_corrected,
                .grad_raw = grads.grad_raw};
        }

        lfs::training::losses::PhotometricLoss::Params params{.lambda_dssim = opt_params.lambda_dssim};
        auto result = photometric_loss_.forward(corrected, gt_image, params);
        if (!result) {
            return std::unexpected(result.error());
        }
        auto [loss_tensor, ctx] = *result;
        return PhotometricLossResult{
            .loss = loss_tensor,
            .grad_corrected = ctx.grad_image,
            .grad_raw = {}};
    }

    std::expected<void, std::string> Trainer::validate_masks() {
        const auto& opt = params_.optimization;
        if (opt.mask_mode == lfs::core::param::MaskMode::None) {
            return {};
        }

        // Segment and ignore does not support mask invert
        if (opt.mask_mode == lfs::core::param::MaskMode::SegmentAndIgnore && opt.invert_masks) {
            LOG_WARN("invert_masks is ignored in SegmentAndIgnore mode (would scramble the mask bands)");
            params_.optimization.invert_masks = false;
        }

        size_t alpha_count = 0;
        size_t masks_found = 0;
        for (const auto& cam : train_dataset_->get_cameras()) {
            if (cam && cam->has_alpha())
                ++alpha_count;
            if (cam && cam->has_mask())
                ++masks_found;
        }

        // Sidecar masks take precedence over the alpha channel (see dataset prefetch),
        // so report them first; alpha-as-mask only covers cameras without a sidecar.
        if (masks_found > 0) {
            LOG_INFO("Found {} masks{}", masks_found, opt.invert_masks ? " (inverted)" : "");
            return {};
        }

        if (opt.use_alpha_as_mask && alpha_count > 0) {
            LOG_INFO("Using alpha channel as mask source ({}/{} cameras){}",
                     alpha_count, train_dataset_->get_cameras().size(),
                     opt.invert_masks ? " (inverted)" : "");
            return {};
        }

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

    std::expected<Trainer::MaskLossResult, std::string> Trainer::compute_photometric_loss_with_mask(
        const lfs::core::Tensor& corrected,
        const lfs::core::Tensor& gt_image,
        const lfs::core::Tensor& mask,
        const lfs::core::Tensor& alpha,
        const lfs::core::param::OptimizationParameters& opt_params,
        const lfs::core::Tensor& raw_rendered) {

        using namespace lfs::core;
        constexpr float ALPHA_CONSISTENCY_WEIGHT = 10.0f;

        const auto mode = opt_params.mask_mode;
        const Tensor mask_2d = mask.ndim() == 3 ? mask.squeeze(0) : mask;
        Tensor mask_2d_th = mask_2d;
        if (mode == param::MaskMode::SegmentAndIgnore) {
            mask_2d_th = mask_2d_th.masked_fill(mask_2d_th <= 250, 0);  // Set all Ignore and Segment to 0
            mask_2d_th = mask_2d_th.masked_fill(mask_2d_th > 250, 255); // Keep everything > 250
        }

        const auto mask_as_float = [](const Tensor t) -> Tensor {
            return (t.dtype() == DataType::UInt8 || t.dtype() == DataType::Bool)
                       ? t.gt(0).to(DataType::Float32)
                       : t;
        };

        Tensor loss, grad_corrected, grad_raw, grad_alpha;
        const bool use_decoupled_appearance_loss =
            raw_rendered.is_valid() &&
            raw_rendered.numel() > 0 &&
            opt_params.lambda_dssim > 0.0f;

        if (mode == param::MaskMode::Segment || mode == param::MaskMode::Ignore || mode == param::MaskMode::SegmentAndIgnore) {
            if (use_decoupled_appearance_loss) {
                auto [loss_tensor, ctx] = lfs::training::kernels::masked_decoupled_fused_l1_ssim_forward(
                    corrected, raw_rendered, gt_image, mask_2d_th, opt_params.lambda_dssim,
                    masked_decoupled_fused_workspace_);
                auto grads = lfs::training::kernels::masked_decoupled_fused_l1_ssim_backward(
                    ctx, masked_decoupled_fused_workspace_);

                grad_corrected = grads.grad_corrected;
                grad_raw = grads.grad_raw;
                loss = loss_tensor;

                if (grad_corrected.ndim() == 4 && corrected.ndim() == 3) {
                    grad_corrected = grad_corrected.squeeze(0);
                }
                if (grad_raw.ndim() == 4 && corrected.ndim() == 3) {
                    grad_raw = grad_raw.squeeze(0);
                }
            } else {
                auto [loss_tensor, ctx] = lfs::training::kernels::masked_fused_l1_ssim_forward(
                    corrected, gt_image, mask_2d_th, opt_params.lambda_dssim, masked_fused_workspace_);

                grad_corrected = lfs::training::kernels::masked_fused_l1_ssim_backward(ctx, masked_fused_workspace_);
                loss = loss_tensor;

                if (grad_corrected.ndim() == 4 && corrected.ndim() == 3) {
                    grad_corrected = grad_corrected.squeeze(0);
                }
            }

            // Segment: opacity penalty for background
            if ((mode == param::MaskMode::Segment || mode == param::MaskMode::SegmentAndIgnore) && alpha.is_valid()) {
                Tensor mask_2d_th_segment = mask_2d;
                if (mode == param::MaskMode::SegmentAndIgnore) {
                    // Values used for ignore (<128) do not contribute to opacity penalty
                    // Values in the range 128<=x<=250 contribute to the opacity penalty
                    // Values > 250 are kept
                    mask_2d_th_segment = mask_2d_th_segment.masked_fill(mask_2d_th_segment < 128, 255);
                    mask_2d_th_segment = mask_2d_th_segment.masked_fill(mask_2d_th_segment >= 128 && mask_2d_th_segment <= 250, 0);
                    mask_2d_th_segment = mask_2d_th_segment.masked_fill(mask_2d_th_segment > 250, 255);
                }
                const Tensor mask_2d_th_segment_f = mask_as_float(mask_2d_th_segment);
                const Tensor alpha_2d = alpha.ndim() == 3 ? alpha.squeeze(0) : alpha;
                const Tensor bg_mask = Tensor::full(mask_2d_th_segment_f.shape(), 1.0f, mask_2d_th_segment_f.device()) - mask_2d_th_segment_f;
                const Tensor penalty_weights = bg_mask.pow(opt_params.mask_opacity_penalty_power);
                const Tensor penalty = (alpha_2d * penalty_weights).mean() * opt_params.mask_opacity_penalty_weight;

                const float inv_pixels = opt_params.mask_opacity_penalty_weight / static_cast<float>(alpha_2d.numel());
                grad_alpha = penalty_weights * inv_pixels;
                loss = loss + penalty;
            }

        } else if (mode == param::MaskMode::AlphaConsistent) {
            // Standard photometric loss
            auto result = compute_photometric_loss_with_gradient(corrected, gt_image, opt_params, raw_rendered);
            if (!result) {
                return std::unexpected(result.error());
            }
            loss = result->loss;
            grad_corrected = result->grad_corrected;
            grad_raw = result->grad_raw;

            // Alpha should match mask
            if (alpha.is_valid()) {
                const Tensor alpha_2d = alpha.ndim() == 3 ? alpha.squeeze(0) : alpha;
                const Tensor mask_f = mask_as_float(mask_2d_th);
                const Tensor alpha_loss = (alpha_2d - mask_f).abs().mean() * ALPHA_CONSISTENCY_WEIGHT;
                loss = loss + alpha_loss;
                grad_alpha = (alpha_2d - mask_f).sign() * (ALPHA_CONSISTENCY_WEIGHT / static_cast<float>(alpha_2d.numel()));
            }
        } else {
            auto fallback = compute_photometric_loss_with_gradient(corrected, gt_image, opt_params, raw_rendered);
            if (!fallback) {
                return std::unexpected(fallback.error());
            }
            return MaskLossResult{
                .loss = fallback->loss,
                .grad_corrected = fallback->grad_corrected,
                .grad_raw = fallback->grad_raw,
                .grad_alpha = {}};
        }

        return MaskLossResult{
            .loss = loss,
            .grad_corrected = grad_corrected,
            .grad_raw = grad_raw,
            .grad_alpha = grad_alpha};
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
        // Blocking flag (not cudaStreamNonBlocking): legacy-stream work — blocking
        // cudaMemcpy readbacks, cold-path loader uploads — must stay implicitly
        // ordered against training kernels. Overlap partners (loader decode,
        // viewer render) use non-blocking streams with explicit event edges.
        cudaStreamCreate(&training_stream_);
        cudaStreamCreateWithFlags(&metrics_stream_, cudaStreamNonBlocking);
        nvtxNameCudaStreamA(training_stream_, "lfs.train");
        nvtxNameCudaStreamA(callback_stream_, "lfs.train.callback");
        nvtxNameCudaStreamA(metrics_stream_, "lfs.metrics");
        createSyncPrimitives();

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
        // Blocking flag (not cudaStreamNonBlocking): legacy-stream work — blocking
        // cudaMemcpy readbacks, cold-path loader uploads — must stay implicitly
        // ordered against training kernels. Overlap partners (loader decode,
        // viewer render) use non-blocking streams with explicit event edges.
        cudaStreamCreate(&training_stream_);
        cudaStreamCreateWithFlags(&metrics_stream_, cudaStreamNonBlocking);
        nvtxNameCudaStreamA(training_stream_, "lfs.train");
        nvtxNameCudaStreamA(callback_stream_, "lfs.train.callback");
        nvtxNameCudaStreamA(metrics_stream_, "lfs.metrics");
        createSyncPrimitives();

        if (!scene.hasTrainingData()) {
            throw std::runtime_error("Scene has no cameras");
        }

        LOG_DEBUG("Trainer constructed from Scene with {} cameras", scene.getAllCameras().size());
    }

    void Trainer::createSyncPrimitives() {
        cudaEventCreateWithFlags(&params_ready_event_, cudaEventDisableTiming);
        for (auto& event : reader_done_events_) {
            cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        }
        for (auto& slot : loss_slots_) {
            slot.pinned = static_cast<float*>(
                lfs::core::PinnedMemoryAllocator::instance().allocate(sizeof(float)));
            cudaEventCreateWithFlags(&slot.done, cudaEventDisableTiming);
        }
    }

    void Trainer::destroySyncPrimitives() {
        std::lock_guard<std::mutex> lock(stream_sync_mutex_);
        params_ready_recorded_ = false;
        reader_done_pending_ = 0;
        viewer_release_semaphore_ = nullptr;
        if (params_ready_event_) {
            cudaEventDestroy(params_ready_event_);
            params_ready_event_ = nullptr;
        }
        for (auto& event : reader_done_events_) {
            if (event) {
                cudaEventDestroy(event);
                event = nullptr;
            }
        }
        for (auto& slot : loss_slots_) {
            if (slot.done) {
                cudaEventDestroy(slot.done);
                slot.done = nullptr;
            }
            if (slot.pinned) {
                lfs::core::PinnedMemoryAllocator::instance().deallocate(slot.pinned, nullptr);
                slot.pinned = nullptr;
            }
            slot.in_flight = false;
        }
    }

    void Trainer::submitLossReadback(const lfs::core::Tensor& total_loss, int iter) {
        LossReadbackSlot& slot = loss_slots_[loss_slot_head_];
        if (!slot.pinned || !slot.done) {
            return;
        }
        if (slot.in_flight) {
            // Ring full: the GPU is LOSS_RING submit intervals behind —
            // explicit backpressure instead of silently dropping the sample.
            // The caller harvests right before submitting, so this slot's
            // value was already consumed once the event completes.
            cudaEventSynchronize(slot.done);
            slot.in_flight = false;
        }
        if (cudaMemcpyAsync(slot.pinned, total_loss.ptr<float>(), sizeof(float),
                            cudaMemcpyDeviceToHost, training_stream_) != cudaSuccess) {
            return;
        }
        if (cudaEventRecord(slot.done, training_stream_) == cudaSuccess) {
            slot.iter = iter;
            slot.in_flight = true;
            loss_slot_head_ = (loss_slot_head_ + 1) % LOSS_RING;
        }
    }

    std::expected<void, std::string> Trainer::harvestLossReadbacks(bool drain, bool in_controller_phase) {
        for (size_t i = 0; i < LOSS_RING; ++i) {
            LossReadbackSlot& slot = loss_slots_[(loss_slot_head_ + i) % LOSS_RING];
            if (!slot.in_flight) {
                continue;
            }
            if (drain) {
                if (cudaEventSynchronize(slot.done) != cudaSuccess) {
                    slot.in_flight = false;
                    continue;
                }
            } else if (cudaEventQuery(slot.done) != cudaSuccess) {
                break;
            }
            slot.in_flight = false;

            const float loss_value = *slot.pinned;
            if (std::isnan(loss_value) || std::isinf(loss_value)) {
                return std::unexpected(std::format("NaN/Inf loss at iteration {}", slot.iter));
            }

            current_loss_ = loss_value;
            if (progress_) {
                progress_->update(
                    slot.iter,
                    loss_value,
                    static_cast<int>(strategy_->get_model().size()),
                    get_progress_phase(slot.iter, in_controller_phase));
            }
            lfs::core::events::state::TrainingProgress{
                .iteration = slot.iter,
                .loss = loss_value,
                .num_gaussians = static_cast<int>(strategy_->get_model().size()),
                .is_refining = strategy_->is_refining(slot.iter)}
                .emit();
        }
        return {};
    }

    bool Trainer::modelAccessLockEnabled() {
        static const bool enabled = [] {
            const char* v = std::getenv("LFS_NO_MODEL_ACCESS_LOCK");
            return !(v && v[0] == '1');
        }();
        return enabled;
    }

    void Trainer::beginModelRead(cudaStream_t reader_stream) {
        std::lock_guard<std::mutex> lock(stream_sync_mutex_);
        if (params_ready_event_ && params_ready_recorded_) {
            cudaStreamWaitEvent(reader_stream, params_ready_event_, 0);
        }
    }

    void Trainer::endModelRead(cudaStream_t reader_stream) {
        std::lock_guard<std::mutex> lock(stream_sync_mutex_);
        cudaEvent_t& slot = reader_done_events_[reader_done_head_];
        if (!slot) {
            return;
        }
        const uint32_t bit = 1u << reader_done_head_;
        if (reader_done_pending_ & bit) {
            // Ring full: the slot's previous record hasn't been consumed by a
            // step yet. Drain it host-side before reuse — re-recording would
            // drop the older reader's edge.
            cudaEventSynchronize(slot);
        }
        if (cudaEventRecord(slot, reader_stream) == cudaSuccess) {
            reader_done_pending_ |= bit;
        }
        reader_done_head_ = (reader_done_head_ + 1) % READER_DONE_RING;
    }

    void Trainer::setViewerReleaseFence(cudaExternalSemaphore_t semaphore) {
        std::lock_guard<std::mutex> lock(stream_sync_mutex_);
        if (viewer_release_semaphore_ == semaphore) {
            return;
        }
        viewer_release_semaphore_ = semaphore;
        viewer_borrow_waited_ = 0;
        // A new fence is a fresh timeline starting at 0 — a borrow value from
        // the previous timeline would make the trainer wait a value the new
        // semaphore never reaches.
        viewer_borrow_value_.store(0, std::memory_order_release);
    }

    void Trainer::publishViewerBorrow(uint64_t value) {
        // Monotonic: prompt per-submit publishes and the frame-scope publisher
        // may interleave; never regress to an older value.
        uint64_t current = viewer_borrow_value_.load(std::memory_order_relaxed);
        while (current < value &&
               !viewer_borrow_value_.compare_exchange_weak(
                   current, value, std::memory_order_release, std::memory_order_relaxed)) {
        }
    }

    void Trainer::recordParamsReady() {
        std::lock_guard<std::mutex> lock(stream_sync_mutex_);
        if (!params_ready_event_) {
            return;
        }
        // training_stream_ is a blocking stream, so the record is also ordered
        // after the legacy-stream rasterizer writes enqueued this step.
        if (cudaEventRecord(params_ready_event_, training_stream_) == cudaSuccess) {
            params_ready_recorded_ = true;
        }
    }

    void Trainer::waitForModelReaders() {
        std::lock_guard<std::mutex> lock(stream_sync_mutex_);
        if (reader_done_pending_ != 0) {
            for (size_t i = 0; i < READER_DONE_RING; ++i) {
                if (reader_done_pending_ & (1u << i)) {
                    cudaStreamWaitEvent(training_stream_, reader_done_events_[i], 0);
                }
            }
            reader_done_pending_ = 0;
        }

        const uint64_t borrow = viewer_borrow_value_.load(std::memory_order_acquire);
        if (viewer_release_semaphore_ && borrow > viewer_borrow_waited_) {
            cudaExternalSemaphoreWaitParams wait_params{};
            wait_params.params.fence.value = borrow;
            if (cudaWaitExternalSemaphoresAsync(&viewer_release_semaphore_, &wait_params, 1,
                                                training_stream_) == cudaSuccess) {
                viewer_borrow_waited_ = borrow;
            }
        }
    }

    bool Trainer::fillCameraLossColors(
        const std::vector<std::shared_ptr<const lfs::core::Camera>>& cameras,
        std::vector<std::array<float, 3>>& colors) const {
        constexpr std::array<float, 3> UNSEEN{1.0f, 1.0f, 1.0f};
        colors.assign(cameras.size(), UNSEEN);

        const auto heatmap = getCameraLossHeatmap();
        if (!heatmap) {
            return true;
        }

        std::shared_lock lock(heatmap->snapshot_mutex);
        if (heatmap->published_colors.empty() || heatmap->published_valid.empty()) {
            return true;
        }

        for (size_t i = 0; i < cameras.size(); ++i) {
            const auto& camera = cameras[i];
            if (!camera) {
                continue;
            }

            const auto it = heatmap->uid_to_slot.find(camera->uid());
            if (it == heatmap->uid_to_slot.end()) {
                continue;
            }

            const size_t slot = it->second;
            if (slot >= heatmap->published_colors.size() ||
                slot >= heatmap->published_valid.size() ||
                !heatmap->published_valid[slot]) {
                continue;
            }

            colors[i] = heatmap->published_colors[slot];
        }

        return true;
    }

    std::expected<void, std::string> Trainer::initialize_camera_loss_heatmap(
        const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras) {
        setCameraLossHeatmap(nullptr);

        auto heatmap = std::make_shared<CameraLossHeatmapState>();
        heatmap->camera_uids.reserve(cameras.size());
        heatmap->uid_to_slot.reserve(cameras.size());

        for (const auto& camera : cameras) {
            if (!camera) {
                continue;
            }

            const int uid = camera->uid();
            if (heatmap->uid_to_slot.contains(uid)) {
                continue;
            }

            const size_t slot = heatmap->camera_uids.size();
            heatmap->camera_uids.push_back(uid);
            heatmap->uid_to_slot.emplace(uid, slot);
        }

        if (heatmap->camera_uids.empty()) {
            return std::unexpected("No cameras available for loss heatmap");
        }

        const lfs::core::TensorShape shape{heatmap->camera_uids.size()};
        heatmap->latest_loss_gpu = lfs::core::Tensor::full(shape, -1.0f, lfs::core::Device::CUDA);
        heatmap->ema_loss_gpu = lfs::core::Tensor::full(shape, -1.0f, lfs::core::Device::CUDA);
        heatmap->ema_loss_stage_cpu = lfs::core::Tensor::full(shape, -1.0f, lfs::core::Device::CPU);
        heatmap->published_colors.resize(heatmap->camera_uids.size());
        heatmap->published_valid.assign(heatmap->camera_uids.size(), 0u);

        if (const cudaError_t err = cudaStreamCreateWithFlags(&heatmap->copy_stream, cudaStreamNonBlocking);
            err != cudaSuccess) {
            return std::unexpected(std::format("Failed to create camera-loss copy stream: {}",
                                               cudaGetErrorString(err)));
        }

        if (const cudaError_t err = cudaEventCreateWithFlags(&heatmap->ready_event, cudaEventDisableTiming);
            err != cudaSuccess) {
            return std::unexpected(std::format("Failed to create camera-loss ready event: {}",
                                               cudaGetErrorString(err)));
        }

        if (const cudaError_t err = cudaEventCreateWithFlags(&heatmap->done_event, cudaEventDisableTiming);
            err != cudaSuccess) {
            return std::unexpected(std::format("Failed to create camera-loss done event: {}",
                                               cudaGetErrorString(err)));
        }

        setCameraLossHeatmap(std::move(heatmap));
        return {};
    }

    void Trainer::update_camera_loss_heatmap(const lfs::core::Camera& camera,
                                             const lfs::core::Tensor& image_loss) {
        const auto heatmap = getCameraLossHeatmap();
        if (!heatmap || !image_loss.is_valid() || image_loss.numel() != 1 ||
            image_loss.device() != lfs::core::Device::CUDA) {
            return;
        }

        const auto it = heatmap->uid_to_slot.find(camera.uid());
        if (it == heatmap->uid_to_slot.end()) {
            return;
        }

        const cudaStream_t stream = image_loss.stream();
        kernels::launch_update_camera_loss_heatmap(
            image_loss.ptr<float>(),
            static_cast<int>(it->second),
            CAMERA_LOSS_EMA_ALPHA,
            heatmap->latest_loss_gpu.ptr<float>(),
            heatmap->ema_loss_gpu.ptr<float>(),
            heatmap->camera_uids.size(),
            stream);

        heatmap->producer_stream = stream;
        heatmap->dirty = true;
    }

    void Trainer::publish_camera_loss_heatmap_snapshot() {
        const auto heatmap = getCameraLossHeatmap();
        if (!heatmap || !heatmap->ema_loss_stage_cpu.is_valid()) {
            return;
        }

        const size_t count = heatmap->camera_uids.size();
        const float* ema_ptr = heatmap->ema_loss_stage_cpu.ptr<float>();

        float min_loss = std::numeric_limits<float>::infinity();
        float max_loss = -std::numeric_limits<float>::infinity();
        size_t seen_count = 0;

        for (size_t i = 0; i < count; ++i) {
            const float loss = ema_ptr[i];
            if (!std::isfinite(loss) || loss < 0.0f) {
                continue;
            }
            min_loss = std::min(min_loss, loss);
            max_loss = std::max(max_loss, loss);
            ++seen_count;
        }

        std::vector<std::array<float, 3>> colors(count);
        std::vector<uint8_t> valid(count, 0u);

        if (seen_count > 0) {
            const bool degenerate_span = (max_loss - min_loss) < 1e-6f;
            const float inv_span = degenerate_span ? 0.0f : 1.0f / (max_loss - min_loss);

            for (size_t i = 0; i < count; ++i) {
                const float loss = ema_ptr[i];
                if (!std::isfinite(loss) || loss < 0.0f) {
                    continue;
                }

                const float t = degenerate_span ? 0.0f : std::clamp((loss - min_loss) * inv_span, 0.0f, 1.0f);
                colors[i] = camera_loss_color(t);
                valid[i] = 1u;
            }
        }

        std::unique_lock lock(heatmap->snapshot_mutex);
        heatmap->published_colors = std::move(colors);
        heatmap->published_valid = std::move(valid);
    }

    void Trainer::maybe_publish_camera_loss_heatmap(const int iter, const bool force) {
        const auto heatmap = getCameraLossHeatmap();
        if (!heatmap) {
            return;
        }

        if (heatmap->copy_in_flight) {
            cudaError_t status = force
                                     ? cudaEventSynchronize(heatmap->done_event)
                                     : cudaEventQuery(heatmap->done_event);
            if (status == cudaSuccess) {
                publish_camera_loss_heatmap_snapshot();
                heatmap->copy_in_flight = false;
            } else if (status != cudaErrorNotReady) {
                LOG_WARN("Camera-loss snapshot query failed: {}", cudaGetErrorString(status));
                heatmap->copy_in_flight = false;
                heatmap->dirty = true;
            }
        }

        // `nullptr` is a valid CUDA default stream, so only gate on actual work state here.
        if (!heatmap->dirty || heatmap->copy_in_flight) {
            return;
        }

        if (!force && (iter % CAMERA_LOSS_PUBLISH_INTERVAL) != 0) {
            return;
        }

        const size_t bytes = heatmap->camera_uids.size() * sizeof(float);

        if (const cudaError_t err = cudaEventRecord(heatmap->ready_event, heatmap->producer_stream);
            err != cudaSuccess) {
            LOG_WARN("Camera-loss ready event failed: {}", cudaGetErrorString(err));
            return;
        }
        if (const cudaError_t err = cudaStreamWaitEvent(heatmap->copy_stream, heatmap->ready_event, 0);
            err != cudaSuccess) {
            LOG_WARN("Camera-loss stream wait failed: {}", cudaGetErrorString(err));
            return;
        }
        if (const cudaError_t err = cudaMemcpyAsync(
                heatmap->ema_loss_stage_cpu.ptr<float>(),
                heatmap->ema_loss_gpu.ptr<float>(),
                bytes,
                cudaMemcpyDeviceToHost,
                heatmap->copy_stream);
            err != cudaSuccess) {
            LOG_WARN("Camera-loss async copy failed: {}", cudaGetErrorString(err));
            return;
        }
        if (const cudaError_t err = cudaEventRecord(heatmap->done_event, heatmap->copy_stream);
            err != cudaSuccess) {
            LOG_WARN("Camera-loss done event failed: {}", cudaGetErrorString(err));
            return;
        }

        heatmap->copy_in_flight = true;
        heatmap->dirty = false;

        if (force) {
            const cudaError_t status = cudaEventSynchronize(heatmap->done_event);
            if (status == cudaSuccess) {
                publish_camera_loss_heatmap_snapshot();
                heatmap->copy_in_flight = false;
            } else {
                LOG_WARN("Camera-loss snapshot sync failed: {}", cudaGetErrorString(status));
                heatmap->dirty = true;
            }
        }
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

            if (params_.optimization.enable_sparsity) {
                const size_t stop_refine_limit = static_cast<size_t>(std::max(0, get_regular_iterations()));
                if (params_.optimization.stop_refine > stop_refine_limit) {
                    LOG_WARN("Sparsity: clamping stop_refine from {} to {} to freeze topology before pruning",
                             params_.optimization.stop_refine, stop_refine_limit);
                    params_.optimization.stop_refine = stop_refine_limit;
                }
            }

            // Create DatasetConfig for lfs::training::CameraDataset
            lfs::training::DatasetConfig dataset_config;
            dataset_config.resize_factor = params.dataset.resize_factor;
            dataset_config.max_width = params.dataset.max_width;
            dataset_config.test_every = params.dataset.test_every;

            // Get source cameras from Scene nodes or base_dataset_
            std::vector<std::shared_ptr<lfs::core::Camera>> source_cameras;
            std::vector<std::shared_ptr<lfs::core::Camera>> train_cameras;
            std::vector<std::shared_ptr<lfs::core::Camera>> val_cameras;
            if (scene_) {
                source_cameras = scene_->getActiveCameras();
                if (source_cameras.empty()) {
                    return std::unexpected("Scene has no active cameras enabled for training");
                }

                if (params.optimization.enable_eval) {
                    for (const auto& camera : source_cameras) {
                        switch (camera->split()) {
                        case lfs::core::CameraSplit::Train:
                            train_cameras.push_back(camera);
                            break;
                        case lfs::core::CameraSplit::Eval:
                            val_cameras.push_back(camera);
                            break;
                        default:
                            assert(false && "Camera split must be Train or Eval");
                            break;
                        }
                    }
                    assert(train_cameras.size() + val_cameras.size() == source_cameras.size());
                }
            } else if (base_dataset_) {
                source_cameras = base_dataset_->get_cameras();
            } else {
                return std::unexpected("No camera source available");
            }

            total_cameras_count_ = source_cameras.size();

            if (auto result = initialize_camera_loss_heatmap(source_cameras); !result) {
                return std::unexpected(result.error());
            }

            // Handle dataset split based on evaluation flag
            if (params.optimization.enable_eval) {
                if (scene_) {
                    train_dataset_ = std::make_shared<CameraDataset>(
                        train_cameras, dataset_config, CameraDataset::Split::ALL);
                    val_dataset_ = std::make_shared<CameraDataset>(
                        val_cameras, dataset_config, CameraDataset::Split::ALL);
                } else {
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
                }
                if (train_dataset_->size() == 0) {
                    return std::unexpected("Evaluation split leaves no training images. Increase Test Every or disable evaluation.");
                }
                if (val_dataset_->size() == 0) {
                    return std::unexpected("Evaluation is enabled but the validation split is empty. Decrease Test Every or disable evaluation.");
                }
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
                syncTrainingSceneTopology(scene_, splat);
            }

            // Re-initialize strategy with new parameters
            strategy_->set_training_dataset(train_dataset_);
            strategy_->initialize(get_runtime_optimization_params());
            apply_frozen_ranges_to_optimizer(splat, strategy_->get_optimizer());
            {
                std::unique_lock<std::shared_mutex> render_lock(render_mutex_);
                if (auto result = ensureModelTensorAllocatorStorage(splat, "strategy initialization"); !result) {
                    return std::unexpected(result.error());
                }
            }
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
                const int regular_iters = get_regular_iterations();
                const int sparsify_steps = get_active_sparsify_steps();
                const int first_sparsify_iter = sparsify_steps > 0 ? regular_iters + 1 : regular_iters;

                const ADMMSparsityOptimizer::Config config{
                    .sparsify_steps = sparsify_steps,
                    .init_rho = params.optimization.init_rho,
                    .prune_ratio = params.optimization.prune_ratio,
                    .update_every = UPDATE_INTERVAL,
                    .start_iteration = get_sparsity_boundary_iteration()};

                sparsity_optimizer_ = SparsityOptimizerFactory::create("admm", config);
                if (sparsity_optimizer_) {
                    LOG_INFO("Sparsity: regular={}, start={}, steps={}, total={}, prune={:.0f}%",
                             regular_iters, first_sparsify_iter, sparsify_steps, get_total_iterations(),
                             params.optimization.prune_ratio * 100);
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
                    get_total_iterations(),
                    /*update_frequency=*/100);
                LOG_DEBUG("Progress bar initialized for {} total iterations", get_total_iterations());
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
                static constexpr const char* MASK_MODE_NAMES[] = {"none", "segment", "ignore", "segment_and_ignore", "alpha_consistent"};
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
                    ctx, get_total_iterations(), is_paused_.load(), is_running_.load(), stop_requested_.load(),
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

    std::shared_ptr<Trainer::CameraLossHeatmapState> Trainer::getCameraLossHeatmap() const {
        std::lock_guard<std::mutex> lock(camera_loss_heatmap_mutex_);
        return camera_loss_heatmap_;
    }

    void Trainer::setCameraLossHeatmap(std::shared_ptr<CameraLossHeatmapState> heatmap) {
        std::lock_guard<std::mutex> lock(camera_loss_heatmap_mutex_);
        camera_loss_heatmap_ = std::move(heatmap);
    }

    std::expected<void, std::string> Trainer::ensureModelTensorAllocatorStorage(
        lfs::core::SplatData& model,
        const std::string_view reason) {
        if (!splat_tensor_allocator_) {
            return {};
        }

        if (auto result = lfs::training::migrateTrainingModelToAllocator(
                params_, model, splat_tensor_allocator_);
            !result) {
            return std::unexpected(std::format(
                "Failed to keep training model in Vulkan-external storage after {}: {}",
                reason,
                result.error()));
        }
        return {};
    }

    std::expected<Trainer::CameraMetricsSnapshot, std::string> Trainer::computeCameraMetrics(
        const lfs::core::Camera& camera,
        const bool include_ssim,
        CameraMetricsAppearanceConfig appearance) {
        if (!initialized_.load() || !strategy_) {
            return std::unexpected("trainer is not initialized");
        }

        auto inputs = load_camera_metrics_inputs(
            camera,
            getGTLoadConfigSnapshot(),
            params_.optimization,
            getActiveImageLoader());
        if (!inputs) {
            return std::unexpected(inputs.error());
        }

        auto gt_image = std::move(inputs->gt_image);
        auto mask = std::move(inputs->mask);

        if (gt_image.device() != lfs::core::Device::CUDA) {
            gt_image = gt_image.to(lfs::core::Device::CUDA);
        }
        if (mask.is_valid() && mask.device() != lfs::core::Device::CUDA) {
            mask = mask.to(lfs::core::Device::CUDA);
        }

        lfs::core::Tensor rendered;
        {
            const std::shared_lock lock(render_mutex_);
            // Exclude the non-refining optimizer writes for the metric read window
            // so the live model can't be mutated mid-render (see getModelAccessMutex).
            std::optional<std::shared_lock<std::shared_mutex>> model_read_lock;
            if (modelAccessLockEnabled()) {
                model_read_lock.emplace(model_access_mutex_);
            }
            // Run the metric render on the dedicated metrics stream (its kernels
            // and tensor ops overlap training; item() readbacks drain it). Cap
            // arena acquisition so a refining iteration holding the arena can't
            // deadlock this reader (which holds render_mutex_ shared) — on
            // timeout the rasterizer throws and the metric is skipped this call.
            const cudaStream_t reader_stream = metrics_stream_ ? metrics_stream_
                                                               : lfs::core::getCurrentCUDAStream();
            std::optional<lfs::core::CUDAStreamGuard> metrics_guard;
            if (metrics_stream_) {
                metrics_guard.emplace(metrics_stream_);
            }
            const lfs::core::RasterizerMemoryArena::ScopedBeginFrameTimeout arena_timeout(100);
            beginModelRead(reader_stream);

            auto& model = strategy_->get_model();
            auto& background = background_;

            try {
                RenderOutput output;
                if (params_.optimization.gut) {
                    output = gsplat_rasterize(
                        camera, model, background,
                        1.0f, false, GsplatRenderMode::RGB, true);
                } else {
                    output = fast_rasterize(
                        camera, model, background, params_.optimization.mip_filter);
                }

                rendered = output.image;
                if (appearance.enabled) {
                    rendered = applyPPISPForViewport(
                        rendered, camera.uid(), appearance.overrides, appearance.use_controller);
                }
                rendered = rendered.clamp(0.0f, 1.0f);
            } catch (const std::exception& e) {
                // Arena busy (refining trainer holds the frame) or render error:
                // skip this metric sample; the panel retries on its next update.
                endModelRead(reader_stream);
                return std::unexpected(std::format("metric render unavailable: {}", e.what()));
            }
            endModelRead(reader_stream);
        }

        CameraMetricsSnapshot snapshot;
        snapshot.used_mask = mask.is_valid();

        try {
            const PSNR psnr_metric(1.0f);
            snapshot.psnr = psnr_metric.compute(rendered, gt_image, mask);

            if (include_ssim) {
                SSIM ssim_metric(true);
                snapshot.ssim = ssim_metric.compute(rendered, gt_image, mask);
            }
        } catch (const std::exception& e) {
            return std::unexpected(e.what());
        }

        if (!std::isfinite(snapshot.psnr) ||
            (snapshot.ssim.has_value() && !std::isfinite(*snapshot.ssim))) {
            return std::unexpected("metric computation produced non-finite values");
        }

        return snapshot;
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

        if (metrics_stream_) {
            cudaStreamSynchronize(metrics_stream_);
            lfs::core::CudaMemoryPool::instance().release_stream(metrics_stream_);
            cudaStreamDestroy(metrics_stream_);
            metrics_stream_ = nullptr;
        }

        if (training_stream_) {
            cudaStreamSynchronize(training_stream_);
            destroySyncPrimitives();
            lfs::core::CudaMemoryPool::instance().release_stream(training_stream_);
            cudaStreamDestroy(training_stream_);
            training_stream_ = nullptr;
        }

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
        setCameraLossHeatmap(nullptr);

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

    TrainingProgress::Phase Trainer::get_progress_phase(
        const int iter,
        const bool in_controller_phase) const {

        if (in_controller_phase) {
            return TrainingProgress::Phase::Controller;
        }

        const bool in_sparsification = get_active_sparsify_steps() > 0 &&
                                       iter > get_sparsity_boundary_iteration();
        if (in_sparsification) {
            return TrainingProgress::Phase::Sparse;
        }

        if (strategy_ && strategy_->is_refining(iter)) {
            return TrainingProgress::Phase::Refine;
        }

        return TrainingProgress::Phase::Train;
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
                progress_->resume(
                    iter,
                    current_loss_.load(),
                    static_cast<int>(strategy_->get_model().size()),
                    get_progress_phase(iter));
            }
            LOG_INFO("Training resumed at iteration {}", iter);
        }

        if (save_requested_.exchange(false)) {
            LOG_INFO("Saving checkpoint and PLY at iteration {}...", iter);
            save_ply(params_.dataset.output_path, "", iter, /*join=*/false, /*save_checkpoint=*/false);
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
            save_ply(params_.dataset.output_path, params_.dataset.output_name, iter, /*join=*/true);
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

        const float w = inv_weight_piecewise(iter, get_total_iterations());
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
            LFS_VRAM_SCOPE("train.step");
            LOG_VRAM_DIFF("train.step");
            if (live_vram_profiler_enabled()) {
                auto& profiler = lfs::diagnostics::VramProfiler::instance();
                profiler.beginIteration(iter);
                profiler.sampleCudaMemory();
            }

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

            // Gate this step's in-place parameter writes behind in-flight model
            // reads (viewer packs, metric renders) — GPU-side waits, ~free once
            // the reads have retired.
            waitForModelReaders();

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
                    ctx, get_total_iterations(), is_paused_.load(), is_running_.load(), stop_requested_.load(),
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

            lfs::core::Tensor* bg_ptr = nullptr;
            {
                nvtxRangePush("background_for_step");
                LFS_VRAM_SCOPE("train.background");
                LOG_VRAM_DIFF("train.background");
                bg_ptr = &background_for_step(iter);
                nvtxRangePop();
            }
            lfs::core::Tensor& bg = *bg_ptr;

            lfs::core::Tensor bg_image;
            if (params_.optimization.bg_mode == lfs::core::param::BackgroundMode::Image) {
                LFS_VRAM_SCOPE("train.background_image");
                LOG_VRAM_DIFF("train.background_image");
                bg_image = get_background_image_for_camera(cam->image_width(), cam->image_height());
            } else if (params_.optimization.bg_mode == lfs::core::param::BackgroundMode::Random) {
                LFS_VRAM_SCOPE("train.random_background");
                LOG_VRAM_DIFF("train.random_background");
                bg_image = get_random_background_for_camera(cam->image_width(), cam->image_height(), iter);
            }

            const bool fastgs_path = !params_.optimization.gut;

            if (!loss_accumulator_.is_valid()) {
                loss_accumulator_ = core::Tensor::zeros({1}, core::Device::CUDA);
            } else {
                loss_accumulator_.zero_();
            }
            if (live_vram_profiler_enabled() && strategy_) {
                record_splat_vram_breakdown(strategy_->get_model());
                record_optimizer_vram_breakdown(strategy_->get_optimizer());
                record_vram_tensor("train.persistent", "loss_accumulator", loss_accumulator_);
                record_vram_tensor("train.persistent", "pipelined_mask", pipelined_mask_);
                record_vram_tensor("train.persistent", "pipelined_depth", pipelined_depth_);
                record_vram_tensor("train.persistent", "background", background_);
                record_vram_tensor("train.persistent", "background_mix_buffer", bg_mix_buffer_);
                record_vram_tensor("train.persistent", "background_image_base", bg_image_base_);
                record_vram_current("train.persistent", "background_image_cache", bg_image_cache_bytes(bg_image_cache_));
                record_pipeline_vram_breakdown(getActiveImageLoader());
            }
            auto& loss_tensor_gpu = loss_accumulator_;
            RenderOutput r_output;
            r_output.camera = cam;
            r_output.target_image = gt_image;
            int tiles_processed = 0;
            const bool in_sparsification = get_active_sparsify_steps() > 0 &&
                                           iter > get_sparsity_boundary_iteration();

            // Determine controller phase before render (does not depend on render results)
            const bool known_ppisp_camera = ppisp_ && ppisp_->is_known_camera(cam->camera_id());
            const int ppisp_cam_idx = known_ppisp_camera ? ppisp_->camera_index(cam->camera_id()) : -1;
            const int ppisp_activation_step = params_.optimization.resolved_ppisp_controller_activation_step(get_total_iterations());
            const bool ppisp_frozen = is_ppisp_frozen();
            const bool in_controller_phase = ppisp_controller_pool_ && known_ppisp_camera &&
                                             params_.optimization.ppisp_use_controller &&
                                             !ppisp_frozen &&
                                             params_.optimization.ppisp_freeze_gaussians_on_distill &&
                                             iter >= ppisp_activation_step &&
                                             ppisp_cam_idx >= 0 &&
                                             ppisp_cam_idx < ppisp_controller_pool_->num_cameras();
            const bool freeze_gaussians_this_iter = ppisp_controller_pool_ &&
                                                    params_.optimization.ppisp_use_controller &&
                                                    params_.optimization.ppisp_freeze_gaussians_on_distill &&
                                                    iter >= ppisp_activation_step;
            const bool use_pixel_error_densification =
                (params_.optimization.strategy == "mcmc") ||
                (params_.optimization.strategy == "igs+") ||
                (core::param::is_mrnf_strategy(params_.optimization.strategy) &&
                 params_.optimization.use_error_map);
            const bool use_ssim_error = use_pixel_error_densification;
            DensificationType densification_type = DensificationType::None;
            if (params_.optimization.strategy == "mcmc")
                densification_type = DensificationType::MCMC;
            else if (core::param::is_mrnf_strategy(params_.optimization.strategy))
                densification_type = DensificationType::MRNF;
            const bool update_gaussians_this_iter = !freeze_gaussians_this_iter;
            const bool run_fastgs_gaussian_backward =
                fastgs_path &&
                update_gaussians_this_iter;

            bool fastgs_strategy_hooks_at_start = false;
            if (fastgs_path && !in_sparsification) {
                LFS_VRAM_SCOPE("train.strategy.fastgs_pre_step");
                LOG_VRAM_DIFF("train.strategy.fastgs_pre_step");
                strategy_->pre_step(iter, r_output);

                // Only exclude the render thread when this step actually mutates model
                // topology (grow/prune → tensor reallocation, moving pointers). In-place
                // parameter updates are ordered against the viewer's reads by the
                // CUDA↔Vulkan interop semaphore, so render_mutex_ is redundant there.
                // Holding it every iteration deadlocks at startup: the render holds a
                // read-lock across its synchronous GPU readback while waiting for the
                // first step's output, which this write-lock — taken before that step —
                // blocks. See trainer.cpp step() lock below; both must be gated.
                std::unique_lock<std::shared_mutex> lock(render_mutex_, std::defer_lock);
                if (strategy_->is_refining(iter)) {
                    lock.lock();
                }
                // Drain in-flight reader events immediately before post_backward's
                // in-place writes — not only at the loop top — so the trainer stream
                // is ordered after any read that began mid-step, collapsing the
                // reader↔writer overlap to a sub-microsecond CPU window. The
                // exclusive lock (when refining) additionally bars new readers.
                waitForModelReaders();
                auto& model = strategy_->get_model();
                const size_t model_size_before = static_cast<size_t>(model.size());
                strategy_->post_backward(iter, r_output);
                fastgs_strategy_hooks_at_start = true;

                if (sparsity_optimizer_ &&
                    sparsity_optimizer_->is_initialized() &&
                    static_cast<size_t>(model.size()) != model_size_before) {
                    LOG_WARN("Sparsity: resetting ADMM state after topology change at iter {} ({} -> {})",
                             iter, model_size_before, model.size());
                    sparsity_optimizer_->reset();
                }
                if (static_cast<size_t>(model.size()) != model_size_before) {
                    syncTrainingSceneTopology(scene_, model);
                }
                if (auto result = ensureModelTensorAllocatorStorage(model, "fastgs strategy post_backward"); !result) {
                    return std::unexpected(result.error());
                }
                // Readers can re-acquire the shared lock the moment the
                // exclusive lock drops — re-mark consistency before that.
                if (lock.owns_lock()) {
                    recordParamsReady();
                }
            }

            FastGSFusedExtraGradients fused_extra_gradients;
            lfs::core::Tensor fused_scale_reg_loss_gpu;
            lfs::core::Tensor fused_opacity_reg_loss_gpu;
            lfs::core::Tensor sparsity_loss_gpu;
            if (fastgs_path) {
                LFS_VRAM_SCOPE("train.regularizers.fastgs_forward_only");
                LOG_VRAM_DIFF("train.regularizers.fastgs_forward_only");
                auto& model = strategy_->get_model();
                if (run_fastgs_gaussian_backward) {
                    fused_extra_gradients.scale_reg_weight = params_.optimization.scale_reg;
                    fused_extra_gradients.opacity_reg_weight = params_.optimization.opacity_reg;
                }

                if (params_.optimization.scale_reg > 0.0f) {
                    auto scale_loss_result = lfs::training::losses::ScaleRegularization::forward_loss_only(
                        model.scaling_raw(),
                        {.weight = params_.optimization.scale_reg});
                    if (!scale_loss_result) {
                        return std::unexpected(scale_loss_result.error());
                    }
                    fused_scale_reg_loss_gpu = *scale_loss_result;
                }
                if (params_.optimization.opacity_reg > 0.0f) {
                    auto opacity_loss_result = lfs::training::losses::OpacityRegularization::forward_loss_only(
                        model.opacity_raw(),
                        {.weight = params_.optimization.opacity_reg});
                    if (!opacity_loss_result) {
                        return std::unexpected(opacity_loss_result.error());
                    }
                    fused_opacity_reg_loss_gpu = *opacity_loss_result;
                }
                if (run_fastgs_gaussian_backward &&
                    sparsity_optimizer_ && sparsity_optimizer_->should_apply_loss(iter)) {
                    auto sparsity_result = compute_sparsity_loss_forward(iter, model);
                    if (!sparsity_result) {
                        return std::unexpected(sparsity_result.error());
                    }
                    auto& [loss_tensor, ctx] = *sparsity_result;
                    sparsity_loss_gpu = std::move(loss_tensor);
                    fused_extra_gradients.sparsity_opa_sigmoid = ctx.opa_sigmoid_ptr;
                    fused_extra_gradients.sparsity_z = ctx.z_ptr;
                    fused_extra_gradients.sparsity_u = ctx.u_ptr;
                    fused_extra_gradients.sparsity_n = static_cast<int>(ctx.n);
                    fused_extra_gradients.sparsity_rho = ctx.rho;
                    fused_extra_gradients.sparsity_grad_loss = 1.0f;
                }
            }

            {
                nvtxRangePush("rasterize");

                lfs::core::Tensor gt_tile = gt_image;
                lfs::core::Tensor bg_tile;
                if (bg_image.is_valid() && !bg_image.is_empty()) {
                    bg_tile = bg_image;
                }

                // Render the tile
                nvtxRangePush("rasterize_forward");

                // Storage for render output (used by both paths)
                RenderOutput output;
                std::optional<FastRasterizeContext> fast_ctx;
                std::optional<GsplatRasterizeContext> gsplat_ctx;

                {
                    LFS_VRAM_SCOPE("train.rasterize_forward");
                    LOG_VRAM_DIFF("train.rasterize_forward");
                    if (params_.optimization.gut) {
                        auto rasterize_result = gsplat_rasterize_forward(
                            *cam, strategy_->get_model(), bg,
                            0, 0, 0, 0,
                            1.0f, false, GsplatRenderMode::RGB, true, bg_tile);

                        if (!rasterize_result) {
                            nvtxRangePop(); // rasterize_forward
                            nvtxRangePop(); // tile
                            return std::unexpected(rasterize_result.error());
                        }

                        output = std::move(rasterize_result->first);
                        gsplat_ctx.emplace(std::move(rasterize_result->second));
                    } else {
                        auto rasterize_result = fast_rasterize_forward(
                            *cam, strategy_->get_model(), bg,
                            0, 0, 0, 0,
                            params_.optimization.mip_filter, bg_tile);

                        // Check for OOM error
                        if (!rasterize_result) {
                            const std::string& error = rasterize_result.error();
                            if (error.find("OUT_OF_MEMORY") != std::string::npos) {
                                nvtxRangePop(); // rasterize_forward
                                nvtxRangePop(); // rasterize

                                LOG_ERROR("OUT OF MEMORY in 3DGS/FastGS training.");
                                LOG_ERROR("Arena error: {}", error);
                                return std::unexpected(error);
                            }
                            // Non-OOM error - propagate
                            nvtxRangePop();
                            nvtxRangePop();
                            return std::unexpected(error);
                        }

                        output = std::move(rasterize_result->first);
                        fast_ctx.emplace(std::move(rasterize_result->second));

                        if (fast_ctx->forward_ctx.n_instances == 0) {
                            fast_ctx->release_forward_context();
                            nvtxRangePop();
                            nvtxRangePop();
                            LOG_DEBUG("Skipping iteration {} - no visible primitives", iter);
                            return iter < get_total_iterations() && !stop_requested_.load() && !stop_token.stop_requested()
                                       ? StepResult::Continue
                                       : StepResult::Stop;
                        }
                    }
                }

                r_output = output; // Save last tile for densification
                r_output.camera = cam;
                r_output.target_image = gt_image;
                nvtxRangePop();

                bool tile_context_cleaned = false;
                auto cleanup_tile_context = [&]() {
                    if (tile_context_cleaned) {
                        return;
                    }
                    tile_context_cleaned = true;

                    if (fast_ctx) {
                        fast_ctx->release_forward_context();
                        fast_ctx.reset();
                    } else if (gsplat_ctx) {
                        auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
                        if (gsplat_ctx->isect_ids_ptr != nullptr) {
                            cudaFree(gsplat_ctx->isect_ids_ptr);
                            gsplat_ctx->isect_ids_ptr = nullptr;
                        }
                        if (gsplat_ctx->flatten_ids_ptr != nullptr) {
                            cudaFree(gsplat_ctx->flatten_ids_ptr);
                            gsplat_ctx->flatten_ids_ptr = nullptr;
                        }
                        arena.end_frame(gsplat_ctx->frame_id, lfs::core::getCurrentCUDAStream());
                        gsplat_ctx.reset();
                    }
                };

                if (in_controller_phase) {
                    // Controller phase: forward through ISP with controller params, photometric loss,
                    // backward only through controller (base params frozen)
                    nvtxRangePush("controller_phase");
                    LFS_VRAM_SCOPE("train.controller_phase");
                    LOG_VRAM_DIFF("train.controller_phase");
                    auto tile_context_guard = makeScopeGuard(cleanup_tile_context);

                    lfs::core::Tensor corrected_image = output.image;
                    if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                        LFS_VRAM_SCOPE("train.bilateral_grid.forward");
                        LOG_VRAM_DIFF("train.bilateral_grid.forward");
                        corrected_image = bilateral_grid_->apply(output.image, cam->uid());
                    }
                    auto ppisp_input = corrected_image;

                    lfs::core::Tensor pred;
                    {
                        LFS_VRAM_SCOPE("train.ppisp_controller.forward");
                        LOG_VRAM_DIFF("train.ppisp_controller.forward");
                        pred = ppisp_controller_pool_->predict(ppisp_cam_idx, corrected_image.unsqueeze(0), 1.0f);
                        corrected_image = ppisp_->apply_with_controller_params(corrected_image, pred, ppisp_cam_idx);
                    }
                    const lfs::core::Tensor raw_loss_input = output.image;

                    // Photometric loss
                    nvtxRangePush("compute_photometric_loss");
                    lfs::core::Tensor tile_loss;
                    lfs::core::Tensor tile_grad;

                    {
                        LFS_VRAM_SCOPE("train.photometric_loss");
                        LOG_VRAM_DIFF("train.photometric_loss");
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
                                    params_.optimization.mask_threshold,
                                    params_.optimization.mask_mode != lfs::core::param::MaskMode::SegmentAndIgnore);
                            }

                            lfs::core::Tensor mask_tile = mask;

                            auto result = compute_photometric_loss_with_mask(
                                corrected_image, gt_tile, mask_tile, output.alpha, params_.optimization, raw_loss_input);
                            if (!result) {
                                nvtxRangePop();
                                nvtxRangePop();
                                nvtxRangePop();
                                return std::unexpected(result.error());
                            }
                            tile_loss = result->loss;
                            tile_grad = result->grad_corrected;
                        } else {
                            auto result = compute_photometric_loss_with_gradient(
                                corrected_image, gt_tile, params_.optimization, raw_loss_input);
                            if (!result) {
                                nvtxRangePop();
                                nvtxRangePop();
                                nvtxRangePop();
                                return std::unexpected(result.error());
                            }
                            tile_loss = result->loss;
                            tile_grad = result->grad_corrected;
                        }
                    }

                    loss_tensor_gpu = loss_tensor_gpu + tile_loss;
                    tiles_processed++;
                    nvtxRangePop(); // compute_photometric_loss
                    if (live_vram_profiler_enabled()) {
                        record_vram_tensor("train.losses", "controller.tile_loss", tile_loss);
                        record_vram_tensor("train.losses", "controller.tile_grad", tile_grad);
                        record_vram_tensor("train.appearance", "ppisp_controller.prediction", pred);
                        record_vram_current("train.losses", "photometric.workspaces",
                                            photometric_workspace_bytes(photometric_loss_));
                        record_vram_current("train.losses", "masked_fused.workspace",
                                            masked_fused_workspace_bytes(masked_fused_workspace_));
                        record_vram_current("train.losses", "decoupled_fused.workspace",
                                            decoupled_fused_workspace_bytes(decoupled_fused_workspace_));
                        record_vram_current("train.losses", "masked_decoupled_fused.workspace",
                                            masked_decoupled_fused_workspace_bytes(masked_decoupled_fused_workspace_));
                    }

                    // ISP backward for controller params
                    {
                        LFS_VRAM_SCOPE("train.ppisp_controller.backward");
                        LOG_VRAM_DIFF("train.ppisp_controller.backward");
                        auto ctrl_grad = ppisp_->backward_with_controller_params(ppisp_input, tile_grad, pred, ppisp_cam_idx);
                        ppisp_controller_pool_->backward(ppisp_cam_idx, ctrl_grad);
                    }

                    nvtxRangePop(); // controller_phase
                } else {
                    // Normal phase: full forward + backward through all components
                    auto tile_context_guard = makeScopeGuard(cleanup_tile_context);

                    lfs::core::Tensor corrected_image = output.image;
                    if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                        nvtxRangePush("bilateral_grid_forward");
                        LFS_VRAM_SCOPE("train.bilateral_grid.forward");
                        LOG_VRAM_DIFF("train.bilateral_grid.forward");
                        corrected_image = bilateral_grid_->apply(output.image, cam->uid());
                        nvtxRangePop();
                    }

                    lfs::core::Tensor ppisp_input;
                    if (ppisp_ && params_.optimization.use_ppisp) {
                        nvtxRangePush("ppisp_forward");
                        LFS_VRAM_SCOPE("train.ppisp.forward");
                        LOG_VRAM_DIFF("train.ppisp.forward");
                        ppisp_input = corrected_image;
                        corrected_image = ppisp_->apply(ppisp_input, cam->camera_id(), cam->uid());
                        nvtxRangePop();
                    }
                    // For decoupled D-SSIM, the active appearance model provides the corrected image
                    // for the L1/luminance terms, while contrast/structure still use the raw render.
                    const lfs::core::Tensor raw_loss_input =
                        ((bilateral_grid_ && params_.optimization.use_bilateral_grid) ||
                         (ppisp_ && params_.optimization.use_ppisp))
                            ? output.image
                            : lfs::core::Tensor{};

                    // Final tonemapping: clamp to [0, 1] for loss computation.
                    // This is redundant when PPISP is active (CRF already clamps), but ensures
                    // valid output range for bilateral grids and raw rasterizer output.
                    corrected_image.clamp_(0.0f, 1.0f);

                    nvtxRangePush("compute_photometric_loss");
                    lfs::core::Tensor tile_loss;
                    lfs::core::Tensor tile_grad;
                    lfs::core::Tensor tile_grad_raw;
                    lfs::core::Tensor tile_grad_alpha;
                    lfs::core::Tensor tile_grad_depth;
                    lfs::core::Tensor tile_error_map;
                    lfs::core::Tensor mask_tile;

                    // 1) Compute photometric loss (populates ssim_map in workspace)
                    const bool use_mask = params_.optimization.mask_mode != lfs::core::param::MaskMode::None &&
                                          (cam->has_mask() || (params_.optimization.use_alpha_as_mask && cam->has_alpha()));
                    const bool used_masked_fused =
                        use_mask &&
                        (params_.optimization.mask_mode == lfs::core::param::MaskMode::Segment ||
                         params_.optimization.mask_mode == lfs::core::param::MaskMode::Ignore ||
                         params_.optimization.mask_mode == lfs::core::param::MaskMode::SegmentAndIgnore) &&
                        params_.optimization.lambda_dssim > 0.0f;
                    {
                        LFS_VRAM_SCOPE("train.photometric_loss");
                        LOG_VRAM_DIFF("train.photometric_loss");
                        if (use_mask) {
                            lfs::core::Tensor mask;
                            if (pipelined_mask_.is_valid() && pipelined_mask_.numel() > 0) {
                                mask = pipelined_mask_;
                            } else {
                                mask = cam->load_and_get_mask(
                                    params_.dataset.resize_factor,
                                    params_.dataset.max_width,
                                    params_.optimization.invert_masks,
                                    params_.optimization.mask_threshold,
                                    params_.optimization.mask_mode != lfs::core::param::MaskMode::SegmentAndIgnore);
                            }

                            mask_tile = mask;

                            auto result = compute_photometric_loss_with_mask(
                                corrected_image, gt_tile, mask_tile, output.alpha, params_.optimization, raw_loss_input);
                            if (!result) {
                                nvtxRangePop();
                                nvtxRangePop();
                                return std::unexpected(result.error());
                            }
                            tile_loss = result->loss;
                            tile_grad = result->grad_corrected;
                            tile_grad_raw = result->grad_raw;
                            tile_grad_alpha = result->grad_alpha;
                        } else {
                            auto result = compute_photometric_loss_with_gradient(
                                corrected_image, gt_tile, params_.optimization, raw_loss_input);
                            if (!result) {
                                nvtxRangePop();
                                nvtxRangePop();
                                return std::unexpected(result.error());
                            }
                            tile_loss = result->loss;
                            tile_grad = result->grad_corrected;
                            tile_grad_raw = result->grad_raw;
                        }
                    }

                    if (run_fastgs_gaussian_backward &&
                        params_.optimization.use_depth_loss &&
                        params_.optimization.depth_loss_weight > 0.0f &&
                        output.depth.is_valid() &&
                        output.depth.numel() > 0 &&
                        output.alpha.is_valid() &&
                        output.alpha.numel() > 0) {
                        LFS_VRAM_SCOPE("train.depth_loss");
                        LOG_VRAM_DIFF("train.depth_loss");

                        lfs::core::Tensor target_depth;
                        if (pipelined_depth_.is_valid() && pipelined_depth_.numel() > 0) {
                            target_depth = pipelined_depth_;
                        } else if (cam->has_depth()) {
                            target_depth = cam->load_and_get_depth(
                                params_.dataset.resize_factor,
                                params_.dataset.max_width);
                        }

                        if (target_depth.is_valid() && target_depth.numel() > 0) {
                            if (target_depth.ndim() == 3 && target_depth.shape()[0] == 1) {
                                target_depth = target_depth.squeeze(0);
                            }
                            if (target_depth.device() != lfs::core::Device::CUDA) {
                                target_depth = target_depth.cuda();
                            }
                            if (!target_depth.is_contiguous()) {
                                target_depth = target_depth.contiguous();
                            }

                            lfs::core::Tensor rendered_depth = output.depth;
                            if (rendered_depth.ndim() == 3 && rendered_depth.shape()[0] == 1) {
                                rendered_depth = rendered_depth.squeeze(0);
                            }
                            if (!rendered_depth.is_contiguous()) {
                                rendered_depth = rendered_depth.contiguous();
                            }

                            lfs::core::Tensor rendered_alpha = output.alpha;
                            if (rendered_alpha.ndim() == 3 && rendered_alpha.shape()[0] == 1) {
                                rendered_alpha = rendered_alpha.squeeze(0);
                            }
                            if (!rendered_alpha.is_contiguous()) {
                                rendered_alpha = rendered_alpha.contiguous();
                            }

                            const bool depth_shape_matches =
                                target_depth.ndim() == 2 &&
                                rendered_depth.ndim() == 2 &&
                                rendered_alpha.ndim() == 2 &&
                                target_depth.shape()[0] == rendered_depth.shape()[0] &&
                                target_depth.shape()[1] == rendered_depth.shape()[1] &&
                                target_depth.shape()[0] == rendered_alpha.shape()[0] &&
                                target_depth.shape()[1] == rendered_alpha.shape()[1];

                            if (depth_shape_matches) {
                                const size_t num_depth_pixels = rendered_depth.numel();
                                const size_t depth_partials =
                                    lfs::training::kernels::depth_loss_partial_count(num_depth_pixels);
                                const cudaStream_t depth_stream = rendered_depth.stream();
                                const auto depth_loss_mode =
                                    depth_loss_mode_from_name(params_.optimization.depth_loss_mode);

                                if (!depth_loss_scalar_.is_valid()) {
                                    depth_loss_scalar_ = lfs::core::Tensor::zeros({1}, lfs::core::Device::CUDA);
                                }
                                depth_loss_scalar_.set_stream(depth_stream);
                                if (!depth_loss_grad_.is_valid() ||
                                    depth_loss_grad_.shape() != rendered_depth.shape()) {
                                    depth_loss_grad_ = lfs::core::Tensor::empty(rendered_depth.shape(), lfs::core::Device::CUDA);
                                }
                                depth_loss_grad_.set_stream(depth_stream);
                                if (!depth_loss_partials_.is_valid() ||
                                    depth_loss_partials_.shape()[0] != depth_partials) {
                                    depth_loss_partials_ = lfs::core::Tensor::empty({depth_partials}, lfs::core::Device::CUDA);
                                }
                                depth_loss_partials_.set_stream(depth_stream);

                                lfs::training::kernels::launch_depth_loss(
                                    rendered_depth.ptr<float>(),
                                    rendered_alpha.ptr<float>(),
                                    target_depth.ptr<float>(),
                                    depth_loss_grad_.ptr<float>(),
                                    depth_loss_scalar_.ptr<float>(),
                                    depth_loss_partials_.ptr<float>(),
                                    num_depth_pixels,
                                    params_.optimization.depth_loss_weight,
                                    depth_loss_mode,
                                    depth_stream);

                                static const bool depth_loss_diag = env_flag_enabled("LFS_DEPTH_LOSS_DIAG");
                                static const int depth_loss_diag_interval =
                                    env_int_or_default("LFS_DEPTH_LOSS_DIAG_INTERVAL", 50);
                                if (depth_loss_diag && (iter == 1 || iter % depth_loss_diag_interval == 0)) {
                                    const auto depth_grad_abs = depth_loss_grad_.abs();
                                    const float valid_count = depth_loss_partials_.slice(0, 0, 1).item<float>();
                                    const float scale_or_norm = depth_loss_partials_.slice(0, 3, 4).item<float>();
                                    const float depth_corr = depth_loss_partials_.slice(0, 4, 5).item<float>();
                                    const float valid_fraction = num_depth_pixels > 0
                                                                     ? valid_count / static_cast<float>(num_depth_pixels)
                                                                     : 0.0f;
                                    LOG_INFO("[DEPTH_LOSS] iter={} mode={} camera='{}' loss={:.6f} corr={:.6f} scale_or_norm={:.6f} valid_pixels={:.0f} valid_fraction={:.3f} target_depth_min={:.6f} target_depth_max={:.6f} target_depth_mean={:.6f} depth_accum_min={:.6f} depth_accum_max={:.6f} depth_accum_mean={:.6f} alpha_accum_min={:.6f} alpha_accum_max={:.6f} alpha_accum_mean={:.6f} grad_depth_abs_max={:.6e} grad_depth_abs_mean={:.6e}",
                                             iter,
                                             depth_loss_mode_name(depth_loss_mode),
                                             cam->image_name(),
                                             depth_loss_scalar_.item<float>(),
                                             depth_corr,
                                             scale_or_norm,
                                             valid_count,
                                             valid_fraction,
                                             target_depth.min().item<float>(),
                                             target_depth.max().item<float>(),
                                             target_depth.mean().item<float>(),
                                             rendered_depth.min().item<float>(),
                                             rendered_depth.max().item<float>(),
                                             rendered_depth.mean().item<float>(),
                                             rendered_alpha.min().item<float>(),
                                             rendered_alpha.max().item<float>(),
                                             rendered_alpha.mean().item<float>(),
                                             depth_grad_abs.max().item<float>(),
                                             depth_grad_abs.mean().item<float>());
                                }

                                tile_grad_depth = depth_loss_grad_;
                                tile_loss = tile_loss + depth_loss_scalar_;
                            } else {
                                LOG_WARN("Skipping depth loss for '{}': rendered depth shape and target depth shape differ",
                                         cam->image_name());
                            }
                        }
                    }

                    // 2) Extract error map from workspace's ssim_map
                    if (use_pixel_error_densification) {
                        LFS_VRAM_SCOPE("train.densification_error_map");
                        LOG_VRAM_DIFF("train.densification_error_map");
                        if (use_ssim_error && params_.optimization.lambda_dssim > 0.0f) {
                            lfs::core::Tensor ssim_map;
                            if (used_masked_fused && raw_loss_input.is_valid()) {
                                ssim_map = masked_decoupled_fused_workspace_.ssim_map;
                            } else if (used_masked_fused) {
                                ssim_map = masked_fused_workspace_.ssim_map;
                            } else if (raw_loss_input.is_valid()) {
                                ssim_map = decoupled_fused_workspace_.ssim_map;
                            } else if (params_.optimization.lambda_dssim < 1.0f) {
                                ssim_map = photometric_loss_.fused_workspace().ssim_map;
                            } else {
                                ssim_map = photometric_loss_.ssim_workspace().ssim_map;
                            }
                            if (ssim_map.shape()[0] == 1 && ssim_map.shape()[1] == 1 &&
                                ssim_map.is_contiguous()) {
                                const size_t H = ssim_map.shape()[2];
                                const size_t W = ssim_map.shape()[3];
                                tile_error_map = ssim_map.reshape({static_cast<int>(H), static_cast<int>(W)});
                                lfs::training::kernels::launch_ssim_to_error_map(ssim_map, tile_error_map);
                            } else {
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
                            const auto gt_for_error = gt_tile.dtype() == lfs::core::DataType::UInt8
                                                          ? gt_tile.to(lfs::core::DataType::Float32) / 255.0f
                                                          : gt_tile;
                            const lfs::core::Tensor abs_diff = (corrected_image - gt_for_error).abs();
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
                            params_.optimization.mask_mode == lfs::core::param::MaskMode::SegmentAndIgnore) {
                            const auto mask_for_error = mask_tile.gt(250).to(lfs::core::DataType::Float32);
                            tile_error_map.mul_(mask_for_error);
                        }

                        if (use_mask &&
                            (params_.optimization.mask_mode == lfs::core::param::MaskMode::Segment ||
                             params_.optimization.mask_mode == lfs::core::param::MaskMode::Ignore)) {
                            const auto mask_for_error =
                                (mask_tile.dtype() == lfs::core::DataType::UInt8 ||
                                 mask_tile.dtype() == lfs::core::DataType::Bool)
                                    ? mask_tile.to(lfs::core::DataType::Float32)
                                    : mask_tile;
                            tile_error_map.mul_(mask_for_error);
                        }
                    }

                    if (tile_error_map.is_valid() && core::param::is_mrnf_strategy(params_.optimization.strategy)) {
                        LFS_VRAM_SCOPE("train.densification_error_map");
                        LOG_VRAM_DIFF("train.densification_error_map.normalize");
                        const auto map_mean = tile_error_map.mean();
                        lfs::training::kernels::launch_normalize_by_device_scalar(
                            tile_error_map.ptr<float>(), tile_error_map.numel(),
                            map_mean.ptr<float>(), 1e-6f);
                    }

                    if (live_vram_profiler_enabled()) {
                        if (fast_ctx) {
                            record_fastgs_vram_breakdown(*fast_ctx,
                                                         output,
                                                         gt_tile,
                                                         bg_tile,
                                                         tile_error_map,
                                                         run_fastgs_gaussian_backward,
                                                         static_cast<std::size_t>(strategy_->get_model().size()));
                        } else if (gsplat_ctx) {
                            record_gsplat_vram_breakdown(*gsplat_ctx, output, gt_tile, bg_tile, tile_error_map);
                        }
                        record_vram_tensor("train.losses", "tile_loss", tile_loss);
                        record_vram_tensor("train.losses", "tile_grad_corrected", tile_grad);
                        record_vram_tensor("train.losses", "tile_grad_raw", tile_grad_raw);
                        record_vram_tensor("train.losses", "tile_grad_alpha", tile_grad_alpha);
                        record_vram_tensor("train.losses", "densification_error_map.live", tile_error_map);
                        record_vram_current("train.losses", "photometric.workspaces",
                                            photometric_workspace_bytes(photometric_loss_));
                        record_vram_current("train.losses", "masked_fused.workspace",
                                            masked_fused_workspace_bytes(masked_fused_workspace_));
                        record_vram_current("train.losses", "decoupled_fused.workspace",
                                            decoupled_fused_workspace_bytes(decoupled_fused_workspace_));
                        record_vram_current("train.losses", "masked_decoupled_fused.workspace",
                                            masked_decoupled_fused_workspace_bytes(masked_decoupled_fused_workspace_));
                        record_vram_current("train.losses", "densification_ssim.workspace",
                                            ssim_map_workspace_bytes(densification_ssim_workspace_));
                        record_vram_tensor("train.losses", "densification_error_map.buffer", densification_error_map_);
                        record_vram_tensor("train.losses", "edge_map_buffer", edge_map_buffer_);
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
                        if (fast_ctx->forward_ctx.sorted_primitive_indices_size > 0) {
                            fastgs_entries.emplace_back("fastgs.sorted_primitive_indices_live",
                                                        fast_ctx->forward_ctx.sorted_primitive_indices_size);
                        }
                        fastgs_entries.emplace_back("fastgs.grad_mean2d_helper", grad_mean2d_helper_bytes);
                        fastgs_entries.emplace_back("fastgs.grad_conic_helper", grad_conic_helper_bytes);
                        if (run_fastgs_gaussian_backward) {
                            fastgs_entries.emplace_back("fastgs.fused_grad_opacity_helper", n_primitives * sizeof(float));
                            fastgs_entries.emplace_back("fastgs.fused_grad_color_helper", n_primitives * 3 * sizeof(float));
                        }
                        fastgs_entries.emplace_back("render.output_image", tensor_reserved_bytes(output.image));
                        fastgs_entries.emplace_back("render.output_alpha", tensor_reserved_bytes(output.alpha));
                        fastgs_entries.emplace_back("render.output_depth", tensor_reserved_bytes(output.depth));
                        fastgs_entries.emplace_back("train.gt_tile", tensor_reserved_bytes(gt_tile));
                        if (bg_image.is_valid() && !bg_image.is_empty()) {
                            fastgs_entries.emplace_back("train.background_image", tensor_reserved_bytes(bg_image));
                        }
                        if (tile_error_map.is_valid()) {
                            fastgs_entries.emplace_back("train.tile_error_map", tensor_reserved_bytes(tile_error_map));
                        }
                        log_entry_bytes("first_fastgs_tile_live", fastgs_entries);
                        if (fast_ctx->forward_ctx.per_instance_sort_total_size > 0) {
                            LOG_INFO("[MEM] fastgs_sort_transient total_forward={:.2f} MiB, released_after_forward={:.2f} MiB, sorted_indices_live={:.2f} MiB",
                                     bytes_to_mib(fast_ctx->forward_ctx.per_instance_sort_total_size),
                                     bytes_to_mib(fast_ctx->forward_ctx.per_instance_sort_scratch_size),
                                     bytes_to_mib(fast_ctx->forward_ctx.sorted_primitive_indices_size));
                        }

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
                        add_tensor_entry(trainer_entries, "trainer.pipelined_depth", pipelined_depth_);
                        add_tensor_entry(trainer_entries, "trainer.depth_loss_grad", depth_loss_grad_);
                        add_tensor_entry(trainer_entries, "trainer.depth_loss_partials", depth_loss_partials_);
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

                        LOG_INFO("[MEM] fastgs counts instances={}, image={}x{}",
                                 fast_ctx->forward_ctx.n_instances,
                                 output.width,
                                 output.height);
                        memory_breakdown_logged_first_raster_ = true;
                    }

                    loss_tensor_gpu = loss_tensor_gpu + tile_loss;
                    tiles_processed++;
                    nvtxRangePop();

                    lfs::core::Tensor raster_grad = tile_grad;
                    if (ppisp_ && params_.optimization.use_ppisp) {
                        nvtxRangePush("ppisp_backward");
                        LFS_VRAM_SCOPE("train.ppisp.backward");
                        LOG_VRAM_DIFF("train.ppisp.backward");
                        raster_grad = ppisp_->backward(ppisp_input, raster_grad, cam->camera_id(), cam->uid());
                        if (ppisp_frozen) {
                            ppisp_->zero_grad();
                        }
                        nvtxRangePop();
                    }

                    if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                        nvtxRangePush("bilateral_grid_backward");
                        LFS_VRAM_SCOPE("train.bilateral_grid.backward");
                        LOG_VRAM_DIFF("train.bilateral_grid.backward");
                        raster_grad = bilateral_grid_->backward(output.image, raster_grad, cam->uid());
                        nvtxRangePop();
                    }

                    if (tile_grad_raw.is_valid() && tile_grad_raw.numel() > 0) {
                        raster_grad = raster_grad + tile_grad_raw;
                    }

                    nvtxRangePush("rasterize_backward");
                    {
                        LFS_VRAM_SCOPE("train.rasterize_backward");
                        LOG_VRAM_DIFF("train.rasterize_backward");
                        if (gsplat_ctx) {
                            auto grad_alpha = tile_grad_alpha.is_valid()
                                                  ? tile_grad_alpha
                                                  : lfs::core::Tensor::zeros_like(output.alpha);
                            tile_context_guard.release();
                            gsplat_rasterize_backward(*gsplat_ctx, raster_grad, grad_alpha,
                                                      strategy_->get_model(), strategy_->get_optimizer(),
                                                      use_pixel_error_densification ? tile_error_map : lfs::core::Tensor{});
                        } else {
                            tile_context_guard.release();
                            if (run_fastgs_gaussian_backward) {
                                // Topology-only locking (see post_backward/step above): the
                                // fused backward updates params in place; the interop
                                // semaphore orders those against the viewer's reads, so the
                                // render thread is only excluded on reallocation iterations.
                                std::unique_lock<std::shared_mutex> model_write_lock(render_mutex_, std::defer_lock);
                                if (strategy_->is_refining(iter))
                                    model_write_lock.lock();
                                fast_rasterize_backward(*fast_ctx, raster_grad, strategy_->get_model(),
                                                        strategy_->get_optimizer(), tile_grad_alpha,
                                                        use_pixel_error_densification ? tile_error_map : lfs::core::Tensor{},
                                                        densification_type,
                                                        iter,
                                                        fused_extra_gradients,
                                                        tile_grad_depth,
                                                        tile_grad_depth.is_valid() && tile_grad_depth.numel() > 0);
                                if (model_write_lock.owns_lock()) {
                                    recordParamsReady();
                                }
                            } else {
                                cleanup_tile_context();
                            }
                        }
                    }
                    nvtxRangePop();
                }

                nvtxRangePop(); // End rasterize
            }

            if (tiles_processed == 0) {
                LOG_DEBUG("Skipping iteration {} - no visible primitives", iter);
                return iter < get_total_iterations() && !stop_requested_.load() && !stop_token.stop_requested()
                           ? StepResult::Continue
                           : StepResult::Stop;
            }

            update_camera_loss_heatmap(*cam, loss_tensor_gpu);
            maybe_publish_camera_loss_heatmap(iter);

            if (in_controller_phase) {
                // Controller phase: only update controller weights
                nvtxRangePush("controller_optimizer_step");
                LFS_VRAM_SCOPE("train.optimizer.ppisp_controller_step");
                LOG_VRAM_DIFF("train.optimizer.ppisp_controller_step");
                ppisp_controller_pool_->optimizer_step(ppisp_cam_idx);
                ppisp_controller_pool_->zero_grad();
                ppisp_controller_pool_->scheduler_step(ppisp_cam_idx);
                nvtxRangePop();
            } else {
                // Normal phase: regularization losses + optimizer steps for all components

                if (params_.optimization.scale_reg > 0.0f) {
                    nvtxRangePush("compute_scale_reg_loss");
                    LFS_VRAM_SCOPE("train.regularizers.scale_loss");
                    LOG_VRAM_DIFF("train.regularizers.scale_loss");
                    if (fastgs_path) {
                        loss_tensor_gpu = loss_tensor_gpu + fused_scale_reg_loss_gpu;
                    } else {
                        auto scale_loss_result = compute_scale_reg_loss(strategy_->get_model(), strategy_->get_optimizer(), params_.optimization);
                        if (!scale_loss_result) {
                            return std::unexpected(scale_loss_result.error());
                        }
                        loss_tensor_gpu = loss_tensor_gpu + *scale_loss_result;
                    }
                    nvtxRangePop();
                }

                if (params_.optimization.opacity_reg > 0.0f) {
                    nvtxRangePush("compute_opacity_reg_loss");
                    LFS_VRAM_SCOPE("train.regularizers.opacity_loss");
                    LOG_VRAM_DIFF("train.regularizers.opacity_loss");
                    if (fastgs_path) {
                        loss_tensor_gpu = loss_tensor_gpu + fused_opacity_reg_loss_gpu;
                    } else {
                        auto opacity_loss_result = compute_opacity_reg_loss(strategy_->get_model(), strategy_->get_optimizer(), params_.optimization);
                        if (!opacity_loss_result) {
                            return std::unexpected(opacity_loss_result.error());
                        }
                        loss_tensor_gpu = loss_tensor_gpu + *opacity_loss_result;
                    }
                    nvtxRangePop();
                }

                if (bilateral_grid_ && params_.optimization.use_bilateral_grid) {
                    nvtxRangePush("bilateral_grid_tv_and_step");
                    LFS_VRAM_SCOPE("train.bilateral_grid.tv_and_step");
                    LOG_VRAM_DIFF("train.bilateral_grid.tv_and_step");
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
                    LFS_VRAM_SCOPE("train.ppisp.reg_and_step");
                    LOG_VRAM_DIFF("train.ppisp.reg_and_step");

                    loss_tensor_gpu = loss_tensor_gpu + ppisp_->reg_loss_gpu();
                    ppisp_->reg_backward();
                    ppisp_->optimizer_step();
                    ppisp_->zero_grad();
                    ppisp_->scheduler_step();

                    nvtxRangePop();
                }
            }

            // Sparsity loss - ALL ON GPU, no CPU sync here
            if (sparsity_optimizer_ &&
                sparsity_optimizer_->should_apply_loss(iter) &&
                (!fastgs_path || update_gaussians_this_iter)) {
                nvtxRangePush("sparsity_loss");
                LFS_VRAM_SCOPE("train.regularizers.sparsity_loss");
                LOG_VRAM_DIFF("train.regularizers.sparsity_loss");
                if (!run_fastgs_gaussian_backward) {
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
                }
                nvtxRangePop();
            }

            // Sparsification phase logging (once per phase transition)
            if (params_.optimization.enable_sparsity) {
                const int first_sparsify_iter = get_sparsity_boundary_iteration() + 1;
                if (get_active_sparsify_steps() > 0 && iter == first_sparsify_iter) {
                    LOG_INFO("Entering sparsification: {} Gaussians, target prune={}%",
                             strategy_->get_model().size(), params_.optimization.prune_ratio * 100);
                }
            }

            // Loss readback at intervals, async: enqueue the D2H into the
            // pinned ring and report harvested samples from earlier iterations
            // — no pipeline stall.
            constexpr int LOSS_SYNC_INTERVAL = 10;
            if (iter % LOSS_SYNC_INTERVAL == 0 || iter == 1) {
                lfs::core::Tensor total_loss = sparsity_loss_gpu.numel() > 0
                                                   ? (loss_tensor_gpu + sparsity_loss_gpu)
                                                   : loss_tensor_gpu;
                if (auto harvested = harvestLossReadbacks(false, in_controller_phase); !harvested) {
                    return std::unexpected(harvested.error());
                }
                submitLossReadback(total_loss, iter);
            }

            if (!in_sparsification && !fastgs_strategy_hooks_at_start) {
                strategy_->pre_step(iter, r_output);
            }

            {
                DeferredEvents deferred;
                {
                    // Same rationale as the post_backward lock above: only block the
                    // render thread when topology actually changes this iteration. The
                    // optimizer step's in-place writes are ordered against the viewer by
                    // the interop semaphore (the render waits for the step's signal before
                    // reading), so the CPU write-lock is needed only for reallocation.
                    std::unique_lock<std::shared_mutex> lock(render_mutex_, std::defer_lock);
                    std::unique_lock<std::shared_mutex> model_write_lock(model_access_mutex_, std::defer_lock);
                    if (strategy_->is_refining(iter)) {
                        lock.lock();
                    } else if (modelAccessLockEnabled()) {
                        // Non-refining in-place writes: hold the model-access lock
                        // exclusive across the optimizer step so viewer/metric
                        // readers (which take it shared) cannot enter mid-write and
                        // tear the model. Refining excludes them via render_mutex_.
                        model_write_lock.lock();
                    }
                    // Drain in-flight reader events immediately before the optimizer
                    // step's in-place writes — not only at the loop top — so the
                    // trainer stream is ordered after any read that began mid-step.
                    // The exclusive lock (when refining) additionally bars new readers.
                    waitForModelReaders();
                    LFS_VRAM_SCOPE("train.optimizer.strategy_step");
                    LOG_VRAM_DIFF("train.optimizer.strategy_step");
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
                            ctx, get_total_iterations(), is_paused_.load(), is_running_.load(), stop_requested_.load(),
                            lfs::training::TrainingPhase::OptimizerStep);
                        lfs::training::ControlBoundary::instance().notify(lfs::training::ControlHook::PreOptimizerStep, ctx);
                    }

                    if (!in_sparsification && !fastgs_strategy_hooks_at_start) {
                        strategy_->post_backward(iter, r_output);
                    }

                    // Skip strategy step if we're in controller distillation phase and freeze is enabled
                    const int ppisp_activation_step = params_.optimization.resolved_ppisp_controller_activation_step(get_total_iterations());
                    const bool freeze_gaussians = ppisp_controller_pool_ &&
                                                  params_.optimization.ppisp_use_controller &&
                                                  params_.optimization.ppisp_freeze_gaussians_on_distill &&
                                                  iter >= ppisp_activation_step;
                    if (!freeze_gaussians) {
                        strategy_->step(iter);
                    }

                    if (sparsity_optimizer_ &&
                        sparsity_optimizer_->is_initialized() &&
                        static_cast<size_t>(model.size()) != model_size_before) {
                        LOG_WARN("Sparsity: resetting ADMM state after topology change at iter {} ({} -> {})",
                                 iter, model_size_before, model.size());
                        sparsity_optimizer_->reset();
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
                    if (auto result = ensureModelTensorAllocatorStorage(model, "strategy step"); !result) {
                        return std::unexpected(result.error());
                    }

                    // End-of-step: parameters are consistent until the next
                    // step's writes; readers wait on this point.
                    recordParamsReady();
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

                const bool save_regular_phase_output = get_active_sparsify_steps() > 0 &&
                                                       iter == get_sparsity_boundary_iteration();
                if (save_regular_phase_output) {
                    LOG_INFO("Saving regular-phase checkpoint and PLY at iteration {} before sparsification", iter);
                    save_ply(params_.dataset.output_path, "", iter, /*join=*/false, /*save_checkpoint=*/false);
                    if (auto result = save_checkpoint(iter); !result) {
                        LOG_WARN("Failed to save regular-phase checkpoint at iteration {}: {}", iter, result.error());
                    }
                }

                // Save checkpoint at specified steps unless the sparsity boundary save already handled it
                for (size_t save_step : params_.optimization.save_steps) {
                    if (iter == static_cast<int>(save_step) &&
                        iter != get_total_iterations() &&
                        !save_regular_phase_output) {
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
                                (params_.dataset.max_width > 0 &&
                                 (cam_to_use->image_height() > params_.dataset.max_width ||
                                  cam_to_use->image_width() > params_.dataset.max_width))) {
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
                    ctx, get_total_iterations(), is_paused_.load(), is_running_.load(), stop_requested_.load(),
                    lfs::training::TrainingPhase::SafeControl);
                lfs::training::ControlBoundary::instance().notify(lfs::training::ControlHook::PostStep, ctx);
            }

            if (live_vram_profiler_enabled() && strategy_) {
                record_splat_vram_breakdown(strategy_->get_model());
                record_optimizer_vram_breakdown(strategy_->get_optimizer());
                lfs::diagnostics::VramProfiler::instance().sampleCudaMemory();
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
            if (iter < get_total_iterations() && !stop_requested_.load() && !stop_token.stop_requested()) {
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
                ctx, get_total_iterations(), is_paused_.load(), is_running_.load(), stop_requested_.load(),
                lfs::training::TrainingPhase::SafeControl);
            lfs::training::ControlBoundary::instance().notify(lfs::training::ControlHook::TrainingStart, ctx);
        }

        try {
            static const bool legacy_stream = []() {
                const char* env = std::getenv("LFS_TRAIN_STREAM_LEGACY");
                return env && env[0] == '1';
            }();
            std::optional<lfs::core::CUDAStreamGuard> stream_guard;
            if (training_stream_ && !legacy_stream) {
                stream_guard.emplace(training_stream_);
                // initialize() ran on another thread on the legacy stream; order
                // all of its work before the first training-stream kernel.
                cudaDeviceSynchronize();
            }

            // Start from current_iteration_ (allows resume from checkpoint)
            int iter = current_iteration_.load() > 0 ? current_iteration_.load() + 1 : 1;
            const RenderMode render_mode = RenderMode::RGB;

            if (progress_) {
                progress_->update(
                    iter,
                    current_loss_.load(),
                    static_cast<int>(strategy_->get_model().size()),
                    get_progress_phase(iter));
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
            PipelinedAuxiliaryImageConfig aux_pipeline_config;
            aux_pipeline_config.load_depths =
                params_.optimization.use_depth_loss &&
                params_.optimization.depth_loss_weight > 0.0f;
            if (aux_pipeline_config.load_depths) {
                LOG_INFO("Depth loss enabled (mode={}, weight={})",
                         depth_loss_mode_name(depth_loss_mode_from_name(params_.optimization.depth_loss_mode)),
                         params_.optimization.depth_loss_weight);
            }
            if (params_.optimization.mask_mode != lfs::core::param::MaskMode::None) {
                aux_pipeline_config.invert_masks = params_.optimization.invert_masks;
                aux_pipeline_config.mask_threshold = params_.optimization.mask_threshold;
                if (params_.optimization.mask_mode == lfs::core::param::MaskMode::SegmentAndIgnore) {
                    aux_pipeline_config.mask_threshold = 0.0f;
                }
                if (params_.optimization.use_alpha_as_mask && alpha_available) {
                    aux_pipeline_config.use_alpha_as_mask = true;
                    aux_pipeline_config.load_masks = true;
                    LOG_INFO("Alpha-as-mask enabled (invert={}, threshold={})",
                             aux_pipeline_config.invert_masks, aux_pipeline_config.mask_threshold);
                } else {
                    aux_pipeline_config.load_masks = true;
                    LOG_INFO("Mask file loading enabled (invert={}, threshold={})",
                             aux_pipeline_config.invert_masks, aux_pipeline_config.mask_threshold);
                }
            }

            auto train_dataloader = create_infinite_pipelined_dataloader(
                train_dataset_, pipelined_config, aux_pipeline_config);
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
            while (iter <= get_total_iterations()) {
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
                pipelined_depth_ = example.depth.has_value() ? std::move(*example.depth) : lfs::core::Tensor();

                if (memory_breakdown_enabled_ && !memory_breakdown_logged_first_batch_) {
                    const auto snapshot = capture_vram_snapshot(true);
                    log_vram_snapshot("after_first_batch_fetch", snapshot);
                    const auto stats = train_dataloader->get_stats();
                    const size_t channels = gt_image.ndim() > 2 ? gt_image.shape()[0] : 1;
                    const size_t height = gt_image.ndim() > 2 ? gt_image.shape()[1] : gt_image.shape()[0];
                    const size_t width = gt_image.ndim() > 2 ? gt_image.shape()[2] : gt_image.shape()[1];
                    LOG_INFO("[MEM] first_batch gt_image={}x{}x{} = {:.2f} MiB, mask = {:.2f} MiB, depth = {:.2f} MiB, output_queue={}, pending_pairs={}, in_flight={}",
                             channels,
                             height,
                             width,
                             bytes_to_mib(tensor_reserved_bytes(gt_image)),
                             bytes_to_mib(tensor_reserved_bytes(pipelined_mask_)),
                             bytes_to_mib(tensor_reserved_bytes(pipelined_depth_)),
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

                        // Device is drained — consume completed loss readbacks
                        // before the retry resubmits into the ring.
                        if (auto harvested = harvestLossReadbacks(true, false); !harvested) {
                            return std::unexpected(harvested.error());
                        }

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
                const bool rotate_checkpoint = get_active_sparsify_steps() == 0;
                save_ply(final_path, params_.dataset.output_name, get_total_iterations(), /*join=*/true,
                         /*save_checkpoint=*/rotate_checkpoint);
            }

            maybe_publish_camera_loss_heatmap(current_iteration_.load(), true);

            if (auto harvested = harvestLossReadbacks(true, false); !harvested) {
                return std::unexpected(harvested.error());
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
                    ctx, get_total_iterations(), is_paused_.load(), is_running_.load(), stop_requested_.load(),
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

    void Trainer::save_ply(const std::filesystem::path& save_path,
                           const std::string& filename,
                           const int iter_num,
                           const bool join_threads,
                           const bool save_checkpoint_file) {

        std::filesystem::path ply_output_path = filename.empty() ? save_path / ("splat_" + std::to_string(iter_num) + ".ply") : save_path / (filename + ".ply");

        const lfs::io::PlySaveOptions ply_options{
            .output_path = ply_output_path,
            .binary = true,
            .async = !join_threads};

        const auto& model = strategy_->get_model();
        const auto export_model = make_ply_export_model(
            model,
            params_.exclude_frozen_add_splats_from_export);
        const auto& model_for_export = export_model ? *export_model : model;
        const auto ply_result = lfs::io::save_ply(model_for_export, ply_options);
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

        if (save_checkpoint_file) {
            auto ckpt_result = lfs::training::save_checkpoint(save_path, iter_num, *strategy_,
                                                              params_for_checkpoint_save(),
                                                              bilateral_grid_.get(), ppisp_.get(), controller_to_save);
            if (!ckpt_result) {
                LOG_WARN("Failed to save checkpoint: {}", ckpt_result.error());
            }
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

        return lfs::training::save_checkpoint(params_.dataset.output_path, iteration, *strategy_,
                                              params_for_checkpoint_save(),
                                              bilateral_grid_.get(), ppisp_.get(), controller_to_save);
    }

    std::expected<void, std::string> Trainer::save_checkpoint_to(const std::filesystem::path& output_path,
                                                                 int iteration) {
        if (!strategy_) {
            return std::unexpected("Cannot save checkpoint: no strategy initialized");
        }

        PPISPControllerPool* controller_to_save = controller_pool_for_save(iteration);

        return lfs::training::save_checkpoint(output_path, iteration, *strategy_,
                                              params_for_checkpoint_save(),
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
        const int camera_idx = is_training_camera ? ppisp_->camera_index(ppisp_->camera_for_frame(camera_uid)) : 0;

        lfs::core::Tensor result;

        if (use_controller && has_controller) {
            const int controller_idx =
                (camera_idx >= 0 && camera_idx < ppisp_controller_pool_->num_cameras()) ? camera_idx : 0;
            const auto controller_params = ppisp_controller_pool_->predict(controller_idx, rgb_chw.unsqueeze(0), 1.0f);
            result = overrides.isIdentity()
                         ? ppisp_->apply_with_controller_params(rgb_chw, controller_params, controller_idx)
                         : ppisp_->apply_with_controller_params_and_overrides(rgb_chw, controller_params, controller_idx,
                                                                              toRenderOverrides(overrides));
        } else if (is_training_camera) {
            const int camera_id = ppisp_->camera_for_frame(camera_uid);
            result = overrides.isIdentity() ? ppisp_->apply(rgb_chw, camera_id, camera_uid)
                                            : ppisp_->apply_with_overrides(rgb_chw, camera_id, camera_uid,
                                                                           toRenderOverrides(overrides));
        } else {
            // Manual viewport overrides should remain usable on novel views even without a controller.
            // Fall back to any learned PPISP frame/camera pair so the user can still inspect exposure,
            // vignette, white-balance, and CRF adjustments on free-orbit renders.
            const int fallback_camera = ppisp_->any_camera_id();
            const int fallback_frame = ppisp_->any_frame_uid();
            result = overrides.isIdentity()
                         ? ppisp_->apply(rgb_chw, fallback_camera, fallback_frame)
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
        return iteration >= params_.optimization.resolved_ppisp_controller_activation_step(get_total_iterations())
                   ? ppisp_controller_pool_.get()
                   : nullptr;
    }

    lfs::core::param::TrainingParameters Trainer::params_for_checkpoint_save() const {
        auto params = params_;
        if (scene_) {
            const auto disabled = scene_->getTrainingDisabledCameraUids();
            params.disabled_camera_uids.assign(disabled.begin(), disabled.end());
        }
        return params;
    }

    void Trainer::save_final_ply_and_checkpoint(const int iteration) {
        save_ply(params_.dataset.output_path, params_.dataset.output_name, iteration, /*join=*/true,
                 /*save_checkpoint=*/false);
        if (auto result = save_checkpoint(iteration); !result) {
            LOG_WARN("Failed to save checkpoint: {}", result.error());
        }
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
            ppisp_controller_pool_.get(), splat_tensor_allocator_);
        if (!result) {
            return result;
        }
        {
            std::unique_lock<std::shared_mutex> render_lock(render_mutex_);
            if (auto storage_result = ensureModelTensorAllocatorStorage(
                    strategy_->get_model(), "checkpoint load");
                !storage_result) {
                return std::unexpected(storage_result.error());
            }
        }

        if (params_.optimization.enable_sparsity) {
            const size_t stop_refine_limit = static_cast<size_t>(std::max(0, get_regular_iterations()));
            if (params_.optimization.stop_refine > stop_refine_limit) {
                LOG_WARN("Sparsity: clamping stop_refine from {} to {} after checkpoint load",
                         params_.optimization.stop_refine, stop_refine_limit);
                params_.optimization.stop_refine = stop_refine_limit;
            }
        }
        sync_strategy_optimization_params();

        current_iteration_ = *result;

        LOG_INFO("Restored training state from checkpoint at iteration {}", *result);
        return result;
    }

} // namespace lfs::training
