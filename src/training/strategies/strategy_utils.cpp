/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "strategy_utils.hpp"
#include "core/assert.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/logger.hpp"
#include "kernels/pruning_kernels.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

namespace lfs::training {

    namespace {
        std::atomic<std::uint64_t> g_frozen_mask_rebuilds{0};

        [[nodiscard]] std::vector<bool> build_frozen_vector(
            const lfs::core::SplatData& splat_data,
            const size_t n,
            size_t* frozen_count = nullptr) {
            std::vector<bool> mask(n, false);
            size_t count = 0;

            for (const auto& range : splat_data.frozen_ranges()) {
                if (range.count == 0 || range.start >= n) {
                    continue;
                }
                const size_t end = range.count > n - range.start ? n : range.start + range.count;
                for (size_t i = range.start; i < end; ++i) {
                    if (!mask[i]) {
                        mask[i] = true;
                        ++count;
                    }
                }
            }

            if (frozen_count) {
                *frozen_count = count;
            }
            return mask;
        }

        [[nodiscard]] size_t frozen_ranges_fingerprint(const lfs::core::SplatData& splat_data) {
            size_t h = 1469598103934665603ull; // FNV-1a offset
            for (const auto& range : splat_data.frozen_ranges()) {
                h ^= range.start + 0x9e3779b97f4a7c15ull;
                h *= 1099511628211ull;
                h ^= range.count + 0x9e3779b97f4a7c15ull;
                h *= 1099511628211ull;
            }
            return h;
        }

        // This process-wide cache avoids rebuilding the frozen mask when its
        // model, size, ranges, and device are unchanged.
        struct FrozenMaskCache {
            const void* key = nullptr;
            size_t n = 0;
            size_t ranges_fp = 0;
            lfs::core::Device device = lfs::core::Device::CUDA;
            lfs::core::Tensor mask;
        };
        FrozenMaskCache g_frozen_mask_cache;

    } // namespace

    std::uint64_t frozen_mask_rebuild_count() noexcept {
        return g_frozen_mask_rebuilds.load(std::memory_order_relaxed);
    }

    void reset_frozen_mask_rebuild_count() noexcept {
        g_frozen_mask_rebuilds.store(0, std::memory_order_relaxed);
    }

    void initialize_gaussians(lfs::core::SplatData& splat_data, int max_cap) {
        // Tensors are already on GPU in the new framework (created with Device::CUDA by default)
        // Gradients are now owned by AdamOptimizer, not SplatData

        // Pre-allocate tensor capacity to avoid reallocations during MCMC operations
        // This eliminates memory fragmentation from varying tensor sizes
        if (max_cap > 0) {
            const size_t capacity = static_cast<size_t>(max_cap);
            LOG_INFO("Pre-allocating tensor capacity for {} Gaussians (parameters)", capacity);

            splat_data.reserve_capacity(capacity);
        }

        // Apply quantization after reserving capacity so the quantized layout
        // inherits the correct capacity.
        lfs::training::sh_value::apply_shN_value_quant(splat_data);
    }

    std::unique_ptr<AdamOptimizer> create_optimizer(
        lfs::core::SplatData& splat_data,
        const lfs::core::param::OptimizationParameters& params) {

        // Create Adam config with per-parameter learning rates
        AdamConfig config;
        config.lr = params.means_lr * splat_data.get_scene_scale(); // Default LR (for means)
        // Use double literals (not float!) to match legacy precision
        config.beta1 = 0.9;
        config.beta2 = 0.999;
        config.eps = 1e-15;

        // Set per-parameter learning rates (matching legacy MCMC strategy)
        config.param_lrs["means"] = params.means_lr * splat_data.get_scene_scale();
        config.param_lrs["sh0"] = params.shs_lr;
        config.param_lrs["shN"] = params.shs_lr / 20.0f; // ShN uses reduced LR (1/20 of SH0)
        config.param_lrs["scaling"] = params.scaling_lr;
        config.param_lrs["rotation"] = params.rotation_lr;
        config.param_lrs["opacity"] = params.opacity_lr;

        // Pre-allocate optimizer state capacity to avoid reallocations during training
        // This dramatically reduces peak memory usage by avoiding double-buffering during growth
        if (params.max_cap > 0) {
            config.initial_capacity = static_cast<size_t>(params.max_cap);
            config.growth_factor = 1.5f; // Still allow growth beyond max_cap if needed
            LOG_INFO("AdamOptimizer: pre-allocating capacity for {} Gaussians (optimizer states)", config.initial_capacity);
        }

        LOG_DEBUG("Creating optimizer with per-parameter LRs:");
        LOG_DEBUG("  means: {:.2e}", config.param_lrs["means"]);
        LOG_DEBUG("  sh0: {:.2e}", config.param_lrs["sh0"]);
        LOG_DEBUG("  shN: {:.2e}", config.param_lrs["shN"]);
        LOG_DEBUG("  scaling: {:.2e}", config.param_lrs["scaling"]);
        LOG_DEBUG("  rotation: {:.2e}", config.param_lrs["rotation"]);
        LOG_DEBUG("  opacity: {:.2e}", config.param_lrs["opacity"]);

        auto optimizer = std::make_unique<AdamOptimizer>(splat_data, config);

        return optimizer;
    }

    std::unique_ptr<ExponentialLR> create_scheduler(
        const lfs::core::param::OptimizationParameters& params,
        AdamOptimizer& optimizer) {

        // Python: gamma = 0.01^(1/max_steps)
        // This means after max_steps, lr will be 0.01 * initial_lr
        const double gamma = std::pow(0.01, 1.0 / params.iterations);

        return std::make_unique<ExponentialLR>(optimizer, gamma, std::vector<ParamType>{ParamType::Means});
    }

    lfs::core::Tensor make_frozen_mask(
        const lfs::core::SplatData& splat_data,
        const size_t n,
        const lfs::core::Device device) {
        if (!splat_data.has_frozen_ranges() || n == 0) {
            return {};
        }

        const size_t ranges_fp = frozen_ranges_fingerprint(splat_data);
        const void* key = static_cast<const void*>(&splat_data);
        if (g_frozen_mask_cache.key == key &&
            g_frozen_mask_cache.n == n &&
            g_frozen_mask_cache.ranges_fp == ranges_fp &&
            g_frozen_mask_cache.device == device &&
            g_frozen_mask_cache.mask.is_valid() &&
            g_frozen_mask_cache.mask.numel() == n) {
            return g_frozen_mask_cache.mask; // share persistent GPU tensor
        }

        size_t frozen_count = 0;
        auto mask = build_frozen_vector(splat_data, n, &frozen_count);
        if (frozen_count == 0) {
            g_frozen_mask_cache = {};
            return {};
        }

        auto tensor = lfs::core::Tensor::from_vector(
            mask,
            lfs::core::TensorShape({n}),
            device);
        tensor.set_name("splat.frozen_mask");
        g_frozen_mask_cache.key = key;
        g_frozen_mask_cache.n = n;
        g_frozen_mask_cache.ranges_fp = ranges_fp;
        g_frozen_mask_cache.device = device;
        g_frozen_mask_cache.mask = tensor;
        g_frozen_mask_rebuilds.fetch_add(1, std::memory_order_relaxed);
        return tensor;
    }

    lfs::core::Tensor make_trainable_mask(
        const lfs::core::SplatData& splat_data,
        const size_t n,
        const lfs::core::Device device) {
        auto frozen_mask = make_frozen_mask(splat_data, n, device);
        return frozen_mask.is_valid()
                   ? frozen_mask.logical_not()
                   : lfs::core::Tensor();
    }

    lfs::core::Tensor exclude_frozen_from_mask(
        const lfs::core::SplatData& splat_data,
        const lfs::core::Tensor& mask) {
        if (!mask.is_valid() || mask.numel() == 0 || !splat_data.has_frozen_ranges()) {
            return mask;
        }

        auto frozen_mask = make_frozen_mask(splat_data, mask.numel(), mask.device());
        return frozen_mask.is_valid()
                   ? mask.logical_and(frozen_mask.logical_not())
                   : mask;
    }

    lfs::core::Tensor zero_frozen_scores(
        const lfs::core::SplatData& splat_data,
        const lfs::core::Tensor& scores) {
        if (!scores.is_valid() || scores.numel() == 0 || !splat_data.has_frozen_ranges()) {
            return scores;
        }

        auto frozen_mask = make_frozen_mask(splat_data, scores.numel(), scores.device());
        return frozen_mask.is_valid()
                   ? scores.masked_fill(frozen_mask, 0.0f)
                   : scores;
    }

    void zero_frozen_scores_inplace(
        const lfs::core::SplatData& splat_data,
        lfs::core::Tensor& scores) {
        if (!scores.is_valid() || scores.numel() == 0 || !splat_data.has_frozen_ranges()) {
            return;
        }

        auto frozen_mask = make_frozen_mask(splat_data, scores.numel(), scores.device());
        if (frozen_mask.is_valid()) {
            scores.masked_fill_(frozen_mask, 0.0f);
        }
    }

    lfs::core::Tensor apply_crop_damping_to_scores(
        const AdamOptimizer& optimizer,
        const lfs::core::Tensor& scores) {
        const auto& crop_mask = optimizer.crop_damping_mask();
        const float scale = optimizer.cropbox_lr_scale();
        if (!crop_mask.is_valid() || crop_mask.numel() == 0 || scale == 1.0f) {
            return scores;
        }

        LFS_ASSERT_MSG(scores.is_valid(), "crop-damped scores must be valid");
        LFS_ASSERT_MSG(scores.ndim() == 1, "crop-damped scores must be one-dimensional");
        LFS_ASSERT_MSG(
            scores.numel() <= crop_mask.numel(),
            "crop damping mask must cover every score row");
        LFS_ASSERT_MSG(
            scores.device() == crop_mask.device(),
            "crop damping mask and scores must use the same device");

        const auto active_mask = scores.numel() == crop_mask.numel()
                                     ? crop_mask
                                     : crop_mask.slice(0, 0, scores.numel());
        if (scale == 0.0f) {
            return scores.masked_fill(active_mask, 0.0f);
        }
        return lfs::core::Tensor::where(active_mask, scores * scale, scores);
    }

    size_t frozen_row_count(const lfs::core::SplatData& splat_data, const size_t n) {
        size_t frozen_count = 0;
        (void)build_frozen_vector(splat_data, n, &frozen_count);
        return frozen_count;
    }

    namespace {
        void apply_frozen_ranges_to_optimizer_impl(
            const lfs::core::SplatData& splat_data,
            AdamOptimizer& optimizer,
            const float* freeze_lr_scale) {
            if (freeze_lr_scale != nullptr) {
                optimizer.set_frozen_lr_scale(*freeze_lr_scale);
            }

            const size_t n = static_cast<size_t>(splat_data.size());
            auto mask = make_frozen_mask(splat_data, n, lfs::core::Device::CUDA);
            if (!mask.is_valid()) {
                optimizer.set_frozen_mask({});
                return;
            }

            const size_t frozen_count = frozen_row_count(splat_data, n);
            optimizer.set_frozen_mask(std::move(mask));
            if (freeze_lr_scale != nullptr && *freeze_lr_scale > 0.0f) {
                LOG_INFO("Frozen training mask: {} / {} Gaussians frozen (LR scale: {})",
                         frozen_count,
                         n,
                         *freeze_lr_scale);
            } else {
                LOG_INFO("Frozen training mask: {} / {} Gaussians frozen", frozen_count, n);
            }
        }
    } // namespace

    void apply_frozen_ranges_to_optimizer(
        const lfs::core::SplatData& splat_data,
        AdamOptimizer& optimizer) {
        apply_frozen_ranges_to_optimizer_impl(splat_data, optimizer, nullptr);
    }

    void apply_frozen_ranges_to_optimizer(
        const lfs::core::SplatData& splat_data,
        AdamOptimizer& optimizer,
        const float freeze_lr_scale) {
        apply_frozen_ranges_to_optimizer_impl(splat_data, optimizer, &freeze_lr_scale);
    }

    void remap_frozen_ranges_after_compaction(
        lfs::core::SplatData& splat_data,
        const lfs::core::Tensor& kept_old_indices,
        const size_t old_size) {
        if (!splat_data.has_frozen_ranges()) {
            return;
        }
        if (!kept_old_indices.is_valid() || kept_old_indices.numel() == 0) {
            splat_data.clear_frozen_ranges();
            return;
        }

        auto kept_i64 = kept_old_indices.dtype() == lfs::core::DataType::Int64
                            ? kept_old_indices
                            : kept_old_indices.to(lfs::core::DataType::Int64);
        splat_data.remap_frozen_ranges_after_keep(old_size, kept_i64.to_vector_int64());
    }

    lfs::core::Tensor compute_dead_mask_from_opacity_and_rotation(
        const lfs::core::Tensor& opacities,
        const lfs::core::Tensor& rotations,
        const float min_opacity) {
        using namespace lfs::core;

        Tensor flat_opacities = opacities;
        if (flat_opacities.ndim() == 2 && flat_opacities.shape()[1] == 1) {
            flat_opacities = flat_opacities.squeeze(-1);
        }

        const size_t n = flat_opacities.numel();
        assert(flat_opacities.ndim() == 1);
        assert(rotations.ndim() == 2 && rotations.shape()[1] == 4);
        assert(rotations.shape()[0] == n);

        auto dead_mask = Tensor::empty({n}, Device::CUDA, DataType::Bool);
        pruning::launch_compute_dead_mask(
            flat_opacities.ptr<float>(),
            rotations.ptr<float>(),
            dead_mask.ptr<uint8_t>(),
            n,
            min_opacity);
        return dead_mask;
    }

    lfs::core::Tensor compute_near_zero_rotation_mask(
        const lfs::core::Tensor& rotations) {
        using namespace lfs::core;

        const size_t n = rotations.shape()[0];
        assert(rotations.ndim() == 2 && rotations.shape()[1] == 4);

        auto near_zero_mask = Tensor::empty({n}, Device::CUDA, DataType::Bool);
        pruning::launch_compute_near_zero_rotation_mask(
            rotations.ptr<float>(),
            near_zero_mask.ptr<uint8_t>(),
            n);
        return near_zero_mask;
    }

    void DensifyNScratch::ensure_n(const size_t n, const lfs::core::Device device) {
        using namespace lfs::core;
        if (n == 0) {
            return;
        }
        n_required = std::max(n_required, n);
        if (n_capacity >= n) {
            return;
        }
        // One-shot to max(n, 1.2×) so densify climb does not re-driver-alloc.
        const size_t new_cap = std::max(
            n, static_cast<size_t>(static_cast<double>(std::max(n_capacity, n)) * 1.2) + 1);
        f32_a = Tensor::zeros_direct(TensorShape({new_cap}), new_cap, device, DataType::Float32);
        bool_a = Tensor::zeros_direct(TensorShape({new_cap}), new_cap, device, DataType::Bool);
        n_capacity = new_cap;
    }

    void DensifyNScratch::ensure_k(const size_t k, const lfs::core::Device device) {
        using namespace lfs::core;
        if (k == 0) {
            return;
        }
        k_required = std::max(k_required, k);
        if (k_capacity >= k) {
            return;
        }
        const size_t new_cap = std::max(
            k, static_cast<size_t>(static_cast<double>(std::max(k_capacity, k)) * 1.2) + 1);
        i64_a = Tensor::empty({new_cap}, device, DataType::Int64);
        i64_b = Tensor::empty({new_cap}, device, DataType::Int64);
        k_capacity = new_cap;
    }

    lfs::core::Tensor DensifyNScratch::f32_a_view(const size_t n) const {
        return f32_a.slice(0, 0, n);
    }
    lfs::core::Tensor DensifyNScratch::bool_a_view(const size_t n) const {
        return bool_a.slice(0, 0, n);
    }
    lfs::core::Tensor DensifyNScratch::i64_a_view(const size_t k) const {
        return i64_a.slice(0, 0, k);
    }
    lfs::core::Tensor DensifyNScratch::i64_b_view(const size_t k) const {
        return i64_b.slice(0, 0, k);
    }

    void DensifyChildWorkspace::ensure(
        const size_t K,
        const size_t sh_rest_in,
        const bool use_shN,
        const bool sh0_flat_layout,
        const lfs::core::Device device) {

        using namespace lfs::core;
        if (K == 0)
            return;

        const size_t need = K;
        constexpr std::size_t base_floats_per_row = 3 + 4 + 3 + 1 + 3;
        required_bytes_high_water = std::max(
            required_bytes_high_water,
            K * (base_floats_per_row + (use_shN ? sh_rest_in * 3 : 0)) * sizeof(float));
        const bool layout_changed =
            (sh0_as_flat != sh0_flat_layout) ||
            (use_shN && sh_rest != sh_rest_in) ||
            (use_shN && (!shN.is_valid() || shN.ndim() != 3));

        if (capacity < need || layout_changed) {
            const size_t new_cap = std::max(
                need,
                static_cast<size_t>(static_cast<double>(std::max(capacity, need)) * 1.2) + 1);
            means = Tensor::empty({new_cap, 3}, device);
            rotations = Tensor::empty({new_cap, 4}, device);
            scales = Tensor::empty({new_cap, 3}, device);
            opacities = Tensor::empty({new_cap}, device);
            sh0_as_flat = sh0_flat_layout;
            if (sh0_flat_layout) {
                sh0_flat = Tensor::empty({new_cap, 3}, device);
                sh0 = Tensor();
            } else {
                sh0 = Tensor::empty({new_cap, 1, 3}, device);
                sh0_flat = Tensor();
            }
            sh_rest = sh_rest_in;
            if (use_shN && sh_rest_in > 0) {
                shN = Tensor::empty({new_cap, sh_rest_in, 3}, device);
            } else {
                shN = Tensor();
            }
            capacity = new_cap;
        } else if (use_shN && sh_rest_in > 0 &&
                   (!shN.is_valid() || shN.shape()[0] < capacity || shN.shape()[1] != sh_rest_in)) {
            sh_rest = sh_rest_in;
            shN = Tensor::empty({capacity, sh_rest_in, 3}, device);
        }
    }

    namespace {
        [[nodiscard]] std::size_t densify_child_bytes(
            const std::size_t rows,
            const std::size_t sh_rest) noexcept {
            // means + rotations + scales + opacity + sh0 + optional rest SH.
            constexpr std::size_t base_floats_per_row = 3 + 4 + 3 + 1 + 3;
            return rows * (base_floats_per_row + sh_rest * 3) * sizeof(float);
        }
    } // namespace

    std::size_t DensifyChildWorkspace::required_bytes() const noexcept {
        return required_bytes_high_water;
    }

    std::size_t DensifyChildWorkspace::resident_bytes() const noexcept {
        return densify_child_bytes(capacity, shN.is_valid() ? sh_rest : 0);
    }

    lfs::core::Tensor DensifyChildWorkspace::means_view(const size_t K) const {
        return means.slice(0, 0, K);
    }
    lfs::core::Tensor DensifyChildWorkspace::rotations_view(const size_t K) const {
        return rotations.slice(0, 0, K);
    }
    lfs::core::Tensor DensifyChildWorkspace::scales_view(const size_t K) const {
        return scales.slice(0, 0, K);
    }
    lfs::core::Tensor DensifyChildWorkspace::sh0_view(const size_t K) const {
        if (sh0_as_flat)
            return sh0_flat.slice(0, 0, K);
        return sh0.slice(0, 0, K);
    }
    lfs::core::Tensor DensifyChildWorkspace::shN_view(const size_t K) const {
        if (!shN.is_valid())
            return lfs::core::Tensor();
        return shN.slice(0, 0, K);
    }
    lfs::core::Tensor DensifyChildWorkspace::opacities_view(const size_t K) const {
        return opacities.slice(0, 0, K);
    }

    void ensure_densification_info_shape_inplace(
        lfs::core::Tensor& densification_info,
        const size_t n,
        const lfs::core::Device device,
        const size_t /*reserve_cols*/) {
        using namespace lfs::core;
        // densification_info is row-major [2, N]: row1 starts at offset N. Growing N
        // cannot use append_zeros (that grows dim0). When shape already matches,
        // reuse the buffer (caller zeros). On mismatch, allocate exact [2,n].
        // Pre-sizing columns would break the rasterizer's [2,N] row offsets, so
        // shape mismatches require an exact allocation.
        if (densification_info.is_valid() &&
            densification_info.ndim() == 2 &&
            densification_info.shape()[0] >= 2 &&
            densification_info.shape()[1] == n) {
            return;
        }
        densification_info = Tensor::zeros({2, n}, device);
        densification_info.set_name("splat.densification_info");
    }

    void ensure_score_buffer_inplace(
        lfs::core::Tensor& scores,
        const size_t n,
        const lfs::core::Device device,
        const size_t reserve_capacity) {
        using namespace lfs::core;
        const size_t desired_cap = reserve_capacity > 0 ? std::max(reserve_capacity, n) : n;

        if (!scores.is_valid() || scores.ndim() != 1 || scores.device() != device ||
            scores.dtype() != DataType::Float32) {
            if (desired_cap > n) {
                scores = Tensor::zeros_direct(TensorShape({n}), desired_cap, device);
            } else {
                scores = Tensor::zeros({n}, device);
            }
            return;
        }

        const size_t cur = scores.numel();
        if (cur == n) {
            if (desired_cap > n && scores.capacity() < desired_cap) {
                scores.reserve(desired_cap);
            }
            return;
        }
        if (cur < n) {
            if (scores.capacity() < desired_cap) {
                if (scores.capacity() == 0 || scores.is_external_storage()) {
                    // Slow path: realloc with headroom.
                    auto fresh = desired_cap > n
                                     ? Tensor::zeros_direct(TensorShape({n}), desired_cap, device)
                                     : Tensor::zeros({n}, device);
                    if (cur > 0) {
                        cudaMemcpy(fresh.ptr<float>(), scores.ptr<float>(),
                                   cur * sizeof(float), cudaMemcpyDeviceToDevice);
                    }
                    scores = std::move(fresh);
                    return;
                }
                scores.reserve(desired_cap);
            }
            scores.append_zeros(n - cur);
            return;
        }
        // Shrink logical size: reallocate exact (rare; refine reset uses zero_ at same n).
        if (desired_cap > n) {
            scores = Tensor::zeros_direct(TensorShape({n}), desired_cap, device);
        } else {
            scores = Tensor::zeros({n}, device);
        }
    }

    int collect_adam_scale_ptrs(AdamOptimizer& optimizer, float* out_ptrs[12]) {
        int n = 0;
        const ParamType types[] = {
            ParamType::Means, ParamType::Sh0, ParamType::ShN,
            ParamType::Scaling, ParamType::Rotation, ParamType::Opacity};
        for (ParamType pt : types) {
            auto* state = optimizer.get_state_mutable(pt);
            if (!state)
                continue;
            if (state->exp_avg_scale.is_valid() && state->exp_avg_scale.numel() > 0 &&
                state->exp_avg_scale.ptr<float>() != nullptr) {
                out_ptrs[n++] = state->exp_avg_scale.ptr<float>();
            }
            if (state->exp_avg_sq_scale.is_valid() && state->exp_avg_sq_scale.numel() > 0 &&
                state->exp_avg_sq_scale.ptr<float>() != nullptr) {
                out_ptrs[n++] = state->exp_avg_sq_scale.ptr<float>();
            }
        }
        return n;
    }

    void zero_adam_grads_at_indices(
        AdamOptimizer& optimizer,
        const lfs::core::Tensor& indices,
        const uint32_t shN_layout_rest) {
        using namespace lfs::core;
        if (!indices.is_valid() || indices.numel() == 0)
            return;

        const ParamType types[] = {
            ParamType::Means, ParamType::Sh0, ParamType::ShN,
            ParamType::Scaling, ParamType::Rotation, ParamType::Opacity};

        for (ParamType pt : types) {
            auto* state = optimizer.get_state_mutable(pt);
            if (!state || !state->grad.is_valid() || state->grad.numel() == 0)
                continue;

            if (pt == ParamType::ShN) {
                if (shN_layout_rest != 0) {
                    auto idx_i32 = indices.dtype() == DataType::Int32
                                       ? indices
                                       : indices.to(DataType::Int32);
                    shN_swizzled_zero_at_indices(
                        state->grad.ptr<float>(), idx_i32.ptr<int>(),
                        idx_i32.numel(), shN_layout_rest);
                }
                continue;
            }

            const auto& shape = state->grad.shape();
            if (shape.rank() == 0)
                continue;
            bool any_zero = false;
            for (size_t d = 0; d < shape.rank(); ++d) {
                if (shape[d] == 0) {
                    any_zero = true;
                    break;
                }
            }
            if (any_zero)
                continue;

            std::vector<size_t> dims = {indices.numel()};
            for (size_t i = 1; i < shape.rank(); ++i)
                dims.push_back(shape[i]);
            auto zeros = Tensor::zeros(TensorShape(dims), state->grad.device());
            state->grad.index_put_(indices, zeros);
        }
    }

} // namespace lfs::training
