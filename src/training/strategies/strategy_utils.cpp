/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "strategy_utils.hpp"
#include "core/logger.hpp"
#include "kernels/pruning_kernels.hpp"
#include <algorithm>
#include <vector>

namespace lfs::training {

    namespace {
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

    } // namespace

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

        size_t frozen_count = 0;
        auto mask = build_frozen_vector(splat_data, n, &frozen_count);
        if (frozen_count == 0) {
            return {};
        }

        auto tensor = lfs::core::Tensor::from_vector(
            mask,
            lfs::core::TensorShape({n}),
            device);
        tensor.set_name("splat.frozen_mask");
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

    size_t frozen_row_count(const lfs::core::SplatData& splat_data, const size_t n) {
        size_t frozen_count = 0;
        (void)build_frozen_vector(splat_data, n, &frozen_count);
        return frozen_count;
    }

    void apply_frozen_ranges_to_optimizer(
        const lfs::core::SplatData& splat_data,
        AdamOptimizer& optimizer) {
        const size_t n = static_cast<size_t>(splat_data.size());
        auto mask = make_frozen_mask(splat_data, n, lfs::core::Device::CUDA);
        if (!mask.is_valid()) {
            optimizer.set_frozen_mask({});
            return;
        }

        const size_t frozen_count = frozen_row_count(splat_data, n);
        optimizer.set_frozen_mask(std::move(mask));
        LOG_INFO("Frozen training mask: {} / {} Gaussians frozen", frozen_count, n);
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

    void update_param_with_optimizer(
        const ParamUpdateFn& param_fn,
        const OptimizerUpdateFn& optimizer_fn,
        std::unique_ptr<AdamOptimizer>& optimizer,
        lfs::core::SplatData& splat_data,
        std::vector<size_t> param_idxs) {

        // CRITICAL: Ensure CUDA device is set for this thread
        // Some operations might spawn TBB threads, and those need CUDA context
        cudaSetDevice(0);

        // Map param index to ParamType
        auto index_to_param_type = [](size_t idx) -> ParamType {
            switch (idx) {
            case 0: return ParamType::Means;
            case 1: return ParamType::Sh0;
            case 2: return ParamType::ShN;
            case 3: return ParamType::Scaling;
            case 4: return ParamType::Rotation;
            case 5: return ParamType::Opacity;
            default:
                LOG_ERROR("Invalid parameter index: {}", idx);
                return ParamType::Means;
            }
        };

        // Get references to all parameters
        // (Gradients are now owned by AdamOptimizer, not SplatData)
        std::array<lfs::core::Tensor*, 6> params = {
            &splat_data.means(),
            &splat_data.sh0(),
            &splat_data.shN(),
            &splat_data.scaling_raw(),
            &splat_data.rotation_raw(),
            &splat_data.opacity_raw()};

        std::array<lfs::core::Tensor, 6> new_params;

        // First pass: Compute new parameters and update optimizer state
        for (auto i : param_idxs) {
            auto param = params[i];
            cudaError_t err_before = cudaGetLastError();
            if (err_before != cudaSuccess) {
                LOG_ERROR("CUDA error before param_fn: {}", cudaGetErrorString(err_before));
            }

            auto param_type = index_to_param_type(i);
            LOG_DEBUG("Calling param_fn for param {}", i);

            auto new_param = param_fn(i, *param);

            cudaError_t err_after = cudaGetLastError();
            if (err_after != cudaSuccess) {
                LOG_ERROR("CUDA error after param_fn({}) [param_type={}]: {}", i, static_cast<int>(param_type), cudaGetErrorString(err_after));
                throw std::runtime_error(std::string("CUDA error in param_fn (param ") + std::to_string(i) + "): " + cudaGetErrorString(err_after));
            }
            new_params[i] = new_param;

            // Modify state in-place (preserves capacity)
            AdamParamState* state = optimizer->get_state_mutable(param_type);
            if (state) {
                optimizer_fn(*state, new_param);
            }
        }

        // Second pass: Update parameters in SplatData
        // (Gradient updates are handled by the optimizer_fn callback which updates optimizer state)
        for (auto i : param_idxs) {
            if (i == 0) {
                splat_data.means() = new_params[i];
            } else if (i == 1) {
                splat_data.sh0() = new_params[i];
            } else if (i == 2) {
                splat_data.shN() = new_params[i];
            } else if (i == 3) {
                splat_data.scaling_raw() = new_params[i];
            } else if (i == 4) {
                splat_data.rotation_raw() = new_params[i];
            } else if (i == 5) {
                splat_data.opacity_raw() = new_params[i];
            }
        }
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

} // namespace lfs::training
