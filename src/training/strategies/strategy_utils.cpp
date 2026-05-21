/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "strategy_utils.hpp"
#include "core/logger.hpp"
#include "kernels/pruning_kernels.hpp"

namespace lfs::training {

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
