/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mcmc.hpp"
#include "core/logger.hpp"
#include "kernels/mcmc_kernels.hpp"
#include "strategy_utils.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace lfs::training {

    namespace {
        [[nodiscard]] inline bool has_zero_dimension(const lfs::core::TensorShape& shape) {
            for (size_t i = 0; i < shape.rank(); ++i) {
                if (shape[i] == 0) {
                    return true;
                }
            }
            return false;
        }

        [[nodiscard]] size_t deleted_mask_capacity(const lfs::core::SplatData& splat_data) {
            const size_t means_capacity = splat_data.means().capacity();
            return means_capacity > 0 ? means_capacity : static_cast<size_t>(splat_data.size());
        }

        void ensure_deleted_mask_size(lfs::core::SplatData& splat_data) {
            const size_t current_size = static_cast<size_t>(splat_data.size());
            const size_t desired_capacity = deleted_mask_capacity(splat_data);
            auto& deleted = splat_data.deleted();

            if (!deleted.is_valid() || deleted.ndim() != 1 || deleted.numel() != current_size) {
                deleted = lfs::core::Tensor::zeros_bool({current_size}, splat_data.means().device());
            }

            deleted.reserve(desired_capacity);
        }

        void set_deleted_mask_rows(
            lfs::core::SplatData& splat_data,
            const lfs::core::Tensor& indices,
            const bool deleted) {
            if (indices.numel() == 0) {
                return;
            }

            ensure_deleted_mask_size(splat_data);
            auto values = deleted
                              ? lfs::core::Tensor::ones_bool({static_cast<size_t>(indices.numel())}, indices.device())
                              : lfs::core::Tensor::zeros_bool({static_cast<size_t>(indices.numel())}, indices.device());
            splat_data.deleted().index_put_(indices, values);
        }

        void append_live_deleted_rows(lfs::core::SplatData& splat_data, const size_t n_rows) {
            if (n_rows == 0 || !splat_data.has_deleted_mask()) {
                return;
            }

            auto& deleted = splat_data.deleted();
            const size_t desired_capacity = std::max(
                deleted_mask_capacity(splat_data),
                static_cast<size_t>(deleted.numel()) + n_rows);
            deleted.reserve(desired_capacity);
            deleted.append_zeros(n_rows);
        }

        void zero_optimizer_state(
            lfs::training::AdamOptimizer& optimizer,
            const ParamType param_type,
            const lfs::core::Tensor& indices) {
            if (indices.numel() == 0) {
                return;
            }

            auto* state = optimizer.get_state_mutable(param_type);
            if (!state) {
                return;
            }

            const auto& shape = state->exp_avg.shape();
            if (has_zero_dimension(shape)) {
                return;
            }

            std::vector<size_t> dims = {static_cast<size_t>(indices.numel())};
            for (size_t i = 1; i < shape.rank(); ++i) {
                dims.push_back(shape[i]);
            }

            auto zeros = lfs::core::Tensor::zeros(lfs::core::TensorShape(dims), state->exp_avg.device());
            state->exp_avg.index_put_(indices, zeros);
            state->exp_avg_sq.index_put_(indices, zeros);
            if (state->grad.is_valid()) {
                state->grad.index_put_(indices, zeros);
            }
        }
    } // anonymous namespace

    MCMC::MCMC(lfs::core::SplatData& splat_data) : _splat_data(&splat_data) {}

    lfs::core::Tensor MCMC::multinomial_sample(const lfs::core::Tensor& weights, int n, bool replacement) {
        // Use the tensor library's built-in multinomial sampling
        return lfs::core::Tensor::multinomial(weights, n, replacement);
    }

    void MCMC::update_optimizer_for_relocate(
        const lfs::core::Tensor& sampled_indices,
        const lfs::core::Tensor& dead_indices,
        ParamType param_type) {

        // Reset optimizer state (exp_avg and exp_avg_sq) for relocated Gaussians
        // Use GPU version for efficiency (indices already on GPU)
        _optimizer->relocate_params_at_indices_gpu(
            param_type,
            sampled_indices.ptr<int64_t>(),
            sampled_indices.numel());
    }

    void MCMC::ensure_densification_info_shape() {
        const size_t n = static_cast<size_t>(_splat_data->size());
        const auto& info = _splat_data->_densification_info;
        if (!info.is_valid() ||
            info.ndim() != 2 ||
            info.shape()[0] < 2 ||
            info.shape()[1] != n) {
            _splat_data->_densification_info = lfs::core::Tensor::zeros({2, n}, _splat_data->means().device());
        }

        if (!_error_score_max.is_valid() ||
            _error_score_max.ndim() != 1 ||
            _error_score_max.numel() != n) {
            _error_score_max = lfs::core::Tensor::zeros({n}, _splat_data->means().device());
            _error_score_windows = 0;
        }
    }

    lfs::core::Tensor MCMC::get_sampling_weights() const {
        using namespace lfs::core;

        const size_t n = static_cast<size_t>(_splat_data->size());
        if (!_error_score_max.is_valid() ||
            _error_score_max.ndim() != 1 ||
            _error_score_max.numel() != n) {
            return Tensor::ones({n}, _splat_data->means().device());
        }

        return _error_score_max.clamp_min(1e-12f);
    }

    int MCMC::relocate_gs() {
        LOG_TIMER("MCMC::relocate_gs");
        using namespace lfs::core;

        // Get opacities (handle both [N] and [N, 1] shapes)
        Tensor opacities;
        {
            LOG_TIMER("relocate_get_opacities");
            opacities = _splat_data->get_opacity();
            if (opacities.ndim() == 2 && opacities.shape()[1] == 1) {
                opacities = opacities.squeeze(-1);
            }
        }

        // Find dead Gaussians: opacity <= min_opacity OR rotation magnitude near zero
        Tensor dead_mask, dead_indices;
        size_t n_dead;
        {
            LOG_TIMER("relocate_find_dead");
            // Fully fused kernel - no intermediate allocations
            const size_t N = opacities.numel();
            dead_mask = Tensor::empty({N}, Device::CUDA, DataType::Bool);
            mcmc::launch_compute_dead_mask(
                opacities.ptr<float>(),
                _splat_data->rotation_raw().ptr<float>(),
                dead_mask.ptr<uint8_t>(),
                N,
                _params->min_opacity);
            dead_indices = dead_mask.nonzero().squeeze(-1);
            n_dead = dead_indices.numel();
        }

        if (n_dead == 0)
            return 0;

        Tensor alive_indices;
        {
            LOG_TIMER("relocate_find_alive");
            Tensor alive_mask = dead_mask.logical_not();
            alive_indices = alive_mask.nonzero().squeeze(-1);
        }

        if (alive_indices.numel() == 0)
            return 0;

        Tensor sampled_idxs, sampled_opacities, sampled_scales;
        {
            LOG_TIMER("relocate_multinomial_sample_and_gather_FUSED");
            const size_t N = opacities.numel();

            // Get source tensors (contiguous)
            Tensor opacities_contig = opacities.contiguous();
            const Tensor sampling_weights = get_sampling_weights();
            Tensor scaling_raw_contig = _splat_data->scaling_raw().contiguous(); // Pass raw scaling, kernel applies exp()

            // Allocate outputs
            sampled_idxs = Tensor::empty({n_dead}, Device::CUDA, DataType::Int64);
            sampled_opacities = Tensor::empty({n_dead}, Device::CUDA, DataType::Float32);
            sampled_scales = Tensor::empty({n_dead, 3}, Device::CUDA, DataType::Float32);

            static thread_local uint64_t seed_counter = 0;
            const uint64_t seed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + seed_counter++;

            // does multinomial sampling + gathering in one pass
            mcmc::launch_multinomial_sample_and_gather(
                sampling_weights.ptr<float>(),
                opacities_contig.ptr<float>(),
                scaling_raw_contig.ptr<float>(), // Pass raw scaling
                alive_indices.ptr<int64_t>(),
                alive_indices.numel(),
                n_dead,
                seed,
                sampled_idxs.ptr<int64_t>(),
                sampled_opacities.ptr<float>(),
                sampled_scales.ptr<float>(),
                N);
        }

        // Count occurrences of each sampled index (how many times each was sampled)
        Tensor ratios;
        {
            LOG_TIMER("relocate_count_occurrences");
            auto ones_N = _ones_int32.slice(0, 0, opacities.numel()).clone();
            ratios = ones_N.index_add_(0, sampled_idxs, _ones_int32.slice(0, 0, sampled_idxs.numel()));
            ratios = ratios.index_select(0, sampled_idxs).contiguous();

            // Clamp ratios to [1, n_max]
            const int n_max = _n_max;
            ratios = ratios.clamp(1, n_max);
        }

        // Allocate output tensors and call CUDA kernel
        Tensor new_opacities, new_scales;
        {
            LOG_TIMER("relocate_cuda_kernel");
            new_opacities = Tensor::empty(sampled_opacities.shape(), Device::CUDA);
            new_scales = Tensor::empty(sampled_scales.shape(), Device::CUDA);

            mcmc::launch_relocation_kernel(
                sampled_opacities.ptr<float>(),
                sampled_scales.ptr<float>(),
                ratios.ptr<int32_t>(),
                new_opacities.ptr<float>(),
                new_scales.ptr<float>(),
                sampled_opacities.numel());
        }

        // Clamp new opacities and compute raw values
        Tensor new_opacity_raw;
        {
            LOG_TIMER("relocate_compute_raw_values");
            new_opacities = new_opacities.clamp(_params->min_opacity, 1.0f - 1e-7f);
            new_opacity_raw = new_opacities.logit(1e-7f);

            if (_splat_data->opacity_raw().ndim() == 2) {
                new_opacity_raw = new_opacity_raw.unsqueeze(-1);
            }
        }

        // Update parameters
        {
            LOG_TIMER("relocate_update_params");
            const int opacity_dim = (_splat_data->opacity_raw().ndim() == 2) ? 1 : 0;
            const size_t N = _splat_data->means().shape()[0]; // Total number of Gaussians

            // Compute log(scales) for the new scales
            Tensor new_scales_log = new_scales.log();

            // Update sampled indices with new opacity/scaling using direct CUDA kernel
            // This preserves tensor capacity (unlike index_put_ which creates new tensors)
            mcmc::launch_update_scaling_opacity(
                sampled_idxs.ptr<int64_t>(),
                new_scales_log.ptr<float>(),
                new_opacity_raw.ptr<float>(),
                _splat_data->scaling_raw().ptr<float>(),
                _splat_data->opacity_raw().ptr<float>(),
                sampled_idxs.numel(),
                opacity_dim,
                N);

            // Copy sampled params to dead slots
            const size_t sh_coeffs = (_splat_data->shN().is_valid() && _splat_data->shN().ndim() >= 2)
                                         ? _splat_data->shN().shape()[1]
                                         : 0;
            mcmc::launch_copy_gaussian_params(
                sampled_idxs.ptr<int64_t>(),
                dead_indices.ptr<int64_t>(),
                _splat_data->means().ptr<float>(),
                _splat_data->sh0().ptr<float>(),
                _splat_data->shN().ptr<float>(),
                _splat_data->scaling_raw().ptr<float>(),
                _splat_data->rotation_raw().ptr<float>(),
                _splat_data->opacity_raw().ptr<float>(),
                dead_indices.numel(),
                sh_coeffs,
                opacity_dim,
                N);
        }

        // Update optimizer states for all parameters
        {
            LOG_TIMER("relocate_update_optimizer");
            update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::Means);
            update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::Sh0);
            update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::ShN);
            update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::Scaling);
            update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::Rotation);
            update_optimizer_for_relocate(sampled_idxs, dead_indices, ParamType::Opacity);
        }

        if (_splat_data->has_deleted_mask()) {
            set_deleted_mask_rows(*_splat_data, dead_indices, false);
        }

        return n_dead;
    }

    int MCMC::add_new_gs() {
        LOG_TIMER("MCMC::add_new_gs");
        using namespace lfs::core;

        if (!_optimizer) {
            LOG_ERROR("MCMC::add_new_gs: optimizer not initialized");
            return 0;
        }

        const int current_n = _splat_data->size();
        const int n_target = std::min(_params->max_cap, static_cast<int>(1.05f * current_n));
        const size_t n_new = std::max(0, n_target - current_n);

        if (n_new == 0)
            return 0;

        // Get opacities (handle both [N] and [N, 1] shapes)
        Tensor opacities;
        {
            LOG_TIMER("add_new_get_opacities");
            opacities = _splat_data->get_opacity();
            if (opacities.ndim() == 2 && opacities.shape()[1] == 1) {
                opacities = opacities.squeeze(-1);
            }
        }

        Tensor sampled_idxs;
        Tensor sampled_opacities;
        Tensor sampled_scales;
        {
            LOG_TIMER("add_new_multinomial_sample_and_gather");

            const size_t N = opacities.numel();

            // Get raw scaling and ensure contiguity
            auto scaling_raw_contig = _splat_data->scaling_raw().contiguous(); // Pass raw scaling, kernel applies exp()
            auto opacities_contig = opacities.contiguous();
            const auto sampling_weights = get_sampling_weights();

            // Allocate output tensors
            sampled_idxs = Tensor::empty({n_new}, Device::CUDA, DataType::Int64);
            sampled_opacities = Tensor::empty({n_new}, Device::CUDA, DataType::Float32);
            sampled_scales = Tensor::empty({n_new, 3}, Device::CUDA, DataType::Float32);

            // Generate random seed
            auto seed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());

            // Call fused CUDA kernel
            mcmc::launch_multinomial_sample_all(
                sampling_weights.ptr<float>(),
                opacities_contig.ptr<float>(),
                scaling_raw_contig.ptr<float>(), // Pass raw scaling
                N,
                n_new,
                seed,
                sampled_idxs.ptr<int64_t>(),
                sampled_opacities.ptr<float>(),
                sampled_scales.ptr<float>());
        }

        // Count occurrences as int32 to avoid float->int conversions in the hot path.
        Tensor ratios;
        {
            LOG_TIMER("add_new_count_occurrences");
            ratios = _ones_int32.slice(0, 0, opacities.numel()).clone();
            ratios = ratios.index_add_(0, sampled_idxs, _ones_int32.slice(0, 0, sampled_idxs.numel()));
            ratios = ratios.index_select(0, sampled_idxs);

            // Clamp in int32 domain
            const int n_max = _n_max;
            ratios = ratios.clamp(1, n_max);
            ratios = ratios.contiguous();
        }

        // Allocate output tensors and call CUDA kernel
        Tensor new_opacities, new_scales;
        {
            LOG_TIMER("add_new_relocation_kernel");
            new_opacities = Tensor::empty(sampled_opacities.shape(), Device::CUDA);
            new_scales = Tensor::empty(sampled_scales.shape(), Device::CUDA);

            mcmc::launch_relocation_kernel(
                sampled_opacities.ptr<float>(),
                sampled_scales.ptr<float>(),
                ratios.ptr<int32_t>(),
                new_opacities.ptr<float>(),
                new_scales.ptr<float>(),
                sampled_opacities.numel());
        }

        // Clamp new opacities and prepare raw values
        Tensor new_opacity_raw, new_scaling_raw;
        {
            LOG_TIMER("add_new_compute_raw_values");
            new_opacities = new_opacities.clamp(_params->min_opacity, 1.0f - 1e-7f);
            new_opacity_raw = new_opacities.logit(1e-7f);
            new_scaling_raw = new_scales.log();

            if (_splat_data->opacity_raw().ndim() == 2) {
                new_opacity_raw = new_opacity_raw.unsqueeze(-1);
            }
        }

        // Update existing Gaussians first (before concatenation)
        {
            LOG_TIMER("add_new_update_original");
            const int opacity_dim = (_splat_data->opacity_raw().ndim() == 2) ? 1 : 0;
            const size_t N = _splat_data->means().shape()[0];

            // Use direct CUDA kernel to preserve tensor capacity
            mcmc::launch_update_scaling_opacity(
                sampled_idxs.ptr<int64_t>(),
                new_scaling_raw.ptr<float>(),
                new_opacity_raw.ptr<float>(),
                _splat_data->scaling_raw().ptr<float>(),
                _splat_data->opacity_raw().ptr<float>(),
                sampled_idxs.numel(),
                opacity_dim,
                N);
        }

        // Use add_new_params_gather() to leverage reserved capacity
        {
            LOG_TIMER("add_new_append_gather");
            // Gather and append parameters for new Gaussians (done after updating opacity/scaling)
            _optimizer->add_new_params_gather(ParamType::Means, sampled_idxs);
            _optimizer->add_new_params_gather(ParamType::Sh0, sampled_idxs);
            _optimizer->add_new_params_gather(ParamType::ShN, sampled_idxs);
            _optimizer->add_new_params_gather(ParamType::Rotation, sampled_idxs);
            _optimizer->add_new_params_gather(ParamType::Opacity, sampled_idxs);
            _optimizer->add_new_params_gather(ParamType::Scaling, sampled_idxs);
        }

        append_live_deleted_rows(*_splat_data, n_new);

        return n_new;
    }

    // Test helper: add_new_gs with explicitly specified indices (no multinomial sampling)
    int MCMC::add_new_gs_with_indices_test(const lfs::core::Tensor& sampled_idxs) {
        LOG_TIMER("MCMC::add_new_gs_with_indices_test");
        using namespace lfs::core;

        if (!_optimizer) {
            LOG_ERROR("add_new_gs_with_indices_test called but optimizer not initialized");
            return 0;
        }

        const int n_new = sampled_idxs.numel();
        if (n_new == 0)
            return 0;

        // Ensure indices are Int64 (test may pass Int32)
        Tensor sampled_idxs_i64 = (sampled_idxs.dtype() == DataType::Int64) ? sampled_idxs : sampled_idxs.to(DataType::Int64);

        // Get opacities
        auto opacities = _splat_data->get_opacity();

        // Get parameters for sampled Gaussians
        auto sampled_opacities = opacities.index_select(0, sampled_idxs_i64);
        auto sampled_scales = _splat_data->get_scaling().index_select(0, sampled_idxs_i64);

        // Ensure cached ones buffer covers current model size
        const size_t required = _splat_data->size();
        if (!_ones_int32.is_valid() || _ones_int32.numel() < required) {
            _ones_int32 = Tensor::ones({required}, Device::CUDA, DataType::Int32);
        }

        // Count occurrences in int32 and keep +1 baseline.
        auto ratios = _ones_int32.slice(0, 0, required).clone();
        ratios.index_add_(0, sampled_idxs_i64, _ones_int32.slice(0, 0, sampled_idxs_i64.numel()));
        ratios = ratios.index_select(0, sampled_idxs_i64);

        // Clamp in int32 domain
        const int n_max = _n_max;
        ratios = ratios.clamp(1, n_max);
        ratios = ratios.contiguous();

        // Call the CUDA relocation function
        Tensor new_opacities, new_scales;
        {
            LOG_TIMER("add_new_relocation");
            new_opacities = Tensor::empty(sampled_opacities.shape(), Device::CUDA);
            new_scales = Tensor::empty(sampled_scales.shape(), Device::CUDA);

            mcmc::launch_relocation_kernel(
                sampled_opacities.ptr<float>(),
                sampled_scales.ptr<float>(),
                ratios.ptr<int32_t>(),
                new_opacities.ptr<float>(),
                new_scales.ptr<float>(),
                sampled_opacities.numel());
        }

        // Clamp new opacities and prepare raw values
        Tensor new_opacity_raw, new_scaling_raw;
        {
            LOG_TIMER("add_new_compute_raw_values");
            new_opacities = new_opacities.clamp(_params->min_opacity, 1.0f - 1e-7f);
            new_opacity_raw = new_opacities.logit(1e-7f);
            new_scaling_raw = new_scales.log();

            if (_splat_data->opacity_raw().ndim() == 2) {
                new_opacity_raw = new_opacity_raw.unsqueeze(-1);
            }
        }

        // Update existing Gaussians first
        {
            LOG_TIMER("add_new_update_original");
            const int opacity_dim = (_splat_data->opacity_raw().ndim() == 2) ? 1 : 0;
            const size_t N = _splat_data->means().shape()[0];

            // Use direct CUDA kernel to preserve tensor capacity
            mcmc::launch_update_scaling_opacity(
                sampled_idxs_i64.ptr<int64_t>(),
                new_scaling_raw.ptr<float>(),
                new_opacity_raw.ptr<float>(),
                _splat_data->scaling_raw().ptr<float>(),
                _splat_data->opacity_raw().ptr<float>(),
                sampled_idxs_i64.numel(),
                opacity_dim,
                N);
        }

        // Use fused append_gather() operation
        {
            LOG_TIMER("add_new_params_gather");
            // Gather opacity/scaling after updating them
            _optimizer->add_new_params_gather(ParamType::Means, sampled_idxs_i64);
            _optimizer->add_new_params_gather(ParamType::Sh0, sampled_idxs_i64);
            _optimizer->add_new_params_gather(ParamType::ShN, sampled_idxs_i64);
            _optimizer->add_new_params_gather(ParamType::Rotation, sampled_idxs_i64);
            _optimizer->add_new_params_gather(ParamType::Opacity, sampled_idxs_i64);
            _optimizer->add_new_params_gather(ParamType::Scaling, sampled_idxs_i64);
        }

        append_live_deleted_rows(*_splat_data, static_cast<size_t>(n_new));

        return n_new;
    }

    void MCMC::inject_noise() {
        LOG_TIMER("MCMC::inject_noise");
        using namespace lfs::core;

        // Get current learning rate from optimizer (after scheduler has updated it)
        const float current_lr = _optimizer->get_lr() * NOISE_LR;

        // Generate noise in pre-allocated buffer
        {
            LOG_TIMER("inject_noise_generate");
            if (_noise_buffer.is_valid() && _noise_buffer.capacity() > 0) {
                // Fill pre-allocated buffer with random values (kernel will use first size() elements)
                _noise_buffer.normal_(0.0f, 1.0f);
            } else {
                // Fallback for non-capacity mode
                _noise_buffer = Tensor::randn(_splat_data->means().shape(), Device::CUDA, DataType::Float32);
            }
        }

        // Call CUDA add_noise kernel (uses first size() elements of buffer)
        {
            LOG_TIMER("inject_noise_cuda_kernel");
            mcmc::launch_add_noise_kernel(
                _splat_data->opacity_raw().ptr<float>(),
                _splat_data->scaling_raw().ptr<float>(),
                _splat_data->rotation_raw().ptr<float>(),
                _noise_buffer.ptr<float>(),
                _splat_data->means().ptr<float>(),
                current_lr,
                _splat_data->size());
        }
    }

    void MCMC::post_backward(int iter, RenderOutput& render_output) {
        LOG_TIMER("MCMC::post_backward");

        // Increment SH degree every sh_degree_interval iterations
        if (iter % _params->sh_degree_interval == 0) {
            _splat_data->increment_sh_degree();
        }

        if (iter == _params->stop_refine) {
            _splat_data->_densification_info = lfs::core::Tensor::empty({0});
            _error_score_max = lfs::core::Tensor::empty({0});
            _error_score_windows = 0;
        }

        if (iter < _params->stop_refine) {
            ensure_densification_info_shape();

            // One training iteration corresponds to one camera view, so info[1] is E_k^pi.
            // Keep the max over views as the densification priority.
            const auto& info = _splat_data->_densification_info;
            if (info.is_valid() &&
                info.ndim() == 2 &&
                info.shape()[0] >= 2 &&
                info.shape()[1] == _error_score_max.numel()) {
                const float* error_row = info.ptr<float>() + info.shape()[1];
                lfs::training::mcmc::launch_elementwise_max_inplace(
                    _error_score_max.ptr<float>(),
                    error_row,
                    _error_score_max.numel());
            }

            // Clear per-view accumulators; they are rebuilt by the next backward pass.
            _splat_data->_densification_info.zero_();
        }

        // Refine Gaussians
        if (is_refining(iter)) {
            const int n_relocated = relocate_gs();
            if (n_relocated > 0) {
                LOG_DEBUG("MCMC: Relocated {} dead Gaussians at iteration {}", n_relocated, iter);
            }

            const int n_added = add_new_gs();
            if (n_added > 0) {
                LOG_DEBUG("MCMC: Added {} new Gaussians at iteration {} (total: {})",
                          n_added, iter, _splat_data->size());
            }
            // Release cached pool memory to avoid bloat (important after add_new_gs)
            lfs::core::Tensor::trim_memory_pool();

            const size_t n = static_cast<size_t>(_splat_data->size());

            if (_error_score_max.numel() < n) {
                const size_t n_new = n - _error_score_max.numel();
                _error_score_max = _error_score_max.cat(
                    lfs::core::Tensor::zeros({n_new}, _splat_data->means().device()),
                    0);
            }

            ++_error_score_windows;
            if (_error_score_windows >= 2) {
                _error_score_max = lfs::core::Tensor::zeros({n}, _splat_data->means().device());
                _error_score_windows = 0;
            }

            _splat_data->_densification_info = lfs::core::Tensor::zeros({2, n}, _splat_data->means().device());
        }

        // Inject noise to positions every iteration
        inject_noise();
    }

    void MCMC::step(int iter) {
        LOG_TIMER("MCMC::step");
        if (iter < _params->iterations) {
            {
                LOG_TIMER("step_optimizer_step");
                _optimizer->step(iter);
            }
            {
                LOG_TIMER("step_zero_grad");
                _optimizer->zero_grad(iter);
            }
            {
                LOG_TIMER("step_scheduler");
                _scheduler->step();
            }
        }
    }

    void MCMC::remove_gaussians(const lfs::core::Tensor& mask) {
        using namespace lfs::core;

        if (!mask.is_valid() || mask.numel() == 0) {
            LOG_DEBUG("MCMC: No Gaussians to remove");
            return;
        }

        const int n_remove = mask.to(DataType::Int32).sum().template item<int>();

        LOG_INFO("MCMC::remove_gaussians called: mask size={}, n_remove={}, current size={}",
                 mask.numel(), n_remove, _splat_data->size());

        if (n_remove == 0) {
            LOG_DEBUG("MCMC: No Gaussians to remove");
            return;
        }

        LOG_DEBUG("MCMC: Removing {} Gaussians", n_remove);

        const Tensor prune_indices = mask.nonzero().squeeze(-1);

        set_deleted_mask_rows(*_splat_data, prune_indices, true);

        auto zero_rotation = Tensor::zeros(
            {static_cast<size_t>(n_remove), 4},
            _splat_data->rotation_raw().device());
        _splat_data->rotation_raw().index_put_(prune_indices, zero_rotation);

        zero_optimizer_state(*_optimizer, ParamType::Means, prune_indices);
        zero_optimizer_state(*_optimizer, ParamType::Sh0, prune_indices);
        zero_optimizer_state(*_optimizer, ParamType::ShN, prune_indices);
        zero_optimizer_state(*_optimizer, ParamType::Scaling, prune_indices);
        zero_optimizer_state(*_optimizer, ParamType::Rotation, prune_indices);
        zero_optimizer_state(*_optimizer, ParamType::Opacity, prune_indices);

        if (_error_score_max.is_valid() &&
            _error_score_max.ndim() == 1 &&
            _error_score_max.numel() >= _splat_data->size()) {
            auto zeros = Tensor::zeros({static_cast<size_t>(n_remove)}, _error_score_max.device());
            _error_score_max.index_put_(prune_indices, zeros);
        }

        LOG_DEBUG("MCMC: soft-deleted {} Gaussians (rotation and optimizer state zeroed)", n_remove);
    }

    void MCMC::initialize(const lfs::core::param::OptimizationParameters& optimParams) {
        using namespace lfs::core;

        _params = std::make_unique<const lfs::core::param::OptimizationParameters>(optimParams);

        // Pre-allocate tensor capacity if max_cap is specified
        if (_params->max_cap > 0) {
            const size_t capacity = static_cast<size_t>(_params->max_cap);
            const size_t current_size = _splat_data->size();
            LOG_INFO("Pre-allocating capacity for {} Gaussians (current size: {}, utilization: {:.1f}%)",
                     capacity, current_size, 100.0f * current_size / capacity);

            try {
                // ELIMINATE ALL POOL ALLOCATIONS: Replace pool-allocated parameters with direct cudaMalloc versions
                LOG_DEBUG("  Replacing pool-allocated parameters with direct cudaMalloc versions:");

                auto replace_with_direct = [capacity](Tensor& param) {
                    // Create new tensor with direct cudaMalloc (ZERO pool usage!)
                    auto new_param = Tensor::zeros_direct(param.shape(), capacity);
                    // Copy data from old pool-allocated tensor to new direct tensor
                    cudaMemcpy(new_param.ptr<float>(), param.ptr<float>(),
                               param.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
                    // Replace (old pool-allocated tensor gets freed)
                    param = new_param;
                };

                replace_with_direct(_splat_data->means());
                replace_with_direct(_splat_data->sh0());
                if (_splat_data->shN().is_valid() && _splat_data->shN().ndim() > 0) {
                    replace_with_direct(_splat_data->shN());
                }
                replace_with_direct(_splat_data->scaling_raw());
                replace_with_direct(_splat_data->rotation_raw());
                replace_with_direct(_splat_data->opacity_raw());

                // Pre-allocate noise buffer [max_cap, 3]
                _noise_buffer = Tensor::zeros_direct(TensorShape({capacity, 3}), capacity);

                LOG_INFO("Pre-allocated capacity: {}/{} Gaussians ({:.1f}%)",
                         current_size, capacity, 100.0f * current_size / capacity);
            } catch (const std::exception& e) {
                LOG_WARN("Failed to pre-allocate capacity: {}. Continuing without pre-allocation.", e.what());
            }
        }

        _n_max = 51;
        mcmc::init_relocation_coefficients(_n_max);

        if (_params->max_cap > 0) {
            _ones_int32 = Tensor::ones({static_cast<size_t>(_params->max_cap)}, Device::CUDA, DataType::Int32);
        }

        _optimizer = create_optimizer(*_splat_data, *_params);
        _optimizer->allocate_gradients(_params->max_cap > 0 ? static_cast<size_t>(_params->max_cap) : 0);
        _scheduler = create_scheduler(*_params, *_optimizer);

        ensure_densification_info_shape();
        if (_splat_data->has_deleted_mask()) {
            ensure_deleted_mask_size(*_splat_data);
        }
        _error_score_windows = 0;

        LOG_INFO("MCMC strategy initialized with {} Gaussians", _splat_data->size());
    }

    bool MCMC::is_refining(int iter) const {
        return (iter < _params->stop_refine &&
                iter > _params->start_refine &&
                iter % _params->refine_every == 0);
    }

    // ===== Serialization =====

    namespace {
        constexpr uint32_t MCMC_MAGIC = 0x4C464D43; // "LFMC"
        constexpr uint32_t MCMC_VERSION = 1;
    } // namespace

    void MCMC::serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&MCMC_MAGIC), sizeof(MCMC_MAGIC));
        os.write(reinterpret_cast<const char*>(&MCMC_VERSION), sizeof(MCMC_VERSION));

        // Serialize optimizer state
        if (_optimizer) {
            uint8_t has_optimizer = 1;
            os.write(reinterpret_cast<const char*>(&has_optimizer), sizeof(has_optimizer));
            _optimizer->serialize(os);
        } else {
            uint8_t has_optimizer = 0;
            os.write(reinterpret_cast<const char*>(&has_optimizer), sizeof(has_optimizer));
        }

        // Serialize scheduler state
        if (_scheduler) {
            uint8_t has_scheduler = 1;
            os.write(reinterpret_cast<const char*>(&has_scheduler), sizeof(has_scheduler));
            _scheduler->serialize(os);
        } else {
            uint8_t has_scheduler = 0;
            os.write(reinterpret_cast<const char*>(&has_scheduler), sizeof(has_scheduler));
        }

        LOG_DEBUG("Serialized MCMC strategy");
    }

    void MCMC::deserialize(std::istream& is) {
        uint32_t magic, version;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != MCMC_MAGIC) {
            throw std::runtime_error("Invalid MCMC checkpoint: wrong magic");
        }
        if (version != MCMC_VERSION) {
            throw std::runtime_error("Unsupported MCMC checkpoint version: " + std::to_string(version));
        }

        // Deserialize optimizer state
        uint8_t has_optimizer;
        is.read(reinterpret_cast<char*>(&has_optimizer), sizeof(has_optimizer));
        if (has_optimizer && _optimizer) {
            _optimizer->deserialize(is);
        }

        // Deserialize scheduler state
        uint8_t has_scheduler;
        is.read(reinterpret_cast<char*>(&has_scheduler), sizeof(has_scheduler));
        if (has_scheduler && _scheduler) {
            _scheduler->deserialize(is);
        }

        LOG_DEBUG("Deserialized MCMC strategy");
    }

    void MCMC::reserve_optimizer_capacity(size_t capacity) {
        if (_optimizer) {
            _optimizer->reserve_capacity(capacity);
            LOG_INFO("Reserved optimizer capacity for {} Gaussians", capacity);
        }
    }

} // namespace lfs::training
