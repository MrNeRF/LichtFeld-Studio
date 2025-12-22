/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "default_strategy.hpp"
#include "core/logger.hpp"
#include "core/parameters.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"
#include "kernels/densification_kernels.hpp"
#include "optimizer/render_output.hpp"
#include "strategy_utils.hpp"

namespace lfs::training {

    namespace {
        // Returns true if shape has any zero dimension (e.g., ShN at sh-degree 0)
        [[nodiscard]] inline bool has_zero_dimension(const lfs::core::TensorShape& shape) {
            for (size_t i = 0; i < shape.rank(); ++i) {
                if (shape[i] == 0)
                    return true;
            }
            return false;
        }

        // Returns true if shN tensor has non-zero coefficients
        [[nodiscard]] inline bool has_shN_coefficients(const lfs::core::Tensor& shN) {
            return shN.is_valid() && shN.ndim() >= 2 && shN.shape()[1] > 0;
        }
    } // anonymous namespace

    DefaultStrategy::DefaultStrategy(lfs::core::SplatData& splat_data) : _splat_data(&splat_data) {}

    void DefaultStrategy::initialize(const lfs::core::param::OptimizationParameters& optimParams) {
        _params = std::make_unique<const lfs::core::param::OptimizationParameters>(optimParams);

        initialize_gaussians(*_splat_data, _params->max_cap);

        _optimizer = create_optimizer(*_splat_data, *_params);
        _optimizer->allocate_gradients(_params->max_cap > 0 ? static_cast<size_t>(_params->max_cap) : 0);
        _scheduler = create_scheduler(*_params, *_optimizer);

        // Initialize densification info: [2, N] tensor for tracking gradients
        _splat_data->_densification_info = lfs::core::Tensor::zeros(
            {2, static_cast<size_t>(_splat_data->size())},
            _splat_data->means().device());

        // Initialize free mask: all slots are active (not free)
        const size_t capacity = _params->max_cap > 0 ? static_cast<size_t>(_params->max_cap)
                                                     : static_cast<size_t>(_splat_data->size());
        _free_mask = lfs::core::Tensor::zeros_bool({capacity}, _splat_data->means().device());
    }

    bool DefaultStrategy::is_refining(int iter) const {
        return (iter < _params->stop_refine &&
                iter > _params->start_refine &&
                iter % _params->refine_every == 0 &&
                iter % _params->reset_every >= _params->pause_refine_after_reset);
    }

    void DefaultStrategy::remove_gaussians(const lfs::core::Tensor& mask) {
        int mask_sum = mask.to(lfs::core::DataType::Int32).sum().template item<int>();

        if (mask_sum == 0) {
            LOG_DEBUG("No Gaussians to remove");
            return;
        }

        LOG_DEBUG("Removing {} Gaussians", mask_sum);
        remove(mask);
    }

    void DefaultStrategy::duplicate(const lfs::core::Tensor& is_duplicated) {
        const lfs::core::Tensor sampled_idxs = is_duplicated.nonzero().squeeze(-1);
        const int64_t num_duplicated = sampled_idxs.shape()[0];

        if (num_duplicated == 0) {
            return; // Nothing to duplicate
        }

        // Try to fill free slots first (in-place with index_put_)
        auto [filled_indices, remaining] = fill_free_slots(sampled_idxs, num_duplicated);
        const int64_t num_filled = num_duplicated - remaining;

        LOG_DEBUG("duplicate(): {} total, {} filled free slots, {} to append", num_duplicated, num_filled, remaining);

        // Append remaining Gaussians in-place
        if (remaining > 0) {
            const auto append_src_indices = sampled_idxs.slice(0, num_filled, num_duplicated);

            // In-place append
            _splat_data->means().append_gather(append_src_indices);
            _splat_data->rotation_raw().append_gather(append_src_indices);
            _splat_data->scaling_raw().append_gather(append_src_indices);
            _splat_data->sh0().append_gather(append_src_indices);
            _splat_data->opacity_raw().append_gather(append_src_indices);

            auto& shN = _splat_data->shN();
            if (has_shN_coefficients(shN)) {
                shN.append_gather(append_src_indices);
            }

            // Initialize optimizer states with zeros
            const size_t n_new = static_cast<size_t>(remaining);
            _optimizer->extend_state_for_new_params(ParamType::Means, n_new);
            _optimizer->extend_state_for_new_params(ParamType::Rotation, n_new);
            _optimizer->extend_state_for_new_params(ParamType::Scaling, n_new);
            _optimizer->extend_state_for_new_params(ParamType::Sh0, n_new);
            _optimizer->extend_state_for_new_params(ParamType::ShN, n_new);
            _optimizer->extend_state_for_new_params(ParamType::Opacity, n_new);
        }
    }

    void DefaultStrategy::split(const lfs::core::Tensor& is_split) {
        const lfs::core::Tensor split_idxs = is_split.nonzero().squeeze(-1);
        const int64_t num_split = split_idxs.shape()[0];

        if (num_split == 0) {
            return; // Nothing to split
        }

        LOG_DEBUG("split(): {} Gaussians to split", num_split);

        // Get SH dimensions
        const bool has_shN = _splat_data->shN().is_valid();
        int shN_dim = 0;
        if (has_shN) {
            const auto& shN_shape = _splat_data->shN().shape();
            if (shN_shape.rank() == 2) {
                shN_dim = shN_shape[1];
            } else if (shN_shape.rank() == 3) {
                shN_dim = shN_shape[1] * shN_shape[2];
            }
        }

        const auto device = _splat_data->means().device();

        // Generate random noise [2, num_split, 3]
        const lfs::core::Tensor random_noise = lfs::core::Tensor::randn(
            {2, static_cast<size_t>(num_split), 3}, device);

        // Allocate temporary tensors for second split results [num_split, ...]
        auto second_positions = lfs::core::Tensor::empty({static_cast<size_t>(num_split), 3}, device);
        auto second_rotations = lfs::core::Tensor::empty({static_cast<size_t>(num_split), 4}, device);
        auto second_scales = lfs::core::Tensor::empty({static_cast<size_t>(num_split), 3}, device);
        auto second_sh0 = lfs::core::Tensor::empty({static_cast<size_t>(num_split), 3}, device);
        lfs::core::Tensor second_shN;
        if (has_shN) {
            second_shN = lfs::core::Tensor::empty({static_cast<size_t>(num_split), static_cast<size_t>(shN_dim)}, device);
        }
        auto second_opacities = lfs::core::Tensor::empty({static_cast<size_t>(num_split)}, device);

        // Skip shN pointers when shN_dim=0 (sh-degree 0)
        const bool use_shN = has_shN && shN_dim > 0;

        // First result modifies in-place, second goes to temporaries
        kernels::launch_split_gaussians_inplace(
            _splat_data->means().ptr<float>(),
            _splat_data->rotation_raw().ptr<float>(),
            _splat_data->scaling_raw().ptr<float>(),
            _splat_data->sh0().ptr<float>(),
            use_shN ? _splat_data->shN().ptr<float>() : nullptr,
            _splat_data->opacity_raw().ptr<float>(),
            second_positions.ptr<float>(),
            second_rotations.ptr<float>(),
            second_scales.ptr<float>(),
            second_sh0.ptr<float>(),
            use_shN ? second_shN.ptr<float>() : nullptr,
            second_opacities.ptr<float>(),
            split_idxs.ptr<int64_t>(),
            random_noise.ptr<float>(),
            static_cast<int>(num_split),
            shN_dim,
            _params->revised_opacity,
            nullptr);

        // Reset optimizer states for split indices
        auto reset_optimizer_state_at_indices = [&](ParamType param_type) {
            auto* state = _optimizer->get_state_mutable(param_type);
            if (!state)
                return;

            const auto& shape = state->exp_avg.shape();
            if (has_zero_dimension(shape))
                return;

            std::vector<size_t> dims = {static_cast<size_t>(num_split)};
            for (size_t i = 1; i < shape.rank(); ++i) {
                dims.push_back(shape[i]);
            }
            auto zeros = lfs::core::Tensor::zeros(lfs::core::TensorShape(dims), state->exp_avg.device());

            state->exp_avg.index_put_(split_idxs, zeros);
            state->exp_avg_sq.index_put_(split_idxs, zeros);
            if (state->grad.is_valid()) {
                state->grad.index_put_(split_idxs, zeros);
            }
        };

        reset_optimizer_state_at_indices(ParamType::Means);
        reset_optimizer_state_at_indices(ParamType::Rotation);
        reset_optimizer_state_at_indices(ParamType::Scaling);
        reset_optimizer_state_at_indices(ParamType::Sh0);
        reset_optimizer_state_at_indices(ParamType::ShN);
        reset_optimizer_state_at_indices(ParamType::Opacity);

        // Now place second split results: fill free slots first, then append
        // Try to fill free slots first
        auto [filled_indices, remaining] = fill_free_slots_with_data(
            second_positions, second_rotations, second_scales,
            second_sh0, second_shN, second_opacities, num_split);

        const int64_t num_filled = num_split - remaining;

        // Append remaining second results
        if (remaining > 0) {
            const size_t old_size = static_cast<size_t>(_splat_data->size());
            const size_t n_remaining = static_cast<size_t>(remaining);

            // Get the remaining data
            const auto append_positions = second_positions.slice(0, num_filled, num_split);
            const auto append_rotations = second_rotations.slice(0, num_filled, num_split);
            const auto append_scales = second_scales.slice(0, num_filled, num_split);
            const auto append_sh0_flat = second_sh0.slice(0, num_filled, num_split);
            const auto append_opacities = second_opacities.slice(0, num_filled, num_split);

            // Create indices for new rows
            std::vector<int> new_indices_vec(n_remaining);
            for (size_t i = 0; i < n_remaining; ++i) {
                new_indices_vec[i] = static_cast<int>(old_size + i);
            }
            const auto new_indices = lfs::core::Tensor::from_vector(
                new_indices_vec, lfs::core::TensorShape({n_remaining}), device);

            // Extend and write data
            _splat_data->means().append_zeros(n_remaining);
            _splat_data->means().index_put_(new_indices, append_positions);

            _splat_data->rotation_raw().append_zeros(n_remaining);
            _splat_data->rotation_raw().index_put_(new_indices, append_rotations);

            _splat_data->scaling_raw().append_zeros(n_remaining);
            _splat_data->scaling_raw().index_put_(new_indices, append_scales);

            const auto append_sh0_reshaped = append_sh0_flat.reshape(
                lfs::core::TensorShape({n_remaining, 1, 3}));
            _splat_data->sh0().append_zeros(n_remaining);
            _splat_data->sh0().index_put_(new_indices, append_sh0_reshaped);

            _splat_data->opacity_raw().append_zeros(n_remaining);
            _splat_data->opacity_raw().index_put_(new_indices, append_opacities);

            if (use_shN) {
                auto append_shN = second_shN.slice(0, num_filled, num_split);
                const auto& shN_shape = _splat_data->shN().shape();
                if (shN_shape.rank() == 3) {
                    append_shN = append_shN.reshape(
                        lfs::core::TensorShape({n_remaining, shN_shape[1], shN_shape[2]}));
                }
                _splat_data->shN().append_zeros(n_remaining);
                _splat_data->shN().index_put_(new_indices, append_shN);
            }

            // Update optimizer states
            _optimizer->extend_state_for_new_params(ParamType::Means, n_remaining);
            _optimizer->extend_state_for_new_params(ParamType::Rotation, n_remaining);
            _optimizer->extend_state_for_new_params(ParamType::Scaling, n_remaining);
            _optimizer->extend_state_for_new_params(ParamType::Sh0, n_remaining);
            _optimizer->extend_state_for_new_params(ParamType::ShN, n_remaining);
            _optimizer->extend_state_for_new_params(ParamType::Opacity, n_remaining);
        }

        LOG_DEBUG("split(): done, {} filled free slots, {} appended", num_filled, remaining);
    }

    std::pair<lfs::core::Tensor, int64_t> DefaultStrategy::fill_free_slots_with_data(
        const lfs::core::Tensor& positions,
        const lfs::core::Tensor& rotations,
        const lfs::core::Tensor& scales,
        const lfs::core::Tensor& sh0,
        const lfs::core::Tensor& shN,
        const lfs::core::Tensor& opacities,
        int64_t count) {

        if (!_free_mask.is_valid() || count == 0) {
            return {lfs::core::Tensor(), count};
        }

        const size_t current_size = static_cast<size_t>(_splat_data->size());

        // Find free slot indices within current size
        auto active_region = _free_mask.slice(0, 0, current_size);
        auto free_indices = active_region.nonzero().squeeze(-1);
        const int64_t num_free = free_indices.numel();

        if (num_free == 0) {
            return {lfs::core::Tensor(), count};
        }

        const int64_t slots_to_fill = std::min(count, num_free);
        auto target_indices = free_indices.slice(0, 0, slots_to_fill);

        // Copy data to free slots
        _splat_data->means().index_put_(target_indices, positions.slice(0, 0, slots_to_fill));
        _splat_data->rotation_raw().index_put_(target_indices, rotations.slice(0, 0, slots_to_fill));
        _splat_data->scaling_raw().index_put_(target_indices, scales.slice(0, 0, slots_to_fill));

        // sh0 needs reshape from [slots_to_fill, 3] to [slots_to_fill, 1, 3]
        auto sh0_reshaped = sh0.slice(0, 0, slots_to_fill).reshape(lfs::core::TensorShape({static_cast<size_t>(slots_to_fill), 1, 3}));
        _splat_data->sh0().index_put_(target_indices, sh0_reshaped);

        _splat_data->opacity_raw().index_put_(target_indices, opacities.slice(0, 0, slots_to_fill));

        if (shN.is_valid() && has_shN_coefficients(_splat_data->shN())) {
            const auto& shN_shape = _splat_data->shN().shape();
            auto shN_slice = shN.slice(0, 0, slots_to_fill).reshape(lfs::core::TensorShape({static_cast<size_t>(slots_to_fill), shN_shape[1], shN_shape[2]}));
            _splat_data->shN().index_put_(target_indices, shN_slice);
        }

        // Reset optimizer states for filled slots
        auto reset_optimizer_state = [&](ParamType param_type) {
            auto* state = _optimizer->get_state_mutable(param_type);
            if (!state)
                return;

            const auto& shape = state->exp_avg.shape();
            if (has_zero_dimension(shape))
                return;

            std::vector<size_t> dims = {static_cast<size_t>(slots_to_fill)};
            for (size_t i = 1; i < shape.rank(); ++i) {
                dims.push_back(shape[i]);
            }
            auto zeros = lfs::core::Tensor::zeros(lfs::core::TensorShape(dims), state->exp_avg.device());

            state->exp_avg.index_put_(target_indices, zeros);
            state->exp_avg_sq.index_put_(target_indices, zeros);
            if (state->grad.is_valid()) {
                state->grad.index_put_(target_indices, zeros);
            }
        };

        reset_optimizer_state(ParamType::Means);
        reset_optimizer_state(ParamType::Rotation);
        reset_optimizer_state(ParamType::Scaling);
        reset_optimizer_state(ParamType::Sh0);
        reset_optimizer_state(ParamType::ShN);
        reset_optimizer_state(ParamType::Opacity);

        // Mark filled slots as active
        auto false_vals = lfs::core::Tensor::zeros_bool({static_cast<size_t>(slots_to_fill)}, target_indices.device());
        _free_mask.index_put_(target_indices, false_vals);

        return {target_indices, count - slots_to_fill};
    }

    void DefaultStrategy::grow_gs(int iter) {
        lfs::core::Tensor numer = _splat_data->_densification_info[1];
        lfs::core::Tensor denom = _splat_data->_densification_info[0];
        const lfs::core::Tensor grads = numer / denom.clamp_min(1.0f);

        lfs::core::Tensor is_grad_high = grads > _params->grad_threshold;

        // Exclude free slots from consideration
        const size_t current_size = static_cast<size_t>(_splat_data->size());
        if (_free_mask.is_valid() && current_size > 0) {
            auto active_free_mask = _free_mask.slice(0, 0, current_size);
            auto is_active = active_free_mask.logical_not(); // true = slot is active (not free)
            is_grad_high = is_grad_high.logical_and(is_active);
        }

        // Get max along last dimension
        const lfs::core::Tensor max_values = _splat_data->get_scaling().max(-1, false);
        const lfs::core::Tensor is_small = max_values <= _params->grow_scale3d * _splat_data->get_scene_scale();
        lfs::core::Tensor is_duplicated = is_grad_high.logical_and(is_small);

        auto num_duplicates = static_cast<int64_t>(is_duplicated.sum_scalar());

        const lfs::core::Tensor is_large = is_small.logical_not();
        lfs::core::Tensor is_split = is_grad_high.logical_and(is_large);
        auto num_split = static_cast<int64_t>(is_split.sum_scalar());

        // Enforce max_cap: limit growth to stay within capacity
        if (_params->max_cap > 0) {
            const int current_n = _splat_data->size();
            // Duplication adds num_duplicates, split replaces num_split with 2*num_split (net +num_split)
            const int64_t potential_new = num_duplicates + num_split;
            const int64_t available = std::max(0, _params->max_cap - current_n);

            if (potential_new > available) {
                // Need to limit - prioritize duplication over split (duplicates small Gaussians)
                if (num_duplicates >= available) {
                    // Can only do partial duplication, no split
                    num_duplicates = available;
                    num_split = 0;
                    // Limit is_duplicated to first 'available' true values
                    auto indices = is_duplicated.nonzero().squeeze(-1);
                    if (indices.numel() > available) {
                        auto keep_indices = indices.slice(0, 0, available);
                        is_duplicated = lfs::core::Tensor::zeros_bool({static_cast<size_t>(current_n)}, is_duplicated.device());
                        auto true_vals = lfs::core::Tensor::ones_bool({static_cast<size_t>(available)}, is_duplicated.device());
                        is_duplicated.index_put_(keep_indices, true_vals);
                    }
                    is_split = lfs::core::Tensor::zeros_bool({static_cast<size_t>(current_n)}, is_split.device());
                } else {
                    // Do all duplications, limit splits
                    const int64_t remaining = available - num_duplicates;
                    num_split = remaining;
                    // Limit is_split to first 'remaining' true values
                    auto indices = is_split.nonzero().squeeze(-1);
                    if (indices.numel() > remaining) {
                        auto keep_indices = indices.slice(0, 0, remaining);
                        is_split = lfs::core::Tensor::zeros_bool({static_cast<size_t>(current_n)}, is_split.device());
                        auto true_vals = lfs::core::Tensor::ones_bool({static_cast<size_t>(remaining)}, is_split.device());
                        is_split.index_put_(keep_indices, true_vals);
                    }
                }
                LOG_DEBUG("max_cap enforcement: limited growth from {} to {} new Gaussians", potential_new, available);
            }
        }

        LOG_DEBUG("grow_gs(): {} duplicates, {} splits", num_duplicates, num_split);

        // First duplicate
        if (num_duplicates > 0) {
            duplicate(is_duplicated);
        }

        // New Gaussians added by duplication will not be split
        auto zeros_to_concat = lfs::core::Tensor::zeros_bool({static_cast<size_t>(num_duplicates)}, is_split.device());
        is_split = is_split.cat(zeros_to_concat, 0);

        if (num_split > 0) {
            split(is_split);
        }
    }

    void DefaultStrategy::remove(const lfs::core::Tensor& is_prune) {
        // Soft deletion: mark slots as free instead of resizing tensors
        // This avoids expensive tensor reallocations during training
        const lfs::core::Tensor prune_indices = is_prune.nonzero().squeeze(-1);
        const int64_t num_pruned = prune_indices.numel();

        if (num_pruned == 0) {
            return;
        }

        // Mark pruned slots as free
        mark_as_free(prune_indices);

        // Zero out quaternion to trigger early exit in preprocessing kernel
        // The rasterizer checks: if (q_norm_sq < 1e-8f) active = false
        // This happens BEFORE expensive covariance computation and gradient computation
        auto zero_rotation = lfs::core::Tensor::zeros(
            {static_cast<size_t>(num_pruned), 4},
            _splat_data->rotation_raw().device());
        _splat_data->rotation_raw().index_put_(prune_indices, zero_rotation);

        // Zero optimizer states in-place (preserves capacity)
        auto zero_optimizer_state = [&](ParamType param_type) {
            auto* state = _optimizer->get_state_mutable(param_type);
            if (!state)
                return;

            const auto& shape = state->exp_avg.shape();
            if (has_zero_dimension(shape))
                return;

            std::vector<size_t> dims = {static_cast<size_t>(num_pruned)};
            for (size_t i = 1; i < shape.rank(); ++i) {
                dims.push_back(shape[i]);
            }
            auto zeros = lfs::core::Tensor::zeros(lfs::core::TensorShape(dims), state->exp_avg.device());

            // Modify in-place to preserve capacity
            state->exp_avg.index_put_(prune_indices, zeros);
            state->exp_avg_sq.index_put_(prune_indices, zeros);
            if (state->grad.is_valid()) {
                state->grad.index_put_(prune_indices, zeros);
            }
        };

        zero_optimizer_state(ParamType::Means);
        zero_optimizer_state(ParamType::Rotation);
        zero_optimizer_state(ParamType::Scaling);
        zero_optimizer_state(ParamType::Sh0);
        zero_optimizer_state(ParamType::ShN);
        zero_optimizer_state(ParamType::Opacity);

        LOG_DEBUG("remove(): soft-deleted {} Gaussians (marked as free, rotation & gradients zeroed)", num_pruned);
    }

    void DefaultStrategy::prune_gs(int iter) {
        // Check for low opacity
        lfs::core::Tensor is_prune = _splat_data->get_opacity() < _params->prune_opacity;

        auto rotation_raw = _splat_data->rotation_raw();
        is_prune = is_prune.logical_or((rotation_raw * rotation_raw).sum(-1, false) < 1e-8f);

        // Check for too large Gaussians
        if (iter > _params->reset_every) {
            const lfs::core::Tensor max_values = _splat_data->get_scaling().max(-1, false);
            lfs::core::Tensor is_too_big = max_values > _params->prune_scale3d * _splat_data->get_scene_scale();
            is_prune = is_prune.logical_or(is_too_big);
        }

        // Exclude already-free slots from pruning (they're already soft-deleted)
        const size_t current_size = static_cast<size_t>(_splat_data->size());
        if (_free_mask.is_valid() && current_size > 0) {
            auto active_free_mask = _free_mask.slice(0, 0, current_size);
            auto is_active = active_free_mask.logical_not(); // true = slot is active
            is_prune = is_prune.logical_and(is_active);      // only prune active slots
        }

        const auto num_prunes = static_cast<int64_t>(is_prune.sum_scalar());
        if (num_prunes > 0) {
            remove(is_prune);
        }
    }

    void DefaultStrategy::reset_opacity() {
        const float threshold = 2.0f * _params->prune_opacity;
        const float logit_threshold = std::log(threshold / (1.0f - threshold));

        // In-place ops preserve capacity
        _splat_data->opacity_raw().clamp_max_(logit_threshold);

        auto* state = _optimizer->get_state_mutable(ParamType::Opacity);
        if (state) {
            state->exp_avg.zero_();
            state->exp_avg_sq.zero_();
        }
    }

    void DefaultStrategy::post_backward(int iter, RenderOutput& render_output) {
        // Increment SH degree every 1000 iterations
        if (iter % _params->sh_degree_interval == 0) {
            _splat_data->increment_sh_degree();
        }

        if (iter == _params->stop_refine) {
            // Reset densification info at the end of refinement. Saves memory and processing time.
            _splat_data->_densification_info = lfs::core::Tensor::empty({0});
        }

        if (iter >= _params->stop_refine) {
            return;
        }

        if (is_refining(iter)) {
            // Reinit if invalid (e.g., checkpoint resume with extended stop_refine)
            const auto& info = _splat_data->_densification_info;
            const size_t n = static_cast<size_t>(_splat_data->size());
            if (!info.is_valid() || info.ndim() != 2 || info.shape()[1] != n) {
                _splat_data->_densification_info = lfs::core::Tensor::zeros({2, n}, _splat_data->means().device());
            }
            grow_gs(iter);
            prune_gs(iter);

            _splat_data->_densification_info = lfs::core::Tensor::zeros(
                {2, static_cast<size_t>(_splat_data->size())},
                _splat_data->means().device());
        }

        if (iter % _params->reset_every == 0 && iter > 0) {
            reset_opacity();
        }
    }

    void DefaultStrategy::step(int iter) {
        if (iter < _params->iterations) {
            _optimizer->step(iter);
            _optimizer->zero_grad(iter);
            _scheduler->step();
        }
    }

    // ===== Serialization =====

    namespace {
        constexpr uint32_t DEFAULT_MAGIC = 0x4C464446; // "LFDF"
        constexpr uint32_t DEFAULT_VERSION = 2;        // v2 adds free_mask serialization
    } // namespace

    void DefaultStrategy::serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&DEFAULT_MAGIC), sizeof(DEFAULT_MAGIC));
        os.write(reinterpret_cast<const char*>(&DEFAULT_VERSION), sizeof(DEFAULT_VERSION));

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

        // Serialize free mask (v2+)
        if (_free_mask.is_valid()) {
            uint8_t has_free_mask = 1;
            os.write(reinterpret_cast<const char*>(&has_free_mask), sizeof(has_free_mask));
            os << _free_mask;
        } else {
            uint8_t has_free_mask = 0;
            os.write(reinterpret_cast<const char*>(&has_free_mask), sizeof(has_free_mask));
        }

        LOG_DEBUG("Serialized DefaultStrategy");
    }

    void DefaultStrategy::deserialize(std::istream& is) {
        uint32_t magic, version;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != DEFAULT_MAGIC) {
            throw std::runtime_error("Invalid DefaultStrategy checkpoint: wrong magic");
        }
        if (version > DEFAULT_VERSION) {
            throw std::runtime_error("Unsupported DefaultStrategy checkpoint version: " + std::to_string(version));
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

        // Deserialize free mask (v2+)
        if (version >= 2) {
            uint8_t has_free_mask;
            is.read(reinterpret_cast<char*>(&has_free_mask), sizeof(has_free_mask));
            if (has_free_mask) {
                is >> _free_mask;
                // Tensor deserialization loads to CPU; move to match splat data device
                if (_splat_data->means().device() == lfs::core::Device::CUDA) {
                    _free_mask = _free_mask.cuda();
                }
            }
        }

        LOG_DEBUG("Deserialized DefaultStrategy (version {})", version);
    }

    void DefaultStrategy::reserve_optimizer_capacity(size_t capacity) {
        if (_optimizer) {
            _optimizer->reserve_capacity(capacity);
            LOG_INFO("Reserved optimizer capacity for {} Gaussians", capacity);
        }
    }

    size_t DefaultStrategy::active_count() const {
        if (!_free_mask.is_valid()) {
            return static_cast<size_t>(_splat_data->size());
        }
        // Count slots that are NOT free (i.e., active)
        // Only count up to the current size (not full capacity)
        const size_t current_size = static_cast<size_t>(_splat_data->size());
        if (current_size == 0)
            return 0;

        auto active_region = _free_mask.slice(0, 0, current_size);
        auto free_count_val = static_cast<size_t>(active_region.sum_scalar());
        return current_size - free_count_val;
    }

    size_t DefaultStrategy::free_count() const {
        if (!_free_mask.is_valid()) {
            return 0;
        }
        // Count free slots within current size
        const size_t current_size = static_cast<size_t>(_splat_data->size());
        if (current_size == 0)
            return 0;

        auto active_region = _free_mask.slice(0, 0, current_size);
        return static_cast<size_t>(active_region.sum_scalar());
    }

    lfs::core::Tensor DefaultStrategy::get_active_indices() const {
        const size_t current_size = static_cast<size_t>(_splat_data->size());
        if (current_size == 0) {
            return lfs::core::Tensor();
        }

        if (!_free_mask.is_valid() || free_count() == 0) {
            // No free mask or no free slots means all slots are active
            // Create all indices using ones_bool -> nonzero
            auto all_active = lfs::core::Tensor::ones_bool({current_size}, _splat_data->means().device());
            return all_active.nonzero().squeeze(-1);
        }

        // Return indices where free_mask is false (i.e., active)
        auto active_region = _free_mask.slice(0, 0, current_size);
        auto is_active = active_region.logical_not();
        return is_active.nonzero().squeeze(-1);
    }

    void DefaultStrategy::mark_as_free(const lfs::core::Tensor& indices) {
        if (!_free_mask.is_valid() || indices.numel() == 0) {
            return;
        }
        // Mark the given indices as free
        auto true_vals = lfs::core::Tensor::ones_bool({static_cast<size_t>(indices.numel())}, indices.device());
        _free_mask.index_put_(indices, true_vals);
    }

    std::pair<lfs::core::Tensor, int64_t> DefaultStrategy::fill_free_slots(
        const lfs::core::Tensor& source_indices, int64_t count) {

        if (!_free_mask.is_valid() || count == 0) {
            // No free slot tracking, all need to be appended
            return {lfs::core::Tensor(), count};
        }

        const size_t current_size = static_cast<size_t>(_splat_data->size());

        // Find free slot indices within current size
        auto active_region = _free_mask.slice(0, 0, current_size);
        auto free_indices = active_region.nonzero().squeeze(-1);
        const int64_t num_free = free_indices.numel();

        if (num_free == 0) {
            // No free slots available
            return {lfs::core::Tensor(), count};
        }

        // Use min(count, num_free) slots
        const int64_t slots_to_fill = std::min(count, num_free);
        auto target_indices = free_indices.slice(0, 0, slots_to_fill);
        auto src_indices = source_indices.slice(0, 0, slots_to_fill);

        // Copy data from source to target slots
        _splat_data->means().index_put_(target_indices, _splat_data->means().index_select(0, src_indices));
        _splat_data->rotation_raw().index_put_(target_indices, _splat_data->rotation_raw().index_select(0, src_indices));
        _splat_data->scaling_raw().index_put_(target_indices, _splat_data->scaling_raw().index_select(0, src_indices));
        _splat_data->sh0().index_put_(target_indices, _splat_data->sh0().index_select(0, src_indices));
        _splat_data->opacity_raw().index_put_(target_indices, _splat_data->opacity_raw().index_select(0, src_indices));

        auto& shN = _splat_data->shN();
        if (has_shN_coefficients(shN)) {
            shN.index_put_(target_indices, shN.index_select(0, src_indices));
        }

        // Reset optimizer states in-place (preserves capacity)
        auto update_optimizer_state = [&](ParamType param_type) {
            auto* state = _optimizer->get_state_mutable(param_type);
            if (!state)
                return;

            const auto& shape = state->exp_avg.shape();
            if (has_zero_dimension(shape))
                return;

            std::vector<size_t> dims = {static_cast<size_t>(slots_to_fill)};
            for (size_t i = 1; i < shape.rank(); ++i) {
                dims.push_back(shape[i]);
            }
            auto zeros = lfs::core::Tensor::zeros(lfs::core::TensorShape(dims), state->exp_avg.device());

            state->exp_avg.index_put_(target_indices, zeros);
            state->exp_avg_sq.index_put_(target_indices, zeros);
        };

        update_optimizer_state(ParamType::Means);
        update_optimizer_state(ParamType::Rotation);
        update_optimizer_state(ParamType::Scaling);
        update_optimizer_state(ParamType::Sh0);
        update_optimizer_state(ParamType::ShN);
        update_optimizer_state(ParamType::Opacity);

        // Mark filled slots as active (not free)
        auto false_vals = lfs::core::Tensor::zeros_bool({static_cast<size_t>(slots_to_fill)}, target_indices.device());
        _free_mask.index_put_(target_indices, false_vals);

        const int64_t remaining = count - slots_to_fill;
        LOG_DEBUG("fill_free_slots: filled {} slots, {} remaining to append", slots_to_fill, remaining);

        return {target_indices, remaining};
    }

} // namespace lfs::training
