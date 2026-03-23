/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs.hpp"
#include "core/logger.hpp"
#include "kernels/lfs_kernels.hpp"
#include "kernels/mcmc_kernels.hpp"
#include "strategy_utils.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace lfs::training {

    LFS::LFS(lfs::core::SplatData& splat_data) : _splat_data(&splat_data) {}

    void LFS::initialize(const lfs::core::param::OptimizationParameters& optimParams) {
        using namespace lfs::core;

        _params = std::make_unique<const lfs::core::param::OptimizationParameters>(optimParams);

        if (_params->max_cap > 0) {
            const size_t capacity = static_cast<size_t>(_params->max_cap);
            const size_t current_size = _splat_data->size();
            LOG_INFO("LFS: pre-allocating capacity for {} Gaussians (current: {}, utilization: {:.1f}%)",
                     capacity, current_size, 100.0f * current_size / capacity);

            auto replace_with_direct = [capacity](Tensor& param) {
                auto new_param = Tensor::zeros_direct(param.shape(), capacity);
                cudaMemcpy(new_param.ptr<float>(), param.ptr<float>(),
                           param.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
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
        }

        _optimizer = create_optimizer(*_splat_data, *_params);
        _optimizer->allocate_gradients(_params->max_cap > 0 ? static_cast<size_t>(_params->max_cap) : 0);
        _scheduler = create_scheduler(*_params, *_optimizer);

        auto compute_gamma = [](double start, double end, const size_t steps) {
            if (steps == 0 || start <= 0.0 || end <= 0.0)
                return 1.0;
            return std::pow(end / start, 1.0 / static_cast<double>(steps));
        };
        _mean_lr_unscaled = _params->means_lr;
        _scale_lr_current = _params->scaling_lr;
        _mean_lr_gamma = compute_gamma(_params->means_lr, _params->means_lr_end, _params->iterations);
        _scale_lr_gamma = compute_gamma(_params->scaling_lr, _params->scaling_lr_end, _params->iterations);

        ensure_densification_info_shape();

        const size_t n = static_cast<size_t>(_splat_data->size());
        _refine_weight_max = Tensor::zeros({n}, _splat_data->means().device());
        _vis_count = Tensor::zeros({n}, _splat_data->means().device());

        compute_bounds();

        LOG_INFO("LFS strategy initialized with {} Gaussians", n);
    }

    void LFS::ensure_densification_info_shape() {
        const size_t n = static_cast<size_t>(_splat_data->size());
        const auto& info = _splat_data->_densification_info;
        if (!info.is_valid() ||
            info.ndim() != 2 ||
            info.shape()[0] < 2 ||
            info.shape()[1] != n) {
            _splat_data->_densification_info = lfs::core::Tensor::zeros({2, n}, _splat_data->means().device());
        }
    }

    void LFS::post_backward(int iter, RenderOutput& /*render_output*/) {
        LOG_TIMER("LFS::post_backward");
        using namespace lfs::core;

        if (iter % _params->sh_degree_interval == 0) {
            _splat_data->increment_sh_degree();
        }

        const float train_t = static_cast<float>(iter) / static_cast<float>(_params->iterations);
        if (train_t > 0.95f) {
            _splat_data->_densification_info = Tensor::empty({0});
            return;
        }

        ensure_densification_info_shape();

        const size_t n = static_cast<size_t>(_splat_data->size());
        const auto& info = _splat_data->_densification_info;

        assert(info.is_valid());
        assert(info.ndim() == 2);
        assert(info.shape()[0] >= 2);
        assert(info.shape()[1] == n);

        if (_refine_weight_max.numel() == n) {
            const float* refine_row = info.ptr<float>() + n;
            mcmc::launch_elementwise_max_inplace(
                _refine_weight_max.ptr<float>(),
                refine_row,
                n);

            const float* vis_row = info.ptr<float>();
            lfs_strategy::launch_elementwise_add_inplace(
                _vis_count.ptr<float>(),
                vis_row,
                n);
        }

        _splat_data->_densification_info.zero_();

        if (_bounds_valid) {
            inject_noise(iter);
        }

        if (is_refining(iter)) {
            refine(iter);
        }
    }

    bool LFS::is_refining(int iter) const {
        if (iter <= 0)
            return false;
        if (iter % _params->refine_every != 0)
            return false;
        const float train_t = static_cast<float>(iter) / static_cast<float>(_params->iterations);
        return train_t <= 0.95f;
    }

    void LFS::refine(int iter) {
        LOG_TIMER("LFS::refine");
        using namespace lfs::core;

        compute_bounds();

        const float max_allowed = _bounds.max_extent * 100.0f;
        const size_t n = static_cast<size_t>(_splat_data->size());

        auto opacities = _splat_data->get_opacity();
        if (opacities.ndim() == 2 && opacities.shape()[1] == 1)
            opacities = opacities.squeeze(-1);
        auto scales = _splat_data->get_scaling();
        const auto& means = _splat_data->means();

        assert(opacities.numel() == n);
        assert(scales.shape()[0] == n && scales.shape()[1] == 3);
        assert(means.shape()[0] == n && means.shape()[1] == 3);

        auto scale_min = scales.min(1);
        auto scale_max = scales.max(1);

        auto center = Tensor::from_vector(
            {_bounds.center[0], _bounds.center[1], _bounds.center[2]},
            TensorShape({1, 3}), Device::CUDA);
        auto dist_from_center = (means - center).abs().max(1);

        auto prune_mask = (opacities < (1.0f / 255.0f)) |
                          (scale_min < 1e-10f) |
                          (scale_max > max_allowed) |
                          (dist_from_center > max_allowed);

        const int pruned_count = static_cast<int>(prune_mask.sum().item());

        if (pruned_count > 0) {
            auto keep_mask = prune_mask.logical_not();
            compact_splats(keep_mask);
            LOG_DEBUG("LFS: pruned {} splats at iter {} (remaining: {})",
                      pruned_count, iter, _splat_data->size());
        }

        // Replacement should stay active even after growth stop.
        grow_and_split(iter, pruned_count);
        enforce_max_cap();
        apply_decay(iter);

        Tensor::trim_memory_pool();

        const size_t new_n = static_cast<size_t>(_splat_data->size());
        _refine_weight_max = Tensor::zeros({new_n}, _splat_data->means().device());
        _vis_count = Tensor::zeros({new_n}, _splat_data->means().device());
        _splat_data->_densification_info = Tensor::zeros({2, new_n}, _splat_data->means().device());
    }

    void LFS::grow_and_split(int iter, int pruned_count) {
        LOG_TIMER("LFS::grow_and_split");
        using namespace lfs::core;

        const size_t n = static_cast<size_t>(_splat_data->size());
        const int desired_total = static_cast<int>(
            std::round(static_cast<float>(
                           ((_refine_weight_max > _params->lfs_growth_grad_threshold) &&
                            (_vis_count > 0.0f))
                               .sum()
                               .item()) *
                       _params->lfs_growth_select_fraction));
        const int budget = (_params->max_cap > 0)
                               ? std::max(0, _params->max_cap - static_cast<int>(n))
                               : INT_MAX;
        const int n_replace = std::min(pruned_count, budget);
        int n_grow = 0;
        lfs::core::Tensor above_threshold;
        if (iter < static_cast<int>(_params->lfs_growth_stop_iter)) {
            above_threshold = (_refine_weight_max > _params->lfs_growth_grad_threshold) &&
                              (_vis_count > 0.0f);
            n_grow = std::max(0, desired_total - pruned_count);
            n_grow = std::min(n_grow, budget - n_replace);
        }
        const int total_K = n_replace + n_grow;

        if (total_K == 0)
            return;

        assert(total_K > 0);
        assert(_params->max_cap <= 0 || n + total_K <= static_cast<size_t>(_params->max_cap));

        auto seed = static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());

        Tensor split_indices;
        Tensor replace_inds;
        Tensor growth_inds;

        if (n_replace > 0) {
            auto opacities = _splat_data->get_opacity();
            if (opacities.ndim() == 2 && opacities.shape()[1] == 1)
                opacities = opacities.squeeze(-1);
            auto replace_weights = opacities * (_vis_count > 0.0f);
            replace_inds = Tensor::empty({static_cast<size_t>(n_replace)}, Device::CUDA, DataType::Int64);
            lfs_strategy::launch_gumbel_topk(
                replace_weights.ptr<float>(), n, n_replace, seed,
                replace_inds.ptr<int64_t>());
        }

        if (n_grow > 0) {
            auto growth_weights = above_threshold * _refine_weight_max;
            growth_inds = Tensor::empty({static_cast<size_t>(n_grow)}, Device::CUDA, DataType::Int64);
            lfs_strategy::launch_gumbel_topk(
                growth_weights.ptr<float>(), n, n_grow, seed + 1,
                growth_inds.ptr<int64_t>());
        }

        if (replace_inds.is_valid() && growth_inds.is_valid()) {
            split_indices = Tensor::cat({replace_inds, growth_inds}, 0);
        } else if (replace_inds.is_valid()) {
            split_indices = replace_inds;
        } else if (growth_inds.is_valid()) {
            split_indices = growth_inds;
        }

        // Match LFS behavior: union replacement+growth indices, removing overlap.
        std::vector<int64_t> split_idx_unique;
        {
            auto split_indices_cpu = split_indices.cpu();
            const int64_t* ptr = split_indices_cpu.ptr<int64_t>();
            const size_t count = split_indices_cpu.numel();

            std::unordered_set<int64_t> seen;
            seen.reserve(count);
            split_idx_unique.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                const int64_t idx = ptr[i];
                if (seen.insert(idx).second) {
                    split_idx_unique.push_back(idx);
                }
            }
        }

        if (_params->max_cap > 0) {
            const size_t max_new = static_cast<size_t>(_params->max_cap) - n;
            if (split_idx_unique.size() > max_new) {
                split_idx_unique.resize(max_new);
                LOG_INFO("LFS: clamped split count to {} for max_cap={}", max_new, _params->max_cap);
            }
        }

        if (split_idx_unique.empty())
            return;

        std::vector<int> split_idx_unique_i32;
        split_idx_unique_i32.reserve(split_idx_unique.size());
        for (const int64_t idx : split_idx_unique) {
            split_idx_unique_i32.push_back(static_cast<int>(idx));
        }
        split_indices = Tensor::from_vector(
                            split_idx_unique_i32,
                            TensorShape({split_idx_unique_i32.size()}),
                            Device::CUDA)
                            .to(DataType::Int64);

        const size_t K = split_idx_unique.size();
        const size_t sh_rest = (_splat_data->shN().is_valid() && _splat_data->shN().ndim() >= 2)
                                   ? _splat_data->shN().shape()[1]
                                   : 0;

        auto child_means = Tensor::empty({K, 3}, Device::CUDA);
        auto child_log_scales = Tensor::empty({K, 3}, Device::CUDA);
        auto child_raw_opacities = Tensor::empty({K}, Device::CUDA);
        auto child_rotations = Tensor::empty({K, 4}, Device::CUDA);
        auto child_sh0 = Tensor::empty({K, 1, 3}, Device::CUDA);
        Tensor child_shN;
        if (sh_rest > 0) {
            child_shN = Tensor::empty({K, sh_rest, 3}, Device::CUDA);
        } else {
            child_shN = Tensor::empty({K, 0, 3}, Device::CUDA);
        }

        lfs_strategy::launch_lfs_split_inplace(
            split_indices.ptr<int64_t>(),
            _splat_data->means().ptr<float>(),
            _splat_data->scaling_raw().ptr<float>(),
            _splat_data->opacity_raw().ptr<float>(),
            _splat_data->rotation_raw().ptr<float>(),
            _splat_data->sh0().ptr<float>(),
            sh_rest > 0 ? _splat_data->shN().ptr<float>() : nullptr,
            child_means.ptr<float>(),
            child_log_scales.ptr<float>(),
            child_raw_opacities.ptr<float>(),
            child_rotations.ptr<float>(),
            child_sh0.ptr<float>(),
            sh_rest > 0 ? child_shN.ptr<float>() : nullptr,
            K, sh_rest, _params->lfs_split_distance);

        if (_splat_data->opacity_raw().ndim() == 2) {
            child_raw_opacities = child_raw_opacities.unsqueeze(-1);
        }

        _optimizer->add_new_params(ParamType::Means, child_means, true);
        _optimizer->add_new_params(ParamType::Sh0, child_sh0, true);
        _optimizer->add_new_params(ParamType::ShN, child_shN, false);
        _optimizer->add_new_params(ParamType::Scaling, child_log_scales, true);
        _optimizer->add_new_params(ParamType::Rotation, child_rotations, true);
        _optimizer->add_new_params(ParamType::Opacity, child_raw_opacities, true);

        LOG_DEBUG("LFS: split {} splats at iter {} (total: {})", K, iter, _splat_data->size());
    }

    void LFS::compact_splats(const lfs::core::Tensor& keep_mask) {
        LOG_TIMER("LFS::compact_splats");
        using namespace lfs::core;

        Tensor valid_indices = keep_mask.nonzero().squeeze(-1);
        const size_t new_size = valid_indices.numel();
        const size_t cap = _params->max_cap > 0 ? static_cast<size_t>(_params->max_cap) : 0;

        auto compact = [&](Tensor& t) {
            if (!t.is_valid() || t.numel() == 0)
                return;
            auto compacted = t.index_select(0, valid_indices).contiguous();
            if (cap > 0)
                compacted.reserve(cap);
            t = std::move(compacted);
        };

        compact(_splat_data->means());
        compact(_splat_data->sh0());
        if (_splat_data->shN().is_valid() && _splat_data->shN().ndim() > 0)
            compact(_splat_data->shN());
        compact(_splat_data->scaling_raw());
        compact(_splat_data->rotation_raw());
        compact(_splat_data->opacity_raw());

        static constexpr ParamType ALL_PARAMS[] = {
            ParamType::Means, ParamType::Sh0, ParamType::ShN,
            ParamType::Scaling, ParamType::Rotation, ParamType::Opacity};

        for (auto pt : ALL_PARAMS) {
            auto* state = _optimizer->get_state_mutable(pt);
            if (!state)
                continue;
            compact(state->exp_avg);
            compact(state->exp_avg_sq);
            if (state->exp_avg.is_valid()) {
                state->grad = Tensor::zeros(state->exp_avg.shape(), state->exp_avg.device());
                if (cap > 0)
                    state->grad.reserve(cap);
            }
            state->size = new_size;
            state->capacity = cap;
        }

        if (_refine_weight_max.is_valid() && _refine_weight_max.numel() > new_size)
            _refine_weight_max = _refine_weight_max.index_select(0, valid_indices).contiguous();
        if (_vis_count.is_valid() && _vis_count.numel() > new_size)
            _vis_count = _vis_count.index_select(0, valid_indices).contiguous();
    }

    void LFS::inject_noise(int /*iter*/) {
        const size_t n = static_cast<size_t>(_splat_data->size());
        if (n == 0)
            return;

        const float lr_mean = static_cast<float>(_optimizer->get_param_lr(ParamType::Means));

        auto seed = static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());

        lfs_strategy::launch_lfs_noise_injection(
            _splat_data->means().ptr<float>(),
            _splat_data->opacity_raw().ptr<float>(),
            _vis_count.ptr<float>(),
            lr_mean,
            _params->lfs_mean_noise_weight,
            _bounds.median_size,
            n, seed);
    }

    void LFS::apply_decay(int iter) {
        const size_t n = static_cast<size_t>(_splat_data->size());
        if (n == 0)
            return;

        const float train_t = static_cast<float>(iter) / static_cast<float>(_params->iterations);

        lfs_strategy::launch_lfs_decay(
            _splat_data->opacity_raw().ptr<float>(),
            _splat_data->scaling_raw().ptr<float>(),
            _params->lfs_opac_decay,
            _params->lfs_scale_decay,
            train_t,
            n);
    }

    void LFS::enforce_max_cap() {
        if (_params->max_cap <= 0)
            return;

        using namespace lfs::core;

        const size_t n = _splat_data->size();
        const size_t cap = static_cast<size_t>(_params->max_cap);
        if (n <= cap)
            return;

        LOG_INFO("LFS: count {} exceeds max_cap {}, pruning excess", n, cap);

        auto opacities = _splat_data->get_opacity();
        if (opacities.ndim() == 2 && opacities.shape()[1] == 1)
            opacities = opacities.squeeze(-1);

        auto seed = static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());

        auto keep_indices = Tensor::empty({cap}, Device::CUDA, DataType::Int64);
        lfs_strategy::launch_gumbel_topk(
            opacities.ptr<float>(), n, cap, seed,
            keep_indices.ptr<int64_t>());

        auto keep_mask = Tensor::zeros_bool({n}, opacities.device());
        auto true_vals = Tensor::ones_bool({cap}, opacities.device());
        keep_mask.index_put_(keep_indices, true_vals);
        compact_splats(keep_mask);

        assert(_splat_data->size() <= cap);
    }

    void LFS::compute_bounds() {
        const size_t n = static_cast<size_t>(_splat_data->size());
        if (n == 0)
            return;

        lfs_strategy::launch_percentile_bounds(
            _splat_data->means().ptr<float>(),
            n,
            _params->lfs_bound_percentile,
            &_bounds);

        _bounds_valid = true;

        _optimizer->set_param_lr(ParamType::Means, _mean_lr_unscaled * _bounds.median_size);
    }

    void LFS::step(int iter) {
        LOG_TIMER("LFS::step");
        if (iter < _params->iterations) {
            _optimizer->step(iter);
            _optimizer->zero_grad(iter);

            _mean_lr_unscaled *= _mean_lr_gamma;
            _scale_lr_current *= _scale_lr_gamma;
            _optimizer->set_param_lr(ParamType::Scaling, _scale_lr_current);
            if (_bounds_valid) {
                _optimizer->set_param_lr(ParamType::Means, _mean_lr_unscaled * _bounds.median_size);
            }
        }
    }

    void LFS::remove_gaussians(const lfs::core::Tensor& mask) {
        using namespace lfs::core;

        Tensor keep_mask = mask.logical_not();
        Tensor keep_indices = keep_mask.nonzero().squeeze(-1);
        const size_t old_size = static_cast<size_t>(_splat_data->size());
        const int n_remove = static_cast<int>(old_size - keep_indices.numel());

        LOG_INFO("LFS::remove_gaussians: mask size={}, n_remove={}, current size={}",
                 mask.numel(), n_remove, _splat_data->size());

        if (n_remove == 0)
            return;

        _splat_data->means() = _splat_data->means().index_select(0, keep_indices).contiguous();
        _splat_data->sh0() = _splat_data->sh0().index_select(0, keep_indices).contiguous();
        if (_splat_data->shN().is_valid()) {
            _splat_data->shN() = _splat_data->shN().index_select(0, keep_indices).contiguous();
        }
        _splat_data->scaling_raw() = _splat_data->scaling_raw().index_select(0, keep_indices).contiguous();
        _splat_data->rotation_raw() = _splat_data->rotation_raw().index_select(0, keep_indices).contiguous();
        _splat_data->opacity_raw() = _splat_data->opacity_raw().index_select(0, keep_indices).contiguous();

        const auto& info = _splat_data->_densification_info;
        if (info.is_valid() && info.ndim() == 2 && info.shape()[1] == old_size) {
            _splat_data->_densification_info = info.index_select(1, keep_indices).contiguous();
        }
        if (_refine_weight_max.is_valid() && _refine_weight_max.numel() == old_size) {
            _refine_weight_max = _refine_weight_max.index_select(0, keep_indices).contiguous();
        }
        if (_vis_count.is_valid() && _vis_count.numel() == old_size) {
            _vis_count = _vis_count.index_select(0, keep_indices).contiguous();
        }

        _optimizer = create_optimizer(*_splat_data, *_params);
        _scheduler = create_scheduler(*_params, *_optimizer);
    }

    namespace {
        constexpr uint32_t LFS_MAGIC = 0x4C464252; // "LFBR"
        constexpr uint32_t LFS_VERSION = 1;
    } // namespace

    void LFS::serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&LFS_MAGIC), sizeof(LFS_MAGIC));
        os.write(reinterpret_cast<const char*>(&LFS_VERSION), sizeof(LFS_VERSION));

        if (_optimizer) {
            uint8_t has_optimizer = 1;
            os.write(reinterpret_cast<const char*>(&has_optimizer), sizeof(has_optimizer));
            _optimizer->serialize(os);
        } else {
            uint8_t has_optimizer = 0;
            os.write(reinterpret_cast<const char*>(&has_optimizer), sizeof(has_optimizer));
        }

        if (_scheduler) {
            uint8_t has_scheduler = 1;
            os.write(reinterpret_cast<const char*>(&has_scheduler), sizeof(has_scheduler));
            _scheduler->serialize(os);
        } else {
            uint8_t has_scheduler = 0;
            os.write(reinterpret_cast<const char*>(&has_scheduler), sizeof(has_scheduler));
        }
    }

    void LFS::deserialize(std::istream& is) {
        uint32_t magic, version;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != LFS_MAGIC)
            throw std::runtime_error("Invalid LFS checkpoint: wrong magic");
        if (version != LFS_VERSION)
            throw std::runtime_error("Unsupported LFS checkpoint version: " + std::to_string(version));

        uint8_t has_optimizer;
        is.read(reinterpret_cast<char*>(&has_optimizer), sizeof(has_optimizer));
        if (has_optimizer && _optimizer)
            _optimizer->deserialize(is);

        uint8_t has_scheduler;
        is.read(reinterpret_cast<char*>(&has_scheduler), sizeof(has_scheduler));
        if (has_scheduler && _scheduler)
            _scheduler->deserialize(is);
    }

    void LFS::reserve_optimizer_capacity(size_t capacity) {
        if (_optimizer) {
            _optimizer->reserve_capacity(capacity);
            LOG_INFO("LFS: reserved optimizer capacity for {} Gaussians", capacity);
        }
    }

} // namespace lfs::training
