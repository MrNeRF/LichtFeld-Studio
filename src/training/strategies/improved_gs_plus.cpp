/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "improved_gs_plus.hpp"

#include "edge_rasterizer.hpp"
#include "gsplat_rasterizer.hpp"
#include "strategy_utils.hpp"

#include "core/tensor/internal/memory_pool.hpp"
#include "io/pipelined_image_loader.hpp"
#include "kernels/densification_kernels.hpp"
#include "kernels/image_kernels.hpp"
#include "optimizer/adam_optimizer.hpp"

#include <numeric>
#include <random>

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

        const float get_percentil_value(const float q_percent, const lfs::core::Tensor tensor) {
            auto [sorted_val, sorted_idx] = tensor.sort();

            const int num_gaussians = static_cast<int>(tensor.shape()[0]);
            const int q_index = std::clamp(static_cast<int>(num_gaussians * q_percent), 0, num_gaussians - 1);
            const float quantile_threshold = sorted_val[q_index].item_as<float>();

            return quantile_threshold;
        }

        struct CannyWorkspace {
            lfs::core::Tensor grayscale;
            lfs::core::Tensor blurred;
            lfs::core::Tensor magnitude;
            lfs::core::Tensor angle;
            lfs::core::Tensor nms_output;
        };

        CannyWorkspace create_canny_workspace(int height, int width) {
            const size_t hw = static_cast<size_t>(height) * static_cast<size_t>(width);
            const auto dev = lfs::core::Device::CUDA;
            const auto dt = lfs::core::DataType::Float32;
            return {
                lfs::core::Tensor::zeros({hw}, dev, dt),
                lfs::core::Tensor::zeros({hw}, dev, dt),
                lfs::core::Tensor::zeros({hw}, dev, dt),
                lfs::core::Tensor::zeros({hw}, dev, dt),
                lfs::core::Tensor::zeros({static_cast<size_t>(height), static_cast<size_t>(width)}, dev, dt)};
        }

        void apply_canny_filter(const lfs::core::Tensor& input_data, CannyWorkspace& ws) {
            assert(input_data.dtype() == lfs::core::DataType::Float32);
            assert(input_data.device() == lfs::core::Device::CUDA);
            assert(input_data.ndim() == 3);

            const int width = input_data.shape()[2];
            const int height = input_data.shape()[1];

            ws.grayscale.zero_();
            ws.blurred.zero_();
            ws.magnitude.zero_();
            ws.angle.zero_();
            ws.nms_output.zero_();

            auto input_contig = input_data.contiguous();
            kernels::launch_grayscale_filter(input_contig.ptr<float>(), ws.grayscale.ptr<float>(), height, width);
            kernels::launch_gausssian_blur(ws.grayscale.ptr<float>(), ws.blurred.ptr<float>(), 3, height, width);
            kernels::launch_sobel_gradient_filter(ws.blurred.ptr<float>(), ws.magnitude.ptr<float>(), ws.angle.ptr<float>(), height, width);
            kernels::launch_nms_kernel(ws.magnitude.ptr<float>(), ws.angle.ptr<float>(), ws.nms_output.ptr<float>(), height, width);
        }

        void normalize_by_positive_median_inplace(lfs::core::Tensor& tensor) {
            tensor.masked_fill_(tensor.isnan(), 0.0f);
            auto valid = tensor.masked_select(tensor > 0.0f);
            if (valid.numel() == 0) {
                tensor.zero_();
                return;
            }
            auto [sorted, _] = valid.sort();
            float median = sorted[valid.numel() / 2].item_as<float>();
            tensor.div_(std::max(median, 1e-9f));
        }
    } // namespace

    ImprovedGSPlus::ImprovedGSPlus(lfs::core::SplatData& splat_data)
        : _splat_data(&splat_data) {}

    std::vector<int64_t> ImprovedGSPlus::get_count_array() {

        const int64_t budget = _params->max_cap;
        this->_initial_points = _splat_data->size();

        this->_total_steps = static_cast<int>((_params->stop_refine - _params->start_refine) / _params->refine_every) + 2;

        std::vector<int64_t> values;
        values.reserve(_total_steps);

        // Equation (2) Taming paper
        const float slope_lower_bound = (budget - _initial_points) / _total_steps;

        const float k = 2 * slope_lower_bound;
        const float a = (budget - _initial_points - k * _total_steps) / (_total_steps * _total_steps);
        const float b = k;
        const float c = static_cast<float>(_initial_points);

        // Set the total number of primitives up to add in each step
        for (int i = 1; i <= _total_steps; i++) {
            values.push_back(static_cast<int64_t>(1 * a * pow(i, 2) + (b * i) + c));
        }

        return values;
    }

    void ImprovedGSPlus::initialize(const lfs::core::param::OptimizationParameters& optimParams) {
        _params = std::make_unique<const lfs::core::param::OptimizationParameters>(optimParams);

        // Initialize Gaussians
        initialize_gaussians(*(_splat_data), _params->max_cap);

        // Create optimizer and scheduler
        _optimizer = create_optimizer(*_splat_data, *_params);
        _optimizer->allocate_gradients(_params->max_cap > 0 ? static_cast<size_t>(_params->max_cap) : 0);

        const double gamma = std::pow(0.1, 1.0 / optimParams.iterations);
        _scheduler = std::make_unique<ExponentialLR>(*_optimizer, gamma, std::vector<ParamType>{ParamType::Means, ParamType::Scaling});

        // Initialize densification info: [2, N] tensor for tracking gradients
        _splat_data->_densification_info = lfs::core::Tensor::zeros(
            {2, static_cast<size_t>(_splat_data->size())},
            _splat_data->means().device());

        // Initialize free mask: all slots are active (not free)
        const size_t capacity = _params->max_cap > 0 ? static_cast<size_t>(_params->max_cap)
                                                     : static_cast<size_t>(_splat_data->size());
        _free_mask = lfs::core::Tensor::zeros_bool({capacity}, _splat_data->means().device());

        // Initialize I-GS+ specifics
        this->_current_step = 0;

        this->_budget_schedule = get_count_array();
    }

    const lfs::core::Tensor ImprovedGSPlus::compute_gaussian_score(const lfs::core::Tensor& gradients) {
        const int64_t N = _splat_data->size();

        auto view_indices = random_cam_indices();
        const int num_views = static_cast<int>(view_indices.size());
        assert(num_views > 0);

        CannyWorkspace canny_ws;
        auto gaussian_scores = lfs::core::Tensor::zeros(
            {static_cast<size_t>(N)}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);

        for (int view = 0; view < num_views; view++) {
            const int idx = view_indices[view];
            lfs::core::Camera* cam = _views->get_camera(idx);

            lfs::core::Tensor image;
            if (_image_loader) {
                lfs::io::LoadParams params;
                params.resize_factor = _views->get_resize_factor();
                params.max_width = _views->get_max_width();
                if (cam->is_undistort_prepared()) {
                    params.undistort = &cam->undistort_params();
                }
                auto cached = _image_loader->decode_cached_image(cam->image_path(), params);
                if (cached) {
                    image = std::move(*cached);
                }
            }
            if (!image.is_valid()) {
                const CameraExample fallback = _views->get(idx);
                image = fallback.data.image;
            }

            const int img_h = image.shape()[1];
            const int img_w = image.shape()[2];
            if (view == 0 ||
                img_h != static_cast<int>(canny_ws.nms_output.shape()[0]) ||
                img_w != static_cast<int>(canny_ws.nms_output.shape()[1])) {
                canny_ws = create_canny_workspace(img_h, img_w);
            }

            apply_canny_filter(image, canny_ws);
            normalize_by_positive_median_inplace(canny_ws.nms_output);

            lfs::core::Tensor bg;
            auto score_render = edge_rasterize(*cam, this->get_model(), bg, canny_ws.nms_output);

            normalize_by_positive_median_inplace(score_render.edges_score);
            gaussian_scores.add_(score_render.edges_score);
        }

        gaussian_scores.div_(static_cast<float>(num_views));
        return gaussian_scores;
    }

    void ImprovedGSPlus::densify_with_score(const lfs::core::Tensor& scores, const lfs::core::Tensor& grads, const int64_t budget) {
        // Get Number of Gaussians to densify
        const lfs::core::Tensor grad_qualifiers = lfs::core::Tensor::where(grads >= _params->grad_threshold,
                                                                           lfs::core::Tensor::ones({1}), lfs::core::Tensor::zeros({1}))
                                                      .to(lfs::core::DataType::Bool);

        const int total_grads = static_cast<int>(grad_qualifiers.sum_scalar());

        // Budget allocation
        const int64_t curr_points = _splat_data->size();
        //  budget caps
        const int64_t curr_budget = std::min(budget, curr_points + total_grads);
        const int64_t budget_for_alloc = curr_budget - curr_points;

        if (budget_for_alloc > 0) {
            LAS_densify(scores, budget_for_alloc, grad_qualifiers, grads);
        }
    }

    void ImprovedGSPlus::LAS_densify(const lfs::core::Tensor& scores, const int64_t budget_for_alloc, const lfs::core::Tensor& grad_mask, const lfs::core::Tensor& grads) {

        lfs::core::Tensor scores_masked;

        if (_current_step < 3) {
            scores_masked = scores;
        } else {
            const lfs::core::Tensor LAS_grad_mask = lfs::core::Tensor::where(grads >= 0.00004,
                                                                             lfs::core::Tensor::ones({1}), lfs::core::Tensor::zeros({1}))
                                                        .to(lfs::core::DataType::Bool);

            scores_masked = scores.masked_fill(~LAS_grad_mask, 0);
        }

        const lfs::core::Tensor sampled_idxs = lfs::core::Tensor::multinomial(scores_masked, budget_for_alloc, false);

        LOG_DEBUG("split(): {} Gaussians to long axis split", budget_for_alloc);

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

        const lfs::core::Device device = _splat_data->means().device();

        // Allocate temporary tensors for split results [budget_for_alloc, ...]
        auto second_positions = lfs::core::Tensor::empty({static_cast<size_t>(budget_for_alloc), 3}, device);
        auto second_rotations = lfs::core::Tensor::empty({static_cast<size_t>(budget_for_alloc), 4}, device);
        auto second_scales = lfs::core::Tensor::empty({static_cast<size_t>(budget_for_alloc), 3}, device);
        auto second_sh0 = lfs::core::Tensor::empty({static_cast<size_t>(budget_for_alloc), 3}, device);
        lfs::core::Tensor second_shN;
        if (has_shN) {
            second_shN = lfs::core::Tensor::empty({static_cast<size_t>(budget_for_alloc), static_cast<size_t>(shN_dim)}, device);
        }
        auto second_opacities = lfs::core::Tensor::empty({static_cast<size_t>(budget_for_alloc)}, device);

        // Skip shN pointer when shN_dim = 0 (sh-degree 0)
        const bool use_shN = has_shN && shN_dim > 0;

        // Kernel launch: First result modifies in-place, seconds will go to temporaries:
        kernels::launch_long_axis_split_gaussians_inplace(
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
            sampled_idxs.ptr<int64_t>(),
            static_cast<int>(budget_for_alloc),
            shN_dim,
            nullptr);

        // Reset optimizer states for long-axis-split indices
        auto reset_optimizer_state_at_indices = [&](ParamType param_type) {
            auto* state = _optimizer->get_state_mutable(param_type);
            if (!state)
                return;

            const auto& shape = state->exp_avg.shape();
            if (has_zero_dimension(shape))
                return;

            std::vector<size_t> dims = {static_cast<size_t>(budget_for_alloc)};
            for (size_t i = 1; i < shape.rank(); ++i) {
                dims.push_back(shape[i]);
            }
            auto zeros = lfs::core::Tensor::zeros(lfs::core::TensorShape(dims), state->exp_avg.device());

            state->exp_avg.index_put_(sampled_idxs, zeros);
            state->exp_avg_sq.index_put_(sampled_idxs, zeros);
            if (state->grad.is_valid()) {
                state->grad.index_put_(sampled_idxs, zeros);
            }
        };

        reset_optimizer_state_at_indices(ParamType::Means);
        reset_optimizer_state_at_indices(ParamType::Rotation);
        reset_optimizer_state_at_indices(ParamType::Scaling);
        reset_optimizer_state_at_indices(ParamType::Sh0);
        reset_optimizer_state_at_indices(ParamType::ShN);
        reset_optimizer_state_at_indices(ParamType::Opacity);

        // Now place second split results: fill free slots first, then append
        auto [filled_indices, remaining] = fill_free_slots_with_data(
            second_positions, second_rotations, second_scales,
            second_sh0, second_shN, second_opacities, budget_for_alloc);

        const int64_t num_filled = budget_for_alloc - remaining;

        // Append remaining second results
        if (remaining > 0) {
            const size_t old_size = static_cast<size_t>(_splat_data->size());
            const size_t n_remaining = static_cast<size_t>(remaining);

            // Get the remaining data
            const auto append_positions = second_positions.slice(0, num_filled, budget_for_alloc);
            const auto append_rotations = second_rotations.slice(0, num_filled, budget_for_alloc);
            const auto append_scales = second_scales.slice(0, num_filled, budget_for_alloc);
            const auto append_sh0_flat = second_sh0.slice(0, num_filled, budget_for_alloc);
            const auto append_opacities = second_opacities.slice(0, num_filled, budget_for_alloc);

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
                auto append_shN = second_shN.slice(0, num_filled, budget_for_alloc);
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

    void ImprovedGSPlus::reset_opacity() {
        const float reset_value = 0.1;
        const float logit_reset_value = std::log(reset_value / (1.0f - reset_value));

        _splat_data->opacity_raw().clamp_max_(logit_reset_value);

        auto* state = _optimizer->get_state_mutable(ParamType::Opacity);
        if (state) {
            state->exp_avg.zero_();
            state->exp_avg_sq.zero_();
        }
    }

    void ImprovedGSPlus::pre_step(int iter, RenderOutput& render_output) {
        if (iter > _params->stop_refine)
            return;
        if (!is_refining(iter))
            return;

        assert(_views && "set_views() must be called before training");

        const lfs::core::Tensor numer = _splat_data->_densification_info[1];
        const lfs::core::Tensor denom = _splat_data->_densification_info[0];
        _precomputed_grads = numer / denom.clamp_min(1.0f);

        _precomputed_scores = compute_gaussian_score(_precomputed_grads);
        _precompute_valid = true;
    }

    void ImprovedGSPlus::post_backward(int iter, RenderOutput& render_output) {

        if (iter % _params->sh_degree_interval == 0) {
            this->_splat_data->increment_sh_degree();
        }

        if (iter > _params->stop_refine) {
            return;
        }

        if (is_refining(iter)) {
            assert(_precompute_valid);

            densify_with_score(_precomputed_scores, _precomputed_grads, get_current_budget());

            opacity_prune(iter);

            lfs::core::Tensor::trim_memory_pool();

            _splat_data->_densification_info = lfs::core::Tensor::zeros(
                {2, static_cast<size_t>(_splat_data->size())},
                _splat_data->means().device());

            this->_current_step++;

            _precomputed_scores = lfs::core::Tensor();
            _precomputed_grads = lfs::core::Tensor();
            _precompute_valid = false;
        }

        if (((iter % _params->reset_every) == 0) && (iter < _params->stop_refine) && (iter > 0)) {
            reset_opacity();
        }

        if (iter == _params->stop_refine) {
            _splat_data->_densification_info = lfs::core::Tensor::empty({0});

            lfs::core::CudaMemoryPool::instance().trim_cached_memory();
        }
    }

    bool ImprovedGSPlus::is_refining(int iter) const {
        return (iter >= _params->start_refine &&
                iter % _params->refine_every == 0 &&
                iter <= _params->stop_refine);
    }

    void ImprovedGSPlus::step(int iter) {
        if (iter < _params->iterations) {
            _optimizer->step(iter);
            _optimizer->zero_grad(iter);
            _scheduler->step();
        }
    }

    void ImprovedGSPlus::remove_gaussians(const lfs::core::Tensor& mask) {
        int mask_sum = mask.to(lfs::core::DataType::Int32).sum().template item<int>();

        if (mask_sum == 0) {
            LOG_DEBUG("No Gaussians to remove");
            return;
        }

        LOG_DEBUG("Removing {} Gaussians", mask_sum);
        remove(mask);
    }

    void ImprovedGSPlus::reserve_optimizer_capacity(size_t capacity) {
        if (_optimizer) {
            _optimizer->reserve_capacity(capacity);
            LOG_INFO("Reserved optimizer capacity for {} Gaussians", capacity);
        }
    }

    std::vector<int> ImprovedGSPlus::random_cam_indices(const int N) const {
        const int num_cam_dataset = _views->size();
        int num_samples = 0;

        if (num_cam_dataset < N) {
            num_samples = num_cam_dataset;
        } else {
            const int min_cam_dataset = 0.08 * num_cam_dataset;
            num_samples = std::max(N, min_cam_dataset);
        }

        std::vector<int> all_indices(num_cam_dataset);
        std::iota(all_indices.begin(), all_indices.end(), 0);

        std::default_random_engine rng(global_seed());
        std::shuffle(all_indices.begin(), all_indices.end(), rng);

        all_indices.resize(num_samples);
        return all_indices;
    }

    // From ImprovedGS but not used
    [[maybe_unused]] void ImprovedGSPlus::prune_post_reset() {
        const float q = 0.2f;
        const lfs::core::Tensor opacity = _splat_data->get_opacity();

        auto [sorted_val, sorted_idx] = opacity.sort();

        int num_gaussians = opacity.shape()[0];
        int q_index = static_cast<int>(num_gaussians * q);

        float quantile_threshold = sorted_val[q_index].item_as<float>();

        const lfs::core::Tensor prune_mask = (opacity < quantile_threshold);

        lfs::training::ImprovedGSPlus::remove(prune_mask);
    }

    void ImprovedGSPlus::opacity_prune(const int iter) {
        if (iter >= _params->stop_refine) {
            return;
        }
        const lfs::core::Tensor prune_mask = (_splat_data->get_opacity() < _params->prune_opacity);
        remove(prune_mask);
    }

    void ImprovedGSPlus::mark_as_free(const lfs::core::Tensor& indices) {
        if (!_free_mask.is_valid() || indices.numel() == 0) {
            return;
        }
        // Mark the given indices as free
        auto true_vals = lfs::core::Tensor::ones_bool({static_cast<size_t>(indices.numel())}, indices.device());
        _free_mask.index_put_(indices, true_vals);
    }

    void ImprovedGSPlus::remove(const lfs::core::Tensor& is_prune) {
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

    std::pair<lfs::core::Tensor, int64_t> ImprovedGSPlus::fill_free_slots_with_data(
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
            const auto n = static_cast<int>(slots_to_fill);
            const auto shN_slice = (shN_shape.rank() == 3)
                                       ? shN.slice(0, 0, slots_to_fill).reshape({n, static_cast<int>(shN_shape[1]), static_cast<int>(shN_shape[2])})
                                       : shN.slice(0, 0, slots_to_fill).reshape({n, static_cast<int>(shN_shape[1])});
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

    // ===== Serialization =====
    void ImprovedGSPlus::serialize(std::ostream& os) const {
        return;
    }

    void ImprovedGSPlus::deserialize(std::istream& is) {
        return;
    }

} // namespace lfs::training