/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mrnf.hpp"
#include "core/logger.hpp"
#include "edge_rasterizer.hpp"
#include "io/pipelined_image_loader.hpp"
#include "kernels/densification_kernels.hpp"
#include "kernels/image_kernels.hpp"
#include "kernels/mcmc_kernels.hpp"
#include "kernels/mrnf_kernels.hpp"
#include "strategy_utils.hpp"
#include "training/dataset.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace lfs::training {

    namespace {
        [[nodiscard]] bool mem_breakdown_enabled() {
            static const bool enabled = [] {
                const char* raw = std::getenv("LFS_MEM_BREAKDOWN");
                if (!raw) {
                    return false;
                }
                const std::string_view value(raw);
                return !value.empty() &&
                       value != "0" && value != "false" && value != "FALSE" &&
                       value != "off" && value != "OFF" &&
                       value != "no" && value != "NO";
            }();
            return enabled;
        }

        [[nodiscard]] double bytes_to_mib(const size_t bytes) {
            return static_cast<double>(bytes) / (1024.0 * 1024.0);
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

        constexpr float MRNF_EDGE_SCORE_WEIGHT = 0.25f;
        constexpr int MRNF_EDGE_MIN_VIEW_SAMPLES = 10;
        constexpr int MRNF_BOUNDS_RECOMPUTE_INTERVAL_REFINES = 5;
        constexpr float MRNF_RAW_OPACITY_PRUNE_THRESHOLD = -5.54126358f; // logit(1 / 255)
        constexpr float MRNF_LOG_MIN_SCALE_THRESHOLD = -23.0258509f;     // log(1e-10)

        [[nodiscard]] double compute_decay_gamma(const double start, const double end, const size_t steps) {
            if (steps == 0 || start <= 0.0 || end <= 0.0) {
                return 1.0;
            }
            return std::pow(end / start, 1.0 / static_cast<double>(steps));
        }

        void reset_vector_buffer(
            lfs::core::Tensor& tensor,
            const size_t size,
            const lfs::core::Device device,
            const size_t reserve_capacity = 0) {
            const size_t desired_capacity = reserve_capacity > 0 ? std::max(reserve_capacity, size) : size;
            const bool needs_new_tensor = !tensor.is_valid() ||
                                          tensor.ndim() != 1 ||
                                          tensor.device() != device ||
                                          tensor.dtype() != lfs::core::DataType::Float32;
            const auto make_fresh = [&]() {
                if (desired_capacity > size) {
                    tensor = lfs::core::Tensor::zeros_direct(lfs::core::TensorShape({size}), desired_capacity, device);
                } else {
                    tensor = lfs::core::Tensor::zeros({size}, device);
                }
            };

            if (needs_new_tensor) {
                make_fresh();
                return;
            }

            const size_t current_size = tensor.numel();
            if (current_size == 0) {
                if (size == 0) {
                    if (desired_capacity > tensor.capacity()) {
                        make_fresh();
                    } else {
                        tensor.zero_();
                    }
                } else if (tensor.capacity() >= desired_capacity) {
                    tensor.append_zeros(size);
                } else {
                    make_fresh();
                }
                return;
            }

            if (current_size == size) {
                if (desired_capacity > size && tensor.capacity() < desired_capacity) {
                    tensor.reserve(desired_capacity);
                }
                tensor.zero_();
                return;
            }

            if (current_size < size) {
                if (tensor.capacity() < desired_capacity) {
                    tensor.reserve(desired_capacity);
                }
                tensor.append_zeros(size - current_size);
                tensor.zero_();
                return;
            }

            make_fresh();
        }

        [[nodiscard]] bool has_zero_dimension(const lfs::core::TensorShape& shape) {
            for (size_t i = 0; i < shape.rank(); ++i) {
                if (shape[i] == 0) {
                    return true;
                }
            }
            return false;
        }

        [[nodiscard]] bool has_shN_coefficients(const lfs::core::Tensor& shN) {
            return shN.is_valid() && shN.ndim() >= 2 && shN.shape()[1] > 0;
        }

        void reset_optimizer_state_at_indices(
            AdamOptimizer& optimizer,
            const ParamType param_type,
            const lfs::core::Tensor& indices) {
            if (!indices.is_valid() || indices.numel() == 0) {
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

            std::vector<size_t> dims = {indices.numel()};
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

        [[nodiscard]] size_t deleted_mask_capacity(
            const lfs::core::SplatData& splat_data,
            const lfs::core::Tensor& free_mask) {
            return free_mask.is_valid() ? static_cast<size_t>(free_mask.numel())
                                        : static_cast<size_t>(splat_data.size());
        }

        void ensure_deleted_mask_size(
            lfs::core::SplatData& splat_data,
            const lfs::core::Tensor& free_mask) {
            const size_t current_size = static_cast<size_t>(splat_data.size());
            const size_t desired_capacity = deleted_mask_capacity(splat_data, free_mask);
            auto& deleted = splat_data.deleted();
            if (!deleted.is_valid() || deleted.ndim() != 1 || deleted.numel() != current_size) {
                deleted = lfs::core::Tensor::zeros_bool({current_size}, splat_data.means().device());
            }
            deleted.reserve(desired_capacity);
        }

        void sync_deleted_mask_from_free_mask(
            lfs::core::SplatData& splat_data,
            const lfs::core::Tensor& free_mask) {
            const size_t current_size = static_cast<size_t>(splat_data.size());
            const size_t desired_capacity = deleted_mask_capacity(splat_data, free_mask);

            if (!free_mask.is_valid()) {
                splat_data.deleted() = lfs::core::Tensor::zeros_bool({current_size}, splat_data.means().device());
                splat_data.deleted().reserve(desired_capacity);
                return;
            }

            splat_data.deleted() = free_mask.slice(0, 0, current_size).clone();
            splat_data.deleted().reserve(desired_capacity);
        }

        void set_deleted_mask_rows(
            lfs::core::SplatData& splat_data,
            const lfs::core::Tensor& free_mask,
            const lfs::core::Tensor& indices,
            const bool deleted) {
            if (!indices.is_valid() || indices.numel() == 0) {
                return;
            }

            ensure_deleted_mask_size(splat_data, free_mask);
            auto values = deleted
                              ? lfs::core::Tensor::ones_bool({static_cast<size_t>(indices.numel())}, indices.device())
                              : lfs::core::Tensor::zeros_bool({static_cast<size_t>(indices.numel())}, indices.device());
            splat_data.deleted().index_put_(indices, values);
        }

        void append_live_deleted_rows(
            lfs::core::SplatData& splat_data,
            const lfs::core::Tensor& free_mask,
            const size_t n_rows) {
            if (n_rows == 0) {
                return;
            }

            ensure_deleted_mask_size(splat_data, free_mask);
            auto& deleted = splat_data.deleted();
            const size_t desired_capacity = std::max(
                deleted_mask_capacity(splat_data, free_mask),
                static_cast<size_t>(deleted.numel()) + n_rows);
            deleted.reserve(desired_capacity);
            deleted.append_zeros(n_rows);
            deleted.reserve(deleted_mask_capacity(splat_data, free_mask));
        }

        struct CannyWorkspace {
            lfs::core::Tensor nms_output;
        };

        [[nodiscard]] CannyWorkspace create_canny_workspace(const int height, const int width) {
            const auto dev = lfs::core::Device::CUDA;
            const auto dt = lfs::core::DataType::Float32;
            return {
                lfs::core::Tensor::zeros({static_cast<size_t>(height), static_cast<size_t>(width)}, dev, dt)};
        }

        void ensure_canny_workspace(lfs::core::Tensor& nms_output, const int height, const int width) {
            if (!nms_output.is_valid() ||
                nms_output.ndim() != 2 ||
                height != static_cast<int>(nms_output.shape()[0]) ||
                width != static_cast<int>(nms_output.shape()[1])) {
                nms_output = lfs::core::Tensor::zeros(
                    {static_cast<size_t>(height), static_cast<size_t>(width)},
                    lfs::core::Device::CUDA,
                    lfs::core::DataType::Float32);
            }
        }

        void apply_canny_filter(const lfs::core::Tensor& input_data, CannyWorkspace& ws) {
            assert(input_data.dtype() == lfs::core::DataType::Float32 ||
                   input_data.dtype() == lfs::core::DataType::UInt8);
            assert(input_data.device() == lfs::core::Device::CUDA);
            assert(input_data.ndim() == 3);
            assert(input_data.shape()[0] >= 3);

            const int width = static_cast<int>(input_data.shape()[2]);
            const int height = static_cast<int>(input_data.shape()[1]);

            auto input_contig = input_data.contiguous();
            if (input_contig.dtype() == lfs::core::DataType::UInt8) {
                kernels::launch_fused_canny_edge_filter_chw(
                    input_contig.ptr<uint8_t>(),
                    ws.nms_output.ptr<float>(),
                    height,
                    width);
            } else {
                kernels::launch_fused_canny_edge_filter_chw(
                    input_contig.ptr<float>(),
                    ws.nms_output.ptr<float>(),
                    height,
                    width);
            }
        }

        void apply_canny_filter(const lfs::core::Tensor& input_data, lfs::core::Tensor& nms_output) {
            assert(input_data.dtype() == lfs::core::DataType::Float32 ||
                   input_data.dtype() == lfs::core::DataType::UInt8);
            assert(input_data.device() == lfs::core::Device::CUDA);
            assert(input_data.ndim() == 3);
            assert(input_data.shape()[0] >= 3);

            const int width = static_cast<int>(input_data.shape()[2]);
            const int height = static_cast<int>(input_data.shape()[1]);

            ensure_canny_workspace(nms_output, height, width);
            auto input_contig = input_data.contiguous();
            if (input_contig.dtype() == lfs::core::DataType::UInt8) {
                kernels::launch_fused_canny_edge_filter_chw(
                    input_contig.ptr<uint8_t>(),
                    nms_output.ptr<float>(),
                    height,
                    width);
            } else {
                kernels::launch_fused_canny_edge_filter_chw(
                    input_contig.ptr<float>(),
                    nms_output.ptr<float>(),
                    height,
                    width);
            }
        }

        void normalize_by_positive_median_inplace(lfs::core::Tensor& tensor) {
            tensor.masked_fill_(tensor.isnan(), 0.0f);
            auto valid = tensor.masked_select(tensor > 0.0f);
            if (valid.numel() == 0) {
                tensor.zero_();
                return;
            }
            auto [sorted, _] = valid.sort();
            if (tensor.device() == lfs::core::Device::CUDA) {
                kernels::launch_normalize_by_device_scalar(
                    tensor.ptr<float>(),
                    tensor.numel(),
                    sorted.ptr<float>() + valid.numel() / 2);
            } else {
                const float median = sorted[valid.numel() / 2].item_as<float>();
                tensor.div_(std::max(median, 1e-9f));
            }
        }

        [[nodiscard]] lfs::core::Tensor normalized_by_positive_median(const lfs::core::Tensor& tensor) {
            auto normalized = tensor.clone();
            normalize_by_positive_median_inplace(normalized);
            return normalized;
        }
    } // namespace

    MRNF::MRNF(lfs::core::SplatData& splat_data) : _splat_data(&splat_data) {}

    void MRNF::initialize(const lfs::core::param::OptimizationParameters& optimParams) {
        using namespace lfs::core;

        _params = std::make_unique<const lfs::core::param::OptimizationParameters>(optimParams);

        if (_params->max_cap > 0) {
            const size_t capacity = static_cast<size_t>(_params->max_cap);
            const size_t current_size = _splat_data->size();
            LOG_INFO("MRNF: pre-allocating capacity for {} Gaussians (current: {}, utilization: {:.1f}%)",
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
        _mean_lr_unscaled = _params->means_lr;
        _scale_lr_current = _params->scaling_lr;
        _mean_lr_gamma = compute_decay_gamma(_params->means_lr, _params->means_lr_end, _params->iterations);
        _scale_lr_gamma = compute_decay_gamma(_params->scaling_lr, _params->scaling_lr_end, _params->iterations);

        ensure_densification_info_shape();

        const size_t capacity = _params->max_cap > 0 ? static_cast<size_t>(_params->max_cap)
                                                     : static_cast<size_t>(_splat_data->size());
        _free_mask = Tensor::zeros_bool({capacity}, _splat_data->means().device());
        sync_deleted_mask_from_free_mask(*_splat_data, _free_mask);

        const size_t n = static_cast<size_t>(_splat_data->size());
        const size_t tracking_capacity = _params->max_cap > 0 ? static_cast<size_t>(_params->max_cap) : 0;
        reset_vector_buffer(_refine_weight_max, n, _splat_data->means().device(), tracking_capacity);
        reset_vector_buffer(_vis_count, n, _splat_data->means().device(), tracking_capacity);

        compute_bounds();

        if (mem_breakdown_enabled()) {
            LOG_INFO("[MEM] MRNF persistent refine_weight_max={:.2f} MiB, vis_count={:.2f} MiB, free_mask={:.2f} MiB, densification_info={:.2f} MiB",
                     bytes_to_mib(tensor_reserved_bytes(_refine_weight_max)),
                     bytes_to_mib(tensor_reserved_bytes(_vis_count)),
                     bytes_to_mib(tensor_reserved_bytes(_free_mask)),
                     bytes_to_mib(tensor_reserved_bytes(_splat_data->_densification_info)));
        }

        LOG_INFO("MRNF strategy initialized with {} Gaussians", n);
    }

    void MRNF::pre_step(int iter, RenderOutput& render_output) {
        _precomputed_edge_scores = lfs::core::Tensor();
        _edge_precompute_valid = false;

        if (!_params || !_params->use_edge_map || iter >= static_cast<int>(_params->stop_refine)) {
            reset_edge_accumulator();
            return;
        }

        if (should_accumulate_edge_sample(iter)) {
            accumulate_edge_sample(iter, render_output);
        }

        if (!is_refining(iter)) {
            return;
        }

        if (_edge_sample_count <= 0 ||
            !_edge_score_sum.is_valid() ||
            _edge_score_sum.ndim() != 1 ||
            _edge_score_sum.numel() != static_cast<size_t>(_splat_data->size())) {
            reset_edge_accumulator();
            return;
        }

        _precomputed_edge_scores = _edge_score_sum.clone();
        _precomputed_edge_scores.div_(static_cast<float>(_edge_sample_count));
        _edge_precompute_valid = true;
        reset_edge_accumulator();
    }

    void MRNF::ensure_densification_info_shape() {
        const size_t n = static_cast<size_t>(_splat_data->size());
        const auto& info = _splat_data->_densification_info;
        if (!info.is_valid() ||
            info.ndim() != 2 ||
            info.shape()[0] < 2 ||
            info.shape()[1] != n) {
            _splat_data->_densification_info = lfs::core::Tensor::zeros({2, n}, _splat_data->means().device());
        }
    }

    int MRNF::edge_target_samples_per_refine_window() const {
        const int refine_window = _params ? static_cast<int>(_params->refine_every) : 1;
        if (!_views || _views->size() == 0) {
            return std::max(1, std::min(refine_window, MRNF_EDGE_MIN_VIEW_SAMPLES));
        }

        const int num_cam_dataset = static_cast<int>(_views->size());
        const int requested_samples = num_cam_dataset < MRNF_EDGE_MIN_VIEW_SAMPLES
                                          ? num_cam_dataset
                                          : std::max(
                                                MRNF_EDGE_MIN_VIEW_SAMPLES,
                                                static_cast<int>(0.08f * static_cast<float>(num_cam_dataset)));
        return std::max(1, std::min(refine_window, requested_samples));
    }

    bool MRNF::should_accumulate_edge_sample(int iter) const {
        if (!_params || !_params->use_edge_map ||
            iter <= static_cast<int>(_params->start_refine) ||
            iter >= static_cast<int>(_params->stop_refine) ||
            _splat_data->size() == 0) {
            return false;
        }

        if (is_refining(iter)) {
            return true;
        }

        const int target_samples = edge_target_samples_per_refine_window();
        const int refine_every = std::max(1, static_cast<int>(_params->refine_every));
        const int stride = std::max(1, refine_every / target_samples);
        return (iter % stride) == 0;
    }

    void MRNF::reset_edge_accumulator() {
        _edge_score_sum = lfs::core::Tensor();
        _edge_sample_count = 0;
        _edge_last_sample_iter = -1;
    }

    void MRNF::accumulate_edge_sample(int iter, const RenderOutput& render_output) {
        using namespace lfs::core;

        if (_edge_last_sample_iter == iter) {
            return;
        }
        if (!render_output.camera ||
            !render_output.target_image.is_valid() ||
            render_output.target_image.device() != Device::CUDA ||
            (render_output.target_image.dtype() != DataType::Float32 &&
             render_output.target_image.dtype() != DataType::UInt8) ||
            render_output.target_image.ndim() != 3 ||
            render_output.target_image.shape()[0] < 3) {
            return;
        }

        const size_t n = static_cast<size_t>(_splat_data->size());
        if (!_edge_score_sum.is_valid() ||
            _edge_score_sum.ndim() != 1 ||
            _edge_score_sum.numel() != n) {
            _edge_score_sum = Tensor::zeros({n}, _splat_data->means().device());
            _edge_sample_count = 0;
        }

        apply_canny_filter(render_output.target_image, _edge_canny_nms_output);
        normalize_by_positive_median_inplace(_edge_canny_nms_output);

        auto score_render = edge_rasterize(
            *render_output.camera,
            this->get_model(),
            _edge_canny_nms_output);
        normalize_by_positive_median_inplace(score_render.edges_score);
        _edge_score_sum.add_(score_render.edges_score);
        ++_edge_sample_count;
        _edge_last_sample_iter = iter;
    }

    void MRNF::post_backward(int iter, RenderOutput& /*render_output*/) {
        LOG_TIMER("MRNF::post_backward");
        using namespace lfs::core;

        if (iter % _params->sh_degree_interval == 0) {
            _splat_data->increment_sh_degree();
        }

        if (iter == static_cast<int>(_params->stop_refine)) {
            _splat_data->_densification_info = Tensor::empty({0});
            _precomputed_edge_scores = Tensor();
            _edge_precompute_valid = false;
            reset_edge_accumulator();
        }

        if (iter >= static_cast<int>(_params->stop_refine)) {
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
            mrnf_strategy::launch_elementwise_add_inplace(
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
            _precomputed_edge_scores = Tensor();
            _edge_precompute_valid = false;
        }
    }

    bool MRNF::is_refining(int iter) const {
        return (iter < static_cast<int>(_params->stop_refine) &&
                iter > static_cast<int>(_params->start_refine) &&
                iter % _params->refine_every == 0);
    }

    void MRNF::refine(int iter) {
        LOG_TIMER("MRNF::refine");
        using namespace lfs::core;

        ++_refine_windows_since_bounds;
        if (!_bounds_valid || _refine_windows_since_bounds >= MRNF_BOUNDS_RECOMPUTE_INTERVAL_REFINES) {
            compute_bounds();
        }

        const float max_allowed = _bounds.max_extent * 100.0f;
        const size_t n = static_cast<size_t>(_splat_data->size());

        auto raw_opacities = _splat_data->opacity_raw();
        if (raw_opacities.ndim() == 2 && raw_opacities.shape()[1] == 1)
            raw_opacities = raw_opacities.squeeze(-1);
        const auto& log_scales = _splat_data->scaling_raw();
        const auto& means = _splat_data->means();
        const float log_max_allowed = std::log(max_allowed);

        assert(raw_opacities.numel() == n);
        assert(log_scales.shape()[0] == n && log_scales.shape()[1] == 3);
        assert(means.shape()[0] == n && means.shape()[1] == 3);

        auto scale_min = log_scales.min(1);
        auto scale_max = log_scales.max(1);

        auto center = Tensor::from_vector(
            {_bounds.center[0], _bounds.center[1], _bounds.center[2]},
            TensorShape({1, 3}), Device::CUDA);
        auto dist_from_center = (means - center).abs().max(1);

        auto prune_mask = (raw_opacities < MRNF_RAW_OPACITY_PRUNE_THRESHOLD) |
                          compute_near_zero_rotation_mask(_splat_data->rotation_raw()) |
                          (scale_min < MRNF_LOG_MIN_SCALE_THRESHOLD) |
                          (scale_max > log_max_allowed) |
                          (dist_from_center > max_allowed);

        if (_free_mask.is_valid() && n > 0) {
            auto active_mask = _free_mask.slice(0, 0, n).logical_not();
            prune_mask = prune_mask.logical_and(active_mask);
        }

        const int pruned_count = static_cast<int>(prune_mask.sum().item());

        if (pruned_count > 0) {
            auto prune_indices = prune_mask.nonzero().squeeze(-1);
            mark_as_free(prune_indices);
            set_deleted_mask_rows(*_splat_data, _free_mask, prune_indices, true);

            // Zero quaternion so deleted rows exit early in preprocessing.
            auto zero_rotation = Tensor::zeros({static_cast<size_t>(pruned_count), 4}, _splat_data->rotation_raw().device());
            _splat_data->rotation_raw().index_put_(prune_indices, zero_rotation);

            reset_optimizer_state_at_indices(*_optimizer, ParamType::Means, prune_indices);
            reset_optimizer_state_at_indices(*_optimizer, ParamType::Sh0, prune_indices);
            reset_optimizer_state_at_indices(*_optimizer, ParamType::ShN, prune_indices);
            reset_optimizer_state_at_indices(*_optimizer, ParamType::Scaling, prune_indices);
            reset_optimizer_state_at_indices(*_optimizer, ParamType::Rotation, prune_indices);
            reset_optimizer_state_at_indices(*_optimizer, ParamType::Opacity, prune_indices);

            LOG_DEBUG("MRNF: soft-pruned {} splats at iter {} (active: {}, total slots: {})",
                      pruned_count, iter, active_count(), _splat_data->size());
        }

        // Replacement should stay active even after growth stop.
        grow_and_split(iter, pruned_count);
        enforce_max_cap();
        apply_decay(iter);

        const size_t new_n = static_cast<size_t>(_splat_data->size());
        const size_t tracking_capacity = _params->max_cap > 0 ? static_cast<size_t>(_params->max_cap) : 0;
        reset_vector_buffer(_refine_weight_max, new_n, _splat_data->means().device(), tracking_capacity);
        reset_vector_buffer(_vis_count, new_n, _splat_data->means().device(), tracking_capacity);
        ensure_densification_info_shape();
        _splat_data->_densification_info.zero_();
    }

    void MRNF::grow_and_split(int iter, int pruned_count) {
        LOG_TIMER("MRNF::grow_and_split");
        using namespace lfs::core;

        const size_t n = static_cast<size_t>(_splat_data->size());
        const size_t current_active = active_count();
        const int desired_total = static_cast<int>(
            std::round(static_cast<float>(
                           ((_refine_weight_max > _params->growth_grad_threshold) &&
                            (_vis_count > 0.0f))
                               .sum()
                               .item()) *
                       _params->grow_fraction));
        const int budget = (_params->max_cap > 0)
                               ? std::max(0, _params->max_cap - static_cast<int>(current_active))
                               : INT_MAX;
        const int requested_replace = std::min(pruned_count, budget);
        int n_grow = 0;
        lfs::core::Tensor above_threshold;
        lfs::core::Tensor active_mask;
        if (_free_mask.is_valid() && n > 0) {
            active_mask = _free_mask.slice(0, 0, n).logical_not();
        }

        auto seed = static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());

        const auto edge_guidance = edge_guidance_factor();

        Tensor split_indices;
        Tensor replace_inds;
        Tensor growth_inds;
        Tensor replace_mask;
        int actual_replace = 0;

        if (requested_replace > 0) {
            auto opacities = _splat_data->get_opacity();
            if (opacities.ndim() == 2 && opacities.shape()[1] == 1)
                opacities = opacities.squeeze(-1);
            auto replace_weights = opacities * (_vis_count > 0.0f);
            if (active_mask.is_valid()) {
                replace_weights = replace_weights * active_mask;
            }
            if (edge_guidance.is_valid()) {
                replace_weights = replace_weights * edge_guidance;
            }
            const int selectable_replace = static_cast<int>(replace_weights.count_nonzero());
            actual_replace = std::min(requested_replace, selectable_replace);
            if (actual_replace > 0) {
                replace_inds = Tensor::empty({static_cast<size_t>(actual_replace)}, Device::CUDA, DataType::Int64);
                mrnf_strategy::launch_gumbel_topk(
                    replace_weights.ptr<float>(), n, actual_replace, seed,
                    replace_inds.ptr<int64_t>());

                replace_mask = Tensor::zeros_bool({n}, Device::CUDA);
                auto true_vals = Tensor::ones_bool({static_cast<size_t>(actual_replace)}, Device::CUDA);
                replace_mask.index_put_(replace_inds, true_vals);
            }
        }

        if (iter < static_cast<int>(_params->grow_until_iter)) {
            above_threshold = (_refine_weight_max > _params->growth_grad_threshold) &&
                              (_vis_count > 0.0f);
            if (active_mask.is_valid()) {
                above_threshold = above_threshold.logical_and(active_mask);
            }
            n_grow = std::max(0, desired_total - actual_replace);
            n_grow = std::min(n_grow, budget - actual_replace);
        }

        if (n_grow > 0) {
            auto growth_weights = above_threshold * _refine_weight_max;
            if (edge_guidance.is_valid()) {
                growth_weights = growth_weights * edge_guidance;
            }

            if (replace_mask.is_valid()) {
                // Keep replacement and growth disjoint on device instead of
                // deduplicating sampled indices on the host.
                growth_weights = growth_weights.masked_fill(replace_mask, 0.0f);
            }

            const int selectable_growth = static_cast<int>(growth_weights.count_nonzero());
            if (selectable_growth > 0) {
                const int growth_budget = std::min(n_grow, selectable_growth);
                growth_inds = Tensor::empty({static_cast<size_t>(growth_budget)}, Device::CUDA, DataType::Int64);
                mrnf_strategy::launch_gumbel_topk(
                    growth_weights.ptr<float>(), n, growth_budget, seed + 1,
                    growth_inds.ptr<int64_t>());
            }
        }

        if (replace_inds.is_valid() && replace_inds.numel() > 0 &&
            growth_inds.is_valid() && growth_inds.numel() > 0) {
            split_indices = Tensor::cat({replace_inds, growth_inds}, 0);
        } else if (replace_inds.is_valid() && replace_inds.numel() > 0) {
            split_indices = replace_inds;
        } else if (growth_inds.is_valid() && growth_inds.numel() > 0) {
            split_indices = growth_inds;
        }

        if (!split_indices.is_valid() || split_indices.numel() == 0)
            return;

        assert(_params->max_cap <= 0 ||
               current_active + split_indices.numel() <= static_cast<size_t>(_params->max_cap));

        const size_t K = split_indices.numel();
        const size_t sh_rest = (_splat_data->shN().is_valid() && _splat_data->shN().ndim() >= 2)
                                   ? _splat_data->shN().shape()[1]
                                   : 0;
        const int shN_dim = static_cast<int>(sh_rest * 3);

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

        kernels::launch_long_axis_split_gaussians_inplace(
            _splat_data->means().ptr<float>(),
            _splat_data->rotation_raw().ptr<float>(),
            _splat_data->scaling_raw().ptr<float>(),
            _splat_data->sh0().ptr<float>(),
            shN_dim > 0 ? _splat_data->shN().ptr<float>() : nullptr,
            _splat_data->opacity_raw().ptr<float>(),
            child_means.ptr<float>(),
            child_rotations.ptr<float>(),
            child_log_scales.ptr<float>(),
            child_sh0.ptr<float>(),
            shN_dim > 0 ? child_shN.ptr<float>() : nullptr,
            child_raw_opacities.ptr<float>(),
            split_indices.ptr<int64_t>(),
            static_cast<int>(K),
            shN_dim);

        reset_optimizer_state_at_indices(*_optimizer, ParamType::Means, split_indices);
        reset_optimizer_state_at_indices(*_optimizer, ParamType::Sh0, split_indices);
        reset_optimizer_state_at_indices(*_optimizer, ParamType::ShN, split_indices);
        reset_optimizer_state_at_indices(*_optimizer, ParamType::Scaling, split_indices);
        reset_optimizer_state_at_indices(*_optimizer, ParamType::Rotation, split_indices);
        reset_optimizer_state_at_indices(*_optimizer, ParamType::Opacity, split_indices);

        size_t append_start = 0;
        if (free_count() > 0) {
            auto [filled_indices, remaining_after_fill] = fill_free_slots_with_data(
                child_means,
                child_rotations,
                child_log_scales,
                child_sh0,
                child_shN,
                child_raw_opacities,
                static_cast<int64_t>(K));
            append_start = K - static_cast<size_t>(remaining_after_fill);
        }

        const size_t n_append = K - append_start;
        if (n_append > 0) {
            append_live_deleted_rows(*_splat_data, _free_mask, n_append);
            if (_free_mask.is_valid() && _free_mask.numel() < _splat_data->size() + n_append) {
                _free_mask.reserve(_splat_data->size() + n_append);
                _free_mask.append_zeros(n_append);
            }

            auto append_means = child_means.slice(0, append_start, K);
            auto append_sh0 = child_sh0.slice(0, append_start, K);
            auto append_shN = child_shN.slice(0, append_start, K);
            auto append_scaling = child_log_scales.slice(0, append_start, K);
            auto append_rotation = child_rotations.slice(0, append_start, K);
            auto append_opacity = child_raw_opacities.slice(0, append_start, K);
            if (_splat_data->opacity_raw().ndim() == 2) {
                append_opacity = append_opacity.unsqueeze(-1);
            }

            _optimizer->add_new_params(ParamType::Means, append_means, true);
            _optimizer->add_new_params(ParamType::Sh0, append_sh0, true);
            _optimizer->add_new_params(ParamType::ShN, append_shN, false);
            _optimizer->add_new_params(ParamType::Scaling, append_scaling, true);
            _optimizer->add_new_params(ParamType::Rotation, append_rotation, true);
            _optimizer->add_new_params(ParamType::Opacity, append_opacity, true);
        }

        LOG_DEBUG("MRNF: split {} splats at iter {} (reused: {}, appended: {}, active: {}, total slots: {})",
                  K, iter, append_start, n_append, active_count(), _splat_data->size());
    }

    void MRNF::compact_splats(const lfs::core::Tensor& keep_mask) {
        LOG_TIMER("MRNF::compact_splats");
        using namespace lfs::core;

        const size_t old_size = static_cast<size_t>(_splat_data->size());
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

        const auto& info = _splat_data->_densification_info;
        if (info.is_valid() && info.ndim() == 2 && info.shape()[1] == old_size) {
            _splat_data->_densification_info = info.index_select(1, valid_indices).contiguous();
        }
        if (_splat_data->has_deleted_mask() && _splat_data->deleted().numel() == old_size) {
            auto compacted_deleted = _splat_data->deleted().index_select(0, valid_indices).contiguous();
            if (cap > 0)
                compacted_deleted.reserve(cap);
            _splat_data->deleted() = std::move(compacted_deleted);
        }
        if (_free_mask.is_valid() && old_size > 0) {
            auto compacted_free = _free_mask.slice(0, 0, old_size).index_select(0, valid_indices).contiguous();
            if (cap > new_size) {
                auto tail = Tensor::zeros_bool({cap - new_size}, compacted_free.device());
                _free_mask = Tensor::cat({compacted_free, tail}, 0);
            } else {
                _free_mask = std::move(compacted_free);
            }
        }
        if (_refine_weight_max.is_valid() && _refine_weight_max.numel() > new_size)
            _refine_weight_max = _refine_weight_max.index_select(0, valid_indices).contiguous();
        if (_vis_count.is_valid() && _vis_count.numel() > new_size)
            _vis_count = _vis_count.index_select(0, valid_indices).contiguous();
        if (_precomputed_edge_scores.is_valid() && _precomputed_edge_scores.numel() > new_size)
            _precomputed_edge_scores = _precomputed_edge_scores.index_select(0, valid_indices).contiguous();
    }

    void MRNF::inject_noise(int /*iter*/) {
        const size_t n = static_cast<size_t>(_splat_data->size());
        if (n == 0)
            return;

        const float lr_mean = static_cast<float>(_optimizer->get_param_lr(ParamType::Means));

        auto seed = static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());

        mrnf_strategy::launch_mrnf_noise_injection(
            _splat_data->means().ptr<float>(),
            _splat_data->opacity_raw().ptr<float>(),
            _vis_count.ptr<float>(),
            lr_mean,
            _params->means_noise_weight,
            _bounds.median_size,
            n, seed);
    }

    void MRNF::apply_decay(int iter) {
        const size_t n = static_cast<size_t>(_splat_data->size());
        if (n == 0)
            return;

        const float train_t = static_cast<float>(iter) / static_cast<float>(_params->iterations);

        mrnf_strategy::launch_mrnf_decay(
            _splat_data->opacity_raw().ptr<float>(),
            _splat_data->scaling_raw().ptr<float>(),
            _params->opacity_decay,
            _params->scale_decay,
            train_t,
            n);
    }

    void MRNF::enforce_max_cap() {
        if (_params->max_cap <= 0)
            return;

        using namespace lfs::core;

        const size_t n = _splat_data->size();
        const size_t cap = static_cast<size_t>(_params->max_cap);
        if (n <= cap)
            return;

        LOG_INFO("MRNF: count {} exceeds max_cap {}, pruning excess", n, cap);

        auto opacities = _splat_data->get_opacity();
        if (opacities.ndim() == 2 && opacities.shape()[1] == 1)
            opacities = opacities.squeeze(-1);

        auto seed = static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());

        auto keep_indices = Tensor::empty({cap}, Device::CUDA, DataType::Int64);
        mrnf_strategy::launch_gumbel_topk(
            opacities.ptr<float>(), n, cap, seed,
            keep_indices.ptr<int64_t>());

        auto keep_mask = Tensor::zeros_bool({n}, opacities.device());
        auto true_vals = Tensor::ones_bool({cap}, opacities.device());
        keep_mask.index_put_(keep_indices, true_vals);
        compact_splats(keep_mask);

        assert(_splat_data->size() <= cap);
    }

    size_t MRNF::active_count() const {
        if (!_free_mask.is_valid()) {
            return static_cast<size_t>(_splat_data->size());
        }

        const size_t current_size = static_cast<size_t>(_splat_data->size());
        if (current_size == 0)
            return 0;

        auto active_region = _free_mask.slice(0, 0, current_size);
        const size_t free_count_val = static_cast<size_t>(active_region.sum_scalar());
        return current_size - free_count_val;
    }

    size_t MRNF::free_count() const {
        if (!_free_mask.is_valid()) {
            return 0;
        }

        const size_t current_size = static_cast<size_t>(_splat_data->size());
        if (current_size == 0)
            return 0;

        auto active_region = _free_mask.slice(0, 0, current_size);
        return static_cast<size_t>(active_region.sum_scalar());
    }

    lfs::core::Tensor MRNF::get_active_indices() const {
        const size_t current_size = static_cast<size_t>(_splat_data->size());
        if (current_size == 0) {
            return {};
        }

        if (!_free_mask.is_valid() || free_count() == 0) {
            auto all_active = lfs::core::Tensor::ones_bool({current_size}, _splat_data->means().device());
            return all_active.nonzero().squeeze(-1);
        }

        auto active_region = _free_mask.slice(0, 0, current_size);
        auto is_active = active_region.logical_not();
        return is_active.nonzero().squeeze(-1);
    }

    void MRNF::mark_as_free(const lfs::core::Tensor& indices) {
        if (!_free_mask.is_valid() || indices.numel() == 0) {
            return;
        }

        auto true_vals = lfs::core::Tensor::ones_bool({static_cast<size_t>(indices.numel())}, indices.device());
        _free_mask.index_put_(indices, true_vals);
    }

    std::pair<lfs::core::Tensor, int64_t> MRNF::fill_free_slots_with_data(
        const lfs::core::Tensor& positions,
        const lfs::core::Tensor& rotations,
        const lfs::core::Tensor& scales,
        const lfs::core::Tensor& sh0,
        const lfs::core::Tensor& shN,
        const lfs::core::Tensor& opacities,
        int64_t count) {

        using namespace lfs::core;

        if (!_free_mask.is_valid() || count == 0) {
            return {Tensor(), count};
        }

        const size_t current_size = static_cast<size_t>(_splat_data->size());
        auto active_region = _free_mask.slice(0, 0, current_size);
        auto free_indices = active_region.nonzero().squeeze(-1);
        const int64_t num_free = free_indices.numel();

        if (num_free == 0) {
            return {Tensor(), count};
        }

        const int64_t slots_to_fill = std::min(count, num_free);
        auto target_indices = free_indices.slice(0, 0, slots_to_fill);

        _splat_data->means().index_put_(target_indices, positions.slice(0, 0, slots_to_fill));
        _splat_data->rotation_raw().index_put_(target_indices, rotations.slice(0, 0, slots_to_fill));
        _splat_data->scaling_raw().index_put_(target_indices, scales.slice(0, 0, slots_to_fill));
        _splat_data->sh0().index_put_(target_indices, sh0.slice(0, 0, slots_to_fill));

        auto opacity_slice = opacities.slice(0, 0, slots_to_fill);
        if (_splat_data->opacity_raw().ndim() == 2) {
            opacity_slice = opacity_slice.unsqueeze(-1);
        }
        _splat_data->opacity_raw().index_put_(target_indices, opacity_slice);

        if (shN.is_valid() && has_shN_coefficients(_splat_data->shN())) {
            _splat_data->shN().index_put_(target_indices, shN.slice(0, 0, slots_to_fill));
        }

        reset_optimizer_state_at_indices(*_optimizer, ParamType::Means, target_indices);
        reset_optimizer_state_at_indices(*_optimizer, ParamType::Sh0, target_indices);
        reset_optimizer_state_at_indices(*_optimizer, ParamType::ShN, target_indices);
        reset_optimizer_state_at_indices(*_optimizer, ParamType::Scaling, target_indices);
        reset_optimizer_state_at_indices(*_optimizer, ParamType::Rotation, target_indices);
        reset_optimizer_state_at_indices(*_optimizer, ParamType::Opacity, target_indices);

        auto false_vals = Tensor::zeros_bool({static_cast<size_t>(slots_to_fill)}, target_indices.device());
        _free_mask.index_put_(target_indices, false_vals);
        set_deleted_mask_rows(*_splat_data, _free_mask, target_indices, false);

        return {target_indices, count - slots_to_fill};
    }

    void MRNF::compute_bounds() {
        const size_t n = active_count();
        if (n == 0) {
            _bounds_valid = false;
            return;
        }

        lfs::core::Tensor active_means = _splat_data->means();
        if (_free_mask.is_valid() && free_count() > 0) {
            auto active_indices = get_active_indices();
            active_means = _splat_data->means().index_select(0, active_indices).contiguous();
        }

        mrnf_strategy::launch_percentile_bounds(
            active_means.ptr<float>(),
            n,
            _params->bounds_percentile,
            &_bounds);

        _bounds_valid = true;
        _refine_windows_since_bounds = 0;

        _optimizer->set_param_lr(ParamType::Means, _mean_lr_unscaled * _bounds.median_size);
    }

    void MRNF::step(int iter) {
        LOG_TIMER("MRNF::step");
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

    void MRNF::remove_gaussians(const lfs::core::Tensor& mask) {
        using namespace lfs::core;

        Tensor keep_mask = mask.logical_not();
        const size_t old_size = static_cast<size_t>(_splat_data->size());
        const int n_remove = static_cast<int>(old_size - keep_mask.to(DataType::Int32).sum().template item<int>());

        LOG_INFO("MRNF::remove_gaussians: mask size={}, n_remove={}, current size={}",
                 mask.numel(), n_remove, _splat_data->size());

        if (n_remove == 0)
            return;

        compact_splats(keep_mask);

        if (_splat_data->size() == 0) {
            _bounds_valid = false;
        } else if (_bounds_valid) {
            compute_bounds();
        }
    }

    lfs::core::Tensor MRNF::compute_edge_scores(const int iter) {
        const int64_t N = static_cast<int64_t>(_splat_data->size());
        if (N <= 0 || active_count() == 0 || !_views || !_image_loader || _views->size() == 0) {
            return {};
        }

        const int num_cam_dataset = static_cast<int>(_views->size());
        int num_samples = 0;
        if (num_cam_dataset < MRNF_EDGE_MIN_VIEW_SAMPLES) {
            num_samples = num_cam_dataset;
        } else {
            const int min_cam_dataset = static_cast<int>(0.08f * static_cast<float>(num_cam_dataset));
            num_samples = std::max(MRNF_EDGE_MIN_VIEW_SAMPLES, min_cam_dataset);
        }
        if (num_samples <= 0) {
            return {};
        }

        std::vector<int> view_indices(num_cam_dataset);
        std::iota(view_indices.begin(), view_indices.end(), 0);
        std::default_random_engine rng(static_cast<unsigned>(iter));
        std::shuffle(view_indices.begin(), view_indices.end(), rng);
        view_indices.resize(num_samples);

        CannyWorkspace canny_ws;
        auto gaussian_scores = lfs::core::Tensor::zeros(
            {static_cast<size_t>(N)}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);

        for (const int dataset_idx : view_indices) {
            lfs::core::Camera* cam = _views->get_camera(static_cast<size_t>(dataset_idx));

            lfs::io::LoadParams params;
            params.resize_factor = _views->get_resize_factor();
            params.max_width = _views->get_max_width();
            params.output_uint8 = true;
            if (cam->is_undistort_prepared()) {
                params.undistort = &cam->undistort_params();
            }

            lfs::core::Tensor image = _image_loader->load_image_immediate(cam->image_path(), params);
            const int img_h = static_cast<int>(image.shape()[1]);
            const int img_w = static_cast<int>(image.shape()[2]);

            if (cam->image_width() != img_w || cam->image_height() != img_h) {
                cam->set_image_dimensions(img_w, img_h);
            }

            if (!canny_ws.nms_output.is_valid() ||
                img_h != static_cast<int>(canny_ws.nms_output.shape()[0]) ||
                img_w != static_cast<int>(canny_ws.nms_output.shape()[1])) {
                canny_ws = create_canny_workspace(img_h, img_w);
            }

            apply_canny_filter(image, canny_ws);
            normalize_by_positive_median_inplace(canny_ws.nms_output);

            auto score_render = edge_rasterize(*cam, this->get_model(), canny_ws.nms_output);
            normalize_by_positive_median_inplace(score_render.edges_score);
            gaussian_scores.add_(score_render.edges_score);
        }

        gaussian_scores.div_(static_cast<float>(num_samples));
        return gaussian_scores;
    }

    lfs::core::Tensor MRNF::edge_guidance_factor() const {
        if (!_params || !_params->use_edge_map || !_edge_precompute_valid) {
            return {};
        }

        const size_t n = static_cast<size_t>(_splat_data->size());
        if (!_precomputed_edge_scores.is_valid() ||
            _precomputed_edge_scores.ndim() != 1 ||
            _precomputed_edge_scores.numel() != n) {
            return {};
        }

        auto normalized_edge = normalized_by_positive_median(_precomputed_edge_scores);
        return normalized_edge.mul(MRNF_EDGE_SCORE_WEIGHT).add(1.0f);
    }

    namespace {
        constexpr uint32_t LFS_MAGIC = 0x4C464252; // "LFBR"
        constexpr uint32_t LFS_VERSION = 3;
    } // namespace

    void MRNF::serialize(std::ostream& os) const {
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

        const uint8_t has_free_mask = _free_mask.is_valid() ? 1 : 0;
        os.write(reinterpret_cast<const char*>(&has_free_mask), sizeof(has_free_mask));
        if (has_free_mask) {
            os << _free_mask;
        }

        os.write(reinterpret_cast<const char*>(&_mean_lr_unscaled), sizeof(_mean_lr_unscaled));
        os.write(reinterpret_cast<const char*>(&_scale_lr_current), sizeof(_scale_lr_current));
    }

    void MRNF::deserialize(std::istream& is) {
        uint32_t magic, version;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != LFS_MAGIC)
            throw std::runtime_error("Invalid MRNF checkpoint: wrong magic");
        if (version == 0 || version > LFS_VERSION)
            throw std::runtime_error("Unsupported MRNF checkpoint version: " + std::to_string(version));

        uint8_t has_optimizer;
        is.read(reinterpret_cast<char*>(&has_optimizer), sizeof(has_optimizer));
        if (has_optimizer && _optimizer)
            _optimizer->deserialize(is);

        uint8_t has_scheduler;
        is.read(reinterpret_cast<char*>(&has_scheduler), sizeof(has_scheduler));
        if (has_scheduler && _scheduler)
            _scheduler->deserialize(is);

        const double optimizer_mean_lr = _optimizer ? _optimizer->get_param_lr(ParamType::Means) : 0.0;
        const double optimizer_scaling_lr = _optimizer ? _optimizer->get_param_lr(ParamType::Scaling) : 0.0;

        if (version >= 2) {
            uint8_t has_free_mask = 0;
            is.read(reinterpret_cast<char*>(&has_free_mask), sizeof(has_free_mask));
            if (has_free_mask) {
                is >> _free_mask;
                if (_free_mask.device() != lfs::core::Device::CUDA) {
                    _free_mask = _free_mask.cuda();
                }
            }
        }
        if (version >= 3) {
            is.read(reinterpret_cast<char*>(&_mean_lr_unscaled), sizeof(_mean_lr_unscaled));
            is.read(reinterpret_cast<char*>(&_scale_lr_current), sizeof(_scale_lr_current));
        } else {
            _mean_lr_unscaled = _params ? _params->means_lr : _mean_lr_unscaled;
            _scale_lr_current = optimizer_scaling_lr > 0.0
                                    ? optimizer_scaling_lr
                                    : (_params ? _params->scaling_lr : _scale_lr_current);
        }

        if (!_free_mask.is_valid()) {
            const size_t capacity = (_params && _params->max_cap > 0)
                                        ? static_cast<size_t>(_params->max_cap)
                                        : static_cast<size_t>(_splat_data->size());
            _free_mask = lfs::core::Tensor::zeros_bool({capacity}, _splat_data->means().device());
        }
        sync_deleted_mask_from_free_mask(*_splat_data, _free_mask);

        const size_t n = static_cast<size_t>(_splat_data->size());
        const size_t tracking_capacity = (_params && _params->max_cap > 0)
                                             ? static_cast<size_t>(_params->max_cap)
                                             : 0;
        reset_vector_buffer(_refine_weight_max, n, _splat_data->means().device(), tracking_capacity);
        reset_vector_buffer(_vis_count, n, _splat_data->means().device(), tracking_capacity);
        ensure_densification_info_shape();
        _precomputed_edge_scores = lfs::core::Tensor();
        _edge_precompute_valid = false;

        if (_splat_data->size() == 0 || active_count() == 0) {
            _bounds_valid = false;
        } else {
            compute_bounds();
        }

        if (version < 3 && _bounds_valid && optimizer_mean_lr > 0.0 && _bounds.median_size > 0.0f) {
            _mean_lr_unscaled = optimizer_mean_lr / static_cast<double>(_bounds.median_size);
        }

        refresh_decay_schedule_from_current_state();

        if (_optimizer) {
            _optimizer->set_param_lr(ParamType::Scaling, _scale_lr_current);
            if (_bounds_valid) {
                _optimizer->set_param_lr(ParamType::Means, _mean_lr_unscaled * _bounds.median_size);
            }
        }
    }

    void MRNF::reserve_optimizer_capacity(size_t capacity) {
        if (_optimizer) {
            _optimizer->reserve_capacity(capacity);
            LOG_INFO("MRNF: reserved optimizer capacity for {} Gaussians", capacity);
        }
    }

    void MRNF::set_optimization_params(const lfs::core::param::OptimizationParameters& params) {
        _params = std::make_unique<const lfs::core::param::OptimizationParameters>(params);

        if (_mean_lr_unscaled <= 0.0) {
            _mean_lr_unscaled = params.means_lr;
        }
        if (_scale_lr_current <= 0.0) {
            _scale_lr_current = params.scaling_lr;
        }

        refresh_decay_schedule_from_current_state();

        if (_optimizer) {
            _optimizer->set_param_lr(ParamType::Scaling, _scale_lr_current);
            if (_bounds_valid) {
                _optimizer->set_param_lr(ParamType::Means, _mean_lr_unscaled * _bounds.median_size);
            }
        }
    }

    void MRNF::refresh_decay_schedule_from_current_state() {
        const int64_t completed_steps = _optimizer ? std::max<int64_t>(0, _optimizer->get_step_count(ParamType::Means))
                                                   : 0;
        const size_t remaining_steps =
            (_params && _params->iterations > static_cast<size_t>(completed_steps))
                ? (_params->iterations - static_cast<size_t>(completed_steps))
                : 0;

        const double mean_lr_end = _params ? _params->means_lr_end : _mean_lr_unscaled;
        const double scaling_lr_end = _params ? _params->scaling_lr_end : _scale_lr_current;
        _mean_lr_gamma = compute_decay_gamma(_mean_lr_unscaled, mean_lr_end, remaining_steps);
        _scale_lr_gamma = compute_decay_gamma(_scale_lr_current, scaling_lr_end, remaining_steps);
    }

} // namespace lfs::training
