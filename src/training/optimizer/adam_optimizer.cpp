/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "adam_optimizer.hpp"
#include "adam_api.h" // fast_lfs::optimizer::adam_step_raw
#include "core/checkpoint_format.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"
#include "diagnostics/vram_profiler.hpp"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace lfs::training {

    namespace {
        constexpr int SH_WARMUP_ITERATIONS = 1000;
        constexpr float DEFAULT_GROWTH_MULTIPLIER = 1.5f;
        constexpr uint8_t QUANTIZED_MOMENT_ZERO_POINT = 128;

        [[nodiscard]] size_t tensor_row_size(const lfs::core::Tensor& tensor) {
            if (!tensor.is_valid() || tensor.ndim() == 0) {
                return 0;
            }
            size_t row_size = 1;
            for (size_t i = 1; i < tensor.shape().rank(); ++i) {
                row_size *= tensor.shape()[i];
            }
            return row_size;
        }

        [[nodiscard]] size_t tensor_allocated_elements(const lfs::core::Tensor& tensor) {
            if (!tensor.is_valid() || tensor.ndim() == 0) {
                return 0;
            }
            const size_t row_size = tensor_row_size(tensor);
            if (row_size == 0) {
                return 0;
            }
            return (tensor.capacity() > 0 ? tensor.capacity() : tensor.shape()[0]) * row_size;
        }

        void fill_quantized_moment_zero_point(lfs::core::Tensor& tensor) {
            if (!tensor.is_valid() || tensor.dtype() != lfs::core::DataType::UInt8) {
                return;
            }
            const size_t elements = tensor_allocated_elements(tensor);
            if (elements == 0) {
                return;
            }
            LFS_CUDA_CHECK(cudaMemsetAsync(
                tensor.ptr<uint8_t>(),
                QUANTIZED_MOMENT_ZERO_POINT,
                elements * sizeof(uint8_t),
                tensor.stream()));
        }

        void fill_quantized_moment_zero_point_range(
            lfs::core::Tensor& tensor,
            const size_t element_offset,
            const size_t elements) {
            if (!tensor.is_valid() || tensor.dtype() != lfs::core::DataType::UInt8 || elements == 0) {
                return;
            }
            LFS_CUDA_CHECK(cudaMemsetAsync(
                tensor.ptr<uint8_t>() + element_offset,
                QUANTIZED_MOMENT_ZERO_POINT,
                elements * sizeof(uint8_t),
                tensor.stream()));
        }
    } // namespace

    AdamOptimizer::AdamOptimizer(lfs::core::SplatData& splat_data, const AdamConfig& config)
        : config_(config),
          splat_data_(splat_data) {}

    void AdamOptimizer::set_frozen_mask(lfs::core::Tensor mask) {
        if (mask.is_valid()) {
            if (mask.dtype() != lfs::core::DataType::Bool || mask.ndim() != 1) {
                throw std::runtime_error("AdamOptimizer frozen mask must be a 1D bool tensor");
            }
            if (mask.device() != lfs::core::Device::CUDA) {
                mask = mask.cuda();
            }
            if (!mask.is_contiguous()) {
                mask = mask.contiguous();
            }
            mask.set_name("adam.frozen_mask");
        }
        frozen_mask_ = std::move(mask);
    }

    void AdamOptimizer::step(const int iteration) {
        LFS_TRACE("kernel.adam.step");
        if (fused_step_iteration_ == iteration) {
            last_step_zeroed_gradients_ = true;
            return;
        }
        last_step_zeroed_gradients_ = false;
        for (const auto type : all_param_types()) {
            step_param(type, iteration);
        }
    }

    size_t AdamOptimizer::compute_state_growth(ParamType type, size_t n_new) const {
        if (type != ParamType::ShN)
            return n_new;
        const auto layout_rest = static_cast<uint32_t>(splat_data_.max_sh_coeffs_rest());
        if (layout_rest == 0)
            return 0;
        const size_t n_old = static_cast<size_t>(splat_data_.size());
        // splat_data_.size() reflects the post-growth N at this point; the caller already
        // resized the SplatData. So n_old here is the new total, and the old N was n_old - n_new.
        if (n_old < n_new)
            return lfs::core::sh_swizzled_float_count(n_old, layout_rest);
        const size_t prev = n_old - n_new;
        return lfs::core::sh_swizzled_float_count(n_old, layout_rest) -
               lfs::core::sh_swizzled_float_count(prev, layout_rest);
    }

    void AdamOptimizer::allocate_gradients() {
        allocate_gradients(config_.initial_capacity);
    }

    void AdamOptimizer::allocate_gradients(const size_t capacity) {
        for (const auto type : all_param_types()) {
            auto& param = get_param(type);
            const auto name = param_name(type);
            auto& state = states_[name];

            if (!param.is_valid()) {
                state = AdamParamState{};
                continue;
            }

            const size_t param_size = param.shape()[0];
            if (type == ParamType::ShN && splat_data_.max_sh_coeffs_rest() == 0) {
                state = AdamParamState{};
                state.size = param_size;
                LOG_INFO("AdamOptimizer: no SH-rest optimizer state for max SH degree 0");
                continue;
            }

            // shN is stored in vksplat swizzled layout as a 1D float tensor; capacity
            // along dim 0 must be expressed in floats, not primitive rows.
            const size_t effective_capacity = (type == ParamType::ShN)
                                                  ? lfs::core::sh_swizzled_float_count(
                                                        capacity,
                                                        static_cast<uint32_t>(splat_data_.max_sh_coeffs_rest()))
                                                  : capacity;
            const size_t alloc_cap = (effective_capacity > param_size) ? effective_capacity : param_size;

            // Handle zero-size tensors (e.g., shN with sh-degree 0 has shape [N, 0, 3]).
            // Moment tensors are persistent Adam state. Gradients are transient and allocated
            // lazily by get_grad(); fused fastgs backward never needs the full gradient buffers.
            //
            // Force the direct/reserved allocator for shN so capacity is recorded even when
            // N already fits in the current swizzled block. Without this, the fast path in
            // extend_state_for_new_params trips on capacity()==0.
            const size_t prim_capacity = (type == ParamType::ShN)
                                             ? std::max(capacity, static_cast<size_t>(splat_data_.size()))
                                             : alloc_cap;
            alloc_quantized_state(type, state, param, alloc_cap, prim_capacity);
            state.exp_avg.set_name("adam." + name + ".exp_avg");
            state.exp_avg_sq.set_name("adam." + name + ".exp_avg_sq");
            state.grad = {};
            state.capacity = alloc_cap;
            state.size = param_size;
            state.step_count = 0;
            LOG_DEBUG("allocate_gradients({}): cap={}", name, state.capacity);
        }
        LOG_DEBUG("Allocated gradients for {} parameter groups", states_.size());
    }

    bool AdamOptimizer::has_gradients() const {
        for (const auto& [_, state] : states_) {
            if (state.grad.is_valid() && state.grad.numel() > 0) {
                return true;
            }
        }
        return false;
    }

    void AdamOptimizer::zero_grad(int /*iteration*/) {
        if (last_step_zeroed_gradients_) {
            last_step_zeroed_gradients_ = false;
            return;
        }
        for (auto& [_, state] : states_) {
            if (state.grad.is_valid() && state.grad.numel() > 0) {
                const size_t bytes = state.size * (state.grad.numel() / state.grad.shape()[0]) * sizeof(float);
                LFS_CUDA_CHECK(cudaMemsetAsync(state.grad.ptr<float>(), 0, bytes, state.grad.stream()));
            }
        }
    }

    lfs::core::Tensor& AdamOptimizer::get_param(ParamType type) {
        switch (type) {
        case ParamType::Means: return splat_data_.means();
        case ParamType::Sh0: return splat_data_.sh0();
        case ParamType::ShN: return splat_data_.shN();
        case ParamType::Scaling: return splat_data_.scaling_raw();
        case ParamType::Rotation: return splat_data_.rotation_raw();
        case ParamType::Opacity: return splat_data_.opacity_raw();
        }
        throw std::runtime_error("Invalid ParamType");
    }

    lfs::core::Tensor& AdamOptimizer::get_grad(ParamType type) {
        const auto name = param_name(type);
        const auto it = states_.find(name);
        if (it == states_.end()) {
            init_state(type, false);
        }
        ensure_grad(type);
        return states_[name].grad;
    }

    std::string AdamOptimizer::param_name(ParamType type) const {
        switch (type) {
        case ParamType::Means: return "means";
        case ParamType::Sh0: return "sh0";
        case ParamType::ShN: return "shN";
        case ParamType::Scaling: return "scaling";
        case ParamType::Rotation: return "rotation";
        case ParamType::Opacity: return "opacity";
        }
        return "unknown";
    }

    // Every param's quantised scale tensor is per-primitive, length = gaussian count, for both
    // contiguous params (shape[0] == N) and swizzled shN (separate 1D moment buffer, scale = N).
    size_t AdamOptimizer::scale_row_count(ParamType /*type*/) const {
        return static_cast<size_t>(splat_data_.size());
    }

    const bool* AdamOptimizer::frozen_mask_ptr() const {
        return frozen_mask_.is_valid() && frozen_mask_.numel() > 0
                   ? frozen_mask_.ptr<bool>()
                   : nullptr;
    }

    int AdamOptimizer::frozen_mask_size() const {
        return frozen_mask_.is_valid()
                   ? static_cast<int>(frozen_mask_.numel())
                   : 0;
    }

    void AdamOptimizer::alloc_quantized_state(ParamType type, AdamParamState& state,
                                              const lfs::core::Tensor& param,
                                              size_t moment_capacity, size_t prim_capacity) {
        const auto& shape = param.shape();
        const size_t moment_rows = shape[0];
        moment_capacity = std::max(moment_capacity, moment_rows);
        const size_t prim_rows = scale_row_count(type);
        prim_capacity = std::max(prim_capacity, prim_rows);

        // shN forces direct allocation so capacity is recorded even when N already fits the
        // current swizzled block (mirrors the pre-quantisation behaviour).
        const bool moment_direct = (moment_capacity > moment_rows) || type == ParamType::ShN;
        if (moment_direct && moment_capacity > 0) {
            state.exp_avg = lfs::core::Tensor::zeros_direct(shape, moment_capacity, param.device(), lfs::core::DataType::UInt8);
            state.exp_avg_sq = lfs::core::Tensor::zeros_direct(shape, moment_capacity, param.device(), lfs::core::DataType::UInt8);
        } else {
            state.exp_avg = lfs::core::Tensor::zeros(shape, param.device(), lfs::core::DataType::UInt8);
            state.exp_avg_sq = lfs::core::Tensor::zeros(shape, param.device(), lfs::core::DataType::UInt8);
        }
        fill_quantized_moment_zero_point(state.exp_avg);

        const lfs::core::TensorShape scale_shape({prim_rows});
        if (prim_capacity > prim_rows) {
            state.exp_avg_scale = lfs::core::Tensor::zeros_direct(scale_shape, prim_capacity, param.device());
            state.exp_avg_sq_scale = lfs::core::Tensor::zeros_direct(scale_shape, prim_capacity, param.device());
        } else {
            state.exp_avg_scale = lfs::core::Tensor::zeros(scale_shape, param.device());
            state.exp_avg_sq_scale = lfs::core::Tensor::zeros(scale_shape, param.device());
        }
    }

    void AdamOptimizer::quantize_float_moments(ParamType type, AdamParamState& state,
                                               lfs::core::Tensor&& exp_avg, lfs::core::Tensor&& exp_avg_sq) {
        if (!exp_avg.is_valid() || !exp_avg_sq.is_valid() || exp_avg.ndim() == 0 ||
            exp_avg.shape()[0] == 0 || exp_avg.numel() == 0) {
            state.exp_avg = {};
            state.exp_avg_sq = {};
            state.exp_avg_scale = {};
            state.exp_avg_sq_scale = {};
            return;
        }

        exp_avg = exp_avg.cuda();
        exp_avg_sq = exp_avg_sq.cuda();
        const size_t prim_rows = scale_row_count(type);

        state.exp_avg = lfs::core::Tensor::empty(exp_avg.shape(), lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
        state.exp_avg_sq = lfs::core::Tensor::empty(exp_avg_sq.shape(), lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
        state.exp_avg_scale = lfs::core::Tensor::empty(lfs::core::TensorShape({prim_rows}), lfs::core::Device::CUDA);
        state.exp_avg_sq_scale = lfs::core::Tensor::empty(lfs::core::TensorShape({prim_rows}), lfs::core::Device::CUDA);

        const cudaStream_t stream = lfs::core::getCurrentCUDAStream();
        lfs::core::waitForCUDAStream(stream, exp_avg.stream());
        lfs::core::waitForCUDAStream(stream, exp_avg_sq.stream());

        if (type == ParamType::ShN) {
            const auto layout_rest = static_cast<uint32_t>(splat_data_.max_sh_coeffs_rest());
            const int slots = static_cast<int>(lfs::core::sh_float4_slots_for_rest(layout_rest));
            fast_lfs::optimizer::quantize_adam_moments_swizzled_raw(
                exp_avg.ptr<float>(), exp_avg_sq.ptr<float>(),
                state.exp_avg.ptr<uint8_t>(), state.exp_avg_scale.ptr<float>(),
                state.exp_avg_sq.ptr<uint8_t>(), state.exp_avg_sq_scale.ptr<float>(),
                static_cast<int>(prim_rows), slots, stream);
        } else {
            const size_t row_size = exp_avg.numel() / exp_avg.shape()[0];
            fast_lfs::optimizer::quantize_adam_moments_raw(
                exp_avg.ptr<float>(), exp_avg_sq.ptr<float>(),
                state.exp_avg.ptr<uint8_t>(), state.exp_avg_scale.ptr<float>(),
                state.exp_avg_sq.ptr<uint8_t>(), state.exp_avg_sq_scale.ptr<float>(),
                static_cast<int>(exp_avg.shape()[0]), static_cast<int>(row_size), stream);
        }

        state.exp_avg.set_stream(stream);
        state.exp_avg_sq.set_stream(stream);
        state.exp_avg_scale.set_stream(stream);
        state.exp_avg_sq_scale.set_stream(stream);
    }

    void AdamOptimizer::init_state(ParamType type, bool allocate_grad) {
        auto& param = get_param(type);
        const auto name = param_name(type);

        if (!param.is_valid()) {
            throw std::runtime_error("init_state: " + name + " not valid");
        }
        if (param.ndim() == 0) {
            throw std::runtime_error("init_state: " + name + " has rank 0");
        }

        auto& state = states_[name];
        const size_t param_size = param.shape()[0];
        const size_t initial_cap = type == ParamType::ShN
                                       ? std::max(
                                             param_size,
                                             lfs::core::sh_swizzled_float_count(
                                                 config_.initial_capacity,
                                                 static_cast<uint32_t>(splat_data_.max_sh_coeffs_rest())))
                                       : compute_new_capacity(0, param_size);

        if (allocate_grad && (!state.grad.is_valid() || state.grad.numel() == 0)) {
            state.grad = (initial_cap > param_size)
                             ? lfs::core::Tensor::zeros_direct(param.shape(), initial_cap)
                             : lfs::core::Tensor::zeros(param.shape(), param.device());
        }

        const size_t prim_capacity = (type == ParamType::ShN)
                                         ? std::max(static_cast<size_t>(splat_data_.size()), config_.initial_capacity)
                                         : std::max(initial_cap, param_size);
        alloc_quantized_state(type, state, param, initial_cap, prim_capacity);
        state.capacity = std::max(initial_cap, param_size);
        state.size = param_size;
        state.step_count = 0;
        if (type == ParamType::ShN) {
            const double mib = (2.0 * static_cast<double>(state.capacity) * sizeof(uint8_t) +
                                2.0 * static_cast<double>(prim_capacity) * sizeof(float)) /
                               (1024.0 * 1024.0);
            LOG_INFO("AdamOptimizer: allocated SH-rest optimizer state at max SH degree {} ({:.2f} MiB)",
                     splat_data_.get_max_sh_degree(), mib);
        }
        LOG_DEBUG("Initialized optimizer state for {}: size={}, capacity={}", name, param_size, state.capacity);
    }

    void AdamOptimizer::ensure_grad(ParamType type) {
        auto& param = get_param(type);
        const auto name = param_name(type);
        if (!states_.contains(name)) {
            init_state(type, false);
        }

        auto& state = states_[name];
        if (state.grad.is_valid() && state.grad.numel() > 0) {
            return;
        }
        if (!param.is_valid() || param.ndim() == 0) {
            throw std::runtime_error("ensure_grad: " + name + " param invalid");
        }

        const size_t param_size = param.shape()[0];
        const size_t alloc_cap = state.capacity > param_size ? state.capacity : param_size;
        state.grad = (alloc_cap > param_size)
                         ? lfs::core::Tensor::zeros_direct(param.shape(), alloc_cap)
                         : lfs::core::Tensor::zeros(param.shape(), param.device());
        state.grad.set_name("adam." + name + ".grad");
        state.size = param_size;
        state.capacity = std::max(state.capacity, alloc_cap);
    }

    void AdamOptimizer::step_param(ParamType type, const int iteration) {
        auto& param = get_param(type);
        if (!param.is_valid() || param.numel() == 0) {
            return;
        }
        if (type == ParamType::ShN &&
            (iteration <= SH_WARMUP_ITERATIONS || splat_data_.active_sh_coeffs_rest() == 0)) {
            return;
        }

        const auto name = param_name(type);
        if (!states_.contains(name)) {
            init_state(type, false);
        }

        auto& state = states_[name];
        if (!state.grad.is_valid() || state.grad.numel() == 0 ||
            !state.exp_avg.is_valid() || state.exp_avg.numel() == 0) {
            return;
        }

        state.step_count++;

        // Skip higher-degree SH during warmup
        if (type == ParamType::ShN && iteration <= SH_WARMUP_ITERATIONS) {
            return;
        }

        const double bias_correction1_rcp = 1.0 / (1.0 - std::pow(config_.beta1, state.step_count));
        const double bias_correction2_sqrt_rcp = 1.0 / std::sqrt(1.0 - std::pow(config_.beta2, state.step_count));
        const float param_lr = static_cast<float>(get_param_lr(type));

        const size_t param_size = param.shape()[0];
        if (param_size != state.size) {
            throw std::runtime_error("Optimizer state desync: " + name);
        }

        cudaStream_t execution_stream = state.grad.stream();
        if (execution_stream == nullptr) {
            execution_stream = param.stream();
        }
        if (execution_stream == nullptr) {
            execution_stream = state.exp_avg.stream();
        }
        lfs::core::waitForCUDAStream(execution_stream, param.stream());
        lfs::core::waitForCUDAStream(execution_stream, state.exp_avg.stream());
        lfs::core::waitForCUDAStream(execution_stream, state.exp_avg_sq.stream());
        lfs::core::waitForCUDAStream(execution_stream, state.exp_avg_scale.stream());
        lfs::core::waitForCUDAStream(execution_stream, state.exp_avg_sq_scale.stream());
        lfs::core::waitForCUDAStream(execution_stream, state.grad.stream());
        if (frozen_mask_.is_valid()) {
            lfs::core::waitForCUDAStream(execution_stream, frozen_mask_.stream());
        }

        if (type == ParamType::ShN) {
            const auto layout_rest = static_cast<uint32_t>(splat_data_.max_sh_coeffs_rest());
            const int slots = static_cast<int>(lfs::core::sh_float4_slots_for_rest(layout_rest));
            fast_lfs::optimizer::adam_step_quantized_swizzled_raw(
                param.ptr<float>(),
                state.exp_avg.ptr<uint8_t>(),
                state.exp_avg_scale.ptr<float>(),
                state.exp_avg_sq.ptr<uint8_t>(),
                state.exp_avg_sq_scale.ptr<float>(),
                state.grad.ptr<float>(),
                frozen_mask_ptr(),
                frozen_mask_size(),
                static_cast<int>(scale_row_count(type)),
                slots,
                param_lr,
                config_.beta1,
                config_.beta2,
                config_.eps,
                bias_correction1_rcp,
                bias_correction2_sqrt_rcp,
                execution_stream);
            param.set_stream(execution_stream);
            state.exp_avg.set_stream(execution_stream);
            state.exp_avg_sq.set_stream(execution_stream);
            state.exp_avg_scale.set_stream(execution_stream);
            state.exp_avg_sq_scale.set_stream(execution_stream);
            state.grad.set_stream(execution_stream);
            return;
        }

        const size_t feature_dim = param.numel() / param_size;
        fast_lfs::optimizer::adam_step_quantized_raw(
            param.ptr<float>(),
            state.exp_avg.ptr<uint8_t>(),
            state.exp_avg_scale.ptr<float>(),
            state.exp_avg_sq.ptr<uint8_t>(),
            state.exp_avg_sq_scale.ptr<float>(),
            state.grad.ptr<float>(),
            frozen_mask_ptr(),
            frozen_mask_size(),
            static_cast<int>(state.size),
            static_cast<int>(feature_dim),
            param_lr,
            config_.beta1,
            config_.beta2,
            config_.eps,
            bias_correction1_rcp,
            bias_correction2_sqrt_rcp,
            execution_stream);
        param.set_stream(execution_stream);
        state.exp_avg.set_stream(execution_stream);
        state.exp_avg_sq.set_stream(execution_stream);
        state.exp_avg_scale.set_stream(execution_stream);
        state.exp_avg_sq_scale.set_stream(execution_stream);
        state.grad.set_stream(execution_stream);
    }

    FastGSFusedAdamState AdamOptimizer::prepare_fastgs_fused_adam(const int iteration) {
        FastGSFusedAdamState fused;
        fused.enabled = true;
        fused.beta1 = static_cast<float>(config_.beta1);
        fused.beta2 = static_cast<float>(config_.beta2);
        fused.eps = static_cast<float>(config_.eps);

        auto prepare_param = [&](ParamType type, const int n_attributes, const bool update_enabled) {
            FastGSFusedAdamParam out;
            auto& param = get_param(type);
            if (!param.is_valid() || param.numel() == 0 || n_attributes <= 0) {
                return out;
            }
            if (!update_enabled) {
                return out;
            }

            const auto name = param_name(type);
            if (!states_.contains(name)) {
                init_state(type, false);
            }
            auto& state = states_[name];
            if (!state.exp_avg.is_valid() || !state.exp_avg_sq.is_valid() ||
                !state.exp_avg_scale.is_valid() || !state.exp_avg_sq_scale.is_valid()) {
                init_state(type, false);
            }

            const size_t param_size = param.shape()[0];
            if (param_size != state.size) {
                throw std::runtime_error("Optimizer state desync before fused Adam: " + name);
            }

            const auto next_step = state.step_count + 1;
            const double bias_correction1_rcp = 1.0 / (1.0 - std::pow(config_.beta1, next_step));
            const double bias_correction2_sqrt_rcp = 1.0 / std::sqrt(1.0 - std::pow(config_.beta2, next_step));

            out.param = param.ptr<float>();
            out.exp_avg_q = state.exp_avg.ptr<uint8_t>();
            out.exp_avg_sq_q = state.exp_avg_sq.ptr<uint8_t>();
            out.exp_avg_scale = state.exp_avg_scale.ptr<float>();
            out.exp_avg_sq_scale = state.exp_avg_sq_scale.ptr<float>();
            out.frozen_mask = frozen_mask_ptr();
            out.frozen_mask_size = frozen_mask_size();
            out.n_elements = static_cast<int>(param.numel());
            out.n_attributes = n_attributes;
            out.step_size = static_cast<float>(get_param_lr(type) * bias_correction1_rcp);
            out.bias_correction2_sqrt_rcp = static_cast<float>(bias_correction2_sqrt_rcp);
            out.enabled = true;
            return out;
        };

        fused.means = prepare_param(ParamType::Means, 3, true);
        fused.sh0 = prepare_param(ParamType::Sh0, 3, true);
        // shN is laid out in swizzled float4 order (vksplat shAt). The fused-backward kernel
        // indexes it via shAt(p, k) float4-slot reads/writes, not via
        // primitive_idx*n_attributes+offset, so n_attributes is informational only.
        const auto active_rest = static_cast<uint32_t>(splat_data_.active_sh_coeffs_rest());
        const auto layout_rest = static_cast<uint32_t>(splat_data_.max_sh_coeffs_rest());
        fused.shN = prepare_param(ParamType::ShN,
                                  static_cast<int>(lfs::core::sh_float4_slots_for_rest(layout_rest) * 4u),
                                  active_rest > 0 && iteration > SH_WARMUP_ITERATIONS);
        fused.scaling = prepare_param(ParamType::Scaling, 3, true);
        fused.rotation = prepare_param(ParamType::Rotation, 4, true);
        fused.opacity = prepare_param(ParamType::Opacity, 1, true);

        fused.enabled = fused.means.enabled || fused.sh0.enabled || fused.shN.enabled ||
                        fused.scaling.enabled || fused.rotation.enabled || fused.opacity.enabled;
        return fused;
    }

    void AdamOptimizer::commit_fastgs_fused_adam(const int iteration) {
        for (const auto type : all_param_types()) {
            auto& param = get_param(type);
            if (!param.is_valid() || param.numel() == 0)
                continue;
            const auto name = param_name(type);
            if (!states_.contains(name)) {
                continue;
            }
            auto& state = states_[name];
            if (state.exp_avg.is_valid() && state.exp_avg_sq.is_valid()) {
                state.step_count++;
            }
        }
        fused_step_iteration_ = iteration;
        last_step_zeroed_gradients_ = true;
    }

    void AdamOptimizer::reset_state_at_indices(ParamType type, const std::vector<int64_t>& indices) {
        if (indices.empty())
            return;

        const auto name = param_name(type);
        if (!states_.contains(name))
            return;

        // Skip ShN when not initialized (sh_degree=0 case)
        if (type == ParamType::ShN) {
            const auto& param = get_param(type);
            if (!param.is_valid() || param.numel() == 0 ||
                splat_data_.max_sh_coeffs_rest() == 0) {
                return; // ShN is empty at max sh-degree 0, nothing to reset
            }
        }

        auto& state = states_[name];

        // Validate tensors before accessing. Contiguous params can reset both quantized bytes and
        // scales directly. Swizzled shN keeps scale-zero reset semantics; inactive high-band bytes
        // are initialized to the signed zero-point separately so later SH activation is neutral.
        if (!state.exp_avg_scale.is_valid() || state.exp_avg_scale.ptr<float>() == nullptr ||
            !state.exp_avg_sq_scale.is_valid() || state.exp_avg_sq_scale.ptr<float>() == nullptr) {
            LOG_WARN("reset_state_at_indices: {} scale tensor is invalid or null", name);
            return;
        }

        const cudaStream_t stream = lfs::core::getCurrentCUDAStream();
        const size_t idx_bytes = indices.size() * sizeof(int64_t);
        int64_t* d_indices = nullptr;
        LFS_CUDA_CHECK(cudaMallocAsync(&d_indices, idx_bytes, stream));
        LFS_CUDA_CHECK(cudaMemcpyAsync(d_indices, indices.data(), idx_bytes, cudaMemcpyHostToDevice, stream));

        lfs::core::waitForCUDAStream(stream, state.exp_avg_scale.stream());
        lfs::core::waitForCUDAStream(stream, state.exp_avg_sq_scale.stream());

        if (type != ParamType::ShN && state.exp_avg.is_valid() && state.exp_avg_sq.is_valid()) {
            const int row_size = static_cast<int>(tensor_row_size(state.exp_avg));
            lfs::core::waitForCUDAStream(stream, state.exp_avg.stream());
            lfs::core::waitForCUDAStream(stream, state.exp_avg_sq.stream());
            fast_lfs::optimizer::zero_quantized_rows_at_indices(
                state.exp_avg.ptr<uint8_t>(),
                state.exp_avg_scale.ptr<float>(),
                d_indices,
                indices.size(),
                row_size,
                QUANTIZED_MOMENT_ZERO_POINT,
                stream);
            fast_lfs::optimizer::zero_quantized_rows_at_indices(
                state.exp_avg_sq.ptr<uint8_t>(),
                state.exp_avg_sq_scale.ptr<float>(),
                d_indices,
                indices.size(),
                row_size,
                0,
                stream);
            state.exp_avg.set_stream(stream);
            state.exp_avg_sq.set_stream(stream);
        } else {
            fast_lfs::optimizer::zero_rows_at_indices(state.exp_avg_scale.ptr<float>(), d_indices, indices.size(), 1, stream);
            fast_lfs::optimizer::zero_rows_at_indices(state.exp_avg_sq_scale.ptr<float>(), d_indices, indices.size(), 1, stream);
        }

        state.exp_avg_scale.set_stream(stream);
        state.exp_avg_sq_scale.set_stream(stream);
        LFS_CUDA_CHECK(cudaFreeAsync(d_indices, stream));
    }

    void AdamOptimizer::extend_state_by_gather(ParamType type, const lfs::core::Tensor& indices) {
        const auto name = param_name(type);
        if (!states_.contains(name))
            return;

        const size_t n_new = indices.numel();
        if (n_new == 0)
            return;

        auto& param = get_param(type);
        auto& state = states_[name];
        const size_t new_size = state.size + n_new;

        if (type == ParamType::ShN &&
            (splat_data_.max_sh_coeffs_rest() == 0 ||
             !state.exp_avg.is_valid() || !state.exp_avg_sq.is_valid())) {
            return;
        }

        if (!param.is_valid() || param.shape().rank() == 0) {
            LOG_WARN("extend_state_by_gather: {} param invalid", name);
            return;
        }
        if (!state.exp_avg.is_valid() || state.exp_avg.ndim() == 0) {
            LOG_WARN("extend_state_by_gather: {} state invalid", name);
            return;
        }

        // Contiguous params only: moment rows == primitive rows == scale rows, so moments and
        // scales grow with the same indices. (shN duplication goes through add_new_params_gather.)
        if (type == ParamType::ShN) {
            LOG_WARN("extend_state_by_gather: shN handled via add_new_params_gather; skipping");
            return;
        }

        // Fast path: use reserved capacity
        const bool grad_has_capacity = !state.grad.is_valid() || state.grad.capacity() > 0;
        const bool all_have_capacity = grad_has_capacity &&
                                       state.exp_avg.capacity() > 0 &&
                                       state.exp_avg_sq.capacity() > 0 &&
                                       state.exp_avg_scale.capacity() > 0 &&
                                       state.exp_avg_sq_scale.capacity() > 0;
        const bool grad_fits = !state.grad.is_valid() || new_size <= state.grad.capacity();
        const bool fits_in_capacity = grad_fits &&
                                      new_size <= state.exp_avg.capacity() &&
                                      new_size <= state.exp_avg_sq.capacity() &&
                                      new_size <= state.exp_avg_scale.capacity() &&
                                      new_size <= state.exp_avg_sq_scale.capacity();
        if (all_have_capacity && fits_in_capacity) {
            // Copy optimizer momentum (bytes + scale) for duplicated Gaussians.
            state.exp_avg.append_gather(indices);
            state.exp_avg_sq.append_gather(indices);
            state.exp_avg_scale.append_gather(indices);
            state.exp_avg_sq_scale.append_gather(indices);
            if (state.grad.is_valid())
                state.grad.append_zeros(n_new);
            state.size = new_size;
            state.capacity = state.exp_avg.capacity();
            LOG_DEBUG("extend_state_by_gather({}): fast path done", name);
            return;
        }
        LOG_WARN("extend_state_by_gather({}): SLOW PATH triggered (all_have_cap={}, fits={})", name, all_have_capacity, fits_in_capacity);

        // Slow path: reallocate via cat (dtype-agnostic; handles uint8 moments + fp32 scales).
        if (state.grad.is_valid()) {
            const auto& shape = param.shape();
            std::vector<size_t> new_dims(shape.dims());
            new_dims[0] = new_size;
            state.grad = lfs::core::Tensor::zeros(lfs::core::TensorShape(new_dims), param.device());
        }
        state.exp_avg = lfs::core::Tensor::cat({state.exp_avg, state.exp_avg.index_select(0, indices)}, 0);
        state.exp_avg_sq = lfs::core::Tensor::cat({state.exp_avg_sq, state.exp_avg_sq.index_select(0, indices)}, 0);
        state.exp_avg_scale = lfs::core::Tensor::cat({state.exp_avg_scale, state.exp_avg_scale.index_select(0, indices)}, 0);
        state.exp_avg_sq_scale = lfs::core::Tensor::cat({state.exp_avg_sq_scale, state.exp_avg_sq_scale.index_select(0, indices)}, 0);
        state.size = new_size;
        state.capacity = 0;
        LOG_DEBUG("extend_state_by_gather: {} slow path, new size = {}", name, new_size);
    }

    void AdamOptimizer::extend_state_for_new_params(ParamType type, const size_t n_new) {
        const auto name = param_name(type);
        if (!states_.contains(name)) {
            LOG_DEBUG("extend_state_for_new_params({}): state not found, skipping", name);
            return;
        }

        // Skip zero-coefficient ShN (sh-degree 0): with swizzled storage the tensor is
        // 1D and may be empty when allocated for SH degree 0.
        if (type == ParamType::ShN) {
            const auto& shN_param = get_param(type);
            if (!shN_param.is_valid() || shN_param.numel() == 0 ||
                splat_data_.max_sh_coeffs_rest() == 0) {
                return;
            }
        }

        auto& param = get_param(type);
        auto& state = states_[name];
        if (type == ParamType::ShN &&
            (splat_data_.max_sh_coeffs_rest() == 0 ||
             !state.exp_avg.is_valid() || !state.exp_avg_sq.is_valid())) {
            return;
        }

        // For swizzled shN, moment growth is measured in floats: (swizzled_floats(N+n_new) -
        // swizzled_floats(N)). For everything else it is n_new rows. Scales always grow by
        // n_new primitives. First-moment bytes must use the signed zero-point (128), not byte
        // zero: inactive shN slots can share a nonzero per-primitive scale once lower SH bands
        // start training.
        const size_t growth = compute_state_growth(type, n_new);
        const size_t new_size = state.size + growth;
        const size_t scale_cur = state.exp_avg_scale.is_valid() ? state.exp_avg_scale.shape()[0] : 0;
        const size_t scale_new = scale_cur + n_new;
        const size_t moment_row_size = tensor_row_size(state.exp_avg);

        if (!param.is_valid() || param.shape().rank() == 0) {
            throw std::runtime_error("extend_state: " + name + " invalid");
        }
        if (!state.exp_avg.is_valid() || state.exp_avg.ndim() == 0) {
            throw std::runtime_error("extend_state: " + name + " state invalid");
        }

        // Fast path: use reserved capacity (all tensors must have capacity)
        const bool grad_has_capacity = !state.grad.is_valid() || state.grad.capacity() > 0;
        const bool all_have_capacity = grad_has_capacity &&
                                       state.exp_avg.capacity() > 0 &&
                                       state.exp_avg_sq.capacity() > 0 &&
                                       state.exp_avg_scale.capacity() > 0 &&
                                       state.exp_avg_sq_scale.capacity() > 0;
        const bool grad_fits = !state.grad.is_valid() || new_size <= state.grad.capacity();
        const bool fits_in_capacity = grad_fits &&
                                      new_size <= state.exp_avg.capacity() &&
                                      new_size <= state.exp_avg_sq.capacity() &&
                                      scale_new <= state.exp_avg_scale.capacity() &&
                                      scale_new <= state.exp_avg_sq_scale.capacity();
        if (all_have_capacity && fits_in_capacity) {
            if (state.grad.is_valid())
                state.grad.append_zeros(growth);
            const size_t moment_offset = state.size * moment_row_size;
            state.exp_avg.append_zeros(growth);
            state.exp_avg_sq.append_zeros(growth);
            fill_quantized_moment_zero_point_range(
                state.exp_avg,
                moment_offset,
                growth * moment_row_size);
            state.exp_avg_scale.append_zeros(n_new);
            state.exp_avg_sq_scale.append_zeros(n_new);
            state.size = new_size;
            state.capacity = state.exp_avg.capacity();
            LOG_DEBUG("extend_state_for_new_params({}): fast path done, new size = {}", name, new_size);
            return;
        }
        LOG_WARN("extend_state_for_new_params({}): SLOW PATH triggered (all_have_cap={}, fits={})", name, all_have_capacity, fits_in_capacity);

        // Slow path: reallocate without extra capacity.
        const auto& shape = param.shape();
        std::vector<size_t> new_dims(shape.dims());
        new_dims[0] = new_size;
        const auto tensor_shape = lfs::core::TensorShape(new_dims);
        if (state.grad.is_valid())
            state.grad = lfs::core::Tensor::zeros(tensor_shape, param.device());

        auto new_exp_avg = lfs::core::Tensor::empty(tensor_shape, param.device(), lfs::core::DataType::UInt8);
        auto new_exp_avg_sq = lfs::core::Tensor::empty(tensor_shape, param.device(), lfs::core::DataType::UInt8);
        const cudaStream_t stream = lfs::core::getCurrentCUDAStream();

        const size_t row_size = shape[0] == 0 ? 0 : param.numel() / shape[0];
        if (state.size > 0 && state.exp_avg.numel() > 0) {
            const size_t old_bytes = state.exp_avg.numel() * sizeof(uint8_t);
            LFS_CUDA_CHECK(cudaMemcpyAsync(new_exp_avg.ptr<uint8_t>(), state.exp_avg.ptr<uint8_t>(), old_bytes, cudaMemcpyDeviceToDevice, stream));
            LFS_CUDA_CHECK(cudaMemcpyAsync(new_exp_avg_sq.ptr<uint8_t>(), state.exp_avg_sq.ptr<uint8_t>(), old_bytes, cudaMemcpyDeviceToDevice, stream));
        }
        const size_t offset = state.exp_avg.numel() * sizeof(uint8_t);
        const size_t new_bytes = growth * row_size * sizeof(uint8_t);
        LFS_CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<char*>(new_exp_avg.ptr<uint8_t>()) + offset, QUANTIZED_MOMENT_ZERO_POINT, new_bytes, stream));
        LFS_CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<char*>(new_exp_avg_sq.ptr<uint8_t>()) + offset, 0, new_bytes, stream));

        new_exp_avg.set_stream(stream);
        new_exp_avg_sq.set_stream(stream);
        state.exp_avg = std::move(new_exp_avg);
        state.exp_avg_sq = std::move(new_exp_avg_sq);

        const auto scale_shape = lfs::core::TensorShape({scale_new});
        auto new_m_scale = lfs::core::Tensor::zeros(scale_shape, param.device());
        auto new_v_scale = lfs::core::Tensor::zeros(scale_shape, param.device());
        if (scale_cur > 0 && state.exp_avg_scale.numel() > 0) {
            LFS_CUDA_CHECK(cudaMemcpyAsync(new_m_scale.ptr<float>(), state.exp_avg_scale.ptr<float>(), scale_cur * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            LFS_CUDA_CHECK(cudaMemcpyAsync(new_v_scale.ptr<float>(), state.exp_avg_sq_scale.ptr<float>(), scale_cur * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        }
        new_m_scale.set_stream(stream);
        new_v_scale.set_stream(stream);
        state.exp_avg_scale = std::move(new_m_scale);
        state.exp_avg_sq_scale = std::move(new_v_scale);

        state.size = new_size;
        state.capacity = 0;
    }

    size_t AdamOptimizer::compute_new_capacity(const size_t current_capacity, const size_t required_size) const {
        if (current_capacity == 0) {
            if (config_.initial_capacity > 0) {
                return std::max(config_.initial_capacity, required_size);
            }
            return static_cast<size_t>(required_size * DEFAULT_GROWTH_MULTIPLIER);
        }
        const size_t grown = static_cast<size_t>(current_capacity * config_.growth_factor);
        return std::max(grown, required_size);
    }

    const AdamParamState* AdamOptimizer::get_state(ParamType type) const {
        const auto name = param_name(type);
        const auto it = states_.find(name);
        return (it != states_.end()) ? &it->second : nullptr;
    }

    AdamParamState* AdamOptimizer::get_state_mutable(ParamType type) {
        const auto name = param_name(type);
        auto it = states_.find(name);
        return (it != states_.end()) ? &it->second : nullptr;
    }

    int64_t AdamOptimizer::get_step_count(ParamType type) const {
        const auto name = param_name(type);
        const auto it = states_.find(name);
        return (it != states_.end()) ? it->second.step_count : 0;
    }

    void AdamOptimizer::set_state(ParamType type, const AdamParamState& state) {
        // Accept legacy fp32 moments by quantising on the way in.
        if (state.exp_avg.is_valid() && state.exp_avg.dtype() == lfs::core::DataType::Float32) {
            AdamParamState converted = state;
            quantize_float_moments(type, converted, state.exp_avg.clone(), state.exp_avg_sq.clone());
            converted.size = state.size;
            converted.capacity = state.size;
            states_[param_name(type)] = std::move(converted);
            return;
        }
        states_[param_name(type)] = state;
    }

    void AdamOptimizer::add_new_params(ParamType type, const lfs::core::Tensor& new_values, const bool validate) {
        auto& param = get_param(type);

        if (validate) {
            if (new_values.ndim() != param.ndim()) {
                throw std::runtime_error("add_new_params: rank mismatch");
            }
            for (size_t i = 1; i < param.ndim(); i++) {
                if (new_values.shape()[i] != param.shape()[i]) {
                    throw std::runtime_error("add_new_params: shape mismatch");
                }
            }
            if (new_values.device() != param.device()) {
                throw std::runtime_error("add_new_params: device mismatch");
            }
        }

        const size_t n_new = new_values.shape()[0];
        if (n_new == 0) {
            return;
        }

        const size_t old_size = param.shape()[0];
        const size_t new_size = old_size + n_new;
        if (param.capacity() >= new_size) {
            param.append_zeros(n_new);
            auto appended = param.slice(0, old_size, new_size);
            appended.copy_from(new_values);
        } else {
            param = lfs::core::Tensor::cat({param, new_values}, 0);
        }
        extend_state_for_new_params(type, n_new);
    }

    void AdamOptimizer::add_new_params_gather(ParamType type, const lfs::core::Tensor& indices) {
        auto& param = get_param(type);

        // ShN: swizzled 1D buffer. append_gather doesn't apply; do swizzle-aware gather.
        if (type == ParamType::ShN) {
            if (!param.is_valid() || param.numel() == 0) {
                // SH degree 0 — no shN data to extend. The param length tracks via sh0.
                return;
            }
            const auto layout_rest = static_cast<uint32_t>(splat_data_.max_sh_coeffs_rest());
            if (layout_rest == 0)
                return;
            const size_t n_new = indices.numel();
            if (n_new == 0)
                return;
            const size_t old_N = static_cast<size_t>(splat_data_.size()) - n_new;
            const size_t new_N = old_N + n_new;
            const size_t old_floats = lfs::core::sh_swizzled_float_count(old_N, layout_rest);
            const size_t new_floats = lfs::core::sh_swizzled_float_count(new_N, layout_rest);
            const size_t growth = new_floats - old_floats;

            // Extend the swizzled param buffer by `growth` zero floats. The new range
            // [old_floats, new_floats) covers all swizzled slots of primitives in
            // [old_N, new_N).
            param.append_zeros(growth);

            const bool indices_are_i64 = indices.dtype() == lfs::core::DataType::Int64;
            lfs::core::Tensor indices_i32;
            const int* indices_i32_ptr = nullptr;
            if (!indices_are_i64) {
                indices_i32 = indices.dtype() == lfs::core::DataType::Int32
                                  ? indices
                                  : indices.to(lfs::core::DataType::Int32);
                indices_i32_ptr = indices_i32.ptr<int>();
            }

            auto gather_new_swizzled_rows = [&](lfs::core::Tensor& tensor) {
                float* ptr = tensor.ptr<float>();
                if (indices_are_i64) {
                    lfs::core::shN_swizzled_gather_self_i64(
                        ptr, ptr, indices.ptr<int64_t>(), n_new, old_N, layout_rest);
                } else {
                    lfs::core::shN_swizzled_gather_self(
                        ptr, ptr, indices_i32_ptr, n_new, old_N, layout_rest);
                }
            };
            auto gather_new_swizzled_rows_u8 = [&](lfs::core::Tensor& tensor) {
                uint8_t* ptr = tensor.ptr<uint8_t>();
                if (indices_are_i64) {
                    lfs::core::shN_swizzled_gather_self_u8_i64(
                        ptr, ptr, indices.ptr<int64_t>(), n_new, old_N, layout_rest);
                } else {
                    lfs::core::shN_swizzled_gather_self_u8(
                        ptr, ptr, indices_i32_ptr, n_new, old_N, layout_rest);
                }
            };

            gather_new_swizzled_rows(param);

            // Extend the Adam state. Moments are uint8 swizzled (gather parent bytes); scales
            // are per-primitive {N} (gather parent scale by primitive index via append_gather).
            const auto name = param_name(type);
            if (states_.contains(name)) {
                auto& state = states_[name];
                if (state.exp_avg.is_valid() && state.exp_avg_sq.is_valid() &&
                    state.exp_avg_scale.is_valid() && state.exp_avg_sq_scale.is_valid()) {
                    const bool fits =
                        state.exp_avg.capacity() >= new_floats &&
                        state.exp_avg_sq.capacity() >= new_floats &&
                        state.exp_avg_scale.capacity() >= new_N &&
                        state.exp_avg_sq_scale.capacity() >= new_N;
                    if (fits) {
                        state.exp_avg.append_zeros(growth);
                        state.exp_avg_sq.append_zeros(growth);
                        gather_new_swizzled_rows_u8(state.exp_avg);
                        gather_new_swizzled_rows_u8(state.exp_avg_sq);
                        state.exp_avg_scale.append_gather(indices);
                        state.exp_avg_sq_scale.append_gather(indices);
                        state.size = new_floats;
                        state.capacity = state.exp_avg.capacity();
                    } else {
                        // Fallback: no reserved capacity. Reallocate at exact size.
                        auto realloc_u8 = [&](lfs::core::Tensor& t) {
                            auto fresh = lfs::core::Tensor::zeros({new_floats}, lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
                            LFS_CUDA_CHECK(cudaMemcpyAsync(
                                fresh.ptr<uint8_t>(), t.ptr<uint8_t>(),
                                old_floats * sizeof(uint8_t),
                                cudaMemcpyDeviceToDevice, nullptr));
                            t = std::move(fresh);
                        };
                        auto realloc_m = [&](lfs::core::Tensor& t) {
                            auto fresh = lfs::core::Tensor::zeros({new_floats}, lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
                            LFS_CUDA_CHECK(cudaMemsetAsync(fresh.ptr<uint8_t>(), QUANTIZED_MOMENT_ZERO_POINT,
                                                           new_floats * sizeof(uint8_t), fresh.stream()));
                            LFS_CUDA_CHECK(cudaMemcpyAsync(
                                fresh.ptr<uint8_t>(), t.ptr<uint8_t>(),
                                old_floats * sizeof(uint8_t),
                                cudaMemcpyDeviceToDevice, nullptr));
                            t = std::move(fresh);
                        };
                        realloc_m(state.exp_avg);
                        realloc_u8(state.exp_avg_sq);
                        gather_new_swizzled_rows_u8(state.exp_avg);
                        gather_new_swizzled_rows_u8(state.exp_avg_sq);
                        state.exp_avg_scale = lfs::core::Tensor::cat(
                            {state.exp_avg_scale, state.exp_avg_scale.index_select(0, indices)}, 0);
                        state.exp_avg_sq_scale = lfs::core::Tensor::cat(
                            {state.exp_avg_sq_scale, state.exp_avg_sq_scale.index_select(0, indices)}, 0);
                        state.size = new_floats;
                        state.capacity = 0;
                    }
                }
            }
            return;
        }

        if (!param.is_valid()) {
            LOG_ERROR("add_new_params_gather: {} not initialized", param_name(type));
            return;
        }

        // Regular case for tensors with data
        if (param.ndim() >= 2 && param.shape()[1] == 0) {
            // This shouldn't happen for non-ShN tensors, but handle gracefully
            return;
        }

        if (indices.device() != param.device()) {
            LOG_ERROR("add_new_params_gather: device mismatch");
            return;
        }

        const size_t n_new = indices.numel();
        param.append_gather(indices);
        extend_state_for_new_params(type, n_new);
    }

    void AdamOptimizer::relocate_params_at_indices(ParamType type, const std::vector<int64_t>& indices) {
        if (indices.empty())
            return;

        const auto& param = get_param(type);
        for (const auto idx : indices) {
            if (idx < 0 || static_cast<size_t>(idx) >= param.shape()[0]) {
                throw std::runtime_error("relocate_params_at_indices: index out of bounds");
            }
        }

        const cudaStream_t stream = lfs::core::getCurrentCUDAStream();
        const size_t idx_bytes = indices.size() * sizeof(int64_t);
        int64_t* d_indices = nullptr;
        LFS_CUDA_CHECK(cudaMallocAsync(&d_indices, idx_bytes, stream));
        LFS_CUDA_CHECK(cudaMemcpyAsync(d_indices, indices.data(), idx_bytes, cudaMemcpyHostToDevice, stream));
        relocate_params_at_indices_gpu(type, d_indices, indices.size());
        LFS_CUDA_CHECK(cudaFreeAsync(d_indices, stream));
    }

    void AdamOptimizer::relocate_params_at_indices_gpu(ParamType type, const int64_t* indices_device, const size_t n_indices) {
        if (n_indices == 0)
            return;

        const auto name = param_name(type);
        if (!states_.contains(name))
            return;

        // Skip ShN when not initialized (sh_degree=0 case): swizzled storage is 1D.
        if (type == ParamType::ShN) {
            const auto& param = get_param(type);
            if (!param.is_valid() || param.numel() == 0 ||
                splat_data_.max_sh_coeffs_rest() == 0) {
                return;
            }
        }

        auto& state = states_[name];

        // Validate scales. Contiguous params reset quantized bytes and scales. Swizzled shN only
        // needs scales reset because its inactive slots are held at the signed zero-point.
        if (!state.exp_avg_scale.is_valid() || state.exp_avg_scale.ptr<float>() == nullptr ||
            !state.exp_avg_sq_scale.is_valid() || state.exp_avg_sq_scale.ptr<float>() == nullptr) {
            LOG_WARN("relocate_params_at_indices_gpu: {} scale tensor is invalid or null", name);
            return;
        }

        const cudaStream_t stream = lfs::core::getCurrentCUDAStream();
        lfs::core::waitForCUDAStream(stream, state.exp_avg_scale.stream());
        lfs::core::waitForCUDAStream(stream, state.exp_avg_sq_scale.stream());

        if (type != ParamType::ShN && state.exp_avg.is_valid() && state.exp_avg_sq.is_valid()) {
            const int row_size = static_cast<int>(tensor_row_size(state.exp_avg));
            lfs::core::waitForCUDAStream(stream, state.exp_avg.stream());
            lfs::core::waitForCUDAStream(stream, state.exp_avg_sq.stream());
            fast_lfs::optimizer::zero_quantized_rows_at_indices(
                state.exp_avg.ptr<uint8_t>(),
                state.exp_avg_scale.ptr<float>(),
                indices_device,
                n_indices,
                row_size,
                QUANTIZED_MOMENT_ZERO_POINT,
                stream);
            fast_lfs::optimizer::zero_quantized_rows_at_indices(
                state.exp_avg_sq.ptr<uint8_t>(),
                state.exp_avg_sq_scale.ptr<float>(),
                indices_device,
                n_indices,
                row_size,
                0,
                stream);
            state.exp_avg.set_stream(stream);
            state.exp_avg_sq.set_stream(stream);
        } else {
            fast_lfs::optimizer::zero_rows_at_indices(state.exp_avg_scale.ptr<float>(), indices_device, n_indices, 1, stream);
            fast_lfs::optimizer::zero_rows_at_indices(state.exp_avg_sq_scale.ptr<float>(), indices_device, n_indices, 1, stream);
        }

        state.exp_avg_scale.set_stream(stream);
        state.exp_avg_sq_scale.set_stream(stream);
    }

    namespace {
        constexpr uint32_t ADAM_STATE_MAGIC = 0x4C464144; // "LFAD"
        // v1: fp32 moments (no scales). v2: uint8 quantised moments + per-primitive fp32 scales.
        constexpr uint32_t ADAM_STATE_VERSION = 2;
    } // namespace

    void AdamOptimizer::serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&ADAM_STATE_MAGIC), sizeof(ADAM_STATE_MAGIC));
        os.write(reinterpret_cast<const char*>(&ADAM_STATE_VERSION), sizeof(ADAM_STATE_VERSION));

        os.write(reinterpret_cast<const char*>(&config_.lr), sizeof(config_.lr));
        os.write(reinterpret_cast<const char*>(&config_.beta1), sizeof(config_.beta1));
        os.write(reinterpret_cast<const char*>(&config_.beta2), sizeof(config_.beta2));
        os.write(reinterpret_cast<const char*>(&config_.eps), sizeof(config_.eps));
        os.write(reinterpret_cast<const char*>(&config_.growth_factor), sizeof(config_.growth_factor));
        os.write(reinterpret_cast<const char*>(&config_.initial_capacity), sizeof(config_.initial_capacity));

        const auto num_param_lrs = static_cast<uint32_t>(config_.param_lrs.size());
        os.write(reinterpret_cast<const char*>(&num_param_lrs), sizeof(num_param_lrs));
        for (const auto& [name, lr] : config_.param_lrs) {
            const auto name_len = static_cast<uint32_t>(name.size());
            os.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
            os.write(name.data(), name_len);
            os.write(reinterpret_cast<const char*>(&lr), sizeof(lr));
        }

        const auto state_complete = [](const AdamParamState& s) {
            return s.exp_avg.is_valid() && s.exp_avg_sq.is_valid() &&
                   s.exp_avg_scale.is_valid() && s.exp_avg_sq_scale.is_valid();
        };

        uint32_t num_states = 0;
        for (const auto& [_, state] : states_) {
            if (state_complete(state))
                ++num_states;
        }
        os.write(reinterpret_cast<const char*>(&num_states), sizeof(num_states));

        for (const auto& [name, state] : states_) {
            if (!state_complete(state))
                continue;

            const auto name_len = static_cast<uint32_t>(name.size());
            os.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
            os.write(name.data(), name_len);
            os.write(reinterpret_cast<const char*>(&state.step_count), sizeof(state.step_count));
            os.write(reinterpret_cast<const char*>(&state.capacity), sizeof(state.capacity));
            os.write(reinterpret_cast<const char*>(&state.size), sizeof(state.size));
            os << state.exp_avg << state.exp_avg_sq
               << state.exp_avg_scale << state.exp_avg_sq_scale;
        }
        LOG_DEBUG("Serialized AdamOptimizer: {} states", num_states);
    }

    void AdamOptimizer::deserialize(std::istream& is) {
        uint32_t magic = 0, version = 0;
        lfs::core::serialization_detail::read_exact(is, &magic, sizeof(magic), "Adam magic");
        lfs::core::serialization_detail::read_exact(is, &version, sizeof(version), "Adam version");

        if (magic != ADAM_STATE_MAGIC) {
            throw std::runtime_error("Invalid AdamOptimizer checkpoint");
        }
        if (version < 1 || version > ADAM_STATE_VERSION) {
            throw std::runtime_error("Unsupported checkpoint version");
        }

        AdamConfig loaded_config;
        lfs::core::serialization_detail::read_exact(is, &loaded_config.lr, sizeof(loaded_config.lr), "Adam learning rate");
        lfs::core::serialization_detail::read_exact(is, &loaded_config.beta1, sizeof(loaded_config.beta1), "Adam beta1");
        lfs::core::serialization_detail::read_exact(is, &loaded_config.beta2, sizeof(loaded_config.beta2), "Adam beta2");
        lfs::core::serialization_detail::read_exact(is, &loaded_config.eps, sizeof(loaded_config.eps), "Adam epsilon");
        lfs::core::serialization_detail::read_exact(
            is, &loaded_config.growth_factor, sizeof(loaded_config.growth_factor), "Adam growth factor");
        lfs::core::serialization_detail::read_exact(
            is, &loaded_config.initial_capacity, sizeof(loaded_config.initial_capacity), "Adam initial capacity");
        if (!std::isfinite(loaded_config.lr) || loaded_config.lr < 0.0f ||
            !std::isfinite(loaded_config.beta1) || loaded_config.beta1 < 0.0 || loaded_config.beta1 >= 1.0 ||
            !std::isfinite(loaded_config.beta2) || loaded_config.beta2 < 0.0 || loaded_config.beta2 >= 1.0 ||
            !std::isfinite(loaded_config.eps) || loaded_config.eps <= 0.0 ||
            !std::isfinite(loaded_config.growth_factor) || loaded_config.growth_factor < 1.0f ||
            loaded_config.initial_capacity > lfs::core::MAX_CHECKPOINT_GAUSSIANS) {
            throw std::runtime_error("Invalid AdamOptimizer checkpoint configuration");
        }

        const auto type_from_name = [](const std::string_view name) -> std::optional<ParamType> {
            if (name == "means")
                return ParamType::Means;
            if (name == "sh0")
                return ParamType::Sh0;
            if (name == "shN")
                return ParamType::ShN;
            if (name == "scaling")
                return ParamType::Scaling;
            if (name == "rotation")
                return ParamType::Rotation;
            if (name == "opacity")
                return ParamType::Opacity;
            return std::nullopt;
        };

        uint32_t num_param_lrs = 0;
        lfs::core::serialization_detail::read_exact(
            is, &num_param_lrs, sizeof(num_param_lrs), "Adam parameter learning-rate count");
        if (num_param_lrs > all_param_types().size())
            throw std::runtime_error("Invalid AdamOptimizer checkpoint: too many parameter learning rates");
        for (uint32_t i = 0; i < num_param_lrs; ++i) {
            uint32_t name_len = 0;
            lfs::core::serialization_detail::read_exact(
                is, &name_len, sizeof(name_len), "Adam parameter name length");
            if (name_len == 0 || name_len > 16)
                throw std::runtime_error("Invalid AdamOptimizer checkpoint: parameter name length is out of bounds");
            std::string name(name_len, '\0');
            lfs::core::serialization_detail::read_exact(is, name.data(), name_len, "Adam parameter name");
            double lr = 0.0;
            lfs::core::serialization_detail::read_exact(is, &lr, sizeof(lr), "Adam parameter learning rate");
            if (!type_from_name(name) || !std::isfinite(lr) || lr < 0.0 ||
                !loaded_config.param_lrs.emplace(name, lr).second) {
                throw std::runtime_error("Invalid AdamOptimizer checkpoint: invalid parameter learning rate");
            }
        }

        uint32_t num_states = 0;
        lfs::core::serialization_detail::read_exact(is, &num_states, sizeof(num_states), "Adam state count");
        if (num_states > all_param_types().size())
            throw std::runtime_error("Invalid AdamOptimizer checkpoint: too many states");

        std::unordered_map<std::string, AdamParamState> loaded_states;
        for (uint32_t i = 0; i < num_states; ++i) {
            uint32_t name_len = 0;
            lfs::core::serialization_detail::read_exact(is, &name_len, sizeof(name_len), "Adam state name length");
            if (name_len == 0 || name_len > 16)
                throw std::runtime_error("Invalid AdamOptimizer checkpoint: state name length is out of bounds");
            std::string name(name_len, '\0');
            lfs::core::serialization_detail::read_exact(is, name.data(), name_len, "Adam state name");
            const auto maybe_type = type_from_name(name);
            if (!maybe_type || loaded_states.contains(name))
                throw std::runtime_error("Invalid AdamOptimizer checkpoint: unknown or duplicate state name");

            AdamParamState state;
            lfs::core::serialization_detail::read_exact(is, &state.step_count, sizeof(state.step_count), "Adam step count");
            lfs::core::serialization_detail::read_exact(is, &state.capacity, sizeof(state.capacity), "Adam state capacity");
            lfs::core::serialization_detail::read_exact(is, &state.size, sizeof(state.size), "Adam state size");
            if (state.step_count < 0 || state.size > state.capacity)
                throw std::runtime_error("Invalid AdamOptimizer checkpoint: inconsistent state bounds");

            const bool is_shN = (name == "shN");
            const ParamType ptype = *maybe_type;
            const auto& parameter = get_param(ptype);
            const size_t expected_state_size = version == 1 && is_shN
                                                   ? static_cast<size_t>(splat_data_.size())
                                                   : parameter.shape()[0];
            if (!parameter.is_valid() || state.size != expected_state_size)
                throw std::runtime_error("Invalid AdamOptimizer checkpoint: state size does not match model");

            if (version == 1) {
                // Legacy fp32 moments (no scales). Read, bridge canonical shN -> swizzled, quantise.
                lfs::core::Tensor favg, favg_sq;
                is >> favg >> favg_sq;
                if (!favg.is_valid() || favg.dtype() != lfs::core::DataType::Float32 ||
                    favg.shape() != favg_sq.shape() || favg_sq.dtype() != lfs::core::DataType::Float32) {
                    throw std::runtime_error("Invalid AdamOptimizer checkpoint: legacy moment schema mismatch");
                }
                if ((!is_shN && favg.shape() != parameter.shape()) ||
                    (is_shN && (favg.ndim() != 3 ||
                                favg.shape()[0] != static_cast<size_t>(splat_data_.size()) ||
                                favg.shape()[2] != 3))) {
                    throw std::runtime_error("Invalid AdamOptimizer checkpoint: legacy moment shape mismatch");
                }
                favg = favg.cuda();
                favg_sq = favg_sq.cuda();

                if (is_shN && favg.is_valid() && favg.ndim() == 3) {
                    const size_t N = static_cast<size_t>(favg.shape()[0]);
                    const uint32_t K = static_cast<uint32_t>(favg.shape()[1]);
                    const uint32_t layout_rest =
                        static_cast<uint32_t>(std::max<size_t>(splat_data_.max_sh_coeffs_rest(), K));
                    const size_t logical_floats = lfs::core::sh_swizzled_float_count(N, layout_rest);
                    auto swizzled_avg = lfs::core::Tensor::zeros({logical_floats}, lfs::core::Device::CUDA);
                    auto swizzled_avg_sq = lfs::core::Tensor::zeros({logical_floats}, lfs::core::Device::CUDA);
                    lfs::core::reorder_sh_to_swizzled(favg.ptr<float>(), swizzled_avg.ptr<float>(), N, K, layout_rest);
                    lfs::core::reorder_sh_to_swizzled(favg_sq.ptr<float>(), swizzled_avg_sq.ptr<float>(), N, K, layout_rest);
                    favg = std::move(swizzled_avg);
                    favg_sq = std::move(swizzled_avg_sq);
                    state.size = logical_floats;
                }
                quantize_float_moments(ptype, state, std::move(favg), std::move(favg_sq));
            } else {
                is >> state.exp_avg >> state.exp_avg_sq >> state.exp_avg_scale >> state.exp_avg_sq_scale;
                const size_t primitive_rows = static_cast<size_t>(splat_data_.size());
                if (state.exp_avg.dtype() != lfs::core::DataType::UInt8 ||
                    state.exp_avg_sq.dtype() != lfs::core::DataType::UInt8 ||
                    state.exp_avg.shape() != parameter.shape() ||
                    state.exp_avg_sq.shape() != parameter.shape() ||
                    state.exp_avg_scale.dtype() != lfs::core::DataType::Float32 ||
                    state.exp_avg_sq_scale.dtype() != lfs::core::DataType::Float32 ||
                    state.exp_avg_scale.ndim() != 1 || state.exp_avg_scale.numel() != primitive_rows ||
                    state.exp_avg_sq_scale.shape() != state.exp_avg_scale.shape()) {
                    throw std::runtime_error("Invalid AdamOptimizer checkpoint: moment schema mismatch");
                }
                state.exp_avg = state.exp_avg.cuda();
                state.exp_avg_sq = state.exp_avg_sq.cuda();
                state.exp_avg_scale = state.exp_avg_scale.cuda();
                state.exp_avg_sq_scale = state.exp_avg_sq_scale.cuda();
            }

            // Serialized capacity is advisory and may be attacker-controlled.
            // The validated checkpoint max_cap is reserved by load_checkpoint
            // after all state has parsed successfully.
            state.capacity = state.exp_avg.is_valid() ? state.exp_avg.shape()[0] : state.size;
            loaded_states.emplace(std::move(name), std::move(state));
        }

        config_ = std::move(loaded_config);
        states_ = std::move(loaded_states);

        // Gradient buffers are transient and allocated lazily by get_grad().
        LOG_DEBUG("Deserialized AdamOptimizer: {} states", num_states);
    }

    void AdamOptimizer::adopt_checkpoint_state(AdamOptimizer& loaded) noexcept {
        static_assert(std::is_nothrow_move_assignable_v<AdamConfig>);
        static_assert(std::is_nothrow_move_assignable_v<decltype(states_)>);
        config_ = std::move(loaded.config_);
        states_ = std::move(loaded.states_);
    }

    void AdamOptimizer::reserve_capacity(const size_t capacity) {
        for (auto& [name, state] : states_) {
            // Moments use float-count capacity for swizzled shN, primitive rows otherwise.
            // Scales are always per-primitive.
            const size_t target_capacity =
                name == "shN" ? lfs::core::sh_swizzled_float_count(
                                    capacity,
                                    static_cast<uint32_t>(splat_data_.max_sh_coeffs_rest()))
                              : capacity;
            if (target_capacity > state.capacity) {
                if (state.grad.is_valid())
                    state.grad.reserve(target_capacity);
                if (state.exp_avg.is_valid())
                    state.exp_avg.reserve(target_capacity);
                if (state.exp_avg_sq.is_valid())
                    state.exp_avg_sq.reserve(target_capacity);
                state.capacity = target_capacity;
            }
            if (state.exp_avg_scale.is_valid() && capacity > state.exp_avg_scale.capacity()) {
                state.exp_avg_scale.reserve(capacity);
                state.exp_avg_sq_scale.reserve(capacity);
            }
        }
    }

    void AdamOptimizer::reset_state(const ParamType type) {
        auto* state = get_state_mutable(type);
        if (!state || !state->exp_avg.is_valid()) {
            return;
        }

        // Zero scales -> all moments dequantise to zero, but m bytes still need the signed
        // zero-point so inactive shN slots stay neutral after another band updates the scale.
        fill_quantized_moment_zero_point(state->exp_avg);
        state->exp_avg_sq.zero_();
        if (state->exp_avg_scale.is_valid())
            state->exp_avg_scale.zero_();
        if (state->exp_avg_sq_scale.is_valid())
            state->exp_avg_sq_scale.zero_();
        state->step_count = 0;
    }

    void AdamOptimizer::invalidate_state(const ParamType type) {
        const auto name = param_name(type);
        auto it = states_.find(name);
        if (it == states_.end()) {
            return;
        }

        it->second.exp_avg = {};
        it->second.exp_avg_sq = {};
        it->second.exp_avg_scale = {};
        it->second.exp_avg_sq_scale = {};
        it->second.grad = {};
        it->second.size = 0;
        it->second.capacity = 0;
        it->second.step_count = 0;
    }

} // namespace lfs::training
