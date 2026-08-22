/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "adam_optimizer.hpp"
#include "adam_api.h"
#include "core/alloc_counter.hpp"
#include "core/assert.hpp"
#include "core/checkpoint_format.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "core/sh_value_quant.hpp"
#include "core/splat_exportable_storage.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "lfs/training/joint_adam_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include <algorithm>
#include <atomic>
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

        std::atomic<uint64_t> g_adam_slow_path_grow_count{0};

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

    } // namespace

    void ensure_joint_bounds_capacity(lfs::core::Tensor& joint_bounds,
                                      const size_t n_prims,
                                      const size_t capacity_prims,
                                      const lfs::core::Device device,
                                      const bool zero_all) {
        using namespace joint_adam;
        const size_t nb = n_bounds_for_prims(n_prims);
        const size_t nb_cap = n_bounds_for_prims(std::max(n_prims, capacity_prims));
        if (nb == 0) {
            joint_bounds = {};
            return;
        }
        lfs::core::alloc_counter::ScopedSite site("joint_bounds");
        if (joint_bounds.is_valid() && joint_bounds.ndim() == 2 &&
            joint_bounds.capacity() >= nb) {
            const size_t cur = joint_bounds.shape()[0];
            if (cur < nb) {
                joint_bounds.append_zeros(nb - cur);
            }
            if (zero_all) {
                joint_bounds.zero_();
            }
            return;
        }
        // Fresh allocation with headroom. Preserve prior block bounds when growing
        // (unless compact zero-init).
        lfs::core::Tensor new_bounds;
        const lfs::core::TensorShape shape({nb, size_t{4}});
        if (nb_cap > nb && nb_cap > 0) {
            new_bounds = lfs::core::Tensor::zeros_direct(shape, nb_cap, device);
        } else {
            new_bounds = lfs::core::Tensor::zeros(shape, device);
        }
        if (!zero_all && joint_bounds.is_valid() && joint_bounds.numel() > 0) {
            // Keep the source alive until the D2D copy finishes. Destroying
            // joint_bounds immediately after cudaMemcpyAsync races the async read.
            const cudaStream_t stream = lfs::core::getCurrentCUDAStream();
            lfs::core::waitForCUDAStream(stream, joint_bounds.stream());
            lfs::core::waitForCUDAStream(stream, new_bounds.stream());
            const size_t copy_n = std::min(joint_bounds.numel(), new_bounds.numel());
            auto old_bounds = std::move(joint_bounds);
            LFS_CUDA_CHECK(cudaMemcpyAsync(
                new_bounds.ptr<float>(), old_bounds.ptr<float>(),
                copy_n * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
            joint_bounds = std::move(new_bounds);
            // old_bounds destroyed after sync
        } else {
            joint_bounds = std::move(new_bounds);
        }
    }

    uint64_t AdamOptimizer::slow_path_grow_count() noexcept {
        return g_adam_slow_path_grow_count.load(std::memory_order_relaxed);
    }

    void AdamOptimizer::reset_slow_path_grow_count() noexcept {
        g_adam_slow_path_grow_count.store(0, std::memory_order_relaxed);
    }

    void AdamOptimizer::note_slow_path_grow(const char* site, const std::string& name) {
        const uint64_t n = g_adam_slow_path_grow_count.fetch_add(1, std::memory_order_relaxed) + 1;
        LOG_WARN("AdamOptimizer slow-path grow #{} at {} for '{}' — capacity was insufficient; "
                 "will re-reserve after this grow to restore the capacity invariant",
                 n, site, name);
    }

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

    void AdamOptimizer::set_frozen_lr_scale(const float scale) {
        if (!std::isfinite(scale) || scale < 0.0f || scale > 1.0f) {
            throw std::runtime_error("AdamOptimizer frozen LR scale must be within [0, 1]");
        }
        frozen_lr_scale_ = scale;
    }

    void AdamOptimizer::set_crop_damping_mask(lfs::core::Tensor mask) {
        if (mask.is_valid()) {
            LFS_ASSERT_MSG(
                mask.dtype() == lfs::core::DataType::Bool && mask.ndim() == 1,
                "AdamOptimizer crop damping mask must be a 1D bool tensor");
            LFS_ASSERT_MSG(
                mask.numel() == static_cast<size_t>(splat_data_.size()),
                "AdamOptimizer crop damping mask must match the model row count");
            if (mask.device() != lfs::core::Device::CUDA) {
                mask = mask.cuda();
            }
            if (!mask.is_contiguous()) {
                mask = mask.contiguous();
            }
            mask.set_name("adam.crop_damping_mask");
        }
        crop_damping_mask_ = std::move(mask);
    }

    void AdamOptimizer::set_cropbox_lr_scale(const float scale) {
        LFS_ASSERT_MSG(
            std::isfinite(scale) && scale >= 0.0f && scale <= 1.0f,
            "AdamOptimizer crop box LR scale must be finite and within [0, 1]");
        cropbox_lr_scale_ = scale;
    }

    void AdamOptimizer::step(const int iteration) {
        LFS_TRACE("kernel.adam.step");
        if (fused_step_iteration_ == iteration) {
            last_step_zeroed_gradients_ = true;
            return;
        }
        last_step_zeroed_gradients_ = false;

        fast_lfs::optimizer::JointContiguousBatchEntry entries[5];
        int n_entries = 0;
        cudaStream_t batch_stream = nullptr;
        const ParamType contiguous[] = {
            ParamType::Means, ParamType::Sh0, ParamType::Scaling,
            ParamType::Rotation, ParamType::Opacity};

        auto prepare_contiguous = [&](ParamType type) {
            auto& param = get_param(type);
            if (!param.is_valid() || param.numel() == 0) {
                return;
            }
            const auto name = param_name(type);
            if (!states_.contains(name)) {
                init_state(type, false);
            }
            auto& state = states_[name];
            if (!state.grad.is_valid() || state.grad.numel() == 0 ||
                !state.exp_avg.is_valid() || state.exp_avg.numel() == 0 ||
                !state.is_joint() || !state.joint_bounds.is_valid()) {
                return;
            }
            auto& param_live = get_param(type);
            const size_t param_size = param_live.shape()[0];
            if (param_size != state.size) {
                throw std::runtime_error("Optimizer state desync: " + name);
            }
            state.step_count++;
            const double bias_correction1_rcp =
                1.0 / (1.0 - std::pow(config_.beta1, state.step_count));
            const double bias_correction2_sqrt_rcp =
                1.0 / std::sqrt(1.0 - std::pow(config_.beta2, state.step_count));
            const float param_lr = static_cast<float>(get_param_lr(type));
            cudaStream_t execution_stream = state.grad.stream();
            if (execution_stream == nullptr) {
                execution_stream = param_live.stream();
            }
            if (execution_stream == nullptr) {
                execution_stream = state.exp_avg.stream();
            }
            if (batch_stream == nullptr) {
                batch_stream = execution_stream;
            }
            lfs::core::waitForCUDAStream(batch_stream, param_live.stream());
            lfs::core::waitForCUDAStream(batch_stream, state.exp_avg.stream());
            lfs::core::waitForCUDAStream(batch_stream, state.joint_bounds.stream());
            lfs::core::waitForCUDAStream(batch_stream, state.grad.stream());
            const size_t feature_dim = param_live.numel() / param_size;
            auto& e = entries[n_entries++];
            e.param = param_live.ptr<float>();
            e.packed = state.exp_avg.ptr<uint8_t>();
            e.bounds = state.joint_bounds.ptr<float>();
            e.grad = state.grad.ptr<float>();
            e.n_prims = static_cast<int>(state.size);
            e.n_attr = static_cast<int>(feature_dim);
            e.lr = param_lr;
            e.bias_correction1_rcp = static_cast<float>(bias_correction1_rcp);
            e.bias_correction2_sqrt_rcp = static_cast<float>(bias_correction2_sqrt_rcp);
            param_live.set_stream(batch_stream);
            state.exp_avg.set_stream(batch_stream);
            state.joint_bounds.set_stream(batch_stream);
            state.grad.set_stream(batch_stream);
        };

        for (const auto type : contiguous) {
            prepare_contiguous(type);
        }
        if (n_entries > 0) {
            if (frozen_mask_.is_valid()) {
                lfs::core::waitForCUDAStream(batch_stream, frozen_mask_.stream());
            }
            if (crop_damping_mask_.is_valid()) {
                crop_damping_mask_.sync_to_stream(batch_stream);
            }
            fast_lfs::optimizer::adam_step_joint_contiguous_batched(
                entries, n_entries,
                frozen_mask_ptr(), frozen_mask_size(), frozen_lr_scale_,
                crop_damping_mask_ptr(), crop_damping_mask_size(), cropbox_lr_scale_,
                static_cast<float>(config_.beta1),
                static_cast<float>(config_.beta2),
                static_cast<float>(config_.eps),
                batch_stream);
        }
        step_param(ParamType::ShN, iteration);
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
            const auto layout_rest =
                static_cast<uint32_t>(splat_data_.max_sh_coeffs_rest());
            if (type == ParamType::ShN && layout_rest == 0) {
                state = AdamParamState{};
                state.size = 0;
                LOG_INFO("AdamOptimizer: no SH-rest optimizer state for max SH degree 0");
                continue;
            }

            // moments always track the float4-swizzle layout. Under q16, param.shape()[0]
            // is the pad-dropped u16 cell count (45/prim for SH3), not the float cell count
            // (48/prim). Never seed state.size / moment tensors from the u16 shape.
            const size_t logical_size =
                (type == ParamType::ShN)
                    ? lfs::core::sh_swizzled_float_count(
                          static_cast<size_t>(splat_data_.size()), layout_rest)
                    : param_size;
            const size_t effective_capacity =
                (type == ParamType::ShN)
                    ? lfs::core::sh_swizzled_float_count(capacity, layout_rest)
                    : capacity;
            const size_t alloc_cap = std::max(effective_capacity, logical_size);

            // Force the direct/reserved allocator for shN so capacity is recorded even when
            // N already fits in the current swizzled block. Without this, the fast path in
            // extend_state_for_new_params trips on capacity()==0.
            const size_t prim_capacity = (type == ParamType::ShN)
                                             ? std::max(capacity, static_cast<size_t>(splat_data_.size()))
                                             : alloc_cap;
            alloc_quantized_state(type, state, param, alloc_cap, prim_capacity);
            state.exp_avg.set_name("adam." + name + ".exp_avg");
            state.grad = {};
            state.capacity = alloc_cap;
            state.size = logical_size;
            state.step_count = 0;
            LOG_DEBUG("allocate_gradients({}): cap={}", name, state.capacity);
        }
        LOG_DEBUG("Allocated gradients for {} parameter groups", states_.size());
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

    const bool* AdamOptimizer::crop_damping_mask_ptr() const {
        return crop_damping_mask_.is_valid() && crop_damping_mask_.numel() > 0
                   ? crop_damping_mask_.ptr<bool>()
                   : nullptr;
    }

    int AdamOptimizer::crop_damping_mask_size() const {
        return crop_damping_mask_.is_valid()
                   ? static_cast<int>(crop_damping_mask_.numel())
                   : 0;
    }

    void AdamOptimizer::alloc_quantized_state(ParamType type, AdamParamState& state,
                                              const lfs::core::Tensor& param,
                                              size_t moment_capacity, size_t prim_capacity) {
        using namespace joint_adam;
        const auto& shape = param.shape();
        const size_t prim_rows = scale_row_count(type);
        prim_capacity = std::max(prim_capacity, prim_rows);

        // shN moments are always float4-swizzle cells, never pad-dropped q16 cells.
        const auto layout_rest =
            static_cast<uint32_t>(splat_data_.max_sh_coeffs_rest());
        const size_t shN_float_rows =
            (type == ParamType::ShN)
                ? lfs::core::sh_swizzled_float_count(prim_rows, layout_rest)
                : 0;
        const size_t moment_rows =
            (type == ParamType::ShN) ? shN_float_rows : shape[0];
        moment_capacity = std::max(moment_capacity, moment_rows);

        // Joint (u, log_s) is the only Adam codec.
        // Non-SH: 16-bit (4 B/cell), shape [N, n_attr * 4] uint8.
        // SH: 8-bit (2 B/cell), shape [swizzled_floats * 2] uint8 (1D).
        const int bits = (type == ParamType::ShN) ? 8 : 16;
        const int bpc = bytes_per_cell(bits);
        state.joint_bits = bits;

        lfs::core::TensorShape packed_shape;
        size_t packed_cap_rows = 0;
        if (type == ParamType::ShN) {
            // 1D: one packed cell per swizzled float element (float layout, not q16).
            packed_shape = lfs::core::TensorShape(
                {shN_float_rows * static_cast<size_t>(bpc)});
            packed_cap_rows = moment_capacity * static_cast<size_t>(bpc);
        } else {
            // Contiguous [N, C] → [N, C * bpc]
            size_t row_elems = 1;
            for (size_t d = 1; d < shape.rank(); ++d)
                row_elems *= shape[d];
            packed_shape = lfs::core::TensorShape({shape[0], row_elems * static_cast<size_t>(bpc)});
            packed_cap_rows = moment_capacity;
        }

        const bool moment_direct = (packed_cap_rows > packed_shape[0]) || type == ParamType::ShN;
        if (moment_direct && packed_cap_rows > 0) {
            state.exp_avg = lfs::core::Tensor::zeros_direct(
                packed_shape, packed_cap_rows, param.device(), lfs::core::DataType::UInt8);
        } else {
            state.exp_avg = lfs::core::Tensor::zeros(
                packed_shape, param.device(), lfs::core::DataType::UInt8);
        }
        // All-zero packed + bounds → free zero moments (no 128 zero-point).

        ensure_joint_bounds_capacity(state.joint_bounds, prim_rows, prim_capacity,
                                     param.device(), /*zero_all=*/false);
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
        const auto layout_rest =
            static_cast<uint32_t>(splat_data_.max_sh_coeffs_rest());
        // Moments track float layout even when param is q16 (u16 cells).
        const size_t logical_size =
            (type == ParamType::ShN)
                ? lfs::core::sh_swizzled_float_count(
                      static_cast<size_t>(splat_data_.size()), layout_rest)
                : param_size;
        const size_t initial_cap = type == ParamType::ShN
                                       ? std::max(
                                             logical_size,
                                             lfs::core::sh_swizzled_float_count(
                                                 config_.initial_capacity,
                                                 layout_rest))
                                       : compute_new_capacity(0, param_size);

        if (allocate_grad && (!state.grad.is_valid() || state.grad.numel() == 0)) {
            // Grad for shN is float-layout sized (fused path rarely needs it).
            if (type == ParamType::ShN) {
                const auto gshape = lfs::core::TensorShape({logical_size});
                state.grad = (initial_cap > logical_size)
                                 ? lfs::core::Tensor::zeros_direct(gshape, initial_cap)
                                 : lfs::core::Tensor::zeros(gshape, param.device());
            } else {
                state.grad = (initial_cap > param_size)
                                 ? lfs::core::Tensor::zeros_direct(param.shape(), initial_cap)
                                 : lfs::core::Tensor::zeros(param.shape(), param.device());
            }
        }

        const size_t prim_capacity = (type == ParamType::ShN)
                                         ? std::max(static_cast<size_t>(splat_data_.size()), config_.initial_capacity)
                                         : std::max(initial_cap, param_size);
        alloc_quantized_state(type, state, param, initial_cap, prim_capacity);
        state.capacity = std::max(initial_cap, logical_size);
        state.size = logical_size;
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

        if (type == ParamType::ShN) {
            const auto layout_rest =
                static_cast<uint32_t>(splat_data_.max_sh_coeffs_rest());
            const size_t float_layout = lfs::core::sh_swizzled_float_count(
                static_cast<size_t>(splat_data_.size()), layout_rest);
            const size_t alloc_cap =
                state.capacity > float_layout ? state.capacity : float_layout;
            const auto gshape = lfs::core::TensorShape({float_layout});
            state.grad = (alloc_cap > float_layout)
                             ? lfs::core::Tensor::zeros_direct(gshape, alloc_cap)
                             : lfs::core::Tensor::zeros(gshape, param.device());
            state.grad.set_name("adam." + name + ".grad");
            if (state.size == 0) {
                state.size = float_layout;
            }
            state.capacity = std::max(state.capacity, alloc_cap);
            return;
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

        auto& param_live = get_param(type);

        const double bias_correction1_rcp = 1.0 / (1.0 - std::pow(config_.beta1, state.step_count));
        const double bias_correction2_sqrt_rcp = 1.0 / std::sqrt(1.0 - std::pow(config_.beta2, state.step_count));
        const float param_lr = static_cast<float>(get_param_lr(type));

        cudaStream_t execution_stream = state.grad.stream();
        if (execution_stream == nullptr) {
            execution_stream = param_live.stream();
        }
        if (execution_stream == nullptr) {
            execution_stream = state.exp_avg.stream();
        }
        lfs::core::waitForCUDAStream(execution_stream, param_live.stream());
        lfs::core::waitForCUDAStream(execution_stream, state.exp_avg.stream());
        if (state.joint_bounds.is_valid())
            lfs::core::waitForCUDAStream(execution_stream, state.joint_bounds.stream());
        lfs::core::waitForCUDAStream(execution_stream, state.grad.stream());
        if (frozen_mask_.is_valid()) {
            lfs::core::waitForCUDAStream(execution_stream, frozen_mask_.stream());
        }
        if (crop_damping_mask_.is_valid()) {
            crop_damping_mask_.sync_to_stream(execution_stream);
        }

        // Contiguous params use adam_step_joint_contiguous_raw. shN uses the
        // standalone joint kernel that shares apply_shN_grads_packed_joint with FastGS.
        if (state.is_joint()) {
            if (type == ParamType::ShN) {
                if (!state.joint_bounds.is_valid()) {
                    throw std::runtime_error("AdamOptimizer::step_param: joint state missing bounds");
                }
                const auto layout_rest =
                    static_cast<uint32_t>(splat_data_.max_sh_coeffs_rest());
                const auto active_rest =
                    static_cast<uint32_t>(splat_data_.active_sh_coeffs_rest());
                const size_t n_live = static_cast<size_t>(splat_data_.size());
                const size_t float_layout =
                    lfs::core::sh_swizzled_float_count(n_live, layout_rest);
                if (state.size != float_layout) {
                    throw std::runtime_error("Optimizer state desync: shN");
                }
                const int sh_layout_slots = static_cast<int>(
                    lfs::core::sh_float4_slots_for_rest(layout_rest));
                const int active_sh_bases = static_cast<int>(active_rest + 1u);

                float* param_ptr = nullptr;
                float* sh_value_bounds = nullptr;
                int sh_value_bits = 0;
                int sh_value_n_cells = 0;
                if (splat_data_.shN_value_quantized() &&
                    splat_data_.shN_value_bounds().is_valid()) {
                    const auto q16 = lfs::core::resolve_q16_bind_ptrs(splat_data_);
                    param_ptr = const_cast<float*>(q16.codes);
                    sh_value_bounds = const_cast<float*>(q16.bounds);
                    sh_value_bits = 16;
                    sh_value_n_cells = static_cast<int>(q16.n_cells_per_prim);
                } else if (splat_data_.shN_ieee_f16()) {
                    param_ptr = static_cast<float*>(
                        lfs::core::resolve_exportable_device_ptr(param_live));
                    sh_value_bits = 16;
                } else {
                    param_ptr = static_cast<float*>(
                        lfs::core::resolve_exportable_device_ptr(param_live));
                }

                fast_lfs::optimizer::adam_step_shN_joint_from_grad(
                    param_ptr,
                    state.exp_avg.ptr<uint8_t>(),
                    state.joint_bounds.ptr<float>(),
                    sh_value_bounds,
                    state.grad.ptr<float>(),
                    frozen_mask_ptr(),
                    frozen_mask_size(),
                    frozen_lr_scale_,
                    crop_damping_mask_ptr(),
                    crop_damping_mask_size(),
                    cropbox_lr_scale_,
                    static_cast<int>(n_live),
                    sh_layout_slots,
                    active_sh_bases,
                    sh_value_bits,
                    sh_value_n_cells,
                    param_lr * static_cast<float>(bias_correction1_rcp),
                    static_cast<float>(config_.beta1),
                    static_cast<float>(config_.beta2),
                    static_cast<float>(config_.eps),
                    static_cast<float>(bias_correction2_sqrt_rcp),
                    execution_stream);
                param_live.set_stream(execution_stream);
                state.exp_avg.set_stream(execution_stream);
                state.joint_bounds.set_stream(execution_stream);
                state.grad.set_stream(execution_stream);
                return;
            }
            if (!state.joint_bounds.is_valid()) {
                throw std::runtime_error("AdamOptimizer::step_param: joint state missing bounds");
            }
            const size_t param_size = param_live.shape()[0];
            if (param_size != state.size) {
                throw std::runtime_error("Optimizer state desync: " + name);
            }
            const size_t feature_dim = param_live.numel() / param_size;
            fast_lfs::optimizer::adam_step_joint_contiguous_raw(
                param_live.ptr<float>(),
                state.exp_avg.ptr<uint8_t>(),
                state.joint_bounds.ptr<float>(),
                state.grad.ptr<float>(),
                frozen_mask_ptr(),
                frozen_mask_size(),
                frozen_lr_scale_,
                crop_damping_mask_ptr(),
                crop_damping_mask_size(),
                cropbox_lr_scale_,
                static_cast<int>(state.size),
                static_cast<int>(feature_dim),
                state.joint_bits,
                param_lr,
                static_cast<float>(config_.beta1),
                static_cast<float>(config_.beta2),
                static_cast<float>(config_.eps),
                static_cast<float>(bias_correction1_rcp),
                static_cast<float>(bias_correction2_sqrt_rcp),
                execution_stream);
            param_live.set_stream(execution_stream);
            state.exp_avg.set_stream(execution_stream);
            state.joint_bounds.set_stream(execution_stream);
            state.grad.set_stream(execution_stream);
            return;
        }

        throw std::runtime_error(
            "AdamOptimizer::step_param: non-joint state is unsupported "
            "(joint (u,log_s) is the only Adam codec)");
    }

    FastGSFusedAdamState AdamOptimizer::prepare_fastgs_fused_adam(
        const int iteration,
        const cudaStream_t execution_stream) {
        if (crop_damping_mask_.is_valid()) {
            crop_damping_mask_.sync_to_stream(execution_stream);
        }

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
            const bool joint_ok = state.is_joint() && state.exp_avg.is_valid() &&
                                  state.joint_bounds.is_valid();
            if (!joint_ok) {
                init_state(type, false);
            }

            // Moments track float4-swizzled cell count even when SH value quant stores
            // pad-dropped u16 params (different numel). After densify grow/relocate/compact
            // or re-encode, heal undersized SH moment bookkeeping + joint_bounds.
            //
            //   HEAL  — moment capacity already covers float_layout(N): advance state.size;
            //           grow joint_bounds logical rows (capacity-first, else realloc bounds).
            //   REBUILD — capacity short: extend_state_for_new_params (growth_factor re-reserve).
            // Moments stay on the float4-swizzle layout; never resize to u16 cell count.
            if (type != ParamType::ShN) {
                if (param.shape()[0] != state.size) {
                    throw std::runtime_error("Optimizer state desync before fused Adam: " + name);
                }
            } else {
                const auto layout_rest =
                    static_cast<uint32_t>(splat_data_.max_sh_coeffs_rest());
                const size_t n_live = static_cast<size_t>(splat_data_.size());
                const size_t float_layout =
                    lfs::core::sh_swizzled_float_count(n_live, layout_rest);
                // q16 param numel ≠ float_layout; only compare state.size to float_layout.
                // state.capacity is float cells; joint exp_avg.capacity is packed bytes
                // (float_cells * bpc) — never compare those units directly.
                const size_t moment_cap_floats =
                    state.capacity > 0
                        ? state.capacity
                        : (state.exp_avg.is_valid()
                               ? (state.is_joint()
                                      ? state.exp_avg.capacity() /
                                            static_cast<size_t>(
                                                joint_adam::bytes_per_cell(state.joint_bits))
                                      : state.exp_avg.capacity())
                               : 0);
                if (state.size != float_layout) {
                    if (moment_cap_floats >= float_layout && state.size < float_layout) {
                        // Capacity reserved at max_cap: advance logical size (new slots
                        // already zero-initialized under joint bounds when pre-allocated).
                        state.size = float_layout;
                    } else if (state.size < float_layout) {
                        // Pad-aware: derive n_new from prim count, not float/N (pad jumps).
                        const size_t floats_per_approx =
                            n_live > 0 ? (float_layout + n_live - 1) / n_live : 0;
                        const size_t n_prev =
                            floats_per_approx > 0 ? state.size / floats_per_approx : 0;
                        // Prefer exact pad inverse via binary search free: use means size
                        // delta if means grew; else extend by remaining prims.
                        size_t n_new = n_live > n_prev ? n_live - n_prev : 0;
                        if (n_new == 0 && float_layout > state.size) {
                            // Fallback: at least one prim of growth.
                            n_new = 1;
                        }
                        if (n_new > 0) {
                            extend_state_for_new_params(type, n_new);
                        }
                        if (state.size != float_layout) {
                            // Last resort: force size when capacity allows after extend.
                            const size_t cap2 =
                                state.capacity > 0
                                    ? state.capacity
                                    : (state.exp_avg.is_valid()
                                           ? (state.is_joint()
                                                  ? state.exp_avg.capacity() /
                                                        static_cast<size_t>(
                                                            joint_adam::bytes_per_cell(
                                                                state.joint_bits))
                                                  : state.exp_avg.capacity())
                                           : 0);
                            if (cap2 >= float_layout && state.size < float_layout) {
                                state.size = float_layout;
                            }
                        }
                        if (state.size != float_layout) {
                            throw std::runtime_error(
                                "Optimizer state desync before fused Adam: " + name +
                                " param_n=" + std::to_string(param.numel()) +
                                " state=" + std::to_string(state.size) +
                                " float_layout=" + std::to_string(float_layout));
                        }
                    } else if (state.size > float_layout) {
                        // Compact path should rebuild; tolerate oversize (use live N).
                        state.size = float_layout;
                    }
                }
                // joint_bounds must cover ceil(N/256) blocks; capacity tracks max_cap.
                if (state.is_joint() && n_live > 0) {
                    const size_t nb = joint_adam::n_bounds_for_prims(n_live);
                    const size_t prim_cap = std::max(
                        n_live,
                        config_.initial_capacity > 0 ? config_.initial_capacity : n_live);
                    const size_t nb_cap = std::max(nb, joint_adam::n_bounds_for_prims(prim_cap));
                    const size_t cur_nb =
                        state.joint_bounds.is_valid() && state.joint_bounds.ndim() >= 1
                            ? state.joint_bounds.shape()[0]
                            : 0;
                    const size_t cur_cap =
                        state.joint_bounds.is_valid() ? state.joint_bounds.capacity() : 0;
                    if (!state.joint_bounds.is_valid() || cur_nb < nb || cur_cap < nb_cap) {
                        if (state.joint_bounds.is_valid() && cur_cap >= nb_cap && cur_nb < nb) {
                            state.joint_bounds.append_zeros(nb - cur_nb);
                        } else {
                            auto new_bounds = lfs::core::Tensor::zeros_direct(
                                lfs::core::TensorShape({nb, size_t{4}}), nb_cap, param.device());
                            if (state.joint_bounds.is_valid() && state.joint_bounds.numel() > 0) {
                                const cudaStream_t bstream = lfs::core::getCurrentCUDAStream();
                                lfs::core::waitForCUDAStream(bstream, state.joint_bounds.stream());
                                lfs::core::waitForCUDAStream(bstream, new_bounds.stream());
                                const size_t copy_n = std::min(state.joint_bounds.numel(),
                                                               new_bounds.numel());
                                auto old_jb = std::move(state.joint_bounds);
                                LFS_CUDA_CHECK(cudaMemcpyAsync(
                                    new_bounds.ptr<float>(), old_jb.ptr<float>(),
                                    copy_n * sizeof(float), cudaMemcpyDeviceToDevice, bstream));
                                LFS_CUDA_CHECK(cudaStreamSynchronize(bstream));
                                state.joint_bounds = std::move(new_bounds);
                            } else {
                                state.joint_bounds = std::move(new_bounds);
                            }
                        }
                    }
                }
            }

            const auto next_step = state.step_count + 1;
            const double bias_correction1_rcp = 1.0 / (1.0 - std::pow(config_.beta1, next_step));
            const double bias_correction2_sqrt_rcp = 1.0 / std::sqrt(1.0 - std::pow(config_.beta2, next_step));

            // SH value quant stores u16 codes as Float16 bit-patterns; use data_ptr.
            // Generation-checked when exportable-backed so a pre-grow raw pointer
            // cannot survive into the fused step.
            out.param = static_cast<float*>(
                lfs::core::resolve_exportable_device_ptr(param));
            out.n_primitives = static_cast<int>(splat_data_.size());
            out.joint_packed = state.exp_avg.ptr<uint8_t>();
            out.joint_bounds = state.joint_bounds.ptr<float>();
            out.joint_bits = state.joint_bits;
            out.frozen_mask = frozen_mask_ptr();
            out.frozen_mask_size = frozen_mask_size();
            out.frozen_lr_scale = frozen_lr_scale_;
            out.crop_damping_mask = crop_damping_mask_ptr();
            out.crop_damping_mask_size = crop_damping_mask_size();
            out.cropbox_lr_scale = cropbox_lr_scale_;
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
        // Generation-checked q16 fetch — never bake a pre-grow exportable pointer.
        if (fused.shN.enabled && splat_data_.shN_value_quantized() &&
            splat_data_.shN_value_bounds().is_valid()) {
            const auto q16 = lfs::core::resolve_q16_bind_ptrs(splat_data_);
            fused.shN.param = const_cast<float*>(q16.codes);
            fused.shN.sh_value_bounds = const_cast<float*>(q16.bounds);
            fused.shN.sh_value_bits = 16;
            fused.shN.sh_value_n_cells = static_cast<int>(q16.n_cells_per_prim);
        } else if (fused.shN.enabled && splat_data_.shN_ieee_f16()) {
            // IEEE f16 float4-swizzle (exportable GUI): half load/store, no bounds.
            fused.shN.param = static_cast<float*>(
                lfs::core::resolve_exportable_device_ptr(splat_data_.shN()));
            fused.shN.sh_value_bits = 16;
            fused.shN.sh_value_bounds = nullptr;
            fused.shN.sh_value_n_cells = 0;
        }
        fused.scaling = prepare_param(ParamType::Scaling, 3, true);
        fused.rotation = prepare_param(ParamType::Rotation, 4, true);
        fused.opacity = prepare_param(ParamType::Opacity, 1, true);

        fused.enabled = fused.means.enabled || fused.sh0.enabled || fused.shN.enabled ||
                        fused.scaling.enabled || fused.rotation.enabled || fused.opacity.enabled;
        return fused;
    }

    void AdamOptimizer::commit_fastgs_fused_adam(const int iteration) {
        // Only bump step_count for parameters that prepare_fastgs_fused_adam
        // actually enabled this iteration. During SH warmup prepare disables shN
        // but the old commit still advanced its counter → first real shN update
        // ran with t≈1001 bias correction instead of t=1.
        const auto active_rest = splat_data_.active_sh_coeffs_rest();
        for (const auto type : all_param_types()) {
            auto& param = get_param(type);
            if (!param.is_valid() || param.numel() == 0)
                continue;
            if (type == ParamType::ShN &&
                (iteration <= SH_WARMUP_ITERATIONS || active_rest == 0)) {
                continue;
            }
            const auto name = param_name(type);
            if (!states_.contains(name)) {
                continue;
            }
            auto& state = states_[name];
            if (state.exp_avg.is_valid() && state.is_joint()) {
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

        if (!state.is_joint() || !state.exp_avg.is_valid() || !state.joint_bounds.is_valid()) {
            LOG_WARN("reset_state_at_indices: {} joint packed/bounds invalid", name);
            return;
        }
        // Encode true (m,v)=(0,0) under current block bounds (u=0,log_s=0).
        const cudaStream_t stream = lfs::core::getCurrentCUDAStream();
        const size_t idx_bytes = indices.size() * sizeof(int64_t);
        int64_t* d_indices = nullptr;
        LFS_CUDA_CHECK(cudaMallocAsync(&d_indices, idx_bytes, stream));
        LFS_CUDA_CHECK(cudaMemcpyAsync(d_indices, indices.data(), idx_bytes, cudaMemcpyHostToDevice, stream));
        lfs::core::waitForCUDAStream(stream, state.exp_avg.stream());
        lfs::core::waitForCUDAStream(stream, state.joint_bounds.stream());
        if (type == ParamType::ShN) {
            const int slots = static_cast<int>(lfs::core::sh_float4_slots_for_rest(
                static_cast<uint32_t>(splat_data_.max_sh_coeffs_rest())));
            if (slots > 0) {
                fast_lfs::optimizer::joint_encode_zero_shN_at_indices(
                    state.exp_avg.ptr<uint8_t>(),
                    state.joint_bounds.ptr<float>(),
                    d_indices,
                    static_cast<int>(indices.size()),
                    slots,
                    state.joint_bits,
                    static_cast<int>(splat_data_.size()),
                    stream);
            }
        } else {
            const int bpc = joint_adam::bytes_per_cell(state.joint_bits);
            const int row_bytes = static_cast<int>(tensor_row_size(state.exp_avg));
            const int n_attr = bpc > 0 ? row_bytes / bpc : 0;
            if (n_attr > 0) {
                fast_lfs::optimizer::joint_encode_zero_rows_at_indices(
                    state.exp_avg.ptr<uint8_t>(),
                    state.joint_bounds.ptr<float>(),
                    d_indices,
                    static_cast<int>(indices.size()),
                    n_attr,
                    state.joint_bits,
                    static_cast<int>(splat_data_.size()),
                    stream);
            }
        }
        state.exp_avg.set_stream(stream);
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
             !state.exp_avg.is_valid())) {
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

        if (state.is_joint()) {
            // Joint packed [N, C*bpc]: gather rows; grow bounds table for new N.
            const bool grad_has_capacity = !state.grad.is_valid() || state.grad.capacity() > 0;
            const bool fits = grad_has_capacity && state.exp_avg.capacity() > 0 &&
                              new_size <= state.exp_avg.capacity() &&
                              (!state.grad.is_valid() || new_size <= state.grad.capacity());
            if (fits) {
                state.exp_avg.append_gather(indices);
                if (state.grad.is_valid())
                    state.grad.append_zeros(n_new);
            } else {
                note_slow_path_grow("extend_state_by_gather(joint)", name);
                state.exp_avg = lfs::core::Tensor::cat(
                    {state.exp_avg, state.exp_avg.index_select(0, indices)}, 0);
                if (state.grad.is_valid()) {
                    const auto& shape = param.shape();
                    std::vector<size_t> new_dims(shape.dims());
                    new_dims[0] = new_size;
                    state.grad = lfs::core::Tensor::zeros(lfs::core::TensorShape(new_dims), param.device());
                }
                const size_t target_cap = compute_new_capacity(new_size, new_size);
                state.exp_avg.reserve(target_cap);
                if (state.grad.is_valid())
                    state.grad.reserve(target_cap);
            }
            // Bounds: grow-only to cover ceil(new_N/256); zero-init new blocks only.
            const size_t old_N = new_size - n_new;
            const size_t prim_cap =
                state.capacity > 0 ? state.capacity
                                   : compute_new_capacity(new_size, new_size);
            ensure_joint_bounds_capacity(state.joint_bounds, new_size, prim_cap,
                                         param.device(), /*zero_all=*/false);
            // raw gather copies codes across blocks with different bounds → garbage.
            // Transcode decode(src bounds) → encode(dst bounds) for the new rows.
            if (n_new > 0 && state.joint_bounds.is_valid()) {
                const int bpc = joint_adam::bytes_per_cell(state.joint_bits);
                const int row_bytes = static_cast<int>(tensor_row_size(state.exp_avg));
                const int n_attr = bpc > 0 ? row_bytes / bpc : 0;
                if (n_attr > 0) {
                    const cudaStream_t stream = lfs::core::getCurrentCUDAStream();
                    lfs::core::waitForCUDAStream(stream, state.exp_avg.stream());
                    lfs::core::waitForCUDAStream(stream, state.joint_bounds.stream());
                    lfs::core::waitForCUDAStream(stream, indices.stream());
                    const bool idx_i64 = indices.dtype() == lfs::core::DataType::Int64;
                    lfs::core::Tensor idx64;
                    const int64_t* idx_ptr = nullptr;
                    if (idx_i64) {
                        idx_ptr = indices.ptr<int64_t>();
                    } else {
                        idx64 = indices.to(lfs::core::DataType::Int64);
                        idx_ptr = idx64.ptr<int64_t>();
                    }
                    fast_lfs::optimizer::joint_transcode_gathered_rows_at_indices(
                        state.exp_avg.ptr<uint8_t>(),
                        state.joint_bounds.ptr<float>(),
                        idx_ptr,
                        static_cast<int>(n_new),
                        static_cast<int>(old_N),
                        n_attr,
                        state.joint_bits,
                        stream);
                    state.exp_avg.set_stream(stream);
                }
            }
            state.size = new_size;
            state.capacity = state.exp_avg.capacity();
            return;
        }

        throw std::runtime_error(
            "extend_state_by_gather: non-joint Adam state is unsupported "
            "(joint (u,log_s) is the only codec)");
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
            (splat_data_.max_sh_coeffs_rest() == 0 || !state.exp_avg.is_valid())) {
            return;
        }

        // For swizzled shN, moment growth is measured in floats: (swizzled_floats(N+n_new) -
        // swizzled_floats(N)). For everything else it is n_new rows. First-moment bytes must
        // use the signed zero-point (128), not byte zero: inactive shN slots can share a
        // nonzero per-primitive scale once lower SH bands start training.
        const size_t growth = compute_state_growth(type, n_new);
        // Joint SH packed is 1D with bpc bytes per float — growth is in float cells * bpc
        // for append along the 1D dim when shape is [floats*bpc].
        size_t packed_growth = growth;
        if (state.is_joint() && type == ParamType::ShN) {
            packed_growth = growth * static_cast<size_t>(joint_adam::bytes_per_cell(state.joint_bits));
        }
        const size_t new_size = state.size + growth;

        if (!param.is_valid() || param.shape().rank() == 0) {
            throw std::runtime_error("extend_state: " + name + " invalid");
        }
        if (!state.exp_avg.is_valid() || state.exp_avg.ndim() == 0) {
            throw std::runtime_error("extend_state: " + name + " state invalid");
        }

        if (state.is_joint()) {
            const size_t packed_new_rows = type == ParamType::ShN
                                               ? state.exp_avg.shape()[0] + packed_growth
                                               : new_size;
            const bool fits = state.exp_avg.capacity() > 0 &&
                              packed_new_rows <= state.exp_avg.capacity() &&
                              (!state.grad.is_valid() || new_size <= state.grad.capacity());
            if (fits) {
                if (state.grad.is_valid())
                    state.grad.append_zeros(growth);
                if (type == ParamType::ShN) {
                    state.exp_avg.append_zeros(packed_growth);
                } else {
                    state.exp_avg.append_zeros(growth);
                }
            } else {
                note_slow_path_grow("extend_state_for_new_params(joint)", name);
                // Re-alloc with growth_factor headroom and restore capacity invariant.
                const size_t prim_cap = compute_new_capacity(
                    static_cast<size_t>(splat_data_.size()),
                    static_cast<size_t>(splat_data_.size()));
                const size_t moment_cap = type == ParamType::ShN
                                              ? lfs::core::sh_swizzled_float_count(
                                                    prim_cap,
                                                    static_cast<uint32_t>(splat_data_.max_sh_coeffs_rest()))
                                              : prim_cap;
                alloc_quantized_state(type, state, param, moment_cap, prim_cap);
                // Ensure tensor capacity is recorded (zeros_direct path).
                if (state.exp_avg.is_valid() && state.exp_avg.capacity() < moment_cap) {
                    state.exp_avg.reserve(moment_cap);
                }
                if (state.grad.is_valid())
                    state.grad.reserve(moment_cap);
            }
            const size_t prim_n = static_cast<size_t>(splat_data_.size());
            const size_t prim_cap =
                std::max(prim_n, state.capacity > 0 ? state.capacity : new_size);
            ensure_joint_bounds_capacity(state.joint_bounds, prim_n, prim_cap,
                                         param.device(), /*zero_all=*/false);
            state.size = new_size;
            state.capacity = state.exp_avg.is_valid() ? state.exp_avg.capacity() : new_size;
            LFS_DEBUG_ASSERT_MSG(state.capacity >= state.size,
                                 "extend_state_for_new_params(joint): capacity < size");

            // raw zero codes under a live mid-block's non-zero bounds do NOT
            // decode to (m,v)=(0,0). Zero-encode the newly appended prim rows.
            if (n_new > 0 && prim_n >= n_new && state.exp_avg.is_valid() &&
                state.joint_bounds.is_valid()) {
                const size_t old_prims = prim_n - n_new;
                std::vector<int64_t> new_idx(n_new);
                for (size_t i = 0; i < n_new; ++i)
                    new_idx[i] = static_cast<int64_t>(old_prims + i);
                const cudaStream_t stream = lfs::core::getCurrentCUDAStream();
                int64_t* d_idx = nullptr;
                const size_t idx_bytes = n_new * sizeof(int64_t);
                LFS_CUDA_CHECK(cudaMallocAsync(&d_idx, idx_bytes, stream));
                LFS_CUDA_CHECK(cudaMemcpyAsync(d_idx, new_idx.data(), idx_bytes,
                                               cudaMemcpyHostToDevice, stream));
                lfs::core::waitForCUDAStream(stream, state.exp_avg.stream());
                lfs::core::waitForCUDAStream(stream, state.joint_bounds.stream());
                if (type == ParamType::ShN) {
                    const int slots = static_cast<int>(lfs::core::sh_float4_slots_for_rest(
                        static_cast<uint32_t>(splat_data_.max_sh_coeffs_rest())));
                    if (slots > 0) {
                        fast_lfs::optimizer::joint_encode_zero_shN_at_indices(
                            state.exp_avg.ptr<uint8_t>(),
                            state.joint_bounds.ptr<float>(),
                            d_idx,
                            static_cast<int>(n_new),
                            slots,
                            state.joint_bits,
                            static_cast<int>(prim_n),
                            stream);
                    }
                } else {
                    const int bpc = joint_adam::bytes_per_cell(state.joint_bits);
                    const int row_bytes = static_cast<int>(tensor_row_size(state.exp_avg));
                    const int n_attr = bpc > 0 ? row_bytes / bpc : 0;
                    if (n_attr > 0) {
                        fast_lfs::optimizer::joint_encode_zero_rows_at_indices(
                            state.exp_avg.ptr<uint8_t>(),
                            state.joint_bounds.ptr<float>(),
                            d_idx,
                            static_cast<int>(n_new),
                            n_attr,
                            state.joint_bits,
                            static_cast<int>(prim_n),
                            stream);
                    }
                }
                state.exp_avg.set_stream(stream);
                state.joint_bounds.set_stream(stream);
                LFS_CUDA_CHECK(cudaFreeAsync(d_idx, stream));
            }
            return;
        }

        throw std::runtime_error(
            "extend_state_for_new_params: non-joint Adam state is unsupported "
            "(joint (u,log_s) is the only codec)");
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

    bool AdamOptimizer::preflight_grow_capacity(const size_t n_new) {
        if (n_new == 0) {
            return true;
        }
        auto& means = get_param(ParamType::Means);
        if (!means.is_valid()) {
            return false;
        }
        const size_t old_size = means.shape()[0];
        const size_t new_size = old_size + n_new;
        if (means.capacity() >= new_size) {
            return true;
        }
        if (!means.is_external_storage()) {
            // Non-exportable paths fall back to Tensor::cat in add_new_params;
            // no capacity-ensure gate to fail.
            return true;
        }
        // Grow/rebind the shared exportable block once before any densify
        // mutation. Failure must leave row counts untouched.
        bool layout_changed = false;
        const bool ok = splat_data_.ensure_param_capacity(new_size, &layout_changed);
#ifndef NDEBUG
        if (ok && layout_changed) {
            auto& means_after = get_param(ParamType::Means);
            LFS_ASSERT_MSG(means_after.is_valid() && means_after.capacity() >= new_size,
                           "preflight_grow_capacity: layout changed but means not ready");
        }
#else
        (void)layout_changed;
#endif
        return ok;
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
        bool layout_changed = false;
        if (param.capacity() < new_size && param.is_external_storage()) {
            // grow exportable block (and rebind) instead of cat, which
            // would orphan the Vulkan zero-copy mapping. Prefer preflight_grow_capacity
            // at densify entry so this path is a last-resort safety net that still
            // throws BEFORE mutating this param (no partial append on failure).
            if (!splat_data_.ensure_param_capacity(new_size, &layout_changed)) {
                throw std::runtime_error(std::format(
                    "add_new_params: external storage capacity {} < needed {} and "
                    "capacity-ensure failed (exportable grow/rebind)",
                    param.capacity(),
                    new_size));
            }
        }
        // Re-fetch after a possible rebind because the in-flight Tensor& `param` is
        // dangling if layout_changed; never use it past this point.
        auto& param_ref = get_param(type);
#ifndef NDEBUG
        if (layout_changed) {
            // Generation advanced: any pre-ensure Tensor& / data_ptr is stale.
            LFS_ASSERT_MSG(param_ref.is_valid() && param_ref.capacity() >= new_size,
                           "add_new_params: layout changed but re-fetched param is not ready");
        }
#else
        (void)layout_changed;
#endif
        if (param_ref.capacity() >= new_size) {
            param_ref.append_zeros(n_new);
            auto appended = param_ref.slice(0, old_size, new_size);
            appended.copy_from(new_values);
        } else {
            param_ref = lfs::core::Tensor::cat({param_ref, new_values}, 0);
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
            // [old_N, new_N). cuda.direct / q16-expand storage cannot reserve in place —
            // reallocate with headroom when capacity is short (cross-block densify).
            if (param.capacity() < new_floats) {
                const size_t target = std::max(
                    compute_new_capacity(new_floats, new_floats), new_floats);
                auto fresh = lfs::core::Tensor::zeros_direct(
                    lfs::core::TensorShape({old_floats}), target, param.device(),
                    param.dtype());
                if (old_floats > 0 && param.is_valid() && param.numel() > 0) {
                    const cudaStream_t stream = lfs::core::getCurrentCUDAStream();
                    lfs::core::waitForCUDAStream(stream, param.stream());
                    const size_t nbytes = old_floats * lfs::core::dtype_size(param.dtype());
                    LFS_CUDA_CHECK(cudaMemcpyAsync(
                        fresh.data_ptr(), param.data_ptr(), nbytes,
                        cudaMemcpyDeviceToDevice, stream));
                    LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
                }
                param = std::move(fresh);
            }
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

            // Param gather is float/q16 specific. Under q16 densify paths dequant first
            // (ensure_shN_fp32_for_mutation); still only gather when float storage.
            if (param.dtype() == lfs::core::DataType::Float32) {
                gather_new_swizzled_rows(param);
            }

            // Extend the Adam state.
            const auto name = param_name(type);
            if (states_.contains(name)) {
                auto& state = states_[name];

                // Joint states grow packed exp_avg directly.
                if (state.is_joint() && state.exp_avg.is_valid()) {
                    const int bpc = joint_adam::bytes_per_cell(state.joint_bits);
                    const size_t packed_growth = growth * static_cast<size_t>(bpc);
                    const size_t packed_new = state.exp_avg.shape()[0] + packed_growth;
                    const bool fits = state.exp_avg.capacity() > 0 &&
                                      packed_new <= state.exp_avg.capacity();
                    if (fits) {
                        state.exp_avg.append_zeros(packed_growth);
                    } else {
                        note_slow_path_grow("add_new_params_gather(shN,joint)", name);
                        const size_t prim_cap = compute_new_capacity(new_N, new_N);
                        const size_t moment_cap =
                            lfs::core::sh_swizzled_float_count(prim_cap, layout_rest);
                        // Preserve existing packed codes then grow.
                        auto old_packed = std::move(state.exp_avg);
                        alloc_quantized_state(type, state, param, moment_cap, prim_cap);
                        if (old_packed.is_valid() && old_packed.numel() > 0 &&
                            state.exp_avg.is_valid()) {
                            const cudaStream_t stream = lfs::core::getCurrentCUDAStream();
                            lfs::core::waitForCUDAStream(stream, old_packed.stream());
                            const size_t copy_n =
                                std::min(old_packed.numel(), state.exp_avg.numel());
                            LFS_CUDA_CHECK(cudaMemcpyAsync(
                                state.exp_avg.ptr<uint8_t>(), old_packed.ptr<uint8_t>(),
                                copy_n * sizeof(uint8_t), cudaMemcpyDeviceToDevice, stream));
                            LFS_CUDA_CHECK(cudaStreamSynchronize(stream));
                        }
                    }
                    ensure_joint_bounds_capacity(state.joint_bounds, new_N,
                                                 std::max(new_N, state.capacity),
                                                 param.device(), /*zero_all=*/false);
                    state.size = new_floats;
                    state.capacity = state.exp_avg.is_valid() ? state.exp_avg.capacity() : new_floats;
                    // Zero-encode new prim rows under (possibly non-zero) block bounds.
                    if (n_new > 0 && state.joint_bounds.is_valid()) {
                        std::vector<int64_t> new_idx(n_new);
                        for (size_t i = 0; i < n_new; ++i)
                            new_idx[i] = static_cast<int64_t>(old_N + i);
                        const cudaStream_t stream = lfs::core::getCurrentCUDAStream();
                        int64_t* d_idx = nullptr;
                        const size_t idx_bytes = n_new * sizeof(int64_t);
                        LFS_CUDA_CHECK(cudaMallocAsync(&d_idx, idx_bytes, stream));
                        LFS_CUDA_CHECK(cudaMemcpyAsync(d_idx, new_idx.data(), idx_bytes,
                                                       cudaMemcpyHostToDevice, stream));
                        const int slots = static_cast<int>(
                            lfs::core::sh_float4_slots_for_rest(layout_rest));
                        if (slots > 0) {
                            lfs::core::waitForCUDAStream(stream, state.exp_avg.stream());
                            lfs::core::waitForCUDAStream(stream, state.joint_bounds.stream());
                            fast_lfs::optimizer::joint_encode_zero_shN_at_indices(
                                state.exp_avg.ptr<uint8_t>(),
                                state.joint_bounds.ptr<float>(),
                                d_idx,
                                static_cast<int>(n_new),
                                slots,
                                state.joint_bits,
                                static_cast<int>(new_N),
                                stream);
                            state.exp_avg.set_stream(stream);
                            state.joint_bounds.set_stream(stream);
                        }
                        LFS_CUDA_CHECK(cudaFreeAsync(d_idx, stream));
                    }
                    LFS_DEBUG_ASSERT_MSG(state.capacity >= state.size,
                                         "add_new_params_gather(shN,joint): capacity < size");
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

        if (state.is_joint()) {
            if (!state.exp_avg.is_valid()) {
                LOG_WARN("relocate_params_at_indices_gpu: {} joint packed invalid", name);
                return;
            }
            if (!state.joint_bounds.is_valid()) {
                LOG_WARN("relocate_params_at_indices_gpu: {} joint bounds invalid", name);
                return;
            }
            const cudaStream_t stream = lfs::core::getCurrentCUDAStream();
            lfs::core::waitForCUDAStream(stream, state.exp_avg.stream());
            lfs::core::waitForCUDAStream(stream, state.joint_bounds.stream());
            if (type == ParamType::ShN) {
                const int slots = static_cast<int>(lfs::core::sh_float4_slots_for_rest(
                    static_cast<uint32_t>(splat_data_.max_sh_coeffs_rest())));
                if (slots > 0) {
                    fast_lfs::optimizer::joint_encode_zero_shN_at_indices(
                        state.exp_avg.ptr<uint8_t>(),
                        state.joint_bounds.ptr<float>(),
                        indices_device,
                        static_cast<int>(n_indices),
                        slots,
                        state.joint_bits,
                        static_cast<int>(splat_data_.size()),
                        stream);
                }
            } else {
                const int bpc = joint_adam::bytes_per_cell(state.joint_bits);
                const int row_bytes = static_cast<int>(tensor_row_size(state.exp_avg));
                const int n_attr = bpc > 0 ? row_bytes / bpc : 0;
                if (n_attr > 0) {
                    fast_lfs::optimizer::joint_encode_zero_rows_at_indices(
                        state.exp_avg.ptr<uint8_t>(),
                        state.joint_bounds.ptr<float>(),
                        indices_device,
                        static_cast<int>(n_indices),
                        n_attr,
                        state.joint_bits,
                        static_cast<int>(splat_data_.size()),
                        stream);
                }
            }
            state.exp_avg.set_stream(stream);
            return;
        }
    }

    namespace {
        constexpr uint32_t ADAM_STATE_MAGIC = 0x4C464144; // "LFAD"
        // v1: fp32 moments (no scales).
        // v2: uint8 quantised moments + per-primitive fp32 scales (legacy codec only).
        // v3: per-state joint_bits marker; joint packs (exp_avg + joint_bounds), legacy keeps v2 tensors.
        constexpr uint32_t ADAM_STATE_VERSION = 3;
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
            return s.is_joint() && s.exp_avg.is_valid() && s.joint_bounds.is_valid();
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
            // v3: joint_bits discriminates packed joint moments from legacy u8+scales.
            const int32_t joint_bits = state.joint_bits;
            os.write(reinterpret_cast<const char*>(&joint_bits), sizeof(joint_bits));
            os << state.exp_avg << state.joint_bounds;
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

            // v3 introduces per-state joint_bits; v1/v2 are always legacy (joint_bits=0).
            int32_t joint_bits = 0;
            if (version >= 3) {
                lfs::core::serialization_detail::read_exact(
                    is, &joint_bits, sizeof(joint_bits), "Adam joint_bits");
                if (joint_bits != 0 && joint_bits != 8 && joint_bits != 16)
                    throw std::runtime_error("Invalid AdamOptimizer checkpoint: unsupported joint_bits");
            }
            state.joint_bits = joint_bits;

            const bool is_shN = (name == "shN");
            const ParamType ptype = *maybe_type;
            const auto& parameter = get_param(ptype);
            // shN state.size is always float4-swizzle cells, never q16 u16 cells.
            // After a fused step the heal sets state.size = float_layout; strategy re-quant
            // before deserialize leaves parameter.shape()[0] as u16 count — do not compare
            // those units.
            const auto layout_rest =
                static_cast<uint32_t>(splat_data_.max_sh_coeffs_rest());
            size_t expected_state_size = parameter.shape()[0];
            if (is_shN) {
                if (version == 1) {
                    expected_state_size = static_cast<size_t>(splat_data_.size());
                } else {
                    expected_state_size = lfs::core::sh_swizzled_float_count(
                        static_cast<size_t>(splat_data_.size()), layout_rest);
                }
            }
            if (!parameter.is_valid() || state.size != expected_state_size)
                throw std::runtime_error("Invalid AdamOptimizer checkpoint: state size does not match model");

            if (version == 1 || !state.is_joint()) {
                throw std::runtime_error(
                    "AdamOptimizer checkpoint uses removed legacy moment codec; "
                    "retrain or convert to joint (u,log_s) checkpoints");
            }
            if (state.is_joint()) {
                // v3 joint: packed (u,log_s) bytes + float4 bounds per 256-splat block.
                is >> state.exp_avg >> state.joint_bounds;
                const size_t primitive_rows = static_cast<size_t>(splat_data_.size());
                const int bpc = joint_adam::bytes_per_cell(state.joint_bits);
                if (bpc <= 0 ||
                    state.exp_avg.dtype() != lfs::core::DataType::UInt8 ||
                    !state.joint_bounds.is_valid() ||
                    state.joint_bounds.dtype() != lfs::core::DataType::Float32 ||
                    state.joint_bounds.ndim() != 2 || state.joint_bounds.shape()[1] != 4) {
                    throw std::runtime_error("Invalid AdamOptimizer checkpoint: joint moment schema mismatch");
                }
                // Accept grow-only bounds tables (shape >= live blocks). Compaction
                // can leave extra zero blocks; never require strict equality.
                const size_t expected_bounds = joint_adam::n_bounds_for_prims(primitive_rows);
                if (!state.joint_bounds.is_valid() ||
                    state.joint_bounds.shape()[0] < expected_bounds) {
                    throw std::runtime_error("Invalid AdamOptimizer checkpoint: joint bounds size mismatch");
                }
                if (is_shN) {
                    // Packed 1D: one cell per float4-swizzle float, bpc bytes each.
                    const size_t expected_bytes =
                        lfs::core::sh_swizzled_float_count(primitive_rows, layout_rest) *
                        static_cast<size_t>(bpc);
                    if (state.exp_avg.ndim() != 1 || state.exp_avg.numel() != expected_bytes ||
                        state.joint_bits != 8) {
                        throw std::runtime_error("Invalid AdamOptimizer checkpoint: joint shN packed shape mismatch");
                    }
                } else {
                    // Contiguous [N, C*bpc]; joint non-SH uses 16-bit cells.
                    size_t row_elems = 1;
                    for (size_t d = 1; d < parameter.shape().rank(); ++d)
                        row_elems *= parameter.shape()[d];
                    if (state.joint_bits != 16 ||
                        state.exp_avg.ndim() != 2 ||
                        state.exp_avg.shape()[0] != parameter.shape()[0] ||
                        state.exp_avg.shape()[1] != row_elems * static_cast<size_t>(bpc)) {
                        throw std::runtime_error("Invalid AdamOptimizer checkpoint: joint packed shape mismatch");
                    }
                }
                state.exp_avg = state.exp_avg.cuda();
                state.joint_bounds = state.joint_bounds.cuda();
            }

            // Serialized capacity is advisory and may be attacker-controlled.
            // The validated checkpoint max_cap is reserved by load_checkpoint
            // after all state has parsed successfully.
            // Joint packed buffers are not row-major param-shaped — keep size as capacity base.
            state.capacity = state.size;
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
        frozen_lr_scale_ = loaded.frozen_lr_scale_;
    }

    void AdamOptimizer::reserve_capacity(const size_t capacity) {
        using namespace joint_adam;

        // Grow a 1D/2D tensor to `cap_rows` along dim0 (or total elems for 1D packed),
        // preserving the live prefix. zeros_direct tensors reject Tensor::reserve, so
        // rebuild via zeros_direct + D2D copy when capacity is insufficient.
        const auto grow_tensor = [](lfs::core::Tensor& t, const size_t cap_rows) {
            if (!t.is_valid() || cap_rows == 0)
                return;
            if (t.capacity() >= cap_rows)
                return;
            const auto shape = t.shape();
            auto grown = lfs::core::Tensor::zeros_direct(shape, cap_rows, t.device(), t.dtype());
            if (t.numel() > 0 && t.data_ptr() && grown.data_ptr()) {
                const size_t elem_bytes = lfs::core::dtype_size(t.dtype());
                if (elem_bytes > 0) {
                    LFS_CUDA_CHECK(cudaMemcpy(
                        grown.data_ptr(), t.data_ptr(),
                        t.numel() * elem_bytes, cudaMemcpyDeviceToDevice));
                }
            }
            if (!t.name().empty())
                grown.set_name(t.name());
            t = std::move(grown);
        };

        for (auto& [name, state] : states_) {
            const bool is_shN = (name == "shN");
            // Moments use float-count capacity for swizzled shN, primitive rows otherwise.
            // Scales / joint_bounds are always per-primitive (bounds table is ceil(N/256)).
            const size_t target_capacity =
                is_shN ? lfs::core::sh_swizzled_float_count(
                             capacity,
                             static_cast<uint32_t>(splat_data_.max_sh_coeffs_rest()))
                       : capacity;

            if (target_capacity > state.capacity) {
                if (state.is_joint()) {
                    const int bpc = bytes_per_cell(state.joint_bits);
                    if (state.exp_avg.is_valid() && bpc > 0) {
                        if (is_shN) {
                            // Packed 1D: capacity is in packed bytes (floats * bpc).
                            grow_tensor(state.exp_avg, target_capacity * static_cast<size_t>(bpc));
                        } else {
                            // Contiguous [N, C*bpc]: capacity is in rows.
                            grow_tensor(state.exp_avg, target_capacity);
                        }
                    }
                    if (state.joint_bounds.is_valid()) {
                        const size_t bounds_cap = n_bounds_for_prims(capacity);
                        grow_tensor(state.joint_bounds, bounds_cap);
                    }
                } else {
                    if (state.grad.is_valid())
                        grow_tensor(state.grad, target_capacity);
                    if (state.exp_avg.is_valid())
                        grow_tensor(state.exp_avg, target_capacity);
                }
                state.capacity = target_capacity;
            }

            if (state.is_joint() && state.joint_bounds.is_valid()) {
                const size_t bounds_cap = n_bounds_for_prims(capacity);
                if (bounds_cap > state.joint_bounds.capacity())
                    grow_tensor(state.joint_bounds, bounds_cap);
            }
        }
    }

    void AdamOptimizer::reset_state(const ParamType type) {
        auto* state = get_state_mutable(type);
        if (!state || !state->exp_avg.is_valid()) {
            return;
        }

        // All-zero packed + bounds → free zero moments (joint only).
        state->exp_avg.zero_();
        if (state->joint_bounds.is_valid())
            state->joint_bounds.zero_();
        state->step_count = 0;
    }

} // namespace lfs::training
