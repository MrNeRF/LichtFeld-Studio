/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "adam_optimizer.hpp"
#include "adam_api.h" // fast_lfs::optimizer::adam_step_raw
#include "core/logger.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>

// Simple CUDA error checking macro
#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            throw std::runtime_error(std::string("CUDA error: ") + \
                                     cudaGetErrorString(err));     \
        }                                                          \
    } while (0)

namespace lfs::training {

    namespace {
        constexpr int SH_WARMUP_ITERATIONS = 1000;
        constexpr float DEFAULT_GROWTH_MULTIPLIER = 1.5f;
    } // namespace

    AdamOptimizer::AdamOptimizer(lfs::core::SplatData& splat_data, const AdamConfig& config)
        : config_(config),
          splat_data_(splat_data) {}

    void AdamOptimizer::step(const int iteration) {
        for (const auto type : all_param_types()) {
            step_param(type, iteration);
        }
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
            const size_t alloc_cap = (capacity > param_size) ? capacity : param_size;

            // Handle zero-size tensors (e.g., shN with sh-degree 0 has shape [N, 0, 3])
            // Still allocate state tensors to maintain valid structure for densification
            if (alloc_cap > param_size) {
                state.grad = lfs::core::Tensor::zeros_direct(param.shape(), alloc_cap);
                state.exp_avg = lfs::core::Tensor::zeros_direct(param.shape(), alloc_cap);
                state.exp_avg_sq = lfs::core::Tensor::zeros_direct(param.shape(), alloc_cap);
            } else {
                state.grad = lfs::core::Tensor::zeros(param.shape(), param.device());
                state.exp_avg = lfs::core::Tensor::zeros(param.shape(), param.device());
                state.exp_avg_sq = lfs::core::Tensor::zeros(param.shape(), param.device());
            }
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
        for (auto& [_, state] : states_) {
            if (state.grad.is_valid() && state.grad.numel() > 0) {
                const size_t bytes = state.size * (state.grad.numel() / state.grad.shape()[0]) * sizeof(float);
                CHECK_CUDA(cudaMemsetAsync(state.grad.ptr<float>(), 0, bytes, lfs::core::getCurrentCUDAStream()));
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
            throw std::runtime_error("get_grad: " + name + " not initialized");
        }
        return it->second.grad;
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

    void AdamOptimizer::init_state(ParamType type) {
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
        const size_t initial_cap = compute_new_capacity(0, param_size);

        if (!state.grad.is_valid() || state.grad.numel() == 0) {
            state.grad = (initial_cap > param_size)
                             ? lfs::core::Tensor::zeros_direct(param.shape(), initial_cap)
                             : lfs::core::Tensor::zeros(param.shape(), param.device());
        }

        if (initial_cap > param_size) {
            state.exp_avg = lfs::core::Tensor::zeros_direct(param.shape(), initial_cap);
            state.exp_avg_sq = lfs::core::Tensor::zeros_direct(param.shape(), initial_cap);
            state.capacity = initial_cap;
        } else {
            state.exp_avg = lfs::core::Tensor::zeros(param.shape(), param.device());
            state.exp_avg_sq = lfs::core::Tensor::zeros(param.shape(), param.device());
            state.capacity = param_size;
        }
        state.size = param_size;
        state.step_count = 0;
        LOG_DEBUG("Initialized optimizer state for {}: size={}, capacity={}", name, param_size, state.capacity);
    }

    void AdamOptimizer::step_param(ParamType type, const int iteration) {
        auto& param = get_param(type);
        if (!param.is_valid() || param.numel() == 0) {
            return;
        }

        const auto name = param_name(type);
        if (!states_.contains(name)) {
            init_state(type);
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

        const size_t feature_dim = param.numel() / param_size;
        const size_t num_elements = state.size * feature_dim;

        fast_lfs::optimizer::adam_step_raw(
            param.ptr<float>(),
            state.exp_avg.ptr<float>(),
            state.exp_avg_sq.ptr<float>(),
            state.grad.ptr<float>(),
            static_cast<int>(num_elements),
            param_lr,
            config_.beta1,
            config_.beta2,
            config_.eps,
            bias_correction1_rcp,
            bias_correction2_sqrt_rcp);
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
            if (!param.is_valid() || (param.ndim() >= 2 && param.shape()[1] == 0)) {
                return; // ShN is empty at sh-degree 0, nothing to reset
            }
        }

        auto& state = states_[name];

        // Validate tensors before accessing
        if (!state.exp_avg.is_valid() || state.exp_avg.ptr<float>() == nullptr) {
            LOG_WARN("reset_state_at_indices: {} exp_avg tensor is invalid or null", name);
            return;
        }
        if (!state.exp_avg_sq.is_valid() || state.exp_avg_sq.ptr<float>() == nullptr) {
            LOG_WARN("reset_state_at_indices: {} exp_avg_sq tensor is invalid or null", name);
            return;
        }

        const auto& shape = state.exp_avg.shape();
        int row_size = 1;
        for (size_t i = 1; i < shape.rank(); i++) {
            row_size *= static_cast<int>(shape[i]);
        }

        int64_t* d_indices;
        CHECK_CUDA(cudaMalloc(&d_indices, indices.size() * sizeof(int64_t)));
        CHECK_CUDA(cudaMemcpy(d_indices, indices.data(), indices.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

        fast_lfs::optimizer::zero_rows_at_indices(state.exp_avg.ptr<float>(), d_indices, indices.size(), row_size);
        fast_lfs::optimizer::zero_rows_at_indices(state.exp_avg_sq.ptr<float>(), d_indices, indices.size(), row_size);

        CHECK_CUDA(cudaFree(d_indices));
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

        if (!param.is_valid() || param.shape().rank() == 0) {
            LOG_WARN("extend_state_by_gather: {} param invalid", name);
            return;
        }
        if (!state.exp_avg.is_valid() || state.exp_avg.ndim() == 0) {
            LOG_WARN("extend_state_by_gather: {} state invalid", name);
            return;
        }

        // Fast path: use reserved capacity
        const bool all_have_capacity = state.grad.capacity() > 0 &&
                                       state.exp_avg.capacity() > 0 &&
                                       state.exp_avg_sq.capacity() > 0;
        const bool fits_in_capacity = new_size <= state.grad.capacity() &&
                                      new_size <= state.exp_avg.capacity() &&
                                      new_size <= state.exp_avg_sq.capacity();
        if (all_have_capacity && fits_in_capacity) {
            // exp_avg and exp_avg_sq: gather from existing (copy optimizer momentum for duplicated Gaussians)
            state.exp_avg.append_gather(indices);
            state.exp_avg_sq.append_gather(indices);
            // grad: append zeros (new Gaussians have no gradients yet)
            state.grad.append_zeros(n_new);
            state.size = new_size;
            state.capacity = state.exp_avg.capacity();
            LOG_DEBUG("extend_state_by_gather({}): fast path done", name);
            return;
        }
        LOG_WARN("extend_state_by_gather({}): SLOW PATH triggered (all_have_cap={}, fits={})", name, all_have_capacity, fits_in_capacity);

        // Slow path: reallocate without extra capacity
        const auto& shape = param.shape();
        std::vector<size_t> new_dims(shape.rank());
        new_dims[0] = new_size;
        for (size_t i = 1; i < shape.rank(); i++) {
            new_dims[i] = shape[i];
        }

        const auto tensor_shape = lfs::core::TensorShape(new_dims);
        state.grad = lfs::core::Tensor::zeros(tensor_shape, param.device());

        auto new_exp_avg = lfs::core::Tensor::empty(tensor_shape, param.device());
        auto new_exp_avg_sq = lfs::core::Tensor::empty(tensor_shape, param.device());

        // Copy old data
        const size_t row_size = param.numel() / shape[0];
        if (state.size > 0 && state.exp_avg.numel() > 0) {
            const size_t old_bytes = state.size * row_size * sizeof(float);
            CHECK_CUDA(cudaMemcpyAsync(new_exp_avg.ptr<float>(), state.exp_avg.ptr<float>(), old_bytes, cudaMemcpyDeviceToDevice, lfs::core::getCurrentCUDAStream()));
            CHECK_CUDA(cudaMemcpyAsync(new_exp_avg_sq.ptr<float>(), state.exp_avg_sq.ptr<float>(), old_bytes, cudaMemcpyDeviceToDevice, lfs::core::getCurrentCUDAStream()));
        }

        // Gather new rows using GPU-native index_select (single GPU operation)
        const auto gathered_avg = state.exp_avg.index_select(0, indices);
        const auto gathered_sq = state.exp_avg_sq.index_select(0, indices);

        // Copy gathered rows to destination (GPU→GPU bulk copy)
        const size_t gathered_bytes = n_new * row_size * sizeof(float);
        const size_t dst_offset = state.size * row_size * sizeof(float);
        CHECK_CUDA(cudaMemcpyAsync(
            reinterpret_cast<char*>(new_exp_avg.ptr<float>()) + dst_offset,
            gathered_avg.ptr<float>(), gathered_bytes, cudaMemcpyDeviceToDevice, lfs::core::getCurrentCUDAStream()));
        CHECK_CUDA(cudaMemcpyAsync(
            reinterpret_cast<char*>(new_exp_avg_sq.ptr<float>()) + dst_offset,
            gathered_sq.ptr<float>(), gathered_bytes, cudaMemcpyDeviceToDevice, lfs::core::getCurrentCUDAStream()));

        state.exp_avg = std::move(new_exp_avg);
        state.exp_avg_sq = std::move(new_exp_avg_sq);
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

        // Skip zero-coefficient ShN (sh-degree 0)
        if (type == ParamType::ShN) {
            const auto& shN_param = get_param(type);
            if (!shN_param.is_valid() || (shN_param.ndim() >= 2 && shN_param.shape()[1] == 0)) {
                return;
            }
        }

        auto& param = get_param(type);
        auto& state = states_[name];
        const size_t new_size = state.size + n_new;

        if (!param.is_valid() || param.shape().rank() == 0) {
            throw std::runtime_error("extend_state: " + name + " invalid");
        }
        if (!state.exp_avg.is_valid() || state.exp_avg.ndim() == 0) {
            throw std::runtime_error("extend_state: " + name + " state invalid");
        }

        // Fast path: use reserved capacity (all tensors must have capacity)
        const bool all_have_capacity = state.grad.capacity() > 0 &&
                                       state.exp_avg.capacity() > 0 &&
                                       state.exp_avg_sq.capacity() > 0;
        const bool fits_in_capacity = new_size <= state.grad.capacity() &&
                                      new_size <= state.exp_avg.capacity() &&
                                      new_size <= state.exp_avg_sq.capacity();
        if (all_have_capacity && fits_in_capacity) {
            state.grad.append_zeros(n_new);
            state.exp_avg.append_zeros(n_new);
            state.exp_avg_sq.append_zeros(n_new);
            state.size = new_size;
            state.capacity = state.exp_avg.capacity();
            LOG_DEBUG("extend_state_for_new_params({}): fast path done, new size = {}", name, new_size);
            return;
        }
        LOG_WARN("extend_state_for_new_params({}): SLOW PATH triggered (all_have_cap={}, fits={})", name, all_have_capacity, fits_in_capacity);

        // Slow path: reallocate without extra capacity (use reserve() for pre-allocation)
        const auto& shape = param.shape();
        std::vector<size_t> new_dims(shape.rank());
        new_dims[0] = new_size;
        for (size_t i = 1; i < shape.rank(); i++) {
            new_dims[i] = shape[i];
        }

        const auto tensor_shape = lfs::core::TensorShape(new_dims);
        state.grad = lfs::core::Tensor::zeros(tensor_shape, param.device());
        auto new_exp_avg = lfs::core::Tensor::empty(tensor_shape, param.device());
        auto new_exp_avg_sq = lfs::core::Tensor::empty(tensor_shape, param.device());

        if (state.size > 0 && state.exp_avg.numel() > 0) {
            const size_t old_bytes = state.exp_avg.numel() * sizeof(float);
            CHECK_CUDA(cudaMemcpyAsync(new_exp_avg.ptr<float>(), state.exp_avg.ptr<float>(), old_bytes, cudaMemcpyDeviceToDevice, lfs::core::getCurrentCUDAStream()));
            CHECK_CUDA(cudaMemcpyAsync(new_exp_avg_sq.ptr<float>(), state.exp_avg_sq.ptr<float>(), old_bytes, cudaMemcpyDeviceToDevice, lfs::core::getCurrentCUDAStream()));
        }

        const size_t row_size = param.numel() / shape[0];
        const size_t offset = state.exp_avg.numel() * sizeof(float);
        const size_t new_bytes = n_new * row_size * sizeof(float);
        CHECK_CUDA(cudaMemsetAsync(reinterpret_cast<char*>(new_exp_avg.ptr<float>()) + offset, 0, new_bytes, lfs::core::getCurrentCUDAStream()));
        CHECK_CUDA(cudaMemsetAsync(reinterpret_cast<char*>(new_exp_avg_sq.ptr<float>()) + offset, 0, new_bytes, lfs::core::getCurrentCUDAStream()));

        state.exp_avg = std::move(new_exp_avg);
        state.exp_avg_sq = std::move(new_exp_avg_sq);
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
        param = lfs::core::Tensor::cat({param, new_values}, 0);
        extend_state_for_new_params(type, n_new);
    }

    void AdamOptimizer::add_new_params_gather(ParamType type, const lfs::core::Tensor& indices) {
        auto& param = get_param(type);

        // Special case: ShN with 0 higher-order SH coefficients (sh_degree=0)
        // The tensor may be either:
        // 1. Completely uninitialized (ndim=0) - need to create [new_size, 0, 3]
        // 2. Exists with shape [N, 0, 3] - need to resize to [new_size, 0, 3]
        // We still need to resize dim 0 to match the new number of gaussians for cat() in get_shs()
        if (type == ParamType::ShN) {
            const bool is_uninitialized = (param.ndim() == 0);
            const bool has_zero_sh_coeffs = (param.ndim() >= 2 && param.shape()[1] == 0);

            if (is_uninitialized || has_zero_sh_coeffs) {
                // sh0 is already updated at this point, so just match its size
                // (don't add n_new again - that would double-count)
                const auto& sh0 = splat_data_.sh0();
                const size_t new_size = sh0.shape()[0];

                LOG_DEBUG("add_new_params_gather: ShN resize to {} (uninitialized={})",
                          new_size, is_uninitialized);

                // Create new tensor with correct shape [new_size, 0, 3]
                auto new_tensor = lfs::core::Tensor::empty(
                    {new_size, 0, 3},
                    lfs::core::Device::CUDA, lfs::core::DataType::Float32);

                // Assign to the reference (this updates splat_data_._shN)
                param = std::move(new_tensor);
                // No optimizer state to extend for empty tensors
                return;
            }
        }

        if (!param.is_valid()) {
            if (type == ParamType::ShN)
                return; // ShN may not be initialized at sh-degree 0
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

        int64_t* d_indices;
        CHECK_CUDA(cudaMalloc(&d_indices, indices.size() * sizeof(int64_t)));
        CHECK_CUDA(cudaMemcpy(d_indices, indices.data(), indices.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
        relocate_params_at_indices_gpu(type, d_indices, indices.size());
        CHECK_CUDA(cudaFree(d_indices));
    }

    void AdamOptimizer::relocate_params_at_indices_gpu(ParamType type, const int64_t* indices_device, const size_t n_indices) {
        if (n_indices == 0)
            return;

        const auto name = param_name(type);
        if (!states_.contains(name))
            return;

        // Skip ShN when not initialized (sh_degree=0 case)
        if (type == ParamType::ShN) {
            const auto& param = get_param(type);
            if (!param.is_valid() || (param.ndim() >= 2 && param.shape()[1] == 0)) {
                return; // ShN is empty at sh-degree 0, nothing to relocate
            }
        }

        auto& state = states_[name];

        // Validate tensors before accessing
        if (!state.exp_avg.is_valid() || state.exp_avg.ptr<float>() == nullptr) {
            LOG_WARN("relocate_params_at_indices_gpu: {} exp_avg tensor is invalid or null", name);
            return;
        }
        if (!state.exp_avg_sq.is_valid() || state.exp_avg_sq.ptr<float>() == nullptr) {
            LOG_WARN("relocate_params_at_indices_gpu: {} exp_avg_sq tensor is invalid or null", name);
            return;
        }

        const auto& shape = state.exp_avg.shape();
        int row_size = 1;
        for (size_t i = 1; i < shape.rank(); i++) {
            row_size *= static_cast<int>(shape[i]);
        }

        // Zero optimizer state (m, v) at relocated indices
        fast_lfs::optimizer::zero_rows_at_indices(state.exp_avg.ptr<float>(), indices_device, n_indices, row_size);
        fast_lfs::optimizer::zero_rows_at_indices(state.exp_avg_sq.ptr<float>(), indices_device, n_indices, row_size);
    }

    namespace {
        constexpr uint32_t ADAM_STATE_MAGIC = 0x4C464144; // "LFAD"
        constexpr uint32_t ADAM_STATE_VERSION = 1;
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

        uint32_t num_states = 0;
        for (const auto& [_, state] : states_) {
            if (state.exp_avg.is_valid() && state.exp_avg_sq.is_valid())
                ++num_states;
        }
        os.write(reinterpret_cast<const char*>(&num_states), sizeof(num_states));

        for (const auto& [name, state] : states_) {
            if (!state.exp_avg.is_valid() || !state.exp_avg_sq.is_valid())
                continue;

            const auto name_len = static_cast<uint32_t>(name.size());
            os.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
            os.write(name.data(), name_len);
            os.write(reinterpret_cast<const char*>(&state.step_count), sizeof(state.step_count));
            os.write(reinterpret_cast<const char*>(&state.capacity), sizeof(state.capacity));
            os.write(reinterpret_cast<const char*>(&state.size), sizeof(state.size));
            os << state.exp_avg << state.exp_avg_sq;
        }
        LOG_DEBUG("Serialized AdamOptimizer: {} states", num_states);
    }

    void AdamOptimizer::deserialize(std::istream& is) {
        uint32_t magic, version;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != ADAM_STATE_MAGIC) {
            throw std::runtime_error("Invalid AdamOptimizer checkpoint");
        }
        if (version != ADAM_STATE_VERSION) {
            throw std::runtime_error("Unsupported checkpoint version");
        }

        is.read(reinterpret_cast<char*>(&config_.lr), sizeof(config_.lr));
        is.read(reinterpret_cast<char*>(&config_.beta1), sizeof(config_.beta1));
        is.read(reinterpret_cast<char*>(&config_.beta2), sizeof(config_.beta2));
        is.read(reinterpret_cast<char*>(&config_.eps), sizeof(config_.eps));
        is.read(reinterpret_cast<char*>(&config_.growth_factor), sizeof(config_.growth_factor));
        is.read(reinterpret_cast<char*>(&config_.initial_capacity), sizeof(config_.initial_capacity));

        uint32_t num_param_lrs;
        is.read(reinterpret_cast<char*>(&num_param_lrs), sizeof(num_param_lrs));
        config_.param_lrs.clear();
        for (uint32_t i = 0; i < num_param_lrs; ++i) {
            uint32_t name_len;
            is.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
            std::string name(name_len, '\0');
            is.read(name.data(), name_len);
            double lr;
            is.read(reinterpret_cast<char*>(&lr), sizeof(lr));
            config_.param_lrs[name] = lr;
        }

        uint32_t num_states;
        is.read(reinterpret_cast<char*>(&num_states), sizeof(num_states));

        states_.clear();
        for (uint32_t i = 0; i < num_states; ++i) {
            uint32_t name_len;
            is.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
            std::string name(name_len, '\0');
            is.read(name.data(), name_len);

            AdamParamState state;
            is.read(reinterpret_cast<char*>(&state.step_count), sizeof(state.step_count));
            is.read(reinterpret_cast<char*>(&state.capacity), sizeof(state.capacity));
            is.read(reinterpret_cast<char*>(&state.size), sizeof(state.size));

            is >> state.exp_avg >> state.exp_avg_sq;
            state.exp_avg = state.exp_avg.cuda();
            state.exp_avg_sq = state.exp_avg_sq.cuda();

            const size_t target_cap = std::max(state.capacity, compute_new_capacity(state.size, state.size));
            if (target_cap > state.size) {
                state.exp_avg.reserve(target_cap);
                state.exp_avg_sq.reserve(target_cap);
                state.capacity = target_cap;
            }
            states_[name] = std::move(state);
        }

        // Allocate gradient buffers (not serialized)
        for (auto& [_, state] : states_) {
            if (state.exp_avg.is_valid()) {
                state.grad = lfs::core::Tensor::zeros_direct(state.exp_avg.shape(), state.capacity);
            }
        }
        LOG_DEBUG("Deserialized AdamOptimizer: {} states", num_states);
    }

    void AdamOptimizer::reserve_capacity(const size_t capacity) {
        for (auto& [_, state] : states_) {
            if (capacity > state.capacity) {
                if (state.grad.is_valid())
                    state.grad.reserve(capacity);
                state.exp_avg.reserve(capacity);
                state.exp_avg_sq.reserve(capacity);
                state.capacity = capacity;
            }
        }
    }

} // namespace lfs::training
