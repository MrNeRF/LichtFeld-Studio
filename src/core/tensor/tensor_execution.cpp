/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_execution.hpp"

#include "backend/vulkan/vk_context.hpp"
#include "backend/vulkan/vk_recorder.hpp"
#include "core/tensor_upload.hpp"
#include "internal/tensor_impl.hpp"

#include <stdexcept>

namespace lfs::core {
    TensorExecutionTarget::TensorExecutionTarget(const TensorWorkQueue& queue)
        : backend_(queue.backend()), target_(queue.native_handle()) {}

    TensorExecutionTarget TensorExecutionTarget::default_queue(GpuBackend backend) {
        return {backend, nullptr};
    }

    TensorExecutionTarget::Scope::Scope(TensorExecutionTarget target)
        : backend_scope_(target.backend()), previous_target_(getCurrentCUDAStream()) {
        if (target.backend() == GpuBackend::Vulkan && target.native_handle() != nullptr) {
            const auto id = reinterpret_cast<uint64_t>(target.native_handle());
            auto context = internal::acquire_vulkan_context();
            if (!context->recorders().owns_queue(id))
                throw std::invalid_argument("Tensor execution target is not a Vulkan queue");
            context->recorders().bind_queue(id);
            rebound_vulkan_ = true;
            return;
        }
        setCurrentCUDAStream(static_cast<cudaStream_t>(target.native_handle()));
    }

    TensorExecutionTarget::Scope::~Scope() {
        if (rebound_vulkan_) {
            if (const auto context = internal::try_live_vulkan_context())
                context->recorders().unbind_queue();
            return;
        }
        setCurrentCUDAStream(static_cast<cudaStream_t>(previous_target_));
    }

    void TensorExecutionTarget::wait() const {
        internal::backend_ops(backend_).synchronize_stream(
            internal::ExecContext{static_cast<cudaStream_t>(target_)});
    }

    void TensorExecutionTarget::wait_for(TensorExecutionTarget producer) const {
        if (backend_ != producer.backend_)
            throw std::invalid_argument("Tensor queue bridge backend mismatch");
        if (backend_ != GpuBackend::CUDA)
            throw std::runtime_error("Explicit tensor queue bridges are unsupported on this backend");
        waitForCUDAStream(static_cast<cudaStream_t>(target_),
                          static_cast<cudaStream_t>(producer.target_));
    }

    void TensorExecutionTarget::set_name(const char* name) const {
        internal::backend_ops(backend_).name_queue(
            internal::ExecContext{static_cast<cudaStream_t>(target_)}, name);
    }

    TensorExecutionTarget Tensor::execution_target() const {
        const auto backend = gpu_backend_of(*this);
        return {backend ? *backend : TensorExecutionTarget::current().backend(), stream()};
    }

    namespace {
        cudaStream_t tensor_target(const Tensor& tensor, TensorExecutionTarget target) {
            if (tensor.device() == Device::GPU && gpu_backend_of(tensor) != target.backend())
                throw std::invalid_argument("Tensor execution target backend mismatch");
            return static_cast<cudaStream_t>(target.native_handle());
        }
    } // namespace

    void Tensor::set_stream(TensorExecutionTarget target) {
        set_stream(tensor_target(*this, target));
    }

    void Tensor::sync_to_stream(TensorExecutionTarget target) const {
        // Vulkan/Metal tensor operations already order storage on their timelines.
        sync_to_stream(tensor_target(*this, target));
    }

    void Tensor::record_stream(TensorExecutionTarget target) const {
        record_stream(tensor_target(*this, target));
    }

    Tensor Tensor::from_blob(void* data, TensorShape shape, Device device, DataType dtype,
                             TensorExecutionTarget target) {
        if (device == Device::GPU && target.backend() != GpuBackend::CUDA)
            throw std::runtime_error("Raw device tensor views are unsupported on this backend");
        return from_blob(data, std::move(shape), device, dtype,
                         static_cast<cudaStream_t>(target.native_handle()));
    }

    Tensor Tensor::to(Device device, TensorExecutionTarget target) const {
        TensorExecutionTarget::Scope scope(target);
        return to(device, tensor_target(*this, target));
    }

    Tensor& Tensor::fill_(float value, TensorExecutionTarget target) {
        return fill_(value, tensor_target(*this, target));
    }

    void push_gpu_range(const char* name) {
        internal::backend_ops(TensorExecutionTarget::current().backend()).push_range(name);
    }

    void pop_gpu_range() {
        internal::backend_ops(TensorExecutionTarget::current().backend()).pop_range();
    }
} // namespace lfs::core
