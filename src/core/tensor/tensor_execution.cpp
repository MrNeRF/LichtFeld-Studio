/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_execution.hpp"

#include "backend/backend_driver.hpp"
#include "backend/gpu_backend_ops.hpp"
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
        : backend_scope_(target.backend()), driver_(&internal::backend_driver(target.backend())), binding_(static_cast<internal::BackendDriver*>(driver_)->bind_target(target.native_handle())) {}
    TensorExecutionTarget::Scope::~Scope() { static_cast<internal::BackendDriver*>(driver_)->unbind_target(binding_); }

    void TensorExecutionTarget::wait() const {
        internal::backend_ops(backend_).synchronize_stream(
            internal::ExecContext{static_cast<cudaStream_t>(target_)});
    }

    void TensorExecutionTarget::wait_for(TensorExecutionTarget producer) const {
        if (backend_ != producer.backend_)
            throw std::invalid_argument("Tensor queue bridge backend mismatch");
        internal::backend_driver(backend_).bridge_targets(target_, producer.target_);
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
        if (device == Device::CPU)
            return from_blob(data, std::move(shape), device, dtype, static_cast<cudaStream_t>(target.native_handle()));
        const size_t width = dtype_size(dtype);
        if (!width || shape.elements() > SIZE_MAX / width)
            throw std::invalid_argument("Raw tensor view byte count overflow");
        const TensorExecutionTarget::Scope scope(target);
        const auto storage = internal::backend_ops(target.backend()).borrow_native_range(data, shape.elements() * width);
        if (!storage.meta)
            return from_blob(data, std::move(shape), device, dtype, static_cast<cudaStream_t>(target.native_handle()));
        if (storage.byte_offset % width)
            throw std::invalid_argument("Raw tensor view offset is not aligned");
        Tensor result = from_blob(storage.data, std::move(shape), Device::CPU, dtype);
        result.device_ = Device::GPU;
        result.storage_offset_ = storage.byte_offset / width;
        result.storage_meta_ = std::shared_ptr<StorageMeta>(const_cast<StorageMeta*>(storage.meta), [](StorageMeta*) {});
        result.view_generation_snapshot_ = storage.meta->generation.load(std::memory_order_relaxed);
        return result;
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
