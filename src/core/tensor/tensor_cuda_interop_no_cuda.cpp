/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_cuda_interop.hpp"

#include "core/cuda_vulkan_interop.hpp"
#include "internal/tensor_impl.hpp"

#include <stdexcept>

namespace lfs::core {
    namespace {
        [[noreturn]] void throw_cuda_unavailable() {
            throw std::runtime_error("CUDA support is not compiled into this build");
        }
    } // namespace

    cudaStream_t getCurrentCUDAStream() { return nullptr; }

    void setCurrentCUDAStream(const cudaStream_t stream) {
        if (stream)
            throw_cuda_unavailable();
    }

    cudaStream_t prepare_inputs_for_stream(
        const std::initializer_list<const Tensor*> inputs,
        const std::optional<cudaStream_t> execution_stream) {
        if (execution_stream.value_or(nullptr))
            throw_cuda_unavailable();
        for (const Tensor* input : inputs) {
            if (!input || !input->is_valid())
                throw TensorError("stream preparation requires valid tensor inputs");
            if (input->device() == Device::GPU &&
                internal::gpu_backend_tag(*input) == GpuBackend::CUDA)
                throw_cuda_unavailable();
        }
        return nullptr;
    }

    size_t cuda_allocation_size(size_t) { return 0; }
    void trim_cuda_memory_pool() {}
    void safe_cuda_pool_deallocate(void*, cudaStream_t) noexcept {}
    bool tensor_uses_cuda_storage(const Tensor&) { return false; }

    TensorCudaStream::TensorCudaStream() { throw_cuda_unavailable(); }
    TensorCudaStream::~TensorCudaStream() = default;
} // namespace lfs::core

namespace lfs::core::cuda {
    bool ExternalMemoryImportScope::active() noexcept { return false; }
} // namespace lfs::core::cuda
