/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/cuda_types.hpp"
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>

#include "core/export.hpp"
#include "core/tensor_fwd.hpp"

namespace lfs::core {

    // Thread-local current CUDA stream (PyTorch-style).
    // Exported from lfs_core so the singleton is shared across DSO boundaries.
    LFS_CORE_API cudaStream_t getCurrentCUDAStream();
    LFS_CORE_API void setCurrentCUDAStream(cudaStream_t stream);

    // Makes execution_stream wait (GPU-side) for work currently enqueued on
    // dependency_stream. Uses pooled events; falls back to a host sync on failure.
    LFS_CORE_API void waitForCUDAStream(cudaStream_t execution_stream, cudaStream_t dependency_stream);

    LFS_CORE_API cudaStream_t prepare_inputs_for_stream(
        std::initializer_list<const Tensor*> inputs,
        std::optional<cudaStream_t> execution_stream = std::nullopt);

    LFS_CORE_API void bridgeStreams(cudaStream_t from, cudaStream_t to);
    // A retired stream handle must not be passed to the driver.
    LFS_CORE_API bool is_stream_retired(cudaStream_t stream) noexcept;
    // Release allocator stream records before destroying an external CUDA stream.
    LFS_CORE_API void release_cuda_stream(cudaStream_t stream);
    LFS_CORE_API size_t cuda_allocation_size(size_t bytes);
    LFS_CORE_API void trim_cuda_memory_pool();
    LFS_CORE_API void safe_cuda_pool_deallocate(void* ptr, cudaStream_t stream = nullptr) noexcept;

    enum class CudaStorageMode : uint8_t {
        Pooled,
        ExactAsync,
        Direct,
    };

    LFS_CORE_API void* allocate_cuda_storage(
        size_t bytes,
        cudaStream_t stream = nullptr,
        CudaStorageMode mode = CudaStorageMode::Pooled,
        const char* label = "tensor.storage",
        const char* operation = "tensor.allocate");
    [[nodiscard]] LFS_CORE_API bool tensor_uses_cuda_storage(const Tensor& tensor);

    class LFS_CORE_API TensorCudaStream {
    public:
        TensorCudaStream();
        ~TensorCudaStream();
        TensorCudaStream(const TensorCudaStream&) = delete;
        TensorCudaStream& operator=(const TensorCudaStream&) = delete;

        [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

    private:
        cudaStream_t stream_ = nullptr;
    };

    class CUDAStreamGuard {
    public:
        explicit CUDAStreamGuard(cudaStream_t stream)
            : prev_stream_(getCurrentCUDAStream()) {
            setCurrentCUDAStream(stream);
        }

        ~CUDAStreamGuard() {
            setCurrentCUDAStream(prev_stream_);
        }

        CUDAStreamGuard(const CUDAStreamGuard&) = delete;
        CUDAStreamGuard& operator=(const CUDAStreamGuard&) = delete;
        CUDAStreamGuard(CUDAStreamGuard&&) = delete;
        CUDAStreamGuard& operator=(CUDAStreamGuard&&) = delete;

    private:
        cudaStream_t prev_stream_;
    };

} // namespace lfs::core
