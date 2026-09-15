/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>
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

    // Synchronous host<->device copy ordered against `stream`, the tensor's
    // home stream. cudaMemcpy runs on the legacy default stream, which a
    // non-blocking home stream neither waits for nor is waited on by: an
    // upload from pageable memory returns while its DMA is still in flight,
    // and a download may read before the home stream's producer finished.
    // Downloads make the legacy stream wait for `stream` first; uploads make
    // `stream` wait for the legacy stream afterwards. The host never waits on
    // `stream` itself, so a gated or capturing home stream cannot deadlock.
    LFS_CORE_API cudaError_t memcpy_ordered(void* dst, const void* src, size_t bytes,
                                            cudaMemcpyKind kind, cudaStream_t stream);

    /**
     * RAII guard for temporarily setting the current CUDA stream
     * (PyTorch's CUDAStreamGuard pattern)
     */
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
