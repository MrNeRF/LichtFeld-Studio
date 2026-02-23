/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>

#include "core/export.hpp"

namespace lfs::core {

    // Thread-local current CUDA stream (PyTorch-style).
    // Exported from lfs_core so the singleton is shared across DSO boundaries.
    LFS_CORE_API cudaStream_t getCurrentCUDAStream();
    LFS_CORE_API void setCurrentCUDAStream(cudaStream_t stream);

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
