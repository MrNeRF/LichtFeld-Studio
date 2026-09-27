/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor_backend.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "core/tensor_upload.hpp"

namespace lfs::io {
    // Synchronous entry points can also be called outside a trainer/worker scope.
    // A CUDA session keeps one non-blocking queue per calling thread, shared by
    // nested decode and cache calls. Any other session does not own a CUDA queue.
    inline cudaStream_t image_execution_stream(void* requested = nullptr) {
        if (requested)
            return static_cast<cudaStream_t>(requested);
        if (const auto current = core::getCurrentCUDAStream())
            return current;
        if (core::default_gpu_backend() != core::GpuBackend::CUDA)
            return nullptr;
        thread_local core::TensorWorkQueue queue(core::GpuBackend::CUDA);
        return static_cast<cudaStream_t>(queue.native_handle());
    }
} // namespace lfs::io
