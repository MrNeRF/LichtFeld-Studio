/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor/internal/cuda_stream_context.hpp"
#include <cuda_runtime.h>

namespace lfs {

    // Launchers fall back to the thread's current CUDA stream when the caller
    // doesn't pass one, so kernel work follows the training/render stream guard
    // instead of silently landing on the legacy default stream. In namespace lfs
    // so every kernel namespace (lfs::training::*, lfs::filters) finds it
    // unqualified.
    inline cudaStream_t resolve_stream(void* stream) {
        return stream ? static_cast<cudaStream_t>(stream) : lfs::core::getCurrentCUDAStream();
    }

    inline cudaStream_t resolve_stream(cudaStream_t stream) {
        return stream ? stream : lfs::core::getCurrentCUDAStream();
    }

} // namespace lfs
