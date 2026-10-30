/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor_cuda_interop.hpp"
#include <array>
#include <cstdint>

namespace lfs::core::internal {
    inline bool cuda_deferred_home_needs_binding(void* home) {
        // Match the original implicit CUDA guard: an explicit current stream
        // wins, and a retired home cannot order the materializer's work.
        return home && !getCurrentCUDAStream() &&
               !is_stream_retired(static_cast<cudaStream_t>(home));
    }

    inline std::array<uintptr_t, 2> bind_cuda_deferred_home(void* home) {
        const auto previous = reinterpret_cast<uintptr_t>(getCurrentCUDAStream());
        setCurrentCUDAStream(static_cast<cudaStream_t>(home));
        return {previous, 0};
    }

    inline void unbind_cuda_deferred_home(std::array<uintptr_t, 2> token) {
        setCurrentCUDAStream(reinterpret_cast<cudaStream_t>(token[0]));
    }
} // namespace lfs::core::internal
