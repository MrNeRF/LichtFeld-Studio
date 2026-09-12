/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <cuda_runtime.h>

namespace lfs::core {

    // Record `stream` as retired ahead of destruction. bridgeStreams skips
    // retired handles instead of probing the driver (a destroyed handle
    // segfaults libcuda on Linux).
    LFS_CORE_API void retire_stream(cudaStream_t stream) noexcept;

    // Remove `stream` from the retired set. Called wherever a caller-live
    // handle enters the API, because the driver reuses pointer values.
    LFS_CORE_API void unretire_stream(cudaStream_t stream) noexcept;

    LFS_CORE_API bool is_stream_retired(cudaStream_t stream) noexcept;

} // namespace lfs::core
