/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <cuda_runtime.h>

namespace lfs::core {

    // Records that `stream` is about to be destroyed. Handles later seen in
    // bridgeStreams are skipped instead of being passed to the driver
    // (a destroyed handle segfaults libcuda on Linux).
    LFS_CORE_API void note_stream_retired(cudaStream_t stream) noexcept;

    // Removes `stream` from the retired set. Called when a handle value
    // re-enters live circulation (the driver reuses heap pointers, so a new
    // cudaStreamCreate can return a previously retired value).
    LFS_CORE_API void note_stream_reused(cudaStream_t stream) noexcept;

    LFS_CORE_API bool is_stream_retired(cudaStream_t stream) noexcept;

} // namespace lfs::core
