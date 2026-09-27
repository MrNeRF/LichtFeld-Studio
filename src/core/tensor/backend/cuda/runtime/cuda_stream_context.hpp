/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"
#include "core/tensor_cuda_interop.hpp"
#include <cuda_runtime.h>
namespace lfs::core {
    // Synchronous host<->device copy ordered on `stream`. The host waits only
    // for that stream, so a gated or capturing home stream cannot deadlock on
    // unrelated default-stream work.
    LFS_LOCAL_SYMBOL cudaError_t memcpy_ordered(void* dst, const void* src, size_t bytes,
                                                cudaMemcpyKind kind, cudaStream_t stream);

} // namespace lfs::core
