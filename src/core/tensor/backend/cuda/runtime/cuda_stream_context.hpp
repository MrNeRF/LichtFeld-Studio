/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"
#include "core/tensor_cuda_interop.hpp"
#include <cuda_runtime.h>
namespace lfs::core {
    // Synchronous host<->device copy ordered against `stream`, the tensor's
    // home stream. cudaMemcpy runs on the legacy default stream, which a
    // non-blocking home stream neither waits for nor is waited on by: an
    // upload from pageable memory returns while its DMA is still in flight,
    // and a download may read before the home stream's producer finished.
    // Downloads make the legacy stream wait for `stream` first; uploads make
    // `stream` wait for the legacy stream afterwards. The host never waits on
    // `stream` itself, so a gated or capturing home stream cannot deadlock.
    LFS_LOCAL_SYMBOL cudaError_t memcpy_ordered(void* dst, const void* src, size_t bytes,
                                                cudaMemcpyKind kind, cudaStream_t stream);

} // namespace lfs::core
