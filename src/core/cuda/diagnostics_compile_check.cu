/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "core/tensor/internal/cuda_memory_guard.hpp"
#include "core/tensor/internal/gpu_config.hpp"
#include "core/tensor/internal/gpu_slab_allocator.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "core/tensor/internal/size_bucketed_pool.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include "memory_arena.hpp"

#include <cuda_runtime.h>

namespace lfs::core::compile_check {

    // This function is intentionally never run, but external linkage ensures
    // its host body is emitted. It mirrors diagnostics calls made from CUDA
    // translation units so nvcc must codegen every caller-site capture macro.
    void compile_cuda_host_diagnostics(void* device_pointer) {
        int device = -1;
        LFS_CUDA_CHECK(cudaGetDevice(&device));
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "compile-check device={}", device);
        LFS_ASSERT(device >= -1);
        LFS_ASSERT_MSG(device_pointer != nullptr, "compile-check pointer");
        LOG_DEBUG("CUDA diagnostics compile-check device=%d", device);

        LFS_ENSURE_CUDA_SUCCESS(cudaSuccess, "compile-check status");
        LFS_ENSURE_CUDA_SUCCESS_MSG(
            cudaSuccess, "compile-check status with context", "context=nvcc host path");
        const CudaCheckState state{};
        LFS_ENSURE_CUDA_SUCCESS_STATE(
            cudaSuccess, state, "compile-check stateful status", "context=nvcc host path");
        LFS_VALIDATE_CUDA_DEVICE_POINTER(device_pointer, "compile-check pointer");
        LFS_VALIDATE_CUDA_DEVICE_POINTER_OPTIONAL(nullptr, "compile-check optional pointer");

        static_assert(sizeof(RasterizerMemoryArena) > 0);
        static_assert(sizeof(CudaDeviceMemory<int>) > 0);
    }

} // namespace lfs::core::compile_check
