/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "core/tensor/internal/cuda_memory_guard.hpp"
#include "core/tensor/internal/gpu_config.hpp"
#include "core/tensor/internal/gpu_slab_allocator.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "core/tensor/internal/size_bucketed_pool.hpp"
#include "core/tensor/internal/tensor_impl.hpp"
#include "core/tensor_trace.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "memory_arena.hpp"

#include <cuda_runtime.h>

namespace lfs::core::compile_check {

    namespace {

        void compile_cuda_logger_arities(const int device) {
            LOG_TRACE("trace zero");
            LOG_TRACE("trace one=%d", device);
            LOG_TRACE("trace many=%d/%d/%d", device, 1, 2);
            LOG_DEBUG("debug zero");
            LOG_DEBUG("debug one=%d", device);
            LOG_DEBUG("debug many=%d/%d/%d", device, 1, 2);
            LOG_INFO("info zero");
            LOG_INFO("info one=%d", device);
            LOG_INFO("info many=%d/%d/%d", device, 1, 2);
            LOG_PERF("perf zero");
            LOG_PERF("perf one=%d", device);
            LOG_PERF("perf many=%d/%d/%d", device, 1, 2);
            LOG_WARN("warn zero");
            LOG_WARN("warn one=%d", device);
            LOG_WARN("warn many=%d/%d/%d", device, 1, 2);
            LOG_ERROR("error zero");
            LOG_ERROR("error one=%d", device);
            LOG_ERROR("error many=%d/%d/%d", device, 1, 2);
            LOG_CRITICAL("critical zero");
            LOG_CRITICAL("critical one=%d", device);
            LOG_CRITICAL("critical many=%d/%d/%d", device, 1, 2);
        }

        void compile_cuda_macro_surface(void* device_pointer,
                                        const int device) {
            LFS_ASSERT(device >= -1);
            LFS_ASSERT_MSG(device_pointer != nullptr, "compile-check pointer");
            LFS_DEBUG_ASSERT(device >= -1);
            LFS_DEBUG_ASSERT_MSG(device_pointer != nullptr, "compile-check debug pointer");

            LOG_TIMER("compile-check timer");
            LOG_TIMER_THRESHOLD("compile-check threshold timer", 0.0);
            LOG_TIMER_TRACE("compile-check trace timer");
            LOG_TIMER_DEBUG("compile-check debug timer");

            const TensorShape shape{1};
            {
                TRACE_OP("compile-check trace", shape);
                TRACE_OP_OUTPUT(shape);
            }
            {
                TRACE_OP2("compile-check trace pair", shape, shape);
                TRACE_OP_OUTPUT(shape);
            }
            TRACE_PRINT_STACK();
            TRACE_PRINT_HISTORY(1);

            LFS_VRAM_SCOPE("compile-check VRAM scope");
            LOG_VRAM_DIFF("compile-check VRAM delta");
            LFS_TRACE("compile-check CPU trace");
            LFS_GPU_TIME("compile-check GPU timer", static_cast<cudaStream_t>(nullptr));
            LFS_GAUGE("compile-check.gauge", device);
            LFS_COUNTER_ADD("compile-check.iter", 1);
            LFS_COUNTER_TOTAL_ADD("compile-check.total", 1);
            LFS_HIST("compile-check.histogram", device);

            LFS_CUDA_BREADCRUMB("compile-check breadcrumb");
            LFS_CUDA_BREADCRUMB_STREAM(
                "compile-check stream breadcrumb", static_cast<cudaStream_t>(nullptr));
        }

    } // namespace

    // This function is intentionally never run, but external linkage ensures
    // its host body is emitted. It mirrors diagnostics calls made from CUDA
    // translation units so nvcc must codegen every caller-site capture macro.
    void compile_cuda_host_diagnostics(void* device_pointer) {
        int device = -1;
        LFS_CUDA_CHECK(cudaGetDevice(&device));
        LFS_CUDA_CHECK_MSG(cudaSuccess, "compile-check zero arguments");
        LFS_CUDA_CHECK_MSG(cudaSuccess, "compile-check device={}", device);
        LFS_CUDA_CHECK_MSG(
            cudaGetLastError(), "compile-check many={}/{}/{}", device, 1, 2);

        compile_cuda_logger_arities(device);
        compile_cuda_macro_surface(device_pointer, device);

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
