/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/cuda_stream_context.hpp"
#include "core/cuda_error.hpp"
#include "internal/cuda_event_pool.hpp"

#include <format>

namespace lfs::core {

    static thread_local cudaStream_t tl_current_stream = nullptr;

    cudaStream_t getCurrentCUDAStream() {
        return tl_current_stream;
    }

    void setCurrentCUDAStream(cudaStream_t stream) {
        tl_current_stream = stream;
    }

    void waitForCUDAStream(cudaStream_t execution_stream, cudaStream_t dependency_stream) {
        if (dependency_stream == nullptr || dependency_stream == execution_stream) {
            return;
        }

        cudaError_t status = cudaErrorUnknown;
        if (cudaEvent_t ready = CudaEventPool::instance().acquire()) {
            const cudaError_t record_status = cudaEventRecord(ready, dependency_stream);
            status = record_status;
            if (record_status == cudaSuccess) {
                const cudaError_t wait_status =
                    cudaStreamWaitEvent(execution_stream, ready, 0);
                status = wait_status;
                if (wait_status != cudaSuccess) {
                    ensure_cuda_success(
                        wait_status, "cudaStreamWaitEvent(tensor dependency)",
                        std::format("dependency_stream={}, execution_stream={}; fallback=stream sync",
                                    static_cast<void*>(dependency_stream),
                                    static_cast<void*>(execution_stream)),
                        LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
                }
            } else {
                ensure_cuda_success(
                    record_status, "cudaEventRecord(tensor dependency)",
                    std::format("dependency_stream={}, execution_stream={}; fallback=stream sync",
                                static_cast<void*>(dependency_stream),
                                static_cast<void*>(execution_stream)),
                    LFS_SOURCE_SITE_CURRENT(), CudaFailureDisposition::LogOnly);
            }
            CudaEventPool::instance().release(ready);
        }

        if (status != cudaSuccess) {
            const cudaError_t sync_status = cudaStreamSynchronize(dependency_stream);
            if (sync_status != cudaSuccess) {
                LFS_ENSURE_CUDA_SUCCESS_MSG(
                    sync_status, "cudaStreamSynchronize(tensor dependency fallback)",
                    std::format("dependency_stream={}, execution_stream={}",
                                static_cast<void*>(dependency_stream),
                                static_cast<void*>(execution_stream)));
            }
        }
    }

} // namespace lfs::core
