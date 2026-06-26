/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/cuda_stream_context.hpp"
#include "internal/cuda_event_pool.hpp"

#include <stdexcept>
#include <string>

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
            status = cudaEventRecord(ready, dependency_stream);
            if (status == cudaSuccess) {
                status = cudaStreamWaitEvent(execution_stream, ready, 0);
            }
            CudaEventPool::instance().release(ready);
        }

        if (status != cudaSuccess) {
            const cudaError_t sync_status = cudaStreamSynchronize(dependency_stream);
            if (sync_status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("Failed to synchronize CUDA streams: ") +
                    cudaGetErrorString(sync_status));
            }
        }
    }

} // namespace lfs::core
