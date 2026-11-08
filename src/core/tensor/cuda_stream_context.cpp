/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "internal/cuda_stream_context.hpp"
#include "core/cuda_error.hpp"
#include "core/pinned_memory_allocator.hpp"
#include "internal/cuda_event_pool.hpp"
#include "internal/stream_lifetime.hpp"
#include "internal/tensor_impl.hpp"

#include <cstring>
#include <format>

namespace lfs::core {

    static thread_local cudaStream_t tl_current_stream = nullptr;

    cudaStream_t getCurrentCUDAStream() {
        return tl_current_stream;
    }

    void setCurrentCUDAStream(cudaStream_t stream) {
        if (stream) {
            unretire_stream(stream);
        }
        tl_current_stream = stream;
    }

    void waitForCUDAStream(cudaStream_t execution_stream, cudaStream_t dependency_stream) {
        unretire_stream(execution_stream);
        unretire_stream(dependency_stream);
        if (dependency_stream == execution_stream) {
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

    cudaError_t memcpy_ordered(void* const dst, const void* const src, const size_t bytes,
                               const cudaMemcpyKind kind, const cudaStream_t stream) {
        if (bytes == 0)
            return cudaSuccess;
        // release_stream drains a home stream before retiring it. Preserve the
        // legacy-copy fallback without passing its destroyed handle to CUDA.
        if (kind == cudaMemcpyDeviceToDevice && !is_stream_retired(stream)) {
            return cudaMemcpyAsync(dst, src, bytes, kind, stream);
        }
        if (kind == cudaMemcpyHostToDevice && !is_stream_retired(stream)) {
            if (dst == nullptr || src == nullptr)
                return cudaErrorInvalidValue;
            cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
            const auto queried = cudaStreamIsCapturing(stream, &capture);
            if (queried != cudaSuccess)
                return queried;
            // A captured H2D node outlives this call, including on graph replay.
            // This raw-pointer API has no graph owner for its temporary staging.
            if (capture != cudaStreamCaptureStatusNone)
                return cudaErrorStreamCaptureUnsupported;
            auto& pinned = PinnedMemoryAllocator::instance();
            void* const staging = pinned.allocate(bytes);
            if (staging == nullptr)
                return cudaErrorMemoryAllocation;
            std::memcpy(staging, src, bytes);
            const auto status = cudaMemcpyAsync(dst, staging, bytes, kind, stream);
            pinned.deallocate(staging, stream); // Reuse waits for the DMA, including malloc fallback.
            return status;
        }
        if (stream != nullptr && kind == cudaMemcpyDeviceToHost) {
            bridgeStreams(stream, nullptr);
        }
        const cudaError_t status = cudaMemcpy(dst, src, bytes, kind);
        return status;
    }

    cudaStream_t prepare_inputs_for_stream(
        const std::initializer_list<const Tensor*> inputs,
        const std::optional<cudaStream_t> requested_stream) {
        // A deferred input enqueues its producer only when it materializes; do that
        // first so the ordering below covers it.
        for (const Tensor* input : inputs) {
            if (input != nullptr && input->is_valid() && input->is_deferred())
                (void)input->data_ptr();
        }
        cudaStream_t execution_stream = requested_stream.has_value()
                                            ? *requested_stream
                                            : getCurrentCUDAStream();
        if (!requested_stream.has_value() && execution_stream == nullptr) {
            for (const Tensor* input : inputs) {
                LFS_ASSERT_MSG(input != nullptr && input->is_valid(),
                               "stream preparation requires valid tensor inputs");
                if (input->device() == Device::CUDA) {
                    execution_stream = input->stream();
                    break;
                }
            }
        }

        for (const Tensor* input : inputs) {
            LFS_ASSERT_MSG(input != nullptr && input->is_valid(),
                           "stream preparation requires valid tensor inputs");
            if (input->device() == Device::CUDA) {
                input->sync_to_stream(execution_stream);
            }
        }
        return execution_stream;
    }

} // namespace lfs::core
