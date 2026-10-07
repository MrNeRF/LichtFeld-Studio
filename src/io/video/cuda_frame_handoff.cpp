/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "cuda_frame_handoff.hpp"

#include "core/include/core/logger.hpp"

#include <cuda.h>

#include <stdexcept>
#include <string>

namespace lfs::io::video {

    namespace {

        [[nodiscard]] std::string driverError(const CUresult result) {
            const char* description = nullptr;
            if (cuGetErrorString(result, &description) == CUDA_SUCCESS && description)
                return description;
            return "CUDA driver error " + std::to_string(static_cast<int>(result));
        }

        void synchronizeDecoderStream(const media::detail::CudaDecodeContext& decoder_context) {
            if (!decoder_context.context)
                throw std::runtime_error("CUDA video decoder context is unavailable");

            const CUresult push_result = cuCtxPushCurrent(static_cast<CUcontext>(decoder_context.context));
            if (push_result != CUDA_SUCCESS) {
                throw std::runtime_error(
                    "Failed to activate CUDA video decoder context: " +
                    driverError(push_result));
            }

            const CUresult sync_result = cuStreamSynchronize(static_cast<CUstream>(decoder_context.stream));
            CUcontext popped_context = nullptr;
            const CUresult pop_result = cuCtxPopCurrent(&popped_context);

            if (sync_result != CUDA_SUCCESS) {
                throw std::runtime_error(
                    "CUDA video decoder synchronization failed: " +
                    driverError(sync_result));
            }
            if (pop_result != CUDA_SUCCESS) {
                throw std::runtime_error(
                    "Failed to restore CUDA context after video decode: " +
                    driverError(pop_result));
            }
        }

    } // namespace

    CudaFrameHandoff::CudaFrameHandoff(
        const media::detail::CudaDecodeContext& decoder_context,
        const cudaStream_t consumer_stream)
        : consumer_stream_(consumer_stream) {
        synchronizeDecoderStream(decoder_context);
        active_ = true;
    }

    CudaFrameHandoff::~CudaFrameHandoff() {
        if (!active_)
            return;

        const cudaError_t result = cudaStreamSynchronize(consumer_stream_);
        if (result != cudaSuccess) {
            LOG_ERROR("CUDA video frame consumer cleanup failed: {}",
                      cudaGetErrorString(result));
        }
    }

    void CudaFrameHandoff::finish() {
        if (!active_)
            return;

        const cudaError_t result = cudaStreamSynchronize(consumer_stream_);
        if (result != cudaSuccess) {
            throw std::runtime_error(
                std::string("CUDA video frame consumer synchronization failed: ") +
                cudaGetErrorString(result));
        }
        active_ = false;
    }

} // namespace lfs::io::video
