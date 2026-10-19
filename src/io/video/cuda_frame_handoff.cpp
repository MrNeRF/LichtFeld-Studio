/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "cuda_frame_handoff.hpp"

#include "core/include/core/logger.hpp"

extern "C" {
#include <libavutil/hwcontext.h>
}

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

        void synchronizeDecoderStream(const AVCUDADeviceContext& decoder_context) {
            if (!decoder_context.cuda_ctx)
                throw std::runtime_error("CUDA video decoder context is unavailable");

            const CUresult push_result = cuCtxPushCurrent(decoder_context.cuda_ctx);
            if (push_result != CUDA_SUCCESS) {
                throw std::runtime_error(
                    "Failed to activate CUDA video decoder context: " +
                    driverError(push_result));
            }

            const CUresult sync_result = cuStreamSynchronize(decoder_context.stream);
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

        [[nodiscard]] const AVCUDADeviceContext& decoderContextForFrame(
            const AVFrame* const frame) {
            if (!frame || !frame->hw_frames_ctx)
                throw std::runtime_error("CUDA video frame has no hardware frame context");

            const auto* const frames_context =
                reinterpret_cast<const AVHWFramesContext*>(frame->hw_frames_ctx->data);
            if (!frames_context || !frames_context->device_ctx ||
                !frames_context->device_ctx->hwctx) {
                throw std::runtime_error("CUDA video frame has no decoder device context");
            }

            return *reinterpret_cast<const AVCUDADeviceContext*>(
                frames_context->device_ctx->hwctx);
        }

    } // namespace

    CudaFrameHandoff::CudaFrameHandoff(const AVFrame* const frame,
                                       const cudaStream_t consumer_stream)
        : CudaFrameHandoff(decoderContextForFrame(frame), consumer_stream) {}

    CudaFrameHandoff::CudaFrameHandoff(
        const AVCUDADeviceContext& decoder_context,
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
