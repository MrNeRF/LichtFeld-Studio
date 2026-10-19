/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/hwcontext_cuda.h>
}

#include <cuda_runtime_api.h>

namespace lfs::io::video {

    // Orders work produced by FFmpeg's CUDA stream before work submitted through
    // the CUDA runtime, and keeps the AVFrame's storage alive until that runtime
    // work has completed.
    class CudaFrameHandoff {
    public:
        explicit CudaFrameHandoff(const AVFrame* frame,
                                  cudaStream_t consumer_stream = nullptr);
        CudaFrameHandoff(const AVCUDADeviceContext& decoder_context,
                         cudaStream_t consumer_stream);
        ~CudaFrameHandoff();

        CudaFrameHandoff(const CudaFrameHandoff&) = delete;
        CudaFrameHandoff& operator=(const CudaFrameHandoff&) = delete;

        void finish();

    private:
        cudaStream_t consumer_stream_ = nullptr;
        bool active_ = false;
    };

} // namespace lfs::io::video
