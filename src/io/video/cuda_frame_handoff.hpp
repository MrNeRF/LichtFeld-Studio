/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "media/cuda_frame.hpp"

#include <cuda_runtime_api.h>

namespace lfs::io::video {

    // Orders the decoder's CUDA stream before the consumer's runtime work.
    // Media keeps decoded storage alive until this handoff has finished; the
    // host adapter never inspects or retains decoder objects.
    class CudaFrameHandoff {
    public:
        explicit CudaFrameHandoff(const media::detail::CudaDecodeContext& decoder_context,
                                  cudaStream_t consumer_stream = nullptr);
        ~CudaFrameHandoff();

        CudaFrameHandoff(const CudaFrameHandoff&) = delete;
        CudaFrameHandoff& operator=(const CudaFrameHandoff&) = delete;

        void finish();

    private:
        cudaStream_t consumer_stream_ = nullptr;
        bool active_ = false;
    };

} // namespace lfs::io::video
