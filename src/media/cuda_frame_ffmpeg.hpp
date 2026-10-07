// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "core/error.hpp"
#include "media/cuda_frame.hpp"
extern "C" {
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
}
#if LFS_HAS_CUDA
extern "C" {
#include <libavutil/hwcontext_cuda.h>
}
#endif

namespace lfs::media::detail {
    // Only the provider interprets the decoder's native frame/device structures.
    inline Result<CudaVideoFrame> cudaFrameView(const AVFrame* frame) {
        auto unavailable = [](std::string text) -> Result<CudaVideoFrame> {
            return make_error({.code = ErrorCode::FailedPrecondition, .domain = ErrorDomain::IO, .detail = std::move(text), .detection = LFS_SOURCE_SITE_CURRENT()});
        };
#if LFS_HAS_CUDA
        if (!frame || frame->format != AV_PIX_FMT_CUDA || !frame->hw_frames_ctx)
            return unavailable("CUDA video frame has no hardware frame context");
        const auto* frames = reinterpret_cast<const AVHWFramesContext*>(frame->hw_frames_ctx->data);
        if (!frames || !frames->device_ctx || !frames->device_ctx->hwctx)
            return unavailable("CUDA video frame has no decoder device context");
        if (frames->format != AV_PIX_FMT_CUDA || frames->sw_format != AV_PIX_FMT_NV12 ||
            frames->device_ctx->type != AV_HWDEVICE_TYPE_CUDA)
            return unavailable("CUDA JPEG handoff requires a decoded NV12 CUDA frame");
        const auto* decoder = reinterpret_cast<const AVCUDADeviceContext*>(frames->device_ctx->hwctx);
        if (!decoder->cuda_ctx || frame->width <= 0 || frame->height <= 0 ||
            !frame->data[0] || !frame->data[1] ||
            frame->linesize[0] < frame->width || frame->linesize[1] < frame->width)
            return unavailable("CUDA video frame has invalid decoder plane storage");
        return CudaVideoFrame{{decoder->cuda_ctx, decoder->stream}, {frame->data[0], frame->data[1]}, {frame->linesize[0], frame->linesize[1]}};
#else
        return unavailable("CUDA video frames are unavailable in this build");
#endif
    }
} // namespace lfs::media::detail
