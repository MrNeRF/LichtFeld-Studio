// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "core/error.hpp"
#include "media/cuda_frame.hpp"
#include "video_color.hpp"
#include <format>
#include <optional>
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
    inline std::optional<ColorMatrix> cudaFrameColorMatrix(const AVFrame* frame, AVColorSpace fallback) {
        const auto matrix = frame->colorspace == AVCOL_SPC_UNSPECIFIED ? fallback : frame->colorspace;
        if (matrix == AVCOL_SPC_BT709)
            return ColorMatrix::Bt709;
        if (matrix == AVCOL_SPC_BT2020_NCL)
            return ColorMatrix::Bt2020Ncl;
        if (matrix == AVCOL_SPC_BT2020_CL)
            return std::nullopt;
        // Match configureVideoToRgbColorimetry's SWS_CS_DEFAULT fallback.
        // NV12 retains its original integer BT.601 kernel for these matrices.
        return ColorMatrix::Smpte170M;
    }
    inline bool cudaFrameColorSupported(const AVFrame* frame, AVColorSpace fallback) {
        return frame && cudaFrameColorMatrix(frame, fallback).has_value() &&
               (frame->color_range == AVCOL_RANGE_UNSPECIFIED || frame->color_range == AVCOL_RANGE_MPEG || frame->color_range == AVCOL_RANGE_JPEG);
    }
    // Only the provider interprets the decoder's native frame/device structures.
    inline Result<CudaVideoFrame> cudaFrameView(const AVFrame* frame, const bool source_full_range = false, const AVColorSpace fallback_matrix = AVCOL_SPC_UNSPECIFIED) {
        auto unavailable = [](std::string text) -> Result<CudaVideoFrame> {
            return make_error({.code = ErrorCode::FailedPrecondition, .domain = ErrorDomain::IO, .detail = std::move(text), .detection = LFS_SOURCE_SITE_CURRENT()});
        };
#if LFS_HAS_CUDA
        if (!frame || frame->format != AV_PIX_FMT_CUDA || !frame->hw_frames_ctx)
            return unavailable(std::format("CUDA video frame requires a hardware frames context (frame_present={}, format={}, frames_context_present={})", frame != nullptr, frame ? frame->format : -1, frame && frame->hw_frames_ctx));
        const auto* frames = reinterpret_cast<const AVHWFramesContext*>(frame->hw_frames_ctx->data);
        if (!frames || !frames->device_ctx || !frames->device_ctx->hwctx)
            return unavailable(std::format("CUDA video frame requires a decoder device context (frames_present={}, device_present={}, hwctx_present={})", frames != nullptr, frames && frames->device_ctx, frames && frames->device_ctx && frames->device_ctx->hwctx));
        if (frames->format != AV_PIX_FMT_CUDA || (frames->sw_format != AV_PIX_FMT_NV12 && frames->sw_format != AV_PIX_FMT_P010LE) ||
            frames->device_ctx->type != AV_HWDEVICE_TYPE_CUDA)
            return unavailable(std::format("CUDA JPEG handoff requires a decoded NV12/P010 CUDA frame (format={}, sw_format={}, device_type={})", static_cast<int>(frames->format), static_cast<int>(frames->sw_format), static_cast<int>(frames->device_ctx->type)));
        const bool p010 = frames->sw_format == AV_PIX_FMT_P010LE;
        const int bytes = p010 ? 2 : 1;
        const auto* decoder = reinterpret_cast<const AVCUDADeviceContext*>(frames->device_ctx->hwctx);
        if (!decoder->cuda_ctx || frame->width <= 0 || frame->height <= 0 ||
            !frame->data[0] || !frame->data[1] ||
            frame->linesize[0] < int64_t(frame->width) * bytes ||
            frame->linesize[1] < ((int64_t(frame->width) + 1) / 2) * 2 * bytes ||
            (p010 && ((frame->linesize[0] | frame->linesize[1]) & 1)))
            return unavailable(std::format("CUDA video frame has invalid decoder plane storage (format={}, width={}, height={}, pitches=[{}, {}], planes_present=[{}, {}], context_present={})", frame->format, frame->width, frame->height, frame->linesize[0], frame->linesize[1], frame->data[0] != nullptr, frame->data[1] != nullptr, decoder->cuda_ctx != nullptr));
        const auto resolved = cudaFrameColorMatrix(frame, fallback_matrix);
        if (!cudaFrameColorSupported(frame, fallback_matrix))
            return unavailable(std::format("CUDA JPEG color conversion does not support matrix/range [{}, {}]", static_cast<int>(frame->colorspace), static_cast<int>(frame->color_range)));
        CudaVideoFrame view{{decoder->cuda_ctx, decoder->stream}, {frame->data[0], frame->data[1]}, {frame->linesize[0], frame->linesize[1]}, frame->color_range == AVCOL_RANGE_JPEG || (frame->color_range == AVCOL_RANGE_UNSPECIFIED && source_full_range)};
        // The host swscale path does not configure chroma position either.
        // Preserve its sampling policy regardless of the signalled location.
        const auto chroma = chromaMapping(ChromaLocation::Left, 1, 1);
        view.color.chroma_y_scale = chroma[2];
        view.color.chroma_y_offset = chroma[3];
        view.layout = p010 ? CudaVideoLayout::P010 : CudaVideoLayout::NV12;
        view.legacy_bt601 = !p010 && *resolved == ColorMatrix::Smpte170M;
        std::array<float, 3> coefficients{};
        yuvLumaCoefficients(*resolved, frame->width, frame->height, coefficients);
        const auto decode = yuvDecodeMatrix(coefficients);
        const double maximum = p010 ? 1023.0 : 255.0;
        const auto levels = yuvDecodeLevels(view.full_range, (maximum + 1) / maximum, false);
        for (int channel = 0; channel < 3; ++channel) {
            view.color.scale[channel] = static_cast<float>(levels.multiplier[channel] / maximum);
            view.color.bias[channel] = static_cast<float>(levels.black[channel] * levels.multiplier[channel]);
            std::copy(decode[channel].begin(), decode[channel].end(), view.color.decode + channel * 3);
        }
        return view;
#else
        return unavailable("CUDA video frames are unavailable in this build");
#endif
    }
} // namespace lfs::media::detail
