// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "linear_frame.hpp"
extern "C" {
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}
#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>

namespace lfs::media::detail {
    namespace {
        Error unsupported(std::string text) {
            return make_error({.code = ErrorCode::Unsupported, .domain = ErrorDomain::IO, .detail = std::move(text), .detection = LFS_SOURCE_SITE_CURRENT()});
        }
        ColorTransfer transfer(AVColorTransferCharacteristic tag) {
            switch (tag) {
            case AVCOL_TRC_LINEAR: return ColorTransfer::Linear;
            case AVCOL_TRC_IEC61966_2_1: return ColorTransfer::Srgb;
            case AVCOL_TRC_BT709: return ColorTransfer::Bt709;
            default: return ColorTransfer::Unspecified;
            }
        }
        ColorPrimaries primaries(AVColorPrimaries tag) {
            switch (tag) {
            case AVCOL_PRI_BT709: return ColorPrimaries::Bt709;
            case AVCOL_PRI_BT2020: return ColorPrimaries::Bt2020;
            default: return ColorPrimaries::Unspecified;
            }
        }
        float inverse(float v, ColorTransfer curve) {
            if (curve == ColorTransfer::Srgb)
                return v <= .04045f ? v / 12.92f : std::pow((v + .055f) / 1.055f, 2.4f);
            if (curve == ColorTransfer::Bt709)
                return v < .081f ? v / 4.5f : std::pow((v + .099f) / 1.099f, 1.f / .45f);
            return v;
        }
    } // namespace
    LinearFrameConverter::LinearFrameConverter(const AVCodecParameters& source, FrameColor overrides) : overrides_(overrides) {
        // Copy only metadata used below, never pointer-owning codec parameters.
        source_.color_trc = source.color_trc;
        source_.color_primaries = source.color_primaries;
        source_.color_space = source.color_space;
        source_.color_range = source.color_range;
    }
    LinearFrameConverter::~LinearFrameConverter() { sws_freeContext(context_); }
    Result<FrameColor> LinearFrameConverter::outputColor() const {
        const auto color = overrides_.primaries != ColorPrimaries::Unspecified ? overrides_.primaries : primaries(source_.color_primaries);
        if (color == ColorPrimaries::Unspecified)
            return unsupported(std::format("Float SDR requires declared BT709 or BT2020 primaries (source={}, override={}); supply an explicit input override if metadata is missing", int(source_.color_primaries), int(overrides_.primaries)));
        return FrameColor{ColorTransfer::Linear, color, AlphaMode::None};
    }
    SinkResult LinearFrameConverter::convert(const AVFrame& frame, int width, int height, uint8_t* destination) {
        const auto* desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(frame.format));
        const auto unsupported = [&](std::string reason) {
            return make_error({.code = ErrorCode::Unsupported, .domain = ErrorDomain::IO, .detail = std::format("{} (format={}, width={}, height={}, transfer={}, primaries={}, matrix={}, range={})", reason, desc ? desc->name : "unknown", frame.width, frame.height, int(frame.color_trc), int(frame.color_primaries), int(frame.colorspace), int(frame.color_range)), .detection = LFS_SOURCE_SITE_CURRENT()});
        };
        if (!desc || (desc->flags & (AV_PIX_FMT_FLAG_HWACCEL | AV_PIX_FMT_FLAG_FLOAT | AV_PIX_FMT_FLAG_ALPHA)) || desc->nb_components < 3 ||
            desc->comp[0].depth > 16)
            return SinkResult::failure(unsupported("Float SDR currently supports integer RGB/YUV video up to 16 bits without alpha"));
        const auto tag = frame.color_trc != AVCOL_TRC_UNSPECIFIED ? frame.color_trc : source_.color_trc;
        if (tag == AVCOL_TRC_SMPTE2084 || tag == AVCOL_TRC_ARIB_STD_B67 || tag == AVCOL_TRC_LOG || tag == AVCOL_TRC_LOG_SQRT || av_frame_get_side_data(&frame, AV_FRAME_DATA_DOVI_METADATA))
            return SinkResult::failure(unsupported("PQ, HLG, logarithmic transfer and Dolby Vision are outside the float SDR profile"));
        const auto curve = overrides_.transfer != ColorTransfer::Unspecified ? overrides_.transfer : transfer(tag);
        if (curve == ColorTransfer::Unspecified)
            return SinkResult::failure(unsupported("Float SDR requires Linear, sRGB or BT709 transfer metadata or an explicit input override"));
        const auto frame_primaries = overrides_.primaries != ColorPrimaries::Unspecified ? overrides_.primaries : primaries(frame.color_primaries != AVCOL_PRI_UNSPECIFIED ? frame.color_primaries : source_.color_primaries);
        auto color = outputColor();
        if (!color || frame_primaries != color->primaries)
            return SinkResult::failure(unsupported("Float SDR primaries changed or are unsupported"));
        if (transfer_ != ColorTransfer::Unspecified && curve != transfer_)
            return SinkResult::failure(unsupported("Float SDR transfer changed during extraction"));
        if (transfer_ == ColorTransfer::Unspecified) {
            transfer_ = curve;
            for (size_t i = 0; i < lut_.size(); ++i)
                lut_[i] = inverse(static_cast<float>(i) / 65535.f, curve);
        }
        if (frame.width <= 0 || frame.height <= 0 || width <= 0 || height <= 0 ||
            frame.width > (std::numeric_limits<int>::max() - 64) / 6 ||
            ((static_cast<uint64_t>(frame.width) * 6 + 63) & ~uint64_t{63}) * frame.height + 64 > 256ULL * 1024 * 1024)
            return SinkResult::failure(make_error({.code = ErrorCode::ResourceExhausted, .domain = ErrorDomain::IO, .detail = "Float SDR RGB48 working buffer exceeds 256 MiB or codec stride limits", .detection = LFS_SOURCE_SITE_CURRENT()}));
        const bool rgb = (desc->flags & AV_PIX_FMT_FLAG_RGB) != 0;
        const auto matrix_tag = frame.colorspace != AVCOL_SPC_UNSPECIFIED ? frame.colorspace : source_.color_space;
        const auto range_tag = frame.color_range != AVCOL_RANGE_UNSPECIFIED ? frame.color_range : source_.color_range;
        int matrix = SWS_CS_ITU709;
        if (!rgb) {
            switch (matrix_tag) {
            case AVCOL_SPC_BT709: matrix = SWS_CS_ITU709; break;
            case AVCOL_SPC_BT470BG:
            case AVCOL_SPC_SMPTE170M: matrix = SWS_CS_ITU601; break;
            case AVCOL_SPC_BT2020_NCL: matrix = SWS_CS_BT2020; break;
            default: return SinkResult::failure(unsupported("Float YUV SDR requires a supported explicit matrix (BT709, BT601 or BT2020 NCL)"));
            }
            if (range_tag != AVCOL_RANGE_JPEG && range_tag != AVCOL_RANGE_MPEG)
                return SinkResult::failure(unsupported("Float YUV SDR requires explicit full/limited range"));
        } else if (range_tag == AVCOL_RANGE_MPEG) {
            return SinkResult::failure(unsupported("Float RGB SDR does not support limited-range RGB"));
        }
        if (!rgb) {
            // Float matrix conversion avoids RGB48 clipping of superwhites and
            // out-of-gamut values. Subsampled chroma needs a separate qualified path.
            if (!(desc->flags & AV_PIX_FMT_FLAG_PLANAR) || desc->log2_chroma_w || desc->log2_chroma_h ||
                (desc->comp[0].depth != 8 && desc->comp[0].depth != 10 && desc->comp[0].depth != 12 && desc->comp[0].depth != 16) || desc->comp[1].depth != desc->comp[0].depth || desc->comp[2].depth != desc->comp[0].depth)
                return SinkResult::failure(unsupported("Float SDR currently supports RGB or planar YUV444 (8-16 bit); subsampled YUV requires a future precision-preserving chroma path"));
            for (int c = 0; c < 3; ++c)
                if (!frame.data[desc->comp[c].plane] || desc->comp[c].step > 2 || desc->comp[c].step < 1 ||
                    std::abs(int64_t(frame.linesize[desc->comp[c].plane])) < int64_t(frame.width) * desc->comp[c].step)
                    return SinkResult::failure(unsupported("Unsupported float YUV444 plane layout"));
            const float kr = matrix == SWS_CS_ITU709 ? .2126f : matrix == SWS_CS_BT2020 ? .2627f
                                                                                        : .299f;
            const float kb = matrix == SWS_CS_ITU709 ? .0722f : matrix == SWS_CS_BT2020 ? .0593f
                                                                                        : .114f;
            const float scale = std::ldexp(1.f, desc->comp[0].depth - 8);
            const float maximum = static_cast<float>((1u << desc->comp[0].depth) - 1u);
            const auto component = [&](int x, int y, int c) {
                const auto& comp = desc->comp[c];
                const auto* p = frame.data[comp.plane] + static_cast<ptrdiff_t>(y) * frame.linesize[comp.plane] + static_cast<ptrdiff_t>(x) * comp.step + comp.offset;
                uint32_t value = p[0];
                if (comp.depth + comp.shift > 8)
                    value = desc->flags & AV_PIX_FMT_FLAG_BE ? (value << 8) | p[1] : value | static_cast<uint32_t>(p[1]) << 8;
                return static_cast<float>((value >> comp.shift) & ((1u << comp.depth) - 1u));
            };
            const auto linearize = [&](float v) {
                if (curve == ColorTransfer::Linear)
                    return v;
                if (v < 0.f || v > 1.f)
                    return inverse(v, curve);
                const float i = v * 65535.f;
                const auto lo = static_cast<size_t>(i);
                return std::lerp(lut_[lo], lut_[std::min(lo + 1, lut_.size() - 1)], i - static_cast<float>(lo));
            };
            const auto sample = [&](int x, int y) {
                const float yy = range_tag == AVCOL_RANGE_MPEG ? (component(x, y, 0) - 16.f * scale) / (219.f * scale) : component(x, y, 0) / maximum;
                const float cb = (component(x, y, 1) - 128.f * scale) / (range_tag == AVCOL_RANGE_MPEG ? 224.f * scale : maximum);
                const float cr = (component(x, y, 2) - 128.f * scale) / (range_tag == AVCOL_RANGE_MPEG ? 224.f * scale : maximum);
                return std::array<float, 3>{linearize(yy + 2.f * (1.f - kr) * cr),
                                            linearize(yy - 2.f * kb * (1.f - kb) / (1.f - kr - kb) * cb - 2.f * kr * (1.f - kr) / (1.f - kr - kb) * cr),
                                            linearize(yy + 2.f * (1.f - kb) * cb)};
            };
            for (int y = 0; y < height; ++y) {
                const float sy = std::clamp((y + .5f) * frame.height / height - .5f, 0.f, static_cast<float>(frame.height - 1));
                const int y0 = static_cast<int>(sy), y1 = std::min(y0 + 1, frame.height - 1);
                for (int x = 0; x < width; ++x) {
                    std::array<float, 3> value{};
                    if (width == frame.width && height == frame.height)
                        value = sample(x, y);
                    else {
                        const float sx = std::clamp((x + .5f) * frame.width / width - .5f, 0.f, static_cast<float>(frame.width - 1));
                        const int x0 = static_cast<int>(sx), x1 = std::min(x0 + 1, frame.width - 1);
                        const auto a = sample(x0, y0), b = sample(x1, y0), c = sample(x0, y1), d = sample(x1, y1);
                        for (int i = 0; i < 3; ++i)
                            value[i] = std::lerp(std::lerp(a[i], b[i], sx - x0), std::lerp(c[i], d[i], sx - x0), sy - y0);
                    }
                    std::memcpy(destination + (static_cast<size_t>(y) * width + x) * 12, value.data(), 12);
                }
            }
            return {};
        }
        if (!context_ || frame.width != cached_width_ || frame.height != cached_height_ || frame.format != cached_format_) {
            sws_freeContext(context_);
            context_ = sws_alloc_context();
            if (!context_)
                return SinkResult::failure(make_error({.code = ErrorCode::ResourceExhausted, .domain = ErrorDomain::IO, .detail = "Cannot allocate float SDR conversion context", .detection = LFS_SOURCE_SITE_CURRENT()}));
            const auto option = [&](const char* name, int value) { return av_opt_set_int(context_, name, value, 0) >= 0; };
            bool valid = option("srcw", frame.width) && option("srch", frame.height) && option("src_format", frame.format) &&
                         option("dstw", frame.width) && option("dsth", frame.height) && option("dst_format", AV_PIX_FMT_RGB48LE) &&
                         option("sws_flags", SWS_BILINEAR | SWS_ACCURATE_RND);
            valid = valid && sws_init_context(context_, nullptr, nullptr) >= 0;
            if (!valid)
                return SinkResult::failure(unsupported("Cannot configure float SDR conversion for this input layout"));
            cached_width_ = frame.width;
            cached_height_ = frame.height;
            cached_format_ = frame.format;
        }
        const auto rgb_stride = (static_cast<size_t>(frame.width) * 6 + 63) & ~size_t{63};
        // SIMD-safe row padding and tail; visible samples remain exactly source width.
        rgb48_.resize(rgb_stride * frame.height + 64);
        uint8_t* planes[4]{rgb48_.data(), nullptr, nullptr, nullptr};
        int strides[4]{static_cast<int>(rgb_stride), 0, 0, 0};
        if (sws_scale(context_, frame.data, frame.linesize, 0, frame.height, planes, strides) != frame.height)
            return SinkResult::failure(make_error({.code = ErrorCode::Unavailable, .domain = ErrorDomain::IO, .detail = "Float SDR conversion failed", .detection = LFS_SOURCE_SITE_CURRENT()}));
        const auto sample = [&](int x, int y, int c) {
            const auto i = static_cast<size_t>(y) * rgb_stride + static_cast<size_t>(x) * 6 + c * 2;
            const auto value = static_cast<uint16_t>(rgb48_[i] | static_cast<uint16_t>(rgb48_[i + 1]) << 8);
            return lut_[value];
        };
        for (int y = 0; y < height; ++y) {
            const float sy = std::clamp((y + .5f) * frame.height / height - .5f, 0.f, static_cast<float>(frame.height - 1));
            const int y0 = static_cast<int>(sy), y1 = std::min(y0 + 1, frame.height - 1);
            for (int x = 0; x < width; ++x) {
                std::array<float, 3> value{};
                if (width == frame.width && height == frame.height) {
                    for (int c = 0; c < 3; ++c)
                        value[c] = sample(x, y, c);
                } else {
                    const float sx = std::clamp((x + .5f) * frame.width / width - .5f, 0.f, static_cast<float>(frame.width - 1));
                    const int x0 = static_cast<int>(sx), x1 = std::min(x0 + 1, frame.width - 1);
                    for (int c = 0; c < 3; ++c)
                        value[c] = std::lerp(std::lerp(sample(x0, y0, c), sample(x1, y0, c), sx - x0),
                                             std::lerp(sample(x0, y1, c), sample(x1, y1, c), sx - x0), sy - y0);
                }
                std::memcpy(destination + (static_cast<size_t>(y) * width + x) * 12, value.data(), 12);
            }
        }
        return {};
    }
} // namespace lfs::media::detail
