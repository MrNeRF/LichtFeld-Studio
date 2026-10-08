// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "linear_frame.hpp"
#include "decoded_video_frame_ffmpeg.hpp"
#include "media/media_options.hpp"
#include "media_backends.hpp"
#include <cstring>
#include <format>
#include <limits>

namespace lfs::media {
    namespace {
        Error unsupported(std::string text) {
            return make_error({.code = ErrorCode::Unsupported, .domain = ErrorDomain::IO, .detail = std::move(text), .detection = LFS_SOURCE_SITE_CURRENT()});
        }
        SinkResult validateProfile(const AVPixFmtDescriptor* desc, ColorTransfer transfer, ColorPrimaries primaries,
                                   ColorMatrix matrix, ColorRange range, FrameColor overrides) {
            const auto fail = [&](std::string_view reason) {
                const auto name = [](const char* text) { return text ? text : "unknown"; };
                return SinkResult::failure(unsupported(std::format("{} (format={}, transfer={}, primaries={}, matrix={}, range={})", reason,
                                                                   desc ? desc->name : "unknown", name(av_color_transfer_name(static_cast<AVColorTransferCharacteristic>(transfer))),
                                                                   name(av_color_primaries_name(static_cast<AVColorPrimaries>(primaries))),
                                                                   name(av_color_space_name(static_cast<AVColorSpace>(matrix))), name(av_color_range_name(static_cast<AVColorRange>(range))))));
            };
            if (!desc || desc->nb_components != 3 || (desc->flags & (AV_PIX_FMT_FLAG_HWACCEL | AV_PIX_FMT_FLAG_FLOAT | AV_PIX_FMT_FLAG_ALPHA | AV_PIX_FMT_FLAG_PAL | AV_PIX_FMT_FLAG_BITSTREAM | AV_PIX_FMT_FLAG_BAYER)) || std::string_view(desc->name).starts_with("xyz"))
                return fail("Linear EXR requires integer RGB/YUV without alpha");
            for (int i = 0; i < 3; ++i)
                if (desc->comp[i].depth < 8 || desc->comp[i].depth > 16 || desc->comp[i].depth + desc->comp[i].shift > 32 || desc->comp[i].step <= 0)
                    return fail("Linear EXR requires 8-16 bit components");
            if (transfer == ColorTransfer::Pq || transfer == ColorTransfer::Hlg || transfer == ColorTransfer::Log || transfer == ColorTransfer::LogSqrt)
                return fail("PQ, HLG and logarithmic transfer are outside the linear SDR profile");
            transfer = overrides.transfer != ColorTransfer::Unspecified ? overrides.transfer : transfer;
            primaries = overrides.primaries != ColorPrimaries::Unspecified ? overrides.primaries : primaries;
            if (transfer != ColorTransfer::Linear && transfer != ColorTransfer::Srgb && transfer != ColorTransfer::Bt709)
                return fail("Linear EXR requires Linear, sRGB or BT709 transfer; supply an input override for missing metadata");
            if (primaries != ColorPrimaries::Bt709 && primaries != ColorPrimaries::Bt2020)
                return fail("Linear EXR requires BT709 or BT2020 primaries; supply an input override for missing metadata");
            if (desc->flags & AV_PIX_FMT_FLAG_RGB) {
                if (range == ColorRange::Limited)
                    return fail("Limited-range RGB is outside the linear SDR profile");
            } else {
                if (!(desc->flags & AV_PIX_FMT_FLAG_PLANAR))
                    return fail("Linear YUV EXR requires planar or semiplanar component storage");
                std::array<float, 3> coefficients{};
                if (matrix == ColorMatrix::Unspecified || !detail::yuvLumaCoefficients(matrix, 0, 0, coefficients))
                    return fail("Linear YUV EXR requires an explicit supported YUV matrix");
                if (range != ColorRange::Full && range != ColorRange::Limited)
                    return fail("Linear YUV EXR requires explicit full or limited range");
            }
            return {};
        }
    } // namespace
    SinkResult validateFloatSdrSource(const StreamDescription& source, FrameColor overrides) {
        const auto tag = [](const std::optional<std::string>& name, auto lookup, int fallback) { return name ? lookup(name->c_str()) : fallback; };
        return validateProfile(source.pixel_format ? av_pix_fmt_desc_get(av_get_pix_fmt(source.pixel_format->c_str())) : nullptr,
                               static_cast<ColorTransfer>(tag(source.color.transfer, av_color_transfer_from_name, AVCOL_TRC_UNSPECIFIED)),
                               static_cast<ColorPrimaries>(tag(source.color.primaries, av_color_primaries_from_name, AVCOL_PRI_UNSPECIFIED)),
                               static_cast<ColorMatrix>(tag(source.color.matrix, av_color_space_from_name, AVCOL_SPC_UNSPECIFIED)),
                               static_cast<ColorRange>(tag(source.color.range, av_color_range_from_name, AVCOL_RANGE_UNSPECIFIED)), overrides);
    }
} // namespace lfs::media
namespace lfs::media::detail {
    LinearFrameConverter::LinearFrameConverter(const AVCodecParameters& source, FrameColor overrides) : overrides_(overrides) {
        source_.color_trc = source.color_trc;
        source_.color_primaries = source.color_primaries;
        source_.color_space = source.color_space;
        source_.color_range = source.color_range;
    }
    LinearFrameConverter::~LinearFrameConverter() = default;
    Result<FrameColor> LinearFrameConverter::outputColor() const {
        const auto color = overrides_.primaries != ColorPrimaries::Unspecified ? overrides_.primaries : static_cast<ColorPrimaries>(source_.color_primaries);
        if (color != ColorPrimaries::Bt709 && color != ColorPrimaries::Bt2020)
            return unsupported("Linear EXR requires declared BT709 or BT2020 primaries; supply an input override for missing metadata");
        return FrameColor{ColorTransfer::Linear, color, AlphaMode::None};
    }
    SinkResult LinearFrameConverter::convert(const AVFrame& frame, int width, int height, uint8_t* destination) {
        const auto* desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(frame.format));
        const auto transfer = static_cast<ColorTransfer>(frame.color_trc != AVCOL_TRC_UNSPECIFIED ? frame.color_trc : source_.color_trc);
        const auto primaries = static_cast<ColorPrimaries>(frame.color_primaries != AVCOL_PRI_UNSPECIFIED ? frame.color_primaries : source_.color_primaries);
        const auto matrix = static_cast<ColorMatrix>(frame.colorspace != AVCOL_SPC_UNSPECIFIED ? frame.colorspace : source_.color_space);
        const auto range = static_cast<ColorRange>(frame.color_range != AVCOL_RANGE_UNSPECIFIED ? frame.color_range : source_.color_range);
        auto valid = validateProfile(desc, transfer, primaries, matrix, range, overrides_);
        if (!valid)
            return valid;
        if (av_frame_get_side_data(&frame, AV_FRAME_DATA_DOVI_METADATA))
            return SinkResult::failure(unsupported("Dolby Vision is outside the linear SDR profile"));
        const auto curve = overrides_.transfer != ColorTransfer::Unspecified ? overrides_.transfer : transfer;
        const auto resolved_primaries = overrides_.primaries != ColorPrimaries::Unspecified ? overrides_.primaries : primaries;
        auto color = outputColor();
        if (!color || resolved_primaries != color->primaries || (transfer_ != ColorTransfer::Unspecified && curve != transfer_))
            return SinkResult::failure(unsupported("Linear SDR transfer or primaries changed during extraction"));
        if (!destination || width <= 0 || height <= 0 || uint64_t(width) * height > std::numeric_limits<uint32_t>::max() / 3)
            return SinkResult::failure(unsupported(std::format("Linear output exceeds address limits (width={}, height={}, component_limit={})", width, height, std::numeric_limits<uint32_t>::max())));
        auto described = describeDecodedVideoFrame(&frame);
        if (!described)
            return SinkResult::failure(std::move(described).error());
        const auto& source = *described;
        VideoColorParameters parameters{};
        parameters.width = frame.width;
        parameters.height = frame.height;
        parameters.rgb = source.rgb;
        parameters.big_endian = source.big_endian;
        parameters.transfer = curve == ColorTransfer::Srgb ? 1 : curve == ColorTransfer::Bt709 ? 2
                                                                                               : 0;
        const auto chroma = chromaMapping(source.chroma_location, source.chroma_w, source.chroma_h);
        parameters.chroma_x_scale = chroma[0];
        parameters.chroma_x_offset = chroma[1];
        parameters.chroma_y_scale = chroma[2];
        parameters.chroma_y_offset = chroma[3];
        std::array<uint64_t, 4> pitches{}, offsets{};
        for (int c = 0; c < 3; ++c) {
            const auto& comp = source.components[c];
            const auto& plane = source.planes[comp.plane];
            pitches[comp.plane] = std::max(pitches[comp.plane], uint64_t(plane.width - 1) * comp.step + comp.offset + (comp.depth + comp.shift + 7) / 8);
        }
        uint64_t bytes = 0;
        for (int i = 0; i < source.plane_count; ++i) {
            offsets[i] = bytes;
            bytes += pitches[i] * source.planes[i].height;
        }
        if (bytes > std::numeric_limits<uint32_t>::max())
            return SinkResult::failure(unsupported(std::format("Linear input exceeds byte-address limits (bytes={}, limit={})", bytes, std::numeric_limits<uint32_t>::max())));
        planes_.resize(static_cast<size_t>(bytes));
        for (int i = 0; i < source.plane_count; ++i)
            for (int y = 0; y < source.planes[i].height; ++y)
                std::memcpy(planes_.data() + offsets[i] + y * pitches[i], source.planes[i].data + y * source.planes[i].pitch, static_cast<size_t>(pitches[i]));
        for (int c = 0; c < 3; ++c) {
            const auto& comp = source.components[c];
            const auto& plane = source.planes[comp.plane];
            parameters.component[c] = {static_cast<uint32_t>(offsets[comp.plane] + comp.offset), static_cast<uint32_t>(pitches[comp.plane]),
                                       static_cast<uint32_t>(comp.step), static_cast<uint32_t>(comp.shift), static_cast<uint32_t>(comp.depth), static_cast<uint32_t>(plane.width), static_cast<uint32_t>(plane.height)};
            const double maximum = (1u << comp.depth) - 1u;
            const auto levels = yuvDecodeLevels(range == ColorRange::Full, (1u << comp.depth) / maximum, false);
            parameters.scale[c] = static_cast<float>((source.rgb ? 1.0 : levels.multiplier[c]) / maximum);
            parameters.bias[c] = source.rgb ? 0.f : static_cast<float>(levels.black[c] * levels.multiplier[c]);
        }
        if (source.rgb)
            parameters.decode[0] = parameters.decode[4] = parameters.decode[8] = 1.f;
        else {
            std::array<float, 3> k{};
            yuvLumaCoefficients(matrix, frame.width, frame.height, k);
            const auto decode = yuvDecodeMatrix(k);
            for (size_t row = 0; row < 3; ++row)
                std::copy(decode[row].begin(), decode[row].end(), parameters.decode + row * 3);
        }
        transfer_ = curve;
        if (!renderer_attempted_) {
            renderer_ = createLinearVideoRenderer();
            renderer_attempted_ = true;
        }
        if (renderer_)
            return renderer_->convert(planes_, parameters, width, height, {destination, static_cast<size_t>(width) * height * 12});
        struct Reader {
            const uint8_t* data;
            uint32_t load(uint32_t address) { return data[address]; }
        };
        VideoColorSampler<Reader> sampler{parameters, {planes_.data()}};
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                std::array<float, 3> value{};
                for (uint32_t c = 0; c < 3; ++c)
                    value[c] = sampler.resized(c, x, y, width, height);
                std::memcpy(destination + (static_cast<size_t>(y) * width + x) * 12, value.data(), 12);
            }
        return {};
    }
} // namespace lfs::media::detail
