// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/path_utils.hpp"
#include "media_probe_ffmpeg.hpp"
extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/display.h>
#include <libavutil/pixdesc.h>
}
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <memory>
#include <string_view>

namespace lfs::media {
    namespace {
        std::optional<Rational> positiveRational(AVRational value) {
            if (value.num <= 0 || value.den <= 0)
                return std::nullopt;
            return Rational{value.num, value.den};
        }
        std::optional<Timestamp> timestamp(int64_t value, AVRational base) {
            const auto rational = positiveRational(base);
            if (value == AV_NOPTS_VALUE || !rational)
                return std::nullopt;
            return Timestamp{value, *rational};
        }
        std::optional<Timestamp> duration(int64_t value, AVRational base) {
            return value >= 0 ? timestamp(value, base) : std::nullopt;
        }
        std::optional<std::string> named(bool available, const char* name) {
            return available && name ? std::optional<std::string>(name) : std::nullopt;
        }
        StreamKind kind(AVMediaType type) {
            switch (type) {
            case AVMEDIA_TYPE_VIDEO: return StreamKind::Video;
            case AVMEDIA_TYPE_AUDIO: return StreamKind::Audio;
            case AVMEDIA_TYPE_SUBTITLE: return StreamKind::Subtitle;
            case AVMEDIA_TYPE_DATA: return StreamKind::Data;
            case AVMEDIA_TYPE_ATTACHMENT: return StreamKind::Attachment;
            default: return StreamKind::Unknown;
            }
        }
        struct Deadline {
            std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
            std::chrono::milliseconds timeout;
            bool expired = false;
            static int interrupt(void* opaque) noexcept {
                auto& state = *static_cast<Deadline*>(opaque);
                state.expired = std::chrono::steady_clock::now() - state.started >= state.timeout;
                return state.expired ? 1 : 0;
            }
        };
        ProbeResult failure(ProbeErrorCode code, int error, const char* operation, bool expired) {
            char text[AV_ERROR_MAX_STRING_SIZE]{};
            av_strerror(error, text, sizeof(text));
            return {std::nullopt, {expired ? ProbeErrorCode::TimedOut : code, error, std::string(operation) + ": " + text}};
        }
    } // namespace

    int legacyQuarterTurn(const Orientation& orientation) {
        if (!orientation.clockwise_degrees || !std::isfinite(*orientation.clockwise_degrees))
            return 0;
        const double reduced = std::fmod(*orientation.clockwise_degrees, 360.0);
        const double rounded = orientation.source == OrientationSource::RotateTag
                                   ? std::trunc(reduced)
                                   : std::round(reduced);
        const int normalized = (static_cast<int>(rounded) % 360 + 360) % 360;
        return normalized == 90 || normalized == 180 || normalized == 270 ? normalized : 0;
    }

    namespace detail {
        Orientation describeDisplayMatrix(const unsigned char* data, std::size_t size) {
            Orientation result;
            result.source = OrientationSource::DisplayMatrix;
            if (!data || size < sizeof(int32_t) * 9)
                return result;
            std::array<int32_t, 9> matrix;
            std::memcpy(matrix.data(), data, sizeof(matrix));
            result.display_matrix = matrix;
            const double angle = -av_display_rotation_get(matrix.data());
            if (std::isfinite(angle))
                result.clockwise_degrees = angle;
            const double determinant = static_cast<double>(matrix[0]) * matrix[4] -
                                       static_cast<double>(matrix[1]) * matrix[3];
            if (determinant != 0)
                result.reflected = determinant < 0;
            return result;
        }

        StreamDescription describeStream(const AVStream* stream) {
            StreamDescription result;
            if (!stream || !stream->codecpar)
                return result;
            const auto& p = *stream->codecpar;
            result.index = stream->index;
            result.kind = kind(p.codec_type);
            result.codec = avcodec_get_name(p.codec_id);
            result.time_base = positiveRational(stream->time_base);
            result.start = timestamp(stream->start_time, stream->time_base);
            result.duration = duration(stream->duration, stream->time_base);
            result.attached_picture = (stream->disposition & AV_DISPOSITION_ATTACHED_PIC) != 0;
            result.default_disposition = (stream->disposition & AV_DISPOSITION_DEFAULT) != 0;
            if (stream->nb_frames > 0)
                result.declared_frame_count = stream->nb_frames;
            if (p.codec_type != AVMEDIA_TYPE_VIDEO)
                return result;
            if (p.width > 0)
                result.width = p.width;
            if (p.height > 0)
                result.height = p.height;
            result.nominal_frame_rate = positiveRational(stream->r_frame_rate);
            result.average_frame_rate = positiveRational(stream->avg_frame_rate);
            result.sample_aspect_ratio = positiveRational(stream->sample_aspect_ratio);
            const auto format = static_cast<AVPixelFormat>(p.format);
            result.pixel_format = named(p.format != AV_PIX_FMT_NONE, av_get_pix_fmt_name(format));
            const auto* desc = av_pix_fmt_desc_get(format);
            if (desc && desc->nb_components > 0) {
                int depth = 0;
                for (int i = 0; i < desc->nb_components; ++i)
                    depth = std::max(depth, desc->comp[i].depth);
                result.color.component_depth = depth;
            } else if (p.bits_per_raw_sample > 0) {
                result.color.component_depth = p.bits_per_raw_sample;
            }
            result.color.primaries = named(p.color_primaries != AVCOL_PRI_UNSPECIFIED, av_color_primaries_name(p.color_primaries));
            result.color.transfer = named(p.color_trc != AVCOL_TRC_UNSPECIFIED, av_color_transfer_name(p.color_trc));
            result.color.matrix = named(p.color_space != AVCOL_SPC_UNSPECIFIED, av_color_space_name(p.color_space));
            result.color.range = named(p.color_range != AVCOL_RANGE_UNSPECIFIED, av_color_range_name(p.color_range));
            for (int i = 0; i < p.nb_coded_side_data; ++i) {
                const auto& side = p.coded_side_data[i];
                if (side.type == AV_PKT_DATA_DISPLAYMATRIX) {
                    result.orientation = describeDisplayMatrix(side.data, side.size);
                    break;
                }
            }
            // Preserve the existing player's tag-first precedence; retain both raw sources.
            const auto* tag = av_dict_get(stream->metadata, "rotate", nullptr, 0);
            if (tag && tag->value) {
                result.orientation.source = OrientationSource::RotateTag;
                result.orientation.rotate_tag = tag->value;
                result.orientation.clockwise_degrees.reset();
                std::string_view value(tag->value);
                const auto first = value.find_first_not_of(" \t\r\n");
                if (first == std::string_view::npos)
                    return result;
                const auto last = value.find_last_not_of(" \t\r\n");
                value = value.substr(first, last - first + 1);
                if (value.front() == '+') {
                    value.remove_prefix(1);
                    if (value.empty() || value.front() == '-')
                        return result;
                }
                double angle = 0;
                const auto parsed = std::from_chars(value.data(), value.data() + value.size(), angle);
                if (parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() && std::isfinite(angle))
                    result.orientation.clockwise_degrees = angle;
            }
            return result;
        }

        int findUsableHeaderVideoStream(const AVFormatContext* context) {
            if (!context)
                return -1;
            for (unsigned int i = 0; i < context->nb_streams; ++i) {
                const auto* stream = context->streams[i];
                const auto* p = stream ? stream->codecpar : nullptr;
                if (p && p->codec_type == AVMEDIA_TYPE_VIDEO && p->codec_id != AV_CODEC_ID_NONE && p->width > 0 && p->height > 0)
                    return static_cast<int>(i);
            }
            return -1;
        }
        void discardNonVideoStreams(AVFormatContext* context, int video_stream_index) {
            if (!context)
                return;
            for (unsigned int i = 0; i < context->nb_streams; ++i)
                if (context->streams[i])
                    context->streams[i]->discard = static_cast<int>(i) == video_stream_index ? AVDISCARD_DEFAULT : AVDISCARD_ALL;
        }
        VideoStreamProbe probeVideoStream(AVFormatContext* context) {
            VideoStreamProbe result;
            if (!context)
                return result;
            result.stream_index = findUsableHeaderVideoStream(context);
            if (result.stream_index < 0) {
                result.ffmpeg_error = avformat_find_stream_info(context, nullptr);
                if (result.ffmpeg_error < 0)
                    return result;
                result.metadata_complete = true;
                result.stream_index = findUsableHeaderVideoStream(context);
            }
            if (result.stream_index < 0)
                return result;
            discardNonVideoStreams(context, result.stream_index);
            if (!result.metadata_complete) {
                result.ffmpeg_error = avformat_find_stream_info(context, nullptr);
                result.metadata_complete = result.ffmpeg_error >= 0;
            }
            return result;
        }

        MediaDescription describeContext(const AVFormatContext* context) {
            MediaDescription result;
            if (!context)
                return result;
            if (context->iformat && context->iformat->name)
                result.container = context->iformat->name;
            result.start = timestamp(context->start_time, {1, AV_TIME_BASE});
            result.duration = duration(context->duration, {1, AV_TIME_BASE});
            const int selected = findUsableHeaderVideoStream(context);
            if (selected >= 0)
                result.selected_video_stream = selected;
            result.streams.reserve(context->nb_streams);
            for (unsigned int i = 0; i < context->nb_streams; ++i)
                result.streams.push_back(describeStream(context->streams[i]));
            return result;
        }
    } // namespace detail

    ProbeResult MediaProbe::inspect(const std::filesystem::path& path, const ProbeOptions& options) {
        if (path.empty() || path.native().find(std::filesystem::path::value_type{}) != std::string::npos || options.timeout.count() <= 0 ||
            (options.depth != ProbeDepth::Headers && options.depth != ProbeDepth::StreamInfo))
            return {std::nullopt, {ProbeErrorCode::InvalidOptions, 0, "Probe requires a path and a positive timeout"}};
        const auto utf8 = core::path_to_utf8(path);
        if (utf8.find('\0') != std::string::npos)
            return {std::nullopt, {ProbeErrorCode::InvalidOptions, 0, "Probe path contains a null byte"}};
        Deadline deadline{std::chrono::steady_clock::now(), options.timeout};
        AVFormatContext* raw = avformat_alloc_context();
        if (!raw)
            return failure(ProbeErrorCode::OpenInput, AVERROR(ENOMEM), "Allocate input", false);
        auto close = [](AVFormatContext* context) { avformat_close_input(&context); };
        raw->interrupt_callback = {Deadline::interrupt, &deadline};
        const int opened = avformat_open_input(&raw, utf8.c_str(), nullptr, nullptr);
        std::unique_ptr<AVFormatContext, decltype(close)> context(raw, close);
        if (opened < 0)
            return failure(ProbeErrorCode::OpenInput, opened, "Open input", deadline.expired);
        if (options.depth == ProbeDepth::StreamInfo) {
            const int probed = avformat_find_stream_info(context.get(), nullptr);
            if (probed < 0)
                return failure(ProbeErrorCode::StreamInfo, probed, "Read stream info", deadline.expired);
        }
        auto description = detail::describeContext(context.get());
        description.stream_info_probed = options.depth == ProbeDepth::StreamInfo;
        return {std::move(description), {}};
    }
} // namespace lfs::media
