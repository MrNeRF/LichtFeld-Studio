// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/path_utils.hpp"
#include "io/video/probed_stream_rewind.hpp"
#include "media/media_probe_ffmpeg.hpp"
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
#include <new>
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
                const auto elapsed = std::chrono::steady_clock::now() - state.started;
                // Convert elapsed time down to milliseconds, not a caller-supplied
                // timeout up to nanoseconds where large positive values overflow.
                state.expired = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed) >= state.timeout;
                return state.expired ? 1 : 0;
            }
        };
        Error invalidOptions(std::string message) {
            return make_error({.code = ErrorCode::InvalidArgument,
                               .domain = ErrorDomain::IO,
                               .detail = std::move(message),
                               .detection = LFS_SOURCE_SITE_CURRENT()});
        }
        Error failure(int error, const char* operation, bool expired) {
            char text[AV_ERROR_MAX_STRING_SIZE]{};
            av_strerror(error, text, sizeof(text));
            const auto code = expired ? ErrorCode::DeadlineExceeded : error == AVERROR(ENOMEM)   ? ErrorCode::ResourceExhausted
                                                                  : error == AVERROR(ENOENT)     ? ErrorCode::NotFound
                                                                  : error == AVERROR(EACCES)     ? ErrorCode::PermissionDenied
                                                                  : error == AVERROR_INVALIDDATA ? ErrorCode::DataLoss
                                                                                                 : ErrorCode::Unavailable;
            return make_error({.code = code,
                               .domain = ErrorDomain::IO,
                               .detail = std::string(operation) + ": " + text,
                               .detection = LFS_SOURCE_SITE_CURRENT(),
                               .fields = SmallFields{}.add("operation", std::string_view(operation)),
                               .native = NativeError{ErrorDomain::IO, error, text}});
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
                std::string_view value(tag->value);
                result.orientation.rotate_tag.reset();
                result.orientation.rotate_tag_limited = value.size() > Orientation::max_rotate_tag_bytes;
                if (!result.orientation.rotate_tag_limited)
                    result.orientation.rotate_tag = value;
                result.orientation.clockwise_degrees.reset();
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
                if (p && p->codec_type == AVMEDIA_TYPE_VIDEO && !(stream->disposition & AV_DISPOSITION_ATTACHED_PIC) &&
                    p->codec_id != AV_CODEC_ID_NONE && p->width > 0 && p->height > 0)
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
            lfs::io::video::limitUntimedVideoProbe(context);
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

    Result<MediaDescription> MediaProbe::inspect(const std::filesystem::path& path, const ProbeOptions& options) {
        if (path.empty() || path.native().find(std::filesystem::path::value_type{}) != std::string::npos || options.timeout.count() <= 0 ||
            (options.depth != ProbeDepth::Headers && options.depth != ProbeDepth::StreamInfo))
            return invalidOptions("Probe requires a valid path, depth and positive timeout");
        // Complete allocating path conversions before creating the FFmpeg context.
        // The native-path NUL check above also covers the display representation.
        std::string native;
        try {
            if (options.timed_metadata_stream) {
                const auto& limits = options.timed_metadata_limits;
                if (*options.timed_metadata_stream < 0 || options.depth != ProbeDepth::Headers ||
                    !limits.bytes || limits.bytes > timed_metadata_maxima.bytes ||
                    !limits.records || limits.records > timed_metadata_maxima.records ||
                    !limits.scan_packets || limits.scan_packets > timed_metadata_maxima.scan_packets)
                    return invalidOptions("Timed metadata requires Headers depth, a nonnegative stream index and positive limits within maxima (stream=" + std::to_string(*options.timed_metadata_stream) + ", bytes=" + std::to_string(limits.bytes) + "/" + std::to_string(timed_metadata_maxima.bytes) + ", records=" + std::to_string(limits.records) + "/" + std::to_string(timed_metadata_maxima.records) + ", scan_packets=" + std::to_string(limits.scan_packets) + "/" + std::to_string(timed_metadata_maxima.scan_packets) + ")");
            }
            native = core::path_to_native_utf8(path);
        } catch (const std::bad_alloc&) {
            return failure(AVERROR(ENOMEM), "Prepare input path", false);
        }
        // Timed collection is scoped to embedded MOV/MP4 tracks. Forcing this
        // demuxer also keeps standalone subtitles on the canonical SRT reader.
        const auto* input_format = options.timed_metadata_stream ? av_find_input_format("mov") : nullptr;
        if (options.timed_metadata_stream && !input_format)
            return failure(AVERROR_DEMUXER_NOT_FOUND, "Locate timed metadata demuxer", false);
        Deadline deadline{std::chrono::steady_clock::now(), options.timeout};
        AVFormatContext* raw = avformat_alloc_context();
        if (!raw)
            return failure(AVERROR(ENOMEM), "Allocate input", false);
        auto close = [](AVFormatContext* context) { avformat_close_input(&context); };
        raw->interrupt_callback = {Deadline::interrupt, &deadline};
        const int opened = avformat_open_input(&raw, native.c_str(), input_format, nullptr);
        std::unique_ptr<AVFormatContext, decltype(close)> context(raw, close);
        if (opened < 0)
            return failure(opened, "Open input", deadline.expired);
        if (options.depth == ProbeDepth::StreamInfo) {
            const int probed = avformat_find_stream_info(context.get(), nullptr);
            if (probed < 0)
                return failure(probed, "Read stream info", deadline.expired);
        }
        try {
            auto description = detail::describeContext(context.get());
            description.stream_info_probed = options.depth == ProbeDepth::StreamInfo;
            if (options.collect_source_metadata)
                detail::collectContextMetadata(context.get(), description, options.metadata_limits);
            if (options.timed_metadata_stream) {
                const auto index = *options.timed_metadata_stream;
                if (static_cast<unsigned>(index) >= context->nb_streams || !context->streams[index] || !context->streams[index]->codecpar)
                    return invalidOptions("Timed metadata stream index is outside the input stream table");
                const auto* stream = context->streams[index];
                const auto* parameters = stream->codecpar;
                if (parameters->codec_type != AVMEDIA_TYPE_DATA && parameters->codec_type != AVMEDIA_TYPE_SUBTITLE)
                    return invalidOptions("Timed metadata collection requires a data or subtitle stream");
                if (stream->time_base.num <= 0 || stream->time_base.den <= 0)
                    return make_error({.code = ErrorCode::DataLoss, .domain = ErrorDomain::IO, .detail = "Timed metadata stream has no valid time base", .detection = LFS_SOURCE_SITE_CURRENT()});
                const auto& limits = options.timed_metadata_limits;
                TimedMetadataObservation observation;
                observation.provider = detail::metadataProviderInfo();
                observation.stream_index = index;
                observation.codec_tag = parameters->codec_tag;
                std::size_t retained = 0, records = 0;
                auto exhausted = [&](std::size_t next_bytes = 0) {
                    return make_error({.code = ErrorCode::ResourceExhausted, .domain = ErrorDomain::IO, .detail = "Timed metadata budget exceeded (retained_bytes=" + std::to_string(retained) + "/" + std::to_string(limits.bytes) + ", next_bytes=" + std::to_string(next_bytes) + ", records=" + std::to_string(records) + "/" + std::to_string(limits.records) + ", scanned_packets=" + std::to_string(observation.packets_examined) + "/" + std::to_string(limits.scan_packets) + ")", .detection = LFS_SOURCE_SITE_CURRENT()});
                };
                const auto retain = [&](const std::uint8_t* data, std::size_t size, std::vector<std::uint8_t>& destination) {
                    if (size > limits.bytes - retained)
                        return false;
                    if (size)
                        destination.assign(data, data + size);
                    retained += size;
                    return true;
                };
                if (parameters->extradata_size < 0 || (parameters->extradata_size && !parameters->extradata))
                    return make_error({.code = ErrorCode::DataLoss, .domain = ErrorDomain::IO, .detail = "Timed metadata codec configuration is invalid", .detection = LFS_SOURCE_SITE_CURRENT()});
                if (!retain(parameters->extradata, static_cast<std::size_t>(parameters->extradata_size), observation.extradata))
                    return exhausted(static_cast<std::size_t>(parameters->extradata_size));
                auto release = [](AVPacket* packet) { av_packet_free(&packet); };
                std::unique_ptr<AVPacket, decltype(release)> packet(av_packet_alloc(), release);
                if (!packet)
                    return failure(AVERROR(ENOMEM), "Allocate metadata packet", false);
                while (true) {
                    if (Deadline::interrupt(&deadline))
                        return failure(AVERROR_EXIT, "Read timed metadata", true);
                    const auto read = av_read_frame(context.get(), packet.get());
                    if (read == AVERROR_EOF) {
                        if (context->pb && context->pb->error < 0 && context->pb->error != AVERROR_EOF)
                            return failure(context->pb->error, "Read timed metadata", deadline.expired);
                        // MOV can report EOF for a missing sample without returning
                        // a corrupt packet. Reuse its parsed index, including any
                        // entries discovered in fragments, rather than reparsing boxes
                        // or assuming declared frame counts equal returned packets.
                        // A missing sample in any read track can stop the demuxer,
                        // truncating even an otherwise valid selected track.
                        const auto input_size = context->pb ? avio_size(context->pb) : AVERROR(EIO);
                        if (input_size < 0)
                            return failure(static_cast<int>(input_size), "Determine timed metadata input size", deadline.expired);
                        for (unsigned track = 0; track < context->nb_streams; ++track) {
                            auto* indexed_stream = context->streams[track];
                            if (!indexed_stream)
                                return failure(AVERROR_INVALIDDATA, "Validate timed metadata stream index", false);
                            const auto entries = avformat_index_get_entries_count(indexed_stream);
                            for (int entry_index = 0; entry_index < entries; ++entry_index) {
                                if (Deadline::interrupt(&deadline))
                                    return failure(AVERROR_EXIT, "Validate timed metadata index", true);
                                const auto* entry = avformat_index_get_entry(indexed_stream, entry_index);
                                if (!entry || entry->pos < 0 || entry->size < 0 || entry->pos > input_size || entry->size > input_size - entry->pos)
                                    return make_error({.code = ErrorCode::DataLoss, .domain = ErrorDomain::IO, .detail = "Timed metadata sample exceeds input bounds (stream=" + std::to_string(track) + ", index=" + std::to_string(entry_index) + ", position=" + std::to_string(entry ? entry->pos : -1) + ", bytes=" + std::to_string(entry ? entry->size : -1) + ", input_bytes=" + std::to_string(input_size) + ")", .detection = LFS_SOURCE_SITE_CURRENT()});
                            }
                        }
                        if (Deadline::interrupt(&deadline))
                            return failure(AVERROR_EXIT, "Read timed metadata", true);
                        break;
                    }
                    if (read < 0)
                        return failure(read, "Read timed metadata", deadline.expired);
                    if (packet->stream_index < 0 || static_cast<unsigned>(packet->stream_index) >= context->nb_streams)
                        return failure(AVERROR_INVALIDDATA, "Read timed metadata stream identity", false);
                    if (++observation.packets_examined > limits.scan_packets)
                        return exhausted();
                    if (packet->stream_index == index) {
                        if (packet->duration < 0 || packet->size < 0 || (packet->size && !packet->data) || packet->side_data_elems < 0 || (packet->side_data_elems && !packet->side_data) || (packet->flags & AV_PKT_FLAG_CORRUPT))
                            return failure(AVERROR_INVALIDDATA, "Read timed metadata packet", false);
                        if (records >= limits.records)
                            return exhausted();
                        ++records;
                        TimedMetadataPacket sample;
                        sample.pts = timestamp(packet->pts, stream->time_base);
                        sample.dts = timestamp(packet->dts, stream->time_base);
                        // AVPacket duration zero means unknown, not a point sample.
                        if (packet->duration > 0)
                            sample.duration = timestamp(packet->duration, stream->time_base);
                        if (packet->pos >= 0)
                            sample.source_offset = packet->pos;
                        sample.flags = packet->flags;
                        if (!retain(packet->data, static_cast<std::size_t>(packet->size), sample.bytes))
                            return exhausted(static_cast<std::size_t>(packet->size));
                        for (int side = 0; side < packet->side_data_elems; ++side) {
                            const auto& block = packet->side_data[side];
                            if (block.size && !block.data)
                                return failure(AVERROR_INVALIDDATA, "Read timed metadata side data", false);
                            if (records >= limits.records)
                                return exhausted();
                            ++records;
                            MetadataSideData owned;
                            owned.type = static_cast<int>(block.type);
                            if (!retain(block.data, block.size, owned.bytes))
                                return exhausted(block.size);
                            sample.side_data.push_back(std::move(owned));
                        }
                        observation.packets.push_back(std::move(sample));
                    }
                    av_packet_unref(packet.get());
                }
                description.timed_metadata = std::move(observation);
            }
            return description;
        } catch (const std::bad_alloc&) {
            return failure(AVERROR(ENOMEM), "Describe input metadata", false);
        }
    }
} // namespace lfs::media
