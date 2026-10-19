/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "video_metadata.hpp"

#include "core/include/core/logger.hpp"
#include "core/path_utils.hpp"
#include "video/probed_stream_rewind.hpp"

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
#include <libavutil/display.h>
#include <libavutil/dovi_meta.h>
#include <libavutil/error.h>
#include <libavutil/pixdesc.h>
}

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <format>

namespace lfs::io::video {

    namespace {

        [[nodiscard]] std::string ffmpegError(const int result) {
            char buffer[AV_ERROR_MAX_STRING_SIZE]{};
            av_strerror(result, buffer, sizeof(buffer));
            return buffer;
        }

        [[nodiscard]] int findUsableHeaderVideoStream(const AVFormatContext* const context) {
            if (!context)
                return -1;

            for (unsigned int i = 0; i < context->nb_streams; ++i) {
                const AVStream* const stream = context->streams[i];
                const AVCodecParameters* const parameters = stream ? stream->codecpar : nullptr;
                if (parameters && parameters->codec_type == AVMEDIA_TYPE_VIDEO &&
                    parameters->codec_id != AV_CODEC_ID_NONE && parameters->width > 0 &&
                    parameters->height > 0) {
                    return static_cast<int>(i);
                }
            }
            return -1;
        }

        void discardNonVideoStreams(AVFormatContext* const context, const int video_stream_idx) {
            if (!context)
                return;

            for (unsigned int i = 0; i < context->nb_streams; ++i)
                context->streams[i]->discard = static_cast<int>(i) == video_stream_idx
                                                   ? AVDISCARD_DEFAULT
                                                   : AVDISCARD_ALL;
        }

        [[nodiscard]] int pixelFormatBitDepth(const AVCodecParameters* const parameters) {
            if (!parameters)
                return 0;
            const AVPixFmtDescriptor* const descriptor =
                av_pix_fmt_desc_get(static_cast<AVPixelFormat>(parameters->format));
            return std::max(descriptor ? descriptor->comp[0].depth : 0,
                            parameters->bits_per_raw_sample);
        }

        [[nodiscard]] bool hasCodedSideData(const AVCodecParameters* const parameters,
                                            const AVPacketSideDataType type) {
            if (!parameters)
                return false;
            for (int i = 0; i < parameters->nb_coded_side_data; ++i) {
                if (parameters->coded_side_data[i].type == type)
                    return true;
            }
            return false;
        }

    } // namespace

    bool probeVideoMetadata(AVFormatContext* const context, VideoMetadata& metadata,
                            std::string& error) {
        metadata = {};
        error.clear();
        if (!context) {
            error = "Video demuxer is not available";
            return false;
        }

        limitUntimedVideoProbe(context);
        metadata.stream_index = findUsableHeaderVideoStream(context);
        bool stream_info_probed = false;
        if (metadata.stream_index < 0) {
            stream_info_probed = true;
            const int probe_result = avformat_find_stream_info(context, nullptr);
            if (probe_result < 0) {
                error = std::format("Failed to read video stream metadata: {}", ffmpegError(probe_result));
                return false;
            }
            metadata.stream_index = findUsableHeaderVideoStream(context);
        }
        if (metadata.stream_index < 0) {
            error = "No usable video stream found";
            return false;
        }

        discardNonVideoStreams(context, metadata.stream_index);
        if (!stream_info_probed && avformat_find_stream_info(context, nullptr) < 0) {
            // Match VideoPlayer's tolerant behavior: a valid header stream may still be
            // usable even when FFmpeg cannot complete its optional metadata probe.
            LOG_WARN("Video metadata probe: could not complete video-only stream metadata probe");
        }

        const AVStream* const stream = context->streams[metadata.stream_index];
        const AVCodecParameters* const parameters = stream->codecpar;
        if (AVDictionaryEntry* const tag = av_dict_get(stream->metadata, "rotate", nullptr, 0);
            tag && tag->value) {
            metadata.rotation = std::atoi(tag->value);
        } else {
            for (int i = 0; i < parameters->nb_coded_side_data; ++i) {
                const AVPacketSideData& side_data = parameters->coded_side_data[i];
                if (side_data.type != AV_PKT_DATA_DISPLAYMATRIX)
                    continue;
                const double angle = av_display_rotation_get(reinterpret_cast<int32_t*>(side_data.data));
                metadata.rotation = static_cast<int>(std::round(-angle));
                break;
            }
        }
        metadata.rotation = ((metadata.rotation % 360) + 360) % 360;
        if (metadata.rotation != 0 && metadata.rotation != 90 &&
            metadata.rotation != 180 && metadata.rotation != 270) {
            metadata.rotation = 0;
        }

        metadata.hdr_format = detectHdrFormat(
            parameters->color_trc, pixelFormatBitDepth(parameters),
            hasCodedSideData(parameters, AV_PKT_DATA_MASTERING_DISPLAY_METADATA),
            hasCodedSideData(parameters, AV_PKT_DATA_CONTENT_LIGHT_LEVEL));
        for (int i = 0; i < parameters->nb_coded_side_data; ++i) {
            const AVPacketSideData& side_data = parameters->coded_side_data[i];
            if (side_data.type != AV_PKT_DATA_DOVI_CONF ||
                side_data.size < static_cast<int>(sizeof(AVDOVIDecoderConfigurationRecord))) {
                continue;
            }
            const auto* const dovi = reinterpret_cast<const AVDOVIDecoderConfigurationRecord*>(side_data.data);
            if (dovi->dv_profile > 0) {
                metadata.has_dolby_vision = true;
                metadata.hdr_format = detectDolbyVisionFormat(parameters->color_trc, dovi->dv_profile,
                                                              dovi->dv_bl_signal_compatibility_id);
            }
            break;
        }
        metadata.is_hdr = isHdrFormat(metadata.hdr_format);

        metadata.fps = av_q2d(stream->avg_frame_rate);
        if (!std::isfinite(metadata.fps) || metadata.fps <= 0.0)
            metadata.fps = av_q2d(stream->r_frame_rate);
        if (!std::isfinite(metadata.fps) || metadata.fps <= 0.0)
            metadata.fps = 30.0;
        metadata.untimed_stream = context->iformat &&
                                  (context->iformat->flags & AVFMT_NOTIMESTAMPS) != 0;
        if (!metadata.untimed_stream && stream->nb_frames > 0) {
            metadata.total_frames = stream->nb_frames;
            metadata.duration = static_cast<double>(metadata.total_frames) / metadata.fps;
        } else if (!metadata.untimed_stream && context->duration != AV_NOPTS_VALUE &&
                   context->duration > 0) {
            metadata.duration = static_cast<double>(context->duration) / AV_TIME_BASE;
        } else if (!metadata.untimed_stream && stream->duration != AV_NOPTS_VALUE &&
                   stream->duration > 0) {
            const double time_base = av_q2d(stream->time_base);
            if (std::isfinite(time_base) && time_base > 0.0)
                metadata.duration = static_cast<double>(stream->duration) * time_base;
        }
        if (metadata.total_frames <= 0 && metadata.duration > 0.0)
            metadata.total_frames = static_cast<int64_t>(std::llround(metadata.duration * metadata.fps));
        return true;
    }

    bool probeVideoFileMetadata(const std::filesystem::path& path, VideoMetadata& metadata,
                                std::string& error) {
        AVFormatContext* context = nullptr;
        const std::string path_utf8 = lfs::core::path_to_native_utf8(path);
        const int open_result = avformat_open_input(&context, path_utf8.c_str(), nullptr, nullptr);
        if (open_result < 0) {
            avformat_close_input(&context);
            error = std::format("Failed to open video: {}", ffmpegError(open_result));
            return false;
        }
        const bool ok = probeVideoMetadata(context, metadata, error);
        avformat_close_input(&context);
        return ok;
    }

} // namespace lfs::io::video
