/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "media/video_frame_extractor.hpp"
#include "core/atomic_file.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "decoded_video_frame_ffmpeg.hpp"
#include "linear_frame.hpp"
#include "media/file_frame_sink.hpp"
#include "media/hdr_renderer.hpp"
#include "media/hdr_tonemap.hpp"
#include "media/media_backends.hpp"
#include "media/media_probe_ffmpeg.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/dovi_meta.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

#include "media/cuda_frame_ffmpeg.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <vector>

namespace lfs::io {

    namespace {
        // Extraction runs off the UI thread and benefits from more parallel
        // HEVC decoding than the latency-sensitive preview path.
        constexpr int MAX_SW_DECODE_THREADS = 8;
        template <size_t Bytes, int Rotation>
        void rotatePackedPixels(const uint8_t* input, std::vector<uint8_t>& output, int width, int height) {
            output.resize(static_cast<size_t>(width) * height * Bytes);
            const int dst_width = Rotation == 180 ? width : height;
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x) {
                    const int dx = Rotation == 180 ? width - 1 - x : Rotation == 90 ? height - 1 - y
                                                                                    : y;
                    const int dy = Rotation == 180 ? height - 1 - y : Rotation == 90 ? x
                                                                                     : width - 1 - x;
                    std::memcpy(output.data() + (static_cast<size_t>(dy) * dst_width + dx) * Bytes,
                                input + (static_cast<size_t>(y) * width + x) * Bytes, Bytes);
                }
        }
        template <size_t Bytes>
        void rotatePackedFormat(const uint8_t* input, std::vector<uint8_t>& output, int width, int height, int rotation) {
            if (rotation == 180)
                rotatePackedPixels<Bytes, 180>(input, output, width, height);
            else if (rotation == 90)
                rotatePackedPixels<Bytes, 90>(input, output, width, height);
            else
                rotatePackedPixels<Bytes, 270>(input, output, width, height);
        }
        void rotatePacked(const uint8_t* input, std::vector<uint8_t>& output, int width, int height, int rotation, size_t bytes) {
            // Dispatch outside the pixel loop so each copy has a fixed size and
            // rotation arithmetic. In particular keep the RGB8 path inexpensive.
            if (bytes == 3)
                rotatePackedFormat<3>(input, output, width, height, rotation);
            else
                rotatePackedFormat<12>(input, output, width, height, rotation);
        }

        [[nodiscard]] double elapsedSeconds(const std::chrono::steady_clock::time_point started) {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        }

        [[nodiscard]] bool configureVideoToRgbColorimetry(
            SwsContext* context, const AVFrame* source,
            const AVColorSpace fallback_colorspace,
            const AVColorRange fallback_range) {
            if (!context || !source)
                return false;
            const auto* pixel_description = av_pix_fmt_desc_get(
                static_cast<AVPixelFormat>(source->format));
            // YUV matrices/ranges do not describe an RGB source. Applying them
            // during RGB resize can change the colors even though no color-space
            // conversion was requested.
            if (pixel_description && (pixel_description->flags & AV_PIX_FMT_FLAG_RGB))
                return true;
            const AVColorSpace colorspace = source->colorspace != AVCOL_SPC_UNSPECIFIED
                                                ? source->colorspace
                                                : fallback_colorspace;
            const AVColorRange color_range = source->color_range != AVCOL_RANGE_UNSPECIFIED
                                                 ? source->color_range
                                                 : fallback_range;
            const int source_matrix = colorspace == AVCOL_SPC_BT2020_NCL ||
                                              colorspace == AVCOL_SPC_BT2020_CL
                                          ? SWS_CS_BT2020
                                      : colorspace == AVCOL_SPC_BT709 ? SWS_CS_ITU709
                                                                      : SWS_CS_DEFAULT;
            const int source_range = color_range == AVCOL_RANGE_JPEG ? 1 : 0;
            return sws_setColorspaceDetails(
                       context, sws_getCoefficients(source_matrix), source_range,
                       sws_getCoefficients(SWS_CS_ITU709), 1, 0, 1 << 16,
                       1 << 16) >= 0;
        }

        void inheritStreamColorimetry(AVFrame* const frame, const AVCodecParameters* const parameters,
                                      const HdrFormat hdr_format) {
            if (!frame || !parameters)
                return;
            if (frame->colorspace == AVCOL_SPC_UNSPECIFIED)
                frame->colorspace = parameters->color_space != AVCOL_SPC_UNSPECIFIED
                                        ? parameters->color_space
                                        : AVCOL_SPC_BT2020_NCL;
            if (frame->color_range == AVCOL_RANGE_UNSPECIFIED)
                frame->color_range = parameters->color_range != AVCOL_RANGE_UNSPECIFIED
                                         ? parameters->color_range
                                         : AVCOL_RANGE_MPEG;
            if (frame->color_primaries == AVCOL_PRI_UNSPECIFIED)
                frame->color_primaries = parameters->color_primaries != AVCOL_PRI_UNSPECIFIED
                                             ? parameters->color_primaries
                                             : AVCOL_PRI_BT2020;
            if (frame->color_trc == AVCOL_TRC_UNSPECIFIED) {
                frame->color_trc = parameters->color_trc != AVCOL_TRC_UNSPECIFIED
                                       ? parameters->color_trc
                                       : (hdr_format == HdrFormat::HLG ||
                                                  hdr_format == HdrFormat::DOLBY_VISION_HLG
                                              ? AVCOL_TRC_ARIB_STD_B67
                                              : AVCOL_TRC_SMPTE2084);
            }
            if (frame->chroma_location == AVCHROMA_LOC_UNSPECIFIED)
                frame->chroma_location = parameters->chroma_location;
        }

        struct ExtractionCancelled final : std::exception {
            [[nodiscard]] const char* what() const noexcept override { return "Extraction stopped"; }
        };

        constexpr int MAX_FILENAME_FRAME_WIDTH = 64;

        void appendFrameNumber(std::string& out, const int frame_number, const int min_width) {
            const std::string number = std::to_string(frame_number);
            if (frame_number < 0 || min_width <= static_cast<int>(number.size())) {
                out += number;
                return;
            }

            out.append(static_cast<size_t>(min_width - static_cast<int>(number.size())), '0');
            out += number;
        }

        [[nodiscard]] double validFrameRate(const AVRational frame_rate) {
            const double fps = av_q2d(frame_rate);
            return std::isfinite(fps) && fps > 0.0 ? fps : 0.0;
        }

        [[nodiscard]] double validDurationSeconds(const int64_t duration, const AVRational time_base) {
            if (duration == AV_NOPTS_VALUE || time_base.num <= 0 || time_base.den <= 0)
                return 0.0;

            const double seconds = static_cast<double>(duration) * av_q2d(time_base);
            return std::isfinite(seconds) && seconds > 0.0 ? seconds : 0.0;
        }

        [[nodiscard]] double frameTimestampSeconds(const AVFrame* const frame,
                                                   const double time_base,
                                                   const int frame_index,
                                                   const double fallback_fps,
                                                   const double stream_start_seconds) {
            int64_t timestamp = frame->best_effort_timestamp;
            if (timestamp == AV_NOPTS_VALUE)
                timestamp = frame->pts;
            if (timestamp != AV_NOPTS_VALUE && std::isfinite(time_base) && time_base > 0.0)
                return static_cast<double>(timestamp) * time_base -
                       stream_start_seconds;
            return static_cast<double>(frame_index) / std::max(fallback_fps, 0.001);
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

        [[nodiscard]] std::string ffmpegError(const int result) {
            char buffer[AV_ERROR_MAX_STRING_SIZE]{};
            av_strerror(result, buffer, sizeof(buffer));
            return buffer;
        }

        [[nodiscard]] int estimateFramesToExtract(const ExtractionMode mode,
                                                  const double start_time,
                                                  const double end_time,
                                                  const double target_fps,
                                                  const int64_t source_frame_count,
                                                  const int frame_step) {
            if (mode == ExtractionMode::FPS)
                return static_cast<int>(std::min<std::size_t>(
                    calculateFpsSampleCount(start_time, end_time, target_fps),
                    static_cast<std::size_t>(std::numeric_limits<int>::max())));

            const int64_t source_frames = std::max<int64_t>(1, source_frame_count);
            const int64_t estimate = 1 + (source_frames - 1) / frame_step;
            return static_cast<int>(
                std::min<int64_t>(estimate, std::numeric_limits<int>::max()));
        }

        enum class DecoderPumpResult {
            Frame,
            EndOfStream,
            Error,
        };

        class DecoderPump {
        public:
            DecoderPump(AVFormatContext* format_context, AVCodecContext* codec_context,
                        AVPacket* packet, const int stream_index)
                : format_context_(format_context),
                  codec_context_(codec_context),
                  packet_(packet),
                  stream_index_(stream_index) {}

            void resetAfterSeek() {
                av_packet_unref(packet_);
                packet_pending_ = false;
                demux_eof_ = false;
                decoder_drain_sent_ = false;
            }

            [[nodiscard]] DecoderPumpResult next(AVFrame* const frame, std::string& error) {
                while (true) {
                    av_frame_unref(frame);
                    const int receive_result = avcodec_receive_frame(codec_context_, frame);
                    if (receive_result == 0)
                        return DecoderPumpResult::Frame;
                    if (receive_result == AVERROR_EOF)
                        return DecoderPumpResult::EndOfStream;
                    if (receive_result != AVERROR(EAGAIN)) {
                        error = "Video decoder receive failed: " + ffmpegError(receive_result);
                        return DecoderPumpResult::Error;
                    }

                    if (packet_pending_) {
                        const int send_result = avcodec_send_packet(codec_context_, packet_);
                        if (send_result == 0) {
                            av_packet_unref(packet_);
                            packet_pending_ = false;
                            continue;
                        }
                        if (send_result == AVERROR(EAGAIN))
                            continue;
                        av_packet_unref(packet_);
                        packet_pending_ = false;
                        error = "Video decoder packet submission failed: " +
                                ffmpegError(send_result);
                        return DecoderPumpResult::Error;
                    }

                    if (demux_eof_) {
                        if (decoder_drain_sent_)
                            return DecoderPumpResult::EndOfStream;
                        const int drain_result = avcodec_send_packet(codec_context_, nullptr);
                        if (drain_result == 0) {
                            decoder_drain_sent_ = true;
                            continue;
                        }
                        if (drain_result == AVERROR_EOF)
                            return DecoderPumpResult::EndOfStream;
                        if (drain_result == AVERROR(EAGAIN))
                            continue;
                        error = "Video decoder drain failed: " + ffmpegError(drain_result);
                        return DecoderPumpResult::Error;
                    }

                    while (true) {
                        const int read_result = av_read_frame(format_context_, packet_);
                        if (read_result == AVERROR_EOF) {
                            demux_eof_ = true;
                            break;
                        }
                        if (read_result < 0) {
                            error = "Video demux failed: " + ffmpegError(read_result);
                            return DecoderPumpResult::Error;
                        }
                        if (packet_->stream_index == stream_index_) {
                            packet_pending_ = true;
                            break;
                        }
                        av_packet_unref(packet_);
                    }
                }
            }

        private:
            AVFormatContext* format_context_;
            AVCodecContext* codec_context_;
            AVPacket* packet_;
            int stream_index_;
            bool packet_pending_ = false;
            bool demux_eof_ = false;
            bool decoder_drain_sent_ = false;
        };

        struct FrameSelectionDecision {
            bool use_sparse_seek;
            const char* reason;
        };

        [[nodiscard]] FrameSelectionDecision chooseFrameSelection(
            const VideoFrameExtractor::Params& params, const double source_fps,
            const double target_fps, const bool timestamps_available,
            const bool using_hw_decode, const int dolby_vision_profile) {
            if (params.sharpness.window_mode)
                return {false, "sharpness_window_requires_candidate_scan"};
            if (params.sharpness.enabled)
                return {false, "sharpness_filter_uses_sequential_scan"};
            if (params.mode != ExtractionMode::FPS)
                return {false, "interval_mode_preserves_frame_index_semantics"};
            if (dolby_vision_profile > 0)
                return {false, "dolby_vision_requires_sequential_decode"};
            if (!using_hw_decode)
                return {false, "software_decoder_requires_sequential_decode"};
            if (!(target_fps > 0.0 && source_fps > 0.0 &&
                  target_fps * 3.0 < source_fps)) {
                return {false, "request_is_not_sparse"};
            }
            if (!timestamps_available)
                return {false, "timestamps_unavailable"};
            return {true, "sparse_fps_keyframe_seek"};
        }

        void write_jpeg_to_file(const std::filesystem::path& path, const std::vector<uint8_t>& data) {
            std::ofstream file(path, std::ios::binary);
            if (!file)
                throw std::runtime_error("Failed to open JPEG output: " + lfs::core::path_to_utf8(path));
            file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
            file.flush();
            if (!file)
                throw std::runtime_error("Failed to write JPEG output: " + lfs::core::path_to_utf8(path));
            file.close();
            if (!file)
                throw std::runtime_error("Failed to close JPEG output: " + lfs::core::path_to_utf8(path));
        }

        const char* get_hw_decoder_name(AVCodecID codec_id) {
            switch (codec_id) {
            case AV_CODEC_ID_H264:
                return "h264_cuvid";
            case AV_CODEC_ID_HEVC:
                return "hevc_cuvid";
            case AV_CODEC_ID_VP8:
                return "vp8_cuvid";
            case AV_CODEC_ID_VP9:
                return "vp9_cuvid";
            case AV_CODEC_ID_AV1:
                return "av1_cuvid";
            case AV_CODEC_ID_MPEG1VIDEO:
                return "mpeg1_cuvid";
            case AV_CODEC_ID_MPEG2VIDEO:
                return "mpeg2_cuvid";
            case AV_CODEC_ID_MPEG4:
                return "mpeg4_cuvid";
            case AV_CODEC_ID_VC1:
                return "vc1_cuvid";
            default:
                return nullptr;
            }
        }

        struct HwDecoderState {
            AVPixelFormat pixel_format = AV_PIX_FMT_NONE;
        };

        AVPixelFormat get_hw_format(AVCodecContext* context, const AVPixelFormat* pix_fmts) {
            const auto* const state = static_cast<const HwDecoderState*>(context->opaque);
            if (!state)
                return AV_PIX_FMT_NONE;
            for (const AVPixelFormat* p = pix_fmts; *p != AV_PIX_FMT_NONE; ++p) {
                if (*p == state->pixel_format)
                    return *p;
            }
#if defined(__APPLE__)
            // A decoder can advertise VideoToolbox yet reject a particular stream profile.
            // Keep FFmpeg's software decoder usable when no hardware format is offered.
            if (state->pixel_format == AV_PIX_FMT_VIDEOTOOLBOX) {
                for (const AVPixelFormat* p = pix_fmts; *p != AV_PIX_FMT_NONE; ++p) {
                    const AVPixFmtDescriptor* const descriptor = av_pix_fmt_desc_get(*p);
                    if (descriptor && !(descriptor->flags & AV_PIX_FMT_FLAG_HWACCEL)) {
                        LOG_WARN("VideoToolbox format unavailable for this stream; using FFmpeg software decode");
                        return *p;
                    }
                }
            }
#endif
            return AV_PIX_FMT_NONE;
        }

        [[nodiscard]] AVPixelFormat hardwareFrameSoftwareFormat(const AVFrame* const frame) {
            if (!frame || !frame->hw_frames_ctx)
                return AV_PIX_FMT_NONE;

            const auto* const frames_ctx =
                reinterpret_cast<const AVHWFramesContext*>(frame->hw_frames_ctx->data);
            if (!frames_ctx)
                return AV_PIX_FMT_NONE;

            return static_cast<AVPixelFormat>(frames_ctx->sw_format);
        }

        [[nodiscard]] const char* pixelFormatName(const AVPixelFormat format) {
            const char* const name = av_get_pix_fmt_name(format);
            return name ? name : "unknown";
        }

        [[nodiscard]] double computeSharpnessScore(const uint8_t* rgb,
                                                   const int w,
                                                   const int h,
                                                   const SharpnessAlgorithm algo) {
            const long long total_pixels = static_cast<long long>(w) * h;
            // Laplacian threshold: pixel needs Laplacian > 10 to count as edge
            // Tenengrad threshold: pixel needs Sobel energy > 40 to count as edge (4x)
            const int lap_threshold = 10;
            const int ten_threshold = 40;
            long long edge_count = 0;

            if (algo == SharpnessAlgorithm::COMBINED) {
                // Single pass: count if EITHER condition is met (no double counting)
                for (int y = 1; y < h - 1; ++y) {
                    for (int x = 1; x < w - 1; ++x) {
                        const uint8_t* const p = rgb + (y * w + x) * 3 + 1;
                        const int lap = std::abs(static_cast<int>(p[0] * 4) - p[-w * 3] - p[3] - p[-3] - p[w * 3]);
                        if (lap > lap_threshold) {
                            ++edge_count;
                            continue;
                        }
                        const int gx = -p[-w * 3 - 3] + p[-w * 3 + 3] - p[-3] * 2 + p[3] * 2 - p[+w * 3 - 3] + p[+w * 3 + 3];
                        const int gy = -p[-w * 3 - 3] - p[-w * 3] * 2 - p[-w * 3 + 3] + p[+w * 3 - 3] + p[+w * 3] * 2 + p[+w * 3 + 3];
                        if (std::abs(gx) + std::abs(gy) > ten_threshold)
                            ++edge_count;
                    }
                }
            } else if (algo == SharpnessAlgorithm::LAPLACIAN) {
                for (int y = 1; y < h - 1; ++y) {
                    for (int x = 1; x < w - 1; ++x) {
                        const uint8_t* const p = rgb + (y * w + x) * 3 + 1;
                        if (std::abs(static_cast<int>(p[0] * 4) - p[-w * 3] - p[3] - p[-3] - p[w * 3]) > lap_threshold)
                            ++edge_count;
                    }
                }
            } else { // TENENGRAD
                for (int y = 1; y < h - 1; ++y) {
                    for (int x = 1; x < w - 1; ++x) {
                        const uint8_t* const p = rgb + (y * w + x) * 3 + 1;
                        const int gx = -p[-w * 3 - 3] + p[-w * 3 + 3] - p[-3] * 2 + p[3] * 2 - p[+w * 3 - 3] + p[+w * 3 + 3];
                        const int gy = -p[-w * 3 - 3] - p[-w * 3] * 2 - p[-w * 3 + 3] + p[+w * 3 - 3] + p[+w * 3] * 2 + p[+w * 3 + 3];
                        if (std::abs(gx) + std::abs(gy) > ten_threshold)
                            ++edge_count;
                    }
                }
            }

            // Edge ratio: percentage of pixels that are part of a sharp edge (0-100)
            return static_cast<double>(edge_count) * 100.0 / static_cast<double>(total_pixels);
        }

    } // namespace

    std::size_t calculateFpsSampleCount(const double start_time, const double end_time,
                                        const double target_fps) {
        if (!std::isfinite(start_time) || !std::isfinite(end_time) ||
            !std::isfinite(target_fps) || target_fps <= 0.0 || end_time <= start_time) {
            return 0;
        }

        const long double scaled_duration =
            static_cast<long double>(end_time - start_time) *
            static_cast<long double>(target_fps);
        const long double exclusive_sample_count =
            std::nextafter(scaled_duration, -std::numeric_limits<long double>::infinity());
        if (exclusive_sample_count >=
            static_cast<long double>(std::numeric_limits<std::size_t>::max() - 1)) {
            return std::numeric_limits<std::size_t>::max();
        }
        return static_cast<std::size_t>(std::floor(exclusive_sample_count)) + 1;
    }

    double fpsSampleTime(const double start_time, const double end_time,
                         const double target_fps, const std::size_t sample_index) {
        const long double sample =
            static_cast<long double>(start_time) +
            static_cast<long double>(sample_index) / target_fps;
        const double rounded_sample = static_cast<double>(sample);
        return rounded_sample < end_time
                   ? rounded_sample
                   : std::nextafter(end_time, start_time);
    }

    bool frameCoversSampleTime(const double frame_time, const double frame_duration,
                               const double sample_time) {
        return std::isfinite(frame_time) && std::isfinite(frame_duration) &&
               std::isfinite(sample_time) && frame_duration > 0.0 &&
               sample_time >= frame_time &&
               static_cast<long double>(sample_time) <
                   static_cast<long double>(frame_time) + frame_duration;
    }

    bool shouldFillRetainedFpsTail(const bool reached_eof, const bool reached_end) {
        return reached_eof || reached_end;
    }

    bool VideoFrameExtractor::validateParams(const Params& params, const int source_width,
                                             const int source_height,
                                             const double stream_time_base,
                                             ValidatedLayout& layout, std::string& error) {
        layout = {};
        error.clear();
        if (source_width <= 0 || source_height <= 0) {
            error = "Source video dimensions must be positive";
            return false;
        }
        if (source_width > std::numeric_limits<int>::max() / 3 ||
            source_height > std::numeric_limits<int>::max() / 3) {
            error = "Source video dimensions exceed supported row-stride limits";
            return false;
        }
        const std::size_t source_width_size =
            static_cast<std::size_t>(source_width);
        const std::size_t source_height_size =
            static_cast<std::size_t>(source_height);
        if (source_width_size >
                std::numeric_limits<std::size_t>::max() / source_height_size ||
            source_width_size * source_height_size >
                std::numeric_limits<std::size_t>::max() / 3) {
            error = "Source RGB frame size exceeds addressable memory";
            return false;
        }
        if (source_width_size * source_height_size >
            static_cast<std::size_t>(std::numeric_limits<int>::max() / 3)) {
            error = "Source video dimensions exceed supported pixel-index limits";
            return false;
        }
        if (!std::isfinite(stream_time_base) || stream_time_base <= 0.0) {
            error = "Video stream time base must be finite and positive";
            return false;
        }
        if (!std::isfinite(params.start_time) || params.start_time < 0.0 ||
            !std::isfinite(params.end_time) ||
            (params.end_time != -1.0 && params.end_time <= params.start_time)) {
            error = "Invalid video trim range";
            return false;
        }
        if (params.rotation != 0 && params.rotation != 90 &&
            params.rotation != 180 && params.rotation != 270) {
            error = "Video rotation must be 0, 90, 180, or 270 degrees";
            return false;
        }
        if (params.jpg_quality < 1 || params.jpg_quality > 100) {
            error = "JPEG quality must be between 1 and 100";
            return false;
        }
        if (!std::isfinite(params.sharpness.threshold) ||
            params.sharpness.threshold < 0.0 ||
            params.sharpness.window_candidates_target < -1) {
            error = "Invalid sharpness parameters";
            return false;
        }

        switch (params.mode) {
        case ExtractionMode::FPS:
            if (!std::isfinite(params.fps) || params.fps <= 0.0) {
                error = "Extraction FPS must be finite and positive";
                return false;
            }
            if (!std::isfinite(1.0 / params.fps) ||
                params.start_time + 1.0 / params.fps <= params.start_time) {
                error = "Extraction FPS exceeds timestamp precision";
                return false;
            }
            if (params.end_time >= 0.0 &&
                calculateFpsSampleCount(params.start_time, params.end_time,
                                        params.fps) >
                    static_cast<std::size_t>(
                        std::numeric_limits<int>::max())) {
                error = "Requested frame count exceeds supported limits";
                return false;
            }
            break;
        case ExtractionMode::INTERVAL:
            if (params.frame_interval <= 0) {
                error = "Frame interval must be positive";
                return false;
            }
            break;
        default:
            error = "Invalid frame extraction mode";
            return false;
        }

        int output_width = source_width;
        int output_height = source_height;
        switch (params.resolution_mode) {
        case ResolutionMode::Original:
            break;
        case ResolutionMode::Scale: {
            if (!std::isfinite(params.scale) || params.scale <= 0.0f) {
                error = "Resolution scale must be finite and positive";
                return false;
            }
            const long double scaled_width =
                static_cast<long double>(source_width) * params.scale;
            const long double scaled_height =
                static_cast<long double>(source_height) * params.scale;
            const long double maximum_even_input =
                static_cast<long double>(std::numeric_limits<int>::max() - 1);
            if (scaled_width < 1.0L || scaled_height < 1.0L ||
                scaled_width > maximum_even_input ||
                scaled_height > maximum_even_input) {
                error = "Scaled video dimensions are out of range";
                return false;
            }
            output_width = (static_cast<int>(scaled_width) + 1) & ~1;
            output_height = (static_cast<int>(scaled_height) + 1) & ~1;
            break;
        }
        case ResolutionMode::Custom:
            if (params.custom_width <= 0 || params.custom_height <= 0) {
                error = "Custom video dimensions must be positive";
                return false;
            }
            output_width = params.custom_width;
            output_height = params.custom_height;
            break;
        default:
            error = "Invalid video resolution mode";
            return false;
        }

        const std::size_t width = static_cast<std::size_t>(output_width);
        const std::size_t height = static_cast<std::size_t>(output_height);
        if (output_width > std::numeric_limits<int>::max() / 3 ||
            output_height > std::numeric_limits<int>::max() / 3) {
            error = "Video dimensions exceed supported row-stride limits";
            return false;
        }
        if (width > std::numeric_limits<std::size_t>::max() / height) {
            error = "Video pixel count exceeds addressable memory";
            return false;
        }
        const std::size_t pixels = width * height;
        if (pixels > std::numeric_limits<std::size_t>::max() / 3) {
            error = "RGB frame size exceeds addressable memory";
            return false;
        }
        if (pixels >
            static_cast<std::size_t>(std::numeric_limits<int>::max() / 3)) {
            error = "Video dimensions exceed supported pixel-index limits";
            return false;
        }
        const long double maximum_timestamp_seconds =
            static_cast<long double>(std::numeric_limits<int64_t>::max()) *
            stream_time_base;
        if (static_cast<long double>(params.start_time) >
                maximum_timestamp_seconds ||
            (params.end_time >= 0.0 &&
             static_cast<long double>(params.end_time) >
                 maximum_timestamp_seconds)) {
            error = "Video trim range exceeds timestamp limits";
            return false;
        }

        layout = {
            .width = output_width,
            .height = output_height,
            .rgb_bytes = pixels * 3,
        };
        return true;
    }

    std::string formatFrameFilenameStem(const std::string_view pattern, const int frame_number) {
        const std::string_view effective_pattern = pattern.empty() ? std::string_view{"frame_%d"} : pattern;
        std::string out;
        out.reserve(effective_pattern.size() + 8);

        bool consumed_value = false;
        for (size_t i = 0; i < effective_pattern.size(); ++i) {
            if (effective_pattern[i] != '%' || i + 1 >= effective_pattern.size()) {
                out.push_back(effective_pattern[i]);
                continue;
            }

            if (effective_pattern[i + 1] == '%') {
                out.push_back('%');
                ++i;
                continue;
            }

            size_t j = i + 1;
            int min_width = 0;
            bool found_value = false;

            if (effective_pattern[j] == 'd') {
                found_value = true;
            } else if (effective_pattern[j] == '0') {
                size_t zero_count = 0;
                while (j < effective_pattern.size() && effective_pattern[j] == '0') {
                    ++zero_count;
                    ++j;
                }

                int parsed_width = 0;
                while (j < effective_pattern.size() && std::isdigit(static_cast<unsigned char>(effective_pattern[j]))) {
                    if (parsed_width < MAX_FILENAME_FRAME_WIDTH)
                        parsed_width = std::min(parsed_width * 10 + (effective_pattern[j] - '0'),
                                                MAX_FILENAME_FRAME_WIDTH);
                    ++j;
                }

                const bool has_d_suffix = j < effective_pattern.size() && effective_pattern[j] == 'd';
                const bool has_legacy_zero_run =
                    parsed_width == 0 && zero_count > 1 &&
                    (j >= effective_pattern.size() ||
                     !std::isalpha(static_cast<unsigned char>(effective_pattern[j])));
                found_value = has_d_suffix || has_legacy_zero_run;

                if (found_value) {
                    min_width = parsed_width > 0
                                    ? parsed_width
                                    : static_cast<int>(has_legacy_zero_run || zero_count > 1
                                                           ? std::min<size_t>(zero_count + 1, MAX_FILENAME_FRAME_WIDTH)
                                                           : 0);
                    if (!has_d_suffix)
                        --j;
                }
            }

            if (found_value) {
                appendFrameNumber(out, frame_number, min_width);
                i = j;
                consumed_value = true;
                continue;
            }

            out.push_back('%');
        }

        if (!consumed_value)
            appendFrameNumber(out, frame_number, 0);

        return out;
    }

    bool isGeneratedFrameFilename(const std::filesystem::path& path, std::string_view pattern) {
        auto extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (extension != ".png" && extension != ".jpg" && extension != ".jpeg" && extension != ".exr")
            return false;
        const auto stem = core::path_to_utf8(path.stem());
        // Reuse the actual formatter, including padding, %% and legacy patterns.
        for (size_t start = 0; start < stem.size(); ++start) {
            int value = 0;
            for (size_t end = start; end < stem.size() && std::isdigit(static_cast<unsigned char>(stem[end])); ++end) {
                const int digit = stem[end] - '0';
                if (value > (std::numeric_limits<int>::max() - digit) / 10)
                    break;
                value = value * 10 + digit;
                if (value > 0 && formatFrameFilenameStem(pattern, value) == stem)
                    return true;
            }
        }
        return false;
    }

    class VideoFrameExtractor::Impl {
    public:
        bool extract(const Params& params, std::string& error, media::FrameSink* provided_sink = nullptr, bool legacy_file_policy = false) {
            outcome_ = ExtractionOutcome::Failed;
            failure_.reset();
            error.clear();
            const bool float_output = params.output_format == media::FramePixelFormat::RGBFloat32;
            const size_t pixel_bytes = float_output ? 12 : 3;
            if ((params.output_format != media::FramePixelFormat::RGB8 && !float_output) ||
                (float_output && (!provided_sink || params.convert_hdr_to_sdr)) ||
                (params.format == ImageFormat::EXR && !float_output)) {
                failure_ = make_error({.code = ErrorCode::Unsupported, .domain = ErrorDomain::IO, .detail = "EXR requires float samples; float extraction requires a CPU sink and the SDR profile", .detection = LFS_SOURCE_SITE_CURRENT()});
                error = std::string(failure_->detail());
                return false;
            }
            const bool external_sink = provided_sink != nullptr;
            const bool custom_sink = external_sink && !legacy_file_policy;
            media::FileFrameSink file_sink({params.output_dir, params.filename_pattern,
                                            params.format == ImageFormat::PNG ? media::FrameFileFormat::PNG : media::FrameFileFormat::JPEG,
                                            params.jpg_quality});
            auto& sink = provided_sink ? *provided_sink : static_cast<media::FrameSink&>(file_sink);
            bool sink_started = false;
            size_t accepted_frames = 0;
            const auto check_sink = [](const media::SinkResult& result) {
                if (!result && result.error().code() == ErrorCode::Cancelled)
                    throw ExtractionCancelled{};
                if (!result)
                    throw std::runtime_error("Frame sink failed: " + std::string(result.error().detail()));
            };
            const auto abort_sink = [&](media::SinkOutcome outcome) {
                if (sink_started) {
                    sink.abort({outcome, accepted_frames, error});
                    sink_started = false;
                }
            };
            const auto extraction_started = std::chrono::steady_clock::now();
            AVFormatContext* fmt_ctx = nullptr;
            AVCodecContext* codec_ctx = nullptr;
            SwsContext* sws_ctx = nullptr;
            AVFrame* frame = nullptr;
            AVFrame* sw_frame = nullptr;
            AVFrame* sparse_previous_frame = nullptr;
            AVPacket* packet = nullptr;
            AVBufferRef* hw_device_ctx = nullptr;

            uint8_t* cpu_contiguous_buffer = nullptr;
            std::vector<uint8_t> rot_buf;
            std::unique_ptr<media::detail::GpuJpegEncoder> gpu_jpeg;
            bool using_hw_decode = false;
            AVHWDeviceType hw_device_type = AV_HWDEVICE_TYPE_NONE;
            HwDecoderState hw_decoder_state;
            const char* decoder_backend = "ffmpeg_software";

            const auto cleanup = [&]() {
                gpu_jpeg.reset();
                if (sws_ctx)
                    sws_freeContext(sws_ctx);
                av_frame_free(&frame);
                av_frame_free(&sw_frame);
                av_frame_free(&sparse_previous_frame);
                av_packet_free(&packet);
                avcodec_free_context(&codec_ctx);
                av_buffer_unref(&hw_device_ctx);
                avformat_close_input(&fmt_ctx);
                delete[] cpu_contiguous_buffer;
                cpu_contiguous_buffer = nullptr;
            };

            try {
                const std::string video_path_utf8 = lfs::core::path_to_utf8(params.video_path);

                if (avformat_open_input(&fmt_ctx, video_path_utf8.c_str(), nullptr,
                                        nullptr) < 0) {
                    error = "Failed to open video file";
                    return false;
                }

                const auto probe = media::detail::probeVideoStream(fmt_ctx);
                const int video_stream_idx = probe.stream_index;
                if (video_stream_idx < 0) {
                    error = probe.ffmpeg_error < 0 ? "Failed to find stream info" : "No video stream found";
                    avformat_close_input(&fmt_ctx);
                    return false;
                }
                AVStream* video_stream = fmt_ctx->streams[video_stream_idx];
                if (!probe.metadata_complete)
                    LOG_WARN("Could not complete video-only stream metadata probe; some source metadata may be unavailable");
                av_seek_frame(fmt_ctx, video_stream_idx, 0, AVSEEK_FLAG_BACKWARD);
                const AVCodecID codec_id = video_stream->codecpar->codec_id;
                int dv_profile = 0;
                int dv_compatibility = 0;
                for (int i = 0; i < video_stream->codecpar->nb_coded_side_data; ++i) {
                    const AVPacketSideData* side_data = &video_stream->codecpar->coded_side_data[i];
                    if (side_data->type == AV_PKT_DATA_DOVI_CONF &&
                        side_data->size >= static_cast<int>(sizeof(AVDOVIDecoderConfigurationRecord))) {
                        const auto* dovi = reinterpret_cast<const AVDOVIDecoderConfigurationRecord*>(side_data->data);
                        dv_profile = dovi->dv_profile;
                        dv_compatibility = dovi->dv_bl_signal_compatibility_id;
                        break;
                    }
                }
                const int source_bit_depth = pixelFormatBitDepth(video_stream->codecpar);
                const bool has_mastering_metadata =
                    hasCodedSideData(video_stream->codecpar,
                                     AV_PKT_DATA_MASTERING_DISPLAY_METADATA);
                const bool has_content_light_metadata =
                    hasCodedSideData(video_stream->codecpar,
                                     AV_PKT_DATA_CONTENT_LIGHT_LEVEL);
                const HdrFormat hdr_format = dv_profile > 0
                                                 ? detectDolbyVisionFormat(video_stream->codecpar->color_trc, dv_profile, dv_compatibility)
                                                 : detectHdrFormat(video_stream->codecpar->color_trc, source_bit_depth,
                                                                   has_mastering_metadata, has_content_light_metadata);
                const bool convert_hdr_to_sdr = params.convert_hdr_to_sdr && isHdrTonemapSupported(hdr_format);
                AVColorSpace source_colorspace = video_stream->codecpar->color_space;
                AVColorRange source_range = video_stream->codecpar->color_range;
                if (convert_hdr_to_sdr && source_colorspace == AVCOL_SPC_UNSPECIFIED)
                    source_colorspace = AVCOL_SPC_BT2020_NCL;
                if (convert_hdr_to_sdr && source_range == AVCOL_RANGE_UNSPECIFIED)
                    source_range = AVCOL_RANGE_MPEG;

                const double stream_time_base = av_q2d(video_stream->time_base);
                VideoFrameExtractor::ValidatedLayout layout;
                std::string validation_error;
                if (!VideoFrameExtractor::validateParams(
                        params, video_stream->codecpar->width,
                        video_stream->codecpar->height, stream_time_base, layout,
                        validation_error)) {
                    error = "Invalid extraction parameters: " + validation_error;
                    avformat_close_input(&fmt_ctx);
                    return false;
                }

                // Decode Dolby Vision in software to preserve per-frame RPU metadata.
                const AVCodec* codec = nullptr;
                const char* hw_decoder_name = dv_profile > 0 || !media::detail::hasGpuJpegBackend() ? nullptr : get_hw_decoder_name(codec_id);
                if (!custom_sink && !float_output && params.allow_hardware_decode && hw_decoder_name) {
                    codec = avcodec_find_decoder_by_name(hw_decoder_name);
                    if (codec) {
                        if (av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr,
                                                   nullptr, 0) == 0) {
                            using_hw_decode = true;
                            hw_device_type = AV_HWDEVICE_TYPE_CUDA;
                            hw_decoder_state.pixel_format = AV_PIX_FMT_CUDA;
                            decoder_backend = "nvdec";
                            LOG_INFO("Using NVDEC hardware decoder: {}", hw_decoder_name);
                        } else {
                            codec = nullptr;
                            LOG_WARN("Failed to create CUDA device context, falling back to CPU");
                        }
                    }
                }

#if defined(__APPLE__)
                if (!custom_sink && !float_output && params.allow_hardware_decode && dv_profile == 0 && !codec) {
                    const AVCodec* const software_codec = avcodec_find_decoder(codec_id);
                    if (software_codec) {
                        for (int i = 0;; ++i) {
                            const AVCodecHWConfig* const config =
                                avcodec_get_hw_config(software_codec, i);
                            if (!config)
                                break;
                            if (!(config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) ||
                                config->device_type != AV_HWDEVICE_TYPE_VIDEOTOOLBOX)
                                continue;
                            if (av_hwdevice_ctx_create(&hw_device_ctx,
                                                       AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
                                                       nullptr, nullptr, 0) == 0) {
                                codec = software_codec;
                                using_hw_decode = true;
                                hw_device_type = AV_HWDEVICE_TYPE_VIDEOTOOLBOX;
                                hw_decoder_state.pixel_format = config->pix_fmt;
                                decoder_backend = "videotoolbox";
                                LOG_INFO("Using VideoToolbox hardware decoder: {}",
                                         software_codec->name);
                            } else {
                                LOG_WARN("Failed to create VideoToolbox device context; falling back to software");
                            }
                            break;
                        }
                    }
                }
#endif

                if (!codec) {
                    codec = avcodec_find_decoder(codec_id);
                    if (!codec) {
                        error = "Unsupported codec";
                        if (hw_device_ctx)
                            av_buffer_unref(&hw_device_ctx);
                        avformat_close_input(&fmt_ctx);
                        return false;
                    }
                    LOG_INFO("Using {} decoder", dv_profile > 0
                                                     ? "FFmpeg Dolby Vision metadata-preserving"
                                                     : "CPU software");
                }

                while (true) {
                    codec_ctx = avcodec_alloc_context3(codec);
                    if (!codec_ctx) {
                        error = "Failed to allocate codec context";
                        if (hw_device_ctx)
                            av_buffer_unref(&hw_device_ctx);
                        avformat_close_input(&fmt_ctx);
                        return false;
                    }

                    if (avcodec_parameters_to_context(codec_ctx, video_stream->codecpar) < 0) {
                        error = "Failed to copy codec parameters";
                        avcodec_free_context(&codec_ctx);
                        if (hw_device_ctx)
                            av_buffer_unref(&hw_device_ctx);
                        avformat_close_input(&fmt_ctx);
                        return false;
                    }

                    if (using_hw_decode) {
                        codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
                        if (!codec_ctx->hw_device_ctx) {
                            error = "Failed to retain hardware video decoder context";
                            avcodec_free_context(&codec_ctx);
                            av_buffer_unref(&hw_device_ctx);
                            avformat_close_input(&fmt_ctx);
                            return false;
                        }
                        codec_ctx->opaque = &hw_decoder_state;
                        codec_ctx->get_format = get_hw_format;
                    } else {
                        const unsigned int hardware_threads = std::max(1U, std::thread::hardware_concurrency());
                        codec_ctx->thread_count = std::min(MAX_SW_DECODE_THREADS,
                                                           static_cast<int>(hardware_threads));
                        codec_ctx->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
                        LOG_INFO("FFmpeg software decoder threads: {}", codec_ctx->thread_count);
                    }
#ifdef AV_CODEC_EXPORT_DATA_DOVI_RPU
                    codec_ctx->export_side_data |= AV_CODEC_EXPORT_DATA_DOVI_RPU;
#endif

                    if (avcodec_open2(codec_ctx, codec, nullptr) >= 0)
                        break;

                    if (!using_hw_decode) {
                        error = "Failed to open codec";
                        avcodec_free_context(&codec_ctx);
                        if (hw_device_ctx)
                            av_buffer_unref(&hw_device_ctx);
                        avformat_close_input(&fmt_ctx);
                        return false;
                    }

                    LOG_WARN("Failed to open {} hardware decoder, falling back to software",
                             av_hwdevice_get_type_name(hw_device_type));
                    avcodec_free_context(&codec_ctx);
                    av_buffer_unref(&hw_device_ctx);
                    using_hw_decode = false;
                    hw_device_type = AV_HWDEVICE_TYPE_NONE;
                    hw_decoder_state.pixel_format = AV_PIX_FMT_NONE;
                    decoder_backend = "ffmpeg_software";
                    codec = avcodec_find_decoder(codec_id);
                    if (!codec) {
                        error = "Unsupported codec";
                        avformat_close_input(&fmt_ctx);
                        return false;
                    }
                    LOG_INFO("Using CPU software decoder");
                }

                const int src_width = codec_ctx->width;
                const int src_height = codec_ctx->height;
                double video_fps = validFrameRate(video_stream->avg_frame_rate);
                if (video_fps <= 0.0)
                    video_fps = validFrameRate(video_stream->r_frame_rate);
                if (video_fps <= 0.0)
                    video_fps = 30.0;
                double video_duration = validDurationSeconds(fmt_ctx->duration, AVRational{1, AV_TIME_BASE});
                if (video_duration <= 0.0)
                    video_duration = validDurationSeconds(video_stream->duration, video_stream->time_base);
                const double time_base = stream_time_base;

                if ((src_width != video_stream->codecpar->width ||
                     src_height != video_stream->codecpar->height) &&
                    !VideoFrameExtractor::validateParams(
                        params, src_width, src_height, time_base, layout,
                        validation_error)) {
                    error = "Invalid extraction parameters: " + validation_error;
                    throw std::invalid_argument(error);
                }
                const int out_width = layout.width;
                const int out_height = layout.height;
                if (float_output && (layout.rgb_bytes > std::numeric_limits<size_t>::max() / 4 || out_width > std::numeric_limits<int>::max() / 12)) {
                    failure_ = make_error({.code = ErrorCode::ResourceExhausted, .domain = ErrorDomain::IO, .detail = std::format("Float output layout exceeds address/stride limits (rgb_bytes={}, width={}, byte_stride_limit={})", layout.rgb_bytes, out_width, std::numeric_limits<int>::max()), .detection = LFS_SOURCE_SITE_CURRENT()});
                    throw std::runtime_error(std::string(failure_->detail()));
                }
                const std::size_t frame_size = layout.rgb_bytes * (float_output ? 4 : 1);
                std::unique_ptr<media::detail::LinearFrameConverter> linear_converter;
                media::FrameColor output_color;
                if (float_output) {
                    if (dv_profile > 0 || video_stream->codecpar->color_trc == AVCOL_TRC_SMPTE2084 || video_stream->codecpar->color_trc == AVCOL_TRC_ARIB_STD_B67 ||
                        video_stream->codecpar->color_trc == AVCOL_TRC_LOG || video_stream->codecpar->color_trc == AVCOL_TRC_LOG_SQRT) {
                        failure_ = make_error({.code = ErrorCode::Unsupported, .domain = ErrorDomain::IO, .detail = "HDR/Dolby Vision is outside the float SDR profile", .detection = LFS_SOURCE_SITE_CURRENT()});
                        throw std::runtime_error(std::string(failure_->detail()));
                    }
                    linear_converter = std::make_unique<media::detail::LinearFrameConverter>(*video_stream->codecpar, params.input_color);
                    const auto color = linear_converter->outputColor();
                    if (!color) {
                        failure_ = color.error();
                        throw std::runtime_error(std::string(failure_->detail()));
                    }
                    output_color = *color;
                }
                const bool needs_scale =
                    out_width != src_width || out_height != src_height;

                const int64_t stream_start_timestamp =
                    video_stream->start_time == AV_NOPTS_VALUE ? 0 : video_stream->start_time;
                const double stream_start_seconds =
                    static_cast<double>(stream_start_timestamp) * time_base;
                const auto stream_timestamp_for_time =
                    [&](const double seconds) {
                        const long double timestamp =
                            static_cast<long double>(stream_start_timestamp) +
                            std::round(static_cast<long double>(seconds) /
                                       time_base);
                        if (timestamp <
                                static_cast<long double>(
                                    std::numeric_limits<int64_t>::min()) ||
                            timestamp >
                                static_cast<long double>(
                                    std::numeric_limits<int64_t>::max())) {
                            throw std::invalid_argument(
                                "Invalid extraction parameters: trim range "
                                "exceeds stream timestamp limits");
                        }
                        return static_cast<int64_t>(timestamp);
                    };

                if (video_duration <= 0.0 && params.end_time < 0.0) {
                    error = "Could not determine video duration";
                    throw std::runtime_error(error);
                }

                const double start_time = params.start_time;
                double end_time =
                    params.end_time < 0.0 ? video_duration : params.end_time;
                if (video_duration > 0.0) {
                    const double duration_tolerance =
                        std::max(1.0e-6, time_base);
                    if (start_time >= video_duration ||
                        end_time > video_duration + duration_tolerance) {
                        error = "Invalid extraction parameters: trim range exceeds video duration";
                        throw std::invalid_argument(error);
                    }
                }
                const double trim_duration = end_time - start_time;
                if (!std::isfinite(trim_duration) || trim_duration <= 0.0) {
                    error = "Invalid extraction parameters: invalid video trim range";
                    throw std::invalid_argument(error);
                }

                int64_t total_frames = video_stream->nb_frames;
                if (total_frames <= 0 || video_duration <= 0.0) {
                    const long double estimated_frames =
                        std::ceil(static_cast<long double>(trim_duration) *
                                  video_fps);
                    total_frames = static_cast<int64_t>(std::min(
                        estimated_frames,
                        static_cast<long double>(
                            std::numeric_limits<int64_t>::max())));
                } else {
                    total_frames = std::max<int64_t>(
                        1, static_cast<int64_t>(
                               static_cast<long double>(total_frames) *
                               trim_duration / video_duration));
                }

                const int frame_step =
                    params.mode == ExtractionMode::INTERVAL
                        ? params.frame_interval
                        : 1;
                const double target_fps =
                    params.mode == ExtractionMode::FPS ? params.fps : video_fps;
                const double target_interval = 1.0 / target_fps;
                double next_capture_time = start_time;
                const std::size_t fps_target_count =
                    params.mode == ExtractionMode::FPS
                        ? calculateFpsSampleCount(start_time, end_time,
                                                  target_fps)
                        : 0;
                if (fps_target_count >
                    static_cast<std::size_t>(
                        std::numeric_limits<int>::max())) {
                    error =
                        "Invalid extraction parameters: requested frame count "
                        "exceeds supported limits";
                    throw std::invalid_argument(error);
                }
                const int estimated_total =
                    params.mode == ExtractionMode::FPS
                        ? static_cast<int>(fps_target_count)
                        : estimateFramesToExtract(
                              params.mode, start_time, end_time, target_fps,
                              total_frames, frame_step);
                // Estimated frames per sliding window (for candidate sampling)
                int window_est_frames = frame_step;
                if (params.mode == ExtractionMode::FPS && video_fps > 0 && target_fps > 0)
                    window_est_frames = static_cast<int>(std::round(video_fps / target_fps));
                window_est_frames = std::max(1, window_est_frames);

                media::SinkSession session;
                if (external_sink)
                    session.source = media::detail::describeContext(fmt_ctx);
                session.source.stream_info_probed = probe.metadata_complete;
                const bool swap_dimensions = params.rotation == 90 || params.rotation == 270;
                session.output = {swap_dimensions ? out_height : out_width,
                                  swap_dimensions ? out_width : out_height,
                                  static_cast<size_t>(swap_dimensions ? out_height : out_width) * pixel_bytes,
                                  params.output_format, output_color};
                session.applied_rotation = params.rotation;
                sink_started = true;
                check_sink(sink.begin(session));

                // Seek to start time if needed
                if (start_time > 0.1) {
                    const int64_t timestamp =
                        stream_timestamp_for_time(start_time);
                    const int seek_result = av_seek_frame(
                        fmt_ctx, video_stream_idx, timestamp,
                        AVSEEK_FLAG_BACKWARD);
                    if (seek_result < 0) {
                        LOG_WARN(
                            "Initial extraction seek failed at {:.2f}s ({}); "
                            "decoding sequentially",
                            start_time, ffmpegError(seek_result));
                    } else {
                        avcodec_flush_buffers(codec_ctx);
                        LOG_INFO("Seeking to start time: {:.2f}s", start_time);
                    }
                }

                if (needs_scale) {
                    LOG_INFO("Output resolution: {}x{} (from {}x{})", out_width, out_height, src_width, src_height);
                }

                frame = av_frame_alloc();
                sw_frame = av_frame_alloc();
                packet = av_packet_alloc();
                if (!frame || !sw_frame || !packet) {
                    error = "Failed to allocate frame/packet";
                    throw std::runtime_error(error);
                }
                DecoderPump decoder_pump(fmt_ctx, codec_ctx, packet, video_stream_idx);

                cpu_contiguous_buffer = new uint8_t[frame_size];

                if (!custom_sink && params.allow_hardware_decode && params.format == ImageFormat::JPG)
                    gpu_jpeg = media::detail::createGpuJpegEncoder({src_width, src_height,
                                                                    out_width, out_height, estimated_total, params.rotation, using_hw_decode,
                                                                    needs_scale, convert_hdr_to_sdr});
                const std::size_t jpeg_batch_size = gpu_jpeg ? gpu_jpeg->capacity() : 0;
                const bool gpu_encoding_enabled = jpeg_batch_size > 0;
                const bool full_gpu_pipeline_available = using_hw_decode && gpu_encoding_enabled &&
                                                         hw_device_type == AV_HWDEVICE_TYPE_CUDA && gpu_jpeg->canConvertHardware() &&
                                                         !needs_scale && !convert_hdr_to_sdr;
                const auto throw_if_cancelled = [&]() {
                    if (params.cancel_requested && params.cancel_requested())
                        throw ExtractionCancelled{};
                };
                std::unique_ptr<HdrLibplaceboRenderer> hdr_renderer;
                std::vector<unsigned char> hdr_sdr_buffer;
                HdrTonemapTiming hdr_timing_total{};
                double cuda_upload_seconds = 0.0;
                double jpeg_encode_seconds = 0.0;
                bool used_gpu_encoding = false;
                double jpeg_write_seconds = 0.0;
                const auto convert_frame_to_output = [&](AVFrame* source) {
                    if (float_output) {
                        const auto converted = linear_converter->convert(*source, out_width, out_height, cpu_contiguous_buffer);
                        if (!converted) {
                            failure_ = converted.error();
                            error = std::string(failure_->detail());
                        }
                        return static_cast<bool>(converted);
                    }
                    if (convert_hdr_to_sdr) {
                        inheritStreamColorimetry(source, video_stream->codecpar, hdr_format);
                        if (!hdr_renderer)
                            hdr_renderer = std::make_unique<HdrLibplaceboRenderer>();
                        std::string renderer_error;
                        HdrTonemapTiming frame_timing{};
                        auto described = media::detail::describeDecodedVideoFrame(source, video_stream);
                        if (!described) {
                            error = std::string(described.error().detail());
                            return false;
                        }
                        if (!hdr_renderer->tonemapToSdr(&*described, hdr_format,
                                                        out_width, out_height,
                                                        hdr_sdr_buffer, renderer_error, &frame_timing)) {
                            LOG_ERROR("HDR extraction renderer failed: {}", renderer_error);
                            error = "HDR extraction renderer failed: " + renderer_error;
                            return false;
                        }
                        hdr_timing_total.initialization_seconds += frame_timing.initialization_seconds;
                        hdr_timing_total.render_seconds += frame_timing.render_seconds;
                        hdr_timing_total.readback_seconds += frame_timing.readback_seconds;
                        hdr_timing_total.rgba_to_rgb_seconds += frame_timing.rgba_to_rgb_seconds;
                        if (hdr_sdr_buffer.size() != frame_size) {
                            error = "HDR extraction renderer returned an invalid frame size";
                            return false;
                        }
                        std::memcpy(cpu_contiguous_buffer, hdr_sdr_buffer.data(), frame_size);
                        return true;
                    }
                    const int source_width =
                        source->width > 0 ? source->width : src_width;
                    const int source_height =
                        source->height > 0 ? source->height : src_height;
                    if (source_width <= 0 || source_height <= 0) {
                        error = "Decoded video frame has invalid dimensions";
                        return false;
                    }
                    sws_ctx = sws_getCachedContext(
                        sws_ctx, source_width, source_height,
                        static_cast<AVPixelFormat>(source->format), out_width,
                        out_height, AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr,
                        nullptr, nullptr);
                    if (!sws_ctx) {
                        error = "Failed to create video scaling context";
                        return false;
                    }
                    if (!configureVideoToRgbColorimetry(
                            sws_ctx, source, source_colorspace,
                            source_range)) {
                        error = "Failed to configure video color conversion";
                        return false;
                    }
                    uint8_t* dst_data[4] = {cpu_contiguous_buffer, nullptr, nullptr, nullptr};
                    int dst_linesize[4] = {out_width * 3, 0, 0, 0};
                    if (sws_scale(sws_ctx, source->data, source->linesize, 0,
                                  source_height, dst_data, dst_linesize) <= 0) {
                        error = "Video color conversion failed";
                        return false;
                    }
                    return true;
                };

                if (full_gpu_pipeline_available) {
                    LOG_INFO("Full GPU pipeline available for NV12 frames: NVDEC decode → GPU color convert → GPU JPEG encode");
                } else if (convert_hdr_to_sdr && gpu_encoding_enabled) {
                    LOG_INFO("HDR hybrid pipeline: {} decode -> libplacebo Vulkan tone map -> host readback -> CUDA JPEG batch",
                             dv_profile > 0 ? "FFmpeg Dolby Vision" : "CPU");
                } else if (using_hw_decode) {
                    LOG_INFO("Hybrid pipeline: {} decode → CPU transfer → {}",
                             decoder_backend,
                             gpu_encoding_enabled ? "GPU encode" : "CPU encode");
                } else if (gpu_encoding_enabled) {
                    LOG_INFO("Using GPU batch JPEG encoding (batch size: {})",
                             jpeg_batch_size);
                } else if (params.format == ImageFormat::JPG) {
                    LOG_INFO("Using CPU JPEG encoding");
                } else {
                    LOG_INFO("Using CPU {} encoding", float_output ? "EXR" : "PNG");
                }

                int in_trim_frame_count = 0;
                int decoded_frame_count = 0;
                int saved_count = 0;
                int skipped_count = 0;
                int written_count = 0;
                int keyframe_seek_count = 0;
                double current_frame_time = 0.0;
                int current_src_frame = 0;
                media::FrameInfo current_info;
                const auto frame_info = [&](const AVFrame* decoded, double seconds) {
                    media::FrameInfo info;
                    const int64_t ticks = decoded->best_effort_timestamp != AV_NOPTS_VALUE ? decoded->best_effort_timestamp : decoded->pts;
                    if (ticks != AV_NOPTS_VALUE) {
                        info.source_timestamp = media::Timestamp{ticks, {video_stream->time_base.num, video_stream->time_base.den}};
                        info.timestamp_origin = decoded->best_effort_timestamp != AV_NOPTS_VALUE ? media::TimestampOrigin::BestEffort : media::TimestampOrigin::Presentation;
                    }
                    info.origin = media::FrameOrigin::Decoded;
                    const auto* description = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(decoded->format));
                    if (description && description->comp[0].depth > 0)
                        info.source_component_depth = description->comp[0].depth;
                    info.decode_index = static_cast<uint64_t>(decoded_frame_count - 1);
                    info.relative_seconds = seconds;
                    info.legacy_source_frame = std::max(1, static_cast<int>(std::llround(seconds * video_fps)) + 1);
                    return info;
                };
                const auto emit_rgb = [&](int width, int height, const void* pixels, media::FrameInfo info, double score) {
                    info.delivery_index = accepted_frames;
                    info.sharpness_score = score;
                    const size_t stride = static_cast<size_t>(width) * pixel_bytes;
                    check_sink(sink.write({{width, height, stride, params.output_format, output_color}, info, std::span<const uint8_t>(static_cast<const uint8_t*>(pixels), stride * height)}));
                    ++accepted_frames;
                };
                std::vector<void*> batch_gpu_ptrs;
                std::vector<std::filesystem::path> batch_filenames;
                struct BatchFrameMeta {
                    double timestamp;
                    int source_frame;
                    double sharpness_score;
                };
                std::vector<BatchFrameMeta> batch_meta;
                std::size_t batch_idx = 0;
                int batch_encode_w = 0, batch_encode_h = 0;
                bool logged_hw_format_fallback = false;
                AVPixelFormat decoded_software_format = AV_PIX_FMT_NONE;
                struct CandidateFrame {
                    std::vector<uint8_t> rgb;
                    std::filesystem::path filename;
                    media::FrameInfo info;
                    double score = 0.0;
                    double timestamp = 0.0;
                    int source_frame = 0;
                };
                std::optional<CandidateFrame> window_best;
                const auto consider_window_frame = [&](const uint8_t* pixels, const media::FrameInfo& info,
                                                       double score, double timestamp, int source_frame) {
                    // max_element retains the first maximum, including equal scores.
                    if (window_best && !(score > window_best->score))
                        return;
                    if (!window_best)
                        window_best.emplace();
                    window_best->rgb.assign(pixels, pixels + frame_size);
                    window_best->info = info;
                    window_best->score = score;
                    window_best->timestamp = timestamp;
                    window_best->source_frame = source_frame;
                };
                std::vector<uint8_t> scoring_proxy;
                int current_window_idx = 0;
                int window_skip_counter = 0;
                int in_window_frame_count = 0;
                struct FrameSaveInfo {
                    std::string filename;
                    double timestamp;
                    int source_frame;
                    double sharpness_score;
                };
                std::vector<FrameSaveInfo> saved_frames;
                std::unordered_set<std::filesystem::path> emitted_filenames;

                const bool seek_timestamps_available =
                    std::isfinite(time_base) && time_base > 0.0 && std::isfinite(video_duration) &&
                    video_duration > 0.0;
                // Dolby Vision RPU application requires continuous sequential decode.
                const FrameSelectionDecision frame_selection = chooseFrameSelection(
                    params, video_fps, target_fps, seek_timestamps_available,
                    using_hw_decode, dv_profile);
                const bool use_sparse_keyframe_seek = frame_selection.use_sparse_seek;
                std::string frame_selection_reason = frame_selection.reason;
                if (use_sparse_keyframe_seek) {
                    LOG_INFO("Sparse FPS extraction: keyframe seek enabled for {} targets; sharpness modes keep sequential decoding",
                             estimated_total);
                } else {
                    LOG_INFO("Frame selection: sequential (reason={}, source_fps={:.3f}, requested_fps={:.3f}, "
                             "estimated_targets={}, estimated_source_frames={})",
                             frame_selection_reason, video_fps, target_fps, estimated_total, total_frames);
                }

                auto flush_jpeg_batch = [&]() {
                    if (batch_gpu_ptrs.empty())
                        return;
                    if (batch_encode_w <= 0 || batch_encode_h <= 0) {
                        throw std::runtime_error("GPU JPEG batch dimensions are invalid");
                    }
                    throw_if_cancelled();

                    const auto jpeg_encode_started = std::chrono::steady_clock::now();
                    auto encoded = gpu_jpeg->encode(batch_gpu_ptrs, batch_encode_w, batch_encode_h,
                                                    params.jpg_quality);
                    jpeg_encode_seconds += elapsedSeconds(jpeg_encode_started);
                    if (encoded.size() != batch_gpu_ptrs.size() ||
                        std::any_of(encoded.begin(), encoded.end(), [](const auto& bytes) { return bytes.empty(); }))
                        throw std::runtime_error("GPU JPEG encoder returned an incomplete batch");

                    for (size_t i = 0; i < encoded.size(); i++) {
                        throw_if_cancelled();
                        if (!encoded[i].empty()) {
                            const auto jpeg_write_started = std::chrono::steady_clock::now();
                            write_jpeg_to_file(batch_filenames[i], encoded[i]);
                            jpeg_write_seconds += elapsedSeconds(jpeg_write_started);
                            ++written_count;
                            ++accepted_frames;
                            used_gpu_encoding = true;
                            if (!custom_sink && params.generate_metadata && i < batch_meta.size()) {
                                saved_frames.push_back({lfs::core::path_to_utf8(batch_filenames[i].filename()),
                                                        batch_meta[i].timestamp,
                                                        batch_meta[i].source_frame,
                                                        batch_meta[i].sharpness_score});
                            }
                        }
                    }

                    batch_gpu_ptrs.clear();
                    batch_filenames.clear();
                    batch_meta.clear();
                    batch_encode_w = 0;
                    batch_encode_h = 0;
                    batch_idx = 0;
                    throw_if_cancelled();
                };

                auto generate_filename = [&](int frame_num) {
                    if (custom_sink)
                        return std::filesystem::path{};
                    std::string ext = params.format == ImageFormat::PNG ? ".png" : params.format == ImageFormat::EXR ? ".exr"
                                                                                                                     : ".jpg";
                    return params.output_dir / (formatFrameFilenameStem(params.filename_pattern, frame_num) + ext);
                };

                const auto source_frame_for_time = [&](const double frame_time) {
                    return std::max(
                        1, static_cast<int>(std::llround(frame_time * video_fps)) + 1);
                };

                auto reserve_output_filename = [&](const std::filesystem::path& filename,
                                                   const int source_frame) {
                    if (custom_sink || emitted_filenames.insert(filename).second)
                        return true;
                    LOG_WARN("Skipping duplicate source frame {} output: {}",
                             source_frame, lfs::core::path_to_utf8(filename));
                    return false;
                };

                auto finish_selected_frame = [&]() {
                    saved_count++;

                    if (params.progress_callback) {
                        params.progress_callback(saved_count + skipped_count, estimated_total, skipped_count);
                    }
                    throw_if_cancelled();
                };

                auto should_extract_frame = [&](const double frame_time) {
                    bool should_extract = false;
                    if (params.mode == ExtractionMode::FPS) {
                        should_extract = frame_time + 1.0e-6 >= next_capture_time;
                        if (should_extract) {
                            do {
                                next_capture_time += target_interval;
                            } while (next_capture_time <= frame_time + 1.0e-6);
                        }
                    } else {
                        should_extract = in_trim_frame_count % frame_step == 0;
                    }
                    in_trim_frame_count++;
                    return should_extract;
                };

                auto flush_window = [&]() {
                    if (!window_best)
                        return;
                    const auto* best = &*window_best;
                    std::filesystem::path fname = generate_filename(best->source_frame);
                    if (!reserve_output_filename(fname, best->source_frame)) {
                        finish_selected_frame();
                        window_best.reset();
                        window_skip_counter = 0;
                        return;
                    }
                    // Geometry preserves every component of RGB8 or float RGB.
                    int write_w = out_width;
                    int write_h = out_height;
                    const uint8_t* write_data = best->rgb.data();
                    if (params.rotation != 0) {
                        rotatePacked(write_data, rot_buf, out_width, out_height, params.rotation, pixel_bytes);
                        if (params.rotation == 90 || params.rotation == 270)
                            std::swap(write_w, write_h);
                        write_data = rot_buf.data();
                    }
                    emit_rgb(write_w, write_h, write_data, best->info, best->score);
                    ++written_count;
                    if (!custom_sink && params.generate_metadata) {
                        saved_frames.push_back({lfs::core::path_to_utf8(fname.filename()), best->timestamp,
                                                best->source_frame, best->score});
                    }
                    finish_selected_frame();
                    window_best.reset();
                    window_skip_counter = 0;
                };

                bool saw_hardware_frame = false;
                auto process_frame_hw = [&](AVFrame* hw_frame) {
                    saw_hardware_frame = true;
                    throw_if_cancelled();
                    std::filesystem::path filename = generate_filename(current_src_frame);

                    const AVPixelFormat hw_sw_format = hardwareFrameSoftwareFormat(hw_frame);
                    if (decoded_software_format == AV_PIX_FMT_NONE &&
                        hw_sw_format != AV_PIX_FMT_NONE) {
                        decoded_software_format = hw_sw_format;
                    }
                    const bool use_full_gpu_pipeline =
                        full_gpu_pipeline_available && hw_sw_format == AV_PIX_FMT_NV12;

                    if (full_gpu_pipeline_available && !use_full_gpu_pipeline &&
                        !logged_hw_format_fallback) {
                        LOG_INFO("Hardware frame format is {}; using CPU transfer/color conversion",
                                 pixelFormatName(hw_sw_format));
                        logged_hw_format_fallback = true;
                    }

                    if (use_full_gpu_pipeline) {
                        const auto hardware_frame = media::detail::cudaFrameView(hw_frame);
                        if (!hardware_frame)
                            throw lfs::Exception(hardware_frame.error());
                        gpu_jpeg->convertHardware(*hardware_frame, params.sharpness.enabled ? cpu_contiguous_buffer : nullptr);
                        double frame_score = 0.0;
                        if (params.sharpness.enabled) {
                            frame_score = computeSharpnessScore(
                                cpu_contiguous_buffer, out_width, out_height, params.sharpness.algorithm);
                            if (params.sharpness.window_mode) {
                                consider_window_frame(cpu_contiguous_buffer, current_info, frame_score, current_frame_time, current_src_frame);
                                gpu_jpeg->finishHardware();
                                return;
                            }
                            if (params.sharpness.threshold > 0.0 && frame_score < params.sharpness.threshold) {
                                ++skipped_count;
                                if (params.progress_callback)
                                    params.progress_callback(saved_count + skipped_count, estimated_total, skipped_count);
                                gpu_jpeg->finishHardware();
                                return;
                            }
                        }

                        if (!reserve_output_filename(filename, current_src_frame)) {
                            finish_selected_frame();
                            gpu_jpeg->finishHardware();
                            return;
                        }

                        if (batch_encode_w == 0) {
                            batch_encode_w = swap_dimensions ? out_height : out_width;
                            batch_encode_h = swap_dimensions ? out_width : out_height;
                        }
                        void* dst_ptr = gpu_jpeg->queueHardware(batch_idx);
                        batch_gpu_ptrs.push_back(dst_ptr);
                        batch_filenames.push_back(filename);
                        batch_meta.push_back({current_frame_time, current_src_frame, frame_score});
                        batch_idx++;

                        if (batch_idx >= jpeg_batch_size) {
                            flush_jpeg_batch();
                        }
                    } else {
                        av_frame_unref(sw_frame);
                        const int transfer_result =
                            av_hwframe_transfer_data(sw_frame, hw_frame, 0);
                        if (transfer_result < 0) {
                            error = "Hardware video frame transfer failed: " +
                                    ffmpegError(transfer_result);
                            throw std::runtime_error(error);
                        }
                        const int props_result = av_frame_copy_props(sw_frame, hw_frame);
                        if (props_result < 0) {
                            av_frame_unref(sw_frame);
                            error = "Hardware video frame metadata copy failed: " +
                                    ffmpegError(props_result);
                            throw std::runtime_error(error);
                        }

                        if (!convert_frame_to_output(sw_frame)) {
                            if (error.empty())
                                error = "Failed to convert decoded video frame";
                            throw std::runtime_error(error);
                        }

                        // --- Sharpness evaluation (hybrid path) ---
                        double frame_score = 0.0;
                        if (params.sharpness.enabled) {
                            frame_score = computeSharpnessScore(
                                cpu_contiguous_buffer, out_width, out_height, params.sharpness.algorithm);
                            if (params.sharpness.window_mode) {
                                consider_window_frame(cpu_contiguous_buffer, current_info, frame_score, current_frame_time, current_src_frame);
                                return;
                            }
                            if (params.sharpness.threshold > 0.0 && frame_score < params.sharpness.threshold) {
                                ++skipped_count;
                                if (params.progress_callback)
                                    params.progress_callback(saved_count + skipped_count, estimated_total, skipped_count);
                                return;
                            }
                        }
                        // --- End sharpness ---

                        if (!reserve_output_filename(filename, current_src_frame)) {
                            finish_selected_frame();
                            return;
                        }

                        // --- Rotation (hybrid HW path) ---
                        int hw_rot_w = out_width;
                        int hw_rot_h = out_height;
                        if (params.rotation != 0) {
                            rot_buf.resize(static_cast<size_t>(out_width) * out_height * 3);
                            if (params.rotation == 180) {
                                for (int y = 0; y < out_height; ++y)
                                    for (int x = 0; x < out_width; ++x) {
                                        const int si = (y * out_width + x) * 3;
                                        const int di = ((out_height - 1 - y) * out_width + (out_width - 1 - x)) * 3;
                                        rot_buf[di + 0] = cpu_contiguous_buffer[si + 0];
                                        rot_buf[di + 1] = cpu_contiguous_buffer[si + 1];
                                        rot_buf[di + 2] = cpu_contiguous_buffer[si + 2];
                                    }
                            } else {
                                const int dst_w = out_height;
                                const int dst_h = out_width;
                                for (int y = 0; y < out_height; ++y)
                                    for (int x = 0; x < out_width; ++x) {
                                        const int si = (y * out_width + x) * 3;
                                        const int di = (params.rotation == 90)
                                                           ? (x * out_height + (out_height - 1 - y)) * 3 // CW
                                                           : ((out_width - 1 - x) * out_height + y) * 3; // CCW
                                        rot_buf[di + 0] = cpu_contiguous_buffer[si + 0];
                                        rot_buf[di + 1] = cpu_contiguous_buffer[si + 1];
                                        rot_buf[di + 2] = cpu_contiguous_buffer[si + 2];
                                    }
                                hw_rot_w = dst_w;
                                hw_rot_h = dst_h;
                            }
                            std::memcpy(cpu_contiguous_buffer, rot_buf.data(),
                                        static_cast<size_t>(out_width) * out_height * 3);
                        }
                        // --- End rotation ---

                        if (gpu_encoding_enabled) {
                            if (batch_encode_w == 0) {
                                batch_encode_w = (hw_rot_w > 0) ? hw_rot_w : out_width;
                                batch_encode_h = (hw_rot_h > 0) ? hw_rot_h : out_height;
                            }
                            const auto cuda_upload_started = std::chrono::steady_clock::now();
                            void* dst_ptr = gpu_jpeg->queueHost(batch_idx, cpu_contiguous_buffer);
                            cuda_upload_seconds += elapsedSeconds(cuda_upload_started);

                            batch_gpu_ptrs.push_back(dst_ptr);
                            batch_filenames.push_back(filename);
                            batch_meta.push_back({current_frame_time, current_src_frame, frame_score});
                            batch_idx++;

                            if (batch_idx >= jpeg_batch_size) {
                                flush_jpeg_batch();
                            }
                        } else {
                            emit_rgb(hw_rot_w, hw_rot_h, cpu_contiguous_buffer, current_info, frame_score);
                            ++written_count;
                            if (!custom_sink && params.generate_metadata) {
                                saved_frames.push_back({lfs::core::path_to_utf8(filename.filename()),
                                                        current_frame_time,
                                                        current_src_frame,
                                                        frame_score});
                            }
                        }
                    }

                    saved_count++;

                    if (params.progress_callback) {
                        params.progress_callback(saved_count + skipped_count, estimated_total, skipped_count);
                    }
                    throw_if_cancelled();
                };

                auto process_frame_sw = [&](AVFrame* decoded_frame) {
                    throw_if_cancelled();
                    if (!convert_frame_to_output(decoded_frame)) {
                        if (error.empty())
                            error = "Failed to convert decoded video frame";
                        throw std::runtime_error(error);
                    }

                    // --- Sharpness evaluation (SW path) ---
                    double frame_score = 0.0;
                    if (params.sharpness.enabled) {
                        const uint8_t* score_pixels = cpu_contiguous_buffer;
                        if (float_output) {
                            // Scoring proxy only. The master remains linear float, including window candidates.
                            scoring_proxy.resize(layout.rgb_bytes);
                            for (size_t i = 0; i < scoring_proxy.size(); ++i) {
                                float value = 0;
                                std::memcpy(&value, cpu_contiguous_buffer + i * sizeof(float), sizeof(float));
                                scoring_proxy[i] = static_cast<uint8_t>(std::lround(std::clamp(value, 0.f, 1.f) * 255.f));
                            }
                            score_pixels = scoring_proxy.data();
                        }
                        frame_score = computeSharpnessScore(score_pixels, out_width, out_height, params.sharpness.algorithm);

                        if (params.sharpness.window_mode) {
                            consider_window_frame(cpu_contiguous_buffer, current_info, frame_score, current_frame_time, current_src_frame);
                            return;
                        }

                        // Threshold mode: discard blurry frames
                        if (params.sharpness.threshold > 0.0 && frame_score < params.sharpness.threshold) {
                            ++skipped_count;
                            if (params.progress_callback)
                                params.progress_callback(saved_count + skipped_count, estimated_total, skipped_count);
                            return;
                        }
                    }
                    // --- End sharpness ---

                    std::filesystem::path filename = generate_filename(current_src_frame);
                    if (!reserve_output_filename(filename, current_src_frame)) {
                        finish_selected_frame();
                        return;
                    }

                    // --- Rotation (SW path) ---
                    int sw_rot_w = out_width;
                    int sw_rot_h = out_height;
                    if (params.rotation != 0) {
                        rotatePacked(cpu_contiguous_buffer, rot_buf, out_width, out_height, params.rotation, pixel_bytes);
                        if (params.rotation == 90 || params.rotation == 270)
                            std::swap(sw_rot_w, sw_rot_h);
                        std::memcpy(cpu_contiguous_buffer, rot_buf.data(), frame_size);
                    }
                    // --- End rotation ---

                    if (gpu_encoding_enabled) {
                        if (batch_encode_w == 0) {
                            batch_encode_w = (sw_rot_w > 0) ? sw_rot_w : out_width;
                            batch_encode_h = (sw_rot_h > 0) ? sw_rot_h : out_height;
                        }
                        const auto cuda_upload_started = std::chrono::steady_clock::now();
                        void* dst_ptr = gpu_jpeg->queueHost(batch_idx, cpu_contiguous_buffer);
                        cuda_upload_seconds += elapsedSeconds(cuda_upload_started);

                        batch_gpu_ptrs.push_back(dst_ptr);
                        batch_filenames.push_back(filename);
                        batch_meta.push_back({current_frame_time, current_src_frame, frame_score});
                        batch_idx++;

                        if (batch_idx >= jpeg_batch_size) {
                            flush_jpeg_batch();
                        }
                    } else {
                        emit_rgb(sw_rot_w, sw_rot_h, cpu_contiguous_buffer, current_info, frame_score);
                        ++written_count;
                        if (!custom_sink && params.generate_metadata) {
                            saved_frames.push_back({lfs::core::path_to_utf8(filename.filename()),
                                                    current_frame_time,
                                                    current_src_frame,
                                                    frame_score});
                        }
                    }

                    finish_selected_frame();
                };

                if (params.mode == ExtractionMode::FPS) {
                    sparse_previous_frame = av_frame_alloc();
                    if (!sparse_previous_frame) {
                        error = "Failed to allocate retained extraction frame";
                        throw std::runtime_error(error);
                    }
                }
                bool has_retained_frame = false;
                media::FrameInfo retained_info;
                double retained_frame_time = 0.0;
                double retained_frame_duration = 0.0;
                const auto retain_frame =
                    [&](AVFrame* const retained_source, const double frame_time,
                        const double frame_duration) {
                        av_frame_unref(sparse_previous_frame);
                        const int ref_result =
                            av_frame_ref(sparse_previous_frame, retained_source);
                        if (ref_result < 0) {
                            error = "Failed to retain extraction frame: " +
                                    ffmpegError(ref_result);
                            throw std::runtime_error(error);
                        }
                        has_retained_frame = true;
                        retained_info = frame_info(retained_source, frame_time);
                        retained_frame_time = frame_time;
                        retained_frame_duration = frame_duration;
                    };

                const auto process_selected_frame = [&](AVFrame* const selected_frame,
                                                        const double frame_time) {
                    current_info = selected_frame == sparse_previous_frame ? retained_info : frame_info(selected_frame, frame_time);
                    current_frame_time = frame_time;
                    current_src_frame = source_frame_for_time(frame_time);
                    if (using_hw_decode &&
                        selected_frame->format == hw_decoder_state.pixel_format)
                        process_frame_hw(selected_frame);
                    else
                        process_frame_sw(selected_frame);
                };

                const auto process_sequential_frame = [&](AVFrame* const decoded_frame) {
                    const double frame_time = frameTimestampSeconds(
                        decoded_frame, time_base, decoded_frame_count - 1,
                        video_fps, stream_start_seconds);
                    if (frame_time < start_time)
                        return false;
                    const bool past_end =
                        params.mode == ExtractionMode::FPS
                            ? frame_time >= end_time
                            : frame_time > end_time;
                    if (past_end)
                        return true;

                    if (params.mode == ExtractionMode::FPS) {
                        const double frame_duration =
                            decoded_frame->duration > 0
                                ? static_cast<double>(decoded_frame->duration) *
                                      time_base
                                : 1.0 / video_fps;
                        retain_frame(decoded_frame, frame_time, frame_duration);
                    }

                    current_info = frame_info(decoded_frame, frame_time);
                    current_frame_time = frame_time;
                    current_src_frame = source_frame_for_time(frame_time);
                    if (params.sharpness.enabled && params.sharpness.window_mode) {
                        const int window_index = params.mode == ExtractionMode::FPS
                                                     ? static_cast<int>(std::floor(
                                                           (frame_time - start_time) / target_interval))
                                                     : in_window_frame_count / frame_step;
                        if (window_index != current_window_idx) {
                            flush_window();
                            current_window_idx = window_index;
                        }

                        ++window_skip_counter;
                        ++in_window_frame_count;
                        int effective_candidates = window_est_frames;
                        if (params.sharpness.window_candidates_target < 0) {
                            const int automatic_target = std::clamp(
                                static_cast<int>(std::round(
                                    std::sqrt(static_cast<double>(window_est_frames)))) *
                                    2,
                                5, 20);
                            effective_candidates =
                                std::min(automatic_target, window_est_frames);
                        } else if (params.sharpness.window_candidates_target > 0) {
                            effective_candidates = std::min(
                                params.sharpness.window_candidates_target,
                                window_est_frames);
                        }
                        if (effective_candidates < window_est_frames) {
                            const int candidate_index = window_skip_counter - 1;
                            const int bucket =
                                candidate_index * effective_candidates / window_est_frames;
                            const int previous_bucket = candidate_index > 0
                                                            ? (candidate_index - 1) * effective_candidates /
                                                                  window_est_frames
                                                            : -1;
                            if (bucket == previous_bucket)
                                return false;
                        }

                        if (using_hw_decode &&
                            decoded_frame->format == hw_decoder_state.pixel_format)
                            process_frame_hw(decoded_frame);
                        else
                            process_frame_sw(decoded_frame);
                    } else if (should_extract_frame(frame_time)) {
                        if (using_hw_decode &&
                            decoded_frame->format == hw_decoder_state.pixel_format)
                            process_frame_hw(decoded_frame);
                        else
                            process_frame_sw(decoded_frame);
                    }
                    return false;
                };

                bool sparse_seek_fallback = false;
                if (use_sparse_keyframe_seek) {
                    for (std::size_t target_index = 0;
                         target_index < fps_target_count; ++target_index) {
                        throw_if_cancelled();
                        const double target_time = fpsSampleTime(
                            start_time, end_time, target_fps, target_index);
                        const int64_t target_timestamp =
                            stream_timestamp_for_time(target_time);
                        const int seek_result = av_seek_frame(
                            fmt_ctx, video_stream_idx, target_timestamp,
                            AVSEEK_FLAG_BACKWARD);
                        if (seek_result < 0) {
                            LOG_WARN(
                                "Sparse extraction seek failed at {:.6f}s ({}); "
                                "continuing sequentially",
                                target_time, ffmpegError(seek_result));
                            sparse_seek_fallback = true;
                            frame_selection_reason =
                                "sparse_seek_failed_sequential_fallback";
                            next_capture_time = target_time;
                            break;
                        }

                        ++keyframe_seek_count;
                        avcodec_flush_buffers(codec_ctx);
                        decoder_pump.resetAfterSeek();
                        av_frame_unref(sparse_previous_frame);
                        has_retained_frame = false;
                        bool found_target = false;

                        while (!found_target) {
                            throw_if_cancelled();
                            const DecoderPumpResult result =
                                decoder_pump.next(frame, error);
                            if (result == DecoderPumpResult::Error)
                                throw std::runtime_error(error);
                            if (result == DecoderPumpResult::EndOfStream) {
                                if (has_retained_frame &&
                                    frameCoversSampleTime(
                                        retained_frame_time,
                                        retained_frame_duration, target_time)) {
                                    process_selected_frame(
                                        sparse_previous_frame,
                                        retained_frame_time);
                                    found_target = true;
                                }
                                break;
                            }

                            ++decoded_frame_count;
                            int64_t timestamp = frame->best_effort_timestamp;
                            if (timestamp == AV_NOPTS_VALUE)
                                timestamp = frame->pts;
                            if (timestamp == AV_NOPTS_VALUE) {
                                error =
                                    "Sparse extraction requires frame timestamps";
                                throw std::runtime_error(error);
                            }

                            const double frame_time =
                                static_cast<double>(timestamp) * time_base -
                                stream_start_seconds;
                            const double frame_duration = frame->duration > 0
                                                              ? static_cast<double>(frame->duration) * time_base
                                                              : 1.0 / video_fps;
                            if (frame_time < target_time) {
                                retain_frame(frame, frame_time, frame_duration);
                                continue;
                            }

                            if (frame_time >= end_time) {
                                if (has_retained_frame &&
                                    frameCoversSampleTime(
                                        retained_frame_time,
                                        retained_frame_duration, target_time)) {
                                    process_selected_frame(
                                        sparse_previous_frame,
                                        retained_frame_time);
                                    found_target = true;
                                }
                                break;
                            }

                            process_selected_frame(frame, frame_time);
                            retain_frame(frame, frame_time, frame_duration);
                            found_target = true;
                        }

                        if (!found_target) {
                            error =
                                "Failed to decode a frame at requested extraction timestamp " +
                                std::to_string(target_time) + "s";
                            throw std::runtime_error(error);
                        }
                    }
                }

                if (!use_sparse_keyframe_seek || sparse_seek_fallback) {
                    bool reached_end = false;
                    bool reached_eof = false;
                    while (!reached_end) {
                        throw_if_cancelled();
                        if (params.mode == ExtractionMode::FPS &&
                            saved_count >= estimated_total) {
                            break;
                        }
                        const DecoderPumpResult result =
                            decoder_pump.next(frame, error);
                        if (result == DecoderPumpResult::EndOfStream) {
                            reached_eof = true;
                            break;
                        }
                        if (result == DecoderPumpResult::Error)
                            throw std::runtime_error(error);
                        ++decoded_frame_count;
                        reached_end = process_sequential_frame(frame);
                    }

                    while (shouldFillRetainedFpsTail(reached_eof, reached_end) &&
                           params.mode == ExtractionMode::FPS &&
                           !params.sharpness.window_mode && has_retained_frame &&
                           saved_count + skipped_count < estimated_total &&
                           next_capture_time < end_time &&
                           frameCoversSampleTime(retained_frame_time,
                                                 retained_frame_duration,
                                                 next_capture_time)) {
                        process_selected_frame(sparse_previous_frame,
                                               retained_frame_time);
                        next_capture_time += target_interval;
                    }
                }

                if (gpu_encoding_enabled) {
                    flush_jpeg_batch();
                }

                // Flush remaining window candidates at end of video
                flush_window();

                if (!custom_sink && params.generate_metadata && !saved_frames.empty()) {
                    try {
                        nlohmann::json root;
                        root["schema_version"] = 2;

                        // Structured fields are intentionally kept alongside
                        // the legacy top-level keys below, so existing tools
                        // can continue reading extraction_metadata.json.
                        AVPixelFormat input_pixel_format =
                            using_hw_decode &&
                                    decoded_software_format != AV_PIX_FMT_NONE
                                ? decoded_software_format
                                : codec_ctx->pix_fmt;
                        if (input_pixel_format == AV_PIX_FMT_NONE)
                            input_pixel_format = static_cast<AVPixelFormat>(video_stream->codecpar->format);
                        const AVPixFmtDescriptor* const input_pixel_descriptor =
                            av_pix_fmt_desc_get(input_pixel_format);
                        const int input_bit_depth = std::max(
                            input_pixel_descriptor
                                ? input_pixel_descriptor->comp[0].depth
                                : 0,
                            video_stream->codecpar->bits_per_raw_sample);
                        const char* const input_primaries_name = av_color_primaries_name(
                            static_cast<AVColorPrimaries>(video_stream->codecpar->color_primaries));
                        const char* const input_transfer_name = av_color_transfer_name(
                            static_cast<AVColorTransferCharacteristic>(video_stream->codecpar->color_trc));
                        const char* const input_space_name = av_color_space_name(
                            static_cast<AVColorSpace>(video_stream->codecpar->color_space));
                        const char* const input_range_name = av_color_range_name(
                            static_cast<AVColorRange>(video_stream->codecpar->color_range));
                        root["input"] = {
                            {"file", lfs::core::path_to_utf8(params.video_path)},
                            {"container", fmt_ctx->iformat && fmt_ctx->iformat->name
                                              ? fmt_ctx->iformat->name
                                              : "unknown"},
                            {"video", {
                                          {"codec", avcodec_get_name(codec_id)},
                                          {"pixel_format", pixelFormatName(input_pixel_format)},
                                          {"bit_depth", input_bit_depth > 0 ? input_bit_depth : source_bit_depth},
                                          {"size", {src_width, src_height}},
                                          {"fps", video_fps},
                                          {"duration_seconds", video_duration},
                                          {"color", {
                                                        {"primaries", {{"code", static_cast<int>(video_stream->codecpar->color_primaries)}, {"name", input_primaries_name ? input_primaries_name : "unknown"}}},
                                                        {"transfer", {{"code", static_cast<int>(video_stream->codecpar->color_trc)}, {"name", input_transfer_name ? input_transfer_name : "unknown"}}},
                                                        {"matrix", {{"code", static_cast<int>(video_stream->codecpar->color_space)}, {"name", input_space_name ? input_space_name : "unknown"}}},
                                                        {"range", {{"code", static_cast<int>(video_stream->codecpar->color_range)}, {"name", input_range_name ? input_range_name : "unknown"}}},
                                                    }},
                                      }},
                            {"hdr", {
                                        {"detected", isHdrFormat(hdr_format)},
                                        {"format", hdrFormatLabel(hdr_format)},
                                        {"dolby_vision_profile", dv_profile},
                                        {"dolby_vision_compatibility_id", dv_compatibility},
                                        {"source_bit_depth", input_bit_depth > 0 ? input_bit_depth : source_bit_depth},
                                    }},
                        };
                        root["output"] = {
                            {"format", params.format == ImageFormat::PNG ? "png" : params.format == ImageFormat::EXR ? "exr"
                                                                                                                     : "jpg"},
                            {"size", (params.rotation == 90 || params.rotation == 270)
                                         ? nlohmann::json{out_height, out_width}
                                         : nlohmann::json{out_width, out_height}},
                            {"bit_depth", 8},
                            {"pixel_format", "rgb24"},
                            {"color_space", convert_hdr_to_sdr
                                                ? "sRGB transfer, BT.709 primaries, full range"
                                                : "source conversion"},
                            {"dithered", convert_hdr_to_sdr},
                            {"hdr_to_sdr", convert_hdr_to_sdr},
                            {"jpeg_quality", params.jpg_quality},
                        };
                        if (float_output) {
                            root["output"]["bit_depth"] = params.exr_precision == media::ExrPrecision::Half ? 16 : 32;
                            root["output"].erase("jpeg_quality");
                            root["output"]["pixel_format"] = "linear_rgb_float32";
                            root["output"]["color_space"] = output_color.primaries == media::ColorPrimaries::Bt709 ? "linear BT709, relative" : "linear BT2020, relative";
                            root["output"]["compression"] = params.exr_compression == media::ExrCompression::ZIP ? "zip" : "none";
                            root["output"]["alpha"] = "none";
                            const int component_depth = input_bit_depth > 0 ? input_bit_depth : source_bit_depth;
                            root["output"]["source_component_depth"] = component_depth > 0 ? nlohmann::json(component_depth) : nlohmann::json(nullptr);
                            root["output"]["color_profile"] = "linear-sdr";
                            root["output"]["input_transfer"] = linear_converter->inputTransfer() == media::ColorTransfer::Linear ? "linear" : linear_converter->inputTransfer() == media::ColorTransfer::Srgb ? "srgb"
                                                                                                                                                                                                              : "bt709";
                            if (linear_converter->inputTransfer() == media::ColorTransfer::Unspecified)
                                root["output"]["input_transfer"] = nullptr;
                            root["output"]["transfer_override"] = params.input_color.transfer != media::ColorTransfer::Unspecified;
                            root["output"]["primaries_override"] = params.input_color.primaries != media::ColorPrimaries::Unspecified;
                            root["output"]["resize_filter"] = "pixel-center bilinear in linear light";
                            root["output"]["sharpness_domain"] = params.sharpness.enabled ? "clipped quantized linear RGB8 proxy" : "disabled";
                        }
                        root["processing"] = {
                            {"decoder", {
                                            {"backend", saw_hardware_frame ? decoder_backend : "ffmpeg_software"},
                                            {"name", codec && codec->name ? codec->name : "unknown"},
                                            {"threads", using_hw_decode ? 0 : codec_ctx->thread_count},
                                        }},
                            {"tone_mapping", {
                                                 {"backend", convert_hdr_to_sdr ? hdr_renderer->backendName() : std::string_view("none")},
                                                 {"enabled", convert_hdr_to_sdr},
                                                 {"output", convert_hdr_to_sdr ? "sRGB transfer, BT.709 primaries, full range, 8-bit dithered" : "source"},
                                             }},
                            {"image_encoder", {
                                                  {"backend", used_gpu_encoding ? "nvimagecodec_cuda" : "cpu"},
                                                  {"batch_size", used_gpu_encoding ? jpeg_batch_size : 1},
                                              }},
                            {"frame_selection", {
                                                    {"strategy", sparse_seek_fallback ? "sequential_fallback" : use_sparse_keyframe_seek ? "keyframe_seek"
                                                                                                                                         : "sequential"},
                                                    {"reason", frame_selection_reason},
                                                    {"keyframe_seek_count", keyframe_seek_count},
                                                    {"source_fps", video_fps},
                                                    {"requested_fps", params.mode == ExtractionMode::FPS ? params.fps : 0.0},
                                                    {"estimated_targets", estimated_total},
                                                    {"estimated_source_frames", total_frames},
                                                    {"sharpness_enabled", params.sharpness.enabled},
                                                    {"sharpness_window_mode", params.sharpness.window_mode},
                                                }},
                        };
                        root["performance"] = {
                            {"decoded_frames", decoded_frame_count},
                            {"selected_frames", saved_count},
                            {"written_frames", written_count},
                            {"discarded_frames", skipped_count},
                            {"timing_seconds", {
                                                   {"total", elapsedSeconds(extraction_started)},
                                                   {"tonemap_initialization", hdr_timing_total.initialization_seconds},
                                                   {"tonemap_render", hdr_timing_total.render_seconds},
                                                   {"tone_map_readback", hdr_timing_total.readback_seconds},
                                                   {"rgba_to_rgb", hdr_timing_total.rgba_to_rgb_seconds},
                                                   {"cuda_upload", cuda_upload_seconds},
                                                   {"jpeg_encode", jpeg_encode_seconds},
                                                   {"file_write", jpeg_write_seconds},
                                               }},
                        };

                        root["source_file"] = lfs::core::path_to_utf8(params.video_path);
                        root["source_fps"] = video_fps;
                        root["trimmed_source_frames"] = total_frames;
                        root["source_duration"] = video_duration;
                        root["source_size"] = {src_width, src_height};
                        root["rotation"] = params.rotation;
                        root["output_size"] = (params.rotation == 90 || params.rotation == 270)
                                                  ? nlohmann::json{out_height, out_width}
                                                  : nlohmann::json{out_width, out_height};
                        root["output_format"] = params.format == ImageFormat::PNG ? "png" : params.format == ImageFormat::EXR ? "exr"
                                                                                                                              : "jpg";
                        if (!float_output)
                            root["output_quality"] = params.jpg_quality;
                        root["filename_pattern"] = params.filename_pattern;
                        if (params.mode == ExtractionMode::FPS) {
                            root["extraction"]["mode"] = "fps";
                            root["extraction"]["fps"] = params.fps;
                        } else {
                            root["extraction"]["mode"] = "interval";
                            root["extraction"]["interval"] = params.frame_interval;
                        }
                        if (params.sharpness.enabled) {
                            std::string algo;
                            switch (params.sharpness.algorithm) {
                            case SharpnessAlgorithm::LAPLACIAN: algo = "laplacian"; break;
                            case SharpnessAlgorithm::TENENGRAD: algo = "tenengrad"; break;
                            case SharpnessAlgorithm::COMBINED: algo = "combined"; break;
                            }
                            root["sharpness"]["algorithm"] = algo;
                            root["sharpness"]["threshold"] = params.sharpness.threshold;
                            root["sharpness"]["window_mode"] = params.sharpness.window_mode;
                            root["sharpness"]["window_candidates_target"] = params.sharpness.window_candidates_target;
                            // Save human-readable mode
                            if (params.sharpness.window_candidates_target < 0)
                                root["sharpness"]["window_candidate_mode"] = "auto";
                            else if (params.sharpness.window_candidates_target == 0)
                                root["sharpness"]["window_candidate_mode"] = "all";
                            else
                                root["sharpness"]["window_candidate_mode"] = "fixed";
                            root["sharpness"]["estimated_window_frames"] = window_est_frames;
                            // Effective candidates per window (for auto and fixed modes)
                            int eff = window_est_frames;
                            if (params.sharpness.window_candidates_target < 0)
                                eff = std::min(std::clamp(static_cast<int>(
                                                              std::round(std::sqrt(static_cast<double>(window_est_frames))) * 2),
                                                          5, 20),
                                               window_est_frames);
                            else if (params.sharpness.window_candidates_target > 0)
                                eff = std::min(params.sharpness.window_candidates_target, window_est_frames);
                            root["sharpness"]["effective_candidates_per_window"] = eff;
                        }

                        auto& frames = root["frames"];
                        for (const auto& f : saved_frames) {
                            frames.push_back({{"file", f.filename},
                                              {"timestamp", f.timestamp},
                                              {"source_frame", f.source_frame},
                                              {"sharpness_score", f.sharpness_score}});
                        }

                        const std::filesystem::path meta_path =
                            params.output_dir / "extraction_metadata.json";
                        if (float_output) {
                            media::ExrOutputOptions options;
                            options.overwrite = params.overwrite_metadata;
                            options.cancelled = params.cancel_requested;
                            const auto result = core::writeTextFileAtomically(meta_path, root.dump(2), {.overwrite = options.overwrite, .durable = false, .create_directories = false, .cancelled = options.cancelled});
                            if (!result) {
                                failure_ = result.error();
                                if (failure_->code() == ErrorCode::Cancelled)
                                    throw ExtractionCancelled{};
                                throw std::runtime_error(std::string(failure_->detail()));
                            }
                        } else {
                            std::ofstream meta_file(meta_path, std::ios::binary);
                            if (meta_file)
                                meta_file << root.dump(2);
                        }
                    } catch (const std::exception& e) {
                        if (float_output)
                            throw;
                        LOG_WARN("Failed to write extraction metadata: {}", e.what());
                    }
                }

                if (skipped_count > 0) {
                    LOG_INFO("Extracted {} frames from video ({} discarded for low sharpness)",
                             written_count, skipped_count);
                } else {
                    LOG_INFO("Extracted {} frames from video", written_count);
                }
                if (convert_hdr_to_sdr) {
                    LOG_INFO("HDR extraction timing: total={:.3f}s decoded={} selected={} written={} init={:.3f}s "
                             "render={:.3f}s readback={:.3f}s rgba_to_rgb={:.3f}s cuda_upload={:.3f}s "
                             "jpeg_encode={:.3f}s file_write={:.3f}s",
                             elapsedSeconds(extraction_started), decoded_frame_count, saved_count, written_count,
                             hdr_timing_total.initialization_seconds, hdr_timing_total.render_seconds,
                             hdr_timing_total.readback_seconds, hdr_timing_total.rgba_to_rgb_seconds,
                             cuda_upload_seconds, jpeg_encode_seconds, jpeg_write_seconds);
                }

                check_sink(sink.complete({media::SinkOutcome::Completed, accepted_frames, {}}));
                sink_started = false;
                cleanup();

                outcome_ = ExtractionOutcome::Completed;
                return true;

            } catch (const ExtractionCancelled& e) {
                cleanup();
                outcome_ = ExtractionOutcome::Cancelled;
                error = e.what();
                abort_sink(media::SinkOutcome::Cancelled);
                return false;
            } catch (const std::filesystem::filesystem_error& e) {
                cleanup();
                // MSVC's what() can contain ACP-encoded paths and localized text.
                // Build this public error from native paths instead of returning
                // bytes that are invalid UTF-8 to the GUI or JSON consumers.
                error = "Filesystem operation failed [" + std::string(e.code().category().name()) +
                        ":" + std::to_string(e.code().value()) + "]";
                if (!e.path1().empty())
                    error += " " + lfs::core::path_to_utf8(e.path1());
                if (!e.path2().empty())
                    error += " -> " + lfs::core::path_to_utf8(e.path2());
                abort_sink(media::SinkOutcome::Failed);
                return false;
            } catch (const std::exception& e) {
                cleanup();
                error = e.what();
                abort_sink(media::SinkOutcome::Failed);
                return false;
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): User-supplied sink/progress callbacks may throw non-standard exceptions; release decoder resources and terminate the sink lifecycle.
                cleanup();
                error = "Unknown extraction failure";
                abort_sink(media::SinkOutcome::Failed);
                return false;
            }
        }

        [[nodiscard]] ExtractionOutcome lastOutcome() const { return outcome_; }
        [[nodiscard]] std::optional<Error> lastError() const { return failure_; }

    private:
        ExtractionOutcome outcome_ = ExtractionOutcome::Failed;
        std::optional<Error> failure_;
    };

    VideoFrameExtractor::VideoFrameExtractor() : impl_(new Impl()) {}
    VideoFrameExtractor::~VideoFrameExtractor() { delete impl_; }

    bool VideoFrameExtractor::extract(const Params& params, std::string& error) {
        return impl_->extract(params, error);
    }

    bool VideoFrameExtractor::extractToSink(const Params& params, media::FrameSink& sink, std::string& error) {
        return impl_->extract(params, error, &sink);
    }

    bool VideoFrameExtractor::extractFilesToSink(const Params& params, media::FrameSink& sink, std::string& error) {
        return impl_->extract(params, error, &sink, true);
    }

    std::optional<Error> VideoFrameExtractor::lastError() const { return impl_->lastError(); }

    ExtractionOutcome VideoFrameExtractor::lastOutcome() const {
        return impl_->lastOutcome();
    }

} // namespace lfs::io
