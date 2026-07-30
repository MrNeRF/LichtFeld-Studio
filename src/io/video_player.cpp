/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "video_player.hpp"
#include "core/include/core/logger.hpp"
#include "core/path_utils.hpp"
#include "hdr_libplacebo.hpp"
#include "hdr_tonemap.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/display.h>
#include <libavutil/dovi_meta.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <queue>
#include <thread>

namespace lfs::io {

    namespace {

        constexpr size_t FRAME_QUEUE_SIZE = 16;
        constexpr int MAX_PREVIEW_HEIGHT = 720;
        constexpr size_t MIN_BUFFERED_FRAMES = 4;
        constexpr int MAX_SW_DECODE_THREADS = 8;

        [[nodiscard]] double monotonicSeconds() {
            return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
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

        [[nodiscard]] bool configureVideoToRgbColorimetry(
            SwsContext* context, const AVFrame* source,
            const AVColorSpace fallback_colorspace,
            const AVColorRange fallback_range) {
            if (!context || !source)
                return false;
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

        const char* getHwDecoderName(const AVCodecID codec_id) {
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

        AVPixelFormat getHwFormat(AVCodecContext*, const AVPixelFormat* pix_fmts) {
            for (const AVPixelFormat* p = pix_fmts; *p != -1; p++) {
                if (*p == AV_PIX_FMT_CUDA) {
                    return *p;
                }
            }
            return AV_PIX_FMT_NONE;
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

    } // namespace

    struct DecodedFrame {
        std::vector<uint8_t> data;
        int channels = 3;
        int width = 0;
        int height = 0;
        bool gpu_rotation = false;
        double pts = 0;
        int64_t frame_number = 0;
        int64_t raw_timestamp = AV_NOPTS_VALUE;
    };

    class VideoPlayer::Impl {
    public:
        enum class DecodeResult {
            Frame,
            EndOfStream,
            Error,
        };

        ~Impl() { close(); }

        bool open(const std::filesystem::path& path) {
            close();
            clearError();

            const std::string path_utf8 = lfs::core::path_to_utf8(path);

            const int open_result = avformat_open_input(&fmt_ctx_, path_utf8.c_str(), nullptr, nullptr);
            if (open_result < 0) {
                setError("Failed to open video: " + ffmpegError(open_result));
                LOG_WARN("VideoPlayer: Failed to open {}", path_utf8);
                return false;
            }

            // Select the video stream before probing codec-level metadata.
            video_stream_idx_ = findUsableHeaderVideoStream(fmt_ctx_);
            if (video_stream_idx_ < 0) {
                // Fallback for containers requiring packet probing.
                if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
                    close();
                    return false;
                }
                video_stream_idx_ = findUsableHeaderVideoStream(fmt_ctx_);
            }

            if (video_stream_idx_ < 0) {
                close();
                return false;
            }

            discardNonVideoStreams(fmt_ctx_, video_stream_idx_);

            // Probe the selected stream for HDR metadata stored in the bitstream.
            if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
                LOG_WARN("VideoPlayer: could not complete video-only stream metadata probe");
            }
            av_seek_frame(fmt_ctx_, video_stream_idx_, 0, AVSEEK_FLAG_BACKWARD);

            AVStream* const stream = fmt_ctx_->streams[video_stream_idx_];
            const AVCodecID codec_id = stream->codecpar->codec_id;

            // Read rotation metadata from stream (metadata dict or display matrix)
            rotation_ = 0;
            AVDictionaryEntry* tag = av_dict_get(stream->metadata, "rotate", nullptr, 0);
            if (tag && tag->value) {
                rotation_ = std::atoi(tag->value);
            } else {
                // Try display matrix side data (common in MP4/MOV from smartphones)
                int32_t* display_matrix = nullptr;

                // Check codecpar coded_side_data
                for (int i = 0; i < stream->codecpar->nb_coded_side_data; i++) {
                    if (stream->codecpar->coded_side_data[i].type == AV_PKT_DATA_DISPLAYMATRIX) {
                        display_matrix = reinterpret_cast<int32_t*>(
                            stream->codecpar->coded_side_data[i].data);
                        break;
                    }
                }

                if (display_matrix) {
                    const double angle = av_display_rotation_get(display_matrix);
                    // av_display_rotation_get returns CCW, our convention is CW
                    rotation_ = static_cast<int>(std::round(-angle));
                }
            }
            rotation_ = ((rotation_ % 360) + 360) % 360; // normalize
            if (rotation_ != 0 && rotation_ != 90 && rotation_ != 180 && rotation_ != 270)
                rotation_ = 0;

            is_hdr_ = false;
            hdr_info_ = "SDR";
            const int source_bit_depth = pixelFormatBitDepth(stream->codecpar);
            const bool has_mastering_metadata =
                hasCodedSideData(stream->codecpar, AV_PKT_DATA_MASTERING_DISPLAY_METADATA);
            const bool has_content_light_metadata =
                hasCodedSideData(stream->codecpar, AV_PKT_DATA_CONTENT_LIGHT_LEVEL);
            hdr_format_ = detectHdrFormat(stream->codecpar->color_trc, source_bit_depth,
                                          has_mastering_metadata, has_content_light_metadata);
            bool has_dolby_vision = false;
            {
                int dv_profile = 0;
                int dv_compatibility = 0;
                for (int i = 0; i < stream->codecpar->nb_coded_side_data; i++) {
                    const AVPacketSideData* side_data = &stream->codecpar->coded_side_data[i];
                    if (side_data->type == AV_PKT_DATA_DOVI_CONF &&
                        side_data->size >= static_cast<int>(sizeof(AVDOVIDecoderConfigurationRecord))) {
                        const auto* dovi = reinterpret_cast<const AVDOVIDecoderConfigurationRecord*>(side_data->data);
                        dv_profile = dovi->dv_profile;
                        dv_compatibility = dovi->dv_bl_signal_compatibility_id;
                        break;
                    }
                }

                if (dv_profile > 0) {
                    is_hdr_ = true;
                    has_dolby_vision = true;
                    hdr_format_ = detectDolbyVisionFormat(stream->codecpar->color_trc, dv_profile,
                                                          dv_compatibility);
                    hdr_info_ = hdrFormatLabel(hdr_format_);
                } else if (isHdrFormat(hdr_format_)) {
                    is_hdr_ = true;
                    hdr_info_ = hdrFormatLabel(hdr_format_);
                }
            }
            if (is_hdr_) {
                LOG_INFO("VideoPlayer: detected HDR format {} (transfer={}, primaries={}, matrix={}, pixel_format={})",
                         hdr_info_, static_cast<int>(stream->codecpar->color_trc),
                         static_cast<int>(stream->codecpar->color_primaries),
                         static_cast<int>(stream->codecpar->color_space),
                         av_get_pix_fmt_name(static_cast<AVPixelFormat>(stream->codecpar->format)));
            }

            // Supply BT.2020 NCL limited-range defaults when NVDEC omits tags.
            source_colorspace_ = stream->codecpar->color_space;
            source_range_ = stream->codecpar->color_range;
            if (is_hdr_ && source_colorspace_ == AVCOL_SPC_UNSPECIFIED)
                source_colorspace_ = AVCOL_SPC_BT2020_NCL;
            if (is_hdr_ && source_range_ == AVCOL_RANGE_UNSPECIFIED)
                source_range_ = AVCOL_RANGE_MPEG;

            const char* hw_decoder_name = getHwDecoderName(codec_id);
            const AVCodec* codec = nullptr;

            // NVDEC cuvid does not reliably preserve Dolby Vision RPU side data.
            if (hw_decoder_name && !has_dolby_vision) {
                codec = avcodec_find_decoder_by_name(hw_decoder_name);
                if (codec &&
                    av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) ==
                        0) {
                    using_hw_decode_ = true;
                    LOG_INFO("VideoPlayer: NVDEC decoder: {}", hw_decoder_name);
                } else {
                    codec = nullptr;
                }
            }

            if (!codec) {
                codec = avcodec_find_decoder(codec_id);
                if (!codec) {
                    close();
                    return false;
                }
                using_hw_decode_ = false;
                LOG_INFO("VideoPlayer: {} decoder", has_dolby_vision
                                                        ? "FFmpeg Dolby Vision metadata-preserving"
                                                        : "CPU");
            }

            codec_ctx_ = avcodec_alloc_context3(codec);
            if (!codec_ctx_) {
                setError("Failed to allocate video decoder context");
                close();
                return false;
            }
            const int parameters_result =
                avcodec_parameters_to_context(codec_ctx_, stream->codecpar);
            if (parameters_result < 0) {
                setError("Failed to configure video decoder: " +
                         ffmpegError(parameters_result));
                close();
                return false;
            }
#ifdef AV_CODEC_EXPORT_DATA_DOVI_RPU
            // Export Dolby Vision RPU side data for libplacebo.
            codec_ctx_->export_side_data |= AV_CODEC_EXPORT_DATA_DOVI_RPU;
#endif

            if (using_hw_decode_) {
                codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
                if (!codec_ctx_->hw_device_ctx) {
                    setError("Failed to retain CUDA video decoder context");
                    close();
                    return false;
                }
                codec_ctx_->get_format = getHwFormat;
            } else {
                const unsigned int hardware_threads = std::max(1U, std::thread::hardware_concurrency());
                codec_ctx_->thread_count = std::min(MAX_SW_DECODE_THREADS,
                                                    static_cast<int>(hardware_threads));
                codec_ctx_->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
                LOG_INFO("VideoPlayer: FFmpeg software decoder threads: {}", codec_ctx_->thread_count);
            }

            const int codec_open_result =
                avcodec_open2(codec_ctx_, codec, nullptr);
            if (codec_open_result < 0) {
                setError("Failed to open video decoder: " +
                         ffmpegError(codec_open_result));
                close();
                return false;
            }

            src_width_ = codec_ctx_->width;
            src_height_ = codec_ctx_->height;
            fps_ = av_q2d(stream->avg_frame_rate);
            if (!std::isfinite(fps_) || fps_ <= 0.0)
                fps_ = av_q2d(stream->r_frame_rate);
            if (!std::isfinite(fps_) || fps_ <= 0.0)
                fps_ = 30.0;
            time_base_ = av_q2d(stream->time_base);
            if (!std::isfinite(time_base_) || time_base_ <= 0.0)
                time_base_ = 1.0 / fps_;
            stream_start_timestamp_ =
                stream->start_time == AV_NOPTS_VALUE ? 0 : stream->start_time;
            frame_duration_timestamp_ = std::max<int64_t>(
                1, static_cast<int64_t>(std::llround(1.0 / (fps_ * time_base_))));
            next_fallback_timestamp_ = stream_start_timestamp_;

            if (src_height_ > MAX_PREVIEW_HEIGHT) {
                const float scale = static_cast<float>(MAX_PREVIEW_HEIGHT) / src_height_;
                width_ = static_cast<int>(src_width_ * scale) & ~1;
                height_ = MAX_PREVIEW_HEIGHT;
            } else {
                width_ = src_width_;
                height_ = src_height_;
            }
            frame_size_ = static_cast<size_t>(width_) * height_ * 3;

            total_frames_ = stream->nb_frames;
            if (total_frames_ > 0) {
                duration_ = static_cast<double>(total_frames_) / fps_;
            } else if (fmt_ctx_->duration != AV_NOPTS_VALUE && fmt_ctx_->duration > 0) {
                duration_ = static_cast<double>(fmt_ctx_->duration) / AV_TIME_BASE;
            } else if (stream->duration != AV_NOPTS_VALUE && stream->duration > 0) {
                duration_ = static_cast<double>(stream->duration) * time_base_;
            } else {
                duration_ = 0.0;
            }
            if (total_frames_ <= 0 && duration_ > 0.0)
                total_frames_ = static_cast<int64_t>(std::llround(duration_ * fps_));

            frame_ = av_frame_alloc();
            sw_frame_ = av_frame_alloc();
            packet_ = av_packet_alloc();
            if (!frame_ || !sw_frame_ || !packet_) {
                setError("Failed to allocate video decoder frames");
                close();
                return false;
            }
            display_buffer_.resize(frame_size_);

            resetDecoderPumpState();
            const DecodeResult first_frame_result = decodeNextFrame();
            if (first_frame_result == DecodeResult::Frame) {
                display_buffer_ = std::move(decoded_frame_.data);
                display_channels_ = decoded_frame_.channels;
                display_width_ = decoded_frame_.width;
                display_height_ = decoded_frame_.height;
                display_gpu_rotation_ = decoded_frame_.gpu_rotation;
                current_time_ = decoded_frame_.pts;
                current_frame_ = decoded_frame_.frame_number;
                current_raw_timestamp_ = decoded_frame_.raw_timestamp;

                // Fallback: check decoded frame for display matrix
                if (rotation_ == 0 && frame_) {
                    for (int i = 0; i < frame_->nb_side_data; i++) {
                        if (frame_->side_data[i]->type == AV_FRAME_DATA_DISPLAYMATRIX) {
                            const double angle = av_display_rotation_get(
                                reinterpret_cast<int32_t*>(frame_->side_data[i]->data));
                            // av_display_rotation_get returns CCW, our convention is CW
                            rotation_ = static_cast<int>(std::round(-angle));
                            break;
                        }
                    }
                    rotation_ = ((rotation_ % 360) + 360) % 360;
                    if (rotation_ != 0 && rotation_ != 90 && rotation_ != 180 && rotation_ != 270)
                        rotation_ = 0;
                }
            } else {
                if (first_frame_result == DecodeResult::EndOfStream)
                    setError("Video contains no decodable frames");
                close();
                return false;
            }

            is_open_ = true;
            eof_reached_ = false;
            playback_start_time_ = -1.0;
            stop_decode_thread_ = false;
            decode_thread_ = std::thread(&Impl::decodeThreadFunc, this);

            return true;
        }

        void close() {
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                stop_decode_thread_ = true;
            }
            queue_cv_.notify_all();

            if (decode_thread_.joinable()) {
                decode_thread_.join();
            }

            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                while (!frame_queue_.empty()) {
                    frame_queue_.pop();
                }
            }

            if (sws_ctx_) {
                sws_freeContext(sws_ctx_);
                sws_ctx_ = nullptr;
            }
            hdr_renderer_.reset();
            resetDecoderPumpState();
            if (frame_) {
                av_frame_free(&frame_);
                frame_ = nullptr;
            }
            if (sw_frame_) {
                av_frame_free(&sw_frame_);
                sw_frame_ = nullptr;
            }
            if (packet_) {
                av_packet_free(&packet_);
                packet_ = nullptr;
            }
            if (codec_ctx_) {
                avcodec_free_context(&codec_ctx_);
                codec_ctx_ = nullptr;
            }
            if (hw_device_ctx_) {
                av_buffer_unref(&hw_device_ctx_);
                hw_device_ctx_ = nullptr;
            }
            if (fmt_ctx_) {
                avformat_close_input(&fmt_ctx_);
                fmt_ctx_ = nullptr;
            }

            video_stream_idx_ = -1;
            is_open_ = false;
            is_playing_ = false;
            using_hw_decode_ = false;
            is_hdr_ = false;
            hdr_to_sdr_ = false;
            preview_rotation_ = 0;
            logged_hdr_rgba_preview_path_ = false;
            hdr_capability_error_reported_ = false;
            hdr_format_ = HdrFormat::SDR;
            source_colorspace_ = AVCOL_SPC_UNSPECIFIED;
            source_range_ = AVCOL_RANGE_UNSPECIFIED;
            current_time_ = 0;
            current_frame_ = 0;
            current_raw_timestamp_ = AV_NOPTS_VALUE;
            stream_start_timestamp_ = 0;
            next_fallback_timestamp_ = 0;
            frame_duration_timestamp_ = 1;
            display_width_ = 0;
            display_height_ = 0;
            display_gpu_rotation_ = false;
            playback_start_time_ = -1.0;
            eof_reached_ = false;
            decode_failed_ = false;
        }

        [[nodiscard]] bool isOpen() const { return is_open_; }

        void play() {
            if (!is_open_) {
                return;
            }

            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait_for(lock, std::chrono::milliseconds(100), [this] {
                    return frame_queue_.size() >= MIN_BUFFERED_FRAMES || eof_reached_ ||
                           stop_decode_thread_;
                });
            }

            is_playing_ = true;
            playback_start_time_ = -1;
        }

        void pause() { is_playing_ = false; }

        void togglePlayPause() {
            if (is_playing_) {
                pause();
            } else {
                play();
            }
        }

        [[nodiscard]] bool isPlaying() const { return is_playing_; }

        void seek(double seconds) {
            if (!is_open_) {
                return;
            }

            pause();
            seconds = std::clamp(seconds, 0.0, duration_);
            const long double timestamp_value =
                static_cast<long double>(stream_start_timestamp_) +
                std::round(static_cast<long double>(seconds) / time_base_);
            const int64_t timestamp =
                timestamp_value <=
                        static_cast<long double>(std::numeric_limits<int64_t>::min())
                    ? std::numeric_limits<int64_t>::min()
                : timestamp_value >=
                        static_cast<long double>(std::numeric_limits<int64_t>::max())
                    ? std::numeric_limits<int64_t>::max()
                    : static_cast<int64_t>(timestamp_value);
            seekTimestamp(timestamp);
        }

        bool rerenderCurrentFrame() {
            if (!is_open_ || current_raw_timestamp_ == AV_NOPTS_VALUE)
                return false;

            const bool resume_playback = is_playing_;
            pause();
            seekTimestamp(current_raw_timestamp_);
            const bool rendered = !decode_failed_ && current_raw_timestamp_ != AV_NOPTS_VALUE;
            if (resume_playback && rendered)
                play();
            return rendered;
        }

        void seekTimestamp(const int64_t timestamp) {
            if (!is_open_)
                return;

            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                while (!frame_queue_.empty()) {
                    frame_queue_.pop();
                }
                seek_target_.store(timestamp);
                seek_requested_.store(true);
                decode_failed_.store(false);
            }
            queue_cv_.notify_all();

            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this] {
                    return stop_decode_thread_ ||
                           (!seek_requested_ &&
                            (!frame_queue_.empty() || eof_reached_ || decode_failed_));
                });

                if (!frame_queue_.empty()) {
                    auto frame = std::move(frame_queue_.front());
                    frame_queue_.pop();
                    displayFrame(std::move(frame));
                }
            }
        }

        void seekFrame(const int64_t frame_number) { seek(static_cast<double>(frame_number) / fps_); }

        void stepForward() {
            if (!is_open_) {
                return;
            }
            pause();

            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (!frame_queue_.empty()) {
                auto frame = std::move(frame_queue_.front());
                frame_queue_.pop();
                lock.unlock();
                queue_cv_.notify_all();

                displayFrame(std::move(frame));
            }
        }

        void stepBackward() {
            if (!is_open_) {
                return;
            }
            pause();
            const int64_t target = std::max<int64_t>(0, current_frame_ - 1);
            seekFrame(target);
        }

        bool update(double /*current_wall_time*/) {
            if (!is_open_) {
                return false;
            }

            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (!is_playing_) {
                return false;
            }

            const double now = monotonicSeconds();
            if (playback_start_time_ < 0.0)
                playback_start_time_ = now - current_time_;
            const double target_pts = now - playback_start_time_;
            const double frame_period = 1.0 / std::max(fps_, 1.0);
            const double presentation_slack = frame_period * 0.5;

            if (frame_queue_.empty()) {
                lock.unlock();
                if (eof_reached_ && target_pts >= duration_ - presentation_slack) {
                    is_playing_ = false;
                }
                return false;
            }

            // Present the newest frame whose PTS is due. When decode/render
            // falls behind, discard older queued frames rather than slowing
            // playback below the source time base.
            bool presented = false;
            while (!frame_queue_.empty() && frame_queue_.front().pts <= target_pts + presentation_slack) {
                auto frame = std::move(frame_queue_.front());
                frame_queue_.pop();
                displayFrame(std::move(frame));
                presented = true;
            }

            lock.unlock();
            queue_cv_.notify_all();

            return presented;
        }

        [[nodiscard]] const uint8_t* currentFrameData() const {
            return display_buffer_.empty() ? nullptr : display_buffer_.data();
        }
        [[nodiscard]] int currentFrameChannels() const { return display_channels_; }
        [[nodiscard]] bool currentFrameHasGpuRotation() const { return display_gpu_rotation_; }

        [[nodiscard]] int width() const { return display_width_; }
        [[nodiscard]] int height() const { return display_height_; }
        [[nodiscard]] int sourceWidth() const { return src_width_; }
        [[nodiscard]] int sourceHeight() const { return src_height_; }
        [[nodiscard]] int rotation() const { return rotation_; }
        [[nodiscard]] bool isHdr() const { return is_hdr_; }
        [[nodiscard]] bool isHdrConversionSupported() {
            if (!is_hdr_ || !isHdrTonemapSupported(hdr_format_))
                return false;
            if (!hdr_renderer_)
                hdr_renderer_ = std::make_unique<HdrLibplaceboRenderer>();
            std::string renderer_error;
            if (hdr_renderer_->isAvailable(renderer_error))
                return true;
            if (!hdr_capability_error_reported_) {
                setError("HDR-to-SDR preview unavailable: " + renderer_error);
                hdr_capability_error_reported_ = true;
            }
            return false;
        }
        [[nodiscard]] std::string hdrInfo() const { return hdr_info_; }
        void setHdrToSdr(const bool enabled) {
            const bool value = enabled && isHdrConversionSupported();
            if (hdr_to_sdr_.exchange(value) != value && hdr_renderer_)
                hdr_renderer_->reset();
        }
        void setPreviewRotation(const int degrees) {
            const int normalized = ((degrees % 360) + 360) % 360;
            const int rotation = normalized - normalized % 90;
            if (preview_rotation_.exchange(rotation) != rotation && hdr_renderer_)
                hdr_renderer_->reset();
        }
        std::string takeError() {
            std::lock_guard lock(error_mutex_);
            std::string error = std::move(last_error_);
            last_error_.clear();
            return error;
        }
        [[nodiscard]] double currentTime() const { return current_time_; }
        [[nodiscard]] double duration() const { return duration_; }
        [[nodiscard]] int64_t currentFrameNumber() const { return current_frame_; }
        [[nodiscard]] int64_t totalFrames() const { return total_frames_; }
        [[nodiscard]] double fps() const { return fps_; }

        std::vector<uint8_t> getThumbnail(const double time, const int max_width) {
            if (!is_open_) {
                return {};
            }

            seek(time);

            if (display_buffer_.empty()) {
                return {};
            }

            if (max_width > 0 && display_width_ > max_width) {
                const int thumb_height = display_height_ * max_width / display_width_;
                std::vector<uint8_t> thumb(static_cast<size_t>(max_width) * thumb_height * 3);

                const AVPixelFormat preview_format =
                    display_channels_ == 4 ? AV_PIX_FMT_RGBA : AV_PIX_FMT_RGB24;
                SwsContext* const thumb_sws =
                    sws_getContext(display_width_, display_height_, preview_format, max_width, thumb_height,
                                   AV_PIX_FMT_RGB24, SWS_BILINEAR, nullptr, nullptr, nullptr);

                if (thumb_sws) {
                    uint8_t* src_data[1] = {display_buffer_.data()};
                    int src_linesize[1] = {display_width_ * display_channels_};
                    uint8_t* dst_data[1] = {thumb.data()};
                    int dst_linesize[1] = {max_width * 3};

                    sws_scale(thumb_sws, src_data, src_linesize, 0, display_height_, dst_data, dst_linesize);
                    sws_freeContext(thumb_sws);
                }
                return thumb;
            }

            return display_buffer_;
        }

    private:
        void displayFrame(DecodedFrame&& frame) {
            display_buffer_ = std::move(frame.data);
            display_channels_ = frame.channels;
            display_width_ = frame.width;
            display_height_ = frame.height;
            display_gpu_rotation_ = frame.gpu_rotation;
            current_time_ = frame.pts;
            current_frame_ = frame.frame_number;
            current_raw_timestamp_ = frame.raw_timestamp;
        }

        void setError(std::string error) {
            std::lock_guard lock(error_mutex_);
            last_error_ = std::move(error);
            decode_failed_.store(true);
        }

        void clearError() {
            std::lock_guard lock(error_mutex_);
            last_error_.clear();
            decode_failed_.store(false);
        }

        void resetDecoderPumpState() {
            if (packet_)
                av_packet_unref(packet_);
            packet_pending_ = false;
            demux_eof_ = false;
            decoder_drain_sent_ = false;
            seek_requested_.store(false);
            seek_target_.store(AV_NOPTS_VALUE);
            skip_video_conversion_ = false;
        }

        [[nodiscard]] int64_t decodedFrameTimestamp() {
            int64_t timestamp = frame_->best_effort_timestamp;
            if (timestamp == AV_NOPTS_VALUE)
                timestamp = frame_->pts;
            if (timestamp == AV_NOPTS_VALUE)
                timestamp = next_fallback_timestamp_;

            const int64_t duration = frame_->duration > 0
                                         ? frame_->duration
                                         : frame_duration_timestamp_;
            const int64_t positive_duration = std::max<int64_t>(duration, 1);
            next_fallback_timestamp_ =
                timestamp > std::numeric_limits<int64_t>::max() - positive_duration
                    ? std::numeric_limits<int64_t>::max()
                    : timestamp + positive_duration;
            return timestamp;
        }

        void setDecodedFrameTiming() {
            decoded_frame_.raw_timestamp = decodedFrameTimestamp();
            const long double relative_timestamp =
                static_cast<long double>(decoded_frame_.raw_timestamp) -
                static_cast<long double>(stream_start_timestamp_);
            const long double presentation_time =
                relative_timestamp * static_cast<long double>(time_base_);
            decoded_frame_.pts = static_cast<double>(presentation_time);
            if (!std::isfinite(decoded_frame_.pts)) {
                decoded_frame_.pts = 0.0;
            }
            const long double frame_number =
                std::round(static_cast<long double>(decoded_frame_.pts) * fps_);
            decoded_frame_.frame_number =
                frame_number <= 0.0L
                    ? 0
                : frame_number >=
                        static_cast<long double>(std::numeric_limits<int64_t>::max())
                    ? std::numeric_limits<int64_t>::max()
                    : static_cast<int64_t>(frame_number);
        }

        void decodeThreadFunc() {
            while (true) {
                std::unique_lock<std::mutex> lock(queue_mutex_);

                queue_cv_.wait(lock, [this] {
                    return stop_decode_thread_ || seek_requested_ ||
                           (!eof_reached_ && !decode_failed_ &&
                            frame_queue_.size() < FRAME_QUEUE_SIZE);
                });

                if (stop_decode_thread_) {
                    break;
                }

                if (seek_requested_) {
                    const int64_t target_timestamp = seek_target_.load();
                    lock.unlock();
                    performSeek(target_timestamp);
                    queue_cv_.notify_all();
                    continue;
                }

                lock.unlock();

                const DecodeResult result = decodeNextFrame();
                if (result == DecodeResult::Frame) {
                    std::lock_guard<std::mutex> qlock(queue_mutex_);
                    if (!seek_requested_)
                        frame_queue_.push(std::move(decoded_frame_));
                } else if (result == DecodeResult::EndOfStream) {
                    eof_reached_ = true;
                    queue_cv_.notify_all();
                } else {
                    decode_failed_ = true;
                    queue_cv_.notify_all();
                }
            }
        }

        void performSeek(const int64_t timestamp) {
            const int seek_result =
                av_seek_frame(fmt_ctx_, video_stream_idx_, timestamp, AVSEEK_FLAG_BACKWARD);
            if (seek_result < 0) {
                setError("Video seek failed: " + ffmpegError(seek_result));
                resetDecoderPumpState();
                return;
            }

            avcodec_flush_buffers(codec_ctx_);
            resetDecoderPumpState();
            eof_reached_ = false;
            decode_failed_ = false;
            next_fallback_timestamp_ = timestamp;
            if (hdr_renderer_)
                hdr_renderer_->reset();

            skip_video_conversion_ = true;
            while (true) {
                const DecodeResult result = decodeNextFrame();
                if (result != DecodeResult::Frame) {
                    if (result == DecodeResult::EndOfStream) {
                        eof_reached_ = true;
                        setError("No decoded frame was available at the requested timestamp");
                    } else {
                        decode_failed_ = true;
                    }
                    break;
                }
                if (decoded_frame_.raw_timestamp >= timestamp) {
                    skip_video_conversion_ = false;
                    if (!convertFrameToBuffer(true))
                        break;
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    frame_queue_.push(std::move(decoded_frame_));
                    break;
                }
            }
            skip_video_conversion_ = false;
        }

        DecodeResult decodeNextFrame() {
            while (true) {
                av_frame_unref(frame_);
                const int receive_result = avcodec_receive_frame(codec_ctx_, frame_);
                if (receive_result == 0) {
                    return convertFrameToBuffer() ? DecodeResult::Frame : DecodeResult::Error;
                }
                if (receive_result == AVERROR_EOF)
                    return DecodeResult::EndOfStream;
                if (receive_result != AVERROR(EAGAIN)) {
                    setError("Video decoder receive failed: " + ffmpegError(receive_result));
                    return DecodeResult::Error;
                }

                if (packet_pending_) {
                    const int send_result = avcodec_send_packet(codec_ctx_, packet_);
                    if (send_result == 0) {
                        av_packet_unref(packet_);
                        packet_pending_ = false;
                        continue;
                    }
                    if (send_result == AVERROR(EAGAIN))
                        continue;

                    av_packet_unref(packet_);
                    packet_pending_ = false;
                    setError("Video decoder packet submission failed: " + ffmpegError(send_result));
                    return DecodeResult::Error;
                }

                if (demux_eof_) {
                    if (decoder_drain_sent_)
                        return DecodeResult::EndOfStream;

                    const int flush_result = avcodec_send_packet(codec_ctx_, nullptr);
                    if (flush_result == 0) {
                        decoder_drain_sent_ = true;
                        continue;
                    }
                    if (flush_result == AVERROR_EOF)
                        return DecodeResult::EndOfStream;
                    if (flush_result == AVERROR(EAGAIN))
                        continue;

                    setError("Video decoder drain failed: " + ffmpegError(flush_result));
                    return DecodeResult::Error;
                }

                while (true) {
                    const int read_result = av_read_frame(fmt_ctx_, packet_);
                    if (read_result == AVERROR_EOF) {
                        demux_eof_ = true;
                        break;
                    }
                    if (read_result < 0) {
                        setError("Video demux failed: " + ffmpegError(read_result));
                        return DecodeResult::Error;
                    }
                    if (packet_->stream_index == video_stream_idx_) {
                        packet_pending_ = true;
                        break;
                    }
                    av_packet_unref(packet_);
                }
            }
        }

        bool convertFrameToBuffer(const bool preserve_timing = false) {
            AVFrame* src_frame = frame_;
            if (!preserve_timing)
                setDecodedFrameTiming();

            if (skip_video_conversion_) {
                decoded_frame_.data.clear();
                decoded_frame_.channels = 3;
                decoded_frame_.width = width_;
                decoded_frame_.height = height_;
                decoded_frame_.gpu_rotation = false;
                return true;
            }

            if (using_hw_decode_ && frame_->format == AV_PIX_FMT_CUDA) {
                av_frame_unref(sw_frame_);
                const int transfer_result = av_hwframe_transfer_data(sw_frame_, frame_, 0);
                if (transfer_result < 0) {
                    setError("Hardware video frame transfer failed: " + ffmpegError(transfer_result));
                    return false;
                }
                const int props_result = av_frame_copy_props(sw_frame_, frame_);
                if (props_result < 0) {
                    av_frame_unref(sw_frame_);
                    setError("Hardware video frame metadata copy failed: " + ffmpegError(props_result));
                    return false;
                }
                src_frame = sw_frame_;
            }

            if (is_hdr_ && hdr_to_sdr_) {
                inheritStreamColorimetry(src_frame, fmt_ctx_->streams[video_stream_idx_]->codecpar, hdr_format_);
                if (!hdr_renderer_)
                    hdr_renderer_ = std::make_unique<HdrLibplaceboRenderer>();
                std::string renderer_error;
                const int rotation = preview_rotation_.load();
                const bool swapped_dimensions = rotation == 90 || rotation == 270;
                const int output_width = swapped_dimensions ? height_ : width_;
                const int output_height = swapped_dimensions ? width_ : height_;
                if (!hdr_renderer_->tonemapToSdrRgba(src_frame, fmt_ctx_->streams[video_stream_idx_],
                                                     hdr_format_,
                                                     output_width, output_height, rotation,
                                                     decoded_frame_.data, renderer_error)) {
                    LOG_ERROR("HDR preview renderer failed: {}", renderer_error);
                    decoded_frame_.data.clear();
                    setError("HDR preview renderer failed: " + renderer_error);
                    return false;
                }
                decoded_frame_.channels = 4;
                decoded_frame_.width = output_width;
                decoded_frame_.height = output_height;
                decoded_frame_.gpu_rotation = true;
                if (!logged_hdr_rgba_preview_path_) {
                    LOG_INFO("HDR preview: libplacebo RGBA8 readback -> Vulkan UI upload "
                             "(CPU RGB/RGBA conversion bypassed, GPU rotation={} deg)",
                             rotation);
                    logged_hdr_rgba_preview_path_ = true;
                }
            } else {
                sws_ctx_ = sws_getCachedContext(
                    sws_ctx_, src_width_, src_height_,
                    static_cast<AVPixelFormat>(src_frame->format), width_, height_,
                    AV_PIX_FMT_RGB24, SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
                if (!sws_ctx_) {
                    setError("Failed to create video color conversion context");
                    return false;
                }
                if (!configureVideoToRgbColorimetry(
                        sws_ctx_, src_frame, source_colorspace_, source_range_)) {
                    setError("Failed to configure video color conversion");
                    return false;
                }
                decoded_frame_.data.resize(frame_size_);
                uint8_t* dst_data[1] = {decoded_frame_.data.data()};
                int dst_linesize[1] = {width_ * 3};
                if (sws_scale(sws_ctx_, src_frame->data, src_frame->linesize, 0, src_height_,
                              dst_data, dst_linesize) <= 0) {
                    setError("Video color conversion failed");
                    decoded_frame_.data.clear();
                    return false;
                }
                decoded_frame_.channels = 3;
                decoded_frame_.width = width_;
                decoded_frame_.height = height_;
                decoded_frame_.gpu_rotation = false;
            }
            return true;
        }

        AVFormatContext* fmt_ctx_ = nullptr;
        AVCodecContext* codec_ctx_ = nullptr;
        AVBufferRef* hw_device_ctx_ = nullptr;
        SwsContext* sws_ctx_ = nullptr;
        AVFrame* frame_ = nullptr;
        AVFrame* sw_frame_ = nullptr;
        AVPacket* packet_ = nullptr;
        bool packet_pending_ = false;
        bool demux_eof_ = false;
        bool decoder_drain_sent_ = false;
        bool using_hw_decode_ = false;

        int video_stream_idx_ = -1;
        int src_width_ = 0;
        int src_height_ = 0;
        int width_ = 0;
        int height_ = 0;
        double fps_ = 0;
        double time_base_ = 0;
        double duration_ = 0;
        int64_t total_frames_ = 0;
        size_t frame_size_ = 0;
        int rotation_ = 0;
        bool is_hdr_ = false;
        std::string hdr_info_;
        HdrFormat hdr_format_ = HdrFormat::SDR;
        AVColorSpace source_colorspace_ = AVCOL_SPC_UNSPECIFIED;
        AVColorRange source_range_ = AVCOL_RANGE_UNSPECIFIED;
        std::atomic<bool> hdr_to_sdr_{false};
        std::atomic<int> preview_rotation_{0};
        bool skip_video_conversion_ = false;
        bool logged_hdr_rgba_preview_path_ = false;
        bool hdr_capability_error_reported_ = false;
        std::unique_ptr<HdrLibplaceboRenderer> hdr_renderer_;

        std::vector<uint8_t> display_buffer_;
        int display_channels_ = 3;
        int display_width_ = 0;
        int display_height_ = 0;
        bool display_gpu_rotation_ = false;
        double current_time_ = 0;
        int64_t current_frame_ = 0;
        int64_t current_raw_timestamp_ = AV_NOPTS_VALUE;
        int64_t stream_start_timestamp_ = 0;
        int64_t next_fallback_timestamp_ = 0;
        int64_t frame_duration_timestamp_ = 1;

        double playback_start_time_ = -1;

        bool is_open_ = false;
        bool is_playing_ = false;

        std::thread decode_thread_;
        std::mutex queue_mutex_;
        std::condition_variable queue_cv_;
        std::queue<DecodedFrame> frame_queue_;
        DecodedFrame decoded_frame_;
        std::atomic<bool> stop_decode_thread_{false};
        std::atomic<bool> eof_reached_{false};
        std::atomic<bool> decode_failed_{false};
        std::atomic<bool> seek_requested_{false};
        std::atomic<int64_t> seek_target_{AV_NOPTS_VALUE};
        std::mutex error_mutex_;
        std::string last_error_;
    };

    VideoPlayer::VideoPlayer() : impl_(std::make_unique<Impl>()) {}
    VideoPlayer::~VideoPlayer() = default;

    bool VideoPlayer::open(const std::filesystem::path& path) { return impl_->open(path); }
    void VideoPlayer::close() { impl_->close(); }
    bool VideoPlayer::isOpen() const { return impl_->isOpen(); }

    void VideoPlayer::play() { impl_->play(); }
    void VideoPlayer::pause() { impl_->pause(); }
    void VideoPlayer::togglePlayPause() { impl_->togglePlayPause(); }
    bool VideoPlayer::isPlaying() const { return impl_->isPlaying(); }

    void VideoPlayer::seek(const double seconds) { impl_->seek(seconds); }
    void VideoPlayer::seekFrame(const int64_t frame_number) { impl_->seekFrame(frame_number); }
    bool VideoPlayer::rerenderCurrentFrame() { return impl_->rerenderCurrentFrame(); }
    void VideoPlayer::stepForward() { impl_->stepForward(); }
    void VideoPlayer::stepBackward() { impl_->stepBackward(); }

    bool VideoPlayer::update(const double current_wall_time) { return impl_->update(current_wall_time); }

    const uint8_t* VideoPlayer::currentFrameData() const { return impl_->currentFrameData(); }
    int VideoPlayer::currentFrameChannels() const { return impl_->currentFrameChannels(); }
    bool VideoPlayer::currentFrameHasGpuRotation() const { return impl_->currentFrameHasGpuRotation(); }
    int VideoPlayer::width() const { return impl_->width(); }
    int VideoPlayer::height() const { return impl_->height(); }
    int VideoPlayer::sourceWidth() const { return impl_->sourceWidth(); }
    int VideoPlayer::sourceHeight() const { return impl_->sourceHeight(); }
    int VideoPlayer::rotation() const { return impl_->rotation(); }
    bool VideoPlayer::isHdr() const { return impl_->isHdr(); }
    bool VideoPlayer::isHdrConversionSupported() { return impl_->isHdrConversionSupported(); }
    std::string VideoPlayer::hdrInfo() const { return impl_->hdrInfo(); }
    void VideoPlayer::setHdrToSdr(const bool enabled) { impl_->setHdrToSdr(enabled); }
    void VideoPlayer::setPreviewRotation(const int degrees) { impl_->setPreviewRotation(degrees); }
    std::string VideoPlayer::takeError() { return impl_->takeError(); }

    double VideoPlayer::currentTime() const { return impl_->currentTime(); }
    double VideoPlayer::duration() const { return impl_->duration(); }
    int64_t VideoPlayer::currentFrameNumber() const { return impl_->currentFrameNumber(); }
    int64_t VideoPlayer::totalFrames() const { return impl_->totalFrames(); }
    double VideoPlayer::fps() const { return impl_->fps(); }

    std::vector<uint8_t> VideoPlayer::getThumbnail(const double time, const int max_width) {
        return impl_->getThumbnail(time, max_width);
    }

} // namespace lfs::io
