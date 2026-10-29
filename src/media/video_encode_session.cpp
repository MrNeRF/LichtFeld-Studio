// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media/video_encode_session.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "media/video_output_extent.hpp"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <exception>
#include <format>
#include <string_view>
#include <thread>
#include <utility>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/spherical.h>
#include <libavutil/stereo3d.h>
}
namespace lfs::media {
    namespace {
        constexpr int DEFAULT_FRAMERATE = 30;
        constexpr int NVENC_FRAME_POOL_SIZE = 4;
        constexpr int NVENC_QP_OFFSET = 3;
        constexpr int nvencAv1Qp(int quality) {
            // Empirical constant-QP alignment with libaom's adaptive CRF on
            // rendered/detail clips. Their scales are not linearly equivalent;
            // content still affects the quality difference between providers.
            constexpr std::array<std::pair<int, int>, 6> points{{{0, 0}, {10, 18}, {18, 40}, {30, 85}, {40, 135}, {51, 205}}};
            for (size_t i = 1; i < points.size(); ++i) {
                const auto [x, y] = points[i - 1];
                const auto [next_x, next_y] = points[i];
                if (quality <= next_x)
                    return y + ((quality - x) * (next_y - y) + (next_x - x) / 2) / (next_x - x);
            }
            return points.back().second;
        }
        Result<void> encodeError(ErrorCode code, std::string text) {
            return Result<void>::failure(make_error({.code = code, .domain = ErrorDomain::IO, .detail = std::move(text), .detection = LFS_SOURCE_SITE_CURRENT()}));
        }
        std::string avErrorText(int code) {
            char text[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(code, text, sizeof(text));
            return text;
        }
        const char* codecName(VideoEncodeCodec codec) {
            return codec == VideoEncodeCodec::AV1 ? "AV1" : codec == VideoEncodeCodec::HEVC ? "HEVC"
                                                                                            : "H.264";
        }
        std::string codecOpenFailure(const AVCodec* codec, int width, int height, const char* error) {
            std::string reason = std::format("{} codec open failed for {}x{}: {}", codec->name, width, height, error);
            if (std::strcmp(codec->name, "h264_nvenc") == 0 && (width > 4096 || height > 4096)) {
                reason += " (NVENC side limit is 4096 pixels)";
            } else if (std::strcmp(codec->name, "libopenh264") == 0 &&
                       static_cast<std::int64_t>(width) * height > 9'437'184) {
                reason += std::format(" (OpenH264 pixel limit is 9437184; requested {}x{})", width, height);
            }
            return reason;
        }

        bool applySphericalMetadata(AVFormatContext* const fmt_ctx, AVStream* const stream) {
            if (!fmt_ctx || !stream || !stream->codecpar)
                return false;
            fmt_ctx->strict_std_compliance = FF_COMPLIANCE_UNOFFICIAL;
            size_t spherical_size = 0;
            auto* const spherical = av_spherical_alloc(&spherical_size);
            if (!spherical)
                return false;
            spherical->projection = AV_SPHERICAL_EQUIRECTANGULAR;
            if (!av_packet_side_data_add(&stream->codecpar->coded_side_data,
                                         &stream->codecpar->nb_coded_side_data,
                                         AV_PKT_DATA_SPHERICAL, spherical, spherical_size, 0)) {
                av_free(spherical);
                return false;
            }
            size_t stereo_size = 0;
            auto* const stereo = av_stereo3d_alloc_size(&stereo_size);
            if (!stereo)
                return false;
            stereo->type = AV_STEREO3D_2D;
            if (!av_packet_side_data_add(&stream->codecpar->coded_side_data,
                                         &stream->codecpar->nb_coded_side_data,
                                         AV_PKT_DATA_STEREO3D, stereo, stereo_size, 0)) {
                av_free(stereo);
                return false;
            }
            return true;
        }

        void applyProvenanceMetadata(AVFormatContext* fmt_ctx, const VideoEncodeOptions& opts) {
            if (!fmt_ctx || opts.comment.empty())
                return;
            if (av_dict_set(&fmt_ctx->metadata, "comment", opts.comment.c_str(), 0) < 0)
                LOG_WARN("Failed to set video provenance comment metadata");
        }
    } // namespace

    class VideoEncodeSession::Impl {
    public:
        ~Impl() { cleanup(); }

        Result<void> open(
            const std::filesystem::path& path,
            const VideoEncodeOptions& opts_in) {

            const VideoEncodeOptions& opts = opts_in;
            if (is_open_)
                return encodeError(ErrorCode::Unavailable, "Encoder is already open");
            width_ = opts.width;
            height_ = opts.height;
            hardware_open_error_.clear();
            framerate_ = opts.framerate;
            framerate_denominator_ = opts.framerate_denominator;
            const bool initialized = opts.preferred_backend == VideoEncodeBackend::Cuda
                                         ? tryInitNvenc(path, opts)
                                         : opts.preferred_backend == VideoEncodeBackend::VideoToolbox && tryInitVideoToolbox(path, opts);
            if (!initialized) {
                cleanup();
                if (opts.preferred_backend != VideoEncodeBackend::Software)
                    LOG_INFO("Requested hardware video encoder unavailable");
                if (opts.codec == VideoEncodeCodec::HEVC) {
                    const auto detail = hardware_open_error_.empty() ? "No requested native provider is available" : hardware_open_error_;
                    return encodeError(ErrorCode::Unavailable, std::format("H.265/HEVC encoding unavailable: {}. HEVC export requires the CUDA backend (NVENC) or macOS (VideoToolbox); no additional software encoder is configured", detail));
                }
                const auto* software = opts.codec == VideoEncodeCodec::AV1 ? avcodec_find_encoder_by_name("libaom-av1") : avcodec_find_encoder(AV_CODEC_ID_H264);
                if (const auto result = initCpuEncoder(path, opts, software); !result) {
                    std::string detail(result.error().detail());
                    if (!hardware_open_error_.empty()) {
                        detail = std::format("{} encoding is unavailable for {}x{}. Native provider: {}; {} fallback: {}",
                                             codecName(opts.codec), opts.width, opts.height, hardware_open_error_, codecName(opts.codec), detail);
                    }
                    cleanup();
                    return encodeError(ErrorCode::Unavailable, std::move(detail));
                }
            }

            is_open_ = true;
            frame_count_ = 0;
            return {};
        }

        Result<void> writeFrame(VideoEncodeWriter& writer) {
            if (!is_open_)
                return encodeError(ErrorCode::FailedPrecondition, "Encoder not open");
            // A queued NVENC surface may still reference the previous storage.
            // Acquire another pool buffer, without copying its pixels, rather
            // than let a producer overwrite memory still owned by the codec.
            if (use_nvenc_ && !av_frame_is_writable(frame_)) {
                av_frame_unref(frame_);
                if (av_hwframe_get_buffer(hw_frames_ctx_, frame_, 0) < 0)
                    return encodeError(ErrorCode::Unavailable, "CUDA encoder frame allocation failed");
            } else if (!use_nvenc_ && av_frame_make_writable(frame_) < 0)
                return encodeError(ErrorCode::Unavailable, "Frame not writable");
            VideoEncodeTarget target;
            target.backend = backend_;
            target.layout = use_nvenc_ ? VideoEncodeLayout::NV12 : VideoEncodeLayout::YUV420P;
            const int count = use_nvenc_ ? 2 : 3;
            for (int plane = 0; plane < count; ++plane) {
                const int row_bytes = plane == 0 || use_nvenc_ ? width_ : width_ / 2;
                if (!frame_->data[plane] || frame_->linesize[plane] < row_bytes)
                    return encodeError(ErrorCode::Internal, std::format("Invalid encoder plane storage (plane={}, data_present={}, linesize={}, row_bytes={})", plane, frame_->data[plane] != nullptr, frame_->linesize[plane], row_bytes));
                target.planes[plane] = {frame_->data[plane], static_cast<std::size_t>(frame_->linesize[plane]),
                                        row_bytes,
                                        plane == 0 ? height_ : height_ / 2};
            }
            if (auto result = writer.write(target); !result)
                return result;
            // NVENC may replace the AVFrame storage above. Restore the known
            // plane interpretation for every submitted frame, including reopen.
            frame_->colorspace = codec_ctx_->colorspace;
            frame_->color_range = codec_ctx_->color_range;
            frame_->pts = frame_count_;
            if (auto result = encodeFrame(frame_); !result)
                return result;
            ++frame_count_;
            return {};
        }

        Result<void> close() {
            if (!is_open_)
                return {};

            std::string close_error;
            if (const auto result = encodeFrame(nullptr); !result) {
                LOG_WARN("Flush error: {}", result.error().detail());
                close_error = result.error().detail();
            }

            if (fmt_ctx_) {
                const int ret = av_write_trailer(fmt_ctx_);
                if (ret < 0 && close_error.empty()) {
                    char err[AV_ERROR_MAX_STRING_SIZE];
                    av_strerror(ret, err, sizeof(err));
                    close_error = std::string("Trailer write failed: ") + err;
                }
            }

            LOG_INFO("Video: {} frames encoded", frame_count_);
            cleanup();
            if (!close_error.empty())
                return encodeError(ErrorCode::Unavailable, std::move(close_error));
            return {};
        }

        [[nodiscard]] bool isOpen() const { return is_open_; }
        bool& inCall() { return in_call_; }

    private:
        [[nodiscard]] double fps() const { return static_cast<double>(framerate_) / framerate_denominator_; }
        // About one keyframe per second.
        [[nodiscard]] int gopSize() const { return std::max(1, static_cast<int>(std::lround(fps()))); }

        bool tryInitNvenc(const std::filesystem::path& path, const VideoEncodeOptions& opts) {
            const AVCodec* const codec = avcodec_find_encoder_by_name(opts.codec == VideoEncodeCodec::AV1 ? "av1_nvenc" : opts.codec == VideoEncodeCodec::HEVC ? "hevc_nvenc"
                                                                                                                                                               : "h264_nvenc");
            if (!codec) {
                hardware_open_error_ = "Requested NVENC encoder is not present in this build";
                LOG_DEBUG("NVENC not available");
                return false;
            }

            const std::string path_utf8 = lfs::core::path_to_native_utf8(path);

            int ret = avformat_alloc_output_context2(&fmt_ctx_, nullptr, "mp4", path_utf8.c_str());
            if (ret < 0 || !fmt_ctx_) {
                hardware_open_error_ = std::format("{} MP4 context allocation failed: {}", codec->name, avErrorText(ret < 0 ? ret : AVERROR(ENOMEM)));
                return false;
            }

            stream_ = avformat_new_stream(fmt_ctx_, nullptr);
            if (!stream_) {
                hardware_open_error_ = "NVENC output stream allocation failed";
                avformat_free_context(fmt_ctx_);
                fmt_ctx_ = nullptr;
                return false;
            }
            stream_->id = 0;

            codec_ctx_ = avcodec_alloc_context3(codec);
            if (!codec_ctx_) {
                hardware_open_error_ = "NVENC codec context allocation failed";
                avformat_free_context(fmt_ctx_);
                fmt_ctx_ = nullptr;
                return false;
            }

            codec_ctx_->width = width_;
            codec_ctx_->height = height_;
            codec_ctx_->time_base = AVRational{framerate_denominator_, framerate_};
            codec_ctx_->framerate = AVRational{framerate_, framerate_denominator_};
            codec_ctx_->sample_aspect_ratio = AVRational{1, 1};
            codec_ctx_->pix_fmt = AV_PIX_FMT_CUDA;
            codec_ctx_->gop_size = gopSize();
            codec_ctx_->max_b_frames = 0;
            codec_ctx_->colorspace = static_cast<AVColorSpace>(opts.matrix);
            codec_ctx_->color_range = static_cast<AVColorRange>(opts.range);

            // Do not silently encode with defaults if a dependency update
            // removes or rejects a requested option.
            const auto checked_option = [&](int status, const char* name) {
                if (status >= 0)
                    return true;
                hardware_open_error_ = std::format("{} option '{}' failed: {}", codec->name, name, avErrorText(status));
                LOG_DEBUG("{}", hardware_open_error_);
                cleanupHwContexts();
                return false;
            };
            // Preserve the legacy H264/HEVC mapping.
            const int qp = opts.codec == VideoEncodeCodec::AV1 ? nvencAv1Qp(opts.crf) : std::min(opts.crf + NVENC_QP_OFFSET, 51);
            if (!checked_option(av_opt_set(codec_ctx_->priv_data, "preset", "p4", 0), "preset") ||
                !checked_option(av_opt_set(codec_ctx_->priv_data, "tune", "hq", 0), "tune") ||
                !checked_option(av_opt_set(codec_ctx_->priv_data, "rc", "constqp", 0), "rc") ||
                !checked_option(av_opt_set_int(codec_ctx_->priv_data, "qp", qp, 0), "qp"))
                return false;

            if (fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
                codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
            }

            ret = av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
            if (ret < 0) {
                hardware_open_error_ = std::format("{} CUDA device initialization failed: {}", codec->name, avErrorText(ret));
                LOG_DEBUG("{}", hardware_open_error_);
                cleanupCodecContext();
                return false;
            }
            codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);

            hw_frames_ctx_ = av_hwframe_ctx_alloc(hw_device_ctx_);
            if (!hw_frames_ctx_) {
                hardware_open_error_ = "NVENC CUDA frame pool allocation failed";
                cleanupHwContexts();
                return false;
            }

            auto* const frames_ctx = reinterpret_cast<AVHWFramesContext*>(hw_frames_ctx_->data);
            frames_ctx->format = AV_PIX_FMT_CUDA;
            frames_ctx->sw_format = AV_PIX_FMT_NV12;
            frames_ctx->width = width_;
            frames_ctx->height = height_;
            frames_ctx->initial_pool_size = NVENC_FRAME_POOL_SIZE;

            ret = av_hwframe_ctx_init(hw_frames_ctx_);
            if (ret < 0) {
                hardware_open_error_ = std::format("{} CUDA frame pool initialization failed for {}x{}: {}", codec->name, width_, height_, avErrorText(ret));
                cleanupHwContexts();
                return false;
            }
            codec_ctx_->hw_frames_ctx = av_buffer_ref(hw_frames_ctx_);

            ret = avcodec_open2(codec_ctx_, codec, nullptr);
            if (ret < 0) {
                char err[AV_ERROR_MAX_STRING_SIZE];
                av_strerror(ret, err, sizeof(err));
                hardware_open_error_ = codecOpenFailure(codec, width_, height_, err);
                LOG_DEBUG("{}", hardware_open_error_);
                cleanupHwContexts();
                return false;
            }

            ret = avcodec_parameters_from_context(stream_->codecpar, codec_ctx_);
            if (ret < 0) {
                hardware_open_error_ = std::format("{} codec parameters copy failed: {}", codec->name, avErrorText(ret));
                cleanupHwContexts();
                return false;
            }
            if (opts.codec == VideoEncodeCodec::HEVC)
                stream_->codecpar->codec_tag = MKTAG('h', 'v', 'c', '1');
            else if (opts.codec == VideoEncodeCodec::AV1)
                stream_->codecpar->codec_tag = MKTAG('a', 'v', '0', '1');
            if (opts.spherical_video && !applySphericalMetadata(fmt_ctx_, stream_)) {
                hardware_open_error_ = "Spherical video metadata setup failed";
                cleanupHwContexts();
                return false;
            }
            stream_->time_base = codec_ctx_->time_base;

            if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE)) {
                ret = avio_open(&fmt_ctx_->pb, path_utf8.c_str(), AVIO_FLAG_WRITE);
                if (ret < 0) {
                    hardware_open_error_ = std::format("{} output file open failed: {}", codec->name, avErrorText(ret));
                    cleanupHwContexts();
                    return false;
                }
            }

            applyProvenanceMetadata(fmt_ctx_, opts);

            ret = avformat_write_header(fmt_ctx_, nullptr);
            if (ret < 0) {
                hardware_open_error_ = std::format("{} MP4 header write failed: {}", codec->name, avErrorText(ret));
                cleanupHwContexts();
                return false;
            }

            frame_ = av_frame_alloc();
            if (!frame_) {
                hardware_open_error_ = "NVENC frame allocation failed";
                cleanupHwContexts();
                return false;
            }
            ret = av_hwframe_get_buffer(hw_frames_ctx_, frame_, 0);
            if (ret < 0) {
                hardware_open_error_ = std::format("{} CUDA frame allocation failed: {}", codec->name, avErrorText(ret));
                cleanupHwContexts();
                return false;
            }

            packet_ = av_packet_alloc();
            if (!packet_) {
                hardware_open_error_ = "NVENC packet allocation failed";
                cleanupHwContexts();
                return false;
            }

            use_nvenc_ = true;
            backend_ = VideoEncodeBackend::Cuda;
            LOG_INFO("NVENC ({}): {}x{} @ {} fps", codec->name, width_, height_, fps());
            return true;
        }

        // Both Apple encoders consume the same YUV420P planes. Prefer the media
        // engine, then Apple's software encoder for unsupported hardware extents.
        bool tryInitVideoToolbox(const std::filesystem::path& path, const VideoEncodeOptions& opts) {
            if (opts.codec == VideoEncodeCodec::AV1) {
                hardware_open_error_ = "FFmpeg 9 has no AV1 VideoToolbox encoder";
                return false;
            }
            const AVCodec* const codec = avcodec_find_encoder_by_name(opts.codec == VideoEncodeCodec::HEVC ? "hevc_videotoolbox" : "h264_videotoolbox");
            if (!codec) {
                hardware_open_error_ = "Requested VideoToolbox encoder is not present in this build";
                return false;
            }
            if (const auto result = initCpuEncoder(path, opts, codec); !result) {
                hardware_open_error_ = std::string(result.error().detail());
                LOG_INFO("VideoToolbox hardware encoder unavailable: {}; retrying with Apple software encoder", hardware_open_error_);
                cleanup();
                if (const auto software = initCpuEncoder(path, opts, codec, true); !software) {
                    hardware_open_error_ += std::format("; Apple software: {}", software.error().detail());
                    return false;
                }
            }
            return true;
        }

        // An encoder fed YUV420P frames from system memory: the software
        // encoder, or VideoToolbox on Macs.
        Result<void> initCpuEncoder(
            const std::filesystem::path& path,
            const VideoEncodeOptions& opts,
            const AVCodec* const codec,
            const bool apple_software = false) {

            const std::string path_utf8 = lfs::core::path_to_native_utf8(path);

            int ret = avformat_alloc_output_context2(&fmt_ctx_, nullptr, "mp4", path_utf8.c_str());
            if (ret < 0 || !fmt_ctx_) {
                return encodeError(ErrorCode::Unavailable, "MP4 context creation failed");
            }

            if (!codec) {
                return encodeError(ErrorCode::Unavailable, opts.codec == VideoEncodeCodec::AV1 ? "AV1 software encoder not present in this build (requires FFmpeg libaom-av1)" : "H.264 encoder not found");
            }
            const bool videotoolbox = std::string_view(codec->name).ends_with("_videotoolbox");
            const bool hardware = videotoolbox && !apple_software;

            stream_ = avformat_new_stream(fmt_ctx_, nullptr);
            if (!stream_) {
                return encodeError(ErrorCode::Unavailable, "Stream creation failed");
            }
            stream_->id = 0;

            codec_ctx_ = avcodec_alloc_context3(codec);
            if (!codec_ctx_) {
                return encodeError(ErrorCode::Unavailable, "Codec context allocation failed");
            }

            codec_ctx_->width = width_;
            codec_ctx_->height = height_;
            codec_ctx_->time_base = AVRational{framerate_denominator_, framerate_};
            codec_ctx_->framerate = AVRational{framerate_, framerate_denominator_};
            codec_ctx_->sample_aspect_ratio = AVRational{1, 1};
            codec_ctx_->pix_fmt = AV_PIX_FMT_YUV420P;
            codec_ctx_->gop_size = gopSize();
            const bool aom = opts.codec == VideoEncodeCodec::AV1;
            codec_ctx_->max_b_frames = videotoolbox || aom ? 0 : 2;
            codec_ctx_->thread_count = aom ? int(std::clamp(std::thread::hardware_concurrency(), 1u, 8u)) : 0;
            codec_ctx_->colorspace = static_cast<AVColorSpace>(opts.matrix);
            codec_ctx_->color_range = static_cast<AVColorRange>(opts.range);

            // Map CRF 18 to 0.1 bits per pixel per frame, with a 250 kbps floor and linear quality scaling.
            const int64_t crf_scale = 51 - opts.crf;
            codec_ctx_->bit_rate = std::max<int64_t>(
                250'000, static_cast<int64_t>(width_) * height_ * framerate_ * crf_scale / (330LL * framerate_denominator_));
            if (aom) {
                // Explicit constant-quality AV1, not the legacy H264 bitrate map.
                // Limit lookahead and per-call workers without adding a global pool.
                codec_ctx_->bit_rate = 0;
                for (const auto& [name, value] : {std::pair{"crf", opts.crf}, std::pair{"cpu-used", 8}, std::pair{"row-mt", 1}, std::pair{"lag-in-frames", 8}})
                    if (const auto status = av_opt_set_int(codec_ctx_->priv_data, name, value, 0); status < 0)
                        return encodeError(ErrorCode::Unavailable, std::format("AV1 libaom option '{}' failed: {}", name, avErrorText(status)));
            } else if (videotoolbox) {
                // Select explicitly so backend() and producer targets describe
                // the actual encoder, including the Apple software fallback.
                if (av_opt_set_int(codec_ctx_->priv_data, "allow_sw", apple_software, 0) < 0 ||
                    av_opt_set_int(codec_ctx_->priv_data, "require_sw", apple_software, 0) < 0)
                    return encodeError(ErrorCode::Unavailable, "VideoToolbox encoder selection failed");
                if (av_opt_set(codec_ctx_->priv_data, "profile", opts.codec == VideoEncodeCodec::HEVC ? "main" : "high", 0) < 0)
                    return encodeError(ErrorCode::Unavailable, "VideoToolbox profile selection failed");
            } else {
                av_opt_set(codec_ctx_->priv_data, "rc_mode", "bitrate", 0);
            }

            if (fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
                codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
            }

            ret = avcodec_open2(codec_ctx_, codec, nullptr);
            if (ret < 0) {
                char err[AV_ERROR_MAX_STRING_SIZE];
                av_strerror(ret, err, sizeof(err));
                return encodeError(ErrorCode::Unavailable, codecOpenFailure(codec, width_, height_, err));
            }

            ret = avcodec_parameters_from_context(stream_->codecpar, codec_ctx_);
            if (ret < 0) {
                return encodeError(ErrorCode::Unavailable, "Codec parameters copy failed");
            }
            if (opts.codec == VideoEncodeCodec::HEVC)
                stream_->codecpar->codec_tag = MKTAG('h', 'v', 'c', '1');
            else if (opts.codec == VideoEncodeCodec::AV1)
                stream_->codecpar->codec_tag = MKTAG('a', 'v', '0', '1');
            if (opts.spherical_video && !applySphericalMetadata(fmt_ctx_, stream_))
                return encodeError(ErrorCode::Unavailable, "Spherical video metadata setup failed");
            stream_->time_base = codec_ctx_->time_base;

            if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE)) {
                ret = avio_open(&fmt_ctx_->pb, path_utf8.c_str(), AVIO_FLAG_WRITE);
                if (ret < 0) {
                    char err[AV_ERROR_MAX_STRING_SIZE];
                    av_strerror(ret, err, sizeof(err));
                    return encodeError(ErrorCode::Unavailable, std::string("File open failed: ") + err);
                }
            }

            applyProvenanceMetadata(fmt_ctx_, opts);

            ret = avformat_write_header(fmt_ctx_, nullptr);
            if (ret < 0) {
                char err[AV_ERROR_MAX_STRING_SIZE];
                av_strerror(ret, err, sizeof(err));
                return encodeError(ErrorCode::Unavailable, std::string("Header write failed: ") + err);
            }

            frame_ = av_frame_alloc();
            if (!frame_) {
                return encodeError(ErrorCode::Unavailable, "Frame allocation failed");
            }
            frame_->format = AV_PIX_FMT_YUV420P;
            frame_->width = width_;
            frame_->height = height_;

            ret = av_frame_get_buffer(frame_, 0);
            if (ret < 0) {
                return encodeError(ErrorCode::Unavailable, "Frame buffer allocation failed");
            }

            packet_ = av_packet_alloc();
            if (!packet_) {
                return encodeError(ErrorCode::Unavailable, "Packet allocation failed");
            }

            backend_ = videotoolbox ? (hardware ? VideoEncodeBackend::VideoToolbox : VideoEncodeBackend::VideoToolboxSoftware)
                                    : VideoEncodeBackend::Software;
            LOG_INFO("{} {} ({}): {}x{} @ {} fps, bitrate {} bps", hardware ? "Hardware" : "Software", codecName(opts.codec), codec->name,
                     width_, height_, fps(), codec_ctx_->bit_rate);
            return {};
        }

        Result<void> encodeFrame(AVFrame* const frame) {
            int ret = avcodec_send_frame(codec_ctx_, frame);
            if (ret < 0) {
                char err[AV_ERROR_MAX_STRING_SIZE];
                av_strerror(ret, err, sizeof(err));
                return encodeError(ErrorCode::Unavailable, std::string("Send frame error: ") + err);
            }

            while (ret >= 0) {
                ret = avcodec_receive_packet(codec_ctx_, packet_);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                    break;
                if (ret < 0) {
                    char err[AV_ERROR_MAX_STRING_SIZE];
                    av_strerror(ret, err, sizeof(err));
                    return encodeError(ErrorCode::Unavailable, std::string("Receive packet error: ") + err);
                }

                if (packet_->duration <= 0)
                    packet_->duration = 1;
                av_packet_rescale_ts(packet_, codec_ctx_->time_base, stream_->time_base);
                packet_->stream_index = stream_->index;

                ret = av_interleaved_write_frame(fmt_ctx_, packet_);
                av_packet_unref(packet_);

                if (ret < 0) {
                    char err[AV_ERROR_MAX_STRING_SIZE];
                    av_strerror(ret, err, sizeof(err));
                    return encodeError(ErrorCode::Unavailable, std::string("Write frame error: ") + err);
                }
            }
            return {};
        }

        void cleanupCodecContext() {
            if (codec_ctx_) {
                avcodec_free_context(&codec_ctx_);
                codec_ctx_ = nullptr;
            }
            if (fmt_ctx_) {
                if (fmt_ctx_->pb && !(fmt_ctx_->oformat->flags & AVFMT_NOFILE))
                    avio_closep(&fmt_ctx_->pb);
                avformat_free_context(fmt_ctx_);
                fmt_ctx_ = nullptr;
            }
        }

        void cleanupHwContexts() {
            cleanupCodecContext();
            if (hw_frames_ctx_) {
                av_buffer_unref(&hw_frames_ctx_);
                hw_frames_ctx_ = nullptr;
            }
            if (hw_device_ctx_) {
                av_buffer_unref(&hw_device_ctx_);
                hw_device_ctx_ = nullptr;
            }
            stream_ = nullptr;
        }

        void cleanup() {
            if (packet_) {
                av_packet_free(&packet_);
                packet_ = nullptr;
            }
            if (frame_) {
                av_frame_free(&frame_);
                frame_ = nullptr;
            }

            cleanupHwContexts();
            is_open_ = false;
            use_nvenc_ = false;
            backend_ = VideoEncodeBackend::Software;
        }

        AVFormatContext* fmt_ctx_ = nullptr;
        AVCodecContext* codec_ctx_ = nullptr;
        AVStream* stream_ = nullptr;
        AVFrame* frame_ = nullptr;
        AVPacket* packet_ = nullptr;
        AVBufferRef* hw_device_ctx_ = nullptr;
        AVBufferRef* hw_frames_ctx_ = nullptr;

        int width_ = 0;
        int height_ = 0;
        int framerate_ = DEFAULT_FRAMERATE;
        int framerate_denominator_ = 1;
        int64_t frame_count_ = 0;
        bool is_open_ = false;
        bool in_call_ = false;
        std::string hardware_open_error_;
        bool use_nvenc_ = false;

    public:
        VideoEncodeBackend backend_ = VideoEncodeBackend::Software;
    };

    VideoEncodeSession::VideoEncodeSession() : impl_(std::make_unique<Impl>()) {}
    VideoEncodeSession::~VideoEncodeSession() = default;
    VideoEncodeSession::VideoEncodeSession(VideoEncodeSession&& other) {
        if (other.impl_ && other.impl_->inCall())
            throw lfs::Exception(make_error({.code = ErrorCode::FailedPrecondition, .domain = ErrorDomain::IO, .detail = "Cannot move encoder during a frame writer callback (writer_active=true)", .detection = LFS_SOURCE_SITE_CURRENT()}));
        impl_ = std::move(other.impl_);
    }
    VideoEncodeSession& VideoEncodeSession::operator=(VideoEncodeSession&& other) {
        if ((impl_ && impl_->inCall()) || (other.impl_ && other.impl_->inCall()))
            throw lfs::Exception(make_error({.code = ErrorCode::FailedPrecondition, .domain = ErrorDomain::IO, .detail = "Cannot replace encoder ownership during a frame writer callback (writer_active=true)", .detection = LFS_SOURCE_SITE_CURRENT()}));
        if (this != &other)
            impl_ = std::move(other.impl_);
        return *this;
    }
    Result<void> VideoEncodeSession::open(const std::filesystem::path& path, const VideoEncodeOptions& options) {
        if (!impl_)
            impl_ = std::make_unique<Impl>();
        if (impl_->inCall())
            return encodeError(ErrorCode::FailedPrecondition, "Cannot open encoder during a frame writer callback (writer_active=true)");
        if (impl_->isOpen())
            return encodeError(ErrorCode::FailedPrecondition, "Encoder is already open");
        if (auto error = io::video::videoOutputExtentError(options.width, options.height))
            return encodeError(ErrorCode::InvalidArgument, std::string(*error));
        if (auto error = io::video::videoEncodingRangeError(options.framerate, options.crf, options.framerate_denominator))
            return encodeError(ErrorCode::InvalidArgument, std::move(*error));
        if (options.codec != VideoEncodeCodec::H264 && options.codec != VideoEncodeCodec::HEVC && options.codec != VideoEncodeCodec::AV1)
            return encodeError(ErrorCode::InvalidArgument, std::format("Unknown video codec (got {})", static_cast<int>(options.codec)));
        if (options.preferred_backend != VideoEncodeBackend::Software &&
            options.preferred_backend != VideoEncodeBackend::Cuda &&
            options.preferred_backend != VideoEncodeBackend::VideoToolbox)
            return encodeError(ErrorCode::InvalidArgument, std::format("Unknown video encoder backend (got {})", static_cast<int>(options.preferred_backend)));
        switch (options.matrix) {
        case ColorMatrix::Rgb:
        case ColorMatrix::Bt709:
        case ColorMatrix::Unspecified:
        case ColorMatrix::Fcc:
        case ColorMatrix::Bt470Bg:
        case ColorMatrix::Smpte170M:
        case ColorMatrix::Smpte240M:
        case ColorMatrix::Ycgco:
        case ColorMatrix::Bt2020Ncl:
        case ColorMatrix::Bt2020Cl:
        case ColorMatrix::Smpte2085:
        case ColorMatrix::ChromaNcl:
        case ColorMatrix::ChromaCl:
        case ColorMatrix::Ictcp:
        case ColorMatrix::YcgcoRe:
        case ColorMatrix::YcgcoRo:
            break;
        default:
            return encodeError(ErrorCode::InvalidArgument, std::format("Unknown video color matrix (got {})", static_cast<int>(options.matrix)));
        }
        if (options.range != ColorRange::Unspecified && options.range != ColorRange::Limited && options.range != ColorRange::Full)
            return encodeError(ErrorCode::InvalidArgument, std::format("Unknown video color range (got {})", static_cast<int>(options.range)));
        auto extension = core::path_to_utf8(path.extension());
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (!extension.empty() && extension != ".mp4")
            return encodeError(ErrorCode::InvalidArgument, "Video export supports only MP4 (.mp4); unsupported filename extension");
        return impl_->open(path, options);
    }
    Result<void> VideoEncodeSession::writeFrame(VideoEncodeWriter& writer) {
        if (!impl_)
            return encodeError(ErrorCode::FailedPrecondition, "Encoder not open");
        if (impl_->inCall())
            return encodeError(ErrorCode::FailedPrecondition, "Cannot write encoder during a frame writer callback (writer_active=true)");
        struct CallGuard {
            bool& active;
            explicit CallGuard(bool& value) : active(value) { active = true; }
            ~CallGuard() { active = false; }
        } guard(impl_->inCall());
        try {
            return impl_->writeFrame(writer);
        } catch (const lfs::Exception& error) {
            return Result<void>::failure(error.error());
        } catch (const std::exception& error) {
            return Result<void>::failure(make_error({.code = ErrorCode::Internal, .domain = ErrorDomain::IO, .detail = error.what(), .detection = LFS_SOURCE_SITE_CURRENT()}));
        } catch (...) {
            return Result<void>::failure(make_error({.code = ErrorCode::Internal, .domain = ErrorDomain::IO, .detail = "Video frame writer threw a nonstandard exception", .detection = LFS_SOURCE_SITE_CURRENT()}));
        }
    }
    Result<void> VideoEncodeSession::close() {
        if (!impl_)
            return {};
        if (impl_->inCall())
            return encodeError(ErrorCode::FailedPrecondition, "Cannot close encoder during a frame writer callback (writer_active=true)");
        return impl_->close();
    }
    bool VideoEncodeSession::isOpen() const { return impl_ && impl_->isOpen(); }
    VideoEncodeBackend VideoEncodeSession::backend() const { return impl_ ? impl_->backend_ : VideoEncodeBackend::Software; }
} // namespace lfs::media
