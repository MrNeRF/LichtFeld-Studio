// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media/video_encode_session.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "media/video_output_extent.hpp"
#include <algorithm>
#include <cctype>
#include <exception>
#include <format>
#include <string_view>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}
namespace lfs::media {
    namespace {
        constexpr int DEFAULT_FRAMERATE = 30;
        constexpr int NVENC_FRAME_POOL_SIZE = 4;
        constexpr int NVENC_QP_OFFSET = 3;
        Result<void> encodeError(ErrorCode code, std::string text) {
            return Result<void>::failure(make_error({.code = code, .domain = ErrorDomain::IO, .detail = std::move(text), .detection = LFS_SOURCE_SITE_CURRENT()}));
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
            framerate_ = opts.framerate;
            const bool initialized = opts.preferred_backend == VideoEncodeBackend::Cuda
                                         ? tryInitNvenc(path, opts)
                                         : opts.preferred_backend == VideoEncodeBackend::VideoToolbox && tryInitVideoToolbox(path, opts);
            if (!initialized) {
                cleanup();
                if (opts.preferred_backend != VideoEncodeBackend::Software)
                    LOG_INFO("Hardware H.264 encoding unavailable, falling back to software H.264");
                if (const auto result = initH264(path, opts, avcodec_find_encoder(AV_CODEC_ID_H264)); !result) {
                    cleanup();
                    return result;
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
        bool tryInitNvenc(const std::filesystem::path& path, const VideoEncodeOptions& opts) {
            const AVCodec* const codec = avcodec_find_encoder_by_name("h264_nvenc");
            if (!codec) {
                LOG_DEBUG("NVENC not available");
                return false;
            }

            const std::string path_utf8 = lfs::core::path_to_utf8(path);

            int ret = avformat_alloc_output_context2(&fmt_ctx_, nullptr, "mp4", path_utf8.c_str());
            if (ret < 0 || !fmt_ctx_)
                return false;

            stream_ = avformat_new_stream(fmt_ctx_, nullptr);
            if (!stream_) {
                avformat_free_context(fmt_ctx_);
                fmt_ctx_ = nullptr;
                return false;
            }
            stream_->id = 0;

            codec_ctx_ = avcodec_alloc_context3(codec);
            if (!codec_ctx_) {
                avformat_free_context(fmt_ctx_);
                fmt_ctx_ = nullptr;
                return false;
            }

            codec_ctx_->width = width_;
            codec_ctx_->height = height_;
            codec_ctx_->time_base = AVRational{1, framerate_};
            codec_ctx_->framerate = AVRational{framerate_, 1};
            codec_ctx_->pix_fmt = AV_PIX_FMT_CUDA;
            codec_ctx_->gop_size = framerate_;
            codec_ctx_->max_b_frames = 0;
            codec_ctx_->colorspace = static_cast<AVColorSpace>(opts.matrix);
            codec_ctx_->color_range = static_cast<AVColorRange>(opts.range);

            av_opt_set(codec_ctx_->priv_data, "preset", "p4", 0);
            av_opt_set(codec_ctx_->priv_data, "tune", "hq", 0);
            av_opt_set(codec_ctx_->priv_data, "rc", "constqp", 0);
            av_opt_set_int(codec_ctx_->priv_data, "qp", opts.crf + NVENC_QP_OFFSET, 0);

            if (fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
                codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
            }

            ret = av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
            if (ret < 0) {
                LOG_DEBUG("CUDA hw context failed");
                cleanupCodecContext();
                return false;
            }
            codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);

            hw_frames_ctx_ = av_hwframe_ctx_alloc(hw_device_ctx_);
            if (!hw_frames_ctx_) {
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
                cleanupHwContexts();
                return false;
            }
            codec_ctx_->hw_frames_ctx = av_buffer_ref(hw_frames_ctx_);

            ret = avcodec_open2(codec_ctx_, codec, nullptr);
            if (ret < 0) {
                LOG_DEBUG("NVENC open failed");
                cleanupHwContexts();
                return false;
            }

            ret = avcodec_parameters_from_context(stream_->codecpar, codec_ctx_);
            if (ret < 0) {
                cleanupHwContexts();
                return false;
            }
            stream_->time_base = codec_ctx_->time_base;

            if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE)) {
                ret = avio_open(&fmt_ctx_->pb, path_utf8.c_str(), AVIO_FLAG_WRITE);
                if (ret < 0) {
                    cleanupHwContexts();
                    return false;
                }
            }

            applyProvenanceMetadata(fmt_ctx_, opts);

            ret = avformat_write_header(fmt_ctx_, nullptr);
            if (ret < 0) {
                cleanupHwContexts();
                return false;
            }

            frame_ = av_frame_alloc();
            if (!frame_) {
                cleanupHwContexts();
                return false;
            }
            ret = av_hwframe_get_buffer(hw_frames_ctx_, frame_, 0);
            if (ret < 0) {
                cleanupHwContexts();
                return false;
            }

            packet_ = av_packet_alloc();
            if (!packet_) {
                cleanupHwContexts();
                return false;
            }

            use_nvenc_ = true;
            backend_ = VideoEncodeBackend::Cuda;
            LOG_INFO("NVENC: {}x{} @ {} fps", width_, height_, framerate_);
            return true;
        }

        // Both Apple encoders consume the same YUV420P planes. Prefer the media
        // engine, then Apple's software encoder for unsupported hardware extents.
        bool tryInitVideoToolbox(const std::filesystem::path& path, const VideoEncodeOptions& opts) {
            const AVCodec* const codec = avcodec_find_encoder_by_name("h264_videotoolbox");
            if (!codec) {
                LOG_DEBUG("VideoToolbox H.264 encoder not available");
                return false;
            }
            if (const auto result = initH264(path, opts, codec); !result) {
                LOG_DEBUG("Hardware VideoToolbox H.264 encoder failed: {}", result.error().detail());
                cleanup();
                LOG_INFO("Hardware VideoToolbox H.264 unavailable, trying Apple software H.264");
                if (const auto software = initH264(path, opts, codec, true); !software) {
                    LOG_DEBUG("Software VideoToolbox H.264 encoder failed: {}", software.error().detail());
                    return false;
                }
            }
            return true;
        }

        // An H.264 encoder fed YUV420P frames from system memory: the software
        // encoder, or VideoToolbox on Macs.
        Result<void> initH264(
            const std::filesystem::path& path,
            const VideoEncodeOptions& opts,
            const AVCodec* const codec,
            const bool apple_software = false) {

            const std::string path_utf8 = lfs::core::path_to_utf8(path);

            int ret = avformat_alloc_output_context2(&fmt_ctx_, nullptr, "mp4", path_utf8.c_str());
            if (ret < 0 || !fmt_ctx_) {
                return encodeError(ErrorCode::Unavailable, "MP4 context creation failed");
            }

            if (!codec) {
                return encodeError(ErrorCode::Unavailable, "H.264 encoder not found");
            }
            const bool videotoolbox = std::string_view(codec->name) == "h264_videotoolbox";
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
            codec_ctx_->time_base = AVRational{1, framerate_};
            codec_ctx_->framerate = AVRational{framerate_, 1};
            codec_ctx_->pix_fmt = AV_PIX_FMT_YUV420P;
            codec_ctx_->gop_size = framerate_;
            codec_ctx_->max_b_frames = videotoolbox ? 0 : 2;
            codec_ctx_->thread_count = 0;
            codec_ctx_->colorspace = static_cast<AVColorSpace>(opts.matrix);
            codec_ctx_->color_range = static_cast<AVColorRange>(opts.range);

            // Map CRF 18 to 0.1 bits per pixel per frame, with a 250 kbps floor and linear quality scaling.
            const int64_t crf_scale = 51 - opts.crf;
            codec_ctx_->bit_rate = std::max<int64_t>(
                250'000, static_cast<int64_t>(width_) * height_ * framerate_ * crf_scale / 330);
            if (videotoolbox) {
                // Select explicitly so backend() and producer targets describe
                // the actual encoder, including the Apple software fallback.
                if (av_opt_set_int(codec_ctx_->priv_data, "allow_sw", apple_software, 0) < 0 ||
                    av_opt_set_int(codec_ctx_->priv_data, "require_sw", apple_software, 0) < 0)
                    return encodeError(ErrorCode::Unavailable, "VideoToolbox encoder selection failed");
                av_opt_set(codec_ctx_->priv_data, "profile", "high", 0);
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
                return encodeError(ErrorCode::Unavailable, std::string("Codec open failed: ") + err);
            }

            ret = avcodec_parameters_from_context(stream_->codecpar, codec_ctx_);
            if (ret < 0) {
                return encodeError(ErrorCode::Unavailable, "Codec parameters copy failed");
            }
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
            LOG_INFO("{} H.264 ({}): {}x{} @ {} fps, bitrate {} bps", hardware ? "Hardware" : "Software", codec->name,
                     width_, height_, framerate_, codec_ctx_->bit_rate);
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
        int64_t frame_count_ = 0;
        bool is_open_ = false;
        bool in_call_ = false;
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
        if (auto error = io::video::videoEncodingRangeError(options.framerate, options.crf))
            return encodeError(ErrorCode::InvalidArgument, std::move(*error));
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
