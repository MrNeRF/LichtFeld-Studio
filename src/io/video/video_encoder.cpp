/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "video_encoder.hpp"
#include "color_convert.cuh"
#include "core/error.hpp"
#include "core/error_reporter.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/provenance.hpp"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <cuda_runtime.h>
#include <format>
#include <optional>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/opt.h>
#include <libavutil/spherical.h>
#include <libavutil/stereo3d.h>
}

namespace lfs::io::video {

    namespace {
        constexpr int DEFAULT_FRAMERATE = 30;
        constexpr int NVENC_FRAME_POOL_SIZE = 4;
        constexpr int NVENC_QP_OFFSET = 3;

        void applyProvenanceMetadata(AVFormatContext* fmt_ctx, const VideoExportOptions& opts) {
            if (!fmt_ctx || !opts.provenance)
                return;
            if (av_dict_set(&fmt_ctx->metadata, "comment",
                            core::provenance_to_json(*opts.provenance).c_str(), 0) < 0) {
                LOG_WARN("Failed to set video provenance comment metadata");
            }
        }

        std::string encoderProbeError(
            const VideoExportOptions& options,
            const std::string& nvenc_error,
            const std::string& fallback_error) {
            return std::format(
                "H.264 encoding is unavailable for {}x{}. h264_nvenc: {}; H.264 fallback: {}",
                options.width, options.height, nvenc_error, fallback_error);
        }

        std::string codecOpenFailure(
            const AVCodec* codec,
            const int width,
            const int height,
            const char* error) {
            std::string reason = std::format("{} codec open failed: {}", codec->name, error);
            // These codec-specific limits make FFmpeg's generic open error actionable; the
            // encoder open result remains the sole capability check.
            if (std::strcmp(codec->name, "h264_nvenc") == 0 && (width > 4096 || height > 4096)) {
                reason += std::format(" (NVENC side limit is 4096 pixels; requested {}x{})", width, height);
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
                                         AV_PKT_DATA_SPHERICAL,
                                         spherical,
                                         spherical_size,
                                         0)) {
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
                                         AV_PKT_DATA_STEREO3D,
                                         stereo,
                                         stereo_size,
                                         0)) {
                av_free(stereo);
                return false;
            }
            return true;
        }

    } // namespace

    class VideoEncoderImpl {
    public:
        ~VideoEncoderImpl() { cleanup(); }

        std::expected<void, std::string> open(
            const std::filesystem::path& path,
            const VideoExportOptions& opts_in) {

            VideoExportOptions opts = opts_in;
            if (!opts.provenance) {
                opts.provenance = core::make_minimal_provenance_stamp();
            }

            if (is_open_)
                return std::unexpected("Encoder is already open");
            if (const auto validation = validateVideoEncodingOptions(opts); !validation)
                return std::unexpected(validation.error());

            auto extension = core::path_to_utf8(path.extension());
            std::transform(extension.begin(), extension.end(), extension.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (!extension.empty() && extension != ".mp4")
                return std::unexpected("Video export supports only MP4 (.mp4); unsupported filename extension");

            width_ = opts.width;
            height_ = opts.height;
            framerate_ = opts.framerate;
            y_plane_bytes_ = static_cast<size_t>(width_) * static_cast<size_t>(height_);
            uv_plane_bytes_ = y_plane_bytes_ / 4;

            if (const auto nvenc_error = tryInitNvenc(path, opts); nvenc_error) {
                cleanup();
                LOG_INFO("NVENC unavailable, falling back to x264");
                if (const auto result = initX264(path, opts); !result) {
                    const std::string fallback_error = result.error();
                    cleanup();
                    return std::unexpected(encoderProbeError(opts, *nvenc_error, fallback_error));
                }
            }

            is_open_ = true;
            frame_count_ = 0;
            return {};
        }

        std::expected<void, std::string> writeFrameGpu(
            const void* const rgb_gpu_ptr,
            const int width,
            const int height,
            const cudaStream_t stream) {

            if (!is_open_) {
                return std::unexpected("Encoder not open");
            }
            if (!rgb_gpu_ptr) {
                return std::unexpected("GPU frame pointer is null");
            }
            if (width != width_ || height != height_) {
                return std::unexpected("Frame size mismatch");
            }

            // Boundary for CUDA launches that now throw lfs::Exception
            // (rgbToNv12Cuda / rgbToYuv420pCuda via LFS_CUDA_LAUNCH_CHECK).
            // Keep the expected-based API non-throwing so the GUI export
            // jthread is not terminate-on-exception (Phase 6B-3c-2).
            try {
                return use_nvenc_ ? writeFrameNvenc(rgb_gpu_ptr, stream)
                                  : writeFrameX264Gpu(rgb_gpu_ptr, stream);
            } catch (const lfs::Exception& e) {
                lfs::Error error = lfs::Error(e.error())
                                       .with_context("write video frame", LFS_SOURCE_SITE_CURRENT(),
                                                     lfs::SmallFields{}
                                                         .add("width", static_cast<std::int64_t>(width_))
                                                         .add("height", static_cast<std::int64_t>(height_))
                                                         .add("frame", frame_count_));
                lfs::core::ErrorReporter::get().report(error, lfs::core::ReportChannel::OwnerLog);
                return std::unexpected(std::string(e.what()));
            } catch (const std::exception& e) {
                return std::unexpected(std::string(e.what()));
            }
        }

        std::expected<void, std::string> close() {
            if (!is_open_)
                return {};

            std::string close_error;
            if (const auto result = encodeFrame(nullptr); !result) {
                LOG_WARN("Flush error: {}", result.error());
                close_error = result.error();
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
                return std::unexpected(std::move(close_error));
            return {};
        }

        [[nodiscard]] bool isOpen() const { return is_open_; }

    private:
        std::optional<std::string> tryInitNvenc(
            const std::filesystem::path& path,
            const VideoExportOptions& opts) {
            const AVCodec* const codec = avcodec_find_encoder_by_name("h264_nvenc");
            if (!codec) {
                return "encoder unavailable";
            }

            const std::string path_utf8 = lfs::core::path_to_native_utf8(path);

            int ret = avformat_alloc_output_context2(
                &fmt_ctx_, nullptr, "mp4", path_utf8.c_str());
            if (ret < 0 || !fmt_ctx_)
                return "MP4 output context creation failed";

            stream_ = avformat_new_stream(fmt_ctx_, nullptr);
            if (!stream_)
                return "stream creation failed";
            stream_->id = 0;

            codec_ctx_ = avcodec_alloc_context3(codec);
            if (!codec_ctx_)
                return "codec context allocation failed";

            codec_ctx_->width = width_;
            codec_ctx_->height = height_;
            codec_ctx_->time_base = AVRational{1, framerate_};
            codec_ctx_->framerate = AVRational{framerate_, 1};
            codec_ctx_->pix_fmt = AV_PIX_FMT_CUDA;
            codec_ctx_->colorspace = AVCOL_SPC_SMPTE170M;
            codec_ctx_->color_range = AVCOL_RANGE_MPEG;
            codec_ctx_->gop_size = framerate_;
            codec_ctx_->max_b_frames = 0;

            av_opt_set(codec_ctx_->priv_data, "preset", "p4", 0);
            av_opt_set(codec_ctx_->priv_data, "tune", "hq", 0);
            av_opt_set(codec_ctx_->priv_data, "rc", "constqp", 0);
            av_opt_set_int(codec_ctx_->priv_data, "qp", opts.crf + NVENC_QP_OFFSET, 0);

            if (fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
                codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
            }

            ret = av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
            if (ret < 0) {
                char err[AV_ERROR_MAX_STRING_SIZE];
                av_strerror(ret, err, sizeof(err));
                return std::string("CUDA hardware context failed: ") + err;
            }
            codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);

            hw_frames_ctx_ = av_hwframe_ctx_alloc(hw_device_ctx_);
            if (!hw_frames_ctx_)
                return "CUDA frame context allocation failed";

            auto* const frames_ctx = reinterpret_cast<AVHWFramesContext*>(hw_frames_ctx_->data);
            frames_ctx->format = AV_PIX_FMT_CUDA;
            frames_ctx->sw_format = AV_PIX_FMT_NV12;
            frames_ctx->width = width_;
            frames_ctx->height = height_;
            frames_ctx->initial_pool_size = NVENC_FRAME_POOL_SIZE;

            ret = av_hwframe_ctx_init(hw_frames_ctx_);
            if (ret < 0) {
                char err[AV_ERROR_MAX_STRING_SIZE];
                av_strerror(ret, err, sizeof(err));
                return std::string("CUDA frame context initialization failed: ") + err;
            }
            codec_ctx_->hw_frames_ctx = av_buffer_ref(hw_frames_ctx_);

            ret = avcodec_open2(codec_ctx_, codec, nullptr);
            if (ret < 0) {
                char err[AV_ERROR_MAX_STRING_SIZE];
                av_strerror(ret, err, sizeof(err));
                return codecOpenFailure(codec, width_, height_, err);
            }

            ret = avcodec_parameters_from_context(stream_->codecpar, codec_ctx_);
            if (ret < 0) {
                char err[AV_ERROR_MAX_STRING_SIZE];
                av_strerror(ret, err, sizeof(err));
                return std::string("codec parameter copy failed: ") + err;
            }
            if (opts.spherical_video && !applySphericalMetadata(fmt_ctx_, stream_))
                return "spherical video metadata setup failed";
            stream_->time_base = codec_ctx_->time_base;

            if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE)) {
                ret = avio_open(&fmt_ctx_->pb, path_utf8.c_str(), AVIO_FLAG_WRITE);
                if (ret < 0) {
                    char err[AV_ERROR_MAX_STRING_SIZE];
                    av_strerror(ret, err, sizeof(err));
                    return std::string("output file open failed: ") + err;
                }
            }

            applyProvenanceMetadata(fmt_ctx_, opts);

            ret = avformat_write_header(fmt_ctx_, nullptr);
            if (ret < 0) {
                char err[AV_ERROR_MAX_STRING_SIZE];
                av_strerror(ret, err, sizeof(err));
                return std::string("MP4 header write failed: ") + err;
            }

            frame_ = av_frame_alloc();
            if (!frame_)
                return "frame allocation failed";
            ret = av_hwframe_get_buffer(hw_frames_ctx_, frame_, 0);
            if (ret < 0) {
                char err[AV_ERROR_MAX_STRING_SIZE];
                av_strerror(ret, err, sizeof(err));
                return std::string("NVENC frame allocation failed: ") + err;
            }

            packet_ = av_packet_alloc();
            if (!packet_)
                return "packet allocation failed";

            use_nvenc_ = true;
            LOG_INFO("NVENC: {}x{} @ {} fps", width_, height_, framerate_);
            return std::nullopt;
        }

        std::expected<void, std::string> initX264(
            const std::filesystem::path& path,
            const VideoExportOptions& opts) {

            const std::string path_utf8 = lfs::core::path_to_native_utf8(path);

            int ret = avformat_alloc_output_context2(
                &fmt_ctx_, nullptr, "mp4", path_utf8.c_str());
            if (ret < 0 || !fmt_ctx_) {
                return std::unexpected("MP4 context creation failed");
            }

            const AVCodec* const codec = avcodec_find_encoder(AV_CODEC_ID_H264);
            if (!codec) {
                return std::unexpected("H.264 encoder not found");
            }

            stream_ = avformat_new_stream(fmt_ctx_, nullptr);
            if (!stream_) {
                return std::unexpected("Stream creation failed");
            }
            stream_->id = 0;

            codec_ctx_ = avcodec_alloc_context3(codec);
            if (!codec_ctx_) {
                return std::unexpected("Codec context allocation failed");
            }

            codec_ctx_->width = width_;
            codec_ctx_->height = height_;
            codec_ctx_->time_base = AVRational{1, framerate_};
            codec_ctx_->framerate = AVRational{framerate_, 1};
            codec_ctx_->pix_fmt = AV_PIX_FMT_YUV420P;
            codec_ctx_->colorspace = AVCOL_SPC_SMPTE170M;
            codec_ctx_->color_range = AVCOL_RANGE_MPEG;
            codec_ctx_->gop_size = framerate_;
            codec_ctx_->max_b_frames = 2;
            codec_ctx_->thread_count = 0;

            av_opt_set_int(codec_ctx_->priv_data, "crf", opts.crf, 0);
            av_opt_set(codec_ctx_->priv_data, "preset", "fast", 0);

            if (fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
                codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
            }

            ret = avcodec_open2(codec_ctx_, codec, nullptr);
            if (ret < 0) {
                char err[AV_ERROR_MAX_STRING_SIZE];
                av_strerror(ret, err, sizeof(err));
                return std::unexpected(codecOpenFailure(codec, width_, height_, err));
            }

            ret = avcodec_parameters_from_context(stream_->codecpar, codec_ctx_);
            if (ret < 0) {
                return std::unexpected("Codec parameters copy failed");
            }
            if (opts.spherical_video && !applySphericalMetadata(fmt_ctx_, stream_))
                return std::unexpected("Spherical video metadata setup failed");
            stream_->time_base = codec_ctx_->time_base;

            if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE)) {
                ret = avio_open(&fmt_ctx_->pb, path_utf8.c_str(), AVIO_FLAG_WRITE);
                if (ret < 0) {
                    char err[AV_ERROR_MAX_STRING_SIZE];
                    av_strerror(ret, err, sizeof(err));
                    return std::unexpected(std::string("File open failed: ") + err);
                }
            }

            applyProvenanceMetadata(fmt_ctx_, opts);

            ret = avformat_write_header(fmt_ctx_, nullptr);
            if (ret < 0) {
                char err[AV_ERROR_MAX_STRING_SIZE];
                av_strerror(ret, err, sizeof(err));
                return std::unexpected(std::string("Header write failed: ") + err);
            }

            frame_ = av_frame_alloc();
            if (!frame_) {
                return std::unexpected("Frame allocation failed");
            }
            frame_->format = AV_PIX_FMT_YUV420P;
            frame_->width = width_;
            frame_->height = height_;

            ret = av_frame_get_buffer(frame_, 0);
            if (ret < 0) {
                return std::unexpected("Frame buffer allocation failed");
            }

            packet_ = av_packet_alloc();
            if (!packet_) {
                return std::unexpected("Packet allocation failed");
            }

            if (const auto allocation = allocateGpuBuffers(); !allocation)
                return allocation;
            LOG_INFO("x264: {}x{} @ {} fps, CRF {}", width_, height_, framerate_, opts.crf);
            return {};
        }

        [[nodiscard]] std::expected<void, std::string> allocateGpuBuffers() {
            if (auto result = checkCuda(cudaMalloc(&y_gpu_, y_plane_bytes_), "Video Y-plane GPU allocation"); !result)
                return result;
            if (auto result = checkCuda(cudaMalloc(&u_gpu_, uv_plane_bytes_), "Video U-plane GPU allocation"); !result)
                return result;
            if (auto result = checkCuda(cudaMalloc(&v_gpu_, uv_plane_bytes_), "Video V-plane GPU allocation"); !result)
                return result;
            if (auto result = checkCuda(cudaMallocHost(&y_pinned_, y_plane_bytes_), "Video Y-plane pinned allocation"); !result)
                return result;
            if (auto result = checkCuda(cudaMallocHost(&u_pinned_, uv_plane_bytes_), "Video U-plane pinned allocation"); !result)
                return result;
            return checkCuda(cudaMallocHost(&v_pinned_, uv_plane_bytes_), "Video V-plane pinned allocation");
        }

        [[nodiscard]] static std::expected<void, std::string> checkCuda(
            const cudaError_t status,
            const char* const operation) {
            if (status == cudaSuccess)
                return {};
            return std::unexpected(std::format(
                "{} failed: {} ({})", operation, cudaGetErrorString(status), cudaGetErrorName(status)));
        }

        std::expected<void, std::string> writeFrameNvenc(
            const void* const rgb_gpu_ptr,
            const cudaStream_t stream) {

            rgbToNv12Cuda(
                static_cast<const float*>(rgb_gpu_ptr),
                frame_->data[0], frame_->data[1],
                width_, height_,
                frame_->linesize[0], frame_->linesize[1],
                stream);
            if (auto result = checkCuda(cudaGetLastError(), "RGB-to-NV12 kernel launch"); !result)
                return result;

            const auto sync_status = stream ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
            if (auto result = checkCuda(sync_status, "RGB-to-NV12 synchronization"); !result)
                return result;

            frame_->pts = frame_count_;
            if (auto result = encodeFrame(frame_); !result)
                return result;
            ++frame_count_;
            return {};
        }

        std::expected<void, std::string> writeFrameX264Gpu(
            const void* const rgb_gpu_ptr,
            const cudaStream_t stream) {

            rgbToYuv420pCuda(
                static_cast<const float*>(rgb_gpu_ptr),
                y_gpu_, u_gpu_, v_gpu_,
                width_, height_, stream);
            if (auto result = checkCuda(cudaGetLastError(), "RGB-to-YUV420P kernel launch"); !result)
                return result;

            if (auto result = checkCuda(
                    cudaMemcpyAsync(y_pinned_, y_gpu_, y_plane_bytes_, cudaMemcpyDeviceToHost, stream),
                    "Video Y-plane copy");
                !result)
                return result;
            if (auto result = checkCuda(
                    cudaMemcpyAsync(u_pinned_, u_gpu_, uv_plane_bytes_, cudaMemcpyDeviceToHost, stream),
                    "Video U-plane copy");
                !result)
                return result;
            if (auto result = checkCuda(
                    cudaMemcpyAsync(v_pinned_, v_gpu_, uv_plane_bytes_, cudaMemcpyDeviceToHost, stream),
                    "Video V-plane copy");
                !result)
                return result;

            const auto sync_status = stream ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
            if (auto result = checkCuda(sync_status, "Video frame copy synchronization"); !result)
                return result;

            const int ret = av_frame_make_writable(frame_);
            if (ret < 0) {
                return std::unexpected("Frame not writable");
            }

            const int half_width = width_ / 2;
            const int half_height = height_ / 2;

            for (int row = 0; row < height_; ++row) {
                memcpy(frame_->data[0] + row * frame_->linesize[0],
                       y_pinned_ + row * width_, width_);
            }
            for (int row = 0; row < half_height; ++row) {
                memcpy(frame_->data[1] + row * frame_->linesize[1],
                       u_pinned_ + row * half_width, half_width);
                memcpy(frame_->data[2] + row * frame_->linesize[2],
                       v_pinned_ + row * half_width, half_width);
            }

            frame_->pts = frame_count_;
            if (auto result = encodeFrame(frame_); !result)
                return result;
            ++frame_count_;
            return {};
        }

        std::expected<void, std::string> encodeFrame(AVFrame* const frame) {
            // Both RGB conversion paths produce BT.601 limited-range planes.
            if (frame) {
                frame->colorspace = AVCOL_SPC_SMPTE170M;
                frame->color_range = AVCOL_RANGE_MPEG;
            }
            int ret = avcodec_send_frame(codec_ctx_, frame);
            if (ret < 0) {
                char err[AV_ERROR_MAX_STRING_SIZE];
                av_strerror(ret, err, sizeof(err));
                return std::unexpected(std::string("Send frame error: ") + err);
            }

            while (ret >= 0) {
                ret = avcodec_receive_packet(codec_ctx_, packet_);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                    break;
                if (ret < 0) {
                    char err[AV_ERROR_MAX_STRING_SIZE];
                    av_strerror(ret, err, sizeof(err));
                    return std::unexpected(std::string("Receive packet error: ") + err);
                }

                // Each submitted frame spans one tick of the fixed-rate codec time base.
                // The muxer cannot infer this from timestamps for a single-frame video.
                if (packet_->duration <= 0)
                    packet_->duration = 1;
                av_packet_rescale_ts(packet_, codec_ctx_->time_base, stream_->time_base);
                packet_->stream_index = stream_->index;

                ret = av_interleaved_write_frame(fmt_ctx_, packet_);
                av_packet_unref(packet_);

                if (ret < 0) {
                    char err[AV_ERROR_MAX_STRING_SIZE];
                    av_strerror(ret, err, sizeof(err));
                    return std::unexpected(std::string("Write frame error: ") + err);
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
            if (y_gpu_) {
                cudaFree(y_gpu_);
                y_gpu_ = nullptr;
            }
            if (u_gpu_) {
                cudaFree(u_gpu_);
                u_gpu_ = nullptr;
            }
            if (v_gpu_) {
                cudaFree(v_gpu_);
                v_gpu_ = nullptr;
            }

            if (y_pinned_) {
                cudaFreeHost(y_pinned_);
                y_pinned_ = nullptr;
            }
            if (u_pinned_) {
                cudaFreeHost(u_pinned_);
                u_pinned_ = nullptr;
            }
            if (v_pinned_) {
                cudaFreeHost(v_pinned_);
                v_pinned_ = nullptr;
            }

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
        }

        AVFormatContext* fmt_ctx_ = nullptr;
        AVCodecContext* codec_ctx_ = nullptr;
        AVStream* stream_ = nullptr;
        AVFrame* frame_ = nullptr;
        AVPacket* packet_ = nullptr;
        AVBufferRef* hw_device_ctx_ = nullptr;
        AVBufferRef* hw_frames_ctx_ = nullptr;

        uint8_t* y_gpu_ = nullptr;
        uint8_t* u_gpu_ = nullptr;
        uint8_t* v_gpu_ = nullptr;

        uint8_t* y_pinned_ = nullptr;
        uint8_t* u_pinned_ = nullptr;
        uint8_t* v_pinned_ = nullptr;

        int width_ = 0;
        int height_ = 0;
        int framerate_ = DEFAULT_FRAMERATE;
        int64_t frame_count_ = 0;
        size_t y_plane_bytes_ = 0;
        size_t uv_plane_bytes_ = 0;
        bool is_open_ = false;
        bool use_nvenc_ = false;
    };

    VideoEncoder::VideoEncoder() : impl_(std::make_unique<VideoEncoderImpl>()) {}
    VideoEncoder::~VideoEncoder() = default;
    VideoEncoder::VideoEncoder(VideoEncoder&&) noexcept = default;
    VideoEncoder& VideoEncoder::operator=(VideoEncoder&&) noexcept = default;

    std::expected<void, std::string> VideoEncoder::open(
        const std::filesystem::path& path, const VideoExportOptions& opts) {
        return impl_->open(path, opts);
    }

    std::expected<void, std::string> VideoEncoder::writeFrame(
        std::span<const uint8_t> /*rgba_data*/, const int /*width*/, const int /*height*/) {
        return std::unexpected("CPU path not implemented - use writeFrameGpu");
    }

    std::expected<void, std::string> VideoEncoder::writeFrameGpu(
        const void* const rgb_gpu_ptr, const int width, const int height, void* const stream) {
        return impl_->writeFrameGpu(rgb_gpu_ptr, width, height, static_cast<cudaStream_t>(stream));
    }

    std::expected<void, std::string> VideoEncoder::close() {
        return impl_->close();
    }

    bool VideoEncoder::isOpen() const {
        return impl_->isOpen();
    }

} // namespace lfs::io::video
