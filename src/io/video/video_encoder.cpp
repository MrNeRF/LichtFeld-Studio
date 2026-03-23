/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "video_encoder.hpp"
#include "color_convert.cuh"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include <cuda_runtime.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/opt.h>
}

namespace lfs::io::video {

    namespace {
        constexpr int DEFAULT_FRAMERATE = 30;
        constexpr int NVENC_FRAME_POOL_SIZE = 4;
        constexpr int NVENC_QP_OFFSET = 3;
    } // namespace

    class VideoEncoderImpl {
    public:
        ~VideoEncoderImpl() { cleanup(); }

        std::expected<void, std::string> open(
            const std::filesystem::path& path,
            const VideoExportOptions& opts) {

            width_ = opts.width;
            height_ = opts.height;
            framerate_ = opts.framerate;

            if (!tryInitNvenc(path, opts)) {
                LOG_INFO("NVENC unavailable, falling back to x264");
                if (const auto result = initX264(path, opts); !result) {
                    return result;
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
            if (width != width_ || height != height_) {
                return std::unexpected("Frame size mismatch");
            }

            return use_nvenc_ ? writeFrameNvenc(rgb_gpu_ptr, stream)
                              : writeFrameX264Gpu(rgb_gpu_ptr, stream);
        }

        std::expected<void, std::string> close() {
            if (!is_open_)
                return {};

            if (const auto result = encodeFrame(nullptr); !result) {
                LOG_WARN("Flush error: {}", result.error());
            }

            if (fmt_ctx_) {
                av_write_trailer(fmt_ctx_);
            }

            LOG_INFO("Video: {} frames encoded", frame_count_);
            cleanup();
            return {};
        }

        [[nodiscard]] bool isOpen() const { return is_open_; }

    private:
        bool tryInitNvenc(const std::filesystem::path& path, const VideoExportOptions& opts) {
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

            ret = avformat_write_header(fmt_ctx_, nullptr);
            if (ret < 0) {
                cleanupHwContexts();
                return false;
            }

            frame_ = av_frame_alloc();
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
            LOG_INFO("NVENC: {}x{} @ {} fps", width_, height_, framerate_);
            return true;
        }

        std::expected<void, std::string> initX264(
            const std::filesystem::path& path,
            const VideoExportOptions& opts) {

            const std::string path_utf8 = lfs::core::path_to_utf8(path);

            int ret = avformat_alloc_output_context2(&fmt_ctx_, nullptr, "mp4", path_utf8.c_str());
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
                return std::unexpected(std::string("Codec open failed: ") + err);
            }

            ret = avcodec_parameters_from_context(stream_->codecpar, codec_ctx_);
            if (ret < 0) {
                return std::unexpected("Codec parameters copy failed");
            }
            stream_->time_base = codec_ctx_->time_base;

            if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE)) {
                ret = avio_open(&fmt_ctx_->pb, path_utf8.c_str(), AVIO_FLAG_WRITE);
                if (ret < 0) {
                    char err[AV_ERROR_MAX_STRING_SIZE];
                    av_strerror(ret, err, sizeof(err));
                    return std::unexpected(std::string("File open failed: ") + err);
                }
            }

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

            allocateGpuBuffers();
            LOG_INFO("x264: {}x{} @ {} fps, CRF {}", width_, height_, framerate_, opts.crf);
            return {};
        }

        void allocateGpuBuffers() {
            const size_t y_size = static_cast<size_t>(width_ * height_);
            const size_t uv_size = static_cast<size_t>((width_ / 2) * (height_ / 2));

            cudaMalloc(&y_gpu_, y_size);
            cudaMalloc(&u_gpu_, uv_size);
            cudaMalloc(&v_gpu_, uv_size);

            cudaMallocHost(&y_pinned_, y_size);
            cudaMallocHost(&u_pinned_, uv_size);
            cudaMallocHost(&v_pinned_, uv_size);
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

            if (stream) {
                cudaStreamSynchronize(stream);
            } else {
                cudaDeviceSynchronize();
            }

            frame_->pts = frame_count_++;
            return encodeFrame(frame_);
        }

        std::expected<void, std::string> writeFrameX264Gpu(
            const void* const rgb_gpu_ptr,
            const cudaStream_t stream) {

            rgbToYuv420pCuda(
                static_cast<const float*>(rgb_gpu_ptr),
                y_gpu_, u_gpu_, v_gpu_,
                width_, height_, stream);

            const size_t y_size = static_cast<size_t>(width_ * height_);
            const size_t uv_size = static_cast<size_t>((width_ / 2) * (height_ / 2));

            cudaMemcpyAsync(y_pinned_, y_gpu_, y_size, cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(u_pinned_, u_gpu_, uv_size, cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(v_pinned_, v_gpu_, uv_size, cudaMemcpyDeviceToHost, stream);

            if (stream) {
                cudaStreamSynchronize(stream);
            } else {
                cudaDeviceSynchronize();
            }

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

            frame_->pts = frame_count_++;
            return encodeFrame(frame_);
        }

        std::expected<void, std::string> encodeFrame(AVFrame* const frame) {
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
