/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "video_encoder.hpp"
#include "core/error.hpp"
#include "core/error_reporter.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/provenance.hpp"
#include "core/tensor_backend.hpp"
#include <algorithm>
#include <array>
#if LFS_HAS_CUDA
#include <cuda_runtime.h>
#endif
#include <format>
#include <string_view>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
#include <libavutil/hwcontext.h>
#if LFS_HAS_CUDA
#include <libavutil/hwcontext_cuda.h>
#endif
#include <libavutil/opt.h>
}

namespace lfs::io::video {

    namespace {
        constexpr int DEFAULT_FRAMERATE = 30;
        constexpr int NVENC_FRAME_POOL_SIZE = 4;
        constexpr int NVENC_QP_OFFSET = 3;

        struct YuvPlanes {
            core::Tensor y;
            core::Tensor u;
            core::Tensor v;
        };

        YuvPlanes rgbToYuv420p(const core::Tensor& rgb) {
            const int height = static_cast<int>(rgb.size(0));
            const int width = static_cast<int>(rgb.size(1));
            const auto bytes = (rgb.clamp(0.0f, 1.0f) * 255.0f + 0.5f).floor();
            const auto channel = [](const core::Tensor& image, const size_t c) {
                return image.slice(2, c, c + 1).reshape({static_cast<int>(image.size(0)), static_cast<int>(image.size(1))});
            };
            const auto y = ((channel(bytes, 0) * 66.0f +
                             channel(bytes, 1) * 129.0f +
                             channel(bytes, 2) * 25.0f + 128.0f) /
                            256.0f)
                               .floor()
                               .add(16.0f)
                               .to(core::DataType::UInt8);
            const auto chroma = (bytes.reshape({height / 2, 2, width / 2, 2, 3})
                                     .sum({1, 3}) /
                                 4.0f)
                                    .floor();
            const auto u = ((channel(chroma, 0) * -38.0f +
                             channel(chroma, 1) * -74.0f +
                             channel(chroma, 2) * 112.0f + 128.0f) /
                            256.0f)
                               .floor()
                               .add(128.0f)
                               .clamp(0.0f, 255.0f)
                               .to(core::DataType::UInt8);
            const auto v = ((channel(chroma, 0) * 112.0f +
                             channel(chroma, 1) * -94.0f +
                             channel(chroma, 2) * -18.0f + 128.0f) /
                            256.0f)
                               .floor()
                               .add(128.0f)
                               .clamp(0.0f, 255.0f)
                               .to(core::DataType::UInt8);
            return {y, u, v};
        }

        void applyProvenanceMetadata(AVFormatContext* fmt_ctx, const VideoExportOptions& opts) {
            if (!fmt_ctx || !opts.provenance)
                return;
            if (av_dict_set(&fmt_ctx->metadata, "comment",
                            core::provenance_to_json(*opts.provenance).c_str(), 0) < 0) {
                LOG_WARN("Failed to set video provenance comment metadata");
            }
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

            const size_t width = static_cast<size_t>(opts.width);
            const size_t height = static_cast<size_t>(opts.height);

            width_ = opts.width;
            height_ = opts.height;
            framerate_ = opts.framerate;
#if LFS_HAS_CUDA
            const bool hardware = core::default_gpu_backend() == core::GpuBackend::CUDA && tryInitNvenc(path, opts);
#elif defined(__APPLE__)
            const bool hardware = tryInitVideoToolbox(path, opts);
#else
            const bool hardware = false;
#endif
            if (!hardware) {
                cleanup();
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

        std::expected<void, std::string> writeFrame(const core::Tensor& rgb_hwc) {

            if (!is_open_) {
                return std::unexpected("Encoder not open");
            }
            if (!rgb_hwc.is_valid() || rgb_hwc.dtype() != core::DataType::Float32 ||
                rgb_hwc.ndim() != 3 || rgb_hwc.size(2) != 3) {
                return std::unexpected("Video frame must be an HWC float32 RGB tensor");
            }
            if (rgb_hwc.size(1) != static_cast<size_t>(width_) ||
                rgb_hwc.size(0) != static_cast<size_t>(height_)) {
                return std::unexpected("Frame size mismatch");
            }

            try {
                const auto frame = use_nvenc_ && rgb_hwc.device() == core::Device::CPU
                                       ? rgb_hwc.gpu()
                                       : rgb_hwc;
                const auto planes = rgbToYuv420p(frame.contiguous());
#if LFS_HAS_CUDA
                return use_nvenc_ ? writeFrameNvenc(planes)
                                  : writeFrameSoftwareH264(planes);
#else
                return writeFrameSoftwareH264(planes);
#endif
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
#if LFS_HAS_CUDA
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

            applyProvenanceMetadata(fmt_ctx_, opts);

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
#endif

#ifdef __APPLE__
        // The Mac's media engine encodes the same YUV420P frames as the
        // software path, from system memory.
        bool tryInitVideoToolbox(const std::filesystem::path& path, const VideoExportOptions& opts) {
            const AVCodec* const codec = avcodec_find_encoder_by_name("h264_videotoolbox");
            if (!codec) {
                LOG_DEBUG("VideoToolbox H.264 encoder not available");
                return false;
            }
            if (const auto result = initH264(path, opts, codec); !result) {
                LOG_DEBUG("VideoToolbox H.264 encoder failed: {}", result.error());
                return false;
            }
            return true;
        }
#endif

        // An H.264 encoder fed YUV420P frames from system memory: the software
        // encoder, or VideoToolbox on Macs.
        std::expected<void, std::string> initH264(
            const std::filesystem::path& path,
            const VideoExportOptions& opts,
            const AVCodec* const codec) {

            const std::string path_utf8 = lfs::core::path_to_utf8(path);

            int ret = avformat_alloc_output_context2(&fmt_ctx_, nullptr, "mp4", path_utf8.c_str());
            if (ret < 0 || !fmt_ctx_) {
                return std::unexpected("MP4 context creation failed");
            }

            if (!codec) {
                return std::unexpected("H.264 encoder not found");
            }
            const bool hardware = std::string_view(codec->name) == "h264_videotoolbox";

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
            codec_ctx_->max_b_frames = hardware ? 0 : 2;
            codec_ctx_->thread_count = 0;

            // Map CRF 18 to 0.1 bits per pixel per frame, with a 250 kbps floor and linear quality scaling.
            const int64_t crf_scale = 51 - opts.crf;
            codec_ctx_->bit_rate = std::max<int64_t>(
                250'000, static_cast<int64_t>(width_) * height_ * framerate_ * crf_scale / 330);
            if (hardware) {
                // Fail rather than fall back to Apple's own software encoder.
                av_opt_set_int(codec_ctx_->priv_data, "allow_sw", 0, 0);
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

            LOG_INFO("{} H.264 ({}): {}x{} @ {} fps, bitrate {} bps", hardware ? "Hardware" : "Software", codec->name,
                     width_, height_, framerate_, codec_ctx_->bit_rate);
            return {};
        }

#if LFS_HAS_CUDA
        [[nodiscard]] static std::expected<void, std::string> checkCuda(
            const cudaError_t status,
            const char* const operation) {
            if (status == cudaSuccess)
                return {};
            return std::unexpected(std::format(
                "{} failed: {} ({})", operation, cudaGetErrorString(status), cudaGetErrorName(status)));
        }

        std::expected<void, std::string> writeFrameNvenc(const YuvPlanes& planes) {
            if (core::gpu_backend_of(planes.y) != core::GpuBackend::CUDA) {
                return std::unexpected("NVENC requires a CUDA tensor frame");
            }
            const auto stream = planes.y.stream();
            auto y = core::Tensor::from_blob(frame_->data[0],
                                             {static_cast<size_t>(height_),
                                              static_cast<size_t>(frame_->linesize[0])},
                                             core::Device::GPU, core::DataType::UInt8, stream);
            auto uv = core::Tensor::from_blob(frame_->data[1],
                                              {static_cast<size_t>(height_ / 2),
                                               static_cast<size_t>(frame_->linesize[1])},
                                              core::Device::GPU, core::DataType::UInt8, stream);
            y.slice(1, 0, width_).copy_from(planes.y);
            uv.slice(1, 0, width_).copy_from(core::Tensor::stack({planes.u, planes.v}, 2).reshape({height_ / 2, width_}));
            const auto sync_status = stream ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
            if (auto result = checkCuda(sync_status, "NVENC frame synchronization"); !result)
                return result;

            frame_->pts = frame_count_;
            if (auto result = encodeFrame(frame_); !result)
                return result;
            ++frame_count_;
            return {};
        }
#endif

        std::expected<void, std::string> writeFrameSoftwareH264(const YuvPlanes& planes) {
            const int ret = av_frame_make_writable(frame_);
            if (ret < 0) {
                return std::unexpected("Frame not writable");
            }

            const auto source = std::array{planes.y.to_pageable_host(),
                                           planes.u.to_pageable_host(),
                                           planes.v.to_pageable_host()};
            for (int plane = 0; plane < 3; ++plane) {
                const size_t rows = plane == 0 ? height_ : height_ / 2;
                const size_t columns = plane == 0 ? width_ : width_ / 2;
                auto target = core::Tensor::from_blob(frame_->data[plane],
                                                      {rows, static_cast<size_t>(frame_->linesize[plane])},
                                                      core::Device::CPU, core::DataType::UInt8);
                target.slice(1, 0, columns).copy_from(source[plane]);
            }

            frame_->pts = frame_count_;
            if (auto result = encodeFrame(frame_); !result)
                return result;
            ++frame_count_;
            return {};
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
        std::span<const uint8_t> rgba_data, const int width, const int height) {
        if (width <= 0 || height <= 0 ||
            rgba_data.size() != static_cast<size_t>(width) * height * 4) {
            return std::unexpected("CPU RGBA frame size mismatch");
        }
        auto rgba = core::Tensor::from_blob(const_cast<uint8_t*>(rgba_data.data()),
                                            {static_cast<size_t>(height), static_cast<size_t>(width), 4},
                                            core::Device::CPU, core::DataType::UInt8);
        auto rgb = rgba.slice(2, 0, 3).to(core::DataType::Float32).div(255.0f);
        return impl_->writeFrame(rgb);
    }

    std::expected<void, std::string> VideoEncoder::writeFrame(const core::Tensor& rgb_hwc) {
        return impl_->writeFrame(rgb_hwc);
    }

    std::expected<void, std::string> VideoEncoder::close() {
        return impl_->close();
    }

    bool VideoEncoder::isOpen() const {
        return impl_->isOpen();
    }

} // namespace lfs::io::video
