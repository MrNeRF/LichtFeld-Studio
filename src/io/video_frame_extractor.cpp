/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "video_frame_extractor.hpp"
#include "core/include/core/logger.hpp"
#include "core/path_utils.hpp"
#include "nvcodec_image_loader.hpp"
#include "video/color_convert.cuh"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <cuda_runtime.h>
#include <stb_image_write.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace lfs::io {

    namespace {
        constexpr int JPEG_BATCH_SIZE = 32;

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

        [[nodiscard]] double frameTimestampSeconds(const AVFrame* const frame,
                                                   const double time_base,
                                                   const int frame_index,
                                                   const double fallback_fps) {
            int64_t timestamp = frame->best_effort_timestamp;
            if (timestamp == AV_NOPTS_VALUE)
                timestamp = frame->pts;
            if (timestamp != AV_NOPTS_VALUE && std::isfinite(time_base) && time_base > 0.0)
                return static_cast<double>(timestamp) * time_base;
            return static_cast<double>(frame_index) / std::max(fallback_fps, 0.001);
        }

        [[nodiscard]] int estimateFramesToExtract(const ExtractionMode mode,
                                                  const double trim_duration,
                                                  const double target_fps,
                                                  const int64_t source_frame_count,
                                                  const int frame_step) {
            if (mode == ExtractionMode::FPS)
                return std::max(1, static_cast<int>(std::ceil(std::max(0.0, trim_duration) * target_fps)));

            const auto source_frames = static_cast<double>(std::max<int64_t>(1, source_frame_count));
            return std::max(1, static_cast<int>(std::ceil(source_frames / static_cast<double>(frame_step))));
        }

        bool write_image_file(const std::filesystem::path& path,
                              int width,
                              int height,
                              const void* data,
                              ImageFormat format,
                              int jpg_quality) {
            const std::string path_utf8 = lfs::core::path_to_utf8(path);
            if (format == ImageFormat::JPG) {
                return stbi_write_jpg(path_utf8.c_str(), width, height, 3, data, jpg_quality) != 0;
            }
            return stbi_write_png(path_utf8.c_str(), width, height, 3, data, width * 3) != 0;
        }

        void write_jpeg_to_file(const std::filesystem::path& path, const std::vector<uint8_t>& data) {
            std::ofstream file(path, std::ios::binary);
            if (file) {
                file.write(reinterpret_cast<const char*>(data.data()),
                           static_cast<std::streamsize>(data.size()));
            }
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

        AVPixelFormat get_hw_format(AVCodecContext*, const AVPixelFormat* pix_fmts) {
            for (const AVPixelFormat* p = pix_fmts; *p != -1; p++) {
                if (*p == AV_PIX_FMT_CUDA)
                    return *p;
            }
            return AV_PIX_FMT_NONE;
        }

    } // namespace

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

    class VideoFrameExtractor::Impl {
    public:
        bool extract(const Params& params, std::string& error) {
            AVFormatContext* fmt_ctx = nullptr;
            AVCodecContext* codec_ctx = nullptr;
            SwsContext* sws_ctx = nullptr;
            AVFrame* frame = nullptr;
            AVFrame* sw_frame = nullptr;
            AVPacket* packet = nullptr;
            AVBufferRef* hw_device_ctx = nullptr;

            uint8_t* gpu_batch_buffer = nullptr;
            uint8_t* gpu_rgb_buffer = nullptr;
            uint8_t* cpu_contiguous_buffer = nullptr;
            std::unique_ptr<NvCodecImageLoader> nvcodec;
            bool using_hw_decode = false;

            try {
                const std::string video_path_utf8 = lfs::core::path_to_utf8(params.video_path);

                if (avformat_open_input(&fmt_ctx, video_path_utf8.c_str(), nullptr,
                                        nullptr) < 0) {
                    error = "Failed to open video file";
                    return false;
                }

                if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
                    error = "Failed to find stream info";
                    avformat_close_input(&fmt_ctx);
                    return false;
                }

                int video_stream_idx = -1;
                for (unsigned int i = 0; i < fmt_ctx->nb_streams; i++) {
                    if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                        video_stream_idx = i;
                        break;
                    }
                }

                if (video_stream_idx == -1) {
                    error = "No video stream found";
                    avformat_close_input(&fmt_ctx);
                    return false;
                }

                AVStream* video_stream = fmt_ctx->streams[video_stream_idx];
                const AVCodecID codec_id = video_stream->codecpar->codec_id;

                // Try hardware decoder first
                const char* hw_decoder_name = get_hw_decoder_name(codec_id);
                const AVCodec* codec = nullptr;

                if (hw_decoder_name) {
                    codec = avcodec_find_decoder_by_name(hw_decoder_name);
                    if (codec) {
                        if (av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr,
                                                   nullptr, 0) == 0) {
                            using_hw_decode = true;
                            LOG_INFO("Using NVDEC hardware decoder: {}", hw_decoder_name);
                        } else {
                            codec = nullptr;
                            LOG_WARN("Failed to create CUDA device context, falling back to CPU");
                        }
                    }
                }

                // Fallback to software decoder
                if (!codec) {
                    codec = avcodec_find_decoder(codec_id);
                    if (!codec) {
                        error = "Unsupported codec";
                        avformat_close_input(&fmt_ctx);
                        return false;
                    }
                    LOG_INFO("Using CPU software decoder");
                }

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
                    codec_ctx->get_format = get_hw_format;
                }

                if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
                    error = "Failed to open codec";
                    avcodec_free_context(&codec_ctx);
                    if (hw_device_ctx)
                        av_buffer_unref(&hw_device_ctx);
                    avformat_close_input(&fmt_ctx);
                    return false;
                }

                std::filesystem::create_directories(params.output_dir);

                const int src_width = codec_ctx->width;
                const int src_height = codec_ctx->height;

                // Calculate output dimensions based on resolution mode
                int out_width = src_width;
                int out_height = src_height;
                if (params.resolution_mode == ResolutionMode::Scale) {
                    out_width = static_cast<int>(src_width * params.scale);
                    out_height = static_cast<int>(src_height * params.scale);
                    out_width = (out_width + 1) & ~1; // Ensure even
                    out_height = (out_height + 1) & ~1;
                } else if (params.resolution_mode == ResolutionMode::Custom) {
                    if (params.custom_width > 0 && params.custom_height > 0) {
                        out_width = params.custom_width;
                        out_height = params.custom_height;
                    }
                }

                const size_t frame_size = static_cast<size_t>(out_width) * out_height * 3;
                const bool needs_scale = out_width != src_width || out_height != src_height;

                double video_fps = validFrameRate(video_stream->avg_frame_rate);
                if (video_fps <= 0.0)
                    video_fps = validFrameRate(video_stream->r_frame_rate);
                if (video_fps <= 0.0)
                    video_fps = 30.0;
                const double video_duration = static_cast<double>(fmt_ctx->duration) / AV_TIME_BASE;
                const double time_base = av_q2d(video_stream->time_base);

                // Handle trim range
                const double start_time = params.start_time;
                const double end_time = params.end_time > 0 ? params.end_time : video_duration;
                const double trim_duration = end_time - start_time;

                int64_t total_frames = video_stream->nb_frames;
                if (total_frames == 0) {
                    total_frames = static_cast<int64_t>(trim_duration * video_fps);
                } else {
                    total_frames = static_cast<int64_t>(trim_duration / video_duration * total_frames);
                }

                const int frame_step = std::max(1, params.frame_interval);
                const double target_fps = std::max(params.fps, 0.001);
                const double target_interval = 1.0 / target_fps;
                double next_capture_time = start_time;
                const int estimated_total = estimateFramesToExtract(params.mode, trim_duration, target_fps,
                                                                    total_frames, frame_step);

                // Seek to start time if needed
                if (start_time > 0.1) {
                    int64_t timestamp = static_cast<int64_t>(start_time / time_base);
                    av_seek_frame(fmt_ctx, video_stream_idx, timestamp, AVSEEK_FLAG_BACKWARD);
                    avcodec_flush_buffers(codec_ctx);
                    LOG_INFO("Seeking to start time: {:.2f}s", start_time);
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

                cpu_contiguous_buffer = new uint8_t[frame_size];

                // Only create sws_ctx for software decode path
                if (!using_hw_decode) {
                    sws_ctx = sws_getContext(src_width, src_height, codec_ctx->pix_fmt,
                                             out_width, out_height, AV_PIX_FMT_RGB24,
                                             SWS_BILINEAR, nullptr, nullptr, nullptr);
                    if (!sws_ctx) {
                        error = "Failed to create scaling context";
                        throw std::runtime_error(error);
                    }
                }

                const bool use_gpu_jpeg =
                    params.format == ImageFormat::JPG && NvCodecImageLoader::is_available();

                if (use_gpu_jpeg) {
                    NvCodecImageLoader::Options opts;
                    nvcodec = std::make_unique<NvCodecImageLoader>(opts);

                    cudaError_t cuda_err =
                        cudaMalloc(&gpu_batch_buffer, JPEG_BATCH_SIZE * frame_size);
                    if (cuda_err != cudaSuccess) {
                        LOG_WARN("Failed to allocate GPU batch buffer, falling back to CPU");
                    }

                    // Allocate RGB conversion buffer for hardware decode (only if no scaling)
                    // GPU NV12→RGB doesn't support scaling, so we fall back to CPU for scaled output
                    if (using_hw_decode && gpu_batch_buffer && !needs_scale) {
                        const size_t src_frame_size = static_cast<size_t>(src_width) * src_height * 3;
                        cuda_err = cudaMalloc(&gpu_rgb_buffer, src_frame_size);
                        if (cuda_err != cudaSuccess) {
                            LOG_WARN("Failed to allocate GPU RGB buffer");
                            gpu_rgb_buffer = nullptr;
                        }
                    }
                }

                const bool gpu_encoding_enabled = use_gpu_jpeg && gpu_batch_buffer != nullptr;
                const bool full_gpu_pipeline = using_hw_decode && gpu_encoding_enabled && gpu_rgb_buffer && !needs_scale;
                const auto throw_if_cancelled = [&]() {
                    if (params.cancel_requested && params.cancel_requested())
                        throw ExtractionCancelled{};
                };

                if (full_gpu_pipeline) {
                    LOG_INFO("Full GPU pipeline: NVDEC decode → GPU color convert → GPU JPEG encode");
                } else if (using_hw_decode) {
                    LOG_INFO("Hybrid pipeline: NVDEC decode → CPU transfer → {}",
                             gpu_encoding_enabled ? "GPU encode" : "CPU encode");
                } else if (gpu_encoding_enabled) {
                    LOG_INFO("Using GPU batch JPEG encoding (batch size: {})", JPEG_BATCH_SIZE);
                } else if (params.format == ImageFormat::JPG) {
                    LOG_INFO("Using CPU JPEG encoding");
                } else {
                    LOG_INFO("Using CPU PNG encoding");
                }

                int in_trim_frame_count = 0;
                int decoded_frame_count = 0;
                int saved_count = 0;

                std::vector<void*> batch_gpu_ptrs;
                std::vector<std::filesystem::path> batch_filenames;
                int batch_idx = 0;

                auto flush_jpeg_batch = [&]() {
                    if (batch_gpu_ptrs.empty())
                        return;
                    throw_if_cancelled();

                    auto encoded = nvcodec->encode_batch_rgb_to_jpeg(batch_gpu_ptrs, out_width, out_height,
                                                                     params.jpg_quality);

                    for (size_t i = 0; i < encoded.size(); i++) {
                        if (!encoded[i].empty()) {
                            write_jpeg_to_file(batch_filenames[i], encoded[i]);
                        }
                    }

                    batch_gpu_ptrs.clear();
                    batch_filenames.clear();
                    batch_idx = 0;
                    throw_if_cancelled();
                };

                auto generate_filename = [&](int frame_num) {
                    std::string ext = params.format == ImageFormat::PNG ? ".png" : ".jpg";
                    return params.output_dir / (formatFrameFilenameStem(params.filename_pattern, frame_num) + ext);
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

                auto process_frame_hw = [&](AVFrame* hw_frame) {
                    throw_if_cancelled();
                    std::filesystem::path filename = generate_filename(saved_count + 1);

                    if (full_gpu_pipeline) {
                        // Full GPU path: NV12 on GPU → RGB on GPU → encode on GPU
                        const uint8_t* y_plane = hw_frame->data[0];
                        const uint8_t* uv_plane = hw_frame->data[1];
                        const int y_pitch = hw_frame->linesize[0];
                        const int uv_pitch = hw_frame->linesize[1];

                        video::nv12ToRgbCuda(y_plane, uv_plane, gpu_rgb_buffer,
                                             src_width, src_height, y_pitch, uv_pitch, nullptr);

                        void* dst_ptr = gpu_batch_buffer + batch_idx * frame_size;
                        cudaMemcpyAsync(dst_ptr, gpu_rgb_buffer, frame_size,
                                        cudaMemcpyDeviceToDevice, nullptr);

                        batch_gpu_ptrs.push_back(dst_ptr);
                        batch_filenames.push_back(filename);
                        batch_idx++;

                        if (batch_idx >= JPEG_BATCH_SIZE) {
                            cudaStreamSynchronize(nullptr);
                            flush_jpeg_batch();
                        }
                    } else {
                        // Transfer from GPU to CPU for processing
                        int ret = av_hwframe_transfer_data(sw_frame, hw_frame, 0);
                        if (ret < 0) {
                            LOG_WARN("Failed to transfer frame from GPU");
                            return;
                        }

                        // NV12→RGB with optional scaling
                        SwsContext* nv12_sws = sws_getContext(
                            src_width, src_height, static_cast<AVPixelFormat>(sw_frame->format),
                            out_width, out_height, AV_PIX_FMT_RGB24, SWS_BILINEAR,
                            nullptr, nullptr, nullptr);

                        if (!nv12_sws) {
                            LOG_WARN("Failed to create NV12 scaling context");
                            return;
                        }

                        uint8_t* dst_data[1] = {cpu_contiguous_buffer};
                        int dst_linesize[1] = {out_width * 3};
                        sws_scale(nv12_sws, sw_frame->data, sw_frame->linesize, 0, src_height,
                                  dst_data, dst_linesize);
                        sws_freeContext(nv12_sws);

                        if (gpu_encoding_enabled) {
                            void* dst_ptr = gpu_batch_buffer + batch_idx * frame_size;
                            cudaMemcpy(dst_ptr, cpu_contiguous_buffer, frame_size,
                                       cudaMemcpyHostToDevice);

                            batch_gpu_ptrs.push_back(dst_ptr);
                            batch_filenames.push_back(filename);
                            batch_idx++;

                            if (batch_idx >= JPEG_BATCH_SIZE) {
                                flush_jpeg_batch();
                            }
                        } else if (!write_image_file(filename, out_width, out_height,
                                                     cpu_contiguous_buffer, params.format,
                                                     params.jpg_quality)) {
                            LOG_WARN("Failed to write extracted frame: {}", lfs::core::path_to_utf8(filename));
                        }
                    }

                    saved_count++;

                    if (params.progress_callback) {
                        params.progress_callback(saved_count, estimated_total);
                    }
                    throw_if_cancelled();
                };

                auto process_frame_sw = [&](AVFrame* decoded_frame) {
                    throw_if_cancelled();
                    uint8_t* dst_data[1] = {cpu_contiguous_buffer};
                    int dst_linesize[1] = {out_width * 3};
                    sws_scale(sws_ctx, decoded_frame->data, decoded_frame->linesize, 0, src_height,
                              dst_data, dst_linesize);

                    std::filesystem::path filename = generate_filename(saved_count + 1);

                    if (gpu_encoding_enabled) {
                        void* dst_ptr = gpu_batch_buffer + batch_idx * frame_size;
                        cudaMemcpy(dst_ptr, cpu_contiguous_buffer, frame_size,
                                   cudaMemcpyHostToDevice);

                        batch_gpu_ptrs.push_back(dst_ptr);
                        batch_filenames.push_back(filename);
                        batch_idx++;

                        if (batch_idx >= JPEG_BATCH_SIZE) {
                            flush_jpeg_batch();
                        }
                    } else if (!write_image_file(filename, out_width, out_height,
                                                 cpu_contiguous_buffer, params.format,
                                                 params.jpg_quality)) {
                        LOG_WARN("Failed to write extracted frame: {}", lfs::core::path_to_utf8(filename));
                    }

                    saved_count++;

                    if (params.progress_callback) {
                        params.progress_callback(saved_count, estimated_total);
                    }
                    throw_if_cancelled();
                };

                bool reached_end = false;
                while (!reached_end && av_read_frame(fmt_ctx, packet) >= 0) {
                    throw_if_cancelled();
                    if (packet->stream_index == video_stream_idx) {
                        if (avcodec_send_packet(codec_ctx, packet) == 0) {
                            while (avcodec_receive_frame(codec_ctx, frame) == 0) {
                                // Check if we've reached the end time
                                const double frame_time =
                                    frameTimestampSeconds(frame, time_base, decoded_frame_count++, video_fps);
                                if (frame_time < start_time) {
                                    // Skip frames before start time (due to keyframe seeking)
                                    continue;
                                }
                                if (frame_time > end_time) {
                                    reached_end = true;
                                    break;
                                }

                                if (should_extract_frame(frame_time)) {
                                    if (using_hw_decode) {
                                        process_frame_hw(frame);
                                    } else {
                                        process_frame_sw(frame);
                                    }
                                }
                            }
                        }
                    }
                    av_packet_unref(packet);
                }

                // Flush decoder (only if we haven't reached end time)
                if (!reached_end) {
                    avcodec_send_packet(codec_ctx, nullptr);
                    while (avcodec_receive_frame(codec_ctx, frame) == 0) {
                        throw_if_cancelled();
                        const double frame_time =
                            frameTimestampSeconds(frame, time_base, decoded_frame_count++, video_fps);
                        if (frame_time < start_time)
                            continue;
                        if (frame_time > end_time)
                            break;

                        if (should_extract_frame(frame_time)) {
                            if (using_hw_decode) {
                                process_frame_hw(frame);
                            } else {
                                process_frame_sw(frame);
                            }
                        }
                    }
                }

                if (gpu_encoding_enabled) {
                    if (full_gpu_pipeline) {
                        cudaStreamSynchronize(nullptr);
                    }
                    flush_jpeg_batch();
                }

                LOG_INFO("Extracted {} frames from video", saved_count);

                // Cleanup
                if (sws_ctx)
                    sws_freeContext(sws_ctx);
                av_frame_free(&frame);
                av_frame_free(&sw_frame);
                av_packet_free(&packet);
                avcodec_free_context(&codec_ctx);
                if (hw_device_ctx)
                    av_buffer_unref(&hw_device_ctx);
                avformat_close_input(&fmt_ctx);
                delete[] cpu_contiguous_buffer;
                if (gpu_rgb_buffer)
                    cudaFree(gpu_rgb_buffer);
                if (gpu_batch_buffer)
                    cudaFree(gpu_batch_buffer);

                return true;

            } catch (const std::exception& e) {
                if (sws_ctx)
                    sws_freeContext(sws_ctx);
                if (frame)
                    av_frame_free(&frame);
                if (sw_frame)
                    av_frame_free(&sw_frame);
                if (packet)
                    av_packet_free(&packet);
                if (codec_ctx)
                    avcodec_free_context(&codec_ctx);
                if (hw_device_ctx)
                    av_buffer_unref(&hw_device_ctx);
                if (fmt_ctx)
                    avformat_close_input(&fmt_ctx);
                delete[] cpu_contiguous_buffer;
                if (gpu_rgb_buffer)
                    cudaFree(gpu_rgb_buffer);
                if (gpu_batch_buffer)
                    cudaFree(gpu_batch_buffer);

                error = e.what();
                return false;
            }
        }
    };

    VideoFrameExtractor::VideoFrameExtractor() : impl_(new Impl()) {}
    VideoFrameExtractor::~VideoFrameExtractor() { delete impl_; }

    bool VideoFrameExtractor::extract(const Params& params, std::string& error) {
        return impl_->extract(params, error);
    }

} // namespace lfs::io
