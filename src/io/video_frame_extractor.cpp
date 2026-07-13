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
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

#include <cuda_runtime.h>
#include <stb_image_write.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>

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
            uint8_t* gpu_rotated_buffer = nullptr;
            uint8_t* cpu_contiguous_buffer = nullptr;
            std::vector<uint8_t> rot_buf;
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
                // Estimated frames per sliding window (for candidate sampling)
                int window_est_frames = frame_step;
                if (params.mode == ExtractionMode::FPS && video_fps > 0 && target_fps > 0)
                    window_est_frames = static_cast<int>(std::round(video_fps / target_fps));
                window_est_frames = std::max(1, window_est_frames);

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
                const bool full_gpu_pipeline_available =
                    using_hw_decode && gpu_encoding_enabled && gpu_rgb_buffer && !needs_scale;
                const auto throw_if_cancelled = [&]() {
                    if (params.cancel_requested && params.cancel_requested())
                        throw ExtractionCancelled{};
                };

                if (full_gpu_pipeline_available) {
                    LOG_INFO("Full GPU pipeline available for NV12 frames: NVDEC decode → GPU color convert → GPU JPEG encode");
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
                int skipped_count = 0;
                int written_count = 0;
                double current_frame_time = 0.0;
                int current_src_frame = 0;
                std::vector<void*> batch_gpu_ptrs;
                std::vector<std::filesystem::path> batch_filenames;
                struct BatchFrameMeta {
                    double timestamp;
                    int source_frame;
                    double sharpness_score;
                };
                std::vector<BatchFrameMeta> batch_meta;
                int batch_idx = 0;
                int batch_encode_w = 0, batch_encode_h = 0;
                bool logged_hw_format_fallback = false;
                bool used_full_gpu_pipeline = false;
                struct CandidateFrame {
                    std::vector<uint8_t> rgb;
                    std::filesystem::path filename;
                    double score = 0.0;
                    double timestamp = 0.0;
                    int source_frame = 0;
                };
                std::vector<CandidateFrame> window_candidates;
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

                auto flush_jpeg_batch = [&]() {
                    if (batch_gpu_ptrs.empty())
                        return;
                    if (batch_encode_w <= 0 || batch_encode_h <= 0) {
                        LOG_ERROR("JPEG batch dimensions not set ({}x{}), skipping {} queued frames",
                                  batch_encode_w, batch_encode_h, batch_gpu_ptrs.size());
                        batch_gpu_ptrs.clear();
                        batch_filenames.clear();
                        batch_meta.clear();
                        batch_idx = 0;
                        return;
                    }
                    throw_if_cancelled();

                    auto encoded = nvcodec->encode_batch_rgb_to_jpeg(batch_gpu_ptrs, batch_encode_w, batch_encode_h,
                                                                     params.jpg_quality);

                    for (size_t i = 0; i < encoded.size(); i++) {
                        if (!encoded[i].empty()) {
                            write_jpeg_to_file(batch_filenames[i], encoded[i]);
                            ++written_count;
                            if (params.generate_metadata && i < batch_meta.size()) {
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

                auto flush_window = [&]() {
                    if (window_candidates.empty())
                        return;
                    const auto best = std::max_element(
                        window_candidates.begin(), window_candidates.end(),
                        [](const CandidateFrame& a, const CandidateFrame& b) {
                            return a.score < b.score;
                        });
                    std::filesystem::path fname = generate_filename(
                        written_count + 1);
                    // Apply rotation to the best window frame before writing
                    int write_w = out_width;
                    int write_h = out_height;
                    const uint8_t* write_data = best->rgb.data();
                    if (params.rotation != 0) {
                        rot_buf.resize(static_cast<size_t>(out_width) * out_height * 3);
                        if (params.rotation == 180) {
                            for (int y = 0; y < out_height; ++y)
                                for (int x = 0; x < out_width; ++x) {
                                    const int si = (y * out_width + x) * 3;
                                    const int di = ((out_height - 1 - y) * out_width + (out_width - 1 - x)) * 3;
                                    rot_buf[di + 0] = best->rgb[si + 0];
                                    rot_buf[di + 1] = best->rgb[si + 1];
                                    rot_buf[di + 2] = best->rgb[si + 2];
                                }
                        } else {
                            const int dst_w = out_height;
                            const int dst_h = out_width;
                            for (int y = 0; y < out_height; ++y)
                                for (int x = 0; x < out_width; ++x) {
                                    const int si = (y * out_width + x) * 3;
                                    const int di = (params.rotation == 90)
                                                       ? (x * out_height + (out_height - 1 - y)) * 3
                                                       : ((out_width - 1 - x) * out_height + y) * 3;
                                    rot_buf[di + 0] = best->rgb[si + 0];
                                    rot_buf[di + 1] = best->rgb[si + 1];
                                    rot_buf[di + 2] = best->rgb[si + 2];
                                }
                            write_w = dst_w;
                            write_h = dst_h;
                        }
                        write_data = rot_buf.data();
                    }
                    if (!write_image_file(fname, write_w, write_h,
                                          write_data, params.format,
                                          params.jpg_quality)) {
                        LOG_WARN("Failed to write sharpest window frame: {}",
                                 lfs::core::path_to_utf8(fname));
                    } else {
                        ++written_count;
                        if (params.generate_metadata) {
                            saved_frames.push_back({lfs::core::path_to_utf8(fname.filename()),
                                                    best->timestamp,
                                                    best->source_frame,
                                                    best->score});
                        }
                    }
                    ++saved_count;
                    if (params.progress_callback)
                        params.progress_callback(saved_count, estimated_total, skipped_count);
                    window_candidates.clear();
                    window_skip_counter = 0;
                    throw_if_cancelled();
                };

                auto process_frame_hw = [&](AVFrame* hw_frame) {
                    throw_if_cancelled();
                    std::filesystem::path filename = generate_filename(saved_count + 1);

                    const AVPixelFormat hw_sw_format = hardwareFrameSoftwareFormat(hw_frame);
                    const bool use_full_gpu_pipeline =
                        full_gpu_pipeline_available && hw_sw_format == AV_PIX_FMT_NV12;

                    if (full_gpu_pipeline_available && !use_full_gpu_pipeline &&
                        !logged_hw_format_fallback) {
                        LOG_INFO("Hardware frame format is {}; using CPU transfer/color conversion",
                                 pixelFormatName(hw_sw_format));
                        logged_hw_format_fallback = true;
                    }

                    if (use_full_gpu_pipeline) {
                        // Full GPU path: NV12 on GPU → RGB on GPU → encode on GPU
                        const uint8_t* y_plane = hw_frame->data[0];
                        const uint8_t* uv_plane = hw_frame->data[1];
                        const int y_pitch = hw_frame->linesize[0];
                        const int uv_pitch = hw_frame->linesize[1];

                        video::nv12ToRgbCuda(y_plane, uv_plane, gpu_rgb_buffer,
                                             src_width, src_height, y_pitch, uv_pitch, nullptr);

                        // --- Sharpness evaluation (full GPU path) ---
                        double frame_score = 0.0;
                        if (params.sharpness.enabled) {
                            // Sharpness computed on CPU after GPU→CPU transfer.
                            // A future GPU-side sharpness kernel could skip this copy,
                            // but for now the hybrid approach keeps the implementation
                            // simple and shared across all paths.
                            cudaMemcpy(cpu_contiguous_buffer, gpu_rgb_buffer, frame_size,
                                       cudaMemcpyDeviceToHost);
                            frame_score = computeSharpnessScore(
                                cpu_contiguous_buffer, out_width, out_height, params.sharpness.algorithm);
                            if (params.sharpness.window_mode) {
                                CandidateFrame cf;
                                cf.rgb.assign(cpu_contiguous_buffer,
                                              cpu_contiguous_buffer + frame_size);
                                cf.score = frame_score;
                                cf.timestamp = current_frame_time;
                                cf.source_frame = current_src_frame;
                                window_candidates.push_back(std::move(cf));
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

                        // --- Rotation (full GPU path) ---
                        const int rot = params.rotation;
                        int batch_w = out_width;
                        int batch_h = out_height;
                        const uint8_t* batch_src = gpu_rgb_buffer;
                        if (rot != 0) {
                            const bool swap = (rot == 90 || rot == 270);
                            const int rw = swap ? out_height : out_width;
                            const int rh = swap ? out_width : out_height;
                            const size_t rot_size = static_cast<size_t>(rw) * rh * 3;
                            if (!gpu_rotated_buffer) {
                                if (cudaMalloc(&gpu_rotated_buffer, rot_size) != cudaSuccess) {
                                    LOG_WARN("Failed to allocate GPU rotation buffer, skipping rotation");
                                    gpu_rotated_buffer = nullptr;
                                }
                            }
                            if (gpu_rotated_buffer) {
                                batch_w = rw;
                                batch_h = rh;
                                batch_src = gpu_rotated_buffer;
                                video::rotateRgbCuda(gpu_rgb_buffer, gpu_rotated_buffer,
                                                     out_width, out_height, rot, nullptr);
                            }
                        }
                        const int batch_frame_size = batch_w * batch_h * 3;
                        // --- End rotation ---

                        if (batch_encode_w == 0) {
                            batch_encode_w = batch_w;
                            batch_encode_h = batch_h;
                        }

                        void* dst_ptr = gpu_batch_buffer + batch_idx * batch_frame_size;
                        cudaError_t cuda_err = cudaMemcpyAsync(dst_ptr, batch_src, batch_frame_size,
                                                               cudaMemcpyDeviceToDevice, nullptr);
                        if (cuda_err != cudaSuccess) {
                            LOG_WARN("Failed to copy GPU RGB frame into JPEG batch buffer: {}",
                                     cudaGetErrorString(cuda_err));
                            return;
                        }

                        cuda_err = cudaStreamSynchronize(nullptr);
                        if (cuda_err != cudaSuccess) {
                            LOG_WARN("Failed to synchronize GPU RGB batch copy: {}",
                                     cudaGetErrorString(cuda_err));
                            return;
                        }

                        batch_gpu_ptrs.push_back(dst_ptr);
                        batch_filenames.push_back(filename);
                        batch_meta.push_back({current_frame_time, current_src_frame, frame_score});
                        batch_idx++;
                        used_full_gpu_pipeline = true;

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

                        // Hardware frame transfer output -> RGB with optional scaling.
                        SwsContext* hw_sws = sws_getContext(
                            src_width, src_height, static_cast<AVPixelFormat>(sw_frame->format),
                            out_width, out_height, AV_PIX_FMT_RGB24, SWS_BILINEAR,
                            nullptr, nullptr, nullptr);

                        if (!hw_sws) {
                            LOG_WARN("Failed to create hardware frame scaling context");
                            return;
                        }

                        uint8_t* dst_data[1] = {cpu_contiguous_buffer};
                        int dst_linesize[1] = {out_width * 3};
                        sws_scale(hw_sws, sw_frame->data, sw_frame->linesize, 0, src_height,
                                  dst_data, dst_linesize);
                        sws_freeContext(hw_sws);

                        // --- Sharpness evaluation (hybrid path) ---
                        double frame_score = 0.0;
                        if (params.sharpness.enabled) {
                            frame_score = computeSharpnessScore(
                                cpu_contiguous_buffer, out_width, out_height, params.sharpness.algorithm);
                            if (params.sharpness.window_mode) {
                                CandidateFrame cf;
                                cf.rgb.assign(cpu_contiguous_buffer,
                                              cpu_contiguous_buffer + frame_size);
                                cf.score = frame_score;
                                cf.timestamp = current_frame_time;
                                cf.source_frame = current_src_frame;
                                window_candidates.push_back(std::move(cf));
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
                            void* dst_ptr = gpu_batch_buffer + batch_idx * frame_size;
                            cudaMemcpy(dst_ptr, cpu_contiguous_buffer, frame_size,
                                       cudaMemcpyHostToDevice);

                            batch_gpu_ptrs.push_back(dst_ptr);
                            batch_filenames.push_back(filename);
                            batch_meta.push_back({current_frame_time, current_src_frame, frame_score});
                            batch_idx++;

                            if (batch_idx >= JPEG_BATCH_SIZE) {
                                flush_jpeg_batch();
                            }
                        } else if (write_image_file(filename, hw_rot_w, hw_rot_h,
                                                    cpu_contiguous_buffer, params.format,
                                                    params.jpg_quality)) {
                            ++written_count;
                            if (params.generate_metadata) {
                                saved_frames.push_back({lfs::core::path_to_utf8(filename.filename()),
                                                        current_frame_time,
                                                        current_src_frame,
                                                        frame_score});
                            }
                        } else {
                            LOG_WARN("Failed to write extracted frame: {}", lfs::core::path_to_utf8(filename));
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
                    uint8_t* dst_data[1] = {cpu_contiguous_buffer};
                    int dst_linesize[1] = {out_width * 3};
                    sws_scale(sws_ctx, decoded_frame->data, decoded_frame->linesize, 0, src_height,
                              dst_data, dst_linesize);

                    // --- Sharpness evaluation (SW path) ---
                    double frame_score = 0.0;
                    if (params.sharpness.enabled) {
                        frame_score = computeSharpnessScore(
                            cpu_contiguous_buffer, out_width, out_height, params.sharpness.algorithm);

                        if (params.sharpness.window_mode) {
                            CandidateFrame cf;
                            cf.rgb.assign(cpu_contiguous_buffer,
                                          cpu_contiguous_buffer + frame_size);
                            cf.score = frame_score;
                            cf.timestamp = current_frame_time;
                            cf.source_frame = current_src_frame;
                            window_candidates.push_back(std::move(cf));
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

                    // --- Rotation (SW path) ---
                    int sw_rot_w = out_width;
                    int sw_rot_h = out_height;
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
                            sw_rot_w = dst_w;
                            sw_rot_h = dst_h;
                        }
                        std::memcpy(cpu_contiguous_buffer, rot_buf.data(),
                                    static_cast<size_t>(out_width) * out_height * 3);
                    }
                    // --- End rotation ---

                    std::filesystem::path filename = generate_filename(saved_count + 1);

                    if (gpu_encoding_enabled) {
                        if (batch_encode_w == 0) {
                            batch_encode_w = (sw_rot_w > 0) ? sw_rot_w : out_width;
                            batch_encode_h = (sw_rot_h > 0) ? sw_rot_h : out_height;
                        }
                        void* dst_ptr = gpu_batch_buffer + batch_idx * frame_size;
                        cudaMemcpy(dst_ptr, cpu_contiguous_buffer, frame_size,
                                   cudaMemcpyHostToDevice);

                        batch_gpu_ptrs.push_back(dst_ptr);
                        batch_filenames.push_back(filename);
                        batch_meta.push_back({current_frame_time, current_src_frame, frame_score});
                        batch_idx++;

                        if (batch_idx >= JPEG_BATCH_SIZE) {
                            flush_jpeg_batch();
                        }
                    } else if (write_image_file(filename, sw_rot_w, sw_rot_h,
                                                cpu_contiguous_buffer, params.format,
                                                params.jpg_quality)) {
                        ++written_count;
                        if (params.generate_metadata) {
                            saved_frames.push_back({lfs::core::path_to_utf8(filename.filename()),
                                                    current_frame_time,
                                                    current_src_frame,
                                                    frame_score});
                        }
                    } else {
                        LOG_WARN("Failed to write extracted frame: {}", lfs::core::path_to_utf8(filename));
                    }

                    saved_count++;

                    if (params.progress_callback) {
                        params.progress_callback(saved_count + skipped_count, estimated_total, skipped_count);
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
                                current_frame_time = frame_time;
                                current_src_frame = decoded_frame_count;

                                if (params.sharpness.enabled && params.sharpness.window_mode) {
                                    int w_idx;
                                    if (params.mode == ExtractionMode::FPS) {
                                        w_idx = static_cast<int>(
                                            std::floor((frame_time - start_time) / target_interval));
                                    } else {
                                        w_idx = in_window_frame_count / frame_step;
                                    }
                                    if (w_idx != current_window_idx) {
                                        flush_window();
                                        current_window_idx = w_idx;
                                    }
                                    if (params.sharpness.window_mode) {
                                        ++window_skip_counter;
                                        ++in_window_frame_count;
                                        // Bucket sampling (zero-based): only process if this frame is a candidate
                                        int effective = window_est_frames;
                                        if (params.sharpness.window_candidates_target < 0) {
                                            const int auto_target = std::clamp(static_cast<int>(std::round(std::sqrt(static_cast<double>(window_est_frames))) * 2), 5, 20);
                                            effective = std::min(auto_target, window_est_frames);
                                        } else if (params.sharpness.window_candidates_target > 0) {
                                            effective = std::min(params.sharpness.window_candidates_target, window_est_frames);
                                        }
                                        if (effective < window_est_frames) {
                                            const int i = window_skip_counter - 1;
                                            const int bucket = i * effective / window_est_frames;
                                            const int prev_bucket = (i > 0) ? ((i - 1) * effective / window_est_frames) : -1;
                                            if (bucket == prev_bucket)
                                                continue;
                                        }
                                    }

                                    if (using_hw_decode)
                                        process_frame_hw(frame);
                                    else
                                        process_frame_sw(frame);
                                } else if (should_extract_frame(frame_time)) {
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
                        current_frame_time = frame_time;
                        current_src_frame = decoded_frame_count;

                        if (params.sharpness.enabled && params.sharpness.window_mode) {
                            int w_idx;
                            if (params.mode == ExtractionMode::FPS) {
                                w_idx = static_cast<int>(
                                    std::floor((frame_time - start_time) / target_interval));
                            } else {
                                w_idx = in_window_frame_count / frame_step;
                            }
                            if (w_idx != current_window_idx) {
                                flush_window();
                                current_window_idx = w_idx;
                            }
                            if (params.sharpness.window_mode) {
                                ++window_skip_counter;
                                ++in_window_frame_count;
                                // Bucket sampling (zero-based): only process if this frame is a candidate
                                int effective = window_est_frames;
                                if (params.sharpness.window_candidates_target < 0) {
                                    const int auto_target = std::clamp(static_cast<int>(std::round(std::sqrt(static_cast<double>(window_est_frames))) * 2), 5, 20);
                                    effective = std::min(auto_target, window_est_frames);
                                } else if (params.sharpness.window_candidates_target > 0) {
                                    effective = std::min(params.sharpness.window_candidates_target, window_est_frames);
                                }
                                if (effective < window_est_frames) {
                                    const int i = window_skip_counter - 1;
                                    const int bucket = i * effective / window_est_frames;
                                    const int prev_bucket = (i > 0) ? ((i - 1) * effective / window_est_frames) : -1;
                                    if (bucket == prev_bucket)
                                        continue;
                                }
                            }

                            if (using_hw_decode)
                                process_frame_hw(frame);
                            else
                                process_frame_sw(frame);
                        } else if (should_extract_frame(frame_time)) {
                            if (using_hw_decode) {
                                process_frame_hw(frame);
                            } else {
                                process_frame_sw(frame);
                            }
                        }
                    }
                }

                if (gpu_encoding_enabled) {
                    if (used_full_gpu_pipeline) {
                        cudaStreamSynchronize(nullptr);
                    }
                    flush_jpeg_batch();
                }

                // Flush remaining window candidates at end of video
                flush_window();

                if (params.generate_metadata && !saved_frames.empty()) {
                    try {
                        nlohmann::json root;
                        root["source_file"] = lfs::core::path_to_utf8(params.video_path);
                        root["source_fps"] = video_fps;
                        root["trimmed_source_frames"] = total_frames;
                        root["source_duration"] = video_duration;
                        root["source_size"] = {src_width, src_height};
                        root["rotation"] = params.rotation;
                        root["output_size"] = (params.rotation == 90 || params.rotation == 270)
                                                  ? nlohmann::json{out_height, out_width}
                                                  : nlohmann::json{out_width, out_height};
                        root["output_format"] = params.format == ImageFormat::PNG ? "png" : "jpg";
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
                        std::ofstream meta_file(meta_path);
                        if (meta_file) {
                            meta_file << root.dump(2);
                        }
                    } catch (const std::exception& e) {
                        LOG_WARN("Failed to write extraction metadata: {}", e.what());
                    }
                }

                if (skipped_count > 0) {
                    LOG_INFO("Extracted {} frames from video ({} discarded for low sharpness)",
                             written_count, skipped_count);
                } else {
                    LOG_INFO("Extracted {} frames from video", written_count);
                }

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
                if (gpu_rotated_buffer)
                    cudaFree(gpu_rotated_buffer);

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
                if (gpu_rotated_buffer)
                    cudaFree(gpu_rotated_buffer);

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
