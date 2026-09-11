/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "hdr_libplacebo.hpp"
#include "core/include/core/logger.hpp"

extern "C" {
#include <libavutil/frame.h>
}

#define PL_LIBAV_IMPLEMENTATION 0
#include <libplacebo/renderer.h>
#include <libplacebo/shaders/dithering.h>
#include <libplacebo/utils/libav.h>
#include <libplacebo/vulkan.h>

#include <array>
#include <chrono>
#include <limits>
#include <mutex>
#include <utility>

namespace lfs::io {

    namespace {
        void libplaceboLogCallback(void*, const enum pl_log_level level, const char* const message) {
            const char* const text = message ? message : "(no message)";
            switch (level) {
            case PL_LOG_FATAL:
            case PL_LOG_ERR:
                LOG_ERROR("libplacebo: {}", text);
                break;
            case PL_LOG_WARN:
                LOG_WARN("libplacebo: {}", text);
                break;
            case PL_LOG_INFO:
                LOG_INFO("libplacebo: {}", text);
                break;
            default:
                break;
            }
        }
    } // namespace

    class HdrLibplaceboRenderer::Impl {
    public:
        ~Impl() {
            if (gpu_) {
                pl_tex_destroy(gpu_, &output_texture_);
                for (auto& texture : source_textures_)
                    pl_tex_destroy(gpu_, &texture);
            }
            pl_renderer_destroy(&renderer_);
            pl_vulkan_destroy(&vulkan_);
            pl_log_destroy(&log_);
        }

        bool isAvailable(std::string& error) {
            std::lock_guard lock(mutex_);
            return initialize(error);
        }

        bool tonemap(const AVFrame* const frame, const AVStream* const stream,
                     const HdrFormat source_format,
                     const int output_width, const int output_height,
                     const int rotation_degrees,
                     std::vector<unsigned char>& output, std::string& error,
                     HdrTonemapTiming* const timing, const bool keep_rgba,
                     const bool temporal_peak_detection) {
            std::lock_guard lock(mutex_);
            if (timing)
                *timing = {};
            if (!frame || output_width <= 0 || output_height <= 0) {
                error = "Invalid HDR frame or output dimensions";
                return false;
            }

            const auto initialization_started = std::chrono::steady_clock::now();
            if (!initialize(error))
                return false;
            if (timing) {
                timing->initialization_seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - initialization_started).count();
            }

            const auto render_started = std::chrono::steady_clock::now();
            pl_frame source{};
            pl_avframe_params map_params{};
            map_params.frame = frame;
            map_params.tex = source_textures_.data();
            map_params.map_dovi = true;
            if (!pl_map_avframe_ex(gpu_, &source, &map_params)) {
                error = "libplacebo could not map the decoded video frame";
                return false;
            }

            const auto unmap_source = [this, &source]() {
                pl_unmap_avframe(gpu_, &source);
            };
            if (source_format == HdrFormat::DOLBY_VISION_NATIVE &&
                source.repr.sys != PL_COLOR_SYSTEM_DOLBYVISION) {
                unmap_source();
                error = "Dolby Vision Profile 5 metadata was not mapped by libplacebo";
                return false;
            }

            if (stream)
                pl_frame_copy_stream_props(&source, stream);

            source.rotation = pl_rotation_normalize(rotation_degrees / 90);

            const bool texture_ready = recreateOutput(output_width, output_height, error);
            if (!texture_ready) {
                unmap_source();
                return false;
            }

            pl_frame target{};
            target.num_planes = 1;
            target.planes[0].texture = output_texture_;
            target.planes[0].components = 4;
            target.planes[0].component_mapping[0] = 0;
            target.planes[0].component_mapping[1] = 1;
            target.planes[0].component_mapping[2] = 2;
            target.planes[0].component_mapping[3] = 3;
            target.repr = pl_color_repr_rgb;
            target.color = pl_color_space_srgb;
            target.crop = {0.0f, 0.0f, static_cast<float>(output_width),
                           static_cast<float>(output_height)};

            pl_render_params render_params = pl_render_default_params;
            render_params.color_map_params = &pl_color_map_default_params;
            render_params.dither_params = &pl_dither_default_params;
            if (!temporal_peak_detection)
                render_params.peak_detect_params = nullptr;
            const bool rendered = pl_render_image(renderer_, &source, &target, &render_params);
            unmap_source();
            if (!rendered) {
                error = "libplacebo failed to render the HDR frame";
                return false;
            }
            if (timing) {
                timing->render_seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - render_started).count();
            }

            std::vector<unsigned char>& readback_buffer = keep_rgba ? output : rgba_buffer_;
            readback_buffer.resize(static_cast<size_t>(output_width) * output_height * 4);
            pl_tex_transfer_params download_params{};
            download_params.tex = output_texture_;
            download_params.row_pitch = static_cast<size_t>(output_width) * 4;
            download_params.ptr = readback_buffer.data();
            const auto readback_started = std::chrono::steady_clock::now();
            if (!pl_tex_download(gpu_, &download_params)) {
                error = "libplacebo failed to read back the SDR frame";
                return false;
            }
            if (timing) {
                timing->readback_seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - readback_started).count();
            }

            if (keep_rgba)
                return true;

            output.resize(static_cast<size_t>(output_width) * output_height * 3);
            const auto rgb_conversion_started = std::chrono::steady_clock::now();
            for (size_t source_index = 0, target_index = 0; source_index < rgba_buffer_.size();
                 source_index += 4, target_index += 3) {
                output[target_index] = rgba_buffer_[source_index];
                output[target_index + 1] = rgba_buffer_[source_index + 1];
                output[target_index + 2] = rgba_buffer_[source_index + 2];
            }
            if (timing) {
                timing->rgba_to_rgb_seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - rgb_conversion_started).count();
            }
            return true;
        }

        void reset() {
            std::lock_guard lock(mutex_);
            if (renderer_)
                pl_renderer_flush_cache(renderer_);
        }

    private:
        bool initialize(std::string& error) {
            if (renderer_)
                return true;
            if (initialization_attempted_) {
                error = initialization_error_;
                return false;
            }
            initialization_attempted_ = true;

            pl_log_params log_params{};
            log_params.log_cb = libplaceboLogCallback;
            log_params.log_level = PL_LOG_WARN;
            pl_log new_log = pl_log_create(PL_API_VER, &log_params);
            if (!new_log)
                return failInitialization("libplacebo could not create its logger", error,
                                          new_log, nullptr, nullptr);

            pl_vulkan new_vulkan = pl_vulkan_create(new_log, nullptr);
            if (!new_vulkan)
                return failInitialization("libplacebo could not create its Vulkan renderer",
                                          error, new_log, new_vulkan, nullptr);

            pl_renderer new_renderer = pl_renderer_create(new_log, new_vulkan->gpu);
            if (!new_renderer)
                return failInitialization("libplacebo could not create its video renderer",
                                          error, new_log, new_vulkan, new_renderer);

            const pl_fmt new_output_format = pl_find_fmt(
                new_vulkan->gpu, PL_FMT_UNORM, 4, 8, 8,
                static_cast<pl_fmt_caps>(PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE));
            if (!new_output_format) {
                return failInitialization("libplacebo could not find an RGBA8 render target",
                                          error, new_log, new_vulkan, new_renderer);
            }

            log_ = new_log;
            vulkan_ = new_vulkan;
            gpu_ = new_vulkan->gpu;
            renderer_ = new_renderer;
            output_format_ = new_output_format;
            return true;
        }

        bool failInitialization(const std::string& message, std::string& error, pl_log log,
                                pl_vulkan vulkan, pl_renderer renderer) {
            pl_renderer_destroy(&renderer);
            pl_vulkan_destroy(&vulkan);
            pl_log_destroy(&log);
            initialization_error_ = message;
            error = message;
            return false;
        }

        bool recreateOutput(const int width, const int height, std::string& error) {
            if (width <= 0 || height <= 0) {
                error = "Invalid libplacebo render target dimensions";
                return false;
            }
            const size_t output_width = static_cast<size_t>(width);
            const size_t output_height = static_cast<size_t>(height);
            if (output_width > std::numeric_limits<size_t>::max() / output_height ||
                output_width * output_height >
                    std::numeric_limits<size_t>::max() / 4) {
                error = "Invalid libplacebo render target dimensions";
                return false;
            }
            pl_tex_params params{};
            params.w = width;
            params.h = height;
            params.format = output_format_;
            params.renderable = true;
            params.host_readable = true;
            params.blit_dst = true;
            if (!pl_tex_recreate(gpu_, &output_texture_, &params)) {
                error = "libplacebo could not allocate the SDR render target";
                return false;
            }
            return true;
        }

        std::mutex mutex_;
        pl_log log_ = nullptr;
        pl_vulkan vulkan_ = nullptr;
        pl_gpu gpu_ = nullptr;
        pl_renderer renderer_ = nullptr;
        pl_fmt output_format_ = nullptr;
        std::array<pl_tex, 4> source_textures_{};
        pl_tex output_texture_ = nullptr;
        std::vector<unsigned char> rgba_buffer_;
        bool initialization_attempted_ = false;
        std::string initialization_error_;
    };

    HdrLibplaceboRenderer::HdrLibplaceboRenderer() : impl_(std::make_unique<Impl>()) {}
    HdrLibplaceboRenderer::~HdrLibplaceboRenderer() = default;

    bool HdrLibplaceboRenderer::isAvailable(std::string& error) {
        return impl_->isAvailable(error);
    }

    bool HdrLibplaceboRenderer::tonemapToSdr(const AVFrame* const frame, const AVStream* const stream,
                                             const HdrFormat source_format,
                                             const int output_width, const int output_height,
                                             std::vector<unsigned char>& output_rgb,
                                             std::string& error, HdrTonemapTiming* const timing) {
        return impl_->tonemap(frame, stream, source_format, output_width, output_height, 0,
                              output_rgb, error, timing, false, false);
    }

    bool HdrLibplaceboRenderer::tonemapToSdrRgba(const AVFrame* const frame, const AVStream* const stream,
                                                 const HdrFormat source_format,
                                                 const int output_width, const int output_height,
                                                 const int rotation_degrees,
                                                 std::vector<unsigned char>& output_rgba,
                                                 std::string& error) {
        return impl_->tonemap(frame, stream, source_format, output_width, output_height,
                              rotation_degrees, output_rgba, error, nullptr, true, true);
    }

    void HdrLibplaceboRenderer::reset() { impl_->reset(); }

} // namespace lfs::io
