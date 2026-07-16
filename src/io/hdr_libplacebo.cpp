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

        bool tonemap(const AVFrame* const frame, const AVStream* const stream,
                     const int output_width, const int output_height,
                     std::vector<unsigned char>& output_rgb, std::string& error,
                     HdrTonemapTiming* const timing) {
            std::lock_guard lock(mutex_);
            if (timing)
                *timing = {};

            const auto initialization_started = std::chrono::steady_clock::now();
            if (!initialize(error))
                return false;
            if (timing) {
                timing->initialization_seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - initialization_started).count();
            }
            if (!frame || output_width <= 0 || output_height <= 0) {
                error = "Invalid HDR frame or output dimensions";
                return false;
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

            // MOV/MP4 commonly carries mastering and Dolby Vision config at
            // stream level. FFmpeg does not copy it to every decoded frame.
            if (stream)
                pl_frame_copy_stream_props(&source, stream);

            // The dialog applies the stream display rotation after receiving
            // RGB data, for both SDR and HDR preview paths. libplacebo also
            // imports FFmpeg display-matrix rotation, which would otherwise
            // rotate here and make the dialog apply it a second time. Render
            // the coded pixel geometry and keep that single UI rotation.
            source.rotation = PL_ROTATION_0;

            const bool texture_ready = recreateOutput(output_width, output_height, error);
            if (!texture_ready) {
                pl_unmap_avframe(gpu_, &source);
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
            const bool rendered = pl_render_image(renderer_, &source, &target, &render_params);
            pl_unmap_avframe(gpu_, &source);
            if (!rendered) {
                error = "libplacebo failed to render the HDR frame";
                return false;
            }
            if (timing) {
                timing->render_seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - render_started).count();
            }

            rgba_buffer_.resize(static_cast<size_t>(output_width) * output_height * 4);
            pl_tex_transfer_params download_params{};
            download_params.tex = output_texture_;
            download_params.row_pitch = static_cast<size_t>(output_width) * 4;
            download_params.ptr = rgba_buffer_.data();
            const auto readback_started = std::chrono::steady_clock::now();
            if (!pl_tex_download(gpu_, &download_params)) {
                error = "libplacebo failed to read back the SDR frame";
                return false;
            }
            if (timing) {
                timing->readback_seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - readback_started).count();
            }

            output_rgb.resize(static_cast<size_t>(output_width) * output_height * 3);
            const auto rgb_conversion_started = std::chrono::steady_clock::now();
            for (size_t source_index = 0, target_index = 0; source_index < rgba_buffer_.size();
                 source_index += 4, target_index += 3) {
                output_rgb[target_index] = rgba_buffer_[source_index];
                output_rgb[target_index + 1] = rgba_buffer_[source_index + 1];
                output_rgb[target_index + 2] = rgba_buffer_[source_index + 2];
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

            pl_log_params log_params{};
            log_params.log_cb = libplaceboLogCallback;
            // Startup details are useful while debugging libplacebo itself but
            // overwhelm normal application logs without helping an end user.
            log_params.log_level = PL_LOG_WARN;
            log_ = pl_log_create(PL_API_VER, &log_params);
            vulkan_ = pl_vulkan_create(log_, nullptr);
            if (!vulkan_) {
                error = "libplacebo could not create its Vulkan renderer";
                return false;
            }
            gpu_ = vulkan_->gpu;
            renderer_ = pl_renderer_create(log_, gpu_);
            if (!renderer_) {
                error = "libplacebo could not create its video renderer";
                return false;
            }
            return true;
        }

        bool recreateOutput(const int width, const int height, std::string& error) {
            const pl_fmt format = pl_find_fmt(gpu_, PL_FMT_UNORM, 4, 8, 8,
                                              static_cast<pl_fmt_caps>(PL_FMT_CAP_RENDERABLE |
                                                                       PL_FMT_CAP_HOST_READABLE));
            if (!format) {
                error = "libplacebo could not find an RGBA8 render target";
                return false;
            }
            pl_tex_params params{};
            params.w = width;
            params.h = height;
            params.format = format;
            params.renderable = true;
            params.host_readable = true;
            // libplacebo clears the render target as part of rendering; this
            // operation is a blit and therefore requires the explicit cap.
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
        std::array<pl_tex, 4> source_textures_{};
        pl_tex output_texture_ = nullptr;
        std::vector<unsigned char> rgba_buffer_;
    };

    HdrLibplaceboRenderer::HdrLibplaceboRenderer() : impl_(std::make_unique<Impl>()) {}
    HdrLibplaceboRenderer::~HdrLibplaceboRenderer() = default;

    bool HdrLibplaceboRenderer::tonemapToSdr(const AVFrame* const frame, const AVStream* const stream,
                                              const int output_width, const int output_height,
                                              std::vector<unsigned char>& output_rgb,
                                              std::string& error, HdrTonemapTiming* const timing) {
        return impl_->tonemap(frame, stream, output_width, output_height, output_rgb, error, timing);
    }

    void HdrLibplaceboRenderer::reset() { impl_->reset(); }

} // namespace lfs::io
