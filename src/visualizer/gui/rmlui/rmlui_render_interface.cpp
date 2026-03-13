/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// clang-format off
#include <glad/glad.h>
// clang-format on

#include "gui/rmlui/rmlui_render_interface.hpp"

#include "core/camera.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "core/logger.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "io/nvcodec_image_loader.hpp"
#include "rendering/cuda_gl_interop.hpp"

#include <stb_image.h>

#include <algorithm>
#include <cassert>
#include <charconv>
#include <sstream>
#include <string_view>
#include <unordered_map>

namespace lfs::vis::gui {

    struct PreviewTextureCache {
        std::unique_ptr<lfs::io::NvCodecImageLoader> nvcodec;
        bool nvcodec_init_failed = false;
        std::unordered_map<Rml::TextureHandle, std::unique_ptr<lfs::rendering::CudaGLInteropTexture>> textures;
    };

    namespace {

        struct PreviewParams {
            int cam_uid = -1;
            int thumb = 0;
            int rf = 1;
            int mw = 0;
            bool ud = false;
            std::string path;
        };

        int parse_int(std::string_view sv, int fallback = 0) {
            int value = fallback;
            std::from_chars(sv.data(), sv.data() + sv.size(), value);
            return value;
        }

        PreviewParams parse_preview_url(const Rml::String& source) {
            PreviewParams p;
            constexpr std::string_view PREFIX = "preview://";
            assert(source.substr(0, PREFIX.size()) == PREFIX);

            const auto query = source.substr(PREFIX.size());

            const auto path_pos = query.find("path=");
            if (path_pos != std::string::npos)
                p.path = query.substr(path_pos + 5);

            const auto params_end = (path_pos != std::string::npos && path_pos > 0) ? path_pos - 1 : query.size();
            const auto params_str = query.substr(0, params_end);
            std::istringstream stream(std::string{params_str});
            std::string pair;
            while (std::getline(stream, pair, '&')) {
                const auto eq = pair.find('=');
                if (eq == std::string::npos)
                    continue;
                const auto key = std::string_view{pair}.substr(0, eq);
                const auto val = std::string_view{pair}.substr(eq + 1);
                if (key == "cam")
                    p.cam_uid = parse_int(val, -1);
                else if (key == "thumb")
                    p.thumb = parse_int(val);
                else if (key == "rf")
                    p.rf = parse_int(val, 1);
                else if (key == "mw")
                    p.mw = parse_int(val);
                else if (key == "ud")
                    p.ud = (val == "1");
            }
            return p;
        }

    } // namespace

    RmlRenderInterface::RmlRenderInterface()
        : preview_cache_(std::make_unique<PreviewTextureCache>()) {
        if (!*this)
            LOG_ERROR("RmlUI GL3 render interface failed to initialize");
    }

    RmlRenderInterface::~RmlRenderInterface() = default;

    void RmlRenderInterface::set_scene(lfs::core::Scene* scene) { scene_ = scene; }

    Rml::TextureHandle RmlRenderInterface::LoadTexture(Rml::Vector2i& dimensions,
                                                       const Rml::String& source) {
        constexpr std::string_view PREVIEW_PREFIX = "preview://";
        if (source.size() > PREVIEW_PREFIX.size() &&
            source.substr(0, PREVIEW_PREFIX.size()) == PREVIEW_PREFIX) {
            auto handle = load_preview_texture(dimensions, source);
            if (handle)
                return handle;
        }

        int w = 0, h = 0, channels = 0;
        unsigned char* data = stbi_load(source.c_str(), &w, &h, &channels, 4);
        if (!data) {
            LOG_WARN("RmlUI LoadTexture failed: {}", source);
            return 0;
        }

        dimensions.x = w;
        dimensions.y = h;

        const int pixel_count = w * h;
        for (int i = 0; i < pixel_count; ++i) {
            unsigned char* p = data + i * 4;
            const unsigned int a = p[3];
            p[0] = static_cast<unsigned char>((p[0] * a + 127) / 255);
            p[1] = static_cast<unsigned char>((p[1] * a + 127) / 255);
            p[2] = static_cast<unsigned char>((p[2] * a + 127) / 255);
        }

        GLuint tex = 0;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glBindTexture(GL_TEXTURE_2D, 0);

        stbi_image_free(data);
        return static_cast<Rml::TextureHandle>(tex);
    }

    void RmlRenderInterface::ReleaseTexture(Rml::TextureHandle texture_handle) {
        if (preview_cache_) {
            auto it = preview_cache_->textures.find(texture_handle);
            if (it != preview_cache_->textures.end()) {
                preview_cache_->textures.erase(it);
                return;
            }
        }
        GLuint tex = static_cast<GLuint>(texture_handle);
        glDeleteTextures(1, &tex);
    }

    Rml::TextureHandle RmlRenderInterface::load_preview_texture(Rml::Vector2i& dimensions,
                                                                const Rml::String& source) {
        assert(preview_cache_);
        const auto params = parse_preview_url(source);
        if (params.path.empty())
            return 0;

        if (!preview_cache_->nvcodec) {
            if (preview_cache_->nvcodec_init_failed)
                return 0;
            try {
                preview_cache_->nvcodec =
                    std::make_unique<lfs::io::NvCodecImageLoader>(lfs::io::NvCodecImageLoader::Options{});
            } catch (const std::exception& e) {
                LOG_WARN("NvCodec init failed for preview: {}", e.what());
                preview_cache_->nvcodec_init_failed = true;
                return 0;
            }
        }

        const int rf = params.thumb > 0 ? 1 : params.rf;
        const int mw = params.thumb > 0 ? params.thumb : params.mw;

        lfs::core::Tensor tensor;
        try {
            tensor = preview_cache_->nvcodec->load_image_gpu(params.path, rf, mw);
        } catch (const std::exception& e) {
            LOG_WARN("GPU decode failed for {}: {}", params.path, e.what());
            return 0;
        }

        if (tensor.is_empty())
            return 0;

        assert(tensor.ndim() == 3);

        if (params.ud && params.thumb == 0 && params.cam_uid >= 0 && scene_) {
            auto cameras = scene_->getAllCameras();
            auto it = std::find_if(cameras.begin(), cameras.end(),
                                   [&](const auto& c) { return c->uid() == params.cam_uid; });
            if (it != cameras.end() && (*it)->has_distortion()) {
                auto& camera = *it;
                if (!camera->is_undistort_precomputed())
                    camera->precompute_undistortion();

                const int tw = static_cast<int>(tensor.shape()[2]);
                const int th = static_cast<int>(tensor.shape()[1]);
                auto scaled =
                    lfs::core::scale_undistort_params(camera->undistort_params(), tw, th);
                tensor = lfs::core::undistort_image(tensor, scaled, nullptr);
            }
        }

        const auto hwc = tensor.permute({1, 2, 0}).contiguous();
        const int h = static_cast<int>(hwc.shape()[0]);
        const int w = static_cast<int>(hwc.shape()[1]);
        assert(h > 0 && w > 0);

        auto interop = std::make_unique<lfs::rendering::CudaGLInteropTexture>();
        if (auto result = interop->init(w, h); !result) {
            LOG_WARN("CudaGLInterop init failed: {}", result.error());
            return 0;
        }

        if (auto result = interop->updateFromTensor(hwc); !result) {
            LOG_WARN("CudaGLInterop upload failed: {}", result.error());
            return 0;
        }

        const auto tex_id = interop->getTextureID();
        const auto handle = static_cast<Rml::TextureHandle>(tex_id);
        dimensions.x = w;
        dimensions.y = h;

        preview_cache_->textures[handle] = std::move(interop);
        return handle;
    }

} // namespace lfs::vis::gui
