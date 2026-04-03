/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gt_texture_cache.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/tensor.hpp"
#include "io/nvcodec_image_loader.hpp"
#include "io/pipelined_image_loader.hpp"
#include <algorithm>
#include <cstring>
#include <glad/glad.h>
#include <optional>
#include <string_view>
#include <vector>

namespace lfs::vis {

    namespace {
        constexpr auto kTensorTextureOrigin = lfs::rendering::TextureOrigin::TopLeft;

        [[nodiscard]] std::string makePipelineLoadSignature(
            const std::filesystem::path& image_path,
            const lfs::io::LoadParams& load_params) {
            auto signature = lfs::core::path_to_utf8(image_path) +
                             ":rf" + std::to_string(std::max(1, load_params.resize_factor)) +
                             "_mw" + std::to_string(load_params.max_width);
            if (load_params.undistort) {
                signature += "_ud";
            }
            return signature;
        }

        [[nodiscard]] std::string makeFallbackLoadSignature(
            const std::filesystem::path& image_path,
            const std::string_view loader_kind) {
            return lfs::core::path_to_utf8(image_path) + ":" + std::string(loader_kind);
        }

        [[nodiscard]] lfs::io::LoadParams normalizeGTLoadParams(const lfs::io::LoadParams& load_params) {
            auto effective_params = load_params;
            effective_params.resize_factor = std::max(1, effective_params.resize_factor);
            if (effective_params.max_width <= 0 || effective_params.max_width > GTTextureCache::MAX_TEXTURE_DIM) {
                effective_params.max_width = GTTextureCache::MAX_TEXTURE_DIM;
            }
            return effective_params;
        }

    } // namespace

    GTTextureCache::GTTextureCache() = default;

    GTTextureCache::~GTTextureCache() {
        clear();
    }

    void GTTextureCache::clear() {
        for (const auto& [id, entry] : texture_cache_) {
            if (!entry.interop_texture && entry.texture_id > 0) {
                glDeleteTextures(1, &entry.texture_id);
            }
        }
        texture_cache_.clear();
    }

    GTTextureCache::TextureInfo GTTextureCache::getGTTexture(
        const int cam_id,
        const std::filesystem::path& image_path,
        lfs::io::PipelinedImageLoader* const pipeline_loader,
        const lfs::io::LoadParams* const load_params) {
        const std::optional<lfs::io::LoadParams> effective_load_params =
            (pipeline_loader && load_params) ? std::optional<lfs::io::LoadParams>(normalizeGTLoadParams(*load_params))
                                             : std::nullopt;
        const std::string requested_signature =
            effective_load_params
                ? makePipelineLoadSignature(image_path, *effective_load_params)
                : makeFallbackLoadSignature(image_path, "fallback");

        if (const auto it = texture_cache_.find(cam_id);
            it != texture_cache_.end() && it->second.load_signature == requested_signature) {
            it->second.last_access = std::chrono::steady_clock::now();
            const auto& entry = it->second;
            const unsigned int tex_id = entry.interop_texture ? entry.interop_texture->getTextureID() : entry.texture_id;
            const glm::vec2 tex_scale = entry.interop_texture
                                            ? glm::vec2(entry.interop_texture->getTexcoordScaleX(),
                                                        entry.interop_texture->getTexcoordScaleY())
                                            : glm::vec2(1.0f);
            return {tex_id, entry.width, entry.height, entry.origin, tex_scale};
        }

        if (const auto it = texture_cache_.find(cam_id); it != texture_cache_.end()) {
            if (!it->second.interop_texture && it->second.texture_id > 0) {
                glDeleteTextures(1, &it->second.texture_id);
            }
            texture_cache_.erase(it);
        }

        if (texture_cache_.size() >= MAX_CACHE_SIZE) {
            evictOldest();
        }

        const auto ext = image_path.extension().string();
        const bool is_jpeg = (ext == ".jpg" || ext == ".jpeg" || ext == ".JPG" || ext == ".JPEG");

        CacheEntry entry;
        entry.last_access = std::chrono::steady_clock::now();

        if (!nvcodec_loader_ && is_jpeg) {
            try {
                constexpr lfs::io::NvCodecImageLoader::Options OPTS{.device_id = 0, .decoder_pool_size = 2};
                nvcodec_loader_ = std::make_unique<lfs::io::NvCodecImageLoader>(OPTS);
            } catch (...) {
                nvcodec_loader_ = nullptr;
            }
        }

        TextureInfo info{};
        if (pipeline_loader && effective_load_params) {
            info = loadTextureFromLoader(*pipeline_loader, image_path, *effective_load_params, entry);
        }
        if (nvcodec_loader_ && is_jpeg && info.texture_id == 0) {
            info = loadTextureGPU(image_path, entry);
        }

        if (info.texture_id == 0) {
            info = loadTexture(image_path);
            if (info.texture_id != 0) {
                entry.texture_id = info.texture_id;
                entry.width = info.width;
                entry.height = info.height;
                entry.origin = lfs::rendering::TextureOrigin::BottomLeft;
            }
        }

        if (info.texture_id == 0) {
            return {};
        }

        entry.load_signature = requested_signature;
        texture_cache_[cam_id] = std::move(entry);
        return info;
    }

    void GTTextureCache::evictOldest() {
        if (texture_cache_.empty()) {
            return;
        }

        const auto oldest = std::min_element(texture_cache_.begin(), texture_cache_.end(),
                                             [](const auto& a, const auto& b) { return a.second.last_access < b.second.last_access; });

        if (!oldest->second.interop_texture && oldest->second.texture_id != 0) {
            glDeleteTextures(1, &oldest->second.texture_id);
        }
        texture_cache_.erase(oldest);
    }

    GTTextureCache::TextureInfo GTTextureCache::loadTexture(const std::filesystem::path& path) {
        if (!std::filesystem::exists(path)) {
            return {};
        }

        try {
            auto [data, width, height, channels] = lfs::core::load_image(path);
            if (!data) {
                return {};
            }

            int out_width = width;
            int out_height = height;
            int scale = 1;
            while (out_width > MAX_TEXTURE_DIM || out_height > MAX_TEXTURE_DIM) {
                out_width /= 2;
                out_height /= 2;
                scale *= 2;
            }

            std::vector<unsigned char> final_data(out_width * out_height * channels);
            const int scale_sq = scale * scale;

            if (scale > 1) {
                for (int y = 0; y < out_height; ++y) {
                    const int src_y = (out_height - 1 - y) * scale;
                    for (int x = 0; x < out_width; ++x) {
                        const int src_x = x * scale;
                        for (int c = 0; c < channels; ++c) {
                            int sum = 0;
                            for (int sy = 0; sy < scale; ++sy) {
                                for (int sx = 0; sx < scale; ++sx) {
                                    sum += data[((src_y + sy) * width + src_x + sx) * channels + c];
                                }
                            }
                            final_data[(y * out_width + x) * channels + c] =
                                static_cast<unsigned char>(sum / scale_sq);
                        }
                    }
                }
            } else {
                const size_t row_size = width * channels;
                for (int y = 0; y < height; ++y) {
                    std::memcpy(final_data.data() + y * row_size,
                                data + (height - 1 - y) * row_size, row_size);
                }
            }

            lfs::core::free_image(data);

            unsigned int texture;
            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);

            const GLenum format = (channels == 1) ? GL_RED : (channels == 4) ? GL_RGBA
                                                                             : GL_RGB;
            const GLenum internal = (channels == 1) ? GL_R8 : (channels == 4) ? GL_RGBA8
                                                                              : GL_RGB8;

            glTexImage2D(GL_TEXTURE_2D, 0, internal, out_width, out_height, 0,
                         format, GL_UNSIGNED_BYTE, final_data.data());
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            return {texture, out_width, out_height};
        } catch (...) {
            return {};
        }
    }

    GTTextureCache::TextureInfo GTTextureCache::loadTextureFromLoader(
        lfs::io::PipelinedImageLoader& loader,
        const std::filesystem::path& path,
        const lfs::io::LoadParams& params,
        CacheEntry& entry) {
        if (!std::filesystem::exists(path)) {
            return {};
        }

        try {
            const auto tensor = loader.load_image_immediate(path, params);
            if (tensor.numel() == 0) {
                return {};
            }

            const auto& shape = tensor.shape();
            const int height = static_cast<int>(shape[1]);
            const int width = static_cast<int>(shape[2]);
            const auto hwc = tensor.permute({1, 2, 0}).contiguous();

            entry.interop_texture = std::make_unique<lfs::rendering::CudaGLInteropTexture>();
            if (auto result = entry.interop_texture->init(width, height); !result) {
                LOG_WARN("Failed to init GT interop texture from loader: {}", result.error());
                entry.interop_texture.reset();
                return loadTextureFromTensor(tensor, entry);
            }

            if (auto result = entry.interop_texture->updateFromTensor(hwc); !result) {
                LOG_WARN("Failed to upload GT texture from loader: {}", result.error());
                entry.interop_texture.reset();
                return loadTextureFromTensor(tensor, entry);
            }

            entry.width = width;
            entry.height = height;
            // CUDA/CPU tensor uploads preserve the conventional top-left image origin,
            // so split-view must flip Y when sampling them as OpenGL textures.
            entry.origin = kTensorTextureOrigin;

            const glm::vec2 tex_scale(
                entry.interop_texture->getTexcoordScaleX(),
                entry.interop_texture->getTexcoordScaleY());

            return {
                entry.interop_texture->getTextureID(),
                width,
                height,
                kTensorTextureOrigin,
                tex_scale};
        } catch (const std::exception& e) {
            LOG_WARN("GT loader path failed for {}: {}", lfs::core::path_to_utf8(path), e.what());
            entry.interop_texture.reset();
            return {};
        }
    }

    GTTextureCache::TextureInfo GTTextureCache::loadTextureFromTensor(const lfs::core::Tensor& tensor, CacheEntry& entry) {
        if (!tensor.is_valid() || tensor.ndim() != 3 || tensor.numel() == 0) {
            return {};
        }

        try {
            lfs::core::Tensor formatted = tensor;
            int channels = 0;
            int width = 0;
            int height = 0;

            const auto& shape = tensor.shape();
            const int first_dim = static_cast<int>(shape[0]);
            const int last_dim = static_cast<int>(shape[2]);

            if (first_dim == 1 || first_dim == 3 || first_dim == 4) {
                channels = first_dim;
                height = static_cast<int>(shape[1]);
                width = static_cast<int>(shape[2]);
                formatted = tensor.permute({1, 2, 0}).contiguous();
            } else if (last_dim == 1 || last_dim == 3 || last_dim == 4) {
                channels = last_dim;
                height = static_cast<int>(shape[0]);
                width = static_cast<int>(shape[1]);
                formatted = tensor.contiguous();
            } else {
                LOG_WARN("Unsupported GT tensor shape for CPU upload: [{}, {}, {}]",
                         static_cast<int>(shape[0]), static_cast<int>(shape[1]), static_cast<int>(shape[2]));
                return {};
            }

            if (formatted.device() == lfs::core::Device::CUDA) {
                formatted = formatted.cpu();
            }
            formatted = formatted.contiguous();

            if (formatted.dtype() != lfs::core::DataType::UInt8) {
                formatted = (formatted.clamp(0.0f, 1.0f) * 255.0f).to(lfs::core::DataType::UInt8);
            }

            unsigned int texture = 0;
            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            const GLenum format = (channels == 1) ? GL_RED : (channels == 4) ? GL_RGBA
                                                                             : GL_RGB;
            const GLenum internal = (channels == 1) ? GL_R8 : (channels == 4) ? GL_RGBA8
                                                                              : GL_RGB8;

            glTexImage2D(GL_TEXTURE_2D, 0, internal, width, height, 0,
                         format, GL_UNSIGNED_BYTE, formatted.ptr<unsigned char>());

            if (const GLenum gl_err = glGetError(); gl_err != GL_NO_ERROR) {
                glBindTexture(GL_TEXTURE_2D, 0);
                glDeleteTextures(1, &texture);
                LOG_WARN("Failed to upload GT texture through CPU path: {}", static_cast<int>(gl_err));
                return {};
            }
            glBindTexture(GL_TEXTURE_2D, 0);

            entry.interop_texture.reset();
            entry.texture_id = texture;
            entry.width = width;
            entry.height = height;
            entry.origin = kTensorTextureOrigin;

            return {texture, width, height, kTensorTextureOrigin};
        } catch (const std::exception& e) {
            LOG_WARN("GT tensor CPU upload failed: {}", e.what());
            return {};
        }
    }

    GTTextureCache::TextureInfo GTTextureCache::loadTextureGPU(const std::filesystem::path& path, CacheEntry& entry) {
        if (!nvcodec_loader_) {
            return {};
        }

        try {
            const auto tensor = nvcodec_loader_->load_image_gpu(path, 1, MAX_TEXTURE_DIM);
            if (tensor.numel() == 0) {
                return {};
            }

            const auto& shape = tensor.shape();
            const int height = static_cast<int>(shape[1]);
            const int width = static_cast<int>(shape[2]);

            const auto hwc = tensor.permute({1, 2, 0}).contiguous();

            entry.interop_texture = std::make_unique<lfs::rendering::CudaGLInteropTexture>();
            if (auto result = entry.interop_texture->init(width, height); !result) {
                LOG_WARN("Failed to init interop texture: {}", result.error());
                entry.interop_texture.reset();
                return {};
            }

            if (auto result = entry.interop_texture->updateFromTensor(hwc); !result) {
                LOG_WARN("Failed to upload to interop texture: {}", result.error());
                entry.interop_texture.reset();
                return {};
            }

            entry.width = width;
            entry.height = height;
            entry.origin = kTensorTextureOrigin;

            const glm::vec2 tex_scale(
                entry.interop_texture->getTexcoordScaleX(),
                entry.interop_texture->getTexcoordScaleY());

            return {
                entry.interop_texture->getTextureID(),
                width,
                height,
                kTensorTextureOrigin,
                tex_scale};
        } catch (const std::exception& e) {
            LOG_WARN("GPU texture load failed: {}", e.what());
            entry.interop_texture.reset();
            return {};
        }
    }

} // namespace lfs::vis
