/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/cuda_gl_interop.hpp"
#include "rendering/frame_contract.hpp"
#include <chrono>
#include <filesystem>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <unordered_map>

namespace lfs::io {
    struct LoadParams;
    class PipelinedImageLoader;
    class NvCodecImageLoader;
} // namespace lfs::io

namespace lfs::core {
    class Tensor;
} // namespace lfs::core

namespace lfs::vis {

    class GTTextureCache {
    public:
        static constexpr int MAX_TEXTURE_DIM = 2048;

        struct TextureInfo {
            unsigned int texture_id = 0;
            int width = 0;
            int height = 0;
            lfs::rendering::TextureOrigin origin = lfs::rendering::TextureOrigin::BottomLeft;
            glm::vec2 texcoord_scale{1.0f};
        };

        GTTextureCache();
        ~GTTextureCache();

        TextureInfo getGTTexture(int cam_id, const std::filesystem::path& image_path,
                                 lfs::io::PipelinedImageLoader* pipeline_loader = nullptr,
                                 const lfs::io::LoadParams* load_params = nullptr);
        void clear();

    private:
        struct CacheEntry {
            std::unique_ptr<lfs::rendering::CudaGLInteropTexture> interop_texture;
            unsigned int texture_id = 0;
            int width = 0;
            int height = 0;
            lfs::rendering::TextureOrigin origin = lfs::rendering::TextureOrigin::BottomLeft;
            std::string load_signature;
            std::chrono::steady_clock::time_point last_access;
        };

        std::unordered_map<int, CacheEntry> texture_cache_;
        std::unique_ptr<lfs::io::NvCodecImageLoader> nvcodec_loader_;
        static constexpr size_t MAX_CACHE_SIZE = 20;

        void evictOldest();
        TextureInfo loadTextureFromLoader(lfs::io::PipelinedImageLoader& loader,
                                          const std::filesystem::path& path,
                                          const lfs::io::LoadParams& params,
                                          CacheEntry& entry);
        TextureInfo loadTextureFromTensor(const lfs::core::Tensor& tensor, CacheEntry& entry);
        TextureInfo loadTexture(const std::filesystem::path& path);
        TextureInfo loadTextureGPU(const std::filesystem::path& path, CacheEntry& entry);
    };

} // namespace lfs::vis
