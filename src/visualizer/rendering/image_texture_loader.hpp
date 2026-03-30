/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/cuda_gl_interop.hpp"

#include <glm/vec2.hpp>

#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>

namespace lfs::core {
    class Tensor;
}

namespace lfs::io {
    struct LoadParams;
    class PipelinedImageLoader;
} // namespace lfs::io

namespace lfs::vis::image_texture {

    struct LoadedTexture {
        std::unique_ptr<lfs::rendering::CudaGLInteropTexture> interop_texture;
        unsigned int texture_id = 0;
        int width = 0;
        int height = 0;
        glm::vec2 texcoord_scale{1.0f};
    };

    [[nodiscard]] std::optional<LoadedTexture> load_texture_from_loader(
        lfs::io::PipelinedImageLoader& loader,
        const std::filesystem::path& path,
        const lfs::io::LoadParams& params,
        std::string_view log_context);

    [[nodiscard]] std::optional<LoadedTexture> load_texture_from_tensor(
        const lfs::core::Tensor& tensor,
        std::string_view log_context);

} // namespace lfs::vis::image_texture
