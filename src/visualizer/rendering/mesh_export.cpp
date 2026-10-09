/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "mesh_offscreen_renderer.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>

namespace lfs::vis {
    glm::mat4 cropProjectionToRect(const glm::mat4& projection,
                                   const glm::ivec2 full_size,
                                   const glm::ivec2 origin,
                                   const glm::ivec2 size) noexcept {
        const glm::vec2 full(full_size);
        const glm::vec2 scale = full / glm::vec2(size);
        const glm::vec2 center{-1.0f + (2.0f * static_cast<float>(origin.x) + static_cast<float>(size.x)) / full.x,
                               1.0f - (2.0f * static_cast<float>(origin.y) + static_cast<float>(size.y)) / full.y};
        glm::mat4 crop(1.0f);
        crop[0][0] = scale.x;
        crop[3][0] = -scale.x * center.x;
        crop[1][1] = scale.y;
        crop[3][1] = -scale.y * center.y;
        return crop * projection;
    }

    void compositeMeshLayer(const MeshLayer& layer,
                            const float* const splat_depth,
                            const std::size_t splat_depth_stride,
                            lfs::core::Tensor& image,
                            const glm::ivec2 origin) {
        assert(layer.rgba.device() == lfs::core::Device::CPU && layer.rgba.ndim() == 3 && layer.rgba.size(0) == 4);
        assert(layer.view_depth.ndim() == 2 && layer.view_depth.size(0) == layer.rgba.size(1) &&
               layer.view_depth.size(1) == layer.rgba.size(2));
        assert(image.device() == lfs::core::Device::CPU && image.dtype() == lfs::core::DataType::UInt8 &&
               image.ndim() == 3 && image.is_contiguous() && (image.size(2) == 3 || image.size(2) == 4));
        const std::size_t rows = layer.view_depth.size(0);
        const std::size_t cols = layer.view_depth.size(1);
        assert(origin.x >= 0 && origin.y >= 0 && static_cast<std::size_t>(origin.y) + rows <= image.size(0) &&
               static_cast<std::size_t>(origin.x) + cols <= image.size(1));
        const std::size_t plane = rows * cols;
        const std::size_t channels = image.size(2);
        const std::size_t image_stride = image.size(1) * channels;
        const float* const rgba = layer.rgba.ptr<float>();
        const float* const mesh_depth = layer.view_depth.ptr<float>();
        auto* const image_origin = image.ptr<std::uint8_t>() + static_cast<std::size_t>(origin.y) * image_stride +
                                   static_cast<std::size_t>(origin.x) * channels;
        for (std::size_t y = 0; y < rows; ++y) {
            const float* const splat_row = splat_depth ? splat_depth + y * splat_depth_stride : nullptr;
            auto* const image_row = image_origin + y * image_stride;
            for (std::size_t x = 0; x < cols; ++x) {
                const std::size_t pixel = y * cols + x;
                if (rgba[3 * plane + pixel] <= 0.0f || (splat_row && !(mesh_depth[pixel] < splat_row[x]))) {
                    continue;
                }
                auto* const out = image_row + x * channels;
                for (std::size_t c = 0; c < 3; ++c) {
                    out[c] = static_cast<std::uint8_t>(std::clamp(rgba[c * plane + pixel], 0.0f, 1.0f) * 255.0f + 0.5f);
                }
                if (channels == 4) {
                    out[3] = 255;
                }
            }
        }
    }

} // namespace lfs::vis
