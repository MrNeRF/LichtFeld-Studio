/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "config.h"

#ifdef LFS_VULKAN_VIEWER_ENABLED

#include "core/export.hpp"
#include "core/tensor.hpp"
#include "rendering/rendering.hpp"
#include "window/vulkan_context.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <glm/glm.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace lfs::vis {

    // Vulkan-native point cloud rasterizer. Renders points as disk-splats via a
    // graphics pipeline with hardware depth test (equivalent to the CUDA
    // atomicMin path), then exposes the color + depth VkImages so the rendering
    // manager can route them through the same external-image plumbing as
    // VkSplat (no CUDA tensor staging on the display path).
    class LFS_VIS_API PointCloudVulkanRenderer {
    public:
        struct RenderResult {
            VkImage image = VK_NULL_HANDLE;
            VkImageView image_view = VK_NULL_HANDLE;
            VkImageLayout image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            std::uint64_t generation = 0;
            VkImage depth_image = VK_NULL_HANDLE;
            VkImageView depth_image_view = VK_NULL_HANDLE;
            VkImageLayout depth_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            std::uint64_t depth_generation = 0;
            glm::ivec2 size{0, 0};
            bool flip_y = false;
        };

        struct CropBox {
            glm::mat4 to_local{1.0f};
            glm::vec3 min{0.0f};
            glm::vec3 max{0.0f};
            bool inverse = false;
            bool desaturate = false;
        };

        struct RenderRequest {
            // Positions/colors are float [N, 3] tensors. Either CUDA or CPU; we
            // copy to Vulkan device-local on first use and cache by pointer.
            const lfs::core::Tensor* positions = nullptr;
            const lfs::core::Tensor* colors = nullptr;

            // Optional model_transforms[K, 16] + per-point transform_indices[N];
            // empty/null disables the transform path.
            const std::vector<glm::mat4>* model_transforms = nullptr;
            const lfs::core::Tensor* transform_indices = nullptr;

            // Optional per-transform visibility (size matches model_transforms).
            const std::vector<bool>* node_visibility_mask = nullptr;

            // Optional crop. When set, points outside the local box are either
            // dropped (default) or rendered desaturated (crop.desaturate).
            std::optional<CropBox> crop;

            glm::mat4 view{1.0f};
            glm::mat4 view_projection{1.0f};
            glm::ivec2 size{0, 0};
            glm::vec3 background_color{0.0f};
            bool transparent_background = false;
            bool orthographic = false;
            float ortho_scale = 1.0f;
            float focal_y = 1.0f;
            float voxel_size = 0.01f;
            float scaling_modifier = 1.0f;
        };

        enum class OutputSlot : std::size_t {
            Main = 0,
            SplitLeft = 1,
            SplitRight = 2,
        };

        PointCloudVulkanRenderer();
        ~PointCloudVulkanRenderer();

        PointCloudVulkanRenderer(const PointCloudVulkanRenderer&) = delete;
        PointCloudVulkanRenderer& operator=(const PointCloudVulkanRenderer&) = delete;

        [[nodiscard]] std::expected<RenderResult, std::string> render(
            VulkanContext& context,
            const RenderRequest& request,
            OutputSlot output_slot = OutputSlot::Main);

        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::vis

#endif // LFS_VULKAN_VIEWER_ENABLED
