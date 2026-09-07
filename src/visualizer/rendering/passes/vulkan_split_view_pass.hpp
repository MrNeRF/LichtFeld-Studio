/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor_fwd.hpp"

#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <vulkan/vulkan.h>

namespace lfs::vis {

    class VulkanContext;

    struct VulkanSplitViewPanel {
        std::shared_ptr<const lfs::core::Tensor> image;
        float start_position = 0.0f;
        float end_position = 1.0f;
        bool normalize_x_to_panel = false;
        bool flip_y = false;
        // When set, the pass binds this VkImageView directly and skips the staging
        // upload path. The caller owns the image and must keep it alive through
        // the viewport pass record/submit.
        VkImageView external_image_view = VK_NULL_HANDLE;
        std::uint64_t external_image_generation = 0;
        // Valid-region UV for padded panel textures (default identity).
        glm::vec2 uv_scale{1.0f, 1.0f};
        glm::vec2 uv_clamp_max{1.0f, 1.0f};
        // Additional mapping from full-content UVs into a clipped render target.
        glm::vec2 texcoord_scale{1.0f, 1.0f};
        glm::vec2 texcoord_offset{0.0f, 0.0f};
        // Set only for a panel rendered at the reconstruction input resolution.
        // Reference and ground-truth panels must remain unfiltered.
        bool spatial_filter = false;
    };

    struct VulkanSplitViewParams {
        bool enabled = false;
        VulkanSplitViewPanel left;
        VulkanSplitViewPanel right;
        float split_position = 0.5f;
        glm::ivec4 content_rect{0, 0, 0, 0}; // x, y, w, h (letterboxed)
        glm::ivec2 coordinate_extent{0, 0};
        glm::vec3 background{0.0f};
        bool exact_texel_sampling = false;
    };

    // GPU split-view composite. Replaces the CPU compositeSplitImages scanline loop
    // with a fragment shader that samples the two panel textures, picks left/right
    // based on x, and overlays the master-style divider/handle/grip.
    class VulkanSplitViewPass {
    public:
        VulkanSplitViewPass();
        ~VulkanSplitViewPass();

        VulkanSplitViewPass(const VulkanSplitViewPass&) = delete;
        VulkanSplitViewPass& operator=(const VulkanSplitViewPass&) = delete;
        VulkanSplitViewPass(VulkanSplitViewPass&&) noexcept;
        VulkanSplitViewPass& operator=(VulkanSplitViewPass&&) noexcept;

        [[nodiscard]] bool init(VulkanContext& context, VkFormat color_format,
                                VkFormat depth_format, VkBuffer screen_quad_buffer);
        void prepare(const VulkanSplitViewParams& params, std::size_t frame_slot);
        // panel_rect and params.content_rect are both in framebuffer-space coords.
        void record(VkCommandBuffer cb, const VkRect2D& panel_rect,
                    const VulkanSplitViewParams& params, std::size_t frame_slot);
        void shutdown();

        [[nodiscard]] bool ready(std::size_t frame_slot) const;
        [[nodiscard]] bool available() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::vis
