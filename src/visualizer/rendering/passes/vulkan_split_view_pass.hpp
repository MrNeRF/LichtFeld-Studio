/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "config.h"

#ifdef LFS_VULKAN_VIEWER_ENABLED
#include "core/tensor.hpp"

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
        // upload path. The view must outlive the frame's record/submit (gui_manager
        // owns the per-frame interop slot it points into).
        VkImageView external_image_view = VK_NULL_HANDLE;
        std::uint64_t external_image_generation = 0;
    };

    struct VulkanSplitViewParams {
        bool enabled = false;
        VulkanSplitViewPanel left;
        VulkanSplitViewPanel right;
        float split_position = 0.5f;
        glm::ivec4 content_rect{0, 0, 0, 0}; // x, y, w, h (letterboxed)
        glm::vec3 background{0.0f};
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
        void prepare(const VulkanSplitViewParams& params);
        // panel_rect and params.content_rect are both in framebuffer-space coords.
        void record(VkCommandBuffer cb, const VkRect2D& panel_rect, const VulkanSplitViewParams& params);
        void shutdown();

        [[nodiscard]] bool ready() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::vis
#endif
