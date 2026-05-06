/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "config.h"

#ifdef LFS_VULKAN_VIEWER_ENABLED
#include "core/tensor.hpp"

#include <cstdint>
#include <memory>
#include <vulkan/vulkan.h>

namespace lfs::vis {

    class VulkanContext;

    struct VulkanDepthBlitParams {
        std::shared_ptr<const lfs::core::Tensor> depth; // [1, H, W] CUDA float
        bool depth_is_ndc = false;
        bool flip_y = false;
        float near_plane = 0.1f;
        float far_plane = 1000.0f;
        // When set, the pass binds this VkImageView (a CUDA/Vulkan interop slot owned
        // by gui_manager) and skips the staging upload path.
        VkImageView external_image_view = VK_NULL_HANDLE;
        std::uint64_t external_image_generation = 0;
    };

    // Writes a sampled depth value into the framebuffer's depth attachment via
    // gl_FragDepth. Lets the mesh pass that follows depth-test against splat depth.
    class VulkanDepthBlitPass {
    public:
        VulkanDepthBlitPass();
        ~VulkanDepthBlitPass();

        VulkanDepthBlitPass(const VulkanDepthBlitPass&) = delete;
        VulkanDepthBlitPass& operator=(const VulkanDepthBlitPass&) = delete;
        VulkanDepthBlitPass(VulkanDepthBlitPass&&) noexcept;
        VulkanDepthBlitPass& operator=(VulkanDepthBlitPass&&) noexcept;

        [[nodiscard]] bool init(VulkanContext& context, VkFormat color_format,
                                VkFormat depth_format, VkBuffer screen_quad_buffer);
        void prepare(const VulkanDepthBlitParams& params);
        void record(VkCommandBuffer cb, VkRect2D rect, const VulkanDepthBlitParams& params);
        void shutdown();

        [[nodiscard]] bool hasDepth() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::vis
#endif
