/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "config.h"

#ifdef LFS_VULKAN_VIEWER_ENABLED
#include <memory>
#include <vulkan/vulkan.h>

namespace lfs::vis {
    class VulkanContext;
    struct VulkanViewportPassParams;

    class VulkanSceneImageUploader {
    public:
        VulkanSceneImageUploader();
        ~VulkanSceneImageUploader();

        VulkanSceneImageUploader(const VulkanSceneImageUploader&) = delete;
        VulkanSceneImageUploader& operator=(const VulkanSceneImageUploader&) = delete;
        VulkanSceneImageUploader(VulkanSceneImageUploader&&) noexcept;
        VulkanSceneImageUploader& operator=(VulkanSceneImageUploader&&) noexcept;

        [[nodiscard]] bool init(VulkanContext& context, VkSampler scene_sampler);
        void shutdown();
        void upload(const VulkanViewportPassParams& params, VkDescriptorSet scene_descriptor_set);
        [[nodiscard]] bool hasImage() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::vis
#endif
