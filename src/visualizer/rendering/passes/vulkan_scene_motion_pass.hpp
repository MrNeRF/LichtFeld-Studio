/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/scene_motion_contract.hpp"

#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <vulkan/vulkan.h>

namespace lfs::vis {

    class VulkanContext;

    struct VulkanSceneMotionParams {
        bool enabled = false;
        VkImageView depth_view = VK_NULL_HANDLE;
        std::uint64_t depth_generation = 0;
        glm::mat4 inverse_current_view_projection{1.0f};
        glm::mat4 previous_view_projection{1.0f};
        glm::ivec2 render_extent{0, 0};
        bool includes_jitter = false;
        bool flip_y = false;
    };

    class VulkanSceneMotionPass {
    public:
        VulkanSceneMotionPass();
        ~VulkanSceneMotionPass();

        VulkanSceneMotionPass(const VulkanSceneMotionPass&) = delete;
        VulkanSceneMotionPass& operator=(const VulkanSceneMotionPass&) = delete;
        VulkanSceneMotionPass(VulkanSceneMotionPass&&) noexcept;
        VulkanSceneMotionPass& operator=(VulkanSceneMotionPass&&) noexcept;

        // Stores the context only. Vulkan objects are created by the first enabled record().
        [[nodiscard]] bool init(VulkanContext& context);
        [[nodiscard]] bool record(VkCommandBuffer command_buffer,
                                  const VulkanSceneMotionParams& params,
                                  std::size_t frame_slot);
        void shutdown();

        [[nodiscard]] VkImageView motionView(std::size_t frame_slot) const;
        [[nodiscard]] SceneMotionContract contract(std::size_t frame_slot) const;
        [[nodiscard]] bool initialized() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::vis
