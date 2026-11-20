/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "vr_hands.hpp"
#include "window/vulkan_context.hpp"
#include <array>
#include <glm/glm.hpp>
#include <memory>
#include <span>

namespace lfs::vis::vr {

    // Draws the hands over the eye images of a VR frame, as see-through hands
    // of light. They are drawn last and in front of the scene: an eye image
    // carries no depth.
    class HandRenderer {
    public:
        struct Eye {
            VkImageView target = VK_NULL_HANDLE; // in COLOR_ATTACHMENT_OPTIMAL
            glm::ivec2 size{0};
            glm::mat4 view_projection{1.0f}; // tracking space to Vulkan clip space
            glm::vec3 position{0.0f};        // in the tracking space
        };

        HandRenderer();
        ~HandRenderer();
        HandRenderer(const HandRenderer&) = delete;
        HandRenderer& operator=(const HandRenderer&) = delete;

        // size: the largest eye image that will be drawn on.
        [[nodiscard]] lfs::Status init(VulkanContext& context, VkFormat color_format, glm::ivec2 size);
        void shutdown();
        [[nodiscard]] bool ready() const;

        // The hands of the coming frame, left then right, each with its linear
        // color; an empty mesh hides a hand. The GPU must be done with the
        // previous frame's draws.
        void setHands(const std::array<std::span<const HandVertex>, 2>& meshes, const std::array<glm::vec3, 2>& colors);
        [[nodiscard]] bool hasHands() const;
        void record(VkCommandBuffer command, const Eye& eye);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::vis::vr
