/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/scene_depth_contract.hpp"
#include "rendering/temporal_frame_tracker.hpp"

#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <limits>
#include <memory>
#include <optional>
#include <vulkan/vulkan.h>

namespace lfs::vis {

    class VulkanContext;

    // Vectors are jitter-free, in top-left render pixels, pointing from the
    // current pixel to its previous-frame position.
    struct VulkanSceneMotionParams {
        bool enabled = false;
        VkImageView depth_view = VK_NULL_HANDLE;
        std::uint64_t depth_generation = 0;
        SceneDepthContract depth;
        glm::mat4 inverse_current_view_projection{1.0f};
        glm::mat4 previous_view_projection{1.0f};
        glm::ivec2 render_extent{0, 0};
        bool flip_y = false;
    };

    [[nodiscard]] inline bool canRecordVulkanSceneMotion(
        const VulkanSceneMotionParams& params) noexcept {
        return params.enabled && params.depth_view != VK_NULL_HANDLE && params.depth.available() &&
               params.depth.valid() && params.depth.storage == SceneDepthStorage::VulkanImage &&
               (params.depth.encoding == SceneDepthEncoding::VulkanNdc ||
                params.depth.encoding == SceneDepthEncoding::LinearView) &&
               params.depth.matchesRenderExtent(params.render_extent);
    }

    [[nodiscard]] constexpr std::optional<std::size_t> temporalMotionResourceSlot(
        const std::size_t frame_slot, const TemporalViewId view) noexcept {
        constexpr auto VIEW_COUNT = static_cast<std::size_t>(TemporalViewId::Count);
        const auto view_index = static_cast<std::size_t>(view);
        if (view_index >= VIEW_COUNT ||
            frame_slot > (std::numeric_limits<std::size_t>::max() - view_index) / VIEW_COUNT) {
            return std::nullopt;
        }
        return frame_slot * VIEW_COUNT + view_index;
    }

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
        [[nodiscard]] VkImage motionImage(std::size_t frame_slot) const;
        [[nodiscard]] bool initialized() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::vis
