/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/scene_temporal_resolve.hpp"
#include "rendering/scene_upscaler_registry.hpp"
#include "rendering/temporal_frame_tracker.hpp"

#include <cstdint>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

namespace lfs::vis {
    class VulkanContext;

    struct VulkanSceneUpscalerResource {
        VkImage image = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        glm::ivec2 valid_extent{0, 0};
        glm::ivec2 allocation_extent{0, 0};
        std::uint64_t generation = 0;

        [[nodiscard]] constexpr bool valid() const {
            return image != VK_NULL_HANDLE && view != VK_NULL_HANDLE &&
                   layout != VK_IMAGE_LAYOUT_UNDEFINED && valid_extent.x > 0 &&
                   valid_extent.y > 0 && allocation_extent.x >= valid_extent.x &&
                   allocation_extent.y >= valid_extent.y;
        }
    };

    struct VulkanSceneUpscalerDispatch {
        TemporalViewId view = TemporalViewId::Main;
        VulkanSceneUpscalerResource color;
        VulkanSceneUpscalerResource depth;
        VulkanSceneUpscalerResource motion;
        glm::ivec2 output_extent{0, 0};
        glm::vec2 jitter_pixels{0.0f};
        glm::vec2 previous_jitter_pixels{0.0f};
        bool motion_includes_jitter = false;
        float exposure = 1.0f;
        float frame_time_seconds = 0.0f;
        std::uint64_t sequence = 0;
        TemporalResetReason reset_reasons = TemporalResetReason::None;
        SceneTemporalQuality quality = SceneTemporalQuality::Balanced;

        [[nodiscard]] constexpr bool valid(
            const SceneUpscalerRequirements requirements) const {
            if (!color.valid() || output_extent.x <= 0 || output_extent.y <= 0 ||
                frame_time_seconds < 0.0f || exposure < 0.0f)
                return false;
            if (requirements.depth && !depth.valid())
                return false;
            if (requirements.motion_vectors && !motion.valid())
                return false;
            return true;
        }
    };

    struct VulkanSceneUpscalerOutput {
        VulkanSceneUpscalerResource color;

        [[nodiscard]] constexpr bool valid(const glm::ivec2 expected_extent) const {
            return color.valid() && color.valid_extent == expected_extent;
        }
    };

    class VulkanSceneUpscalerAdapter : public SceneUpscalerAdapter {
    public:
        ~VulkanSceneUpscalerAdapter() override = default;

        [[nodiscard]] virtual SceneUpscalerAvailability initialize(
            VulkanContext& context) noexcept = 0;
        [[nodiscard]] virtual bool record(
            VkCommandBuffer command_buffer,
            const VulkanSceneUpscalerDispatch& dispatch) noexcept = 0;
        [[nodiscard]] virtual VulkanSceneUpscalerOutput output(
            TemporalViewId view) const noexcept = 0;
        virtual void reset(TemporalViewId view, TemporalResetReason reasons) noexcept = 0;
        virtual void shutdown() noexcept = 0;
    };
} // namespace lfs::vis
