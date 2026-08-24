/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/scene_temporal_plan.hpp"
#include "rendering/temporal_frame_tracker.hpp"
#include "vulkan_scene_depth_history_contract.hpp"

#include <algorithm>
#include <cstddef>
#include <glm/glm.hpp>
#include <memory>
#include <optional>
#include <vulkan/vulkan.h>

namespace lfs::vis {
    class VulkanContext;

    [[nodiscard]] constexpr bool validTemporalViewId(const TemporalViewId view) noexcept {
        return static_cast<std::size_t>(view) <
               static_cast<std::size_t>(TemporalViewId::Count);
    }

    struct VulkanSceneTemporalResourceStats {
        std::size_t history_bytes = 0;
        std::size_t history_images = 0;
        std::size_t resident_views = 0;
        bool static_resources_initialized = false;
    };

    [[nodiscard]] constexpr std::size_t nextTemporalHistoryWriteIndex(
        const bool has_history, const std::size_t read_index) {
        return has_history ? 1u - std::min<std::size_t>(read_index, 1u) : 0u;
    }

    [[nodiscard]] constexpr glm::vec4 temporalCurrentUvTransform(
        const glm::ivec2 render_extent, const glm::ivec2 allocation_extent) {
        if (render_extent.x <= 0 || render_extent.y <= 0 || allocation_extent.x <= 0 ||
            allocation_extent.y <= 0 || render_extent.x > allocation_extent.x ||
            render_extent.y > allocation_extent.y) {
            return {};
        }
        return {
            static_cast<float>(render_extent.x) / allocation_extent.x,
            static_cast<float>(render_extent.y) / allocation_extent.y,
            (static_cast<float>(render_extent.x) - 0.5f) / allocation_extent.x,
            (static_cast<float>(render_extent.y) - 0.5f) / allocation_extent.y,
        };
    }

    struct VulkanSceneTemporalResolveParams {
        bool enabled = false;
        TemporalViewId view = TemporalViewId::Main;
        VkImageView current_color_view = VK_NULL_HANDLE;
        VkImageView motion_view = VK_NULL_HANDLE;
        VkImageLayout current_color_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkImageLayout motion_layout = VK_IMAGE_LAYOUT_GENERAL;
        glm::ivec2 render_extent{0, 0};
        glm::ivec2 output_extent{0, 0};
        glm::ivec2 current_allocation_extent{0, 0};
        glm::vec2 current_jitter_ndc{0.0f};
        glm::vec2 previous_jitter_ndc{0.0f};
        bool jitter_flip_y = false;
        std::uint64_t sequence = 0;
        bool history_valid = false;
        float history_weight = 0.9f;
        float motion_rejection_pixels = 128.0f;
        float motion_confidence_pixels = 0.30f;
        float current_sharpness = 0.10f;
        VulkanSceneDepthHistoryParams current_depth;
        float depth_relative_threshold = 0.01f;
        float depth_absolute_threshold = 1e-4f;
    };

    [[nodiscard]] inline bool validTemporalDepthInputs(
        const VulkanSceneTemporalResolveParams& params) noexcept {
        return !params.current_depth.enabled ||
               (canRecordVulkanSceneDepthHistory(params.current_depth) &&
                params.current_depth.view == params.view &&
                params.current_depth.depth.matchesRenderExtent(params.render_extent));
    }

    [[nodiscard]] constexpr std::optional<std::size_t> temporalDepthHistoryResourceSlot(
        const TemporalViewId view, const std::size_t ping_index) noexcept {
        if (!validTemporalViewId(view) || ping_index > 1)
            return std::nullopt;
        return static_cast<std::size_t>(view) * 2 + ping_index;
    }

    class VulkanSceneTemporalResolvePass {
    public:
        VulkanSceneTemporalResolvePass();
        ~VulkanSceneTemporalResolvePass();
        VulkanSceneTemporalResolvePass(const VulkanSceneTemporalResolvePass&) = delete;
        VulkanSceneTemporalResolvePass& operator=(const VulkanSceneTemporalResolvePass&) = delete;
        VulkanSceneTemporalResolvePass(VulkanSceneTemporalResolvePass&&) noexcept;
        VulkanSceneTemporalResolvePass& operator=(VulkanSceneTemporalResolvePass&&) noexcept;

        [[nodiscard]] bool init(VulkanContext& context);
        [[nodiscard]] bool record(VkCommandBuffer command_buffer,
                                  const VulkanSceneTemporalResolveParams& params);
        void reset(TemporalViewId view);
        void resetAll();
        // Retain immutable pipeline state while freeing all per-view color and
        // depth history allocations after submitted users have retired.
        void releaseHistory();
        void shutdown();

        [[nodiscard]] VkImageView outputView(TemporalViewId view) const;
        [[nodiscard]] SceneHistoryContract contract(TemporalViewId view) const;
        [[nodiscard]] bool initialized() const;
        [[nodiscard]] VulkanSceneTemporalResourceStats resourceStats() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::vis
