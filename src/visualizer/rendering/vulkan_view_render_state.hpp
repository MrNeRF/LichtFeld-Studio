/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "passes/vulkan_depth_blit_pass.hpp"
#include "passes/vulkan_environment_pass.hpp"
#include "passes/vulkan_mesh_pass.hpp"
#include "passes/vulkan_split_view_pass.hpp"
#include "scene_renderer.hpp"
#include "view_render_state.hpp"
#include "vulkan_viewport_frame_resources.hpp"

#include <mutex>
#include <vulkan/vulkan.h>

namespace lfs::vis {
    struct VulkanMeshFrame {
        struct TemporalFrame {
            TemporalFrameInput input;
            SceneTemporalResolveSettings resolve_settings;
            SceneTemporalQuality quality = SceneTemporalQuality::Balanced;
        };

        glm::mat4 view_projection{1.0f};
        glm::vec3 camera_position{0.0f};
        std::vector<VulkanMeshDrawItem> items;
        std::vector<VulkanMeshViewportPanel> panels;
        VulkanEnvironmentParams environment;
        VulkanDepthBlitParams depth_blit;
        VulkanSplitViewParams split_view;
        std::optional<TemporalFrame> temporal;
        std::array<std::shared_ptr<void>, 2> split_output_lifetimes;

        void retainSplitOutputs(SceneRenderer* renderer) {
            if (!renderer || !split_view.enabled)
                return;
            split_output_lifetimes = {
                renderer->retainOutputImage(SceneImageViewHandle::fromNative(split_view.left.external_image_view)),
                renderer->retainOutputImage(SceneImageViewHandle::fromNative(split_view.right.external_image_view))};
        }
    };

    struct ViewportReferenceState {
        ViewportInteropService viewport_interop_;
        std::shared_ptr<VulkanViewportFrameResources> pending_frame_resources;
        VkImage external_viewport_image = VK_NULL_HANDLE;
        VkImageView external_viewport_image_view = VK_NULL_HANDLE;
        VkImageLayout external_viewport_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        std::uint64_t external_viewport_image_generation = 0;
        mutable std::mutex mesh_frame_mutex;
        VulkanMeshFrame mesh_frame;

        void clearSceneOutput() {
            external_viewport_image = VK_NULL_HANDLE;
            external_viewport_image_view = VK_NULL_HANDLE;
            external_viewport_image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            external_viewport_image_generation = 0;
        }
        [[nodiscard]] std::optional<std::uint64_t>
        sceneOutputGeneration() const {
            if (external_viewport_image == VK_NULL_HANDLE)
                return std::nullopt;
            return external_viewport_image_generation;
        }
    };

    using VulkanViewRenderState = ViewportReferenceState;
    inline VulkanViewRenderState& vulkanViewRenderState(ViewRenderState& view) {
        auto state = view.reference_state_;
        if (!state) {
            state = std::make_shared<VulkanViewRenderState>();
            view.reference_state_ = state;
        }
        return *state;
    }

    inline const VulkanViewRenderState* vulkanViewRenderStateOrNull(
        const ViewRenderState& view) noexcept {
        return view.reference_state_.get();
    }
} // namespace lfs::vis
