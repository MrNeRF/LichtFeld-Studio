/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "camera_interaction_service.hpp"
#include "core/tensor.hpp"
#include "core/tensor_image.hpp"
#include "depth_window_state.hpp"
#include "dirty_flags.hpp"
#include "framerate_controller.hpp"
#include "gt_comparison_state.hpp"
#include "internal/viewport.hpp"
#include "passes/vulkan_depth_blit_pass.hpp"
#include "passes/vulkan_environment_pass.hpp"
#include "passes/vulkan_mesh_pass.hpp"
#include "passes/vulkan_split_view_pass.hpp"
#include "render_animation_state.hpp"
#include "render_target_id.hpp"
#include "rendering/rendering.hpp"
#include "rendering/scene_temporal_resolve.hpp"
#include "rendering/scene_upscaler_registry.hpp"
#include "rendering/screen_overlay_renderer.hpp"
#include "rendering/temporal_frame_tracker.hpp"
#include "rendering_types.hpp"
#include "split_view_service.hpp"
#include "stale_frame_guard.hpp"
#include "view_source.hpp"
#include "viewport_artifact_service.hpp"
#include "viewport_frame_lifecycle_service.hpp"
#include "viewport_interaction_context.hpp"
#include "viewport_interop_service.hpp"
#include "viewport_overlay_service.hpp"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace lfs::vis {
    struct FramebufferViewportRect {
        glm::ivec2 top_left{0, 0};
        glm::ivec2 size{0, 0};

        [[nodiscard]] bool valid() const { return size.x > 0 && size.y > 0; }
    };
    struct VulkanMeshFrame {
        struct TemporalFrame {
            TemporalFrameInput input;
            SceneTemporalResolveSettings resolve_settings;
            SceneTemporalQuality quality = SceneTemporalQuality::Balanced;
        };

        glm::mat4 view_projection{1.0f};
        glm::vec3 camera_position{0.0f};
        std::vector<lfs::vis::VulkanMeshDrawItem> items;
        std::vector<lfs::vis::VulkanMeshViewportPanel> panels;
        lfs::vis::VulkanEnvironmentParams environment;
        lfs::vis::VulkanDepthBlitParams depth_blit;
        lfs::vis::VulkanSplitViewParams split_view;
        std::optional<TemporalFrame> temporal;
    };
    struct GTPresentedView {
        GTRenderCamera camera;
        glm::ivec2 size{0, 0};
    };
    struct ViewRenderState {
        ViewportOverlayService viewport_overlay_service_;
        ViewId id = kNoView;
        std::chrono::steady_clock::time_point last_visible{};
        std::optional<RenderSettings> rendered_settings;
        lfs::rendering::ScreenOverlayRenderer screen_overlay_renderer_;
        mutable FramerateController framerate_controller_;
        std::shared_ptr<const lfs::core::Tensor> vulkan_viewport_image_;
        std::uint64_t vulkan_viewport_image_generation_ = 0;
        std::string last_logged_vksplat_render_error_;
        StaleFrameGuard vksplat_stale_frame_guard_;
        DirtyMask parked_arena_retry_ = 0;
        std::atomic<DirtyMask> training_refresh_dirty_{0};
        std::uint64_t viewport_projection_generation_ = 1;
        std::uint64_t temporal_scene_revision_ = 1;
        TemporalConvergenceController temporal_convergence_;
        std::atomic<std::uint64_t> temporal_camera_cut_generation_{0};
        std::uint64_t consumed_temporal_camera_cut_generation_ = 0;
        bool scene_reconstruction_request_logged_ = false;
        std::string last_scene_reconstruction_backend_;
        std::string last_scene_reconstruction_preset_;
        RenderTargetId main_render_target_{};
        RenderTargetId split_left_render_target_{};
        RenderTargetId split_right_render_target_{};
        VkImage vulkan_external_viewport_image_ = VK_NULL_HANDLE;
        VkImageView vulkan_external_viewport_image_view_ = VK_NULL_HANDLE;
        VkImageLayout vulkan_external_viewport_image_layout_ =
            VK_IMAGE_LAYOUT_UNDEFINED;
        std::uint64_t vulkan_external_viewport_image_generation_ = 0;
        std::uint64_t split_view_image_generation_ = 0;
        std::uint64_t split_left_image_generation_ = 0;
        std::uint64_t split_right_image_generation_ = 0;
        const lfs::core::Tensor* split_left_source_ = nullptr;
        glm::ivec2 split_left_source_size_{0, 0};
        int split_left_source_camera_uid_ = -1;
        bool split_left_source_undistorted_ = false;
        glm::ivec2 split_right_source_size_{0, 0};
        glm::ivec2 vulkan_viewport_image_size_{0, 0};
        glm::ivec2 vulkan_viewport_image_alloc_size_{0, 0};
        glm::ivec2 vulkan_viewport_coordinate_size_{0, 0};
        bool vulkan_viewport_image_flip_y_ = false;
        glm::ivec2 vulkan_gt_comparison_content_size_{0, 0};
        std::optional<GTPresentedView> vulkan_gt_comparison_selection_view_;
        GTComparisonActualSizeState gt_comparison_actual_size_state_;
        std::optional<GTComparisonActualFrameSnapshot> gt_comparison_published_actual_frame_;
        std::uint64_t gt_async_depth_ticket_ = 0;
        lfs::core::Tensor gt_async_depth_dest_{};
        GTComparisonMode gt_async_ticket_mode_ = GTComparisonMode::RGB;
        std::optional<lfs::rendering::CameraIntrinsics> gt_async_ticket_intrinsics_;
        bool gt_async_ticket_flip_y_ = false;
        lfs::rendering::FrameMetadata gt_async_ticket_metadata_{};
        std::optional<GTPresentedView> gt_async_ticket_view_;
        // Keep the displayed image and its camera together until the next ticket arrives.
        std::shared_ptr<lfs::core::Tensor> gt_async_held_display_;
        bool gt_async_held_flip_y_ = false;
        lfs::rendering::FrameMetadata gt_async_held_metadata_{};
        std::optional<GTPresentedView> gt_async_held_view_;
        std::atomic<uint32_t> dirty_mask_{DirtyFlag::ALL};
        RenderAnimationState animation_state_;
        FramebufferViewportRect framebuffer_viewport_rect_;
        ViewportArtifactService viewport_artifact_service_;
        ViewportInteropService viewport_interop_;
        SplitViewService split_view_service_;
        ViewportFrameLifecycleService frame_lifecycle_service_;
        int depth_window_preview_count_ = 0;
        std::optional<DepthWindowState> depth_window_drag_backup_;
        uint64_t depth_window_drag_owner_ = 0;
        uint64_t depth_window_last_drag_token_ = 0;
        uint64_t depth_window_projection_generation_ = 0;
        uint64_t depth_window_mode_epoch_ = 0;
        SceneUpscalerSelection scene_upscaler_runtime_selection_{};
        mutable std::mutex depth_window_transition_mutex_;
        mutable std::mutex vulkan_mesh_frame_mutex_;
        VulkanMeshFrame vulkan_mesh_frame_;
        ViewportInteractionContext viewport_interaction_context_;
    };
} // namespace lfs::vis
