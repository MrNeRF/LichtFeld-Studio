/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "camera_interaction_service.hpp"
#include "core/tensor.hpp"
#include "core/tensor_image.hpp"
#include "depth_window_state.hpp"
#include "dirty_flags.hpp"
#include "framerate_controller.hpp"
#include "internal/viewport.hpp"
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
#include "viewport_draw_types.hpp"
#include "viewport_frame_lifecycle_service.hpp"
#include "viewport_interaction_context.hpp"
#include "viewport_overlay_service.hpp"
#include "viewport_reference_state.hpp"
#include "viewport_frame_desc.hpp"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace lfs::vis {
    class TensorSceneTemporalPipeline;
    class MetalSceneUpscaler;
    struct FramebufferViewportRect {
        glm::ivec2 top_left{0, 0};
        glm::ivec2 size{0, 0};

        [[nodiscard]] bool valid() const { return size.x > 0 && size.y > 0; }
    };
    struct GTPresentedView {
        GTRenderCamera camera;
        glm::ivec2 size{0, 0};
    };
    struct ViewRenderState {
        [[nodiscard]] std::uint64_t presentedImageGeneration() const {
            if (const auto generation = referenceSceneOutputGeneration(*this))
                return *generation;
            return vulkan_viewport_image_generation_;
        }
        ViewportOverlayService viewport_overlay_service_;
        ViewId id = kNoView;
        glm::ivec2 last_nonzero_viewport_size_{0, 0};
        glm::ivec2 requested_viewport_size_{0, 0};
        std::chrono::steady_clock::time_point last_visible{};
        std::optional<RenderSettings> rendered_settings;
        lfs::rendering::ScreenOverlayRenderer screen_overlay_renderer_;
        mutable FramerateController framerate_controller_;
        std::shared_ptr<const lfs::core::Tensor> vulkan_viewport_image_;
        // Linear view depth and environment background matching
        // vulkan_viewport_image_ (tensor compositor).
        std::shared_ptr<const lfs::core::Tensor> viewport_depth_image_;
        ViewportEnvironment viewport_environment_;
        ViewportMeshPassDesc viewport_meshes_;
        ViewportSplitView split_view_;
        std::uint64_t vulkan_viewport_image_generation_ = 0;
        std::string last_logged_vksplat_render_error_;
        StaleFrameGuard vksplat_stale_frame_guard_;
        DirtyMask parked_arena_retry_ = 0;
        // Over-budget navigation: render when the camera has rested.
        bool camera_settle_pending_ = false;
        glm::mat3 last_navigation_rotation_{1.0f};
        glm::vec3 last_navigation_translation_{0.0f};
        bool navigation_pose_valid_ = false;
        std::chrono::steady_clock::time_point camera_settle_deadline_{};
        bool temporal_settle_pending_ = false;
        std::chrono::steady_clock::time_point temporal_settle_deadline_{};
        std::atomic<DirtyMask> training_refresh_dirty_{0};
        std::atomic<double> training_preview_turn_ms_{0.0};
        int last_training_preview_iteration_ = -1;
        bool has_training_preview_iteration_ = false;
        std::uint64_t last_rendered_input_fingerprint_ = 0;
        bool has_rendered_input_fingerprint_ = false;
        std::uint64_t viewport_projection_generation_ = 1;
        std::uint64_t temporal_scene_revision_ = 1;
        TemporalConvergenceController temporal_convergence_;
        std::shared_ptr<TensorSceneTemporalPipeline> tensor_temporal_pipeline_;
        std::shared_ptr<MetalSceneUpscaler> metal_scene_upscaler_;
        std::atomic<std::uint64_t> temporal_camera_cut_generation_{0};
        std::uint64_t consumed_temporal_camera_cut_generation_ = 0;
        bool scene_reconstruction_request_logged_ = false;
        std::string last_scene_reconstruction_backend_;
        std::string last_scene_reconstruction_preset_;
        RenderTargetId main_render_target_{};
        RenderTargetId split_left_render_target_{};
        RenderTargetId split_right_render_target_{};
        std::shared_ptr<ViewportReferenceState> reference_state_;
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
        SplitViewService split_view_service_;
        ViewportFrameLifecycleService frame_lifecycle_service_;
        int depth_window_preview_count_ = 0;
        std::optional<DepthWindowState> depth_window_drag_backup_;
        uint64_t depth_window_drag_owner_ = 0;
        uint64_t depth_window_last_drag_token_ = 0;
        uint64_t depth_window_projection_generation_ = 0;
        uint64_t depth_window_mode_epoch_ = 0;
        SceneUpscalerSelection scene_upscaler_runtime_selection_{};
        // The requested reconstruction cannot serve the current view mode.
        bool scene_upscaler_mode_unsupported_ = false;
        bool scene_upscaler_runtime_failed_ = false;
        std::string scene_upscaler_runtime_config_;
        mutable std::mutex depth_window_transition_mutex_;
        ViewportInteractionContext viewport_interaction_context_;
    };
} // namespace lfs::vis
