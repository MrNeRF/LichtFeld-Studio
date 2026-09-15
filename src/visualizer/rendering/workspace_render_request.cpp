/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "workspace_render_request.hpp"
#include "temporal_frame_tracker.hpp"
#include "viewport_request_builder.hpp"

#include <stdexcept>

namespace lfs::vis {

    lfs::rendering::FrameView makeWorkspaceFrameView(
        const PaneSnapshot& pane, const glm::ivec2 extent, const glm::vec3 background_color) {
        if (pane.id == kInvalidViewId || extent.x <= 0 || extent.y <= 0) {
            throw std::invalid_argument("Workspace render requires a live view and positive extent");
        }
        std::optional<lfs::rendering::CameraIntrinsics> containment;
        if (!pane.projection.orthographic) {
            const auto [fx, fy] =
                lfs::rendering::computePixelFocalLengths(extent, pane.projection.focal_length_mm);
            containment = lfs::rendering::CameraIntrinsics{
                .focal_x = fx,
                .focal_y = fy,
                .center_x = 0.5f * static_cast<float>(extent.x),
                .center_y = 0.5f * static_cast<float>(extent.y)};
        }
        // Orthographic zoom is measured in logical viewport pixels per world
        // unit. Keep its field of view when raster resolution changes during
        // resize, upscaling, or display scaling.
        const float ortho_scale = pane.projection.ortho_scale *
                                  (pane.window_size.y > 0
                                       ? static_cast<float>(extent.y) / pane.window_size.y
                                       : 1.0f);
        return {.rotation = pane.rotation,
                .translation = pane.translation,
                .size = extent,
                .focal_length_mm = pane.projection.focal_length_mm,
                .intrinsics_override = std::nullopt,
                .containment_intrinsics = containment,
                .near_plane = pane.projection.near_plane,
                .far_plane = pane.projection.far_plane,
                .orthographic = pane.projection.orthographic,
                .ortho_scale = ortho_scale,
                .background_color = background_color};
    }

    RenderSettings workspaceRenderSettings(const RenderSettings& shared_settings,
                                           const PaneSnapshot& pane) {
        RenderSettings settings = shared_settings;
        settings.split_view_mode = SplitViewMode::Disabled;
        settings.focal_length_mm = pane.projection.focal_length_mm;
        settings.orthographic = pane.projection.orthographic;
        settings.ortho_scale = pane.projection.ortho_scale;
        settings.equirectangular = pane.projection.equirectangular;
        settings.depth_clip_enabled = pane.projection.far_plane != lfs::rendering::DEFAULT_FAR_PLANE;
        settings.depth_clip_far = pane.projection.far_plane;
        settings.depth_filter_min.z = -pane.depth.far_plane;
        settings.depth_filter_max.z = -pane.depth.near_plane;
        settings.depth_filter_scale_x = pane.depth.scale_x;
        settings.depth_filter_scale_y = pane.depth.scale_y;
        settings.depth_filter_offset_x = pane.depth.offset_x;
        settings.depth_filter_offset_y = pane.depth.offset_y;
        settings.grid_plane = pane.grid_plane;
        return settings;
    }

    lfs::rendering::ViewportRenderRequest buildWorkspaceRenderRequest(
        const FrameContext& shared_scene, const PaneSnapshot& pane,
        const glm::ivec2 render_extent, const glm::vec2 jitter_pixels,
        const std::optional<ViewId> interaction_view) {
        const auto frame_view =
            makeWorkspaceFrameView(pane, render_extent, shared_scene.settings.background_color);
        Viewport camera(render_extent.x, render_extent.y);
        camera.setViewMatrix(pane.rotation, pane.translation);
        camera.camera.setPivot(pane.pivot);
        camera.ortho_scale_override = frame_view.ortho_scale;
        camera.frameBufferSize = render_extent;
        auto settings = workspaceRenderSettings(shared_scene.settings, pane);
        settings.ortho_scale = frame_view.ortho_scale;

        const bool owns_interaction = interaction_view == pane.id;
        auto cursor = owns_interaction ? shared_scene.cursor_preview : CursorPreviewState{};
        // Legacy comparison panel numbers do not describe workspace views.
        cursor.panel.reset();
        const FrameContext pane_context{
            .viewport = camera,
            .render_lock_held = shared_scene.render_lock_held,
            .scene_manager = shared_scene.scene_manager,
            .model = shared_scene.model,
            .scene_state = shared_scene.scene_state,
            .settings = std::move(settings),
            .render_size = render_extent,
            .viewport_pos = {pane.rect.x, pane.rect.y},
            .frame_dirty = shared_scene.frame_dirty,
            .training_active = shared_scene.training_active,
            .depth_window_drag_preview = owns_interaction && shared_scene.depth_window_drag_preview,
            .cursor_preview = cursor,
            .gizmo = shared_scene.gizmo,
            .hovered_camera_id = owns_interaction ? shared_scene.hovered_camera_id : -1,
            .current_camera_id = owns_interaction ? shared_scene.current_camera_id : -1,
            .hovered_gaussian_id = owns_interaction ? shared_scene.hovered_gaussian_id : -1,
            .selection_flash_intensity = shared_scene.selection_flash_intensity,
            .view_panels = {}};

        auto request = buildViewportRenderRequest(pane_context, render_extent);
        // The request borrows transforms until its sequential render completes.
        // Never return the address of the temporary pane_context's vector.
        request.scene.model_transforms = &shared_scene.scene_state.model_transforms;
        request.frame_view = applySceneViewJitter(frame_view, jitter_pixels);
        request.raster_backend =
            lfs::rendering::normalizeViewerRasterBackend(request.raster_backend, request.gut);
        request.gut = lfs::rendering::isGutBackend(request.raster_backend);
        return request;
    }

} // namespace lfs::vis
