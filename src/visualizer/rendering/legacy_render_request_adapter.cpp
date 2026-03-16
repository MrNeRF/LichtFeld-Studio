/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "legacy_render_request_adapter.hpp"
#include "scene/scene_manager.hpp"

namespace lfs::vis {

    namespace {

        void applyViewportCropBox(lfs::rendering::ViewportRenderRequest& request, const FrameContext& ctx) {
            if (!ctx.scene_manager || !(ctx.settings.use_crop_box || ctx.settings.show_crop_box)) {
                return;
            }

            const auto& cropboxes = ctx.scene_state.cropboxes;
            const size_t idx = (ctx.scene_state.selected_cropbox_index >= 0)
                                   ? static_cast<size_t>(ctx.scene_state.selected_cropbox_index)
                                   : 0;

            if (idx >= cropboxes.size() || !cropboxes[idx].data) {
                return;
            }

            const auto& cb = cropboxes[idx];
            request.crop_box = lfs::rendering::BoundingBox{
                .min = cb.data->min,
                .max = cb.data->max,
                .transform = glm::inverse(cb.world_transform)};
            request.crop_inverse = cb.data->inverse;
            request.crop_desaturate =
                ctx.settings.show_crop_box && !ctx.settings.use_crop_box && ctx.settings.desaturate_cropping;
            request.crop_parent_node_index =
                ctx.scene_manager->getScene().getVisibleNodeIndex(cb.parent_splat_id);
        }

        void applyPointCloudCropBox(lfs::rendering::PointCloudRenderRequest& request, const FrameContext& ctx) {
            if (!ctx.scene_manager || !(ctx.settings.use_crop_box || ctx.settings.show_crop_box)) {
                return;
            }

            const auto& cropboxes = ctx.scene_state.cropboxes;
            const size_t idx = (ctx.scene_state.selected_cropbox_index >= 0)
                                   ? static_cast<size_t>(ctx.scene_state.selected_cropbox_index)
                                   : 0;

            if (idx >= cropboxes.size() || !cropboxes[idx].data) {
                return;
            }

            const auto& cb = cropboxes[idx];
            request.crop_box = lfs::rendering::BoundingBox{
                .min = cb.data->min,
                .max = cb.data->max,
                .transform = glm::inverse(cb.world_transform)};
            request.crop_inverse = cb.data->inverse;
            request.crop_desaturate =
                ctx.settings.show_crop_box && !ctx.settings.use_crop_box && ctx.settings.desaturate_cropping;
        }

        void applyViewportEllipsoid(lfs::rendering::ViewportRenderRequest& request, const FrameContext& ctx) {
            if (!ctx.scene_manager || !(ctx.settings.use_ellipsoid || ctx.settings.show_ellipsoid)) {
                return;
            }

            const auto& scene = ctx.scene_manager->getScene();
            const auto visible_ellipsoids = scene.getVisibleEllipsoids();
            const core::NodeId selected_ellipsoid_id = ctx.scene_manager->getSelectedNodeEllipsoidId();
            for (const auto& el : visible_ellipsoids) {
                if (!el.data) {
                    continue;
                }
                if (selected_ellipsoid_id != core::NULL_NODE && el.node_id != selected_ellipsoid_id) {
                    continue;
                }
                request.ellipsoid = lfs::rendering::Ellipsoid{
                    .radii = el.data->radii,
                    .transform = glm::inverse(el.world_transform)};
                request.ellipsoid_inverse = el.data->inverse;
                request.ellipsoid_desaturate = ctx.settings.show_ellipsoid &&
                                               !ctx.settings.use_ellipsoid &&
                                               ctx.settings.desaturate_cropping;
                request.ellipsoid_parent_node_index = scene.getVisibleNodeIndex(el.parent_splat_id);
                return;
            }
        }

        void applyViewportDepthFilter(lfs::rendering::ViewportRenderRequest& request, const FrameContext& ctx) {
            if (!ctx.settings.depth_filter_enabled) {
                return;
            }

            request.depth_filter = lfs::rendering::BoundingBox{
                .min = ctx.settings.depth_filter_min,
                .max = ctx.settings.depth_filter_max,
                .transform = ctx.settings.depth_filter_transform.inv().toMat4()};
        }

    } // namespace

    lfs::rendering::ViewportRenderRequest buildViewportRenderRequest(const FrameContext& ctx,
                                                                     const glm::ivec2 render_size) {
        auto frame_view = ctx.makeFrameView();
        frame_view.size = render_size;

        lfs::rendering::ViewportRenderRequest request{
            .frame_view = frame_view,
            .scaling_modifier = ctx.settings.scaling_modifier,
            .antialiasing = ctx.settings.antialiasing,
            .mip_filter = ctx.settings.mip_filter,
            .sh_degree = ctx.settings.sh_degree,
            .point_cloud_mode = ctx.settings.point_cloud_mode,
            .voxel_size = ctx.settings.voxel_size,
            .gut = ctx.settings.gut,
            .equirectangular = ctx.settings.equirectangular,
            .show_rings = ctx.settings.show_rings,
            .ring_width = ctx.settings.ring_width,
            .show_center_markers = ctx.settings.show_center_markers,
            .model_transforms = &ctx.scene_state.model_transforms,
            .transform_indices = ctx.scene_state.transform_indices,
            .selection_mask = ctx.scene_state.selection_mask,
            .output_screen_positions = ctx.brush.output_screen_positions,
            .brush =
                {.active = ctx.brush.active,
                 .cursor = {ctx.brush.x, ctx.brush.y},
                 .radius = ctx.brush.radius,
                 .add_mode = ctx.brush.add_mode,
                 .selection_tensor = ctx.brush.preview_selection ? ctx.brush.preview_selection
                                                                 : ctx.brush.selection_tensor,
                 .saturation_mode = ctx.brush.saturation_mode,
                 .saturation_amount = ctx.brush.saturation_amount,
                 .selection_mode_rings =
                     (ctx.brush.selection_mode == lfs::rendering::SelectionMode::Rings)},
            .crop_box = std::nullopt,
            .ellipsoid = std::nullopt,
            .depth_filter = std::nullopt,
            .selected_node_mask = (ctx.settings.desaturate_unselected ||
                                   ctx.selection_flash_intensity > 0.0f)
                                      ? ctx.scene_state.selected_node_mask
                                      : std::vector<bool>{},
            .node_visibility_mask = ctx.scene_state.node_visibility_mask,
            .desaturate_unselected = ctx.settings.desaturate_unselected,
            .selection_flash_intensity = ctx.selection_flash_intensity,
            .hovered_depth_id = nullptr,
            .highlight_gaussian_id = (ctx.brush.selection_mode == lfs::rendering::SelectionMode::Rings)
                                         ? ctx.hovered_gaussian_id
                                         : -1};

        applyViewportCropBox(request, ctx);
        applyViewportEllipsoid(request, ctx);
        applyViewportDepthFilter(request, ctx);
        return request;
    }

    lfs::rendering::PointCloudRenderRequest buildPointCloudRenderRequest(
        const FrameContext& ctx, const std::vector<glm::mat4>& model_transforms) {
        const auto frame_view = ctx.makeFrameView();

        lfs::rendering::PointCloudRenderRequest request{
            .frame_view = frame_view,
            .scaling_modifier = ctx.settings.scaling_modifier,
            .voxel_size = ctx.settings.voxel_size,
            .equirectangular = ctx.settings.equirectangular,
            .model_transforms = &model_transforms,
            .crop_box = std::nullopt};

        applyPointCloudCropBox(request, ctx);
        return request;
    }

} // namespace lfs::vis
