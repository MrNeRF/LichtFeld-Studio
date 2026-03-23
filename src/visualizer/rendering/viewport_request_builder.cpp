/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "viewport_request_builder.hpp"
#include "scene/scene_manager.hpp"

namespace lfs::vis {

    namespace {

        void applyGaussianCropBox(lfs::rendering::GaussianFilterState& filters, const FrameContext& ctx) {
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
            filters.crop_region = lfs::rendering::GaussianScopedBoxFilter{
                .bounds =
                    {.min = cb.data->min,
                     .max = cb.data->max,
                     .transform = glm::inverse(cb.world_transform)},
                .inverse = cb.data->inverse,
                .desaturate =
                    ctx.settings.show_crop_box && !ctx.settings.use_crop_box && ctx.settings.desaturate_cropping,
                .parent_node_index = ctx.scene_manager->getScene().getVisibleNodeIndex(cb.parent_splat_id)};
        }

        void applyPointCloudCropBox(lfs::rendering::PointCloudFilterState& filters, const FrameContext& ctx) {
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
            filters.crop_box = lfs::rendering::BoundingBox{
                .min = cb.data->min,
                .max = cb.data->max,
                .transform = glm::inverse(cb.world_transform)};
            filters.crop_inverse = cb.data->inverse;
            filters.crop_desaturate =
                ctx.settings.show_crop_box && !ctx.settings.use_crop_box && ctx.settings.desaturate_cropping;
        }

        void applyGaussianEllipsoid(lfs::rendering::GaussianFilterState& filters, const FrameContext& ctx) {
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
                filters.ellipsoid_region = lfs::rendering::GaussianScopedEllipsoidFilter{
                    .bounds =
                        {.radii = el.data->radii,
                         .transform = glm::inverse(el.world_transform)},
                    .inverse = el.data->inverse,
                    .desaturate = ctx.settings.show_ellipsoid &&
                                  !ctx.settings.use_ellipsoid &&
                                  ctx.settings.desaturate_cropping,
                    .parent_node_index = scene.getVisibleNodeIndex(el.parent_splat_id)};
                return;
            }
        }

        void applyGaussianViewVolume(lfs::rendering::GaussianFilterState& filters, const FrameContext& ctx) {
            if (!ctx.settings.depth_filter_enabled) {
                return;
            }

            filters.view_volume = lfs::rendering::BoundingBox{
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
            .gut = ctx.settings.gut,
            .equirectangular = ctx.settings.equirectangular,
            .scene =
                {.model_transforms = &ctx.scene_state.model_transforms,
                 .transform_indices = ctx.scene_state.transform_indices,
                 .node_visibility_mask = ctx.scene_state.node_visibility_mask},
            .filters = {},
            .overlay =
                {.markers =
                     {.show_rings = ctx.settings.show_rings,
                      .ring_width = ctx.settings.ring_width,
                      .show_center_markers = ctx.settings.show_center_markers},
                 .cursor =
                     {.enabled = ctx.cursor_preview.active,
                      .cursor = {ctx.cursor_preview.x, ctx.cursor_preview.y},
                      .radius = ctx.cursor_preview.radius,
                      .saturation_preview = ctx.cursor_preview.saturation_mode,
                      .saturation_amount = ctx.cursor_preview.saturation_amount},
                 .emphasis =
                     {.mask = ctx.scene_state.selection_mask,
                      .transient_mask =
                          {.mask = ctx.cursor_preview.preview_selection ? ctx.cursor_preview.preview_selection
                                                                        : ctx.cursor_preview.selection_tensor,
                           .additive = ctx.cursor_preview.add_mode},
                      .emphasized_node_mask = (ctx.settings.desaturate_unselected ||
                                               ctx.selection_flash_intensity > 0.0f)
                                                  ? ctx.scene_state.selected_node_mask
                                                  : std::vector<bool>{},
                      .dim_non_emphasized = ctx.settings.desaturate_unselected,
                      .flash_intensity = ctx.selection_flash_intensity,
                      .focused_gaussian_id =
                          (ctx.cursor_preview.selection_mode == SelectionPreviewMode::Rings)
                              ? ctx.hovered_gaussian_id
                              : -1}}};

        applyGaussianCropBox(request.filters, ctx);
        applyGaussianEllipsoid(request.filters, ctx);
        applyGaussianViewVolume(request.filters, ctx);
        return request;
    }

    lfs::rendering::HoveredGaussianQueryRequest buildHoveredGaussianQueryRequest(
        const FrameContext& ctx, const glm::ivec2 render_size) {
        auto frame_view = ctx.makeFrameView();
        frame_view.size = render_size;

        lfs::rendering::HoveredGaussianQueryRequest request{
            .frame_view = frame_view,
            .scaling_modifier = ctx.settings.scaling_modifier,
            .mip_filter = ctx.settings.mip_filter,
            .sh_degree = ctx.settings.sh_degree,
            .gut = ctx.settings.gut,
            .equirectangular = ctx.settings.equirectangular,
            .scene =
                {.model_transforms = &ctx.scene_state.model_transforms,
                 .transform_indices = ctx.scene_state.transform_indices,
                 .node_visibility_mask = ctx.scene_state.node_visibility_mask},
            .filters = {},
            .cursor = {ctx.cursor_preview.x, ctx.cursor_preview.y},
        };

        applyGaussianCropBox(request.filters, ctx);
        applyGaussianEllipsoid(request.filters, ctx);
        applyGaussianViewVolume(request.filters, ctx);
        return request;
    }

    lfs::rendering::SplitViewGaussianPanelRenderState buildSplitViewGaussianPanelRenderState(
        const FrameContext& ctx, const glm::ivec2 render_size) {
        const auto request = buildViewportRenderRequest(ctx, render_size);
        return lfs::rendering::SplitViewGaussianPanelRenderState{
            .frame_view = request.frame_view,
            .scaling_modifier = request.scaling_modifier,
            .antialiasing = request.antialiasing,
            .mip_filter = request.mip_filter,
            .sh_degree = request.sh_degree,
            .gut = request.gut,
            .equirectangular = request.equirectangular,
            .filters = request.filters,
            .overlay = request.overlay};
    }

    lfs::rendering::SplitViewPointCloudPanelRenderState buildSplitViewPointCloudPanelRenderState(
        const FrameContext& ctx, const glm::ivec2 render_size) {
        auto frame_view = ctx.makeFrameView();
        frame_view.size = render_size;

        lfs::rendering::SplitViewPointCloudPanelRenderState state{
            .frame_view = frame_view,
            .render =
                {.scaling_modifier = ctx.settings.scaling_modifier,
                 .voxel_size = ctx.settings.voxel_size,
                 .equirectangular = ctx.settings.equirectangular},
            .filters = {}};
        applyPointCloudCropBox(state.filters, ctx);
        return state;
    }

    lfs::rendering::PointCloudRenderRequest buildPointCloudRenderRequest(
        const FrameContext& ctx, const glm::ivec2 render_size, const std::vector<glm::mat4>& model_transforms) {
        auto frame_view = ctx.makeFrameView();
        frame_view.size = render_size;

        lfs::rendering::PointCloudRenderRequest request{
            .frame_view = frame_view,
            .render =
                {.scaling_modifier = ctx.settings.scaling_modifier,
                 .voxel_size = ctx.settings.voxel_size,
                 .equirectangular = ctx.settings.equirectangular},
            .scene =
                {.model_transforms = &model_transforms,
                 .transform_indices = ctx.scene_state.transform_indices},
            .filters = {}};

        applyPointCloudCropBox(request.filters, ctx);
        return request;
    }

} // namespace lfs::vis
