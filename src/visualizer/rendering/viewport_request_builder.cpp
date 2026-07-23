/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "viewport_request_builder.hpp"
#include "scene/scene_manager.hpp"

namespace lfs::vis {

    namespace {
        [[nodiscard]] bool panelMatches(const std::optional<SplitViewPanelId> preview_panel,
                                        const std::optional<SplitViewPanelId> render_panel) {
            return !preview_panel || !render_panel || *preview_panel == *render_panel;
        }

        [[nodiscard]] const lfs::core::Scene::RenderableCropBox* findEnabledPointCloudCropBox(
            const std::vector<lfs::core::Scene::RenderableCropBox>& cropboxes,
            const core::NodeId node_id) {
            if (node_id == core::NULL_NODE) {
                return nullptr;
            }
            for (const auto& cb : cropboxes) {
                if (cb.node_id == node_id && cb.data && cb.data->enabled) {
                    return &cb;
                }
            }
            return nullptr;
        }
        [[nodiscard]] const lfs::core::Scene::RenderableCropBox* singleEnabledCropBox(
            const std::vector<lfs::core::Scene::RenderableCropBox>& cropboxes) {
            const lfs::core::Scene::RenderableCropBox* selected = nullptr;
            for (const auto& cb : cropboxes) {
                if (!cb.data || !cb.data->enabled || !cb.parent_effectively_visible) {
                    continue;
                }
                if (selected) {
                    return nullptr;
                }
                selected = &cb;
            }
            return selected;
        }

        [[nodiscard]] const lfs::core::Scene::RenderableEllipsoid* findEnabledPointCloudEllipsoid(
            const std::vector<lfs::core::Scene::RenderableEllipsoid>& ellipsoids,
            const core::NodeId node_id) {
            if (node_id == core::NULL_NODE) {
                return nullptr;
            }
            for (const auto& el : ellipsoids) {
                if (el.node_id == node_id && el.data && el.data->enabled) {
                    return &el;
                }
            }
            return nullptr;
        }
        [[nodiscard]] const lfs::core::Scene::RenderableEllipsoid* singleEnabledEllipsoid(
            const std::vector<lfs::core::Scene::RenderableEllipsoid>& ellipsoids) {
            const lfs::core::Scene::RenderableEllipsoid* selected = nullptr;
            for (const auto& el : ellipsoids) {
                if (!el.data || !el.data->enabled || !el.parent_effectively_visible) {
                    continue;
                }
                if (selected) {
                    return nullptr;
                }
                selected = &el;
            }
            return selected;
        }

        [[nodiscard]] const lfs::core::Scene::RenderableCropBox* activePointCloudCropBoxFilter(const FrameContext& ctx) {
            const auto& cropboxes = ctx.scene_state.cropboxes;
            const core::NodeId selected_cropbox_id =
                ctx.scene_manager ? ctx.scene_manager->getSelectedNodeCropBoxId() : core::NULL_NODE;
            if (const auto* const cb = findEnabledPointCloudCropBox(cropboxes, selected_cropbox_id)) {
                return cb;
            }

            const auto selected_idx = ctx.scene_state.selected_cropbox_index;
            if (selected_idx >= 0) {
                const size_t idx = static_cast<size_t>(selected_idx);
                if (idx < cropboxes.size() && cropboxes[idx].data && cropboxes[idx].data->enabled) {
                    return &cropboxes[idx];
                }
            }

            if (ctx.scene_manager && ctx.scene_manager->hasSelectedNode()) {
                return nullptr;
            }
            return singleEnabledCropBox(cropboxes);
        }
        [[nodiscard]] const lfs::core::Scene::RenderableEllipsoid* activePointCloudEllipsoidFilter(const FrameContext& ctx) {
            const auto& ellipsoids = ctx.scene_state.ellipsoids;
            const core::NodeId selected_ellipsoid_id =
                ctx.scene_manager ? ctx.scene_manager->getSelectedNodeEllipsoidId() : core::NULL_NODE;
            if (const auto* const el = findEnabledPointCloudEllipsoid(ellipsoids, selected_ellipsoid_id)) {
                return el;
            }

            if (ctx.scene_manager && ctx.scene_manager->hasSelectedNode()) {
                return nullptr;
            }
            return singleEnabledEllipsoid(ellipsoids);
        }
        void upsertScopedCropBox(std::vector<lfs::rendering::GaussianScopedBoxFilter>& filters,
                                 lfs::rendering::GaussianScopedBoxFilter filter) {
            if (filter.parent_node_index < 0) {
                return;
            }
            for (auto& existing : filters) {
                if (existing.parent_node_index == filter.parent_node_index) {
                    existing = std::move(filter);
                    return;
                }
            }
            filters.push_back(std::move(filter));
        }

        void applyGaussianCropBox(lfs::rendering::GaussianFilterState& filters, const FrameContext& ctx) {
            for (const auto& cb : ctx.scene_state.cropboxes) {
                if (!cb.data || !cb.data->enabled || cb.parent_node_index < 0) {
                    continue;
                }
                filters.crop_regions.push_back(lfs::rendering::GaussianScopedBoxFilter{
                    .bounds =
                        {.min = cb.data->min,
                         .max = cb.data->max,
                         .transform = glm::inverse(cb.world_transform)},
                    .inverse = cb.data->inverse,
                    .desaturate = ctx.settings.desaturate_cropping,
                    .parent_node_index = cb.parent_node_index});
            }

            if (ctx.gizmo.cropbox_active && ctx.gizmo.cropbox_affects_render &&
                ctx.gizmo.cropbox_parent_node_index >= 0) {
                upsertScopedCropBox(filters.crop_regions,
                                    lfs::rendering::GaussianScopedBoxFilter{
                                        .bounds =
                                            {.min = ctx.gizmo.cropbox_min,
                                             .max = ctx.gizmo.cropbox_max,
                                             .transform = glm::inverse(ctx.gizmo.cropbox_transform)},
                                        .inverse = false,
                                        .desaturate = ctx.settings.desaturate_cropping,
                                        .parent_node_index = ctx.gizmo.cropbox_parent_node_index});
            }

            if (!filters.crop_regions.empty()) {
                filters.crop_region = filters.crop_regions.front();
            }
        }
        void applyPointCloudCropVolume(lfs::rendering::PointCloudFilterState& filters, const FrameContext& ctx) {
            if (ctx.gizmo.cropbox_active && ctx.gizmo.cropbox_affects_render) {
                filters.crop_box = lfs::rendering::BoundingBox{
                    .min = ctx.gizmo.cropbox_min,
                    .max = ctx.gizmo.cropbox_max,
                    .transform = glm::inverse(ctx.gizmo.cropbox_transform)};
                filters.crop_ellipsoid.reset();
                filters.crop_inverse = false;
                filters.crop_desaturate = ctx.settings.desaturate_cropping;
                return;
            }

            if (ctx.gizmo.ellipsoid_active && ctx.gizmo.ellipsoid_affects_render) {
                filters.crop_ellipsoid = lfs::rendering::Ellipsoid{
                    .radii = ctx.gizmo.ellipsoid_radii,
                    .transform = glm::inverse(ctx.gizmo.ellipsoid_transform)};
                filters.crop_box.reset();
                filters.crop_inverse = false;
                filters.crop_desaturate = ctx.settings.desaturate_cropping;
                return;
            }

            if (const auto* const cb = activePointCloudCropBoxFilter(ctx)) {
                filters.crop_box = lfs::rendering::BoundingBox{
                    .min = cb->data->min,
                    .max = cb->data->max,
                    .transform = glm::inverse(cb->world_transform)};
                filters.crop_ellipsoid.reset();
                filters.crop_inverse = cb->data->inverse;
                filters.crop_desaturate = ctx.settings.desaturate_cropping;
                return;
            }

            const auto* const el = activePointCloudEllipsoidFilter(ctx);
            if (!el) {
                return;
            }

            filters.crop_ellipsoid = lfs::rendering::Ellipsoid{
                .radii = el->data->radii,
                .transform = glm::inverse(el->world_transform)};
            filters.crop_box.reset();
            filters.crop_inverse = el->data->inverse;
            filters.crop_desaturate = ctx.settings.desaturate_cropping;
        }

        void upsertScopedEllipsoid(std::vector<lfs::rendering::GaussianScopedEllipsoidFilter>& filters,
                                   lfs::rendering::GaussianScopedEllipsoidFilter filter) {
            if (filter.parent_node_index < 0) {
                return;
            }
            for (auto& existing : filters) {
                if (existing.parent_node_index == filter.parent_node_index) {
                    existing = std::move(filter);
                    return;
                }
            }
            filters.push_back(std::move(filter));
        }

        void applyGaussianEllipsoid(lfs::rendering::GaussianFilterState& filters, const FrameContext& ctx) {
            for (const auto& el : ctx.scene_state.ellipsoids) {
                if (!el.data || !el.data->enabled || el.parent_node_index < 0) {
                    continue;
                }
                filters.ellipsoid_regions.push_back(lfs::rendering::GaussianScopedEllipsoidFilter{
                    .bounds =
                        {.radii = el.data->radii,
                         .transform = glm::inverse(el.world_transform)},
                    .inverse = el.data->inverse,
                    .desaturate = ctx.settings.desaturate_cropping,
                    .parent_node_index = el.parent_node_index});
            }

            if (ctx.gizmo.ellipsoid_active && ctx.gizmo.ellipsoid_affects_render &&
                ctx.gizmo.ellipsoid_parent_node_index >= 0) {
                upsertScopedEllipsoid(filters.ellipsoid_regions,
                                      lfs::rendering::GaussianScopedEllipsoidFilter{
                                          .bounds =
                                              {.radii = ctx.gizmo.ellipsoid_radii,
                                               .transform = glm::inverse(ctx.gizmo.ellipsoid_transform)},
                                          .inverse = false,
                                          .desaturate = ctx.settings.desaturate_cropping,
                                          .parent_node_index = ctx.gizmo.ellipsoid_parent_node_index});
            }

            if (!filters.ellipsoid_regions.empty()) {
                filters.ellipsoid_region = filters.ellipsoid_regions.front();
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
            filters.cull_outside_view_volume = ctx.settings.hide_outside_depth_box;
        }

        void populateSelectionColors(
            std::array<glm::vec4, lfs::rendering::kSelectionColorTableCount>& colors,
            const FrameContext& ctx) {
            colors[0] = glm::vec4(ctx.settings.selection_color_center_marker, 1.0f);
            colors[lfs::rendering::kSelectionPreviewColorIndex] =
                glm::vec4(ctx.settings.selection_color_preview, 1.0f);
            constexpr float kSelectedHoverRedBias = 0.65f;
            const glm::vec3 selected_hover_color =
                ctx.settings.selection_color_committed * (1.0f - kSelectedHoverRedBias) +
                glm::vec3(1.0f, 0.02f, 0.02f) * kSelectedHoverRedBias;
            colors[lfs::rendering::kSelectionSelectedHoverColorIndex] =
                glm::vec4(selected_hover_color, 1.0f);
            if (ctx.scene_manager) {
                for (const auto& group : ctx.scene_manager->getScene().getSelectionGroups()) {
                    const auto index = static_cast<std::size_t>(group.id);
                    if (index < lfs::rendering::kSelectionGroupColorCount) {
                        colors[index] = glm::vec4(group.color, 1.0f);
                    }
                }
            } else {
                colors[1] = glm::vec4(ctx.settings.selection_color_committed, 1.0f);
            }
        }

    } // namespace

    lfs::rendering::ViewportRenderRequest buildViewportRenderRequest(const FrameContext& ctx,
                                                                     const glm::ivec2 render_size,
                                                                     const Viewport* const source_viewport,
                                                                     const std::optional<SplitViewPanelId> render_panel) {
        const Viewport& viewport = source_viewport ? *source_viewport : ctx.viewport;
        const auto frame_view = ctx.makeFrameView(viewport, render_size);
        const bool selection_overlay_enabled = !ctx.training_active;
        const bool overlay_visible =
            selection_overlay_enabled && panelMatches(ctx.cursor_preview.panel, render_panel);
        const bool ring_selection_mode = ctx.cursor_preview.selection_mode == SelectionPreviewMode::Rings;

        lfs::rendering::ViewportRenderRequest request{
            .frame_view = frame_view,
            .scaling_modifier = ctx.settings.scaling_modifier,
            .antialiasing = ctx.settings.antialiasing,
            .mip_filter = ctx.settings.mip_filter,
            .sh_degree = ctx.settings.sh_degree,
            .raster_backend = ctx.settings.raster_backend,
            .gut = ctx.settings.gut ||
                   lfs::rendering::isGutBackend(ctx.settings.raster_backend),
            .equirectangular = ctx.settings.equirectangular,
            .scene =
                {.model_transforms = &ctx.scene_state.model_transforms,
                 .transform_indices = ctx.scene_state.transform_indices,
                 .node_visibility_mask = ctx.scene_state.node_visibility_mask},
            .filters = {},
            .overlay =
                {.markers =
                     {.show_rings = ctx.settings.show_rings || ring_selection_mode,
                      .ring_width = ctx.settings.ring_width,
                      .show_center_markers = ctx.settings.show_center_markers},
                 .cursor =
                     {.enabled = ctx.cursor_preview.active && overlay_visible,
                      .cursor = {ctx.cursor_preview.x, ctx.cursor_preview.y},
                      .radius = ctx.cursor_preview.radius,
                      .saturation_preview = ctx.cursor_preview.saturation_mode,
                      .saturation_amount = ctx.cursor_preview.saturation_amount},
                 .emphasis =
                     {.mask = selection_overlay_enabled ? ctx.scene_state.selection_mask : nullptr,
                      .transient_mask =
                          {.mask = selection_overlay_enabled
                                       ? (ctx.cursor_preview.preview_selection ? ctx.cursor_preview.preview_selection
                                                                               : ctx.cursor_preview.selection_tensor)
                                       : nullptr,
                           .additive = selection_overlay_enabled && ctx.cursor_preview.add_mode},
                      .emphasized_node_mask = (selection_overlay_enabled &&
                                                       (ctx.settings.desaturate_unselected ||
                                                        ctx.selection_flash_intensity > 0.0f)
                                                   ? ctx.scene_state.selected_node_mask
                                                   : std::vector<bool>{}),
                      .dim_non_emphasized = selection_overlay_enabled && ctx.settings.desaturate_unselected,
                      .flash_intensity = selection_overlay_enabled ? ctx.selection_flash_intensity : 0.0f,
                      .focused_gaussian_id = (selection_overlay_enabled && ring_selection_mode && overlay_visible)
                                                 ? ctx.cursor_preview.focused_gaussian_id
                                                 : -1}},
            .transparent_background = environmentBackgroundUsesTransparentViewerCompositing(ctx.settings),
            .depth_view = ctx.settings.depth_view,
            .depth_view_min = ctx.settings.depth_view_min,
            .depth_view_max = ctx.settings.depth_view_max,
            .depth_visualization_mode = ctx.settings.depth_visualization_mode};

        if (selection_overlay_enabled ||
            request.overlay.markers.show_rings ||
            request.overlay.markers.show_center_markers) {
            populateSelectionColors(request.overlay.selection_colors, ctx);
        }

        applyGaussianCropBox(request.filters, ctx);
        applyGaussianEllipsoid(request.filters, ctx);
        applyGaussianViewVolume(request.filters, ctx);
        return request;
    }

    lfs::rendering::SplitViewGaussianPanelRenderState buildSplitViewGaussianPanelRenderState(
        const FrameContext& ctx, const glm::ivec2 render_size,
        const Viewport* const source_viewport,
        const std::optional<SplitViewPanelId> render_panel) {
        const auto request = buildViewportRenderRequest(ctx, render_size, source_viewport, render_panel);
        return lfs::rendering::SplitViewGaussianPanelRenderState{
            .frame_view = request.frame_view,
            .scaling_modifier = request.scaling_modifier,
            .antialiasing = request.antialiasing,
            .mip_filter = request.mip_filter,
            .sh_degree = request.sh_degree,
            .raster_backend = request.raster_backend,
            .gut = request.gut,
            .equirectangular = request.equirectangular,
            .scene = request.scene,
            .filters = request.filters,
            .overlay = request.overlay};
    }

    lfs::rendering::SplitViewPointCloudPanelRenderState buildSplitViewPointCloudPanelRenderState(
        const FrameContext& ctx, const glm::ivec2 render_size, const Viewport* const source_viewport) {
        const Viewport& viewport = source_viewport ? *source_viewport : ctx.viewport;
        const auto frame_view = ctx.makeFrameView(viewport, render_size);

        lfs::rendering::SplitViewPointCloudPanelRenderState state{
            .frame_view = frame_view,
            .render =
                {.scaling_modifier = ctx.settings.scaling_modifier,
                 .voxel_size = ctx.settings.voxel_size,
                 .equirectangular = ctx.settings.equirectangular},
            .scene =
                {.model_transforms = &ctx.scene_state.model_transforms,
                 .transform_indices = ctx.scene_state.transform_indices,
                 .node_visibility_mask = ctx.scene_state.node_visibility_mask},
            .filters = {},
            .overlay = {}};
        if (!ctx.training_active) {
            state.overlay.selection_mask = ctx.scene_state.selection_mask;
            state.overlay.transient_mask.mask = ctx.cursor_preview.preview_selection
                                                    ? ctx.cursor_preview.preview_selection
                                                    : ctx.cursor_preview.selection_tensor;
            state.overlay.transient_mask.additive = ctx.cursor_preview.add_mode;
            populateSelectionColors(state.overlay.selection_colors, ctx);
        }
        applyPointCloudCropVolume(state.filters, ctx);
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
                 .transform_indices = ctx.scene_state.transform_indices,
                 .node_visibility_mask = ctx.scene_state.node_visibility_mask},
            .filters = {},
            .overlay = {},
            .transparent_background = environmentBackgroundUsesTransparentViewerCompositing(ctx.settings)};

        if (!ctx.training_active) {
            request.overlay.selection_mask = ctx.scene_state.selection_mask;
            request.overlay.transient_mask.mask = ctx.cursor_preview.preview_selection
                                                      ? ctx.cursor_preview.preview_selection
                                                      : ctx.cursor_preview.selection_tensor;
            request.overlay.transient_mask.additive = ctx.cursor_preview.add_mode;
            populateSelectionColors(request.overlay.selection_colors, ctx);
        }

        applyPointCloudCropVolume(request.filters, ctx);
        return request;
    }

} // namespace lfs::vis
