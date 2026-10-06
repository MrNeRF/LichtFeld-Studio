/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "viewport_request_builder.hpp"
#include "core/camera.hpp"
#include "frustum_depth_coverage.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "rendering/model_renderability.hpp"
#include "scene/scene_manager.hpp"
#include "temporal_frame_tracker.hpp"
#include <algorithm>
#include <array>
#include <type_traits>
#include <vector>

namespace lfs::vis {

    namespace {
        [[nodiscard]] std::vector<uint32_t> frustumDepthSampleMask(
            const FrameContext& ctx, const lfs::rendering::FrameView& view) {
            // Offscreen/custom projections retain full exact depth.
            if (!ctx.scene_manager || !ctx.viewport_region || view.orthographic || ctx.settings.equirectangular ||
                ctx.viewport_region->width <= 0 || ctx.viewport_region->height <= 0 ||
                view.subregion_full_size != glm::ivec2(0) || view.size.x <= 0 || view.size.y <= 0)
                return {};
            const auto& scene = ctx.scene_manager->getScene();
            const auto& cameras = scene.getVisibleCamerasCached();
            auto transforms = ctx.scene_state.camera_scene_transforms;
            if (transforms.size() != cameras.size()) {
                transforms = scene.getVisibleCameraSceneTransforms();
                if (transforms.size() != cameras.size())
                    transforms.assign(cameras.size(), glm::mat4(1));
                for (auto& transform : transforms)
                    transform = lfs::rendering::dataWorldTransformToVisualizerWorld(transform);
            }
            struct Cache {
                const lfs::core::Scene* scene = nullptr;
                uint64_t generation = 0, camera_generation = 0;
                float scale = 0;
                std::vector<std::weak_ptr<const lfs::core::Camera>> cameras;
                std::vector<glm::mat4> transforms;
                std::vector<std::array<glm::vec3, 5>> points;
                bool full = false;
            };
            // Camera extrinsics may be on the GPU. Rebuild only when the scene
            // or calibration changes, not every time the viewer camera moves.
            static thread_local Cache cache;
            const float scale = ctx.settings.camera_frustum_scale;
            if (cache.scene != &scene || cache.generation != scene.renderGeneration() ||
                cache.camera_generation != scene.cameraListGeneration() || cache.scale != scale ||
                cache.cameras.size() != cameras.size() ||
                !std::equal(cache.cameras.begin(), cache.cameras.end(), cameras.begin(),
                            [](const auto& cached, const auto& camera) { return cached.lock() == camera; }) ||
                cache.transforms != transforms) {
                cache = {.scene = &scene, .generation = scene.renderGeneration(), .camera_generation = scene.cameraListGeneration(), .scale = scale, .cameras = {}, .transforms = transforms, .points = {}};
                cache.cameras.assign(cameras.begin(), cameras.end());
                for (size_t i = 0; i < cameras.size(); ++i) {
                    if (!cameras[i] || scale <= 0)
                        continue;
                    const auto& camera = *cameras[i];
                    if (camera.camera_model_type() == lfs::core::CameraModelType::EQUIRECTANGULAR) {
                        cache.full = true;
                        break;
                    }
                    if (camera.camera_width() <= 0 || camera.camera_height() <= 0 || camera.FoVy() <= 0)
                        continue;
                    auto r = camera.R();
                    auto t = camera.T();
                    if (!r.is_valid() || !t.is_valid() || r.numel() < 9 || t.numel() < 3 ||
                        r.dtype() != lfs::core::DataType::Float32 || t.dtype() != lfs::core::DataType::Float32) {
                        cache.full = true;
                        break;
                    }
                    const auto host_r = r.cpu().contiguous();
                    const auto host_t = t.cpu().contiguous();
                    glm::mat4 world_to_camera(1);
                    for (int row = 0; row < 3; ++row) {
                        for (int col = 0; col < 3; ++col)
                            world_to_camera[col][row] = host_r.ptr<float>()[row * 3 + col];
                        world_to_camera[3][row] = host_t.ptr<float>()[row];
                    }
                    const auto model = transforms[i] * glm::inverse(world_to_camera) *
                                       lfs::rendering::DATA_TO_VISUALIZER_CAMERA_AXES_4;
                    const float h = std::tan(camera.FoVy() * 0.5f) * scale;
                    const float w = h * float(camera.camera_width()) / camera.camera_height();
                    std::array<glm::vec3, 5> points{
                        glm::vec3(0),
                        {-w, -h, -scale},
                        {w, -h, -scale},
                        {w, h, -scale},
                        {-w, h, -scale}};
                    for (auto& point : points)
                        point = glm::vec3(model * glm::vec4(point, 1));
                    cache.points.push_back(points);
                }
            }
            if (cache.full)
                return {};
            // Visible line radius is 0.75 + 1.0 AA pixels; four render pixels
            // also cover bilinear depth neighbours and projection rounding.
            const float pixel_scale = std::max({1.0f, ctx.settings.render_scale,
                                                view.size.x / ctx.viewport_region->width,
                                                view.size.y / ctx.viewport_region->height});
            FrustumDepthCoverage coverage(view.size, 4.0f * pixel_scale);
            const auto [fx, fy] = lfs::rendering::computePixelFocalLengths(view.size, view.focal_length_mm);
            const auto project = [&](glm::vec3 p) {
                return glm::vec2(view.size) * 0.5f + glm::vec2(p.x, -p.y) *
                                                         glm::vec2(fx, fy) / -p.z;
            };
            constexpr int edges[][2] = {{0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 2}, {2, 3}, {3, 4}, {4, 1}};
            constexpr float near_z = -1e-4f;
            for (auto points : cache.points) {
                if (glm::distance(points[0], view.translation) < scale * 0.1f)
                    continue;
                bool quad_visible = true;
                std::array<glm::vec2, 4> projected;
                for (size_t i = 0; i < points.size(); ++i) {
                    auto& point = points[i];
                    point = glm::transpose(view.rotation) * (point - view.translation);
                    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
                        return {}; // Invalid projection: retain the full-depth contract.
                    if (i > 0) {
                        quad_visible &= point.z < near_z;
                        if (point.z < near_z) {
                            const auto p = project(point);
                            if (!std::isfinite(p.x) || !std::isfinite(p.y))
                                return {};
                            projected[i - 1] = p;
                        }
                    }
                }
                // Include the image plane even while its thumbnail is loading.
                if (quad_visible)
                    coverage.quad(projected);
                for (const auto& edge : edges) {
                    auto a = points[edge[0]], b = points[edge[1]];
                    if (a.z >= near_z && b.z >= near_z)
                        continue;
                    if (a.z >= near_z) {
                        a = glm::mix(a, b, (near_z - a.z) / (b.z - a.z));
                        a.z = near_z;
                    } else if (b.z >= near_z) {
                        b = glm::mix(b, a, (near_z - b.z) / (a.z - b.z));
                        b.z = near_z;
                    }
                    const auto pa = project(a), pb = project(b);
                    if (!std::isfinite(pa.x) || !std::isfinite(pa.y) || !std::isfinite(pb.x) || !std::isfinite(pb.y))
                        return {};
                    coverage.line(pa, pb);
                }
            }
            return coverage.take();
        }

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
        void applyGaussianViewVolume(lfs::rendering::GaussianFilterState& filters,
                                     const FrameContext& ctx) {
            // While GT comparison mode is active the depth filter's render effect
            // (dim/hide/containment, and the drag-preview lane with it) is fully
            // suspended. Settings are never mutated — dropping the filters from the
            // request restores everything the moment GT ends.
            if (!ctx.settings.depth_filter_enabled ||
                splitViewUsesGTComparison(ctx.settings.split_view_mode)) {
                return;
            }

            float depth_near = -ctx.settings.depth_filter_max.z;
            float depth_far = -ctx.settings.depth_filter_min.z;
            float scale_x = ctx.settings.depth_filter_scale_x;
            float scale_y = ctx.settings.depth_filter_scale_y;
            float offset_x = ctx.settings.depth_filter_offset_x;
            float offset_y = ctx.settings.depth_filter_offset_y;

            filters.view_volume = lfs::rendering::BoundingBox{
                .min = {ctx.settings.depth_filter_min.x,
                        ctx.settings.depth_filter_min.y,
                        -depth_far},
                .max = {ctx.settings.depth_filter_max.x,
                        ctx.settings.depth_filter_max.y,
                        -depth_near},
                .transform = ctx.settings.depth_filter_transform.inv().toMat4()};
            filters.screen_window = lfs::rendering::SelectionScreenWindow{
                .scale_x = scale_x,
                .scale_y = scale_y,
                .offset_x = offset_x,
                .offset_y = offset_y,
                .drag_preview = ctx.depth_window_drag_preview};
            filters.cull_outside_view_volume = ctx.settings.depth_filter_viz_mode == 2;
            filters.dim_outside_view_volume = ctx.settings.depth_filter_viz_mode == 1;
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

    lfs::rendering::ViewportRenderRequest buildViewportRenderRequest(
        const FrameContext& ctx,
        const glm::ivec2 render_size,
        const Viewport* const source_viewport,
        const std::optional<SplitViewPanelId> render_panel) {
        return buildViewportRenderRequest(
            ctx, render_size, source_viewport, render_panel, {0, 0}, {0, 0});
    }

    lfs::rendering::ViewportRenderRequest buildViewportRenderRequest(const FrameContext& ctx,
                                                                     const glm::ivec2 render_size,
                                                                     const Viewport* const source_viewport,
                                                                     const std::optional<SplitViewPanelId> render_panel,
                                                                     const glm::ivec2 subregion_origin,
                                                                     const glm::ivec2 subregion_full_size) {
        const Viewport& viewport = source_viewport ? *source_viewport : ctx.viewport;
        const auto frame_view = applySceneViewJitter(
            ctx.makeFrameView(viewport, render_size), ctx.scene_jitter_pixels);
        const bool selection_overlay_enabled = !ctx.training_active;
        const bool overlay_visible =
            selection_overlay_enabled && panelMatches(ctx.cursor_preview.panel, render_panel);
        const bool ring_selection_mode = ctx.cursor_preview.selection_mode == SelectionPreviewMode::Rings;

        lfs::rendering::ViewportRenderRequest request{
            .frame_view = frame_view,
            .color_exposure = ctx.settings.color_exposure,
            .color_tonemapping = ctx.settings.color_tonemapping,
            .splat_render_profile = ctx.settings.splat_render_profile,
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
                 .node_visibility_mask = ctx.scene_state.node_visibility_mask,
                 .node_active_sh_degrees = ctx.scene_state.node_active_sh_degrees},
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
                                                 : -1},
                 // Same FrameContext snapshot as emphasis.mask — no cross-frame lag.
                 .has_selection = selection_overlay_enabled && ctx.scene_state.has_selection},
            .transparent_background = environmentBackgroundUsesTransparentViewerCompositing(ctx.settings),
            .depth_view = ctx.settings.depth_view,
            .require_exact_depth = ctx.settings.show_camera_frustums ||
                                   std::any_of(ctx.scene_state.meshes.begin(), ctx.scene_state.meshes.end(),
                                               [](const auto& mesh) { return mesh.mesh != nullptr; }),
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
        request.frame_view.subregion_origin = subregion_origin;
        request.frame_view.subregion_full_size = subregion_full_size;
        if (ctx.settings.show_camera_frustums && !ctx.training_active && !request.gut && !request.depth_view &&
            ctx.settings.split_view_mode == SplitViewMode::Disabled &&
            !render_panel && (!source_viewport || source_viewport == &ctx.viewport) &&
            std::none_of(ctx.scene_state.meshes.begin(), ctx.scene_state.meshes.end(),
                         [](const auto& mesh) { return mesh.mesh != nullptr; }))
            request.exact_depth_sample_mask = frustumDepthSampleMask(ctx, request.frame_view);
        return request;
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

    void applyPlyComparisonNodeScope(
        lfs::rendering::GaussianSceneState& scene,
        lfs::rendering::GaussianFilterState& filters,
        lfs::rendering::GaussianOverlayState& overlay,
        const FrameContext& ctx,
        const core::SceneNode& node,
        const int visible_index) {
        // An owned model has one transform slot. Discard aggregate indices,
        // visibility and SH limits; the renderer uses this model's active SH
        // degree. The caller supplies the node's world transform separately.
        scene = {};
        const auto keep_matching = [visible_index](auto& regions, auto& primary) {
            using Filter = std::decay_t<decltype(regions[0])>;
            std::vector<Filter> kept;
            kept.reserve(regions.size());
            for (auto& region : regions) {
                if (region.parent_node_index == visible_index) {
                    region.parent_node_index = 0;
                    kept.push_back(std::move(region));
                }
            }
            regions = std::move(kept);
            if (!regions.empty()) {
                primary = regions.front();
            } else {
                primary.reset();
            }
        };
        keep_matching(filters.crop_regions, filters.crop_region);
        keep_matching(filters.ellipsoid_regions, filters.ellipsoid_region);

        overlay.emphasis.mask.reset();
        overlay.has_selection = false;
        if (ctx.scene_manager) {
            overlay.emphasis.mask =
                ctx.scene_manager->getScene().selectionMaskSliceForNode(node.id);
            overlay.has_selection =
                overlay.emphasis.mask && overlay.emphasis.mask->is_valid() &&
                overlay.emphasis.mask->numel() > 0 && ctx.scene_state.has_selection;
        }

        const bool selected =
            visible_index >= 0 &&
            static_cast<size_t>(visible_index) < ctx.scene_state.selected_node_mask.size() &&
            ctx.scene_state.selected_node_mask[static_cast<size_t>(visible_index)];
        overlay.emphasis.emphasized_node_mask =
            overlay.emphasis.dim_non_emphasized ? std::vector<bool>{selected} : std::vector<bool>{};
        size_t offset = 0;
        size_t count = 0;
        if (ctx.scene_manager) {
            for (const auto& slot : ctx.scene_manager->getScene().getVisibleSplatNodeSlots()) {
                if (!slot.node || !slot.node->model) {
                    continue;
                }
                const auto node_count = static_cast<size_t>(slot.node->model->size());
                if (slot.node->id == node.id) {
                    count = node_count;
                    break;
                }
                offset += node_count;
            }
        }
        auto& transient = overlay.emphasis.transient_mask;
        if (transient.mask && transient.mask->is_valid() &&
            transient.mask->ndim() == 1 && count > 0 &&
            offset + count <= transient.mask->numel()) {
            transient.owned_mask = std::make_shared<core::Tensor>(
                transient.mask->slice(0, offset, offset + count));
            transient.mask = transient.owned_mask.get();
        } else {
            transient = {};
        }
        const int focused = overlay.emphasis.focused_gaussian_id;
        overlay.emphasis.focused_gaussian_id =
            focused >= 0 && static_cast<size_t>(focused) >= offset &&
                    static_cast<size_t>(focused) < offset + count
                ? static_cast<int>(static_cast<size_t>(focused) - offset)
                : -1;
    }

    PlyComparisonDepthSample resolvePlyComparisonDepthSample(
        const core::Scene& scene,
        const size_t split_view_offset,
        const SplitViewPanelId panel) {
        PlyComparisonDepthSample sample;
        const auto visible_nodes = scene.getVisibleSplatNodeSlots();
        const auto pair = plyComparisonPairForOffset(visible_nodes.size(), split_view_offset);
        if (!pair) {
            return sample;
        }
        const size_t index = panel == SplitViewPanelId::Right ? pair->second : pair->first;
        const auto& slot = visible_nodes[index];
        sample.node = slot.node;
        sample.visible_index = static_cast<int>(slot.slot_index);
        if (slot.node && hasRenderableGaussians(slot.node->model.get())) {
            sample.model = slot.node->model.get();
            sample.uses_owned_node_model = true;
        } else {
            sample.model = scene.peekCombinedModel();
        }
        return sample;
    }

    void scopeSceneRenderStateToVisibleSplatNode(
        SceneRenderState& state,
        const core::Scene& scene,
        const core::SceneNode& node,
        const int visible_index,
        const glm::mat4& visualizer_world_transform) {
        state.combined_model = node.model.get();
        state.model_transforms = {visualizer_world_transform};
        state.node_active_sh_degrees = node.model
                                           ? std::vector<int>{node.model->get_active_sh_degree()}
                                           : std::vector<int>{};
        state.transform_indices.reset();
        state.node_visibility_mask.clear();
        state.visible_splat_count = hasRenderableGaussians(node.model.get()) ? 1 : 0;

        const auto keep_matching = [visible_index](auto& items) {
            using Item = std::decay_t<decltype(items[0])>;
            std::vector<Item> kept;
            kept.reserve(items.size());
            for (auto& item : items) {
                if (item.parent_node_index == visible_index) {
                    item.parent_node_index = 0;
                    kept.push_back(std::move(item));
                }
            }
            items = std::move(kept);
        };
        keep_matching(state.cropboxes);
        keep_matching(state.ellipsoids);
        state.selected_cropbox_index = -1;
        for (size_t i = 0; i < state.cropboxes.size(); ++i) {
            if (state.cropboxes[i].data && state.cropboxes[i].data->enabled) {
                state.selected_cropbox_index = static_cast<int>(i);
                break;
            }
        }

        state.selection_mask = scene.selectionMaskSliceForNode(node.id);
        const bool selected =
            visible_index >= 0 &&
            static_cast<size_t>(visible_index) < state.selected_node_mask.size() &&
            state.selected_node_mask[static_cast<size_t>(visible_index)];
        state.selected_node_mask = {selected};
        state.has_selection = scene.hasSelection() && state.selection_mask &&
                              state.selection_mask->is_valid();
    }

    PointSceneRenderer::RenderRequest buildPointSceneRequest(const lfs::rendering::PointCloudRenderRequest& frame,
                                                             const RenderSettings& settings) {
        const auto& view = frame.frame_view;
        const glm::mat4 view_matrix = view.getViewMatrix();
        const auto intrinsics = view.getCameraIntrinsics();
        glm::mat4 projection = lfs::rendering::createProjectionMatrix(
            view.size, lfs::rendering::focalLengthToVFov(view.focal_length_mm), view.orthographic,
            view.ortho_scale, view.near_plane, view.far_plane);
        if (!view.orthographic) {
            projection[0][0] = 2.0f * intrinsics.focal_x / view.size.x;
            projection[1][1] = 2.0f * intrinsics.focal_y / view.size.y;
            projection[2][0] = 1.0f - 2.0f * intrinsics.center_x / view.size.x;
            projection[2][1] = 2.0f * intrinsics.center_y / view.size.y - 1.0f;
        }
        // The projection is OpenGL NDC (Y up); images have a top-left origin.
        glm::mat4 clip_y_flip(1.0f);
        clip_y_flip[1][1] = -1.0f;
        PointSceneRenderer::RenderRequest request{};
        request.model_transforms = frame.scene.model_transforms;
        request.transform_indices = frame.scene.transform_indices.get();
        request.node_visibility_mask = &frame.scene.node_visibility_mask;
        request.selection_mask = frame.overlay.selection_mask.get();
        request.preview_selection_mask = frame.overlay.transient_mask.mask;
        request.selection_colors = &frame.overlay.selection_colors;
        request.preview_selection_additive = frame.overlay.transient_mask.additive;
        if (frame.filters.crop_box) {
            request.crop = PointSceneRenderer::CropBox{.to_local = frame.filters.crop_box->transform,
                                                       .min = frame.filters.crop_box->min,
                                                       .max = frame.filters.crop_box->max,
                                                       .inverse = frame.filters.crop_inverse,
                                                       .desaturate = frame.filters.crop_desaturate};
        } else if (frame.filters.crop_ellipsoid) {
            request.crop_ellipsoid = PointSceneRenderer::CropEllipsoid{.to_local = frame.filters.crop_ellipsoid->transform,
                                                                       .radii = frame.filters.crop_ellipsoid->radii,
                                                                       .inverse = frame.filters.crop_inverse,
                                                                       .desaturate = frame.filters.crop_desaturate};
        }
        request.view = view_matrix;
        request.view_projection = clip_y_flip * projection * view_matrix;
        request.size = view.size;
        request.background_color = view.background_color;
        request.transparent_background = frame.transparent_background;
        request.orthographic = view.orthographic;
        request.ortho_scale = view.ortho_scale;
        request.focal_y = intrinsics.focal_y;
        request.voxel_size = frame.render.voxel_size;
        request.scaling_modifier = frame.render.scaling_modifier;
        request.depth_view = settings.depth_view;
        request.depth_view_min = settings.depth_view_min;
        request.depth_view_max = settings.depth_view_max;
        request.depth_visualization_mode = settings.depth_visualization_mode;
        return request;
    }

    ViewportMeshPassDesc buildViewportMeshes(const FrameContext& ctx, const RenderSettings& settings) {
        ViewportMeshPassDesc frame;
        const auto vp_data = ctx.makeViewportData();
        frame.view_projection = vp_data.getProjectionMatrix() * vp_data.getViewMatrix();
        frame.camera_position = vp_data.translation;
        const auto& meshes = ctx.scene_state.meshes;
        const auto& selected_nodes = ctx.scene_state.selected_node_mask;
        const bool any_selected = std::ranges::any_of(meshes, [](const auto& mesh) { return mesh.is_selected; }) ||
                                  std::ranges::any_of(selected_nodes, [](const bool selected) { return selected; });
        const bool dim_non_emphasized = settings.desaturate_unselected && any_selected;
        const glm::vec3 headlight_dir = glm::length(vp_data.translation) > 1e-6f
                                            ? glm::normalize(vp_data.translation)
                                            : settings.mesh_light_dir;
        frame.items.reserve(meshes.size());
        for (const auto& mesh : meshes) {
            if (!mesh.mesh)
                continue;
            frame.items.push_back({
                .mesh = mesh.mesh,
                .model = mesh.transform,
                .light_dir = headlight_dir,
                .light_intensity = settings.mesh_light_intensity,
                .ambient = settings.mesh_ambient,
                .backface_culling = settings.mesh_backface_culling,
                .is_emphasized = mesh.is_selected,
                .dim_non_emphasized = dim_non_emphasized,
                .flash_intensity = ctx.selection_flash_intensity,
                .wireframe_overlay = settings.mesh_wireframe,
                .wireframe_color = settings.mesh_wireframe_color,
                .wireframe_width = settings.mesh_wireframe_width,
                .shadow_enabled = settings.mesh_shadow_enabled,
                .shadow_map_resolution = settings.mesh_shadow_resolution,
            });
        }
        return frame;
    }

    ViewportEnvironment buildViewportEnvironment(const FrameContext& ctx, const RenderSettings& settings,
                                                 const bool enabled) {
        const auto vp_data = ctx.makeViewportData();
        const auto frame_view = ctx.makeFrameView();
        ViewportEnvironment environment{
            .enabled = enabled,
            .map_path = settings.environment_map_path,
            .camera_to_world = vp_data.rotation,
            .viewport_size = glm::vec2(static_cast<float>(frame_view.size.x), static_cast<float>(frame_view.size.y)),
            .exposure = settings.environment_exposure,
            .rotation_radians = glm::radians(settings.environment_rotation_degrees),
            .equirectangular_view = settings.equirectangular,
        };
        if (frame_view.intrinsics_override.has_value() && !frame_view.orthographic) {
            const auto& intr = *frame_view.intrinsics_override;
            environment.intrinsics = glm::vec4(intr.focal_x, intr.focal_y, intr.center_x, intr.center_y);
        } else {
            const auto [fx, fy] = lfs::rendering::computePixelFocalLengths(frame_view.size, frame_view.focal_length_mm);
            environment.intrinsics = glm::vec4(fx, fy, frame_view.size.x * 0.5f, frame_view.size.y * 0.5f);
        }
        return environment;
    }

} // namespace lfs::vis
