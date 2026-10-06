/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "render_target_id.hpp"
#include "scene_temporal_resolve.hpp"
#include "scene_upscaler_registry.hpp"
#include "temporal_frame_tracker.hpp"
#include "viewport_draw_types.hpp"
#include <optional>

namespace lfs::vis {
    struct ViewportSceneOutputDesc {
        RenderTargetId target{};
        std::shared_ptr<const lfs::core::Tensor> color, depth;
        glm::ivec2 size{0}, allocation_size{0};
        uint64_t generation = 0;
        bool flip_y = false;
    };
    struct ViewportDepth {
        std::shared_ptr<const lfs::core::Tensor> depth;
        bool depth_is_ndc = false, flip_y = false;
        float near_plane = 0.1f, far_plane = 1000.0f;
        glm::vec2 uv_scale{1}, uv_clamp_max{1};
    };
    struct ViewportTemporalRequest {
        RenderTargetId target{};
        TemporalFrameInput input;
        SceneTemporalResolveSettings settings;
        SceneTemporalQuality quality = SceneTemporalQuality::Balanced;
    };
    struct ViewportPluginRequest {
        ViewportTemporalRequest temporal;
        SceneUpscalerBackend plugin = SceneUpscalerBackend::Native;
    };
    struct ViewportSplitPanel {
        RenderTargetId target{};
        std::shared_ptr<const lfs::core::Tensor> image, depth;
        float start_position = 0, end_position = 1;
        bool normalize_x_to_panel = false, flip_y = false, spatial_filter = false;
        glm::ivec2 image_size{0}, allocation_size{0};
        glm::vec2 uv_scale{1}, uv_clamp_max{1}, texcoord_scale{1}, texcoord_offset{0};
    };
    struct ViewportSplitView {
        bool enabled = false, loss_visualization = false;
        bool exact_texel_sampling = false;
        ViewportSplitPanel left, right;
        float split_position = 0.5f;
        glm::ivec4 content_rect{0};
        glm::ivec2 coordinate_extent{0};
        glm::vec3 background{0};
    };

    struct ViewportFrameDesc {
        std::size_t frame_slot = 0;
        glm::uvec2 framebuffer_extent{0, 0};
        std::vector<ViewportSceneOutputDesc> scene_outputs;
        bool export_locked = false;
        glm::vec2 viewport_pos{0.0f, 0.0f};
        glm::vec2 viewport_size{0.0f, 0.0f};
        glm::vec2 framebuffer_scale{1.0f, 1.0f};
        glm::vec3 background_color{0.0f, 0.0f, 0.0f};

        std::shared_ptr<const lfs::core::Tensor> scene_image;
        glm::ivec2 scene_image_size{0, 0};
        // Bucketed image extent for padded output slots (defaults to size).
        glm::ivec2 scene_image_alloc_size{0, 0};
        bool scene_image_flip_y = false;
        // Interactive resize deliberately keeps the last complete interop image
        // until the render extent settles. Do not replace that binding with an
        // incompletely prepared image during the deferral window.
        bool preserve_scene_image_binding = false;
        SceneUpscalerBackend scene_upscaler = SceneUpscalerBackend::Native;
        bool scene_upscaler_mode_unsupported = false;
        std::optional<ViewportTemporalRequest> temporal;
        std::array<std::optional<ViewportTemporalRequest>, 2> split_temporal;
        std::optional<ViewportPluginRequest> plugin;
        std::array<std::optional<ViewportPluginRequest>, 2> split_plugin;

        bool grid_enabled = false;
        glm::mat4 grid_view{1.0f};
        glm::mat4 grid_projection{1.0f};
        glm::mat4 grid_view_projection{1.0f};
        glm::vec3 grid_view_position{0.0f, 0.0f, 0.0f};
        int grid_plane = 2;
        float grid_opacity = 1.0f;
        bool grid_orthographic = false;
        std::vector<ViewportGridOverlay> grid_overlays;

        bool vignette_enabled = false;
        float vignette_intensity = 0.0f;
        float vignette_radius = 0.75f;
        float vignette_softness = 0.5f;

        std::vector<ViewportOverlayVertex> overlay_triangles;
        std::vector<ViewportShapeOverlayVertex> shape_overlay_triangles;
        std::vector<ViewportShapeOverlayVertex> ui_shape_overlay_triangles;
        // Number of trailing overlay_triangles to draw after viewport UI overlays.
        std::uint32_t post_ui_overlay_vertex_count = 0;
        std::vector<ViewportPivotOverlay> pivot_overlays;
        std::vector<ViewportTexturedOverlay> textured_overlays;
        // Textured overlays drawn in the UI phase (after ui_shape_overlay): screen-space
        // text and icons that must layer above overlay fills and gizmo shapes.
        std::vector<ViewportTexturedOverlay> ui_textured_overlays;
        // Immutable for the duration of prepare/record. GUI frustum caches keep
        // this shared block alive so an idle frame does not copy 1740 entries.
        std::shared_ptr<const ViewportFrustumOverlayData> frustum_overlay_data;
        std::vector<ViewportFrustumInstance> frustum_instances;
        std::vector<ViewportFrustumBatch> frustum_batches;

        // GPU-rendered meshes drawn into the same color/depth attachments as the
        // viewport pass.
        glm::mat4 mesh_view_projection{1.0f};
        glm::vec3 mesh_camera_position{0.0f};
        std::vector<ViewportMeshDrawItem> mesh_items;
        std::vector<ViewportMeshPanel> mesh_panels;

        // GPU-rendered equirect environment background. Replaces the old CPU
        // `renderEnvironmentBackground` per-pixel sampling loop.
        ViewportEnvironment environment;

        // Splat depth blit. When set, mesh draws will depth-test against the splat
        // depth surface, so meshes occluded by splats render correctly.
        ViewportDepth depth_blit;

        // Split-view composite. When enabled, replaces the single-tensor scene quad
        // blit with a two-panel composite (master-style divider/handle/grip in-shader).
        ViewportSplitView split_view;
    };

} // namespace lfs::vis
