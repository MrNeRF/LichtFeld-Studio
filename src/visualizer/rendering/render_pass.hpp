/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "dirty_flags.hpp"
#include "internal/viewport.hpp"
#include "rendering_types.hpp"
#include "scene/scene_render_state.hpp"
#include <chrono>
#include <optional>
#include <rendering/rendering.hpp>
#include <shared_mutex>

namespace lfs::core {
    class Tensor;
    class SplatData;
} // namespace lfs::core

namespace lfs::vis {

    class SceneManager;

    struct BrushState {
        bool active = false;
        float x = 0, y = 0, radius = 0;
        bool add_mode = true;
        lfs::core::Tensor* selection_tensor = nullptr;
        lfs::core::Tensor* preview_selection = nullptr;
        bool saturation_mode = false;
        float saturation_amount = 0;
        lfs::rendering::SelectionMode selection_mode{};
        bool output_screen_positions = false;
    };

    struct GizmoState {
        bool cropbox_active = false;
        glm::vec3 cropbox_min{0}, cropbox_max{0};
        glm::mat4 cropbox_transform{1};
        bool ellipsoid_active = false;
        glm::vec3 ellipsoid_radii{1};
        glm::mat4 ellipsoid_transform{1};
    };

    struct PickState {
        bool requested = false;
        glm::vec2 pos{-1};
        int hovered_camera_id = -1;
    };

    struct FrameContext {
        const Viewport& viewport;
        const ViewportRegion* viewport_region = nullptr;

        SceneManager* scene_manager = nullptr;
        const lfs::core::SplatData* model = nullptr;
        SceneRenderState scene_state;

        RenderSettings settings;
        glm::ivec2 render_size;
        glm::ivec2 viewport_pos;
        DirtyMask frame_dirty = 0;

        BrushState brush;
        GizmoState gizmo;
        PickState pick;
        int current_camera_id = -1;
        int hovered_gaussian_id = -1;
        float selection_flash_intensity = 0;
        unsigned int cached_render_texture = 0;

        [[nodiscard]] lfs::rendering::ViewportData makeViewportData() const {
            return {.rotation = viewport.getRotationMatrix(),
                    .translation = viewport.getTranslation(),
                    .size = render_size,
                    .focal_length_mm = settings.focal_length_mm,
                    .orthographic = settings.orthographic,
                    .ortho_scale = settings.ortho_scale};
        }
    };

    // Pass execution order (defined in RenderingManager constructor):
    //   [pre] SplatRasterPass — GT comparison pre-render (before loop, at GT dimensions)
    //   [0]   SplitViewPass   — Side-by-side views (blocks all downstream if active)
    //   [1]   SplatRasterPass — Render splats to offscreen FBO
    //   [2]   PointCloudPass  — Pre-training point cloud (mutually exclusive with splats)
    //   [3]   PresentPass     — Blit cached splat image to screen
    //   [4]   MeshPass        — Render meshes, composite with splats
    //   [5]   OverlayPass     — Grid, crop boxes, frustums, pivot, axes
    //
    // Inter-pass coordination flags:
    //   split_view_executed — Set by SplitViewPass. Skips SplatRaster/PointCloud/Present/Mesh.
    //   splats_presented    — Set by PresentPass/PointCloudPass. Tells MeshPass to composite.
    //   splat_pre_rendered  — Set by GT pre-render. Skips SplatRasterPass in the loop.
    struct FrameResources {
        lfs::rendering::RenderResult cached_result;
        glm::ivec2 cached_result_size{0};
        bool render_texture_valid = false;
        bool splats_presented = false;
        bool split_view_executed = false;
        bool splat_pre_rendered = false;
        std::optional<GTComparisonContext> gt_context;

        int hovered_gaussian_id = -1;
        int hovered_camera_id = -1;
        bool pick_consumed = false;
        SplitViewInfo split_info;

        DirtyMask additional_dirty = 0;
        std::optional<std::chrono::steady_clock::time_point> pivot_animation_end;
    };

    std::optional<std::shared_lock<std::shared_mutex>> acquireRenderLock(const FrameContext& ctx);

    class RenderPass {
    public:
        virtual ~RenderPass() = default;
        [[nodiscard]] virtual const char* name() const = 0;
        [[nodiscard]] virtual DirtyMask sensitivity() const = 0;

        [[nodiscard]] virtual bool shouldExecute(DirtyMask frame_dirty, const FrameContext& ctx) const {
            return (frame_dirty & sensitivity()) != 0;
        }

        virtual void execute(lfs::rendering::RenderingEngine& engine,
                             const FrameContext& ctx,
                             FrameResources& res) = 0;
    };

} // namespace lfs::vis
