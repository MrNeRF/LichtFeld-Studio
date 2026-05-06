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
#include <rendering/frame_contract.hpp>
#include <rendering/rendering.hpp>
#include <vector>

namespace lfs::core {
    class Tensor;
    class SplatData;
} // namespace lfs::core

namespace lfs::vis {

    class SceneManager;

    struct CursorPreviewState {
        bool active = false;
        float x = 0, y = 0, radius = 0;
        bool add_mode = true;
        lfs::core::Tensor* selection_tensor = nullptr;
        lfs::core::Tensor* preview_selection = nullptr;
        bool saturation_mode = false;
        float saturation_amount = 0;
        std::optional<SplitViewPanelId> panel;
        int focused_gaussian_id = -1;
        SelectionPreviewMode selection_mode{};
    };

    struct GizmoState {
        bool cropbox_active = false;
        glm::vec3 cropbox_min{0}, cropbox_max{0};
        glm::mat4 cropbox_transform{1};
        bool ellipsoid_active = false;
        glm::vec3 ellipsoid_radii{1};
        glm::mat4 ellipsoid_transform{1};
    };

    struct FrameViewPanel {
        SplitViewPanelId panel = SplitViewPanelId::Left;
        const Viewport* viewport = nullptr;
        glm::ivec2 render_size{0};
        glm::ivec2 viewport_offset{0};
        float start_position = 0.0f;
        float end_position = 1.0f;
        int grid_plane = 1;

        [[nodiscard]] bool valid() const {
            return viewport != nullptr && render_size.x > 0 && render_size.y > 0;
        }
    };

    struct FrameContext {
        const Viewport& viewport;
        const ViewportRegion* viewport_region = nullptr;
        bool render_lock_held = false;

        SceneManager* scene_manager = nullptr;
        const lfs::core::SplatData* model = nullptr;
        SceneRenderState scene_state;

        RenderSettings settings;
        glm::ivec2 render_size;
        glm::ivec2 viewport_pos;
        DirtyMask frame_dirty = 0;

        CursorPreviewState cursor_preview;
        GizmoState gizmo;
        int hovered_camera_id = -1;
        int current_camera_id = -1;
        int hovered_gaussian_id = -1;
        float selection_flash_intensity = 0;
        std::vector<FrameViewPanel> view_panels;

        [[nodiscard]] const FrameViewPanel* findViewPanel(const SplitViewPanelId panel_id) const {
            for (const auto& panel : view_panels) {
                if (panel.panel == panel_id && panel.valid()) {
                    return &panel;
                }
            }
            return nullptr;
        }

        [[nodiscard]] lfs::rendering::FrameView makeFrameView(const Viewport& source,
                                                              const glm::ivec2 size) const {
            return {.rotation = source.getRotationMatrix(),
                    .translation = source.getTranslation(),
                    .size = size,
                    .focal_length_mm = settings.focal_length_mm,
                    .intrinsics_override = std::nullopt,
                    .near_plane = lfs::rendering::DEFAULT_NEAR_PLANE,
                    .far_plane = settings.depth_clip_enabled ? settings.depth_clip_far
                                                             : lfs::rendering::DEFAULT_FAR_PLANE,
                    .orthographic = settings.orthographic,
                    .ortho_scale = settings.ortho_scale,
                    .background_color = settings.background_color};
        }

        [[nodiscard]] lfs::rendering::FrameView makeFrameView() const {
            return makeFrameView(viewport, render_size);
        }

        [[nodiscard]] lfs::rendering::FrameView makeFrameView(const FrameViewPanel& panel) const {
            return makeFrameView(*panel.viewport, panel.render_size);
        }

        [[nodiscard]] lfs::rendering::ViewportData makeViewportData(const Viewport& source,
                                                                    const glm::ivec2 size) const {
            const auto frame_view = makeFrameView(source, size);
            return {.rotation = frame_view.rotation,
                    .translation = frame_view.translation,
                    .size = frame_view.size,
                    .focal_length_mm = frame_view.focal_length_mm,
                    .orthographic = frame_view.orthographic,
                    .ortho_scale = frame_view.ortho_scale};
        }

        [[nodiscard]] lfs::rendering::ViewportData makeViewportData() const {
            return makeViewportData(viewport, render_size);
        }

        [[nodiscard]] lfs::rendering::ViewportData makeViewportData(const FrameViewPanel& panel) const {
            return makeViewportData(*panel.viewport, panel.render_size);
        }
    };

    struct CachedRenderPanelMetadata {
        std::shared_ptr<lfs::core::Tensor> depth;
        float start_position = 0.0f;
        float end_position = 1.0f;

        [[nodiscard]] bool valid() const {
            return end_position > start_position;
        }
    };

    struct CachedRenderMetadata {
        std::array<CachedRenderPanelMetadata, 2> depth_panels{};
        size_t depth_panel_count = 0;
        bool valid = false;
        bool depth_is_ndc = false;
        float near_plane = lfs::rendering::DEFAULT_NEAR_PLANE;
        float far_plane = lfs::rendering::DEFAULT_FAR_PLANE;
        bool orthographic = false;
        bool color_has_alpha = false;

        [[nodiscard]] const std::shared_ptr<lfs::core::Tensor>& primaryDepth() const {
            return depth_panels[0].depth;
        }
    };

    [[nodiscard]] inline CachedRenderMetadata makeCachedRenderMetadata(const lfs::rendering::FrameMetadata& result) {
        CachedRenderMetadata metadata{
            .depth_panel_count = result.depth_panel_count,
            .valid = result.valid,
            .depth_is_ndc = result.depth_is_ndc,
            .near_plane = result.near_plane,
            .far_plane = result.far_plane,
            .orthographic = result.orthographic,
            .color_has_alpha = result.color_has_alpha,
        };
        for (size_t i = 0; i < result.depth_panel_count && i < metadata.depth_panels.size(); ++i) {
            metadata.depth_panels[i] = {
                .depth = result.depth_panels[i].depth,
                .start_position = result.depth_panels[i].start_position,
                .end_position = result.depth_panels[i].end_position,
            };
        }
        return metadata;
    }

    struct FrameResources {
        CachedRenderMetadata cached_metadata;
        std::optional<lfs::rendering::GpuFrame> cached_gpu_frame;
        glm::ivec2 cached_result_size{0};
        bool splats_presented = false;
        bool split_view_executed = false;
        bool splat_pre_rendered = false;
        std::optional<GTComparisonContext> gt_context;

        int hovered_gaussian_id = -1;
        SplitViewInfo split_info;

        DirtyMask additional_dirty = 0;
        std::optional<std::chrono::steady_clock::time_point> pivot_animation_end;
    };

    inline void applyGTComparisonRenderCamera(
        lfs::rendering::FrameView& frame_view,
        bool& equirectangular,
        const std::optional<GTComparisonContext>& gt_context) {
        if (!gt_context || !gt_context->render_camera.has_value()) {
            return;
        }

        const auto& render_camera = *gt_context->render_camera;
        frame_view.rotation = render_camera.rotation;
        frame_view.translation = render_camera.translation;
        frame_view.intrinsics_override = render_camera.intrinsics;
        frame_view.orthographic = false;
        frame_view.ortho_scale = lfs::rendering::DEFAULT_ORTHO_SCALE;
        equirectangular = render_camera.equirectangular;
    }

} // namespace lfs::vis
