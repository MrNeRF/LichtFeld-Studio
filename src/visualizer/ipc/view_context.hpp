/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "visualizer/rendering/rendering_manager.hpp"

#include <array>
#include <functional>
#include <memory>
#include <optional>

namespace lfs::core {
    class Tensor;
}

namespace lfs::vis {

    struct ViewInfo {
        std::array<float, 9> rotation;
        std::array<float, 3> translation;
        std::array<float, 3> pivot;
        int width;
        int height;
        float fov;
    };

    struct SetViewParams {
        std::array<float, 3> eye;
        std::array<float, 3> target;
        std::array<float, 3> up;
    };

    using SetViewCallback = std::function<void(const SetViewParams&)>;
    using SetFovCallback = std::function<void(float)>;

    struct ViewportRender {
        std::shared_ptr<lfs::core::Tensor> image;
        std::shared_ptr<lfs::core::Tensor> screen_positions;
    };

    using GetViewCallback = std::function<std::optional<ViewInfo>()>;
    using GetViewportRenderCallback = std::function<std::optional<ViewportRender>()>;

    LFS_VIS_API void set_view_callback(GetViewCallback callback);
    LFS_VIS_API void set_viewport_render_callback(GetViewportRenderCallback callback);
    [[nodiscard]] LFS_VIS_API std::optional<ViewInfo> get_current_view_info();
    [[nodiscard]] LFS_VIS_API std::optional<ViewportRender> get_viewport_render();

    LFS_VIS_API void set_set_view_callback(SetViewCallback callback);
    LFS_VIS_API void set_set_fov_callback(SetFovCallback callback);
    LFS_VIS_API void apply_set_view(const SetViewParams& params);
    LFS_VIS_API void apply_set_fov(float fov_degrees);

    struct RenderSettingsProxy {
        float focal_length_mm = 35.0f;
        float scaling_modifier = 1.0f;
        bool antialiasing = false;
        bool mip_filter = false;
        int sh_degree = 3;
        float render_scale = 1.0f;
        bool show_crop_box = false;
        bool use_crop_box = false;
        bool desaturate_unselected = false;
        bool desaturate_cropping = true;
        std::array<float, 3> background_color{0.0f, 0.0f, 0.0f};
        bool show_coord_axes = false;
        float axes_size = 2.0f;
        bool show_grid = true;
        int grid_plane = 1;
        float grid_opacity = 0.5f;
        bool point_cloud_mode = false;
        float voxel_size = 0.01f;
        bool show_rings = false;
        float ring_width = 0.01f;
        bool show_center_markers = false;
        bool show_camera_frustums = true;
        float camera_frustum_scale = 0.25f;
        std::array<float, 3> train_camera_color{1.0f, 1.0f, 1.0f};
        std::array<float, 3> eval_camera_color{1.0f, 0.0f, 0.0f};
        bool show_pivot = false;
        float split_position = 0.5f;
        bool gut = false;
        bool equirectangular = false;
        bool orthographic = false;
        float ortho_scale = 100.0f;
        std::array<float, 3> selection_color_committed{0.859f, 0.325f, 0.325f};
        std::array<float, 3> selection_color_preview{0.0f, 0.871f, 0.298f};
        std::array<float, 3> selection_color_center_marker{0.0f, 0.604f, 0.733f};
        bool depth_clip_enabled = false;
        float depth_clip_far = 100.0f;

        bool apply_appearance_correction = false;
        int ppisp_mode = 1;
        PPISPOverrides ppisp;

        bool mesh_wireframe = false;
        std::array<float, 3> mesh_wireframe_color{0.2f, 0.2f, 0.2f};
        float mesh_wireframe_width = 1.0f;
        std::array<float, 3> mesh_light_dir{0.3f, 1.0f, 0.5f};
        float mesh_light_intensity = 0.7f;
        float mesh_ambient = 0.4f;
        bool mesh_backface_culling = true;
        bool mesh_shadow_enabled = false;
        int mesh_shadow_resolution = 2048;
    };

    using GetRenderSettingsCallback = std::function<std::optional<RenderSettingsProxy>()>;
    using SetRenderSettingsCallback = std::function<void(const RenderSettingsProxy&)>;

    LFS_VIS_API void set_render_settings_callbacks(GetRenderSettingsCallback get_cb, SetRenderSettingsCallback set_cb);
    [[nodiscard]] LFS_VIS_API std::optional<RenderSettingsProxy> get_render_settings();
    LFS_VIS_API void update_render_settings(const RenderSettingsProxy& settings);

} // namespace lfs::vis
