/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "geometry/euclidean_transform.hpp"
#include "rendering/render_constants.hpp"
#include <array>
#include <glm/glm.hpp>
#include <string>

namespace lfs::vis {

    constexpr int GPU_ALIGNMENT = 16;

    enum class SplitViewMode {
        Disabled,
        PLYComparison,
        GTComparison
    };

    struct PPISPOverrides {
        // Exposure (Section 4.1)
        float exposure_offset = 0.0f; // EV stops (-3 to +3)

        // Vignetting (Section 4.2)
        bool vignette_enabled = true;
        float vignette_strength = 1.0f; // 0.0 to 2.0

        // Color Correction (Section 4.3) - 4 chromaticity control points
        // White point (neutral) - intuitive temperature/tint controls
        float wb_temperature = 0.0f; // -1.0 to +1.0 (cool to warm)
        float wb_tint = 0.0f;        // -1.0 to +1.0 (green to magenta)
        // RGB primary offsets - direct chromaticity manipulation
        float color_red_x = 0.0f;   // -0.5 to +0.5
        float color_red_y = 0.0f;   // -0.5 to +0.5
        float color_green_x = 0.0f; // -0.5 to +0.5
        float color_green_y = 0.0f; // -0.5 to +0.5
        float color_blue_x = 0.0f;  // -0.5 to +0.5
        float color_blue_y = 0.0f;  // -0.5 to +0.5

        // CRF (Section 4.4) - piecewise power curve per channel
        float gamma_multiplier = 1.0f; // 0.5 to 2.5 (overall gamma)
        float gamma_red = 0.0f;        // -0.5 to +0.5 (per-channel offset)
        float gamma_green = 0.0f;      // -0.5 to +0.5
        float gamma_blue = 0.0f;       // -0.5 to +0.5
        float crf_toe = 0.0f;          // -1.0 to +1.0 (shadow compression)
        float crf_shoulder = 0.0f;     // -1.0 to +1.0 (highlight roll-off)

        [[nodiscard]] bool isIdentity() const {
            return exposure_offset == 0.0f && vignette_enabled && vignette_strength == 1.0f &&
                   wb_temperature == 0.0f && wb_tint == 0.0f && color_red_x == 0.0f && color_red_y == 0.0f &&
                   color_green_x == 0.0f && color_green_y == 0.0f && color_blue_x == 0.0f && color_blue_y == 0.0f &&
                   gamma_multiplier == 1.0f && gamma_red == 0.0f && gamma_green == 0.0f && gamma_blue == 0.0f &&
                   crf_toe == 0.0f && crf_shoulder == 0.0f;
        }
    };

    struct RenderSettings {
        // Core rendering settings
        float focal_length_mm = lfs::rendering::DEFAULT_FOCAL_LENGTH_MM;
        float scaling_modifier = 1.0f;
        bool antialiasing = false;
        bool mip_filter = false;
        int sh_degree = 3;
        float render_scale = 1.0f; // Viewer resolution scale (0.25-1.0), does not affect training

        // Crop box (data stored in scene graph CropBoxData, these are UI toggles only)
        bool show_crop_box = false;
        bool use_crop_box = false;
        // Ellipsoid (data stored in scene graph EllipsoidData, these are UI toggles only)
        bool show_ellipsoid = false;
        bool use_ellipsoid = false;
        bool desaturate_unselected = false;     // Desaturate unselected PLYs when one is selected
        bool desaturate_cropping = true;        // Desaturate outside crop box/ellipsoid instead of hiding
        bool crop_filter_for_selection = false; // Use crop box/ellipsoid as selection filter

        // Appearance correction (PPISP)
        bool apply_appearance_correction = false;
        enum class PPISPMode { MANUAL = 0,
                               AUTO = 1 };
        PPISPMode ppisp_mode = PPISPMode::AUTO;
        PPISPOverrides ppisp_overrides;

        // Background
        glm::vec3 background_color = glm::vec3(0.0f, 0.0f, 0.0f);

        // Coordinate axes
        bool show_coord_axes = false;
        float axes_size = 2.0f;
        std::array<bool, 3> axes_visibility = {true, true, true};

        // Grid
        bool show_grid = true;
        int grid_plane = 1;
        float grid_opacity = 0.5f;

        // Point cloud
        bool point_cloud_mode = false;
        float voxel_size = 0.01f;

        // Ring mode (only active in splat mode)
        bool show_rings = false;
        float ring_width = 0.01f;
        bool show_center_markers = false;

        // Camera frustums
        bool show_camera_frustums = true; // Master toggle for camera frustum rendering
        float camera_frustum_scale = 0.25f;
        glm::vec3 train_camera_color = glm::vec3(1.0f, 1.0f, 1.0f);
        glm::vec3 eval_camera_color = glm::vec3(1.0f, 0.0f, 0.0f);

        // Pivot point visualization
        bool show_pivot = false;

        // Split view
        SplitViewMode split_view_mode = SplitViewMode::Disabled;
        float split_position = 0.5f;
        size_t split_view_offset = 0;

        bool gut = false;
        bool equirectangular = false;
        bool orthographic = false;
        float ortho_scale = 100.0f; // Pixels per world unit (larger = more zoomed in)

        // Selection colors (RGB: committed=219,83,83 preview=0,222,76 center=0,154,187)
        glm::vec3 selection_color_committed{0.859f, 0.325f, 0.325f};
        glm::vec3 selection_color_preview{0.0f, 0.871f, 0.298f};
        glm::vec3 selection_color_center_marker{0.0f, 0.604f, 0.733f};

        // Depth clipping
        bool depth_clip_enabled = false;
        float depth_clip_far = 100.0f;

        bool mesh_wireframe = false;
        glm::vec3 mesh_wireframe_color{0.2f};
        float mesh_wireframe_width = 1.0f;
        glm::vec3 mesh_light_dir{0.3f, 1.0f, 0.5f};
        float mesh_light_intensity = 0.7f;
        float mesh_ambient = 0.4f;
        bool mesh_backface_culling = true;
        bool mesh_shadow_enabled = false;
        int mesh_shadow_resolution = 2048;

        // Depth filter (Selection tool only - separate from crop box)
        bool depth_filter_enabled = false;
        glm::vec3 depth_filter_min = glm::vec3(-50.0f, -10000.0f, 0.0f);
        glm::vec3 depth_filter_max = glm::vec3(50.0f, 10000.0f, 100.0f);
        lfs::geometry::EuclideanTransform depth_filter_transform;
    };

    struct SplitViewInfo {
        bool enabled = false;
        std::string left_name;
        std::string right_name;
    };

    struct ViewportRegion {
        float x, y, width, height;
    };

    struct GTComparisonContext {
        unsigned int gt_texture_id = 0;
        glm::ivec2 dimensions{0, 0};
        glm::ivec2 gpu_aligned_dims{0, 0};
        glm::vec2 render_texcoord_scale{1.0f, 1.0f};
        glm::vec2 gt_texcoord_scale{1.0f, 1.0f};
        bool gt_needs_flip = false;

        [[nodiscard]] bool valid() const { return gt_texture_id != 0 && dimensions.x > 0 && dimensions.y > 0; }
    };

} // namespace lfs::vis
