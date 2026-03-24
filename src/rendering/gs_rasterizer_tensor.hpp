/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "rasterizer/rasterization/include/gsplat_forward.h"
#include "rendering/render_constants.hpp"
#include <tuple>

namespace lfs::rendering {

    using lfs::core::Tensor;

    std::tuple<Tensor, Tensor> rasterize_tensor(
        const lfs::core::Camera& viewpoint_camera,
        const lfs::core::SplatData& gaussian_model,
        const Tensor& bg_color,
        int sh_degree_override = -1,
        bool show_rings = false,
        float ring_width = 0.01f,
        const Tensor* model_transforms = nullptr,
        const Tensor* transform_indices = nullptr,
        const Tensor* selection_mask = nullptr,
        Tensor* screen_positions_out = nullptr,
        bool cursor_active = false,
        float cursor_x = 0.0f,
        float cursor_y = 0.0f,
        float cursor_radius = 0.0f,
        bool preview_selection_add_mode = true,
        Tensor* preview_selection_out = nullptr,
        bool cursor_saturation_preview = false,
        float cursor_saturation_amount = 0.0f,
        bool show_center_markers = false,
        const Tensor* crop_box_transform = nullptr,
        const Tensor* crop_box_min = nullptr,
        const Tensor* crop_box_max = nullptr,
        bool crop_inverse = false,
        bool crop_desaturate = false,
        int crop_parent_node_index = -1,
        const Tensor* ellipsoid_transform = nullptr,
        const Tensor* ellipsoid_radii = nullptr,
        bool ellipsoid_inverse = false,
        bool ellipsoid_desaturate = false,
        int ellipsoid_parent_node_index = -1,
        const Tensor* view_volume_transform = nullptr,
        const Tensor* view_volume_min = nullptr,
        const Tensor* view_volume_max = nullptr,
        bool view_volume_cull = false,
        const Tensor* deleted_mask = nullptr,
        unsigned long long* hovered_depth_id = nullptr,
        int focused_gaussian_id = -1,
        float far_plane = DEFAULT_FAR_PLANE,
        const std::vector<bool>& emphasized_node_mask = {},
        bool dim_non_emphasized = false,
        const std::vector<bool>& node_visibility_mask = {},
        float emphasis_flash_intensity = 0.0f,
        bool orthographic = false,
        float ortho_scale = 1.0f,
        bool mip_filter = false);

    // GUT rasterization for viewer (forward-only, no training dependency)
    struct GutRenderOutput {
        Tensor image; // [3, H, W]
        Tensor depth; // [1, H, W]
    };

    GutRenderOutput gut_rasterize_tensor(
        const lfs::core::Camera& camera,
        const lfs::core::SplatData& model,
        const Tensor& bg_color,
        int sh_degree_override = -1,
        float scaling_modifier = 1.0f,
        GutCameraModel camera_model = GutCameraModel::PINHOLE,
        const Tensor* model_transforms = nullptr,
        const Tensor* transform_indices = nullptr,
        const std::vector<bool>& node_visibility_mask = {});

} // namespace lfs::rendering
