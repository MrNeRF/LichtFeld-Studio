/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "rendering/render_constants.hpp"
#include <tuple>

namespace lfs::rendering {

    using lfs::core::Tensor;

    std::tuple<Tensor, Tensor> rasterize_tensor(
        const lfs::core::Camera& viewpoint_camera,
        const lfs::core::SplatData& gaussian_model,
        const Tensor& bg_color,
        bool show_rings = false,
        float ring_width = 0.01f,
        const Tensor* model_transforms = nullptr,
        const Tensor* transform_indices = nullptr,
        const Tensor* selection_mask = nullptr,
        Tensor* screen_positions_out = nullptr,
        bool brush_active = false,
        float brush_x = 0.0f,
        float brush_y = 0.0f,
        float brush_radius = 0.0f,
        bool brush_add_mode = true,
        Tensor* brush_selection_out = nullptr,
        bool brush_saturation_mode = false,
        float brush_saturation_amount = 0.0f,
        bool selection_mode_rings = false,
        bool show_center_markers = false,
        const Tensor* crop_box_transform = nullptr,
        const Tensor* crop_box_min = nullptr,
        const Tensor* crop_box_max = nullptr,
        bool crop_inverse = false,
        bool crop_desaturate = false,
        const Tensor* depth_filter_transform = nullptr,
        const Tensor* depth_filter_min = nullptr,
        const Tensor* depth_filter_max = nullptr,
        const Tensor* deleted_mask = nullptr,
        unsigned long long* hovered_depth_id = nullptr,
        int highlight_gaussian_id = -1,
        float far_plane = DEFAULT_FAR_PLANE,
        const std::vector<bool>& selected_node_mask = {},
        bool desaturate_unselected = false,
        float selection_flash_intensity = 0.0f,
        bool orthographic = false,
        float ortho_scale = 1.0f);

    // GUT rasterization for viewer (forward-only, no training dependency)
    struct GutRenderOutput {
        Tensor image; // [3, H, W]
        Tensor depth; // [1, H, W]
    };

    GutRenderOutput gut_rasterize_tensor(
        const lfs::core::Camera& camera,
        const lfs::core::SplatData& model,
        const Tensor& bg_color,
        float scaling_modifier = 1.0f);

} // namespace lfs::rendering
