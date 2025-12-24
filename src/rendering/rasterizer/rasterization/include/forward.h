/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "helper_math.h"
#include <cstdint>
#include <functional>

namespace lfs::rendering {

    void brush_select(
        const float2* screen_positions,
        float mouse_x,
        float mouse_y,
        float radius,
        uint8_t* selection_out,
        int n_primitives);

    // Select Gaussians inside rectangle (sets true for points inside)
    void rect_select(
        const float2* positions,
        float x0, float y0, float x1, float y1,
        bool* selection,
        int n_primitives);

    // Select Gaussians inside rectangle with add/remove mode
    void rect_select_mode(
        const float2* positions,
        float x0, float y0, float x1, float y1,
        bool* selection,
        int n_primitives,
        bool add_mode);

    // Set single selection element on GPU
    void set_selection_element(bool* selection, int index, bool value);

    // Select Gaussians inside polygon (sets true for points inside)
    void polygon_select(
        const float2* positions,
        const float2* polygon,
        int num_vertices,
        bool* selection,
        int n_primitives);

    // Select Gaussians inside polygon with add/remove mode
    void polygon_select_mode(
        const float2* positions,
        const float2* polygon,
        int num_vertices,
        bool* selection,
        int n_primitives,
        bool add_mode);

    void forward(
        std::function<char*(size_t)> per_primitive_buffers_func,
        std::function<char*(size_t)> per_tile_buffers_func,
        std::function<char*(size_t)> per_instance_buffers_func,
        const float3* means,
        const float3* scales_raw,
        const float4* rotations_raw,
        const float* opacities_raw,
        const float3* sh_coefficients_0,
        const float3* sh_coefficients_rest,
        const float4* w2c,
        const float3* cam_position,
        float* image,
        float* alpha,
        float* depth,
        int n_primitives,
        int active_sh_bases,
        int total_bases_sh_rest,
        int width,
        int height,
        float fx,
        float fy,
        float cx,
        float cy,
        float near,
        float far,
        bool show_rings = false,
        float ring_width = 0.01f,
        const float* model_transforms = nullptr,
        const int* transform_indices = nullptr,
        int num_transforms = 0,
        const uint8_t* selection_mask = nullptr,
        float2* screen_positions_out = nullptr,
        bool brush_active = false,
        float brush_x = 0.0f,
        float brush_y = 0.0f,
        float brush_radius = 0.0f,
        bool brush_add_mode = true,
        bool* brush_selection_out = nullptr,
        bool brush_saturation_mode = false,
        float brush_saturation_amount = 0.0f,
        bool selection_mode_rings = false,
        bool show_center_markers = false,
        const float* crop_box_transform = nullptr,
        const float3* crop_box_min = nullptr,
        const float3* crop_box_max = nullptr,
        bool crop_inverse = false,
        bool crop_desaturate = false,
        const float* depth_filter_transform = nullptr,
        const float3* depth_filter_min = nullptr,
        const float3* depth_filter_max = nullptr,
        const bool* deleted_mask = nullptr,
        unsigned long long* hovered_depth_id = nullptr,
        int highlight_gaussian_id = -1,
        const bool* selected_node_mask = nullptr,
        int num_selected_nodes = 0,
        bool desaturate_unselected = false,
        float selection_flash_intensity = 0.0f,
        bool orthographic = false,
        float ortho_scale = 1.0f,
        bool mip_filter = false);

} // namespace lfs::rendering