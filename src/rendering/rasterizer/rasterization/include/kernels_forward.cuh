/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "buffer_utils.h"
#include "helper_math.h"
#include "kernel_utils.cuh"
#include "rasterization_config.h"
#include "utils.h"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace lfs::rendering::kernels::forward {

    // Selection status encoding: bits 0-6 = group ID, bit 7 = preview flag
    constexpr uint8_t SELECTION_PREVIEW_FLAG = 0x80;
    constexpr uint8_t SELECTION_GROUP_MASK = 0x7F;
    constexpr float SELECTION_PREVIEW_BLEND = 0.9f;
    constexpr float SELECTION_COMMITTED_BLEND = 0.8f;

    __global__ void preprocess_cu(
        const float3* means,
        const float3* raw_scales,
        const float4* raw_rotations,
        const float* raw_opacities,
        const float3* sh_coefficients_0,
        const float3* sh_coefficients_rest,
        const float4* w2c,
        const float3* cam_position,
        uint* primitive_depth_keys,
        uint* primitive_indices,
        uint* primitive_n_touched_tiles,
        ushort4* primitive_screen_bounds,
        float2* primitive_mean2d,
        float4* primitive_conic_opacity,
        float3* primitive_color,
        float* primitive_depth,
        bool* primitive_outside_crop,
        uint8_t* primitive_selection_status,
        uint* n_visible_primitives,
        uint* n_instances,
        const uint n_primitives,
        const uint grid_width,
        const uint grid_height,
        const uint active_sh_bases,
        const uint total_bases_sh_rest,
        const float w,
        const float h,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        const float near_, // near and far are macros in windowns
        const float far_,
        const float* model_transforms, // Array of 4x4 transforms (row-major), one per node
        const int* transform_indices,  // Per-Gaussian index into transforms array [N]
        const int num_transforms,      // Number of transforms in array
        const uint8_t* selection_mask,
        const bool brush_active,
        const float brush_x,
        const float brush_y,
        const float brush_radius_sq,
        const bool brush_add_mode,
        bool* brush_selection_out,
        const bool brush_saturation_mode,
        const float brush_saturation_amount,
        const bool selection_mode_rings,
        const float* crop_box_transform,
        const float3* crop_box_min,
        const float3* crop_box_max,
        const bool crop_inverse,
        const bool crop_desaturate,
        const float* depth_filter_transform,
        const float3* depth_filter_min,
        const float3* depth_filter_max,
        const bool* deleted_mask,
        const int highlight_gaussian_id,
        unsigned long long* hovered_depth_id,
        const bool* selected_node_mask,
        const int num_selected_nodes,
        const bool desaturate_unselected,
        const bool orthographic,
        const float ortho_scale) {
        auto primitive_idx = cg::this_grid().thread_rank();
        bool active = true;
        if (primitive_idx >= n_primitives) {
            active = false;
            primitive_idx = n_primitives - 1;
        }

        // Soft deletion mask culling - skip deleted gaussians
        if (active && deleted_mask != nullptr && deleted_mask[primitive_idx]) {
            active = false;
        }

        if (active)
            primitive_n_touched_tiles[primitive_idx] = 0;

        float3 mean3d = means[primitive_idx];

        // Apply model transform
        mat3x3 model_rot = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        bool has_transform = false;
        if (model_transforms != nullptr && num_transforms > 0) {
            const int transform_idx = transform_indices != nullptr
                                          ? min(max(transform_indices[primitive_idx], 0), num_transforms - 1)
                                          : 0;
            const float* const m = model_transforms + transform_idx * 16;
            has_transform = m[0] != 1.0f || m[5] != 1.0f || m[10] != 1.0f ||
                            m[1] != 0.0f || m[2] != 0.0f || m[3] != 0.0f ||
                            m[4] != 0.0f || m[6] != 0.0f || m[7] != 0.0f ||
                            m[8] != 0.0f || m[9] != 0.0f || m[11] != 0.0f;
            if (has_transform) {
                model_rot = {m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]};
                const float3 t = make_float3(m[3], m[7], m[11]);
                mean3d = make_float3(
                    model_rot.m11 * mean3d.x + model_rot.m12 * mean3d.y + model_rot.m13 * mean3d.z + t.x,
                    model_rot.m21 * mean3d.x + model_rot.m22 * mean3d.y + model_rot.m23 * mean3d.z + t.y,
                    model_rot.m31 * mean3d.x + model_rot.m32 * mean3d.y + model_rot.m33 * mean3d.z + t.z);
            }
        }

        // Crop box test (on transformed position)
        bool outside_crop = false;
        if (active && crop_box_transform != nullptr) {
            const float3 bmin = *crop_box_min;
            const float3 bmax = *crop_box_max;
            const float* const c = crop_box_transform;
            const float lx = c[0] * mean3d.x + c[1] * mean3d.y + c[2] * mean3d.z + c[3];
            const float ly = c[4] * mean3d.x + c[5] * mean3d.y + c[6] * mean3d.z + c[7];
            const float lz = c[8] * mean3d.x + c[9] * mean3d.y + c[10] * mean3d.z + c[11];
            const bool inside = lx >= bmin.x && lx <= bmax.x &&
                                ly >= bmin.y && ly <= bmax.y &&
                                lz >= bmin.z && lz <= bmax.z;
            outside_crop = inside == crop_inverse;
            if (outside_crop && !crop_desaturate)
                active = false;
        }

        // Depth filter (desaturate only, no culling)
        if (active && depth_filter_transform != nullptr) {
            const float3 dmin = *depth_filter_min;
            const float3 dmax = *depth_filter_max;
            const float* const d = depth_filter_transform;
            const float dx = d[0] * mean3d.x + d[1] * mean3d.y + d[2] * mean3d.z + d[3];
            const float dy = d[4] * mean3d.x + d[5] * mean3d.y + d[6] * mean3d.z + d[7];
            const float dz = d[8] * mean3d.x + d[9] * mean3d.y + d[10] * mean3d.z + d[11];
            const bool inside = dx >= dmin.x && dx <= dmax.x &&
                                dy >= dmin.y && dy <= dmax.y &&
                                dz >= dmin.z && dz <= dmax.z;
            if (!inside)
                outside_crop = true;
        }

        // Mark unselected nodes for desaturation
        if (active && desaturate_unselected && selected_node_mask != nullptr && num_selected_nodes > 0 && transform_indices != nullptr) {
            const int node_idx = transform_indices[primitive_idx];
            if (node_idx >= 0 && node_idx < num_selected_nodes && !selected_node_mask[node_idx]) {
                outside_crop = true;
            }
        }

        // z culling
        const float4 w2c_r3 = w2c[2];
        const float depth = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y + w2c_r3.z * mean3d.z + w2c_r3.w;
        if (depth < near_ || depth > far_)
            active = false;

        // early exit if whole warp is inactive
        if (__ballot_sync(0xffffffffu, active) == 0)
            return;

        // load opacity
        const float raw_opacity = raw_opacities[primitive_idx];
        const float opacity = 1.0f / (1.0f + expf(-raw_opacity));
        if (opacity < config::min_alpha_threshold)
            active = false;

        // compute 3d covariance from raw scale and rotation
        const float3 raw_scale = raw_scales[primitive_idx];
        const float3 variance = make_float3(expf(2.0f * raw_scale.x), expf(2.0f * raw_scale.y), expf(2.0f * raw_scale.z));
        auto [qr, qx, qy, qz] = raw_rotations[primitive_idx];
        const float qrr_raw = qr * qr, qxx_raw = qx * qx, qyy_raw = qy * qy, qzz_raw = qz * qz;
        const float q_norm_sq = qrr_raw + qxx_raw + qyy_raw + qzz_raw;
        if (q_norm_sq < 1e-8f)
            active = false;
        const float qxx = 2.0f * qxx_raw / q_norm_sq, qyy = 2.0f * qyy_raw / q_norm_sq, qzz = 2.0f * qzz_raw / q_norm_sq;
        const float qxy = 2.0f * qx * qy / q_norm_sq, qxz = 2.0f * qx * qz / q_norm_sq, qyz = 2.0f * qy * qz / q_norm_sq;
        const float qrx = 2.0f * qr * qx / q_norm_sq, qry = 2.0f * qr * qy / q_norm_sq, qrz = 2.0f * qr * qz / q_norm_sq;
        const mat3x3 rotation = {
            1.0f - (qyy + qzz), qxy - qrz, qry + qxz,
            qrz + qxy, 1.0f - (qxx + qzz), qyz - qrx,
            qxz - qry, qrx + qyz, 1.0f - (qxx + qyy)};
        const mat3x3 rotation_scaled = {
            rotation.m11 * variance.x, rotation.m12 * variance.y, rotation.m13 * variance.z,
            rotation.m21 * variance.x, rotation.m22 * variance.y, rotation.m23 * variance.z,
            rotation.m31 * variance.x, rotation.m32 * variance.y, rotation.m33 * variance.z};

        // Compute cov3d = R * S^2 * R^T (symmetric, store upper triangle)
        mat3x3_triu cov3d{
            rotation_scaled.m11 * rotation.m11 + rotation_scaled.m12 * rotation.m12 + rotation_scaled.m13 * rotation.m13,
            rotation_scaled.m11 * rotation.m21 + rotation_scaled.m12 * rotation.m22 + rotation_scaled.m13 * rotation.m23,
            rotation_scaled.m11 * rotation.m31 + rotation_scaled.m12 * rotation.m32 + rotation_scaled.m13 * rotation.m33,
            rotation_scaled.m21 * rotation.m21 + rotation_scaled.m22 * rotation.m22 + rotation_scaled.m23 * rotation.m23,
            rotation_scaled.m21 * rotation.m31 + rotation_scaled.m22 * rotation.m32 + rotation_scaled.m23 * rotation.m33,
            rotation_scaled.m31 * rotation.m31 + rotation_scaled.m32 * rotation.m32 + rotation_scaled.m33 * rotation.m33,
        };

        // Apply model transform to covariance: cov' = M * cov * M^T
        if (has_transform) {
            // Expand cov3d to full 3x3 symmetric matrix
            const mat3x3 cov_full = {
                cov3d.m11, cov3d.m12, cov3d.m13,
                cov3d.m12, cov3d.m22, cov3d.m23,
                cov3d.m13, cov3d.m23, cov3d.m33};

            // Compute M * cov (3x3 * 3x3)
            const mat3x3 m_cov = {
                model_rot.m11 * cov_full.m11 + model_rot.m12 * cov_full.m21 + model_rot.m13 * cov_full.m31,
                model_rot.m11 * cov_full.m12 + model_rot.m12 * cov_full.m22 + model_rot.m13 * cov_full.m32,
                model_rot.m11 * cov_full.m13 + model_rot.m12 * cov_full.m23 + model_rot.m13 * cov_full.m33,
                model_rot.m21 * cov_full.m11 + model_rot.m22 * cov_full.m21 + model_rot.m23 * cov_full.m31,
                model_rot.m21 * cov_full.m12 + model_rot.m22 * cov_full.m22 + model_rot.m23 * cov_full.m32,
                model_rot.m21 * cov_full.m13 + model_rot.m22 * cov_full.m23 + model_rot.m23 * cov_full.m33,
                model_rot.m31 * cov_full.m11 + model_rot.m32 * cov_full.m21 + model_rot.m33 * cov_full.m31,
                model_rot.m31 * cov_full.m12 + model_rot.m32 * cov_full.m22 + model_rot.m33 * cov_full.m32,
                model_rot.m31 * cov_full.m13 + model_rot.m32 * cov_full.m23 + model_rot.m33 * cov_full.m33};

            // Compute (M * cov) * M^T - result is symmetric, only compute upper triangle
            cov3d = {
                m_cov.m11 * model_rot.m11 + m_cov.m12 * model_rot.m12 + m_cov.m13 * model_rot.m13,
                m_cov.m11 * model_rot.m21 + m_cov.m12 * model_rot.m22 + m_cov.m13 * model_rot.m23,
                m_cov.m11 * model_rot.m31 + m_cov.m12 * model_rot.m32 + m_cov.m13 * model_rot.m33,
                m_cov.m21 * model_rot.m21 + m_cov.m22 * model_rot.m22 + m_cov.m23 * model_rot.m23,
                m_cov.m21 * model_rot.m31 + m_cov.m22 * model_rot.m32 + m_cov.m23 * model_rot.m33,
                m_cov.m31 * model_rot.m31 + m_cov.m32 * model_rot.m32 + m_cov.m33 * model_rot.m33};
        }

        // Camera-space position
        const float4 w2c_r1 = w2c[0];
        const float4 w2c_r2 = w2c[1];
        const float cam_x = w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w;
        const float cam_y = w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w;

        // EWA splatting: project 3D Gaussian to 2D
        const auto proj = orthographic
                              ? kernels::project_orthographic(cam_x, cam_y, cx, cy, ortho_scale, w2c_r1, w2c_r2)
                              : kernels::project_perspective(cam_x, cam_y, depth, fx, fy, cx, cy, w, h, w2c_r1, w2c_r2, w2c_r3);
        const float2 mean2d = proj.mean2d;
        float3 cov2d = kernels::project_cov3d(proj.jw_r1, proj.jw_r2, cov3d);
        cov2d.x += config::dilation;
        cov2d.z += config::dilation;
        const float determinant = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
        if (determinant < 1e-8f)
            active = false;
        const float3 conic = make_float3(
            cov2d.z / determinant,
            -cov2d.y / determinant,
            cov2d.x / determinant);

        // compute bounds
        const float power_threshold = logf(opacity * config::min_alpha_threshold_rcp);
        const float power_threshold_factor = sqrtf(2.0f * power_threshold);
        float extent_x = fmaxf(power_threshold_factor * sqrtf(cov2d.x) - 0.5f, 0.0f);
        float extent_y = fmaxf(power_threshold_factor * sqrtf(cov2d.z) - 0.5f, 0.0f);
        const uint4 screen_bounds = make_uint4(
            min(grid_width, static_cast<uint>(max(0, __float2int_rd((mean2d.x - extent_x) / static_cast<float>(config::tile_width))))),   // x_min
            min(grid_width, static_cast<uint>(max(0, __float2int_ru((mean2d.x + extent_x) / static_cast<float>(config::tile_width))))),   // x_max
            min(grid_height, static_cast<uint>(max(0, __float2int_rd((mean2d.y - extent_y) / static_cast<float>(config::tile_height))))), // y_min
            min(grid_height, static_cast<uint>(max(0, __float2int_ru((mean2d.y + extent_y) / static_cast<float>(config::tile_height)))))  // y_max
        );
        const uint n_touched_tiles_max = (screen_bounds.y - screen_bounds.x) * (screen_bounds.w - screen_bounds.z);
        if (n_touched_tiles_max == 0)
            active = false;

        // early exit if whole warp is inactive
        if (__ballot_sync(0xffffffffu, active) == 0)
            return;

        // compute exact number of tiles the primitive overlaps
        const uint n_touched_tiles = compute_exact_n_touched_tiles(
            mean2d, conic, screen_bounds,
            power_threshold, n_touched_tiles_max, active);

        // cooperative threads no longer needed
        if (n_touched_tiles == 0 || !active)
            return;

        // store results
        primitive_n_touched_tiles[primitive_idx] = n_touched_tiles;
        primitive_screen_bounds[primitive_idx] = make_ushort4(
            static_cast<ushort>(screen_bounds.x),
            static_cast<ushort>(screen_bounds.y),
            static_cast<ushort>(screen_bounds.z),
            static_cast<ushort>(screen_bounds.w));
        primitive_mean2d[primitive_idx] = mean2d;
        primitive_conic_opacity[primitive_idx] = make_float4(conic, opacity);
        float3 color = convert_sh_to_color(
            sh_coefficients_0, sh_coefficients_rest,
            mean3d, cam_position[0],
            primitive_idx, active_sh_bases, total_bases_sh_rest);

        // Brush hit test
        const bool selectable = !outside_crop;
        bool under_brush = false;
        if (brush_active) {
            const float dx = mean2d.x - brush_x;
            const float dy = mean2d.y - brush_y;
            under_brush = (dx * dx + dy * dy <= brush_radius_sq);
        }

        // Mark gaussians under brush for selection (add/remove determined by endStroke)
        if (under_brush && brush_selection_out != nullptr && selectable) {
            brush_selection_out[primitive_idx] = true;
        }

        // Saturation preview
        if (brush_saturation_mode && under_brush && selectable) {
            const float lum = 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
            const float sat = 1.0f + brush_saturation_amount;
            color.x = fmaxf(0.0f, fminf(1.0f, lum + sat * (color.x - lum)));
            color.y = fmaxf(0.0f, fminf(1.0f, lum + sat * (color.y - lum)));
            color.z = fmaxf(0.0f, fminf(1.0f, lum + sat * (color.z - lum)));
        }

        // Depth filter: dim outside gaussians
        if (outside_crop) {
            constexpr float DEPTH_FILTER_BRIGHTNESS = 0.25f;
            const float lum = 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
            color = make_float3(lum * DEPTH_FILTER_BRIGHTNESS, lum * DEPTH_FILTER_BRIGHTNESS, lum * DEPTH_FILTER_BRIGHTNESS);
        }

        // Ring mode hover detection
        if (selection_mode_rings && brush_active && hovered_depth_id != nullptr && selectable) {
            if (brush_x >= mean2d.x - extent_x && brush_x <= mean2d.x + extent_x &&
                brush_y >= mean2d.y - extent_y && brush_y <= mean2d.y + extent_y) {
                const float2 delta = make_float2(brush_x - mean2d.x, brush_y - mean2d.y);
                const float sigma = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
                                    conic.y * delta.x * delta.y;
                if (sigma >= 0.0f) {
                    const float hover_alpha = opacity * expf(-sigma);
                    if (hover_alpha >= config::min_alpha_threshold) {
                        const unsigned int depth_bits = __float_as_uint(depth);
                        const unsigned long long packed =
                            (static_cast<unsigned long long>(depth_bits) << 32) | primitive_idx;
                        atomicMin(hovered_depth_id, packed);
                    }
                }
            }
        }

        // Encode selection status
        uint8_t sel_status = 0;
        if (!brush_saturation_mode) {
            const uint8_t group_id = selection_mask ? selection_mask[primitive_idx] : 0;
            const bool is_committed = group_id > 0;
            const bool is_in_preview = brush_selection_out && brush_selection_out[primitive_idx];
            const bool is_ring_highlight = selection_mode_rings && selectable &&
                                           highlight_gaussian_id == static_cast<int>(primitive_idx);

            const bool is_preview = (is_in_preview && !is_committed && brush_add_mode) ||
                                    (is_in_preview && is_committed && !brush_add_mode) ||
                                    (under_brush && selectable && brush_add_mode && !is_committed) ||
                                    (under_brush && selectable && !brush_add_mode && is_committed) ||
                                    is_ring_highlight;

            sel_status = (group_id & SELECTION_GROUP_MASK) | (is_preview ? SELECTION_PREVIEW_FLAG : 0);
        }

        primitive_color[primitive_idx] = color;
        primitive_depth[primitive_idx] = depth;
        primitive_outside_crop[primitive_idx] = outside_crop;
        primitive_selection_status[primitive_idx] = sel_status;

        const uint offset = atomicAdd(n_visible_primitives, 1);
        const uint depth_key = __float_as_uint(depth);
        primitive_depth_keys[offset] = depth_key;
        primitive_indices[offset] = primitive_idx;
        atomicAdd(n_instances, n_touched_tiles);
    }

    __global__ void apply_depth_ordering_cu(
        const uint* primitive_indices_sorted,
        const uint* primitive_n_touched_tiles,
        uint* primitive_offset,
        const uint n_visible_primitives) {
        auto idx = cg::this_grid().thread_rank();
        if (idx >= n_visible_primitives)
            return;
        const uint primitive_idx = primitive_indices_sorted[idx];
        primitive_offset[idx] = primitive_n_touched_tiles[primitive_idx];
    }

    // based on https://github.com/r4dl/StopThePop-Rasterization/blob/d8cad09919ff49b11be3d693d1e71fa792f559bb/cuda_rasterizer/stopthepop/stopthepop_common.cuh#L325
    __global__ void create_instances_cu(
        const uint* primitive_indices_sorted,
        const uint* primitive_offsets,
        const ushort4* primitive_screen_bounds,
        const float2* primitive_mean2d,
        const float4* primitive_conic_opacity,
        ushort* instance_keys,
        uint* instance_primitive_indices,
        const uint grid_width,
        const uint n_visible_primitives) {
        auto block = cg::this_thread_block();
        auto warp = cg::tiled_partition<32u>(block);
        uint idx = cg::this_grid().thread_rank();

        bool active = true;
        if (idx >= n_visible_primitives) {
            active = false;
            idx = n_visible_primitives - 1;
        }

        if (__ballot_sync(0xffffffffu, active) == 0)
            return;

        const uint primitive_idx = primitive_indices_sorted[idx];

        const ushort4 screen_bounds = primitive_screen_bounds[primitive_idx];
        const uint screen_bounds_width = static_cast<uint>(screen_bounds.y - screen_bounds.x);
        const uint tile_count = static_cast<uint>(screen_bounds.w - screen_bounds.z) * screen_bounds_width;

        __shared__ ushort4 collected_screen_bounds[config::block_size_create_instances];
        __shared__ float2 collected_mean2d_shifted[config::block_size_create_instances];
        __shared__ float4 collected_conic_opacity[config::block_size_create_instances];
        collected_screen_bounds[block.thread_rank()] = screen_bounds;
        collected_mean2d_shifted[block.thread_rank()] = primitive_mean2d[primitive_idx] - 0.5f;
        collected_conic_opacity[block.thread_rank()] = primitive_conic_opacity[primitive_idx];

        uint current_write_offset = primitive_offsets[idx];

        if (active) {
            const float2 mean2d_shifted = collected_mean2d_shifted[block.thread_rank()];
            const float4 conic_opacity = collected_conic_opacity[block.thread_rank()];
            const float3 conic = make_float3(conic_opacity);
            const float power_threshold = logf(conic_opacity.w * config::min_alpha_threshold_rcp);

            for (uint instance_idx = 0; instance_idx < tile_count && instance_idx < config::n_sequential_threshold; instance_idx++) {
                const uint tile_y = screen_bounds.z + (instance_idx / screen_bounds_width);
                const uint tile_x = screen_bounds.x + (instance_idx % screen_bounds_width);
                if (will_primitive_contribute(mean2d_shifted, conic, tile_x, tile_y, power_threshold)) {
                    const ushort tile_key = static_cast<ushort>(tile_y * grid_width + tile_x);
                    instance_keys[current_write_offset] = tile_key;
                    instance_primitive_indices[current_write_offset] = primitive_idx;
                    current_write_offset++;
                }
            }
        }

        const uint lane_idx = cg::this_thread_block().thread_rank() % 32u;
        const uint warp_idx = cg::this_thread_block().thread_rank() / 32u;
        const uint lane_mask_allprev_excl = 0xffffffffu >> (32u - lane_idx);
        const int compute_cooperatively = active && tile_count > config::n_sequential_threshold;
        const uint remaining_threads = __ballot_sync(0xffffffffu, compute_cooperatively);
        if (remaining_threads == 0)
            return;

        const uint n_remaining_threads = __popc(remaining_threads);
        for (int n = 0; n < n_remaining_threads && n < 32; n++) {
            int current_lane = __fns(remaining_threads, 0, n + 1);
            uint primitive_idx_coop = __shfl_sync(0xffffffffu, primitive_idx, current_lane);
            uint current_write_offset_coop = __shfl_sync(0xffffffffu, current_write_offset, current_lane);

            const ushort4 screen_bounds_coop = collected_screen_bounds[warp.meta_group_rank() * 32 + current_lane];
            const uint screen_bounds_width_coop = static_cast<uint>(screen_bounds_coop.y - screen_bounds_coop.x);
            const uint tile_count_coop = screen_bounds_width_coop * static_cast<uint>(screen_bounds_coop.w - screen_bounds_coop.z);

            const float2 mean2d_shifted_coop = collected_mean2d_shifted[warp.meta_group_rank() * 32 + current_lane];
            const float4 conic_opacity_coop = collected_conic_opacity[warp.meta_group_rank() * 32 + current_lane];
            const float3 conic_coop = make_float3(conic_opacity_coop);
            const float power_threshold_coop = logf(conic_opacity_coop.w * config::min_alpha_threshold_rcp);

            const uint remaining_tile_count = tile_count_coop - config::n_sequential_threshold;
            const int n_iterations = div_round_up(remaining_tile_count, 32u);
            for (int i = 0; i < n_iterations; i++) {
                const int instance_idx = i * 32 + lane_idx + config::n_sequential_threshold;
                const int active_current = instance_idx < tile_count_coop;
                const uint tile_y = screen_bounds_coop.z + (instance_idx / screen_bounds_width_coop);
                const uint tile_x = screen_bounds_coop.x + (instance_idx % screen_bounds_width_coop);
                const uint write = active_current && will_primitive_contribute(mean2d_shifted_coop, conic_coop, tile_x, tile_y, power_threshold_coop);
                const uint write_ballot = __ballot_sync(0xffffffffu, write);
                const uint n_writes = __popc(write_ballot);
                const uint write_offset_current = __popc(write_ballot & lane_mask_allprev_excl);
                const uint write_offset = current_write_offset_coop + write_offset_current;
                if (write) {
                    const ushort tile_key = static_cast<ushort>(tile_y * grid_width + tile_x);
                    instance_keys[write_offset] = tile_key;
                    instance_primitive_indices[write_offset] = primitive_idx_coop;
                }
                current_write_offset_coop += n_writes;
            }

            __syncwarp();
        }
    }

    __global__ void extract_instance_ranges_cu(
        const ushort* instance_keys,
        uint2* tile_instance_ranges,
        const uint n_instances) {
        auto instance_idx = cg::this_grid().thread_rank();
        if (instance_idx >= n_instances)
            return;
        const ushort instance_tile_idx = instance_keys[instance_idx];
        if (instance_idx == 0)
            tile_instance_ranges[instance_tile_idx].x = 0;
        else {
            const ushort previous_instance_tile_idx = instance_keys[instance_idx - 1];
            if (instance_tile_idx != previous_instance_tile_idx) {
                tile_instance_ranges[previous_instance_tile_idx].y = instance_idx;
                tile_instance_ranges[instance_tile_idx].x = instance_idx;
            }
        }
        if (instance_idx == n_instances - 1)
            tile_instance_ranges[instance_tile_idx].y = n_instances;
    }

    __global__ void extract_bucket_counts(
        uint2* tile_instance_ranges,
        uint* tile_n_buckets,
        const uint n_tiles) {
        auto tile_idx = cg::this_grid().thread_rank();
        if (tile_idx >= n_tiles)
            return;
        const uint2 instance_range = tile_instance_ranges[tile_idx];
        const uint n_buckets = div_round_up(instance_range.y - instance_range.x, 32u);
        tile_n_buckets[tile_idx] = n_buckets;
    }

    __global__ void __launch_bounds__(config::block_size_blend) blend_cu(
        const uint2* tile_instance_ranges,
        const uint* instance_primitive_indices,
        const float2* primitive_mean2d,
        const float4* primitive_conic_opacity,
        const float3* primitive_color,
        const float* primitive_depth,
        const bool* primitive_outside_crop,
        const uint8_t* primitive_selection_status,
        float* image,
        float* alpha_map,
        float* depth_map,
        const uint width,
        const uint height,
        const uint grid_width,
        const bool show_rings,
        const float ring_width,
        const bool show_center_markers,
        const float selection_flash_intensity,
        const int* transform_indices,
        const bool* selected_node_mask,
        const int num_selected_nodes) {
        auto block = cg::this_thread_block();
        const dim3 group_index = block.group_index();
        const dim3 thread_index = block.thread_index();
        const uint thread_rank = block.thread_rank();
        const uint2 pixel_coords = make_uint2(group_index.x * config::tile_width + thread_index.x, group_index.y * config::tile_height + thread_index.y);
        const bool inside = pixel_coords.x < width && pixel_coords.y < height;
        const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y)) + 0.5f;

        const uint tile_idx = group_index.y * grid_width + group_index.x;
        const uint2 tile_range = tile_instance_ranges[tile_idx];
        const int n_points_total = tile_range.y - tile_range.x;

        __shared__ float2 collected_mean2d[config::block_size_blend];
        __shared__ float4 collected_conic_opacity[config::block_size_blend];
        __shared__ float3 collected_color[config::block_size_blend];
        __shared__ float collected_depth[config::block_size_blend];
        __shared__ bool collected_outside_crop[config::block_size_blend];
        __shared__ uint8_t collected_selection_status[config::block_size_blend];
        __shared__ uint collected_primitive_idx[config::block_size_blend];
        // initialize local storage
        float3 color_pixel = make_float3(0.0f);
        float depth_pixel = 1e10f; // Median depth (at 50% accumulated alpha)
        float accumulated_alpha = 0.0f;
        float transmittance = 1.0f;
        bool done = !inside;
        // collaborative loading and processing
        for (int n_points_remaining = n_points_total, current_fetch_idx = tile_range.x + thread_rank; n_points_remaining > 0; n_points_remaining -= config::block_size_blend, current_fetch_idx += config::block_size_blend) {
            if (__syncthreads_count(done) == config::block_size_blend)
                break;
            if (current_fetch_idx < tile_range.y) {
                const uint primitive_idx = instance_primitive_indices[current_fetch_idx];
                collected_mean2d[thread_rank] = primitive_mean2d[primitive_idx];
                collected_conic_opacity[thread_rank] = primitive_conic_opacity[primitive_idx];
                collected_color[thread_rank] = fmaxf(primitive_color[primitive_idx], 0.0f);
                collected_depth[thread_rank] = primitive_depth[primitive_idx];
                collected_outside_crop[thread_rank] = primitive_outside_crop[primitive_idx];
                collected_selection_status[thread_rank] = primitive_selection_status[primitive_idx];
                collected_primitive_idx[thread_rank] = primitive_idx;
            }
            block.sync();
            const int current_batch_size = min(config::block_size_blend, n_points_remaining);
            for (int j = 0; !done && j < current_batch_size; ++j) {
                const float4 conic_opacity = collected_conic_opacity[j];
                const float3 conic = make_float3(conic_opacity);
                const float2 delta = collected_mean2d[j] - pixel;
                const float opacity = conic_opacity.w;
                const float sigma_over_2 = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
                if (sigma_over_2 < 0.0f)
                    continue;
                float gaussian = expf(-sigma_over_2);

                // Early alpha check before ring mode
                float alpha = fminf(opacity * gaussian, config::max_fragment_alpha);
                if (alpha < config::min_alpha_threshold)
                    continue;

                const bool is_outside = collected_outside_crop[j];

                // Ring mode (skip for outside gaussians)
                if (show_rings && !is_outside) {
                    const float boundary_gaussian = config::min_alpha_threshold / opacity;
                    const float ring_outer = boundary_gaussian * (1.0f + ring_width * 10.0f);
                    const float ring_inner = boundary_gaussian * (1.0f - ring_width * 10.0f);
                    if (gaussian < ring_outer && gaussian > ring_inner) {
                        alpha = fminf(0.8f, config::max_fragment_alpha);
                    }
                }
                const float next_transmittance = transmittance * (1.0f - alpha);
                if (next_transmittance < config::transmittance_threshold) {
                    done = true;
                    continue;
                }

                float3 final_color = collected_color[j];
                const uint8_t sel_status = collected_selection_status[j];
                const uint8_t group_id = sel_status & SELECTION_GROUP_MASK;
                const bool is_preview = (sel_status & SELECTION_PREVIEW_FLAG) != 0;

                // Center markers mode
                if (show_center_markers && !is_outside) {
                    constexpr float INNER_RADIUS_SQ = 2.25f;
                    constexpr float OUTER_RADIUS_SQ = 6.25f;
                    constexpr float OUTLINE_DARKEN = 0.4f;

                    const float dist_sq = delta.x * delta.x + delta.y * delta.y;
                    if (dist_sq > OUTER_RADIUS_SQ)
                        continue;

                    final_color = is_preview ? config::SELECTION_COLOR_PREVIEW
                                             : config::SELECTION_GROUP_COLORS[group_id];
                    if (dist_sq > INNER_RADIUS_SQ)
                        final_color = final_color * OUTLINE_DARKEN;

                    color_pixel = final_color;
                    depth_pixel = collected_depth[j];  // Set depth for center markers
                    transmittance = 0.0f;
                    done = true;
                    continue;
                }

                // Selection coloring
                if (is_preview) {
                    final_color = lerp(final_color, config::SELECTION_COLOR_PREVIEW, SELECTION_PREVIEW_BLEND);
                } else if (group_id > 0) {
                    final_color = lerp(final_color, config::SELECTION_GROUP_COLORS[group_id], SELECTION_COMMITTED_BLEND);
                }

                // Selection flash highlight
                if (selection_flash_intensity > 0.0f && selected_node_mask != nullptr && transform_indices != nullptr) {
                    const int node_idx = transform_indices[collected_primitive_idx[j]];
                    if (node_idx >= 0 && node_idx < num_selected_nodes && selected_node_mask[node_idx]) {
                        constexpr float3 SELECTION_FLASH_COLOR = {1.0f, 0.95f, 0.6f};
                        final_color = lerp(final_color, SELECTION_FLASH_COLOR, selection_flash_intensity * 0.5f);
                    }
                }

                color_pixel += transmittance * alpha * final_color;

                // Median depth: pick depth when accumulated alpha crosses 50%
                // This gives the depth at the "middle" of the opacity distribution
                const float contribution = transmittance * alpha;
                const float new_accumulated = accumulated_alpha + contribution;
                if (accumulated_alpha < 0.5f && new_accumulated >= 0.5f) {
                    depth_pixel = collected_depth[j];
                }
                accumulated_alpha = new_accumulated;
                transmittance = next_transmittance;
            }
        }
        if (inside) {
            const int pixel_idx = width * pixel_coords.y + pixel_coords.x;
            const int n_pixels = width * height;

            // store results
            image[pixel_idx] = color_pixel.x;
            image[pixel_idx + n_pixels] = color_pixel.y;
            image[pixel_idx + n_pixels * 2] = color_pixel.z;
            alpha_map[pixel_idx] = 1.0f - transmittance;
            depth_map[pixel_idx] = depth_pixel;
        }
    }

} // namespace lfs::rendering::kernels::forward