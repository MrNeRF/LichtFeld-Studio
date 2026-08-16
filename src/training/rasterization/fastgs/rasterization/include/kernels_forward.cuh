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
#include <cstdint>
namespace cg = cooperative_groups;

namespace fast_lfs::rasterization::kernels::forward {

    __device__ __forceinline__ void report_forward_status(
        FastGSForwardStatus* __restrict__ status,
        const unsigned int flag,
        const uint source_index,
        const uint tile_index,
        const std::uint64_t value,
        const uint4 bounds,
        const uint expected_count,
        const uint actual_count) {
        report_fastgs_status(
            status,
            flag,
            source_index,
            tile_index,
            value,
            bounds,
            expected_count,
            actual_count);
    }

    __device__ __forceinline__ uint quantize_depth_key(float depth, const uint depth_bits) {
        if (depth_bits == 0)
            return 0;

        constexpr uint FLOAT32_FRACTION_BITS = 23;
        constexpr uint FLOAT32_FRACTION_MASK = (1u << FLOAT32_FRACTION_BITS) - 1u;
        constexpr uint FLOAT32_BELOW_TWO = 0x3fffffffu;

        float normalized_depth = (2.0f * depth + 1.0f) / (depth + 1.0f);
        normalized_depth = fminf(fmaxf(normalized_depth, 1.0f), __uint_as_float(FLOAT32_BELOW_TWO));
        const uint fraction = __float_as_uint(normalized_depth) & FLOAT32_FRACTION_MASK;
        return fraction >> (FLOAT32_FRACTION_BITS - depth_bits);
    }

    __device__ __forceinline__ InstanceKey make_instance_key(const uint tile_key, const uint depth_key, const uint depth_bits) {
        return (static_cast<InstanceKey>(tile_key) << depth_bits) | static_cast<InstanceKey>(depth_key);
    }

    __global__ void preprocess_cu(
        const float3* __restrict__ means,
        const float3* __restrict__ raw_scales,
        const float4* __restrict__ raw_rotations,
        const float* __restrict__ raw_opacities,
        const float3* __restrict__ sh_coefficients_0,
        const float4* __restrict__ sh_coefficients_rest, // float4 OR bitcast half/u16
        const float2* __restrict__ sh_value_bounds,      // null = fp32 or IEEE f16
        const uint sh_value_n_cells,
        const uint sh_value_bits, // 0=fp32, 16+bounds=q16, 16+null bounds=IEEE f16
        const float4* __restrict__ w2c,
        const float3* __restrict__ cam_position,
        uint* __restrict__ primitive_depth_keys,
        float* __restrict__ primitive_depths,
        std::uint64_t* __restrict__ primitive_n_touched_tiles,
        ushort4* __restrict__ primitive_screen_bounds,
        PackedMeanBBox* __restrict__ primitive_mean2d,
        float4* __restrict__ primitive_conic_opacity,
        float4* __restrict__ primitive_color,
        float3* __restrict__ primitive_normals,
        const uint n_primitives,
        const uint grid_width,
        const uint grid_height,
        const uint active_sh_bases,
        const uint sh_layout_slots,
        const float w,
        const float h,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        const float near_, // near and far are macros in windowns
        const float far_,
        const uint depth_bits,
        const bool mip_filter) {
        auto primitive_idx = cg::this_grid().thread_rank();
        bool active = true;
        if (primitive_idx >= n_primitives) {
            active = false;
            primitive_idx = n_primitives - 1;
        }

        if (active)
            primitive_n_touched_tiles[primitive_idx] = 0;

        // load 3d mean
        const float3 mean3d = means[primitive_idx];

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
        const float3 raw_scale = active ? raw_scales[primitive_idx] : make_float3(0.0f, 0.0f, 0.0f);
        const float3 clamped_scale = make_float3(
            fminf(raw_scale.x, config::max_raw_scale),
            fminf(raw_scale.y, config::max_raw_scale),
            fminf(raw_scale.z, config::max_raw_scale));
        const float3 variance = make_float3(expf(2.0f * clamped_scale.x), expf(2.0f * clamped_scale.y), expf(2.0f * clamped_scale.z));
        const float4 raw_rotation = raw_rotations[primitive_idx];
        const float qr = raw_rotation.x;
        const float qx = raw_rotation.y;
        const float qy = raw_rotation.z;
        const float qz = raw_rotation.w;
        const float qrr_raw = qr * qr, qxx_raw = qx * qx, qyy_raw = qy * qy, qzz_raw = qz * qz;
        const float q_norm_sq = qrr_raw + qxx_raw + qyy_raw + qzz_raw;
        if (q_norm_sq < 1e-8f)
            active = false;
        if (__ballot_sync(0xffffffffu, active) == 0)
            return;
        const float q_norm_sq_safe = fmaxf(q_norm_sq, 1e-8f);
        const float qxx = 2.0f * qxx_raw / q_norm_sq_safe, qyy = 2.0f * qyy_raw / q_norm_sq_safe, qzz = 2.0f * qzz_raw / q_norm_sq_safe;
        const float qxy = 2.0f * qx * qy / q_norm_sq_safe, qxz = 2.0f * qx * qz / q_norm_sq_safe, qyz = 2.0f * qy * qz / q_norm_sq_safe;
        const float qrx = 2.0f * qr * qx / q_norm_sq_safe, qry = 2.0f * qr * qy / q_norm_sq_safe, qrz = 2.0f * qr * qz / q_norm_sq_safe;
        const mat3x3 rotation = {
            1.0f - (qyy + qzz), qxy - qrz, qry + qxz,
            qrz + qxy, 1.0f - (qxx + qzz), qyz - qrx,
            qxz - qry, qrx + qyz, 1.0f - (qxx + qyy)};
        const mat3x3 rotation_scaled = {
            rotation.m11 * variance.x, rotation.m12 * variance.y, rotation.m13 * variance.z,
            rotation.m21 * variance.x, rotation.m22 * variance.y, rotation.m23 * variance.z,
            rotation.m31 * variance.x, rotation.m32 * variance.y, rotation.m33 * variance.z};
        const mat3x3_triu cov3d{
            rotation_scaled.m11 * rotation.m11 + rotation_scaled.m12 * rotation.m12 + rotation_scaled.m13 * rotation.m13,
            rotation_scaled.m11 * rotation.m21 + rotation_scaled.m12 * rotation.m22 + rotation_scaled.m13 * rotation.m23,
            rotation_scaled.m11 * rotation.m31 + rotation_scaled.m12 * rotation.m32 + rotation_scaled.m13 * rotation.m33,
            rotation_scaled.m21 * rotation.m21 + rotation_scaled.m22 * rotation.m22 + rotation_scaled.m23 * rotation.m23,
            rotation_scaled.m21 * rotation.m31 + rotation_scaled.m22 * rotation.m32 + rotation_scaled.m23 * rotation.m33,
            rotation_scaled.m31 * rotation.m31 + rotation_scaled.m32 * rotation.m32 + rotation_scaled.m33 * rotation.m33,
        };

        // compute 2d mean in normalized image coordinates
        const float4 w2c_r1 = w2c[0];
        const float x = (w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w) / depth;
        const float4 w2c_r2 = w2c[1];
        const float y = (w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w) / depth;

        // ewa splatting
        const float clip_left = (-0.15f * w - cx) / fx;
        const float clip_right = (1.15f * w - cx) / fx;
        const float clip_top = (-0.15f * h - cy) / fy;
        const float clip_bottom = (1.15f * h - cy) / fy;
        const float tx = clamp(x, clip_left, clip_right);
        const float ty = clamp(y, clip_top, clip_bottom);
        const float j11 = fx / depth;
        const float j13 = -j11 * tx;
        const float j22 = fy / depth;
        const float j23 = -j22 * ty;
        const float3 jw_r1 = make_float3(
            j11 * w2c_r1.x + j13 * w2c_r3.x,
            j11 * w2c_r1.y + j13 * w2c_r3.y,
            j11 * w2c_r1.z + j13 * w2c_r3.z);
        const float3 jw_r2 = make_float3(
            j22 * w2c_r2.x + j23 * w2c_r3.x,
            j22 * w2c_r2.y + j23 * w2c_r3.y,
            j22 * w2c_r2.z + j23 * w2c_r3.z);
        const float3 jwc_r1 = make_float3(
            jw_r1.x * cov3d.m11 + jw_r1.y * cov3d.m12 + jw_r1.z * cov3d.m13,
            jw_r1.x * cov3d.m12 + jw_r1.y * cov3d.m22 + jw_r1.z * cov3d.m23,
            jw_r1.x * cov3d.m13 + jw_r1.y * cov3d.m23 + jw_r1.z * cov3d.m33);
        const float3 jwc_r2 = make_float3(
            jw_r2.x * cov3d.m11 + jw_r2.y * cov3d.m12 + jw_r2.z * cov3d.m13,
            jw_r2.x * cov3d.m12 + jw_r2.y * cov3d.m22 + jw_r2.z * cov3d.m23,
            jw_r2.x * cov3d.m13 + jw_r2.y * cov3d.m23 + jw_r2.z * cov3d.m33);
        float3 cov2d = make_float3(dot(jwc_r1, jw_r1), dot(jwc_r1, jw_r2), dot(jwc_r2, jw_r2));

        // Mip filter: use smaller dilation and compensate opacity
        const float det_raw = mip_filter ? fmaxf(cov2d.x * cov2d.z - cov2d.y * cov2d.y, 0.0f) : 0.0f;
        const float kernel_size = mip_filter ? config::dilation_mip_filter : config::dilation;
        cov2d.x += kernel_size;
        cov2d.z += kernel_size;
        const float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
        if (det < config::min_cov2d_determinant)
            active = false;
        const float det_rcp = 1.0f / det;
        const float output_opacity = mip_filter ? opacity * sqrtf(det_raw * det_rcp) : opacity;
        if (output_opacity < config::min_alpha_threshold)
            active = false;

        const float3 conic = make_float3(cov2d.z * det_rcp, -cov2d.y * det_rcp, cov2d.x * det_rcp);
        const float2 mean2d = make_float2(x * fx + cx, y * fy + cy);

        // Compute bounds
        const float power_threshold = logf(output_opacity * config::min_alpha_threshold_rcp);
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
            power_threshold, active);

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
        // Conservative pixel AABB of the contribution ellipse (pixel centers at +0.5).
        // Full extent (no -0.5) so warp cull never drops a contributing splat.
        const float extent_x_full = power_threshold_factor * sqrtf(cov2d.x);
        const float extent_y_full = power_threshold_factor * sqrtf(cov2d.z);
        // i contributes when mean - extent <= i+0.5 <= mean + extent
        // => i in [ceil(mean-extent-0.5), floor(mean+extent-0.5)]
        // Store half-open [x_min, x_max) with conservative floor/ceil.
        const int px_min = max(0, __float2int_rd(mean2d.x - extent_x_full - 0.5f));
        const int py_min = max(0, __float2int_rd(mean2d.y - extent_y_full - 0.5f));
        const int px_max = max(px_min, __float2int_ru(mean2d.x + extent_x_full - 0.5f) + 1);
        const int py_max = max(py_min, __float2int_ru(mean2d.y + extent_y_full - 0.5f) + 1);
        PackedMeanBBox packed{};
        packed.mean2d = mean2d;
        packed.pixel_bbox = make_ushort4(
            static_cast<ushort>(min(px_min, 65535)),
            static_cast<ushort>(min(px_max, 65535)),
            static_cast<ushort>(min(py_min, 65535)),
            static_cast<ushort>(min(py_max, 65535)));
        primitive_mean2d[primitive_idx] = packed;
        primitive_conic_opacity[primitive_idx] = make_float4(conic, output_opacity);
        // Pad float3 color → float4 for 128-bit loads (w unused).
        const float3 sh_color = convert_sh_to_color(
            sh_coefficients_0, sh_coefficients_rest,
            mean3d, cam_position[0],
            primitive_idx, active_sh_bases, sh_layout_slots,
            sh_value_bounds, sh_value_n_cells, sh_value_bits);
        primitive_color[primitive_idx] = make_float4(sh_color, 0.0f);
        primitive_depth_keys[primitive_idx] = quantize_depth_key(depth, depth_bits);
        primitive_depths[primitive_idx] = depth;

        // Camera-space unit normal: rotation column of the smallest axis, oriented toward the camera.
        if (primitive_normals != nullptr) {
            const float3 axis = (variance.x <= variance.y && variance.x <= variance.z)
                                    ? make_float3(rotation.m11, rotation.m21, rotation.m31)
                                : (variance.y <= variance.z)
                                    ? make_float3(rotation.m12, rotation.m22, rotation.m32)
                                    : make_float3(rotation.m13, rotation.m23, rotation.m33);
            const float3 view_dir = mean3d - cam_position[0];
            const float3 normal_world = dot(axis, view_dir) > 0.0f
                                            ? make_float3(-axis.x, -axis.y, -axis.z)
                                            : axis;
            primitive_normals[primitive_idx] = make_float3(
                w2c_r1.x * normal_world.x + w2c_r1.y * normal_world.y + w2c_r1.z * normal_world.z,
                w2c_r2.x * normal_world.x + w2c_r2.y * normal_world.y + w2c_r2.z * normal_world.z,
                w2c_r3.x * normal_world.x + w2c_r3.y * normal_world.y + w2c_r3.z * normal_world.z);
        }
    }

    // based on https://github.com/r4dl/StopThePop-Rasterization/blob/d8cad09919ff49b11be3d693d1e71fa792f559bb/cuda_rasterizer/stopthepop/stopthepop_common.cuh#L325
    __global__ void create_instances_cu(
        const std::uint64_t* __restrict__ primitive_n_touched_tiles,
        const std::uint64_t* __restrict__ primitive_offsets,
        const uint* __restrict__ primitive_depth_keys,
        const ushort4* __restrict__ primitive_screen_bounds,
        const PackedMeanBBox* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        InstanceKey* __restrict__ instance_keys,
        uint* __restrict__ instance_primitive_indices,
        FastGSForwardStatus* __restrict__ status,
        const uint grid_width,
        const uint depth_bits,
        const uint n_primitives,
        const uint max_instances) {
        uint idx = cg::this_grid().thread_rank();

        bool active = true;
        if (idx >= n_primitives) {
            active = false;
            idx = n_primitives - 1;
        }

        const uint primitive_idx = idx;
        const uint n_touched_tiles = active ? static_cast<uint>(primitive_n_touched_tiles[primitive_idx]) : 0;
        active = active && n_touched_tiles > 0;

        if (__ballot_sync(0xffffffffu, active) == 0)
            return;

        const ushort4 screen_bounds = active ? primitive_screen_bounds[primitive_idx] : make_ushort4(0, 0, 0, 0);
        const uint4 diagnostic_bounds = make_uint4(screen_bounds.x, screen_bounds.y, screen_bounds.z, screen_bounds.w);
        const uint depth_key = active ? primitive_depth_keys[primitive_idx] : 0;
        const uint write_offset_end_raw = active ? static_cast<uint>(primitive_offsets[idx]) : 0;
        // clamp to sort-buffer capacity; flag overflow for host re-run.
        if (active && write_offset_end_raw > max_instances) {
            report_forward_status(
                status,
                kFastGSForwardStatusSortCapacityOverflow,
                primitive_idx,
                0,
                write_offset_end_raw,
                diagnostic_bounds,
                max_instances,
                write_offset_end_raw);
        }
        const uint write_offset_end =
            write_offset_end_raw > max_instances ? max_instances : write_offset_end_raw;

        const float2 mean2d_shifted = active ? primitive_mean2d[primitive_idx].mean2d - 0.5f : make_float2(0.0f, 0.0f);
        const float4 conic_opacity_loaded = active ? primitive_conic_opacity[primitive_idx] : make_float4(0.0f, 0.0f, 0.0f, config::min_alpha_threshold);
        const float3 conic = make_float3(conic_opacity_loaded);
        const float power_threshold_precomputed = logf(conic_opacity_loaded.w * config::min_alpha_threshold_rcp);
        const float radius_sq = 2.0f * power_threshold_precomputed;

        uint current_write_offset = idx == 0 ? 0 : static_cast<uint>(primitive_offsets[idx - 1]);
        if (current_write_offset > max_instances) {
            current_write_offset = max_instances;
        }
        if (active) {
            const uint screen_bounds_width = static_cast<uint>(screen_bounds.y - screen_bounds.x);
            const uint screen_bounds_height = static_cast<uint>(screen_bounds.w - screen_bounds.z);

            if (screen_bounds_height <= screen_bounds_width) {
                for (uint tile_y = screen_bounds.z; tile_y < screen_bounds.w; tile_y++) {
                    const float y0 = static_cast<float>(tile_y * config::tile_height) - mean2d_shifted.y;
                    const float y1 = y0 + static_cast<float>(config::tile_height);
                    const float2 bound = ellipse_range_bound(conic, radius_sq, y0, y1);
                    const uint min_x = floor_tile_clamped(bound.x + mean2d_shifted.x, screen_bounds.x, screen_bounds.y, config::tile_width);
                    const uint max_x = ceil_tile_clamped(bound.y + mean2d_shifted.x, screen_bounds.x, screen_bounds.y, config::tile_width);
                    for (uint tile_x = min_x; tile_x < max_x && current_write_offset < write_offset_end; tile_x++) {
                        const uint tile_key = tile_y * grid_width + tile_x;
                        instance_keys[current_write_offset] = make_instance_key(tile_key, depth_key, depth_bits);
                        instance_primitive_indices[current_write_offset] = primitive_idx;
                        current_write_offset++;
                    }
                }
            } else {
                const float3 conic_transposed = make_float3(conic.z, conic.y, conic.x);
                for (uint tile_x = screen_bounds.x; tile_x < screen_bounds.y; tile_x++) {
                    const float x0 = static_cast<float>(tile_x * config::tile_width) - mean2d_shifted.x;
                    const float x1 = x0 + static_cast<float>(config::tile_width);
                    const float2 bound = ellipse_range_bound(conic_transposed, radius_sq, x0, x1);
                    const uint min_y = floor_tile_clamped(bound.x + mean2d_shifted.y, screen_bounds.z, screen_bounds.w, config::tile_height);
                    const uint max_y = ceil_tile_clamped(bound.y + mean2d_shifted.y, screen_bounds.z, screen_bounds.w, config::tile_height);
                    for (uint tile_y = min_y; tile_y < max_y && current_write_offset < write_offset_end; tile_y++) {
                        const uint tile_key = tile_y * grid_width + tile_x;
                        instance_keys[current_write_offset] = make_instance_key(tile_key, depth_key, depth_bits);
                        instance_primitive_indices[current_write_offset] = primitive_idx;
                        current_write_offset++;
                    }
                }
            }

            if (current_write_offset != write_offset_end) {
                report_forward_status(
                    status,
                    kFastGSForwardStatusInstanceWriteMismatch,
                    primitive_idx,
                    0,
                    current_write_offset,
                    diagnostic_bounds,
                    write_offset_end,
                    current_write_offset);
            }
        }
    }

    // n_instances_upper sizes the grid; real count is read from d_n_instances
    __global__ void extract_instance_ranges_cu(
        const InstanceKey* instance_keys,
        uint2* tile_instance_ranges,
        FastGSForwardStatus* __restrict__ status,
        const uint depth_bits,
        const uint n_tiles,
        const uint n_instances_upper,
        const std::uint64_t* __restrict__ d_n_instances) {
        auto instance_idx = cg::this_grid().thread_rank();
        if (instance_idx >= n_instances_upper)
            return;
        const uint n_instances = d_n_instances != nullptr
                                     ? static_cast<uint>(*d_n_instances)
                                     : n_instances_upper;
        if (instance_idx >= n_instances)
            return;
        const uint instance_tile_idx = static_cast<uint>(instance_keys[instance_idx] >> depth_bits);
        if (instance_tile_idx >= n_tiles) {
            report_forward_status(
                status,
                kFastGSForwardStatusTileIndexOutOfRange,
                instance_idx,
                instance_tile_idx,
                instance_tile_idx,
                make_uint4(0, 0, 0, 0),
                n_tiles,
                0);
            return;
        }
        if (instance_idx == 0)
            tile_instance_ranges[instance_tile_idx].x = 0;
        else {
            const uint previous_instance_tile_idx = static_cast<uint>(instance_keys[instance_idx - 1] >> depth_bits);
            if (previous_instance_tile_idx >= n_tiles) {
                report_forward_status(
                    status,
                    kFastGSForwardStatusTileIndexOutOfRange,
                    instance_idx - 1,
                    previous_instance_tile_idx,
                    previous_instance_tile_idx,
                    make_uint4(0, 0, 0, 0),
                    n_tiles,
                    0);
                return;
            }
            if (instance_tile_idx != previous_instance_tile_idx) {
                tile_instance_ranges[previous_instance_tile_idx].y = instance_idx;
                tile_instance_ranges[instance_tile_idx].x = instance_idx;
            }
        }
        if (instance_idx == n_instances - 1)
            tile_instance_ranges[instance_tile_idx].y = n_instances;
    }

    // Warp-level sub-tile culling + 128-bit splat layout for forward blend.
    // Source: Yang, Drettakis, Bernstein, "Warp-Level Culling for Efficient Blending
    // in 3D Gaussian Splatting", ACM CGIT 9(4):54, 2026, doi:10.1145/3820019.
    //
    // 16×16 tile → 8×4 sub-tiles (32 px = 1 warp). Warp ballots splat×sub-tile AABB
    // intersection 32-at-a-time; non-intersecting splats skip full data/alpha eval.
    // Output is bit-identical to the pre-cull kernel when the AABB is conservative.
    //
    // warp_cull_mode: 0 = enabled (production), 1 = disabled (mask all-1s, reference),
    //                 2 = deliberately incorrect empty mask for negative coverage.
    // blend_batch_size_runtime: multiple of 32 in [32, block_size_blend]; 0 → config default.
    template <bool kRenderNormal>
    __global__ void __launch_bounds__(config::block_size_blend) blend_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ instance_primitive_indices,
        const PackedMeanBBox* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        const float4* __restrict__ primitive_color,
        const float* __restrict__ primitive_depths,
        const float3* __restrict__ primitive_normals,
        float* __restrict__ image,
        float* __restrict__ alpha_map,
        float* __restrict__ depth_map,
        float* __restrict__ normal_map,
        uint* __restrict__ tile_n_contributions,
        float* __restrict__ tile_final_transmittance,
        // fuse background compose into the final write.
        // out = fg + T * bg  (equivalent to fg + (1-alpha)*bg).
        // bg_image (CHW [3,H,W]) wins over solid bg_color [3]; either may be null → black.
        const float* __restrict__ bg_color,
        const float* __restrict__ bg_image,
        const uint width,
        const uint height,
        const uint grid_width,
        const int warp_cull_mode,
        const int blend_batch_size_runtime) {
        auto block = cg::this_thread_block();
        const dim3 group_index = block.group_index();
        const uint thread_rank = block.thread_rank();

        // 8×4 sub-tile mapping: 8 warps tile a 16×16 block (2 cols × 4 rows).
        static_assert(config::tile_width == 16 && config::tile_height == 16, "warp sub-tile layout assumes 16×16 tiles");
        static_assert(config::warp_subtile_width == 8 && config::warp_subtile_height == 4, "warp sub-tile is 8×4");
        static_assert(config::block_size_blend == 256, "block_size_blend must be 256");
        const uint lane_id = thread_rank & 31u;
        const uint warp_id = thread_rank >> 5;
        const uint subtile_origin_x = (warp_id & 1u) * static_cast<uint>(config::warp_subtile_width);
        const uint subtile_origin_y = (warp_id >> 1) * static_cast<uint>(config::warp_subtile_height);
        const uint local_x = lane_id & 7u;
        const uint local_y = lane_id >> 3;
        const uint tile_local_x = subtile_origin_x + local_x;
        const uint tile_local_y = subtile_origin_y + local_y;
        // Geometric pixel rank (dy*16+dx) for transmittance / backward layout.
        const uint pixel_rank = tile_local_y * static_cast<uint>(config::tile_width) + tile_local_x;

        const uint2 pixel_coords = make_uint2(
            group_index.x * config::tile_width + tile_local_x,
            group_index.y * config::tile_height + tile_local_y);
        const bool inside = pixel_coords.x < width && pixel_coords.y < height;
        const float2 pixel = make_float2(__uint2float_rn(pixel_coords.x), __uint2float_rn(pixel_coords.y)) + 0.5f;

        // Absolute sub-tile AABB (half-open) for ballot culling.
        const uint sub_x0 = group_index.x * config::tile_width + subtile_origin_x;
        const uint sub_y0 = group_index.y * config::tile_height + subtile_origin_y;
        const uint sub_x1 = sub_x0 + static_cast<uint>(config::warp_subtile_width);
        const uint sub_y1 = sub_y0 + static_cast<uint>(config::warp_subtile_height);

        const uint tile_idx = group_index.y * grid_width + group_index.x;
        const uint2 tile_range = tile_instance_ranges[tile_idx];
        const int n_points_total = static_cast<int>(tile_range.y - tile_range.x);

        int batch_size = blend_batch_size_runtime > 0 ? blend_batch_size_runtime : config::blend_batch_size;
        // Clamp to legal warp-multiple range without diverging the block.
        batch_size = max(32, min(batch_size, config::block_size_blend));
        batch_size = batch_size & ~31; // force multiple of 32

        // Shared memory: sized to max batch (256); only first batch_size slots used.
        // 128-bit packed geom + float4 color (padded) + float4 conic_opacity.
        __shared__ PackedMeanBBox collected_geom[config::block_size_blend];
        __shared__ float4 collected_conic_opacity[config::block_size_blend];
        __shared__ float4 collected_color[config::block_size_blend];
        __shared__ float collected_depth[config::block_size_blend];
        __shared__ float3 collected_normal[kRenderNormal ? config::block_size_blend : 1];

        float3 color_pixel = make_float3(0.0f);
        float depth_pixel = 0.0f;
        float3 normal_pixel = make_float3(0.0f);
        float transmittance = 1.0f;
        uint n_possible_contributions = 0;
        uint n_contributions = 0;
        bool done = !inside;

        for (int n_points_remaining = n_points_total, batch_base = 0;
             n_points_remaining > 0;
             n_points_remaining -= batch_size, batch_base += batch_size) {
            if (__syncthreads_count(done) == config::block_size_blend)
                break;

            // Collaborative fetch: first batch_size threads load one splat each.
            if (static_cast<int>(thread_rank) < batch_size) {
                const int fetch_idx = static_cast<int>(tile_range.x) + batch_base + static_cast<int>(thread_rank);
                if (fetch_idx < static_cast<int>(tile_range.y)) {
                    const uint primitive_idx = instance_primitive_indices[fetch_idx];
                    collected_geom[thread_rank] = primitive_mean2d[primitive_idx];
                    collected_conic_opacity[thread_rank] = primitive_conic_opacity[primitive_idx];
                    // Clamp RGB; keep w=0 padding.
                    const float4 raw_c = primitive_color[primitive_idx];
                    const float3 clamped = fminf(fmaxf(make_float3(raw_c), 0.0f), config::max_blend_color);
                    collected_color[thread_rank] = make_float4(clamped, 0.0f);
                    collected_depth[thread_rank] = primitive_depths[primitive_idx];
                    if constexpr (kRenderNormal) {
                        collected_normal[thread_rank] = primitive_normals[primitive_idx];
                    }
                }
            }
            block.sync();

            const int current_batch_size = min(batch_size, n_points_remaining);

            // Warp iterates the batch 32 splats at a time with ballot culling.
            for (int j_base = 0; j_base < current_batch_size; j_base += 32) {
                const int j_test = j_base + static_cast<int>(lane_id);
                bool intersects = false;
                if (j_test < current_batch_size) {
                    const ushort4 bb = collected_geom[j_test].pixel_bbox;
                    // Half-open AABB overlap: [bb.x, bb.y) × [bb.z, bb.w) vs sub-tile.
                    intersects = (static_cast<uint>(bb.x) < sub_x1) &&
                                 (static_cast<uint>(bb.y) > sub_x0) &&
                                 (static_cast<uint>(bb.z) < sub_y1) &&
                                 (static_cast<uint>(bb.w) > sub_y0);
                }
                unsigned mask = __ballot_sync(0xffffffffu, intersects);
                // Test hooks: 1 = disable cull (process all), 2 = empty mask (wrong).
                if (warp_cull_mode == 1) {
                    mask = 0xffffffffu;
                } else if (warp_cull_mode == 2) {
                    mask = 0u;
                }

                // Per-pixel composite; always advance n_possible for every splat index
                // so last_contributor stays bit-identical to the uncull path.
                for (int k = 0; !done && k < 32; ++k) {
                    const int j = j_base + k;
                    if (j >= current_batch_size)
                        break;
                    n_possible_contributions++;
                    if (((mask >> k) & 1u) == 0u)
                        continue;

                    const float4 conic_opacity = collected_conic_opacity[j];
                    const float3 conic = make_float3(conic_opacity);
                    const float2 delta = collected_geom[j].mean2d - pixel;
                    const float opacity = conic_opacity.w;
                    const float sigma_over_2 = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
                    if (sigma_over_2 < 0.0f)
                        continue;
                    const float gaussian = expf(-sigma_over_2);
                    const float alpha = fminf(opacity * gaussian, config::max_fragment_alpha);
                    if (alpha < config::min_alpha_threshold)
                        continue;
                    const float weight = transmittance * alpha;
                    const float4 c = collected_color[j];
                    color_pixel += weight * make_float3(c);
                    depth_pixel += weight * collected_depth[j];
                    if constexpr (kRenderNormal) {
                        normal_pixel += weight * collected_normal[j];
                    }
                    transmittance *= (1.0f - alpha);
                    n_contributions = n_possible_contributions;
                    if (transmittance < config::transmittance_threshold) {
                        done = true;
                    }
                }
            }
        }

        if (inside) {
            const int pixel_idx = width * pixel_coords.y + pixel_coords.x;
            const int n_pixels = width * height;
            float3 bg = make_float3(0.0f, 0.0f, 0.0f);
            if (bg_image != nullptr) {
                bg = make_float3(bg_image[pixel_idx],
                                 bg_image[pixel_idx + n_pixels],
                                 bg_image[pixel_idx + 2 * n_pixels]);
            } else if (bg_color != nullptr) {
                bg = make_float3(bg_color[0], bg_color[1], bg_color[2]);
            }
            image[pixel_idx] = color_pixel.x + transmittance * bg.x;
            image[pixel_idx + n_pixels] = color_pixel.y + transmittance * bg.y;
            image[pixel_idx + n_pixels * 2] = color_pixel.z + transmittance * bg.z;
            alpha_map[pixel_idx] = 1.0f - transmittance;
            depth_map[pixel_idx] = depth_pixel;
            if constexpr (kRenderNormal) {
                normal_map[pixel_idx] = normal_pixel.x;
                normal_map[pixel_idx + n_pixels] = normal_pixel.y;
                normal_map[pixel_idx + n_pixels * 2] = normal_pixel.z;
            }
            tile_n_contributions[pixel_idx] = n_contributions;
        }
        tile_final_transmittance[tile_idx * config::block_size_blend + pixel_rank] = inside ? transmittance : 1.0f;
    }

} // namespace fast_lfs::rasterization::kernels::forward
