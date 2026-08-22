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
        const float clip_left,
        const float clip_right,
        const float clip_top,
        const float clip_bottom,
        const float near_, // near and far are macros in windowns
        const float far_,
        const uint depth_bits,
        const bool mip_filter) {
        (void)w;
        (void)h;
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
        const float inv_q_norm_sq = 2.0f / q_norm_sq_safe;
        const float qxx = qxx_raw * inv_q_norm_sq, qyy = qyy_raw * inv_q_norm_sq, qzz = qzz_raw * inv_q_norm_sq;
        const float qxy = qx * qy * inv_q_norm_sq, qxz = qx * qz * inv_q_norm_sq, qyz = qy * qz * inv_q_norm_sq;
        const float qrx = qr * qx * inv_q_norm_sq, qry = qr * qy * inv_q_norm_sq, qrz = qr * qz * inv_q_norm_sq;
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
        const float inv_depth = 1.0f / depth;
        const float4 w2c_r1 = w2c[0];
        const float x = (w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w) * inv_depth;
        const float4 w2c_r2 = w2c[1];
        const float y = (w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w) * inv_depth;

        // ewa splatting (clip box is grid-uniform; computed once on the host)
        const float tx = clamp(x, clip_left, clip_right);
        const float ty = clamp(y, clip_top, clip_bottom);
        const float j11 = fx * inv_depth;
        const float j13 = -j11 * tx;
        const float j22 = fy * inv_depth;
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
        const uint4 screen_bounds = compute_screen_tile_bounds(
            mean2d, extent_x, extent_y, grid_width, grid_height);
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

    __device__ __forceinline__ void emit_ellipse_tile_span(
        const uint2 span,
        const uint scan_index,
        const bool along_x,
        const uint grid_width,
        const uint depth_key,
        const uint depth_bits,
        const uint primitive_idx,
        uint& write_at,
        const uint write_end,
        InstanceKey* __restrict__ instance_keys,
        uint* __restrict__ instance_primitive_indices) {
        for (uint t = span.x; t < span.y && write_at < write_end; t++) {
            const uint tile_key = along_x ? (scan_index * grid_width + t) : (t * grid_width + scan_index);
            instance_keys[write_at] = make_instance_key(tile_key, depth_key, depth_bits);
            instance_primitive_indices[write_at] = primitive_idx;
            write_at++;
        }
    }

    // Long-tail primitives: 32 lanes cooperate over rows of one source lane
    // at a time. Isolated so the serial emit path does not inherit its
    // register footprint.
    __device__ __noinline__ void create_instances_coop_warp(
        const bool active,
        const uint n_scan,
        const uint scan0,
        const uint cross0,
        const uint cross1,
        const uint write_offset_end,
        const uint current_write_offset,
        const uint primitive_idx,
        const uint depth_key,
        const bool scan_along_x,
        const float2 mean2d_shifted,
        const float3 conic,
        const float radius_sq,
        const uint grid_width,
        const uint depth_bits,
        const uint4 diagnostic_bounds,
        InstanceKey* __restrict__ instance_keys,
        uint* __restrict__ instance_primitive_indices,
        FastGSForwardStatus* __restrict__ status) {
        const uint lane = threadIdx.x & 31u;
        for (int src = 0; src < 32; ++src) {
            const bool src_active = __shfl_sync(0xffffffffu, static_cast<unsigned>(active), src) != 0u;
            const uint src_n_scan = __shfl_sync(0xffffffffu, n_scan, src);
            if (!src_active || src_n_scan == 0u)
                continue;
            const uint src_scan0 = __shfl_sync(0xffffffffu, scan0, src);
            const uint src_cross0 = __shfl_sync(0xffffffffu, cross0, src);
            const uint src_cross1 = __shfl_sync(0xffffffffu, cross1, src);
            const uint src_write_end = __shfl_sync(0xffffffffu, write_offset_end, src);
            const uint src_write = __shfl_sync(0xffffffffu, current_write_offset, src);
            const uint src_prim = __shfl_sync(0xffffffffu, primitive_idx, src);
            const uint src_dkey = __shfl_sync(0xffffffffu, depth_key, src);
            const bool src_along_x = __shfl_sync(0xffffffffu, static_cast<unsigned>(scan_along_x), src) != 0u;
            const float2 src_mean = make_float2(
                __shfl_sync(0xffffffffu, mean2d_shifted.x, src),
                __shfl_sync(0xffffffffu, mean2d_shifted.y, src));
            const float3 src_conic = make_float3(
                __shfl_sync(0xffffffffu, conic.x, src),
                __shfl_sync(0xffffffffu, conic.y, src),
                __shfl_sync(0xffffffffu, conic.z, src));
            const float src_radius_sq = __shfl_sync(0xffffffffu, radius_sq, src);

            uint emitted_unclamped = 0u;
            for (uint row0 = 0; row0 < src_n_scan; row0 += 32u) {
                const uint row = row0 + lane;
                uint2 span = make_uint2(0, 0);
                uint cnt = 0u;
                const uint scan_index = src_scan0 + row;
                if (row < src_n_scan) {
                    span = ellipse_touched_tile_span(
                        src_conic, src_radius_sq, src_mean, src_along_x, scan_index, src_cross0, src_cross1);
                    cnt = tile_span_count(span);
                }
                const uint excl = warp_exclusive_scan_uint(cnt);
                uint write_at = src_write + emitted_unclamped + excl;
                if (row < src_n_scan) {
                    emit_ellipse_tile_span(
                        span, scan_index, src_along_x, grid_width, src_dkey, depth_bits, src_prim,
                        write_at, src_write_end, instance_keys, instance_primitive_indices);
                }
                emitted_unclamped += lfs::core::warp_ops::warp_reduce_sum(cnt);
            }

            if (lane == static_cast<uint>(src)) {
                const uint actual = src_write + emitted_unclamped > src_write_end
                                        ? src_write_end
                                        : src_write + emitted_unclamped;
                if (actual != src_write_end) {
                    report_forward_status(
                        status,
                        kFastGSForwardStatusInstanceWriteMismatch,
                        src_prim,
                        0,
                        actual,
                        diagnostic_bounds,
                        src_write_end,
                        actual);
                }
            }
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

        const uint screen_bounds_width = static_cast<uint>(screen_bounds.y - screen_bounds.x);
        const uint screen_bounds_height = static_cast<uint>(screen_bounds.w - screen_bounds.z);
        const bool scan_along_x = screen_bounds_height <= screen_bounds_width;
        const uint scan0 = scan_along_x ? static_cast<uint>(screen_bounds.z) : static_cast<uint>(screen_bounds.x);
        const uint scan1 = scan_along_x ? static_cast<uint>(screen_bounds.w) : static_cast<uint>(screen_bounds.y);
        const uint cross0 = scan_along_x ? static_cast<uint>(screen_bounds.x) : static_cast<uint>(screen_bounds.z);
        const uint cross1 = scan_along_x ? static_cast<uint>(screen_bounds.y) : static_cast<uint>(screen_bounds.w);
        const uint n_scan = active && scan1 >= scan0 ? scan1 - scan0 : 0u;
        const uint max_scan = lfs::core::warp_ops::warp_reduce_max(n_scan);

        if (max_scan < 32u) {
            if (active) {
                for (uint scan_index = scan0; scan_index < scan1 && current_write_offset < write_offset_end; scan_index++) {
                    const uint2 span = ellipse_touched_tile_span(
                        conic, radius_sq, mean2d_shifted, scan_along_x, scan_index, cross0, cross1);
                    emit_ellipse_tile_span(
                        span, scan_index, scan_along_x, grid_width, depth_key, depth_bits, primitive_idx,
                        current_write_offset, write_offset_end, instance_keys, instance_primitive_indices);
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
        } else {
            create_instances_coop_warp(
                active, n_scan, scan0, cross0, cross1, write_offset_end, current_write_offset,
                primitive_idx, depth_key, scan_along_x, mean2d_shifted, conic, radius_sq,
                grid_width, depth_bits, diagnostic_bounds, instance_keys, instance_primitive_indices, status);
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

    // Dual-pixel forward blend: replaced a 256-thread one-pixel-per-thread variant.
    // The 2-pixel shape mirrors blend_backward_cu (128 threads, 4 warps × 2 pixels).
    //
    // warp_cull_mode: 0 = enabled (production), 1 = disabled (mask all-1s, reference),
    //                 2 = deliberately incorrect empty mask for negative coverage.
    // blend_batch_size_runtime: multiple of 32 in [32, block_size_blend_forward]; 0 → config default.
    template <bool kRenderNormal>
    __global__ void __launch_bounds__(config::block_size_blend_forward) blend_cu(
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

        static_assert(config::tile_width == 16 && config::tile_height == 16,
                      "warp sub-tile layout assumes 16×16 tiles");
        static_assert(config::warp_subtile_width == 8 && config::warp_subtile_height == 4,
                      "warp sub-tile is 8×4");
        static_assert(config::block_size_blend_forward == 128,
                      "blend forward is 128 threads (4 warps x 2 pixels)");
        static_assert(config::block_size_blend_forward % 32 == 0,
                      "blend forward block size must be a multiple of warp size");
        static_assert(config::block_size_blend / config::block_size_blend_forward == 2,
                      "each thread owns 2 pixels");

        const uint lane_id = thread_rank & 31u;
        const uint warp_id = thread_rank >> 5; // 0..3
        const uint local_x = lane_id & 7u;
        const uint local_y = lane_id >> 3;
        // Two sub-tiles per warp: warp w → subtiles w and w+4 (same 2×2 XY packing as forward).
        const uint st0 = warp_id;     // 0..3
        const uint st1 = warp_id + 4; // 4..7
        const uint st0_ox = (st0 & 1u) * static_cast<uint>(config::warp_subtile_width);
        const uint st0_oy = (st0 >> 1) * static_cast<uint>(config::warp_subtile_height);
        const uint st1_ox = (st1 & 1u) * static_cast<uint>(config::warp_subtile_width);
        const uint st1_oy = (st1 >> 1) * static_cast<uint>(config::warp_subtile_height);
        const uint tlx0 = st0_ox + local_x;
        const uint tly0 = st0_oy + local_y;
        const uint tlx1 = st1_ox + local_x;
        const uint tly1 = st1_oy + local_y;
        const uint pixel_rank0 = tly0 * static_cast<uint>(config::tile_width) + tlx0;
        const uint pixel_rank1 = tly1 * static_cast<uint>(config::tile_width) + tlx1;

        const uint2 start_pixel_coords = {
            group_index.x * static_cast<uint>(config::tile_width),
            group_index.y * static_cast<uint>(config::tile_height)};
        const uint2 pix0 = {start_pixel_coords.x + tlx0, start_pixel_coords.y + tly0};
        const uint2 pix1 = {start_pixel_coords.x + tlx1, start_pixel_coords.y + tly1};
        const bool inside0 = pix0.x < width && pix0.y < height;
        const bool inside1 = pix1.x < width && pix1.y < height;
        const float2 pixel0 = make_float2(__uint2float_rn(pix0.x), __uint2float_rn(pix0.y)) + 0.5f;
        const float2 pixel1 = make_float2(__uint2float_rn(pix1.x), __uint2float_rn(pix1.y)) + 0.5f;

        const uint sub0_ix0 = start_pixel_coords.x + st0_ox;
        const uint sub0_iy0 = start_pixel_coords.y + st0_oy;
        const uint sub0_ix1 = sub0_ix0 + static_cast<uint>(config::warp_subtile_width);
        const uint sub0_iy1 = sub0_iy0 + static_cast<uint>(config::warp_subtile_height);
        const uint sub1_ix0 = start_pixel_coords.x + st1_ox;
        const uint sub1_iy0 = start_pixel_coords.y + st1_oy;
        const uint sub1_ix1 = sub1_ix0 + static_cast<uint>(config::warp_subtile_width);
        const uint sub1_iy1 = sub1_iy0 + static_cast<uint>(config::warp_subtile_height);

        const uint tile_idx = group_index.y * grid_width + group_index.x;
        const uint2 tile_range = tile_instance_ranges[tile_idx];
        const int n_points_total = static_cast<int>(tile_range.y - tile_range.x);

        int batch_size = blend_batch_size_runtime > 0 ? blend_batch_size_runtime : config::blend_batch_size;
        batch_size = max(32, min(batch_size, config::block_size_blend_forward));
        batch_size = batch_size & ~31; // force multiple of 32

        __shared__ float2 collected_mean2d[config::block_size_blend_forward];
        __shared__ ushort4 s_bbox[config::block_size_blend_forward];
        __shared__ float4 collected_conic_opacity[config::block_size_blend_forward];
        __shared__ float4 collected_color[config::block_size_blend_forward];
        __shared__ float collected_depth[config::block_size_blend_forward];
        __shared__ float3 collected_normal[kRenderNormal ? config::block_size_blend_forward : 1];

        float3 color_pixel0 = make_float3(0.0f);
        float3 color_pixel1 = make_float3(0.0f);
        float depth_pixel0 = 0.0f;
        float depth_pixel1 = 0.0f;
        float3 normal_pixel0 = make_float3(0.0f);
        float3 normal_pixel1 = make_float3(0.0f);
        float transmittance0 = 1.0f;
        float transmittance1 = 1.0f;
        uint n_possible_contributions0 = 0;
        uint n_possible_contributions1 = 0;
        uint n_contributions0 = 0;
        uint n_contributions1 = 0;
        bool done0 = !inside0;
        bool done1 = !inside1;

        for (int n_points_remaining = n_points_total, batch_base = 0;
             n_points_remaining > 0;
             n_points_remaining -= batch_size, batch_base += batch_size) {
            if (__syncthreads_count(done0 && done1) == config::block_size_blend_forward)
                break;

            if (static_cast<int>(thread_rank) < batch_size) {
                const int fetch_idx = static_cast<int>(tile_range.x) + batch_base + static_cast<int>(thread_rank);
                if (fetch_idx < static_cast<int>(tile_range.y)) {
                    const uint primitive_idx = instance_primitive_indices[fetch_idx];
                    const PackedMeanBBox geom = primitive_mean2d[primitive_idx];
                    collected_mean2d[thread_rank] = geom.mean2d;
                    s_bbox[thread_rank] = geom.pixel_bbox;
                    float4 conic_opacity = primitive_conic_opacity[primitive_idx];
                    // Fold 0.5 into the diagonal of the staged conic so the per-pixel
                    // quadric is cxx*dx^2 + cxy*dx*dy + czz*dy^2. The stored primitive
                    // conic is left unscaled for backward. conic.y is already the 1.0
                    // coefficient of dx*dy (the 0.5 and the 2 cancel). *0.5 is exact
                    // for finite normals and commutes with a correctly rounded FMA.
                    conic_opacity.x *= 0.5f;
                    conic_opacity.z *= 0.5f;
                    collected_conic_opacity[thread_rank] = conic_opacity;
                    const float4 raw_c = primitive_color[primitive_idx];
                    const float3 clamped = fminf(fmaxf(make_float3(raw_c), 0.0f), config::max_blend_color);
                    // log(opacity*255): skip __expf when sigma/2 is already past the
                    // alpha=1/255 contour. Dead .w slot; pairs that pass still run the
                    // exact opacity*exp test.
                    collected_color[thread_rank] = make_float4(
                        clamped, logf(conic_opacity.w * config::min_alpha_threshold_rcp));
                    collected_depth[thread_rank] = primitive_depths[primitive_idx];
                    if constexpr (kRenderNormal) {
                        collected_normal[thread_rank] = primitive_normals[primitive_idx];
                    }
                }
            }
            block.sync();

            const int current_batch_size = min(batch_size, n_points_remaining);

            for (int j_base = 0; j_base < current_batch_size; j_base += 32) {
                const int j_test = j_base + static_cast<int>(lane_id);
                bool intersects0 = false;
                bool intersects1 = false;
                if (j_test < current_batch_size) {
                    const ushort4 bb = s_bbox[j_test];
                    intersects0 = (static_cast<uint>(bb.x) < sub0_ix1) &&
                                  (static_cast<uint>(bb.y) > sub0_ix0) &&
                                  (static_cast<uint>(bb.z) < sub0_iy1) &&
                                  (static_cast<uint>(bb.w) > sub0_iy0);
                    intersects1 = (static_cast<uint>(bb.x) < sub1_ix1) &&
                                  (static_cast<uint>(bb.y) > sub1_ix0) &&
                                  (static_cast<uint>(bb.z) < sub1_iy1) &&
                                  (static_cast<uint>(bb.w) > sub1_iy0);
                }
                unsigned mask0 = __ballot_sync(0xffffffffu, intersects0);
                unsigned mask1 = __ballot_sync(0xffffffffu, intersects1);
                if (warp_cull_mode == 1) {
                    mask0 = mask1 = 0xffffffffu;
                } else if (warp_cull_mode == 2) {
                    mask0 = mask1 = 0u;
                }

                for (int k = 0; k < 32; ++k) {
                    const int j = j_base + k;
                    if (j >= current_batch_size)
                        break;

                    const bool walk0 = !done0;
                    const bool walk1 = !done1;
                    if (walk0)
                        n_possible_contributions0++;
                    if (walk1)
                        n_possible_contributions1++;
                    const bool hit0 = walk0 && (((mask0 >> k) & 1u) != 0u);
                    const bool hit1 = walk1 && (((mask1 >> k) & 1u) != 0u);
                    if (!hit0 && !hit1)
                        continue;

                    const float4 conic_opacity = collected_conic_opacity[j];
                    const float3 conic = make_float3(conic_opacity);
                    const float2 mean2d = collected_mean2d[j];
                    const float opacity = conic_opacity.w;
                    const float4 c = collected_color[j];
                    const float depth = collected_depth[j];
                    float3 normal = make_float3(0.0f);
                    if constexpr (kRenderNormal) {
                        normal = collected_normal[j];
                    }
                    // One-sided slack: skip only when even a 2-ulp-high __expf still
                    // fails alpha >= 1/255. Bound is ~1.3e-6 (2 ulp __expf ≈ 2^-22 in
                    // the exponent, 1 ulp logf at log(255)≈5.54 is 2^-20, plus 0.5 ulp
                    // of opacity*255); 1e-5 is ~8x that. Survivors always take the
                    // exact test below.
                    constexpr float kLogAlphaGateEps = 1.0e-5f;
                    const float log_alpha_pt = c.w;

                    if (hit0) {
                        const float2 delta = mean2d - pixel0;
                        const float sigma_over_2 = (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
                        if (!(sigma_over_2 < 0.0f) && !(sigma_over_2 > log_alpha_pt + kLogAlphaGateEps)) {
                            // __expf: -6.7% blend kernel time vs expf at equal 30k PSNR/SSIM (bonsai A/B).
                            const float gaussian = __expf(-sigma_over_2);
                            const float alpha = fminf(opacity * gaussian, config::max_fragment_alpha);
                            if (!(alpha < config::min_alpha_threshold)) {
                                const float weight = transmittance0 * alpha;
                                color_pixel0 += weight * make_float3(c);
                                depth_pixel0 += weight * depth;
                                if constexpr (kRenderNormal) {
                                    normal_pixel0 += weight * normal;
                                }
                                transmittance0 *= (1.0f - alpha);
                                n_contributions0 = n_possible_contributions0;
                                if (transmittance0 < config::transmittance_threshold) {
                                    done0 = true;
                                }
                            }
                        }
                    }
                    if (hit1) {
                        const float2 delta = mean2d - pixel1;
                        const float sigma_over_2 = (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
                        if (!(sigma_over_2 < 0.0f) && !(sigma_over_2 > log_alpha_pt + kLogAlphaGateEps)) {
                            const float gaussian = __expf(-sigma_over_2);
                            const float alpha = fminf(opacity * gaussian, config::max_fragment_alpha);
                            if (!(alpha < config::min_alpha_threshold)) {
                                const float weight = transmittance1 * alpha;
                                color_pixel1 += weight * make_float3(c);
                                depth_pixel1 += weight * depth;
                                if constexpr (kRenderNormal) {
                                    normal_pixel1 += weight * normal;
                                }
                                transmittance1 *= (1.0f - alpha);
                                n_contributions1 = n_possible_contributions1;
                                if (transmittance1 < config::transmittance_threshold) {
                                    done1 = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        if (inside0) {
            const int pixel_idx = width * pix0.y + pix0.x;
            const int n_pixels = width * height;
            float3 bg = make_float3(0.0f, 0.0f, 0.0f);
            if (bg_image != nullptr) {
                bg = make_float3(bg_image[pixel_idx],
                                 bg_image[pixel_idx + n_pixels],
                                 bg_image[pixel_idx + 2 * n_pixels]);
            } else if (bg_color != nullptr) {
                bg = make_float3(bg_color[0], bg_color[1], bg_color[2]);
            }
            image[pixel_idx] = color_pixel0.x + transmittance0 * bg.x;
            image[pixel_idx + n_pixels] = color_pixel0.y + transmittance0 * bg.y;
            image[pixel_idx + n_pixels * 2] = color_pixel0.z + transmittance0 * bg.z;
            alpha_map[pixel_idx] = 1.0f - transmittance0;
            depth_map[pixel_idx] = depth_pixel0;
            if constexpr (kRenderNormal) {
                normal_map[pixel_idx] = normal_pixel0.x;
                normal_map[pixel_idx + n_pixels] = normal_pixel0.y;
                normal_map[pixel_idx + n_pixels * 2] = normal_pixel0.z;
            }
            tile_n_contributions[pixel_idx] = n_contributions0;
        }
        tile_final_transmittance[tile_idx * config::block_size_blend + pixel_rank0] = inside0 ? transmittance0 : 1.0f;

        if (inside1) {
            const int pixel_idx = width * pix1.y + pix1.x;
            const int n_pixels = width * height;
            float3 bg = make_float3(0.0f, 0.0f, 0.0f);
            if (bg_image != nullptr) {
                bg = make_float3(bg_image[pixel_idx],
                                 bg_image[pixel_idx + n_pixels],
                                 bg_image[pixel_idx + 2 * n_pixels]);
            } else if (bg_color != nullptr) {
                bg = make_float3(bg_color[0], bg_color[1], bg_color[2]);
            }
            image[pixel_idx] = color_pixel1.x + transmittance1 * bg.x;
            image[pixel_idx + n_pixels] = color_pixel1.y + transmittance1 * bg.y;
            image[pixel_idx + n_pixels * 2] = color_pixel1.z + transmittance1 * bg.z;
            alpha_map[pixel_idx] = 1.0f - transmittance1;
            depth_map[pixel_idx] = depth_pixel1;
            if constexpr (kRenderNormal) {
                normal_map[pixel_idx] = normal_pixel1.x;
                normal_map[pixel_idx + n_pixels] = normal_pixel1.y;
                normal_map[pixel_idx + n_pixels * 2] = normal_pixel1.z;
            }
            tile_n_contributions[pixel_idx] = n_contributions1;
        }
        tile_final_transmittance[tile_idx * config::block_size_blend + pixel_rank1] = inside1 ? transmittance1 : 1.0f;
    }

} // namespace fast_lfs::rasterization::kernels::forward
