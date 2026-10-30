/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "blend_bookkeeping.cuh"
#include "buffer_utils.h"
#include "gaussian_evaluation.cuh"
#include "helper_math.h"
#include "instance_sort.cuh"
#include "kernel_utils.cuh"
#include "lfs/training/screen_share.cuh"
#include "rasterization_config.h"
#include "utils.h"
#include <cooperative_groups.h>
#include <cstdint>
#include <cub/block/block_scan.cuh>
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

    __device__ __forceinline__ uint depth_sort_key(float depth) {
        return __float_as_uint(depth);
    }

    __global__ void gather_depth_ordered_counts_cu(
        const uint* __restrict__ depth_order,
        const uint* __restrict__ work_indices,
        const std::uint64_t* __restrict__ counts,
        std::uint64_t* __restrict__ ordered_counts,
        const uint count) {
        const uint index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < count)
            ordered_counts[index] = counts[work_indices[depth_order[index]]];
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
        const uint* __restrict__ primitive_indices,
        uint* __restrict__ primitive_visible_mask,
        uint* __restrict__ primitive_depth_keys,
        float* __restrict__ primitive_depths,
        std::uint64_t* __restrict__ primitive_n_touched_tiles,
        ushort4* __restrict__ primitive_screen_bounds,
        ProjectedRecord* __restrict__ projected,
        float3* __restrict__ primitive_normals,
        const uint n_work_items,
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
        const bool mip_filter,
        const float dilation_scale,
        float* __restrict__ max_screen_share) {
        (void)w;
        (void)h;
        auto work_idx = cg::this_grid().thread_rank();
        bool active = work_idx < n_work_items;
        if (!active) {
            work_idx = n_work_items - 1;
        }
        const uint primitive_idx = primitive_indices != nullptr
                                       ? primitive_indices[work_idx]
                                       : work_idx;
        if (!active) {
            active = false;
        }

        if (active && primitive_n_touched_tiles != nullptr)
            primitive_n_touched_tiles[work_idx] = 0;

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
        const float4 raw_rotation = raw_rotations[primitive_idx];
        if (!isfinite(mean3d.x) || !isfinite(mean3d.y) || !isfinite(mean3d.z) ||
            !isfinite(raw_scale.x) || !isfinite(raw_scale.y) || !isfinite(raw_scale.z) ||
            !isfinite(raw_rotation.x) || !isfinite(raw_rotation.y) ||
            !isfinite(raw_rotation.z) || !isfinite(raw_rotation.w) || !isfinite(raw_opacity))
            active = false;
        if (__ballot_sync(0xffffffffu, active) == 0)
            return;
        const float3 clamped_scale = make_float3(
            fminf(raw_scale.x, config::max_raw_scale),
            fminf(raw_scale.y, config::max_raw_scale),
            fminf(raw_scale.z, config::max_raw_scale));
        const float3 variance = make_float3(expf(2.0f * clamped_scale.x), expf(2.0f * clamped_scale.y), expf(2.0f * clamped_scale.z));
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
        const float kernel_size = (mip_filter ? config::dilation_mip_filter : config::dilation) * dilation_scale;
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

        if (max_screen_share != nullptr) {
            const float3 campos = cam_position[0];
            const float share = lfs::training::gaussian_screen_share(
                mean3d.x, mean3d.y, mean3d.z,
                campos.x, campos.y, campos.z,
                raw_scale.x, raw_scale.y, raw_scale.z,
                raw_opacity);
            lfs::training::atomic_max_float(max_screen_share + primitive_idx, share);
        }

        // store results
        if (primitive_visible_mask != nullptr) {
            atomicOr(&primitive_visible_mask[primitive_idx >> 5],
                     1u << (primitive_idx & 31u));
        }
        if (primitive_n_touched_tiles != nullptr)
            primitive_n_touched_tiles[work_idx] = n_touched_tiles;
        if (primitive_screen_bounds == nullptr)
            return;
        primitive_screen_bounds[work_idx] = make_ushort4(
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
        projected[work_idx].geometry = packed;
        projected[work_idx].conic_opacity = make_float4(conic, output_opacity);
        // Pad float3 color → float4 for 128-bit loads (w unused).
        const float3 sh_color = convert_sh_to_color(
            sh_coefficients_0, sh_coefficients_rest,
            mean3d, cam_position[0],
            primitive_idx, active_sh_bases, sh_layout_slots,
            sh_value_bounds, sh_value_n_cells, sh_value_bits);
        projected[work_idx].color = make_float4(sh_color, 0.0f);
        primitive_depth_keys[work_idx] = depth_sort_key(depth);
        primitive_depths[work_idx] = depth;

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
            primitive_normals[work_idx] = make_float3(
                w2c_r1.x * normal_world.x + w2c_r1.y * normal_world.y + w2c_r1.z * normal_world.z,
                w2c_r2.x * normal_world.x + w2c_r2.y * normal_world.y + w2c_r2.z * normal_world.z,
                w2c_r3.x * normal_world.x + w2c_r3.y * normal_world.y + w2c_r3.z * normal_world.z);
        }
    }

    template <typename Key>
    __device__ __forceinline__ void emit_ellipse_tile_span(
        const uint2 span,
        const uint scan_index,
        const bool along_x,
        const uint grid_width,
        const uint depth_key,
        const uint primitive_idx,
        uint& write_at,
        const uint write_end,
        Key* __restrict__ instance_keys,
        uint* __restrict__ instance_primitive_indices) {
        for (uint t = span.x; t < span.y && write_at < write_end; t++) {
            const uint tile_key = along_x ? (scan_index * grid_width + t) : (t * grid_width + scan_index);
            instance_keys[write_at] = make_instance_key<Key>(tile_key, depth_key);
            instance_primitive_indices[write_at] = primitive_idx;
            write_at++;
        }
    }

    // Long-tail primitives: 32 lanes cooperate over rows of one source lane
    // at a time. Isolated so the serial emit path does not inherit its
    // register footprint.
    template <typename Key>
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
        const uint4 diagnostic_bounds,
        Key* __restrict__ instance_keys,
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
                        span, scan_index, src_along_x, grid_width, src_dkey, src_prim,
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
    template <typename Key, bool DepthOrdered>
    __global__ void create_instances_cu(
        const std::uint64_t* __restrict__ primitive_n_touched_tiles,
        const std::uint64_t* __restrict__ primitive_offsets,
        const uint* __restrict__ primitive_depth_keys,
        const ushort4* __restrict__ primitive_screen_bounds,
        const ProjectedRecord* __restrict__ projected,
        const uint* __restrict__ visible_primitive_indices,
        const uint* __restrict__ primitive_work_indices,
        Key* __restrict__ instance_keys,
        uint* __restrict__ instance_primitive_indices,
        FastGSForwardStatus* __restrict__ status,
        const uint grid_width,
        const uint n_visible,
        const uint max_instances) {
        uint idx = cg::this_grid().thread_rank();

        bool active = idx < n_visible;
        if (!active) {
            active = false;
            idx = n_visible - 1;
        }

        const uint primitive_idx = visible_primitive_indices[idx];
        const uint work_idx = DepthOrdered ? primitive_work_indices[primitive_idx] : idx;
        const uint n_touched_tiles = active ? static_cast<uint>(primitive_n_touched_tiles[work_idx]) : 0;
        active = active && n_touched_tiles > 0;

        if (__ballot_sync(0xffffffffu, active) == 0)
            return;

        const ushort4 screen_bounds = active ? primitive_screen_bounds[work_idx] : make_ushort4(0, 0, 0, 0);
        const uint4 diagnostic_bounds = make_uint4(screen_bounds.x, screen_bounds.y, screen_bounds.z, screen_bounds.w);
        const uint depth_key = active ? primitive_depth_keys[work_idx] : 0;
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

        const float2 mean2d_shifted = active ? projected[work_idx].geometry.mean2d - 0.5f : make_float2(0.0f, 0.0f);
        const float4 conic_opacity_loaded = active ? projected[work_idx].conic_opacity : make_float4(0.0f, 0.0f, 0.0f, config::min_alpha_threshold);
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
                        span, scan_index, scan_along_x, grid_width, depth_key, primitive_idx,
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
                grid_width, diagnostic_bounds, instance_keys, instance_primitive_indices, status);
        }
    }

    // Front-to-back blend. Each warp owns one 8x8 quadrant of the 16x16 tile and
    // each thread two vertically adjacent pixels of one column, so both rows
    // share one AlphaColumn per splat. The block stages one batch of splats with
    // an 8-bit mask of the 8x4 regions their pixel bounds touch, computed once
    // per splat; each warp then walks the set bits of its two regions. Splats
    // that no pixel blended are compacted out of the tile list in place, and
    // each kept entry records the regions it reached for the backward pass.
    //
    // warp_cull_mode: 0 = enabled (production), 1 = disabled (mask all-1s, reference),
    //                 2 = deliberately incorrect empty mask for negative coverage.
    // blend_batch_size_runtime: multiple of 32 in [32, block_size_blend_forward]; 0 → config default.
    template <bool kRenderNormal, bool kRenderDepth>
    __global__ void __launch_bounds__(config::block_size_blend_forward) blend_cu(
        uint2* __restrict__ tile_instance_ranges,
        uint* __restrict__ instance_primitive_indices,
        unsigned char* __restrict__ contributor_masks,
        uint* __restrict__ contributor_positions,
        const uint* __restrict__ primitive_work_indices,
        const ProjectedRecord* __restrict__ projected,
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
        static_assert(config::tile_width == 16 && config::tile_height == 16, "quadrant layout assumes 16x16 tiles");
        static_assert(config::block_size_blend_forward == 128, "four warps, one 8x8 quadrant each");
        static_assert(config::warp_subtile_width == 8 && config::warp_subtile_height == 4,
                      "coverage bits address 8x4 regions");

        const uint thread_rank = block.thread_rank();
        const uint lane = thread_rank & 31u;
        const uint warp = thread_rank >> 5;
        const uint tile_x = (warp & 1u) * 8u + (lane & 7u);
        const uint tile_y0 = (warp >> 1) * 8u + 2u * (lane >> 3);
        const uint top_region = (warp & 1u) + 4u * (warp >> 1);
        const uint region = top_region + (lane < 16u ? 0u : 2u);

        const uint tile_idx = blockIdx.y * grid_width + blockIdx.x;
        const uint origin_x = blockIdx.x * static_cast<uint>(config::tile_width);
        const uint origin_y = blockIdx.y * static_cast<uint>(config::tile_height);
        const uint pixel_x = origin_x + tile_x;
        const uint pixel_y0 = origin_y + tile_y0;
        const float center_x = __uint2float_rn(pixel_x) + 0.5f;
        const float center_y0 = __uint2float_rn(pixel_y0) + 0.5f;
        bool inside[2];
        float transmittance[2];
        float3 color[2], normal[2];
        float depth[2];
        uint contributions[2];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            inside[k] = pixel_x < width && pixel_y0 + k < height;
            transmittance[k] = signed_transmittance(1.0f, !inside[k]);
            color[k] = make_float3(0.0f);
            normal[k] = make_float3(0.0f);
            depth[k] = 0.0f;
            contributions[k] = 0u;
        }

        const uint2 tile_range = tile_instance_ranges[tile_idx];
        const int n_points_total = static_cast<int>(tile_range.y - tile_range.x);
        int batch_size = blend_batch_size_runtime > 0 ? blend_batch_size_runtime : config::blend_batch_size;
        batch_size = max(32, min(batch_size, config::block_size_blend_forward)) & ~31;

        using ContributorScan = cub::BlockScan<uint, config::block_size_blend_forward>;
        __shared__ typename ContributorScan::TempStorage contributor_scan;
        __shared__ uint collected_ids[config::block_size_blend_forward];
        __shared__ uint collected_masks[config::block_size_blend_forward];
        __shared__ unsigned char collected_regions[config::block_size_blend_forward];
        __shared__ float2 collected_mean2d[config::block_size_blend_forward];
        __shared__ float4 collected_conic_opacity[config::block_size_blend_forward];
        __shared__ float4 collected_color[config::block_size_blend_forward];
        __shared__ float collected_depth[config::block_size_blend_forward];
        __shared__ float3 collected_normal[kRenderNormal ? config::block_size_blend_forward : 1];
        uint kept_count = 0;

        for (int n_points_remaining = n_points_total, batch_base = 0;
             n_points_remaining > 0;
             n_points_remaining -= batch_size, batch_base += batch_size) {
            if (__syncthreads_count(transmittance_complete(transmittance[0]) && transmittance_complete(transmittance[1])) ==
                config::block_size_blend_forward)
                break;

            collected_masks[thread_rank] = 0u;
            if (static_cast<int>(thread_rank) < batch_size) {
                const int fetch_idx = static_cast<int>(tile_range.x) + batch_base + static_cast<int>(thread_rank);
                unsigned regions = 0u;
                if (fetch_idx < static_cast<int>(tile_range.y)) {
                    const uint primitive_idx = instance_primitive_indices[fetch_idx];
                    collected_ids[thread_rank] = primitive_idx;
                    const uint work_idx = primitive_work_indices[primitive_idx];
                    const ProjectedRecord record = projected[work_idx];
                    const ushort4 bb = record.geometry.pixel_bbox;
                    // Pixel bounds are half-open [x_min, x_max) x [y_min, y_max).
#pragma unroll
                    for (uint band = 0; band < 4u; ++band) {
                        const uint y0 = origin_y + band * 4u;
                        if (static_cast<uint>(bb.z) < y0 + 4u && static_cast<uint>(bb.w) > y0) {
                            if (static_cast<uint>(bb.x) < origin_x + 8u && static_cast<uint>(bb.y) > origin_x)
                                regions |= 1u << (2u * band);
                            if (static_cast<uint>(bb.x) < origin_x + 16u && static_cast<uint>(bb.y) > origin_x + 8u)
                                regions |= 2u << (2u * band);
                        }
                    }
                    collected_mean2d[thread_rank] = record.geometry.mean2d;
                    collected_conic_opacity[thread_rank] =
                        make_float4(base2_conic(make_float3(record.conic_opacity)), record.conic_opacity.w);
                    collected_color[thread_rank] =
                        make_float4(fminf(fmaxf(make_float3(record.color), 0.0f), config::max_blend_color), 0.0f);
                    collected_depth[thread_rank] = primitive_depths[work_idx];
                    if constexpr (kRenderNormal)
                        collected_normal[thread_rank] = primitive_normals[work_idx];
                }
                collected_regions[thread_rank] = static_cast<unsigned char>(regions);
            }
            block.sync();

            const int current_batch_size = min(batch_size, n_points_remaining);
            for (int j_base = 0; j_base < current_batch_size; j_base += 32) {
                const int j_test = j_base + static_cast<int>(lane);
                const unsigned regions = j_test < current_batch_size ? collected_regions[j_test] : 0u;
                unsigned mask_top = __ballot_sync(0xffffffffu, (regions >> top_region) & 1u);
                unsigned mask_bottom = __ballot_sync(0xffffffffu, (regions >> (top_region + 2u)) & 1u);
                if (warp_cull_mode == 1) {
                    const unsigned valid = 0xffffffffu >> (32 - min(32, current_batch_size - j_base));
                    mask_top = mask_bottom = valid;
                } else if (warp_cull_mode == 2) {
                    mask_top = mask_bottom = 0u;
                }
                const unsigned mine = lane < 16u ? mask_top : mask_bottom;
                unsigned remaining = mask_top | mask_bottom;
                unsigned blended = 0u;
                while (remaining) {
                    const int k = __ffs(remaining) - 1;
                    remaining &= remaining - 1;
                    const int j = j_base + k;
                    const bool reaches = ((mine >> k) & 1u) != 0u;
                    const float4 conic_opacity = collected_conic_opacity[j];
                    const float3 conic = make_float3(conic_opacity);
                    const float2 mean2d = collected_mean2d[j];
                    const float3 splat_color = make_float3(collected_color[j]);
                    const float splat_depth = collected_depth[j];
                    float3 splat_normal = make_float3(0.0f);
                    if constexpr (kRenderNormal)
                        splat_normal = collected_normal[j];
                    const AlphaColumn column = AlphaColumn::at(conic, conic_opacity.w, mean2d.x - center_x);
                    const float dy0 = mean2d.y - center_y0;
                    bool accepted_any = false;
#pragma unroll
                    for (int row = 0; row < 2; ++row) {
                        const bool live = reaches && !transmittance_complete(transmittance[row]);
                        const float unclamped = live ? column.alpha(conic.z, dy0 - static_cast<float>(row)) : 0.0f;
                        const bool accepted = live && alpha_accepted(unclamped);
                        const float alpha = accepted ? clamped_alpha(unclamped) : 0.0f;
                        const float magnitude = transmittance_magnitude(transmittance[row]);
                        const float weight = magnitude * alpha;
                        color[row] += weight * splat_color;
                        depth[row] += weight * splat_depth;
                        if constexpr (kRenderNormal)
                            normal[row] += weight * splat_normal;
                        const float after = magnitude * (1.0f - alpha);
                        const bool complete = transmittance_complete(transmittance[row]) ||
                                              (accepted && after < config::transmittance_threshold);
                        transmittance[row] = signed_transmittance(after, complete);
                        contributions[row] = accepted ? static_cast<uint>(batch_base + j + 1) : contributions[row];
                        accepted_any = accepted_any || accepted;
                    }
                    blended |= accepted_any ? 1u << k : 0u;
                }
                // OR within each half-warp: lanes 0-15 hold the top region, 16-31 the bottom one.
#pragma unroll
                for (int offset = 8; offset > 0; offset >>= 1)
                    blended |= __shfl_xor_sync(0xffffffffu, blended, offset);
                const unsigned blended_top = __shfl_sync(0xffffffffu, blended, 0);
                const unsigned blended_bottom = __shfl_sync(0xffffffffu, blended, 16);
                const unsigned bit = 1u << lane;
                const unsigned coverage = ((blended_top & bit) ? 1u << top_region : 0u) |
                                          ((blended_bottom & bit) ? 1u << (top_region + 2u) : 0u);
                if (j_test < current_batch_size && coverage != 0u)
                    atomicOr(collected_masks + j_test, coverage);
            }

            // All geometry and IDs for this batch are staged before any in-place
            // write. The destination never reaches a future batch.
            block.sync();
            const uint coverage = collected_masks[thread_rank];
            const bool kept = static_cast<int>(thread_rank) < current_batch_size && coverage != 0u;
            uint rank, total;
            ContributorScan(contributor_scan).ExclusiveSum(kept ? 1u : 0u, rank, total);
            if (kept) {
                const uint destination = tile_range.x + kept_count + rank;
                instance_primitive_indices[destination] = collected_ids[thread_rank];
                contributor_masks[destination] = static_cast<unsigned char>(coverage);
                contributor_positions[destination] = batch_base + thread_rank;
            }
            kept_count += total;
            block.sync();
        }
        if (thread_rank == 0)
            tile_instance_ranges[tile_idx].y = tile_range.x + kept_count;

        const int n_pixels = width * height;
#pragma unroll
        for (int row = 0; row < 2; ++row) {
            const float final_transmittance = transmittance_magnitude(transmittance[row]);
            const uint pixel_rank = (tile_y0 + row) * static_cast<uint>(config::tile_width) + tile_x;
            tile_final_transmittance[tile_idx * config::block_size_blend + pixel_rank] = inside[row] ? final_transmittance : 1.0f;
            if (!inside[row])
                continue;
            const int pixel_idx = width * (pixel_y0 + row) + pixel_x;
            float3 bg = make_float3(0.0f, 0.0f, 0.0f);
            if (bg_image != nullptr) {
                bg = make_float3(bg_image[pixel_idx], bg_image[pixel_idx + n_pixels], bg_image[pixel_idx + 2 * n_pixels]);
            } else if (bg_color != nullptr) {
                bg = make_float3(bg_color[0], bg_color[1], bg_color[2]);
            }
            image[pixel_idx] = color[row].x + final_transmittance * bg.x;
            image[pixel_idx + n_pixels] = color[row].y + final_transmittance * bg.y;
            image[pixel_idx + n_pixels * 2] = color[row].z + final_transmittance * bg.z;
            alpha_map[pixel_idx] = 1.0f - final_transmittance;
            if constexpr (kRenderDepth)
                depth_map[pixel_idx] = depth[row];
            if constexpr (kRenderNormal) {
                normal_map[pixel_idx] = normal[row].x;
                normal_map[pixel_idx + n_pixels] = normal[row].y;
                normal_map[pixel_idx + n_pixels * 2] = normal[row].z;
            }
            tile_n_contributions[pixel_idx] = contributions[row];
        }
    }

} // namespace fast_lfs::rasterization::kernels::forward
