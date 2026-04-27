/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "buffer_utils.h"
#include "edge_rasterization_config.h"
#include "helper_math.h"
#include "kernel_utils.cuh"
#include "utils.h"
#include <cooperative_groups.h>
#include <cstdint>
namespace cg = cooperative_groups;

namespace edge_compute::rasterization::kernels::forward {

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
        const float4* __restrict__ w2c,
        uint* __restrict__ primitive_depth_keys,
        uint* __restrict__ primitive_n_touched_tiles,
        ushort4* __restrict__ primitive_screen_bounds,
        float2* __restrict__ primitive_mean2d,
        float4* __restrict__ primitive_conic_opacity,
        const uint n_primitives,
        const uint grid_width,
        const uint grid_height,
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
        auto [qr, qx, qy, qz] = raw_rotations[primitive_idx];
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

        // Skip primitives whose exact tile range collapsed after ellipse refinement.
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
        primitive_conic_opacity[primitive_idx] = make_float4(conic, output_opacity);
        primitive_depth_keys[primitive_idx] = quantize_depth_key(depth, depth_bits);
    }

    // based on https://github.com/r4dl/StopThePop-Rasterization/blob/d8cad09919ff49b11be3d693d1e71fa792f559bb/cuda_rasterizer/stopthepop/stopthepop_common.cuh#L325
    __global__ void create_instances_cu(
        const uint* __restrict__ primitive_n_touched_tiles,
        const uint* __restrict__ primitive_offsets,
        const uint* __restrict__ primitive_depth_keys,
        const ushort4* __restrict__ primitive_screen_bounds,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        InstanceKey* __restrict__ instance_keys,
        uint* __restrict__ instance_primitive_indices,
        const uint grid_width,
        const uint depth_bits,
        const uint n_primitives) {
        uint idx = cg::this_grid().thread_rank();

        bool active = true;
        if (idx >= n_primitives) {
            active = false;
            idx = n_primitives - 1;
        }

        const uint primitive_idx = idx;
        const uint n_touched_tiles = active ? primitive_n_touched_tiles[primitive_idx] : 0;
        active = active && n_touched_tiles > 0;

        if (__ballot_sync(0xffffffffu, active) == 0)
            return;

        const ushort4 screen_bounds = active ? primitive_screen_bounds[primitive_idx] : make_ushort4(0, 0, 0, 0);
        const uint depth_key = active ? primitive_depth_keys[primitive_idx] : 0;
        const uint write_offset_end = active ? primitive_offsets[idx] : 0;

        const float2 mean2d_shifted = active ? primitive_mean2d[primitive_idx] - 0.5f : make_float2(0.0f, 0.0f);
        const float4 conic_opacity_loaded = active ? primitive_conic_opacity[primitive_idx] : make_float4(0.0f, 0.0f, 0.0f, config::min_alpha_threshold);
        const float3 conic = make_float3(conic_opacity_loaded);
        const float power_threshold_precomputed = logf(conic_opacity_loaded.w * config::min_alpha_threshold_rcp);
        const float radius_sq = 2.0f * power_threshold_precomputed;

        uint current_write_offset = idx == 0 ? 0 : primitive_offsets[idx - 1];

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
        }
    }

    __global__ void extract_instance_ranges_cu(
        const InstanceKey* instance_keys,
        uint2* tile_instance_ranges,
        const uint depth_bits,
        const uint n_instances) {
        auto instance_idx = cg::this_grid().thread_rank();
        if (instance_idx >= n_instances)
            return;
        const uint instance_tile_idx = static_cast<uint>(instance_keys[instance_idx] >> depth_bits);
        if (instance_idx == 0)
            tile_instance_ranges[instance_tile_idx].x = 0;
        else {
            const uint previous_instance_tile_idx = static_cast<uint>(instance_keys[instance_idx - 1] >> depth_bits);
            if (instance_tile_idx != previous_instance_tile_idx) {
                tile_instance_ranges[previous_instance_tile_idx].y = instance_idx;
                tile_instance_ranges[instance_tile_idx].x = instance_idx;
            }
        }
        if (instance_idx == n_instances - 1)
            tile_instance_ranges[instance_tile_idx].y = n_instances;
    }

    __global__ void __launch_bounds__(config::block_size_blend) edge_blend_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ instance_primitive_indices,
        const float2* __restrict__ primitive_mean2d,
        const float4* __restrict__ primitive_conic_opacity,
        const uint width,
        const uint height,
        const uint grid_width,
        const float* __restrict__ pixel_weights,
        float* __restrict__ accum_weights) {
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

        // Fetch pixel weight once in register
        float thread_pixel_weight = 0.0f;
        int pixel_idx = 0;
        if (inside) {
            pixel_idx = width * pixel_coords.y + pixel_coords.x;
            thread_pixel_weight = pixel_weights[pixel_idx];
        }

        // setup shared memory
        __shared__ float2 collected_mean2d[config::block_size_blend];
        __shared__ float4 collected_conic_opacity[config::block_size_blend];
        __shared__ uint collected_primitive_idx[config::block_size_blend];

        // initialize local storage
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
                if (sigma_over_2 < 0.0f) {
                    continue;
                }

                const float gaussian = expf(-sigma_over_2);
                const float alpha = fminf(opacity * gaussian, config::max_fragment_alpha);
                if (alpha < config::min_alpha_threshold) {
                    continue;
                }

                const float contribution_factor = transmittance * alpha;

                transmittance *= (1.0f - alpha);

                atomicAdd(&accum_weights[collected_primitive_idx[j]], thread_pixel_weight * contribution_factor);

                if (transmittance < config::transmittance_threshold) {
                    done = true;
                    continue;
                }
            }
        }
    }

} // namespace edge_compute::rasterization::kernels::forward
