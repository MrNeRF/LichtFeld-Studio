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

namespace fast_gs::rasterization::kernels::forward {

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
        const float far_) {
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
        float3 cov2d = make_float3(
            dot(jwc_r1, jw_r1),
            dot(jwc_r1, jw_r2),
            dot(jwc_r2, jw_r2));
        cov2d.x += config::dilation;
        cov2d.z += config::dilation;
        const float determinant = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
        if (determinant < 1e-8f)
            active = false;
        const float3 conic = make_float3(
            cov2d.z / determinant,
            -cov2d.y / determinant,
            cov2d.x / determinant);

        // 2d mean in screen space
        const float2 mean2d = make_float2(
            x * fx + cx,
            y * fy + cy);

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
        primitive_color[primitive_idx] = convert_sh_to_color(
            sh_coefficients_0, sh_coefficients_rest,
            mean3d, cam_position[0],
            primitive_idx, active_sh_bases, total_bases_sh_rest);

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
        const uint* tile_bucket_offsets,
        const uint* instance_primitive_indices,
        const float2* primitive_mean2d,
        const float4* primitive_conic_opacity,
        const float3* primitive_color,
        float* image,
        float* alpha_map,
        uint* tile_max_n_contributions,
        uint* tile_n_contributions,
        uint* bucket_tile_index,
        float4* bucket_color_transmittance,
        const uint width,
        const uint height,
        const uint grid_width) {
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

        uint bucket_offset = tile_idx == 0 ? 0 : tile_bucket_offsets[tile_idx - 1];
        const int n_buckets = div_round_up(n_points_total, 32); // re-computing is faster than reading from tile_n_buckets
        for (int n_buckets_remaining = n_buckets, current_bucket_idx = thread_rank; n_buckets_remaining > 0; n_buckets_remaining -= config::block_size_blend, current_bucket_idx += config::block_size_blend) {
            if (current_bucket_idx < n_buckets)
                bucket_tile_index[bucket_offset + current_bucket_idx] = tile_idx;
        }

        // setup shared memory
        __shared__ float2 collected_mean2d[config::block_size_blend];
        __shared__ float4 collected_conic_opacity[config::block_size_blend];
        __shared__ float3 collected_color[config::block_size_blend];
        // initialize local storage
        float3 color_pixel = make_float3(0.0f);
        float transmittance = 1.0f;
        uint n_possible_contributions = 0;
        uint n_contributions = 0;
        bool done = !inside;
        // collaborative loading and processing
        for (int n_points_remaining = n_points_total, current_fetch_idx = tile_range.x + thread_rank; n_points_remaining > 0; n_points_remaining -= config::block_size_blend, current_fetch_idx += config::block_size_blend) {
            if (__syncthreads_count(done) == config::block_size_blend)
                break;
            if (current_fetch_idx < tile_range.y) {
                const uint primitive_idx = instance_primitive_indices[current_fetch_idx];
                collected_mean2d[thread_rank] = primitive_mean2d[primitive_idx];
                collected_conic_opacity[thread_rank] = primitive_conic_opacity[primitive_idx];
                const float3 color = fmaxf(primitive_color[primitive_idx], 0.0f);
                collected_color[thread_rank] = color;
            }
            block.sync();
            const int current_batch_size = min(config::block_size_blend, n_points_remaining);
            for (int j = 0; !done && j < current_batch_size; ++j) {
                if (j % 32 == 0) {
                    const float4 current_color_transmittance = make_float4(color_pixel, transmittance);
                    bucket_color_transmittance[bucket_offset * config::block_size_blend + thread_rank] = current_color_transmittance;
                    bucket_offset++;
                }
                n_possible_contributions++;
                const float4 conic_opacity = collected_conic_opacity[j];
                const float3 conic = make_float3(conic_opacity);
                const float2 delta = collected_mean2d[j] - pixel;
                const float opacity = conic_opacity.w;
                const float sigma_over_2 = 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) + conic.y * delta.x * delta.y;
                if (sigma_over_2 < 0.0f)
                    continue;
                const float gaussian = expf(-sigma_over_2);
                const float alpha = fminf(opacity * gaussian, config::max_fragment_alpha);
                if (alpha < config::min_alpha_threshold)
                    continue;
                const float next_transmittance = transmittance * (1.0f - alpha);
                if (next_transmittance < config::transmittance_threshold) {
                    done = true;
                    continue;
                }
                color_pixel += transmittance * alpha * collected_color[j];
                transmittance = next_transmittance;
                n_contributions = n_possible_contributions;
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
            tile_n_contributions[pixel_idx] = n_contributions;
        }

        // max reduce the number of contributions
        typedef cub::BlockReduce<uint, config::tile_width, cub::BLOCK_REDUCE_WARP_REDUCTIONS, config::tile_height> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        n_contributions = BlockReduce(temp_storage).Reduce(n_contributions, thrust::maximum<uint>());
        if (thread_rank == 0)
            tile_max_n_contributions[tile_idx] = n_contributions;
    }

} // namespace fast_gs::rasterization::kernels::forward
