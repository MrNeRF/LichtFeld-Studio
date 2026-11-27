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

    __device__ __forceinline__ std::uint64_t load_acquire_u64(const std::uint64_t* ptr) {
        std::uint64_t val;
        asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(val) : "l"(ptr) : "memory");
        return val;
    }

    __device__ __forceinline__ void store_release_u64(std::uint64_t* ptr, const std::uint64_t val) {
        asm volatile("st.release.gpu.global.u64 [%0], %1;" ::"l"(ptr), "l"(val) : "memory");
    }

    __global__ void iota_u32_cu(uint* __restrict__ out, const uint count) {
        const uint index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < count)
            out[index] = index;
    }

    __global__ void gather_depth_ordered_counts_cu(
        const uint* __restrict__ depth_work_order,
        const uint* __restrict__ counts,
        uint* __restrict__ ordered_counts,
        const uint count) {
        const uint index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < count)
            ordered_counts[index] = counts[depth_work_order[index]];
    }

    __global__ void prepare_tile_lpt_keys_cu(
        const uint2* __restrict__ tile_instance_ranges,
        uint* __restrict__ tile_counts,
        uint* __restrict__ tile_ids,
        const uint n_tiles) {
        const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_tiles) {
            const uint2 r = tile_instance_ranges[idx];
            tile_counts[idx] = (r.y > r.x) ? (r.y - r.x) : 0u;
            tile_ids[idx] = idx;
        }
    }

    // Single-pass forward preprocess + conservative frustum cull + compact 64-bit
    // decoupled-lookback visibility compaction. Assigns deterministic ascending work_idx
    // with 8-byte lookback tiles (no per-instance prefix scan or dead offset store).
    template <int ACTIVE_SH_BASES, bool SH_Q16>
    __global__ void __launch_bounds__(config::block_size_preprocess, 4) preprocess_single_pass_cu(
        const float3* __restrict__ means,
        const float3* __restrict__ raw_scales,
        const float4* __restrict__ raw_rotations,
        const float* __restrict__ raw_opacities,
        const float3* __restrict__ sh_coefficients_0,
        const float4* __restrict__ sh_coefficients_rest,
        const float2* __restrict__ sh_value_bounds,
        const uint sh_value_n_cells,
        const uint sh_value_bits,
        const float4* __restrict__ w2c,
        const float3* __restrict__ cam_position,
        ScanHeader* __restrict__ scan_header,
        std::uint64_t* __restrict__ scan_tiles,
        std::uint8_t* __restrict__ active_block_flag,
        uint* __restrict__ active_block_ids,
        uint* __restrict__ primitive_work_indices,
        uint* __restrict__ visible_indices,
        uint* __restrict__ primitive_depth_keys,
        float* __restrict__ primitive_depths,
        uint* __restrict__ primitive_n_touched_tiles,
        ushort4* __restrict__ primitive_screen_bounds,
        ProjectedRecord* __restrict__ projected,
        float3* __restrict__ primitive_normals,
        const uint n_primitives,
        const uint visible_cap,
        const uint n_blocks,
        const uint grid_width,
        const uint grid_height,
        const uint sh_layout_slots,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        const float clip_left,
        const float clip_right,
        const float clip_top,
        const float clip_bottom,
        const float near_,
        const float far_,
        const bool mip_filter,
        const float dilation_scale,
        float* __restrict__ max_screen_share) {
        static_assert(config::block_size_preprocess == 256, "single-pass preprocess assumes 256-thread blocks");
        const uint tid = threadIdx.x;
        const uint lane = tid & 31u;
        const uint warp = tid >> 5;

        __shared__ uint s_bid;
        if (tid == 0u) {
            s_bid = atomicAdd(&scan_header->next_block, 1u);
        }
        __syncthreads();
        const uint bid = s_bid;
        const uint primitive_idx = bid * 256u + tid;
        const bool in_range = primitive_idx < n_primitives;
        bool active = in_range;

        const float4 w2c_r1 = w2c[0];
        const float4 w2c_r2 = w2c[1];
        const float4 w2c_r3 = w2c[2];

        const float3 mean3d = active ? means[primitive_idx] : make_float3(0.0f, 0.0f, 0.0f);
        const float depth = w2c_r3.x * mean3d.x + w2c_r3.y * mean3d.y + w2c_r3.z * mean3d.z + w2c_r3.w;
        if (!isfinite(mean3d.x) || !isfinite(mean3d.y) || !isfinite(mean3d.z) ||
            depth < near_ || depth > far_) {
            active = false;
        }

        uint n_touched_tiles = 0u;
        uint4 screen_bounds = make_uint4(0u, 0u, 0u, 0u);
        float2 mean2d = make_float2(0.0f, 0.0f);
        float3 conic = make_float3(0.0f, 0.0f, 0.0f);
        float output_opacity = 0.0f;
        float extent_x_full = 0.0f;
        float extent_y_full = 0.0f;
        float3 cam_normal = make_float3(0.0f, 0.0f, 0.0f);
        float screen_share_val = 0.0f;

        if (__ballot_sync(0xffffffffu, active) != 0u) {
            const float raw_opacity = active ? raw_opacities[primitive_idx] : -100.0f;
            const float opacity = 1.0f / (1.0f + expf(-raw_opacity));
            if (!isfinite(raw_opacity) || opacity < config::min_alpha_threshold) {
                active = false;
            }

            if (__ballot_sync(0xffffffffu, active) != 0u) {
                const float3 raw_scale = active ? raw_scales[primitive_idx] : make_float3(0.0f, 0.0f, 0.0f);
                if (!isfinite(raw_scale.x) || !isfinite(raw_scale.y) || !isfinite(raw_scale.z)) {
                    active = false;
                }

                const float inv_depth = active ? (1.0f / depth) : 0.0f;
                const float x = (w2c_r1.x * mean3d.x + w2c_r1.y * mean3d.y + w2c_r1.z * mean3d.z + w2c_r1.w) * inv_depth;
                const float y = (w2c_r2.x * mean3d.x + w2c_r2.y * mean3d.y + w2c_r2.z * mean3d.z + w2c_r2.w) * inv_depth;
                mean2d = make_float2(x * fx + cx, y * fy + cy);

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

                const float3 clamped_scale = make_float3(
                    fminf(raw_scale.x, config::max_raw_scale),
                    fminf(raw_scale.y, config::max_raw_scale),
                    fminf(raw_scale.z, config::max_raw_scale));
                const float kernel_size = (mip_filter ? config::dilation_mip_filter : config::dilation) * dilation_scale;

                // Conservative frustum rejection using isotropic upper bound on cov2d before loading rotations.
                const float w_tiles = static_cast<float>(grid_width * static_cast<uint>(config::tile_width));
                const float h_tiles = static_cast<float>(grid_height * static_cast<uint>(config::tile_height));
                const float dist_x = fmaxf(-mean2d.x, mean2d.x - w_tiles);
                const float dist_y = fmaxf(-mean2d.y, mean2d.y - h_tiles);
                if (active && (dist_x > 0.0f || dist_y > 0.0f)) {
                    const float max_s = fmaxf(fmaxf(clamped_scale.x, clamped_scale.y), clamped_scale.z);
                    const float max_v = expf(2.0f * max_s);
                    const float max_ext_x = 3.34f * sqrtf(max_v * dot(jw_r1, jw_r1) + kernel_size);
                    const float max_ext_y = 3.34f * sqrtf(max_v * dot(jw_r2, jw_r2) + kernel_size);
                    if (dist_x > max_ext_x + 1.0f || dist_y > max_ext_y + 1.0f) {
                        active = false;
                    }
                }

                if (__ballot_sync(0xffffffffu, active) != 0u) {
                    const float4 raw_rotation = active ? raw_rotations[primitive_idx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    if (!isfinite(raw_rotation.x) || !isfinite(raw_rotation.y) ||
                        !isfinite(raw_rotation.z) || !isfinite(raw_rotation.w)) {
                        active = false;
                    }
                    const float qr = raw_rotation.x;
                    const float qx = raw_rotation.y;
                    const float qy = raw_rotation.z;
                    const float qz = raw_rotation.w;
                    const float qrr_raw = qr * qr, qxx_raw = qx * qx, qyy_raw = qy * qy, qzz_raw = qz * qz;
                    const float q_norm_sq = qrr_raw + qxx_raw + qyy_raw + qzz_raw;
                    if (q_norm_sq < 1e-8f) {
                        active = false;
                    }

                    if (__ballot_sync(0xffffffffu, active) != 0u) {
                        const float3 variance = make_float3(
                            expf(2.0f * clamped_scale.x),
                            expf(2.0f * clamped_scale.y),
                            expf(2.0f * clamped_scale.z));
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
                        const float3 jwc_r1 = make_float3(
                            jw_r1.x * cov3d.m11 + jw_r1.y * cov3d.m12 + jw_r1.z * cov3d.m13,
                            jw_r1.x * cov3d.m12 + jw_r1.y * cov3d.m22 + jw_r1.z * cov3d.m23,
                            jw_r1.x * cov3d.m13 + jw_r1.y * cov3d.m23 + jw_r1.z * cov3d.m33);
                        const float3 jwc_r2 = make_float3(
                            jw_r2.x * cov3d.m11 + jw_r2.y * cov3d.m12 + jw_r2.z * cov3d.m13,
                            jw_r2.x * cov3d.m12 + jw_r2.y * cov3d.m22 + jw_r2.z * cov3d.m23,
                            jw_r2.x * cov3d.m13 + jw_r2.y * cov3d.m23 + jw_r2.z * cov3d.m33);
                        float3 cov2d = make_float3(dot(jwc_r1, jw_r1), dot(jwc_r1, jw_r2), dot(jwc_r2, jw_r2));

                        const float det_raw = mip_filter ? fmaxf(cov2d.x * cov2d.z - cov2d.y * cov2d.y, 0.0f) : 0.0f;
                        cov2d.x += kernel_size;
                        cov2d.z += kernel_size;
                        const float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
                        if (det < config::min_cov2d_determinant) {
                            active = false;
                        }
                        const float det_rcp = 1.0f / det;
                        output_opacity = mip_filter ? opacity * sqrtf(det_raw * det_rcp) : opacity;
                        if (output_opacity < config::min_alpha_threshold) {
                            active = false;
                        }

                        conic = make_float3(cov2d.z * det_rcp, -cov2d.y * det_rcp, cov2d.x * det_rcp);
                        const float power_threshold = logf(output_opacity * config::min_alpha_threshold_rcp);
                        const float power_threshold_factor = sqrtf(2.0f * power_threshold);
                        const float extent_x = fmaxf(power_threshold_factor * sqrtf(cov2d.x) - 0.5f, 0.0f);
                        const float extent_y = fmaxf(power_threshold_factor * sqrtf(cov2d.z) - 0.5f, 0.0f);
                        screen_bounds = compute_screen_tile_bounds(
                            mean2d, extent_x, extent_y, grid_width, grid_height);
                        const uint n_touched_tiles_max =
                            (screen_bounds.y - screen_bounds.x) * (screen_bounds.w - screen_bounds.z);
                        if (n_touched_tiles_max == 0u) {
                            active = false;
                        }

                        if (__ballot_sync(0xffffffffu, active) != 0u) {
                            n_touched_tiles = compute_exact_n_touched_tiles(
                                mean2d, conic, screen_bounds, power_threshold, active);
                            if (n_touched_tiles == 0u) {
                                active = false;
                            }
                            if (active) {
                                extent_x_full = power_threshold_factor * sqrtf(cov2d.x);
                                extent_y_full = power_threshold_factor * sqrtf(cov2d.z);
                                if (max_screen_share != nullptr) {
                                    const float3 campos = cam_position[0];
                                    screen_share_val = lfs::training::gaussian_screen_share(
                                        mean3d.x, mean3d.y, mean3d.z,
                                        campos.x, campos.y, campos.z,
                                        raw_scale.x, raw_scale.y, raw_scale.z,
                                        raw_opacity);
                                }
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
                                    cam_normal = make_float3(
                                        w2c_r1.x * normal_world.x + w2c_r1.y * normal_world.y + w2c_r1.z * normal_world.z,
                                        w2c_r2.x * normal_world.x + w2c_r2.y * normal_world.y + w2c_r2.z * normal_world.z,
                                        w2c_r3.x * normal_world.x + w2c_r3.y * normal_world.y + w2c_r3.z * normal_world.z);
                                }
                            }
                        }
                    }
                }
            }
        }

        const bool visible = active;
        const unsigned vis_ballot = __ballot_sync(0xffffffffu, visible);
        const uint lane_vis_excl = __popc(vis_ballot & ((1u << lane) - 1u));
        const uint warp_vis_total = __popc(vis_ballot);

        std::uint64_t warp_inst = visible ? static_cast<std::uint64_t>(n_touched_tiles) : 0ull;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_inst += __shfl_down_sync(0xffffffffu, warp_inst, offset);
        }

        __shared__ uint s_warp_vis[8];
        __shared__ uint s_warp_vis_off[8];
        __shared__ std::uint64_t s_warp_inst[8];
        __shared__ uint s_prefix_vis;

        if (lane == 0u) {
            s_warp_vis[warp] = warp_vis_total;
            s_warp_inst[warp] = warp_inst;
        }
        __syncthreads();

        if (warp == 0u) {
            uint block_vis = 0u;
            if (lane == 0u) {
                std::uint64_t block_inst = 0ull;
#pragma unroll
                for (uint w = 0u; w < 8u; ++w) {
                    s_warp_vis_off[w] = block_vis;
                    block_vis += s_warp_vis[w];
                    block_inst += s_warp_inst[w];
                }
                if (block_vis > 0u) {
                    if (active_block_flag != nullptr)
                        active_block_flag[bid] = 1u;
                    if (active_block_ids != nullptr)
                        active_block_ids[atomicAdd(&scan_header->n_active_blocks, 1u)] = bid;
                    if (block_inst > 0ull) {
                        atomicAdd(reinterpret_cast<unsigned long long*>(&scan_header->n_instances),
                                  static_cast<unsigned long long>(block_inst));
                    }
                }
                store_release_u64(&scan_tiles[bid],
                                  ((bid == 0u ? 2ull : 1ull) << 62) | static_cast<std::uint64_t>(block_vis));
            }
            block_vis = __shfl_sync(0xffffffffu, block_vis, 0);
            uint excl_vis = 0u;
            if (bid != 0u) {
                // Warp-parallel decoupled lookback: each probe inspects 32 predecessors
                // and stops at the nearest inclusive prefix (deterministic integer sum).
                int look_base = static_cast<int>(bid) - 1;
                while (true) {
                    const int look = look_base - static_cast<int>(lane);
                    std::uint64_t word = 2ull << 62;
                    if (look >= 0) {
                        do {
                            word = load_acquire_u64(&scan_tiles[look]);
                        } while ((word >> 62) == 0ull);
                    }
                    const unsigned inc_mask = __ballot_sync(0xffffffffu, (word >> 62) == 2ull);
                    const int first_inc = (inc_mask != 0u) ? (__ffs(inc_mask) - 1) : 32;
                    uint add = (static_cast<int>(lane) <= first_inc) ? static_cast<uint>(word & 0xffffffffull) : 0u;
#pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        add += __shfl_xor_sync(0xffffffffu, add, offset);
                    excl_vis += add;
                    if (inc_mask != 0u)
                        break;
                    look_base -= 32;
                }
                if (lane == 0u)
                    store_release_u64(&scan_tiles[bid], (2ull << 62) | static_cast<std::uint64_t>(excl_vis + block_vis));
            }
            if (lane == 0u) {
                s_prefix_vis = excl_vis;
                if (bid == n_blocks - 1u) {
                    scan_header->n_visible = excl_vis + block_vis;
                }
            }
        }
        __syncthreads();

        if (!in_range)
            return;

        const uint work_idx = s_prefix_vis + s_warp_vis_off[warp] + lane_vis_excl;
        primitive_work_indices[primitive_idx] = visible ? work_idx : 0xffffffffu;
        if (!visible)
            return;

        if (max_screen_share != nullptr) {
            lfs::training::atomic_max_float(max_screen_share + primitive_idx, screen_share_val);
        }
        if (work_idx >= visible_cap)
            return;

        visible_indices[work_idx] = primitive_idx;
        primitive_n_touched_tiles[work_idx] = n_touched_tiles;
        primitive_screen_bounds[work_idx] = make_ushort4(
            static_cast<ushort>(screen_bounds.x),
            static_cast<ushort>(screen_bounds.y),
            static_cast<ushort>(screen_bounds.z),
            static_cast<ushort>(screen_bounds.w));

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
        const float3 sh_color = convert_sh_to_color_fast<ACTIVE_SH_BASES, SH_Q16>(
            sh_coefficients_0, sh_coefficients_rest,
            mean3d, cam_position[0],
            primitive_idx, sh_layout_slots,
            sh_value_bounds, sh_value_n_cells, sh_value_bits);
        projected[work_idx].color = make_float4(sh_color, 0.0f);
        primitive_depth_keys[work_idx] = depth_sort_key(depth);
        primitive_depths[work_idx] = depth;
        if (primitive_normals != nullptr) {
            primitive_normals[work_idx] = cam_normal;
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
        const uint* __restrict__ primitive_n_touched_tiles,
        const uint* __restrict__ primitive_offsets,
        const uint* __restrict__ primitive_depth_keys,
        const ushort4* __restrict__ primitive_screen_bounds,
        const ProjectedRecord* __restrict__ projected,
        const uint* __restrict__ visible_primitive_indices,
        const uint* __restrict__ depth_work_order,
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

        const uint work_idx = DepthOrdered ? depth_work_order[idx] : idx;
        const uint primitive_idx = active ? visible_primitive_indices[work_idx] : 0u;
        const uint n_touched_tiles = active ? primitive_n_touched_tiles[work_idx] : 0u;
        active = active && n_touched_tiles > 0u;

        if (__ballot_sync(0xffffffffu, active) == 0u)
            return;

        const ushort4 screen_bounds = active ? primitive_screen_bounds[work_idx] : make_ushort4(0, 0, 0, 0);
        const uint4 diagnostic_bounds = make_uint4(screen_bounds.x, screen_bounds.y, screen_bounds.z, screen_bounds.w);
        // Tile-only keys (depth-ordered path) never read the depth key.
        constexpr bool kPackedKey = sizeof(Key) == sizeof(InstanceKey);
        const uint depth_key = (kPackedKey && active) ? primitive_depth_keys[work_idx] : 0u;
        const uint write_offset_end_raw = active ? primitive_offsets[idx] : 0u;
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

        uint current_write_offset = idx == 0 ? 0u : primitive_offsets[idx - 1];
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
        const uint box_area = screen_bounds_width * screen_bounds_height;
        const bool full_box = active && (n_touched_tiles == box_area) && (n_scan < 32u);

        // Fast path: when every active lane's ellipse covers its entire tile bounding box,
        // emit the box directly without loading projected[work_idx] or running logf/sqrtf.
        if (__ballot_sync(0xffffffffu, active && !full_box) == 0u) {
            if (active) {
                for (uint scan_index = scan0; scan_index < scan1 && current_write_offset < write_offset_end; scan_index++) {
                    emit_ellipse_tile_span(
                        make_uint2(cross0, cross1), scan_index, scan_along_x, grid_width, depth_key, primitive_idx,
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
            return;
        }

        const bool load_ellipse = active && !full_box;
        const float2 mean2d_shifted = load_ellipse ? projected[work_idx].geometry.mean2d - 0.5f : make_float2(0.0f, 0.0f);
        const float4 conic_opacity_loaded = load_ellipse ? projected[work_idx].conic_opacity : make_float4(0.0f, 0.0f, 0.0f, config::min_alpha_threshold);
        const float3 conic = make_float3(conic_opacity_loaded);
        const float power_threshold_precomputed = load_ellipse ? logf(conic_opacity_loaded.w * config::min_alpha_threshold_rcp) : 0.0f;
        const float radius_sq = 2.0f * power_threshold_precomputed;

        const uint max_scan = lfs::core::warp_ops::warp_reduce_max(n_scan);

        if (max_scan < 32u) {
            if (active) {
                if (full_box) {
                    for (uint scan_index = scan0; scan_index < scan1 && current_write_offset < write_offset_end; scan_index++) {
                        emit_ellipse_tile_span(
                            make_uint2(cross0, cross1), scan_index, scan_along_x, grid_width, depth_key, primitive_idx,
                            current_write_offset, write_offset_end, instance_keys, instance_primitive_indices);
                    }
                } else {
                    for (uint scan_index = scan0; scan_index < scan1 && current_write_offset < write_offset_end; scan_index++) {
                        const uint2 span = ellipse_touched_tile_span(
                            conic, radius_sq, mean2d_shifted, scan_along_x, scan_index, cross0, cross1);
                        emit_ellipse_tile_span(
                            span, scan_index, scan_along_x, grid_width, depth_key, primitive_idx,
                            current_write_offset, write_offset_end, instance_keys, instance_primitive_indices);
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
        } else {
            if (full_box) {
                for (uint scan_index = scan0; scan_index < scan1 && current_write_offset < write_offset_end; scan_index++) {
                    emit_ellipse_tile_span(
                        make_uint2(cross0, cross1), scan_index, scan_along_x, grid_width, depth_key, primitive_idx,
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
            create_instances_coop_warp(
                load_ellipse, load_ellipse ? n_scan : 0u, scan0, cross0, cross1, write_offset_end, current_write_offset,
                primitive_idx, depth_key, scan_along_x, mean2d_shifted, conic, radius_sq,
                grid_width, diagnostic_bounds, instance_keys, instance_primitive_indices, status);
        }
    }

    // Per-pixel background: CHW image if given, else solid color, else black.
    __device__ __forceinline__ float3 load_pixel_background(
        const float* bg_image,
        const float* bg_color,
        const int pixel_idx,
        const int n_pixels) {
        if (bg_image != nullptr)
            return make_float3(bg_image[pixel_idx], bg_image[pixel_idx + n_pixels], bg_image[pixel_idx + 2 * n_pixels]);
        if (bg_color != nullptr)
            return make_float3(bg_color[0], bg_color[1], bg_color[2]);
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    // Front-to-back blend. Eight warps own one 8x4 sub-tile each (1 pixel/thread),
    // staging up to 256 splats per batch with per-chunk warp coverage masks (no SMEM
    // atomics) and ballot compaction.
    //
    // warp_cull_mode: 0 = enabled (production), 1 = disabled (mask all-1s, reference),
    //                 2 = deliberately incorrect empty mask for negative coverage.
    // blend_batch_size_runtime: multiple of 32 in [32, block_size_blend_forward]; 0 -> config default.
    template <bool kRenderNormal, bool kRenderDepth>
    __global__ void __launch_bounds__(config::block_size_blend_forward, (kRenderNormal || kRenderDepth) ? 3 : 5) blend_cu(
        uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ tile_order,
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
        static_assert(config::tile_width == 16 && config::tile_height == 16, "sub-tile layout assumes 16x16 tiles");
        static_assert(config::block_size_blend_forward == 256, "eight warps, one 8x4 sub-tile each");
        static_assert(config::warp_subtile_width == 8 && config::warp_subtile_height == 4,
                      "coverage bits address 8x4 regions");

        const uint thread_rank = block.thread_rank();
        const uint lane = thread_rank & 31u;
        const uint warp = thread_rank >> 5;
        const uint tile_x = (warp & 1u) * 8u + (lane & 7u);
        const uint tile_y = (warp >> 1) * 4u + (lane >> 3);

        const uint tile_idx = (tile_order != nullptr) ? tile_order[blockIdx.x] : blockIdx.x;
        const uint tile_col = tile_idx % grid_width;
        const uint tile_row = tile_idx / grid_width;
        const uint origin_x = tile_col * static_cast<uint>(config::tile_width);
        const uint origin_y = tile_row * static_cast<uint>(config::tile_height);
        const uint pixel_x = origin_x + tile_x;
        const uint pixel_y = origin_y + tile_y;
        const float center_x = __uint2float_rn(pixel_x) + 0.5f;
        // Match the dual-row (mean2d.y - center_y0) - row_off rounding to preserve exact pixel values.
        const float center_y_pair = __uint2float_rn(pixel_y & ~1u) + 0.5f;
        const float dy_row_off = (pixel_y & 1u) ? 1.0f : 0.0f;

        const bool inside = pixel_x < width && pixel_y < height;
        float transmittance = signed_transmittance(1.0f, !inside);
        float3 color = make_float3(0.0f);
        float3 normal = make_float3(0.0f);
        float depth = 0.0f;
        uint contributions = 0u;

        const uint2 tile_range = tile_instance_ranges[tile_idx];
        const int n_points_total = static_cast<int>(tile_range.y - tile_range.x);
        int batch_size = blend_batch_size_runtime > 0 ? blend_batch_size_runtime : config::blend_batch_size;
        batch_size = max(32, min(batch_size, config::block_size_blend_forward)) & ~31;

        __shared__ uint collected_ids[config::block_size_blend_forward];
        __shared__ unsigned char collected_regions[config::block_size_blend_forward];
        __shared__ float2 collected_mean2d[config::block_size_blend_forward];
        __shared__ float4 collected_conic_opacity[config::block_size_blend_forward];
        __shared__ float3 collected_color[config::block_size_blend_forward];
        __shared__ float collected_depth[kRenderDepth ? config::block_size_blend_forward : 1];
        __shared__ float3 collected_normal[kRenderNormal ? config::block_size_blend_forward : 1];
        __shared__ uint s_chunk_blended[8][8];
        uint kept_count = 0u;

        for (int n_points_remaining = n_points_total, batch_base = 0;
             n_points_remaining > 0;
             n_points_remaining -= batch_size, batch_base += batch_size) {
            if (static_cast<int>(thread_rank) < batch_size) {
                const int fetch_idx = static_cast<int>(tile_range.x) + batch_base + static_cast<int>(thread_rank);
                unsigned regions = 0u;
                if (fetch_idx < static_cast<int>(tile_range.y)) {
                    const uint primitive_idx = instance_primitive_indices[fetch_idx];
                    collected_ids[thread_rank] = primitive_idx;
                    const uint work_idx = primitive_work_indices[primitive_idx];
                    const ProjectedRecord record = projected[work_idx];
                    const ushort4 bb = record.geometry.pixel_bbox;
#pragma unroll
                    for (uint band = 0u; band < 4u; ++band) {
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
                        fminf(fmaxf(make_float3(record.color), 0.0f), config::max_blend_color);
                    if constexpr (kRenderDepth)
                        collected_depth[thread_rank] = primitive_depths[work_idx];
                    if constexpr (kRenderNormal)
                        collected_normal[thread_rank] = primitive_normals[work_idx];
                }
                collected_regions[thread_rank] = static_cast<unsigned char>(regions);
            }
            block.sync();

            const int current_batch_size = min(batch_size, n_points_remaining);
            for (int chunk = 0, j_base = 0; j_base < current_batch_size; ++chunk, j_base += 32) {
                const int j_test = j_base + static_cast<int>(lane);
                const unsigned regions = j_test < current_batch_size ? collected_regions[j_test] : 0u;
                unsigned mine = __ballot_sync(0xffffffffu, (regions >> warp) & 1u);
                if (warp_cull_mode == 1) {
                    mine = 0xffffffffu >> (32 - min(32, current_batch_size - j_base));
                } else if (warp_cull_mode == 2) {
                    mine = 0u;
                }
                if (!__any_sync(0xffffffffu, !transmittance_complete(transmittance))) {
                    mine = 0u;
                }
                unsigned blended = 0u;
                while (mine) {
                    const int k = __ffs(mine) - 1;
                    mine &= mine - 1;
                    const int j = j_base + k;
                    const float4 conic_opacity = collected_conic_opacity[j];
                    const float3 conic = make_float3(conic_opacity);
                    const float2 mean2d = collected_mean2d[j];
                    const float3 splat_color = collected_color[j];
                    float splat_depth = 0.0f;
                    if constexpr (kRenderDepth)
                        splat_depth = collected_depth[j];
                    float3 splat_normal = make_float3(0.0f);
                    if constexpr (kRenderNormal)
                        splat_normal = collected_normal[j];

                    const bool live = !transmittance_complete(transmittance);
                    const AlphaColumn column = AlphaColumn::at(conic, conic_opacity.w, mean2d.x - center_x);
                    const float dy = (mean2d.y - center_y_pair) - dy_row_off;
                    const float unclamped = live ? column.alpha(conic.z, dy) : 0.0f;
                    const bool accepted = live && alpha_accepted(unclamped);
                    const float alpha = accepted ? clamped_alpha(unclamped) : 0.0f;
                    const float magnitude = transmittance_magnitude(transmittance);
                    const float weight = magnitude * alpha;
                    color += weight * splat_color;
                    if constexpr (kRenderDepth)
                        depth += weight * splat_depth;
                    if constexpr (kRenderNormal)
                        normal += weight * splat_normal;
                    const float after = magnitude * (1.0f - alpha);
                    const bool complete = transmittance_complete(transmittance) ||
                                          (accepted && after < config::transmittance_threshold);
                    transmittance = signed_transmittance(after, complete);
                    contributions = accepted ? static_cast<uint>(batch_base + j + 1) : contributions;
                    blended |= accepted ? (1u << k) : 0u;
                }
                __syncwarp();
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    blended |= __shfl_xor_sync(0xffffffffu, blended, offset);
                if (lane == 0u)
                    s_chunk_blended[chunk][warp] = blended;
            }

            // Single barrier serves both s_chunk_blended visibility and all-done early-exit check.
            const bool all_done =
                (__syncthreads_count(transmittance_complete(transmittance)) == config::block_size_blend_forward);
            const uint n_chunks = static_cast<uint>((current_batch_size + 31) >> 5);
            uint coverage = 0u;
            if (static_cast<int>(thread_rank) < current_batch_size) {
                const unsigned bit = 1u << lane;
#pragma unroll
                for (uint r = 0u; r < 8u; ++r) {
                    if (s_chunk_blended[warp][r] & bit)
                        coverage |= 1u << r;
                }
            }
            const bool kept = coverage != 0u;
            const unsigned kept_mask = __ballot_sync(0xffffffffu, kept);
            const uint lane_rank = __popc(kept_mask & ((1u << lane) - 1u));

            // Compute per-chunk kept counts directly from s_chunk_blended without a second barrier.
            uint chunk_cnt = 0u;
            if (lane < n_chunks) {
                const uint any_blended =
                    s_chunk_blended[lane][0] | s_chunk_blended[lane][1] |
                    s_chunk_blended[lane][2] | s_chunk_blended[lane][3] |
                    s_chunk_blended[lane][4] | s_chunk_blended[lane][5] |
                    s_chunk_blended[lane][6] | s_chunk_blended[lane][7];
                chunk_cnt = __popc(any_blended);
            }
            uint warp_offset = 0u;
            uint total = 0u;
#pragma unroll
            for (uint w = 0u; w < 8u; ++w) {
                const uint c = __shfl_sync(0xffffffffu, chunk_cnt, w);
                if (w < warp)
                    warp_offset += c;
                total += c;
            }
            if (kept) {
                const uint destination = tile_range.x + kept_count + warp_offset + lane_rank;
                instance_primitive_indices[destination] = collected_ids[thread_rank];
                contributor_masks[destination] = static_cast<unsigned char>(coverage);
                contributor_positions[destination] = batch_base + thread_rank;
            }
            kept_count += total;
            if (all_done)
                break;
        }
        if (thread_rank == 0u)
            tile_instance_ranges[tile_idx].y = tile_range.x + kept_count;

        const int n_pixels = width * height;
        const float final_transmittance = transmittance_magnitude(transmittance);
        const uint pixel_rank = tile_y * static_cast<uint>(config::tile_width) + tile_x;
        tile_final_transmittance[tile_idx * config::block_size_blend + pixel_rank] = inside ? final_transmittance : 1.0f;
        if (!inside)
            return;
        const int pixel_idx = width * pixel_y + pixel_x;
        const float3 bg = load_pixel_background(bg_image, bg_color, pixel_idx, n_pixels);
        image[pixel_idx] = color.x + final_transmittance * bg.x;
        image[pixel_idx + n_pixels] = color.y + final_transmittance * bg.y;
        image[pixel_idx + n_pixels * 2] = color.z + final_transmittance * bg.z;
        alpha_map[pixel_idx] = 1.0f - final_transmittance;
        if constexpr (kRenderDepth)
            depth_map[pixel_idx] = depth;
        if constexpr (kRenderNormal) {
            normal_map[pixel_idx] = normal.x;
            normal_map[pixel_idx + n_pixels] = normal.y;
            normal_map[pixel_idx + n_pixels * 2] = normal.z;
        }
        tile_n_contributions[pixel_idx] = contributions;
    }

    // Front-to-back RGB blend split into four 64-thread quadrants per 16x16 tile: each
    // CTA owns one 16x4 band (two 8x4 warps), exits as soon as its own 64 pixels are
    // complete, and leaves the tile instance list uncompacted. Per-instance coverage
    // bits are OR-ed into contributor_masks (zeroed by the caller), so the entry index
    // is the instance's original position and no position array is needed.
    __global__ void __launch_bounds__(64, 20) blend_quadrant_cu(
        const uint2* __restrict__ tile_instance_ranges,
        const uint* __restrict__ tile_order,
        const uint* __restrict__ instance_primitive_indices,
        unsigned char* __restrict__ contributor_masks,
        const uint* __restrict__ primitive_work_indices,
        const ProjectedRecord* __restrict__ projected,
        float* __restrict__ image,
        float* __restrict__ alpha_map,
        uint* __restrict__ tile_n_contributions,
        float* __restrict__ tile_final_transmittance,
        const float* __restrict__ bg_color,
        const float* __restrict__ bg_image,
        const uint width,
        const uint height,
        const uint grid_width) {
        static_assert(config::tile_width == 16 && config::tile_height == 16, "quadrant layout assumes 16x16 tiles");
        static_assert(config::warp_subtile_width == 8 && config::warp_subtile_height == 4,
                      "coverage bits address 8x4 regions");
        constexpr int kBatch = 64;

        const uint thread_rank = threadIdx.x;
        const uint lane = thread_rank & 31u;
        const uint local_warp = thread_rank >> 5;
        const uint tile_pos = blockIdx.x >> 2;
        const uint band = blockIdx.x & 3u;
        const uint sub_base = band << 1;
        const uint warp = sub_base + local_warp;
        const uint tile_x = (warp & 1u) * 8u + (lane & 7u);
        const uint tile_y = (warp >> 1) * 4u + (lane >> 3);

        const uint tile_idx = (tile_order != nullptr) ? tile_order[tile_pos] : tile_pos;
        const uint origin_x = (tile_idx % grid_width) * static_cast<uint>(config::tile_width);
        const uint origin_y = (tile_idx / grid_width) * static_cast<uint>(config::tile_height);
        const uint pixel_x = origin_x + tile_x;
        const uint pixel_y = origin_y + tile_y;
        const float center_x = __uint2float_rn(pixel_x) + 0.5f;
        // Match the dual-row (mean2d.y - center_y0) - row_off rounding to preserve exact pixel values.
        const float center_y_pair = __uint2float_rn(pixel_y & ~1u) + 0.5f;
        const float dy_row_off = (pixel_y & 1u) ? 1.0f : 0.0f;

        const bool inside = pixel_x < width && pixel_y < height;
        float transmittance = signed_transmittance(1.0f, !inside);
        float3 color = make_float3(0.0f);
        uint contributions = 0u;

        const uint2 tile_range = tile_instance_ranges[tile_idx];
        const int n_points_total = static_cast<int>(tile_range.y - tile_range.x);
        const uint band_y0 = origin_y + band * 4u;

        __shared__ float2 collected_mean2d[kBatch];
        __shared__ float4 collected_conic_opacity[kBatch];
        __shared__ float3 collected_color[kBatch];
        __shared__ unsigned char collected_regions[kBatch];
        __shared__ uint s_chunk_blended[2][2];

        for (int batch_base = 0; batch_base < n_points_total; batch_base += kBatch) {
            const int fetch_idx = static_cast<int>(tile_range.x) + batch_base + static_cast<int>(thread_rank);
            {
                unsigned regions = 0u;
                if (fetch_idx < static_cast<int>(tile_range.y)) {
                    const uint primitive_idx = instance_primitive_indices[fetch_idx];
                    const uint work_idx = primitive_work_indices[primitive_idx];
                    const ProjectedRecord record = projected[work_idx];
                    const ushort4 bb = record.geometry.pixel_bbox;
                    if (static_cast<uint>(bb.z) < band_y0 + 4u && static_cast<uint>(bb.w) > band_y0) {
                        if (static_cast<uint>(bb.x) < origin_x + 8u && static_cast<uint>(bb.y) > origin_x)
                            regions |= 1u;
                        if (static_cast<uint>(bb.x) < origin_x + 16u && static_cast<uint>(bb.y) > origin_x + 8u)
                            regions |= 2u;
                    }
                    collected_mean2d[thread_rank] = record.geometry.mean2d;
                    collected_conic_opacity[thread_rank] =
                        make_float4(base2_conic(make_float3(record.conic_opacity)), record.conic_opacity.w);
                    collected_color[thread_rank] =
                        fminf(fmaxf(make_float3(record.color), 0.0f), config::max_blend_color);
                }
                collected_regions[thread_rank] = static_cast<unsigned char>(regions);
            }
            __syncthreads();

            const int current_batch_size = min(kBatch, n_points_total - batch_base);
#pragma unroll
            for (int chunk = 0; chunk < 2; ++chunk) {
                const int j_base = chunk * 32;
                unsigned blended = 0u;
                if (j_base < current_batch_size) {
                    const int j_test = j_base + static_cast<int>(lane);
                    const unsigned regions = j_test < current_batch_size ? collected_regions[j_test] : 0u;
                    unsigned mine = __ballot_sync(0xffffffffu, (regions >> local_warp) & 1u);
                    if (!__any_sync(0xffffffffu, !transmittance_complete(transmittance)))
                        mine = 0u;
                    while (mine) {
                        const int k = __ffs(mine) - 1;
                        mine &= mine - 1;
                        const int j = j_base + k;
                        const float4 conic_opacity = collected_conic_opacity[j];
                        const float3 conic = make_float3(conic_opacity);
                        const float2 mean2d = collected_mean2d[j];
                        const float3 splat_color = collected_color[j];

                        const bool live = !transmittance_complete(transmittance);
                        const AlphaColumn column = AlphaColumn::at(conic, conic_opacity.w, mean2d.x - center_x);
                        const float dy = (mean2d.y - center_y_pair) - dy_row_off;
                        const float unclamped = live ? column.alpha(conic.z, dy) : 0.0f;
                        const bool accepted = live && alpha_accepted(unclamped);
                        const float alpha = accepted ? clamped_alpha(unclamped) : 0.0f;
                        const float magnitude = transmittance_magnitude(transmittance);
                        const float weight = magnitude * alpha;
                        color += weight * splat_color;
                        const float after = magnitude * (1.0f - alpha);
                        const bool complete = transmittance_complete(transmittance) ||
                                              (accepted && after < config::transmittance_threshold);
                        transmittance = signed_transmittance(after, complete);
                        contributions = accepted ? static_cast<uint>(batch_base + j + 1) : contributions;
                        blended |= accepted ? (1u << k) : 0u;
                    }
                    __syncwarp();
#if __CUDA_ARCH__ >= 800
                    blended = __reduce_or_sync(0xffffffffu, blended);
#else
#pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        blended |= __shfl_xor_sync(0xffffffffu, blended, offset);
#endif
                }
                if (lane == 0u)
                    s_chunk_blended[chunk][local_warp] = blended;
            }

            // One barrier publishes the chunk masks and checks whether the band is done.
            const bool all_done = __syncthreads_count(transmittance_complete(transmittance)) == kBatch;
            {
                // Consecutive lanes own consecutive mask bytes; OR the up to four bytes
                // of one 32-bit word inside the warp so its lowest lane issues one atomic.
                const uint chunk = thread_rank >> 5;
                const uint cov2 = static_cast<int>(thread_rank) < current_batch_size
                                      ? ((s_chunk_blended[chunk][0] >> lane) & 1u) |
                                            (((s_chunk_blended[chunk][1] >> lane) & 1u) << 1)
                                      : 0u;
                const uint idx = static_cast<uint>(fetch_idx);
                const uint byte_in_word = idx & 3u;
                uint word_bits = (cov2 << sub_base) << (byte_in_word * 8u);
                const uint next1 = __shfl_down_sync(0xffffffffu, word_bits, 1);
                if (byte_in_word != 3u && lane != 31u)
                    word_bits |= next1;
                const uint next2 = __shfl_down_sync(0xffffffffu, word_bits, 2);
                if (byte_in_word < 2u && lane < 30u)
                    word_bits |= next2;
                if ((byte_in_word == 0u || lane == 0u) && word_bits != 0u)
                    atomicOr(reinterpret_cast<unsigned int*>(contributor_masks) + (idx >> 2), word_bits);
            }
            if (all_done)
                break;
        }

        const int n_pixels = width * height;
        const float final_transmittance = transmittance_magnitude(transmittance);
        const uint pixel_rank = tile_y * static_cast<uint>(config::tile_width) + tile_x;
        tile_final_transmittance[tile_idx * config::block_size_blend + pixel_rank] = inside ? final_transmittance : 1.0f;
        if (!inside)
            return;
        const int pixel_idx = width * pixel_y + pixel_x;
        const float3 bg = load_pixel_background(bg_image, bg_color, pixel_idx, n_pixels);
        image[pixel_idx] = color.x + final_transmittance * bg.x;
        image[pixel_idx + n_pixels] = color.y + final_transmittance * bg.y;
        image[pixel_idx + n_pixels * 2] = color.z + final_transmittance * bg.z;
        alpha_map[pixel_idx] = 1.0f - final_transmittance;
        tile_n_contributions[pixel_idx] = contributions;
    }

} // namespace fast_lfs::rasterization::kernels::forward
