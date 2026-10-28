/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../../../kernels/kernel_stream.hpp"
#include "buffer_utils.h"
#include "helper_math.h"
#include "kernel_utils.cuh"
#include "lfs/core/warp_reduce.cuh"
#include "pop_scores.h"
#include "rasterization_config.h"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace fast_lfs::rasterization {
    namespace {
        // Same tile walk as the backward blend: 4 warps x 2 pixels cover a 16x16
        // tile and each warp owns two 8x4 sub-tiles, so splats are culled and
        // reduced per warp instead of every pixel issuing its own atomic.
        constexpr int kPopBlock = 128;
        constexpr int kPopWarps = kPopBlock / 32;
        static_assert(config::tile_width == 16 && config::tile_height == 16, "pop scores assume 16x16 tiles");
        static_assert(config::warp_subtile_width == 8 && config::warp_subtile_height == 4, "pop scores assume 8x4 sub-tiles");
        static_assert(config::block_size_blend == 2 * kPopBlock, "each thread owns two pixels");

        __global__ void __launch_bounds__(kPopBlock) pop_scores_cu(
            const uint2* __restrict__ tile_instance_ranges,
            const uint* __restrict__ instance_primitive_indices,
            const uint* __restrict__ primitive_work_indices,
            const PackedMeanBBox* __restrict__ primitive_mean2d,
            const float4* __restrict__ primitive_conic_opacity,
            const float4* __restrict__ primitive_color,
            const uint* __restrict__ tile_n_contributions,
            const float* __restrict__ tile_final_transmittance,
            const float* __restrict__ background,
            const float* __restrict__ background_image,
            double* __restrict__ scores,
            const uint n_instances,
            const uint n_primitives,
            const uint n_visible,
            const uint width,
            const uint height,
            const uint grid_width) {
            auto block = cg::this_thread_block();
            const uint tile_idx = block.group_index().x;
            const uint thread_rank = block.thread_rank();
            const uint2 range = tile_instance_ranges[tile_idx];
            if (range.x >= range.y || range.y > n_instances)
                return;
            const int tile_n_primitives = static_cast<int>(range.y - range.x);

            const uint lane_id = thread_rank & 31u;
            const uint warp_id = thread_rank >> 5;
            const uint local_x = lane_id & 7u;
            const uint local_y = lane_id >> 3;
            const uint st0 = warp_id;
            const uint st1 = warp_id + 4;
            const uint st0_ox = (st0 & 1u) * config::warp_subtile_width;
            const uint st0_oy = (st0 >> 1) * config::warp_subtile_height;
            const uint st1_ox = (st1 & 1u) * config::warp_subtile_width;
            const uint st1_oy = (st1 >> 1) * config::warp_subtile_height;
            const uint tile_x0 = (tile_idx % grid_width) * config::tile_width;
            const uint tile_y0 = (tile_idx / grid_width) * config::tile_height;
            const uint2 pix0 = {tile_x0 + st0_ox + local_x, tile_y0 + st0_oy + local_y};
            const uint2 pix1 = {tile_x0 + st1_ox + local_x, tile_y0 + st1_oy + local_y};
            const bool inside0 = pix0.x < width && pix0.y < height;
            const bool inside1 = pix1.x < width && pix1.y < height;
            const uint n_pixels = width * height;
            const uint pixel_idx0 = inside0 ? width * pix0.y + pix0.x : 0u;
            const uint pixel_idx1 = inside1 ? width * pix1.y + pix1.x : 0u;
            const float2 pixel0 = make_float2(static_cast<float>(pix0.x) + 0.5f, static_cast<float>(pix0.y) + 0.5f);
            const float2 pixel1 = make_float2(static_cast<float>(pix1.x) + 0.5f, static_cast<float>(pix1.y) + 0.5f);
            const uint rank0 = (st0_oy + local_y) * config::tile_width + st0_ox + local_x;
            const uint rank1 = (st1_oy + local_y) * config::tile_width + st1_ox + local_x;

            const float sub_w = static_cast<float>(config::warp_subtile_width);
            const float sub_h = static_cast<float>(config::warp_subtile_height);
            const uint sub0_ix0 = tile_x0 + st0_ox, sub0_iy0 = tile_y0 + st0_oy;
            const uint sub1_ix0 = tile_x0 + st1_ox, sub1_iy0 = tile_y0 + st1_oy;
            const uint sub0_ix1 = sub0_ix0 + config::warp_subtile_width, sub0_iy1 = sub0_iy0 + config::warp_subtile_height;
            const uint sub1_ix1 = sub1_ix0 + config::warp_subtile_width, sub1_iy1 = sub1_iy0 + config::warp_subtile_height;

            const uint last0 = inside0 ? tile_n_contributions[pixel_idx0] : 0u;
            const uint last1 = inside1 ? tile_n_contributions[pixel_idx1] : 0u;
            float T0 = inside0 ? tile_final_transmittance[tile_idx * config::block_size_blend + rank0] : 1.0f;
            float T1 = inside1 ? tile_final_transmittance[tile_idx * config::block_size_blend + rank1] : 1.0f;
            const auto background_at = [&](const uint pixel_idx) {
                if (background_image)
                    return make_float3(background_image[pixel_idx], background_image[n_pixels + pixel_idx],
                                       background_image[2 * n_pixels + pixel_idx]);
                return background ? make_float3(background[0], background[1], background[2]) : make_float3(0.0f);
            };
            float3 behind0 = inside0 ? background_at(pixel_idx0) : make_float3(0.0f);
            float3 behind1 = inside1 ? background_at(pixel_idx1) : make_float3(0.0f);

            uint max_last = ::max(last0, last1);
#pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
                max_last = ::max(max_last, __shfl_xor_sync(0xffffffffu, max_last, offset));
            __shared__ uint s_warp_last[kPopWarps];
            __shared__ int s_count;
            if (lane_id == 0u)
                s_warp_last[warp_id] = max_last;
            block.sync();
            if (thread_rank == 0u) {
                uint m = 0u;
#pragma unroll
                for (int w = 0; w < kPopWarps; ++w)
                    m = ::max(m, s_warp_last[w]);
                s_count = min(tile_n_primitives, static_cast<int>(m));
            }
            block.sync();
            const int count = s_count;
            if (count <= 0)
                return;

            __shared__ uint s_prim_idx[kPopBlock];
            __shared__ float2 s_mean2d[kPopBlock];
            __shared__ ushort4 s_bbox[kPopBlock];
            __shared__ float4 s_conic_opacity[kPopBlock];
            __shared__ float3 s_color[kPopBlock];
            __shared__ unsigned char s_valid[kPopBlock];

            // Back to front, so the color behind each splat is known when it is scored.
            for (int batch_base = 0; batch_base < count; batch_base += kPopBlock) {
                const int n_batch = min(kPopBlock, count - batch_base);
                if (static_cast<int>(thread_rank) < n_batch) {
                    const int tile_primitive_idx = count - 1 - (batch_base + static_cast<int>(thread_rank));
                    const uint primitive_idx = instance_primitive_indices[range.x + static_cast<uint>(tile_primitive_idx)];
                    const uint work_idx = primitive_idx < n_primitives ? primitive_work_indices[primitive_idx] : n_visible;
                    const bool valid = work_idx < n_visible;
                    s_valid[thread_rank] = valid ? 1u : 0u;
                    s_prim_idx[thread_rank] = valid ? primitive_idx : 0u;
                    if (valid) {
                        const PackedMeanBBox geom = primitive_mean2d[work_idx];
                        s_mean2d[thread_rank] = geom.mean2d;
                        s_bbox[thread_rank] = geom.pixel_bbox;
                        s_conic_opacity[thread_rank] = primitive_conic_opacity[work_idx];
                        s_color[thread_rank] = fminf(fmaxf(make_float3(primitive_color[work_idx]), 0.0f), config::max_blend_color);
                    }
                }
                block.sync();

                for (int j_base = 0; j_base < n_batch; j_base += 32) {
                    const int j_test = j_base + static_cast<int>(lane_id);
                    bool hit0 = false, hit1 = false;
                    if (j_test < n_batch && s_valid[j_test]) {
                        const ushort4 bb = s_bbox[j_test];
                        const float4 co = s_conic_opacity[j_test];
                        const float3 conic = make_float3(co);
                        if (bb.x < sub0_ix1 && bb.y > sub0_ix0 && bb.z < sub0_iy1 && bb.w > sub0_iy0)
                            hit0 = kernels::splat_overlaps_subtile_ellipse(s_mean2d[j_test], conic, co.w,
                                                                           static_cast<float>(sub0_ix0), static_cast<float>(sub0_iy0), sub_w, sub_h);
                        if (bb.x < sub1_ix1 && bb.y > sub1_ix0 && bb.z < sub1_iy1 && bb.w > sub1_iy0)
                            hit1 = kernels::splat_overlaps_subtile_ellipse(s_mean2d[j_test], conic, co.w,
                                                                           static_cast<float>(sub1_ix0), static_cast<float>(sub1_iy0), sub_w, sub_h);
                    }
                    const unsigned mask0 = __ballot_sync(0xffffffffu, hit0);
                    const unsigned mask1 = __ballot_sync(0xffffffffu, hit1);
                    unsigned live = mask0 | mask1;
                    while (live != 0u) {
                        const int k = __ffs(live) - 1;
                        live &= live - 1u;
                        const int j = j_base + k;
                        const int tile_primitive_idx = count - 1 - (batch_base + j);
                        const float2 mean2d = s_mean2d[j];
                        const float4 co = s_conic_opacity[j];
                        const float3 color = s_color[j];
                        bool contributed = false;
                        const auto score_pixel = [&](const bool use, const bool inside, const uint last,
                                                     const float2 pixel, float& T, float3& behind) -> float {
                            if (!use || !inside || static_cast<uint>(tile_primitive_idx) >= last)
                                return 0.0f;
                            const float2 d = mean2d - pixel;
                            const float sigma_over_2 = 0.5f * (co.x * d.x * d.x + co.z * d.y * d.y) + co.y * d.x * d.y;
                            if (!(sigma_over_2 >= 0.0f))
                                return 0.0f;
                            const float alpha = fminf(co.w * __expf(-sigma_over_2), config::max_fragment_alpha);
                            if (!(alpha >= config::min_alpha_threshold))
                                return 0.0f;
                            contributed = true;
                            T /= 1.0f - alpha;
                            const float3 delta = (T * alpha) * (color - behind);
                            behind = alpha * color + (1.0f - alpha) * behind;
                            return dot(delta, delta);
                        };
                        float score = score_pixel(((mask0 >> k) & 1u) != 0u, inside0, last0, pixel0, T0, behind0);
                        score += score_pixel(((mask1 >> k) & 1u) != 0u, inside1, last1, pixel1, T1, behind1);
                        const unsigned contributors = __ballot_sync(0xffffffffu, contributed);
                        if (contributors == 0u)
                            continue;
                        score = __popc(contributors) == 1u
                                    ? __shfl_sync(0xffffffffu, score, __ffs(contributors) - 1)
                                    : lfs::core::warp_ops::warp_reduce_sum(score);
                        if (lane_id == 0u)
                            atomicAdd(scores + s_prim_idx[j], static_cast<double>(score));
                    }
                }
                block.sync();
            }
        }
    } // namespace

    void accumulate_pop_scores(const ForwardContext& ctx, int width, int height,
                               const float* background, const float* background_image,
                               double* scores, size_t n_primitives, cudaStream_t stream) {
        stream = lfs::resolve_stream(stream);
        if (ctx.n_instances == 0)
            return;
        char* primitive_blob = static_cast<char*>(ctx.per_primitive_buffers);
        char* tile_blob = static_cast<char*>(ctx.per_tile_buffers);
        const auto primitive = PerPrimitiveBuffers::from_persistent_blob(primitive_blob, ctx.n_visible);
        const uint grid_width = (width + config::tile_width - 1) / config::tile_width;
        const uint grid_height = (height + config::tile_height - 1) / config::tile_height;
        const auto tile = PerTileBuffers::from_blob(tile_blob, static_cast<int>(grid_width * grid_height));
        pop_scores_cu<<<grid_width * grid_height, kPopBlock, 0, stream>>>(
            tile.instance_ranges, static_cast<const uint*>(ctx.sorted_primitive_indices), ctx.primitive_work_indices,
            primitive.mean2d, primitive.conic_opacity, primitive.color, tile.n_contributions, tile.final_transmittance,
            background, background_image, scores, static_cast<uint>(ctx.n_instances), static_cast<uint>(n_primitives),
            static_cast<uint>(ctx.n_visible), static_cast<uint>(width), static_cast<uint>(height), grid_width);
        LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.pop_scores");
    }
} // namespace fast_lfs::rasterization
