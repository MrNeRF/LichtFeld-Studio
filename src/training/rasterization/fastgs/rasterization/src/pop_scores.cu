/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../../../kernels/kernel_stream.hpp"
#include "buffer_utils.h"
#include "pop_scores.h"

namespace fast_lfs::rasterization {
    __global__ void pop_scores_kernel(PerPrimitiveBuffers primitive, PerTileBuffers tile,
                                      const uint* ids, const uint* work_ids, int width, int height,
                                      const float* background, const float* background_image, double* scores) {
        const int pix = blockIdx.x * blockDim.x + threadIdx.x;
        const int pixels = width * height;
        if (pix >= pixels)
            return;
        const int x = pix % width, y = pix / width;
        const int tile_id = (y / config::tile_height) * ((width + config::tile_width - 1) / config::tile_width) + x / config::tile_width;
        const int rank = (y % config::tile_height) * config::tile_width + x % config::tile_width;
        double T = tile.final_transmittance[tile_id * config::block_size_blend + rank];
        const uint begin = tile.instance_ranges[tile_id].x;
        const uint count = tile.n_contributions[pix];
        double behind[3];
        for (int c = 0; c < 3; ++c)
            behind[c] = background_image ? background_image[c * pixels + pix] : (background ? background[c] : 0.f);
        for (int64_t i = int64_t(begin) + count - 1; i >= begin; --i) {
            const uint g = ids[i], work = work_ids[g];
            const auto geom = primitive.mean2d[work];
            const auto bb = geom.pixel_bbox;
            const int sx = x / 8 * 8, sy = y / 4 * 4;
            if (!(bb.x < sx + 8 && bb.y > sx && bb.z < sy + 4 && bb.w > sy))
                continue;
            const float4 q = primitive.conic_opacity[work];
            const float dx = geom.mean2d.x - (float(x) + 0.5f), dy = geom.mean2d.y - (float(y) + 0.5f);
            const float sigma = ((q.x * 0.5f) * dx * dx + (q.z * 0.5f) * dy * dy) + q.y * dx * dy;
            if (sigma < 0.f || sigma > logf(q.w * config::min_alpha_threshold_rcp) + 1.e-5f)
                continue;
            const float a = fminf(q.w * __expf(-sigma), config::max_fragment_alpha);
            if (a < config::min_alpha_threshold)
                continue;
            T /= (1.f - a);
            const float4 raw = primitive.color[work];
            const float colors[3] = {raw.x, raw.y, raw.z};
            double score = 0;
            for (int c = 0; c < 3; ++c) {
                const double color = fminf(fmaxf(colors[c], 0.f), config::max_blend_color);
                const double delta = T * a * (color - behind[c]);
                score += delta * delta;
                behind[c] = a * color + (1.f - a) * behind[c];
            }
            atomicAdd(scores + g, score);
        }
    }
    void accumulate_pop_scores(const ForwardContext& ctx, int width, int height,
                               const float* background, const float* background_image,
                               double* scores, cudaStream_t stream) {
        stream = lfs::resolve_stream(stream);
        if (ctx.n_instances == 0)
            return;
        char* primitive_blob = static_cast<char*>(ctx.per_primitive_buffers);
        char* tile_blob = static_cast<char*>(ctx.per_tile_buffers);
        const auto primitive = PerPrimitiveBuffers::from_persistent_blob(primitive_blob, ctx.n_visible);
        const int tiles = ((width + 15) / 16) * ((height + 15) / 16);
        const auto tile = PerTileBuffers::from_blob(tile_blob, tiles);
        pop_scores_kernel<<<(width * height + 255) / 256, 256, 0, stream>>>(primitive, tile,
                                                                            static_cast<const uint*>(ctx.sorted_primitive_indices), ctx.primitive_work_indices,
                                                                            width, height, background, background_image, scores);
        LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.pop_scores");
    }
} // namespace fast_lfs::rasterization
