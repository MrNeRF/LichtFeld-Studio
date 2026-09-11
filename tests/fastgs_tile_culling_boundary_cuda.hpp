/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Device-side probes of FastGS tile-culling helpers. The helpers live in
// kernel_utils.cuh; this TU launches 1-thread kernels so tests can assert
// closed-interval behaviour at exact tile/pixel-center boundaries.

namespace fastgs_tile_culling_boundary_test {

    constexpr int kMaxTiles = 64;

    struct FloorCeilSample {
        float coord;
        std::uint32_t min_tile;
        std::uint32_t max_tile;
        std::uint32_t tile_size;
        std::uint32_t floor_tile;
        std::uint32_t ceil_tile;
    };

    struct OverlapSample {
        float inv_a;
        float inv_b;
        float inv_c;
        float x0;
        float x1;
        float y0;
        float y1;
        int overlaps;
    };

    struct SplatSubtileSample {
        float mean_x;
        float mean_y;
        float conic_x;
        float conic_y;
        float conic_z;
        float opacity;
        float sub_x0;
        float sub_y0;
        float sub_w;
        float sub_h;
        int overlaps;
        float power_threshold;
    };

    struct AgreementSample {
        float mean_x;
        float mean_y;
        float conic_x;
        float conic_y;
        float conic_z;
        float opacity;
        std::uint32_t width;
        std::uint32_t height;
        std::uint32_t count;
        std::uint32_t n_instances;
        std::uint32_t n_blend_tiles;
        std::uint32_t n_missing_blend_tiles;
        std::uint32_t n_bwd_drops;
        std::uint32_t n_blend_pixels;
        float power_threshold;
        std::uint32_t instance_tiles[kMaxTiles];
        std::uint32_t blend_tiles[kMaxTiles];
    };

    cudaError_t eval_floor_ceil(FloorCeilSample* samples, int n);
    cudaError_t eval_ellipse_overlap(OverlapSample* samples, int n);
    cudaError_t eval_splat_subtile(SplatSubtileSample* samples, int n);
    cudaError_t eval_agreement(AgreementSample* samples, int n);

} // namespace fastgs_tile_culling_boundary_test
