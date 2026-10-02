/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/geometry_types.hpp"

#include <cuda_runtime.h>

namespace lfs::training::kernels {

    // Depth-normal consistency: alpha-weighted cosine between the rendered
    // normal map and the normal derived from the rendered expected depth
    // (E = accum/alpha, unprojected through the intrinsics, central-difference
    // cross product, oriented toward the camera with a detached sign):
    //   L = weight * sum(alpha * (1 - cos(n_render, n_depth))) / max(sum(alpha), 1)
    // Gradients flow BOTH ways: into the rendered normal map (added to
    // grad_normal) and through n_depth into the accumulated-depth and alpha
    // maps (atomically added to grad_depth_accum / grad_alpha), which is what
    // couples splat orientation to actual surface position.
    // All grad buffers must be pre-initialized (zeroed or holding other loss
    // gradients); this kernel accumulates into them.
    void launch_normal_consistency_loss(
        const float* rendered_normal,      // [3*H*W] CHW accumulated normals
        const float* rendered_depth_accum, // [H*W] accumulated depth
        const float* rendered_alpha,       // [H*W]
        float* grad_normal,                // [3*H*W] CHW, accumulated into
        float* grad_depth_accum,           // [H*W], accumulated into
        float* grad_alpha,                 // [H*W], accumulated into
        float* loss_out,                   // [1], written
        float* partial_sums,               // [normal_consistency_partial_count(num_pixels)]
        int width,
        int height,
        float fx,
        float fy,
        float cx,
        float cy,
        float weight,
        cudaStream_t stream = nullptr,
        const float* pixel_weight = nullptr);

    // Prior-supervised depth normals: alpha-weighted cosine between the prior
    // normal map and the normal derived from rendered expected depth. Gradients
    // flow only through n_depth into grad_depth_accum / grad_alpha.
    // All grad buffers must be pre-initialized; this kernel accumulates.
    void launch_normal_prior_depth_loss(
        const float* prior_normal,         // [3*H*W] CHW prior normals
        const float* rendered_depth_accum, // [H*W] accumulated depth
        const float* rendered_alpha,       // [H*W]
        float* grad_depth_accum,           // [H*W], accumulated into
        float* grad_alpha,                 // [H*W], accumulated into
        float* loss_out,                   // [1], written
        float* partial_sums,               // [normal_consistency_partial_count(num_pixels)]
        int width,
        int height,
        float fx,
        float fy,
        float cx,
        float cy,
        float weight,
        cudaStream_t stream = nullptr,
        const float* pixel_weight = nullptr);

} // namespace lfs::training::kernels
