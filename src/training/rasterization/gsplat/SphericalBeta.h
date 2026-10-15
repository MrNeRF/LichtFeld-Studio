/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// Spherical Beta view-dependent color (Beta Splatting, Liu et al.). Each splat carries
// `num_lobes` raw lobes [r, g, b, theta, phi, beta] that add a specular term to the
// SH base color already written to `colors`:
//
//   C = C_base + sum_l softplus(rgb_l) * max(dot(v, d_l), 0)^(4 * exp(beta_l))
//
// with d_l = (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)), v the normalized
// view direction (mean - camera) and softplus using beta = 10 * ln(2), matching the
// reference implementation so exported PLYs render identically in other viewers.

namespace gsplat_lfs {

    // Adds the lobe contribution in place to the first three channels of `colors`.
    void launch_spherical_beta_fwd_kernel(
        uint32_t num_lobes,
        uint32_t color_stride,
        uint32_t N,             // splats; sb_params rows
        int64_t total_elements, // C * N
        const float* dirs,      // [C * N, 3] unnormalized view directions
        const float* sb_params, // [N, num_lobes * 6]
        const bool* masks,      // [C * N] optional
        float* colors,          // [C * N, color_stride] in/out
        cudaStream_t stream = nullptr);

    // Accumulates dL/d(sb_params) into the zero-initialized `v_sb_params`. The gradient
    // with respect to the base color is the identity, so `v_colors` is left untouched
    // for the SH backward pass.
    void launch_spherical_beta_bwd_kernel(
        uint32_t num_lobes,
        uint32_t color_stride,
        uint32_t N,
        int64_t total_elements,
        const float* dirs,      // [C * N, 3]
        const float* sb_params, // [N, num_lobes * 6]
        const bool* masks,      // [C * N] optional
        const float* v_colors,  // [C * N, color_stride]
        float* v_sb_params,     // [N, num_lobes * 6] accumulated
        cudaStream_t stream = nullptr);

} // namespace gsplat_lfs
