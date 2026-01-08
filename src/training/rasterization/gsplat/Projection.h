/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "Common.h"
#include <cstdint>
#include <cuda_runtime.h>

namespace gsplat_lfs {

    void launch_projection_ut_3dgs_fused_kernel(
        // inputs
        const float* means,     // [N, 3]
        const float* quats,     // [N, 4]
        const float* scales,    // [N, 3]
        const float* opacities, // [N] optional (can be nullptr)
        const float* viewmats0, // [C, 4, 4]
        const float* viewmats1, // [C, 4, 4] optional
        const float* Ks,        // [C, 3, 3]
        uint32_t N,
        uint32_t C,
        uint32_t image_width,
        uint32_t image_height,
        float eps2d,
        float near_plane,
        float far_plane,
        float radius_clip,
        CameraModelType camera_model,
        const UnscentedTransformParameters& ut_params,
        ShutterType rs_type,
        const float* radial_coeffs,     // optional
        const float* tangential_coeffs, // optional
        const float* thin_prism_coeffs, // optional
        // outputs
        int32_t* radii,       // [C, N, 2]
        float* means2d,       // [C, N, 2]
        float* depths,        // [C, N]
        float* conics,        // [C, N, 3]
        float* compensations, // [C, N] optional (can be nullptr)
        cudaStream_t stream = nullptr);

} // namespace gsplat_lfs
