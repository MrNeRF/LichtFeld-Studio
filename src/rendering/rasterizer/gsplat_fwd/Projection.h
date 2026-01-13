/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "Common.h"
#include <cstdint>
#include <cuda_runtime.h>

namespace gsplat_fwd {

    void launch_projection_ut_3dgs_fused_kernel(
        // inputs
        const float* means,     // [N_total, 3]
        const float* quats,     // [N_total, 4]
        const float* scales,    // [N_total, 3]
        const float* opacities, // [N_total] optional (can be nullptr)
        const float* viewmats0, // [C, 4, 4]
        const float* viewmats1, // [C, 4, 4] optional
        const float* Ks,        // [C, 3, 3]
        uint32_t N_total,       // Total gaussians in input arrays
        uint32_t M,             // Visible gaussians to process (M <= N_total)
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
        const float* radial_coeffs,       // optional
        const float* tangential_coeffs,   // optional
        const float* thin_prism_coeffs,   // optional
        const int* transform_indices,     // [N_total] optional
        const bool* node_visibility_mask, // optional
        int num_visibility_nodes,
        const int* visible_indices, // [M] maps output idx → global gaussian idx (nullptr = all)
        // outputs (sized to [C, M, ...])
        int32_t* radii,       // [C, M, 2]
        float* means2d,       // [C, M, 2]
        float* depths,        // [C, M]
        float* conics,        // [C, M, 3]
        float* compensations, // [C, M] optional (can be nullptr)
        cudaStream_t stream = nullptr);

} // namespace gsplat_fwd
