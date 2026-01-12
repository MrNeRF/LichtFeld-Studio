/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "Projection.h"
#include "Common.h"
#include "Ops.h"

#include <cuda_runtime.h>

namespace gsplat_fwd {

    void projection_ut_3dgs_fused(
        const float* means,
        const float* quats,
        const float* scales,
        const float* opacities,
        const float* viewmats0,
        const float* viewmats1,
        const float* Ks,
        uint32_t N_total,
        uint32_t M,
        uint32_t C,
        uint32_t image_width,
        uint32_t image_height,
        float eps2d,
        float near_plane,
        float far_plane,
        float radius_clip,
        bool calc_compensations,
        CameraModelType camera_model,
        const UnscentedTransformParameters& ut_params,
        ShutterType rs_type,
        const float* radial_coeffs,
        const float* tangential_coeffs,
        const float* thin_prism_coeffs,
        const int* transform_indices,
        const bool* node_visibility_mask,
        int num_visibility_nodes,
        const int* visible_indices,
        int32_t* radii,
        float* means2d,
        float* depths,
        float* conics,
        float* compensations,
        cudaStream_t stream) {
        GSPLAT_CHECK_CUDA_PTR(means, "means");
        GSPLAT_CHECK_CUDA_PTR(quats, "quats");
        GSPLAT_CHECK_CUDA_PTR(scales, "scales");
        GSPLAT_CHECK_CUDA_PTR(viewmats0, "viewmats0");
        GSPLAT_CHECK_CUDA_PTR(Ks, "Ks");
        GSPLAT_CHECK_CUDA_PTR(radii, "radii");
        GSPLAT_CHECK_CUDA_PTR(means2d, "means2d");
        GSPLAT_CHECK_CUDA_PTR(depths, "depths");
        GSPLAT_CHECK_CUDA_PTR(conics, "conics");

        if (M == 0 || C == 0) {
            return;
        }

        launch_projection_ut_3dgs_fused_kernel(
            means, quats, scales, opacities,
            viewmats0, viewmats1, Ks,
            N_total, M, C, image_width, image_height,
            eps2d, near_plane, far_plane, radius_clip,
            camera_model, ut_params, rs_type,
            radial_coeffs, tangential_coeffs, thin_prism_coeffs,
            transform_indices, node_visibility_mask, num_visibility_nodes,
            visible_indices,
            radii, means2d, depths, conics, compensations,
            stream);
    }

} // namespace gsplat_fwd
