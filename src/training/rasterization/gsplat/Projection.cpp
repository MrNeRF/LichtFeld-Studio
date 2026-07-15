/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "Projection.h"
#include "Common.h"
#include "Ops.h"

#include <cuda_runtime.h>

namespace gsplat_lfs {

    void projection_ut_3dgs_fused(
        const float* means,
        const float* quats,
        const float* scales,
        const float* opacities,
        const float* viewmats0,
        const float* viewmats1,
        const float* Ks,
        uint32_t N,
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
        int32_t* radii,
        float* means2d,
        float* depths,
        float* conics,
        float* compensations,
        cudaStream_t stream) {
        gsplat_lfs::debug_validate_cuda_pointer(means, "means");
        gsplat_lfs::debug_validate_cuda_pointer(quats, "quats");
        gsplat_lfs::debug_validate_cuda_pointer(scales, "scales");
        gsplat_lfs::debug_validate_cuda_pointer(viewmats0, "viewmats0");
        gsplat_lfs::debug_validate_cuda_pointer(Ks, "Ks");
        gsplat_lfs::debug_validate_cuda_pointer(radii, "radii");
        gsplat_lfs::debug_validate_cuda_pointer(means2d, "means2d");
        gsplat_lfs::debug_validate_cuda_pointer(depths, "depths");
        gsplat_lfs::debug_validate_cuda_pointer(conics, "conics");

        if (N == 0 || C == 0) {
            return;
        }

        launch_projection_ut_3dgs_fused_kernel(
            means, quats, scales, opacities,
            viewmats0, viewmats1, Ks,
            N, C, image_width, image_height,
            eps2d, near_plane, far_plane, radius_clip,
            camera_model, ut_params, rs_type,
            radial_coeffs, tangential_coeffs, thin_prism_coeffs,
            radii, means2d, depths, conics, compensations,
            stream);
    }

} // namespace gsplat_lfs
