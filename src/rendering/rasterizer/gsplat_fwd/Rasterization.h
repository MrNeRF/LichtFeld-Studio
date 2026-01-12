/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Forward-only GUT rasterization for viewer (no backward pass)

#pragma once

#include "Common.h"
#include <cstdint>
#include <cuda_runtime.h>

namespace gsplat_fwd {

    /////////////////////////////////////////////////
    // rasterize_to_pixels_from_world_3dgs - Forward Only
    /////////////////////////////////////////////////

    template <uint32_t CDIM>
    void launch_rasterize_to_pixels_from_world_3dgs_fwd_kernel(
        const float* means,
        const float* quats,
        const float* scales,
        const float* colors,
        const float* opacities,
        const float* backgrounds,
        const bool* masks,
        const float* depths,
        uint32_t C,
        uint32_t N,
        uint32_t n_isects,
        uint32_t image_width,
        uint32_t image_height,
        uint32_t tile_size,
        const float* viewmats0,
        const float* viewmats1,
        const float* Ks,
        CameraModelType camera_model,
        const UnscentedTransformParameters& ut_params,
        ShutterType rs_type,
        const float* radial_coeffs,
        const float* tangential_coeffs,
        const float* thin_prism_coeffs,
        const int32_t* tile_offsets,
        const int32_t* flatten_ids,
        const int32_t* visible_indices, // [M] maps g -> global gaussian idx, nullptr = direct
        float* renders,
        float* alphas,
        int32_t* last_ids,
        float* median_depths,
        cudaStream_t stream = nullptr);

} // namespace gsplat_fwd
