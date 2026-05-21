/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace gsplat_fwd {

    void launch_spherical_harmonics_swizzled_fwd_kernel(
        uint32_t degrees_to_use,
        const float* dirs,              // [M, 3]
        const float* sh0,               // [N_total, 1, 3] or [N_total, 3]
        const float* sh_rest_swizzled,  // resident swizzled shN rest layout
        const bool* masks,              // [M] optional
        const int32_t* visible_indices, // [M] maps elem_id -> global_idx, nullptr = direct
        int64_t n_elements,             // M (visible gaussians)
        float* colors,                  // [M, 3]
        cudaStream_t stream = nullptr);

} // namespace gsplat_fwd
