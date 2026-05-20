/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "SphericalHarmonics.h"
#include "Common.h"
#include "Ops.h"

#include <cuda_runtime.h>

namespace gsplat_fwd {

    void spherical_harmonics_swizzled_fwd(
        uint32_t degrees_to_use,
        const float* dirs,
        const float* sh0,
        const float* sh_rest_swizzled,
        const bool* masks,
        const int32_t* visible_indices,
        int64_t total_elements,
        float* colors,
        cudaStream_t stream) {
        if (total_elements == 0) {
            return;
        }

        launch_spherical_harmonics_swizzled_fwd_kernel(
            degrees_to_use,
            dirs, sh0, sh_rest_swizzled, masks, visible_indices,
            total_elements,
            colors, stream);
    }

} // namespace gsplat_fwd
