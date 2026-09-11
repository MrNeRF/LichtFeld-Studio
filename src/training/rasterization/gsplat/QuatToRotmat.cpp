/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "QuatToRotmat.h"
#include "Common.h"
#include "Ops.h"

#include <cuda_runtime.h>

namespace gsplat_lfs {

    void quats_to_rotmats(
        const float* quats,
        int64_t N,
        float* rotmats,
        cudaStream_t stream) {
        gsplat_lfs::debug_validate_cuda_pointer(quats, "quats");
        gsplat_lfs::debug_validate_cuda_pointer(rotmats, "rotmats");

        if (N == 0) {
            return;
        }

        launch_quats_to_rotmats_kernel(quats, N, rotmats, stream);
    }

} // namespace gsplat_lfs
