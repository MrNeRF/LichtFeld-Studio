/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

namespace lfs::vis::vksplat::detail {

    [[nodiscard]] cudaError_t launchPackActivatedRotations(
        const float* rotation_raw,
        float* rotations_dst,
        std::size_t num_splats,
        cudaStream_t stream);

    [[nodiscard]] cudaError_t launchPackScalesOpacs(
        const float* scaling_raw,
        const float* opacity_raw,
        float* scales_opacs_dst,
        std::size_t num_splats,
        cudaStream_t stream);

} // namespace lfs::vis::vksplat::detail
