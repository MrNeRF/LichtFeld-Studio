/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda/splat_transform.hpp"
#include "core/cuda_error.hpp"

namespace lfs::core::cuda {
    namespace {
#include "core/device/affine_splat.cuh"
    } // namespace
    void transform_splat_geometry(const splat_transform::LinearTransform& matrix,
                                  const float* scales, const float* rotations,
                                  float* out_scales, float* out_rotations,
                                  std::size_t count, cudaStream_t stream, const float* matrices) {
        if (!count)
            return;
        transform_geometry<<<(count + 255) / 256, 256, 0, stream>>>(
            matrix, scales, rotations, out_scales, out_rotations, count, matrices);
        LFS_CUDA_CHECK(cudaGetLastError());
    }
} // namespace lfs::core::cuda
