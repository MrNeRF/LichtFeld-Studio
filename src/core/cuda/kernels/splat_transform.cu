/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda/splat_transform.hpp"
#include "core/cuda_error.hpp"

namespace lfs::core::cuda {
    namespace {
        __global__ void transform_geometry(splat_transform::LinearTransform matrix,
                                           const float* scales, const float* rotations,
                                           float* out_scales, float* out_rotations,
                                           std::size_t count, const float* matrices) {
            const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < count) {
                if (matrices)
                    for (int k = 0; k < 9; ++k)
                        matrix.rows[k] = matrices[9 * i + k];
                splat_transform::affine_geometry(matrix, scales + 3 * i, rotations + 4 * i,
                                                 out_scales + 3 * i, out_rotations + 4 * i);
            }
        }
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
