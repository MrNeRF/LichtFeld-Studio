/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda_error.hpp"
#include "internal/point_projection.hpp"
#include "tensor_projection.hpp"

namespace lfs::core::tensor_ops {
    namespace {
        constexpr int kBlockSize = 256;
        __global__ void projectPointsKernel(const float* points, float* output, size_t count,
                                            PointProjection projection, const float* transforms,
                                            size_t transform_count, const int32_t* indices,
                                            const uint8_t* visibility, size_t visibility_count) {
            const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < count)
                internal::projectPoint(points, output, i, projection, transforms, transform_count, indices, visibility, visibility_count);
        }
    } // namespace
    void launch_project_points(const float* points, float* output, size_t count,
                               const PointProjection& projection, const float* transforms,
                               size_t transform_count, const int32_t* indices,
                               const uint8_t* visibility, size_t visibility_count, cudaStream_t stream) {
        projectPointsKernel<<<unsigned((count + kBlockSize - 1) / kBlockSize), kBlockSize, 0, stream>>>(
            points, output, count, projection, transforms, transform_count, indices, visibility, visibility_count);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.project_points");
    }
} // namespace lfs::core::tensor_ops
