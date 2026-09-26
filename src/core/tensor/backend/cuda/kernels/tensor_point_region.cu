/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda_error.hpp"
#include "internal/point_region.hpp"
#include "tensor_point_region.hpp"

namespace lfs::core::tensor_ops {
    namespace {
        constexpr int kBlockSize = 256;

        template <PointRegion2DKind Kind>
        __global__ void markPoints2DKernel(uint8_t* mask, const float2* points, const int count,
                                           const PointRegion2D region, const float2* geometry,
                                           const int geometry_count) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= count)
                return;
            const float2 pos = points[i];
            const bool inside = internal::pointInRegion<Kind>(pos.x, pos.y, region, reinterpret_cast<const float*>(geometry), geometry_count);
            if (inside)
                mask[i] = 1;
        }
    } // namespace

    void launch_mark_points_2d(uint8_t* mask, const float* points, const size_t count,
                               const PointRegion2D& region, const float* geometry,
                               const size_t geometry_count, const cudaStream_t stream) {
        const auto launch = [&]<PointRegion2DKind Kind>() {
            markPoints2DKernel<Kind><<<static_cast<unsigned>((count + kBlockSize - 1) / kBlockSize), kBlockSize, 0, stream>>>(
                mask, reinterpret_cast<const float2*>(points), static_cast<int>(count), region,
                reinterpret_cast<const float2*>(geometry), static_cast<int>(geometry_count));
            LFS_CUDA_LAUNCH_CHECK(stream, "tensor.mark_points_2d");
        };
        switch (region.kind) {
        case PointRegion2DKind::Disk: launch.template operator()<PointRegion2DKind::Disk>(); break;
        case PointRegion2DKind::Rectangle: launch.template operator()<PointRegion2DKind::Rectangle>(); break;
        case PointRegion2DKind::Polygon: launch.template operator()<PointRegion2DKind::Polygon>(); break;
        case PointRegion2DKind::Disks: launch.template operator()<PointRegion2DKind::Disks>(); break;
        }
    }
} // namespace lfs::core::tensor_ops
