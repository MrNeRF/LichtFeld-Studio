/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../facade_trace.hpp"
#include "../gpu_backend_ops.hpp"
#include "core/assert.hpp"
#include "kernels/tensor_point_region.hpp"
#include "kernels/tensor_projection.hpp"
#include "kernels/tensor_spatial.hpp"

namespace lfs::core::internal {
    namespace {
        template <class T>
        T* cuda_pointer(const StorageRef storage) {
            LFS_ASSERT_MSG(storage.backend == GpuBackend::CUDA, "CUDA spatial adapter requires CUDA storage");
            return reinterpret_cast<T*>(static_cast<unsigned char*>(storage.data) + storage.byte_offset);
        }
    } // namespace

    void CudaBackendOps::radius_neighbors(const StorageRef points, const StorageRef references,
                                          const StorageRef heads, const StorageRef next, const StorageRef output,
                                          const size_t count, const size_t buckets, const float radius,
                                          const ExecContext context) {
        LFS_FACADE_TRACE(radius_neighbors);
        tensor_ops::launch_radius_neighbors(cuda_pointer<const float>(points), cuda_pointer<const uint8_t>(references),
                                            cuda_pointer<int32_t>(heads), cuda_pointer<int32_t>(next), cuda_pointer<bool>(output),
                                            count, buckets, radius, context.cuda_stream);
    }

    void CudaBackendOps::project_points(const StorageRef points, const StorageRef output, const size_t count,
                                        const PointProjection& projection,
                                        const StorageRef* transforms, const size_t transform_count,
                                        const StorageRef* indices, const StorageRef* visibility,
                                        const size_t visibility_count, const ExecContext context) {
        LFS_FACADE_TRACE(project_points);
        tensor_ops::launch_project_points(
            cuda_pointer<const float>(points), cuda_pointer<float>(output), count, projection,
            transforms ? cuda_pointer<const float>(*transforms) : nullptr, transform_count,
            indices ? cuda_pointer<const int32_t>(*indices) : nullptr,
            visibility ? cuda_pointer<const uint8_t>(*visibility) : nullptr, visibility_count, context.cuda_stream);
    }
    void CudaBackendOps::mark_points_2d(const StorageRef mask, const StorageRef points, const size_t count,
                                        const PointRegion2D& region, const StorageRef* geometry,
                                        const size_t geometry_count, const ExecContext context) {
        LFS_FACADE_TRACE(mark_points_2d);
        tensor_ops::launch_mark_points_2d(cuda_pointer<uint8_t>(mask), cuda_pointer<const float>(points), count,
                                          region, geometry ? cuda_pointer<const float>(*geometry) : nullptr,
                                          geometry_count, context.cuda_stream);
    }
} // namespace lfs::core::internal
