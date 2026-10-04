/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../facade_trace.hpp"
#include "../gpu_backend_ops.hpp"

#include "../../internal/tensor_impl.hpp"
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

    void CudaBackendOps::point_neighbor_spacing(const StorageRef points, const StorageRef references,
                                                const StorageRef heads, const StorageRef next, const StorageRef output,
                                                const size_t count, const size_t buckets, const float cell_width,
                                                const ExecContext context) {
        LFS_FACADE_TRACE(point_neighbor_spacing);
        tensor_ops::launch_point_neighbor_spacing(cuda_pointer<const float>(points), cuda_pointer<const uint8_t>(references),
                                                  cuda_pointer<int32_t>(heads), cuda_pointer<int32_t>(next), cuda_pointer<float>(output),
                                                  count, buckets, cell_width, context.cuda_stream);
    }

    void CudaBackendOps::radius_neighbors(const StorageRef points, const StorageRef references,
                                          const StorageRef heads, const StorageRef next, const StorageRef output,
                                          const size_t count, const size_t buckets, const float radius,
                                          const bool exclude_self, const std::optional<StorageRef> queries,
                                          const ExecContext context) {
        LFS_FACADE_TRACE(radius_neighbors);
        tensor_ops::launch_radius_neighbors(cuda_pointer<const float>(points), cuda_pointer<const uint8_t>(references),
                                            cuda_pointer<int32_t>(heads), cuda_pointer<int32_t>(next), cuda_pointer<bool>(output),
                                            count, buckets, radius, exclude_self,
                                            queries ? cuda_pointer<const uint8_t>(*queries) : nullptr, context.cuda_stream);
    }

    void CudaBackendOps::radius_neighbor_counts(const StorageRef points, const StorageRef references,
                                                const StorageRef heads, const StorageRef next, const StorageRef output,
                                                const size_t count, const size_t buckets, const float radius,
                                                const int32_t max_count, const std::optional<StorageRef> queries,
                                                const ExecContext context) {
        LFS_FACADE_TRACE(radius_neighbor_counts);
        tensor_ops::launch_radius_neighbor_counts(cuda_pointer<const float>(points), cuda_pointer<const uint8_t>(references),
                                                  cuda_pointer<int32_t>(heads), cuda_pointer<int32_t>(next), cuda_pointer<int32_t>(output),
                                                  count, buckets, radius, max_count,
                                                  queries ? cuda_pointer<const uint8_t>(*queries) : nullptr, context.cuda_stream);
    }

    void CudaBackendOps::radius_neighbor_min(const StorageRef points, const StorageRef values,
                                             const StorageRef references, const StorageRef heads,
                                             const StorageRef next, const StorageRef output,
                                             const size_t count, const size_t buckets, const float radius,
                                             const std::optional<StorageRef> radii, const ExecContext context) {
        LFS_FACADE_TRACE(radius_neighbor_min);
        tensor_ops::launch_radius_neighbor_min(
            cuda_pointer<const float>(points), cuda_pointer<const void>(values),
            values.dtype == DataType::Float32, cuda_pointer<const uint8_t>(references),
            cuda_pointer<int32_t>(heads), cuda_pointer<int32_t>(next), cuda_pointer<void>(output),
            count, buckets, radius, radii ? cuda_pointer<const float>(*radii) : nullptr, context.cuda_stream);
    }

    bool CudaBackendOps::radius_connected_components(const StorageRef points, const StorageRef references,
                                                     const StorageRef heads, const StorageRef next,
                                                     const StorageRef labels, const size_t count,
                                                     const size_t buckets, const float radius,
                                                     const ExecContext context) {
        LFS_FACADE_TRACE(radius_connected_components);
        tensor_ops::launch_radius_connected_components(
            cuda_pointer<const float>(points), cuda_pointer<const uint8_t>(references), cuda_pointer<int32_t>(heads),
            cuda_pointer<int32_t>(next), cuda_pointer<int32_t>(labels), count, buckets, radius, context.cuda_stream);
        return true;
    }

    bool CudaBackendOps::point_tree_components(const StorageRef points, const StorageRef sorted, const StorageRef boxes,
                                               const StorageRef box_radii, const StorageRef visit,
                                               const StorageRef sorted_radii, const StorageRef radii,
                                               const StorageRef labels, const PointTreeProgram& program,
                                               const ExecContext context) {
        LFS_FACADE_TRACE(point_tree_components);
        tensor_ops::launch_point_tree_components(
            cuda_pointer<const float>(points), cuda_pointer<const float>(sorted), cuda_pointer<const float>(boxes),
            cuda_pointer<const float>(box_radii), cuda_pointer<const int32_t>(visit),
            cuda_pointer<const float>(sorted_radii), cuda_pointer<const float>(radii), cuda_pointer<int32_t>(labels),
            program, context.cuda_stream);
        return true;
    }

    bool CudaBackendOps::point_tree_counts(const StorageRef points, const StorageRef sorted, const StorageRef boxes,
                                           const StorageRef visit, const StorageRef radii,
                                           const std::optional<StorageRef> queries, const StorageRef output,
                                           const PointTreeProgram& program, const ExecContext context) {
        LFS_FACADE_TRACE(point_tree_counts);
        tensor_ops::launch_point_tree_counts(
            cuda_pointer<const float>(points), cuda_pointer<const float>(sorted), cuda_pointer<const float>(boxes),
            cuda_pointer<const int32_t>(visit), cuda_pointer<const float>(radii),
            queries ? cuda_pointer<const uint8_t>(*queries) : nullptr, cuda_pointer<int32_t>(output), program,
            context.cuda_stream);
        return true;
    }

    bool CudaBackendOps::point_tree_spacing(const StorageRef points, const StorageRef sorted, const StorageRef boxes,
                                            const StorageRef visit, const StorageRef output,
                                            const PointTreeProgram& program, const ExecContext context) {
        LFS_FACADE_TRACE(point_tree_spacing);
        tensor_ops::launch_point_tree_spacing(cuda_pointer<const float>(points), cuda_pointer<const float>(sorted),
                                              cuda_pointer<const float>(boxes), cuda_pointer<const int32_t>(visit),
                                              cuda_pointer<float>(output), program, context.cuda_stream);
        return true;
    }

    bool CudaBackendOps::triangle_tree_parity(const StorageRef points, const StorageRef visit, const StorageRef triangles,
                                              const StorageRef boxes, const StorageRef output,
                                              const PointTreeProgram& program, const ExecContext context) {
        LFS_FACADE_TRACE(triangle_tree_parity);
        tensor_ops::launch_triangle_tree_parity(cuda_pointer<const float>(points), cuda_pointer<const int32_t>(visit),
                                                cuda_pointer<const float>(triangles), cuda_pointer<const float>(boxes),
                                                cuda_pointer<int32_t>(output), program, context.cuda_stream);
        return true;
    }

    void CudaBackendOps::nearest_point_indices(StorageRef q, StorageRef t, StorageRef h, StorageRef n, StorageRef o,
                                               size_t nq, size_t nt, size_t buckets, float width, ExecContext context) {
        LFS_FACADE_TRACE(nearest_point_indices);
        tensor_ops::launch_nearest_point_indices(cuda_pointer<const float>(q), cuda_pointer<const float>(t),
                                                 cuda_pointer<int32_t>(h), cuda_pointer<int32_t>(n), cuda_pointer<int32_t>(o), nq, nt, buckets, width, context.cuda_stream);
    }
    void CudaBackendOps::camera_frustum_counts(StorageRef p, StorageRef c, StorageRef o, size_t n, size_t cameras,
                                               float maximum, ExecContext context) {
        LFS_FACADE_TRACE(camera_frustum_counts);
        tensor_ops::launch_camera_frustum_counts(cuda_pointer<const float>(p), cuda_pointer<const float>(c),
                                                 cuda_pointer<int32_t>(o), n, cameras, maximum, context.cuda_stream);
    }

    // CUDA builds rasterize point clouds with the renderer's own kernel.
    void CudaBackendOps::rasterize_points(const PointRasterProgram&, ExecContext) {
        throw TensorError("CUDA rasterizes point clouds in the renderer");
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
