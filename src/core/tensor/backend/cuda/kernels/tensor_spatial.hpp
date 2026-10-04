/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"

#include "core/tensor/backend/descriptors.hpp"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::core::tensor_ops {
    void launch_nearest_point_indices(const float*, const float*, int32_t*, int32_t*, int32_t*, size_t, size_t, size_t, float, cudaStream_t);
    void launch_camera_frustum_counts(const float*, const float*, int32_t*, size_t, size_t, float, cudaStream_t);
    void launch_point_neighbor_spacing(const float* points, const uint8_t* references,
                                       int32_t* heads, int32_t* next, float* output,
                                       size_t count, size_t buckets, float cell_width, cudaStream_t stream);
    void launch_radius_neighbors(const float* points, const uint8_t* references,
                                 int32_t* heads, int32_t* next, bool* output,
                                 size_t count, size_t buckets, float radius, bool exclude_self,
                                 const uint8_t* queries, cudaStream_t stream);
    void launch_radius_neighbor_counts(const float* points, const uint8_t* references,
                                       int32_t* heads, int32_t* next, int32_t* output,
                                       size_t count, size_t buckets, float radius, int32_t max_count,
                                       const uint8_t* queries, cudaStream_t stream);
    void launch_radius_neighbor_min(const float* points, const void* values, uint8_t value_is_float,
                                    const uint8_t* references, int32_t* heads, int32_t* next, void* output,
                                    size_t count, size_t buckets, float radius, const float* radii, cudaStream_t stream);
    void launch_radius_connected_components(const float* points, const uint8_t* references, int32_t* heads,
                                            int32_t* next, int32_t* labels, size_t count, size_t buckets, float radius,
                                            cudaStream_t stream);
    void launch_point_tree_components(const float* points, const float* sorted, const float* boxes, const float* box_radii,
                                      const int32_t* visit, const float* sorted_radii, const float* radii, int32_t* labels,
                                      const internal::PointTreeProgram& tree, cudaStream_t stream);
    void launch_point_tree_counts(const float* points, const float* sorted, const float* boxes, const int32_t* visit,
                                  const float* radii, const uint8_t* queries, int32_t* output,
                                  const internal::PointTreeProgram& tree, cudaStream_t stream);
} // namespace lfs::core::tensor_ops
