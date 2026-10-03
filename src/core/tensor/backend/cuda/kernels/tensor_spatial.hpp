/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::core::tensor_ops {
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
} // namespace lfs::core::tensor_ops
