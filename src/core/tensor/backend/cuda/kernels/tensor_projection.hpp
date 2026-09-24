/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"

#include "core/tensor_spatial.hpp"
#include <cstddef>
#include <cuda_runtime.h>

namespace lfs::core::tensor_ops {
    void launch_project_points(const float* points, float* output, size_t count,
                               const PointProjection& projection,
                               const float* transforms, size_t transform_count,
                               const int32_t* indices,
                               const uint8_t* visibility, size_t visibility_count,
                               cudaStream_t stream);
}
