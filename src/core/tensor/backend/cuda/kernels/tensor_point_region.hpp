/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"

#include "core/tensor_spatial.hpp"
#include <cstddef>
#include <cuda_runtime.h>

namespace lfs::core::tensor_ops {
    void launch_mark_points_2d(uint8_t* mask, const float* points, size_t count,
                               const PointRegion2D& region, const float* geometry,
                               size_t geometry_count, cudaStream_t stream);
}
