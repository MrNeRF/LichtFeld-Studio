/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::core::tensor_ops {
    void launch_radius_neighbors(const float* points, const uint8_t* references,
                                 int32_t* heads, int32_t* next, bool* output,
                                 size_t count, size_t buckets, float radius, cudaStream_t stream);
}
