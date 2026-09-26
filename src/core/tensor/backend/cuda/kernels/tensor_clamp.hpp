/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"
#include <cstddef>
#include <cuda_runtime.h>

namespace lfs::core::tensor_ops {
    void launch_clamp_fused(const int* src, int* dst, int min_val, int max_val, size_t n, cudaStream_t stream);
}
