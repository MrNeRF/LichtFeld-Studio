/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"
#include <cstddef>
#include <cuda_runtime_api.h>

namespace lfs::core::internal {
    void launch_where_scalar(void* output, const void* condition, float value, const void* source,
                             size_t count, bool half, cudaStream_t stream);
}
