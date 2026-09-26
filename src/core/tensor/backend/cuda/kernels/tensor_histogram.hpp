/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::core::tensor_ops {
    void launch_histogram_u8(const uint8_t* values, int32_t* counts, size_t size,
                             cudaStream_t stream);
} // namespace lfs::core::tensor_ops
