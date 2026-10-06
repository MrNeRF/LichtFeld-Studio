/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include "core/tensor_fwd.hpp"
#include <cstdint>

namespace lfs::core {
    // Stateless UInt32 mixing of Int32 [N] index bits XOR seed. Returns Float32
    // [N] in [0,1), identical on CPU/CUDA/Metal/Vulkan, without advancing any RNG.
    // Uses the high 24 hash bits, exactly representable in Float32. Preserves the
    // input device/backend and supports strided and empty index tensors.
    LFS_CORE_API Tensor random_from_indices(const Tensor& indices, uint32_t seed);
} // namespace lfs::core
