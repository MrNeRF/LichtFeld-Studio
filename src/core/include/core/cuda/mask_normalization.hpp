/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"

namespace lfs::core::cuda {
    // Correctly rounded k/255, independent of CUDA fast-math. One conversion
    // pass on the input stream, preserving shape and without a float temporary.
    LFS_CUDA_API Tensor normalize_mask_bytes(const Tensor& mask);
    // Binary mask comparison preserves subnormal cutoffs even in fast-math builds.
    LFS_CUDA_API Tensor threshold_mask(const Tensor& mask, float threshold);
} // namespace lfs::core::cuda
