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
    // Alpha's existing 16-bit cache uses the same correctly rounded value domain.
    LFS_CUDA_API void normalize_mask_u16(const std::uint16_t* input, float* output, size_t count, cudaStream_t stream);
    // Put a sidecar on the existing truncated-byte cache grid in place. Optional
    // byte output is already ready for the lossless cache encoder.
    LFS_CUDA_API Tensor quantize_mask_to_byte_grid(Tensor& mask, bool keep_bytes = true);
    // Produce the same bytes without changing a source mask shared with a resize.
    LFS_CUDA_API Tensor mask_values_to_byte_grid(const Tensor& mask);
} // namespace lfs::core::cuda
