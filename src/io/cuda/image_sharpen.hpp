/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/image_sharpen.hpp"
#include "core/tensor.hpp"

namespace lfs::io {
    // Sharpens the luma of a contiguous [3,H,W] UInt8 or Float32 CUDA image and adds the same change to R, G and B.
    // Runs on `stream` without a host sync. Float32 results are put on the 16-bit grid when quantize_u16 is set.
    [[nodiscard]] lfs::core::Tensor sharpen_image(const lfs::core::Tensor& image, const lfs::core::SharpenParams& params,
                                                  bool quantize_u16, cudaStream_t stream);
} // namespace lfs::io
