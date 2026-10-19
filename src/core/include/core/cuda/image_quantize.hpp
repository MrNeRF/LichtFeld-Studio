/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"

namespace lfs::core::cuda {

    /// Interleaved bytes [H, W, C] of a CUDA float image [C, H, W], each uint8(clamp(v, 0, 1) * 255 + 0.5),
    /// written in one pass so no float intermediate is allocated. Runs on the image's stream.
    LFS_CUDA_API Tensor quantize_to_interleaved_bytes(const Tensor& image);

} // namespace lfs::core::cuda
