/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::io::cuda {

    // Fused kernel: uint8 HWC -> float32 CHW normalized [0,1]
    void launch_uint8_hwc_to_float32_chw(
        const uint8_t* input,
        float* output,
        size_t height,
        size_t width,
        size_t channels,
        cudaStream_t stream = nullptr);

} // namespace lfs::io::cuda
