/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

__device__ __forceinline__ std::uint8_t mask_grid_byte(const float value) {
    return static_cast<std::uint8_t>(fminf(fmaxf(lfs::core::device::mul_rn(value, 255.f), 0.f), 255.f));
}
template <bool KeepBytes>
__global__ void quantize_mask_grid_kernel(float* values, std::uint8_t* bytes, size_t count) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const auto byte = mask_grid_byte(values[i]);
    values[i] = normalize_mask_byte(byte);
    if constexpr (KeepBytes)
        bytes[i] = byte;
}
