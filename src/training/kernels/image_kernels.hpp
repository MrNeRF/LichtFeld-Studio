/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "core/tensor_fwd.hpp"
#include "lfs/training/ops/domain_types.hpp"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::training::kernels {

    void launch_fused_canny_edge_filter_chw(
        const float* d_input_chw,
        float* d_output_hw,
        const int height,
        const int width,
        cudaStream_t stream = nullptr);
    void launch_fused_canny_edge_filter_chw(
        const uint8_t* d_input_chw,
        float* d_output_hw,
        const int height,
        const int width,
        cudaStream_t stream = nullptr);
    // Divides d_data by *d_scalar in place; the whole kernel no-ops when
    // *d_scalar <= skip_below (device-side replacement for host mean/median
    // readback + branch).
    void launch_normalize_by_device_scalar(
        float* d_data,
        const std::size_t n,
        const float* d_scalar,
        float skip_below = 0.0f,
        cudaStream_t stream = nullptr);

    // Edge guidance for one view: Canny edges of a CHW image (uint8 or float32) written to
    // `edges` [H,W], zeroed where `photometric_mask` gives no photometric weight, then divided
    // by the median of the remaining positive edges. An invalid mask keeps every pixel.
    void compute_edge_weight_map(
        const lfs::core::Tensor& image,
        const lfs::core::Tensor& photometric_mask,
        lfs::gpu_ops::MaskPhotoMode mask_mode,
        lfs::core::Tensor& edges,
        cudaStream_t stream);

    // Rounds float values in [0, 1] to the nearest 8-bit level, as saving and reloading the image would.
    lfs::core::Tensor quantize_to_8bit_grid(const lfs::core::Tensor& image);
    // Clamps to [0, 1] and rounds to the nearest of `levels` + 1 evenly spaced values.
    lfs::core::Tensor quantize_to_grid(const lfs::core::Tensor& image, float levels);

} // namespace lfs::training::kernels
