/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

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
    void launch_normalize_by_device_scalar(
        float* d_data,
        const std::size_t n,
        const float* d_scalar,
        cudaStream_t stream = nullptr);

} // namespace lfs::training::kernels
