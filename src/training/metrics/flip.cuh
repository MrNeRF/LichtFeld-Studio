/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"

#include <cuda_runtime.h>
#include <numbers>

namespace lfs::training {

    // A 0.7 m wide 3840 pixel display viewed from 0.7 m, the FLIP default.
    inline constexpr float FLIP_DEFAULT_PIXELS_PER_DEGREE = 3840.0f * std::numbers::pi_v<float> / 180.0f;

    // Per-pixel LDR-FLIP (Andersson et al. 2020) between sRGB-encoded CUDA Float32 [3,H,W] images in [0,1].
    // Returns [H,W] errors in [0,1].
    [[nodiscard]] lfs::core::Tensor flip_error_map(const lfs::core::Tensor& reference,
                                                   const lfs::core::Tensor& test,
                                                   float pixels_per_degree = FLIP_DEFAULT_PIXELS_PER_DEGREE,
                                                   cudaStream_t stream = nullptr);

    // Magma-coloured UInt8 [3,H,W] image of a [H,W] FLIP error map.
    [[nodiscard]] lfs::core::Tensor flip_error_image(const lfs::core::Tensor& error_map, cudaStream_t stream = nullptr);

} // namespace lfs::training
