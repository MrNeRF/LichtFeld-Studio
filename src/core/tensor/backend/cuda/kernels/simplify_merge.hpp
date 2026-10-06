/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace lfs::core::tensor_ops {
    void launch_simplify_merge(const float* means, const float* scales, const float* rotation, const float* opacity,
                               const float* appearance, const int32_t* offsets, const int32_t* members, float* out_means,
                               float* out_scales, float* out_rotation, float* out_opacity, float* out_appearance,
                               uint32_t groups, uint32_t app_dim, cudaStream_t stream);
} // namespace lfs::core::tensor_ops
