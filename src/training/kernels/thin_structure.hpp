/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"

namespace lfs::training::kernels {

    struct LFS_CUDA_API RidgeWorkspace {
        lfs::core::Tensor luminance;
        lfs::core::Tensor horizontal;
        lfs::core::Tensor reduction;
        void ensure_size(size_t height, size_t width);
    };

    LFS_CUDA_API float structure_base_denominator(const lfs::core::Tensor& base_weight, int height, int width, bool valid_padding);

    LFS_CUDA_API void ridge_structure_map(const lfs::core::Tensor& image,
                                          lfs::core::Tensor& output,
                                          RidgeWorkspace& workspace);

    LFS_CUDA_API void structure_photometric_weight(const lfs::core::Tensor& structure,
                                                   const lfs::core::Tensor& base_weight,
                                                   lfs::core::Tensor& output,
                                                   float gain, bool valid_padding);

} // namespace lfs::training::kernels
