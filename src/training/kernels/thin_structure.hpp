/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"

namespace lfs::training::kernels {

    // The separable passes run over row bands so the intermediate stays a few MiB at any image size.
    struct LFS_CUDA_API RidgeWorkspace {
        lfs::core::Tensor horizontal;
        lfs::core::Tensor reduction;
        size_t band_bytes = size_t{16} << 20;
        [[nodiscard]] size_t band_rows(size_t height, size_t width) const;
        void ensure_size(size_t band_rows, size_t width);
    };

    LFS_CUDA_API float structure_base_denominator(const lfs::core::Tensor& base_weight, int height, int width, bool valid_padding);

    LFS_CUDA_API void ridge_structure_map(const lfs::core::Tensor& image,
                                          lfs::core::Tensor& output,
                                          RidgeWorkspace& workspace);

    LFS_CUDA_API void structure_photometric_weight(const lfs::core::Tensor& structure,
                                                   const lfs::core::Tensor& base_weight,
                                                   lfs::core::Tensor& output,
                                                   float gain, bool valid_padding);

    LFS_CUDA_API void structure_densification_weight(lfs::core::Tensor& error,
                                                     const lfs::core::Tensor& structure,
                                                     float gain);

} // namespace lfs::training::kernels
