/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/cuda_types.hpp"
#include "core/tensor.hpp"
#include "core/tensor_image.hpp"

namespace lfs::core {

    LFS_CORE_API Tensor undistort_image(const Tensor& src, const UndistortParams& params, cudaStream_t stream);

    LFS_CORE_API Tensor undistort_image_region(
        const Tensor& source,
        const UndistortParams& params,
        int destination_x,
        int destination_y,
        int width,
        int height,
        cudaStream_t stream);

    LFS_CORE_API Tensor undistort_mask(const Tensor& src, const UndistortParams& params, cudaStream_t stream);

} // namespace lfs::core
