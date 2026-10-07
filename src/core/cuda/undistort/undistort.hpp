/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera_types.h"
#include "core/cuda_types.hpp"
#include "core/tensor.hpp"
#include "core/tensor_image.hpp"

namespace lfs::core::cuda {
    LFS_CORE_API Tensor undistort_image(const Tensor& src, const UndistortParams& params,
                                        cudaStream_t stream);

    LFS_CORE_API Tensor distort_image_to_source(const Tensor& src, const UndistortParams& params,
                                                Tensor& validity_mask, cudaStream_t stream);

    LFS_CORE_API Tensor undistort_mask_area(const Tensor& src, const UndistortParams& params, cudaStream_t stream);

    LFS_CORE_API Tensor undistort_depth_area(const Tensor& src, const UndistortParams& params, cudaStream_t stream);

    LFS_CORE_API Tensor undistort_normal_area(const Tensor& src, const UndistortParams& params, cudaStream_t stream);

    LFS_CORE_API Tensor distort_mask_to_source_area(const Tensor& src, const UndistortParams& params,
                                                    cudaStream_t stream);

    LFS_CORE_API Tensor distort_depth_to_source_area(const Tensor& src, const UndistortParams& params,
                                                     cudaStream_t stream);

    LFS_CORE_API Tensor distort_normal_to_source_area(const Tensor& src, const UndistortParams& params,
                                                      cudaStream_t stream);

    LFS_CORE_API Tensor undistort_image_region(
        const Tensor& source,
        const UndistortParams& params,
        int destination_x,
        int destination_y,
        int width,
        int height,
        cudaStream_t stream);

    LFS_CORE_API Tensor undistort_mask(const Tensor& src, const UndistortParams& params, cudaStream_t stream);

    Tensor inverse_distortion_sample_map(const UndistortParams& params, cudaStream_t stream);

} // namespace lfs::core::cuda
