/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/camera_types.h"
#include "core/export.hpp"
#include "core/tensor_fwd.hpp"

namespace lfs::core {
    struct UndistortParams {
        float src_fx, src_fy, src_cx, src_cy;
        float dst_fx, dst_fy, dst_cx, dst_cy;
        int src_width, src_height;
        int dst_width, dst_height;
        CameraModelType model_type;
        float distortion[12];
        int num_distortion;
        bool crop_solve_failed = false;
    };

    LFS_CORE_API UndistortParams compute_undistort_params(
        float fx, float fy, float cx, float cy,
        int width, int height,
        const Tensor& radial, const Tensor& tangential,
        CameraModelType model, float blank_pixels = 0.0f);

    LFS_CORE_API UndistortParams scale_undistort_params(
        const UndistortParams& params, const int actual_src_width, const int actual_src_height,
        const int max_width = 0);

    struct UndistortGrid {
        int width;
        int height;
        float scale_x;
        float scale_y;
    };

    LFS_CORE_API UndistortGrid compute_undistort_grid(
        const UndistortParams& params, int resize_factor, int max_width);

    LFS_CORE_API UndistortParams prepare_undistort_params(
        const UndistortParams& params, int actual_src_width, int actual_src_height,
        int resize_factor, int max_width);

    // Maps normalized camera coordinates through the encoded camera model.
    LFS_CORE_API void distort_normalized_point(
        const UndistortParams& params, float x, float y, float& distorted_x, float& distorted_y);

    // Inverts a distorted image pixel to normalized camera coordinates.
    LFS_CORE_API bool undistort_image_point(
        const UndistortParams& params, float image_x, float image_y,
        float& normalized_x, float& normalized_y);

    LFS_CORE_API Tensor undistort_image(const Tensor& src, const UndistortParams& params,
                                        void* stream);

    LFS_CORE_API Tensor distort_image_to_source(const Tensor& src, const UndistortParams& params,
                                                Tensor& validity_mask, void* stream);

    LFS_CORE_API Tensor undistort_mask_area(const Tensor& src, const UndistortParams& params, void* stream);

    LFS_CORE_API Tensor undistort_depth_area(const Tensor& src, const UndistortParams& params, void* stream);

    LFS_CORE_API Tensor undistort_normal_area(const Tensor& src, const UndistortParams& params, void* stream);

    LFS_CORE_API Tensor distort_mask_to_source_area(const Tensor& src, const UndistortParams& params,
                                                    void* stream);

    LFS_CORE_API Tensor distort_depth_to_source_area(const Tensor& src, const UndistortParams& params,
                                                     void* stream);

    LFS_CORE_API Tensor distort_normal_to_source_area(const Tensor& src, const UndistortParams& params,
                                                      void* stream);

    LFS_CORE_API Tensor undistort_mask(const Tensor& src, const UndistortParams& params, void* stream);

    namespace internal {
        LFS_CORE_API Tensor warp_image_tensor(const Tensor& input, const UndistortParams& params, int mode, bool inverse, Tensor* validity);
        LFS_CORE_API Tensor undistort_image_tensor(const Tensor& input, const UndistortParams& params, bool mask);
        LFS_CORE_API Tensor resize_image_prior_tensor(const Tensor& input, int height, int width, bool normal);
    } // namespace internal
} // namespace lfs::core
