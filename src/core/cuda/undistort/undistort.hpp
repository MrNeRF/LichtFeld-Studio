/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera_types.h"
#include "core/tensor.hpp"
#include <cuda_runtime.h>

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

    struct UndistortGrid {
        int width;
        int height;
        float scale_x;
        float scale_y;
    };

    UndistortParams compute_undistort_params(
        float fx, float fy, float cx, float cy,
        int width, int height,
        const Tensor& radial, const Tensor& tangential,
        CameraModelType model, float blank_pixels = 0.0f);

    UndistortGrid compute_undistort_grid(
        const UndistortParams& params, int resize_factor, int max_width);

    UndistortParams prepare_undistort_params(
        const UndistortParams& params, int actual_src_width, int actual_src_height,
        int resize_factor, int max_width);

    // Maps normalized camera coordinates through the encoded camera model.
    void distort_normalized_point(
        const UndistortParams& params, float x, float y, float& distorted_x, float& distorted_y);

    // Inverts a distorted image pixel to normalized camera coordinates.
    bool undistort_image_point(
        const UndistortParams& params, float image_x, float image_y,
        float& normalized_x, float& normalized_y);

    Tensor undistort_image(const Tensor& src, const UndistortParams& params,
                           cudaStream_t stream);

    Tensor distort_image_to_source(const Tensor& src, const UndistortParams& params,
                                   Tensor& validity_mask, cudaStream_t stream);

    Tensor undistort_mask_area(const Tensor& src, const UndistortParams& params, cudaStream_t stream);

    Tensor undistort_depth_area(const Tensor& src, const UndistortParams& params, cudaStream_t stream);

    Tensor undistort_normal_area(const Tensor& src, const UndistortParams& params, cudaStream_t stream);

    Tensor distort_mask_to_source_area(const Tensor& src, const UndistortParams& params,
                                       cudaStream_t stream);

    Tensor distort_depth_to_source_area(const Tensor& src, const UndistortParams& params,
                                        cudaStream_t stream);

    Tensor distort_normal_to_source_area(const Tensor& src, const UndistortParams& params,
                                         cudaStream_t stream);

    Tensor inverse_distortion_sample_map(const UndistortParams& params, cudaStream_t stream);

} // namespace lfs::core
