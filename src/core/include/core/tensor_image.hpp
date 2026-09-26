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

    namespace internal {
        LFS_CORE_API Tensor undistort_image_tensor(const Tensor& input, const UndistortParams& params, bool mask);
        LFS_CORE_API Tensor resize_image_prior_tensor(const Tensor& input, int height, int width, bool normal);
    } // namespace internal
} // namespace lfs::core
