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
    };

    UndistortParams compute_undistort_params(
        float fx, float fy, float cx, float cy,
        int width, int height,
        const Tensor& radial, const Tensor& tangential,
        CameraModelType model, float blank_pixels = 0.0f);

    Tensor undistort_image(const Tensor& src, const UndistortParams& params, cudaStream_t stream);

    Tensor undistort_mask(const Tensor& src, const UndistortParams& params, cudaStream_t stream);

} // namespace lfs::core
