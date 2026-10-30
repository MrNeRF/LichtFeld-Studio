/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "core/device/portability.cuh"
#include "core/distortion_model.hpp"
#include "runtime_api.h"
namespace hip_lens {
    using lfs::core::CameraModelType;
    using UndistortParams = LfsHipLens;
    namespace detail = lfs::core::detail;
#include "core/device/lens.cuh"
    template <typename T, int Channels = 0>
    __global__ void __launch_bounds__(BLOCK_DIM* BLOCK_DIM)
        undistort_image_kernel(
            const T* __restrict__ src,
            float* __restrict__ dst,
            const int runtime_channels,
            const int output_y_offset,
            const int output_y_end,
            const int quadrature,
            const UndistortParams params) {
        undistort_image_kernel_impl<true, Channels>(src, dst, runtime_channels, output_y_offset, quadrature, params, output_y_end);
    }

    __global__ void __launch_bounds__(BLOCK_DIM* BLOCK_DIM)
        distort_image_to_source_kernel(
            const float* __restrict__ src,
            float* __restrict__ dst,
            uint8_t* __restrict__ validity,
            const int channels,
            const int output_y_offset,
            const int output_y_end,
            const UndistortParams params) {
        distort_image_to_source_kernel_impl<true>(src, dst, validity, channels, output_y_offset, params, output_y_end);
    }

    __global__ void __launch_bounds__(BLOCK_DIM* BLOCK_DIM)
        undistort_area_kernel(
            const float* __restrict__ src,
            float* __restrict__ dst,
            const int channels,
            const AreaFilterMode mode,
            const int output_y_offset,
            const int output_y_end,
            const int quadrature,
            const UndistortParams params) {
        undistort_area_kernel_impl<true>(src, dst, channels, mode, output_y_offset, quadrature, params, output_y_end);
    }

    __global__ void __launch_bounds__(BLOCK_DIM* BLOCK_DIM)
        distort_area_to_source_kernel(
            const float* __restrict__ src,
            float* __restrict__ dst,
            const int channels,
            const AreaFilterMode mode,
            const int output_y_offset,
            const int output_y_end,
            const UndistortParams params) {
        distort_area_to_source_kernel_impl<true>(src, dst, channels, mode, output_y_offset, params, output_y_end);
    }

    __global__ void __launch_bounds__(BLOCK_DIM* BLOCK_DIM)
        inverse_distortion_sample_map_kernel(
            float2* __restrict__ samples,
            const UndistortParams params) {
        inverse_distortion_sample_map_kernel_impl<true>(samples, params);
    }
} // namespace hip_lens
