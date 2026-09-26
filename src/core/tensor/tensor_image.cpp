/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_image.hpp"
#if !LFS_HAS_CUDA
#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/cuda/undistort/undistort.hpp"
#endif
#include "internal/image_resample.hpp"
#include "internal/tensor_impl.hpp"
#include <limits>

namespace lfs::core::internal {
    Tensor undistort_image_tensor(const Tensor& input, const UndistortParams& p, const bool mask) {
        LFS_ASSERT_MSG(input.is_valid() && input.dtype() == DataType::Float32 &&
                           input.ndim() == (mask ? 2u : 3u) && p.src_width > 0 && p.src_height > 0 &&
                           p.dst_width > 0 && p.dst_height > 0 &&
                           input.size(input.ndim() - 1) == size_t(p.src_width) && input.size(input.ndim() - 2) == size_t(p.src_height),
                       "Undistortion requires float CHW images or HW masks matching the camera");
        LFS_ASSERT_MSG(size_t(p.src_width) * p.src_height <= INT32_MAX && size_t(p.dst_width) * p.dst_height <= INT32_MAX,
                       "Undistortion pixel count exceeds int32");
        const auto source = input.contiguous();
        if (input.device() == Device::GPU) {
            pin_operands({&source});
            const auto stream = prepare_inputs_for_stream({&source}, source.stream());
            return backend_ops_for(source).image_undistort(source, p, mask, ExecContext{stream});
        }
        const int channels = mask ? 1 : int(input.size(0));
        auto output = Tensor::empty(mask ? TensorShape{size_t(p.dst_height), size_t(p.dst_width)} : TensorShape{size_t(channels), size_t(p.dst_height), size_t(p.dst_width)}, Device::CPU);
        for (int y = 0; y < p.dst_height; ++y)
            for (int x = 0; x < p.dst_width; ++x)
                image_math::undistort(source.ptr<float>(), output.ptr<float>(), channels, p, x, y);
        return output;
    }
    Tensor resize_image_prior_tensor(const Tensor& input, int height, int width, bool normal) {
        LFS_ASSERT_MSG(input.is_valid() && input.dtype() == DataType::Float32 &&
                           input.ndim() == (normal ? 3u : 2u) && (!normal || input.size(0) == 3) && height > 0 && width > 0 &&
                           input.size(input.ndim() - 1) > 0 && input.size(input.ndim() - 2) > 0,
                       "Prior resize requires float depth HW or normals CHW and positive dimensions");
        LFS_ASSERT_MSG(input.numel() <= INT32_MAX && size_t(height) * width <= INT32_MAX,
                       "Prior resize pixel count exceeds int32");
        const auto source = input.contiguous();
        if (input.device() == Device::GPU) {
            pin_operands({&source});
            const auto stream = prepare_inputs_for_stream({&source}, source.stream());
            return backend_ops_for(source).image_resize_prior(source, height, width, normal, ExecContext{stream});
        }
        auto output = Tensor::empty(normal ? TensorShape{3, size_t(height), size_t(width)} : TensorShape{size_t(height), size_t(width)}, Device::CPU);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                image_math::resize_prior(source.ptr<float>(), output.ptr<float>(), int(input.size(input.ndim() - 1)),
                                         int(input.size(input.ndim() - 2)), width, height, normal, x, y);
        return output;
    }
} // namespace lfs::core::internal

#if !LFS_HAS_CUDA
// Builds without CUDA lack lanczos_resize.cu and undistort.cu, whose wrappers
// send every non-CUDA tensor to the portable implementations above; these
// definitions do the same.
namespace lfs::core {
    Tensor resize_depth_prior(const Tensor& input, const int output_h, const int output_w, cudaStream_t) {
        return internal::resize_image_prior_tensor(input, output_h, output_w, false);
    }

    Tensor resize_normal_prior(const Tensor& input, const int output_h, const int output_w, cudaStream_t) {
        return internal::resize_image_prior_tensor(input, output_h, output_w, true);
    }

    Tensor undistort_image(const Tensor& src, const UndistortParams& params, cudaStream_t) {
        return internal::undistort_image_tensor(src, params, false);
    }

    Tensor undistort_mask(const Tensor& src, const UndistortParams& params, cudaStream_t) {
        return internal::undistort_image_tensor(src, params, true);
    }
} // namespace lfs::core
#endif
