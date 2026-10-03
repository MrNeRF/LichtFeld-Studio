/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_image.hpp"
#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "internal/image_resample.hpp"
#include "internal/tensor_impl.hpp"
#include "internal/undistort_resample.hpp"
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
        return mask ? undistort_mask_area(input, p, input.stream())
                    : undistort_image(input, p, input.stream());
    }

    Tensor warp_image_tensor(const Tensor& input, const UndistortParams& p, int mode, bool inverse, Tensor* validity) {
        const auto source = input.contiguous();
        if (!inverse && mode == 0 && warp_math::is_identity_resample(p))
            return source.clone();
        if (input.device() == Device::GPU) {
            pin_operands({&source});
            const auto stream = prepare_inputs_for_stream({&source}, source.stream());
            return backend_ops_for(source).image_warp(source, p, mode, inverse, validity, ExecContext{stream});
        }
        const bool scalar = input.ndim() == 2;
        const int channels = scalar ? 1 : int(input.size(0));
        const int width = inverse ? p.src_width : p.dst_width;
        const int height = inverse ? p.src_height : p.dst_height;
        auto output = Tensor::zeros(scalar ? TensorShape{size_t(height), size_t(width)} : TensorShape{size_t(channels), size_t(height), size_t(width)}, Device::CPU);
        if (validity)
            *validity = Tensor::zeros({size_t(height), size_t(width)}, Device::CPU, DataType::UInt8);
        if (!inverse && mode == 0 && warp_math::is_identity_resample(p))
            return source.clone();
        const int quadrature = warp_math::area_quadrature(p);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (mode == 0) {
                    if (inverse)
                        warp_math::distort_image_to_source_kernel(source.ptr<float>(), output.ptr<float>(), validity->ptr<uint8_t>(), channels, x, y, p);
                    else
                        warp_math::undistort_image_kernel(source.ptr<float>(), output.ptr<float>(), channels, x, y, quadrature, p);
                } else {
                    const auto filter = static_cast<warp_math::AreaFilterMode>(mode - 1);
                    if (inverse)
                        warp_math::distort_area_to_source_kernel(source.ptr<float>(), output.ptr<float>(), channels, filter, x, y, p);
                    else
                        warp_math::undistort_area_kernel(source.ptr<float>(), output.ptr<float>(), channels, filter, x, y, quadrature, p);
                }
            }
        }
        return output;
    }

    Tensor GpuBackendOps::image_warp(const Tensor& input, const UndistortParams& params, int mode, bool inverse, Tensor* validity, ExecContext) {
        const GpuBackendScope scope(*gpu_backend_of(input));
        auto result = warp_image_tensor(input.cpu(), params, mode, inverse, validity).to(Device::GPU);
        if (validity)
            *validity = validity->to(Device::GPU);
        return result;
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
// Builds without CUDA use the portable prior resize implementations.
namespace lfs::core {
    Tensor resize_depth_prior(const Tensor& input, const int output_h, const int output_w, cudaStream_t) {
        return internal::resize_image_prior_tensor(input, output_h, output_w, false);
    }

    Tensor resize_normal_prior(const Tensor& input, const int output_h, const int output_w, cudaStream_t) {
        return internal::resize_image_prior_tensor(input, output_h, output_w, true);
    }
} // namespace lfs::core
#endif

namespace lfs::core {
    Tensor undistort_image(const Tensor& src, const UndistortParams& params, void* stream) {
#if LFS_HAS_CUDA
        if (gpu_backend_of(src) == GpuBackend::CUDA)
            return cuda::undistort_image(src, params, static_cast<cudaStream_t>(stream));
#else
        (void)stream;
#endif
        return internal::warp_image_tensor(src, params, 0, false, nullptr);
    }

    Tensor undistort_mask(const Tensor& src, const UndistortParams& params, void* stream) {
#if LFS_HAS_CUDA
        if (gpu_backend_of(src) == GpuBackend::CUDA)
            return cuda::undistort_mask(src, params, static_cast<cudaStream_t>(stream));
#else
        (void)stream;
#endif
        return internal::warp_image_tensor(src, params, 1, false, nullptr);
    }

    Tensor undistort_mask_area(const Tensor& src, const UndistortParams& params, void* stream) {
#if LFS_HAS_CUDA
        if (gpu_backend_of(src) == GpuBackend::CUDA)
            return cuda::undistort_mask_area(src, params, static_cast<cudaStream_t>(stream));
#else
        (void)stream;
#endif
        return internal::warp_image_tensor(src, params, 1, false, nullptr);
    }

    Tensor undistort_depth_area(const Tensor& src, const UndistortParams& params, void* stream) {
#if LFS_HAS_CUDA
        if (gpu_backend_of(src) == GpuBackend::CUDA)
            return cuda::undistort_depth_area(src, params, static_cast<cudaStream_t>(stream));
#else
        (void)stream;
#endif
        return internal::warp_image_tensor(src, params, 2, false, nullptr);
    }

    Tensor undistort_normal_area(const Tensor& src, const UndistortParams& params, void* stream) {
#if LFS_HAS_CUDA
        if (gpu_backend_of(src) == GpuBackend::CUDA)
            return cuda::undistort_normal_area(src, params, static_cast<cudaStream_t>(stream));
#else
        (void)stream;
#endif
        return internal::warp_image_tensor(src, params, 3, false, nullptr);
    }

    Tensor distort_mask_to_source_area(const Tensor& src, const UndistortParams& params, void* stream) {
#if LFS_HAS_CUDA
        if (gpu_backend_of(src) == GpuBackend::CUDA)
            return cuda::distort_mask_to_source_area(src, params, static_cast<cudaStream_t>(stream));
#else
        (void)stream;
#endif
        return internal::warp_image_tensor(src, params, 1, true, nullptr);
    }

    Tensor distort_depth_to_source_area(const Tensor& src, const UndistortParams& params, void* stream) {
#if LFS_HAS_CUDA
        if (gpu_backend_of(src) == GpuBackend::CUDA)
            return cuda::distort_depth_to_source_area(src, params, static_cast<cudaStream_t>(stream));
#else
        (void)stream;
#endif
        return internal::warp_image_tensor(src, params, 2, true, nullptr);
    }

    Tensor distort_normal_to_source_area(const Tensor& src, const UndistortParams& params, void* stream) {
#if LFS_HAS_CUDA
        if (gpu_backend_of(src) == GpuBackend::CUDA)
            return cuda::distort_normal_to_source_area(src, params, static_cast<cudaStream_t>(stream));
#else
        (void)stream;
#endif
        return internal::warp_image_tensor(src, params, 3, true, nullptr);
    }

    Tensor distort_image_to_source(const Tensor& src, const UndistortParams& params, Tensor& validity, void* stream) {
#if LFS_HAS_CUDA
        if (gpu_backend_of(src) == GpuBackend::CUDA)
            return cuda::distort_image_to_source(src, params, validity, static_cast<cudaStream_t>(stream));
#else
        (void)stream;
#endif
        return internal::warp_image_tensor(src, params, 0, true, &validity);
    }
} // namespace lfs::core
