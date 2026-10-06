/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_image.hpp"
#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/cuda/undistort/undistort.hpp"
#include "internal/image_resample.hpp"
#include "internal/tensor_impl.hpp"
#include "internal/undistort_resample.hpp"
#include <limits>
#include <stdexcept>

namespace lfs::core::internal {
    Tensor undistort_image_region_tensor(
        const Tensor& input, const UndistortParams& params,
        const int destination_x, const int destination_y, const int width, const int height) {
        if (!input.is_valid() || input.ndim() != 3 || input.size(0) < 1 || input.size(0) > 4 ||
            (input.dtype() != DataType::UInt8 && input.dtype() != DataType::Float32) ||
            params.src_width <= 0 || params.src_height <= 0 ||
            input.size(1) != size_t(params.src_height) || input.size(2) != size_t(params.src_width)) {
            throw std::invalid_argument("undistort_image requires a CHW UInt8 or Float32 tensor matching its parameters");
        }
        if (params.dst_width <= 0 || params.dst_height <= 0 ||
            destination_x < 0 || destination_y < 0 || width <= 0 || height <= 0 ||
            int64_t(destination_x) + width > params.dst_width ||
            int64_t(destination_y) + height > params.dst_height) {
            throw std::invalid_argument("undistort_image destination region is outside the full output");
        }
        // Match the full-image fast path: a crop of an identity warp must not
        // enter the area filter and blur pixels that the full call copies exactly.
        if (warp_math::is_identity_resample(params)) {
            auto crop = input.slice(1, destination_y, destination_y + height)
                            .slice(2, destination_x, destination_x + width)
                            .contiguous();
            return crop.dtype() == DataType::UInt8 ? crop.to(DataType::Float32).div(255.0f) : crop.clone();
        }
        auto region = params;
        region.dst_cx -= destination_x;
        region.dst_cy -= destination_y;
        region.dst_width = width;
        region.dst_height = height;
        return warp_image_tensor(input, region, 0, false, nullptr);
    }

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
        const bool rgb8 = source.dtype() == DataType::UInt8;
        LFS_ASSERT_MSG(!rgb8 || (!inverse && mode == 0 && source.ndim() == 3),
                       "UInt8 image warps support forward RGB undistortion only");
        if (!inverse && mode == 0 && warp_math::is_identity_resample(p))
            return rgb8 ? source.to(DataType::Float32).div(255.0f) : source.clone();
        if (input.device() == Device::GPU) {
            pin_operands({&source});
            const auto stream = prepare_inputs_for_stream({&source}, source.stream());
            return backend_ops_for(source).image_warp(source, p, mode, inverse, validity, ExecContext{stream});
        }
        const bool scalar = input.ndim() == 2;
        const int channels = scalar ? 1 : int(input.size(0));
        if (mode == 4) {
            auto samples = Tensor::empty({size_t(p.src_height), size_t(p.src_width), 2}, Device::CPU);
            auto* values = samples.ptr<float>();
            for (int y = 0; y < p.src_height; ++y) {
                for (int x = 0; x < p.src_width; ++x) {
                    float nx, ny;
                    if (!undistort_image_point(p, x + 0.5f, y + 0.5f, nx, ny))
                        nx = ny = std::numeric_limits<float>::quiet_NaN();
                    const auto i = 2 * (size_t(y) * p.src_width + x);
                    values[i] = nx;
                    values[i + 1] = ny;
                }
            }
            return samples;
        }
        const int width = inverse ? p.src_width : p.dst_width;
        const int height = inverse ? p.src_height : p.dst_height;
        auto output = Tensor::zeros(scalar ? TensorShape{size_t(height), size_t(width)} : TensorShape{size_t(channels), size_t(height), size_t(width)}, Device::CPU);
        if (validity)
            *validity = Tensor::zeros({size_t(height), size_t(width)}, Device::CPU, DataType::UInt8);
        const int quadrature = warp_math::area_quadrature(p);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (mode == 0) {
                    if (inverse)
                        warp_math::distort_image_to_source_kernel(source.ptr<float>(), output.ptr<float>(), validity->ptr<uint8_t>(), channels, x, y, p);
                    else if (rgb8)
                        warp_math::undistort_image_kernel(source.ptr<uint8_t>(), output.ptr<float>(), channels, x, y, quadrature, p);
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
    Tensor inverse_distortion_sample_map(const UndistortParams& params, void* stream) {
#if LFS_HAS_CUDA
        if (default_gpu_backend() == GpuBackend::CUDA)
            return cuda::inverse_distortion_sample_map(params, static_cast<cudaStream_t>(stream));
#endif
        auto source = Tensor::empty({1}, Device::GPU);
        return internal::backend_ops_for(source).image_warp(source, params, 4, true, nullptr, internal::ExecContext{});
    }

    Tensor undistort_image_region(
        const Tensor& source, const UndistortParams& params,
        int x, int y, int width, int height, void* stream) {
#if LFS_HAS_CUDA
        if (gpu_backend_of(source) == GpuBackend::CUDA)
            return cuda::undistort_image_region(source, params, x, y, width, height, static_cast<cudaStream_t>(stream));
#else
        (void)stream;
#endif
        return internal::undistort_image_region_tensor(source, params, x, y, width, height);
    }

    Tensor undistort_image(const Tensor& src, const UndistortParams& params, void* stream) {
#if LFS_HAS_CUDA
        if (gpu_backend_of(src) == GpuBackend::CUDA)
            return cuda::undistort_image(src, params, static_cast<cudaStream_t>(stream));
#else
        (void)stream;
#endif
        return internal::undistort_image_region_tensor(src, params, 0, 0, params.dst_width, params.dst_height);
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
