/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/gpu_backend_fwd.hpp"
#include "core/logger.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "distortion_model.cuh"
#include "undistort.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nvtx3/nvToolsExt.h>
#include <stdexcept>
#include <type_traits>

namespace lfs::core::cuda {

    namespace {

#include "core/device/lens.cuh"

        template <typename T>
        __global__ void __launch_bounds__(BLOCK_DIM* BLOCK_DIM)
            undistort_image_kernel(
                const T* __restrict__ src,
                float* __restrict__ dst,
                const int channels,
                const int output_y_offset,
                const int quadrature,
                const UndistortParams params) {
            undistort_image_kernel_impl<false>(src, dst, channels, output_y_offset, quadrature, params, 0);
        }

        __global__ void __launch_bounds__(BLOCK_DIM* BLOCK_DIM)
            distort_image_to_source_kernel(
                const float* __restrict__ src,
                float* __restrict__ dst,
                uint8_t* __restrict__ validity,
                const int channels,
                const int output_y_offset,
                const __grid_constant__ UndistortParams params) {
            distort_image_to_source_kernel_impl<false>(src, dst, validity, channels, output_y_offset, params, 0);
        }

        __global__ void __launch_bounds__(BLOCK_DIM* BLOCK_DIM)
            undistort_area_kernel(
                const float* __restrict__ src,
                float* __restrict__ dst,
                const int channels,
                const AreaFilterMode mode,
                const int output_y_offset,
                const int quadrature,
                const UndistortParams params) {
            undistort_area_kernel_impl<false>(src, dst, channels, mode, output_y_offset, quadrature, params, 0);
        }

        __global__ void __launch_bounds__(BLOCK_DIM* BLOCK_DIM)
            distort_area_to_source_kernel(
                const float* __restrict__ src,
                float* __restrict__ dst,
                const int channels,
                const AreaFilterMode mode,
                const int output_y_offset,
                const __grid_constant__ UndistortParams params) {
            distort_area_to_source_kernel_impl<false>(src, dst, channels, mode, output_y_offset, params, 0);
        }

        __global__ void __launch_bounds__(BLOCK_DIM* BLOCK_DIM)
            inverse_distortion_sample_map_kernel(
                float2* __restrict__ samples,
                const __grid_constant__ UndistortParams params) {
            inverse_distortion_sample_map_kernel_impl<false>(samples, params);
        }
        bool is_identity_resample(const UndistortParams& params) {
            if (static_cast<CameraModelType>(params.model_type) != CameraModelType::PINHOLE ||
                params.src_width != params.dst_width ||
                params.src_height != params.dst_height ||
                params.src_fx != params.dst_fx || params.src_fy != params.dst_fy ||
                params.src_cx != params.dst_cx || params.src_cy != params.dst_cy) {
                return false;
            }
            for (int i = 0; i < params.num_distortion; ++i) {
                if (params.distortion[i] != 0.0f)
                    return false;
            }
            return true;
        }

    } // anonymous namespace

    Tensor undistort_image_region(
        const Tensor& src, const UndistortParams& full_params,
        const int x, const int y, const int width, const int height, cudaStream_t stream) {
        if (!src.is_valid() || src.ndim() != 3 || src.device() != Device::GPU ||
            (src.dtype() != DataType::UInt8 && src.dtype() != DataType::Float32) ||
            src.size(0) == 0 || src.size(0) > 4 || full_params.src_width <= 0 || full_params.src_height <= 0 ||
            src.size(1) != size_t(full_params.src_height) || src.size(2) != size_t(full_params.src_width) ||
            x < 0 || y < 0 || width <= 0 || height <= 0 ||
            int64_t(x) + width > full_params.dst_width || int64_t(y) + height > full_params.dst_height) {
            throw std::invalid_argument("Invalid undistort image source or destination region");
        }
        const GpuBackendScope backend_scope(GpuBackend::CUDA);
        const CUDAStreamGuard stream_guard(stream);
        const auto input = src.contiguous();
        input.sync_to_stream(stream);
        auto params = full_params;
        params.dst_cx -= x;
        params.dst_cy -= y;
        params.dst_width = width;
        params.dst_height = height;
        const int channels = static_cast<int>(input.size(0));
        if (is_identity_resample(full_params)) {
            auto crop = input.slice(1, y, y + height).slice(2, x, x + width).contiguous();
            return crop.dtype() == DataType::UInt8 ? crop.to(DataType::Float32).div(255.0f) : crop.clone();
        }
        auto dst = Tensor::empty({size_t(channels), size_t(height), size_t(width)}, Device::GPU, DataType::Float32);
        const dim3 block(BLOCK_DIM, BLOCK_DIM);
        for (int output_y = 0; output_y < height; output_y += OUTPUT_TILE_ROWS) {
            const int tile_rows = std::min(OUTPUT_TILE_ROWS, height - output_y);
            const dim3 grid((width + BLOCK_DIM - 1) / BLOCK_DIM, (tile_rows + BLOCK_DIM - 1) / BLOCK_DIM);
            if (input.dtype() == DataType::UInt8) {
                undistort_image_kernel<<<grid, block, 0, stream>>>(
                    input.ptr<std::uint8_t>(), dst.ptr<float>(), channels, output_y, area_quadrature(params), params);
                if (cudaGetLastError() != cudaSuccess)
                    throw std::runtime_error("undistort image UInt8 kernel launch failed");
            } else {
                undistort_image_kernel<<<grid, block, 0, stream>>>(
                    input.ptr<float>(), dst.ptr<float>(), channels, output_y, area_quadrature(params), params);
                if (cudaGetLastError() != cudaSuccess)
                    throw std::runtime_error("undistort image Float32 kernel launch failed");
            }
        }
        return dst;
    }

    Tensor undistort_image(const Tensor& src, const UndistortParams& params, cudaStream_t stream) {
        return undistort_image_region(src, params, 0, 0, params.dst_width, params.dst_height, stream);
    }

    Tensor distort_image_to_source(
        const Tensor& src, const UndistortParams& params,
        Tensor& validity_mask, cudaStream_t stream) {
        const GpuBackendScope backend_scope(GpuBackend::CUDA);
        assert(src.is_valid());
        assert(src.ndim() == 3);
        assert(src.device() == Device::GPU);
        assert(src.dtype() == DataType::Float32);
        const CUDAStreamGuard stream_guard(stream);
        const auto input = src.contiguous();
        input.sync_to_stream(stream);
        const int channels = static_cast<int>(input.shape()[0]);
        assert(channels > 0 && channels <= 4);
        assert(static_cast<int>(src.shape()[1]) == params.dst_height);
        assert(static_cast<int>(src.shape()[2]) == params.dst_width);

        auto dst = Tensor::zeros(
            {static_cast<size_t>(channels), static_cast<size_t>(params.src_height),
             static_cast<size_t>(params.src_width)},
            Device::GPU, DataType::Float32);
        validity_mask = Tensor::zeros(
            {static_cast<size_t>(params.src_height), static_cast<size_t>(params.src_width)},
            Device::GPU, DataType::UInt8);
        const dim3 block(BLOCK_DIM, BLOCK_DIM);
        for (int output_y = 0; output_y < params.src_height; output_y += OUTPUT_TILE_ROWS) {
            const int tile_rows = std::min(OUTPUT_TILE_ROWS, params.src_height - output_y);
            const dim3 grid(
                (params.src_width + BLOCK_DIM - 1) / BLOCK_DIM,
                (tile_rows + BLOCK_DIM - 1) / BLOCK_DIM);
            distort_image_to_source_kernel<<<grid, block, 0, stream>>>(
                input.ptr<float>(), dst.ptr<float>(), validity_mask.ptr<uint8_t>(),
                channels, output_y, params);
        }
        const cudaError_t error = cudaGetLastError();
        assert(error == cudaSuccess && "distort_image_to_source_kernel launch failed");
        return dst;
    }

    namespace {
        Tensor launch_undistort_area(
            const Tensor& src, const UndistortParams& params,
            const AreaFilterMode mode, cudaStream_t stream) {
            const GpuBackendScope backend_scope(GpuBackend::CUDA);
            assert(src.is_valid());
            assert(src.device() == Device::GPU);
            assert(src.dtype() == DataType::Float32);
            const CUDAStreamGuard stream_guard(stream);
            const auto input = src.contiguous();
            input.sync_to_stream(stream);
            const bool scalar = input.ndim() == 2;
            const int channels = scalar ? 1 : static_cast<int>(input.shape()[0]);
            assert((scalar || src.ndim() == 3) && channels > 0 && channels <= 4);
            assert(static_cast<int>(src.shape()[src.ndim() - 2]) == params.src_height);
            assert(static_cast<int>(src.shape()[src.ndim() - 1]) == params.src_width);
            if (mode == AreaFilterMode::NORMAL)
                assert(!scalar && channels == 3);

            TensorShape output_shape = scalar
                                           ? TensorShape({static_cast<size_t>(params.dst_height),
                                                          static_cast<size_t>(params.dst_width)})
                                           : TensorShape({static_cast<size_t>(channels),
                                                          static_cast<size_t>(params.dst_height),
                                                          static_cast<size_t>(params.dst_width)});
            auto dst = Tensor::zeros(output_shape, Device::GPU, DataType::Float32);
            const dim3 block(BLOCK_DIM, BLOCK_DIM);
            for (int output_y = 0; output_y < params.dst_height; output_y += OUTPUT_TILE_ROWS) {
                const int tile_rows = std::min(OUTPUT_TILE_ROWS, params.dst_height - output_y);
                const dim3 grid(
                    (params.dst_width + BLOCK_DIM - 1) / BLOCK_DIM,
                    (tile_rows + BLOCK_DIM - 1) / BLOCK_DIM);
                undistort_area_kernel<<<grid, block, 0, stream>>>(
                    input.ptr<float>(), dst.ptr<float>(), channels, mode, output_y,
                    area_quadrature(params), params);
            }
            const cudaError_t error = cudaGetLastError();
            assert(error == cudaSuccess && "undistort_area_kernel launch failed");
            return dst;
        }
    } // namespace

    Tensor undistort_mask_area(
        const Tensor& src, const UndistortParams& params, cudaStream_t stream) {
        return launch_undistort_area(src, params, AreaFilterMode::NONNEGATIVE, stream);
    }

    Tensor undistort_depth_area(
        const Tensor& src, const UndistortParams& params, cudaStream_t stream) {
        return launch_undistort_area(src, params, AreaFilterMode::DEPTH, stream);
    }

    Tensor undistort_normal_area(
        const Tensor& src, const UndistortParams& params, cudaStream_t stream) {
        return launch_undistort_area(src, params, AreaFilterMode::NORMAL, stream);
    }

    namespace {
        Tensor launch_distort_area_to_source(
            const Tensor& src, const UndistortParams& params,
            const AreaFilterMode mode, cudaStream_t stream) {
            const GpuBackendScope backend_scope(GpuBackend::CUDA);
            assert(src.is_valid());
            assert(src.device() == Device::GPU);
            assert(src.dtype() == DataType::Float32);
            const CUDAStreamGuard stream_guard(stream);
            const auto input = src.contiguous();
            input.sync_to_stream(stream);
            const bool scalar = input.ndim() == 2;
            const int channels = scalar ? 1 : static_cast<int>(input.shape()[0]);
            assert((scalar || src.ndim() == 3) && channels > 0 && channels <= 4);
            assert(static_cast<int>(src.shape()[src.ndim() - 2]) == params.dst_height);
            assert(static_cast<int>(src.shape()[src.ndim() - 1]) == params.dst_width);
            if (mode == AreaFilterMode::NORMAL)
                assert(!scalar && channels == 3);

            TensorShape output_shape = scalar
                                           ? TensorShape({static_cast<size_t>(params.src_height),
                                                          static_cast<size_t>(params.src_width)})
                                           : TensorShape({static_cast<size_t>(channels),
                                                          static_cast<size_t>(params.src_height),
                                                          static_cast<size_t>(params.src_width)});
            auto dst = Tensor::zeros(output_shape, Device::GPU, DataType::Float32);
            const dim3 block(BLOCK_DIM, BLOCK_DIM);
            for (int output_y = 0; output_y < params.src_height; output_y += OUTPUT_TILE_ROWS) {
                const int tile_rows = std::min(OUTPUT_TILE_ROWS, params.src_height - output_y);
                const dim3 grid(
                    (params.src_width + BLOCK_DIM - 1) / BLOCK_DIM,
                    (tile_rows + BLOCK_DIM - 1) / BLOCK_DIM);
                distort_area_to_source_kernel<<<grid, block, 0, stream>>>(
                    input.ptr<float>(), dst.ptr<float>(), channels, mode, output_y, params);
            }
            const cudaError_t error = cudaGetLastError();
            assert(error == cudaSuccess && "distort_area_to_source_kernel launch failed");
            return dst;
        }
    } // namespace

    Tensor distort_mask_to_source_area(
        const Tensor& src, const UndistortParams& params, cudaStream_t stream) {
        return launch_distort_area_to_source(
            src, params, AreaFilterMode::NONNEGATIVE, stream);
    }

    Tensor distort_depth_to_source_area(
        const Tensor& src, const UndistortParams& params, cudaStream_t stream) {
        return launch_distort_area_to_source(src, params, AreaFilterMode::DEPTH, stream);
    }

    Tensor distort_normal_to_source_area(
        const Tensor& src, const UndistortParams& params, cudaStream_t stream) {
        return launch_distort_area_to_source(src, params, AreaFilterMode::NORMAL, stream);
    }

    Tensor undistort_mask(const Tensor& src, const UndistortParams& params, cudaStream_t stream) {
        return undistort_mask_area(src, params, stream);
    }

    Tensor inverse_distortion_sample_map(const UndistortParams& params, cudaStream_t stream) {
        assert(params.src_width > 0 && params.src_height > 0);
        nvtxRangePush("inverse_distortion_sample_map");
        const CUDAStreamGuard stream_guard(stream);
        auto samples = Tensor::empty(
            {static_cast<size_t>(params.src_height), static_cast<size_t>(params.src_width), 2},
            Device::GPU, DataType::Float32);
        const dim3 block(BLOCK_DIM, BLOCK_DIM);
        const dim3 grid(
            (params.src_width + BLOCK_DIM - 1) / BLOCK_DIM,
            (params.src_height + BLOCK_DIM - 1) / BLOCK_DIM);
        inverse_distortion_sample_map_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<float2*>(samples.ptr<float>()), params);
        const cudaError_t error = cudaGetLastError();
        assert(error == cudaSuccess && "inverse_distortion_sample_map_kernel launch failed");
        nvtxRangePop();
        return samples;
    }

} // namespace lfs::core::cuda
