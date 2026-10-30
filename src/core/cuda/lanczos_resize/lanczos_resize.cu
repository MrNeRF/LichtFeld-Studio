#include "core/cuda/mask_normalization.cuh"
#include <type_traits>
/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-FileCopyrightText: 2025 Youyu Chen (original Lanczos implementation)
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-License-Identifier: MIT (original Lanczos implementation)
 */

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "core/gpu_backend_fwd.hpp"
#include "core/logger.hpp"
#include "core/tensor/backend/cuda/runtime/cuda_memory_guard.hpp"
#include "core/tensor_image.hpp"
#include "lanczos_resize.hpp"

#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>
#include <optional>

#define BLOCK_X      16
#define BLOCK_Y      16
#define NUM_CHANNELS 3

namespace cg = cooperative_groups;

namespace {
#include "core/device/resize_prior.cuh"

    struct CoefficientLayout {
        uint32_t stride;
        size_t count;
    };

    std::optional<CoefficientLayout> coefficient_layout(
        const int input_size,
        const int output_size,
        const int kernel_size) {
        if (input_size <= 0 || output_size <= 0 || kernel_size <= 0) {
            return std::nullopt;
        }

        const long double support =
            2.0L * static_cast<long double>(kernel_size) * input_size / output_size;
        const long double stride_value = std::ceil(support) + 1.0L;
        if (!std::isfinite(stride_value) || stride_value > std::numeric_limits<uint32_t>::max()) {
            return std::nullopt;
        }

        const auto stride = static_cast<uint32_t>(stride_value);
        const auto output_count = static_cast<size_t>(output_size);
        if (stride > std::numeric_limits<uint32_t>::max() / output_count) {
            return std::nullopt;
        }
        const size_t count = output_count * stride;
        if (count > std::numeric_limits<size_t>::max() / sizeof(float)) {
            return std::nullopt;
        }
        return CoefficientLayout{.stride = stride, .count = count};
    }

    lfs::core::CudaDeviceMemory<float> allocate_coefficients(
        const size_t count,
        const std::string_view axis) {
        float* pointer = nullptr;
        const cudaError_t status = cudaMalloc(&pointer, count * sizeof(float));
        if (status != cudaSuccess) {
            LFS_ENSURE_CUDA_SUCCESS_MSG(
                status,
                "cudaMalloc(Lanczos coefficients)",
                lfs::core::detail::format_cuda_safe(
                    "axis={}, element_count={}, requested_bytes={}",
                    axis, count, count * sizeof(float)));
        }

        lfs::core::CudaDeviceMemory<float> allocation;
        allocation.reset(pointer, count);
        return allocation;
    }

#include "core/device/resize_pixels.cuh"

} // namespace

namespace lfs::core {

    namespace detail {

#include "core/device/lanczos.cuh"

        template <typename T, bool ExactAlpha = false>
        static Tensor lanczos_resize_impl(
            const Tensor& input,
            int output_h,
            int output_w,
            int kernel_size,
            cudaStream_t cuda_stream) {
            const int input_h = static_cast<int>(input.size(0));
            const int input_w = static_cast<int>(input.size(1));
            const int channels = static_cast<int>(input.size(2));

            if (channels < 1 || channels > 4) {
                LOG_ERROR("lanczos_resize: Expected 1 to 4 channels, got %d", channels);
                return Tensor();
            }

            auto output = Tensor::empty(
                TensorShape({static_cast<size_t>(channels),
                             static_cast<size_t>(output_h),
                             static_cast<size_t>(output_w)}),
                Device::GPU, DataType::Float32);
            if (output.stream() != cuda_stream) {
                output.set_stream(cuda_stream);
            }

            LFS_CUDA_CHECK_MSG(
                cudaMemsetAsync(output.data_ptr(), 0, output.bytes(), cuda_stream),
                "Lanczos RGB output bytes={}", output.bytes());

            const auto layout_x = coefficient_layout(input_w, output_w, kernel_size);
            const auto layout_y = coefficient_layout(input_h, output_h, kernel_size);
            if (!layout_x || !layout_y) {
                LOG_ERROR("lanczos_resize: coefficient layout overflow");
                return {};
            }

            auto coef_x = allocate_coefficients(layout_x->count, "x");
            auto coef_y = allocate_coefficients(layout_y->count, "y");

            const int threads = BLOCK_X * BLOCK_Y;
            detail::PreComputeCoef<<<(output_w + threads - 1) / threads, threads, 0, cuda_stream>>>(
                input_w, output_w, kernel_size, layout_x->stride, coef_x.get());
            LFS_CUDA_CHECK_MSG(cudaGetLastError(), "Lanczos x coefficient kernel launch");
            detail::PreComputeCoef<<<(output_h + threads - 1) / threads, threads, 0, cuda_stream>>>(
                input_h, output_h, kernel_size, layout_y->stride, coef_y.get());
            LFS_CUDA_CHECK_MSG(cudaGetLastError(), "Lanczos y coefficient kernel launch");

            const dim3 tile_grid((output_w + BLOCK_X - 1) / BLOCK_X,
                                 (output_h + BLOCK_Y - 1) / BLOCK_Y);
            const dim3 block(BLOCK_X, BLOCK_Y, 1);

            const auto launch = [&]<uint32_t CHANNELS>() {
                detail::LanczosResampleCUDA<CHANNELS, T, ExactAlpha>
                    <<<tile_grid, block, 0, cuda_stream>>>(
                        input_h, input_w, output_h, output_w, kernel_size,
                        layout_x->stride, layout_y->stride,
                        coef_x.get(), coef_y.get(),
                        input.ptr<T>(), output.ptr<float>());
                LFS_CUDA_CHECK_MSG(cudaGetLastError(), "Lanczos interleaved resample kernel launch");
            };
            switch (channels) {
            case 1: launch.template operator()<1>(); break;
            case 2: launch.template operator()<2>(); break;
            case 3: launch.template operator()<3>(); break;
            default: launch.template operator()<4>(); break;
            }
            LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(cuda_stream), "Lanczos interleaved resample completion");
            return output;
        }

    } // namespace detail

    Tensor lanczos_resize(
        const Tensor& input,
        int output_h,
        int output_w,
        int kernel_size,
        cudaStream_t cuda_stream, bool normalize_mask_alpha) {

        if (!input.is_valid() || input.device() != Device::GPU) {
            LOG_ERROR("lanczos_resize: Input must be a valid CUDA tensor");
            return Tensor();
        }
        if (input.ndim() != 3) {
            LOG_ERROR("lanczos_resize: Input must be 3D tensor [H, W, C]");
            return Tensor();
        }
        if (output_h <= 0 || output_w <= 0 || kernel_size <= 0) {
            LOG_ERROR("lanczos_resize: Output dimensions and kernel size must be positive");
            return {};
        }
        if (input.size(0) == 0 || input.size(1) == 0 ||
            input.size(0) > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            input.size(1) > static_cast<size_t>(std::numeric_limits<int>::max())) {
            LOG_ERROR("lanczos_resize: Input dimensions must be non-zero and fit in int");
            return {};
        }

        if (input.dtype() == DataType::UInt8 && normalize_mask_alpha) {
            return detail::lanczos_resize_impl<uint8_t, true>(input, output_h, output_w, kernel_size, cuda_stream);
        } else if (input.dtype() == DataType::UInt8) {
            return detail::lanczos_resize_impl<uint8_t>(
                input, output_h, output_w, kernel_size, cuda_stream);
        } else if (input.dtype() == DataType::Float32) {
            return detail::lanczos_resize_impl<float>(
                input, output_h, output_w, kernel_size, cuda_stream);
        } else {
            LOG_ERROR("lanczos_resize: Unsupported dtype: %s",
                      dtype_name(input.dtype()));
            return Tensor();
        }
    }

    Tensor lanczos_resize_grayscale(
        const Tensor& input,
        int output_h,
        int output_w,
        int kernel_size,
        cudaStream_t cuda_stream, bool normalize_mask) {

        if (!input.is_valid() || input.device() != Device::GPU) {
            LOG_ERROR("lanczos_resize_grayscale: Input must be a valid CUDA tensor");
            return Tensor();
        }

        if (input.dtype() != DataType::UInt8 && input.dtype() != DataType::Float32) {
            LOG_ERROR("lanczos_resize_grayscale: Input must be UInt8 or Float32 dtype");
            return Tensor();
        }

        if (input.ndim() != 2) {
            LOG_ERROR("lanczos_resize_grayscale: Input must be 2D tensor [H, W]");
            return Tensor();
        }
        if (output_h <= 0 || output_w <= 0 || kernel_size <= 0) {
            LOG_ERROR("lanczos_resize_grayscale: Output dimensions and kernel size must be positive");
            return {};
        }
        if (input.size(0) == 0 || input.size(1) == 0 ||
            input.size(0) > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            input.size(1) > static_cast<size_t>(std::numeric_limits<int>::max())) {
            LOG_ERROR("lanczos_resize_grayscale: Input dimensions must be non-zero and fit in int");
            return {};
        }

        const int input_h = static_cast<int>(input.size(0));
        const int input_w = static_cast<int>(input.size(1));

        auto output = Tensor::empty(
            TensorShape({static_cast<size_t>(output_h), static_cast<size_t>(output_w)}),
            Device::GPU,
            DataType::Float32);
        if (output.stream() != cuda_stream) {
            output.set_stream(cuda_stream);
        }

        LFS_CUDA_CHECK_MSG(
            cudaMemsetAsync(output.data_ptr(), 0, output.bytes(), cuda_stream),
            "Lanczos grayscale output bytes={}", output.bytes());

        const auto layout_x = coefficient_layout(input_w, output_w, kernel_size);
        const auto layout_y = coefficient_layout(input_h, output_h, kernel_size);
        if (!layout_x || !layout_y) {
            LOG_ERROR("lanczos_resize_grayscale: coefficient layout overflow");
            return {};
        }

        auto coef_x = allocate_coefficients(layout_x->count, "x");
        auto coef_y = allocate_coefficients(layout_y->count, "y");

        detail::PreComputeCoef<<<(output_w + BLOCK_X * BLOCK_Y - 1) / (BLOCK_X * BLOCK_Y), BLOCK_X * BLOCK_Y, 0, cuda_stream>>>(
            input_w, output_w, kernel_size, layout_x->stride, coef_x.get());
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "Lanczos grayscale x coefficient kernel launch");

        detail::PreComputeCoef<<<(output_h + BLOCK_X * BLOCK_Y - 1) / (BLOCK_X * BLOCK_Y), BLOCK_X * BLOCK_Y, 0, cuda_stream>>>(
            input_h, output_h, kernel_size, layout_y->stride, coef_y.get());
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "Lanczos grayscale y coefficient kernel launch");

        const dim3 tile_grid((output_w + BLOCK_X - 1) / BLOCK_X, (output_h + BLOCK_Y - 1) / BLOCK_Y);
        const dim3 block(BLOCK_X, BLOCK_Y, 1);

        if (input.dtype() == DataType::UInt8) {
            if (normalize_mask) {
                detail::LanczosResampleGrayscaleCUDA<uint8_t, true><<<tile_grid, block, 0, cuda_stream>>>(
                    input_h, input_w, output_h, output_w, kernel_size,
                    layout_x->stride, layout_y->stride,
                    coef_x.get(), coef_y.get(),
                    input.ptr<uint8_t>(), 1.0f / 255.0f, output.ptr<float>());
                LFS_CUDA_LAUNCH_CHECK(cuda_stream, "core.lanczos.grayscale_resample");
            } else {
                detail::LanczosResampleGrayscaleCUDA<uint8_t><<<tile_grid, block, 0, cuda_stream>>>(
                    input_h, input_w, output_h, output_w, kernel_size,
                    layout_x->stride, layout_y->stride,
                    coef_x.get(), coef_y.get(),
                    input.ptr<uint8_t>(), 1.0f / 255.0f, output.ptr<float>());
                LFS_CUDA_LAUNCH_CHECK(cuda_stream, "core.lanczos.grayscale_resample");
            }
        } else {
            detail::LanczosResampleGrayscaleCUDA<float><<<tile_grid, block, 0, cuda_stream>>>(
                input_h, input_w, output_h, output_w, kernel_size,
                layout_x->stride, layout_y->stride,
                coef_x.get(), coef_y.get(),
                input.ptr<float>(), 1.0f, output.ptr<float>());
        }
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "Lanczos grayscale resample kernel launch");
        LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(cuda_stream), "Lanczos grayscale resample completion");

        return output;
    }

    Tensor lanczos_resize_float_chw(
        const Tensor& input,
        int output_h,
        int output_w,
        int kernel_size,
        cudaStream_t cuda_stream) {

        if (!input.is_valid() || input.device() != Device::GPU) {
            LOG_ERROR("lanczos_resize_float_chw: Input must be a valid CUDA tensor");
            return Tensor();
        }
        if (input.dtype() != DataType::Float32) {
            LOG_ERROR("lanczos_resize_float_chw: Input must be Float32 dtype");
            return Tensor();
        }
        if (input.ndim() != 3) {
            LOG_ERROR("lanczos_resize_float_chw: Input must be 3D tensor [C, H, W]");
            return Tensor();
        }
        if (output_h <= 0 || output_w <= 0 || kernel_size <= 0) {
            LOG_ERROR("lanczos_resize_float_chw: Output dimensions and kernel size must be positive");
            return {};
        }
        if (input.size(1) == 0 || input.size(2) == 0 ||
            input.size(1) > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            input.size(2) > static_cast<size_t>(std::numeric_limits<int>::max())) {
            LOG_ERROR("lanczos_resize_float_chw: Input dimensions must be non-zero and fit in int");
            return {};
        }

        const int channels = static_cast<int>(input.size(0));
        const int input_h = static_cast<int>(input.size(1));
        const int input_w = static_cast<int>(input.size(2));
        if (channels != 3 && channels != 4) {
            LOG_ERROR("lanczos_resize_float_chw: Only 3- or 4-channel images supported, got %d", channels);
            return Tensor();
        }

        auto output = Tensor::empty(
            TensorShape({static_cast<size_t>(channels),
                         static_cast<size_t>(output_h),
                         static_cast<size_t>(output_w)}),
            Device::GPU, DataType::Float32);
        if (output.stream() != cuda_stream) {
            output.set_stream(cuda_stream);
        }

        LFS_CUDA_CHECK_MSG(
            cudaMemsetAsync(output.data_ptr(), 0, output.bytes(), cuda_stream),
            "Lanczos CHW output bytes={}", output.bytes());

        const auto layout_x = coefficient_layout(input_w, output_w, kernel_size);
        const auto layout_y = coefficient_layout(input_h, output_h, kernel_size);
        if (!layout_x || !layout_y) {
            LOG_ERROR("lanczos_resize_float_chw: coefficient layout overflow");
            return {};
        }

        auto coef_x = allocate_coefficients(layout_x->count, "x");
        auto coef_y = allocate_coefficients(layout_y->count, "y");

        const int threads = BLOCK_X * BLOCK_Y;
        detail::PreComputeCoef<<<(output_w + threads - 1) / threads, threads, 0, cuda_stream>>>(
            input_w, output_w, kernel_size, layout_x->stride, coef_x.get());
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "Lanczos CHW x coefficient kernel launch");
        detail::PreComputeCoef<<<(output_h + threads - 1) / threads, threads, 0, cuda_stream>>>(
            input_h, output_h, kernel_size, layout_y->stride, coef_y.get());
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "Lanczos CHW y coefficient kernel launch");

        const dim3 tile_grid((output_w + BLOCK_X - 1) / BLOCK_X, (output_h + BLOCK_Y - 1) / BLOCK_Y);
        const dim3 block(BLOCK_X, BLOCK_Y, 1);

        if (channels == 3) {
            detail::LanczosResamplePlanarCUDA<3><<<tile_grid, block, 0, cuda_stream>>>(
                input_h, input_w, output_h, output_w, kernel_size,
                layout_x->stride, layout_y->stride,
                coef_x.get(), coef_y.get(), input.ptr<float>(), output.ptr<float>());
            LFS_CUDA_CHECK_MSG(cudaGetLastError(), "Lanczos CHW resample kernel launch");
        } else {
            detail::LanczosResamplePlanarCUDA<4><<<tile_grid, block, 0, cuda_stream>>>(
                input_h, input_w, output_h, output_w, kernel_size,
                layout_x->stride, layout_y->stride,
                coef_x.get(), coef_y.get(), input.ptr<float>(), output.ptr<float>());
            LFS_CUDA_CHECK_MSG(cudaGetLastError(), "Lanczos CHW resample kernel launch");
        }
        LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(cuda_stream), "Lanczos CHW resample completion");

        return output;
    }

    template <int Channels>
    Tensor resize_prior(const Tensor& input, int output_h, int output_w, cudaStream_t stream) {
        if (!input.is_valid() || input.device() != Device::CUDA || input.dtype() != DataType::Float32 ||
            !input.is_contiguous() || output_h <= 0 || output_w <= 0 ||
            input.ndim() != (Channels == 1 ? 2 : 3) || (Channels == 3 && input.size(0) != 3)) {
            throw std::invalid_argument("Prior resize requires contiguous CUDA float depth [H,W] or normals [3,H,W]");
        }
        const int h = static_cast<int>(input.size(input.ndim() - 2));
        const int w = static_cast<int>(input.size(input.ndim() - 1));
        const auto shape = Channels == 1
                               ? TensorShape({static_cast<size_t>(output_h), static_cast<size_t>(output_w)})
                               : TensorShape({3, static_cast<size_t>(output_h), static_cast<size_t>(output_w)});
        auto output = Tensor::empty(shape, Device::CUDA, DataType::Float32);
        output.set_stream(stream);
        const dim3 block(16, 16);
        const dim3 grid((output_w + 15) / 16, (output_h + 15) / 16);
        resize_prior_kernel<Channels><<<grid, block, 0, stream>>>(input.ptr<float>(), output.ptr<float>(), w, h, output_w, output_h);
        LFS_CUDA_CHECK_MSG(cudaGetLastError(), "Prior resample kernel launch");
        // Match the existing resize helpers' lifetime/stream completion contract.
        LFS_CUDA_CHECK_MSG(cudaStreamSynchronize(stream), "Prior resample completion");
        return output;
    }

    Tensor resize_depth_prior(const Tensor& input, int output_h, int output_w, cudaStream_t stream) {
        if (gpu_backend_of(input) != GpuBackend::CUDA) {
            return internal::resize_image_prior_tensor(input, output_h, output_w, false);
        }
        const GpuBackendScope backend_scope(GpuBackend::CUDA);

        return resize_prior<1>(input, output_h, output_w, stream);
    }

    Tensor resize_normal_prior(const Tensor& input, int output_h, int output_w, cudaStream_t stream) {
        if (gpu_backend_of(input) != GpuBackend::CUDA) {
            return internal::resize_image_prior_tensor(input, output_h, output_w, true);
        }
        const GpuBackendScope backend_scope(GpuBackend::CUDA);

        return resize_prior<3>(input, output_h, output_w, stream);
    }

} // namespace lfs::core
