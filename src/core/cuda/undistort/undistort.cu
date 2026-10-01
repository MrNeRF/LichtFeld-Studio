/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/gpu_backend_fwd.hpp"
#include "core/logger.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "core/tensor_image.hpp"
#include "undistort.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>
#include <nvtx3/nvToolsExt.h>
#include <stdexcept>

namespace lfs::core {

    namespace {

        constexpr int BLOCK_DIM = 16;
        constexpr float PIXEL_CENTER_OFFSET = 0.5f;

        class ScopedNvtxRange {
        public:
            explicit ScopedNvtxRange(const char* name) { nvtxRangePush(name); }
            ~ScopedNvtxRange() { nvtxRangePop(); }
            ScopedNvtxRange(const ScopedNvtxRange&) = delete;
            ScopedNvtxRange& operator=(const ScopedNvtxRange&) = delete;
        };

        // COLMAP sensor/models.h (BSD-3 licensed formulas)
        __device__ void apply_distortion_pinhole(
            const float x, const float y,
            const float* __restrict__ dist, const int num_dist,
            float& dx, float& dy) {

            const float r2 = x * x + y * y;
            const float r4 = r2 * r2;
            const float r6 = r4 * r2;

            const float k1 = num_dist > 0 ? dist[0] : 0.0f;
            const float k2 = num_dist > 1 ? dist[1] : 0.0f;
            const float k3 = num_dist > 2 ? dist[2] : 0.0f;
            const float radial = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;

            const float p1 = num_dist > 3 ? dist[3] : 0.0f;
            const float p2 = num_dist > 4 ? dist[4] : 0.0f;

            dx = x * radial + 2.0f * p1 * x * y + p2 * (r2 + 2.0f * x * x);
            dy = y * radial + p1 * (r2 + 2.0f * y * y) + 2.0f * p2 * x * y;
        }

        __device__ void apply_distortion_fisheye(
            const float x, const float y,
            const float* __restrict__ dist, const int num_dist,
            float& dx, float& dy) {

            const float r = sqrtf(x * x + y * y);
            if (r < 1e-8f) {
                dx = x;
                dy = y;
                return;
            }

            const float theta = atanf(r);
            const float theta2 = theta * theta;
            const float theta4 = theta2 * theta2;
            const float theta6 = theta4 * theta2;
            const float theta8 = theta4 * theta4;

            const float k1 = num_dist > 0 ? dist[0] : 0.0f;
            const float k2 = num_dist > 1 ? dist[1] : 0.0f;
            const float k3 = num_dist > 2 ? dist[2] : 0.0f;
            const float k4 = num_dist > 3 ? dist[3] : 0.0f;

            const float theta_d = theta * (1.0f + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
            const float scale = theta_d / r;

            dx = x * scale;
            dy = y * scale;
        }

        __device__ void apply_distortion_thin_prism_fisheye(
            const float x, const float y,
            const float* __restrict__ dist, const int num_dist,
            float& dx, float& dy) {

            const float r = sqrtf(x * x + y * y);
            if (r < 1e-8f) {
                dx = x;
                dy = y;
                return;
            }

            const float theta = atanf(r);
            const float theta2 = theta * theta;
            const float theta4 = theta2 * theta2;
            const float theta6 = theta4 * theta2;
            const float theta8 = theta4 * theta4;

            const float k1 = num_dist > 0 ? dist[0] : 0.0f;
            const float k2 = num_dist > 1 ? dist[1] : 0.0f;
            const float k3 = num_dist > 2 ? dist[2] : 0.0f;
            const float k4 = num_dist > 3 ? dist[3] : 0.0f;

            const float theta_d = theta * (1.0f + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
            const float scale = theta_d / r;

            float xd = x * scale;
            float yd = y * scale;

            const float p1 = num_dist > 4 ? dist[4] : 0.0f;
            const float p2 = num_dist > 5 ? dist[5] : 0.0f;
            const float r2 = xd * xd + yd * yd;
            xd += 2.0f * p1 * xd * yd + p2 * (r2 + 2.0f * xd * xd);
            yd += p1 * (r2 + 2.0f * yd * yd) + 2.0f * p2 * xd * yd;

            const float s1 = num_dist > 6 ? dist[6] : 0.0f;
            const float s2 = num_dist > 7 ? dist[7] : 0.0f;
            const float s3 = num_dist > 8 ? dist[8] : 0.0f;
            const float s4 = num_dist > 9 ? dist[9] : 0.0f;
            const float r2d = xd * xd + yd * yd;
            const float r4d = r2d * r2d;
            xd += s1 * r2d + s2 * r4d;
            yd += s3 * r2d + s4 * r4d;

            dx = xd;
            dy = yd;
        }

        __device__ void apply_distortion(
            const float x, const float y,
            const CameraModelType model,
            const float* __restrict__ dist, const int num_dist,
            float& dx, float& dy) {

            switch (model) {
            case CameraModelType::PINHOLE:
                apply_distortion_pinhole(x, y, dist, num_dist, dx, dy);
                break;
            case CameraModelType::FISHEYE:
                apply_distortion_fisheye(x, y, dist, num_dist, dx, dy);
                break;
            case CameraModelType::THIN_PRISM_FISHEYE:
                apply_distortion_thin_prism_fisheye(x, y, dist, num_dist, dx, dy);
                break;
            default:
                dx = x;
                dy = y;
                break;
            }
        }

        template <typename T>
        __device__ float normalized_sample(const T value) {
            return static_cast<float>(value);
        }

        template <>
        __device__ float normalized_sample<std::uint8_t>(const std::uint8_t value) {
            return static_cast<float>(value) / 255.0f;
        }

        template <typename T>
        __device__ float bilinear_sample(
            const T* __restrict__ src,
            const int width, const int height, const int stride,
            const float sx, const float sy) {

            const auto get_pixel_constant_border = [&](const int y, const int x) {
                if (x >= 0 && y >= 0 && x < width && y < height) {
                    return normalized_sample(src[y * stride + x]);
                }
                return 0.0f;
            };

            const float x0f = floorf(sx);
            const float y0f = floorf(sy);
            const int x0 = static_cast<int>(x0f);
            const int y0 = static_cast<int>(y0f);
            const int x1 = x0 + 1;
            const int y1 = y0 + 1;

            const float fx = sx - x0f;
            const float fy = sy - y0f;

            const float v00 = get_pixel_constant_border(y0, x0);
            const float v01 = get_pixel_constant_border(y0, x1);
            const float v10 = get_pixel_constant_border(y1, x0);
            const float v11 = get_pixel_constant_border(y1, x1);

            return (1.0f - fy) * ((1.0f - fx) * v00 + fx * v01) +
                   fy * ((1.0f - fx) * v10 + fx * v11);
        }

        template <typename T>
        __global__ void __launch_bounds__(BLOCK_DIM* BLOCK_DIM)
            undistort_image_kernel(
                const T* __restrict__ src,
                float* __restrict__ dst,
                const int channels,
                const UndistortParams params,
                const int destination_x,
                const int destination_y,
                const int region_width,
                const int region_height) {

            const int local_x = blockIdx.x * BLOCK_DIM + threadIdx.x;
            const int local_y = blockIdx.y * BLOCK_DIM + threadIdx.y;
            if (local_x >= region_width || local_y >= region_height)
                return;

            const int global_x = destination_x + local_x;
            const int global_y = destination_y + local_y;
            const float pixel_x = static_cast<float>(global_x) + PIXEL_CENTER_OFFSET;
            const float pixel_y = static_cast<float>(global_y) + PIXEL_CENTER_OFFSET;
            const float nx = (pixel_x - params.dst_cx) / params.dst_fx;
            const float ny = (pixel_y - params.dst_cy) / params.dst_fy;

            float dnx, dny;
            apply_distortion(nx, ny, params.model_type, params.distortion, params.num_distortion, dnx, dny);

            const float sx = dnx * params.src_fx + params.src_cx - PIXEL_CENTER_OFFSET;
            const float sy = dny * params.src_fy + params.src_cy - PIXEL_CENTER_OFFSET;
            const int src_plane = params.src_height * params.src_width;
            const int dst_plane = region_height * region_width;
            for (int c = 0; c < channels; ++c) {
                dst[c * dst_plane + local_y * region_width + local_x] =
                    bilinear_sample(
                        src + c * src_plane,
                        params.src_width,
                        params.src_height,
                        params.src_width,
                        sx,
                        sy);
            }
        }

        Tensor undistort_image_impl(
            const Tensor& source,
            const UndistortParams& params,
            const int destination_x,
            const int destination_y,
            const int width,
            const int height,
            cudaStream_t requested_stream,
            const char* range_name) {
            if (!source.is_valid()) {
                throw std::invalid_argument(
                    "undistort_image requires a valid source tensor");
            }
            if (source.ndim() != 3) {
                throw std::invalid_argument(
                    "undistort_image requires a rank-3 CHW tensor");
            }
            if (source.device() != Device::GPU) {
                throw std::invalid_argument(
                    "undistort_image requires a CUDA tensor");
            }
            if (!source.is_contiguous()) {
                throw std::invalid_argument(
                    "undistort_image requires a contiguous CHW tensor");
            }
            if (source.dtype() != DataType::UInt8 &&
                source.dtype() != DataType::Float32) {
                throw std::invalid_argument(
                    "undistort_image requires UInt8 or Float32 input");
            }
            if (params.src_width <= 0 || params.src_height <= 0 ||
                source.shape()[1] != static_cast<std::size_t>(params.src_height) ||
                source.shape()[2] != static_cast<std::size_t>(params.src_width)) {
                throw std::invalid_argument(
                    "undistort_image source dimensions do not match its parameters");
            }
            if (params.dst_width <= 0 || params.dst_height <= 0 ||
                destination_x < 0 || destination_y < 0 || width <= 0 || height <= 0 ||
                static_cast<std::int64_t>(destination_x) + width > params.dst_width ||
                static_cast<std::int64_t>(destination_y) + height > params.dst_height) {
                throw std::invalid_argument(
                    "undistort_image destination region is outside the full output");
            }

            const GpuBackendScope backend_scope(GpuBackend::CUDA);
            const ScopedNvtxRange nvtx_range(range_name);
            const cudaStream_t execution_stream =
                prepare_inputs_for_stream({&source}, requested_stream);
            const CUDAStreamGuard stream_guard(execution_stream);
            const int channels = static_cast<int>(source.shape()[0]);
            Tensor destination = Tensor::empty(
                {static_cast<size_t>(channels),
                 static_cast<size_t>(height),
                 static_cast<size_t>(width)},
                Device::GPU,
                DataType::Float32);

            const dim3 block(BLOCK_DIM, BLOCK_DIM);
            const dim3 grid(
                (width + BLOCK_DIM - 1) / BLOCK_DIM,
                (height + BLOCK_DIM - 1) / BLOCK_DIM);
            if (source.dtype() == DataType::UInt8) {
                undistort_image_kernel<<<grid, block, 0, execution_stream>>>(
                    source.ptr<std::uint8_t>(), destination.ptr<float>(), channels, params,
                    destination_x, destination_y, width, height);
            } else {
                undistort_image_kernel<<<grid, block, 0, execution_stream>>>(
                    source.ptr<float>(), destination.ptr<float>(), channels, params,
                    destination_x, destination_y, width, height);
            }
            if (cudaGetLastError() != cudaSuccess) {
                throw std::runtime_error("undistort image kernel launch failed");
            }
            return destination;
        }

        __global__ void __launch_bounds__(BLOCK_DIM* BLOCK_DIM)
            undistort_mask_kernel(
                const float* __restrict__ src,
                float* __restrict__ dst,
                const UndistortParams params) {

            const int ox = blockIdx.x * BLOCK_DIM + threadIdx.x;
            const int oy = blockIdx.y * BLOCK_DIM + threadIdx.y;

            if (ox >= params.dst_width || oy >= params.dst_height)
                return;

            const float pixel_x = static_cast<float>(ox) + PIXEL_CENTER_OFFSET;
            const float pixel_y = static_cast<float>(oy) + PIXEL_CENTER_OFFSET;
            const float nx = (pixel_x - params.dst_cx) / params.dst_fx;
            const float ny = (pixel_y - params.dst_cy) / params.dst_fy;

            float dnx, dny;
            apply_distortion(nx, ny, params.model_type, params.distortion, params.num_distortion, dnx, dny);

            const float sx = dnx * params.src_fx + params.src_cx - PIXEL_CENTER_OFFSET;
            const float sy = dny * params.src_fy + params.src_cy - PIXEL_CENTER_OFFSET;

            dst[oy * params.dst_width + ox] =
                bilinear_sample(src, params.src_width, params.src_height, params.src_width, sx, sy);
        }

    } // anonymous namespace

    Tensor undistort_image(const Tensor& src, const UndistortParams& params, cudaStream_t stream) {
        if (src.is_valid() && gpu_backend_of(src) != GpuBackend::CUDA) {
            return internal::undistort_image_region_tensor(
                src, params, 0, 0, params.dst_width, params.dst_height);
        }
        return undistort_image_impl(
            src,
            params,
            0,
            0,
            params.dst_width,
            params.dst_height,
            stream,
            "undistort_image");
    }

    Tensor undistort_image_region(
        const Tensor& source,
        const UndistortParams& params,
        const int destination_x,
        const int destination_y,
        const int width,
        const int height,
        cudaStream_t stream) {
        if (source.is_valid() && gpu_backend_of(source) != GpuBackend::CUDA) {
            return internal::undistort_image_region_tensor(
                source, params, destination_x, destination_y, width, height);
        }
        return undistort_image_impl(
            source,
            params,
            destination_x,
            destination_y,
            width,
            height,
            stream,
            "undistort_image_region");
    }

    Tensor undistort_mask(const Tensor& src, const UndistortParams& params, cudaStream_t stream) {
        if (gpu_backend_of(src) != GpuBackend::CUDA) {
            return internal::undistort_image_tensor(src, params, true);
        }
        const GpuBackendScope backend_scope(GpuBackend::CUDA);

        assert(src.is_valid());
        assert(src.ndim() == 2);
        assert(src.device() == Device::GPU);
        assert(static_cast<int>(src.shape()[0]) == params.src_height);
        assert(static_cast<int>(src.shape()[1]) == params.src_width);

        nvtxRangePush("undistort_mask");

        auto dst = Tensor::zeros(
            {static_cast<size_t>(params.dst_height),
             static_cast<size_t>(params.dst_width)},
            Device::GPU);

        const dim3 block(BLOCK_DIM, BLOCK_DIM);
        const dim3 grid(
            (params.dst_width + BLOCK_DIM - 1) / BLOCK_DIM,
            (params.dst_height + BLOCK_DIM - 1) / BLOCK_DIM);

        undistort_mask_kernel<<<grid, block, 0, stream>>>(
            src.ptr<float>(), dst.ptr<float>(), params);

        const cudaError_t err = cudaGetLastError();
        assert(err == cudaSuccess && "undistort_mask_kernel launch failed");

        nvtxRangePop();
        return dst;
    }

} // namespace lfs::core
