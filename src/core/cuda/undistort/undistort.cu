/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "undistort.hpp"

#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nvtx3/nvToolsExt.h>

namespace lfs::core {

    namespace {

        constexpr int BLOCK_DIM = 16;

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

        __device__ float bilinear_sample(
            const float* __restrict__ src,
            const int width, const int height, const int stride,
            const float sx, const float sy) {

            const float x0f = floorf(sx);
            const float y0f = floorf(sy);
            const int x0 = static_cast<int>(x0f);
            const int y0 = static_cast<int>(y0f);
            const int x1 = x0 + 1;
            const int y1 = y0 + 1;

            if (x0 < 0 || y0 < 0 || x1 >= width || y1 >= height)
                return 0.0f;

            const float fx = sx - x0f;
            const float fy = sy - y0f;

            const float v00 = src[y0 * stride + x0];
            const float v01 = src[y0 * stride + x1];
            const float v10 = src[y1 * stride + x0];
            const float v11 = src[y1 * stride + x1];

            return (1.0f - fy) * ((1.0f - fx) * v00 + fx * v01) +
                   fy * ((1.0f - fx) * v10 + fx * v11);
        }

        __global__ void __launch_bounds__(BLOCK_DIM* BLOCK_DIM)
            undistort_image_kernel(
                const float* __restrict__ src,
                float* __restrict__ dst,
                const int channels,
                const UndistortParams params) {

            const int ox = blockIdx.x * BLOCK_DIM + threadIdx.x;
            const int oy = blockIdx.y * BLOCK_DIM + threadIdx.y;

            if (ox >= params.dst_width || oy >= params.dst_height)
                return;

            const float nx = (static_cast<float>(ox) - params.dst_cx) / params.dst_fx;
            const float ny = (static_cast<float>(oy) - params.dst_cy) / params.dst_fy;

            float dnx, dny;
            apply_distortion(nx, ny, params.model_type, params.distortion, params.num_distortion, dnx, dny);

            const float sx = dnx * params.src_fx + params.src_cx;
            const float sy = dny * params.src_fy + params.src_cy;

            const int src_plane = params.src_height * params.src_width;
            const int dst_plane = params.dst_height * params.dst_width;

            for (int c = 0; c < channels; ++c) {
                dst[c * dst_plane + oy * params.dst_width + ox] =
                    bilinear_sample(src + c * src_plane, params.src_width, params.src_height, params.src_width, sx, sy);
            }
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

            const float nx = (static_cast<float>(ox) - params.dst_cx) / params.dst_fx;
            const float ny = (static_cast<float>(oy) - params.dst_cy) / params.dst_fy;

            float dnx, dny;
            apply_distortion(nx, ny, params.model_type, params.distortion, params.num_distortion, dnx, dny);

            const float sx = dnx * params.src_fx + params.src_cx;
            const float sy = dny * params.src_fy + params.src_cy;

            dst[oy * params.dst_width + ox] =
                bilinear_sample(src, params.src_width, params.src_height, params.src_width, sx, sy);
        }

        void apply_distortion_cpu(
            const float x, const float y,
            const CameraModelType model,
            const float* dist, const int num_dist,
            float& dx, float& dy) {

            switch (model) {
            case CameraModelType::PINHOLE: {
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
                break;
            }
            case CameraModelType::FISHEYE: {
                const float r = std::sqrt(x * x + y * y);
                if (r < 1e-8f) {
                    dx = x;
                    dy = y;
                    return;
                }
                const float theta = std::atan(r);
                const float theta2 = theta * theta;
                const float k1 = num_dist > 0 ? dist[0] : 0.0f;
                const float k2 = num_dist > 1 ? dist[1] : 0.0f;
                const float k3 = num_dist > 2 ? dist[2] : 0.0f;
                const float k4 = num_dist > 3 ? dist[3] : 0.0f;
                const float theta_d = theta * (1.0f + k1 * theta2 + k2 * theta2 * theta2 +
                                               k3 * theta2 * theta2 * theta2 + k4 * theta2 * theta2 * theta2 * theta2);
                const float scale = theta_d / r;
                dx = x * scale;
                dy = y * scale;
                break;
            }
            case CameraModelType::THIN_PRISM_FISHEYE: {
                const float r = std::sqrt(x * x + y * y);
                if (r < 1e-8f) {
                    dx = x;
                    dy = y;
                    return;
                }
                const float theta = std::atan(r);
                const float theta2 = theta * theta;
                const float k1 = num_dist > 0 ? dist[0] : 0.0f;
                const float k2 = num_dist > 1 ? dist[1] : 0.0f;
                const float k3 = num_dist > 2 ? dist[2] : 0.0f;
                const float k4 = num_dist > 3 ? dist[3] : 0.0f;
                const float theta_d = theta * (1.0f + k1 * theta2 + k2 * theta2 * theta2 +
                                               k3 * theta2 * theta2 * theta2 + k4 * theta2 * theta2 * theta2 * theta2);
                const float scale = theta_d / r;
                float xd = x * scale;
                float yd = y * scale;
                const float p1 = num_dist > 4 ? dist[4] : 0.0f;
                const float p2 = num_dist > 5 ? dist[5] : 0.0f;
                const float r2d = xd * xd + yd * yd;
                xd += 2.0f * p1 * xd * yd + p2 * (r2d + 2.0f * xd * xd);
                yd += p1 * (r2d + 2.0f * yd * yd) + 2.0f * p2 * xd * yd;
                const float s1 = num_dist > 6 ? dist[6] : 0.0f;
                const float s2 = num_dist > 7 ? dist[7] : 0.0f;
                const float s3 = num_dist > 8 ? dist[8] : 0.0f;
                const float s4 = num_dist > 9 ? dist[9] : 0.0f;
                const float r4d = r2d * r2d;
                xd += s1 * r2d + s2 * r4d;
                yd += s3 * r2d + s4 * r4d;
                dx = xd;
                dy = yd;
                break;
            }
            default:
                dx = x;
                dy = y;
                break;
            }
        }

    } // anonymous namespace

    UndistortParams compute_undistort_params(
        float fx, float fy, float cx, float cy,
        int width, int height,
        const Tensor& radial, const Tensor& tangential,
        CameraModelType model, float blank_pixels) {

        UndistortParams params{};
        params.src_fx = fx;
        params.src_fy = fy;
        params.src_cx = cx;
        params.src_cy = cy;
        params.src_width = width;
        params.src_height = height;
        params.model_type = model;

        // Coefficient layout per model:
        // PINHOLE:            [k1, k2, k3, p1, p2]               indices 0-4
        // FISHEYE:            [k1, k2, k3, k4]                   indices 0-3
        // THIN_PRISM_FISHEYE: [k1, k2, k3, k4, p1, p2, s1..s4]  indices 0-9
        std::memset(params.distortion, 0, sizeof(params.distortion));
        params.num_distortion = 0;

        std::vector<float> rad_vec, tan_vec;
        if (radial.is_valid() && radial.numel() > 0) {
            auto rad_cpu = radial.cpu();
            auto rad_acc = rad_cpu.accessor<float, 1>();
            for (size_t i = 0; i < rad_cpu.numel(); ++i)
                rad_vec.push_back(rad_acc(i));
        }
        if (tangential.is_valid() && tangential.numel() > 0) {
            auto tan_cpu = tangential.cpu();
            auto tan_acc = tan_cpu.accessor<float, 1>();
            for (size_t i = 0; i < tan_cpu.numel(); ++i)
                tan_vec.push_back(tan_acc(i));
        }

        const auto place = [&](int idx, float val) {
            assert(idx < 12);
            params.distortion[idx] = val;
            params.num_distortion = std::max(params.num_distortion, idx + 1);
        };

        switch (model) {
        case CameraModelType::PINHOLE:
            for (size_t i = 0; i < rad_vec.size() && i < 3; ++i)
                place(static_cast<int>(i), rad_vec[i]);
            for (size_t i = 0; i < tan_vec.size() && i < 2; ++i)
                place(3 + static_cast<int>(i), tan_vec[i]);
            break;

        case CameraModelType::FISHEYE:
            for (size_t i = 0; i < rad_vec.size() && i < 4; ++i)
                place(static_cast<int>(i), rad_vec[i]);
            break;

        case CameraModelType::THIN_PRISM_FISHEYE:
            for (size_t i = 0; i < rad_vec.size() && i < 4; ++i)
                place(static_cast<int>(i), rad_vec[i]);
            for (size_t i = 0; i < tan_vec.size() && i < 6; ++i)
                place(4 + static_cast<int>(i), tan_vec[i]);
            break;

        default:
            break;
        }

        float min_x = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float min_y = std::numeric_limits<float>::max();
        float max_y = std::numeric_limits<float>::lowest();

        const auto trace_pixel = [&](const float px, const float py) {
            const float nx = (px - cx) / fx;
            const float ny = (py - cy) / fy;
            float dnx, dny;
            apply_distortion_cpu(nx, ny, model, params.distortion, params.num_distortion, dnx, dny);
            min_x = std::min(min_x, dnx);
            max_x = std::max(max_x, dnx);
            min_y = std::min(min_y, dny);
            max_y = std::max(max_y, dny);
        };

        for (int x = 0; x < width; ++x) {
            trace_pixel(static_cast<float>(x), 0.0f);
            trace_pixel(static_cast<float>(x), static_cast<float>(height - 1));
        }
        for (int y = 0; y < height; ++y) {
            trace_pixel(0.0f, static_cast<float>(y));
            trace_pixel(static_cast<float>(width - 1), static_cast<float>(y));
        }

        const float scale_x = 1.0f / (1.0f + blank_pixels * (max_x - min_x));
        const float scale_y = 1.0f / (1.0f + blank_pixels * (max_y - min_y));
        const float scale = std::min(scale_x, scale_y);

        params.dst_fx = fx * scale;
        params.dst_fy = fy * scale;
        params.dst_width = std::clamp(
            static_cast<int>(std::round((max_x - min_x) * params.dst_fx)), 1, width * 2);
        params.dst_height = std::clamp(
            static_cast<int>(std::round((max_y - min_y) * params.dst_fy)), 1, height * 2);
        params.dst_cx = static_cast<float>(params.dst_width) * 0.5f;
        params.dst_cy = static_cast<float>(params.dst_height) * 0.5f;

        LOG_INFO("Undistort: %dx%d -> %dx%d, fx=%.1f->%.1f, fy=%.1f->%.1f",
                 width, height, params.dst_width, params.dst_height,
                 fx, params.dst_fx, fy, params.dst_fy);

        return params;
    }

    UndistortParams scale_undistort_params(
        const UndistortParams& params, const int actual_src_width, const int actual_src_height) {

        if (actual_src_width == params.src_width && actual_src_height == params.src_height)
            return params;

        assert(actual_src_width > 0 && actual_src_height > 0);

        const float sx = static_cast<float>(actual_src_width) / static_cast<float>(params.src_width);
        const float sy = static_cast<float>(actual_src_height) / static_cast<float>(params.src_height);

        UndistortParams scaled = params;
        scaled.src_fx = params.src_fx * sx;
        scaled.src_fy = params.src_fy * sy;
        scaled.src_cx = params.src_cx * sx;
        scaled.src_cy = params.src_cy * sy;
        scaled.src_width = actual_src_width;
        scaled.src_height = actual_src_height;

        scaled.dst_fx = params.dst_fx * sx;
        scaled.dst_fy = params.dst_fy * sy;
        scaled.dst_width = std::max(1, static_cast<int>(std::lroundf(params.dst_width * sx)));
        scaled.dst_height = std::max(1, static_cast<int>(std::lroundf(params.dst_height * sy)));
        scaled.dst_cx = scaled.dst_width * 0.5f;
        scaled.dst_cy = scaled.dst_height * 0.5f;

        return scaled;
    }

    Tensor undistort_image(const Tensor& src, const UndistortParams& params, cudaStream_t stream) {
        assert(src.is_valid());
        assert(src.ndim() == 3);
        assert(src.device() == Device::CUDA);

        const int channels = static_cast<int>(src.shape()[0]);
        assert(static_cast<int>(src.shape()[1]) == params.src_height);
        assert(static_cast<int>(src.shape()[2]) == params.src_width);

        nvtxRangePush("undistort_image");

        auto dst = Tensor::zeros(
            {static_cast<size_t>(channels),
             static_cast<size_t>(params.dst_height),
             static_cast<size_t>(params.dst_width)},
            Device::CUDA);

        const dim3 block(BLOCK_DIM, BLOCK_DIM);
        const dim3 grid(
            (params.dst_width + BLOCK_DIM - 1) / BLOCK_DIM,
            (params.dst_height + BLOCK_DIM - 1) / BLOCK_DIM);

        undistort_image_kernel<<<grid, block, 0, stream>>>(
            src.ptr<float>(), dst.ptr<float>(), channels, params);

        const cudaError_t err = cudaGetLastError();
        assert(err == cudaSuccess && "undistort_image_kernel launch failed");

        nvtxRangePop();
        return dst;
    }

    Tensor undistort_mask(const Tensor& src, const UndistortParams& params, cudaStream_t stream) {
        assert(src.is_valid());
        assert(src.ndim() == 2);
        assert(src.device() == Device::CUDA);
        assert(static_cast<int>(src.shape()[0]) == params.src_height);
        assert(static_cast<int>(src.shape()[1]) == params.src_width);

        nvtxRangePush("undistort_mask");

        auto dst = Tensor::zeros(
            {static_cast<size_t>(params.dst_height),
             static_cast<size_t>(params.dst_width)},
            Device::CUDA);

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
