/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "undistort.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>
#include <nvtx3/nvToolsExt.h>

namespace lfs::core {

    namespace {

        constexpr int BLOCK_DIM = 16;
        constexpr float PIXEL_CENTER_OFFSET = 0.5f;
        constexpr float NEWTON_EPSILON = 1e-6f;
        constexpr float MAX_FISHEYE_THETA = 1.56079632679f;
        constexpr int MAX_NEWTON_ITERATIONS = 20;
        constexpr float COLMAP_MIN_SCALE = 0.2f;
        constexpr float COLMAP_MAX_SCALE = 2.0f;

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

            const auto get_pixel_constant_border = [&](const int y, const int x) {
                if (x >= 0 && y >= 0 && x < width && y < height) {
                    return src[y * stride + x];
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

            const float pixel_x = static_cast<float>(ox) + PIXEL_CENTER_OFFSET;
            const float pixel_y = static_cast<float>(oy) + PIXEL_CENTER_OFFSET;
            const float nx = (pixel_x - params.dst_cx) / params.dst_fx;
            const float ny = (pixel_y - params.dst_cy) / params.dst_fy;

            float dnx, dny;
            apply_distortion(nx, ny, params.model_type, params.distortion, params.num_distortion, dnx, dny);

            const float sx = dnx * params.src_fx + params.src_cx - PIXEL_CENTER_OFFSET;
            const float sy = dny * params.src_fy + params.src_cy - PIXEL_CENTER_OFFSET;

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

        bool cam_from_img_pinhole_cpu(
            const float img_x, const float img_y,
            const float fx, const float fy,
            const float cx, const float cy,
            const float* dist, const int num_dist,
            float& ux, float& uy) {

            if (num_dist <= 0) {
                ux = (img_x - cx) / fx;
                uy = (img_y - cy) / fy;
                return true;
            }

            const float xd = (img_x - cx) / fx;
            const float yd = (img_y - cy) / fy;
            ux = xd;
            uy = yd;

            for (int iter = 0; iter < MAX_NEWTON_ITERATIONS; ++iter) {
                const float r2 = ux * ux + uy * uy;
                const float r4 = r2 * r2;
                const float r6 = r4 * r2;

                const float k1 = dist[0];
                const float k2 = num_dist > 1 ? dist[1] : 0.0f;
                const float k3 = num_dist > 2 ? dist[2] : 0.0f;
                const float p1 = num_dist > 3 ? dist[3] : 0.0f;
                const float p2 = num_dist > 4 ? dist[4] : 0.0f;

                const float radial = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;
                const float d_radial_dr2 = k1 + 2.0f * k2 * r2 + 3.0f * k3 * r4;
                const float d_radial_dx = 2.0f * ux * d_radial_dr2;
                const float d_radial_dy = 2.0f * uy * d_radial_dr2;

                const float fx_residual =
                    ux * radial + 2.0f * p1 * ux * uy + p2 * (r2 + 2.0f * ux * ux) - xd;
                const float fy_residual =
                    uy * radial + p1 * (r2 + 2.0f * uy * uy) + 2.0f * p2 * ux * uy - yd;

                const float j00 = radial + ux * d_radial_dx + 2.0f * p1 * uy + 6.0f * p2 * ux;
                const float j01 = ux * d_radial_dy + 2.0f * p1 * ux + 2.0f * p2 * uy;
                const float j10 = uy * d_radial_dx + 2.0f * p2 * uy + 2.0f * p1 * ux;
                const float j11 = radial + uy * d_radial_dy + 6.0f * p1 * uy + 2.0f * p2 * ux;

                const float det = j00 * j11 - j01 * j10;
                if (std::fabs(det) < NEWTON_EPSILON) {
                    return false;
                }

                const float step_x = (j11 * fx_residual - j01 * fy_residual) / det;
                const float step_y = (-j10 * fx_residual + j00 * fy_residual) / det;
                ux -= step_x;
                uy -= step_y;

                if (std::fabs(step_x) < NEWTON_EPSILON && std::fabs(step_y) < NEWTON_EPSILON) {
                    return std::isfinite(ux) && std::isfinite(uy);
                }
            }

            return std::isfinite(ux) && std::isfinite(uy);
        }

        bool cam_from_img_fisheye_cpu(
            const float img_x, const float img_y,
            const float fx, const float fy,
            const float cx, const float cy,
            const float* dist, const int num_dist,
            float& ux, float& uy) {

            const float xd = (img_x - cx) / fx;
            const float yd = (img_y - cy) / fy;
            const float rd = std::sqrt(xd * xd + yd * yd);
            if (rd < NEWTON_EPSILON) {
                ux = 0.0f;
                uy = 0.0f;
                return true;
            }

            float theta = rd;
            for (int iter = 0; iter < MAX_NEWTON_ITERATIONS; ++iter) {
                const float theta2 = theta * theta;
                const float theta4 = theta2 * theta2;
                const float theta6 = theta4 * theta2;
                const float theta8 = theta4 * theta4;

                const float k1 = num_dist > 0 ? dist[0] : 0.0f;
                const float k2 = num_dist > 1 ? dist[1] : 0.0f;
                const float k3 = num_dist > 2 ? dist[2] : 0.0f;
                const float k4 = num_dist > 3 ? dist[3] : 0.0f;

                const float theta_d =
                    theta * (1.0f + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
                const float derivative =
                    1.0f + 3.0f * k1 * theta2 + 5.0f * k2 * theta4 + 7.0f * k3 * theta6 +
                    9.0f * k4 * theta8;
                if (std::fabs(derivative) < NEWTON_EPSILON) {
                    return false;
                }

                const float step = (theta_d - rd) / derivative;
                theta -= step;
                if (std::fabs(step) < NEWTON_EPSILON) {
                    break;
                }
            }

            if (!std::isfinite(theta) || theta < 0.0f || theta >= MAX_FISHEYE_THETA) {
                return false;
            }

            const float r = std::tan(theta);
            const float scale = r / rd;
            ux = xd * scale;
            uy = yd * scale;
            return std::isfinite(ux) && std::isfinite(uy);
        }

        bool cam_from_img_generic_cpu(
            const float img_x, const float img_y,
            const float fx, const float fy,
            const float cx, const float cy,
            const CameraModelType model,
            const float* dist, const int num_dist,
            float& ux, float& uy) {

            const float xd = (img_x - cx) / fx;
            const float yd = (img_y - cy) / fy;
            ux = xd;
            uy = yd;

            for (int iter = 0; iter < MAX_NEWTON_ITERATIONS; ++iter) {
                float fx_eval, fy_eval;
                apply_distortion_cpu(ux, uy, model, dist, num_dist, fx_eval, fy_eval);

                const float residual_x = fx_eval - xd;
                const float residual_y = fy_eval - yd;
                if (std::fabs(residual_x) < NEWTON_EPSILON &&
                    std::fabs(residual_y) < NEWTON_EPSILON) {
                    return std::isfinite(ux) && std::isfinite(uy);
                }

                constexpr float jacobian_step = 1e-4f;
                float fx_dx, fy_dx;
                float fx_dy, fy_dy;
                apply_distortion_cpu(ux + jacobian_step, uy, model, dist, num_dist, fx_dx, fy_dx);
                apply_distortion_cpu(ux, uy + jacobian_step, model, dist, num_dist, fx_dy, fy_dy);

                const float j00 = (fx_dx - fx_eval) / jacobian_step;
                const float j10 = (fy_dx - fy_eval) / jacobian_step;
                const float j01 = (fx_dy - fx_eval) / jacobian_step;
                const float j11 = (fy_dy - fy_eval) / jacobian_step;

                const float det = j00 * j11 - j01 * j10;
                if (std::fabs(det) < NEWTON_EPSILON) {
                    return false;
                }

                const float step_x = (j11 * residual_x - j01 * residual_y) / det;
                const float step_y = (-j10 * residual_x + j00 * residual_y) / det;
                ux -= step_x;
                uy -= step_y;

                if (std::fabs(step_x) < NEWTON_EPSILON && std::fabs(step_y) < NEWTON_EPSILON) {
                    return std::isfinite(ux) && std::isfinite(uy);
                }
            }

            return std::isfinite(ux) && std::isfinite(uy);
        }

        bool cam_from_img_cpu(
            const float img_x, const float img_y,
            const float fx, const float fy,
            const float cx, const float cy,
            const CameraModelType model,
            const float* dist, const int num_dist,
            float& ux, float& uy) {

            switch (model) {
            case CameraModelType::PINHOLE:
                return cam_from_img_pinhole_cpu(img_x, img_y, fx, fy, cx, cy, dist, num_dist, ux, uy);
            case CameraModelType::FISHEYE:
                return cam_from_img_fisheye_cpu(img_x, img_y, fx, fy, cx, cy, dist, num_dist, ux, uy);
            case CameraModelType::THIN_PRISM_FISHEYE:
                return cam_from_img_generic_cpu(img_x, img_y, fx, fy, cx, cy, model, dist, num_dist, ux, uy);
            default:
                ux = (img_x - cx) / fx;
                uy = (img_y - cy) / fy;
                return true;
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

        params.dst_fx = fx;
        params.dst_fy = fy;
        params.dst_cx = cx;
        params.dst_cy = cy;
        params.dst_width = width;
        params.dst_height = height;

        const bool needs_undistorted_crop =
            model != CameraModelType::PINHOLE || params.num_distortion > 0;
        if (!needs_undistorted_crop) {
            return params;
        }

        float left_min_x = std::numeric_limits<float>::max();
        float left_max_x = std::numeric_limits<float>::lowest();
        float right_min_x = std::numeric_limits<float>::max();
        float right_max_x = std::numeric_limits<float>::lowest();
        float top_min_y = std::numeric_limits<float>::max();
        float top_max_y = std::numeric_limits<float>::lowest();
        float bottom_min_y = std::numeric_limits<float>::max();
        float bottom_max_y = std::numeric_limits<float>::lowest();
        bool found_valid_border_point = false;
        bool left_has_valid_sample = false;
        bool right_has_valid_sample = false;
        bool top_has_valid_sample = false;
        bool bottom_has_valid_sample = false;

        const auto trace_pixel = [&](const float px, const float py, float& min_axis, float& max_axis,
                                     const bool trace_x_axis, bool& edge_has_valid_sample) {
            float ux, uy;
            if (!cam_from_img_cpu(
                    px, py, fx, fy, cx, cy, model, params.distortion, params.num_distortion, ux, uy)) {
                return;
            }

            found_valid_border_point = true;
            edge_has_valid_sample = true;
            const float undistorted_x = fx * ux + cx;
            const float undistorted_y = fy * uy + cy;
            const float value = trace_x_axis ? undistorted_x : undistorted_y;
            min_axis = std::min(min_axis, value);
            max_axis = std::max(max_axis, value);
        };

        for (int y = 0; y < height; ++y) {
            const float py = static_cast<float>(y) + PIXEL_CENTER_OFFSET;
            trace_pixel(PIXEL_CENTER_OFFSET, py, left_min_x, left_max_x, true, left_has_valid_sample);
            trace_pixel(static_cast<float>(width) - PIXEL_CENTER_OFFSET, py,
                        right_min_x, right_max_x, true, right_has_valid_sample);
        }
        for (int x = 0; x < width; ++x) {
            const float px = static_cast<float>(x) + PIXEL_CENTER_OFFSET;
            trace_pixel(px, PIXEL_CENTER_OFFSET, top_min_y, top_max_y, false, top_has_valid_sample);
            trace_pixel(px, static_cast<float>(height) - PIXEL_CENTER_OFFSET,
                        bottom_min_y, bottom_max_y, false, bottom_has_valid_sample);
        }

        if (!found_valid_border_point) {
            LOG_WARN("Undistort crop solve found no valid border samples, keeping original intrinsics");
            return params;
        }

        const auto accumulate_scale_candidate = [&](const float numerator, const float denominator,
                                                    const bool take_min, float& result, bool& has_result) {
            if (!std::isfinite(numerator) || !std::isfinite(denominator) ||
                std::fabs(denominator) <= NEWTON_EPSILON) {
                return;
            }

            const float candidate = numerator / denominator;
            if (!std::isfinite(candidate) || candidate <= NEWTON_EPSILON) {
                return;
            }

            if (!has_result) {
                result = candidate;
                has_result = true;
            } else if (take_min) {
                result = std::min(result, candidate);
            } else {
                result = std::max(result, candidate);
            }
        };

        const auto resolve_axis_scale = [&](const char* axis_name,
                                            const float min_scale_candidate, const bool has_min_scale_candidate,
                                            const float max_scale_candidate, const bool has_max_scale_candidate) {
            if (!has_min_scale_candidate && !has_max_scale_candidate) {
                LOG_WARN("Undistort crop solve found no valid {}-axis scale candidates, keeping axis scale at 1.0",
                         axis_name);
                return 1.0f;
            }

            const float min_scale = has_min_scale_candidate ? min_scale_candidate : max_scale_candidate;
            const float max_scale = has_max_scale_candidate ? max_scale_candidate : min_scale_candidate;
            if (!has_min_scale_candidate || !has_max_scale_candidate) {
                LOG_WARN("Undistort crop solve found incomplete {}-axis border constraints, reusing available scale candidate",
                         axis_name);
            }

            const float blended_scale = min_scale * blank_pixels + max_scale * (1.0f - blank_pixels);
            if (!std::isfinite(blended_scale) || std::fabs(blended_scale) <= NEWTON_EPSILON) {
                LOG_WARN("Undistort crop solve produced invalid {}-axis blended scale, keeping axis scale at 1.0",
                         axis_name);
                return 1.0f;
            }

            const float scale = 1.0f / blended_scale;
            if (!std::isfinite(scale)) {
                LOG_WARN("Undistort crop solve produced non-finite {}-axis scale, keeping axis scale at 1.0",
                         axis_name);
                return 1.0f;
            }

            return std::clamp(scale, COLMAP_MIN_SCALE, COLMAP_MAX_SCALE);
        };

        float min_scale_x_candidate = 1.0f;
        float max_scale_x_candidate = 1.0f;
        float min_scale_y_candidate = 1.0f;
        float max_scale_y_candidate = 1.0f;
        bool has_min_scale_x_candidate = false;
        bool has_max_scale_x_candidate = false;
        bool has_min_scale_y_candidate = false;
        bool has_max_scale_y_candidate = false;

        if (left_has_valid_sample) {
            accumulate_scale_candidate(cx, cx - left_min_x, true,
                                       min_scale_x_candidate, has_min_scale_x_candidate);
            accumulate_scale_candidate(cx, cx - left_max_x, false,
                                       max_scale_x_candidate, has_max_scale_x_candidate);
        }
        if (right_has_valid_sample) {
            const float right_extent = static_cast<float>(width) - PIXEL_CENTER_OFFSET - cx;
            accumulate_scale_candidate(right_extent, right_max_x - cx, true,
                                       min_scale_x_candidate, has_min_scale_x_candidate);
            accumulate_scale_candidate(right_extent, right_min_x - cx, false,
                                       max_scale_x_candidate, has_max_scale_x_candidate);
        }
        if (top_has_valid_sample) {
            accumulate_scale_candidate(cy, cy - top_min_y, true,
                                       min_scale_y_candidate, has_min_scale_y_candidate);
            accumulate_scale_candidate(cy, cy - top_max_y, false,
                                       max_scale_y_candidate, has_max_scale_y_candidate);
        }
        if (bottom_has_valid_sample) {
            const float bottom_extent = static_cast<float>(height) - PIXEL_CENTER_OFFSET - cy;
            accumulate_scale_candidate(bottom_extent, bottom_max_y - cy, true,
                                       min_scale_y_candidate, has_min_scale_y_candidate);
            accumulate_scale_candidate(bottom_extent, bottom_min_y - cy, false,
                                       max_scale_y_candidate, has_max_scale_y_candidate);
        }

        const float scale_x = resolve_axis_scale(
            "x", min_scale_x_candidate, has_min_scale_x_candidate,
            max_scale_x_candidate, has_max_scale_x_candidate);
        const float scale_y = resolve_axis_scale(
            "y", min_scale_y_candidate, has_min_scale_y_candidate,
            max_scale_y_candidate, has_max_scale_y_candidate);

        params.dst_width = std::max(1, static_cast<int>(scale_x * static_cast<float>(width)));
        params.dst_height = std::max(1, static_cast<int>(scale_y * static_cast<float>(height)));
        params.dst_cx = cx * static_cast<float>(params.dst_width) / static_cast<float>(width);
        params.dst_cy = cy * static_cast<float>(params.dst_height) / static_cast<float>(height);

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
        scaled.dst_cx = params.dst_cx * sx;
        scaled.dst_cy = params.dst_cy * sy;

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
