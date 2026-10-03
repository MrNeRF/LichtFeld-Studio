/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Host-side undistortion parameters: the COLMAP-style undistorted camera for a
// distorted one, and its rescaling to the image size actually loaded. Every
// build needs these; only the CUDA resampling kernels live in undistort.cu.

#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "core/tensor_image.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lfs::core {

    namespace {

        constexpr float PIXEL_CENTER_OFFSET = 0.5f;
        constexpr float NEWTON_EPSILON = 1e-6f;
        constexpr float MAX_FISHEYE_THETA = 1.56079632679f;
        constexpr int MAX_NEWTON_ITERATIONS = 20;
        constexpr int THIN_PRISM_SEED_ITERATIONS = 5;
        constexpr float INVERSE_RESIDUAL_PIXELS = 5.0e-4f;
        // Float32 Newton stalls near 6e-4 px at 8k image scales; accepting up to 1e-2 px keeps
        // those pixels valid while the geometric error stays far below sampling resolution.
        constexpr float INVERSE_ACCEPT_PIXELS = 1.0e-2f;
        constexpr float INVERSE_JACOBIAN_STEP = 1.0e-4f;
        constexpr float INVERSE_MAX_STEP = 2.0f;
        constexpr float COLMAP_MIN_SCALE = 0.2f;
        constexpr float COLMAP_MAX_SCALE = 2.0f;
        constexpr int AREA_QUADRATURE = 8;
        constexpr int LANCZOS_RADIUS = 3;
        constexpr int OUTPUT_TILE_ROWS = 256;
        constexpr float MIN_SIGNED_WEIGHT_RATIO = 1.0e-4f;
        constexpr float EVALUATION_MIN_COVERAGE = 0.999f;

        // COLMAP sensor/models.h (BSD-3 licensed formulas)
        void apply_distortion_pinhole(
            const float x, const float y,
            const float* __restrict__ dist, const int num_dist,
            float& dx, float& dy) {

            const float r2 = x * x + y * y;
            const float r4 = r2 * r2;
            const float r6 = r4 * r2;

            const float k1 = num_dist > 0 ? dist[0] : 0.0f;
            const float k2 = num_dist > 1 ? dist[1] : 0.0f;
            const float k3 = num_dist > 2 ? dist[2] : 0.0f;
            const float numerator = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;
            float radial = numerator;
            if (num_dist >= 6) {
                const float denominator =
                    1.0f + dist[3] * r2 + dist[4] * r4 + dist[5] * r6;
                radial = numerator / denominator;
            }

            const int tangential_offset = num_dist >= 6 ? 6 : 3;
            const float p1 = num_dist > tangential_offset ? dist[tangential_offset] : 0.0f;
            const float p2 = num_dist > tangential_offset + 1 ? dist[tangential_offset + 1] : 0.0f;

            dx = x * radial + 2.0f * p1 * x * y + p2 * (r2 + 2.0f * x * x);
            dy = y * radial + p1 * (r2 + 2.0f * y * y) + 2.0f * p2 * x * y;
        }

        void thin_prism_increment(
            const float uu, const float vv,
            const float* __restrict__ dist, const int num_dist,
            float& tx, float& ty) {
            const float p1 = num_dist > 4 ? dist[4] : 0.0f;
            const float p2 = num_dist > 5 ? dist[5] : 0.0f;
            const float sx1 = num_dist > 6 ? dist[6] : 0.0f;
            const float sx2 = num_dist > 7 ? dist[7] : 0.0f;
            const float sy1 = num_dist > 8 ? dist[8] : 0.0f;
            const float sy2 = num_dist > 9 ? dist[9] : 0.0f;
            const float u2 = uu * uu;
            const float uv = uu * vv;
            const float v2 = vv * vv;
            const float r2 = u2 + v2;
            const float r4 = r2 * r2;
            tx = 2.0f * p1 * uv + p2 * (r2 + 2.0f * u2) + sx1 * r2 + sx2 * r4;
            ty = 2.0f * p2 * uv + p1 * (r2 + 2.0f * v2) + sy1 * r2 + sy2 * r4;
        }

        void thin_prism_fisheye_from_theta_point(
            const float uu, const float vv,
            const float* __restrict__ dist, const int num_dist,
            float& dx, float& dy) {
            const float k1 = num_dist > 0 ? dist[0] : 0.0f;
            const float k2 = num_dist > 1 ? dist[1] : 0.0f;
            const float k3 = num_dist > 2 ? dist[2] : 0.0f;
            const float k4 = num_dist > 3 ? dist[3] : 0.0f;
            const float r2 = uu * uu + vv * vv;
            const float r4 = r2 * r2;
            const float r6 = r4 * r2;
            const float r8 = r6 * r2;
            const float radial = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
            float tx, ty;
            thin_prism_increment(uu, vv, dist, num_dist, tx, ty);
            dx = uu + uu * radial + tx;
            dy = vv + vv * radial + ty;
        }

        bool solve_fisheye_theta(
            const float theta_d, const float* __restrict__ dist, float& theta) {
            theta = fminf(theta_d, MAX_FISHEYE_THETA);
            for (int iter = 0; iter < MAX_NEWTON_ITERATIONS; ++iter) {
                const float theta2 = theta * theta;
                const float theta4 = theta2 * theta2;
                const float theta6 = theta4 * theta2;
                const float theta8 = theta4 * theta4;
                const float residual = theta * (1.0f + dist[0] * theta2 + dist[1] * theta4 +
                                                dist[2] * theta6 + dist[3] * theta8) -
                                       theta_d;
                const float slope = 1.0f + 3.0f * dist[0] * theta2 + 5.0f * dist[1] * theta4 +
                                    7.0f * dist[2] * theta6 + 9.0f * dist[3] * theta8;
                if (!std::isfinite(residual) || !std::isfinite(slope) || fabsf(slope) < NEWTON_EPSILON)
                    return false;
                const float step = residual / slope;
                theta -= step;
                if (fabsf(step) < 1e-7f)
                    break;
            }
            return theta > 0.0f && theta < MAX_FISHEYE_THETA;
        }

        bool inverse_fisheye_radial(
            const float xd, const float yd, const float* __restrict__ dist,
            float& ux, float& uy) {
            const float theta_d = sqrtf(xd * xd + yd * yd);
            if (theta_d < 1e-8f) {
                ux = xd;
                uy = yd;
                return true;
            }
            float theta;
            if (!solve_fisheye_theta(theta_d, dist, theta))
                return false;
            const float scale = tanf(theta) / theta_d;
            ux = xd * scale;
            uy = yd * scale;
            return true;
        }

        bool inverse_thin_prism_seed(
            const float xd, const float yd, const float* __restrict__ dist, const int num_dist,
            float& ux, float& uy) {
            float wx = xd;
            float wy = yd;
            float theta = 0.0f;
            for (int iter = 0; iter < THIN_PRISM_SEED_ITERATIONS; ++iter) {
                float tx, ty;
                thin_prism_increment(wx, wy, dist, num_dist, tx, ty);
                const float zx = xd - tx;
                const float zy = yd - ty;
                const float theta_d = hypotf(zx, zy);
                if (theta_d < 1e-8f) {
                    wx = zx;
                    wy = zy;
                    theta = theta_d;
                    continue;
                }
                if (!solve_fisheye_theta(theta_d, dist, theta))
                    return false;
                wx = zx * theta / theta_d;
                wy = zy * theta / theta_d;
            }
            const float scale = theta > 1e-8f ? tanf(theta) / theta : 1.0f;
            ux = wx * scale;
            uy = wy * scale;
            return true;
        }

        bool seed_inverse_distortion(
            const float xd, const float yd, const CameraModelType model,
            const float* __restrict__ dist, const int num_dist,
            float& ux, float& uy) {
            if (model == CameraModelType::THIN_PRISM_FISHEYE && num_dist > 4)
                return inverse_thin_prism_seed(xd, yd, dist, num_dist, ux, uy);
            if (model == CameraModelType::FISHEYE || model == CameraModelType::THIN_PRISM_FISHEYE)
                return inverse_fisheye_radial(xd, yd, dist, ux, uy);
            ux = xd;
            uy = yd;
            return true;
        }

        void apply_distortion_cpu(
            const float x, const float y,
            const CameraModelType model,
            const float* dist, const int num_dist,
            float& dx, float& dy) {

            switch (model) {
            case CameraModelType::PINHOLE: {
                apply_distortion_pinhole(x, y, dist, num_dist, dx, dy);
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
                const float theta_over_r = std::atan(r) / r;
                thin_prism_fisheye_from_theta_point(
                    x * theta_over_r, y * theta_over_r, dist, num_dist, dx, dy);
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
                const bool rational = num_dist >= 6;
                const float k4 = rational ? dist[3] : 0.0f;
                const float k5 = rational ? dist[4] : 0.0f;
                const float k6 = rational ? dist[5] : 0.0f;
                const int tangential_offset = rational ? 6 : 3;
                const float p1 = num_dist > tangential_offset ? dist[tangential_offset] : 0.0f;
                const float p2 = num_dist > tangential_offset + 1 ? dist[tangential_offset + 1] : 0.0f;

                const float numerator = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;
                const float d_numerator_dr2 = k1 + 2.0f * k2 * r2 + 3.0f * k3 * r4;
                float radial = numerator;
                float d_radial_dr2 = d_numerator_dr2;
                if (rational) {
                    const float denominator = 1.0f + k4 * r2 + k5 * r4 + k6 * r6;
                    if (!std::isfinite(denominator) ||
                        std::fabs(denominator) < NEWTON_EPSILON) {
                        return false;
                    }
                    const float d_denominator_dr2 =
                        k4 + 2.0f * k5 * r2 + 3.0f * k6 * r4;
                    radial = numerator / denominator;
                    d_radial_dr2 =
                        (d_numerator_dr2 * denominator -
                         numerator * d_denominator_dr2) /
                        (denominator * denominator);
                }
                if (!std::isfinite(radial) || !std::isfinite(d_radial_dr2)) {
                    return false;
                }
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
            if (!seed_inverse_distortion(xd, yd, model, dist, num_dist, ux, uy))
                return false;

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

    } // namespace

    void distort_normalized_point(
        const UndistortParams& params,
        const float x,
        const float y,
        float& distorted_x,
        float& distorted_y) {
        apply_distortion_cpu(
            x, y, params.model_type, params.distortion, params.num_distortion,
            distorted_x, distorted_y);
    }

    bool undistort_image_point(
        const UndistortParams& params,
        const float image_x,
        const float image_y,
        float& normalized_x,
        float& normalized_y) {
        return cam_from_img_cpu(
            image_x, image_y,
            params.src_fx, params.src_fy, params.src_cx, params.src_cy,
            params.model_type, params.distortion, params.num_distortion,
            normalized_x, normalized_y);
    }

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
        // PINHOLE polynomial: [k1, k2, k3, p1, p2]               indices 0-4
        // PINHOLE rational:   [k1, k2, k3, k4, k5, k6, p1, p2]  indices 0-7
        // FISHEYE:            [k1, k2, k3, k4]                   indices 0-3
        // THIN_PRISM_FISHEYE: [k1, k2, k3, k4, p1, p2, sx1, sx2, sy1, sy2]  indices 0-9
        std::memset(params.distortion, 0, sizeof(params.distortion));
        params.num_distortion = 0;

        std::vector<float> rad_vec, tan_vec;
        if (radial.is_valid() && radial.numel() > 0) {
            assert(radial.ndim() == 1);
            auto rad_cpu = radial.cpu();
            auto rad_acc = rad_cpu.accessor<float, 1>();
            for (size_t i = 0; i < rad_cpu.numel(); ++i)
                rad_vec.push_back(rad_acc(i));
        }
        if (tangential.is_valid() && tangential.numel() > 0) {
            assert(tangential.ndim() == 1);
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
        case CameraModelType::PINHOLE: {
            if (rad_vec.size() > 3 && rad_vec.size() != 6) {
                throw std::invalid_argument(
                    "Pinhole distortion requires at most three polynomial radial coefficients or six rational radial coefficients");
            }
            if (!tan_vec.empty() && tan_vec.size() != 2) {
                throw std::invalid_argument(
                    "Pinhole distortion supports exactly two tangential coefficients");
            }
            const bool rational =
                rad_vec.size() == 6 &&
                (rad_vec[3] != 0.0f || rad_vec[4] != 0.0f || rad_vec[5] != 0.0f);
            const size_t radial_count = rational ? rad_vec.size() : std::min<size_t>(rad_vec.size(), 3);
            for (size_t i = 0; i < radial_count; ++i)
                place(static_cast<int>(i), rad_vec[i]);
            const int tangential_offset = rational ? 6 : 3;
            for (size_t i = 0; i < tan_vec.size(); ++i)
                place(tangential_offset + static_cast<int>(i), tan_vec[i]);
            break;
        }

        case CameraModelType::FISHEYE:
            if (rad_vec.size() > 4) {
                throw std::invalid_argument(
                    "Fisheye distortion supports at most four radial coefficients");
            }
            if (std::any_of(tan_vec.begin(), tan_vec.end(), [](const float value) {
                    return value != 0.0f;
                })) {
                throw std::invalid_argument(
                    "Fisheye distortion does not support tangential coefficients");
            }
            for (size_t i = 0; i < rad_vec.size(); ++i)
                place(static_cast<int>(i), rad_vec[i]);
            break;

        case CameraModelType::THIN_PRISM_FISHEYE: {
            // Tangential tuple in COLMAP order {p1, p2, sx1, sy1}, optionally extended by {sx2, sy2}.
            constexpr std::array<int, 6> thin_prism_slots = {4, 5, 6, 8, 7, 9};
            if (rad_vec.size() > 4 || tan_vec.size() > thin_prism_slots.size()) {
                throw std::invalid_argument(
                    "Thin prism fisheye distortion supports four radial and six tangential or prism coefficients");
            }
            for (size_t i = 0; i < rad_vec.size(); ++i)
                place(static_cast<int>(i), rad_vec[i]);
            for (size_t i = 0; i < tan_vec.size(); ++i)
                place(thin_prism_slots[i], tan_vec[i]);
            break;
        }

        default:
            if (!rad_vec.empty() || !tan_vec.empty()) {
                throw std::invalid_argument(
                    "Distortion coefficients are unsupported by this camera model");
            }
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
            if (!undistort_image_point(params, px, py, ux, uy)) {
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
            params.crop_solve_failed = true;
            LOG_DEBUG("Undistort crop solve found no valid border samples, keeping original intrinsics");
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

    UndistortGrid compute_undistort_grid(
        const UndistortParams& params, const int resize_factor, const int max_width) {
        assert(params.dst_width > 0 && params.dst_height > 0);
        const float resize_scale = 1.0f / static_cast<float>(std::max(1, resize_factor));
        const int largest_crop_dimension = std::max(params.dst_width, params.dst_height);
        const float width_scale = max_width > 0
                                      ? static_cast<float>(max_width) /
                                            static_cast<float>(largest_crop_dimension)
                                      : 1.0f;
        const float scale = std::min({1.0f, resize_scale, width_scale});
        const int width = std::max(
            1, static_cast<int>(std::lround(static_cast<double>(params.dst_width) * scale)));
        const int height = std::max(
            1, static_cast<int>(std::lround(static_cast<double>(params.dst_height) * scale)));
        return {
            .width = width,
            .height = height,
            .scale_x = static_cast<float>(width) / static_cast<float>(params.dst_width),
            .scale_y = static_cast<float>(height) / static_cast<float>(params.dst_height)};
    }

    UndistortParams prepare_undistort_params(
        const UndistortParams& params,
        const int actual_src_width,
        const int actual_src_height,
        const int resize_factor,
        const int max_width) {
        const auto grid = compute_undistort_grid(params, resize_factor, max_width);
        assert(actual_src_width > 0 && actual_src_height > 0);
        assert(grid.width > 0 && grid.height > 0);
        UndistortParams scaled = params;
        const float src_scale_x = static_cast<float>(actual_src_width) /
                                  static_cast<float>(params.src_width);
        const float src_scale_y = static_cast<float>(actual_src_height) /
                                  static_cast<float>(params.src_height);
        scaled.src_fx = params.src_fx * src_scale_x;
        scaled.src_fy = params.src_fy * src_scale_y;
        scaled.src_cx = params.src_cx * src_scale_x;
        scaled.src_cy = params.src_cy * src_scale_y;
        scaled.src_width = actual_src_width;
        scaled.src_height = actual_src_height;
        scaled.dst_fx = params.dst_fx * grid.scale_x;
        scaled.dst_fy = params.dst_fy * grid.scale_y;
        scaled.dst_cx = params.dst_cx * grid.scale_x;
        scaled.dst_cy = params.dst_cy * grid.scale_y;
        scaled.dst_width = grid.width;
        scaled.dst_height = grid.height;
        return scaled;
    }

    UndistortParams scale_undistort_params(
        const UndistortParams& params, const int actual_src_width, const int actual_src_height,
        const int max_width) {
        UndistortParams scaled = params;

        if (actual_src_width != params.src_width || actual_src_height != params.src_height) {
            assert(actual_src_width > 0 && actual_src_height > 0);

            const float sx = static_cast<float>(actual_src_width) / static_cast<float>(params.src_width);
            const float sy = static_cast<float>(actual_src_height) / static_cast<float>(params.src_height);

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
        }

        if (max_width > 0 && (scaled.dst_width > max_width || scaled.dst_height > max_width)) {
            const int old_width = scaled.dst_width;
            const int old_height = scaled.dst_height;
            int new_width;
            int new_height;
            if (old_width > old_height) {
                new_width = max_width;
                new_height = std::max(
                    1, static_cast<int>(static_cast<std::int64_t>(old_height) * max_width / old_width));
            } else {
                new_height = max_width;
                new_width = std::max(
                    1, static_cast<int>(static_cast<std::int64_t>(old_width) * max_width / old_height));
            }

            const float dst_sx = static_cast<float>(new_width) / static_cast<float>(old_width);
            const float dst_sy = static_cast<float>(new_height) / static_cast<float>(old_height);
            scaled.dst_fx *= dst_sx;
            scaled.dst_fy *= dst_sy;
            scaled.dst_cx *= dst_sx;
            scaled.dst_cy *= dst_sy;
            scaled.dst_width = new_width;
            scaled.dst_height = new_height;
        }

        return scaled;
    }

} // namespace lfs::core
