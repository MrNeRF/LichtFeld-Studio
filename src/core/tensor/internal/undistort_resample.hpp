/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor_image.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
namespace lfs::core::internal::warp_math {
    using std::isfinite;
    using std::max;
    using std::min;
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
    inline void apply_distortion_pinhole(
        const float x, const float y,
        const float* dist, const int num_dist,
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

    inline void apply_distortion_fisheye(
        const float x, const float y,
        const float* dist, const int num_dist,
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

    inline void thin_prism_increment(
        const float uu, const float vv,
        const float* dist, const int num_dist,
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

    inline void thin_prism_fisheye_from_theta_point(
        const float uu, const float vv,
        const float* dist, const int num_dist,
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

    inline bool solve_fisheye_theta(
        const float theta_d, const float* dist, float& theta) {
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
            if (!isfinite(residual) || !isfinite(slope) || fabsf(slope) < NEWTON_EPSILON)
                return false;
            const float step = residual / slope;
            theta -= step;
            if (fabsf(step) < 1e-7f)
                break;
        }
        return theta > 0.0f && theta < MAX_FISHEYE_THETA;
    }

    inline bool inverse_fisheye_radial(
        const float xd, const float yd, const float* dist,
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

    inline bool inverse_thin_prism_seed(
        const float xd, const float yd, const float* dist, const int num_dist,
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

    inline bool seed_inverse_distortion(
        const float xd, const float yd, const CameraModelType model,
        const float* dist, const int num_dist,
        float& ux, float& uy) {
        if (model == CameraModelType::THIN_PRISM_FISHEYE && num_dist > 4)
            return inverse_thin_prism_seed(xd, yd, dist, num_dist, ux, uy);
        if (model == CameraModelType::FISHEYE || model == CameraModelType::THIN_PRISM_FISHEYE)
            return inverse_fisheye_radial(xd, yd, dist, ux, uy);
        ux = xd;
        uy = yd;
        return true;
    }

    inline void apply_distortion_thin_prism_fisheye(
        const float x, const float y,
        const float* dist, const int num_dist,
        float& dx, float& dy) {

        const float r = sqrtf(x * x + y * y);
        if (r < 1e-8f) {
            dx = x;
            dy = y;
            return;
        }

        const float theta_over_r = atanf(r) / r;
        thin_prism_fisheye_from_theta_point(
            x * theta_over_r, y * theta_over_r, dist, num_dist, dx, dy);
    }

    inline void apply_distortion(
        const float x, const float y,
        const CameraModelType model,
        const float* dist, const int num_dist,
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

    inline float bilinear_sample_renormalized(
        const float* src,
        const int width, const int height, const int stride,
        const float sx, const float sy,
        const bool positive_only = false) {
        const int x0 = static_cast<int>(floorf(sx));
        const int y0 = static_cast<int>(floorf(sy));
        const float fx = sx - static_cast<float>(x0);
        const float fy = sy - static_cast<float>(y0);
        float value = 0.0f;
        float weight_sum = 0.0f;
        for (int dy = 0; dy < 2; ++dy) {
            const int y = y0 + dy;
            const float wy = dy == 0 ? 1.0f - fy : fy;
            for (int dx = 0; dx < 2; ++dx) {
                const int x = x0 + dx;
                if (x < 0 || x >= width || y < 0 || y >= height)
                    continue;
                const float sample = src[y * stride + x];
                if (!isfinite(sample) || (positive_only && sample <= 0.0f))
                    continue;
                const float wx = dx == 0 ? 1.0f - fx : fx;
                const float weight = wx * wy;
                value += sample * weight;
                weight_sum += weight;
            }
        }
        if (weight_sum > 1.0e-8f)
            return value / weight_sum;
        if (positive_only)
            return 0.0f;
        const int nearest_x = min(max(static_cast<int>(floorf(sx + 0.5f)), 0), width - 1);
        const int nearest_y = min(max(static_cast<int>(floorf(sy + 0.5f)), 0), height - 1);
        return src[nearest_y * stride + nearest_x];
    }

    inline float lanczos3_weight(const float value) {
        const float absolute_value = fabsf(value);
        if (absolute_value >= static_cast<float>(LANCZOS_RADIUS))
            return 0.0f;
        if (absolute_value < 1.0e-6f)
            return 1.0f;
        // Match CUDA's sinpif zeros and keep near-integer taps accurate: border
        // normalization can amplify float sine argument-reduction error.
        if (absolute_value == floorf(absolute_value))
            return 0.0f;
        constexpr double PI = 3.14159265358979323846;
        const double pi_value = PI * static_cast<double>(value);
        return static_cast<float>(std::sin(pi_value) / pi_value *
                                  (std::sin(pi_value / LANCZOS_RADIUS) /
                                   (pi_value / LANCZOS_RADIUS)));
    }

    inline bool lanczos3_sample(
        const float* src,
        const int width, const int height, const int channels,
        const float sx, const float sy,
        float* values, float& absolute_inside, float& absolute_full) {
        const int base_x = static_cast<int>(floorf(sx));
        const int base_y = static_cast<int>(floorf(sy));
        float weights_x[6];
        float weights_y[6];
        for (int i = 0; i < 6; ++i) {
            weights_x[i] = lanczos3_weight(sx - static_cast<float>(base_x + i - 2));
            weights_y[i] = lanczos3_weight(sy - static_cast<float>(base_y + i - 2));
        }

        float weighted[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float signed_inside = 0.0f;
        absolute_inside = 0.0f;
        absolute_full = 0.0f;
        const int plane = width * height;
        for (int j = 0; j < 6; ++j) {
            const int y = base_y + j - 2;
            for (int i = 0; i < 6; ++i) {
                const int x = base_x + i - 2;
                const float weight = weights_x[i] * weights_y[j];
                absolute_full += fabsf(weight);
                if (x < 0 || x >= width || y < 0 || y >= height)
                    continue;
                signed_inside += weight;
                absolute_inside += fabsf(weight);
                const int index = y * width + x;
                for (int channel = 0; channel < channels; ++channel)
                    weighted[channel] += src[channel * plane + index] * weight;
            }
        }

        if (fabsf(signed_inside) > MIN_SIGNED_WEIGHT_RATIO * absolute_inside) {
            for (int channel = 0; channel < channels; ++channel)
                values[channel] = weighted[channel] / signed_inside;
            return true;
        }

        for (int channel = 0; channel < channels; ++channel) {
            values[channel] = bilinear_sample_renormalized(
                src + channel * plane, width, height, width, sx, sy);
        }
        return false;
    }

    inline void
    undistort_image_kernel(
        const float* src,
        float* dst,
        const int channels,
        const int ox, const int oy,
        const int quadrature,
        const UndistortParams params) {
        if (ox >= params.dst_width || oy >= params.dst_height)
            return;

        float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int qy = 0; qy < quadrature; ++qy) {
            for (int qx = 0; qx < quadrature; ++qx) {
                const float pixel_x = static_cast<float>(ox) +
                                      (static_cast<float>(qx) + 0.5f) / quadrature;
                const float pixel_y = static_cast<float>(oy) +
                                      (static_cast<float>(qy) + 0.5f) / quadrature;
                const float nx = (pixel_x - params.dst_cx) / params.dst_fx;
                const float ny = (pixel_y - params.dst_cy) / params.dst_fy;
                float dnx, dny;
                apply_distortion(nx, ny, params.model_type, params.distortion,
                                 params.num_distortion, dnx, dny);
                // CUDA fuses this projection. Separate rounding can move a tap
                // exactly onto a sinc zero just outside the image border.
                const float sx = std::fma(dnx, params.src_fx, params.src_cx) - PIXEL_CENTER_OFFSET;
                const float sy = std::fma(dny, params.src_fy, params.src_cy) - PIXEL_CENTER_OFFSET;
                float sample[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                float absolute_inside = 0.0f;
                float absolute_full = 0.0f;
                (void)lanczos3_sample(
                    src, params.src_width, params.src_height, channels,
                    sx, sy, sample, absolute_inside, absolute_full);
                for (int channel = 0; channel < channels; ++channel)
                    result[channel] += sample[channel];
            }
        }

        const int output_index = oy * params.dst_width + ox;
        const int output_plane = params.dst_width * params.dst_height;
        const float inverse_samples = 1.0f / static_cast<float>(quadrature * quadrature);
        for (int channel = 0; channel < channels; ++channel)
            dst[channel * output_plane + output_index] = result[channel] * inverse_samples;
    }

    // The fisheye radial model is one-dimensional in the incidence angle: solving it in theta
    // stays well conditioned up to the angle limit, where Newton in the image plane cannot reach
    // the root because r = tan(theta) grows without bound.

    inline bool inverse_distortion(
        const float xd, const float yd, const UndistortParams& params,
        float& ux, float& uy) {
        if (!seed_inverse_distortion(xd, yd, params.model_type, params.distortion,
                                     params.num_distortion, ux, uy))
            return false;

        float previous_error_px = INFINITY;
        for (int iter = 0; iter < MAX_NEWTON_ITERATIONS; ++iter) {
            float eval_x, eval_y;
            apply_distortion(ux, uy, params.model_type, params.distortion,
                             params.num_distortion, eval_x, eval_y);
            const float rx = eval_x - xd;
            const float ry = eval_y - yd;
            const float error_px = hypotf(rx * params.src_fx, ry * params.src_fy);
            if (!isfinite(error_px))
                return false;
            const bool stalled = error_px <= INVERSE_ACCEPT_PIXELS && error_px >= previous_error_px;
            if (error_px <= INVERSE_RESIDUAL_PIXELS || stalled)
                break;
            previous_error_px = error_px;

            float xp_x, xp_y, xm_x, xm_y;
            float yp_x, yp_y, ym_x, ym_y;
            apply_distortion(ux + INVERSE_JACOBIAN_STEP, uy,
                             params.model_type, params.distortion,
                             params.num_distortion, xp_x, xp_y);
            apply_distortion(ux - INVERSE_JACOBIAN_STEP, uy,
                             params.model_type, params.distortion,
                             params.num_distortion, xm_x, xm_y);
            apply_distortion(ux, uy + INVERSE_JACOBIAN_STEP,
                             params.model_type, params.distortion,
                             params.num_distortion, yp_x, yp_y);
            apply_distortion(ux, uy - INVERSE_JACOBIAN_STEP,
                             params.model_type, params.distortion,
                             params.num_distortion, ym_x, ym_y);

            const float inverse_step = 0.5f / INVERSE_JACOBIAN_STEP;
            const float j00 = (xp_x - xm_x) * inverse_step;
            const float j10 = (xp_y - xm_y) * inverse_step;
            const float j01 = (yp_x - ym_x) * inverse_step;
            const float j11 = (yp_y - ym_y) * inverse_step;
            const float det = j00 * j11 - j01 * j10;
            if (!isfinite(det) || fabsf(det) < NEWTON_EPSILON)
                return false;

            float step_x = (j11 * rx - j01 * ry) / det;
            float step_y = (-j10 * rx + j00 * ry) / det;
            if (!isfinite(step_x) || !isfinite(step_y))
                return false;
            const float step_length = fmaxf(fabsf(step_x), fabsf(step_y));
            if (step_length > INVERSE_MAX_STEP) {
                step_x *= INVERSE_MAX_STEP / step_length;
                step_y *= INVERSE_MAX_STEP / step_length;
            }
            ux -= step_x;
            uy -= step_y;
            if (!isfinite(ux) || !isfinite(uy))
                return false;
        }

        float final_x, final_y;
        apply_distortion(ux, uy, params.model_type, params.distortion,
                         params.num_distortion, final_x, final_y);
        const float final_error_px = hypotf(
            (final_x - xd) * params.src_fx,
            (final_y - yd) * params.src_fy);
        if (!isfinite(final_error_px) || final_error_px > INVERSE_ACCEPT_PIXELS)
            return false;
        if ((params.model_type == CameraModelType::FISHEYE ||
             params.model_type == CameraModelType::THIN_PRISM_FISHEYE) &&
            atanf(hypotf(ux, uy)) >= MAX_FISHEYE_THETA)
            return false;
        return true;
    }

    inline void
    distort_image_to_source_kernel(
        const float* src,
        float* dst,
        uint8_t* validity,
        const int channels,
        const int ox, const int oy,
        const UndistortParams params) {
        if (ox >= params.src_width || oy >= params.src_height)
            return;

        float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float absolute_inside_sum = 0.0f;
        float absolute_full_sum = 0.0f;
        bool converged = true;
        for (int qy = 0; qy < AREA_QUADRATURE && converged; ++qy) {
            for (int qx = 0; qx < AREA_QUADRATURE; ++qx) {
                const float pixel_x = static_cast<float>(ox) +
                                      (static_cast<float>(qx) + 0.5f) / AREA_QUADRATURE;
                const float pixel_y = static_cast<float>(oy) +
                                      (static_cast<float>(qy) + 0.5f) / AREA_QUADRATURE;
                const float xd = (pixel_x - params.src_cx) / params.src_fx;
                const float yd = (pixel_y - params.src_cy) / params.src_fy;
                float ux, uy;
                if (!inverse_distortion(xd, yd, params, ux, uy)) {
                    converged = false;
                    break;
                }
                const float sx = ux * params.dst_fx + params.dst_cx - PIXEL_CENTER_OFFSET;
                const float sy = uy * params.dst_fy + params.dst_cy - PIXEL_CENTER_OFFSET;
                if (!isfinite(sx) || !isfinite(sy)) {
                    converged = false;
                    break;
                }
                float sample[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                float absolute_inside = 0.0f;
                float absolute_full = 0.0f;
                if (!lanczos3_sample(
                        src, params.dst_width, params.dst_height, channels,
                        sx, sy, sample, absolute_inside, absolute_full)) {
                    converged = false;
                    break;
                }
                absolute_inside_sum += absolute_inside;
                absolute_full_sum += absolute_full;
                for (int channel = 0; channel < channels; ++channel)
                    result[channel] += sample[channel];
            }
        }

        const int output_index = oy * params.src_width + ox;
        const int output_plane = params.src_width * params.src_height;
        const float coverage = absolute_full_sum > 0.0f
                                   ? absolute_inside_sum / absolute_full_sum
                                   : 0.0f;
        if (!converged || coverage < EVALUATION_MIN_COVERAGE) {
            validity[output_index] = 0;
            for (int channel = 0; channel < channels; ++channel)
                dst[channel * output_plane + output_index] = 0.0f;
            return;
        }

        validity[output_index] = 1;
        constexpr float inverse_samples = 1.0f / (AREA_QUADRATURE * AREA_QUADRATURE);
        for (int channel = 0; channel < channels; ++channel)
            dst[channel * output_plane + output_index] = result[channel] * inverse_samples;
    }

    enum class AreaFilterMode : int {
        NONNEGATIVE,
        DEPTH,
        NORMAL
    };

    inline void
    undistort_area_kernel(
        const float* src,
        float* dst,
        const int channels,
        const AreaFilterMode mode,
        const int ox, const int oy,
        const int quadrature,
        const UndistortParams params) {
        if (ox >= params.dst_width || oy >= params.dst_height)
            return;

        float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float valid_samples = 0.0f;
        const int input_plane = params.src_width * params.src_height;
        for (int qy = 0; qy < quadrature; ++qy) {
            for (int qx = 0; qx < quadrature; ++qx) {
                const float pixel_x = static_cast<float>(ox) +
                                      (static_cast<float>(qx) + 0.5f) / quadrature;
                const float pixel_y = static_cast<float>(oy) +
                                      (static_cast<float>(qy) + 0.5f) / quadrature;
                const float nx = (pixel_x - params.dst_cx) / params.dst_fx;
                const float ny = (pixel_y - params.dst_cy) / params.dst_fy;
                float dnx, dny;
                apply_distortion(nx, ny, params.model_type, params.distortion,
                                 params.num_distortion, dnx, dny);
                const float sx = dnx * params.src_fx + params.src_cx - PIXEL_CENTER_OFFSET;
                const float sy = dny * params.src_fy + params.src_cy - PIXEL_CENTER_OFFSET;
                if (mode == AreaFilterMode::DEPTH) {
                    const float sample = bilinear_sample_renormalized(
                        src, params.src_width, params.src_height, params.src_width,
                        sx, sy, true);
                    if (sample > 0.0f && isfinite(sample)) {
                        result[0] += sample;
                        valid_samples += 1.0f;
                    }
                } else {
                    for (int channel = 0; channel < channels; ++channel) {
                        result[channel] += bilinear_sample_renormalized(
                            src + channel * input_plane,
                            params.src_width, params.src_height, params.src_width,
                            sx, sy);
                    }
                    valid_samples += 1.0f;
                }
            }
        }

        const int output_index = oy * params.dst_width + ox;
        const int output_plane = params.dst_width * params.dst_height;
        if (valid_samples <= 0.0f) {
            for (int channel = 0; channel < channels; ++channel)
                dst[channel * output_plane + output_index] = 0.0f;
            return;
        }
        for (int channel = 0; channel < channels; ++channel)
            result[channel] /= valid_samples;
        if (mode == AreaFilterMode::NORMAL) {
            const float norm = sqrtf(result[0] * result[0] + result[1] * result[1] +
                                     result[2] * result[2]);
            if (norm > 1.0e-8f && isfinite(norm)) {
                result[0] /= norm;
                result[1] /= norm;
                result[2] /= norm;
            } else {
                result[0] = result[1] = result[2] = 0.0f;
            }
        }
        for (int channel = 0; channel < channels; ++channel)
            dst[channel * output_plane + output_index] = result[channel];
    }

    inline void
    distort_area_to_source_kernel(
        const float* src,
        float* dst,
        const int channels,
        const AreaFilterMode mode,
        const int ox, const int oy,
        const UndistortParams params) {
        if (ox >= params.src_width || oy >= params.src_height)
            return;

        float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float valid_samples = 0.0f;
        const int input_plane = params.dst_width * params.dst_height;
        bool converged = true;
        for (int qy = 0; qy < AREA_QUADRATURE && converged; ++qy) {
            for (int qx = 0; qx < AREA_QUADRATURE; ++qx) {
                const float pixel_x = static_cast<float>(ox) +
                                      (static_cast<float>(qx) + 0.5f) / AREA_QUADRATURE;
                const float pixel_y = static_cast<float>(oy) +
                                      (static_cast<float>(qy) + 0.5f) / AREA_QUADRATURE;
                const float xd = (pixel_x - params.src_cx) / params.src_fx;
                const float yd = (pixel_y - params.src_cy) / params.src_fy;
                float ux, uy;
                if (!inverse_distortion(xd, yd, params, ux, uy)) {
                    converged = false;
                    break;
                }
                const float sx = ux * params.dst_fx + params.dst_cx - PIXEL_CENTER_OFFSET;
                const float sy = uy * params.dst_fy + params.dst_cy - PIXEL_CENTER_OFFSET;
                if (mode == AreaFilterMode::DEPTH) {
                    const float sample = bilinear_sample_renormalized(
                        src, params.dst_width, params.dst_height, params.dst_width,
                        sx, sy, true);
                    if (sample > 0.0f && isfinite(sample)) {
                        result[0] += sample;
                        valid_samples += 1.0f;
                    }
                } else {
                    for (int channel = 0; channel < channels; ++channel) {
                        result[channel] += bilinear_sample_renormalized(
                            src + channel * input_plane,
                            params.dst_width, params.dst_height, params.dst_width,
                            sx, sy);
                    }
                    valid_samples += 1.0f;
                }
            }
        }

        const int output_index = oy * params.src_width + ox;
        const int output_plane = params.src_width * params.src_height;
        if (!converged || valid_samples <= 0.0f) {
            for (int channel = 0; channel < channels; ++channel)
                dst[channel * output_plane + output_index] = 0.0f;
            return;
        }
        for (int channel = 0; channel < channels; ++channel)
            result[channel] /= valid_samples;
        if (mode == AreaFilterMode::NORMAL) {
            const float norm = sqrtf(result[0] * result[0] + result[1] * result[1] +
                                     result[2] * result[2]);
            if (norm > 1.0e-8f && isfinite(norm)) {
                result[0] /= norm;
                result[1] /= norm;
                result[2] /= norm;
            } else {
                result[0] = result[1] = result[2] = 0.0f;
            }
        }
        for (int channel = 0; channel < channels; ++channel)
            dst[channel * output_plane + output_index] = result[channel];
    }

    // At least one sample per source pixel along each axis, so strong minification still
    // integrates the whole footprint instead of aliasing on a fixed sample lattice.
    inline int area_quadrature(const UndistortParams& params) {
        const float minification = std::max(params.src_fx / params.dst_fx,
                                            params.src_fy / params.dst_fy);
        return std::max(AREA_QUADRATURE, static_cast<int>(std::ceil(minification)));
    }

    inline bool is_identity_resample(const UndistortParams& params) {
        if (params.model_type != CameraModelType::PINHOLE ||
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

} // namespace lfs::core::internal::warp_math
