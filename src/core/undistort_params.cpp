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
        constexpr int MAX_NEWTON_ITERATIONS = 20;
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

        // Host twin of the branch-aware inverse in undistort.cu, so the crop matches the CUDA warps
        // on every backend. Pixels past the first turning point of the lens map, or past the
        // horizon, have no ray.
        constexpr float BRANCH_MAX_THETA = 1.570795f;
        constexpr int BRANCH_SAMPLES = 1024;
        constexpr float NO_BOUND = std::numeric_limits<float>::infinity();

        using RadialBranch = UndistortRadialBranch;

        void radial_mapping(const float r, const UndistortParams& params,
                            float& value, float& derivative, float& denominator) {
            const auto* k = params.distortion;
            const bool fisheye = params.model_type != CameraModelType::PINHOLE;
            const float s = r * r;
            const float k1 = params.num_distortion > 0 ? k[0] : 0.0f;
            const float k2 = params.num_distortion > 1 ? k[1] : 0.0f;
            const float k3 = params.num_distortion > 2 ? k[2] : 0.0f;
            const float k4 = fisheye && params.num_distortion > 3 ? k[3] : 0.0f;
            const float numerator = 1.0f + s * (k1 + s * (k2 + s * (k3 + s * k4)));
            const float dn = k1 + s * (2.0f * k2 + s * (3.0f * k3 + s * 4.0f * k4));
            const bool rational = !fisheye && params.num_distortion >= 6;
            denominator = rational ? 1.0f + s * (k[3] + s * (k[4] + s * k[5])) : 1.0f;
            const float dd = rational ? k[3] + s * (2.0f * k[4] + s * 3.0f * k[5]) : 0.0f;
            const float radial = numerator / denominator;
            value = r * radial;
            derivative = radial + 2.0f * s * (dn - radial * dd) / denominator;
        }

        RadialBranch radial_branch(const UndistortParams& params) {
            const bool fisheye = params.model_type != CameraModelType::PINHOLE;
            RadialBranch branch{fisheye ? BRANCH_MAX_THETA : NO_BOUND, NO_BOUND};
            // Scanning atan(r) covers pinhole radii without a lens-specific cutoff.
            int first = BRANCH_SAMPLES + 1;
            for (int i = 1; i <= BRANCH_SAMPLES; ++i) {
                const float angle = BRANCH_MAX_THETA * (static_cast<float>(i) / BRANCH_SAMPLES);
                float value, derivative, denominator;
                radial_mapping(fisheye ? angle : std::tan(angle), params, value, derivative, denominator);
                if (derivative <= 0.0f || denominator <= 0.0f) {
                    first = i;
                    break;
                }
            }
            if (first <= BRANCH_SAMPLES) {
                float lo = BRANCH_MAX_THETA * (static_cast<float>(first - 1) / BRANCH_SAMPLES);
                float hi = BRANCH_MAX_THETA * (static_cast<float>(first) / BRANCH_SAMPLES);
                for (int i = 0; i < 24; ++i) {
                    const float mid = 0.5f * (lo + hi);
                    float value, derivative, denominator;
                    radial_mapping(fisheye ? mid : std::tan(mid), params, value, derivative, denominator);
                    if (derivative > 0.0f && denominator > 0.0f)
                        lo = mid;
                    else
                        hi = mid;
                }
                branch.bound = fisheye ? lo : std::tan(lo);
            }
            if (std::isfinite(branch.bound)) {
                float derivative, denominator;
                radial_mapping(branch.bound, params, branch.maximum, derivative, denominator);
            }
            return branch;
        }

        bool branch_inverse_distortion(
            const float xd, const float yd, const UndistortParams& params, const RadialBranch& branch,
            float& ux, float& uy) {
            const bool fisheye = params.model_type != CameraModelType::PINHOLE;
            const bool prism = params.model_type == CameraModelType::THIN_PRISM_FISHEYE;
            const int offset = params.num_distortion >= 6 ? 6 : 3;
            bool coupled = !fisheye &&
                           ((params.num_distortion > offset && params.distortion[offset] != 0.0f) ||
                            (params.num_distortion > offset + 1 && params.distortion[offset + 1] != 0.0f));
            if (prism) {
                for (int i = 4; i < params.num_distortion; ++i)
                    coupled |= params.distortion[i] != 0.0f;
            }
            const float rd = std::sqrt(xd * xd + yd * yd);
            if (!std::isfinite(rd) || (!coupled && rd >= branch.maximum))
                return false;
            if (rd < 1e-8f) {
                ux = xd;
                uy = yd;
                return true;
            }
            const float px = params.src_fx * xd;
            const float py = params.src_fy * yd;
            const float pixel_scale = std::sqrt(px * px + py * py) / rd;
            float lo = 0.0f;
            float hi = branch.bound;
            if (!std::isfinite(hi)) {
                hi = std::fmax(1.0f, 2.0f * rd);
                for (int i = 0; i < 32; ++i) {
                    float value, derivative, denominator;
                    radial_mapping(hi, params, value, derivative, denominator);
                    if (value >= rd)
                        break;
                    hi *= 2.0f;
                }
            }
            float r = std::isfinite(branch.maximum) ? rd * (branch.bound / branch.maximum) : rd;
            r = std::fmin(r, hi);
            if (rd >= branch.maximum) {
                r = std::nextafter(branch.bound, 0.0f);
            } else {
                float previous_error_px = NO_BOUND;
                for (int i = 0; i < 40; ++i) {
                    float value, derivative, denominator;
                    radial_mapping(r, params, value, derivative, denominator);
                    const float residual = value - rd;
                    const float error_px = std::fabs(residual) * pixel_scale;
                    if (error_px <= INVERSE_RESIDUAL_PIXELS ||
                        (error_px <= INVERSE_ACCEPT_PIXELS && error_px >= previous_error_px))
                        break;
                    previous_error_px = error_px;
                    if (residual < 0.0f)
                        lo = r;
                    else
                        hi = r;
                    const float next = r - residual / derivative;
                    r = next > lo && next < hi ? next : 0.5f * (lo + hi);
                }
            }
            if (!coupled) {
                float value, derivative, denominator;
                radial_mapping(r, params, value, derivative, denominator);
                if (!(r < branch.bound) || !std::isfinite(value) ||
                    std::fabs(value - rd) * pixel_scale > INVERSE_ACCEPT_PIXELS)
                    return false;
                const float scale = (fisheye ? std::tan(r) : r) / rd;
                ux = xd * scale;
                uy = yd * scale;
                return std::isfinite(ux) && std::isfinite(uy);
            }
            ux = xd * r / rd;
            uy = yd * r / rd;

            // Coupled roots must converge in pixels inside the first radial branch with det J > 0.
            // Prism refinement stays in theta coordinates so a horizon seed does not overflow tan(theta).
            const auto forward = [&](const float x, const float y, float& dx, float& dy) {
                if (prism)
                    thin_prism_fisheye_from_theta_point(x, y, params.distortion, params.num_distortion, dx, dy);
                else
                    apply_distortion_cpu(x, y, params.model_type, params.distortion, params.num_distortion, dx, dy);
            };
            float previous_error_px = NO_BOUND;
            for (int iter = 0; iter <= MAX_NEWTON_ITERATIONS; ++iter) {
                float eval_x, eval_y;
                forward(ux, uy, eval_x, eval_y);
                const float rx = eval_x - xd;
                const float ry = eval_y - yd;
                const float error_px = std::hypot(rx * params.src_fx, ry * params.src_fy);
                if (!std::isfinite(error_px))
                    return false;
                float xp_x, xp_y, xm_x, xm_y;
                float yp_x, yp_y, ym_x, ym_y;
                forward(ux + INVERSE_JACOBIAN_STEP, uy, xp_x, xp_y);
                forward(ux - INVERSE_JACOBIAN_STEP, uy, xm_x, xm_y);
                forward(ux, uy + INVERSE_JACOBIAN_STEP, yp_x, yp_y);
                forward(ux, uy - INVERSE_JACOBIAN_STEP, ym_x, ym_y);
                const float inverse_step = 0.5f / INVERSE_JACOBIAN_STEP;
                const float j00 = (xp_x - xm_x) * inverse_step;
                const float j10 = (xp_y - xm_y) * inverse_step;
                const float j01 = (yp_x - ym_x) * inverse_step;
                const float j11 = (yp_y - ym_y) * inverse_step;
                const float det = j00 * j11 - j01 * j10;
                if (!std::isfinite(det) || std::fabs(det) < NEWTON_EPSILON)
                    return false;
                const bool stalled = error_px <= INVERSE_ACCEPT_PIXELS && error_px >= previous_error_px;
                if (error_px <= INVERSE_RESIDUAL_PIXELS || stalled || iter == MAX_NEWTON_ITERATIONS) {
                    const float radius = std::hypot(ux, uy);
                    if (error_px > INVERSE_ACCEPT_PIXELS || det <= 0.0f || !(radius < branch.bound))
                        return false;
                    if (prism && radius > 1e-8f) {
                        const float scale = std::tan(radius) / radius;
                        ux *= scale;
                        uy *= scale;
                    }
                    return std::isfinite(ux) && std::isfinite(uy);
                }
                previous_error_px = error_px;
                float step_x = (j11 * rx - j01 * ry) / det;
                float step_y = (-j10 * rx + j00 * ry) / det;
                if (!std::isfinite(step_x) || !std::isfinite(step_y))
                    return false;
                const float step_length = std::fmax(std::fabs(step_x), std::fabs(step_y));
                if (step_length > INVERSE_MAX_STEP) {
                    step_x *= INVERSE_MAX_STEP / step_length;
                    step_y *= INVERSE_MAX_STEP / step_length;
                }
                ux -= step_x;
                uy -= step_y;
            }
            return false;
        }

        // Traces the frame border through the branch-aware inverse; border samples without a ray
        // take the fold boundary. Keeps COLMAP's scale rules and fails below the minimum scale.
        bool solve_undistort_crop(const UndistortParams& params, const float blank_pixels,
                                  int& crop_width, int& crop_height) {
            const auto branch = radial_branch(params);
            const float radius = params.model_type == CameraModelType::PINHOLE ? branch.bound : std::tan(branch.bound);
            std::array<float, 4> low;
            std::array<float, 4> high;
            low.fill(NO_BOUND);
            high.fill(-NO_BOUND);
            for (int edge = 0; edge < 4; ++edge) {
                const int length = edge < 2 ? params.src_height : params.src_width;
                for (int i = 0; i < length; ++i) {
                    const float px = edge < 2 ? (edge == 0 ? PIXEL_CENTER_OFFSET : params.src_width - PIXEL_CENTER_OFFSET)
                                              : i + PIXEL_CENTER_OFFSET;
                    const float py = edge >= 2 ? (edge == 2 ? PIXEL_CENTER_OFFSET : params.src_height - PIXEL_CENTER_OFFSET)
                                               : i + PIXEL_CENTER_OFFSET;
                    const float xd = (px - params.src_cx) / params.src_fx;
                    const float yd = (py - params.src_cy) / params.src_fy;
                    float ux, uy;
                    if (!branch_inverse_distortion(xd, yd, params, branch, ux, uy)) {
                        const float rd = std::hypot(xd, yd);
                        if (!std::isfinite(radius) || rd <= 0.0f)
                            continue;
                        ux = xd * radius / rd;
                        uy = yd * radius / rd;
                    }
                    const float value = edge < 2 ? params.src_fx * ux : params.src_fy * uy;
                    low[edge] = std::fmin(low[edge], value);
                    high[edge] = std::fmax(high[edge], value);
                }
            }
            float scale[2];
            for (int axis = 0; axis < 2; ++axis) {
                const float center = axis == 0 ? params.src_cx : params.src_cy;
                const float size = static_cast<float>(axis == 0 ? params.src_width : params.src_height);
                const int near_edge = 2 * axis;
                const int far_edge = near_edge + 1;
                const float outer = std::fmin(center / -low[near_edge], (size - PIXEL_CENTER_OFFSET - center) / high[far_edge]);
                const float inner = std::fmax(center / -high[near_edge], (size - PIXEL_CENTER_OFFSET - center) / low[far_edge]);
                const float blended = outer * blank_pixels + inner * (1.0f - blank_pixels);
                if (!std::isfinite(blended) || blended <= 0.0f)
                    return false;
                scale[axis] = std::fmin(COLMAP_MAX_SCALE, std::fmax(COLMAP_MIN_SCALE, 1.0f / blended));
            }
            const float corner = std::hypot(
                std::fmax(params.src_cx, params.src_width - params.src_cx) * scale[0] / params.src_fx,
                std::fmax(params.src_cy, params.src_height - params.src_cy) * scale[1] / params.src_fy);
            if (corner >= radius) {
                const float shrink = std::nextafter(radius, 0.0f) / corner;
                scale[0] *= shrink;
                scale[1] *= shrink;
            }
            if (scale[0] < COLMAP_MIN_SCALE || scale[1] < COLMAP_MIN_SCALE)
                return false;
            crop_width = std::max(1, static_cast<int>(params.src_width * scale[0]));
            crop_height = std::max(1, static_cast<int>(params.src_height * scale[1]));
            return true;
        }

    } // namespace

    UndistortRadialBranch undistort_radial_branch(const UndistortParams& params) {
        return radial_branch(params);
    }

    bool undistort_normalized_point(
        const UndistortParams& params,
        const UndistortRadialBranch& branch,
        const float distorted_x,
        const float distorted_y,
        float& normalized_x,
        float& normalized_y) {
        return branch_inverse_distortion(distorted_x, distorted_y, params, branch, normalized_x, normalized_y);
    }

    bool undistort_image_point(
        const UndistortParams& params,
        const float image_x,
        const float image_y,
        float& normalized_x,
        float& normalized_y) {
        return branch_inverse_distortion(
            (image_x - params.src_cx) / params.src_fx, (image_y - params.src_cy) / params.src_fy,
            params, radial_branch(params), normalized_x, normalized_y);
    }

    UndistortParams detail::initialize_undistort_params(
        float fx, float fy, float cx, float cy,
        int width, int height,
        const Tensor& radial, const Tensor& tangential,
        CameraModelType model) {

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

        return params;
    }

    UndistortParams compute_undistort_params(
        float fx, float fy, float cx, float cy,
        int width, int height,
        const Tensor& radial, const Tensor& tangential,
        CameraModelType model, float blank_pixels) {

        UndistortParams params = detail::initialize_undistort_params(
            fx, fy, cx, cy, width, height, radial, tangential, model);

        const bool distorted =
            model != CameraModelType::PINHOLE ||
            std::any_of(params.distortion, params.distortion + params.num_distortion,
                        [](const float value) { return value != 0.0f; });
        if (!distorted) {
            return params;
        }

        int crop_width = 0;
        int crop_height = 0;
        if (width <= 0 || height <= 0 || fx <= 0.0f || fy <= 0.0f ||
            !solve_undistort_crop(params, blank_pixels, crop_width, crop_height)) {
            params.crop_solve_failed = true;
            return params;
        }
        params.dst_width = crop_width;
        params.dst_height = crop_height;
        params.dst_cx = cx * static_cast<float>(params.dst_width) / static_cast<float>(width);
        params.dst_cy = cy * static_cast<float>(params.dst_height) / static_cast<float>(height);

        LOG_DEBUG("Undistort: {}x{} -> {}x{}, fx={:.1f}->{:.1f}, fy={:.1f}->{:.1f}",
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
