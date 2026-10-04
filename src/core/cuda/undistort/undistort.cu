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
#include <limits>
#include <nvtx3/nvToolsExt.h>
#include <stdexcept>

namespace lfs::core::cuda {

    namespace {

        constexpr int BLOCK_DIM = 16;
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

        using detail::apply_distortion;
        using detail::apply_distortion_pinhole;
        using detail::thin_prism_fisheye_from_theta_point;
        using detail::thin_prism_increment;

        __device__ float bilinear_sample_renormalized(
            const float* __restrict__ src,
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

        __device__ float lanczos3_weight(const float value) {
            const float absolute_value = fabsf(value);
            if (absolute_value >= static_cast<float>(LANCZOS_RADIUS))
                return 0.0f;
            if (absolute_value < 1.0e-6f)
                return 1.0f;
            constexpr float PI = 3.14159265358979323846f;
            const float pi_value = PI * value;
            // Preserve sinc zeros near integer taps even with CUDA fast math.
            // Border renormalization amplifies sine argument-reduction error.
            return sinpif(value) / pi_value *
                   (sinpif(value / static_cast<float>(LANCZOS_RADIUS)) /
                    (pi_value / static_cast<float>(LANCZOS_RADIUS)));
        }

        __device__ bool lanczos3_sample(
            const float* __restrict__ src,
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

        __global__ void __launch_bounds__(BLOCK_DIM* BLOCK_DIM)
            undistort_image_kernel(
                const float* __restrict__ src,
                float* __restrict__ dst,
                const int channels,
                const int output_y_offset,
                const int quadrature,
                const UndistortParams params) {
            const int ox = blockIdx.x * BLOCK_DIM + threadIdx.x;
            const int oy = output_y_offset + blockIdx.y * BLOCK_DIM + threadIdx.y;
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
                    const float sx = dnx * params.src_fx + params.src_cx - PIXEL_CENTER_OFFSET;
                    const float sy = dny * params.src_fy + params.src_cy - PIXEL_CENTER_OFFSET;
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
        __host__ __device__ bool solve_fisheye_theta(
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
                if (!isfinite(residual) || !isfinite(slope) || fabsf(slope) < NEWTON_EPSILON)
                    return false;
                const float step = residual / slope;
                theta -= step;
                if (fabsf(step) < 1e-7f)
                    break;
            }
            return theta > 0.0f && theta < MAX_FISHEYE_THETA;
        }

        __host__ __device__ bool inverse_fisheye_radial(
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

        // The prism and tangential terms move the distorted radius away from the radial polynomial,
        // which seeds the Newton solve near the angle limit at r = tan(theta) too far out to recover
        // or rejects the ray outright; strip them by fixed-point iteration on the theta point first.
        __host__ __device__ bool inverse_thin_prism_seed(
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

        __host__ __device__ bool seed_inverse_distortion(
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

        __device__ bool inverse_distortion(
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

        __global__ void __launch_bounds__(BLOCK_DIM* BLOCK_DIM)
            distort_image_to_source_kernel(
                const float* __restrict__ src,
                float* __restrict__ dst,
                uint8_t* __restrict__ validity,
                const int channels,
                const int output_y_offset,
                const UndistortParams params) {
            const int ox = blockIdx.x * BLOCK_DIM + threadIdx.x;
            const int oy = output_y_offset + blockIdx.y * BLOCK_DIM + threadIdx.y;
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

        __global__ void __launch_bounds__(BLOCK_DIM* BLOCK_DIM)
            undistort_area_kernel(
                const float* __restrict__ src,
                float* __restrict__ dst,
                const int channels,
                const AreaFilterMode mode,
                const int output_y_offset,
                const int quadrature,
                const UndistortParams params) {
            const int ox = blockIdx.x * BLOCK_DIM + threadIdx.x;
            const int oy = output_y_offset + blockIdx.y * BLOCK_DIM + threadIdx.y;
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

        __global__ void __launch_bounds__(BLOCK_DIM* BLOCK_DIM)
            distort_area_to_source_kernel(
                const float* __restrict__ src,
                float* __restrict__ dst,
                const int channels,
                const AreaFilterMode mode,
                const int output_y_offset,
                const UndistortParams params) {
            const int ox = blockIdx.x * BLOCK_DIM + threadIdx.x;
            const int oy = output_y_offset + blockIdx.y * BLOCK_DIM + threadIdx.y;
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
        int area_quadrature(const UndistortParams& params) {
            const float minification = std::max(params.src_fx / params.dst_fx,
                                                params.src_fy / params.dst_fy);
            return std::max(AREA_QUADRATURE, static_cast<int>(std::ceil(minification)));
        }

        bool is_identity_resample(const UndistortParams& params) {
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

        __global__ void __launch_bounds__(BLOCK_DIM* BLOCK_DIM)
            inverse_distortion_sample_map_kernel(
                float2* __restrict__ samples,
                const UndistortParams params) {
            const int ox = blockIdx.x * BLOCK_DIM + threadIdx.x;
            const int oy = blockIdx.y * BLOCK_DIM + threadIdx.y;
            if (ox >= params.src_width || oy >= params.src_height)
                return;
            const float xd = (static_cast<float>(ox) + PIXEL_CENTER_OFFSET - params.src_cx) /
                             params.src_fx;
            const float yd = (static_cast<float>(oy) + PIXEL_CENTER_OFFSET - params.src_cy) /
                             params.src_fy;
            float ux, uy;
            if (!inverse_distortion(xd, yd, params, ux, uy)) {
                ux = nanf("");
                uy = nanf("");
            }
            samples[oy * params.src_width + ox] = make_float2(ux, uy);
        }

    } // anonymous namespace

    Tensor undistort_image(
        const Tensor& src, const UndistortParams& params, cudaStream_t stream) {
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
        assert(static_cast<int>(src.shape()[1]) == params.src_height);
        assert(static_cast<int>(src.shape()[2]) == params.src_width);
        if (is_identity_resample(params))
            return input.clone();

        nvtxRangePush("undistort_image");
        auto dst = Tensor::empty(
            {static_cast<size_t>(channels), static_cast<size_t>(params.dst_height),
             static_cast<size_t>(params.dst_width)},
            Device::GPU, DataType::Float32);
        const dim3 block(BLOCK_DIM, BLOCK_DIM);
        for (int output_y = 0; output_y < params.dst_height; output_y += OUTPUT_TILE_ROWS) {
            const int tile_rows = std::min(OUTPUT_TILE_ROWS, params.dst_height - output_y);
            const dim3 grid(
                (params.dst_width + BLOCK_DIM - 1) / BLOCK_DIM,
                (tile_rows + BLOCK_DIM - 1) / BLOCK_DIM);
            undistort_image_kernel<<<grid, block, 0, stream>>>(
                input.ptr<float>(), dst.ptr<float>(), channels, output_y,
                area_quadrature(params), params);
        }
        const cudaError_t error = cudaGetLastError();
        assert(error == cudaSuccess && "undistort_image_kernel launch failed");
        nvtxRangePop();
        return dst;
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
