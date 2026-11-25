/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/distortion_model.hpp"
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
    constexpr float COLMAP_MIN_SCALE = 0.2f;
    constexpr float COLMAP_MAX_SCALE = 2.0f;
    constexpr int AREA_QUADRATURE = 8;
    constexpr int LANCZOS_RADIUS = 3;
    constexpr int OUTPUT_TILE_ROWS = 256;
    constexpr float MIN_SIGNED_WEIGHT_RATIO = 1.0e-4f;

    inline float image_sample(const float value) { return value; }
    inline float image_sample(const uint8_t value) { return static_cast<float>(value) / 255.0f; }

    template <typename Sample>
    inline float bilinear_sample_renormalized(
        const Sample* src,
        const int width, const int height, const int stride,
        const float sx, const float sy,
        const bool positive_only = false, const bool panorama = false) {
        const int x0 = static_cast<int>(floorf(sx));
        const int y0 = static_cast<int>(floorf(sy));
        const float fx = sx - static_cast<float>(x0);
        const float fy = sy - static_cast<float>(y0);
        float value = 0.0f;
        float weight_sum = 0.0f;
        for (int dy = 0; dy < 2; ++dy) {
            const float wy = dy == 0 ? 1.0f - fy : fy;
            for (int dx = 0; dx < 2; ++dx) {
                int y = y0 + dy;
                int x = x0 + dx;
                if (!detail::remap_tap(x, y, width, height, panorama))
                    continue;
                const float sample = image_sample(src[y * stride + x]);
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
        int nearest_x = static_cast<int>(floorf(sx + 0.5f));
        int nearest_y = static_cast<int>(floorf(sy + 0.5f));
        if (!detail::remap_tap(nearest_x, nearest_y, width, height, panorama)) {
            nearest_x = min(max(nearest_x, 0), width - 1);
            nearest_y = min(max(nearest_y, 0), height - 1);
        }
        return image_sample(src[nearest_y * stride + nearest_x]);
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

    template <typename Sample>
    inline bool lanczos3_sample(
        const Sample* src,
        const int width, const int height, const int channels,
        const float sx, const float sy,
        float* values, float& absolute_inside, float& absolute_full, const bool panorama = false) {
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
            for (int i = 0; i < 6; ++i) {
                int y = base_y + j - 2;
                int x = base_x + i - 2;
                const float weight = weights_x[i] * weights_y[j];
                absolute_full += fabsf(weight);
                if (!detail::remap_tap(x, y, width, height, panorama))
                    continue;
                signed_inside += weight;
                absolute_inside += fabsf(weight);
                const int index = y * width + x;
                for (int channel = 0; channel < channels; ++channel)
                    weighted[channel] += image_sample(src[channel * plane + index]) * weight;
            }
        }

        if (fabsf(signed_inside) > MIN_SIGNED_WEIGHT_RATIO * absolute_inside) {
            for (int channel = 0; channel < channels; ++channel)
                values[channel] = weighted[channel] / signed_inside;
            return true;
        }

        for (int channel = 0; channel < channels; ++channel) {
            values[channel] = bilinear_sample_renormalized(
                src + channel * plane, width, height, width, sx, sy, false, panorama);
        }
        return false;
    }

    template <typename Sample>
    inline void
    undistort_image_kernel(
        const Sample* src,
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
                const auto [sx, sy] = detail::source_position<true>(pixel_x, pixel_y, params);
                float sample[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                float absolute_inside = 0.0f;
                float absolute_full = 0.0f;
                (void)lanczos3_sample(
                    src, params.src_width, params.src_height, channels,
                    sx, sy, sample, absolute_inside, absolute_full, params.view && static_cast<CameraModelType>(params.model_type) == CameraModelType::EQUIRECTANGULAR);
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

        const bool panorama = params.view && static_cast<CameraModelType>(params.model_type) == CameraModelType::EQUIRECTANGULAR;
        int samples = quadrature;
        if (panorama) {
            // A face pixel spans 1 / cos(latitude) as many panorama columns as rows, so it takes that many more
            // samples per axis to keep one per column it covers, as the CUDA and Slang area filters do.
            const float nx = (static_cast<float>(ox) + 0.5f - params.dst_cx) / params.dst_fx;
            const float ny = (static_cast<float>(oy) + 0.5f - params.dst_cy) / params.dst_fy;
            const auto* r = params.view_rotation;
            const float x = r[0] * nx + r[1] * ny + r[2];
            const float y = r[3] * nx + r[4] * ny + r[5];
            const float z = r[6] * nx + r[7] * ny + r[8];
            const float cos_latitude = std::hypot(x, z) / std::sqrt(x * x + y * y + z * z);
            const float columns = max(params.src_fx / params.dst_fx, params.src_fy / params.dst_fy) /
                                  max(cos_latitude, 1.0f / 64);
            samples = max(quadrature, min(64, static_cast<int>(std::ceil(columns))));
        }
        float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float valid_samples = 0.0f;
        const int input_plane = params.src_width * params.src_height;
        for (int qy = 0; qy < samples; ++qy) {
            for (int qx = 0; qx < samples; ++qx) {
                const float pixel_x = static_cast<float>(ox) +
                                      (static_cast<float>(qx) + 0.5f) / samples;
                const float pixel_y = static_cast<float>(oy) +
                                      (static_cast<float>(qy) + 0.5f) / samples;
                const auto [sx, sy] = detail::source_position(pixel_x, pixel_y, params);
                if (mode == AreaFilterMode::DEPTH) {
                    const float sample = bilinear_sample_renormalized(
                        src, params.src_width, params.src_height, params.src_width,
                        sx, sy, true, params.view && static_cast<CameraModelType>(params.model_type) == CameraModelType::EQUIRECTANGULAR);
                    if (sample > 0.0f && isfinite(sample)) {
                        result[0] += sample;
                        valid_samples += 1.0f;
                    }
                } else {
                    for (int channel = 0; channel < channels; ++channel) {
                        result[channel] += bilinear_sample_renormalized(
                            src + channel * input_plane,
                            params.src_width, params.src_height, params.src_width,
                            sx, sy, false, params.view && static_cast<CameraModelType>(params.model_type) == CameraModelType::EQUIRECTANGULAR);
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

    // At least one sample per source pixel along each axis, so strong minification still
    // integrates the whole footprint instead of aliasing on a fixed sample lattice.
    inline int area_quadrature(const UndistortParams& params) {
        const float minification = std::max(params.src_fx / params.dst_fx,
                                            params.src_fy / params.dst_fy);
        return std::max(AREA_QUADRATURE, static_cast<int>(std::ceil(minification)));
    }

    inline bool is_identity_resample(const UndistortParams& params) {
        if (params.view || params.model_type != CameraModelType::PINHOLE ||
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
