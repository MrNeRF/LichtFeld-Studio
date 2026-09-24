/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor_image.hpp"

#include <algorithm>
#include "core/tensor/internal/private_access.hpp"
#include <cmath>
namespace lfs::core::internal::image_math {
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
        const float radial = 1.0f + k1 * r2 + k2 * r4 + k3 * r6;

        const float p1 = num_dist > 3 ? dist[3] : 0.0f;
        const float p2 = num_dist > 4 ? dist[4] : 0.0f;

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

    inline float bilinear_sample(
        const float* src,
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

    inline void undistort(const float* src, float* dst, int channels, const UndistortParams& p, int x, int y) {
        float dx, dy;
        apply_distortion((x + 0.5f - p.dst_cx) / p.dst_fx, (y + 0.5f - p.dst_cy) / p.dst_fy,
                         p.model_type, p.distortion, p.num_distortion, dx, dy);
        const float sx = dx * p.src_fx + p.src_cx - 0.5f, sy = dy * p.src_fy + p.src_cy - 0.5f;
        for (int c = 0; c < channels; ++c)
            dst[c * p.dst_width * p.dst_height + y * p.dst_width + x] =
                bilinear_sample(src + c * p.src_width * p.src_height, p.src_width, p.src_height, p.src_width, sx, sy);
    }
    inline bool valid(const float* src, size_t i, size_t plane, bool normal) {
        if (!std::isfinite(src[i]))
            return false;
        if (!normal)
            return src[i] > 0;
        return std::isfinite(src[plane + i]) && std::isfinite(src[2 * plane + i]) &&
               src[i] * src[i] + src[plane + i] * src[plane + i] + src[2 * plane + i] * src[2 * plane + i] >= 0.25f;
    }
    inline void resize_prior(const float* src, float* dst, int sw, int sh, int dw, int dh, bool normal, int x, int y) {
        const size_t sp = size_t(sw) * sh, dp = size_t(dw) * dh, out = size_t(y) * dw + x;
        const float sx = fmaxf(0, fminf(sw - 1.0f, (x + 0.5f) * sw / dw - 0.5f));
        const float sy = fmaxf(0, fminf(sh - 1.0f, (y + 0.5f) * sh / dh - 0.5f));
        const int x0 = int(sx), y0 = int(sy), channels = normal ? 3 : 1;
        float value[3]{}, weight = 0;
        if (valid(src, size_t(int(sy + 0.5f)) * sw + int(sx + 0.5f), sp, normal)) {
            for (int j = 0; j < 2; ++j)
                for (int i = 0; i < 2; ++i) {
                    const size_t index = size_t(std::min(y0 + j, sh - 1)) * sw + std::min(x0 + i, sw - 1);
                    if (!valid(src, index, sp, normal))
                        continue;
                    const float w = (i ? sx - x0 : 1.0f - (sx - x0)) * (j ? sy - y0 : 1.0f - (sy - y0));
                    weight += w;
                    for (int c = 0; c < channels; ++c)
                        value[c] += w * src[c * sp + index];
                }
        }
        if (normal)
            weight = sqrtf(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
        for (int c = 0; c < channels; ++c)
            dst[c * dp + out] = weight > 1e-8f ? value[c] / weight : 0;
    }
} // namespace lfs::core::internal::image_math
