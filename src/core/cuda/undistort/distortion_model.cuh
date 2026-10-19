/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera_types.h"

#include <cmath>

// Forward lens models shared by the undistortion kernels and the mesh evaluation mask.
namespace lfs::core::detail {

    // COLMAP sensor/models.h (BSD-3 licensed formulas)
    __host__ __device__ inline void apply_distortion_pinhole(
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

    __host__ __device__ inline void apply_distortion_fisheye(
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

    // COLMAP ThinPrismFisheyeCameraModel evaluates the tangential and prism terms on the
    // theta-scaled point (uu, vv), |(uu, vv)| = theta, independently of the radial polynomial.
    __host__ __device__ inline void thin_prism_increment(
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

    __host__ __device__ inline void thin_prism_fisheye_from_theta_point(
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

    __host__ __device__ inline void apply_distortion_thin_prism_fisheye(
        const float x, const float y,
        const float* __restrict__ dist, const int num_dist,
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

    __host__ __device__ inline void apply_distortion(
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

} // namespace lfs::core::detail
