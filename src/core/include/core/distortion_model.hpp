/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera_types.h"

#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LFS_DISTORTION_HD __host__ __device__
#else
#define LFS_DISTORTION_HD
#endif

// Forward lens models shared by geometry, undistortion and mesh evaluation.
namespace lfs::core::detail {

    // COLMAP sensor/models.h (BSD-3 licensed formulas)
    template <typename Scalar>
    LFS_DISTORTION_HD inline void apply_distortion_pinhole(
        const Scalar x, const Scalar y,
        const Scalar* __restrict dist, const int num_dist,
        Scalar& dx, Scalar& dy) {

        const Scalar r2 = x * x + y * y;
        const Scalar r4 = r2 * r2;
        const Scalar r6 = r4 * r2;

        const Scalar k1 = num_dist > 0 ? dist[0] : Scalar(0.0f);
        const Scalar k2 = num_dist > 1 ? dist[1] : Scalar(0.0f);
        const Scalar k3 = num_dist > 2 ? dist[2] : Scalar(0.0f);
        const Scalar numerator = Scalar(1.0f) + k1 * r2 + k2 * r4 + k3 * r6;
        Scalar radial = numerator;
        if (num_dist >= 6) {
            const Scalar denominator =
                Scalar(1.0f) + dist[3] * r2 + dist[4] * r4 + dist[5] * r6;
            radial = numerator / denominator;
        }

        const int tangential_offset = num_dist >= 6 ? 6 : 3;
        const Scalar p1 = num_dist > tangential_offset ? dist[tangential_offset] : Scalar(0.0f);
        const Scalar p2 = num_dist > tangential_offset + 1 ? dist[tangential_offset + 1] : Scalar(0.0f);

        dx = x * radial + Scalar(2.0f) * p1 * x * y + p2 * (r2 + Scalar(2.0f) * x * x);
        dy = y * radial + p1 * (r2 + Scalar(2.0f) * y * y) + Scalar(2.0f) * p2 * x * y;
    }

    template <typename Scalar>
    LFS_DISTORTION_HD inline void apply_distortion_fisheye(
        const Scalar x, const Scalar y,
        const Scalar* __restrict dist, const int num_dist,
        Scalar& dx, Scalar& dy) {

        const Scalar r = std::sqrt(x * x + y * y);
        if (r < Scalar(1e-8)) {
            dx = x;
            dy = y;
            return;
        }

        const Scalar theta = std::atan(r);
        const Scalar theta2 = theta * theta;
        const Scalar theta4 = theta2 * theta2;
        const Scalar theta6 = theta4 * theta2;
        const Scalar theta8 = theta4 * theta4;

        const Scalar k1 = num_dist > 0 ? dist[0] : Scalar(0.0f);
        const Scalar k2 = num_dist > 1 ? dist[1] : Scalar(0.0f);
        const Scalar k3 = num_dist > 2 ? dist[2] : Scalar(0.0f);
        const Scalar k4 = num_dist > 3 ? dist[3] : Scalar(0.0f);

        const Scalar theta_d = theta * (Scalar(1.0f) + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
        const Scalar scale = theta_d / r;

        dx = x * scale;
        dy = y * scale;
    }

    // COLMAP ThinPrismFisheyeCameraModel evaluates the tangential and prism terms on the
    // theta-scaled point (uu, vv), |(uu, vv)| = theta, independently of the radial polynomial.
    template <typename Scalar>
    LFS_DISTORTION_HD inline void thin_prism_increment(
        const Scalar uu, const Scalar vv,
        const Scalar* __restrict dist, const int num_dist,
        Scalar& tx, Scalar& ty) {
        const Scalar p1 = num_dist > 4 ? dist[4] : Scalar(0.0f);
        const Scalar p2 = num_dist > 5 ? dist[5] : Scalar(0.0f);
        const Scalar sx1 = num_dist > 6 ? dist[6] : Scalar(0.0f);
        const Scalar sx2 = num_dist > 7 ? dist[7] : Scalar(0.0f);
        const Scalar sy1 = num_dist > 8 ? dist[8] : Scalar(0.0f);
        const Scalar sy2 = num_dist > 9 ? dist[9] : Scalar(0.0f);
        const Scalar u2 = uu * uu;
        const Scalar uv = uu * vv;
        const Scalar v2 = vv * vv;
        const Scalar r2 = u2 + v2;
        const Scalar r4 = r2 * r2;
        tx = Scalar(2.0f) * p1 * uv + p2 * (r2 + Scalar(2.0f) * u2) + sx1 * r2 + sx2 * r4;
        ty = Scalar(2.0f) * p2 * uv + p1 * (r2 + Scalar(2.0f) * v2) + sy1 * r2 + sy2 * r4;
    }

    template <typename Scalar>
    LFS_DISTORTION_HD inline void thin_prism_fisheye_from_theta_point(
        const Scalar uu, const Scalar vv,
        const Scalar* __restrict dist, const int num_dist,
        Scalar& dx, Scalar& dy) {
        const Scalar k1 = num_dist > 0 ? dist[0] : Scalar(0.0f);
        const Scalar k2 = num_dist > 1 ? dist[1] : Scalar(0.0f);
        const Scalar k3 = num_dist > 2 ? dist[2] : Scalar(0.0f);
        const Scalar k4 = num_dist > 3 ? dist[3] : Scalar(0.0f);
        const Scalar r2 = uu * uu + vv * vv;
        const Scalar r4 = r2 * r2;
        const Scalar r6 = r4 * r2;
        const Scalar r8 = r6 * r2;
        const Scalar radial = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
        Scalar tx, ty;
        thin_prism_increment(uu, vv, dist, num_dist, tx, ty);
        dx = uu + uu * radial + tx;
        dy = vv + vv * radial + ty;
    }

    template <typename Scalar>
    LFS_DISTORTION_HD inline void apply_distortion_thin_prism_fisheye(
        const Scalar x, const Scalar y,
        const Scalar* __restrict dist, const int num_dist,
        Scalar& dx, Scalar& dy) {

        const Scalar r = std::sqrt(x * x + y * y);
        if (r < Scalar(1e-8)) {
            dx = x;
            dy = y;
            return;
        }

        const Scalar theta_over_r = std::atan(r) / r;
        thin_prism_fisheye_from_theta_point(
            x * theta_over_r, y * theta_over_r, dist, num_dist, dx, dy);
    }

    template <typename Scalar>
    LFS_DISTORTION_HD inline void project_fisheye_ray(
        Scalar x, Scalar y, Scalar z, CameraModelType model, const Scalar* dist, int count,
        Scalar& dx, Scalar& dy) {
        const Scalar r = std::hypot(x, y);
        const Scalar scale = r > 0 ? std::atan2(r, z) / r : Scalar(0);
        thin_prism_fisheye_from_theta_point(x * scale, y * scale, dist,
                                            model == CameraModelType::FISHEYE && count > 4 ? 4 : count, dx, dy);
    }

    template <typename Scalar>
    LFS_DISTORTION_HD inline void apply_distortion(
        const Scalar x, const Scalar y,
        const CameraModelType model,
        const Scalar* __restrict dist, const int num_dist,
        Scalar& dx, Scalar& dy) {

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

namespace lfs::core {

    // The angle off the axis at which the radial polynomial of a fisheye, theta (1 + k1 theta^2 + k2 theta^4 +
    // k3 theta^6 + k4 theta^8), stops increasing, or pi if it increases all the way: beyond it the lens folds rays
    // back onto radii it already holds. The slope is scanned from the axis for its first sign change, so a later
    // turning point is never taken for it, and the crossing is refined by bisection. k holds k1..k4.
    inline double fisheye_radial_limit(const float* k) {
        constexpr double pi = 3.14159265358979323846;
        const auto slope = [k](const double theta) {
            const double s = theta * theta;
            return 1.0 + s * (3.0 * k[0] + s * (5.0 * k[1] + s * (7.0 * k[2] + s * 9.0 * k[3])));
        };
        constexpr int steps = 4096;
        for (int i = 1; i <= steps; ++i) {
            double hi = pi * i / steps;
            if (slope(hi) > 0.0)
                continue;
            double lo = pi * (i - 1) / steps;
            for (int j = 0; j < 64; ++j) {
                const double mid = 0.5 * (lo + hi);
                (slope(mid) > 0.0 ? lo : hi) = mid;
            }
            return lo;
        }
        return pi;
    }

} // namespace lfs::core

#undef LFS_DISTORTION_HD
