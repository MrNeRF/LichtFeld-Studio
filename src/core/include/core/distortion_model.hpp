/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "core/device/portability.cuh"

#include "core/camera_types.h"

#include <cmath>

#if LFS_DEVICE_COMPILER
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

    struct SourcePosition {
        float x, y;
    };

    template <bool FusedProjection = false, typename Lens>
    LFS_DISTORTION_HD inline SourcePosition source_position(float pixel_x, float pixel_y, const Lens& p) {
        const float nx = (pixel_x - p.dst_cx) / p.dst_fx;
        const float ny = (pixel_y - p.dst_cy) / p.dst_fy;
        const auto model = static_cast<CameraModelType>(p.model_type);
        float dx, dy;
        if (p.view) {
            const auto* r = p.view_rotation;
            const float x = r[0] * nx + r[1] * ny + r[2];
            const float y = r[3] * nx + r[4] * ny + r[5];
            const float z = r[6] * nx + r[7] * ny + r[8];
            if (model == CameraModelType::EQUIRECTANGULAR) {
                dx = std::atan2(x, z);
                dy = std::atan2(y, std::hypot(x, z));
            } else {
                project_fisheye_ray(x, y, z, model, p.distortion, p.num_distortion, dx, dy);
            }
        } else {
            apply_distortion(nx, ny, model, p.distortion, p.num_distortion, dx, dy);
        }
        if constexpr (FusedProjection) {
            return {std::fma(dx, p.src_fx, p.src_cx) - 0.5f, std::fma(dy, p.src_fy, p.src_cy) - 0.5f};
        }
        return {dx * p.src_fx + p.src_cx - 0.5f, dy * p.src_fy + p.src_cy - 0.5f};
    }

    LFS_DISTORTION_HD inline bool remap_tap(int& x, int& y, int width, int height, bool panorama) {
        if (x >= 0 && x < width && y >= 0 && y < height)
            return true;
        if (!panorama)
            return false;
        while (y < 0 || y >= height) {
            y = y < 0 ? -1 - y : height - 1 - (y - height);
            x += width / 2;
        }
        x %= width;
        if (x < 0)
            x += width;
        return true;
    }
} // namespace lfs::core::detail

#undef LFS_DISTORTION_HD
