/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cmath>

// Geometry of the match strategy's optimizer step, shared by the fused backward and the tests.
namespace lfs::training::match_geometry {

#ifdef __CUDACC__
#define LFS_MATCH_HD __host__ __device__ __forceinline__
#else
#define LFS_MATCH_HD inline
#endif

    // Footprint extent in standard deviations: sqrt(2 ln(max(255 opacity, floor))).
    LFS_MATCH_HD float footprint_extent(const float opacity, const float floor) {
        return sqrtf(2.0f * logf(fmaxf(255.0f * opacity, floor)));
    }

    // Angular radius of a splat seen from distance d: rho / (max(d, rho) + sqrt(max(d^2 - rho^2, 0))),
    // rho = exp(max log scale) x footprint_extent(opacity, 1). About rho / (2 d) far away.
    LFS_MATCH_HD float view_radius(const float max_log_scale, const float opacity, const float distance) {
        const float rho = expf(max_log_scale) * footprint_extent(opacity, 1.0f);
        const float d2 = distance * distance;
        const float radius = rho / (fmaxf(distance, rho) + sqrtf(fmaxf(d2 - rho * rho, 0.0f)));
        return radius < INFINITY ? radius : 0.0f; // also 0 for NaN
    }

    // Row-major rotation of a raw (w, x, y, z) quaternion; false for a zero quaternion.
    LFS_MATCH_HD bool rotation(const float q[4], float R[9]) {
        const float norm2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
        if (!(norm2 > 0.0f))
            return false;
        const float inv = 1.0f / sqrtf(norm2);
        const float w = q[0] * inv, x = q[1] * inv, y = q[2] * inv, z = q[3] * inv;
        R[0] = 1.0f - 2.0f * (y * y + z * z);
        R[1] = 2.0f * (x * y - w * z);
        R[2] = 2.0f * (x * z + w * y);
        R[3] = 2.0f * (x * y + w * z);
        R[4] = 1.0f - 2.0f * (x * x + z * z);
        R[5] = 2.0f * (y * z - w * x);
        R[6] = 2.0f * (x * z - w * y);
        R[7] = 2.0f * (y * z + w * x);
        R[8] = 1.0f - 2.0f * (x * x + y * y);
        return true;
    }

    // First-moment input of the camera-distance position step: e R diag(exp(s)) R^T g with
    // e = footprint_extent(opacity, 1.00001). Its second-moment input is (0.6 r g)^2 per axis, r the
    // view radius, so the first Adam step is about lr x e sigma_dir / (0.6 r) ~ 3.3 lr x distance along the
    // long axis, whatever the splat's size.
    LFS_MATCH_HD void angular_first_moment(const float q[4], const float log_scales[3], const float opacity,
                                           const float g[3], float out[3]) {
        float R[9];
        if (!rotation(q, R)) {
            out[0] = out[1] = out[2] = 0.0f;
            return;
        }
        const float extent = footprint_extent(opacity, 1.00001f);
        float local[3];
        for (int a = 0; a < 3; ++a)
            local[a] = expf(log_scales[a]) * (R[a] * g[0] + R[3 + a] * g[1] + R[6 + a] * g[2]);
        for (int d = 0; d < 3; ++d)
            out[d] = extent * (R[3 * d] * local[0] + R[3 * d + 1] * local[1] + R[3 * d + 2] * local[2]);
    }

    LFS_MATCH_HD float angular_second_moment_scale(const float radius) {
        return (0.6f * radius) * (0.6f * radius);
    }

    // On-screen size hinge: push = weight x log2(radius / limit) above the limit, else 0. Axis i of the scale
    // gradient gains push x exp(2 (s_i - s_max)) x (sqrt(v_hat_i) + eps).
    LFS_MATCH_HD float hinge_push(const float radius, const float limit, const float weight) {
        return (weight > 0.0f && radius > limit) ? weight * log2f(radius / limit) : 0.0f;
    }

#undef LFS_MATCH_HD

} // namespace lfs::training::match_geometry
