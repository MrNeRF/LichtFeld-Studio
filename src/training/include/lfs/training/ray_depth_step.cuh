/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cmath>

#ifdef __CUDACC__
#define LFS_RAY_DEPTH_FN __host__ __device__ inline
#else
#define LFS_RAY_DEPTH_FN inline
#endif

// Ray-depth coordinate of a splat. A view's image does not change when a splat is scaled
// about that view's camera centre (position and all three scales by the same factor), so a
// splat far from the cameras has an almost flat loss along "slide out along the viewing ray
// and grow in proportion". Adam normalizes x, y, z and the log-scales one by one and cannot
// follow the weak parallax signal along that direction. One scalar t per splat, with its own
// normalized step, moves the splat along it about an anchor a, the running centre of the
// cameras that see the splat:
//     x <- x + expm1(dt) (x - a),    log-scales <- log-scales + dt.
// The step is added after the ordinary optimizer step; geometry and gradient are those the
// image was rendered with.
namespace lfs::training::ray_depth {

    // Per-splat state row: anchor xyz, baseline spread B2, moments m and v. A row filled
    // with kUnobserved (negative v) belongs to a splat no camera has trained yet.
    inline constexpr int kStateStride = 6;
    inline constexpr float kUnobserved = -1.0f;
    inline constexpr float kBeta1 = 0.9f;
    inline constexpr float kBeta2 = 0.999f;
    // Weight of a new view in the running anchor and baseline spread.
    inline constexpr float kObserverRate = 1.0f / 32.0f;
    // Sideways shift allowed per step, in units of the row's position learning rate.
    inline constexpr float kSidewaysSteps = 10.0f / 3.0f;
    inline constexpr float kMinCrossVariance = 1e-30f;
    // Smallest J = B2 / vperp at which an outward move applies. Below it the observers' baseline spread is under
    // a tenth of the splat's footprint across the ray: the views cannot see a further move, so nothing in the loss
    // would stop the splat drifting out. Inward moves always apply.
    inline constexpr float kMinOutwardResolution = 0.01f;

    // A proposal that leaves the row alone, as stored between the two halves of an unfused step.
    inline constexpr float kNotObserved = 1e30f;

    struct Proposal {
        float dt = 0.0f;
        // Squared camera baseline across the viewing ray; negative: the view leaves the
        // observer statistics untouched.
        float bperp2 = -1.0f;
    };

    // Step 9, baseline part: B2 follows the running mean of bperp2. B2 is not used by the move,
    // so it is updated as soon as the proposal no longer reads it.
    LFS_RAY_DEPTH_FN void observe_baseline(float* row, const float bperp2) {
        row[3] = row[3] == 0.0f && bperp2 > 0.0f ? bperp2 : row[3] + (bperp2 - row[3]) * kObserverRate;
    }

    // expm1 keeps a small step of a far splat from rounding away against |x - a|.
    LFS_RAY_DEPTH_FN void move(float* mean, float* log_scale, const float* anchor, const float dt) {
        const float e = ::expm1f(dt);
        for (int k = 0; k < 3; ++k) {
            mean[k] += e * (mean[k] - anchor[k]);
            log_scale[k] += dt;
        }
    }

    // Steps 1 to 7 for one splat whose data gradient on its mean, g_x, is nonzero. lr is the
    // coordinate's step bound for this row; step_bound is the row's position learning rate in
    // world units times kSidewaysSteps (both zero for a frozen row). Updates the moments.
    LFS_RAY_DEPTH_FN Proposal propose(float* row, const float* x, const float* log_scale, const float* quat,
                                      const float* camera, const float* g_x, const float lr,
                                      const float step_bound) {
        Proposal p;
        if (!(g_x[0] != 0.0f || g_x[1] != 0.0f || g_x[2] != 0.0f) || !(step_bound > 0.0f))
            return p;
        if (row[5] < 0.0f) {
            for (int k = 0; k < 3; ++k)
                row[k] = camera[k];
            row[3] = row[4] = row[5] = 0.0f;
            return p;
        }
        float b[3], r[3];
        float bb = 0.0f, d2 = 0.0f;
        for (int k = 0; k < 3; ++k) {
            b[k] = camera[k] - row[k];
            r[k] = x[k] - camera[k];
            bb += b[k] * b[k];
            d2 += r[k] * r[k];
        }
        if (!(d2 > 0.0f))
            return p;
        const float inv_d = 1.0f / ::sqrtf(d2);
        float rb = 0.0f, g = 0.0f;
        for (int k = 0; k < 3; ++k) {
            r[k] *= inv_d;
            rb += r[k] * b[k];
            // The part of g_x that reaches the splat through the image, alpha and normal maps
            // satisfies (x - c).g_x + sum(g_log_scale) = 0, so b.g_x is its exact derivative along
            // the coordinate, (x - a).g_x + sum(g_log_scale). The part through the splat's own
            // rendered depth z (a depth-normal consistency prior) would add z dL/dz; it is left out
            // here and keeps acting through the ordinary step.
            g += b[k] * g_x[k];
        }
        p.bperp2 = ::fmaxf(bb - rb * rb, 0.0f);

        const float m = kBeta1 * row[4] + (1.0f - kBeta1) * g;
        const float v = row[5] == 0.0f ? g * g : kBeta2 * row[5] + (1.0f - kBeta2) * g * g;
        row[4] = m;
        row[5] = v;
        if (!(v > 0.0f)) {
            observe_baseline(row, p.bperp2);
            return p;
        }
        const float u = ::fminf(::fmaxf(m / ::sqrtf(v), -1.0f), 1.0f);
        const float dt0 = -lr * u;

        // Variance of the splat across the viewing ray, sum_k s_k^2 |R_k x r|^2 / 2, from
        // non-negative terms so a needle along the ray does not cancel.
        const float qn = quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3];
        const float s = qn > 0.0f ? 2.0f / qn : 0.0f;
        const float w = quat[0], qx = quat[1], qy = quat[2], qz = quat[3];
        const float axes[3][3] = {
            {1.0f - s * (qy * qy + qz * qz), s * (qx * qy + w * qz), s * (qx * qz - w * qy)},
            {s * (qx * qy - w * qz), 1.0f - s * (qx * qx + qz * qz), s * (qy * qz + w * qx)},
            {s * (qx * qz + w * qy), s * (qy * qz - w * qx), 1.0f - s * (qx * qx + qy * qy)}};
        float vperp = 0.0f;
        for (int k = 0; k < 3; ++k) {
            const float* a = axes[k];
            const float cx = a[1] * r[2] - a[2] * r[1];
            const float cy = a[2] * r[0] - a[0] * r[2];
            const float cz = a[0] * r[1] - a[1] * r[0];
            vperp += ::expf(2.0f * log_scale[k]) * (cx * cx + cy * cy + cz * cz);
        }
        vperp = ::fmaxf(0.5f * vperp, kMinCrossVariance);
        // B2 against the footprint: views that cannot tell the depth take the full inward step and
        // no outward one, a splat the ordinary step already triangulates fades out quadratically.
        const float resolution = row[3] / vperp;
        const float dt1 = dt0 > 0.0f && resolution < kMinOutwardResolution ? 0.0f : dt0 / (1.0f + resolution);
        // The typical observer may see the splat shift sideways by at most step_bound.
        const float bmax = ::sqrtf(::fmaxf(row[3], p.bperp2));
        const float q = bmax * dt1 / step_bound;
        p.dt = dt1 / ::sqrtf(1.0f + q * q);
        observe_baseline(row, p.bperp2);
        return p;
    }

    // An outward move stops where the splat would leave the first half of the rasterizer's
    // depth range (far_limit, world units) seen from this camera; inward moves always apply.
    LFS_RAY_DEPTH_FN bool outward_move_allowed(const float* mean, const float* anchor, const float* camera,
                                               const float dt, const float far_limit) {
        if (!(dt > 0.0f))
            return true;
        const float e = ::expm1f(dt);
        const float half = 0.5f * far_limit;
        float d2 = 0.0f;
        for (int k = 0; k < 3; ++k) {
            const float moved = mean[k] + e * (mean[k] - anchor[k]) - camera[k];
            d2 += moved * moved;
        }
        return d2 <= half * half;
    }

    // Steps 8 and 9, after the ordinary step: the move about the current anchor, then this
    // view enters the running anchor. observed: the proposal touched the row's statistics.
    LFS_RAY_DEPTH_FN void commit(float* row, float* mean, float* log_scale, const float* camera,
                                 const float dt, const bool observed, const float far_limit) {
        if (!observed)
            return;
        if (dt != 0.0f && outward_move_allowed(mean, row, camera, dt, far_limit))
            move(mean, log_scale, row, dt);
        for (int k = 0; k < 3; ++k)
            row[k] += (camera[k] - row[k]) * kObserverRate;
    }

    // The coordinate's share of the position step of a row with the given frozen / crop-box
    // damping; zero when the row takes no ordinary step.
    LFS_RAY_DEPTH_FN float row_lr_scale(const unsigned row,
                                        const bool* frozen_mask, const int frozen_mask_size,
                                        const float frozen_lr_scale,
                                        const bool* crop_mask, const int crop_mask_size,
                                        const float crop_lr_scale) {
        float scale = 1.0f;
        if (frozen_mask != nullptr && row < static_cast<unsigned>(frozen_mask_size) && frozen_mask[row])
            scale *= frozen_lr_scale;
        if (crop_mask != nullptr && row < static_cast<unsigned>(crop_mask_size) && crop_mask[row])
            scale *= crop_lr_scale;
        return scale;
    }

} // namespace lfs::training::ray_depth

#undef LFS_RAY_DEPTH_FN
