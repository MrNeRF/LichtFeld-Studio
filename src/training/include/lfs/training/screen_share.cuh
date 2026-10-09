/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cmath>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace lfs::training {

    // Dimensionless angular / screen-share of a 3D Gaussian. Saturates toward 1
    // when the splat swallows the camera (d ≲ r). Host and device.
    [[nodiscard]]
#ifdef __CUDACC__
    __host__ __device__
#endif
        inline float gaussian_screen_share(
            const float mean_x,
            const float mean_y,
            const float mean_z,
            const float cam_x,
            const float cam_y,
            const float cam_z,
            const float log_sx,
            const float log_sy,
            const float log_sz,
            const float opacity_raw) {
#ifdef __CUDA_ARCH__
        const float max_log = fmaxf(log_sx, fmaxf(log_sy, log_sz));
        const float opacity = 1.0f / (1.0f + expf(-opacity_raw));
        const float extend = sqrtf(2.0f * logf(fmaxf(255.0f * opacity, 1.0f)));
        const float r = expf(max_log) * extend;
        const float dx = mean_x - cam_x;
        const float dy = mean_y - cam_y;
        const float dz = mean_z - cam_z;
        const float d = sqrtf(dx * dx + dy * dy + dz * dz);
        const float denom = fmaxf(d, r) + sqrtf(fmaxf(d * d - r * r, 0.0f));
        if (!(denom > 0.0f) || !(r > 0.0f))
            return 0.0f;
        return fminf(fmaxf(r / denom, 0.0f), 1.0f);
#else
        const float max_log = std::fmax(log_sx, std::fmax(log_sy, log_sz));
        const float opacity = 1.0f / (1.0f + std::exp(-opacity_raw));
        const float extend = std::sqrt(2.0f * std::log(std::fmax(255.0f * opacity, 1.0f)));
        const float r = std::exp(max_log) * extend;
        const float dx = mean_x - cam_x;
        const float dy = mean_y - cam_y;
        const float dz = mean_z - cam_z;
        const float d = std::sqrt(dx * dx + dy * dy + dz * dz);
        const float denom = std::fmax(d, r) + std::sqrt(std::fmax(d * d - r * r, 0.0f));
        if (!(denom > 0.0f) || !(r > 0.0f))
            return 0.0f;
        const float share = r / denom;
        return share < 0.0f ? 0.0f : (share > 1.0f ? 1.0f : share);
#endif
    }

    [[nodiscard]]
#ifdef __CUDACC__
    __host__ __device__
#endif
        inline bool screen_share_cap_active(const float max_screen_share) {
        return max_screen_share > 0.0f && max_screen_share < 1.0f;
    }

#ifdef __CUDACC__
    __device__ __forceinline__ void atomic_max_float(float* addr, const float val) {
        int* const addr_i = reinterpret_cast<int*>(addr);
        int old = *addr_i;
        while (true) {
            const float old_f = __int_as_float(old);
            if (old_f >= val)
                return;
            const int assumed = old;
            old = atomicCAS(addr_i, assumed, __float_as_int(val));
            if (old == assumed)
                return;
        }
    }

    __device__ __forceinline__ float screen_share_hinge_extra_grad(
        const float share,
        const float limit,
        const float penalty,
        const float old_v,
        const float bias_correction2_sqrt_rcp,
        const float eps) {
        if (!(limit > 0.0f) || !(limit < 1.0f) || !(share > limit) || !(penalty > 0.0f))
            return 0.0f;
        const float hinge = penalty * log2f(share / limit);
        const float denom = sqrtf(old_v) * bias_correction2_sqrt_rcp + eps;
        return hinge * denom;
    }
#endif

} // namespace lfs::training
