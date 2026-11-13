/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// A camera's pose offset (r, d) from its start pose W0 | b0 (row-major 4x4 world-to-camera), in the start camera's
// axes, r in radians and d the centre's move in scene units:
//   W = E W0,  b = E (b0 - d),  E = exp(-[r]x),  centre c = -W^T b.
// The rasterizer reports dL/d(w2c) and dL/d(centre) as independent inputs; offset_grad chains them to dL/d(r, d).

#include <cmath>

#if defined(__CUDACC__)
#define LFS_POSE_HD __host__ __device__ __forceinline__
#else
#define LFS_POSE_HD inline
#endif

namespace lfs::training::camera_pose {

    // exp([phi]x), row-major; exactly the identity at phi = 0.
    template <class T>
    LFS_POSE_HD void exp_so3(const T phi[3], T E[9]) {
        const T th2 = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
        T A, B;
        if (th2 < T(1e-8)) {
            A = T(1) - th2 / T(6);
            B = T(0.5) - th2 / T(24);
        } else {
            const T th = std::sqrt(th2);
            A = std::sin(th) / th;
            B = (T(1) - std::cos(th)) / th2;
        }
        const T x = phi[0], y = phi[1], z = phi[2];
        E[0] = T(1) + B * (x * x - th2);
        E[1] = B * x * y - A * z;
        E[2] = B * x * z + A * y;
        E[3] = B * x * y + A * z;
        E[4] = T(1) + B * (y * y - th2);
        E[5] = B * y * z - A * x;
        E[6] = B * x * z - A * y;
        E[7] = B * y * z + A * x;
        E[8] = T(1) + B * (z * z - th2);
    }

    template <class T>
    LFS_POSE_HD void compose(const T w2c0[16], const T off[6], T w2c[16], T c[3]) {
        const T phi[3] = {-off[0], -off[1], -off[2]};
        T E[9];
        exp_so3(phi, E);
        const T u[3] = {w2c0[3] - off[3], w2c0[7] - off[4], w2c0[11] - off[5]};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j)
                w2c[i * 4 + j] = E[i * 3] * w2c0[j] + E[i * 3 + 1] * w2c0[4 + j] + E[i * 3 + 2] * w2c0[8 + j];
            w2c[i * 4 + 3] = E[i * 3] * u[0] + E[i * 3 + 1] * u[1] + E[i * 3 + 2] * u[2];
        }
        for (int j = 0; j < 4; ++j)
            w2c[12 + j] = w2c0[12 + j];
        for (int j = 0; j < 3; ++j)
            c[j] = -(w2c[j] * w2c[3] + w2c[4 + j] * w2c[7] + w2c[8 + j] * w2c[11]);
    }

    // dL/d(dr, dd) of the left perturbation W(δ) = exp(-[dr]x) W, b(δ) = exp(-[dr]x) (b - dd), c(δ) = c + W^T dd
    // from g_w2c (rows 0-2 of dL/d(W | b)) and g_c = dL/dc.
    template <class T>
    LFS_POSE_HD void perturbation_grad(const T w2c[16], const T g_w2c[16], const T g_c[3], T g[6]) {
        // N_ij = W_i . (dL/dW)_j
        T N[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                N[i][j] = w2c[i * 4] * g_w2c[j * 4] + w2c[i * 4 + 1] * g_w2c[j * 4 + 1] + w2c[i * 4 + 2] * g_w2c[j * 4 + 2];
        const T b[3] = {w2c[3], w2c[7], w2c[11]};
        const T gb[3] = {g_w2c[3], g_w2c[7], g_w2c[11]};
        g[0] = N[2][1] - N[1][2] + gb[1] * b[2] - gb[2] * b[1];
        g[1] = N[0][2] - N[2][0] + gb[2] * b[0] - gb[0] * b[2];
        g[2] = N[1][0] - N[0][1] + gb[0] * b[1] - gb[1] * b[0];
        for (int i = 0; i < 3; ++i)
            g[3 + i] = -gb[i] + w2c[i * 4] * g_c[0] + w2c[i * 4 + 1] * g_c[1] + w2c[i * 4 + 2] * g_c[2];
    }

    // dL/d(dr, dd) of the left perturbation at the offset -> dL/d(r, d):
    // dL/dr = J_l(-r)^T dL/ddr (exact SO(3) left Jacobian), dL/dd = E^T dL/ddd.
    template <class T>
    LFS_POSE_HD void chain(const T off[6], const T g[6], T out[6]) {
        const T phi[3] = {-off[0], -off[1], -off[2]};
        const T th2 = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
        T a, b;
        if (th2 < T(1e-6)) {
            a = T(0.5) - th2 / T(24);
            b = T(1) / T(6) - th2 / T(120);
        } else {
            const T th = std::sqrt(th2);
            a = (T(1) - std::cos(th)) / th2;
            b = (th - std::sin(th)) / (th2 * th);
        }
        // J_l(phi)^T g = g - a (phi x g) + b (phi x (phi x g))
        const T pg[3] = {phi[1] * g[2] - phi[2] * g[1], phi[2] * g[0] - phi[0] * g[2], phi[0] * g[1] - phi[1] * g[0]};
        const T ppg[3] = {phi[1] * pg[2] - phi[2] * pg[1], phi[2] * pg[0] - phi[0] * pg[2], phi[0] * pg[1] - phi[1] * pg[0]};
        for (int k = 0; k < 3; ++k)
            out[k] = g[k] - a * pg[k] + b * ppg[k];
        T E[9];
        exp_so3(phi, E);
        for (int k = 0; k < 3; ++k)
            out[3 + k] = E[k] * g[3] + E[3 + k] * g[4] + E[6 + k] * g[5];
    }

    // dL/d(r, d) at the offset from the rasterizer's gradient of the pose it drew, compose(w2c0, off).
    template <class T>
    LFS_POSE_HD void offset_grad(const T w2c0[16], const T off[6], const T g_w2c[16], const T g_c[3], T out[6]) {
        T w2c[16], c[3], g[6];
        compose(w2c0, off, w2c, c);
        perturbation_grad(w2c, g_w2c, g_c, g);
        chain(off, g, out);
    }

} // namespace lfs::training::camera_pose
