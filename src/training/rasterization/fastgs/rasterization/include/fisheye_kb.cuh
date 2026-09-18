/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// Kannala-Brandt (COLMAP OPENCV_FISHEYE) projection, Jacobian, and dJ/dP.
// Single source of truth for FastGS native fisheye: forward, backward, and tests
// all include this header. Do not duplicate these formulas elsewhere.
//
// Camera point P = (x, y, z) is in the OpenCV camera frame (P = W * mean).
// Depth for FISHEYE is ray length D = |P|, not camera-z. Gaussians past the
// polynomial foldover (theta_d' <= 0) are culled. The EWA affine footprint
// is knowingly approximate near the image rim. The fisheye footprint is
// composited without the 2D dilation.

#include <cmath>
#include <cuda_runtime.h>

#ifndef __CUDACC__
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

namespace fast_lfs::rasterization::fisheye_kb {

    constexpr double kPi = 3.14159265358979323846;
    constexpr double kThetaMaxMarginRad = 0.5 * kPi / 180.0; // 0.5 degree
    constexpr float kOnAxisDFloat = 1.0e-4f;
    constexpr double kOnAxisDDouble = 1.0e-8;
    constexpr float kMinRayLength = 1.0e-8f;

    template <typename T>
    __host__ __device__ inline T kb_on_axis_d() {
        if constexpr (sizeof(T) == sizeof(float)) {
            return static_cast<T>(kOnAxisDFloat);
        } else {
            return static_cast<T>(kOnAxisDDouble);
        }
    }

    template <typename T>
    __host__ __device__ inline T kb_abs(T x) {
#ifdef __CUDA_ARCH__
        if constexpr (sizeof(T) == sizeof(float)) {
            return fabsf(x);
        } else {
            return fabs(x);
        }
#else
        return std::abs(x);
#endif
    }

    template <typename T>
    __host__ __device__ inline T kb_sqrt(T x) {
#ifdef __CUDA_ARCH__
        if constexpr (sizeof(T) == sizeof(float)) {
            return sqrtf(x);
        } else {
            return sqrt(x);
        }
#else
        return std::sqrt(x);
#endif
    }

    template <typename T>
    __host__ __device__ inline T kb_hypot(T x, T y) {
        return kb_sqrt(x * x + y * y);
    }

    template <typename T>
    __host__ __device__ inline T kb_atan2(T y, T x) {
#ifdef __CUDA_ARCH__
        if constexpr (sizeof(T) == sizeof(float)) {
            return atan2f(y, x);
        } else {
            return atan2(y, x);
        }
#else
        return std::atan2(y, x);
#endif
    }

    template <typename T>
    __host__ __device__ inline T kb_min(T a, T b) {
        return a < b ? a : b;
    }

    template <typename T>
    __host__ __device__ inline T kb_max(T a, T b) {
        return a > b ? a : b;
    }

    // theta_d = theta * (1 + t2*(k1 + t2*(k2 + t2*(k3 + t2*k4)))), t2 = theta^2
    template <typename T>
    __host__ __device__ inline T kb_theta_d(T theta, T k1, T k2, T k3, T k4) {
        const T t2 = theta * theta;
        return theta * (T(1) + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))));
    }

    // d(theta_d)/d(theta)
    template <typename T>
    __host__ __device__ inline T kb_theta_d_prime(T theta, T k1, T k2, T k3, T k4) {
        const T t2 = theta * theta;
        return T(1) + t2 * (T(3) * k1 + t2 * (T(5) * k2 + t2 * (T(7) * k3 + t2 * T(9) * k4)));
    }

    // d^2(theta_d)/d(theta)^2
    template <typename T>
    __host__ __device__ inline T kb_theta_d_dbl_prime(T theta, T k1, T k2, T k3, T k4) {
        const T t2 = theta * theta;
        return theta * (T(6) * k1 + t2 * (T(20) * k2 + t2 * (T(42) * k3 + t2 * T(72) * k4)));
    }

    template <typename T>
    struct KbProjection {
        T px, py;
        T J00, J01, J02, J10, J11, J12;
        T theta, theta_d, theta_d_prime, theta_d_dbl_prime;
        T x, y, z, d, D, D2, d2;
        T A, B;
        bool on_axis;
        bool finite;
    };

    template <typename T>
    __host__ __device__ inline KbProjection<T> kb_project(
        T x, T y, T z,
        T fx, T fy, T cx, T cy,
        T k1, T k2, T k3, T k4) {
        KbProjection<T> o{};
        o.x = x;
        o.y = y;
        o.z = z;
        o.d2 = x * x + y * y;
        o.d = kb_sqrt(o.d2);
        o.D2 = o.d2 + z * z;
        o.D = kb_sqrt(o.D2);
        o.theta = kb_atan2(o.d, z);
        o.px = cx;
        o.py = cy;
        o.finite = o.D >= static_cast<T>(kMinRayLength);
        o.on_axis = o.d < kb_on_axis_d<T>();

        if (!o.finite) {
            return o;
        }

        if (o.on_axis) {
            // Closed-form on-axis limit: general formulas are 0/0.
            // J = diag(fx/z, fy/z, 0), pixel = (cx, cy). Same branch on host
            // and device. |z| below the ray-length guard is not finite.
            if (kb_abs(z) < static_cast<T>(kMinRayLength)) {
                o.finite = false;
                return o;
            }
            const T inv_z = T(1) / z;
            o.J00 = fx * inv_z;
            o.J01 = T(0);
            o.J02 = T(0);
            o.J10 = T(0);
            o.J11 = fy * inv_z;
            o.J12 = T(0);
            o.theta_d = kb_theta_d(o.theta, k1, k2, k3, k4);
            o.theta_d_prime = kb_theta_d_prime(o.theta, k1, k2, k3, k4);
            o.theta_d_dbl_prime = kb_theta_d_dbl_prime(o.theta, k1, k2, k3, k4);
            o.A = T(0);
            o.B = T(0);
            return o;
        }

        o.theta_d = kb_theta_d(o.theta, k1, k2, k3, k4);
        o.theta_d_prime = kb_theta_d_prime(o.theta, k1, k2, k3, k4);
        o.theta_d_dbl_prime = kb_theta_d_dbl_prime(o.theta, k1, k2, k3, k4);

        o.px = cx + fx * o.theta_d * x / o.d;
        o.py = cy + fy * o.theta_d * y / o.d;

        o.A = o.theta_d_prime / (o.D2 * o.d2);
        o.B = o.theta_d / (o.d * o.d * o.d);

        const T Axyz = o.A * x * y * z;
        const T Bxy = o.B * x * y;
        o.J00 = fx * (o.A * x * x * z + o.B * y * y);
        o.J01 = fx * (Axyz - Bxy);
        o.J02 = -fx * o.theta_d_prime * x / o.D2;
        o.J10 = fy * (Axyz - Bxy);
        o.J11 = fy * (o.A * y * y * z + o.B * x * x);
        o.J12 = -fy * o.theta_d_prime * y / o.D2;
        return o;
    }

    // First theta in (0, pi) where theta_d' <= 0, else pi.
    template <typename T>
    __host__ __device__ inline T kb_theta_foldover(T k1, T k2, T k3, T k4) {
        const T pi = static_cast<T>(kPi);
        constexpr int kSamples = 128;
        T prev = T(0);
        for (int i = 1; i <= kSamples; ++i) {
            const T th = pi * static_cast<T>(i) / static_cast<T>(kSamples);
            if (kb_theta_d_prime(th, k1, k2, k3, k4) <= T(0)) {
                T lo = prev;
                T hi = th;
                for (int j = 0; j < 40; ++j) {
                    const T mid = T(0.5) * (lo + hi);
                    if (kb_theta_d_prime(mid, k1, k2, k3, k4) <= T(0)) {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                return hi;
            }
            prev = th;
        }
        return pi;
    }

    // min(foldover, theta whose image radius reaches the farthest corner) + 0.5 deg,
    // never exceeding foldover. Corner solve uses f = max(fx, fy).
    template <typename T>
    __host__ __device__ inline T kb_theta_max(
        T k1, T k2, T k3, T k4,
        T width, T height,
        T fx, T fy, T cx, T cy) {
        const T fold = kb_theta_foldover(k1, k2, k3, k4);
        const T r00 = kb_hypot(cx, cy);
        const T r10 = kb_hypot(width - cx, cy);
        const T r01 = kb_hypot(cx, height - cy);
        const T r11 = kb_hypot(width - cx, height - cy);
        const T r_target = kb_max(kb_max(r00, r10), kb_max(r01, r11));
        const T f = kb_max(fx, fy); // conservative when fx != fy

        const T fold_safe = fold * T(0.999999);
        T theta_corner;
        if (f * kb_theta_d(fold_safe, k1, k2, k3, k4) < r_target) {
            theta_corner = fold;
        } else {
            T lo = T(0);
            T hi = fold;
            for (int i = 0; i < 50; ++i) {
                const T mid = T(0.5) * (lo + hi);
                if (f * kb_theta_d(mid, k1, k2, k3, k4) < r_target) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            theta_corner = hi;
            for (int i = 0; i < 6; ++i) {
                const T g = f * kb_theta_d(theta_corner, k1, k2, k3, k4) - r_target;
                const T gp = f * kb_theta_d_prime(theta_corner, k1, k2, k3, k4);
                if (kb_abs(gp) < T(1e-20)) {
                    break;
                }
                const T next = theta_corner - g / gp;
                if (next <= T(0) || next >= fold) {
                    break;
                }
                theta_corner = next;
            }
        }

        T tmax = kb_min(fold, theta_corner) + static_cast<T>(kThetaMaxMarginRad);
        // Stay strictly below foldover so theta_d' stays positive in kernels.
        if (tmax >= fold) {
            tmax = fold_safe;
        }
        if (tmax < T(1e-4)) {
            tmax = T(1e-4);
        }
        return tmax;
    }

    // Host wrapper: theta_max in double with the full image and unadjusted
    // principal point (not a training tile).
    __host__ inline float kb_theta_max_from_intrinsics(
        float k1, float k2, float k3, float k4,
        int width, int height,
        float fx, float fy, float cx, float cy) {
        return static_cast<float>(kb_theta_max(
            static_cast<double>(k1), static_cast<double>(k2),
            static_cast<double>(k3), static_cast<double>(k4),
            static_cast<double>(width), static_cast<double>(height),
            static_cast<double>(fx), static_cast<double>(fy),
            static_cast<double>(cx), static_cast<double>(cy)));
    }

    // Shared dJ/dP temporaries. dC, dE, dF, dGx, dGy are derivatives of the
    // unscaled J factors (J00 = fx*C, J01 = fx*E, J02 = fx*Gx, ...).
    template <typename T>
    struct KbJPartials {
        T dC[3];
        T dE[3];
        T dF[3];
        T dGx[3];
        T dGy[3];
        bool on_axis;
        T inv_z2;
    };

    template <typename T>
    __host__ __device__ inline KbJPartials<T> kb_j_partials(const KbProjection<T>& p) {
        KbJPartials<T> o{};
        o.on_axis = p.on_axis;
        if (!p.finite) {
            return o;
        }
        if (p.on_axis) {
            o.inv_z2 = T(1) / (p.z * p.z);
            return o;
        }

        const T x = p.x, y = p.y, z = p.z;
        const T d = p.d, D2 = p.D2, d2 = p.d2;
        const T tdp = p.theta_d_prime;
        const T td = p.theta_d;
        const T tdpp = p.theta_d_dbl_prime;
        const T A = p.A;
        const T B = p.B;
        const T inv_D2 = T(1) / D2;
        const T inv_d = T(1) / d;

        // dtheta/dP = (x z, y z, -d^2) / (D^2 d)
        const T dth_x = x * z * inv_D2 * inv_d;
        const T dth_y = y * z * inv_D2 * inv_d;
        const T dth_z = -d * inv_D2;
        const T dtdp_x = tdpp * dth_x;
        const T dtdp_y = tdpp * dth_y;
        const T dtdp_z = tdpp * dth_z;
        const T dtd_x = tdp * dth_x;
        const T dtd_y = tdp * dth_y;
        const T dtd_z = tdp * dth_z;

        const T dD2_x = T(2) * x, dD2_y = T(2) * y, dD2_z = T(2) * z;
        const T dd2_x = T(2) * x, dd2_y = T(2) * y;
        const T dd_x = x * inv_d, dd_y = y * inv_d;

        // A = theta_d' / (D^2 d^2), B = theta_d / d^3. Form dA/dB without
        // dividing by theta_d' or theta_d (those hit 0 at foldover / origin).
        const T inv_D2_d2 = inv_D2 / d2;
        const T inv_d3 = inv_d * inv_d * inv_d;
        const T dA_x = dtdp_x * inv_D2_d2 - A * (dD2_x * inv_D2 + dd2_x / d2);
        const T dA_y = dtdp_y * inv_D2_d2 - A * (dD2_y * inv_D2 + dd2_y / d2);
        const T dA_z = dtdp_z * inv_D2_d2 - A * (dD2_z * inv_D2);
        const T dB_x = dtd_x * inv_d3 - T(3) * B * dd_x * inv_d;
        const T dB_y = dtd_y * inv_d3 - T(3) * B * dd_y * inv_d;
        const T dB_z = dtd_z * inv_d3;

        // C = A x^2 z + B y^2
        o.dC[0] = dA_x * x * x * z + A * T(2) * x * z + dB_x * y * y;
        o.dC[1] = dA_y * x * x * z + dB_y * y * y + B * T(2) * y;
        o.dC[2] = dA_z * x * x * z + A * x * x + dB_z * y * y;
        // E = A x y z - B x y
        o.dE[0] = dA_x * x * y * z + A * y * z - dB_x * x * y - B * y;
        o.dE[1] = dA_y * x * y * z + A * x * z - dB_y * x * y - B * x;
        o.dE[2] = dA_z * x * y * z + A * x * y - dB_z * x * y;
        // F = A y^2 z + B x^2
        o.dF[0] = dA_x * y * y * z + dB_x * x * x + B * T(2) * x;
        o.dF[1] = dA_y * y * y * z + A * T(2) * y * z + dB_y * x * x;
        o.dF[2] = dA_z * y * y * z + A * y * y + dB_z * x * x;
        // Gx = -theta_d' * x / D^2
        const T inv_D2_sq = inv_D2 * inv_D2;
        o.dGx[0] = -dtdp_x * x * inv_D2 - tdp * inv_D2 + tdp * x * T(2) * x * inv_D2_sq;
        o.dGx[1] = -dtdp_y * x * inv_D2 + tdp * x * T(2) * y * inv_D2_sq;
        o.dGx[2] = -dtdp_z * x * inv_D2 + tdp * x * T(2) * z * inv_D2_sq;
        // Gy = -theta_d' * y / D^2
        o.dGy[0] = -dtdp_x * y * inv_D2 + tdp * y * T(2) * x * inv_D2_sq;
        o.dGy[1] = -dtdp_y * y * inv_D2 - tdp * inv_D2 + tdp * y * T(2) * y * inv_D2_sq;
        o.dGy[2] = -dtdp_z * y * inv_D2 + tdp * y * T(2) * z * inv_D2_sq;
        return o;
    }

    // dJ_ij / dP_k as a (2,3,3) tensor. Used by tests against golden dJdP_fd.
    template <typename T>
    __host__ __device__ inline void kb_dJ_dP(
        const KbProjection<T>& p,
        T fx, T fy,
        T dJdP[2][3][3]) {
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    dJdP[i][j][k] = T(0);
                }
            }
        }
        const KbJPartials<T> g = kb_j_partials(p);
        if (!p.finite) {
            return;
        }
        if (g.on_axis) {
            dJdP[0][0][2] = -fx * g.inv_z2;
            dJdP[1][1][2] = -fy * g.inv_z2;
            return;
        }
        for (int k = 0; k < 3; ++k) {
            dJdP[0][0][k] = fx * g.dC[k];
            dJdP[0][1][k] = fx * g.dE[k];
            dJdP[0][2][k] = fx * g.dGx[k];
            dJdP[1][0][k] = fy * g.dE[k];
            dJdP[1][1][k] = fy * g.dF[k];
            dJdP[1][2][k] = fy * g.dGy[k];
        }
    }

    // Contract dL_dJ through dJ/dP into dL_dP. Shared temporaries computed once.
    template <typename T>
    __host__ __device__ inline void kb_contract_dL_dJ(
        const KbProjection<T>& p,
        T fx, T fy,
        T dL_dJ00, T dL_dJ01, T dL_dJ02,
        T dL_dJ10, T dL_dJ11, T dL_dJ12,
        T& dL_dx, T& dL_dy, T& dL_dz) {
        const KbJPartials<T> g = kb_j_partials(p);
        if (!p.finite) {
            return;
        }
        if (g.on_axis) {
            dL_dz += dL_dJ00 * (-fx * g.inv_z2) + dL_dJ11 * (-fy * g.inv_z2);
            return;
        }
        const T sx0 = fx * dL_dJ00;
        const T sx1 = fx * dL_dJ01;
        const T sx2 = fx * dL_dJ02;
        const T sy0 = fy * dL_dJ10;
        const T sy1 = fy * dL_dJ11;
        const T sy2 = fy * dL_dJ12;
        dL_dx += sx0 * g.dC[0] + sx1 * g.dE[0] + sx2 * g.dGx[0] + sy0 * g.dE[0] + sy1 * g.dF[0] + sy2 * g.dGy[0];
        dL_dy += sx0 * g.dC[1] + sx1 * g.dE[1] + sx2 * g.dGx[1] + sy0 * g.dE[1] + sy1 * g.dF[1] + sy2 * g.dGy[1];
        dL_dz += sx0 * g.dC[2] + sx1 * g.dE[2] + sx2 * g.dGx[2] + sy0 * g.dE[2] + sy1 * g.dF[2] + sy2 * g.dGy[2];
    }

} // namespace fast_lfs::rasterization::fisheye_kb
