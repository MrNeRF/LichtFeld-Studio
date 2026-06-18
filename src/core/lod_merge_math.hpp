/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Shared scalar math for the LOD tree builders (bhatt_lod.cpp, octree_lod.cpp).
// Moved verbatim from bhatt_lod.cpp so both builders merge with bit-identical
// conventions (eigen decomposition, quaternion order, lodOpacity weights).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace lfs::core::lodmath {

    constexpr float kMinScale = 1e-12f;
    constexpr float kMinQuatNorm = 1e-12f;
    constexpr float kEllipsoidAreaP = 1.6075f;
    constexpr int kJacobiIterations = 10;
    constexpr float kMinEval = 1e-18f;
    constexpr float kEpsCov = 1e-8f;
    constexpr float SH_C0 = 0.28209479177387814f;
    constexpr float kPi = 3.14159265358979323846f;

    [[nodiscard]] inline float sigmoid(const float x) {
        if (x >= 0.0f) {
            const float z = std::exp(-x);
            return 1.0f / (1.0f + z);
        }
        const float z = std::exp(x);
        return z / (1.0f + z);
    }

    [[nodiscard]] inline float clamp_scale_raw(const float raw) {
        return std::clamp(raw, -30.0f, 30.0f);
    }

    [[nodiscard]] inline float activated_scale(const float raw) {
        return std::max(std::exp(clamp_scale_raw(raw)), kMinScale);
    }

    [[nodiscard]] inline float ellipsoid_area(const float sx, const float sy, const float sz) {
        const float t1 = std::pow(sx * sy, kEllipsoidAreaP);
        const float t2 = std::pow(sx * sz, kEllipsoidAreaP);
        const float t3 = std::pow(sy * sz, kEllipsoidAreaP);
        return 4.0f * kPi * std::pow((t1 + t2 + t3) / 3.0f, 1.0f / kEllipsoidAreaP);
    }

    [[nodiscard]] inline float lod_opacity(const float opacity) {
        if (opacity > 1.0f) {
            constexpr float kE = 2.718281828459045f;
            return std::sqrt(1.0f + kE * std::log(opacity));
        }
        return 1.0f;
    }

    inline void quat_to_rotmat(const float qw, const float qx, const float qy, const float qz, std::array<float, 9>& out) {
        const float xx = qx * qx;
        const float yy = qy * qy;
        const float zz = qz * qz;
        const float wx = qw * qx;
        const float wy = qw * qy;
        const float wz = qw * qz;
        const float xy = qx * qy;
        const float xz = qx * qz;
        const float yz = qy * qz;

        out[0] = 1.0f - 2.0f * (yy + zz);
        out[1] = 2.0f * (xy - wz);
        out[2] = 2.0f * (xz + wy);
        out[3] = 2.0f * (xy + wz);
        out[4] = 1.0f - 2.0f * (xx + zz);
        out[5] = 2.0f * (yz - wx);
        out[6] = 2.0f * (xz - wy);
        out[7] = 2.0f * (yz + wx);
        out[8] = 1.0f - 2.0f * (xx + yy);
    }

    inline void sigma_from_rot_var(const std::array<float, 9>& R,
                                   const float vx,
                                   const float vy,
                                   const float vz,
                                   std::array<float, 9>& out) {
        const std::array<float, 3> variance = {vx, vy, vz};
        std::array<float, 9> scaled{};
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                const size_t idx = static_cast<size_t>(row * 3 + col);
                scaled[idx] = R[idx] * variance[static_cast<size_t>(col)];
            }
        }
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                out[static_cast<size_t>(row * 3 + col)] =
                    scaled[static_cast<size_t>(row * 3 + 0)] * R[static_cast<size_t>(col * 3 + 0)] +
                    scaled[static_cast<size_t>(row * 3 + 1)] * R[static_cast<size_t>(col * 3 + 1)] +
                    scaled[static_cast<size_t>(row * 3 + 2)] * R[static_cast<size_t>(col * 3 + 2)];
            }
        }
    }

    [[nodiscard]] inline float det3(const std::array<float, 9>& A) {
        return A[0] * (A[4] * A[8] - A[5] * A[7]) -
               A[1] * (A[3] * A[8] - A[5] * A[6]) +
               A[2] * (A[3] * A[7] - A[4] * A[6]);
    }

    struct Eigen3x3 {
        std::array<float, 3> values{};
        std::array<float, 9> vectors{};
    };

    [[nodiscard]] inline Eigen3x3 sort_eigendecomposition(const Eigen3x3& out) {
        std::array<int, 3> order = {0, 1, 2};
        std::sort(order.begin(), order.end(), [&](const int lhs, const int rhs) {
            if (out.values[static_cast<size_t>(lhs)] != out.values[static_cast<size_t>(rhs)])
                return out.values[static_cast<size_t>(lhs)] > out.values[static_cast<size_t>(rhs)];
            return lhs < rhs;
        });

        Eigen3x3 sorted;
        for (int col = 0; col < 3; ++col) {
            const int src_col = order[static_cast<size_t>(col)];
            sorted.values[static_cast<size_t>(col)] = out.values[static_cast<size_t>(src_col)];
            for (int row = 0; row < 3; ++row)
                sorted.vectors[static_cast<size_t>(row * 3 + col)] = out.vectors[static_cast<size_t>(row * 3 + src_col)];
        }

        if (det3(sorted.vectors) < 0.0f) {
            sorted.vectors[2] *= -1.0f;
            sorted.vectors[5] *= -1.0f;
            sorted.vectors[8] *= -1.0f;
        }
        return sorted;
    }

    [[nodiscard]] inline Eigen3x3 eigen_symmetric_3x3_jacobi(const std::array<float, 9>& Ain) {
        std::array<float, 9> A = Ain;
        std::array<float, 9> V = {
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
            0.0f,
            0.0f,
            0.0f,
            1.0f,
        };

        for (int iter = 0; iter < kJacobiIterations; ++iter) {
            int p = 0;
            int q = 1;
            float max_abs = std::abs(A[1]);
            if (std::abs(A[2]) > max_abs) {
                p = 0;
                q = 2;
                max_abs = std::abs(A[2]);
            }
            if (std::abs(A[5]) > max_abs) {
                p = 1;
                q = 2;
                max_abs = std::abs(A[5]);
            }
            if (max_abs < 1e-12f)
                break;

            const int pp = 3 * p + p;
            const int qq = 3 * q + q;
            const int pq = 3 * p + q;
            const float app = A[static_cast<size_t>(pp)];
            const float aqq = A[static_cast<size_t>(qq)];
            const float apq = A[static_cast<size_t>(pq)];
            const float tau = (aqq - app) / (2.0f * apq);
            const float t = std::copysign(1.0f, tau) / (std::abs(tau) + std::sqrt(1.0f + tau * tau));
            const float c = 1.0f / std::sqrt(1.0f + t * t);
            const float s = t * c;

            for (int k = 0; k < 3; ++k) {
                if (k == p || k == q)
                    continue;
                const int kp = 3 * k + p;
                const int kq = 3 * k + q;
                const float akp = A[static_cast<size_t>(kp)];
                const float akq = A[static_cast<size_t>(kq)];
                A[static_cast<size_t>(kp)] = c * akp - s * akq;
                A[static_cast<size_t>(3 * p + k)] = A[static_cast<size_t>(kp)];
                A[static_cast<size_t>(kq)] = s * akp + c * akq;
                A[static_cast<size_t>(3 * q + k)] = A[static_cast<size_t>(kq)];
            }

            A[static_cast<size_t>(pp)] = c * c * app - 2.0f * s * c * apq + s * s * aqq;
            A[static_cast<size_t>(qq)] = s * s * app + 2.0f * s * c * apq + c * c * aqq;
            A[static_cast<size_t>(pq)] = 0.0f;
            A[static_cast<size_t>(3 * q + p)] = 0.0f;

            for (int k = 0; k < 3; ++k) {
                const int kp = 3 * k + p;
                const int kq = 3 * k + q;
                const float vkp = V[static_cast<size_t>(kp)];
                const float vkq = V[static_cast<size_t>(kq)];
                V[static_cast<size_t>(kp)] = c * vkp - s * vkq;
                V[static_cast<size_t>(kq)] = s * vkp + c * vkq;
            }
        }

        Eigen3x3 out;
        out.values = {A[0], A[4], A[8]};
        out.vectors = V;
        return sort_eigendecomposition(out);
    }

    [[nodiscard]] inline Eigen3x3 eigen_symmetric_3x3(const std::array<float, 9>& Ain) {
        return eigen_symmetric_3x3_jacobi(Ain);
    }

    inline void rotmat_to_quat(const std::array<float, 9>& R, std::array<float, 4>& out) {
        const float m00 = R[0];
        const float m11 = R[4];
        const float m22 = R[8];
        const float tr = m00 + m11 + m22;
        float qw = 0.0f;
        float qx = 0.0f;
        float qy = 0.0f;
        float qz = 0.0f;

        if (tr > 0.0f) {
            const float S = std::sqrt(tr + 1.0f) * 2.0f;
            qw = 0.25f * S;
            qx = (R[7] - R[5]) / S;
            qy = (R[2] - R[6]) / S;
            qz = (R[3] - R[1]) / S;
        } else if (R[0] > R[4] && R[0] > R[8]) {
            const float S = std::sqrt(1.0f + R[0] - R[4] - R[8]) * 2.0f;
            qw = (R[7] - R[5]) / S;
            qx = 0.25f * S;
            qy = (R[1] + R[3]) / S;
            qz = (R[2] + R[6]) / S;
        } else if (R[4] > R[8]) {
            const float S = std::sqrt(1.0f + R[4] - R[0] - R[8]) * 2.0f;
            qw = (R[2] - R[6]) / S;
            qx = (R[1] + R[3]) / S;
            qy = 0.25f * S;
            qz = (R[5] + R[7]) / S;
        } else {
            const float S = std::sqrt(1.0f + R[8] - R[0] - R[4]) * 2.0f;
            qw = (R[3] - R[1]) / S;
            qx = (R[2] + R[6]) / S;
            qy = (R[5] + R[7]) / S;
            qz = 0.25f * S;
        }

        const float inv_n = 1.0f / std::max(std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz), kMinQuatNorm);
        out[0] = qw * inv_n;
        out[1] = qx * inv_n;
        out[2] = qy * inv_n;
        out[3] = qz * inv_n;
    }

    inline void decompose_sigma_to_raw_scale_quat(const std::array<float, 9>& sigma,
                                                  std::array<float, 3>& scaling_raw,
                                                  std::array<float, 4>& rotation_raw) {
        const auto eig = eigen_symmetric_3x3(sigma);
        std::array<float, 3> evals = {
            std::max(eig.values[0], kMinEval),
            std::max(eig.values[1], kMinEval),
            std::max(eig.values[2], kMinEval),
        };

        scaling_raw[0] = std::log(std::max(std::sqrt(evals[0]), kMinScale));
        scaling_raw[1] = std::log(std::max(std::sqrt(evals[1]), kMinScale));
        scaling_raw[2] = std::log(std::max(std::sqrt(evals[2]), kMinScale));
        rotmat_to_quat(eig.vectors, rotation_raw);
    }

    // Bhattacharyya-distance similarity with color modulation; cov holds the 6
    // unique covariance elements {xx, xy, xz, yy, yz, zz}. Moved verbatim from
    // BhattLodWorkset::similarity so both builders pair with identical metrics.
    [[nodiscard]] inline float bhatt_similarity(
        const float* cov_a, const float det_a, const float* mean_a, const float* rgb_a,
        const float* cov_b, const float det_b, const float* mean_b, const float* rgb_b) {
        const float m00 = 0.5f * (cov_a[0] + cov_b[0]);
        const float m01 = 0.5f * (cov_a[1] + cov_b[1]);
        const float m02 = 0.5f * (cov_a[2] + cov_b[2]);
        const float m11 = 0.5f * (cov_a[3] + cov_b[3]);
        const float m12 = 0.5f * (cov_a[4] + cov_b[4]);
        const float m22 = 0.5f * (cov_a[5] + cov_b[5]);

        const float C00 = m11 * m22 - m12 * m12;
        const float C01 = m02 * m12 - m01 * m22;
        const float C02 = m01 * m12 - m02 * m11;
        const float C11 = m00 * m22 - m02 * m02;
        const float C12 = m01 * m02 - m00 * m12;
        const float C22 = m00 * m11 - m01 * m01;

        const float det = m00 * C00 + m01 * C01 + m02 * C02;
        const float det_sigma = det;

        if (det_sigma <= kEpsCov || det_a <= kEpsCov || det_b <= kEpsCov ||
            !std::isfinite(det_sigma) || !std::isfinite(det_a) || !std::isfinite(det_b)) {
            return 0.0f;
        }

        if (std::abs(det) < kEpsCov) {
            return 0.0f;
        }

        const float inv_det = 1.0f / det;
        const float inv_xx = C00 * inv_det;
        const float inv_yy = C11 * inv_det;
        const float inv_zz = C22 * inv_det;
        const float inv_xy = C01 * inv_det;
        const float inv_xz = C02 * inv_det;
        const float inv_yz = C12 * inv_det;

        const float dx = mean_b[0] - mean_a[0];
        const float dy = mean_b[1] - mean_a[1];
        const float dz = mean_b[2] - mean_a[2];

        const float quad = inv_xx * dx * dx + inv_yy * dy * dy + inv_zz * dz * dz + 2.0f * inv_xy * dx * dy + 2.0f * inv_xz * dx * dz + 2.0f * inv_yz * dy * dz;

        const float term1 = 0.125f * quad;
        const float term2 = 0.5f * std::log(det_sigma / std::sqrt(det_a * det_b));
        const float distance = term1 + term2;
        const float spatial = std::exp(-distance);

        const float dr = rgb_a[0] - rgb_b[0];
        const float dg = rgb_a[1] - rgb_b[1];
        const float db = rgb_a[2] - rgb_b[2];
        const float color_delta2 = dr * dr + dg * dg + db * db;

        const float metric = spatial * std::exp(-color_delta2);
        if (std::isnan(metric) || !std::isfinite(metric)) {
            return 0.0f;
        }
        return metric;
    }

    // Compute symmetric 3x3 covariance from scale + quaternion and store 6 unique elements + det
    inline void compute_covariance_from_scale_quat(
        float sx, float sy, float sz,
        float qw, float qx, float qy, float qz,
        float& out_xx, float& out_xy, float& out_xz,
        float& out_yy, float& out_yz, float& out_zz,
        float& out_det) {

        std::array<float, 9> R;
        quat_to_rotmat(qw, qx, qy, qz, R);
        const float sx2 = sx * sx;
        const float sy2 = sy * sy;
        const float sz2 = sz * sz;
        std::array<float, 9> cov;
        sigma_from_rot_var(R, sx2, sy2, sz2, cov);

        out_xx = cov[0];
        out_xy = cov[1];
        out_xz = cov[2];
        out_yy = cov[4];
        out_yz = cov[5];
        out_zz = cov[8];
        out_det = det3(cov);
    }

} // namespace lfs::core::lodmath
