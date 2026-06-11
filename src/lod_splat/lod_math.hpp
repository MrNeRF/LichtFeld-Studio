/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Minimal self-contained math for the LOD splat module.
// Deliberately avoids lfs_tensor (CUDA-only) and external deps so the module
// builds standalone on any host, including macOS where the main app cannot.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace lfs::lod {

    struct Vec3 {
        float x = 0.f, y = 0.f, z = 0.f;

        Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
        Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
        Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
        Vec3& operator+=(const Vec3& o) {
            x += o.x;
            y += o.y;
            z += o.z;
            return *this;
        }
        float dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
        Vec3 cross(const Vec3& o) const {
            return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
        }
        float norm() const { return std::sqrt(dot(*this)); }
        Vec3 normalized() const {
            const float n = norm();
            return n > 1e-20f ? (*this) * (1.f / n) : Vec3{0, 0, 1};
        }
        Vec3 cwiseMin(const Vec3& o) const { return {std::min(x, o.x), std::min(y, o.y), std::min(z, o.z)}; }
        Vec3 cwiseMax(const Vec3& o) const { return {std::max(x, o.x), std::max(y, o.y), std::max(z, o.z)}; }
        float maxComp() const { return std::max(x, std::max(y, z)); }
    };

    // Column-major 3x3, m[col][row] flattened as m[3*col + row].
    struct Mat3 {
        float m[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

        static Mat3 identity() { return {}; }
        float& at(int r, int c) { return m[3 * c + r]; }
        float at(int r, int c) const { return m[3 * c + r]; }

        Vec3 operator*(const Vec3& v) const {
            return {m[0] * v.x + m[3] * v.y + m[6] * v.z,
                    m[1] * v.x + m[4] * v.y + m[7] * v.z,
                    m[2] * v.x + m[5] * v.y + m[8] * v.z};
        }
        Mat3 operator*(const Mat3& o) const {
            Mat3 r;
            for (int c = 0; c < 3; ++c)
                for (int row = 0; row < 3; ++row)
                    r.at(row, c) = at(row, 0) * o.at(0, c) + at(row, 1) * o.at(1, c) + at(row, 2) * o.at(2, c);
            return r;
        }
        Mat3 transposed() const {
            Mat3 r;
            for (int c = 0; c < 3; ++c)
                for (int row = 0; row < 3; ++row)
                    r.at(row, c) = at(c, row);
            return r;
        }
        float det() const {
            return at(0, 0) * (at(1, 1) * at(2, 2) - at(1, 2) * at(2, 1)) -
                   at(0, 1) * (at(1, 0) * at(2, 2) - at(1, 2) * at(2, 0)) +
                   at(0, 2) * (at(1, 0) * at(2, 1) - at(1, 1) * at(2, 0));
        }
    };

    // Quaternion stored (w, x, y, z) — same convention as LichtFeld's PLY rot_0..rot_3.
    struct Quat {
        float w = 1.f, x = 0.f, y = 0.f, z = 0.f;

        Quat normalized() const {
            const float n = std::sqrt(w * w + x * x + y * y + z * z);
            if (n < 1e-20f)
                return {1, 0, 0, 0};
            return {w / n, x / n, y / n, z / n};
        }

        Mat3 toMat3() const {
            const Quat q = normalized();
            Mat3 r;
            const float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
            const float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
            const float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
            r.at(0, 0) = 1 - 2 * (yy + zz);
            r.at(0, 1) = 2 * (xy - wz);
            r.at(0, 2) = 2 * (xz + wy);
            r.at(1, 0) = 2 * (xy + wz);
            r.at(1, 1) = 1 - 2 * (xx + zz);
            r.at(1, 2) = 2 * (yz - wx);
            r.at(2, 0) = 2 * (xz - wy);
            r.at(2, 1) = 2 * (yz + wx);
            r.at(2, 2) = 1 - 2 * (xx + yy);
            return r;
        }

        static Quat fromMat3(const Mat3& a) {
            Quat q;
            const float tr = a.at(0, 0) + a.at(1, 1) + a.at(2, 2);
            if (tr > 0.f) {
                float s = std::sqrt(tr + 1.f) * 2.f;
                q.w = 0.25f * s;
                q.x = (a.at(2, 1) - a.at(1, 2)) / s;
                q.y = (a.at(0, 2) - a.at(2, 0)) / s;
                q.z = (a.at(1, 0) - a.at(0, 1)) / s;
            } else if (a.at(0, 0) > a.at(1, 1) && a.at(0, 0) > a.at(2, 2)) {
                float s = std::sqrt(1.f + a.at(0, 0) - a.at(1, 1) - a.at(2, 2)) * 2.f;
                q.w = (a.at(2, 1) - a.at(1, 2)) / s;
                q.x = 0.25f * s;
                q.y = (a.at(0, 1) + a.at(1, 0)) / s;
                q.z = (a.at(0, 2) + a.at(2, 0)) / s;
            } else if (a.at(1, 1) > a.at(2, 2)) {
                float s = std::sqrt(1.f + a.at(1, 1) - a.at(0, 0) - a.at(2, 2)) * 2.f;
                q.w = (a.at(0, 2) - a.at(2, 0)) / s;
                q.x = (a.at(0, 1) + a.at(1, 0)) / s;
                q.y = 0.25f * s;
                q.z = (a.at(1, 2) + a.at(2, 1)) / s;
            } else {
                float s = std::sqrt(1.f + a.at(2, 2) - a.at(0, 0) - a.at(1, 1)) * 2.f;
                q.w = (a.at(1, 0) - a.at(0, 1)) / s;
                q.x = (a.at(0, 2) + a.at(2, 0)) / s;
                q.y = (a.at(1, 2) + a.at(2, 1)) / s;
                q.z = 0.25f * s;
            }
            return q.normalized();
        }
    };

    // Symmetric 3x3 stored as upper triangle: xx, xy, xz, yy, yz, zz.
    struct Sym3 {
        float v[6] = {0, 0, 0, 0, 0, 0};

        static Sym3 fromCovariance(const Mat3& rot, const Vec3& scale) {
            // Sigma = R * diag(s^2) * R^T
            Mat3 rs = rot;
            for (int row = 0; row < 3; ++row) {
                rs.at(row, 0) *= scale.x;
                rs.at(row, 1) *= scale.y;
                rs.at(row, 2) *= scale.z;
            }
            const Mat3 sig = rs * rs.transposed();
            return {{sig.at(0, 0), sig.at(0, 1), sig.at(0, 2), sig.at(1, 1), sig.at(1, 2), sig.at(2, 2)}};
        }

        Mat3 toMat3() const {
            Mat3 m;
            m.at(0, 0) = v[0];
            m.at(0, 1) = v[1];
            m.at(0, 2) = v[2];
            m.at(1, 0) = v[1];
            m.at(1, 1) = v[3];
            m.at(1, 2) = v[4];
            m.at(2, 0) = v[2];
            m.at(2, 1) = v[4];
            m.at(2, 2) = v[5];
            return m;
        }

        Sym3& addScaled(const Sym3& o, float s) {
            for (int i = 0; i < 6; ++i)
                v[i] += o.v[i] * s;
            return *this;
        }
        Sym3& addOuterScaled(const Vec3& d, float s) {
            v[0] += s * d.x * d.x;
            v[1] += s * d.x * d.y;
            v[2] += s * d.x * d.z;
            v[3] += s * d.y * d.y;
            v[4] += s * d.y * d.z;
            v[5] += s * d.z * d.z;
            return *this;
        }
        Sym3& scale(float s) {
            for (int i = 0; i < 6; ++i)
                v[i] *= s;
            return *this;
        }
    };

    inline float detSym3(const Sym3& s) {
        return s.v[0] * (s.v[3] * s.v[5] - s.v[4] * s.v[4]) -
               s.v[1] * (s.v[1] * s.v[5] - s.v[4] * s.v[2]) +
               s.v[2] * (s.v[1] * s.v[4] - s.v[3] * s.v[2]);
    }

    // Bhattacharyya distance between Gaussians (mu_a, A) and (mu_b, B) given the
    // precomputed determinants of A and B. Lower = more similar. This is the
    // similarity LichtFeld's bhatt_lod builder clusters by.
    inline float bhattDistance(const Vec3& da, const Sym3& A, float detA, const Sym3& B, float detB) {
        Sym3 M; // (A + B) / 2
        for (int i = 0; i < 6; ++i)
            M.v[i] = 0.5f * (A.v[i] + B.v[i]);
        const float detM = detSym3(M);
        if (detM <= 1e-30f || detA <= 1e-30f || detB <= 1e-30f)
            return 1e30f;
        // solve M x = da via adjugate
        const float inv = 1.f / detM;
        const float m00 = (M.v[3] * M.v[5] - M.v[4] * M.v[4]) * inv;
        const float m01 = (M.v[2] * M.v[4] - M.v[1] * M.v[5]) * inv;
        const float m02 = (M.v[1] * M.v[4] - M.v[2] * M.v[3]) * inv;
        const float m11 = (M.v[0] * M.v[5] - M.v[2] * M.v[2]) * inv;
        const float m12 = (M.v[1] * M.v[2] - M.v[0] * M.v[4]) * inv;
        const float m22 = (M.v[0] * M.v[3] - M.v[1] * M.v[1]) * inv;
        const float xx = m00 * da.x + m01 * da.y + m02 * da.z;
        const float xy = m01 * da.x + m11 * da.y + m12 * da.z;
        const float xz = m02 * da.x + m12 * da.y + m22 * da.z;
        const float maha = da.x * xx + da.y * xy + da.z * xz;
        return 0.125f * maha + 0.5f * std::log(detM / std::sqrt(detA * detB));
    }

    // Jacobi eigendecomposition of a symmetric 3x3. Returns eigenvalues in `eval`
    // (descending) and corresponding eigenvectors as columns of `evec` (right-handed).
    inline void eigenSym3(const Sym3& s, Vec3& eval, Mat3& evec) {
        float a[3][3] = {{s.v[0], s.v[1], s.v[2]}, {s.v[1], s.v[3], s.v[4]}, {s.v[2], s.v[4], s.v[5]}};
        float q[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        for (int iter = 0; iter < 32; ++iter) {
            // largest off-diagonal
            int p = 0, r = 1;
            float off = std::fabs(a[0][1]);
            if (std::fabs(a[0][2]) > off) {
                off = std::fabs(a[0][2]);
                p = 0;
                r = 2;
            }
            if (std::fabs(a[1][2]) > off) {
                off = std::fabs(a[1][2]);
                p = 1;
                r = 2;
            }
            if (off < 1e-12f)
                break;
            const float app = a[p][p], arr = a[r][r], apr = a[p][r];
            const float theta = 0.5f * (arr - app) / apr;
            const float t = (theta >= 0 ? 1.f : -1.f) / (std::fabs(theta) + std::sqrt(theta * theta + 1.f));
            const float c = 1.f / std::sqrt(t * t + 1.f), sn = t * c;
            for (int k = 0; k < 3; ++k) {
                const float akp = a[k][p], akr = a[k][r];
                a[k][p] = c * akp - sn * akr;
                a[k][r] = sn * akp + c * akr;
            }
            for (int k = 0; k < 3; ++k) {
                const float apk = a[p][k], ark = a[r][k];
                a[p][k] = c * apk - sn * ark;
                a[r][k] = sn * apk + c * ark;
            }
            for (int k = 0; k < 3; ++k) {
                const float qkp = q[k][p], qkr = q[k][r];
                q[k][p] = c * qkp - sn * qkr;
                q[k][r] = sn * qkp + c * qkr;
            }
        }
        int order[3] = {0, 1, 2};
        float ev[3] = {a[0][0], a[1][1], a[2][2]};
        std::sort(order, order + 3, [&](int i, int j) { return ev[i] > ev[j]; });
        eval = {ev[order[0]], ev[order[1]], ev[order[2]]};
        for (int c = 0; c < 3; ++c)
            for (int row = 0; row < 3; ++row)
                evec.at(row, c) = q[row][order[c]];
        // enforce right-handed basis so the quaternion conversion is valid
        if (evec.det() < 0.f)
            for (int row = 0; row < 3; ++row)
                evec.at(row, 2) = -evec.at(row, 2);
    }

    inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
    inline float logit(float p) {
        p = std::clamp(p, 1e-5f, 1.f - 1e-5f);
        return std::log(p / (1.f - p));
    }

    // 21-bit-per-axis Morton code (63 bits total).
    inline uint64_t expandBits21(uint64_t v) {
        v &= 0x1fffff;
        v = (v | v << 32) & 0x1f00000000ffffULL;
        v = (v | v << 16) & 0x1f0000ff0000ffULL;
        v = (v | v << 8) & 0x100f00f00f00f00fULL;
        v = (v | v << 4) & 0x10c30c30c30c30c3ULL;
        v = (v | v << 2) & 0x1249249249249249ULL;
        return v;
    }
    inline uint64_t morton3D(const Vec3& p, const Vec3& bbMin, const Vec3& bbInvExtent) {
        const float fx = std::clamp((p.x - bbMin.x) * bbInvExtent.x, 0.f, 1.f) * ((1 << 21) - 1);
        const float fy = std::clamp((p.y - bbMin.y) * bbInvExtent.y, 0.f, 1.f) * ((1 << 21) - 1);
        const float fz = std::clamp((p.z - bbMin.z) * bbInvExtent.z, 0.f, 1.f) * ((1 << 21) - 1);
        return (expandBits21((uint64_t)fx) << 2) | (expandBits21((uint64_t)fy) << 1) | expandBits21((uint64_t)fz);
    }

} // namespace lfs::lod
