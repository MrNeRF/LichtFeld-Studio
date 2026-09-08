/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

// Host/device adaptation of core/lod_merge_math.hpp's Jacobi procedure.
// Both decimation backends use this single adapter. Reference differences:
// binary64 scratch, 24 iterations, and JavaScript Math.sign(0) == 0.
// Keep the core LOD builders' float/10-iteration behavior untouched.
namespace lfs::io::decimate {
    DEC_HD inline void decompose(double* a, double* scales, double* quaternion) {
        double v[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        for (int iter = 0; iter < 24; ++iter) {
            int p = 0, q = 1;
            double max_abs = fabs(a[1]);
            if (fabs(a[2]) > max_abs) {
                p = 0;
                q = 2;
                max_abs = fabs(a[2]);
            }
            if (fabs(a[5]) > max_abs) {
                p = 1;
                q = 2;
                max_abs = fabs(a[5]);
            }
            if (max_abs < 1e-12)
                break;
            int pp = 3 * p + p, qq = 3 * q + q, pq = 3 * p + q;
            double app = a[pp], aqq = a[qq], apq = a[pq], tau = (aqq - app) / (2 * apq);
            double t = (double(tau > 0) - double(tau < 0)) / (fabs(tau) + sqrt(1 + tau * tau));
            double c = 1 / sqrt(1 + t * t), s = t * c;
            for (int k = 0; k < 3; ++k) {
                if (k == p || k == q)
                    continue;
                int kp = 3 * k + p, kq = 3 * k + q;
                double akp = a[kp], akq = a[kq];
                a[kp] = c * akp - s * akq;
                a[3 * p + k] = a[kp];
                a[kq] = s * akp + c * akq;
                a[3 * q + k] = a[kq];
            }
            a[pp] = c * c * app - 2 * s * c * apq + s * s * aqq;
            a[qq] = s * s * app + 2 * s * c * apq + c * c * aqq;
            a[pq] = a[3 * q + p] = 0;
            for (int k = 0; k < 3; ++k) {
                int kp = 3 * k + p, kq = 3 * k + q;
                double vkp = v[kp], vkq = v[kq];
                v[kp] = c * vkp - s * vkq;
                v[kq] = s * vkp + c * vkq;
            }
        }
        int order[3] = {0, 1, 2};
        for (int i = 1; i < 3; ++i) {
            int key = order[i], j = i;
            while (j > 0 && a[4 * key] > a[4 * order[j - 1]]) {
                order[j] = order[j - 1];
                --j;
            }
            order[j] = key;
        }
        double r[9];
        for (int c = 0; c < 3; ++c) {
            scales[c] = sqrt(hi(a[4 * order[c]], 1e-18));
            for (int row = 0; row < 3; ++row)
                r[row * 3 + c] = v[row * 3 + order[c]];
        }
        if (det(r) < 0) {
            r[2] *= -1;
            r[5] *= -1;
            r[8] *= -1;
        }
        double w, x, y, z, tr = r[0] + r[4] + r[8];
        if (tr > 0) {
            double s = sqrt(tr + 1) * 2;
            w = 0.25 * s;
            x = (r[7] - r[5]) / s;
            y = (r[2] - r[6]) / s;
            z = (r[3] - r[1]) / s;
        } else if (r[0] > r[4] && r[0] > r[8]) {
            double s = sqrt(1 + r[0] - r[4] - r[8]) * 2;
            w = (r[7] - r[5]) / s;
            x = 0.25 * s;
            y = (r[1] + r[3]) / s;
            z = (r[2] + r[6]) / s;
        } else if (r[4] > r[8]) {
            double s = sqrt(1 + r[4] - r[0] - r[8]) * 2;
            w = (r[2] - r[6]) / s;
            x = (r[1] + r[3]) / s;
            y = 0.25 * s;
            z = (r[5] + r[7]) / s;
        } else {
            double s = sqrt(1 + r[8] - r[0] - r[4]) * 2;
            w = (r[3] - r[1]) / s;
            x = (r[2] + r[6]) / s;
            y = (r[5] + r[7]) / s;
            z = 0.25 * s;
        }
        double inv = 1 / hi(sqrt(w * w + x * x + y * y + z * z), 1e-12);
        quaternion[0] = w * inv;
        quaternion[1] = x * inv;
        quaternion[2] = y * inv;
        quaternion[3] = z * inv;
    }
} // namespace lfs::io::decimate
