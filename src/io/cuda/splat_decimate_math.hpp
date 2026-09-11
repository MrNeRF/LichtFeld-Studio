/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "splat_decimate_internal.hpp"
#include <cmath>

#ifdef __CUDACC__
#define DEC_HD __host__ __device__
#else
#define DEC_HD
#endif

namespace lfs::io::decimate {
    // Reference covariance/mass cache entries are Float32Array, even though
    // formulas run in double. Keep these rounding points on both backends;
    // derived samples and self-densities retain their original double precision.
    struct Cache {
        float r[9], v[3], inv[3], sigma[9], logdet, mass;
        double sample[3], self_logpdf;
    };
    DEC_HD inline double hi(double a, double b) { return a > b ? a : b; }
    DEC_HD inline double lo(double a, double b) { return a < b ? a : b; }
    DEC_HD inline double sigmoid(double x) { return 1 / (1 + exp(-x)); }
    DEC_HD inline double area(double x, double y, double z) {
        return 12.566370614359172 * pow((pow(x * y, 1.6075) + pow(x * z, 1.6075) + pow(y * z, 1.6075)) / 3, 1 / 1.6075);
    }
    template <class T>
    DEC_HD inline void rotation(const float* q, T* r) {
        double w = q[0], x = q[1], y = q[2], z = q[3];
        double inv = 1 / hi(sqrt(w * w + x * x + y * y + z * z), 1e-12);
        w *= inv;
        x *= inv;
        y *= inv;
        z *= inv;
        r[0] = 1 - 2 * (y * y + z * z);
        r[1] = 2 * (x * y - w * z);
        r[2] = 2 * (x * z + w * y);
        r[3] = 2 * (x * y + w * z);
        r[4] = 1 - 2 * (x * x + z * z);
        r[5] = 2 * (y * z - w * x);
        r[6] = 2 * (x * z - w * y);
        r[7] = 2 * (y * z + w * x);
        r[8] = 1 - 2 * (x * x + y * y);
    }
    template <class R, class S>
    DEC_HD inline void covariance(const R* r, const double* v, S* s) {
        for (int a = 0; a < 3; ++a)
            for (int b = a; b < 3; ++b) {
                s[a * 3 + b] = double(r[a * 3]) * r[b * 3] * v[0] + double(r[a * 3 + 1]) * r[b * 3 + 1] * v[1] + double(r[a * 3 + 2]) * r[b * 3 + 2] * v[2];
                s[b * 3 + a] = s[a * 3 + b];
            }
    }
    DEC_HD inline double mass(View v, uint32_t i, double eps) {
        const float* s = v.scale + size_t(i) * 3;
        return sigmoid(v.opacity[i]) * area(hi(exp(double(s[0])), 1e-12), hi(exp(double(s[1])), 1e-12), hi(exp(double(s[2])), 1e-12)) + eps;
    }
    DEC_HD inline double logpdf(const double* x, const float* m, const Cache& c);
    DEC_HD inline Cache cache_one(View v, uint32_t i) {
        Cache c;
        double variance[3], ld = 0;
        for (int a = 0; a < 3; ++a) {
            double s = hi(exp(double(v.scale[size_t(i) * 3 + a])), 1e-12);
            variance[a] = s * s + 1e-8;
            c.v[a] = variance[a];
            c.inv[a] = 1 / hi(variance[a], 1e-30);
            ld += log(hi(variance[a], 1e-30));
        }
        c.logdet = ld;
        c.mass = mass(v, i, 1e-12);
        rotation(v.rot + size_t(i) * 4, c.r);
        covariance(c.r, variance, c.sigma);
        // Each edge uses the same fixed sample and its density under this
        // Gaussian. Cache them in double, retaining the original rounding.
        const double z[3] = {1.6264323081902676, 0.0033697340332619848, 1.0509958442185130};
        double scaled[3];
        for (int a = 0; a < 3; ++a)
            scaled[a] = z[a] * sqrt(hi(c.v[a], 0));
        const float* position = v.pos + size_t(i) * 3;
        for (int a = 0; a < 3; ++a)
            c.sample[a] = position[a] + scaled[0] * c.r[a * 3] + scaled[1] * c.r[a * 3 + 1] + scaled[2] * c.r[a * 3 + 2];
        c.self_logpdf = logpdf(c.sample, position, c);
        return c;
    }
    DEC_HD inline double det(const double* a) {
        return a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
    }
    DEC_HD inline double logpdf(const double* x, const float* m, const Cache& c) {
        double d[3] = {x[0] - m[0], x[1] - m[1], x[2] - m[2]}, q = 0;
        for (int a = 0; a < 3; ++a) {
            double y = d[0] * c.r[a] + d[1] * c.r[3 + a] + d[2] * c.r[6 + a];
            q += y * y * c.inv[a];
        }
        return -0.5 * (3 * 1.8378770664093453 + c.logdet + q);
    }
    DEC_HD inline double logadd(double a, double b) {
        double m = hi(a, b);
        return m + log(exp(a - m) + exp(b - m));
    }
    DEC_HD inline float color(View v, uint32_t i, int c) {
        return c < 3 ? v.dc[size_t(i) * 3 + c] : v.sh[size_t(i) * v.rest * 3 + c - 3];
    }
    DEC_HD inline float edge(View v, const Cache* cache, uint32_t i, uint32_t j) {
        const Cache &a = cache[i], &b = cache[j];
        double w = double(a.mass) + b.mass;
        double p = hi(1e-12, lo(1 - 1e-12, a.mass / (w > 0 ? w : 1))), q = 1 - p;
        double lp = log(p), lq = log(q), d[3], e[3], s[9];
        const float *u = v.pos + size_t(i) * 3, *t = v.pos + size_t(j) * 3;
        for (int c = 0; c < 3; ++c) {
            double m = p * u[c] + q * t[c];
            d[c] = u[c] - m;
            e[c] = t[c] - m;
        }
        for (int c = 0; c < 9; ++c)
            s[c] = p * a.sigma[c] + q * b.sigma[c] + p * d[c / 3] * d[c % 3] + q * e[c / 3] * e[c % 3];
        s[1] = s[3] = 0.5 * (s[1] + s[3]);
        s[2] = s[6] = 0.5 * (s[2] + s[6]);
        s[5] = s[7] = 0.5 * (s[5] + s[7]);
        s[0] += 1e-8;
        s[4] += 1e-8;
        s[8] += 1e-8;
        double cost = p * logadd(lp + a.self_logpdf, lq + logpdf(a.sample, t, b)) + q * logadd(lp + logpdf(b.sample, u, a), lq + b.self_logpdf) + 0.5 * (3 * 1.8378770664093453 + log(hi(det(s), 1e-30)) + 3);
        for (int c = 0; c < 3 + v.rest * 3; ++c) {
            double delta = double(color(v, i, c)) - color(v, j, c);
            cost += delta * delta;
        }
        return float(cost);
    }
    DEC_HD inline void decompose(double* s, double* scales, double* quaternion);
    DEC_HD inline void merge_one(View v, const uint32_t* ids, int count, View out, uint32_t row) {
        double weights[4], w = 0, mean[3] = {}, sig[9] = {};
        for (int m = 0; m < count; ++m) {
            weights[m] = mass(v, ids[m], 1e-30);
            w += weights[m];
        }
        for (int m = 0; m < count; ++m) {
            weights[m] /= w;
            for (int a = 0; a < 3; ++a)
                mean[a] += weights[m] * v.pos[size_t(ids[m]) * 3 + a];
        }
        for (int m = 0; m < count; ++m) {
            uint32_t i = ids[m];
            double r[9], variance[3], si[9], d[3];
            rotation(v.rot + size_t(i) * 4, r);
            for (int a = 0; a < 3; ++a) {
                double sc = hi(exp(double(v.scale[size_t(i) * 3 + a])), 1e-12);
                variance[a] = sc * sc;
                d[a] = v.pos[size_t(i) * 3 + a] - mean[a];
            }
            covariance(r, variance, si);
            for (int a = 0; a < 9; ++a)
                sig[a] += weights[m] * (d[a / 3] * d[a % 3] + si[a]);
        }
        sig[0] += 1e-8;
        sig[4] += 1e-8;
        sig[8] += 1e-8;
        double scales[3], quat[4];
        decompose(sig, scales, quat);
        for (int a = 0; a < 3; ++a) {
            out.pos[size_t(row) * 3 + a] = mean[a];
            out.scale[size_t(row) * 3 + a] = log(scales[a]);
        }
        for (int a = 0; a < 4; ++a)
            out.rot[size_t(row) * 4 + a] = quat[a];
        double alpha = hi(1e-7, lo(1 - 1e-7, w / hi(area(scales[0], scales[1], scales[2]), 1e-30)));
        out.opacity[row] = log(alpha / (1 - alpha));
        for (int c = 0; c < 3 + v.rest * 3; ++c) {
            double acc = 0;
            for (int m = 0; m < count; ++m)
                acc += weights[m] * color(v, ids[m], c);
            if (c < 3)
                out.dc[size_t(row) * 3 + c] = acc;
            else
                out.sh[size_t(row) * v.rest * 3 + c - 3] = acc;
        }
    }
    DEC_HD inline void copy_one(View v, uint32_t i, View out, uint32_t row) {
        for (int a = 0; a < 3; ++a) {
            out.pos[size_t(row) * 3 + a] = v.pos[size_t(i) * 3 + a];
            out.scale[size_t(row) * 3 + a] = v.scale[size_t(i) * 3 + a];
            out.dc[size_t(row) * 3 + a] = v.dc[size_t(i) * 3 + a];
        }
        for (int a = 0; a < 4; ++a)
            out.rot[size_t(row) * 4 + a] = v.rot[size_t(i) * 4 + a];
        out.opacity[row] = v.opacity[i];
        for (int a = 0; a < v.rest * 3; ++a)
            out.sh[size_t(row) * v.rest * 3 + a] = v.sh[size_t(i) * v.rest * 3 + a];
    }
} // namespace lfs::io::decimate

#include "splat_decimate_eigen.hpp"
#undef DEC_HD
