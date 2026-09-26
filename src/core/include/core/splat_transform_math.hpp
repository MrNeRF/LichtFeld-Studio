/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <cmath>

#ifdef __CUDACC__
#define LFS_TRANSFORM_INLINE __host__ __device__ inline
#else
#define LFS_TRANSFORM_INLINE inline
#endif

namespace lfs::core::splat_transform {
    struct LinearTransform {
        float rows[9];
    };

    // Factor A R diag(scale) directly with one-sided Jacobi rotations. Forming
    // and diagonalizing its covariance instead squares the condition number
    // and loses the short axes of very thin splats. Right rotations preserve
    // B B^T, and the resulting orthogonal columns are the exported splat axes.
    LFS_TRANSFORM_INLINE void affine_geometry(const LinearTransform& a,
                                              const float* log_scale, const float* quaternion,
                                              float* result_scale, float* result_quaternion) {
        double w = quaternion[0], x = quaternion[1], y = quaternion[2], z = quaternion[3];
        const double qnorm = sqrt(w * w + x * x + y * y + z * z);
        if (qnorm > 0) {
            w /= qnorm;
            x /= qnorm;
            y /= qnorm;
            z /= qnorm;
        } else {
            w = 1;
            x = y = z = 0;
        }
        const double rotation[9] = {
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)};
        const double largest_log = fmax(double(log_scale[0]), fmax(double(log_scale[1]), double(log_scale[2])));
        if (largest_log == -INFINITY) {
            // Activated zero scales have log-scale -inf. Avoid -inf - -inf
            // when all three axes have zero extent.
            for (int j = 0; j < 3; ++j)
                result_scale[j] = -INFINITY;
            result_quaternion[0] = 1;
            for (int j = 1; j < 4; ++j)
                result_quaternion[j] = 0;
            return;
        }
        double b[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                b[3 * i + j] = 0;
                for (int k = 0; k < 3; ++k)
                    b[3 * i + j] += double(a.rows[3 * i + k]) * rotation[3 * k + j];
                b[3 * i + j] *= exp(double(log_scale[j]) - largest_log);
            }
        for (int sweep = 0; sweep < 12; ++sweep) {
            bool changed = false;
            for (int p = 0; p < 2; ++p)
                for (int q = p + 1; q < 3; ++q) {
                    double aa = 0, bb = 0, ab = 0;
                    for (int k = 0; k < 3; ++k) {
                        aa += b[3 * k + p] * b[3 * k + p];
                        bb += b[3 * k + q] * b[3 * k + q];
                        ab += b[3 * k + p] * b[3 * k + q];
                    }
                    if (fabs(ab) <= 1e-14 * sqrt(aa) * sqrt(bb))
                        continue;
                    const double delta = 0.5 * (bb - aa);
                    const double t = ab / (delta + copysign(hypot(delta, ab), delta));
                    const double c = 1 / sqrt(1 + t * t), s = t * c;
                    for (int k = 0; k < 3; ++k) {
                        const double u = b[3 * k + p], v = b[3 * k + q];
                        b[3 * k + p] = c * u - s * v;
                        b[3 * k + q] = s * u + c * v;
                    }
                    changed = true;
                }
            if (!changed)
                break;
        }
        double length[3];
        for (int j = 0; j < 3; ++j)
            length[j] = hypot(hypot(b[j], b[3 + j]), b[6 + j]);
        // Descending axes give deterministic output and put null axes last.
        for (int p = 0; p < 2; ++p)
            for (int q = p + 1; q < 3; ++q)
                if (length[p] < length[q]) {
                    double tmp = length[p];
                    length[p] = length[q];
                    length[q] = tmp;
                    for (int k = 0; k < 3; ++k) {
                        tmp = b[3 * k + p];
                        b[3 * k + p] = b[3 * k + q];
                        b[3 * k + q] = tmp;
                    }
                }
        for (int j = 0; j < 3; ++j)
            result_scale[j] = float(log(length[j]) + largest_log);
        double u[3] = {1, 0, 0}, v[3] = {0, 1, 0};
        if (length[0] > 0)
            for (int k = 0; k < 3; ++k)
                u[k] = b[3 * k] / length[0];
        if (length[1] > 0) {
            for (int k = 0; k < 3; ++k)
                v[k] = b[3 * k + 1] / length[1];
        } else {
            int axis = fabs(u[0]) < fabs(u[1]) ? 0 : 1;
            if (fabs(u[2]) < fabs(u[axis]))
                axis = 2;
            for (int k = 0; k < 3; ++k)
                v[k] = (k == axis ? 1.0 : 0.0) - u[k] * u[axis];
            const double norm = hypot(hypot(v[0], v[1]), v[2]);
            for (int k = 0; k < 3; ++k)
                v[k] /= norm;
        }
        // A right-handed frame also represents reflections: changing an axis
        // sign leaves the Gaussian covariance unchanged.
        const double r[9] = {u[0], v[0], u[1] * v[2] - u[2] * v[1],
                             u[1], v[1], u[2] * v[0] - u[0] * v[2],
                             u[2], v[2], u[0] * v[1] - u[1] * v[0]};
        double q[4];
        if (r[0] + r[4] + r[8] > 0) {
            const double s = 2 * sqrt(1 + r[0] + r[4] + r[8]);
            q[0] = s / 4;
            q[1] = (r[7] - r[5]) / s;
            q[2] = (r[2] - r[6]) / s;
            q[3] = (r[3] - r[1]) / s;
        } else {
            int i = r[0] > r[4] ? 0 : 1;
            if (r[8] > r[3 * i + i])
                i = 2;
            const int j = (i + 1) % 3, k = (i + 2) % 3;
            const double s = 2 * sqrt(1 + r[3 * i + i] - r[3 * j + j] - r[3 * k + k]);
            q[0] = (r[3 * k + j] - r[3 * j + k]) / s;
            q[i + 1] = s / 4;
            q[j + 1] = (r[3 * j + i] + r[3 * i + j]) / s;
            q[k + 1] = (r[3 * k + i] + r[3 * i + k]) / s;
        }
        const double norm = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        for (int i = 0; i < 4; ++i)
            result_quaternion[i] = float(q[i] / (q[0] < 0 ? -norm : norm));
    }
} // namespace lfs::core::splat_transform

#undef LFS_TRANSFORM_INLINE
