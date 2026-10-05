/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda_error.hpp"
#include "simplify_merge.hpp"

#include <cfloat>

// One thread per group: the moment-matching merge of splat_simplify.cpp (merge_voxel_groups), with the
// same order of accumulation over the members.
namespace lfs::core::tensor_ops {
    namespace {
        constexpr float kEpsCov = 1e-8f;
        constexpr float kMinScale = 1e-12f;
        constexpr float kMinQuatNorm = 1e-12f;
        constexpr float kMinEval = 1e-18f;
        constexpr int kJacobiIterations = 32;
        constexpr unsigned kBlock = 128;

        __device__ void quat_to_rotmat(const float qw, const float qx, const float qy, const float qz, float* out) {
            const float xx = qx * qx, yy = qy * qy, zz = qz * qz;
            const float wx = qw * qx, wy = qw * qy, wz = qw * qz;
            const float xy = qx * qy, xz = qx * qz, yz = qy * qz;
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

        __device__ void sigma_from_rot_var(const float* R, const float vx, const float vy, const float vz, float* out) {
            const float variance[3] = {vx, vy, vz};
            float scaled[9];
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 3; ++col)
                    scaled[row * 3 + col] = __fmul_rn(R[row * 3 + col], variance[col]);
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 3; ++col) {
                    float sum = 0.0f;
                    sum = fmaf(scaled[row * 3 + 0], R[col * 3 + 0], sum);
                    sum = fmaf(scaled[row * 3 + 1], R[col * 3 + 1], sum);
                    sum = fmaf(scaled[row * 3 + 2], R[col * 3 + 2], sum);
                    out[row * 3 + col] = sum;
                }
        }

        __device__ float det3(const float* A) {
            return A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6]) +
                   A[2] * (A[3] * A[7] - A[4] * A[6]);
        }

        // Jacobi eigendecomposition, eigenvalues descending (ties by column), right-handed vectors.
        __device__ void eigen_symmetric(const float* input, float* values, float* vectors) {
            float A[9], V[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
            for (int i = 0; i < 9; ++i)
                A[i] = input[i];
            for (int iter = 0; iter < kJacobiIterations; ++iter) {
                int p = 0, q = 1;
                float max_abs = fabsf(A[1]);
                if (fabsf(A[2]) > max_abs) {
                    p = 0;
                    q = 2;
                    max_abs = fabsf(A[2]);
                }
                if (fabsf(A[5]) > max_abs) {
                    p = 1;
                    q = 2;
                    max_abs = fabsf(A[5]);
                }
                if (max_abs < 1e-12f)
                    break;
                const int pp = 3 * p + p, qq = 3 * q + q, pq = 3 * p + q;
                const float app = A[pp], aqq = A[qq], apq = A[pq];
                const float tau = (aqq - app) / (2.0f * apq);
                const float t = copysignf(1.0f, tau) / (fabsf(tau) + sqrtf(1.0f + tau * tau));
                const float c = 1.0f / sqrtf(1.0f + t * t);
                const float s = t * c;
                for (int k = 0; k < 3; ++k) {
                    if (k == p || k == q)
                        continue;
                    const int kp = 3 * k + p, kq = 3 * k + q;
                    const float akp = A[kp], akq = A[kq];
                    A[kp] = c * akp - s * akq;
                    A[3 * p + k] = A[kp];
                    A[kq] = s * akp + c * akq;
                    A[3 * q + k] = A[kq];
                }
                A[pp] = c * c * app - 2.0f * s * c * apq + s * s * aqq;
                A[qq] = s * s * app + 2.0f * s * c * apq + c * c * aqq;
                A[pq] = 0.0f;
                A[3 * q + p] = 0.0f;
                for (int k = 0; k < 3; ++k) {
                    const int kp = 3 * k + p, kq = 3 * k + q;
                    const float vkp = V[kp], vkq = V[kq];
                    V[kp] = c * vkp - s * vkq;
                    V[kq] = s * vkp + c * vkq;
                }
            }
            const float raw[3] = {A[0], A[4], A[8]};
            int order[3] = {0, 1, 2};
            for (int i = 1; i < 3; ++i)
                for (int j = i; j > 0; --j) {
                    const int lhs = order[j], rhs = order[j - 1];
                    if (!(raw[lhs] > raw[rhs] || (raw[lhs] == raw[rhs] && lhs < rhs)))
                        break;
                    order[j] = rhs;
                    order[j - 1] = lhs;
                }
            for (int col = 0; col < 3; ++col) {
                values[col] = raw[order[col]];
                for (int row = 0; row < 3; ++row)
                    vectors[row * 3 + col] = V[row * 3 + order[col]];
            }
            if (det3(vectors) < 0.0f) {
                vectors[2] *= -1.0f;
                vectors[5] *= -1.0f;
                vectors[8] *= -1.0f;
            }
        }

        __device__ void rotmat_to_quat(const float* R, float* out) {
            const float tr = R[0] + R[4] + R[8];
            float qw, qx, qy, qz;
            if (tr > 0.0f) {
                const float S = sqrtf(tr + 1.0f) * 2.0f;
                qw = 0.25f * S;
                qx = (R[7] - R[5]) / S;
                qy = (R[2] - R[6]) / S;
                qz = (R[3] - R[1]) / S;
            } else if (R[0] > R[4] && R[0] > R[8]) {
                const float S = sqrtf(1.0f + R[0] - R[4] - R[8]) * 2.0f;
                qw = (R[7] - R[5]) / S;
                qx = 0.25f * S;
                qy = (R[1] + R[3]) / S;
                qz = (R[2] + R[6]) / S;
            } else if (R[4] > R[8]) {
                const float S = sqrtf(1.0f + R[4] - R[0] - R[8]) * 2.0f;
                qw = (R[2] - R[6]) / S;
                qx = (R[1] + R[3]) / S;
                qy = 0.25f * S;
                qz = (R[5] + R[7]) / S;
            } else {
                const float S = sqrtf(1.0f + R[8] - R[0] - R[4]) * 2.0f;
                qw = (R[3] - R[1]) / S;
                qx = (R[2] + R[6]) / S;
                qy = (R[5] + R[7]) / S;
                qz = 0.25f * S;
            }
            const float inv_n = 1.0f / fmaxf(sqrtf(qw * qw + qx * qx + qy * qy + qz * qz), kMinQuatNorm);
            out[0] = qw * inv_n;
            out[1] = qx * inv_n;
            out[2] = qy * inv_n;
            out[3] = qz * inv_n;
        }

        // decompose_sigma_to_raw_scale_quat, then activated_scale of each raw scale.
        __device__ void decompose(const float* sigma, const bool allow_collapsed, float* scales, float* rotation) {
            float values[3], vectors[9];
            eigen_symmetric(sigma, values, vectors);
            const float zero_tolerance =
                allow_collapsed ? fmaxf(fmaxf(fmaxf(values[0], values[1]), values[2]), 0.0f) * (8.0f * FLT_EPSILON) : 0.0f;
            for (int axis = 0; axis < 3; ++axis) {
                const float variance = fmaxf(values[axis], allow_collapsed ? 0.0f : kMinEval);
                if (variance <= zero_tolerance) {
                    scales[axis] = 0.0f;
                } else {
                    const float raw = logf(fmaxf(sqrtf(variance), kMinScale));
                    scales[axis] = fmaxf(expf(fminf(fmaxf(raw, -30.0f), 30.0f)), kMinScale);
                }
            }
            rotmat_to_quat(vectors, rotation);
        }

        __device__ float member_weight(const float* scales, const float* opacity, const int32_t row, bool& collapsed) {
            const float* s = scales + static_cast<size_t>(row) * 3;
            collapsed = s[0] == 0.0f || s[1] == 0.0f || s[2] == 0.0f;
            const float volume = fmaxf(s[0], kMinScale) * fmaxf(s[1], kMinScale) * fmaxf(s[2], kMinScale);
            return volume * opacity[row];
        }

        __global__ void simplify_merge_kernel(const float* __restrict__ means, const float* __restrict__ scales,
                                              const float* __restrict__ rotation, const float* __restrict__ opacity,
                                              const float* __restrict__ appearance, const int32_t* __restrict__ offsets,
                                              const int32_t* __restrict__ members, float* __restrict__ out_means,
                                              float* __restrict__ out_scales, float* __restrict__ out_rotation,
                                              float* __restrict__ out_opacity, float* __restrict__ out_appearance,
                                              const uint32_t groups, const uint32_t app_dim) {
            const uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
            if (g >= groups)
                return;
            const int32_t begin = offsets[g], end = offsets[g + 1];
            const size_t o = g;
            if (end - begin == 1) {
                const size_t row = static_cast<size_t>(members[begin]);
                for (int a = 0; a < 3; ++a) {
                    out_means[o * 3 + a] = means[row * 3 + a];
                    out_scales[o * 3 + a] = scales[row * 3 + a];
                }
                for (int a = 0; a < 4; ++a)
                    out_rotation[o * 4 + a] = rotation[row * 4 + a];
                out_opacity[o] = opacity[row];
                for (uint32_t k = 0; k < app_dim; ++k)
                    out_appearance[o * app_dim + k] = appearance[row * app_dim + k];
                return;
            }
            // Weights by volume and opacity; collapsed members get normalized relative weights.
            float total = 0.0f, largest = -FLT_MAX;
            bool has_collapsed = false;
            for (int32_t m = begin; m < end; ++m) {
                bool collapsed;
                const float w = member_weight(scales, opacity, members[m], collapsed);
                has_collapsed |= collapsed;
                total += w;
                largest = fmaxf(largest, w);
            }
            const bool relative = total < 1e-30f && has_collapsed;
            if (total < 1e-30f && !has_collapsed) {
                total = 1e-30f;
            } else if (relative) {
                total = 0.0f;
                for (int32_t m = begin; m < end; ++m) {
                    bool collapsed;
                    const float w = member_weight(scales, opacity, members[m], collapsed);
                    total += largest > 0.0f ? w / largest : 1.0f;
                }
            }
            const auto weight = [&](const int32_t row) {
                bool collapsed;
                float w = member_weight(scales, opacity, row, collapsed);
                if (relative)
                    w = largest > 0.0f ? w / largest : 1.0f;
                return w / total;
            };

            const size_t origin = static_cast<size_t>(members[begin]) * 3;
            const float ox = has_collapsed ? means[origin] : 0.0f;
            const float oy = has_collapsed ? means[origin + 1] : 0.0f;
            const float oz = has_collapsed ? means[origin + 2] : 0.0f;
            float cx = 0.0f, cy = 0.0f, cz = 0.0f;
            for (int32_t m = begin; m < end; ++m) {
                const int32_t row = members[m];
                const float w = weight(row);
                const size_t r3 = static_cast<size_t>(row) * 3;
                cx += w * (means[r3] - ox);
                cy += w * (means[r3 + 1] - oy);
                cz += w * (means[r3 + 2] - oz);
            }
            cx += ox;
            cy += oy;
            cz += oz;
            out_means[o * 3] = cx;
            out_means[o * 3 + 1] = cy;
            out_means[o * 3 + 2] = cz;

            float sigma[9] = {};
            for (int32_t m = begin; m < end; ++m) {
                const int32_t row = members[m];
                const float w = weight(row);
                const size_t r3 = static_cast<size_t>(row) * 3, r4 = static_cast<size_t>(row) * 4;
                const float sx = scales[r3], sy = scales[r3 + 1], sz = scales[r3 + 2];
                float R[9], sig[9];
                quat_to_rotmat(rotation[r4], rotation[r4 + 1], rotation[r4 + 2], rotation[r4 + 3], R);
                sigma_from_rot_var(R, sx * sx, sy * sy, sz * sz, sig);
                const float dx = means[r3] - cx, dy = means[r3 + 1] - cy, dz = means[r3 + 2] - cz;
                sig[0] += dx * dx;
                sig[1] += dx * dy;
                sig[2] += dx * dz;
                sig[3] += dy * dx;
                sig[4] += dy * dy;
                sig[5] += dy * dz;
                sig[6] += dz * dx;
                sig[7] += dz * dy;
                sig[8] += dz * dz;
                for (int a = 0; a < 9; ++a)
                    sigma[a] += w * sig[a];
            }
            sigma[1] = sigma[3] = 0.5f * (sigma[1] + sigma[3]);
            sigma[2] = sigma[6] = 0.5f * (sigma[2] + sigma[6]);
            sigma[5] = sigma[7] = 0.5f * (sigma[5] + sigma[7]);
            if (!has_collapsed) {
                sigma[0] += kEpsCov;
                sigma[4] += kEpsCov;
                sigma[8] += kEpsCov;
            }
            decompose(sigma, has_collapsed, out_scales + o * 3, out_rotation + o * 4);

            float transmitted = 1.0f;
            for (int32_t m = begin; m < end; ++m)
                transmitted *= 1.0f - opacity[members[m]];
            out_opacity[o] = fminf(fmaxf(1.0f - transmitted, 0.0f), 1.0f);

            for (uint32_t k = 0; k < app_dim; ++k)
                out_appearance[o * app_dim + k] = 0.0f;
            for (int32_t m = begin; m < end; ++m) {
                const int32_t row = members[m];
                const float w = weight(row);
                for (uint32_t k = 0; k < app_dim; ++k)
                    out_appearance[o * app_dim + k] += w * appearance[static_cast<size_t>(row) * app_dim + k];
            }
        }
    } // namespace

    void launch_simplify_merge(const float* means, const float* scales, const float* rotation, const float* opacity,
                               const float* appearance, const int32_t* offsets, const int32_t* members, float* out_means,
                               float* out_scales, float* out_rotation, float* out_opacity, float* out_appearance,
                               const uint32_t groups, const uint32_t app_dim, const cudaStream_t stream) {
        if (groups == 0)
            return;
        simplify_merge_kernel<<<(groups + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
            means, scales, rotation, opacity, appearance, offsets, members, out_means, out_scales, out_rotation,
            out_opacity, out_appearance, groups, app_dim);
        LFS_CUDA_LAUNCH_CHECK(stream, "tensor.simplify_merge");
    }
} // namespace lfs::core::tensor_ops
