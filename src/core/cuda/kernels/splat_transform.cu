/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/cuda/splat_transform.hpp"
#include "core/cuda_error.hpp"

namespace lfs::core::cuda {
    namespace {
        // The float algorithm of affine_splat_geometry.slang: the shared double math runs at a small fraction of
        // the float rate on consumer GPUs. Explicitly rounded operations keep the compensated sums and the
        // polynomials exact under fast math.
        struct Pair {
            float high, low;
        };

        __device__ Pair two_sum(const float a, const float b) {
            const float total = __fadd_rn(a, b);
            const float part = __fsub_rn(total, a);
            return {total, __fadd_rn(__fsub_rn(a, __fsub_rn(total, part)), __fsub_rn(b, part))};
        }

        __device__ Pair add_pair(const Pair a, const Pair b) {
            const Pair sum = two_sum(a.high, b.high);
            return two_sum(sum.high, __fadd_rn(__fadd_rn(a.low, b.low), sum.low));
        }

        __device__ float exp_difference(const float a, const float b) {
            if (a == -INFINITY)
                return 0.0f;
            const Pair delta = two_sum(a, -b);
            if (delta.high < -104.0f)
                return 0.0f;
            const int k = int(roundf(__fmul_rn(delta.high, 1.4426950408889634f)));
            const float r = __fadd_rn(__fmaf_rn(-float(k), 0.693147182464599609375f, delta.high),
                                      __fadd_rn(delta.low, __fmul_rn(float(k), 1.904654323148236e-9f)));
            // Range reduction bounds abs(r) by ln(2)/2.
            float poly = 1.0f / 40320.0f;
            poly = __fmaf_rn(poly, r, 1.0f / 5040.0f);
            poly = __fmaf_rn(poly, r, 1.0f / 720.0f);
            poly = __fmaf_rn(poly, r, 1.0f / 120.0f);
            poly = __fmaf_rn(poly, r, 1.0f / 24.0f);
            poly = __fmaf_rn(poly, r, 1.0f / 6.0f);
            poly = __fmaf_rn(poly, r, 0.5f);
            poly = __fmaf_rn(poly, r, 1.0f);
            poly = __fmaf_rn(poly, r, 1.0f);
            return ldexpf(poly, k);
        }

        __device__ Pair log_pair(const float value) {
            if (value == 0.0f)
                return {-INFINITY, 0.0f};
            const unsigned bits = __float_as_uint(value);
            int exponent = int((bits >> 23) & 255u) - 127;
            float m = __uint_as_float((bits & 0x7fffffu) | 0x3f800000u);
            if (m > 1.41421356237f) {
                m = __fmul_rn(m, 0.5f);
                ++exponent;
            }
            const float z = __fdiv_rn(__fsub_rn(m, 1.0f), __fadd_rn(m, 1.0f)), z2 = __fmul_rn(z, z);
            float poly = 1.0f / 13.0f;
            poly = __fmaf_rn(poly, z2, 1.0f / 11.0f);
            poly = __fmaf_rn(poly, z2, 1.0f / 9.0f);
            poly = __fmaf_rn(poly, z2, 1.0f / 7.0f);
            poly = __fmaf_rn(poly, z2, 1.0f / 5.0f);
            poly = __fmaf_rn(poly, z2, 1.0f / 3.0f);
            const float fraction = __fmul_rn(__fmul_rn(2.0f, z), __fmaf_rn(poly, z2, 1.0f));
            const float high = __fmul_rn(float(exponent), 0.693147182464599609375f);
            const float low = __fadd_rn(__fadd_rn(__fmaf_rn(float(exponent), 0.693147182464599609375f, -high),
                                                  __fmul_rn(float(exponent), -1.904654323148236e-9f)),
                                        fraction);
            return {high, low};
        }

        __device__ float restore_log_scale(const float length, const Pair linear_log, const float largest_log) {
            if (length == 0.0f)
                return -INFINITY;
            return add_pair(add_pair(log_pair(length), linear_log), {largest_log, 0.0f}).high;
        }

        __device__ float dot3(const float3 a, const float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
        __device__ float3 scale3(const float3 a, const float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
        __device__ float3 sub3(const float3 a, const float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
        __device__ float3 div3(const float3 a, const float s) {
            return make_float3(__fdiv_rn(a.x, s), __fdiv_rn(a.y, s), __fdiv_rn(a.z, s));
        }

        __device__ float scaled_length(const float3 v) {
            const float scale = fmaxf(fabsf(v.x), fmaxf(fabsf(v.y), fabsf(v.z)));
            if (scale == 0.0f)
                return 0.0f;
            const float3 unit = div3(v, scale);
            return scale * __fsqrt_rn(dot3(unit, unit));
        }

        __device__ bool orthogonalize(float3& a, float3& b) {
            const float aa = dot3(a, a), bb = dot3(b, b), ab = dot3(a, b);
            if (fabsf(ab) <= 2e-7f * __fsqrt_rn(aa) * __fsqrt_rn(bb))
                return false;
            const float delta = 0.5f * (bb - aa);
            const float scale = fmaxf(fabsf(delta), fabsf(ab));
            const float d = __fdiv_rn(delta, scale), e = __fdiv_rn(ab, scale);
            const float h = __fsqrt_rn(d * d + e * e);
            const float t = __fdiv_rn(e, d + (delta < 0.0f ? -h : h));
            const float c = rsqrtf(1.0f + t * t), s = t * c;
            const float3 old = a;
            a = sub3(scale3(old, c), scale3(b, s));
            b = make_float3(s * old.x + c * b.x, s * old.y + c * b.y, s * old.z + c * b.z);
            return true;
        }

        __device__ void sort_axes(float3& a, float3& b, float& la, float& lb) {
            if (la < lb) {
                const float3 v = a;
                a = b;
                b = v;
                const float l = la;
                la = lb;
                lb = l;
            }
        }

        __device__ float3 normalized(const float3 v) { return div3(v, __fsqrt_rn(dot3(v, v))); }

        __device__ void affine(const float* linear, const float* log_scale, const float* quaternion,
                               float* result_scale, float* result_quaternion) {
            float4 q = make_float4(quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
            const float qm = fmaxf(fmaxf(fabsf(q.x), fabsf(q.y)), fmaxf(fabsf(q.z), fabsf(q.w)));
            if (qm > 0.0f) {
                q = make_float4(__fdiv_rn(q.x, qm), __fdiv_rn(q.y, qm), __fdiv_rn(q.z, qm), __fdiv_rn(q.w, qm));
                const float n = __fsqrt_rn(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
                q = make_float4(__fdiv_rn(q.x, n), __fdiv_rn(q.y, n), __fdiv_rn(q.z, n), __fdiv_rn(q.w, n));
            } else {
                q = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
            }
            const float w = q.x, x = q.y, y = q.z, z = q.w;
            // Difference of squares preserves zero diagonal entries at equal quaternion components.
            const float3 r0 = make_float3((w * w + x * x) - (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y));
            const float3 r1 = make_float3(2 * (x * y - w * z), (w * w + y * y) - (x * x + z * z), 2 * (y * z + w * x));
            const float3 r2 = make_float3(2 * (x * z + w * y), 2 * (y * z - w * x), (w * w + z * z) - (x * x + y * y));
            const float largest_log = fmaxf(log_scale[0], fmaxf(log_scale[1], log_scale[2]));
            float linear_scale = 0.0f;
            for (int k = 0; k < 9; ++k)
                linear_scale = fmaxf(linear_scale, fabsf(linear[k]));
            if (linear_scale == 0.0f || largest_log == -INFINITY) {
                for (int j = 0; j < 3; ++j)
                    result_scale[j] = -INFINITY;
                result_quaternion[0] = 1.0f;
                for (int j = 1; j < 4; ++j)
                    result_quaternion[j] = 0.0f;
                return;
            }
            // Common scale factors leave singular vectors unchanged and bound matrix entries.
            const float3 a0 = div3(make_float3(linear[0], linear[1], linear[2]), linear_scale);
            const float3 a1 = div3(make_float3(linear[3], linear[4], linear[5]), linear_scale);
            const float3 a2 = div3(make_float3(linear[6], linear[7], linear[8]), linear_scale);
            float3 b0 = scale3(make_float3(dot3(a0, r0), dot3(a1, r0), dot3(a2, r0)), exp_difference(log_scale[0], largest_log));
            float3 b1 = scale3(make_float3(dot3(a0, r1), dot3(a1, r1), dot3(a2, r1)), exp_difference(log_scale[1], largest_log));
            float3 b2 = scale3(make_float3(dot3(a0, r2), dot3(a1, r2), dot3(a2, r2)), exp_difference(log_scale[2], largest_log));
            for (int sweep = 0; sweep < 8; ++sweep) {
                bool changed = orthogonalize(b0, b1);
                changed = orthogonalize(b0, b2) || changed;
                changed = orthogonalize(b1, b2) || changed;
                if (!changed)
                    break;
            }
            float l0 = scaled_length(b0), l1 = scaled_length(b1), l2 = scaled_length(b2);
            sort_axes(b0, b1, l0, l1);
            sort_axes(b0, b2, l0, l2);
            sort_axes(b1, b2, l1, l2);
            const Pair linear_log = log_pair(linear_scale);
            result_scale[0] = restore_log_scale(l0, linear_log, largest_log);
            result_scale[1] = restore_log_scale(l1, linear_log, largest_log);
            result_scale[2] = restore_log_scale(l2, linear_log, largest_log);
            const float3 u = l0 > 0.0f ? div3(b0, l0) : make_float3(1.0f, 0.0f, 0.0f);
            float3 v = l1 > 0.0f ? div3(b1, l1) : make_float3(0.0f, 0.0f, 0.0f);
            // The null-space basis is arbitrary; the completed frame must remain orthonormal.
            v = sub3(v, scale3(u, dot3(u, v)));
            if (dot3(v, v) < 1e-12f) {
                const float3 axis = fabsf(u.x) <= fabsf(u.y) && fabsf(u.x) <= fabsf(u.z) ? make_float3(1.0f, 0.0f, 0.0f)
                                    : fabsf(u.y) <= fabsf(u.z)                           ? make_float3(0.0f, 1.0f, 0.0f)
                                                                                         : make_float3(0.0f, 0.0f, 1.0f);
                v = sub3(axis, scale3(u, dot3(u, axis)));
            }
            v = normalized(v);
            const float3 t = make_float3(u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x);
            // A right-handed frame represents reflections without changing the covariance.
            const float trace = u.x + v.y + t.z;
            float4 r;
            if (trace > 0.0f) {
                const float s = 2.0f * __fsqrt_rn(1.0f + trace);
                r = make_float4(s / 4.0f, __fdiv_rn(v.z - t.y, s), __fdiv_rn(t.x - u.z, s), __fdiv_rn(u.y - v.x, s));
            } else if (u.x > v.y && u.x > t.z) {
                const float s = 2.0f * __fsqrt_rn(1.0f + u.x - v.y - t.z);
                r = make_float4(__fdiv_rn(v.z - t.y, s), s / 4.0f, __fdiv_rn(u.y + v.x, s), __fdiv_rn(t.x + u.z, s));
            } else if (v.y > t.z) {
                const float s = 2.0f * __fsqrt_rn(1.0f + v.y - u.x - t.z);
                r = make_float4(__fdiv_rn(t.x - u.z, s), __fdiv_rn(u.y + v.x, s), s / 4.0f, __fdiv_rn(v.z + t.y, s));
            } else {
                const float s = 2.0f * __fsqrt_rn(1.0f + t.z - u.x - v.y);
                r = make_float4(__fdiv_rn(u.y - v.x, s), __fdiv_rn(t.x + u.z, s), __fdiv_rn(v.z + t.y, s), s / 4.0f);
            }
            const float rn = __fsqrt_rn(r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w);
            r = make_float4(__fdiv_rn(r.x, rn), __fdiv_rn(r.y, rn), __fdiv_rn(r.z, rn), __fdiv_rn(r.w, rn));
            if (r.x < 0.0f)
                r = make_float4(-r.x, -r.y, -r.z, -r.w);
            result_quaternion[0] = r.x;
            result_quaternion[1] = r.y;
            result_quaternion[2] = r.z;
            result_quaternion[3] = r.w;
        }

        __global__ void transform_geometry(const splat_transform::LinearTransform matrix,
                                           const float* scales, const float* rotations,
                                           float* out_scales, float* out_rotations,
                                           const std::size_t count, const float* matrices) {
            const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= count)
                return;
            float linear[9], s[3], q[4], os[3], oq[4];
            for (int k = 0; k < 9; ++k)
                linear[k] = matrices ? matrices[9 * i + k] : matrix.rows[k];
            for (int k = 0; k < 3; ++k)
                s[k] = scales[3 * i + k];
            for (int k = 0; k < 4; ++k)
                q[k] = rotations[4 * i + k];
            affine(linear, s, q, os, oq);
            for (int k = 0; k < 3; ++k)
                out_scales[3 * i + k] = os[k];
            for (int k = 0; k < 4; ++k)
                out_rotations[4 * i + k] = oq[k];
        }
    } // namespace
    void transform_splat_geometry(const splat_transform::LinearTransform& matrix,
                                  const float* scales, const float* rotations,
                                  float* out_scales, float* out_rotations,
                                  std::size_t count, cudaStream_t stream, const float* matrices) {
        if (!count)
            return;
        transform_geometry<<<(count + 255) / 256, 256, 0, stream>>>(
            matrix, scales, rotations, out_scales, out_rotations, count, matrices);
        LFS_CUDA_CHECK(cudaGetLastError());
    }
} // namespace lfs::core::cuda
