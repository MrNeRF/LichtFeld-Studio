/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cuda_runtime.h>

#include "Common.h"
#include "SphericalBeta.h"

namespace gsplat_lfs {

    namespace {
        constexpr uint32_t kSbParamsPerLobe = 6;
        constexpr uint32_t kSbBlockSize = 256;
        // F.softplus(x, beta=ln(2) * 10) from the reference; keeps lobe colors non-negative.
        constexpr float kSbSoftplusBeta = 6.931471805599453f;
        constexpr float kSbSoftplusThreshold = 20.0f;
        // Lower bound on dot(v, d) inside the angular gradient: for sharpness exponents
        // below one, dot^(exponent - 1) is unbounded as the lobe edge approaches.
        constexpr float kSbMinDot = 1e-6f;

        __device__ __forceinline__ float sb_softplus(const float x) {
            const float bx = kSbSoftplusBeta * x;
            return bx > kSbSoftplusThreshold ? x : log1pf(__expf(bx)) / kSbSoftplusBeta;
        }

        __device__ __forceinline__ float sb_softplus_grad(const float x) {
            const float bx = kSbSoftplusBeta * x;
            return bx > kSbSoftplusThreshold ? 1.0f : 1.0f / (1.0f + __expf(-bx));
        }

        struct SbLobe {
            float dx, dy, dz;     // lobe direction
            float st, ct, sp, cp; // sin/cos of theta and phi
            float exponent;       // 4 * exp(beta)
        };

        __device__ __forceinline__ SbLobe sb_lobe(const float* lobe) {
            SbLobe out;
            __sincosf(lobe[3], &out.st, &out.ct);
            __sincosf(lobe[4], &out.sp, &out.cp);
            out.dx = out.st * out.cp;
            out.dy = out.st * out.sp;
            out.dz = out.ct;
            out.exponent = 4.0f * __expf(lobe[5]);
            return out;
        }

        __device__ __forceinline__ void normalized_dir(const float* dirs, const uint32_t idx,
                                                       float& vx, float& vy, float& vz) {
            const float x = dirs[idx * 3 + 0];
            const float y = dirs[idx * 3 + 1];
            const float z = dirs[idx * 3 + 2];
            const float inorm = rsqrtf(fmaxf(x * x + y * y + z * z, 1e-20f));
            vx = x * inorm;
            vy = y * inorm;
            vz = z * inorm;
        }

        __global__ void spherical_beta_fwd_kernel(
            const uint32_t total,
            const uint32_t N,
            const uint32_t num_lobes,
            const uint32_t color_stride,
            const float* __restrict__ dirs,
            const float* __restrict__ sb_params,
            const bool* __restrict__ masks,
            float* __restrict__ colors) {
            const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= total || (masks != nullptr && !masks[idx])) {
                return;
            }

            float vx, vy, vz;
            normalized_dir(dirs, idx, vx, vy, vz);

            const float* params = sb_params + static_cast<size_t>(idx % N) * num_lobes * kSbParamsPerLobe;
            float r = 0.0f, g = 0.0f, b = 0.0f;
            for (uint32_t l = 0; l < num_lobes; ++l) {
                const float* lobe = params + l * kSbParamsPerLobe;
                const SbLobe s = sb_lobe(lobe);
                const float dot = vx * s.dx + vy * s.dy + vz * s.dz;
                if (dot <= 0.0f) {
                    continue;
                }
                const float term = __powf(dot, s.exponent);
                r += term * sb_softplus(lobe[0]);
                g += term * sb_softplus(lobe[1]);
                b += term * sb_softplus(lobe[2]);
            }

            float* out = colors + static_cast<size_t>(idx) * color_stride;
            out[0] += r;
            out[1] += g;
            out[2] += b;
        }

        __global__ void spherical_beta_bwd_kernel(
            const uint32_t total,
            const uint32_t N,
            const uint32_t num_lobes,
            const uint32_t color_stride,
            const bool atomic_accumulate,
            const float* __restrict__ dirs,
            const float* __restrict__ sb_params,
            const bool* __restrict__ masks,
            const float* __restrict__ v_colors,
            float* __restrict__ v_sb_params) {
            const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= total || (masks != nullptr && !masks[idx])) {
                return;
            }

            const float* v_color = v_colors + static_cast<size_t>(idx) * color_stride;
            const float vcr = v_color[0];
            const float vcg = v_color[1];
            const float vcb = v_color[2];
            if (vcr == 0.0f && vcg == 0.0f && vcb == 0.0f) {
                return;
            }

            float vx, vy, vz;
            normalized_dir(dirs, idx, vx, vy, vz);

            const size_t row = static_cast<size_t>(idx % N) * num_lobes * kSbParamsPerLobe;
            const float* params = sb_params + row;
            float* v_params = v_sb_params + row;
            for (uint32_t l = 0; l < num_lobes; ++l) {
                const float* lobe = params + l * kSbParamsPerLobe;
                const SbLobe s = sb_lobe(lobe);
                const float dot = vx * s.dx + vy * s.dy + vz * s.dz;
                if (dot <= 0.0f) {
                    continue;
                }
                const float term = __powf(dot, s.exponent);
                const float r = sb_softplus(lobe[0]);
                const float g = sb_softplus(lobe[1]);
                const float b = sb_softplus(lobe[2]);
                const float color_grad = vcr * r + vcg * g + vcb * b;

                // d(term)/d(dot) = exponent * dot^(exponent - 1)
                const float d_term_d_dot = s.exponent * __powf(fmaxf(dot, kSbMinDot), s.exponent - 1.0f);
                const float d_dot_d_theta = vx * s.ct * s.cp + vy * s.ct * s.sp - vz * s.st;
                const float d_dot_d_phi = -vx * s.st * s.sp + vy * s.st * s.cp;
                // d(term)/d(beta) = term * ln(dot) * d(exponent)/d(beta), exponent' = exponent
                const float d_term_d_beta = term * __logf(dot) * s.exponent;

                const float grads[kSbParamsPerLobe] = {
                    vcr * term * sb_softplus_grad(lobe[0]),
                    vcg * term * sb_softplus_grad(lobe[1]),
                    vcb * term * sb_softplus_grad(lobe[2]),
                    color_grad * d_term_d_dot * d_dot_d_theta,
                    color_grad * d_term_d_dot * d_dot_d_phi,
                    color_grad * d_term_d_beta,
                };
                float* v_lobe = v_params + l * kSbParamsPerLobe;
                if (atomic_accumulate) {
#pragma unroll
                    for (uint32_t k = 0; k < kSbParamsPerLobe; ++k) {
                        atomicAdd(v_lobe + k, grads[k]);
                    }
                } else {
#pragma unroll
                    for (uint32_t k = 0; k < kSbParamsPerLobe; ++k) {
                        v_lobe[k] += grads[k];
                    }
                }
            }
        }

        uint32_t checked_total(const int64_t total_elements, const uint32_t N) {
            LFS_ASSERT_MSG(total_elements >= 0 &&
                               total_elements <= static_cast<int64_t>(UINT32_MAX),
                           "spherical beta element count exceeds 32-bit indexing");
            LFS_ASSERT_MSG(N > 0 && total_elements % N == 0,
                           "spherical beta element count must be a multiple of the splat count");
            return static_cast<uint32_t>(total_elements);
        }
    } // namespace

    void launch_spherical_beta_fwd_kernel(
        const uint32_t num_lobes,
        const uint32_t color_stride,
        const uint32_t N,
        const int64_t total_elements,
        const float* dirs,
        const float* sb_params,
        const bool* masks,
        float* colors,
        cudaStream_t stream) {
        if (num_lobes == 0 || N == 0 || total_elements == 0) {
            return;
        }
        LFS_ASSERT_MSG(color_stride >= 3, "spherical beta requires RGB color channels");
        const uint32_t total = checked_total(total_elements, N);
        const uint32_t blocks = (total + kSbBlockSize - 1) / kSbBlockSize;
        spherical_beta_fwd_kernel<<<blocks, kSbBlockSize, 0, stream>>>(
            total, N, num_lobes, color_stride, dirs, sb_params, masks, colors);
        LFS_CUDA_LAUNCH_CHECK(stream, "gsplat.spherical_beta_fwd");
    }

    void launch_spherical_beta_bwd_kernel(
        const uint32_t num_lobes,
        const uint32_t color_stride,
        const uint32_t N,
        const int64_t total_elements,
        const float* dirs,
        const float* sb_params,
        const bool* masks,
        const float* v_colors,
        float* v_sb_params,
        cudaStream_t stream) {
        if (num_lobes == 0 || N == 0 || total_elements == 0) {
            return;
        }
        LFS_ASSERT_MSG(color_stride >= 3, "spherical beta requires RGB color channels");
        const uint32_t total = checked_total(total_elements, N);
        const uint32_t blocks = (total + kSbBlockSize - 1) / kSbBlockSize;
        // A single camera owns each output row; several cameras share rows.
        const bool atomic_accumulate = total > N;
        spherical_beta_bwd_kernel<<<blocks, kSbBlockSize, 0, stream>>>(
            total, N, num_lobes, color_stride, atomic_accumulate,
            dirs, sb_params, masks, v_colors, v_sb_params);
        LFS_CUDA_LAUNCH_CHECK(stream, "gsplat.spherical_beta_bwd");
    }

} // namespace gsplat_lfs
