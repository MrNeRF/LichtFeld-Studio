/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "Common.h"
#include "SphericalHarmonics.h"
#include "Utils.cuh"

namespace gsplat_fwd {

    namespace cg = cooperative_groups;

    // SH basis constants (Sloan, JCGT 2013)
    constexpr float SH_C0 = 0.2820947917738781f;
    constexpr float SH_C1 = 0.48860251190292f;
    constexpr float SH_DC_OFFSET = 0.5f; // 3DGS stores colors as (color - 0.5) / C0

    template <typename scalar_t>
    __device__ void sh_coeffs_to_color_fast(
        const uint32_t degree,
        const uint32_t c,
        const vec3& dir,
        const scalar_t* __restrict__ coeffs,
        scalar_t* __restrict__ colors) {
        float result = SH_C0 * coeffs[c];
        if (degree >= 1) {
            // Normally rsqrt is faster than sqrt, but --use_fast_math will optimize
            // sqrt on single precision, so we use sqrt here.
            float inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
            float x = dir.x * inorm;
            float y = dir.y * inorm;
            float z = dir.z * inorm;

            result += SH_C1 * (-y * coeffs[1 * 3 + c] +
                               z * coeffs[2 * 3 + c] - x * coeffs[3 * 3 + c]);
            if (degree >= 2) {
                float z2 = z * z;

                float fTmp0B = -1.092548430592079f * z;
                float fC1 = x * x - y * y;
                float fS1 = 2.f * x * y;
                float pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);
                float pSH7 = fTmp0B * x;
                float pSH5 = fTmp0B * y;
                float pSH8 = 0.5462742152960395f * fC1;
                float pSH4 = 0.5462742152960395f * fS1;

                result += pSH4 * coeffs[4 * 3 + c] + pSH5 * coeffs[5 * 3 + c] +
                          pSH6 * coeffs[6 * 3 + c] + pSH7 * coeffs[7 * 3 + c] +
                          pSH8 * coeffs[8 * 3 + c];
                if (degree >= 3) {
                    float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
                    float fTmp1B = 1.445305721320277f * z;
                    float fC2 = x * fC1 - y * fS1;
                    float fS2 = x * fS1 + y * fC1;
                    float pSH12 =
                        z * (1.865881662950577f * z2 - 1.119528997770346f);
                    float pSH13 = fTmp0C * x;
                    float pSH11 = fTmp0C * y;
                    float pSH14 = fTmp1B * fC1;
                    float pSH10 = fTmp1B * fS1;
                    float pSH15 = -0.5900435899266435f * fC2;
                    float pSH9 = -0.5900435899266435f * fS2;

                    result +=
                        pSH9 * coeffs[9 * 3 + c] + pSH10 * coeffs[10 * 3 + c] +
                        pSH11 * coeffs[11 * 3 + c] + pSH12 * coeffs[12 * 3 + c] +
                        pSH13 * coeffs[13 * 3 + c] + pSH14 * coeffs[14 * 3 + c] +
                        pSH15 * coeffs[15 * 3 + c];

                    if (degree >= 4) {
                        float fTmp0D =
                            z * (-4.683325804901025f * z2 + 2.007139630671868f);
                        float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
                        float fTmp2B = -1.770130769779931f * z;
                        float fC3 = x * fC2 - y * fS2;
                        float fS3 = x * fS2 + y * fC2;
                        float pSH20 =
                            (1.984313483298443f * z * pSH12 -
                             1.006230589874905f * pSH6);
                        float pSH21 = fTmp0D * x;
                        float pSH19 = fTmp0D * y;
                        float pSH22 = fTmp1C * fC1;
                        float pSH18 = fTmp1C * fS1;
                        float pSH23 = fTmp2B * fC2;
                        float pSH17 = fTmp2B * fS2;
                        float pSH24 = 0.6258357354491763f * fC3;
                        float pSH16 = 0.6258357354491763f * fS3;

                        result += pSH16 * coeffs[16 * 3 + c] +
                                  pSH17 * coeffs[17 * 3 + c] +
                                  pSH18 * coeffs[18 * 3 + c] +
                                  pSH19 * coeffs[19 * 3 + c] +
                                  pSH20 * coeffs[20 * 3 + c] +
                                  pSH21 * coeffs[21 * 3 + c] +
                                  pSH22 * coeffs[22 * 3 + c] +
                                  pSH23 * coeffs[23 * 3 + c] +
                                  pSH24 * coeffs[24 * 3 + c];
                    }
                }
            }
        }

        colors[c] = result + SH_DC_OFFSET;
    }

    template <typename scalar_t>
    __device__ void sh_coeffs_to_color_fast_vjp(
        const uint32_t degree,
        const uint32_t c,
        const vec3& dir,
        const scalar_t* __restrict__ coeffs,
        const scalar_t* __restrict__ v_colors,
        scalar_t* __restrict__ v_coeffs,
        vec3* __restrict__ v_dir) {
        const float v_colors_local = v_colors[c];

        v_coeffs[c] = SH_C0 * v_colors_local;
        if (degree < 1) {
            return;
        }
        float inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
        float x = dir.x * inorm;
        float y = dir.y * inorm;
        float z = dir.z * inorm;
        float v_x = 0.f, v_y = 0.f, v_z = 0.f;

        v_coeffs[1 * 3 + c] = -SH_C1 * y * v_colors_local;
        v_coeffs[2 * 3 + c] = SH_C1 * z * v_colors_local;
        v_coeffs[3 * 3 + c] = -SH_C1 * x * v_colors_local;

        if (v_dir != nullptr) {
            v_x += -SH_C1 * coeffs[3 * 3 + c] * v_colors_local;
            v_y += -SH_C1 * coeffs[1 * 3 + c] * v_colors_local;
            v_z += SH_C1 * coeffs[2 * 3 + c] * v_colors_local;
        }
        if (degree < 2) {
            if (v_dir != nullptr) {
                vec3 dir_n = vec3(x, y, z);
                vec3 v_dir_n = vec3(v_x, v_y, v_z);
                vec3 v_d = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

                v_dir->x = v_d.x;
                v_dir->y = v_d.y;
                v_dir->z = v_d.z;
            }
            return;
        }

        float z2 = z * z;
        float fTmp0B = -1.092548430592079f * z;
        float fC1 = x * x - y * y;
        float fS1 = 2.f * x * y;
        float pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);
        float pSH7 = fTmp0B * x;
        float pSH5 = fTmp0B * y;
        float pSH8 = 0.5462742152960395f * fC1;
        float pSH4 = 0.5462742152960395f * fS1;
        v_coeffs[4 * 3 + c] = pSH4 * v_colors_local;
        v_coeffs[5 * 3 + c] = pSH5 * v_colors_local;
        v_coeffs[6 * 3 + c] = pSH6 * v_colors_local;
        v_coeffs[7 * 3 + c] = pSH7 * v_colors_local;
        v_coeffs[8 * 3 + c] = pSH8 * v_colors_local;

        float fTmp0B_z, fC1_x, fC1_y, fS1_x, fS1_y, pSH6_z, pSH7_x, pSH7_z, pSH5_y,
            pSH5_z, pSH8_x, pSH8_y, pSH4_x, pSH4_y;
        if (v_dir != nullptr) {
            fTmp0B_z = -1.092548430592079f;
            fC1_x = 2.f * x;
            fC1_y = -2.f * y;
            fS1_x = 2.f * y;
            fS1_y = 2.f * x;
            pSH6_z = 2.f * 0.9461746957575601f * z;
            pSH7_x = fTmp0B;
            pSH7_z = fTmp0B_z * x;
            pSH5_y = fTmp0B;
            pSH5_z = fTmp0B_z * y;
            pSH8_x = 0.5462742152960395f * fC1_x;
            pSH8_y = 0.5462742152960395f * fC1_y;
            pSH4_x = 0.5462742152960395f * fS1_x;
            pSH4_y = 0.5462742152960395f * fS1_y;

            v_x += v_colors_local *
                   (pSH4_x * coeffs[4 * 3 + c] + pSH8_x * coeffs[8 * 3 + c] +
                    pSH7_x * coeffs[7 * 3 + c]);
            v_y += v_colors_local *
                   (pSH4_y * coeffs[4 * 3 + c] + pSH8_y * coeffs[8 * 3 + c] +
                    pSH5_y * coeffs[5 * 3 + c]);
            v_z += v_colors_local *
                   (pSH6_z * coeffs[6 * 3 + c] + pSH7_z * coeffs[7 * 3 + c] +
                    pSH5_z * coeffs[5 * 3 + c]);
        }

        if (degree < 3) {
            if (v_dir != nullptr) {
                vec3 dir_n = vec3(x, y, z);
                vec3 v_dir_n = vec3(v_x, v_y, v_z);
                vec3 v_d = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

                v_dir->x = v_d.x;
                v_dir->y = v_d.y;
                v_dir->z = v_d.z;
            }
            return;
        }

        float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
        float fTmp1B = 1.445305721320277f * z;
        float fC2 = x * fC1 - y * fS1;
        float fS2 = x * fS1 + y * fC1;
        float pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
        float pSH13 = fTmp0C * x;
        float pSH11 = fTmp0C * y;
        float pSH14 = fTmp1B * fC1;
        float pSH10 = fTmp1B * fS1;
        float pSH15 = -0.5900435899266435f * fC2;
        float pSH9 = -0.5900435899266435f * fS2;
        v_coeffs[9 * 3 + c] = pSH9 * v_colors_local;
        v_coeffs[10 * 3 + c] = pSH10 * v_colors_local;
        v_coeffs[11 * 3 + c] = pSH11 * v_colors_local;
        v_coeffs[12 * 3 + c] = pSH12 * v_colors_local;
        v_coeffs[13 * 3 + c] = pSH13 * v_colors_local;
        v_coeffs[14 * 3 + c] = pSH14 * v_colors_local;
        v_coeffs[15 * 3 + c] = pSH15 * v_colors_local;

        float fTmp0C_z, fTmp1B_z, fC2_x, fC2_y, fS2_x, fS2_y, pSH12_z, pSH13_x,
            pSH13_z, pSH11_y, pSH11_z, pSH14_x, pSH14_y, pSH14_z, pSH10_x, pSH10_y,
            pSH10_z, pSH15_x, pSH15_y, pSH9_x, pSH9_y;
        if (v_dir != nullptr) {
            fTmp0C_z = -2.285228997322329f * 2.f * z;
            fTmp1B_z = 1.445305721320277f;
            fC2_x = fC1 + x * fC1_x - y * fS1_x;
            fC2_y = x * fC1_y - fS1 - y * fS1_y;
            fS2_x = fS1 + x * fS1_x + y * fC1_x;
            fS2_y = x * fS1_y + fC1 + y * fC1_y;
            pSH12_z = 3.f * 1.865881662950577f * z2 - 1.119528997770346f;
            pSH13_x = fTmp0C;
            pSH13_z = fTmp0C_z * x;
            pSH11_y = fTmp0C;
            pSH11_z = fTmp0C_z * y;
            pSH14_x = fTmp1B * fC1_x;
            pSH14_y = fTmp1B * fC1_y;
            pSH14_z = fTmp1B_z * fC1;
            pSH10_x = fTmp1B * fS1_x;
            pSH10_y = fTmp1B * fS1_y;
            pSH10_z = fTmp1B_z * fS1;
            pSH15_x = -0.5900435899266435f * fC2_x;
            pSH15_y = -0.5900435899266435f * fC2_y;
            pSH9_x = -0.5900435899266435f * fS2_x;
            pSH9_y = -0.5900435899266435f * fS2_y;

            v_x += v_colors_local *
                   (pSH9_x * coeffs[9 * 3 + c] + pSH15_x * coeffs[15 * 3 + c] +
                    pSH10_x * coeffs[10 * 3 + c] + pSH14_x * coeffs[14 * 3 + c] +
                    pSH13_x * coeffs[13 * 3 + c]);

            v_y += v_colors_local *
                   (pSH9_y * coeffs[9 * 3 + c] + pSH15_y * coeffs[15 * 3 + c] +
                    pSH10_y * coeffs[10 * 3 + c] + pSH14_y * coeffs[14 * 3 + c] +
                    pSH11_y * coeffs[11 * 3 + c]);

            v_z += v_colors_local *
                   (pSH12_z * coeffs[12 * 3 + c] + pSH13_z * coeffs[13 * 3 + c] +
                    pSH11_z * coeffs[11 * 3 + c] + pSH14_z * coeffs[14 * 3 + c] +
                    pSH10_z * coeffs[10 * 3 + c]);
        }

        if (degree < 4) {
            if (v_dir != nullptr) {
                vec3 dir_n = vec3(x, y, z);
                vec3 v_dir_n = vec3(v_x, v_y, v_z);
                vec3 v_d = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

                v_dir->x = v_d.x;
                v_dir->y = v_d.y;
                v_dir->z = v_d.z;
            }
            return;
        }

        float fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
        float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
        float fTmp2B = -1.770130769779931f * z;
        float fC3 = x * fC2 - y * fS2;
        float fS3 = x * fS2 + y * fC2;
        float pSH20 = (1.984313483298443f * z * pSH12 + -1.006230589874905f * pSH6);
        float pSH21 = fTmp0D * x;
        float pSH19 = fTmp0D * y;
        float pSH22 = fTmp1C * fC1;
        float pSH18 = fTmp1C * fS1;
        float pSH23 = fTmp2B * fC2;
        float pSH17 = fTmp2B * fS2;
        float pSH24 = 0.6258357354491763f * fC3;
        float pSH16 = 0.6258357354491763f * fS3;
        v_coeffs[16 * 3 + c] = pSH16 * v_colors_local;
        v_coeffs[17 * 3 + c] = pSH17 * v_colors_local;
        v_coeffs[18 * 3 + c] = pSH18 * v_colors_local;
        v_coeffs[19 * 3 + c] = pSH19 * v_colors_local;
        v_coeffs[20 * 3 + c] = pSH20 * v_colors_local;
        v_coeffs[21 * 3 + c] = pSH21 * v_colors_local;
        v_coeffs[22 * 3 + c] = pSH22 * v_colors_local;
        v_coeffs[23 * 3 + c] = pSH23 * v_colors_local;
        v_coeffs[24 * 3 + c] = pSH24 * v_colors_local;

        float fTmp0D_z, fTmp1C_z, fTmp2B_z, fC3_x, fC3_y, fS3_x, fS3_y, pSH20_z,
            pSH21_x, pSH21_z, pSH19_y, pSH19_z, pSH22_x, pSH22_y, pSH22_z, pSH18_x,
            pSH18_y, pSH18_z, pSH23_x, pSH23_y, pSH23_z, pSH17_x, pSH17_y, pSH17_z,
            pSH24_x, pSH24_y, pSH16_x, pSH16_y;
        if (v_dir != nullptr) {
            fTmp0D_z = 3.f * -4.683325804901025f * z2 + 2.007139630671868f;
            fTmp1C_z = 2.f * 3.31161143515146f * z;
            fTmp2B_z = -1.770130769779931f;
            fC3_x = fC2 + x * fC2_x - y * fS2_x;
            fC3_y = x * fC2_y - fS2 - y * fS2_y;
            fS3_x = fS2 + y * fC2_x + x * fS2_x;
            fS3_y = x * fS2_y + fC2 + y * fC2_y;
            pSH20_z = 1.984313483298443f * (pSH12 + z * pSH12_z) +
                      -1.006230589874905f * pSH6_z;
            pSH21_x = fTmp0D;
            pSH21_z = fTmp0D_z * x;
            pSH19_y = fTmp0D;
            pSH19_z = fTmp0D_z * y;
            pSH22_x = fTmp1C * fC1_x;
            pSH22_y = fTmp1C * fC1_y;
            pSH22_z = fTmp1C_z * fC1;
            pSH18_x = fTmp1C * fS1_x;
            pSH18_y = fTmp1C * fS1_y;
            pSH18_z = fTmp1C_z * fS1;
            pSH23_x = fTmp2B * fC2_x;
            pSH23_y = fTmp2B * fC2_y;
            pSH23_z = fTmp2B_z * fC2;
            pSH17_x = fTmp2B * fS2_x;
            pSH17_y = fTmp2B * fS2_y;
            pSH17_z = fTmp2B_z * fS2;
            pSH24_x = 0.6258357354491763f * fC3_x;
            pSH24_y = 0.6258357354491763f * fC3_y;
            pSH16_x = 0.6258357354491763f * fS3_x;
            pSH16_y = 0.6258357354491763f * fS3_y;

            v_x += v_colors_local *
                   (pSH16_x * coeffs[16 * 3 + c] + pSH24_x * coeffs[24 * 3 + c] +
                    pSH17_x * coeffs[17 * 3 + c] + pSH23_x * coeffs[23 * 3 + c] +
                    pSH18_x * coeffs[18 * 3 + c] + pSH22_x * coeffs[22 * 3 + c] +
                    pSH21_x * coeffs[21 * 3 + c]);
            v_y += v_colors_local *
                   (pSH16_y * coeffs[16 * 3 + c] + pSH24_y * coeffs[24 * 3 + c] +
                    pSH17_y * coeffs[17 * 3 + c] + pSH23_y * coeffs[23 * 3 + c] +
                    pSH18_y * coeffs[18 * 3 + c] + pSH22_y * coeffs[22 * 3 + c] +
                    pSH19_y * coeffs[19 * 3 + c]);
            v_z += v_colors_local *
                   (pSH20_z * coeffs[20 * 3 + c] + pSH21_z * coeffs[21 * 3 + c] +
                    pSH19_z * coeffs[19 * 3 + c] + pSH22_z * coeffs[22 * 3 + c] +
                    pSH18_z * coeffs[18 * 3 + c] + pSH23_z * coeffs[23 * 3 + c] +
                    pSH17_z * coeffs[17 * 3 + c]);

            vec3 dir_n = vec3(x, y, z);
            vec3 v_dir_n = vec3(v_x, v_y, v_z);
            vec3 v_d = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

            v_dir->x = v_d.x;
            v_dir->y = v_d.y;
            v_dir->z = v_d.z;
        }
    }

    template <typename scalar_t>
    __global__ void spherical_harmonics_fwd_kernel(
        const uint32_t M,
        const uint32_t K,
        const uint32_t degrees_to_use,
        const vec3* __restrict__ dirs,               // [M, 3]
        const scalar_t* __restrict__ coeffs,         // [N_total, K, 3] (N-sized)
        const bool* __restrict__ masks,              // [M]
        const int32_t* __restrict__ visible_indices, // [M] maps elem_id -> global_idx, nullptr = direct
        scalar_t* __restrict__ colors                // [M, 3]
    ) {
        // parallelize over M * 3
        uint32_t idx = cg::this_grid().thread_rank();
        if (idx >= M * 3) {
            return;
        }
        uint32_t elem_id = idx / 3;
        uint32_t c = idx % 3; // color channel
        if (masks != nullptr && !masks[elem_id]) {
            return;
        }

        // Map to global index for sh_coeffs access
        const uint32_t global_id = (visible_indices != nullptr)
                                       ? static_cast<uint32_t>(visible_indices[elem_id])
                                       : elem_id;

        // Guard against nullptr dirs - only read when degree > 0 and dirs is valid
        // When dirs is nullptr, only SH0 (degree=0) can be evaluated
        vec3 dir = (degrees_to_use > 0 && dirs != nullptr) ? dirs[elem_id] : vec3{0.f, 0.f, 1.f};
        sh_coeffs_to_color_fast(
            dirs != nullptr ? degrees_to_use : 0u, // Only use higher SH degrees if dirs available
            c,
            dir,
            coeffs + global_id * K * 3,
            colors + elem_id * 3);
    }

    void launch_spherical_harmonics_fwd_kernel(
        uint32_t degrees_to_use,
        const float* dirs,              // [M, 3]
        const float* coeffs,            // [N_total, K, 3] (N-sized when using visible_indices)
        const bool* masks,              // [M] optional (can be nullptr)
        const int32_t* visible_indices, // [M] maps elem_id -> global_idx, nullptr = direct
        int64_t total_elements,         // M (visible gaussians)
        int32_t K,
        float* colors, // [M, 3]
        cudaStream_t stream) {
        const uint32_t M = static_cast<uint32_t>(total_elements);

        // parallelize over M * 3
        int64_t n_elements = M * 3;
        dim3 threads(256);
        dim3 grid((n_elements + threads.x - 1) / threads.x);
        int64_t shmem_size = 0; // No shared memory used in this kernel

        if (n_elements == 0) {
            // skip the kernel launch if there are no elements
            return;
        }

        spherical_harmonics_fwd_kernel<float>
            <<<grid, threads, shmem_size, stream>>>(
                M,
                static_cast<uint32_t>(K),
                degrees_to_use,
                reinterpret_cast<const vec3*>(dirs),
                coeffs,
                masks,
                visible_indices,
                colors);
    }

    template <typename scalar_t>
    __global__ void spherical_harmonics_bwd_kernel(
        const uint32_t N,
        const uint32_t K,
        const uint32_t degrees_to_use,
        const vec3* __restrict__ dirs,         // [N, 3]
        const scalar_t* __restrict__ coeffs,   // [N, K, 3]
        const bool* __restrict__ masks,        // [N]
        const scalar_t* __restrict__ v_colors, // [N, 3
        scalar_t* __restrict__ v_coeffs,       // [N, K, 3]
        scalar_t* __restrict__ v_dirs          // [N, 3] optional
    ) {
        // parallelize over N * 3
        uint32_t idx = cg::this_grid().thread_rank();
        if (idx >= N * 3) {
            return;
        }
        uint32_t elem_id = idx / 3;
        uint32_t c = idx % 3; // color channel
        if (masks != nullptr && !masks[elem_id]) {
            return;
        }

        // Guard against nullptr dirs - only read when degree > 0 and dirs is valid
        // When dirs is nullptr, only SH0 (degree=0) gradients can be computed
        vec3 dir = (degrees_to_use > 0 && dirs != nullptr) ? dirs[elem_id] : vec3{0.f, 0.f, 1.f};
        uint32_t effective_degree = dirs != nullptr ? degrees_to_use : 0u;

        vec3 v_dir = {0.f, 0.f, 0.f};
        sh_coeffs_to_color_fast_vjp(
            effective_degree,
            c,
            dir,
            coeffs + elem_id * K * 3,
            v_colors + elem_id * 3,
            v_coeffs + elem_id * K * 3,
            (v_dirs == nullptr || dirs == nullptr) ? nullptr : &v_dir);
        if (v_dirs != nullptr && dirs != nullptr) {
            atomicAdd(v_dirs + elem_id * 3, v_dir.x);
            atomicAdd(v_dirs + elem_id * 3 + 1, v_dir.y);
            atomicAdd(v_dirs + elem_id * 3 + 2, v_dir.z);
        }
    }

    void launch_spherical_harmonics_bwd_kernel(
        uint32_t degrees_to_use,
        const float* dirs,      // [N, 3]
        const float* coeffs,    // [N, K, 3]
        const bool* masks,      // [N] optional (can be nullptr)
        const float* v_colors,  // [N, 3]
        int64_t total_elements, // N
        int32_t K,
        bool compute_v_dirs,
        float* v_coeffs, // [N, K, 3]
        float* v_dirs,   // [N, 3] optional (can be nullptr)
        cudaStream_t stream) {
        const uint32_t N = static_cast<uint32_t>(total_elements);

        // parallelize over N * 3
        int64_t n_elements = N * 3;
        dim3 threads(256);
        dim3 grid((n_elements + threads.x - 1) / threads.x);
        int64_t shmem_size = 0; // No shared memory used in this kernel

        if (n_elements == 0) {
            // skip the kernel launch if there are no elements
            return;
        }

        spherical_harmonics_bwd_kernel<float>
            <<<grid, threads, shmem_size, stream>>>(
                N,
                static_cast<uint32_t>(K),
                degrees_to_use,
                reinterpret_cast<const vec3*>(dirs),
                coeffs,
                masks,
                v_colors,
                v_coeffs,
                compute_v_dirs ? v_dirs : nullptr);
    }

    // Compute viewing directions: dir = mean - camera_position
    __global__ void compute_view_dirs_kernel(
        const float* __restrict__ means,
        const float* __restrict__ viewmats,
        const uint32_t C,
        const uint32_t M,                        // Visible gaussians to process
        const int* __restrict__ visible_indices, // [M] maps output idx → global gaussian idx
        float* __restrict__ dirs) {
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= C * M)
            return;

        const uint32_t c = idx / M;
        const uint32_t out_n = idx % M; // output gaussian index

        // Map to global gaussian index if using visibility filtering
        const uint32_t global_n = (visible_indices != nullptr)
                                      ? static_cast<uint32_t>(visible_indices[out_n])
                                      : out_n;

        const float* vm = viewmats + c * 16;

        // Extract R and t from viewmat [4, 4] (row-major)
        const float R00 = vm[0], R01 = vm[1], R02 = vm[2], tx = vm[3];
        const float R10 = vm[4], R11 = vm[5], R12 = vm[6], ty = vm[7];
        const float R20 = vm[8], R21 = vm[9], R22 = vm[10], tz = vm[11];

        // Camera position: campos = -R^T * t
        const float campos_x = -(R00 * tx + R10 * ty + R20 * tz);
        const float campos_y = -(R01 * tx + R11 * ty + R21 * tz);
        const float campos_z = -(R02 * tx + R12 * ty + R22 * tz);

        // Read from global gaussian index
        const float mx = means[global_n * 3 + 0];
        const float my = means[global_n * 3 + 1];
        const float mz = means[global_n * 3 + 2];

        // Write to compacted output
        const uint32_t out_idx = (c * M + out_n) * 3;
        dirs[out_idx + 0] = mx - campos_x;
        dirs[out_idx + 1] = my - campos_y;
        dirs[out_idx + 2] = mz - campos_z;
    }

    void compute_view_dirs(
        const float* means,
        const float* viewmats,
        const uint32_t C,
        const uint32_t N_total,
        const uint32_t M,
        const int* visible_indices,
        float* dirs,
        cudaStream_t stream) {
        if (C * M == 0)
            return;

        constexpr uint32_t BLOCK_SIZE = 256;
        const uint32_t num_blocks = (C * M + BLOCK_SIZE - 1) / BLOCK_SIZE;

        compute_view_dirs_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            means, viewmats, C, M, visible_indices, dirs);
    }

} // namespace gsplat_fwd
