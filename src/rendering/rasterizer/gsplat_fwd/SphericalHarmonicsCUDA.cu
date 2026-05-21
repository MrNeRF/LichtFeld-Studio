/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "Common.h"
#include "SphericalHarmonics.h"
#include "Utils.cuh"
#include "rasterizer_constants.cuh"

namespace gsplat_fwd {

    namespace cg = cooperative_groups;

    // SH basis constants (Sloan, JCGT 2013)
    constexpr float SH_C0 = 0.2820947917738781f;
    constexpr float SH_C1 = 0.48860251190292f;
    constexpr float SH_DC_OFFSET = 0.5f; // 3DGS stores colors as (color - 0.5) / C0
    constexpr uint32_t kShReorderSize = 32u;
    constexpr uint32_t kShRestFloat4PerPrimitive = 12u;

    __device__ __forceinline__ uint32_t shAt(const uint32_t primitive_idx, const uint32_t float4_slot) {
        const uint32_t block = primitive_idx / kShReorderSize;
        const uint32_t lane = primitive_idx % kShReorderSize;
        return block * (kShRestFloat4PerPrimitive * kShReorderSize) + float4_slot * kShReorderSize + lane;
    }

    __device__ __forceinline__ float float4_component(const float4 v, const uint32_t component) {
        switch (component) {
        case 0:
            return v.x;
        case 1:
            return v.y;
        case 2:
            return v.z;
        default:
            return v.w;
        }
    }

    __device__ __forceinline__ float swizzled_rest_coeff_channel(
        const float4* __restrict__ sh_rest,
        const uint32_t primitive_idx,
        const uint32_t rest_coeff_idx,
        const uint32_t channel) {
        if (sh_rest == nullptr) {
            return 0.0f;
        }
        const uint32_t offset = rest_coeff_idx * 3u + channel;
        const uint32_t slot = offset / 4u;
        const uint32_t component = offset % 4u;
        return float4_component(sh_rest[shAt(primitive_idx, slot)], component);
    }

    template <typename scalar_t>
    __global__ void spherical_harmonics_swizzled_fwd_kernel(
        const uint32_t M,
        const uint32_t degrees_to_use,
        const vec3* __restrict__ dirs,
        const scalar_t* __restrict__ sh0,
        const float4* __restrict__ sh_rest,
        const bool* __restrict__ masks,
        const int32_t* __restrict__ visible_indices,
        scalar_t* __restrict__ colors) {
        const uint32_t idx = cg::this_grid().thread_rank();
        if (idx >= M * 3) {
            return;
        }
        const uint32_t elem_id = idx / 3;
        const uint32_t c = idx % 3;
        if (masks != nullptr && !masks[elem_id]) {
            return;
        }

        const uint32_t global_id = (visible_indices != nullptr)
                                       ? static_cast<uint32_t>(visible_indices[elem_id])
                                       : elem_id;
        const vec3 dir = (degrees_to_use > 0 && dirs != nullptr) ? dirs[elem_id] : vec3{0.f, 0.f, 1.f};
        const uint32_t effective_degree = dirs != nullptr ? degrees_to_use : 0u;

        float result = SH_C0 * sh0[global_id * 3u + c];
        if (effective_degree >= 1) {
            const float inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
            const float x = dir.x * inorm;
            const float y = dir.y * inorm;
            const float z = dir.z * inorm;

            const auto coeff = [&](const uint32_t rest_idx) -> float {
                return swizzled_rest_coeff_channel(sh_rest, global_id, rest_idx, c);
            };

            result += SH_C1 * (-y * coeff(0) + z * coeff(1) - x * coeff(2));
            if (effective_degree >= 2) {
                const float z2 = z * z;
                const float fTmp0B = -1.092548430592079f * z;
                const float fC1 = x * x - y * y;
                const float fS1 = 2.f * x * y;
                const float pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);
                const float pSH7 = fTmp0B * x;
                const float pSH5 = fTmp0B * y;
                const float pSH8 = 0.5462742152960395f * fC1;
                const float pSH4 = 0.5462742152960395f * fS1;

                result += pSH4 * coeff(3) + pSH5 * coeff(4) +
                          pSH6 * coeff(5) + pSH7 * coeff(6) +
                          pSH8 * coeff(7);
                if (effective_degree >= 3) {
                    const float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
                    const float fTmp1B = 1.445305721320277f * z;
                    const float fC2 = x * fC1 - y * fS1;
                    const float fS2 = x * fS1 + y * fC1;
                    const float pSH12 =
                        z * (1.865881662950577f * z2 - 1.119528997770346f);
                    const float pSH13 = fTmp0C * x;
                    const float pSH11 = fTmp0C * y;
                    const float pSH14 = fTmp1B * fC1;
                    const float pSH10 = fTmp1B * fS1;
                    const float pSH15 = -0.5900435899266435f * fC2;
                    const float pSH9 = -0.5900435899266435f * fS2;

                    result +=
                        pSH9 * coeff(8) + pSH10 * coeff(9) +
                        pSH11 * coeff(10) + pSH12 * coeff(11) +
                        pSH13 * coeff(12) + pSH14 * coeff(13) +
                        pSH15 * coeff(14);
                }
            }
        }

        colors[idx] = result + SH_DC_OFFSET;
    }

    void launch_spherical_harmonics_swizzled_fwd_kernel(
        uint32_t degrees_to_use,
        const float* dirs,
        const float* sh0,
        const float* sh_rest_swizzled,
        const bool* masks,
        const int32_t* visible_indices,
        int64_t total_elements,
        float* colors,
        cudaStream_t stream) {
        const uint32_t M = static_cast<uint32_t>(total_elements);
        const int64_t n_elements = M * 3;
        if (n_elements == 0) {
            return;
        }

        dim3 threads(256);
        dim3 grid((n_elements + threads.x - 1) / threads.x);
        spherical_harmonics_swizzled_fwd_kernel<float>
            <<<grid, threads, 0, stream>>>(
                M,
                degrees_to_use,
                reinterpret_cast<const vec3*>(dirs),
                sh0,
                reinterpret_cast<const float4*>(sh_rest_swizzled),
                masks,
                visible_indices,
                colors);
    }

    using lfs::rendering::extract_rotation_row_major;
    using lfs::rendering::has_non_identity_transform;

    // Compute viewing directions for SH. When model transforms are provided,
    // directions are mapped to local space to keep SH object-locked.
    __global__ void compute_view_dirs_kernel(
        const float* __restrict__ means,
        const float* __restrict__ viewmats,
        const uint32_t C,
        const uint32_t M,                           // Visible gaussians to process
        const float* __restrict__ model_transforms, // [num_transforms, 4, 4] row-major optional
        const int* __restrict__ transform_indices,  // [N_total] optional
        const int num_transforms,
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

        float dir_world_x = mx - campos_x;
        float dir_world_y = my - campos_y;
        float dir_world_z = mz - campos_z;
        float dir_x = dir_world_x;
        float dir_y = dir_world_y;
        float dir_z = dir_world_z;

        if (model_transforms != nullptr && num_transforms > 0) {
            const int transform_idx = transform_indices != nullptr
                                          ? min(max(transform_indices[global_n], 0), num_transforms - 1)
                                          : 0;
            const float* const m = model_transforms + transform_idx * 16;
            if (has_non_identity_transform(m)) {
                const float mean_world_x = m[0] * mx + m[1] * my + m[2] * mz + m[3];
                const float mean_world_y = m[4] * mx + m[5] * my + m[6] * mz + m[7];
                const float mean_world_z = m[8] * mx + m[9] * my + m[10] * mz + m[11];
                dir_world_x = mean_world_x - campos_x;
                dir_world_y = mean_world_y - campos_y;
                dir_world_z = mean_world_z - campos_z;

                float rot[9];
                if (extract_rotation_row_major(m, rot)) {
                    // SH is object-locked by rotation only (matches export transform behavior).
                    dir_x = rot[0] * dir_world_x + rot[3] * dir_world_y + rot[6] * dir_world_z;
                    dir_y = rot[1] * dir_world_x + rot[4] * dir_world_y + rot[7] * dir_world_z;
                    dir_z = rot[2] * dir_world_x + rot[5] * dir_world_y + rot[8] * dir_world_z;
                } else {
                    dir_x = dir_world_x;
                    dir_y = dir_world_y;
                    dir_z = dir_world_z;
                }
            }
        }

        // Write to compacted output
        const uint32_t out_idx = (c * M + out_n) * 3;
        dirs[out_idx + 0] = dir_x;
        dirs[out_idx + 1] = dir_y;
        dirs[out_idx + 2] = dir_z;
    }

    void compute_view_dirs(
        const float* means,
        const float* viewmats,
        const uint32_t C,
        const uint32_t N_total,
        const uint32_t M,
        const float* model_transforms,
        const int* transform_indices,
        const int num_transforms,
        const int* visible_indices,
        float* dirs,
        cudaStream_t stream) {
        if (C * M == 0)
            return;

        constexpr uint32_t BLOCK_SIZE = 256;
        const uint32_t num_blocks = (C * M + BLOCK_SIZE - 1) / BLOCK_SIZE;

        compute_view_dirs_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            means, viewmats, C, M,
            model_transforms, transform_indices, num_transforms,
            visible_indices, dirs);
    }

} // namespace gsplat_fwd
