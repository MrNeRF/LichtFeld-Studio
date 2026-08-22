/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "Cameras.cuh"
#include "Common.h"
#include "FromWorldRay.cuh"
#include "Rasterization.h"
#include "Utils.cuh"
#include "core/cuda_safe_format.hpp"

// Use standard CUDA atomic add
#ifndef gpuAtomicAdd
#define gpuAtomicAdd atomicAdd
#endif

namespace gsplat_lfs {

    namespace cg = cooperative_groups;

    template <uint32_t CDIM, typename scalar_t, bool kPinholeGlobal>
    __global__ void rasterize_to_pixels_from_world_3dgs_bwd_kernel(
        const uint32_t C,
        const uint32_t N,
        const uint32_t n_isects,
        const bool packed,
        // fwd inputs
        const vec3* __restrict__ means,           // [N, 3]
        const vec4* __restrict__ quats,           // [N, 4]
        const vec3* __restrict__ scales,          // [N, 3]
        const scalar_t* __restrict__ colors,      // [C, N, CDIM] or [nnz, CDIM]
        const scalar_t* __restrict__ opacities,   // [C, N] or [nnz]
        const scalar_t* __restrict__ backgrounds, // [C, CDIM] - solid color (mutually exclusive with bg_images)
        const scalar_t* __restrict__ bg_images,   // [C, CDIM, H, W] - per-pixel background (mutually exclusive with backgrounds)
        const bool* __restrict__ masks,           // [C, tile_height, tile_width]
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        const uint32_t tile_width,
        const uint32_t tile_height,
        // camera model
        const scalar_t* __restrict__ viewmats0, // [C, 4, 4]
        const scalar_t* __restrict__ viewmats1, // [C, 4, 4] optional for rolling shutter
        const scalar_t* __restrict__ Ks,        // [C, 3, 3]
        const CameraModelType camera_model_type,
        // uncented transform
        const UnscentedTransformParameters ut_params,
        const ShutterType rs_type,
        const scalar_t* __restrict__ radial_coeffs,     // [C, 6] or [C, 4] optional
        const scalar_t* __restrict__ tangential_coeffs, // [C, 2] optional
        const scalar_t* __restrict__ thin_prism_coeffs, // [C, 2] optional
        // intersections
        const int32_t* __restrict__ tile_offsets, // [C, tile_height, tile_width]
        const int32_t* __restrict__ flatten_ids,  // [n_isects]
        // fwd outputs
        const scalar_t* __restrict__ render_alphas, // [C, image_height, image_width, 1]
        const int32_t* __restrict__ last_ids,       // [C, image_height, image_width]
        // grad outputs
        const scalar_t* __restrict__ v_render_colors, // [C, CDIM, image_height, image_width]
        const scalar_t* __restrict__ v_render_alphas, // [C, image_height, image_width, 1]
        // grad inputs
        vec3* __restrict__ v_means,                          // [N, 3]
        vec4* __restrict__ v_quats,                          // [N, 4]
        vec3* __restrict__ v_scales,                         // [N, 3]
        scalar_t* __restrict__ v_colors,                     // [C, N, CDIM] or [nnz, CDIM]
        scalar_t* __restrict__ v_opacities,                  // [C, N] or [nnz]
        float* __restrict__ densification_info,              // [2, N] flattened or nullptr
        const scalar_t* __restrict__ densification_error_map // [H, W] or nullptr
    ) {
        auto block = cg::this_thread_block();
        uint32_t cid = block.group_index().x;
        uint32_t tile_id =
            block.group_index().y * tile_width + block.group_index().z;
        uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
        uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

        tile_offsets += cid * tile_height * tile_width;
        render_alphas += cid * image_height * image_width;
        last_ids += cid * image_height * image_width;
        v_render_colors += cid * image_height * image_width * CDIM;
        v_render_alphas += cid * image_height * image_width;
        if (backgrounds != nullptr) {
            backgrounds += cid * CDIM;
        }
        if (bg_images != nullptr) {
            bg_images += cid * CDIM * image_height * image_width;
        }
        if (masks != nullptr) {
            masks += cid * tile_height * tile_width;
        }

        // when the mask is provided, do nothing and return if
        // this tile is labeled as False
        if (masks != nullptr && !masks[tile_id]) {
            return;
        }

        const float px = (float)j + 0.5f;
        const float py = (float)i + 0.5f;
        // clamp this value to the last pixel
        const int32_t pix_id =
            min(i * image_width + j, image_width * image_height - 1);

        const WorldRay ray = from_world_pixel_ray<kPinholeGlobal>(
            camera_model_type, rs_type, image_width, image_height, px, py,
            viewmats0, viewmats1, Ks, cid,
            radial_coeffs, tangential_coeffs, thin_prism_coeffs);
        const vec3 ray_d = ray.ray_dir;
        const vec3 ray_o = ray.ray_org;

        // keep not rasterizing threads around for reading data
        const bool active = (i < image_height && j < image_width) && ray.valid_flag;

        // have all threads in tile process the same gaussians in batches
        // first collect gaussians between range.x and range.y in batches
        // which gaussians to look through in this tile
        int32_t range_start = tile_offsets[tile_id];
        int32_t range_end =
            (cid == C - 1) && (tile_id == tile_width * tile_height - 1)
                ? n_isects
                : tile_offsets[tile_id + 1];
        const uint32_t block_size = block.size();
        const uint32_t num_batches =
            (range_end - range_start + block_size - 1) / block_size;

        extern __shared__ int s[];
        int32_t* id_batch = (int32_t*)s; // [block_size]
        vec4* xyz_opacity_batch =
            reinterpret_cast<vec4*>(&id_batch[block_size]); // [block_size]
        vec3* scale_batch =
            reinterpret_cast<vec3*>(&xyz_opacity_batch[block_size]); // [block_size]
        vec4* quat_batch =
            reinterpret_cast<vec4*>(&scale_batch[block_size]); // [block_size]
        mat3* mt_batch =
            reinterpret_cast<mat3*>(&quat_batch[block_size]); // [block_size]
        float* rgbs_batch =
            reinterpret_cast<float*>(&mt_batch[block_size]); // [block_size * CDIM]

        // this is the T AFTER the last gaussian in this pixel
        float T_final = 1.0f - render_alphas[pix_id];
        float T = T_final;
        // the contribution from gaussians behind the current one
        float buffer[CDIM] = {0.f};
        // index of last gaussian to contribute to this pixel
        const int32_t bin_final = active ? last_ids[pix_id] : 0;

        // df/d_out for this pixel
        float v_render_c[CDIM];
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            v_render_c[k] = v_render_colors[chw_pix(k, pix_id, image_height, image_width)];
        }
        const float v_render_a = v_render_alphas[pix_id];

        const bool have_bg = bg_images != nullptr || backgrounds != nullptr;
        float bg_accum = 0.f;
        if (have_bg) {
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                float bg_val;
                if (bg_images != nullptr) {
                    bg_val = bg_images[k * image_height * image_width + pix_id];
                } else {
                    bg_val = backgrounds[k];
                }
                bg_accum += bg_val * v_render_c[k];
            }
        }
        const bool do_dens = densification_info != nullptr && densification_error_map != nullptr;
        const float pixel_error = do_dens ? densification_error_map[pix_id] : 0.f;

        // collect and process batches of gaussians
        // each thread loads one gaussian at a time before rasterizing
        const uint32_t tr = block.thread_rank();
        cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
        const int32_t warp_bin_final =
            cg::reduce(warp, bin_final, cg::greater<int>());
        for (uint32_t b = 0; b < num_batches; ++b) {
            // resync all threads before writing next batch of shared mem
            block.sync();

            // each thread fetch 1 gaussian from back to front
            // 0 index will be furthest back in batch
            // index of gaussian to load
            // batch end is the index of the last gaussian in the batch
            // These values can be negative so must be int32 instead of uint32
            const int32_t batch_end = range_end - 1 - block_size * b;
            const int32_t batch_size = min(block_size, batch_end + 1 - range_start);
            const int32_t idx = batch_end - tr;
            if (idx >= range_start) {
                int32_t g = flatten_ids[idx]; // flatten index in [C * N] or [nnz]
                id_batch[tr] = g;
                const vec3 xyz = means[g];
                const float opac = activated_opacity(opacities[g]);
                xyz_opacity_batch[tr] = {xyz.x, xyz.y, xyz.z, opac};
                const vec3 scale_act = activated_scale(scales[g]);
                const vec4 quat = quats[g];
                scale_batch[tr] = scale_act;
                quat_batch[tr] = quat;
                const mat3 R = quat_to_rotmat(quat);
                const mat3 S = mat3(
                    1.0f / scale_act[0], 0.f, 0.f,
                    0.f, 1.0f / scale_act[1], 0.f,
                    0.f, 0.f, 1.0f / scale_act[2]);
                mt_batch[tr] = glm::transpose(R * S);
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    rgbs_batch[tr * CDIM + k] = colors[g * CDIM + k];
                }
            }
            // wait for other threads to collect the gaussians in batch
            block.sync();
            // process gaussians in the current batch for this pixel
            // 0 index is the furthest back gaussian in the batch
            for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size;
                 ++t) {
                bool valid = active;
                if (batch_end - static_cast<int32_t>(t) > bin_final) {
                    valid = false;
                }
                float alpha = 0.f;
                float opac = 0.f;
                float vis = 0.f;
                vec3 xyz, o_minus_mu, gro, grd, grd_n, gcrod;
                mat3 Mt;
                if (valid) {
                    const vec4 xyz_opac = xyz_opacity_batch[t];
                    opac = xyz_opac[3];
                    xyz = {xyz_opac[0], xyz_opac[1], xyz_opac[2]};
                    Mt = mt_batch[t];
                    o_minus_mu = ray_o - xyz;
                    gro = Mt * o_minus_mu;
                    grd = Mt * ray_d;
                    grd_n = safe_normalize(grd);
                    gcrod = glm::cross(grd_n, gro);
                    const float grayDist = glm::dot(gcrod, gcrod);
                    const float power = -0.5f * grayDist;
                    vis = __expf(power);
                    alpha = min(0.999f, opac * vis);
                    if (power > 0.f || alpha < 1.f / 255.f) {
                        valid = false;
                    }
                }

                const unsigned valid_mask = __ballot_sync(0xffffffffu, valid);
                if (valid_mask == 0u) {
                    continue;
                }

                float v_rgb_local[CDIM] = {0.f};
                vec3 v_mean_local = {0.f, 0.f, 0.f};
                vec3 v_scale_local = {0.f, 0.f, 0.f};
                vec4 v_quat_local = {0.f, 0.f, 0.f, 0.f};
                float v_opacity_local = 0.f;
                float densification_weight_local = 0.f;
                float densification_error_weighted_local = 0.f;
                if (valid) {
                    const float ra = 1.0f / (1.0f - alpha);
                    T *= ra;
                    const float fac = alpha * T;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        v_rgb_local[k] = fac * v_render_c[k];
                    }
                    if (do_dens) {
                        densification_weight_local = fac;
                        densification_error_weighted_local = fac * pixel_error;
                    }
                    float v_alpha = 0.f;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        v_alpha += (rgbs_batch[t * CDIM + k] * T - buffer[k] * ra) *
                                   v_render_c[k];
                    }
                    v_alpha += T_final * ra * v_render_a;
                    if (have_bg) {
                        v_alpha += -T_final * ra * bg_accum;
                    }

                    if (opac * vis <= 0.999f) {
                        const float v_vis = opac * v_alpha;
                        const float v_gradDist = -0.5f * vis * v_vis;
                        const vec3 v_gcrod = 2.0f * v_gradDist * gcrod;
                        const vec3 v_grd_n = -glm::cross(v_gcrod, gro);
                        const vec3 v_gro = glm::cross(v_gcrod, grd_n);
                        const vec3 v_grd = safe_normalize_bw(grd, v_grd_n);
                        const mat3 v_Mt = glm::outerProduct(v_grd, ray_d) +
                                          glm::outerProduct(v_gro, o_minus_mu);
                        const vec3 v_o_minus_mu = glm::transpose(Mt) * v_gro;
                        v_mean_local += -v_o_minus_mu;
                        const vec3 scale_act = scale_batch[t];
                        const vec4 quat = quat_batch[t];
                        const mat3 R = quat_to_rotmat(quat);
                        quat_scale_to_preci_half_vjp(
                            quat, scale_act, R, glm::transpose(v_Mt), v_quat_local, v_scale_local);
                        v_opacity_local = vis * v_alpha;
                    }

#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        buffer[k] += rgbs_batch[t * CDIM + k] * fac;
                    }
                }
                warpSum<CDIM>(v_rgb_local, warp);
                warpSum(v_mean_local, warp);
                warpSum(v_scale_local, warp);
                warpSum(v_quat_local, warp);
                warpSum(v_opacity_local, warp);
                warpSum(densification_weight_local, warp);
                warpSum(densification_error_weighted_local, warp);
                if (warp.thread_rank() == 0) {
                    const int32_t g = id_batch[t];
                    float* v_rgb_ptr = (float*)(v_colors) + CDIM * g;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                    }
                    float* v_mean_ptr = (float*)(v_means) + 3 * g;
                    gpuAtomicAdd(v_mean_ptr, v_mean_local.x);
                    gpuAtomicAdd(v_mean_ptr + 1, v_mean_local.y);
                    gpuAtomicAdd(v_mean_ptr + 2, v_mean_local.z);

                    float* v_scale_ptr = (float*)(v_scales) + 3 * g;
                    const vec3 scale_act = scale_batch[t];
                    gpuAtomicAdd(v_scale_ptr, v_scale_local.x * scale_act.x);
                    gpuAtomicAdd(v_scale_ptr + 1, v_scale_local.y * scale_act.y);
                    gpuAtomicAdd(v_scale_ptr + 2, v_scale_local.z * scale_act.z);

                    float* v_quat_ptr = (float*)(v_quats) + 4 * g;
                    gpuAtomicAdd(v_quat_ptr, v_quat_local.x);
                    gpuAtomicAdd(v_quat_ptr + 1, v_quat_local.y);
                    gpuAtomicAdd(v_quat_ptr + 2, v_quat_local.z);
                    gpuAtomicAdd(v_quat_ptr + 3, v_quat_local.w);

                    const float opac_act = xyz_opacity_batch[t][3];
                    gpuAtomicAdd(v_opacities + g, v_opacity_local * opac_act * (1.0f - opac_act));
                    if (do_dens) {
                        gpuAtomicAdd(densification_info + g, densification_weight_local);
                        gpuAtomicAdd(densification_info + N + g, densification_error_weighted_local);
                    }
                }
            }
        }
    }

    ////////////////////////////////////////////////////////////////
    // Launch Function
    ////////////////////////////////////////////////////////////////

    template <uint32_t CDIM>
    void launch_rasterize_to_pixels_from_world_3dgs_bwd_kernel(
        const float* means,
        const float* quats,
        const float* scales,
        const float* colors,
        const float* opacities,
        const float* backgrounds,
        const float* bg_images,
        const bool* masks,
        uint32_t C,
        uint32_t N,
        uint32_t n_isects,
        uint32_t image_width,
        uint32_t image_height,
        uint32_t tile_size,
        const float* viewmats0,
        const float* viewmats1,
        const float* Ks,
        CameraModelType camera_model,
        const UnscentedTransformParameters& ut_params,
        ShutterType rs_type,
        const float* radial_coeffs,
        const float* tangential_coeffs,
        const float* thin_prism_coeffs,
        const int32_t* tile_offsets,
        const int32_t* flatten_ids,
        const float* render_alphas,
        const int32_t* last_ids,
        const float* v_render_colors,
        const float* v_render_alphas,
        float* v_means,
        float* v_quats,
        float* v_scales,
        float* v_colors,
        float* v_opacities,
        float* densification_info,
        const float* densification_error_map,
        cudaStream_t stream) {
        const bool packed = false; // Only support non-packed for now
        const uint32_t tile_width = (image_width + tile_size - 1) / tile_size;
        const uint32_t tile_height = (image_height + tile_size - 1) / tile_size;

        // Each block covers a tile on the image
        dim3 threads = {tile_size, tile_size, 1};
        dim3 grid = {C, tile_height, tile_width};

        int64_t shmem_size =
            tile_size * tile_size *
            (sizeof(int32_t) + sizeof(vec4) + sizeof(vec3) + sizeof(vec4) +
             sizeof(mat3) + sizeof(float) * CDIM);
        // Atomics contend past 2 blocks/SM; pad so occupancy stays at 2.
        if constexpr (CDIM == 3) {
            constexpr int64_t kOccCapBytes = 49152;
            if (shmem_size < kOccCapBytes) {
                shmem_size = kOccCapBytes;
            }
        }

        if (n_isects == 0) {
            // Skip kernel launch if no intersections
            return;
        }

        const bool pinhole_global = is_pinhole_global_launch(
            camera_model, rs_type, viewmats1,
            radial_coeffs, tangential_coeffs, thin_prism_coeffs);

        auto launch = [&](auto kernel) {
            auto err = cudaFuncSetAttribute(
                kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shmem_size);
            if (err != cudaSuccess) {
                lfs::core::ensure_cuda_success(
                    err, "cudaFuncSetAttribute(gsplat backward shared memory)",
                    lfs::core::detail::format_cuda_safe(
                        "requested_bytes={}, try lowering tile_size", shmem_size),
                    LFS_SOURCE_SITE_CURRENT(),
                    lfs::core::CudaFailureDisposition::LogOnly);
                return;
            }
            kernel<<<grid, threads, shmem_size, stream>>>(
                C,
                N,
                n_isects,
                packed,
                reinterpret_cast<const vec3*>(means),
                reinterpret_cast<const vec4*>(quats),
                reinterpret_cast<const vec3*>(scales),
                colors,
                opacities,
                backgrounds,
                bg_images,
                masks,
                image_width,
                image_height,
                tile_size,
                tile_width,
                tile_height,
                viewmats0,
                viewmats1,
                Ks,
                camera_model,
                ut_params,
                rs_type,
                radial_coeffs,
                tangential_coeffs,
                thin_prism_coeffs,
                tile_offsets,
                flatten_ids,
                render_alphas,
                last_ids,
                v_render_colors,
                v_render_alphas,
                reinterpret_cast<vec3*>(v_means),
                reinterpret_cast<vec4*>(v_quats),
                reinterpret_cast<vec3*>(v_scales),
                v_colors,
                v_opacities,
                densification_info,
                densification_error_map);
            LFS_CUDA_LAUNCH_CHECK(stream, "gsplat.rasterize_to_pixels_bwd");
        };

        if constexpr (CDIM == 3) {
            if (pinhole_global) {
                launch(rasterize_to_pixels_from_world_3dgs_bwd_kernel<CDIM, float, true>);
                return;
            }
        }
        launch(rasterize_to_pixels_from_world_3dgs_bwd_kernel<CDIM, float, false>);
    }

    ////////////////////////////////////////////////////////////////
    // Explicit Instantiations
    ////////////////////////////////////////////////////////////////

#define __INS__(CDIM)                                                          \
    template void launch_rasterize_to_pixels_from_world_3dgs_bwd_kernel<CDIM>( \
        const float* means,                                                    \
        const float* quats,                                                    \
        const float* scales,                                                   \
        const float* colors,                                                   \
        const float* opacities,                                                \
        const float* backgrounds,                                              \
        const float* bg_images,                                                \
        const bool* masks,                                                     \
        uint32_t C,                                                            \
        uint32_t N,                                                            \
        uint32_t n_isects,                                                     \
        uint32_t image_width,                                                  \
        uint32_t image_height,                                                 \
        uint32_t tile_size,                                                    \
        const float* viewmats0,                                                \
        const float* viewmats1,                                                \
        const float* Ks,                                                       \
        CameraModelType camera_model,                                          \
        const UnscentedTransformParameters& ut_params,                         \
        ShutterType rs_type,                                                   \
        const float* radial_coeffs,                                            \
        const float* tangential_coeffs,                                        \
        const float* thin_prism_coeffs,                                        \
        const int32_t* tile_offsets,                                           \
        const int32_t* flatten_ids,                                            \
        const float* render_alphas,                                            \
        const int32_t* last_ids,                                               \
        const float* v_render_colors,                                          \
        const float* v_render_alphas,                                          \
        float* v_means,                                                        \
        float* v_quats,                                                        \
        float* v_scales,                                                       \
        float* v_colors,                                                       \
        float* v_opacities,                                                    \
        float* densification_info,                                             \
        const float* densification_error_map,                                  \
        cudaStream_t stream);

    __INS__(1)
    __INS__(2)
    __INS__(3)
    __INS__(4)
    __INS__(5)
    __INS__(8)
    __INS__(9)
    __INS__(16)
    __INS__(17)
    __INS__(32)
    __INS__(33)
    __INS__(64)
    __INS__(65)
    __INS__(128)
    __INS__(129)
    __INS__(256)
    __INS__(257)
    __INS__(512)
    __INS__(513)
#undef __INS__

} // namespace gsplat_lfs
