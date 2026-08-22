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

// Use standard CUDA atomic add
#ifndef gpuAtomicAdd
#define gpuAtomicAdd atomicAdd
#endif

namespace gsplat_lfs {

    namespace cg = cooperative_groups;

    // 16x16 tile, 128 threads, two horizontally adjacent pixels per thread.
    constexpr uint32_t kBwdDualThreads = 128;
    constexpr uint32_t kBwdDualBatch = 256;

    struct PixelBwd {
        bool active;
        int32_t pix_id;
        int32_t bin_final;
        vec3 ray_o;
        vec3 ray_d;
        float T;
        float T_final;
        float buffer[3];
        float v_render_c[3];
        float v_render_a;
        float bg_accum;
        float pixel_error;
    };

    template <bool kPerfectPinhole>
    __device__ __forceinline__ void init_pixel_bwd(
        PixelBwd& pix,
        const uint32_t i,
        const uint32_t j,
        const uint32_t image_width,
        const uint32_t image_height,
        const CameraModelType camera_model_type,
        const ShutterType rs_type,
        const float* viewmats0,
        const float* viewmats1,
        const float* Ks,
        const uint32_t cid,
        const float* radial_coeffs,
        const float* tangential_coeffs,
        const float* thin_prism_coeffs,
        const float* render_alphas,
        const int32_t* last_ids,
        const float* v_render_colors,
        const float* v_render_alphas,
        const float* backgrounds,
        const float* bg_images,
        const bool have_bg,
        const bool do_dens,
        const float* densification_error_map) {
        const float px = (float)j + 0.5f;
        const float py = (float)i + 0.5f;
        pix.pix_id = min(i * image_width + j, image_width * image_height - 1);
        const WorldRay ray = from_world_pixel_ray<kPerfectPinhole>(
            camera_model_type, rs_type, image_width, image_height, px, py,
            viewmats0, viewmats1, Ks, cid,
            radial_coeffs, tangential_coeffs, thin_prism_coeffs);
        pix.ray_d = ray.ray_dir;
        pix.ray_o = ray.ray_org;
        pix.active = (i < image_height && j < image_width) && ray.valid_flag;
        pix.T_final = 1.0f - render_alphas[pix.pix_id];
        pix.T = pix.T_final;
#pragma unroll
        for (uint32_t k = 0; k < 3; ++k) {
            pix.buffer[k] = 0.f;
            pix.v_render_c[k] = v_render_colors[chw_pix(k, pix.pix_id, image_height, image_width)];
        }
        pix.bin_final = pix.active ? last_ids[pix.pix_id] : 0;
        pix.v_render_a = v_render_alphas[pix.pix_id];
        pix.bg_accum = 0.f;
        if (have_bg) {
#pragma unroll
            for (uint32_t k = 0; k < 3; ++k) {
                float bg_val;
                if (bg_images != nullptr) {
                    bg_val = bg_images[k * image_height * image_width + pix.pix_id];
                } else {
                    bg_val = backgrounds[k];
                }
                pix.bg_accum += bg_val * pix.v_render_c[k];
            }
        }
        pix.pixel_error = do_dens ? densification_error_map[pix.pix_id] : 0.f;
    }

    __device__ __forceinline__ bool contribute_pixel(
        PixelBwd& pix,
        const int32_t isect_idx,
        const float opac,
        const mat3 Mt,
        const vec3 gro,
        const vec3 o_minus_mu,
        const vec3 scale_act,
        const vec4 quat,
        const float rgb0,
        const float rgb1,
        const float rgb2,
        const bool have_bg,
        const bool do_dens,
        float v_rgb[3],
        vec3& v_mean,
        vec3& v_scale,
        vec4& v_quat,
        float& v_opacity,
        float& dens_w,
        float& dens_e,
        mat3& R,
        bool& have_R) {
        bool valid = pix.active;
        if (isect_idx > pix.bin_final) {
            valid = false;
        }
        if (!valid) {
            return false;
        }
        const vec3 grd = Mt * pix.ray_d;
        const vec3 grd_n = safe_normalize(grd);
        const vec3 gcrod = glm::cross(grd_n, gro);
        const float grayDist = glm::dot(gcrod, gcrod);
        const float power = -0.5f * grayDist;
        const float vis = __expf(power);
        const float alpha = min(0.999f, opac * vis);
        if (power > 0.f || alpha < 1.f / 255.f) {
            return false;
        }

        const float ra = 1.0f / (1.0f - alpha);
        pix.T *= ra;
        const float fac = alpha * pix.T;
        v_rgb[0] += fac * pix.v_render_c[0];
        v_rgb[1] += fac * pix.v_render_c[1];
        v_rgb[2] += fac * pix.v_render_c[2];
        if (do_dens) {
            dens_w += fac;
            dens_e += fac * pix.pixel_error;
        }
        float v_alpha = (rgb0 * pix.T - pix.buffer[0] * ra) * pix.v_render_c[0] +
                        (rgb1 * pix.T - pix.buffer[1] * ra) * pix.v_render_c[1] +
                        (rgb2 * pix.T - pix.buffer[2] * ra) * pix.v_render_c[2];
        v_alpha += pix.T_final * ra * pix.v_render_a;
        if (have_bg) {
            v_alpha += -pix.T_final * ra * pix.bg_accum;
        }

        if (opac * vis <= 0.999f) {
            const float v_vis = opac * v_alpha;
            const float v_gradDist = -0.5f * vis * v_vis;
            const vec3 v_gcrod = 2.0f * v_gradDist * gcrod;
            const vec3 v_grd_n = -glm::cross(v_gcrod, gro);
            const vec3 v_gro = glm::cross(v_gcrod, grd_n);
            const vec3 v_grd = safe_normalize_bw(grd, v_grd_n);
            const mat3 v_Mt = glm::outerProduct(v_grd, pix.ray_d) +
                              glm::outerProduct(v_gro, o_minus_mu);
            const vec3 v_o_minus_mu = glm::transpose(Mt) * v_gro;
            v_mean += -v_o_minus_mu;
            if (!have_R) {
                R = quat_to_rotmat(quat);
                have_R = true;
            }
            quat_scale_to_preci_half_vjp(
                quat, scale_act, R, glm::transpose(v_Mt), v_quat, v_scale);
            v_opacity += vis * v_alpha;
        }

        pix.buffer[0] += rgb0 * fac;
        pix.buffer[1] += rgb1 * fac;
        pix.buffer[2] += rgb2 * fac;
        return true;
    }

    template <bool kSharedOrigin>
    __device__ __forceinline__ void contribute_pixel_pair(
        PixelBwd& pix0,
        PixelBwd& pix1,
        const int32_t isect_idx,
        const vec4 xyz_opac,
        const mat3 Mt,
        const vec3 scale_act,
        const vec4 quat,
        const float rgb0,
        const float rgb1,
        const float rgb2,
        const bool have_bg,
        const bool do_dens,
        float v_rgb[3],
        vec3& v_mean,
        vec3& v_scale,
        vec4& v_quat,
        float& v_opacity,
        float& dens_w,
        float& dens_e,
        bool& hit0,
        bool& hit1) {
        const float opac = xyz_opac[3];
        const vec3 xyz = {xyz_opac[0], xyz_opac[1], xyz_opac[2]};
        vec3 o_minus_mu0, o_minus_mu1, gro0, gro1;
        if constexpr (kSharedOrigin) {
            const vec3 ray_o = pix0.active ? pix0.ray_o : pix1.ray_o;
            o_minus_mu0 = ray_o - xyz;
            gro0 = Mt * o_minus_mu0;
            o_minus_mu1 = o_minus_mu0;
            gro1 = gro0;
        } else {
            o_minus_mu0 = pix0.ray_o - xyz;
            gro0 = Mt * o_minus_mu0;
            o_minus_mu1 = pix1.ray_o - xyz;
            gro1 = Mt * o_minus_mu1;
        }
        mat3 R;
        bool have_R = false;
        hit0 = contribute_pixel(
            pix0, isect_idx, opac, Mt, gro0, o_minus_mu0, scale_act, quat,
            rgb0, rgb1, rgb2, have_bg, do_dens, v_rgb, v_mean, v_scale, v_quat,
            v_opacity, dens_w, dens_e, R, have_R);
        hit1 = contribute_pixel(
            pix1, isect_idx, opac, Mt, gro1, o_minus_mu1, scale_act, quat,
            rgb0, rgb1, rgb2, have_bg, do_dens, v_rgb, v_mean, v_scale, v_quat,
            v_opacity, dens_w, dens_e, R, have_R);
    }

    __device__ __forceinline__ float reduce_field(float v, const unsigned n_contrib, const int src_lane) {
        if (n_contrib == 1u) {
            return __shfl_sync(0xffffffffu, v, src_lane);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            v += __shfl_xor_sync(0xffffffffu, v, offset);
        }
        return v;
    }

    __device__ __forceinline__ void flush_splat_grads(
        const int32_t g,
        const uint32_t N,
        const bool do_dens,
        const float v_rgb[3],
        const vec3 v_mean,
        const vec3 v_scale,
        const vec4 v_quat,
        const float v_opacity,
        const float dens_w,
        const float dens_e,
        const vec3 scale_act,
        const float opac_act,
        vec3* v_means,
        vec4* v_quats,
        vec3* v_scales,
        float* v_colors,
        float* v_opacities,
        float* densification_info) {
        float* v_rgb_ptr = v_colors + 3 * g;
        gpuAtomicAdd(v_rgb_ptr, v_rgb[0]);
        gpuAtomicAdd(v_rgb_ptr + 1, v_rgb[1]);
        gpuAtomicAdd(v_rgb_ptr + 2, v_rgb[2]);
        float* v_mean_ptr = (float*)(v_means) + 3 * g;
        gpuAtomicAdd(v_mean_ptr, v_mean.x);
        gpuAtomicAdd(v_mean_ptr + 1, v_mean.y);
        gpuAtomicAdd(v_mean_ptr + 2, v_mean.z);
        float* v_scale_ptr = (float*)(v_scales) + 3 * g;
        gpuAtomicAdd(v_scale_ptr, v_scale.x * scale_act.x);
        gpuAtomicAdd(v_scale_ptr + 1, v_scale.y * scale_act.y);
        gpuAtomicAdd(v_scale_ptr + 2, v_scale.z * scale_act.z);
        float* v_quat_ptr = (float*)(v_quats) + 4 * g;
        gpuAtomicAdd(v_quat_ptr, v_quat.x);
        gpuAtomicAdd(v_quat_ptr + 1, v_quat.y);
        gpuAtomicAdd(v_quat_ptr + 2, v_quat.z);
        gpuAtomicAdd(v_quat_ptr + 3, v_quat.w);
        gpuAtomicAdd(v_opacities + g, v_opacity * opac_act * (1.0f - opac_act));
        if (do_dens) {
            gpuAtomicAdd(densification_info + g, dens_w);
            gpuAtomicAdd(densification_info + N + g, dens_e);
        }
    }

    template <typename scalar_t, bool kSharedOrigin, bool kPerfectPinhole>
    __global__ void rasterize_to_pixels_from_world_3dgs_bwd_dual_kernel(
        const uint32_t C,
        const uint32_t N,
        const uint32_t n_isects,
        const bool packed,
        const vec3* __restrict__ means,
        const vec4* __restrict__ quats,
        const vec3* __restrict__ scales,
        const scalar_t* __restrict__ colors,
        const scalar_t* __restrict__ opacities,
        const scalar_t* __restrict__ backgrounds,
        const scalar_t* __restrict__ bg_images,
        const bool* __restrict__ masks,
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        const uint32_t tile_width,
        const uint32_t tile_height,
        const scalar_t* __restrict__ viewmats0,
        const scalar_t* __restrict__ viewmats1,
        const scalar_t* __restrict__ Ks,
        const CameraModelType camera_model_type,
        const UnscentedTransformParameters ut_params,
        const ShutterType rs_type,
        const scalar_t* __restrict__ radial_coeffs,
        const scalar_t* __restrict__ tangential_coeffs,
        const scalar_t* __restrict__ thin_prism_coeffs,
        const int32_t* __restrict__ tile_offsets,
        const int32_t* __restrict__ flatten_ids,
        const scalar_t* __restrict__ render_alphas,
        const int32_t* __restrict__ last_ids,
        const scalar_t* __restrict__ v_render_colors,
        const scalar_t* __restrict__ v_render_alphas,
        vec3* __restrict__ v_means,
        vec4* __restrict__ v_quats,
        vec3* __restrict__ v_scales,
        scalar_t* __restrict__ v_colors,
        scalar_t* __restrict__ v_opacities,
        float* __restrict__ densification_info,
        const scalar_t* __restrict__ densification_error_map) {
        auto block = cg::this_thread_block();
        const uint32_t cid = block.group_index().x;
        const uint32_t tile_id =
            block.group_index().y * tile_width + block.group_index().z;
        const uint32_t tx = block.thread_index().x;
        const uint32_t ty = block.thread_index().y;
        const uint32_t i = block.group_index().y * tile_size + ty;
        const uint32_t j0 = block.group_index().z * tile_size + tx * 2u;
        const uint32_t j1 = j0 + 1u;

        tile_offsets += cid * tile_height * tile_width;
        render_alphas += cid * image_height * image_width;
        last_ids += cid * image_height * image_width;
        v_render_colors += cid * image_height * image_width * 3u;
        v_render_alphas += cid * image_height * image_width;
        if (backgrounds != nullptr) {
            backgrounds += cid * 3u;
        }
        if (bg_images != nullptr) {
            bg_images += cid * 3u * image_height * image_width;
        }
        if (masks != nullptr) {
            masks += cid * tile_height * tile_width;
        }
        if (masks != nullptr && !masks[tile_id]) {
            return;
        }

        const bool have_bg = bg_images != nullptr || backgrounds != nullptr;
        const bool do_dens = densification_info != nullptr && densification_error_map != nullptr;

        PixelBwd pix0, pix1;
        init_pixel_bwd<kPerfectPinhole>(
            pix0, i, j0, image_width, image_height,
            camera_model_type, rs_type, viewmats0, viewmats1, Ks, cid,
            radial_coeffs, tangential_coeffs, thin_prism_coeffs,
            render_alphas, last_ids, v_render_colors, v_render_alphas,
            backgrounds, bg_images, have_bg, do_dens, densification_error_map);
        init_pixel_bwd<kPerfectPinhole>(
            pix1, i, j1, image_width, image_height,
            camera_model_type, rs_type, viewmats0, viewmats1, Ks, cid,
            radial_coeffs, tangential_coeffs, thin_prism_coeffs,
            render_alphas, last_ids, v_render_colors, v_render_alphas,
            backgrounds, bg_images, have_bg, do_dens, densification_error_map);

        const int32_t range_start = tile_offsets[tile_id];
        const int32_t range_end = tile_offsets[tile_id + 1];
        const uint32_t num_batches =
            (range_end - range_start + kBwdDualBatch - 1) / kBwdDualBatch;

        extern __shared__ int s[];
        int32_t* id_batch = (int32_t*)s;
        vec4* xyz_opacity_batch = reinterpret_cast<vec4*>(&id_batch[kBwdDualBatch]);
        vec3* scale_batch = reinterpret_cast<vec3*>(&xyz_opacity_batch[kBwdDualBatch]);
        vec4* quat_batch = reinterpret_cast<vec4*>(&scale_batch[kBwdDualBatch]);
        mat3* mt_batch = reinterpret_cast<mat3*>(&quat_batch[kBwdDualBatch]);
        float* rgbs_batch = reinterpret_cast<float*>(&mt_batch[kBwdDualBatch]);

        const uint32_t tr = block.thread_rank();
        cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
        const int32_t warp_bin_final =
            cg::reduce(warp, max(pix0.bin_final, pix1.bin_final), cg::greater<int>());

        auto stage_splat = [&](const int32_t idx, const uint32_t slot) {
            if (idx >= range_start) {
                const int32_t g = flatten_ids[idx];
                id_batch[slot] = g;
                const vec3 xyz = means[g];
                const float opac = activated_opacity(opacities[g]);
                xyz_opacity_batch[slot] = {xyz.x, xyz.y, xyz.z, opac};
                const vec3 scale_act = activated_scale(scales[g]);
                const vec4 quat = quats[g];
                scale_batch[slot] = scale_act;
                quat_batch[slot] = quat;
                const mat3 R = quat_to_rotmat(quat);
                const mat3 S = mat3(
                    1.0f / scale_act[0], 0.f, 0.f,
                    0.f, 1.0f / scale_act[1], 0.f,
                    0.f, 0.f, 1.0f / scale_act[2]);
                mt_batch[slot] = glm::transpose(R * S);
                rgbs_batch[slot * 3u + 0] = colors[g * 3u + 0];
                rgbs_batch[slot * 3u + 1] = colors[g * 3u + 1];
                rgbs_batch[slot * 3u + 2] = colors[g * 3u + 2];
            }
        };

        for (uint32_t b = 0; b < num_batches; ++b) {
            block.sync();
            const int32_t batch_end = range_end - 1 - static_cast<int32_t>(kBwdDualBatch * b);
            const int32_t batch_size = min(static_cast<int32_t>(kBwdDualBatch), batch_end + 1 - range_start);
            stage_splat(batch_end - static_cast<int32_t>(tr), tr);
            stage_splat(batch_end - static_cast<int32_t>(tr + kBwdDualThreads), tr + kBwdDualThreads);
            block.sync();

            for (uint32_t t = max(0, batch_end - warp_bin_final); t < static_cast<uint32_t>(batch_size); ++t) {
                const int32_t isect_idx = batch_end - static_cast<int32_t>(t);
                const vec4 xyz_opac = xyz_opacity_batch[t];
                const mat3 Mt = mt_batch[t];
                const vec3 scale_act = scale_batch[t];
                const vec4 quat = quat_batch[t];
                const float rgb0 = rgbs_batch[t * 3u + 0];
                const float rgb1 = rgbs_batch[t * 3u + 1];
                const float rgb2 = rgbs_batch[t * 3u + 2];

                float v_rgb_local[3] = {0.f, 0.f, 0.f};
                vec3 v_mean_local = {0.f, 0.f, 0.f};
                vec3 v_scale_local = {0.f, 0.f, 0.f};
                vec4 v_quat_local = {0.f, 0.f, 0.f, 0.f};
                float v_opacity_local = 0.f;
                float densification_weight_local = 0.f;
                float densification_error_weighted_local = 0.f;

                bool hit0 = false;
                bool hit1 = false;
                contribute_pixel_pair<kSharedOrigin>(
                    pix0, pix1, isect_idx, xyz_opac, Mt, scale_act, quat, rgb0, rgb1, rgb2,
                    have_bg, do_dens, v_rgb_local, v_mean_local, v_scale_local, v_quat_local,
                    v_opacity_local, densification_weight_local, densification_error_weighted_local,
                    hit0, hit1);

                const unsigned valid_mask = __ballot_sync(0xffffffffu, hit0 || hit1);
                if (valid_mask == 0u) {
                    continue;
                }
                const unsigned n_contrib = __popc(valid_mask);
                const int src_lane = __ffs(valid_mask) - 1;
                v_rgb_local[0] = reduce_field(v_rgb_local[0], n_contrib, src_lane);
                v_rgb_local[1] = reduce_field(v_rgb_local[1], n_contrib, src_lane);
                v_rgb_local[2] = reduce_field(v_rgb_local[2], n_contrib, src_lane);
                v_mean_local.x = reduce_field(v_mean_local.x, n_contrib, src_lane);
                v_mean_local.y = reduce_field(v_mean_local.y, n_contrib, src_lane);
                v_mean_local.z = reduce_field(v_mean_local.z, n_contrib, src_lane);
                v_scale_local.x = reduce_field(v_scale_local.x, n_contrib, src_lane);
                v_scale_local.y = reduce_field(v_scale_local.y, n_contrib, src_lane);
                v_scale_local.z = reduce_field(v_scale_local.z, n_contrib, src_lane);
                v_quat_local.x = reduce_field(v_quat_local.x, n_contrib, src_lane);
                v_quat_local.y = reduce_field(v_quat_local.y, n_contrib, src_lane);
                v_quat_local.z = reduce_field(v_quat_local.z, n_contrib, src_lane);
                v_quat_local.w = reduce_field(v_quat_local.w, n_contrib, src_lane);
                v_opacity_local = reduce_field(v_opacity_local, n_contrib, src_lane);
                densification_weight_local = reduce_field(densification_weight_local, n_contrib, src_lane);
                densification_error_weighted_local =
                    reduce_field(densification_error_weighted_local, n_contrib, src_lane);
                if (warp.thread_rank() == 0) {
                    flush_splat_grads(
                        id_batch[t], N, do_dens, v_rgb_local, v_mean_local, v_scale_local,
                        v_quat_local, v_opacity_local, densification_weight_local,
                        densification_error_weighted_local, scale_act, xyz_opac[3],
                        v_means, v_quats, v_scales, v_colors, v_opacities, densification_info);
                }
            }
        }
    }

    template <uint32_t CDIM, typename scalar_t, bool kPerfectPinhole>
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

        const WorldRay ray = from_world_pixel_ray<kPerfectPinhole>(
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
        int32_t range_end = tile_offsets[tile_id + 1];
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

        if (n_isects == 0) {
            // Skip kernel launch if no intersections
            return;
        }

        const bool global_shutter = is_global_shutter_launch(rs_type, viewmats1);
        const bool perfect_pinhole = is_perfect_pinhole_launch(
            camera_model, radial_coeffs, tangential_coeffs, thin_prism_coeffs);

        auto launch_args = [&](auto kernel, dim3 grid, dim3 threads, int64_t shmem_size) {
            set_kernel_max_dynamic_smem(
                kernel, static_cast<int>(shmem_size), "gsplat backward");
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
            if (tile_size == 16) {
                dim3 threads = {8, 16, 1};
                dim3 grid = {C, tile_height, tile_width};
                int64_t shmem_size =
                    int64_t(kBwdDualBatch) *
                    (sizeof(int32_t) + sizeof(vec4) + sizeof(vec3) + sizeof(vec4) +
                     sizeof(mat3) + sizeof(float) * CDIM);
                if (global_shutter) {
                    if (perfect_pinhole) {
                        launch_args(
                            rasterize_to_pixels_from_world_3dgs_bwd_dual_kernel<float, true, true>,
                            grid, threads, shmem_size);
                    } else {
                        launch_args(
                            rasterize_to_pixels_from_world_3dgs_bwd_dual_kernel<float, true, false>,
                            grid, threads, shmem_size);
                    }
                    return;
                }
                launch_args(
                    rasterize_to_pixels_from_world_3dgs_bwd_dual_kernel<float, false, false>,
                    grid, threads, shmem_size);
                return;
            }
        }

        dim3 threads = {tile_size, tile_size, 1};
        dim3 grid = {C, tile_height, tile_width};
        int64_t shmem_size =
            tile_size * tile_size *
            (sizeof(int32_t) + sizeof(vec4) + sizeof(vec3) + sizeof(vec4) +
             sizeof(mat3) + sizeof(float) * CDIM);
        if constexpr (CDIM == 3) {
            constexpr int64_t kOccCapBytes = 49152;
            if (shmem_size < kOccCapBytes) {
                shmem_size = kOccCapBytes;
            }
        }
        if constexpr (CDIM == 3) {
            if (global_shutter && perfect_pinhole) {
                launch_args(
                    rasterize_to_pixels_from_world_3dgs_bwd_kernel<CDIM, float, true>,
                    grid, threads, shmem_size);
                return;
            }
        }
        launch_args(
            rasterize_to_pixels_from_world_3dgs_bwd_kernel<CDIM, float, false>,
            grid, threads, shmem_size);
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
