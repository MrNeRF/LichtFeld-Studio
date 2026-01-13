/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cooperative_groups.h>
#include <cstdio>
#include <cuda_runtime.h>

#include "Cameras.cuh"
#include "Common.h"
#include "Rasterization.h"
#include "Utils.cuh"

namespace gsplat_fwd {

    namespace cg = cooperative_groups;

    namespace {
        constexpr float FAR_DEPTH = 1e10f;
    }

    ////////////////////////////////////////////////////////////////
    // Forward Kernel
    ////////////////////////////////////////////////////////////////

    template <uint32_t CDIM, typename scalar_t>
    __global__ void rasterize_to_pixels_from_world_3dgs_fwd_kernel(
        const uint32_t C,
        const uint32_t N,
        const uint32_t n_isects,
        const bool packed,
        const vec3* __restrict__ means,           // [N_total, 3] (N-sized, use visible_indices for access)
        const vec4* __restrict__ quats,           // [N_total, 4] (N-sized, use visible_indices for access)
        const vec3* __restrict__ scales,          // [N_total, 3] (N-sized, use visible_indices for access)
        const scalar_t* __restrict__ colors,      // [C, M, CDIM] or [nnz, CDIM] (M-sized from SH)
        const scalar_t* __restrict__ opacities,   // [N_total] (N-sized, use visible_indices for access)
        const scalar_t* __restrict__ backgrounds, // [C, CDIM]
        const bool* __restrict__ masks,           // [C, tile_height, tile_width]
        const scalar_t* __restrict__ depths,      // [C, M] per-gaussian depths (M-sized from projection)
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
        // indirect indexing for visibility filtering
        const int32_t* __restrict__ visible_indices, // [M] maps g -> global gaussian idx, nullptr = direct
        scalar_t* __restrict__ render_colors,        // [C, image_height, image_width, CDIM]
        scalar_t* __restrict__ render_alphas,        // [C, image_height, image_width, 1]
        int32_t* __restrict__ last_ids,              // [C, image_height, image_width]
        scalar_t* __restrict__ median_depths         // [C, image_height, image_width]
    ) {
        // each thread draws one pixel, but also timeshares caching gaussians in a
        // shared tile

        auto block = cg::this_thread_block();
        int32_t cid = block.group_index().x;
        int32_t tile_id =
            block.group_index().y * tile_width + block.group_index().z;
        uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
        uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

        tile_offsets += cid * tile_height * tile_width;
        render_colors += cid * image_height * image_width * CDIM;
        render_alphas += cid * image_height * image_width;
        last_ids += cid * image_height * image_width;
        if (median_depths != nullptr) {
            median_depths += cid * image_height * image_width;
        }
        if (depths != nullptr) {
            depths += cid * N;
        }
        if (backgrounds != nullptr) {
            backgrounds += cid * CDIM;
        }
        if (masks != nullptr) {
            masks += cid * tile_height * tile_width;
        }

        float px = (float)j + 0.5f;
        float py = (float)i + 0.5f;
        int32_t pix_id = i * image_width + j;

        // Create rolling shutter parameter
        auto rs_params = RollingShutterParameters(
            viewmats0 + cid * 16,
            viewmats1 == nullptr ? nullptr : viewmats1 + cid * 16);
        // shift pointers to the current camera. note that glm is colume-major.
        const vec2 focal_length = {Ks[cid * 9 + 0], Ks[cid * 9 + 4]};
        const vec2 principal_point = {Ks[cid * 9 + 2], Ks[cid * 9 + 5]};

        // Create ray from pixel
        WorldRay ray;
        if (camera_model_type == CameraModelType::PINHOLE) {
            if (radial_coeffs == nullptr && tangential_coeffs == nullptr && thin_prism_coeffs == nullptr) {
                PerfectPinholeCameraModel::Parameters cm_params = {};
                cm_params.resolution = {image_width, image_height};
                cm_params.shutter_type = rs_type;
                cm_params.principal_point = {principal_point.x, principal_point.y};
                cm_params.focal_length = {focal_length.x, focal_length.y};
                PerfectPinholeCameraModel camera_model(cm_params);
                ray = camera_model.image_point_to_world_ray_shutter_pose(vec2(px, py), rs_params);
            } else {
                OpenCVPinholeCameraModel<>::Parameters cm_params = {};
                cm_params.resolution = {image_width, image_height};
                cm_params.shutter_type = rs_type;
                cm_params.principal_point = {principal_point.x, principal_point.y};
                cm_params.focal_length = {focal_length.x, focal_length.y};
                if (radial_coeffs != nullptr) {
                    cm_params.radial_coeffs = make_array<float, 6>(radial_coeffs + cid * 6);
                }
                if (tangential_coeffs != nullptr) {
                    cm_params.tangential_coeffs = make_array<float, 2>(tangential_coeffs + cid * 2);
                }
                if (thin_prism_coeffs != nullptr) {
                    cm_params.thin_prism_coeffs = make_array<float, 4>(thin_prism_coeffs + cid * 4);
                }
                OpenCVPinholeCameraModel camera_model(cm_params);
                ray = camera_model.image_point_to_world_ray_shutter_pose(vec2(px, py), rs_params);
            }
        } else if (camera_model_type == CameraModelType::FISHEYE) {
            OpenCVFisheyeCameraModel<>::Parameters cm_params = {};
            cm_params.resolution = {image_width, image_height};
            cm_params.shutter_type = rs_type;
            cm_params.principal_point = {principal_point.x, principal_point.y};
            cm_params.focal_length = {focal_length.x, focal_length.y};
            if (radial_coeffs != nullptr) {
                cm_params.radial_coeffs = make_array<float, 4>(radial_coeffs + cid * 4);
            }
            OpenCVFisheyeCameraModel camera_model(cm_params);
            ray = camera_model.image_point_to_world_ray_shutter_pose(vec2(px, py), rs_params);
        } else if (camera_model_type == CameraModelType::EQUIRECTANGULAR) {
            EquirectangularCameraModel::Parameters cm_params = {};
            cm_params.resolution = {image_width, image_height};
            cm_params.shutter_type = rs_type;
            EquirectangularCameraModel camera_model(cm_params);
            ray = camera_model.image_point_to_world_ray_shutter_pose(vec2(px, py), rs_params);
        } else if (camera_model_type == CameraModelType::THIN_PRISM_FISHEYE) {
            ThinPrismFisheyeCameraModel<>::Parameters cm_params = {};
            cm_params.resolution = {image_width, image_height};
            cm_params.shutter_type = rs_type;
            cm_params.principal_point = {principal_point.x, principal_point.y};
            cm_params.focal_length = {focal_length.x, focal_length.y};
            if (radial_coeffs != nullptr) {
                cm_params.radial_coeffs = make_array<float, 4>(radial_coeffs + cid * 4);
            }
            if (thin_prism_coeffs != nullptr) {
                cm_params.thin_prism_coeffs = make_array<float, 4>(thin_prism_coeffs + cid * 4);
            }
            ThinPrismFisheyeCameraModel camera_model(cm_params);
            ray = camera_model.image_point_to_world_ray_shutter_pose(vec2(px, py), rs_params);
        } else {
            // should never reach here
            assert(false);
            return;
        }
        const vec3 ray_d = ray.ray_dir;
        const vec3 ray_o = ray.ray_org;

        // return if out of bounds
        // keep not rasterizing threads around for reading data
        bool inside = (i < image_height && j < image_width);
        bool done = (!inside) || (!ray.valid_flag);

        // when the mask is provided, render the background color and return
        // if this tile is labeled as False
        if (masks != nullptr && inside && !masks[tile_id]) {
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                render_colors[pix_id * CDIM + k] =
                    backgrounds == nullptr ? 0.0f : backgrounds[k];
            }
            return;
        }

        // have all threads in tile process the same gaussians in batches
        // first collect gaussians between range.x and range.y in batches
        // which gaussians to look through in this tile
        int32_t range_start = tile_offsets[tile_id];
        int32_t range_end =
            (cid == C - 1) && (tile_id == tile_width * tile_height - 1)
                ? n_isects
                : tile_offsets[tile_id + 1];
        const uint32_t block_size = block.size();
        uint32_t num_batches =
            (range_end - range_start + block_size - 1) / block_size;

        extern __shared__ int s[];
        int32_t* id_batch = (int32_t*)s; // [block_size]
        vec4* xyz_opacity_batch =
            reinterpret_cast<vec4*>(&id_batch[block_size]); // [block_size]
        mat3* iscl_rot_batch =
            reinterpret_cast<mat3*>(&xyz_opacity_batch[block_size]); // [block_size]

        float T = 1.0f;
        uint32_t cur_idx = 0;
        float median_depth = FAR_DEPTH;

        // collect and process batches of gaussians
        // each thread loads one gaussian at a time before rasterizing its
        // designated pixel
        uint32_t tr = block.thread_rank();

        float pix_out[CDIM] = {0.f};
        for (uint32_t b = 0; b < num_batches; ++b) {
            // resync all threads before beginning next batch
            // end early if entire tile is done
            if (__syncthreads_count(done) >= block_size) {
                break;
            }

            // each thread fetch 1 gaussian from front to back
            // index of gaussian to load
            uint32_t batch_start = range_start + block_size * b;
            uint32_t idx = batch_start + tr;
            if (idx < range_end) {
                // g indexes into M-sized arrays (colors, depths)
                int32_t g = flatten_ids[idx]; // flatten index in [0..M-1]
                id_batch[tr] = g;

                // global_g indexes into N-sized arrays (means, quats, scales, opacities)
                const int32_t global_g = (visible_indices != nullptr) ? visible_indices[g] : g;
                const vec3 xyz = means[global_g];
                const float opac = opacities[global_g];
                xyz_opacity_batch[tr] = {xyz.x, xyz.y, xyz.z, opac};

                const vec4 quat = quats[global_g];
                vec3 scale = scales[global_g];

                mat3 R = quat_to_rotmat(quat);
                mat3 S = mat3(
                    1.0f / scale[0],
                    0.f,
                    0.f,
                    0.f,
                    1.0f / scale[1],
                    0.f,
                    0.f,
                    0.f,
                    1.0f / scale[2]);
                mat3 iscl_rot = S * glm::transpose(R);
                iscl_rot_batch[tr] = iscl_rot;
            }

            // wait for other threads to collect the gaussians in batch
            block.sync();

            // process gaussians in the current batch for this pixel
            uint32_t batch_size = min(block_size, range_end - batch_start);
            for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
                const vec4 xyz_opac = xyz_opacity_batch[t];
                const float opac = xyz_opac[3];
                const vec3 xyz = {xyz_opac[0], xyz_opac[1], xyz_opac[2]};
                const mat3 iscl_rot = iscl_rot_batch[t];

                const vec3 gro = iscl_rot * (ray_o - xyz);
                const vec3 grd = safe_normalize(iscl_rot * ray_d);
                const vec3 gcrod = glm::cross(grd, gro);
                const float grayDist = glm::dot(gcrod, gcrod);
                const float power = -0.5f * grayDist;

                float alpha = min(0.999f, opac * __expf(power));
                if (alpha < 1.f / 255.f) {
                    continue;
                }

                const float next_T = T * (1.0f - alpha);
                const int32_t g = id_batch[t];

                // Median depth: check before early exit to catch high-opacity cases
                if (depths != nullptr && T > 0.5f && next_T <= 0.5f) {
                    median_depth = depths[g];
                }

                if (next_T <= 1e-4f) {
                    done = true;
                    break;
                }

                const float vis = alpha * T;
                const float* c_ptr = colors + g * CDIM;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    pix_out[k] += c_ptr[k] * vis;
                }
                cur_idx = batch_start + t;
                T = next_T;
            }
        }

        if (inside) {
            render_alphas[pix_id] = 1.0f - T;
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                render_colors[pix_id * CDIM + k] =
                    backgrounds == nullptr ? pix_out[k]
                                           : (pix_out[k] + T * backgrounds[k]);
            }
            last_ids[pix_id] = static_cast<int32_t>(cur_idx);
            if (median_depths != nullptr) {
                median_depths[pix_id] = median_depth;
            }
        }
    }

    ////////////////////////////////////////////////////////////////
    // Launch Function
    ////////////////////////////////////////////////////////////////

    template <uint32_t CDIM>
    void launch_rasterize_to_pixels_from_world_3dgs_fwd_kernel(
        const float* means,
        const float* quats,
        const float* scales,
        const float* colors,
        const float* opacities,
        const float* backgrounds,
        const bool* masks,
        const float* depths,
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
        const int32_t* visible_indices,
        float* renders,
        float* alphas,
        int32_t* last_ids,
        float* median_depths,
        cudaStream_t stream) {
        const bool packed = false; // Only support non-packed for now
        const uint32_t tile_width = (image_width + tile_size - 1) / tile_size;
        const uint32_t tile_height = (image_height + tile_size - 1) / tile_size;

        // Each block covers a tile on the image. In total there are
        // C * tile_height * tile_width blocks.
        dim3 threads = {tile_size, tile_size, 1};
        dim3 grid = {C, tile_height, tile_width};

        int64_t shmem_size =
            tile_size * tile_size *
            (sizeof(int32_t) + sizeof(vec4) + sizeof(mat3));

        if (n_isects == 0) {
            // Skip kernel launch if no intersections
            // Still need to clear output buffers
            cudaMemsetAsync(renders, 0, C * image_height * image_width * CDIM * sizeof(float), stream);
            cudaMemsetAsync(alphas, 0, C * image_height * image_width * sizeof(float), stream);
            cudaMemsetAsync(last_ids, 0, C * image_height * image_width * sizeof(int32_t), stream);
            return;
        }

        auto err = cudaFuncSetAttribute(
            rasterize_to_pixels_from_world_3dgs_fwd_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "GSPLAT ERROR: Failed to set maximum shared memory size "
                    "(requested %ld bytes), try lowering tile_size. CUDA error: %s\n",
                    shmem_size, cudaGetErrorString(err));
            return;
        }

        rasterize_to_pixels_from_world_3dgs_fwd_kernel<CDIM, float>
            <<<grid, threads, shmem_size, stream>>>(
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
                masks,
                depths,
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
                visible_indices,
                renders,
                alphas,
                last_ids,
                median_depths);
    }

    ////////////////////////////////////////////////////////////////
    // Explicit Instantiations
    ////////////////////////////////////////////////////////////////

#define __INS__(CDIM)                                                          \
    template void launch_rasterize_to_pixels_from_world_3dgs_fwd_kernel<CDIM>( \
        const float* means,                                                    \
        const float* quats,                                                    \
        const float* scales,                                                   \
        const float* colors,                                                   \
        const float* opacities,                                                \
        const float* backgrounds,                                              \
        const bool* masks,                                                     \
        const float* depths,                                                   \
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
        const int32_t* visible_indices,                                        \
        float* renders,                                                        \
        float* alphas,                                                         \
        int32_t* last_ids,                                                     \
        float* median_depths,                                                  \
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

} // namespace gsplat_fwd
