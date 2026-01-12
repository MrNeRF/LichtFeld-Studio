/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "Cameras.cuh"
#include "Common.h"
#include "Projection.h"
#include "Utils.cuh"

namespace gsplat_fwd {

    namespace cg = cooperative_groups;

    template <typename scalar_t>
    __global__ void projection_ut_3dgs_fused_kernel(
        const uint32_t C,
        const uint32_t N_total,                 // Total gaussians in input arrays
        const uint32_t M,                       // Visible gaussians to process
        const scalar_t* __restrict__ means,     // [N_total, 3]
        const scalar_t* __restrict__ quats,     // [N_total, 4]
        const scalar_t* __restrict__ scales,    // [N_total, 3]
        const scalar_t* __restrict__ opacities, // [N_total] optional
        const scalar_t* __restrict__ viewmats0, // [C, 4, 4]
        const scalar_t* __restrict__ viewmats1, // [C, 4, 4] optional for rolling shutter
        const scalar_t* __restrict__ Ks,        // [C, 3, 3]
        const uint32_t image_width,
        const uint32_t image_height,
        const float eps2d,
        const float near_plane,
        const float far_plane,
        const float radius_clip,
        const CameraModelType camera_model_type,
        // uncented transform
        const UnscentedTransformParameters ut_params,
        const ShutterType rs_type,
        const scalar_t* __restrict__ radial_coeffs,     // [C, 6] or [C, 4] optional
        const scalar_t* __restrict__ tangential_coeffs, // [C, 2] optional
        const scalar_t* __restrict__ thin_prism_coeffs, // [C, 2] optional
        // node visibility culling (used when visible_indices is null)
        const int* __restrict__ transform_indices,     // [N_total] optional
        const bool* __restrict__ node_visibility_mask, // [num_visibility_nodes] optional
        const int num_visibility_nodes,
        // indirect indexing
        const int* __restrict__ visible_indices, // [M] maps output idx → global gaussian idx
        // outputs (sized to [C, M, ...])
        int32_t* __restrict__ radii,         // [C, M, 2]
        scalar_t* __restrict__ means2d,      // [C, M, 2]
        scalar_t* __restrict__ depths,       // [C, M]
        scalar_t* __restrict__ conics,       // [C, M, 3]
        scalar_t* __restrict__ compensations // [C, M] optional
    ) {
        // parallelize over C * M
        uint32_t idx = cg::this_grid().thread_rank();
        if (idx >= C * M) {
            return;
        }
        const uint32_t cid = idx / M;     // camera id
        const uint32_t out_gid = idx % M; // output gaussian index (0..M-1)

        // Map to global gaussian index if using visibility filtering
        const uint32_t gid = (visible_indices != nullptr)
                                 ? static_cast<uint32_t>(visible_indices[out_gid])
                                 : out_gid;

        // Node visibility check only when not using visible_indices (already pre-filtered)
        if (visible_indices == nullptr && node_visibility_mask != nullptr &&
            transform_indices != nullptr && num_visibility_nodes > 0) {
            const int node_idx = transform_indices[gid];
            if (node_idx >= 0 && node_idx < num_visibility_nodes && !node_visibility_mask[node_idx]) {
                radii[idx * 2] = 0;
                radii[idx * 2 + 1] = 0;
                return;
            }
        }

        // Read from global gaussian index
        const glm::fvec3 mean = glm::make_vec3(means + gid * 3);
        const glm::fvec3 scale = glm::make_vec3(scales + gid * 3);
        glm::fquat quat = glm::fquat{
            quats[gid * 4 + 0],
            quats[gid * 4 + 1],
            quats[gid * 4 + 2],
            quats[gid * 4 + 3]}; // w,x,y,z quaternion
        quat = glm::normalize(quat);

        // shift pointers to the current camera. note that glm is colume-major.
        const vec2 focal_length = {Ks[cid * 9 + 0], Ks[cid * 9 + 4]};
        const vec2 principal_point = {Ks[cid * 9 + 2], Ks[cid * 9 + 5]};

        // Create rolling shutter parameter
        auto rs_params = RollingShutterParameters(
            viewmats0 + cid * 16,
            viewmats1 == nullptr ? nullptr : viewmats1 + cid * 16);

        // transform Gaussian center to camera space
        // Interpolate to *center* shutter pose as single per-Gaussian camera pose
        const auto shutter_pose = interpolate_shutter_pose(0.5f, rs_params);
        const vec3 mean_c = glm::rotate(shutter_pose.q, mean) + shutter_pose.t;
        if ((mean_c.z < near_plane && camera_model_type != CameraModelType::EQUIRECTANGULAR) || mean_c.z > far_plane) {
            radii[idx * 2] = 0;
            radii[idx * 2 + 1] = 0;
            return;
        }

        // projection using uncented transform
        ImageGaussianReturn image_gaussian_return;
        if (camera_model_type == CameraModelType::PINHOLE) {
            if (radial_coeffs == nullptr && tangential_coeffs == nullptr && thin_prism_coeffs == nullptr) {
                PerfectPinholeCameraModel::Parameters cm_params = {};
                cm_params.resolution = {image_width, image_height};
                cm_params.shutter_type = rs_type;
                cm_params.principal_point = {principal_point.x, principal_point.y};
                cm_params.focal_length = {focal_length.x, focal_length.y};
                PerfectPinholeCameraModel camera_model(cm_params);
                image_gaussian_return =
                    world_gaussian_to_image_gaussian_unscented_transform_shutter_pose(
                        camera_model, rs_params, ut_params, mean, scale, quat);
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
                image_gaussian_return =
                    world_gaussian_to_image_gaussian_unscented_transform_shutter_pose(
                        camera_model, rs_params, ut_params, mean, scale, quat);
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
            image_gaussian_return =
                world_gaussian_to_image_gaussian_unscented_transform_shutter_pose(
                    camera_model, rs_params, ut_params, mean, scale, quat);
        } else if (camera_model_type == CameraModelType::EQUIRECTANGULAR) {
            EquirectangularCameraModel::Parameters cm_params = {};
            cm_params.resolution = {image_width, image_height};
            EquirectangularCameraModel camera_model(cm_params);
            image_gaussian_return =
                world_gaussian_to_image_gaussian_unscented_transform_shutter_pose(
                    camera_model, rs_params, ut_params, mean, scale, quat);
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
            image_gaussian_return =
                world_gaussian_to_image_gaussian_unscented_transform_shutter_pose(
                    camera_model, rs_params, ut_params, mean, scale, quat);
        } else {
            // should never reach here
            assert(false);
            return;
        }

        auto [mean2d, covar2d, valid_ut] = image_gaussian_return;
        if (!valid_ut) {
            radii[idx * 2] = 0;
            radii[idx * 2 + 1] = 0;
            return;
        }

        float compensation;
        float det = add_blur(eps2d, covar2d, compensation);
        if (det <= 0.f) {
            radii[idx * 2] = 0;
            radii[idx * 2 + 1] = 0;
            return;
        }

        // compute the inverse of the 2d covariance
        mat2 covar2d_inv = glm::inverse(covar2d);

        float extend = 3.33f;
        if (opacities != nullptr) {
            float opacity = opacities[gid];
            opacity *= compensation;
            if (opacity < ALPHA_THRESHOLD) {
                radii[idx * 2] = 0;
                radii[idx * 2 + 1] = 0;
                return;
            }
            // Compute opacity-aware bounding box.
            // https://arxiv.org/pdf/2402.00525 Section B.2
            extend = min(extend, sqrt(2.0f * __logf(opacity / ALPHA_THRESHOLD)));
        }

        // compute tight rectangular bounding box (non differentiable)
        // https://arxiv.org/pdf/2402.00525
        float b = 0.5f * (covar2d[0][0] + covar2d[1][1]);
        float tmp = sqrtf(max(0.01f, b * b - det));
        float v1 = b + tmp; // larger eigenvalue
        float r1 = extend * sqrtf(v1);
        float radius_x = ceilf(min(extend * sqrtf(covar2d[0][0]), r1));
        float radius_y = ceilf(min(extend * sqrtf(covar2d[1][1]), r1));

        if (radius_x <= radius_clip && radius_y <= radius_clip) {
            radii[idx * 2] = 0;
            radii[idx * 2 + 1] = 0;
            return;
        }

        // mask out gaussians outside the image region
        if (mean2d.x + radius_x <= 0 || mean2d.x - radius_x >= image_width ||
            mean2d.y + radius_y <= 0 || mean2d.y - radius_y >= image_height) {
            radii[idx * 2] = 0;
            radii[idx * 2 + 1] = 0;
            return;
        }

        // write to outputs
        radii[idx * 2] = (int32_t)radius_x;
        radii[idx * 2 + 1] = (int32_t)radius_y;
        means2d[idx * 2] = mean2d.x;
        means2d[idx * 2 + 1] = mean2d.y;
        depths[idx] = mean_c.z;
        conics[idx * 3] = covar2d_inv[0][0];
        conics[idx * 3 + 1] = covar2d_inv[0][1];
        conics[idx * 3 + 2] = covar2d_inv[1][1];
        if (compensations != nullptr) {
            compensations[idx] = compensation;
        }
    }

    void launch_projection_ut_3dgs_fused_kernel(
        // inputs
        const float* means,     // [N_total, 3]
        const float* quats,     // [N_total, 4]
        const float* scales,    // [N_total, 3]
        const float* opacities, // [N_total] optional (can be nullptr)
        const float* viewmats0, // [C, 4, 4]
        const float* viewmats1, // [C, 4, 4] optional for rolling shutter (can be nullptr)
        const float* Ks,        // [C, 3, 3]
        uint32_t N_total,       // Total gaussians in input arrays
        uint32_t M,             // Visible gaussians to process
        uint32_t C,
        uint32_t image_width,
        uint32_t image_height,
        float eps2d,
        float near_plane,
        float far_plane,
        float radius_clip,
        CameraModelType camera_model,
        // uncented transform
        const UnscentedTransformParameters& ut_params,
        ShutterType rs_type,
        const float* radial_coeffs,     // [C, 6] or [C, 4] optional (can be nullptr)
        const float* tangential_coeffs, // [C, 2] optional (can be nullptr)
        const float* thin_prism_coeffs, // [C, 2] optional (can be nullptr)
        // node visibility culling
        const int* transform_indices,     // [N_total] optional (can be nullptr)
        const bool* node_visibility_mask, // [num_visibility_nodes] optional (can be nullptr)
        int num_visibility_nodes,
        const int* visible_indices, // [M] maps output idx → global gaussian idx (nullptr = all visible)
        // outputs (sized to [C, M, ...])
        int32_t* radii,       // [C, M, 2]
        float* means2d,       // [C, M, 2]
        float* depths,        // [C, M]
        float* conics,        // [C, M, 3]
        float* compensations, // [C, M] optional (can be nullptr)
        cudaStream_t stream) {
        int64_t n_elements = C * M;
        dim3 threads(256);
        dim3 grid((n_elements + threads.x - 1) / threads.x);
        int64_t shmem_size = 0; // No shared memory used in this kernel

        if (n_elements == 0) {
            // skip the kernel launch if there are no elements
            return;
        }

        projection_ut_3dgs_fused_kernel<float>
            <<<grid, threads, shmem_size, stream>>>(
                C,
                N_total,
                M,
                means,
                quats,
                scales,
                opacities,
                viewmats0,
                viewmats1,
                Ks,
                image_width,
                image_height,
                eps2d,
                near_plane,
                far_plane,
                radius_clip,
                camera_model,
                ut_params,
                rs_type,
                radial_coeffs,
                tangential_coeffs,
                thin_prism_coeffs,
                transform_indices,
                node_visibility_mask,
                num_visibility_nodes,
                visible_indices,
                radii,
                means2d,
                depths,
                conics,
                compensations);
    }

} // namespace gsplat_fwd
