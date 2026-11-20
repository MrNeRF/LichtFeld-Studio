/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "Cameras.cuh"
#include "Common.h"
#include "Projection.h"
#include "Utils.cuh"

namespace gsplat_lfs {

    namespace cg = cooperative_groups;

    // Pixels per world unit along two directions perpendicular to the view ray through the splat centre, from the
    // camera model's own projection: forward differences with a step of 1e-3 of the distance. Zero if a point fails.
    template <class CameraModel>
    inline __device__ glm::fmat2 view_plane_jacobian(
        CameraModel const& camera_model, RollingShutterParameters const& rs_params, const float margin,
        const glm::fvec3& mean, const glm::fvec3& mean_c, const glm::fquat& world_to_camera) {
        const glm::fvec3 dir = glm::normalize(mean_c);
        const glm::fvec3 u_c = glm::normalize(glm::cross(dir, fabsf(dir.x) < 0.9f ? glm::fvec3(1.f, 0.f, 0.f) : glm::fvec3(0.f, 1.f, 0.f)));
        const glm::fquat camera_to_world = glm::inverse(world_to_camera);
        const float h = 1e-3f * glm::length(mean_c);
        auto project = [&](const glm::fvec3& point) {
            return camera_model.template world_point_to_image_point_shutter_pose<>(point, rs_params, margin);
        };
        const auto p0 = project(mean);
        const auto pu = project(mean + h * glm::rotate(camera_to_world, u_c));
        const auto pv = project(mean + h * glm::rotate(camera_to_world, glm::cross(dir, u_c)));
        if (!p0.valid_flag || !pu.valid_flag || !pv.valid_flag)
            return glm::fmat2(0.f);
        return glm::fmat2((pu.imagePoint - p0.imagePoint) / h, (pv.imagePoint - p0.imagePoint) / h);
    }

    // sqrt(N(s) / N(s')): N is the determinant of the splat's covariance marginalised perpendicular to the view
    // direction d (splat frame), so the factor keeps the splat's integrated contribution when every squared scale
    // grows by sigma2. Written without divisions so needle-thin splats stay finite.
    inline __device__ float mip_compensation(const vec3 scale, const vec4 quat, const vec3 view_dir, const float sigma2) {
        const vec3 d = glm::transpose(quat_to_rotmat(quat)) * view_dir;
        const vec3 a = scale * scale;
        const vec3 b = a + sigma2;
        const float n = d.x * d.x * a.y * a.z + d.y * d.y * a.x * a.z + d.z * d.z * a.x * a.y;
        const float n_dilated = d.x * d.x * b.y * b.z + d.y * d.y * b.x * b.z + d.z * d.z * b.x * b.y;
        return n_dilated > 0.f ? sqrtf(n / n_dilated) : 1.f;
    }

    template <typename scalar_t>
    __global__ void projection_ut_3dgs_fused_kernel(
        const uint32_t C,
        const uint32_t N,
        const scalar_t* __restrict__ means,     // [N, 3]
        const scalar_t* __restrict__ quats,     // [N, 4]
        const scalar_t* __restrict__ scales,    // [N, 3]
        const scalar_t* __restrict__ opacities, // [N] optional
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
        // outputs
        int32_t* __restrict__ radii,          // [C, N, 2]
        scalar_t* __restrict__ means2d,       // [C, N, 2]
        scalar_t* __restrict__ depths,        // [C, N]
        scalar_t* __restrict__ conics,        // [C, N, 3]
        scalar_t* __restrict__ compensations, // [C, N] optional
        scalar_t* __restrict__ dilations      // [C, N] optional: mip filter on, world variance added to s^2
    ) {
        // parallelize over C * N.
        uint32_t idx = cg::this_grid().thread_rank();
        if (idx >= C * N) {
            return;
        }
        const uint32_t cid = idx / N; // camera id
        const uint32_t gid = idx % N; // gaussian id

        // shift pointers to the current gaussian
        const glm::fvec3 mean = glm::make_vec3(means + gid * 3);
        const glm::fvec3 scale_raw = glm::make_vec3(scales + gid * 3);
        const glm::fvec3 scale = glm::fvec3(
            expf(scale_raw.x), expf(scale_raw.y), expf(scale_raw.z));
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
        const float depth = camera_model_type == CameraModelType::PINHOLE ? mean_c.z : glm::length(mean_c);
        if (depth < near_plane || depth > far_plane) {
            radii[idx * 2] = 0;
            radii[idx * 2 + 1] = 0;
            return;
        }

        // projection using uncented transform
        ImageGaussianReturn image_gaussian_return;
        glm::fmat2 J(0.f); // mip filter: pixels per world unit at the splat
        auto project = [&](auto const& camera_model) {
            if (dilations != nullptr)
                J = view_plane_jacobian(camera_model, rs_params, ut_params.in_image_margin_factor, mean, mean_c, shutter_pose.q);
            return world_gaussian_to_image_gaussian_unscented_transform_shutter_pose(
                camera_model, rs_params, ut_params, mean, scale, quat);
        };
        if (camera_model_type == CameraModelType::PINHOLE) {
            if (radial_coeffs == nullptr && tangential_coeffs == nullptr && thin_prism_coeffs == nullptr) {
                PerfectPinholeCameraModel::Parameters cm_params = {};
                cm_params.resolution = {image_width, image_height};
                cm_params.shutter_type = rs_type;
                cm_params.principal_point = {principal_point.x, principal_point.y};
                cm_params.focal_length = {focal_length.x, focal_length.y};
                PerfectPinholeCameraModel camera_model(cm_params);
                image_gaussian_return = project(camera_model);
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
                image_gaussian_return = project(camera_model);
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
            image_gaussian_return = project(camera_model);
        } else if (camera_model_type == CameraModelType::EQUIRECTANGULAR) {
            // For equirectangular cameras in tile mode, the K matrix encodes tile information:
            //   K[0][0] (focal_length.x) = full_image_width
            //   K[1][1] (focal_length.y) = full_image_height
            //   K[0][2] (principal_point.x) = tile_x_offset
            //   K[1][2] (principal_point.y) = tile_y_offset
            // This avoids changing all function interfaces for a camera-specific fix.
            // For non-tiled rendering, full_image_width/height == image_width/height and offsets are 0.
            const uint32_t full_image_width = static_cast<uint32_t>(focal_length.x);
            const uint32_t full_image_height = static_cast<uint32_t>(focal_length.y);
            const float tile_x_offset = principal_point.x;
            const float tile_y_offset = principal_point.y;

            assert(full_image_width >= image_width && "full_image_width must be >= image_width");
            assert(full_image_height >= image_height && "full_image_height must be >= image_height");

            EquirectangularCameraModel::Parameters cm_params = {};
            cm_params.resolution = {full_image_width, full_image_height};
            EquirectangularCameraModel camera_model(cm_params);
            image_gaussian_return = project(camera_model);

            // Convert from full image coordinates to tile-local coordinates
            image_gaussian_return.mean.x -= tile_x_offset;
            image_gaussian_return.mean.y -= tile_y_offset;
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
            image_gaussian_return = project(camera_model);
        } else {
            // should never reach here
            assert(false);
            return;
        }

        const glm::fvec2 mean2d = image_gaussian_return.mean;
        glm::fmat2 covar2d = image_gaussian_return.covariance;
        const bool valid_ut = image_gaussian_return.valid;
        if (!valid_ut) {
            radii[idx * 2] = 0;
            radii[idx * 2 + 1] = 0;
            return;
        }

        // Mip filter: eps2d px^2 in the image is the isotropic world variance sigma2 = eps2d / |det J| at the
        // splat. The ray kernels add it to every squared scale; the bounds take its largest image eigenvalue.
        float sigma2 = 0.f;
        float blur = eps2d;
        if (dilations != nullptr) {
            const float det_J = J[0][0] * J[1][1] - J[1][0] * J[0][1];
            const float variance = eps2d / fabsf(det_J);
            if (isfinite(variance)) {
                const glm::fmat2 JJt = J * glm::transpose(J);
                const float half_trace = 0.5f * (JJt[0][0] + JJt[1][1]);
                sigma2 = variance;
                blur = max(eps2d, variance * (half_trace + sqrtf(max(0.f, half_trace * half_trace - det_J * det_J))));
            }
        }

        float compensation;
        float det = add_blur(blur, covar2d, compensation);
        if (det <= 0.f) {
            radii[idx * 2] = 0;
            radii[idx * 2 + 1] = 0;
            return;
        }

        // compute the inverse of the 2d covariance
        mat2 covar2d_inv = glm::inverse(covar2d);

        if (dilations != nullptr) {
            const vec3 view_dir = glm::rotate(glm::inverse(shutter_pose.q), glm::normalize(mean_c));
            compensation = mip_compensation(scale, vec4(quat.w, quat.x, quat.y, quat.z), view_dir, sigma2);
        }

        float extend = 3.33f;
        if (opacities != nullptr) {
            float opacity = activated_opacity(opacities[gid]);
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
        depths[idx] = depth;
        conics[idx * 3] = covar2d_inv[0][0];
        conics[idx * 3 + 1] = covar2d_inv[0][1];
        conics[idx * 3 + 2] = covar2d_inv[1][1];
        if (compensations != nullptr) {
            compensations[idx] = compensation;
        }
        if (dilations != nullptr) {
            dilations[idx] = sigma2;
        }
    }

    void launch_projection_ut_3dgs_fused_kernel(
        // inputs
        const float* means,     // [N, 3]
        const float* quats,     // [N, 4]
        const float* scales,    // [N, 3]
        const float* opacities, // [N] optional (can be nullptr)
        const float* viewmats0, // [C, 4, 4]
        const float* viewmats1, // [C, 4, 4] optional for rolling shutter (can be nullptr)
        const float* Ks,        // [C, 3, 3]
        uint32_t N,
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
        // outputs
        int32_t* radii,       // [C, N, 2]
        float* means2d,       // [C, N, 2]
        float* depths,        // [C, N]
        float* conics,        // [C, N, 3]
        float* compensations, // [C, N] optional (can be nullptr)
        float* dilations,     // [C, N] optional (can be nullptr)
        cudaStream_t stream) {
        int64_t n_elements = C * N;
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
                N,
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
                radii,
                means2d,
                depths,
                conics,
                compensations,
                dilations);
        LFS_CUDA_LAUNCH_CHECK(stream, "gsplat.projection_ut_3dgs_fused");
    }

} // namespace gsplat_lfs
