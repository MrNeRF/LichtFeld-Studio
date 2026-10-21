/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "roi_weight_map.hpp"

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "kernel_stream.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>

namespace lfs::training::kernels {
    namespace {

        constexpr int kThreadsPerBlock = 256;
        // Below this accumulated alpha a pixel shows mostly background colour, not a surface.
        constexpr float kMinSurfaceAlpha = 0.5f;
        constexpr size_t kMaxBlocks = 65535;
        constexpr float kParallelEpsilon = 1.0e-8f;

        struct Affine3x4 {
            float4 rows[3];
        };

        [[nodiscard]] Affine3x4 pack_affine(const glm::mat4& matrix) {
            return Affine3x4{{
                {matrix[0][0], matrix[1][0], matrix[2][0], matrix[3][0]},
                {matrix[0][1], matrix[1][1], matrix[2][1], matrix[3][1]},
                {matrix[0][2], matrix[1][2], matrix[2][2], matrix[3][2]},
            }};
        }

        __device__ __forceinline__ float3 transform_point(
            const Affine3x4& transform,
            const float3 point) {
            return make_float3(
                transform.rows[0].x * point.x + transform.rows[0].y * point.y +
                    transform.rows[0].z * point.z + transform.rows[0].w,
                transform.rows[1].x * point.x + transform.rows[1].y * point.y +
                    transform.rows[1].z * point.z + transform.rows[1].w,
                transform.rows[2].x * point.x + transform.rows[2].y * point.y +
                    transform.rows[2].z * point.z + transform.rows[2].w);
        }

        __device__ __forceinline__ float3 transform_direction(
            const Affine3x4& transform,
            const float3 direction) {
            return make_float3(
                transform.rows[0].x * direction.x + transform.rows[0].y * direction.y +
                    transform.rows[0].z * direction.z,
                transform.rows[1].x * direction.x + transform.rows[1].y * direction.y +
                    transform.rows[1].z * direction.z,
                transform.rows[2].x * direction.x + transform.rows[2].y * direction.y +
                    transform.rows[2].z * direction.z);
        }

        __device__ __forceinline__ bool ray_intersects_aabb(
            const float3 origin,
            const float3 direction,
            const float3 bounds_min,
            const float3 bounds_max) {
            float t_enter = -FLT_MAX;
            float t_exit = FLT_MAX;

            const float origins[3] = {origin.x, origin.y, origin.z};
            const float directions[3] = {direction.x, direction.y, direction.z};
            const float mins[3] = {bounds_min.x, bounds_min.y, bounds_min.z};
            const float maxs[3] = {bounds_max.x, bounds_max.y, bounds_max.z};

#pragma unroll
            for (int axis = 0; axis < 3; ++axis) {
                if (fabsf(directions[axis]) < kParallelEpsilon) {
                    if (origins[axis] < mins[axis] || origins[axis] > maxs[axis]) {
                        return false;
                    }
                    continue;
                }

                const float inv_direction = 1.0f / directions[axis];
                float t0 = (mins[axis] - origins[axis]) * inv_direction;
                float t1 = (maxs[axis] - origins[axis]) * inv_direction;
                if (t0 > t1) {
                    const float tmp = t0;
                    t0 = t1;
                    t1 = tmp;
                }
                t_enter = fmaxf(t_enter, t0);
                t_exit = fminf(t_exit, t1);
                if (t_exit < t_enter) {
                    return false;
                }
            }

            return t_exit >= 0.0f;
        }

        __global__ void roi_weight_map_kernel(
            const float* __restrict__ world_to_camera,
            const float* __restrict__ camera_position,
            const float fx,
            const float fy,
            const float cx,
            const float cy,
            const int width,
            const int height,
            const Affine3x4 world_to_cropbox,
            const float3 crop_min,
            const float3 crop_max,
            const bool inverse,
            const float outside_weight,
            const float* __restrict__ camera_rays,
            const float* __restrict__ surface_depth,
            const float* __restrict__ surface_alpha,
            float* __restrict__ output) {
            const size_t num_pixels = static_cast<size_t>(width) * height;
            for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
                 idx < num_pixels;
                 idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
                const int x = static_cast<int>(idx % width);
                const int y = static_cast<int>(idx / width);
                float3 camera_direction;
                if (camera_rays) {
                    camera_direction = make_float3(
                        camera_rays[3 * idx], camera_rays[3 * idx + 1], camera_rays[3 * idx + 2]);
                    // The lens has no ray for this pixel.
                    if (!isfinite(camera_direction.x) || !isfinite(camera_direction.y) ||
                        !isfinite(camera_direction.z)) {
                        output[idx] = inverse ? 1.0f : outside_weight;
                        continue;
                    }
                } else {
                    camera_direction = make_float3(
                        (static_cast<float>(x) + 0.5f - cx) / fx,
                        (static_cast<float>(y) + 0.5f - cy) / fy,
                        1.0f);
                }

                const float3 world_direction = make_float3(
                    world_to_camera[0] * camera_direction.x +
                        world_to_camera[4] * camera_direction.y +
                        world_to_camera[8] * camera_direction.z,
                    world_to_camera[1] * camera_direction.x +
                        world_to_camera[5] * camera_direction.y +
                        world_to_camera[9] * camera_direction.z,
                    world_to_camera[2] * camera_direction.x +
                        world_to_camera[6] * camera_direction.y +
                        world_to_camera[10] * camera_direction.z);
                const float3 world_origin = make_float3(
                    camera_position[0], camera_position[1], camera_position[2]);
                const float3 origin_in_box = transform_point(world_to_cropbox, world_origin);
                const float3 direction_in_box = transform_direction(world_to_cropbox, world_direction);
                bool in_roi;
                if (surface_depth && surface_alpha) {
                    // Mostly background colour: no surface, so nothing in the box is seen here.
                    const float alpha = surface_alpha[idx];
                    if (!(alpha >= kMinSurfaceAlpha)) {
                        in_roi = false;
                    } else {
                        const float t = surface_depth[idx] / alpha;
                        const float3 p = make_float3(origin_in_box.x + t * direction_in_box.x,
                                                     origin_in_box.y + t * direction_in_box.y,
                                                     origin_in_box.z + t * direction_in_box.z);
                        in_roi = p.x >= crop_min.x && p.x <= crop_max.x && p.y >= crop_min.y &&
                                 p.y <= crop_max.y && p.z >= crop_min.z && p.z <= crop_max.z;
                    }
                } else {
                    in_roi = ray_intersects_aabb(origin_in_box, direction_in_box, crop_min, crop_max);
                }
                const bool full_weight = inverse ? !in_roi : in_roi;
                output[idx] = full_weight ? 1.0f : outside_weight;
            }
        }

        // The ray from the camera through a splat centre meets the box when the splat lies inside
        // it, or in front of or behind it in that view: every such splat shades full-weight pixels.
        // margin grows the box by the splat's footprint (shrinks it when negative), so splats
        // that overlap the box edge in a view count as well and in-box splats need not spill
        // over the ground just beside it.
        __device__ __forceinline__ bool splat_sees_box(
            const float3 camera_in_box,
            const float3 centre_in_box,
            const float3 crop_min,
            const float3 crop_max,
            const float margin) {
            const float3 lo = make_float3(crop_min.x - margin, crop_min.y - margin, crop_min.z - margin);
            const float3 hi = make_float3(crop_max.x + margin, crop_max.y + margin, crop_max.z + margin);
            if (lo.x > hi.x || lo.y > hi.y || lo.z > hi.z)
                return false;
            const float3 direction = make_float3(
                centre_in_box.x - camera_in_box.x,
                centre_in_box.y - camera_in_box.y,
                centre_in_box.z - camera_in_box.z);
            return ray_intersects_aabb(camera_in_box, direction, lo, hi);
        }

        // Three standard deviations of the splat's largest axis, in box units.
        __device__ __forceinline__ float splat_margin(
            const float* __restrict__ scaling_raw, const size_t i, const float margin_scale) {
            if (!scaling_raw)
                return 0.0f;
            const float largest = fmaxf(scaling_raw[3 * i], fmaxf(scaling_raw[3 * i + 1], scaling_raw[3 * i + 2]));
            return 3.0f * expf(largest) * margin_scale;
        }

        __global__ void roi_view_reject_kernel(
            const float* __restrict__ means,
            const float* __restrict__ scaling_raw,
            const size_t n,
            const Affine3x4 model_to_cropbox,
            const float margin_scale,
            const Affine3x4 world_to_cropbox,
            const float* __restrict__ camera_position,
            const float3 crop_min,
            const float3 crop_max,
            const bool inverse,
            const bool* __restrict__ deleted,
            bool* __restrict__ rejected) {
            const float3 camera_in_box = transform_point(
                world_to_cropbox,
                make_float3(camera_position[0], camera_position[1], camera_position[2]));
            for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
                 i += static_cast<size_t>(blockDim.x) * gridDim.x) {
                if (deleted && deleted[i]) {
                    rejected[i] = false;
                    continue;
                }
                const float3 centre = transform_point(
                    model_to_cropbox, make_float3(means[3 * i], means[3 * i + 1], means[3 * i + 2]));
                // Inverse boxes shrink instead, so the splats on their rim train with the outside.
                const float margin = splat_margin(scaling_raw, i, margin_scale);
                const bool sees = splat_sees_box(
                    camera_in_box, centre, crop_min, crop_max, inverse ? -margin : margin);
                rejected[i] = inverse ? sees : !sees;
            }
        }

        __global__ void roi_relevance_reject_kernel(
            const float* __restrict__ means,
            const float* __restrict__ scaling_raw,
            const size_t n,
            const Affine3x4 model_to_cropbox,
            const float margin_scale,
            const float* __restrict__ cameras_in_box,
            const int num_cameras,
            const float3 crop_min,
            const float3 crop_max,
            const bool inverse,
            const bool* __restrict__ deleted,
            bool* __restrict__ rejected,
            bool* __restrict__ in_roi) {
            for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
                 i += static_cast<size_t>(blockDim.x) * gridDim.x) {
                if (deleted && deleted[i]) {
                    rejected[i] = false;
                    if (in_roi)
                        in_roi[i] = false;
                    continue;
                }
                const float3 centre = transform_point(
                    model_to_cropbox, make_float3(means[3 * i], means[3 * i + 1], means[3 * i + 2]));
                if (in_roi) {
                    const bool inside = centre.x >= crop_min.x && centre.x <= crop_max.x &&
                                        centre.y >= crop_min.y && centre.y <= crop_max.y &&
                                        centre.z >= crop_min.z && centre.z <= crop_max.z;
                    in_roi[i] = inverse ? !inside : inside;
                }
                // Inverse boxes keep splats that some camera sees outside the box.
                const float margin = splat_margin(scaling_raw, i, margin_scale);
                bool relevant = false;
                for (int c = 0; c < num_cameras && !relevant; ++c) {
                    const float3 camera = make_float3(
                        cameras_in_box[3 * c], cameras_in_box[3 * c + 1], cameras_in_box[3 * c + 2]);
                    const bool sees = splat_sees_box(
                        camera, centre, crop_min, crop_max, inverse ? -margin : margin);
                    relevant = inverse ? !sees : sees;
                }
                rejected[i] = !relevant;
            }
        }

    } // namespace

    void launch_roi_weight_map(
        const float* world_to_camera,
        const float* camera_position,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        const int width,
        const int height,
        const glm::mat4& world_to_cropbox,
        const glm::vec3& crop_min,
        const glm::vec3& crop_max,
        const bool inverse,
        const float outside_weight,
        const float* camera_rays,
        const float* surface_depth,
        const float* surface_alpha,
        float* output,
        cudaStream_t stream) {
        LFS_ASSERT_MSG(world_to_camera != nullptr, "ROI world-to-camera pointer must not be null");
        LFS_ASSERT_MSG(camera_position != nullptr, "ROI camera-position pointer must not be null");
        LFS_ASSERT_MSG(output != nullptr, "ROI output pointer must not be null");
        LFS_ASSERT_MSG(width > 0 && height > 0, "ROI dimensions must be positive");
        LFS_ASSERT_MSG(std::isfinite(fx) && fx > 0.0f, "ROI focal length fx must be positive and finite");
        LFS_ASSERT_MSG(std::isfinite(fy) && fy > 0.0f, "ROI focal length fy must be positive and finite");
        LFS_ASSERT_MSG(
            std::isfinite(outside_weight) && outside_weight >= 0.0f && outside_weight <= 1.0f,
            "ROI outside weight must be finite and within [0, 1]");

        stream = resolve_stream(stream);
        const size_t num_pixels = static_cast<size_t>(width) * height;
        const int num_blocks = static_cast<int>(
            std::min((num_pixels + kThreadsPerBlock - 1) / kThreadsPerBlock, kMaxBlocks));
        roi_weight_map_kernel<<<num_blocks, kThreadsPerBlock, 0, stream>>>(
            world_to_camera,
            camera_position,
            fx,
            fy,
            cx,
            cy,
            width,
            height,
            pack_affine(world_to_cropbox),
            make_float3(crop_min.x, crop_min.y, crop_min.z),
            make_float3(crop_max.x, crop_max.y, crop_max.z),
            inverse,
            outside_weight,
            camera_rays,
            surface_depth,
            surface_alpha,
            output);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.roi_weight_map");
    }

    float box_units_per_model_unit(const glm::mat4& model_to_cropbox) {
        // Largest stretch of the linear part: power iteration on A^T A.
        const glm::mat3 a(model_to_cropbox);
        const glm::mat3 ata = glm::transpose(a) * a;
        glm::vec3 v(1.0f, 0.7f, 0.4f);
        for (int i = 0; i < 32; ++i) {
            const glm::vec3 next = ata * v;
            const float length = glm::length(next);
            if (!(length > 0.0f))
                return 0.0f;
            v = next / length;
        }
        return std::sqrt(glm::length(ata * v));
    }

    void launch_roi_view_reject_mask(
        const float* means,
        const float* scaling_raw,
        const size_t n,
        const glm::mat4& model_to_cropbox,
        const glm::mat4& world_to_cropbox,
        const float* camera_position,
        const glm::vec3& crop_min,
        const glm::vec3& crop_max,
        const bool inverse,
        const bool* deleted,
        bool* rejected,
        cudaStream_t stream) {
        if (n == 0)
            return;
        LFS_ASSERT_MSG(means != nullptr && camera_position != nullptr && rejected != nullptr,
                       "ROI view mask pointers must not be null");
        stream = resolve_stream(stream);
        const int num_blocks = static_cast<int>(
            std::min((n + kThreadsPerBlock - 1) / kThreadsPerBlock, kMaxBlocks));
        roi_view_reject_kernel<<<num_blocks, kThreadsPerBlock, 0, stream>>>(
            means, scaling_raw, n, pack_affine(model_to_cropbox), box_units_per_model_unit(model_to_cropbox),
            pack_affine(world_to_cropbox), camera_position,
            make_float3(crop_min.x, crop_min.y, crop_min.z),
            make_float3(crop_max.x, crop_max.y, crop_max.z),
            inverse, deleted, rejected);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.roi_view_reject_mask");
    }

    void launch_roi_relevance_reject_mask(
        const float* means,
        const float* scaling_raw,
        const size_t n,
        const glm::mat4& model_to_cropbox,
        const float* cameras_in_box,
        const int num_cameras,
        const glm::vec3& crop_min,
        const glm::vec3& crop_max,
        const bool inverse,
        const bool* deleted,
        bool* rejected,
        bool* in_roi,
        cudaStream_t stream) {
        if (n == 0)
            return;
        LFS_ASSERT_MSG(means != nullptr && rejected != nullptr && (num_cameras == 0 || cameras_in_box != nullptr),
                       "ROI relevance mask pointers must not be null");
        stream = resolve_stream(stream);
        const int num_blocks = static_cast<int>(
            std::min((n + kThreadsPerBlock - 1) / kThreadsPerBlock, kMaxBlocks));
        roi_relevance_reject_kernel<<<num_blocks, kThreadsPerBlock, 0, stream>>>(
            means, scaling_raw, n, pack_affine(model_to_cropbox), box_units_per_model_unit(model_to_cropbox),
            cameras_in_box, num_cameras,
            make_float3(crop_min.x, crop_min.y, crop_min.z),
            make_float3(crop_max.x, crop_max.y, crop_max.z),
            inverse, deleted, rejected, in_roi);
        LFS_CUDA_LAUNCH_CHECK(stream, "training.roi_relevance_reject_mask");
    }

} // namespace lfs::training::kernels
