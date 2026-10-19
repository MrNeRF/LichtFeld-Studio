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

} // namespace lfs::training::kernels
