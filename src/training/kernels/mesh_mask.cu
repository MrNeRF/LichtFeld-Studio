/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mesh_mask.hpp"

#include "core/cuda/undistort/distortion_model.cuh"

#include "core/assert.hpp"
#include "core/cuda_error.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "training/kernels/kernel_stream.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>
#include <nvtx3/nvToolsExt.h>

namespace lfs::training::kernels {
    namespace {

        constexpr int THREADS_PER_BLOCK = 256;
        constexpr int MAX_GRID_BLOCKS = 65535;
        constexpr int MAX_PERSISTENT_BLOCKS = 2048;
        constexpr int SMALL_BOX_PIXELS = 64;
        constexpr int MAX_POLYGON_VERTICES = 8;
        constexpr int DISTORTED_BOUNDS_MARGIN = 2;
        constexpr int MAX_EDGE_SAMPLES = 65536;

        struct CameraData {
            float4 rows[3];
            float fx;
            float fy;
            float cx;
            float cy;
            int width;
            int height;
            float guard_min_x;
            float guard_max_x;
            float guard_min_y;
            float guard_max_y;
            bool distorted;
            lfs::core::UndistortParams distortion;
        };

        struct PreparedPolygon {
            float2 vertices[MAX_POLYGON_VERTICES];
            int count;
            int min_x;
            int max_x;
            int min_y;
            int max_y;
        };

        __device__ __forceinline__ float3 transform_point(
            const CameraData& camera, const float3 point) {
            return make_float3(
                camera.rows[0].x * point.x + camera.rows[0].y * point.y +
                    camera.rows[0].z * point.z + camera.rows[0].w,
                camera.rows[1].x * point.x + camera.rows[1].y * point.y +
                    camera.rows[1].z * point.z + camera.rows[1].w,
                camera.rows[2].x * point.x + camera.rows[2].y * point.y +
                    camera.rows[2].z * point.z + camera.rows[2].w);
        }

        __device__ __forceinline__ float plane_distance(
            const float3 point, const int plane, const CameraData& camera,
            const float z_near) {
            switch (plane) {
            case 0:
                return point.z - z_near;
            case 1:
                return point.x - camera.guard_min_x * point.z;
            case 2:
                return camera.guard_max_x * point.z - point.x;
            case 3:
                return point.y - camera.guard_min_y * point.z;
            default:
                return camera.guard_max_y * point.z - point.y;
            }
        }

        __device__ int clip_against_plane(
            const float3* input, const int input_count, float3* output,
            const int plane, const CameraData& camera, const float z_near) {
            int output_count = 0;
            float3 previous = input[input_count - 1];
            float previous_distance = plane_distance(previous, plane, camera, z_near);
            bool previous_inside = previous_distance >= 0.0f;
            for (int i = 0; i < input_count; ++i) {
                const float3 current = input[i];
                const float current_distance = plane_distance(current, plane, camera, z_near);
                const bool current_inside = current_distance >= 0.0f;
                if (current_inside != previous_inside) {
                    const float t = previous_distance / (previous_distance - current_distance);
                    output[output_count++] = make_float3(
                        previous.x + t * (current.x - previous.x),
                        previous.y + t * (current.y - previous.y),
                        previous.z + t * (current.z - previous.z));
                }
                if (current_inside)
                    output[output_count++] = current;
                previous = current;
                previous_distance = current_distance;
                previous_inside = current_inside;
            }
            return output_count;
        }

        __device__ __forceinline__ void mark_covered(
            uint8_t* __restrict__ mask, const int pixel_index) {
            mask[pixel_index] = 1;
        }

        __device__ __forceinline__ void apply_distortion(
            const float x, const float y,
            const lfs::core::UndistortParams& params,
            float& dx, float& dy) {
            lfs::core::detail::apply_distortion(
                x, y, params.model_type, params.distortion, params.num_distortion, dx, dy);
        }

        __device__ bool compute_pixel_bounds(
            const PreparedPolygon& polygon, const CameraData& camera,
            int& min_x, int& max_x, int& min_y, int& max_y) {
            float bounds_min_x = INFINITY;
            float bounds_max_x = -INFINITY;
            float bounds_min_y = INFINITY;
            float bounds_max_y = -INFINITY;

            if (!camera.distorted) {
                for (int i = 0; i < polygon.count; ++i) {
                    const float px = polygon.vertices[i].x * camera.fx + camera.cx;
                    const float py = polygon.vertices[i].y * camera.fy + camera.cy;
                    bounds_min_x = fminf(bounds_min_x, px);
                    bounds_max_x = fmaxf(bounds_max_x, px);
                    bounds_min_y = fminf(bounds_min_y, py);
                    bounds_max_y = fmaxf(bounds_max_y, py);
                }
            } else {
                const auto& params = camera.distortion;
                const float sample_fx = fmaxf(fabsf(params.src_fx), fabsf(params.dst_fx));
                const float sample_fy = fmaxf(fabsf(params.src_fy), fabsf(params.dst_fy));
                for (int edge = 0; edge < polygon.count; ++edge) {
                    const float2 a = polygon.vertices[edge];
                    const float2 b = polygon.vertices[(edge + 1) % polygon.count];
                    const float edge_pixels = fmaxf(
                        fabsf(b.x - a.x) * sample_fx,
                        fabsf(b.y - a.y) * sample_fy);
                    const int steps = max(1, min(MAX_EDGE_SAMPLES, static_cast<int>(ceilf(edge_pixels))));
                    for (int sample = 0; sample <= steps; ++sample) {
                        const float t = static_cast<float>(sample) / static_cast<float>(steps);
                        const float x = a.x + t * (b.x - a.x);
                        const float y = a.y + t * (b.y - a.y);
                        float dx, dy;
                        apply_distortion(x, y, params, dx, dy);
                        const float px = dx * params.src_fx + params.src_cx;
                        const float py = dy * params.src_fy + params.src_cy;
                        if (!isfinite(px) || !isfinite(py))
                            continue;
                        bounds_min_x = fminf(bounds_min_x, px);
                        bounds_max_x = fmaxf(bounds_max_x, px);
                        bounds_min_y = fminf(bounds_min_y, py);
                        bounds_max_y = fmaxf(bounds_max_y, py);
                    }
                }
            }

            if (!isfinite(bounds_min_x) || !isfinite(bounds_max_x) ||
                !isfinite(bounds_min_y) || !isfinite(bounds_max_y))
                return false;
            const float margin = camera.distorted
                                     ? static_cast<float>(DISTORTED_BOUNDS_MARGIN)
                                     : 0.0f;
            const float first_x = bounds_min_x - 0.5f - margin;
            const float last_x = bounds_max_x - 0.5f + margin;
            const float first_y = bounds_min_y - 0.5f - margin;
            const float last_y = bounds_max_y - 0.5f + margin;
            if (last_x < 0.0f || last_y < 0.0f ||
                first_x > static_cast<float>(camera.width - 1) ||
                first_y > static_cast<float>(camera.height - 1))
                return false;
            min_x = max(0, static_cast<int>(floorf(fmaxf(first_x, -1.0f))));
            max_x = min(camera.width - 1,
                        static_cast<int>(ceilf(fminf(last_x, static_cast<float>(camera.width)))));
            min_y = max(0, static_cast<int>(floorf(fmaxf(first_y, -1.0f))));
            max_y = min(camera.height - 1,
                        static_cast<int>(ceilf(fminf(last_y, static_cast<float>(camera.height)))));
            return min_x <= max_x && min_y <= max_y;
        }

        __device__ bool prepare_triangle(
            const float* __restrict__ vertices,
            const int32_t* __restrict__ indices,
            const int vertex_count,
            const int face_index,
            const CameraData& camera,
            const float z_near,
            PreparedPolygon& prepared) {
            float3 clipped_a[MAX_POLYGON_VERTICES];
            float3 clipped_b[MAX_POLYGON_VERTICES];
            int count = 3;
            for (int corner = 0; corner < 3; ++corner) {
                const int32_t vertex_index = indices[3 * face_index + corner];
                if (vertex_index < 0 || vertex_index >= vertex_count)
                    return false;
                const float* const vertex = vertices + 3 * vertex_index;
                clipped_a[corner] = transform_point(
                    camera, make_float3(vertex[0], vertex[1], vertex[2]));
            }

            float3* input = clipped_a;
            float3* output = clipped_b;
            for (int plane = 0; plane < 5; ++plane) {
                count = clip_against_plane(input, count, output, plane, camera, z_near);
                if (count < 3)
                    return false;
                float3* const temporary = input;
                input = output;
                output = temporary;
            }

            prepared.count = count;
            float twice_area = 0.0f;
            for (int i = 0; i < count; ++i) {
                const float inverse_z = 1.0f / input[i].z;
                prepared.vertices[i] = make_float2(
                    input[i].x * inverse_z, input[i].y * inverse_z);
            }
            for (int i = 0; i < count; ++i) {
                const float2 a = prepared.vertices[i];
                const float2 b = prepared.vertices[(i + 1) % count];
                twice_area += a.x * b.y - a.y * b.x;
            }
            if (!isfinite(twice_area) || twice_area == 0.0f)
                return false;
            if (twice_area < 0.0f) {
                for (int i = 0; i < count / 2; ++i) {
                    const float2 temporary = prepared.vertices[i];
                    prepared.vertices[i] = prepared.vertices[count - 1 - i];
                    prepared.vertices[count - 1 - i] = temporary;
                }
            }

            return compute_pixel_bounds(
                prepared, camera,
                prepared.min_x, prepared.max_x, prepared.min_y, prepared.max_y);
        }

        __device__ __forceinline__ float edge_function(
            const float2 a, const float2 b, const float2 point) {
            const float dx = b.x - a.x;
            const float dy = b.y - a.y;
            return fmaf(dx, point.y - a.y, -dy * (point.x - a.x));
        }

        __device__ __forceinline__ bool inside_edge(
            const float2 a, const float2 b, const float2 point) {
            const float dx = b.x - a.x;
            const float dy = b.y - a.y;
            const float px = point.x - a.x;
            const float py = point.y - a.y;
            constexpr float ROUNDING_SCALE = 4.0f * 1.1920928955078125e-7f;
            const float tolerance = ROUNDING_SCALE *
                                    (fabsf(dx * py) + fabsf(dy * px));
            return fmaf(dx, py, -dy * px) >= -tolerance;
        }

        __device__ bool point_in_polygon_fan(
            const PreparedPolygon& polygon, const float2 point) {
            const float2 first = polygon.vertices[0];
            for (int triangle = 1; triangle + 1 < polygon.count; ++triangle) {
                const float2 second = polygon.vertices[triangle];
                const float2 third = polygon.vertices[triangle + 1];
                const float area = edge_function(first, second, third);
                if (area == 0.0f)
                    continue;
                if (inside_edge(first, second, point) &&
                    inside_edge(second, third, point) &&
                    inside_edge(third, first, point))
                    return true;
            }
            return false;
        }

        __device__ __forceinline__ float2 sample_point(
            const int x, const int y, const CameraData& camera,
            const float* __restrict__ inverse_sample_map) {
            if (camera.distorted) {
                const int index = 2 * (y * camera.width + x);
                return make_float2(inverse_sample_map[index], inverse_sample_map[index + 1]);
            }
            return make_float2(
                (static_cast<float>(x) + 0.5f - camera.cx) / camera.fx,
                (static_cast<float>(y) + 0.5f - camera.cy) / camera.fy);
        }

        __device__ void rasterize_serial(
            const PreparedPolygon& polygon,
            const CameraData& camera,
            const float* __restrict__ inverse_sample_map,
            uint8_t* __restrict__ mask) {
            for (int y = polygon.min_y; y <= polygon.max_y; ++y) {
                for (int x = polygon.min_x; x <= polygon.max_x; ++x) {
                    const float2 sample = sample_point(x, y, camera, inverse_sample_map);
                    if (isfinite(sample.x) && isfinite(sample.y) &&
                        point_in_polygon_fan(polygon, sample))
                        mark_covered(mask, y * camera.width + x);
                }
            }
        }

        __global__ void prepare_and_rasterize_small_kernel(
            const float* __restrict__ vertices,
            const int32_t* __restrict__ indices,
            const int vertex_count,
            const int face_count,
            const CameraData camera,
            const float z_near,
            const float* __restrict__ inverse_sample_map,
            int32_t* __restrict__ large_faces,
            int32_t* __restrict__ counters,
            uint8_t* __restrict__ mask) {
            for (int face_index = blockIdx.x * blockDim.x + threadIdx.x;
                 face_index < face_count;
                 face_index += blockDim.x * gridDim.x) {
                PreparedPolygon polygon;
                if (!prepare_triangle(
                        vertices, indices, vertex_count, face_index,
                        camera, z_near, polygon))
                    continue;
                const int64_t pixels =
                    static_cast<int64_t>(polygon.max_x - polygon.min_x + 1) *
                    static_cast<int64_t>(polygon.max_y - polygon.min_y + 1);
                if (pixels <= SMALL_BOX_PIXELS) {
                    rasterize_serial(
                        polygon, camera, inverse_sample_map, mask);
                } else {
                    const int slot = atomicAdd(counters, 1);
                    large_faces[slot] = face_index;
                }
            }
        }

        __global__ void rasterize_large_kernel(
            const float* __restrict__ vertices,
            const int32_t* __restrict__ indices,
            const int vertex_count,
            const CameraData camera,
            const float z_near,
            const float* __restrict__ inverse_sample_map,
            const int32_t* __restrict__ large_faces,
            int32_t* __restrict__ counters,
            uint8_t* __restrict__ mask) {
            __shared__ PreparedPolygon polygon;
            __shared__ int face_index;
            __shared__ bool prepared;

            while (true) {
                if (threadIdx.x == 0) {
                    const int queue_index = atomicAdd(counters + 1, 1);
                    face_index = queue_index < counters[0] ? large_faces[queue_index] : -1;
                    prepared = face_index >= 0 && prepare_triangle(
                                                      vertices, indices, vertex_count,
                                                      face_index, camera, z_near, polygon);
                }
                __syncthreads();
                if (face_index < 0)
                    return;
                if (prepared) {
                    const int box_width = polygon.max_x - polygon.min_x + 1;
                    for (int y = polygon.min_y; y <= polygon.max_y; ++y) {
                        for (int box_x = threadIdx.x; box_x < box_width;
                             box_x += blockDim.x) {
                            const int x = polygon.min_x + box_x;
                            const float2 sample = sample_point(
                                x, y, camera, inverse_sample_map);
                            if (isfinite(sample.x) && isfinite(sample.y) &&
                                point_in_polygon_fan(polygon, sample))
                                mark_covered(mask, y * camera.width + x);
                        }
                    }
                }
                __syncthreads();
            }
        }

        CameraData pack_camera(
            const MeshMaskCamera& camera,
            const lfs::core::UndistortParams* distortion,
            const float2 sample_extent) {
            CameraData packed{};
            for (int row = 0; row < 3; ++row) {
                packed.rows[row] = make_float4(
                    camera.world_to_camera[4 * row],
                    camera.world_to_camera[4 * row + 1],
                    camera.world_to_camera[4 * row + 2],
                    camera.world_to_camera[4 * row + 3]);
            }
            packed.fx = camera.fx;
            packed.fy = camera.fy;
            packed.cx = camera.cx;
            packed.cy = camera.cy;
            packed.width = camera.width;
            packed.height = camera.height;
            packed.distorted = distortion != nullptr;
            if (distortion)
                packed.distortion = *distortion;

            const auto expand_guard = [&](const float fx, const float fy,
                                          const float cx, const float cy,
                                          const int width, const int height) {
                packed.guard_min_x = std::min(
                    packed.guard_min_x, (-static_cast<float>(width) - cx) / fx);
                packed.guard_max_x = std::max(
                    packed.guard_max_x, (2.0f * width - cx) / fx);
                packed.guard_min_y = std::min(
                    packed.guard_min_y, (-static_cast<float>(height) - cy) / fy);
                packed.guard_max_y = std::max(
                    packed.guard_max_y, (2.0f * height - cy) / fy);
            };
            packed.guard_min_x = std::numeric_limits<float>::infinity();
            packed.guard_max_x = -std::numeric_limits<float>::infinity();
            packed.guard_min_y = std::numeric_limits<float>::infinity();
            packed.guard_max_y = -std::numeric_limits<float>::infinity();
            expand_guard(camera.fx, camera.fy, camera.cx, camera.cy,
                         camera.width, camera.height);
            if (distortion) {
                expand_guard(
                    distortion->dst_fx, distortion->dst_fy,
                    distortion->dst_cx, distortion->dst_cy,
                    distortion->dst_width, distortion->dst_height);
                expand_guard(
                    distortion->src_fx, distortion->src_fy,
                    distortion->src_cx, distortion->src_cy,
                    distortion->src_width, distortion->src_height);
                // Wide-angle rays can land far outside every frame; the guard must reach them.
                packed.guard_min_x = std::min(packed.guard_min_x, -2.0f * sample_extent.x);
                packed.guard_max_x = std::max(packed.guard_max_x, 2.0f * sample_extent.x);
                packed.guard_min_y = std::min(packed.guard_min_y, -2.0f * sample_extent.y);
                packed.guard_max_y = std::max(packed.guard_max_y, 2.0f * sample_extent.y);
            }
            return packed;
        }

        lfs::core::Tensor rasterize_mesh_coverage_impl(
            const lfs::core::Tensor& vertices,
            const lfs::core::Tensor& indices,
            const MeshMaskCamera& camera,
            const lfs::core::Tensor* inverse_sample_map,
            const lfs::core::UndistortParams* distortion,
            const float z_near,
            cudaStream_t stream) {
            using lfs::core::DataType;
            using lfs::core::Device;
            using lfs::core::Tensor;

            LFS_ASSERT_MSG(vertices.is_valid(), "Mesh mask vertices must be valid");
            LFS_ASSERT_MSG(vertices.ndim() == 2 && vertices.shape()[1] == 3,
                           "Mesh mask vertices must have shape [V,3]");
            LFS_ASSERT_MSG(vertices.device() == Device::GPU &&
                               vertices.dtype() == DataType::Float32,
                           "Mesh mask vertices must be GPU Float32");
            LFS_ASSERT_MSG(indices.is_valid(), "Mesh mask indices must be valid");
            LFS_ASSERT_MSG(indices.ndim() == 2 && indices.shape()[1] == 3,
                           "Mesh mask indices must have shape [F,3]");
            LFS_ASSERT_MSG(indices.device() == Device::GPU &&
                               indices.dtype() == DataType::Int32,
                           "Mesh mask indices must be GPU Int32");
            LFS_ASSERT_MSG(vertices.shape()[0] <= static_cast<size_t>(std::numeric_limits<int>::max()),
                           "Mesh mask vertex count exceeds Int32 range");
            LFS_ASSERT_MSG(indices.shape()[0] <= static_cast<size_t>(std::numeric_limits<int>::max()),
                           "Mesh mask face count exceeds Int32 range");
            LFS_ASSERT_MSG(camera.width > 0 && camera.height > 0,
                           "Mesh mask output dimensions must be positive");
            LFS_ASSERT_MSG(std::isfinite(camera.fx) && camera.fx > 0.0f &&
                               std::isfinite(camera.fy) && camera.fy > 0.0f,
                           "Mesh mask focal lengths must be positive and finite");
            LFS_ASSERT_MSG(std::isfinite(camera.cx) && std::isfinite(camera.cy),
                           "Mesh mask principal point must be finite");
            LFS_ASSERT_MSG(std::isfinite(z_near) && z_near > 0.0f,
                           "Mesh mask near plane must be positive and finite");
            LFS_ASSERT_MSG((inverse_sample_map == nullptr) == (distortion == nullptr),
                           "Mesh mask sample map and distortion parameters must be provided together");
            if (inverse_sample_map) {
                LFS_ASSERT_MSG(inverse_sample_map->is_valid(),
                               "Mesh mask inverse sample map must be valid");
                LFS_ASSERT_MSG(
                    inverse_sample_map->device() == Device::GPU &&
                        inverse_sample_map->dtype() == DataType::Float32 &&
                        inverse_sample_map->ndim() == 3 &&
                        inverse_sample_map->shape()[0] == static_cast<size_t>(camera.height) &&
                        inverse_sample_map->shape()[1] == static_cast<size_t>(camera.width) &&
                        inverse_sample_map->shape()[2] == 2,
                    "Mesh mask inverse sample map must be GPU Float32 [H,W,2]");
                LFS_ASSERT_MSG(
                    distortion->src_width == camera.width &&
                        distortion->src_height == camera.height,
                    "Mesh mask distortion source dimensions must match the output");
                LFS_ASSERT_MSG(
                    distortion->src_width > 0 && distortion->src_height > 0 &&
                        distortion->dst_width > 0 && distortion->dst_height > 0 &&
                        std::isfinite(distortion->src_fx) && distortion->src_fx > 0.0f &&
                        std::isfinite(distortion->src_fy) && distortion->src_fy > 0.0f &&
                        std::isfinite(distortion->dst_fx) && distortion->dst_fx > 0.0f &&
                        std::isfinite(distortion->dst_fy) && distortion->dst_fy > 0.0f,
                    "Mesh mask distortion dimensions and focal lengths must be positive");
            }

            stream = resolve_stream(stream);
            nvtxRangePush("rasterize_mesh_coverage");
            const lfs::core::CUDAStreamGuard stream_guard(stream);
            vertices.sync_to_stream(stream);
            indices.sync_to_stream(stream);
            if (inverse_sample_map)
                inverse_sample_map->sync_to_stream(stream);

            const auto input_vertices = vertices.contiguous();
            const auto input_indices = indices.contiguous();
            const Tensor samples = inverse_sample_map
                                       ? inverse_sample_map->contiguous()
                                       : Tensor{};
            auto mask = Tensor::zeros(
                {static_cast<size_t>(camera.height), static_cast<size_t>(camera.width)},
                Device::GPU, DataType::UInt8);
            const int face_count = static_cast<int>(input_indices.shape()[0]);
            if (face_count == 0) {
                nvtxRangePop();
                return mask;
            }

            auto large_faces = Tensor::empty(
                {static_cast<size_t>(face_count)}, Device::GPU, DataType::Int32);
            auto counters = Tensor::zeros({size_t{2}}, Device::GPU, DataType::Int32);
            const int64_t pixel_count =
                static_cast<int64_t>(camera.width) * camera.height;
            LFS_ASSERT_MSG(
                pixel_count <= std::numeric_limits<int>::max(),
                "Mesh mask output pixel count exceeds Int32 range");
            float2 sample_extent = make_float2(0.0f, 0.0f);
            if (inverse_sample_map) {
                const auto magnitude = samples.abs();
                const auto extent = magnitude.masked_fill(magnitude.ne(magnitude), 0.0f).max({0, 1}).cpu();
                sample_extent = make_float2(extent.ptr<float>()[0], extent.ptr<float>()[1]);
            }
            const CameraData packed_camera = pack_camera(camera, distortion, sample_extent);
            const int prepare_blocks = std::min(
                MAX_GRID_BLOCKS,
                (face_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
            prepare_and_rasterize_small_kernel<<<
                prepare_blocks, THREADS_PER_BLOCK, 0, stream>>>(
                input_vertices.ptr<float>(), input_indices.ptr<int32_t>(),
                static_cast<int>(input_vertices.shape()[0]), face_count,
                packed_camera, z_near,
                samples.is_valid() ? samples.ptr<float>() : nullptr,
                large_faces.ptr<int32_t>(), counters.ptr<int32_t>(),
                mask.ptr<uint8_t>());
            LFS_CUDA_LAUNCH_CHECK(stream, "training.mesh_mask.prepare");

            const int large_blocks = std::min(MAX_PERSISTENT_BLOCKS, face_count);
            rasterize_large_kernel<<<large_blocks, THREADS_PER_BLOCK, 0, stream>>>(
                input_vertices.ptr<float>(), input_indices.ptr<int32_t>(),
                static_cast<int>(input_vertices.shape()[0]), packed_camera, z_near,
                samples.is_valid() ? samples.ptr<float>() : nullptr,
                large_faces.ptr<int32_t>(), counters.ptr<int32_t>(),
                mask.ptr<uint8_t>());
            LFS_CUDA_LAUNCH_CHECK(stream, "training.mesh_mask.large");
            nvtxRangePop();
            return mask;
        }

    } // namespace

    lfs::core::Tensor rasterize_mesh_coverage(
        const lfs::core::Tensor& vertices,
        const lfs::core::Tensor& indices,
        const MeshMaskCamera& camera,
        const float z_near,
        const cudaStream_t stream) {
        return rasterize_mesh_coverage_impl(
            vertices, indices, camera, nullptr, nullptr, z_near, stream);
    }

    lfs::core::Tensor rasterize_mesh_coverage(
        const lfs::core::Tensor& vertices,
        const lfs::core::Tensor& indices,
        const MeshMaskCamera& camera,
        const lfs::core::Tensor& inverse_sample_map,
        const lfs::core::UndistortParams& distortion,
        const float z_near,
        const cudaStream_t stream) {
        return rasterize_mesh_coverage_impl(
            vertices, indices, camera, &inverse_sample_map, &distortion,
            z_near, stream);
    }

    namespace {
        constexpr int POINT_THREADS = 256;

        __global__ void splat_points_kernel(const float* __restrict__ means, const int count,
                                            const MeshMaskCamera camera, const int radius,
                                            const bool distorted, const lfs::core::UndistortParams distortion,
                                            uint8_t* __restrict__ mask) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= count)
                return;
            const float* p = means + 3 * i;
            const auto& m = camera.world_to_camera;
            const float x = m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3];
            const float y = m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7];
            const float z = m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11];
            if (!(z > 1.0e-6f))
                return;
            float nx = x / z;
            float ny = y / z;
            if (distorted) {
                float dx, dy;
                apply_distortion(nx, ny, distortion, dx, dy);
                nx = dx;
                ny = dy;
            }
            const float u = camera.fx * nx + camera.cx;
            const float v = camera.fy * ny + camera.cy;
            if (!isfinite(u) || !isfinite(v) || u < -radius || v < -radius ||
                u >= camera.width + radius || v >= camera.height + radius)
                return;
            const int cu = static_cast<int>(floorf(u));
            const int cv = static_cast<int>(floorf(v));
            for (int dy = -radius; dy <= radius; ++dy) {
                const int py = cv + dy;
                if (py < 0 || py >= camera.height)
                    continue;
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int px = cu + dx;
                    if (px >= 0 && px < camera.width && dx * dx + dy * dy <= radius * radius)
                        mask[static_cast<size_t>(py) * camera.width + px] = 1;
                }
            }
        }

        // One separable pass of a square max (dilate) or min (erode) filter; outside pixels count as
        // 0 for dilation and 1 for erosion so closing keeps regions that touch the image border.
        __global__ void morphology_pass_kernel(const uint8_t* __restrict__ input, uint8_t* __restrict__ output,
                                               const int width, const int height, const int reach,
                                               const bool horizontal, const bool dilate) {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= width * height)
                return;
            const int x = index % width;
            const int y = index / width;
            uint8_t value = dilate ? 0 : 1;
            for (int offset = -reach; offset <= reach; ++offset) {
                const int sx = horizontal ? x + offset : x;
                const int sy = horizontal ? y : y + offset;
                const uint8_t sample = (sx < 0 || sx >= width || sy < 0 || sy >= height)
                                           ? (dilate ? 0 : 1)
                                           : input[static_cast<size_t>(sy) * width + sx];
                value = dilate ? max(value, sample) : min(value, sample);
            }
            output[index] = value;
        }
    } // namespace

    lfs::core::Tensor splat_point_coverage(const lfs::core::Tensor& means, const MeshMaskCamera& camera, const int radius,
                                           const int close, const lfs::core::UndistortParams* distortion,
                                           cudaStream_t stream) {
        LFS_ASSERT_MSG(means.device() == lfs::core::Device::GPU && means.dtype() == lfs::core::DataType::Float32 &&
                           means.ndim() == 2 && means.shape()[1] == 3,
                       "Point mask means must be a CUDA Float32 [N,3] tensor");
        LFS_ASSERT_MSG(camera.width > 0 && camera.height > 0 && radius >= 0 && close >= 0,
                       "Point mask camera size and radii must be valid");
        stream = resolve_stream(stream);
        nvtxRangePush("splat_point_coverage");
        const lfs::core::CUDAStreamGuard stream_guard(stream);
        means.sync_to_stream(stream);
        const auto points = means.contiguous();
        const size_t pixels = static_cast<size_t>(camera.width) * camera.height;
        auto mask = lfs::core::Tensor::zeros({static_cast<size_t>(camera.height), static_cast<size_t>(camera.width)},
                                             lfs::core::Device::GPU, lfs::core::DataType::UInt8);
        const int count = static_cast<int>(points.shape()[0]);
        if (count > 0) {
            splat_points_kernel<<<(count + POINT_THREADS - 1) / POINT_THREADS, POINT_THREADS, 0, stream>>>(
                points.ptr<float>(), count, camera, radius, distortion != nullptr,
                distortion ? *distortion : lfs::core::UndistortParams{}, mask.ptr<uint8_t>());
            LFS_CUDA_LAUNCH_CHECK(stream, "training.point_mask.splat");
        }
        if (close > 0) {
            auto scratch = lfs::core::Tensor::empty(mask.shape(), lfs::core::Device::GPU, lfs::core::DataType::UInt8);
            const int blocks = static_cast<int>((pixels + POINT_THREADS - 1) / POINT_THREADS);
            for (const bool dilate : {true, false}) {
                morphology_pass_kernel<<<blocks, POINT_THREADS, 0, stream>>>(
                    mask.ptr<uint8_t>(), scratch.ptr<uint8_t>(), camera.width, camera.height, close, true, dilate);
                LFS_CUDA_LAUNCH_CHECK(stream, "training.point_mask.close_rows");
                morphology_pass_kernel<<<blocks, POINT_THREADS, 0, stream>>>(
                    scratch.ptr<uint8_t>(), mask.ptr<uint8_t>(), camera.width, camera.height, close, false, dilate);
                LFS_CUDA_LAUNCH_CHECK(stream, "training.point_mask.close_columns");
            }
        }
        nvtxRangePop();
        return mask;
    }

} // namespace lfs::training::kernels
