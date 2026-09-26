/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../facade_trace.hpp"
#include "core/tensor_spatial.hpp"
#include "vk_backend_ops.hpp"
#include "vk_context.hpp"
#include "vk_ops_common.hpp"
#include "vk_pipelines.hpp"
#include "vk_recorder.hpp"

#include <array>
#include <cstddef>

namespace lfs::core::internal {
    namespace {
        struct RadiusPush {
            uint64_t points;
            uint64_t references;
            uint64_t heads;
            uint64_t next;
            uint64_t output;
            uint32_t count;
            uint32_t bucket_mask;
            float radius;
            uint32_t padding;
        };
        static_assert(sizeof(RadiusPush) == 56);

        struct ProjectionPush {
            std::array<float, 4> row0;
            std::array<float, 4> row1;
            std::array<float, 4> row2;
            uint64_t points;
            uint64_t output;
            uint64_t transforms;
            uint64_t indices;
            uint64_t visibility;
            std::array<float, 2> scale;
            std::array<float, 2> center;
            float invalid_value;
            float near_distance;
            uint32_t count;
            uint32_t transform_count;
            uint32_t visibility_count;
            uint32_t padding;
        };
        static_assert(sizeof(ProjectionPush) == 128);
        static_assert(offsetof(ProjectionPush, points) == 48);
        static_assert(offsetof(ProjectionPush, scale) == 88);
    } // namespace

    void VulkanBackendOps::radius_neighbors(const StorageRef points, const StorageRef references,
                                            const StorageRef heads, const StorageRef next, const StorageRef output,
                                            const size_t count, const size_t buckets, const float radius, ExecContext) {
        LFS_FACADE_TRACE(radius_neighbors);
        const auto context = acquire_vulkan_context();
        const RadiusPush push{
            .points = vk::address(points),
            .references = vk::address(references),
            .heads = vk::address(heads),
            .next = vk::address(next),
            .output = vk::address(output),
            .count = static_cast<uint32_t>(count),
            .bucket_mask = static_cast<uint32_t>(buckets - 1),
            .radius = radius,
        };
        const auto dispatch = [&](const uint32_t mode, const std::span<const StorageRef> reads,
                                  const std::span<const StorageRef> writes, const size_t work) {
            const std::array constants{mode};
            const auto& pipeline = context->pipelines().specialized("radius_neighbors", sizeof(push), constants);
            context->recorders().record(reads, writes, [&](const VkCommandBuffer command) {
                vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
                vkCmdPushConstants(command, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
                vkCmdDispatch(command, vk::dispatch_groups(*context, work), 1, 1);
            });
        };
        const std::array build_reads{points, references, heads};
        const std::array build_writes{heads, next};
        dispatch(0, build_reads, build_writes, count);
        const std::array query_reads{points, references, heads, next};
        const std::array query_writes{output};
        // Four byte results per invocation: one owner writes each output word.
        dispatch(1, query_reads, query_writes, (count + 3) / 4);
    }

    void VulkanBackendOps::rasterize_points(const PointRasterProgram& program, ExecContext) {
        LFS_FACADE_TRACE(rasterize_points);
        const auto context = acquire_vulkan_context();
        struct Push {
            uint64_t positions, colors, parameters, scratch, image, depth, transforms, indices, visibility, deleted;
            uint32_t count, width, height, channels, transform_count, visibility_count, flags;
            float ortho_scale, focal_y, voxel_size, far_plane;
            uint32_t padding;
        };
        static_assert(sizeof(Push) == 128);
        const auto address = [](const std::optional<StorageRef>& storage) {
            return storage ? vk::address(*storage) : uint64_t{0};
        };
        const Push push{
            .positions = vk::address(program.positions),
            .colors = vk::address(program.colors),
            .parameters = vk::address(program.parameters),
            .scratch = vk::address(program.scratch),
            .image = vk::address(program.image),
            .depth = vk::address(program.depth),
            .transforms = address(program.transforms),
            .indices = address(program.indices),
            .visibility = address(program.visibility),
            .deleted = address(program.deleted),
            .count = static_cast<uint32_t>(program.count),
            .width = program.width,
            .height = program.height,
            .channels = program.channels,
            .transform_count = program.transform_count,
            .visibility_count = program.visibility_count,
            .flags = program.flags,
            .ortho_scale = program.ortho_scale,
            .focal_y = program.focal_y,
            .voxel_size = program.voxel_size,
            .far_plane = program.far_plane,
            .padding = 0,
        };
        std::array<StorageRef, 7> reads{program.positions, program.colors, program.parameters};
        size_t read_count = 3;
        for (const auto& operand : {program.transforms, program.indices, program.visibility, program.deleted}) {
            if (operand)
                reads[read_count++] = *operand;
        }
        const std::array writes{program.scratch, program.image, program.depth};
        const size_t pixels = static_cast<size_t>(program.width) * program.height;
        // Clear, nearest depth, winning color, then the image.
        for (const uint32_t phase : {0u, 1u, 2u, 3u}) {
            const std::array constants{phase};
            const auto& pipeline = context->pipelines().specialized("point_raster", sizeof(push), constants);
            const size_t threads = phase == 1 || phase == 2 ? program.count : pixels;
            context->recorders().record(std::span(reads.data(), read_count), writes, [&](VkCommandBuffer command) {
                vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
                vkCmdPushConstants(command, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
                vkCmdDispatch(command, vk::dispatch_groups(*context, threads), 1, 1);
            });
        }
    }

    void VulkanBackendOps::project_points(const StorageRef points, const StorageRef output, const size_t count,
                                          const PointProjection& projection,
                                          const StorageRef* transforms, const size_t transform_count,
                                          const StorageRef* indices, const StorageRef* visibility,
                                          const size_t visibility_count, ExecContext) {
        LFS_FACADE_TRACE(project_points);
        const auto context = acquire_vulkan_context();
        const auto& r = projection.rotation;
        const auto& t = projection.translation;
        std::array<float, 2> scale{projection.focal_x, projection.focal_y};
        if (projection.model == PointProjectionModel::Orthographic) {
            scale = {projection.ortho_scale, projection.ortho_scale};
        } else if (projection.model == PointProjectionModel::Equirectangular) {
            scale = {static_cast<float>(projection.width), static_cast<float>(projection.height)};
        }
        const ProjectionPush push{
            .row0 = {r[0], r[1], r[2], t[0]},
            .row1 = {r[3], r[4], r[5], t[1]},
            .row2 = {r[6], r[7], r[8], t[2]},
            .points = vk::address(points),
            .output = vk::address(output),
            .transforms = transforms ? vk::address(*transforms) : 0,
            .indices = indices ? vk::address(*indices) : 0,
            .visibility = visibility ? vk::address(*visibility) : 0,
            .scale = scale,
            .center = {projection.center_x, projection.center_y},
            .invalid_value = projection.invalid_value,
            .near_distance = projection.near_distance,
            .count = static_cast<uint32_t>(count),
            .transform_count = static_cast<uint32_t>(transform_count),
            .visibility_count = static_cast<uint32_t>(visibility_count),
            .padding = 0,
        };
        std::array<StorageRef, 4> reads{points};
        size_t read_count = 1;
        for (const auto* storage : {transforms, indices, visibility}) {
            if (storage) {
                reads[read_count++] = *storage;
            }
        }
        const std::array writes{output};
        const std::array constants{static_cast<uint32_t>(projection.model), uint32_t(transforms != nullptr),
                                   uint32_t(indices != nullptr), uint32_t(visibility != nullptr)};
        const auto& pipeline = context->pipelines().specialized("project_points", sizeof(push), constants);
        context->recorders().record(std::span(reads.data(), read_count), writes, [&](VkCommandBuffer command) {
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
            vkCmdPushConstants(command, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            vkCmdDispatch(command, vk::dispatch_groups(*context, count), 1, 1);
        });
    }
    void VulkanBackendOps::mark_points_2d(const StorageRef mask, const StorageRef points, const size_t count,
                                          const PointRegion2D& region, const StorageRef* geometry,
                                          const size_t geometry_count, ExecContext) {
        LFS_FACADE_TRACE(mark_points_2d);
        const auto context = acquire_vulkan_context();
        struct Push {
            uint64_t mask, points, geometry;
            uint32_t count, geometry_count;
            std::array<float, 4> bounds;
            float radius_sq, minimum_coordinate;
        };
        static_assert(sizeof(Push) == 56);
        const Push push{
            .mask = vk::address(mask),
            .points = vk::address(points),
            .geometry = geometry ? vk::address(*geometry) : 0,
            .count = static_cast<uint32_t>(count),
            .geometry_count = static_cast<uint32_t>(geometry_count),
            .bounds = {region.x0, region.y0, region.x1, region.y1},
            .radius_sq = region.radius * region.radius,
            .minimum_coordinate = region.minimum_coordinate,
        };
        const std::array constants{static_cast<uint32_t>(region.kind)};
        const auto& pipeline = context->pipelines().specialized("point_region", sizeof(push), constants);
        std::array reads{mask, points, geometry ? *geometry : points};
        const std::array writes{mask};
        context->recorders().record(reads, writes, [&](const VkCommandBuffer command) {
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
            vkCmdPushConstants(command, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            const size_t words = (count + (push.mask & 3) + 3) / 4;
            vkCmdDispatch(command, vk::dispatch_groups(*context, words), 1, 1);
        });
    }
} // namespace lfs::core::internal
