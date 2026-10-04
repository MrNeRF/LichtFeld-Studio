/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../facade_trace.hpp"
#include "vk_backend_ops.hpp"
#include "vk_ops_common.hpp"
#include "vk_pipelines.hpp"
#include "vk_recorder.hpp"
namespace lfs::core::internal {
    void VulkanBackendOps::affine_splat_geometry(StorageRef scales, StorageRef rotations, StorageRef out_scales, StorageRef out_rotations,
                                                 const splat_transform::LinearTransform& linear, size_t n, std::optional<StorageRef> matrices, ExecContext) {
        LFS_FACADE_TRACE(affine_splat_geometry);
        const auto context = acquire_vulkan_context();
        struct Push {
            uint64_t scales, rotations, out_scales, out_rotations;
            float linear[9];
            uint32_t count;
            uint64_t matrices;
        };
        static_assert(sizeof(Push) == 80);
        Push push{vk::address(scales), vk::address(rotations), vk::address(out_scales), vk::address(out_rotations), {}, vk::checked_u32(n, "affine splat count exceeds uint32"), matrices ? vk::address(*matrices) : 0};
        std::copy(std::begin(linear.rows), std::end(linear.rows), push.linear);
        const auto& pipeline = context->pipelines().specialized("affine_splat_geometry", sizeof(push), {});
        std::vector<StorageRef> reads{scales, rotations};
        if (matrices)
            reads.push_back(*matrices);
        const std::array writes{out_scales, out_rotations};
        context->recorders().record(reads, writes, [&](VkCommandBuffer command) {
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
            vkCmdPushConstants(command, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            vkCmdDispatch(command, vk::dispatch_groups(*context, n), 1, 1);
        });
    }
} // namespace lfs::core::internal
