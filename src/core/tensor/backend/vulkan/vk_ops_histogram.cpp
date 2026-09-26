/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../facade_trace.hpp"
#include "vk_backend_ops.hpp"
#include "vk_ops_common.hpp"
#include "vk_pipelines.hpp"
#include "vk_recorder.hpp"

namespace lfs::core::internal {
    void VulkanBackendOps::histogram_u8(const StorageRef values, const StorageRef counts,
                                        const size_t size, ExecContext) {
        LFS_FACADE_TRACE(histogram_u8);
        struct Push {
            uint64_t input;
            uint64_t output;
            uint32_t size;
            uint32_t padding;
        };
        static_assert(sizeof(Push) == 24);
        const Push push{vk::address(values), vk::address(counts), static_cast<uint32_t>(size), 0};
        const auto context = acquire_vulkan_context();
        const auto& pipeline = context->pipelines().specialized("histogram_u8", sizeof(push), std::span<const uint32_t>{});
        const std::array reads{values, counts};
        const std::array writes{counts};
        context->recorders().record(reads, writes, [&](const VkCommandBuffer command) {
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
            vkCmdPushConstants(command, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            vkCmdDispatch(command, std::min(vk::dispatch_groups(*context, size), 4096u), 1, 1);
        });
    }
} // namespace lfs::core::internal
