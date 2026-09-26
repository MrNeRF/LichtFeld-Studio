/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../facade_trace.hpp"
#include "internal/point_filter.hpp"
#include "vk_backend_ops.hpp"
#include "vk_ops_common.hpp"
#include "vk_pipelines.hpp"
#include "vk_recorder.hpp"

namespace lfs::core::internal {
    void VulkanBackendOps::filter_points(StorageRef mask, const PointFilterProgram& p, ExecContext) {
        LFS_FACADE_TRACE(filter_points);
        const auto context = acquire_vulkan_context();
        struct Push {
            uint64_t mask;
            std::array<uint64_t, FilterInputCount> inputs;
            uint64_t window;
            uint32_t count, transform_count, allowed_count, flags;
        };
        static_assert(sizeof(Push) == 104);
        Push push{.mask = vk::address(mask),
                  .count = p.count,
                  .transform_count = p.transform_count,
                  .allowed_count = p.allowed_count,
                  .flags = p.flags};
        std::array<StorageRef, 11> reads;
        size_t count = 0;
        reads[count++] = mask;
        for (size_t i = 0; i < p.inputs.size(); ++i)
            if (p.inputs[i]) {
                push.inputs[i] = vk::address(*p.inputs[i]);
                reads[count++] = *p.inputs[i];
            }
        std::optional<vk::ScopedAllocation> window;
        if (p.flags & LFS_FILTER_WINDOW) {
            auto data = pointFilterWindowBlock(p.window);
            window.emplace(*context, sizeof(data));
            context->memory().copy_host_to_device(CopyRequest{.src = raw_storage_ref(data.data()),
                                                              .dst = window->storage(),
                                                              .bytes = sizeof(data),
                                                              .synchronous = false});
            reads[count++] = window->storage();
            push.window = vk::address(window->storage());
        }
        const std::array constants{push.flags, static_cast<uint32_t>(p.window.projection.model)};
        const auto& pipeline = context->pipelines().specialized("filter_points", sizeof(push), constants);
        context->recorders().record(std::span(reads.data(), count), std::span(&mask, 1), [&](VkCommandBuffer command) {
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
            vkCmdPushConstants(command, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            const size_t words = (p.count + (push.mask & 3) + 3) / 4;
            vkCmdDispatch(command, std::min(vk::dispatch_groups(*context, words), 4096u), 1, 1);
        });
    }
} // namespace lfs::core::internal
