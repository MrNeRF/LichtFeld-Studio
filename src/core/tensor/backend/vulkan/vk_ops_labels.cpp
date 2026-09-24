/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../facade_trace.hpp"
#include "vk_backend_ops.hpp"
#include "vk_ops_common.hpp"
#include "vk_pipelines.hpp"
#include "vk_recorder.hpp"

namespace lfs::core::internal {
    void VulkanBackendOps::update_labels(const StorageRef output, const StorageRef selected,
                                         const LabelUpdateProgram& p, ExecContext) {
        LFS_FACADE_TRACE(update_labels);
        const auto context = acquire_vulkan_context();
        struct Push {
            uint64_t output, selected, existing, locked, indices, categories, allowed;
            uint32_t count, output_count, allowed_count, label, mode, padding;
        };
        static_assert(sizeof(Push) == 80);
        const auto address = [](const std::optional<StorageRef>& storage) {
            return storage ? vk::address(*storage) : uint64_t{0};
        };
        const Push push{.output = vk::address(output),
                        .selected = vk::address(selected),
                        .existing = address(p.existing),
                        .locked = address(p.locked),
                        .indices = address(p.indices),
                        .categories = address(p.categories),
                        .allowed = address(p.allowed),
                        .count = static_cast<uint32_t>(p.count),
                        .output_count = static_cast<uint32_t>(p.output_count),
                        .allowed_count = static_cast<uint32_t>(p.allowed_count),
                        .label = p.label,
                        .mode = p.mode,
                        .padding = 0};
        std::array<StorageRef, 7> reads;
        size_t read_count = 0;
        reads[read_count++] = output;
        reads[read_count++] = selected;
        for (const auto* input : {&p.existing, &p.locked, &p.indices, &p.categories, &p.allowed})
            if (*input)
                reads[read_count++] = **input;
        const std::array writes{output};
        const auto dispatch = [&](uint32_t phase) {
            const std::array constants{phase};
            const auto& pipeline = context->pipelines().specialized("update_labels", sizeof(push), constants);
            context->recorders().record(std::span(reads.data(), read_count), writes, [&](VkCommandBuffer command) {
                vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
                vkCmdPushConstants(command, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
                const size_t work = phase == 0 || phase == 3 ? (p.output_count + (push.output & 3) + 3) / 4 : p.count;
                vkCmdDispatch(command, std::min(vk::dispatch_groups(*context, work), 4096u), 1, 1);
            });
        };
        if (!p.indices)
            dispatch(0);
        else {
            dispatch(3);
            if (p.mode == 2)
                dispatch(1);
            dispatch(2);
        }
    }
} // namespace lfs::core::internal
