/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../facade_trace.hpp"
#include "vk_backend_ops.hpp"
#include "vk_ops_common.hpp"
#include "vk_pipelines.hpp"
#include "vk_recorder.hpp"
namespace lfs::core::internal {
    void VulkanBackendOps::sh_codec(StorageRef source, StorageRef destination, const ShCodecProgram& program, ExecContext) {
        LFS_FACADE_TRACE(sh_codec);
        const auto& p = program.codec;
        const auto context = acquire_vulkan_context();
        struct Push {
            uint64_t source, destination, indices, source_bounds, destination_bounds;
            uint32_t source_rows, destination_rows, count;
            uint32_t source_offset, destination_offset, padding;
        };
        static_assert(sizeof(Push) == 64);
        const auto address = [](const std::optional<StorageRef>& s) { return s ? vk::address(*s) : uint64_t{0}; };
        const Push push{vk::address(source), vk::address(destination), address(program.indices), address(program.source_bounds), address(program.destination_bounds),
                        uint32_t(p.source_rows), uint32_t(p.destination_rows), uint32_t(p.count),
                        uint32_t(p.source_offset), uint32_t(p.destination_offset), 0};
        const std::array constants{uint32_t(p.source_format), uint32_t(p.destination_format),
                                   program.indices ? (program.indices->dtype == DataType::Int64 ? 2u : 1u) : 0u, uint32_t(p.scatter), p.source_rest, p.destination_rest};
        const bool encode = p.destination_format == ShFormat::Q16;
        const auto& pipeline = context->pipelines().specialized(encode ? "sh_encode" : "sh_codec", sizeof(push), constants);
        std::array<StorageRef, 4> reads{source, destination, source, source};
        size_t count = 2;
        if (program.indices)
            reads[count++] = *program.indices;
        if (program.source_bounds)
            reads[count++] = *program.source_bounds;
        const std::array writes{destination, program.destination_bounds.value_or(destination)};
        context->recorders().record(std::span(reads.data(), count), std::span(writes.data(), encode ? 2 : 1), [&](VkCommandBuffer command) {
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
            vkCmdPushConstants(command, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            const size_t work = ((p.count + 31) / 32) * 32 * ((p.destination_rest * 3 + 3) / 4) * 4;
            vkCmdDispatch(command, encode ? uint32_t((p.count + 255) / 256) : std::min(vk::dispatch_groups(*context, work), 4096u), 1, 1);
        });
    }
} // namespace lfs::core::internal
