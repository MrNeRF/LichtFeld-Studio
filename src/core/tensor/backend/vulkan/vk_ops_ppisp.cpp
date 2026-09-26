/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../facade_trace.hpp"
#include "core/tensor_ppisp.hpp"
#include "vk_backend_ops.hpp"
#include "vk_ops_common.hpp"
#include "vk_pipelines.hpp"
#include "vk_recorder.hpp"
namespace lfs::core::internal {
    void VulkanBackendOps::ppisp_apply(StorageRef input, StorageRef output, int width, int height, const PpispParams& p, ExecContext) {
        LFS_FACADE_TRACE(ppisp_apply);
        const auto context = acquire_vulkan_context();
        vk::ScopedAllocation parameters(*context, sizeof(p));
        context->memory().copy_host_to_device(CopyRequest{.src = raw_storage_ref(const_cast<PpispParams*>(&p)),
                                                          .dst = parameters.storage(),
                                                          .bytes = sizeof(p),
                                                          .synchronous = false});
        struct Push {
            uint64_t input, output, parameters;
            int width, height;
        };
        const Push push{vk::address(input), vk::address(output), vk::address(parameters.storage()), width, height};
        const auto& pipeline = context->pipelines().specialized("ppisp_apply", sizeof(push), {});
        const std::array reads{input, parameters.storage()};
        const std::array writes{output};
        context->recorders().record(reads, writes, [&](VkCommandBuffer command) {
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
            vkCmdPushConstants(command, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            vkCmdDispatch(command, vk::dispatch_groups(*context, size_t(width) * height), 1, 1);
        });
    }
} // namespace lfs::core::internal
