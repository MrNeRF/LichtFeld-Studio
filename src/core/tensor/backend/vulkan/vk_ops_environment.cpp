/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../facade_trace.hpp"
#include "core/tensor_environment.hpp"
#include "vk_backend_ops.hpp"
#include "vk_ops_common.hpp"
#include "vk_pipelines.hpp"
#include "vk_recorder.hpp"
namespace lfs::core::internal {
    void VulkanBackendOps::environment_composite(StorageRef rgb, StorageRef alpha, StorageRef environment,
                                                 StorageRef output, const EnvironmentCompositeParams& p, ExecContext) {
        LFS_FACADE_TRACE(environment_composite);
        struct Push {
            uint64_t rgb, alpha, environment, output;
            EnvironmentCompositeParams p;
            uint32_t padding;
        };
        static_assert(sizeof(Push) == 128);
        const Push push{vk::address(rgb), vk::address(alpha), vk::address(environment), vk::address(output), p, 0};
        const auto context = acquire_vulkan_context();
        const std::array constants{uint32_t(p.equirect_view)};
        const auto& pipeline = context->pipelines().specialized("environment_composite", sizeof(push), constants);
        const std::array reads{rgb, alpha, environment};
        const std::array writes{output};
        context->recorders().record(reads, writes, [&](VkCommandBuffer command) {
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
            vkCmdPushConstants(command, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
            vkCmdDispatch(command, vk::dispatch_groups(*context, (size_t(p.band_width) * p.band_height + 3) / 4), 1, 1);
        });
    }
} // namespace lfs::core::internal
