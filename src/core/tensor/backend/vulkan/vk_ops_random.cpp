/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../facade_trace.hpp"
#include "vk_backend_ops.hpp"

#include "../../internal/tensor_impl.hpp"
#include "core/assert.hpp"
#include "vk_context.hpp"
#include "vk_memory.hpp"
#include "vk_ops_common.hpp"
#include "vk_pipelines.hpp"
#include "vk_recorder.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>

namespace lfs::core::internal {
    namespace {
        using vk::address;
        using vk::checked_u32;
        using vk::dispatch_groups;

        // random.slang kinds.
        constexpr uint32_t kUniform = 0;
        constexpr uint32_t kRandint = 2;
        constexpr uint32_t kNormal = 3;

        struct RandomPush {
            uint64_t output_address;
            uint64_t seed;
            uint32_t count;
            int32_t low;
            int32_t high;
            float first;
            float second;
            uint32_t pad0;
        };
        static_assert(sizeof(RandomPush) == 40);

        void record_random(VulkanContext& context, const uint32_t kind, const RandomPush& push,
                           const std::span<const StorageRef> reads,
                           const std::span<const StorageRef> writes, const uint32_t groups) {
            const std::array constants{kind};
            const VulkanPipeline& pipeline =
                context.pipelines().specialized("random", sizeof(RandomPush), constants);
            context.recorders().record(
                reads, writes, [&](const VkCommandBuffer command) {
                    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                      pipeline.pipeline);
                    vkCmdPushConstants(command, pipeline.layout,
                                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                       sizeof(push), &push);
                    vkCmdDispatch(command, groups, 1, 1);
                });
        }

        // Elementwise draws: every element is one Philox block keyed by the seed.
        void draw_elements(const uint32_t kind, const StorageRef output,
                           const RandomProgram& program, const uint64_t seed) {
            if (program.count == 0) {
                return;
            }
            const auto context = acquire_vulkan_context();
            const RandomPush push{
                .output_address = address(output),
                .seed = seed,
                .count = checked_u32(program.count, "Vulkan random count exceeds uint32"),
                .low = program.low,
                .high = program.high,
                .first = program.first,
                .second = program.second,
            };
            const std::array writes{output};
            record_random(*context, kind, push, {}, writes, dispatch_groups(*context, program.count));
        }

    } // namespace

    void VulkanBackendOps::uniform(
        const StorageRef output, const RandomProgram& program, ExecContext) {
        LFS_FACADE_TRACE(uniform);
        draw_elements(kUniform, output, program, program.seed);
    }

    void VulkanBackendOps::randint(
        const StorageRef output, const RandomProgram& program, ExecContext) {
        LFS_FACADE_TRACE(randint);
        draw_elements(kRandint, output, program, program.seed);
    }

    void VulkanBackendOps::normal(
        const StorageRef output, StorageRef, const RandomProgram& program, ExecContext) {
        LFS_FACADE_TRACE(normal);
        // The CUDA path draws from the process generator's stream; Vulkan draws
        // every element from its own Philox block under the program seed, so odd
        // counts need no scratch.
        draw_elements(kNormal, output, program, program.seed);
    }

} // namespace lfs::core::internal
