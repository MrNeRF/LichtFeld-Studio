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
#include <span>

namespace lfs::core::internal {
    namespace {
        using vk::address;
        using vk::checked_u32;
        using vk::dispatch_groups;
        using vk::kLocalSize;

        // Lines up to this length sort in one workgroup's shared memory
        // (sort.slang); longer lines take the multi-block radix sort (radix.slang).
        constexpr size_t kBitonicCapacity = 2048;
        constexpr size_t kRadixBlockElements = kLocalSize * 8;
        constexpr uint32_t kRadixDigits = 16;
        constexpr uint32_t kRadixPasses = 8;

        // Phases of radix.slang.
        constexpr uint32_t kExtractPhase = 0;
        constexpr uint32_t kHistogramPhase = 1;
        constexpr uint32_t kScanPhase = 2;
        constexpr uint32_t kScatterPhase = 3;
        constexpr uint32_t kGatherPhase = 4;
        constexpr uint32_t kWritePhase = 5;
        constexpr uint32_t kFloatKeys = 0;

        struct SortPush {
            uint64_t values_address;
            uint64_t indices_address;
            uint32_t lines;
            uint32_t dim_size;
            uint32_t inner;
            uint32_t descending;
            uint32_t pad0;
            uint32_t pad1;
        };
        static_assert(sizeof(SortPush) == 40);

        struct RadixPush {
            uint64_t values_address;
            uint64_t indices_address;
            uint64_t keys_a_address;
            uint64_t keys_b_address;
            uint64_t positions_a_address;
            uint64_t positions_b_address;
            uint64_t histogram_address;
            uint32_t lines;
            uint32_t dim_size;
            uint32_t inner;
            uint32_t blocks_per_line;
            uint32_t shift;
            uint32_t parity;
            uint32_t descending;
            uint32_t total;
        };
        static_assert(sizeof(RadixPush) == 88);

        struct Lines {
            size_t count;
            size_t dim_size;
            size_t inner;
            bool descending;
        };

        void sort_in_shared(VulkanContext& context, const StorageRef values,
                            const StorageRef indices, const Lines& lines) {
            const SortPush push{
                .values_address = address(values),
                .indices_address = address(indices),
                .lines = checked_u32(lines.count, "Vulkan sort line count exceeds uint32"),
                .dim_size = checked_u32(lines.dim_size, "Vulkan sort size exceeds uint32"),
                .inner = checked_u32(lines.inner, "Vulkan sort inner size exceeds uint32"),
                .descending = lines.descending ? 1u : 0u,
            };
            const VulkanPipeline& pipeline =
                context.pipelines().specialized("sort", sizeof(SortPush), {});
            const uint32_t groups = static_cast<uint32_t>(
                std::min<size_t>(lines.count, context.caps().max_workgroup_count[0]));
            const std::array reads{values};
            const std::array writes{values, indices};
            context.recorders().record(
                reads, writes, [&](const VkCommandBuffer command) {
                    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
                    vkCmdPushConstants(command, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                       sizeof(push), &push);
                    vkCmdDispatch(command, groups, 1, 1);
                });
        }

        void sort_radix(VulkanContext& context, const StorageRef values,
                        const StorageRef indices, const Lines& lines) {
            const size_t total = lines.count * lines.dim_size;
            const size_t blocks_per_line =
                (lines.dim_size + kRadixBlockElements - 1) / kRadixBlockElements;
            const size_t blocks = lines.count * blocks_per_line;
            const StorageRef keys_a = context.memory().allocate(total * sizeof(uint32_t), 16, {});
            const StorageRef keys_b = context.memory().allocate(total * sizeof(uint32_t), 16, {});
            const StorageRef positions_a = context.memory().allocate(total * sizeof(uint32_t), 16, {});
            const StorageRef positions_b = context.memory().allocate(total * sizeof(uint32_t), 16, {});
            const StorageRef histogram = context.memory().allocate(
                lines.count * kRadixDigits * blocks_per_line * sizeof(uint32_t), 16, {});
            RadixPush push{
                .values_address = address(values),
                .indices_address = address(indices),
                .keys_a_address = address(keys_a),
                .keys_b_address = address(keys_b),
                .positions_a_address = address(positions_a),
                .positions_b_address = address(positions_b),
                .histogram_address = address(histogram),
                .lines = checked_u32(lines.count, "Vulkan sort line count exceeds uint32"),
                .dim_size = checked_u32(lines.dim_size, "Vulkan sort size exceeds uint32"),
                .inner = checked_u32(lines.inner, "Vulkan sort inner size exceeds uint32"),
                .blocks_per_line = checked_u32(blocks_per_line, "Vulkan sort block count exceeds uint32"),
                .descending = lines.descending ? 1u : 0u,
                .total = checked_u32(total, "Vulkan sort element count exceeds uint32"),
            };
            const std::array scratch{keys_a, keys_b, positions_a, positions_b, histogram};
            const auto dispatch = [&](const uint32_t phase, const uint32_t groups,
                                      const std::span<const StorageRef> reads,
                                      const std::span<const StorageRef> writes) {
                const std::array constants{phase, kFloatKeys};
                const VulkanPipeline& pipeline =
                    context.pipelines().specialized("radix", sizeof(RadixPush), constants);
                context.recorders().record(
                    reads, writes, [&](const VkCommandBuffer command) {
                        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
                        vkCmdPushConstants(command, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                           sizeof(push), &push);
                        vkCmdDispatch(command, groups, 1, 1);
                    });
            };
            const uint32_t element_groups = dispatch_groups(context, total);
            const uint32_t block_groups = static_cast<uint32_t>(
                std::min<size_t>(blocks, context.caps().max_workgroup_count[0]));
            const uint32_t line_groups = static_cast<uint32_t>(
                std::min<size_t>(lines.count, context.caps().max_workgroup_count[0]));
            {
                const std::array reads{values};
                const std::array writes{keys_a, positions_a};
                dispatch(kExtractPhase, element_groups, reads, writes);
            }
            for (uint32_t pass = 0; pass < kRadixPasses; ++pass) {
                push.shift = pass * 4;
                push.parity = pass & 1u;
                const StorageRef source_keys = push.parity == 0 ? keys_a : keys_b;
                const StorageRef source_positions = push.parity == 0 ? positions_a : positions_b;
                const StorageRef target_keys = push.parity == 0 ? keys_b : keys_a;
                const StorageRef target_positions = push.parity == 0 ? positions_b : positions_a;
                {
                    const std::array reads{source_keys};
                    const std::array writes{histogram};
                    dispatch(kHistogramPhase, block_groups, reads, writes);
                }
                {
                    const std::array reads{histogram};
                    const std::array writes{histogram};
                    dispatch(kScanPhase, line_groups, reads, writes);
                }
                {
                    const std::array reads{source_keys, source_positions, histogram};
                    const std::array writes{target_keys, target_positions};
                    dispatch(kScatterPhase, block_groups, reads, writes);
                }
            }
            {
                const std::array reads{values, positions_a};
                const std::array writes{keys_b};
                dispatch(kGatherPhase, element_groups, reads, writes);
            }
            {
                const std::array reads{keys_b, positions_a};
                const std::array writes{values, indices};
                dispatch(kWritePhase, element_groups, reads, writes);
            }
            for (const StorageRef storage : scratch) {
                context.memory().deallocate(storage);
            }
        }

        void sort_lines(const StorageRef values, const StorageRef indices, const Lines& lines) {
            LFS_ASSERT_MSG(values.dtype == DataType::Float32 && indices.dtype == DataType::Int64,
                           "Vulkan sort requires Float32 values and Int64 indices");
            if (lines.count == 0 || lines.dim_size == 0) {
                return;
            }
            const auto context = acquire_vulkan_context();
            if (lines.dim_size <= kBitonicCapacity) {
                sort_in_shared(*context, values, indices, lines);
            } else {
                sort_radix(*context, values, indices, lines);
            }
        }
    } // namespace

    void VulkanBackendOps::sort_1d(
        const StorageRef values, const StorageRef indices, const size_t count,
        const SortProgram& program, ExecContext) {
        LFS_FACADE_TRACE(sort_1d);
        sort_lines(values, indices,
                   Lines{.count = 1, .dim_size = count, .inner = 1, .descending = program.descending});
    }

    void VulkanBackendOps::sort_2d(
        const StorageRef values, const StorageRef indices, const SortProgram& program,
        ExecContext) {
        LFS_FACADE_TRACE(sort_2d);
        sort_lines(values, indices,
                   Lines{.count = program.outer_size * program.inner_size,
                         .dim_size = program.dim_size,
                         .inner = program.inner_size,
                         .descending = program.descending});
    }

} // namespace lfs::core::internal
