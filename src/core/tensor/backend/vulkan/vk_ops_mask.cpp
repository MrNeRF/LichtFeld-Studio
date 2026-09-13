/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../facade_trace.hpp"
#include "vk_backend_ops.hpp"

#include "../../internal/tensor_impl.hpp"
#include "core/assert.hpp"
#include "vk_context.hpp"
#include "vk_memory.hpp"
#include "vk_ops_common.hpp"
#include "vk_ops_index_common.hpp"
#include "vk_pipelines.hpp"
#include "vk_recorder.hpp"

#include <array>
#include <span>

namespace lfs::core::internal {
    namespace {
        using vk::address;
        using vk::checked_u32;
        using vk::dispatch_groups;

        // Modes of mask.slang.
        constexpr uint32_t kFillMode = 0;
        constexpr uint32_t kAndLiveMode = 1;
        constexpr uint32_t kCompactSelectMode = 2;
        constexpr uint32_t kCompactScatterMode = 3;
        constexpr uint32_t kNonzeroMode = 4;
        constexpr uint32_t kScanMode = 5;

        constexpr uint32_t kBytePredicate = 0;
        constexpr uint32_t kFloatPredicate = 1;

        struct MaskPush {
            uint64_t data_address;
            uint64_t mask_address;
            uint64_t source_address;
            uint64_t scan_address;
            uint32_t count;
            uint32_t fill_low;
            uint32_t fill_high;
            uint32_t pad0;
        };
        static_assert(sizeof(MaskPush) == 48);

        struct WherePush {
            uint64_t condition_address;
            uint64_t x_address;
            uint64_t y_address;
            uint64_t output_address;
            std::array<uint32_t, MAX_TENSOR_RANK> condition_dims;
            std::array<uint32_t, MAX_TENSOR_RANK> x_dims;
            std::array<uint32_t, MAX_TENSOR_RANK> y_dims;
            std::array<uint32_t, MAX_TENSOR_RANK> output_dims;
            uint32_t condition_rank;
            uint32_t x_rank;
            uint32_t y_rank;
            uint32_t output_rank;
            uint32_t count;
            uint32_t pad0;
        };
        static_assert(sizeof(WherePush) == 184);

        using vk_index::fill_bits;
        using vk_index::shader_dims;
        using vk_index::shader_dtype;

        void record_mask(VulkanContext& context, const uint32_t mode, const DataType dtype,
                         const uint32_t predicate, const MaskPush& push,
                         const std::span<const StorageRef> reads,
                         const std::span<const StorageRef> writes, const uint32_t groups) {
            const std::array constants{mode, shader_dtype(dtype), predicate};
            const VulkanPipeline& pipeline =
                context.pipelines().specialized("mask", sizeof(MaskPush), constants);
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

        // Exclusive scan of the predicate over count elements, computed by one
        // workgroup so compaction offsets are deterministic; the total lands in a
        // 16-byte scratch for callers that return the selected count.
        struct Compaction {
            StorageRef scan;
            StorageRef total;
        };

        Compaction scan_predicate(VulkanContext& context, const uint32_t predicate,
                                  const StorageRef mask, const size_t count) {
            Compaction result{
                .scan = context.memory().allocate(count * sizeof(uint32_t), 16, {}),
                .total = context.memory().allocate(16, 16, {}),
            };
            const MaskPush push{
                .mask_address = address(mask),
                .source_address = address(result.total),
                .scan_address = address(result.scan),
                .count = checked_u32(count, "Vulkan mask count exceeds uint32"),
            };
            const std::array reads{mask};
            const std::array writes{result.scan, result.total};
            record_mask(context, kScanMode, DataType::UInt8, predicate, push, reads, writes, 1);
            return result;
        }

        uint32_t read_total(VulkanContext& context, const StorageRef total) {
            uint32_t value = 0;
            context.memory().copy_device_to_host(CopyRequest{
                .src = total,
                .dst = raw_storage_ref(&value),
                .bytes = sizeof(value),
                .synchronous = true,
                .operation = "tensor.mask.count",
            });
            return value;
        }

        size_t compact_nonzero(const uint32_t predicate, const StorageRef input,
                               const StorageRef output, const MaskProgram& program) {
            if (program.count == 0 || program.selected_count == 0) {
                return 0;
            }
            const auto context = acquire_vulkan_context();
            const Compaction compaction = scan_predicate(*context, predicate, input, program.count);
            const MaskPush push{
                .mask_address = address(input),
                .source_address = address(output),
                .scan_address = address(compaction.scan),
                .count = checked_u32(program.count, "Vulkan nonzero count exceeds uint32"),
            };
            const std::array reads{input, compaction.scan};
            const std::array writes{output};
            record_mask(*context, kNonzeroMode, DataType::Int64, predicate, push, reads, writes,
                        dispatch_groups(*context, program.count));
            const uint32_t total = read_total(*context, compaction.total);
            context->memory().deallocate(compaction.scan);
            context->memory().deallocate(compaction.total);
            return total;
        }
    } // namespace

    void VulkanBackendOps::masked_fill(
        const StorageRef output, const StorageRef mask, const MaskProgram& program, ExecContext) {
        LFS_FACADE_TRACE(masked_fill);
        if (program.count == 0) {
            return;
        }
        const auto context = acquire_vulkan_context();
        const auto [low, high] = fill_bits(output.dtype, program.value);
        const MaskPush push{
            .data_address = address(output),
            .mask_address = address(mask),
            .count = checked_u32(program.count, "Vulkan masked_fill count exceeds uint32"),
            .fill_low = low,
            .fill_high = high,
        };
        const std::array reads{mask};
        const std::array writes{output};
        record_mask(*context, kFillMode, output.dtype, kBytePredicate, push, reads, writes,
                    dispatch_groups(*context, program.count));
    }

    size_t VulkanBackendOps::masked_select(
        const StorageRef input, const StorageRef mask, const StorageRef output,
        const MaskProgram& program, ExecContext) {
        LFS_FACADE_TRACE(masked_select);
        if (program.count == 0 || program.selected_count == 0) {
            return 0;
        }
        const auto context = acquire_vulkan_context();
        const Compaction compaction = scan_predicate(*context, kBytePredicate, mask, program.count);
        const MaskPush push{
            .data_address = address(input),
            .mask_address = address(mask),
            .source_address = address(output),
            .scan_address = address(compaction.scan),
            .count = checked_u32(program.count, "Vulkan masked_select count exceeds uint32"),
        };
        const std::array reads{input, mask, compaction.scan};
        const std::array writes{output};
        record_mask(*context, kCompactSelectMode, input.dtype, kBytePredicate, push, reads, writes,
                    dispatch_groups(*context, program.count));
        context->memory().deallocate(compaction.scan);
        context->memory().deallocate(compaction.total);
        // The host sized the output from the same mask; like CUDA the launch trusts it.
        return program.selected_count;
    }

    void VulkanBackendOps::masked_scatter(
        const StorageRef output, const StorageRef mask, const StorageRef source,
        const MaskProgram& program, ExecContext) {
        LFS_FACADE_TRACE(masked_scatter);
        if (program.count == 0 || program.selected_count == 0) {
            return;
        }
        const auto context = acquire_vulkan_context();
        const Compaction compaction = scan_predicate(*context, kBytePredicate, mask, program.count);
        const MaskPush push{
            .data_address = address(output),
            .mask_address = address(mask),
            .source_address = address(source),
            .scan_address = address(compaction.scan),
            .count = checked_u32(program.count, "Vulkan masked_scatter count exceeds uint32"),
        };
        const std::array reads{mask, source, compaction.scan};
        const std::array writes{output};
        record_mask(*context, kCompactScatterMode, output.dtype, kBytePredicate, push, reads, writes,
                    dispatch_groups(*context, program.count));
        context->memory().deallocate(compaction.scan);
        context->memory().deallocate(compaction.total);
    }

    void VulkanBackendOps::and_live(
        const StorageRef mask, const StorageRef live_mask, const MaskProgram& program, ExecContext) {
        LFS_FACADE_TRACE(and_live);
        if (program.count == 0) {
            return;
        }
        const auto context = acquire_vulkan_context();
        const MaskPush push{
            .data_address = address(mask),
            .mask_address = address(live_mask),
            .count = checked_u32(program.count, "Vulkan and_live count exceeds uint32"),
        };
        const std::array reads{live_mask, mask};
        const std::array writes{mask};
        record_mask(*context, kAndLiveMode, DataType::UInt8, kBytePredicate, push, reads, writes,
                    dispatch_groups(*context, program.count));
    }

    void VulkanBackendOps::where(
        const StorageRef condition, const StorageRef x, const StorageRef y,
        const StorageRef output, const StridedLayout& condition_layout,
        const StridedLayout& x_layout, const StridedLayout& y_layout,
        const StridedLayout& output_layout, ExecContext) {
        LFS_FACADE_TRACE(where);
        LFS_ASSERT_MSG(x.dtype == DataType::Float32 && y.dtype == DataType::Float32 &&
                           output.dtype == DataType::Float32,
                       "Vulkan where supports only Float32 operands");
        if (output_layout.element_count == 0) {
            return;
        }
        const auto context = acquire_vulkan_context();
        const WherePush push{
            .condition_address = address(condition),
            .x_address = address(x),
            .y_address = address(y),
            .output_address = address(output),
            .condition_dims = shader_dims(condition_layout),
            .x_dims = shader_dims(x_layout),
            .y_dims = shader_dims(y_layout),
            .output_dims = shader_dims(output_layout),
            .condition_rank = static_cast<uint32_t>(condition_layout.rank),
            .x_rank = static_cast<uint32_t>(x_layout.rank),
            .y_rank = static_cast<uint32_t>(y_layout.rank),
            .output_rank = static_cast<uint32_t>(output_layout.rank),
            .count = checked_u32(output_layout.element_count, "Vulkan where count exceeds uint32"),
        };
        const VulkanPipeline& pipeline =
            context->pipelines().specialized("where", sizeof(WherePush), {});
        const std::array reads{condition, x, y};
        const std::array writes{output};
        context->recorders().record(
            reads, writes, [&](const VkCommandBuffer command) {
                vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
                vkCmdPushConstants(command, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                   sizeof(push), &push);
                vkCmdDispatch(command, dispatch_groups(*context, output_layout.element_count), 1, 1);
            });
    }

    size_t VulkanBackendOps::nonzero(
        const StorageRef input, const StorageRef output, const MaskProgram& program, ExecContext) {
        LFS_FACADE_TRACE(nonzero);
        LFS_ASSERT_MSG(input.dtype == DataType::Float32, "Vulkan nonzero supports only Float32");
        return compact_nonzero(kFloatPredicate, input, output, program);
    }

    size_t VulkanBackendOps::nonzero_bool(
        const StorageRef input, const StorageRef output, const MaskProgram& program, ExecContext) {
        LFS_FACADE_TRACE(nonzero_bool);
        return compact_nonzero(kBytePredicate, input, output, program);
    }

} // namespace lfs::core::internal
