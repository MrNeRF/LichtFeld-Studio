/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../facade_trace.hpp"
#include "../tensor_vulkan_interop.hpp"
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
#include <format>
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

        using vk_index::shader_dims;
        using vk_index::shader_dtype;

        // The where operands right-aligned to the output's rank, with output
        // axes of size 1 dropped and neighbouring axes merged where every
        // operand is either full or broadcast along both. Operands that match
        // the output set a bit of `direct`: they read at the output index.
        struct WhereShape {
            std::array<std::array<uint32_t, MAX_TENSOR_RANK>, 4> dims{};
            uint32_t rank = 0;
            uint32_t direct = 0;
        };

        WhereShape collapse_where(const std::array<const StridedLayout*, 3> operands, const StridedLayout& output) {
            const size_t rank = output.rank;
            std::array<std::array<size_t, MAX_TENSOR_RANK>, 3> padded{};
            for (size_t operand = 0; operand < 3; ++operand) {
                const StridedLayout& layout = *operands[operand];
                LFS_ASSERT_MSG(layout.rank <= rank,
                               std::format("where operand rank {} exceeds the output rank {}", layout.rank, rank));
                for (size_t axis = 0; axis < rank; ++axis) {
                    const size_t source = axis + layout.rank;
                    padded[operand][axis] = source >= rank ? layout.dims[source - rank] : 1;
                }
            }
            WhereShape shape;
            std::array<bool, 3> previous_full{};
            for (size_t axis = 0; axis < rank; ++axis) {
                if (output.dims[axis] == 1)
                    continue;
                std::array<bool, 3> full{};
                for (size_t operand = 0; operand < 3; ++operand)
                    full[operand] = padded[operand][axis] != 1;
                if (shape.rank > 0 && full == previous_full) {
                    const uint32_t last = shape.rank - 1;
                    shape.dims[3][last] = checked_u32(shape.dims[3][last] * output.dims[axis],
                                                      "Vulkan where extent exceeds uint32");
                    for (size_t operand = 0; operand < 3; ++operand)
                        shape.dims[operand][last] = full[operand] ? shape.dims[3][last] : 1u;
                    continue;
                }
                shape.dims[3][shape.rank] = checked_u32(output.dims[axis], "Vulkan where extent exceeds uint32");
                for (size_t operand = 0; operand < 3; ++operand)
                    shape.dims[operand][shape.rank] = full[operand] ? shape.dims[3][shape.rank] : 1u;
                previous_full = full;
                ++shape.rank;
            }
            if (shape.rank == 0) {
                shape.rank = 1;
                for (auto& dims : shape.dims)
                    dims[0] = 1;
            }
            for (size_t operand = 0; operand < 3; ++operand) {
                bool matches = true;
                for (uint32_t axis = 0; axis < shape.rank; ++axis)
                    matches = matches && shape.dims[operand][axis] == shape.dims[3][axis];
                shape.direct |= matches ? 1u << operand : 0u;
            }
            return shape;
        }

        void record_mask(VulkanContext& context, const uint32_t mode, const DataType dtype,
                         const uint32_t predicate, const MaskPush& push,
                         const std::span<const StorageRef> reads,
                         const std::span<const StorageRef> writes, const uint32_t groups) {
            const bool packed = (mode == kFillMode || mode == kAndLiveMode) && dtype_size(dtype) == 1;
            const size_t work = (push.count + (push.data_address & 3u) + 3u) / 4u;
            const uint32_t dispatch_count = packed ? dispatch_groups(context, work) : groups;
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
                    vkCmdDispatch(command, dispatch_count, 1, 1);
                });
        }

        StorageRef scan_predicate(VulkanContext& context, const uint32_t predicate,
                                  const StorageRef mask, const size_t count) {
            StorageRef scan = context.memory().allocate(count * sizeof(uint32_t), 16, {});
            scan.dtype = DataType::Int32;
            const MaskPush push{
                .mask_address = address(mask),
                .scan_address = address(scan),
                .count = checked_u32(count, "Vulkan mask count exceeds uint32"),
            };
            const std::array reads{mask};
            const std::array writes{scan};
            record_mask(context, kScanMode, DataType::UInt8, predicate, push, reads, writes,
                        dispatch_groups(context, count));
            StridedLayout layout{};
            layout.rank = 1;
            layout.dims[0] = count;
            layout.strides[0] = 1;
            layout.element_count = count;
            backend_ops(GpuBackend::Vulkan).cumsum(scan, layout, 0, {});
            return scan;
        }

        uint32_t read_total(const StorageRef total) {
            uint32_t value = 0;
            backend_ops(GpuBackend::Vulkan).copy_device_to_host(CopyRequest{
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
            const StorageRef scan = scan_predicate(*context, predicate, input, program.count);
            const MaskPush push{
                .mask_address = address(input),
                .source_address = address(output),
                .scan_address = address(scan),
                .count = checked_u32(program.count, "Vulkan nonzero count exceeds uint32"),
            };
            const std::array reads{input, scan};
            const std::array writes{output};
            record_mask(*context, kNonzeroMode, DataType::Int64, predicate, push, reads, writes,
                        dispatch_groups(*context, program.count));
            const uint32_t total = read_total(offset_storage_ref(scan, (program.count - 1) * sizeof(uint32_t)));
            context->memory().deallocate(scan);
            return total;
        }
    } // namespace

    void vulkan_where_into(Tensor& output, const Tensor& condition, float value, const Tensor& source) {
        LFS_FACADE_TRACE(where);
        pin_operands({&output, &condition, &source});
        const auto context = acquire_vulkan_context();
        const auto [low, high] = fill_bits(output.dtype(), scalar_operand(value));
        const auto destination = storage_ref(output);
        const auto mask = storage_ref(condition);
        const auto input = storage_ref(source);
        const MaskPush push{
            .data_address = address(destination),
            .mask_address = address(mask),
            .source_address = address(input),
            .count = checked_u32(output.numel(), "Vulkan where_into count exceeds uint32"),
            .fill_low = low,
            .fill_high = high,
        };
        const std::array reads{mask, input};
        const std::array writes{destination};
        record_mask(*context, 6, output.dtype(), kBytePredicate, push, reads, writes,
                    dispatch_groups(*context, output.numel()));
    }

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
        const StorageRef scan = scan_predicate(*context, kBytePredicate, mask, program.count);
        const MaskPush push{
            .data_address = address(input),
            .mask_address = address(mask),
            .source_address = address(output),
            .scan_address = address(scan),
            .count = checked_u32(program.count, "Vulkan masked_select count exceeds uint32"),
        };
        const std::array reads{input, mask, scan};
        const std::array writes{output};
        record_mask(*context, kCompactSelectMode, input.dtype, kBytePredicate, push, reads, writes,
                    dispatch_groups(*context, program.count));
        context->memory().deallocate(scan);
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
        const StorageRef scan = scan_predicate(*context, kBytePredicate, mask, program.count);
        const MaskPush push{
            .data_address = address(output),
            .mask_address = address(mask),
            .source_address = address(source),
            .scan_address = address(scan),
            .count = checked_u32(program.count, "Vulkan masked_scatter count exceeds uint32"),
        };
        const std::array reads{mask, source, scan};
        const std::array writes{output};
        record_mask(*context, kCompactScatterMode, output.dtype, kBytePredicate, push, reads, writes,
                    dispatch_groups(*context, program.count));
        context->memory().deallocate(scan);
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
        LFS_ASSERT_MSG(condition.dtype == DataType::Bool && x.dtype == y.dtype &&
                           x.dtype == output.dtype,
                       "Vulkan where requires a Bool condition and matching value dtypes");
        if (output_layout.element_count == 0) {
            return;
        }
        const auto context = acquire_vulkan_context();
        const WhereShape shape = collapse_where({&condition_layout, &x_layout, &y_layout}, output_layout);
        const WherePush push{
            .condition_address = address(condition),
            .x_address = address(x),
            .y_address = address(y),
            .output_address = address(output),
            .condition_dims = shape.dims[0],
            .x_dims = shape.dims[1],
            .y_dims = shape.dims[2],
            .output_dims = shape.dims[3],
            .condition_rank = shape.rank,
            .x_rank = shape.rank,
            .y_rank = shape.rank,
            .output_rank = shape.rank,
            .count = checked_u32(output_layout.element_count, "Vulkan where count exceeds uint32"),
        };
        const VulkanPipeline& pipeline =
            context->pipelines().specialized("where", sizeof(WherePush),
                                             std::array{static_cast<uint32_t>(output.dtype), shape.direct});
        const std::array reads{condition, x, y};
        const std::array writes{output};
        context->recorders().record(
            reads, writes, [&](const VkCommandBuffer command) {
                vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
                vkCmdPushConstants(command, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                   sizeof(push), &push);
                vkCmdDispatch(command, dispatch_groups(*context, dtype_size(output.dtype) == 1 ? (output_layout.element_count + (push.output_address & 3u) + 3u) / 4u : output_layout.element_count), 1, 1);
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
