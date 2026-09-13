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

#include <algorithm>
#include <array>
#include <span>

namespace lfs::core::internal {
    namespace {
        using vk::address;
        using vk::checked_u32;
        using vk::dispatch_groups;
        using vk::kLocalSize;
        using vk_index::fill_bits;
        using vk_index::shader_dims;
        using vk_index::shader_dtype;

        // Modes of index.slang.
        constexpr uint32_t kGatherMode = 0;
        constexpr uint32_t kTakeMode = 1;
        constexpr uint32_t kIndexSelectMode = 2;
        constexpr uint32_t kScatterAssignMode = 3;
        constexpr uint32_t kScatterAddMode = 4;
        constexpr uint32_t kIndexPutMode = 5;
        constexpr uint32_t kIndexFillMode = 6;
        constexpr uint32_t kSortedRunMode = 7;

        constexpr uint32_t kUnaryNone = 0;
        constexpr uint32_t kUnaryAbs = 1;
        constexpr uint32_t kUnarySqrt = 2;
        constexpr uint32_t kUnaryNeg = 3;

        constexpr uint32_t kRunAssign = 0;
        constexpr uint32_t kRunAdd = 1;

        constexpr int kBoundaryAssert = 0;

        // radix.slang phases and key kinds used to group duplicate targets.
        constexpr uint32_t kExtractPhase = 0;
        constexpr uint32_t kHistogramPhase = 1;
        constexpr uint32_t kScanPhase = 2;
        constexpr uint32_t kScatterPhase = 3;
        constexpr uint32_t kIndexKeys = 1;
        constexpr size_t kRadixBlockElements = kLocalSize * 8;
        constexpr uint32_t kRadixDigits = 16;
        constexpr uint32_t kRadixPasses = 8;

        struct IndexPush {
            uint64_t input_address;
            uint64_t index_address;
            uint64_t value_address;
            uint64_t fault_address;
            uint64_t keys_address;
            uint64_t positions_address;
            uint32_t outer;
            uint32_t dim_size;
            uint32_t inner;
            uint32_t index_size;
            uint32_t total;
            uint32_t rank;
            uint32_t index_rank;
            uint32_t dim;
            uint32_t fill_low;
            uint32_t fill_high;
            uint32_t input_size;
            uint32_t op_id;
            std::array<uint32_t, MAX_TENSOR_RANK> input_dims;
            std::array<uint32_t, MAX_TENSOR_RANK> index_dims;
        };
        static_assert(sizeof(IndexPush) == 160);

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

        struct Geometry {
            size_t outer = 1;
            size_t dim_size = 0;
            size_t inner = 1;
        };

        Geometry geometry(const StridedLayout& layout, const int dim) {
            LFS_ASSERT_MSG(dim >= 0 && static_cast<size_t>(dim) < layout.rank,
                           "index dimension is out of range");
            Geometry result;
            for (size_t axis = 0; axis < layout.rank; ++axis) {
                if (static_cast<int>(axis) < dim) {
                    result.outer *= layout.dims[axis];
                } else if (static_cast<int>(axis) > dim) {
                    result.inner *= layout.dims[axis];
                }
            }
            result.dim_size = layout.dims[static_cast<size_t>(dim)];
            return result;
        }

        struct Launch {
            uint32_t mode;
            DataType dtype;
            uint32_t unary = kUnaryNone;
            uint32_t boundary = 0;
            uint32_t run_op = kRunAssign;
            bool atomic_float = false;
            size_t total = 0;
            IndexPush push{};
        };

        void record_index(VulkanContext& context, const Launch& launch,
                          const std::span<const StorageRef> reads,
                          const std::span<const StorageRef> writes) {
            if (launch.total == 0) {
                return;
            }
            const std::array constants{
                launch.mode,
                shader_dtype(launch.dtype),
                launch.unary,
                launch.boundary,
                launch.run_op,
            };
            const VulkanPipeline& pipeline = context.pipelines().specialized(
                launch.atomic_float ? "index_atomic" : "index", sizeof(IndexPush), constants);
            IndexPush push = launch.push;
            push.total = checked_u32(launch.total, "Vulkan index operation count exceeds uint32");
            context.recorders().record(
                reads, writes, [&](const VkCommandBuffer command) {
                    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                      pipeline.pipeline);
                    vkCmdPushConstants(command, pipeline.layout,
                                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                       sizeof(push), &push);
                    vkCmdDispatch(command, dispatch_groups(context, launch.total), 1, 1);
                });
        }

        // Sorts the int32 index array by target (stable) with radix.slang so
        // duplicate targets form runs; returns the sorted keys and positions.
        struct SortedIndices {
            StorageRef keys;
            StorageRef positions;
            std::array<StorageRef, 3> scratch;
        };

        SortedIndices sort_indices(VulkanContext& context, const StorageRef indices,
                                   const size_t count) {
            const size_t blocks = (count + kRadixBlockElements - 1) / kRadixBlockElements;
            const StorageRef keys_a = context.memory().allocate(count * sizeof(uint32_t), 16, {});
            const StorageRef keys_b = context.memory().allocate(count * sizeof(uint32_t), 16, {});
            const StorageRef positions_a = context.memory().allocate(count * sizeof(uint32_t), 16, {});
            const StorageRef positions_b = context.memory().allocate(count * sizeof(uint32_t), 16, {});
            const StorageRef histogram =
                context.memory().allocate(kRadixDigits * blocks * sizeof(uint32_t), 16, {});
            RadixPush push{
                .values_address = address(indices),
                .keys_a_address = address(keys_a),
                .keys_b_address = address(keys_b),
                .positions_a_address = address(positions_a),
                .positions_b_address = address(positions_b),
                .histogram_address = address(histogram),
                .lines = 1,
                .dim_size = checked_u32(count, "Vulkan index count exceeds uint32"),
                .inner = 1,
                .blocks_per_line = checked_u32(blocks, "Vulkan index block count exceeds uint32"),
                .total = checked_u32(count, "Vulkan index count exceeds uint32"),
            };
            const auto dispatch = [&](const uint32_t phase, const uint32_t groups,
                                      const std::span<const StorageRef> reads,
                                      const std::span<const StorageRef> writes) {
                const std::array constants{phase, kIndexKeys};
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
            const uint32_t block_groups = static_cast<uint32_t>(
                std::min<size_t>(blocks, context.caps().max_workgroup_count[0]));
            {
                const std::array reads{indices};
                const std::array writes{keys_a, positions_a};
                dispatch(kExtractPhase, dispatch_groups(context, count), reads, writes);
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
                    dispatch(kScanPhase, 1, reads, writes);
                }
                {
                    const std::array reads{source_keys, source_positions, histogram};
                    const std::array writes{target_keys, target_positions};
                    dispatch(kScatterPhase, block_groups, reads, writes);
                }
            }
            return SortedIndices{
                .keys = keys_a,
                .positions = positions_a,
                .scratch = {keys_b, positions_b, histogram},
            };
        }

        // Scatter-family launches cover outer * index_size * inner source elements.
        Launch scatter_launch(const uint32_t mode, const StorageRef output,
                              const StorageRef indices, const StorageRef source,
                              const StridedLayout& output_layout, const int dim,
                              const size_t index_size) {
            const Geometry shape = geometry(output_layout, dim);
            Launch launch{.mode = mode, .dtype = output.dtype};
            launch.total = shape.outer * index_size * shape.inner;
            launch.push.input_address = address(output);
            launch.push.index_address = address(indices);
            launch.push.value_address = mode == kIndexFillMode ? 0 : address(source);
            launch.push.outer = checked_u32(shape.outer, "Vulkan scatter outer size exceeds uint32");
            launch.push.dim_size = checked_u32(shape.dim_size, "Vulkan scatter dimension exceeds uint32");
            launch.push.inner = checked_u32(shape.inner, "Vulkan scatter inner size exceeds uint32");
            launch.push.index_size = checked_u32(index_size, "Vulkan scatter index count exceeds uint32");
            return launch;
        }

        // Deterministic scatter: every duplicate target is served by one thread
        // that applies its sources in position order (last index wins for
        // assignment, sequential accumulation otherwise).
        void scatter_sorted(VulkanContext& context, Launch launch, const uint32_t run_op,
                            const StorageRef output, const StorageRef indices,
                            const StorageRef source) {
            if (launch.total == 0) {
                return;
            }
            const SortedIndices sorted = sort_indices(context, indices, launch.push.index_size);
            launch.mode = kSortedRunMode;
            launch.run_op = run_op;
            launch.push.keys_address = address(sorted.keys);
            launch.push.positions_address = address(sorted.positions);
            const std::array reads{indices, source, sorted.keys, sorted.positions};
            const std::array writes{output};
            record_index(context, launch, reads, writes);
            context.memory().deallocate(sorted.keys);
            context.memory().deallocate(sorted.positions);
            for (const StorageRef storage : sorted.scratch) {
                context.memory().deallocate(storage);
            }
        }

        void scatter_add(VulkanContext& context, Launch launch, const StorageRef output,
                         const StorageRef indices, const StorageRef source) {
            LFS_ASSERT_MSG(output.dtype == DataType::Float32 || output.dtype == DataType::Int32 ||
                               output.dtype == DataType::UInt8 || output.dtype == DataType::Bool,
                           "Vulkan scatter add supports Float32, Int32 and byte tensors");
            // Integer adds are order-independent, so their atomics stay deterministic;
            // float adds use the device atomic when present and otherwise the
            // sorted-run path, which matches a sequential CPU reference.
            if (output.dtype == DataType::Float32 && !context.caps().shader_atomic_float) {
                scatter_sorted(context, launch, kRunAdd, output, indices, source);
                return;
            }
            launch.mode = kScatterAddMode;
            launch.atomic_float = output.dtype == DataType::Float32;
            const std::array reads{indices, source};
            const std::array writes{output};
            record_index(context, launch, reads, writes);
        }
    } // namespace

    void VulkanBackendOps::gather(
        const StorageRef input, const StorageRef indices, const StorageRef output,
        const StridedLayout& input_layout, const StridedLayout& index_layout,
        const IndexProgram& program, ExecContext) {
        LFS_FACADE_TRACE(gather);
        LFS_ASSERT_MSG(input_layout.rank > 0 && input_layout.rank <= MAX_TENSOR_RANK &&
                           index_layout.rank <= MAX_TENSOR_RANK,
                       "gather rank exceeds MAX_TENSOR_RANK");
        const auto context = acquire_vulkan_context();
        Launch launch{.mode = kGatherMode, .dtype = input.dtype};
        launch.boundary = static_cast<uint32_t>(program.boundary_mode);
        launch.total = program.total_elements;
        launch.push.input_address = address(input);
        launch.push.index_address = address(indices);
        launch.push.value_address = address(output);
        launch.push.fault_address =
            program.boundary_mode == kBoundaryAssert ? context->fault_address() : 0;
        launch.push.rank = static_cast<uint32_t>(input_layout.rank);
        launch.push.index_rank = static_cast<uint32_t>(index_layout.rank);
        launch.push.dim = static_cast<uint32_t>(program.dim);
        launch.push.input_dims = shader_dims(input_layout);
        launch.push.index_dims = shader_dims(index_layout);
        const std::array reads{input, indices};
        const std::array writes{output};
        record_index(*context, launch, reads, writes);
    }

    void VulkanBackendOps::gather_fused_unary(
        const StorageRef input, const StorageRef indices, const StorageRef output,
        const PointwiseOp unary, const IndexProgram& program, ExecContext) {
        LFS_FACADE_TRACE(gather_fused_unary);
        LFS_ASSERT_MSG(input.dtype == DataType::Float32 && output.dtype == DataType::Float32,
                       "Vulkan fused gather supports only Float32");
        uint32_t unary_code = kUnaryNone;
        switch (unary) {
        case PointwiseOp::Abs: unary_code = kUnaryAbs; break;
        case PointwiseOp::Sqrt: unary_code = kUnarySqrt; break;
        case PointwiseOp::Neg: unary_code = kUnaryNeg; break;
        default: LFS_ASSERT_MSG(false, "unsupported fused gather unary operation");
        }
        const auto context = acquire_vulkan_context();
        Launch launch{.mode = kTakeMode, .dtype = DataType::Float32, .unary = unary_code};
        launch.total = program.index_size;
        launch.push.input_address = address(input);
        launch.push.index_address = address(indices);
        launch.push.value_address = address(output);
        launch.push.input_size = checked_u32(program.input_size, "Vulkan gather input size exceeds uint32");
        const std::array reads{input, indices};
        const std::array writes{output};
        record_index(*context, launch, reads, writes);
    }

    void VulkanBackendOps::take(
        const StorageRef input, const StorageRef indices, const StorageRef output,
        const IndexProgram& program, ExecContext) {
        LFS_FACADE_TRACE(take);
        const auto context = acquire_vulkan_context();
        Launch launch{.mode = kTakeMode, .dtype = input.dtype};
        launch.total = program.index_size;
        launch.push.input_address = address(input);
        launch.push.index_address = address(indices);
        launch.push.value_address = address(output);
        launch.push.input_size = checked_u32(program.input_size, "Vulkan take input size exceeds uint32");
        const std::array reads{input, indices};
        const std::array writes{output};
        record_index(*context, launch, reads, writes);
    }

    void VulkanBackendOps::index_select(
        const StorageRef input, const StorageRef indices, const StorageRef output,
        const StridedLayout& input_layout, const IndexProgram& program, ExecContext) {
        LFS_FACADE_TRACE(index_select);
        if (program.index_size == 0) {
            return;
        }
        const auto context = acquire_vulkan_context();
        const Geometry shape = geometry(input_layout, program.dim);
        Launch launch{.mode = kIndexSelectMode, .dtype = input.dtype};
        launch.boundary = static_cast<uint32_t>(program.boundary_mode);
        launch.total = shape.outer * program.index_size * shape.inner;
        launch.push.input_address = address(input);
        launch.push.index_address = address(indices);
        launch.push.value_address = address(output);
        // Assert mode records the first out-of-range index; the record is raised
        // at the next synchronization (check_fault_buffer), like the CUDA harvest.
        launch.push.fault_address =
            program.boundary_mode == kBoundaryAssert ? context->fault_address() : 0;
        launch.push.outer = checked_u32(shape.outer, "Vulkan index_select outer size exceeds uint32");
        launch.push.dim_size = checked_u32(shape.dim_size, "Vulkan index_select dimension exceeds uint32");
        launch.push.inner = checked_u32(shape.inner, "Vulkan index_select inner size exceeds uint32");
        launch.push.index_size = checked_u32(program.index_size, "Vulkan index_select index count exceeds uint32");
        const std::array reads{input, indices};
        const std::array writes{output};
        record_index(*context, launch, reads, writes);
    }

    void VulkanBackendOps::scatter(
        const StorageRef output, const StorageRef indices, const StorageRef source,
        const StridedLayout& output_layout, const StridedLayout& source_layout,
        const IndexProgram& program, ExecContext) {
        LFS_FACADE_TRACE(scatter);
        LFS_ASSERT_MSG(output_layout.rank > 0 && output_layout.rank <= MAX_TENSOR_RANK,
                       "scatter rank exceeds MAX_TENSOR_RANK");
        const auto context = acquire_vulkan_context();
        const Launch launch = scatter_launch(kScatterAssignMode, output, indices, source,
                                             output_layout, program.dim,
                                             source_layout.dims[static_cast<size_t>(program.dim)]);
        if (program.scatter_mode == static_cast<int>(ScatterMode::Add)) {
            scatter_add(*context, launch, output, indices, source);
            return;
        }
        // Assignment with duplicate targets is deterministic: the highest source
        // position wins, as in the CPU reference.
        scatter_sorted(*context, launch, kRunAssign, output, indices, source);
    }

    void VulkanBackendOps::index_copy(
        const StorageRef output, const StorageRef indices, const StorageRef source,
        const StridedLayout& output_layout, const IndexProgram& program, ExecContext) {
        LFS_FACADE_TRACE(index_copy);
        const auto context = acquire_vulkan_context();
        const Launch launch = scatter_launch(kScatterAssignMode, output, indices, source,
                                             output_layout, program.dim, program.index_size);
        scatter_sorted(*context, launch, kRunAssign, output, indices, source);
    }

    void VulkanBackendOps::index_add(
        const StorageRef output, const StorageRef indices, const StorageRef source,
        const StridedLayout& output_layout, const IndexProgram& program, ExecContext) {
        LFS_FACADE_TRACE(index_add);
        LFS_ASSERT_MSG(output.dtype == DataType::Float32 || output.dtype == DataType::Int32,
                       "Vulkan index_add supports only Float32 and Int32");
        const auto context = acquire_vulkan_context();
        const Launch launch = scatter_launch(kScatterAddMode, output, indices, source, output_layout,
                                             program.dim, program.index_size);
        scatter_add(*context, launch, output, indices, source);
    }

    void VulkanBackendOps::index_fill(
        const StorageRef output, const StorageRef indices, const StridedLayout& output_layout,
        const IndexProgram& program, const ScalarOperand value, ExecContext) {
        LFS_FACADE_TRACE(index_fill);
        const auto context = acquire_vulkan_context();
        Launch launch = scatter_launch(kIndexFillMode, output, indices, StorageRef{}, output_layout,
                                       program.dim, program.index_size);
        const auto [low, high] = fill_bits(output.dtype, value);
        launch.push.fill_low = low;
        launch.push.fill_high = high;
        const std::array reads{indices};
        const std::array writes{output};
        record_index(*context, launch, reads, writes);
    }

    void VulkanBackendOps::index_put(
        const StorageRef output, const StorageRef indices, const StorageRef values,
        const IndexProgram& program, ExecContext) {
        LFS_FACADE_TRACE(index_put);
        const auto context = acquire_vulkan_context();
        Launch launch{.mode = kIndexPutMode, .dtype = output.dtype};
        launch.total = program.index_size;
        launch.push.input_address = address(output);
        launch.push.index_address = address(indices);
        launch.push.value_address = address(values);
        launch.push.input_size = checked_u32(program.input_size, "Vulkan index_put size exceeds uint32");
        const std::array reads{indices, values};
        const std::array writes{output};
        record_index(*context, launch, reads, writes);
    }

} // namespace lfs::core::internal
