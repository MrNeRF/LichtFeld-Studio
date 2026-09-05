/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "../facade_trace.hpp"
#include "vk_backend_ops.hpp"

#include "../../internal/tensor_impl.hpp"
#include "../../internal/tensor_ops.hpp"
#include "core/assert.hpp"
#include "vk_context.hpp"
#include "vk_memory.hpp"
#include "vk_ops_common.hpp"
#include "vk_pipelines.hpp"
#include "vk_recorder.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace lfs::core::internal {
    namespace {
        using vk::address;
        using vk::checked_u32;
        using vk::dispatch_groups;
        using vk::kLocalSize;

        // Shader modes of reduce.slang.
        constexpr uint32_t kPartialMode = 0;
        constexpr uint32_t kSegmentedMode = 2;
        constexpr uint32_t kStridedMode = 3;
        constexpr uint32_t kGeneralMode = 4;
        // Shader element code for float2 (sum, compensation) partials.
        constexpr uint32_t kPairCode = 8;

        constexpr size_t kMaxPartials = 1024;
        constexpr size_t kElementsPerPartial = kLocalSize * 8;
        constexpr size_t kSingleGroupFullReduce = 4096;
        constexpr size_t kSegmentedThreshold = 64;
        constexpr size_t kSplitOutputLimit = 4096;
        constexpr size_t kSplitReduceThreshold = 1024;
        constexpr size_t kMaxSplits = 64;

        struct ReducePush {
            uint64_t input_address;
            uint64_t output_address;
            uint64_t chain_address;
            uint32_t outer;
            uint32_t reduce;
            uint32_t inner;
            uint32_t count;
            uint32_t chain_length;
            uint32_t split_chunk;
            uint32_t reduced_mask;
            uint32_t rank;
            float mean_scale;
            uint32_t pad0;
            std::array<uint32_t, MAX_TENSOR_RANK> dims;
        };
        static_assert(sizeof(ReducePush) == 96);

        struct CountPush {
            uint64_t input_address;
            uint64_t result_address;
            uint32_t count;
            uint32_t pad0;
            uint32_t pad1;
            uint32_t pad2;
        };
        static_assert(sizeof(CountPush) == 32);

        struct ScanPush {
            uint64_t data_address;
            uint32_t outer;
            uint32_t dim;
            uint32_t inner;
            uint32_t lines;
            uint32_t pad0;
            uint32_t pad1;
        };
        static_assert(sizeof(ScanPush) == 32);

        uint32_t shader_dtype(const DataType dtype) {
            switch (dtype) {
            case DataType::Float32: return 0;
            case DataType::Int32: return 2;
            case DataType::Int64: return 3;
            case DataType::UInt8:
            case DataType::Bool: return 5;
            default: break;
            }
            LFS_ASSERT_MSG(false, "Vulkan reduction received an unsupported dtype");
            return 0;
        }

        bool logical_op(const ReduceOp op) {
            return op == ReduceOp::Any || op == ReduceOp::All;
        }

        // Partial results keep the accumulator width: (sum, compensation) pairs
        // for float sums, float for float max/min/prod, int64 for integer,
        // boolean and logical reductions. Every partial slot is 8 bytes.
        uint32_t partial_code(const DataType input, const ReduceOp op) {
            if (input != DataType::Float32 || logical_op(op)) {
                return shader_dtype(DataType::Int64);
            }
            return op == ReduceOp::Sum || op == ReduceOp::Mean ? kPairCode
                                                               : shader_dtype(DataType::Float32);
        }

        float mean_scale_for(const ReduceOp op, const size_t reduce) {
            return op == ReduceOp::Mean ? 1.0f / static_cast<float>(reduce) : 1.0f;
        }

        struct Chain {
            vk::ScopedAllocation table;
            uint32_t length = 0;
            std::vector<StorageRef> reads;
        };

        Chain make_chain(VulkanContext& context,
                         const tensor_ops::FusedPointwiseOpChain& chain,
                         const std::span<const StorageRef> rhs_storages) {
            Chain result{
                .table = vk::upload_chain(context, chain),
                .length = static_cast<uint32_t>(chain.num_ops),
            };
            result.reads.reserve(1 + rhs_storages.size());
            result.reads.push_back(result.table.storage());
            result.reads.insert(result.reads.end(), rhs_storages.begin(), rhs_storages.end());
            return result;
        }

        struct Stage {
            ReduceOp op;
            uint32_t input_code;
            uint32_t output_code;
            uint32_t mode;
            StorageRef input;
            StorageRef output;
            const Chain* chain = nullptr;
            uint32_t groups_x = 1;
            uint32_t groups_y = 1;
            ReducePush push{};
        };

        void record_stage(VulkanContext& context, Stage stage) {
            LFS_ASSERT_MSG(static_cast<uint8_t>(stage.op) <= static_cast<uint8_t>(ReduceOp::All),
                           "Vulkan reduction received an unsupported operation");
            stage.push.input_address = address(stage.input);
            stage.push.output_address = address(stage.output);
            if (stage.chain != nullptr) {
                stage.push.chain_address = address(stage.chain->table.storage());
                stage.push.chain_length = stage.chain->length;
            }
            const std::array constants{
                static_cast<uint32_t>(stage.op),
                stage.input_code,
                stage.output_code,
                stage.mode,
                stage.chain != nullptr ? 1u : 0u,
            };
            const VulkanPipeline& pipeline =
                context.pipelines().specialized("reduce", sizeof(ReducePush), constants);
            std::vector<StorageRef> reads{stage.input};
            if (stage.chain != nullptr) {
                reads.insert(reads.end(), stage.chain->reads.begin(), stage.chain->reads.end());
            }
            const std::array writes{stage.output};
            context.recorders().record(
                reads, writes, [&](const VkCommandBuffer command) {
                    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                      pipeline.pipeline);
                    vkCmdPushConstants(command, pipeline.layout,
                                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                       sizeof(stage.push), &stage.push);
                    vkCmdDispatch(command, stage.groups_x, stage.groups_y, 1);
                });
        }

        // Reduces count contiguous elements into output[0]. Large inputs go
        // through per-workgroup partials so the result is order-deterministic.
        void reduce_full(VulkanContext& context, const ReduceOp op,
                         const StorageRef input, const DataType input_dtype,
                         const size_t count, const StorageRef output,
                         const DataType output_dtype, const Chain* const chain) {
            if (count == 0) {
                return;
            }
            const float mean_scale = mean_scale_for(op, count);
            if (count <= kSingleGroupFullReduce) {
                Stage stage{
                    .op = op,
                    .input_code = shader_dtype(input_dtype),
                    .output_code = shader_dtype(output_dtype),
                    .mode = kSegmentedMode,
                    .input = input,
                    .output = output,
                    .chain = chain,
                };
                stage.push.outer = 1;
                stage.push.reduce = checked_u32(count, "Vulkan reduction count exceeds uint32");
                stage.push.inner = 1;
                stage.push.mean_scale = mean_scale;
                record_stage(context, stage);
                return;
            }
            const size_t groups =
                std::min(kMaxPartials, (count + kElementsPerPartial - 1) / kElementsPerPartial);
            const uint32_t partial = partial_code(input_dtype, op);
            const vk::ScopedAllocation partial_block(context, groups * sizeof(int64_t));
            const StorageRef partials = partial_block.storage();
            Stage first{
                .op = op,
                .input_code = shader_dtype(input_dtype),
                .output_code = partial,
                .mode = kPartialMode,
                .input = input,
                .output = partials,
                .chain = chain,
                .groups_x = static_cast<uint32_t>(groups),
            };
            first.push.count = checked_u32(count, "Vulkan reduction count exceeds uint32");
            first.push.mean_scale = 1.0f;
            record_stage(context, first);
            Stage second{
                .op = op,
                .input_code = partial,
                .output_code = shader_dtype(output_dtype),
                .mode = kSegmentedMode,
                .input = partials,
                .output = output,
            };
            second.push.outer = 1;
            second.push.reduce = static_cast<uint32_t>(groups);
            second.push.inner = 1;
            second.push.mean_scale = mean_scale;
            record_stage(context, second);
        }

        // Reduces the middle extent of an (outer, reduce, inner) view into
        // outer * inner outputs.
        void reduce_axes(VulkanContext& context, const ReduceOp op,
                         const StorageRef input, const DataType input_dtype,
                         const StorageRef output, const DataType output_dtype,
                         const size_t outer, const size_t reduce, const size_t inner,
                         const Chain* const chain) {
            if (outer == 0 || reduce == 0 || inner == 0) {
                return;
            }
            if (outer == 1 && inner == 1) {
                reduce_full(context, op, input, input_dtype, reduce, output, output_dtype, chain);
                return;
            }
            const float mean_scale = mean_scale_for(op, reduce);
            const uint32_t outer32 = checked_u32(outer, "Vulkan reduction outer size exceeds uint32");
            const uint32_t reduce32 = checked_u32(reduce, "Vulkan reduction size exceeds uint32");
            const uint32_t inner32 = checked_u32(inner, "Vulkan reduction inner size exceeds uint32");
            if (inner == 1 && reduce >= kSegmentedThreshold) {
                Stage stage{
                    .op = op,
                    .input_code = shader_dtype(input_dtype),
                    .output_code = shader_dtype(output_dtype),
                    .mode = kSegmentedMode,
                    .input = input,
                    .output = output,
                    .chain = chain,
                    .groups_x = static_cast<uint32_t>(std::min<size_t>(
                        outer, context.caps().max_workgroup_count[0])),
                };
                stage.push.outer = outer32;
                stage.push.reduce = reduce32;
                stage.push.inner = 1;
                stage.push.mean_scale = mean_scale;
                record_stage(context, stage);
                return;
            }
            const size_t outputs = outer * inner;
            const uint32_t outputs32 = checked_u32(outputs, "Vulkan reduction output count exceeds uint32");
            // Few outputs over a long reduce extent starve the device of threads;
            // the reduce range is split across workgroup rows into partials that
            // a second strided pass combines.
            size_t splits = 1;
            if (outputs < kSplitOutputLimit && reduce >= kSplitReduceThreshold) {
                splits = std::min(kMaxSplits, (reduce + kLocalSize - 1) / kLocalSize);
            }
            if (splits == 1) {
                Stage stage{
                    .op = op,
                    .input_code = shader_dtype(input_dtype),
                    .output_code = shader_dtype(output_dtype),
                    .mode = kStridedMode,
                    .input = input,
                    .output = output,
                    .chain = chain,
                    .groups_x = dispatch_groups(context, outputs),
                };
                stage.push.outer = outer32;
                stage.push.reduce = reduce32;
                stage.push.inner = inner32;
                stage.push.split_chunk = reduce32;
                stage.push.mean_scale = mean_scale;
                record_stage(context, stage);
                return;
            }
            const uint32_t partial = partial_code(input_dtype, op);
            const vk::ScopedAllocation partial_block(context, splits * outputs * sizeof(int64_t));
            const StorageRef partials = partial_block.storage();
            Stage first{
                .op = op,
                .input_code = shader_dtype(input_dtype),
                .output_code = partial,
                .mode = kStridedMode,
                .input = input,
                .output = partials,
                .chain = chain,
                .groups_x = dispatch_groups(context, outputs),
                .groups_y = static_cast<uint32_t>(splits),
            };
            first.push.outer = outer32;
            first.push.reduce = reduce32;
            first.push.inner = inner32;
            first.push.split_chunk = static_cast<uint32_t>((reduce + splits - 1) / splits);
            first.push.mean_scale = 1.0f;
            record_stage(context, first);
            Stage second{
                .op = op,
                .input_code = partial,
                .output_code = shader_dtype(output_dtype),
                .mode = kStridedMode,
                .input = partials,
                .output = output,
                .groups_x = dispatch_groups(context, outputs),
            };
            second.push.outer = 1;
            second.push.reduce = static_cast<uint32_t>(splits);
            second.push.inner = outputs32;
            second.push.split_chunk = static_cast<uint32_t>(splits);
            second.push.mean_scale = mean_scale;
            record_stage(context, second);
        }

        // Reduces the axes of reduced_mask, which are not one contiguous run: the
        // kept axes are permute-copied to the front so the contiguous-run path,
        // with its splits, does the work (one thread per output over a strided
        // walk was two orders of magnitude slower).
        void reduce_general(VulkanBackendOps& ops, VulkanContext& context, const ReduceOp op,
                            const StorageRef input, const DataType input_dtype,
                            const StorageRef output, const DataType output_dtype,
                            const StridedLayout& layout, const uint32_t reduced_mask) {
            size_t outputs = 1;
            size_t reduce = 1;
            std::array<size_t, MAX_TENSOR_RANK> strides{};
            size_t stride = 1;
            for (size_t axis = layout.rank; axis-- > 0;) {
                strides[axis] = stride;
                stride *= layout.dims[axis];
                ((reduced_mask >> axis) & 1u ? reduce : outputs) *= layout.dims[axis];
            }
            if (outputs == 0 || reduce == 0) {
                return;
            }
            StridedLayout permuted{};
            permuted.rank = layout.rank;
            permuted.element_count = layout.element_count;
            size_t position = 0;
            for (const bool reduced_pass : {false, true}) {
                for (size_t axis = 0; axis < layout.rank; ++axis) {
                    if ((((reduced_mask >> axis) & 1u) != 0u) == reduced_pass) {
                        permuted.dims[position] = layout.dims[axis];
                        permuted.strides[position] = strides[axis];
                        ++position;
                    }
                }
            }
            const vk::ScopedAllocation scratch(
                context, layout.element_count * dtype_size(input_dtype));
            StorageRef permuted_input = scratch.storage();
            permuted_input.dtype = input_dtype;
            ops.strided_copy(input, permuted_input, permuted, {});
            reduce_axes(context, op, permuted_input, input_dtype, output, output_dtype,
                        outputs, reduce, 1, nullptr);
        }

        float scalar_reduce(const ReduceOp op, const StorageRef input, const size_t count) {
            LFS_ASSERT_MSG(input.dtype == DataType::Float32,
                           "Vulkan scalar reduction requires Float32 input");
            if (count == 0) {
                switch (op) {
                case ReduceOp::Max: return -std::numeric_limits<float>::infinity();
                case ReduceOp::Min: return std::numeric_limits<float>::infinity();
                default: return 0.0f;
                }
            }
            const auto context = acquire_vulkan_context();
            const vk::ScopedAllocation scratch_block(*context, 16);
            const StorageRef scratch = scratch_block.storage();
            reduce_full(*context, op, input, DataType::Float32, count, scratch,
                        DataType::Float32, nullptr);
            float value = 0.0f;
            context->memory().copy_device_to_host(CopyRequest{
                .src = scratch,
                .dst = raw_storage_ref(&value),
                .bytes = sizeof(value),
                .synchronous = true,
                .operation = "tensor.reduce.scalar",
            });
            return value;
        }

        // count.slang kinds: 0 nonzero bytes, 1 nonzero floats, 2 NaN present,
        // 3 infinity present. Workgroups add their tallies to one zeroed counter.
        uint32_t count_matches(const uint32_t kind, const StorageRef input, const size_t count) {
            if (count == 0) {
                return 0;
            }
            const auto context = acquire_vulkan_context();
            const vk::ScopedAllocation scratch_block(*context, 16);
            const StorageRef scratch = scratch_block.storage();
            context->memory().memset(FillRequest{
                .dst = scratch,
                .bytes = sizeof(uint32_t),
                .value = 0,
                .operation = "tensor.count.zero",
            });
            const std::array constants{kind};
            const VulkanPipeline& pipeline =
                context->pipelines().specialized("count", sizeof(CountPush), constants);
            const CountPush push{
                .input_address = address(input),
                .result_address = address(scratch),
                .count = checked_u32(count, "Vulkan count exceeds uint32"),
            };
            const std::array reads{input};
            const std::array writes{scratch};
            context->recorders().record(
                reads, writes, [&](const VkCommandBuffer command) {
                    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                      pipeline.pipeline);
                    vkCmdPushConstants(command, pipeline.layout,
                                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                       sizeof(push), &push);
                    vkCmdDispatch(command, dispatch_groups(*context, count), 1, 1);
                });
            uint32_t value = 0;
            context->memory().copy_device_to_host(CopyRequest{
                .src = scratch,
                .dst = raw_storage_ref(&value),
                .bytes = sizeof(value),
                .synchronous = true,
                .operation = "tensor.count.readback",
            });
            return value;
        }
    } // namespace

    float VulkanBackendOps::sum_scalar(const StorageRef input, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(sum_scalar);
        return scalar_reduce(ReduceOp::Sum, input, count);
    }

    float VulkanBackendOps::mean_scalar(const StorageRef input, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(mean_scalar);
        return scalar_reduce(ReduceOp::Mean, input, count);
    }

    float VulkanBackendOps::max_scalar(const StorageRef input, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(max_scalar);
        return scalar_reduce(ReduceOp::Max, input, count);
    }

    float VulkanBackendOps::min_scalar(const StorageRef input, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(min_scalar);
        return scalar_reduce(ReduceOp::Min, input, count);
    }

    void VulkanBackendOps::reduce(
        const StorageRef input, const StorageRef output,
        const StridedLayout& input_layout, const ReduceProgram& program, ExecContext) {
        LFS_FACADE_TRACE(reduce);
        LFS_ASSERT_MSG(program.axis_count <= MAX_TENSOR_RANK && input_layout.rank <= MAX_TENSOR_RANK,
                       "reduction axis count exceeds MAX_TENSOR_RANK");
        LFS_ASSERT_MSG(input.dtype != DataType::Float16,
                       "Vulkan backend: Float16 reduction is not implemented yet");
        LFS_ASSERT_MSG(output.dtype == program.result_dtype,
                       "Vulkan reduction output storage dtype does not match the program");
        if (input_layout.element_count == 0) {
            return;
        }
        const auto context = acquire_vulkan_context();
        const size_t rank = input_layout.rank;
        if (program.axis_count == 0 || program.axis_count == rank) {
            reduce_full(*context, program.op, input, input.dtype, input_layout.element_count,
                        output, program.result_dtype, nullptr);
            return;
        }
        std::array<int, MAX_TENSOR_RANK> axes{};
        std::copy_n(program.axes.begin(), program.axis_count, axes.begin());
        std::sort(axes.begin(), axes.begin() + program.axis_count);
        uint32_t reduced_mask = 0;
        bool contiguous_run = true;
        for (size_t i = 0; i < program.axis_count; ++i) {
            LFS_ASSERT_MSG(axes[i] >= 0 && axes[i] < static_cast<int>(rank),
                           "reduction axis is out of range");
            reduced_mask |= 1u << static_cast<unsigned>(axes[i]);
            contiguous_run = contiguous_run && (i == 0 || axes[i] == axes[i - 1] + 1);
        }
        if (!contiguous_run) {
            reduce_general(*this, *context, program.op, input, input.dtype, output,
                           program.result_dtype, input_layout, reduced_mask);
            return;
        }
        const size_t first = static_cast<size_t>(axes[0]);
        const size_t last = static_cast<size_t>(axes[program.axis_count - 1]);
        size_t outer = 1;
        size_t reduce = 1;
        size_t inner = 1;
        for (size_t axis = 0; axis < rank; ++axis) {
            (axis < first ? outer : axis <= last ? reduce
                                                 : inner) *= input_layout.dims[axis];
        }
        reduce_axes(*context, program.op, input, input.dtype, output, program.result_dtype,
                    outer, reduce, inner, nullptr);
    }

    void VulkanBackendOps::column_reduce(
        const StorageRef input, const StorageRef output, const size_t rows,
        const size_t columns, const ReduceProgram& program, ExecContext) {
        LFS_FACADE_TRACE(column_reduce);
        const auto context = acquire_vulkan_context();
        reduce_axes(*context, program.op, input, DataType::Float32, output, DataType::Float32,
                    1, rows, columns, nullptr);
    }

    void VulkanBackendOps::strided_reduce(
        const StorageRef input, const StorageRef output, const size_t outer_size,
        const size_t reduce_size, const size_t inner_size, const ReduceProgram& program,
        ExecContext) {
        LFS_FACADE_TRACE(strided_reduce);
        const auto context = acquire_vulkan_context();
        reduce_axes(*context, program.op, input, DataType::Float32, output, DataType::Float32,
                    outer_size, reduce_size, inner_size, nullptr);
    }

    void VulkanBackendOps::fused_transform_reduce(
        const StorageRef input, const StorageRef output, const size_t count,
        const tensor_ops::FusedPointwiseOpChain& chain, const ReduceProgram& program,
        const std::span<const StorageRef> rhs_storages, ExecContext) {
        LFS_FACADE_TRACE(fused_transform_reduce);
        if (count == 0) {
            return;
        }
        const auto context = acquire_vulkan_context();
        const Chain descriptors = make_chain(*context, chain, rhs_storages);
        reduce_full(*context, program.op, input, DataType::Float32, count, output,
                    DataType::Float32, &descriptors);
    }

    void VulkanBackendOps::fused_segmented_transform_reduce(
        const StorageRef input, const StorageRef output, const size_t segment_count,
        const size_t segment_size, const tensor_ops::FusedPointwiseOpChain& chain,
        const ReduceProgram& program, const std::span<const StorageRef> rhs_storages,
        ExecContext) {
        LFS_FACADE_TRACE(fused_segmented_transform_reduce);
        if (segment_count == 0 || segment_size == 0) {
            return;
        }
        const auto context = acquire_vulkan_context();
        const Chain descriptors = make_chain(*context, chain, rhs_storages);
        reduce_axes(*context, program.op, input, DataType::Float32, output, DataType::Float32,
                    segment_count, segment_size, 1, &descriptors);
    }

    size_t VulkanBackendOps::count_nonzero_bool(
        const StorageRef input, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(count_nonzero_bool);
        return count_matches(0, input, count);
    }

    size_t VulkanBackendOps::count_nonzero_float(
        const StorageRef input, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(count_nonzero_float);
        return count_matches(1, input, count);
    }

    bool VulkanBackendOps::has_nan(const StorageRef input, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(has_nan);
        return count_matches(2, input, count) != 0;
    }

    bool VulkanBackendOps::has_inf(const StorageRef input, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(has_inf);
        return count_matches(3, input, count) != 0;
    }

    void VulkanBackendOps::cumsum(
        const StorageRef data, const StridedLayout& layout, const int dim, ExecContext) {
        LFS_FACADE_TRACE(cumsum);
        LFS_ASSERT_MSG(data.dtype == DataType::Float32 || data.dtype == DataType::Int32,
                       "Vulkan cumsum supports Float32 and Int32");
        LFS_ASSERT_MSG(dim >= 0 && static_cast<size_t>(dim) < layout.rank,
                       "cumsum dimension is out of range");
        size_t outer = 1;
        size_t inner = 1;
        for (size_t axis = 0; axis < layout.rank; ++axis) {
            if (axis < static_cast<size_t>(dim)) {
                outer *= layout.dims[axis];
            } else if (axis > static_cast<size_t>(dim)) {
                inner *= layout.dims[axis];
            }
        }
        const size_t size = layout.dims[static_cast<size_t>(dim)];
        const size_t lines = outer * inner;
        if (lines == 0 || size == 0) {
            return;
        }
        const auto context = acquire_vulkan_context();
        // Short lines scan one per thread; any longer line gets a workgroup and
        // scans in chunks with a carried total (the line loop covers any count).
        const uint32_t mode = size <= 32 ? 0u : 1u;
        const std::array constants{shader_dtype(data.dtype), mode};
        const VulkanPipeline& pipeline =
            context->pipelines().specialized("scan", sizeof(ScanPush), constants);
        const ScanPush push{
            .data_address = address(data),
            .outer = checked_u32(outer, "Vulkan cumsum outer size exceeds uint32"),
            .dim = checked_u32(size, "Vulkan cumsum size exceeds uint32"),
            .inner = checked_u32(inner, "Vulkan cumsum inner size exceeds uint32"),
            .lines = checked_u32(lines, "Vulkan cumsum line count exceeds uint32"),
        };
        const uint32_t groups =
            mode == 0 ? dispatch_groups(*context, lines)
                      : static_cast<uint32_t>(std::min<size_t>(
                            lines, context->caps().max_workgroup_count[0]));
        const std::array reads{data};
        const std::array writes{data};
        context->recorders().record(
            reads, writes, [&](const VkCommandBuffer command) {
                vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  pipeline.pipeline);
                vkCmdPushConstants(command, pipeline.layout,
                                   VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                   sizeof(push), &push);
                vkCmdDispatch(command, groups, 1, 1);
            });
    }

} // namespace lfs::core::internal
