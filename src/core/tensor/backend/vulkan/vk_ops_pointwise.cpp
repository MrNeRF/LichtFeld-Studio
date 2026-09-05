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
#include <bit>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>

namespace lfs::core::internal {
    namespace {
        using vk::address;
        using vk::checked_u32;
        using vk::dispatch_groups;
        using vk::kLocalSize;

        uint64_t scalar_integer(const ScalarOperand scalar) {
            switch (scalar.kind) {
            case ScalarKind::Float: return 0;
            case ScalarKind::Int32:
                return static_cast<uint64_t>(static_cast<int64_t>(scalar.value.int32_value));
            case ScalarKind::Int64: return static_cast<uint64_t>(scalar.value.int64_value);
            case ScalarKind::Bool: return scalar.value.bool_value ? 1 : 0;
            }
            return 0;
        }

        float scalar_float(const ScalarOperand scalar) {
            switch (scalar.kind) {
            case ScalarKind::Float: return scalar.value.float_value;
            case ScalarKind::Int32: return static_cast<float>(scalar.value.int32_value);
            case ScalarKind::Int64: return static_cast<float>(scalar.value.int64_value);
            case ScalarKind::Bool: return scalar.value.bool_value ? 1.0f : 0.0f;
            }
            return 0.0f;
        }

        struct PointwisePush {
            uint64_t lhs_address;
            uint64_t rhs_address;
            uint64_t output_address;
            uint64_t scalar_int64;
            float scalar_float;
            uint32_t count;
            uint32_t scalar_kind;
            uint32_t flags;
        };
        static_assert(sizeof(PointwisePush) == 48);

        bool force_fp32_half() {
            const char* const value = std::getenv("LFS_VULKAN_FORCE_FP32_HALF");
            return value != nullptr && std::strcmp(value, "1") == 0;
        }

        void dispatch_pointwise(const PointwiseProgram& program,
                                const StorageRef lhs, const StorageRef rhs,
                                const StorageRef output, const size_t count,
                                const uint32_t arity) {
            if (count == 0) {
                return;
            }
            LFS_ASSERT_MSG(lhs.dtype == program.in_dtype &&
                               output.dtype == program.out_dtype,
                           "Vulkan pointwise program dtype does not match storage");
            if (arity == 2) {
                LFS_ASSERT_MSG(rhs.dtype == program.in_dtype,
                               "Vulkan pointwise rhs dtype does not match program");
            }
            const uint64_t lhs_address = address(lhs);
            const uint64_t rhs_address = arity == 2 ? address(rhs) : 0;
            const uint64_t output_address = address(output);
            const bool vectorized = program.in_dtype == DataType::Float32 &&
                                    count % 4 == 0 &&
                                    (lhs_address & 15u) == 0 &&
                                    (output_address & 15u) == 0 &&
                                    (arity != 2 || (rhs_address & 15u) == 0) &&
                                    program.scalar.scalar_on_right;
            const auto context = acquire_vulkan_context();
            const bool native_half =
                program.in_dtype == DataType::Float16 &&
                program.out_dtype == DataType::Float16 && arity == 2 &&
                static_cast<uint32_t>(program.op) >= 4u &&
                static_cast<uint32_t>(program.op) <= 7u &&
                context->caps().shader_float16 && !force_fp32_half();
            const std::array constants{
                static_cast<uint32_t>(program.op),
                static_cast<uint32_t>(program.in_dtype),
                static_cast<uint32_t>(program.out_dtype),
                arity,
                vectorized ? 1u : 0u,
                native_half ? 1u : 0u,
            };
            const VulkanPipeline& pipeline = context->pipelines().specialized(
                native_half ? "pointwise_half" : "pointwise",
                sizeof(PointwisePush), constants);
            // The kernel maps one invocation to one element (or one float4 group), so
            // a count above the device's group limit is dispatched in chunks with the
            // operand addresses advanced per chunk. Chunk boundaries are multiples of
            // four elements and of four bytes for every dtype.
            const uint64_t chunk_elements =
                static_cast<uint64_t>(context->caps().max_workgroup_count[0]) * kLocalSize *
                (vectorized ? 4u : 1u);
            const size_t in_bytes = dtype_size(program.in_dtype);
            const size_t out_bytes = dtype_size(program.out_dtype);
            std::array<StorageRef, 2> reads{lhs, rhs};
            const size_t read_count = arity == 2 ? 2 : 1;
            const std::array writes{output};
            for (size_t offset = 0; offset < count; offset += chunk_elements) {
                const size_t chunk = std::min<size_t>(count - offset, chunk_elements);
                const PointwisePush push{
                    .lhs_address = lhs_address + offset * in_bytes,
                    .rhs_address = arity == 2 ? rhs_address + offset * in_bytes : rhs_address,
                    .output_address = output_address + offset * out_bytes,
                    .scalar_int64 = scalar_integer(program.scalar),
                    .scalar_float = scalar_float(program.scalar),
                    .count = checked_u32(chunk, "Vulkan pointwise count exceeds uint32"),
                    .scalar_kind = static_cast<uint32_t>(program.scalar.kind),
                    .flags = program.scalar.scalar_on_right ? 1u : 0u,
                };
                context->recorders().record(
                    std::span<const StorageRef>(reads.data(), read_count), writes,
                    [&](const VkCommandBuffer command) {
                        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                          pipeline.pipeline);
                        vkCmdPushConstants(command, pipeline.layout,
                                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                           sizeof(push), &push);
                        vkCmdDispatch(command,
                                      dispatch_groups(*context,
                                                      vectorized ? chunk / 4 : chunk),
                                      1, 1);
                    });
            }
        }

        struct BroadcastPush {
            uint64_t lhs_address;
            uint64_t rhs_address;
            uint64_t output_address;
            std::array<uint32_t, MAX_TENSOR_RANK> lhs_dims;
            std::array<uint32_t, MAX_TENSOR_RANK> rhs_dims;
            std::array<uint32_t, MAX_TENSOR_RANK> output_dims;
            uint32_t lhs_rank;
            uint32_t rhs_rank;
            uint32_t output_rank;
            uint32_t count;
        };
        static_assert(sizeof(BroadcastPush) == 136);

        std::array<uint32_t, MAX_TENSOR_RANK> shader_dims(
            const StridedLayout& layout) {
            LFS_ASSERT_MSG(layout.rank <= MAX_TENSOR_RANK,
                           "Vulkan layout rank exceeds MAX_TENSOR_RANK");
            std::array<uint32_t, MAX_TENSOR_RANK> result{};
            for (size_t i = 0; i < layout.rank; ++i) {
                result[i] = checked_u32(layout.dims[i],
                                        "Vulkan dimension exceeds uint32");
            }
            return result;
        }

        uint64_t fill_pattern(const DataType dtype, const ScalarOperand value) {
            uint64_t pattern = 0;
            switch (dtype) {
            case DataType::Float32: {
                const float converted = scalar_float(value);
                std::memcpy(&pattern, &converted, sizeof(converted));
                return pattern;
            }
            case DataType::Float16: {
                const __half converted = __float2half_rn(scalar_float(value));
                std::memcpy(&pattern, &converted, sizeof(converted));
                return pattern;
            }
            case DataType::Int32: {
                const int32_t converted = static_cast<int32_t>(scalar_integer(value));
                std::memcpy(&pattern, &converted, sizeof(converted));
                return pattern;
            }
            case DataType::UInt32: {
                const uint32_t converted = static_cast<uint32_t>(scalar_integer(value));
                std::memcpy(&pattern, &converted, sizeof(converted));
                return pattern;
            }
            case DataType::Int64: return scalar_integer(value);
            case DataType::Bool: return scalar_float(value) != 0.0f ? 1 : 0;
            case DataType::UInt8: return static_cast<uint8_t>(scalar_integer(value));
            }
            return 0;
        }

        struct FillPush {
            uint64_t output_address;
            uint64_t pattern;
            std::array<uint64_t, MAX_TENSOR_RANK> dims;
            std::array<uint64_t, MAX_TENSOR_RANK> strides;
            uint32_t count;
            uint32_t rank;
            uint32_t strided;
            uint32_t padding;
        };
        static_assert(sizeof(FillPush) == 160);

        bool is_contiguous(const StridedLayout& layout) {
            size_t expected = 1;
            for (size_t reverse = 0; reverse < layout.rank; ++reverse) {
                const size_t dimension = layout.rank - reverse - 1;
                if (layout.dims[dimension] > 1 &&
                    layout.strides[dimension] != expected) {
                    return false;
                }
                expected *= layout.dims[dimension];
            }
            return true;
        }

        void dispatch_fill(const StorageRef output, const StridedLayout& layout,
                           const ScalarOperand value) {
            if (layout.element_count == 0) {
                return;
            }
            LFS_ASSERT_MSG(layout.rank <= MAX_TENSOR_RANK,
                           "Vulkan strided fill rank exceeds MAX_TENSOR_RANK");
            FillPush push{
                .output_address = address(output),
                .pattern = fill_pattern(output.dtype, value),
                .count = checked_u32(layout.element_count,
                                     "Vulkan fill count exceeds uint32"),
                .rank = static_cast<uint32_t>(layout.rank),
                .strided = is_contiguous(layout) ? 0u : 1u,
            };
            std::copy_n(layout.dims.begin(), layout.rank, push.dims.begin());
            std::copy_n(layout.strides.begin(), layout.rank, push.strides.begin());
            const std::array constants{static_cast<uint32_t>(output.dtype)};
            const auto context = acquire_vulkan_context();
            const VulkanPipeline& pipeline =
                context->pipelines().specialized("fill", sizeof(FillPush), constants);
            const std::array writes{output};
            context->recorders().record(
                {}, writes, [&](const VkCommandBuffer command) {
                    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                      pipeline.pipeline);
                    vkCmdPushConstants(command, pipeline.layout,
                                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                       sizeof(push), &push);
                    vkCmdDispatch(command,
                                  dispatch_groups(*context, layout.element_count),
                                  1, 1);
                });
        }

        bool conversion_supported(const DataType input, const DataType output) {
            if (input == output) {
                return input == DataType::Bool;
            }
            switch (input) {
            case DataType::Float32:
                return output == DataType::Float16 || output == DataType::Int32 ||
                       output == DataType::Int64 || output == DataType::UInt8 ||
                       output == DataType::UInt32;
            case DataType::Float16:
                return output == DataType::Float32 || output == DataType::Int32 ||
                       output == DataType::Int64 || output == DataType::UInt8;
            case DataType::Int32:
                return output == DataType::Float32 || output == DataType::Float16 ||
                       output == DataType::Int64 || output == DataType::UInt8;
            case DataType::Int64:
                return output == DataType::Float32 || output == DataType::Float16 ||
                       output == DataType::Int32 || output == DataType::UInt8;
            case DataType::UInt8:
            case DataType::Bool:
                return output == DataType::Float32 || output == DataType::Float16 ||
                       output == DataType::Int32 || output == DataType::Int64 ||
                       output == DataType::Bool;
            case DataType::UInt32:
                return output == DataType::Float32 || output == DataType::Int64;
            }
            return false;
        }

        struct ConvertPush {
            uint64_t input_address;
            uint64_t output_address;
            uint32_t count;
            uint32_t padding;
        };
        static_assert(sizeof(ConvertPush) == 24);
    } // namespace

    void VulkanBackendOps::unary(const PointwiseProgram& program,
                                 const StorageRef input, const StorageRef output,
                                 const size_t count, ExecContext) {
        LFS_FACADE_TRACE(unary);
        dispatch_pointwise(program, input, {}, output, count, 1);
    }

    void VulkanBackendOps::binary(const PointwiseProgram& program,
                                  const StorageRef lhs, const StorageRef rhs,
                                  const StorageRef output, const size_t count,
                                  ExecContext) {
        LFS_FACADE_TRACE(binary);
        dispatch_pointwise(program, lhs, rhs, output, count, 2);
    }

    void VulkanBackendOps::scalar(const PointwiseProgram& program,
                                  const StorageRef input, const StorageRef output,
                                  const size_t count, ExecContext) {
        LFS_FACADE_TRACE(scalar);
        dispatch_pointwise(program, input, {}, output, count, 3);
    }

    void VulkanBackendOps::broadcast_binary(
        const PointwiseProgram& program,
        const StorageRef lhs, const StridedLayout& lhs_layout,
        const StorageRef rhs, const StridedLayout& rhs_layout,
        const StorageRef output, const StridedLayout& output_layout,
        ExecContext) {
        LFS_FACADE_TRACE(broadcast_binary);
        if (output_layout.element_count == 0) {
            return;
        }
        LFS_ASSERT_MSG(lhs.dtype == program.in_dtype && rhs.dtype == program.in_dtype &&
                           output.dtype == program.out_dtype,
                       "Vulkan broadcast dtype does not match pointwise program");
        BroadcastPush push{
            .lhs_address = address(lhs),
            .rhs_address = address(rhs),
            .output_address = address(output),
            .lhs_dims = shader_dims(lhs_layout),
            .rhs_dims = shader_dims(rhs_layout),
            .output_dims = shader_dims(output_layout),
            .lhs_rank = static_cast<uint32_t>(lhs_layout.rank),
            .rhs_rank = static_cast<uint32_t>(rhs_layout.rank),
            .output_rank = static_cast<uint32_t>(output_layout.rank),
            .count = checked_u32(output_layout.element_count,
                                 "Vulkan broadcast count exceeds uint32"),
        };
        const std::array constants{
            static_cast<uint32_t>(program.op),
            static_cast<uint32_t>(program.in_dtype),
            static_cast<uint32_t>(program.out_dtype),
        };
        const auto context = acquire_vulkan_context();
        const VulkanPipeline& pipeline =
            context->pipelines().specialized("broadcast", sizeof(BroadcastPush), constants);
        const std::array reads{lhs, rhs};
        const std::array writes{output};
        context->recorders().record(
            reads, writes, [&](const VkCommandBuffer command) {
                vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  pipeline.pipeline);
                vkCmdPushConstants(command, pipeline.layout,
                                   VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                   sizeof(push), &push);
                vkCmdDispatch(command,
                              dispatch_groups(*context, output_layout.element_count),
                              1, 1);
            });
    }

    void VulkanBackendOps::fused_pointwise_chain(
        const StorageRef input, const StorageRef output, const size_t count,
        const tensor_ops::FusedPointwiseOpChain& chain,
        const std::span<const StorageRef> rhs_storages, ExecContext) {
        LFS_FACADE_TRACE(fused_pointwise_chain);
        if (count == 0) {
            return;
        }
        LFS_ASSERT_MSG(input.dtype == DataType::Float32 &&
                           output.dtype == DataType::Float32 && chain.num_ops > 0 &&
                           chain.num_ops <= tensor_ops::FUSED_POINTWISE_MAX_OPS,
                       "Vulkan fused pointwise chain requires Float32 and 1 to 16 operations");
        const auto context = acquire_vulkan_context();
        const StorageRef metadata = vk::upload_chain(*context, chain);
        const std::array constants{0u, 0u, 0u, 4u, 0u, 0u};
        const VulkanPipeline& pipeline =
            context->pipelines().specialized("pointwise", sizeof(PointwisePush), constants);
        // Every rhs operand is listed as a read so the recorder flushes its producer
        // and stamps its last use; the chain descriptor only carries its address.
        // Tensor rhs operands are indexed by the element's global index in the shader,
        // so the chain is dispatched in chunks with the input and output advanced but
        // the descriptor table unchanged; rhs addresses are absolute per element.
        std::vector<StorageRef> reads{input, metadata};
        reads.insert(reads.end(), rhs_storages.begin(), rhs_storages.end());
        const std::array writes{output};
        const uint64_t chunk_elements =
            static_cast<uint64_t>(context->caps().max_workgroup_count[0]) * kLocalSize;
        for (size_t offset = 0; offset < count; offset += chunk_elements) {
            const size_t chunk = std::min<size_t>(count - offset, chunk_elements);
            const PointwisePush push{
                .lhs_address = address(input) + offset * sizeof(float),
                .rhs_address = address(metadata),
                .output_address = address(output) + offset * sizeof(float),
                .count = checked_u32(chunk, "Vulkan fused pointwise count exceeds uint32"),
                .scalar_kind = static_cast<uint32_t>(chain.num_ops),
                .flags = checked_u32(offset, "Vulkan fused pointwise offset exceeds uint32"),
            };
            context->recorders().record(
                reads, writes, [&](const VkCommandBuffer command) {
                    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                      pipeline.pipeline);
                    vkCmdPushConstants(command, pipeline.layout,
                                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                       sizeof(push), &push);
                    vkCmdDispatch(command, dispatch_groups(*context, chunk), 1, 1);
                });
        }
        context->memory().deallocate(metadata);
    }

    void VulkanBackendOps::clamp_scalar(
        const StorageRef data, const ScalarOperand minimum,
        const ScalarOperand maximum, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(clamp_scalar);
        clamp_fused(data, data, minimum, maximum, count, {});
    }

    void VulkanBackendOps::clamp_fused(
        const StorageRef input, const StorageRef output,
        const ScalarOperand minimum, const ScalarOperand maximum,
        const size_t count, ExecContext) {
        LFS_FACADE_TRACE(clamp_fused);
        if (count == 0) {
            return;
        }
        LFS_ASSERT_MSG(input.dtype == DataType::Float32 &&
                           output.dtype == DataType::Float32 &&
                           minimum.kind == ScalarKind::Float &&
                           maximum.kind == ScalarKind::Float,
                       "Vulkan floating clamp requires Float32 operands");
        const std::array constants{0u, 0u, 0u, 5u, 0u, 0u};
        const auto context = acquire_vulkan_context();
        const VulkanPipeline& pipeline =
            context->pipelines().specialized("pointwise", sizeof(PointwisePush), constants);
        const PointwisePush push{
            .lhs_address = address(input),
            .output_address = address(output),
            .scalar_int64 = std::bit_cast<uint32_t>(maximum.value.float_value),
            .scalar_float = minimum.value.float_value,
            .count = checked_u32(count, "Vulkan clamp count exceeds uint32"),
        };
        const std::array reads{input};
        const std::array writes{output};
        context->recorders().record(
            reads, writes, [&](const VkCommandBuffer command) {
                vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  pipeline.pipeline);
                vkCmdPushConstants(command, pipeline.layout,
                                   VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                   sizeof(push), &push);
                vkCmdDispatch(command, dispatch_groups(*context, count), 1, 1);
            });
    }

    void VulkanBackendOps::clamp_scalar_int(
        const StorageRef data, const ScalarOperand minimum,
        const ScalarOperand maximum, const size_t count, ExecContext) {
        LFS_FACADE_TRACE(clamp_scalar_int);
        if (count == 0) {
            return;
        }
        LFS_ASSERT_MSG(data.dtype == DataType::Int32 &&
                           minimum.kind == ScalarKind::Int32 &&
                           maximum.kind == ScalarKind::Int32,
                       "Vulkan integer clamp requires Int32 operands");
        const std::array constants{0u, 2u, 2u, 5u, 0u, 0u};
        const auto context = acquire_vulkan_context();
        const VulkanPipeline& pipeline =
            context->pipelines().specialized("pointwise", sizeof(PointwisePush), constants);
        const uint64_t bounds =
            static_cast<uint32_t>(minimum.value.int32_value) |
            (static_cast<uint64_t>(static_cast<uint32_t>(maximum.value.int32_value)) << 32);
        const PointwisePush push{
            .lhs_address = address(data),
            .output_address = address(data),
            .scalar_int64 = bounds,
            .count = checked_u32(count, "Vulkan integer clamp count exceeds uint32"),
        };
        const std::array reads{data};
        const std::array writes{data};
        context->recorders().record(
            reads, writes, [&](const VkCommandBuffer command) {
                vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  pipeline.pipeline);
                vkCmdPushConstants(command, pipeline.layout,
                                   VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                   sizeof(push), &push);
                vkCmdDispatch(command, dispatch_groups(*context, count), 1, 1);
            });
    }

    void VulkanBackendOps::convert_type(const StorageRef input,
                                        const StorageRef output,
                                        const size_t count, ExecContext) {
        LFS_FACADE_TRACE(convert_type);
        if (count == 0) {
            return;
        }
        LFS_ASSERT_MSG(conversion_supported(input.dtype, output.dtype),
                       "Vulkan dtype conversion pair has no CUDA instantiation");
        const std::array constants{static_cast<uint32_t>(input.dtype),
                                   static_cast<uint32_t>(output.dtype)};
        const auto context = acquire_vulkan_context();
        const VulkanPipeline& pipeline =
            context->pipelines().specialized("convert", sizeof(ConvertPush), constants);
        const ConvertPush push{
            .input_address = address(input),
            .output_address = address(output),
            .count = checked_u32(count, "Vulkan conversion count exceeds uint32"),
        };
        const std::array reads{input};
        const std::array writes{output};
        context->recorders().record(
            reads, writes, [&](const VkCommandBuffer command) {
                vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  pipeline.pipeline);
                vkCmdPushConstants(command, pipeline.layout,
                                   VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                   sizeof(push), &push);
                vkCmdDispatch(command, dispatch_groups(*context, count), 1, 1);
            });
    }

    void VulkanBackendOps::fill_strided(const StorageRef output,
                                        const StridedLayout& layout,
                                        const ScalarOperand value, ExecContext) {
        LFS_FACADE_TRACE(fill_strided);
        dispatch_fill(output, layout, value);
    }

    void VulkanBackendOps::load_fill(const StorageRef output, const size_t count,
                                     const ScalarOperand value, ExecContext) {
        LFS_FACADE_TRACE(load_fill);
        StridedLayout layout{};
        layout.rank = 1;
        layout.dims[0] = count;
        layout.strides[0] = 1;
        layout.element_count = count;
        dispatch_fill(output, layout, value);
    }

} // namespace lfs::core::internal
