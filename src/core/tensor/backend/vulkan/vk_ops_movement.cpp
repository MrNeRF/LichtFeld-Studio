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
#include <cstddef>
#include <limits>
#include <span>

namespace lfs::core::internal {
    namespace {
        using vk::address;
        using vk::checked_u32;
        using vk::dispatch_groups;
        using vk::kLocalSize;

        struct StridedPush {
            uint64_t input_address;
            uint64_t output_address;
            std::array<uint64_t, MAX_TENSOR_RANK> dims;
            std::array<uint64_t, MAX_TENSOR_RANK> strides;
            uint32_t rank;
            uint32_t count;
        };
        static_assert(sizeof(StridedPush) == 152);

        void validate_layout(const StridedLayout& layout) {
            LFS_ASSERT_MSG(layout.rank > 0 && layout.rank <= MAX_TENSOR_RANK,
                           "Vulkan strided movement requires rank 1 through MAX_TENSOR_RANK");
            for (size_t i = 0; i < layout.rank; ++i) {
                LFS_ASSERT_MSG(layout.dims[i] > 0,
                               "Vulkan strided movement requires non-zero dimensions");
            }
        }

        size_t required_elements(const StridedLayout& layout) {
            if (layout.element_count == 0) {
                return 0;
            }
            size_t last = 0;
            for (size_t i = 0; i < layout.rank; ++i) {
                if (layout.strides[i] == 0) {
                    continue;
                }
                LFS_ASSERT_MSG(
                    layout.dims[i] - 1 <=
                        (std::numeric_limits<size_t>::max() - last) / layout.strides[i],
                    "Vulkan strided layout extent overflows size_t");
                last += (layout.dims[i] - 1) * layout.strides[i];
            }
            return last + 1;
        }

        void dispatch_strided(const StorageRef input, const StorageRef output,
                              const StridedLayout& layout, const bool scatter,
                              const DataType input_dtype,
                              const DataType output_dtype) {
            if (layout.element_count == 0) {
                return;
            }
            validate_layout(layout);
            StridedPush push{
                .input_address = address(input),
                .output_address = address(output),
                .rank = static_cast<uint32_t>(layout.rank),
                .count = checked_u32(layout.element_count,
                                     "Vulkan strided count exceeds uint32"),
            };
            std::copy_n(layout.dims.begin(), layout.rank, push.dims.begin());
            std::copy_n(layout.strides.begin(), layout.rank, push.strides.begin());
            const std::array constants{static_cast<uint32_t>(input_dtype),
                                       static_cast<uint32_t>(output_dtype),
                                       scatter ? 1u : 0u};
            const auto context = acquire_vulkan_context();
            const VulkanPipeline& pipeline = context->pipelines().specialized(
                "strided_copy", sizeof(StridedPush), constants);
            const std::array reads{input};
            const std::array writes{output};
            context->recorders().record(
                reads, writes, [&](const VkCommandBuffer command) {
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

        struct CatPadPush {
            uint64_t input_address;
            uint64_t output_address;
            std::array<uint32_t, MAX_TENSOR_RANK> input_dims;
            std::array<uint32_t, MAX_TENSOR_RANK> input_strides;
            std::array<uint32_t, MAX_TENSOR_RANK> output_strides;
            std::array<uint32_t, MAX_TENSOR_RANK> pad_before;
            uint32_t count;
            uint32_t rank;
            uint32_t input_block;
            uint32_t output_block;
            uint32_t output_offset;
            uint32_t padding;
        };
        static_assert(sizeof(CatPadPush) == 168);

        std::array<uint32_t, MAX_TENSOR_RANK> narrow_layout_values(
            const std::array<size_t, MAX_TENSOR_RANK>& values,
            const size_t rank, const char* const description) {
            std::array<uint32_t, MAX_TENSOR_RANK> result{};
            for (size_t i = 0; i < rank; ++i) {
                result[i] = checked_u32(values[i], description);
            }
            return result;
        }

        void dispatch_cat_input(const StorageRef input, const StorageRef output,
                                const size_t input_block, const size_t output_block,
                                const size_t output_offset, const size_t count) {
            if (count == 0) {
                return;
            }
            const CatPadPush push{
                .input_address = address(input),
                .output_address = address(output),
                .count = checked_u32(count, "Vulkan cat count exceeds uint32"),
                .input_block = checked_u32(input_block,
                                           "Vulkan cat input block exceeds uint32"),
                .output_block = checked_u32(output_block,
                                            "Vulkan cat output block exceeds uint32"),
                .output_offset = checked_u32(output_offset,
                                             "Vulkan cat output offset exceeds uint32"),
            };
            const std::array constants{static_cast<uint32_t>(input.dtype), 0u};
            const auto context = acquire_vulkan_context();
            const VulkanPipeline& pipeline = context->pipelines().specialized(
                "cat_pad", sizeof(CatPadPush), constants);
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
    } // namespace

    void VulkanBackendOps::strided_copy(
        const StorageRef input, const StorageRef output,
        const StridedLayout& input_layout, ExecContext) {
        LFS_FACADE_TRACE(strided_copy);
        LFS_ASSERT_MSG(input.dtype == output.dtype,
                       "Vulkan strided copy requires matching dtypes");
        dispatch_strided(input, output, input_layout, false,
                         input.dtype, output.dtype);
    }

    void VulkanBackendOps::strided_copy_immediate(
        const StorageRef input, const StorageRef output,
        const StridedLayout& input_layout, ExecContext context) {
        LFS_FACADE_TRACE(strided_copy_immediate);
        strided_copy(input, output, input_layout, context);
    }

    void VulkanBackendOps::strided_upload(
        const StorageRef host_input, const StorageRef output,
        const StridedLayout& input_layout, ExecContext) {
        LFS_FACADE_TRACE(strided_upload);
        if (input_layout.element_count == 0) {
            return;
        }
        validate_layout(input_layout);
        const size_t source_elements = required_elements(input_layout);
        LFS_ASSERT_MSG(source_elements <=
                           std::numeric_limits<size_t>::max() / dtype_size(output.dtype),
                       "Vulkan strided upload byte size overflows size_t");
        const size_t source_bytes = source_elements * dtype_size(output.dtype);
        const auto context = acquire_vulkan_context();
        StorageRef staging = context->memory().allocate(source_bytes, 16, {});
        staging.dtype = output.dtype;
        context->memory().copy_host_to_device(CopyRequest{
            .src = raw_storage_ref(
                static_cast<std::byte*>(host_input.data) + host_input.byte_offset,
                output.dtype),
            .dst = staging,
            .bytes = source_bytes,
            .synchronous = false,
        });
        dispatch_strided(staging, output, input_layout, false,
                         output.dtype, output.dtype);
        context->memory().deallocate(staging);
    }

    void VulkanBackendOps::strided_scatter(
        const StorageRef input, const StorageRef output,
        const StridedLayout& output_layout, ExecContext) {
        LFS_FACADE_TRACE(strided_scatter);
        LFS_ASSERT_MSG(input.dtype == output.dtype,
                       "Vulkan strided scatter requires matching dtypes");
        dispatch_strided(input, output, output_layout, true,
                         input.dtype, output.dtype);
    }

    void VulkanBackendOps::strided_scatter_immediate(
        const StorageRef input, const StorageRef output,
        const StridedLayout& output_layout, ExecContext context) {
        LFS_FACADE_TRACE(strided_scatter_immediate);
        strided_scatter(input, output, output_layout, context);
    }

    void VulkanBackendOps::strided_scatter_int32_to_float32(
        const StorageRef input, const StorageRef output,
        const StridedLayout& output_layout, ExecContext) {
        LFS_FACADE_TRACE(strided_scatter_int32_to_float32);
        LFS_ASSERT_MSG(input.dtype == DataType::Int32 &&
                           output.dtype == DataType::Float32 &&
                           output_layout.rank == 2,
                       "Vulkan fused Int32 to Float32 scatter requires rank 2");
        dispatch_strided(input, output, output_layout, true,
                         DataType::Int32, DataType::Float32);
    }

    void VulkanBackendOps::cat_last_dim(
        const StorageRef output, const std::span<const StorageRef> inputs,
        const std::span<const StridedLayout> layouts, const size_t num_rows,
        const size_t row_size, const size_t element_size, ExecContext) {
        LFS_FACADE_TRACE(cat_last_dim);
        LFS_ASSERT_MSG(inputs.size() == layouts.size() && !inputs.empty(),
                       "Vulkan cat requires matching non-empty input metadata");
        LFS_ASSERT_MSG(element_size == dtype_size(output.dtype),
                       "Vulkan cat element size does not match output dtype");
        size_t output_offset = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            LFS_ASSERT_MSG(inputs[i].dtype == output.dtype && layouts[i].rank > 0,
                           "Vulkan cat input dtype or rank is invalid");
            const size_t input_block = layouts[i].dims[layouts[i].rank - 1];
            dispatch_cat_input(inputs[i], output, input_block, row_size,
                               output_offset, num_rows * input_block);
            output_offset += input_block;
        }
        LFS_ASSERT_MSG(output_offset == row_size,
                       "Vulkan last-dimension cat widths do not sum to output width");
    }

    void VulkanBackendOps::cat_middle_dim(
        const StorageRef output, const std::span<const StorageRef> inputs,
        const std::span<const StridedLayout> layouts, const size_t outer_size,
        const size_t inner_size, const int dim, const size_t element_size,
        ExecContext) {
        LFS_FACADE_TRACE(cat_middle_dim);
        LFS_ASSERT_MSG(inputs.size() == layouts.size() && !inputs.empty(),
                       "Vulkan cat requires matching non-empty input metadata");
        LFS_ASSERT_MSG(dim >= 0 && element_size == dtype_size(output.dtype),
                       "Vulkan middle-dimension cat metadata is invalid");
        size_t total_dimension = 0;
        for (const StridedLayout& layout : layouts) {
            LFS_ASSERT_MSG(static_cast<size_t>(dim) < layout.rank,
                           "Vulkan cat dimension is outside an input rank");
            total_dimension += layout.dims[dim];
        }
        size_t dimension_offset = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            LFS_ASSERT_MSG(inputs[i].dtype == output.dtype,
                           "Vulkan cat inputs must match output dtype");
            const size_t input_dimension = layouts[i].dims[dim];
            const size_t input_block = input_dimension * inner_size;
            dispatch_cat_input(inputs[i], output, input_block,
                               total_dimension * inner_size,
                               dimension_offset * inner_size,
                               outer_size * input_block);
            dimension_offset += input_dimension;
        }
    }

    void VulkanBackendOps::pad(
        const StorageRef input, const StorageRef output,
        const StridedLayout& input_layout,
        const StridedLayout& output_layout,
        const std::array<size_t, MAX_TENSOR_RANK>& pad_before,
        ExecContext) {
        LFS_FACADE_TRACE(pad);
        if (input_layout.element_count == 0) {
            return;
        }
        LFS_ASSERT_MSG(input.dtype == DataType::Float32 &&
                           output.dtype == DataType::Float32 &&
                           input_layout.rank == output_layout.rank &&
                           input_layout.rank <= MAX_TENSOR_RANK,
                       "Vulkan pad requires matching rank Float32 tensors");
        const CatPadPush push{
            .input_address = address(input),
            .output_address = address(output),
            .input_dims = narrow_layout_values(
                input_layout.dims, input_layout.rank,
                "Vulkan pad input dimension exceeds uint32"),
            .input_strides = narrow_layout_values(
                input_layout.strides, input_layout.rank,
                "Vulkan pad input stride exceeds uint32"),
            .output_strides = narrow_layout_values(
                output_layout.strides, output_layout.rank,
                "Vulkan pad output stride exceeds uint32"),
            .pad_before = narrow_layout_values(
                pad_before, input_layout.rank,
                "Vulkan pad width exceeds uint32"),
            .count = checked_u32(input_layout.element_count,
                                 "Vulkan pad count exceeds uint32"),
            .rank = static_cast<uint32_t>(input_layout.rank),
        };
        const std::array constants{static_cast<uint32_t>(input.dtype), 1u};
        const auto context = acquire_vulkan_context();
        const VulkanPipeline& pipeline = context->pipelines().specialized(
            "cat_pad", sizeof(CatPadPush), constants);
        const std::array reads{input};
        const std::array writes{output};
        context->recorders().record(
            reads, writes, [&](const VkCommandBuffer command) {
                vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  pipeline.pipeline);
                vkCmdPushConstants(command, pipeline.layout,
                                   VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                   sizeof(push), &push);
                vkCmdDispatch(command,
                              dispatch_groups(*context, input_layout.element_count),
                              1, 1);
            });
    }

} // namespace lfs::core::internal
