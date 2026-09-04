/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vk_backend_ops.hpp"

#include "../../internal/tensor_impl.hpp"
#include "core/assert.hpp"
#include "vk_context.hpp"
#include "vk_memory.hpp"
#include "vk_pipelines.hpp"
#include "vk_recorder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <utility>
#include <vector>

namespace lfs::core::internal {
    namespace {
        [[noreturn]] void not_implemented(const char* const entry) {
            LFS_ASSERT_MSG(false,
                           std::format("Vulkan backend: {} is not implemented yet", entry));
            std::unreachable();
        }

        bool is_contiguous_layout(const StridedLayout& layout) {
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

        size_t logical_offset(const StridedLayout& layout, size_t linear) {
            size_t offset = 0;
            for (size_t reverse = 0; reverse < layout.rank; ++reverse) {
                const size_t dimension = layout.rank - reverse - 1;
                const size_t coordinate = layout.dims[dimension] == 0
                                              ? 0
                                              : linear % layout.dims[dimension];
                if (layout.dims[dimension] != 0) {
                    linear /= layout.dims[dimension];
                }
                offset += coordinate * layout.strides[dimension];
            }
            return offset;
        }

        size_t required_source_bytes(const StorageRef storage,
                                     const StridedLayout& layout) {
            size_t maximum_element = 0;
            for (size_t dimension = 0; dimension < layout.rank; ++dimension) {
                if (layout.dims[dimension] != 0) {
                    maximum_element +=
                        (layout.dims[dimension] - 1) * layout.strides[dimension];
                }
            }
            return storage.byte_offset +
                   (layout.element_count == 0 ? 0
                                              : (maximum_element + 1) * dtype_size(storage.dtype));
        }

        double scalar_as_double(const ScalarOperand value) {
            switch (value.kind) {
            case ScalarKind::Float: return value.value.float_value;
            case ScalarKind::Int32: return value.value.int32_value;
            case ScalarKind::Int64: return static_cast<double>(value.value.int64_value);
            case ScalarKind::Bool: return value.value.bool_value ? 1.0 : 0.0;
            }
            return 0.0;
        }

        uint64_t fill_pattern(const DataType dtype, const ScalarOperand value) {
            uint64_t pattern = 0;
            const double number = scalar_as_double(value);
            switch (dtype) {
            case DataType::Float32: {
                const float converted = static_cast<float>(number);
                std::memcpy(&pattern, &converted, sizeof(converted));
                break;
            }
            case DataType::Float16: {
                const __half converted = __float2half(static_cast<float>(number));
                std::memcpy(&pattern, &converted, sizeof(converted));
                break;
            }
            case DataType::Int32: {
                const int32_t converted = static_cast<int32_t>(number);
                std::memcpy(&pattern, &converted, sizeof(converted));
                break;
            }
            case DataType::UInt32: {
                const uint32_t converted = static_cast<uint32_t>(number);
                std::memcpy(&pattern, &converted, sizeof(converted));
                break;
            }
            case DataType::Int64: {
                const int64_t converted = static_cast<int64_t>(number);
                std::memcpy(&pattern, &converted, sizeof(converted));
                break;
            }
            case DataType::Bool: {
                const uint8_t converted = number != 0.0 ? 1 : 0;
                std::memcpy(&pattern, &converted, sizeof(converted));
                break;
            }
            case DataType::UInt8: {
                const uint8_t converted = static_cast<uint8_t>(number);
                std::memcpy(&pattern, &converted, sizeof(converted));
                break;
            }
            }
            const size_t element_size = dtype_size(dtype);
            if (element_size == 1) {
                pattern *= 0x0101010101010101ull;
            } else if (element_size == 2) {
                pattern |= pattern << 16;
                pattern |= pattern << 32;
            } else if (element_size == 4) {
                pattern |= pattern << 32;
            }
            return pattern;
        }

        struct FillPush {
            uint64_t output_address;
            uint64_t pattern;
            uint32_t byte_count;
            uint32_t element_size;
        };
        static_assert(sizeof(FillPush) == 24);

        void fill_contiguous(const StorageRef output, const size_t count,
                             const ScalarOperand value) {
            if (count == 0) {
                return;
            }
            const size_t element_size = dtype_size(output.dtype);
            const size_t byte_count = count * element_size;
            LFS_ASSERT_MSG(byte_count <= std::numeric_limits<uint32_t>::max(),
                           "Vulkan fill kernel byte count exceeds its phase-P3 limit");
            const auto context = acquire_vulkan_context();
            const uint64_t address = output.meta->gpu_descriptor.base_address +
                                     output.byte_offset;
            if ((address & 3u) != 0 || (byte_count & 3u) != 0) {
                const uint64_t pattern = fill_pattern(output.dtype, value);
                std::vector<std::byte> host(byte_count);
                for (size_t offset = 0; offset < byte_count;
                     offset += element_size) {
                    std::memcpy(host.data() + offset, &pattern, element_size);
                }
                context->memory().copy_host_to_device(CopyRequest{
                    .src = raw_storage_ref(host.data(), output.dtype),
                    .dst = output,
                    .bytes = byte_count,
                    .synchronous = true,
                });
                return;
            }
            const VulkanPipeline& pipeline = context->pipelines().fill();
            const FillPush push{
                .output_address = address,
                .pattern = fill_pattern(output.dtype, value),
                .byte_count = static_cast<uint32_t>(byte_count),
                .element_size = static_cast<uint32_t>(element_size),
            };
            const std::array writes{output};
            context->recorders().record(
                {}, writes, [&](const VkCommandBuffer command) {
                    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                      pipeline.pipeline);
                    vkCmdPushConstants(command, pipeline.layout,
                                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                       sizeof(push), &push);
                    vkCmdDispatch(command,
                                  ((push.byte_count / 4) + pipeline.local_size_x - 1) /
                                      pipeline.local_size_x,
                                  1, 1);
                });
        }

        void copy_strided_vulkan(const StorageRef input, const StorageRef output,
                                 const StridedLayout& layout) {
            const size_t element_size = dtype_size(input.dtype);
            const size_t output_bytes = layout.element_count * element_size;
            auto context = acquire_vulkan_context();
            if (is_contiguous_layout(layout)) {
                context->memory().copy_device_to_device(CopyRequest{
                    .src = input,
                    .dst = output,
                    .bytes = output_bytes,
                    .synchronous = false,
                });
                return;
            }
            const size_t source_bytes = required_source_bytes(input, layout);
            LFS_ASSERT_MSG(input.meta != nullptr &&
                               source_bytes <= input.meta->gpu_descriptor.byte_size,
                           "Vulkan strided copy reads outside the source allocation");
            std::vector<std::byte> source(source_bytes);
            StorageRef source_base = input;
            source_base.byte_offset = 0;
            context->memory().copy_device_to_host(CopyRequest{
                .src = source_base,
                .dst = raw_storage_ref(source.data()),
                .bytes = source_bytes,
                .synchronous = true,
            });
            std::vector<std::byte> dense(output_bytes);
            for (size_t index = 0; index < layout.element_count; ++index) {
                const size_t source_offset =
                    input.byte_offset + logical_offset(layout, index) * element_size;
                std::memcpy(dense.data() + index * element_size,
                            source.data() + source_offset, element_size);
            }
            context->memory().copy_host_to_device(CopyRequest{
                .src = raw_storage_ref(dense.data(), input.dtype),
                .dst = output,
                .bytes = output_bytes,
                .synchronous = true,
            });
        }
    } // namespace

#define LFS_VK_NOTIMPL_VOID(Name, Parameters) \
    void VulkanBackendOps::Name Parameters { not_implemented(#Name); }
#define LFS_VK_NOTIMPL_FLOAT(Name, Parameters) \
    float VulkanBackendOps::Name Parameters { not_implemented(#Name); }
#define LFS_VK_NOTIMPL_SIZE(Name, Parameters) \
    size_t VulkanBackendOps::Name Parameters { not_implemented(#Name); }
#define LFS_VK_NOTIMPL_BOOL(Name, Parameters) \
    bool VulkanBackendOps::Name Parameters { not_implemented(#Name); }

    LFS_VK_NOTIMPL_VOID(unary, (const PointwiseProgram&, StorageRef, StorageRef, size_t, ExecContext))
    LFS_VK_NOTIMPL_VOID(binary, (const PointwiseProgram&, StorageRef, StorageRef, StorageRef, size_t, ExecContext))
    LFS_VK_NOTIMPL_VOID(scalar, (const PointwiseProgram&, StorageRef, StorageRef, size_t, ExecContext))
    LFS_VK_NOTIMPL_VOID(broadcast_binary, (const PointwiseProgram&, StorageRef, const StridedLayout&, StorageRef, const StridedLayout&, StorageRef, const StridedLayout&, ExecContext))
    LFS_VK_NOTIMPL_VOID(fused_pointwise_chain, (StorageRef, StorageRef, size_t, const tensor_ops::FusedPointwiseOpChain&, ExecContext))
    LFS_VK_NOTIMPL_VOID(clamp_scalar, (StorageRef, ScalarOperand, ScalarOperand, size_t, ExecContext))
    LFS_VK_NOTIMPL_VOID(clamp_fused, (StorageRef, StorageRef, ScalarOperand, ScalarOperand, size_t, ExecContext))
    LFS_VK_NOTIMPL_VOID(clamp_scalar_int, (StorageRef, ScalarOperand, ScalarOperand, size_t, ExecContext))

    void VulkanBackendOps::convert_type(const StorageRef input,
                                        const StorageRef output,
                                        const size_t count, ExecContext) {
        if (input.dtype != output.dtype) {
            not_implemented("convert_type");
        }
        acquire_vulkan_context()->memory().copy_device_to_device(CopyRequest{
            .src = input,
            .dst = output,
            .bytes = count * dtype_size(input.dtype),
            .synchronous = false,
        });
    }

    void VulkanBackendOps::fill_strided(const StorageRef output,
                                        const StridedLayout& layout,
                                        const ScalarOperand value, ExecContext) {
        if (is_contiguous_layout(layout)) {
            fill_contiguous(output, layout.element_count, value);
            return;
        }
        const size_t allocation_bytes = output.meta->gpu_descriptor.byte_size;
        std::vector<std::byte> data(allocation_bytes);
        auto context = acquire_vulkan_context();
        StorageRef base = output;
        base.byte_offset = 0;
        context->memory().copy_device_to_host(CopyRequest{
            .src = base,
            .dst = raw_storage_ref(data.data()),
            .bytes = allocation_bytes,
            .synchronous = true,
        });
        const size_t element_size = dtype_size(output.dtype);
        const uint64_t pattern = fill_pattern(output.dtype, value);
        for (size_t index = 0; index < layout.element_count; ++index) {
            const size_t byte_offset = output.byte_offset +
                                       logical_offset(layout, index) * element_size;
            LFS_ASSERT_MSG(byte_offset + element_size <= data.size(),
                           "Vulkan strided fill writes outside its allocation");
            std::memcpy(data.data() + byte_offset, &pattern, element_size);
        }
        context->memory().copy_host_to_device(CopyRequest{
            .src = raw_storage_ref(data.data()),
            .dst = base,
            .bytes = allocation_bytes,
            .synchronous = true,
        });
    }

    void VulkanBackendOps::load_fill(const StorageRef output, const size_t count,
                                     const ScalarOperand value, ExecContext) {
        fill_contiguous(output, count, value);
    }

    LFS_VK_NOTIMPL_FLOAT(sum_scalar, (StorageRef, size_t, ExecContext))
    LFS_VK_NOTIMPL_FLOAT(mean_scalar, (StorageRef, size_t, ExecContext))
    LFS_VK_NOTIMPL_FLOAT(max_scalar, (StorageRef, size_t, ExecContext))
    LFS_VK_NOTIMPL_FLOAT(min_scalar, (StorageRef, size_t, ExecContext))
    LFS_VK_NOTIMPL_VOID(reduce, (StorageRef, StorageRef, const StridedLayout&, const ReduceProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(column_reduce, (StorageRef, StorageRef, size_t, size_t, const ReduceProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(strided_reduce, (StorageRef, StorageRef, size_t, size_t, size_t, const ReduceProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(fused_transform_reduce, (StorageRef, StorageRef, size_t, const tensor_ops::FusedPointwiseOpChain&, const ReduceProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(fused_segmented_transform_reduce, (StorageRef, StorageRef, size_t, size_t, const tensor_ops::FusedPointwiseOpChain&, const ReduceProgram&, ExecContext))
    LFS_VK_NOTIMPL_SIZE(count_nonzero_bool, (StorageRef, size_t, ExecContext))
    LFS_VK_NOTIMPL_SIZE(count_nonzero_float, (StorageRef, size_t, ExecContext))
    LFS_VK_NOTIMPL_BOOL(has_nan, (StorageRef, size_t, ExecContext))
    LFS_VK_NOTIMPL_BOOL(has_inf, (StorageRef, size_t, ExecContext))
    LFS_VK_NOTIMPL_VOID(cumsum, (StorageRef, const StridedLayout&, int, ExecContext))
    LFS_VK_NOTIMPL_VOID(sort_1d, (StorageRef, StorageRef, size_t, const SortProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(sort_2d, (StorageRef, StorageRef, const SortProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(sgemm, (StorageRef, StorageRef, StorageRef, const GemmProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(sgemm_tn, (StorageRef, StorageRef, StorageRef, const GemmProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(sgemm_batched, (StorageRef, StorageRef, StorageRef, const GemmProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(sgemm_bias_relu, (StorageRef, StorageRef, StorageRef, StorageRef, const GemmProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(dot_product, (StorageRef, StorageRef, StorageRef, size_t, ExecContext))
    LFS_VK_NOTIMPL_VOID(diag, (StorageRef, StorageRef, size_t, ExecContext))
    LFS_VK_NOTIMPL_VOID(eye, (StorageRef, size_t, size_t, ExecContext))
    LFS_VK_NOTIMPL_VOID(cdist, (StorageRef, StorageRef, StorageRef, size_t, size_t, size_t, float, ExecContext))
    LFS_VK_NOTIMPL_VOID(max_pool2d, (StorageRef, StorageRef, const PoolProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(adaptive_avg_pool2d, (StorageRef, StorageRef, const PoolProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(bias_add, (StorageRef, StorageRef, StorageRef, int, int, int, ExecContext))
    LFS_VK_NOTIMPL_VOID(bias_relu, (StorageRef, StorageRef, StorageRef, int, int, int, ExecContext))
    LFS_VK_NOTIMPL_VOID(relu, (StorageRef, StorageRef, int, ExecContext))
    LFS_VK_NOTIMPL_VOID(uniform, (StorageRef, const RandomProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(bernoulli, (StorageRef, const RandomProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(randint, (StorageRef, const RandomProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(multinomial, (StorageRef, StorageRef, const RandomProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(normal, (StorageRef, StorageRef, const RandomProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(gather, (StorageRef, StorageRef, StorageRef, const StridedLayout&, const StridedLayout&, const IndexProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(gather_fused_unary, (StorageRef, StorageRef, StorageRef, PointwiseOp, const IndexProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(take, (StorageRef, StorageRef, StorageRef, const IndexProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(index_select, (StorageRef, StorageRef, StorageRef, const StridedLayout&, const IndexProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(scatter, (StorageRef, StorageRef, StorageRef, const StridedLayout&, const StridedLayout&, const IndexProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(index_copy, (StorageRef, StorageRef, StorageRef, const StridedLayout&, const IndexProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(index_add, (StorageRef, StorageRef, StorageRef, const StridedLayout&, const IndexProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(index_fill, (StorageRef, StorageRef, const StridedLayout&, const IndexProgram&, ScalarOperand, ExecContext))
    LFS_VK_NOTIMPL_VOID(index_put, (StorageRef, StorageRef, StorageRef, const IndexProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(strided_scatter, (StorageRef, StorageRef, const StridedLayout&, ExecContext))
    LFS_VK_NOTIMPL_VOID(strided_scatter_immediate, (StorageRef, StorageRef, const StridedLayout&, ExecContext))
    LFS_VK_NOTIMPL_VOID(strided_scatter_int32_to_float32, (StorageRef, StorageRef, const StridedLayout&, ExecContext))
    LFS_VK_NOTIMPL_VOID(masked_fill, (StorageRef, StorageRef, const MaskProgram&, ExecContext))
    LFS_VK_NOTIMPL_SIZE(masked_select, (StorageRef, StorageRef, StorageRef, const MaskProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(masked_scatter, (StorageRef, StorageRef, StorageRef, const MaskProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(and_live, (StorageRef, StorageRef, const MaskProgram&, ExecContext))
    LFS_VK_NOTIMPL_VOID(where, (StorageRef, StorageRef, StorageRef, StorageRef, const StridedLayout&, const StridedLayout&, const StridedLayout&, const StridedLayout&, ExecContext))
    LFS_VK_NOTIMPL_SIZE(nonzero, (StorageRef, StorageRef, const MaskProgram&, ExecContext))
    LFS_VK_NOTIMPL_SIZE(nonzero_bool, (StorageRef, StorageRef, const MaskProgram&, ExecContext))

    void VulkanBackendOps::strided_copy(const StorageRef input,
                                        const StorageRef output,
                                        const StridedLayout& input_layout,
                                        ExecContext) {
        copy_strided_vulkan(input, output, input_layout);
    }

    void VulkanBackendOps::strided_copy_immediate(
        const StorageRef input, const StorageRef output,
        const StridedLayout& input_layout, ExecContext) {
        copy_strided_vulkan(input, output, input_layout);
    }

    void VulkanBackendOps::strided_upload(const StorageRef host_input,
                                          const StorageRef output,
                                          const StridedLayout& input_layout,
                                          ExecContext) {
        const size_t element_size = dtype_size(output.dtype);
        const auto* source = static_cast<const std::byte*>(host_input.data) +
                             host_input.byte_offset;
        if (is_contiguous_layout(input_layout)) {
            acquire_vulkan_context()->memory().copy_host_to_device(CopyRequest{
                .src = raw_storage_ref(const_cast<std::byte*>(source), output.dtype),
                .dst = output,
                .bytes = input_layout.element_count * element_size,
                .synchronous = true,
            });
            return;
        }
        std::vector<std::byte> dense(input_layout.element_count * element_size);
        for (size_t index = 0; index < input_layout.element_count; ++index) {
            std::memcpy(dense.data() + index * element_size,
                        source + logical_offset(input_layout, index) * element_size,
                        element_size);
        }
        acquire_vulkan_context()->memory().copy_host_to_device(CopyRequest{
            .src = raw_storage_ref(dense.data(), output.dtype),
            .dst = output,
            .bytes = dense.size(),
            .synchronous = true,
        });
    }

    LFS_VK_NOTIMPL_VOID(cat_last_dim, (StorageRef, std::span<const StorageRef>, std::span<const StridedLayout>, size_t, size_t, size_t, ExecContext))
    LFS_VK_NOTIMPL_VOID(cat_middle_dim, (StorageRef, std::span<const StorageRef>, std::span<const StridedLayout>, size_t, size_t, int, size_t, ExecContext))
    LFS_VK_NOTIMPL_VOID(pad, (StorageRef, StorageRef, const StridedLayout&, const StridedLayout&, const std::array<size_t, MAX_TENSOR_RANK>&, ExecContext))

#undef LFS_VK_NOTIMPL_VOID
#undef LFS_VK_NOTIMPL_FLOAT
#undef LFS_VK_NOTIMPL_SIZE
#undef LFS_VK_NOTIMPL_BOOL

} // namespace lfs::core::internal
