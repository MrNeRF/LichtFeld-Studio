/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "execution_plan.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <unordered_set>

namespace lfs::onnx_vulkan::detail {
    namespace {
        [[nodiscard]] VkDeviceSize align_up(const VkDeviceSize value, const VkDeviceSize alignment) {
            return (value + alignment - 1) / alignment * alignment;
        }

        [[nodiscard]] std::vector<std::int64_t>
        contiguous_strides(const std::span<const std::int64_t> shape) {
            std::vector<std::int64_t> strides(shape.size());
            std::int64_t stride = 1;
            for (std::size_t axis = shape.size(); axis-- > 0;) {
                strides[axis] = stride;
                stride *= shape[axis];
            }
            return strides;
        }

        [[nodiscard]] std::uint16_t float_to_half(const float value) {
            const auto bits = std::bit_cast<std::uint32_t>(value);
            const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
            const auto exponent = static_cast<int>((bits >> 23) & 0xffu);
            const auto mantissa = bits & 0x7fffffu;
            if (exponent == 0xff)
                return static_cast<std::uint16_t>(sign | (mantissa == 0 ? 0x7c00u : 0x7e00u));
            const auto half_exponent = exponent - 127 + 15;
            if (half_exponent >= 31)
                return static_cast<std::uint16_t>(sign | 0x7c00u);
            if (half_exponent <= 0) {
                if (half_exponent < -10)
                    return sign;
                const auto normalized = mantissa | 0x800000u;
                const auto shift = 14 - half_exponent;
                auto rounded = normalized >> shift;
                const auto remainder = normalized & ((1u << shift) - 1u);
                const auto halfway = 1u << (shift - 1);
                if (remainder > halfway || (remainder == halfway && (rounded & 1u) != 0u))
                    ++rounded;
                return static_cast<std::uint16_t>(sign | rounded);
            }
            auto rounded = (static_cast<std::uint32_t>(half_exponent) << 10) | (mantissa >> 13);
            const auto remainder = mantissa & 0x1fffu;
            if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u) != 0u))
                ++rounded;
            return static_cast<std::uint16_t>(sign | rounded);
        }
    }

    std::expected<WeightStore, Error>
    upload_weights(const Model& model, const VulkanRuntime& runtime) {
        WeightStore result;
        const auto alignment = runtime.storage_alignment();
        std::unordered_set<std::string_view> matrix_weights;
        if (runtime.native_fp16_storage_enabled()) {
            for (const auto& node : model.graph.nodes) {
                if ((node.op_type == "MatMul" || node.op_type == "Gemm") && node.inputs.size() > 1)
                    matrix_weights.emplace(node.inputs[1]);
            }
        }
        const auto is_fp16_matrix_weight = [&](const TensorData& tensor) {
            return tensor.type == ElementType::Float32 && tensor.shape.size() >= 2 &&
                   tensor.bytes.size() >= 4096 && matrix_weights.contains(tensor.name);
        };
        VkDeviceSize total = 0;
        for (const auto& tensor : model.graph.initializers) {
            if (tensor.type != ElementType::Float32)
                continue;
            total = align_up(total, alignment);
            if (tensor.bytes.size() > runtime.maximum_storage_range())
                return std::unexpected(Error{ErrorCode::UnsupportedModel,
                                             "initializer '" + tensor.name +
                                                 "' exceeds maxStorageBufferRange on this Vulkan device",
                                             {}, "Vulkan maxStorageBufferRange"});
            if (tensor.bytes.size() > std::numeric_limits<VkDeviceSize>::max() - total)
                return std::unexpected(Error{ErrorCode::MalformedModel, "packed initializer size overflows"});
            total += tensor.bytes.size();
        }
        auto buffer = runtime.create_buffer(total,
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (!buffer)
            return std::unexpected(buffer.error());
        result.buffer = std::move(*buffer);

        VkDeviceSize fp16_total = 0;
        if (runtime.native_fp16_storage_enabled()) {
            for (const auto& tensor : model.graph.initializers) {
                if (!is_fp16_matrix_weight(tensor))
                    continue;
                const auto packed_bytes = align_up(std::max<VkDeviceSize>(4, tensor.bytes.size() / 2), 4);
                if (packed_bytes > runtime.maximum_storage_range())
                    continue;
                fp16_total = align_up(fp16_total, alignment);
                if (packed_bytes > std::numeric_limits<VkDeviceSize>::max() - fp16_total)
                    return std::unexpected(Error{ErrorCode::MalformedModel, "packed FP16 initializer size overflows"});
                fp16_total += packed_bytes;
            }
            if (fp16_total != 0) {
                auto fp16_buffer = runtime.create_buffer(fp16_total,
                                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                if (!fp16_buffer)
                    return std::unexpected(fp16_buffer.error());
                result.fp16_buffer = std::move(*fp16_buffer);
            }
        }

        VkDeviceSize offset = 0;
        for (const auto& tensor : model.graph.initializers) {
            if (tensor.type != ElementType::Float32)
                continue;
            offset = align_up(offset, alignment);
            if (auto uploaded = runtime.upload(result.buffer, offset, tensor.bytes); !uploaded)
                return std::unexpected(uploaded.error());
            result.tensors.emplace(tensor.name,
                                   DeviceTensor{
                                       .binding = {result.buffer.buffer, offset,
                                                   std::max<VkDeviceSize>(4, tensor.bytes.size())},
                                       .type = tensor.type,
                                       .shape = tensor.shape,
                                       .strides = contiguous_strides(tensor.shape),
                                   });
            offset += tensor.bytes.size();
        }
        VkDeviceSize fp16_offset = 0;
        for (const auto& tensor : model.graph.initializers) {
            if (!is_fp16_matrix_weight(tensor) || !result.fp16_buffer.buffer)
                continue;
            fp16_offset = align_up(fp16_offset, alignment);
            const auto count = tensor.bytes.size() / sizeof(float);
            const auto packed_bytes = static_cast<std::size_t>(
                align_up(std::max<VkDeviceSize>(4, count * sizeof(std::uint16_t)), 4));
            if (packed_bytes > runtime.maximum_storage_range())
                continue;
            std::vector<std::byte> packed(packed_bytes);
            for (std::size_t index = 0; index < count; ++index) {
                float value;
                std::memcpy(&value, tensor.bytes.data() + index * sizeof(float), sizeof(float));
                const auto half = float_to_half(value);
                std::memcpy(packed.data() + index * sizeof(std::uint16_t), &half, sizeof(half));
            }
            if (auto uploaded = runtime.upload(result.fp16_buffer, fp16_offset, packed); !uploaded)
                return std::unexpected(uploaded.error());
            result.fp16_tensors.emplace(tensor.name,
                                        DeviceTensor{
                                            .binding = {result.fp16_buffer.buffer, fp16_offset, packed_bytes},
                                            .type = tensor.type,
                                            .shape = tensor.shape,
                                            .strides = contiguous_strides(tensor.shape),
                                        });
            fp16_offset += packed_bytes;
        }
        return result;
    }

} // namespace lfs::onnx_vulkan::detail
