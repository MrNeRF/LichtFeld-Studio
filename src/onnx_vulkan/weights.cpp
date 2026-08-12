/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "execution_plan.hpp"

#include <algorithm>
#include <limits>

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
    } // namespace

    std::expected<WeightStore, Error>
    upload_weights(const Model& model, const VulkanRuntime& runtime) {
        WeightStore result;
        const auto alignment = runtime.storage_alignment();
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
        return result;
    }

} // namespace lfs::onnx_vulkan::detail
