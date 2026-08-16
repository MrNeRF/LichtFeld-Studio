/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "execution_plan.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_map>
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

        [[nodiscard]] std::unordered_set<std::string>
        packed_matmul_weights(const Model& model) {
            std::unordered_map<std::string, std::size_t> uses;
            for (const auto& node : model.graph.nodes)
                for (const auto& input : node.inputs)
                    if (!input.empty())
                        ++uses[input];

            std::unordered_set<std::string> eligible;
            for (const auto& node : model.graph.nodes) {
                if (node.op_type != "MatMul" || node.inputs.size() < 2 || uses[node.inputs[1]] != 1)
                    continue;
                const auto initializer = std::ranges::find(model.graph.initializers, node.inputs[1],
                                                           &TensorData::name);
                if (initializer == model.graph.initializers.end() ||
                    initializer->type != ElementType::Float32 || initializer->shape.size() != 2 ||
                    initializer->shape[0] < 256 || initializer->shape[1] < 256 ||
                    // Packing did not improve square/expansion projections in
                    // representative MoGe measurements, so avoid duplicating
                    // those large weights. Contraction layers are the path
                    // where the packed layout can still win device tuning.
                    initializer->shape[0] <= initializer->shape[1])
                    continue;
                eligible.emplace(initializer->name);
            }
            return eligible;
        }

        [[nodiscard]] std::expected<std::vector<std::byte>, Error>
        pack_matmul_b(const TensorData& tensor) {
            constexpr std::size_t kBlockK = 16;
            constexpr std::size_t kBlockN = 64;
            const auto k_size = static_cast<std::size_t>(tensor.shape[0]);
            const auto n_size = static_cast<std::size_t>(tensor.shape[1]);
            if (k_size > std::numeric_limits<std::size_t>::max() / n_size ||
                k_size * n_size > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
                tensor.bytes.size() != k_size * n_size * sizeof(float))
                return std::unexpected(Error{ErrorCode::MalformedModel,
                                             "MatMul initializer '" + tensor.name + "' has invalid byte size"});

            const auto k_blocks = (k_size + kBlockK - 1) / kBlockK;
            const auto n_blocks = (n_size + kBlockN - 1) / kBlockN;
            if (n_blocks > std::numeric_limits<std::size_t>::max() / k_blocks ||
                n_blocks * k_blocks > std::numeric_limits<std::size_t>::max() / (kBlockK * kBlockN * sizeof(float)))
                return std::unexpected(Error{ErrorCode::MalformedModel,
                                             "packed MatMul initializer size overflows"});
            std::vector<std::byte> packed(n_blocks * k_blocks * kBlockK * kBlockN * sizeof(float));
            for (std::size_t n_block = 0; n_block < n_blocks; ++n_block) {
                const auto n_origin = n_block * kBlockN;
                const auto columns = std::min(kBlockN, n_size - n_origin);
                for (std::size_t k_block = 0; k_block < k_blocks; ++k_block) {
                    for (std::size_t inner_k = 0; inner_k < kBlockK; ++inner_k) {
                        const auto k = k_block * kBlockK + inner_k;
                        if (k >= k_size)
                            break;
                        const auto source = (k * n_size + n_origin) * sizeof(float);
                        const auto destination =
                            ((((n_block * k_blocks + k_block) * kBlockK + inner_k) * kBlockN)) * sizeof(float);
                        std::memcpy(packed.data() + destination, tensor.bytes.data() + source,
                                    columns * sizeof(float));
                    }
                }
            }
            return packed;
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

        const auto packed_names = packed_matmul_weights(model);
        VkDeviceSize packed_total = 0;
        std::unordered_map<std::string, std::vector<std::byte>> packed_bytes;
        for (const auto& tensor : model.graph.initializers) {
            if (!packed_names.contains(tensor.name))
                continue;
            auto packed = pack_matmul_b(tensor);
            if (!packed)
                return std::unexpected(packed.error());
            packed_total = align_up(packed_total, alignment);
            if (packed->size() > runtime.maximum_storage_range() ||
                packed->size() > std::numeric_limits<VkDeviceSize>::max() - packed_total) {
                packed_total = 0;
                packed_bytes.clear();
                break;
            }
            packed_total += packed->size();
            packed_bytes.emplace(tensor.name, std::move(*packed));
        }
        if (packed_total != 0) {
            auto packed_buffer = runtime.create_buffer(packed_total,
                                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (packed_buffer) {
                VkDeviceSize packed_offset = 0;
                bool uploaded_all = true;
                std::unordered_map<std::string, DeviceTensor> packed_tensors;
                for (const auto& tensor : model.graph.initializers) {
                    const auto bytes = packed_bytes.find(tensor.name);
                    if (bytes == packed_bytes.end())
                        continue;
                    packed_offset = align_up(packed_offset, alignment);
                    if (auto uploaded = runtime.upload(*packed_buffer, packed_offset, bytes->second); !uploaded) {
                        uploaded_all = false;
                        break;
                    }
                    packed_tensors.emplace(
                        tensor.name,
                        DeviceTensor{
                            .binding = {packed_buffer->buffer, packed_offset,
                                        std::max<VkDeviceSize>(4, bytes->second.size())},
                            .type = tensor.type,
                            .shape = tensor.shape,
                            .strides = {},
                        });
                    packed_offset += bytes->second.size();
                }
                if (uploaded_all) {
                    result.packed_buffer = std::move(*packed_buffer);
                    result.packed_tensors = std::move(packed_tensors);
                }
            }
        }
        return result;
    }

} // namespace lfs::onnx_vulkan::detail
