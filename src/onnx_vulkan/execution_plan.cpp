/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "execution_plan.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace lfs::onnx_vulkan::detail {
    namespace {
        constexpr std::size_t kKernelRank = 8;
        constexpr std::size_t kParameterWords = 128;
        constexpr std::int64_t kMaximumPortableFusedAttentionTokens = 64;

        [[nodiscard]] Error execution_error(std::string message, const Node* node = nullptr) {
            if (node)
                message = "node '" + node->name + "' (" + node->op_type + "): " + message;
            return {ErrorCode::ExecutionFailure, std::move(message), node ? node->name : std::string{}, {}};
        }

        [[nodiscard]] std::expected<std::size_t, Error>
        element_count(const std::span<const std::int64_t> shape) {
            std::size_t count = 1;
            for (const auto extent : shape) {
                if (extent < 0)
                    return std::unexpected(execution_error("dynamic extent was not resolved"));
                if (extent == 0)
                    return std::size_t{0};
                if (count > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(extent))
                    return std::unexpected(execution_error("tensor element count overflows"));
                count *= static_cast<std::size_t>(extent);
            }
            return count;
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

        void collect_captures(const Graph& graph, std::unordered_set<std::string>& captures) {
            std::unordered_set<std::string> local;
            for (const auto& input : graph.inputs) local.emplace(input.name);
            for (const auto& initializer : graph.initializers) local.emplace(initializer.name);
            for (const auto& node : graph.nodes)
                for (const auto& output : node.outputs)
                    if (!output.empty()) local.emplace(output);
            for (const auto& node : graph.nodes) {
                for (const auto& input : node.inputs)
                    if (!input.empty() && !local.contains(input)) captures.emplace(input);
                for (const auto& attribute : node.attributes) {
                    const auto* nested = std::get_if<std::shared_ptr<Graph>>(&attribute.value);
                    if (!nested || !*nested)
                        continue;
                    std::unordered_set<std::string> nested_captures;
                    collect_captures(**nested, nested_captures);
                    for (const auto& capture : nested_captures)
                        if (!local.contains(capture)) captures.emplace(capture);
                }
            }
        }

        [[nodiscard]] VkDeviceSize align_up(const VkDeviceSize value, const VkDeviceSize alignment) {
            return (value + alignment - 1) / alignment * alignment;
        }

        struct HostTensor {
            ElementType type = ElementType::Float32;
            std::vector<std::int64_t> shape;
            std::span<const std::byte> bytes;
            std::shared_ptr<std::vector<std::byte>> owner;
        };

        [[nodiscard]] HostTensor host_tensor(const TensorData& tensor) {
            return {tensor.type, tensor.shape, tensor.bytes, tensor.external_owner};
        }

        [[nodiscard]] HostTensor owned_host(const ElementType type,
                                            std::vector<std::int64_t> shape,
                                            std::vector<std::byte> bytes) {
            auto owner = std::make_shared<std::vector<std::byte>>(std::move(bytes));
            return {type, std::move(shape), *owner, std::move(owner)};
        }

        [[nodiscard]] long double read_number(const HostTensor& tensor, const std::size_t index) {
            switch (tensor.type) {
            case ElementType::Float32: return tensor.bytes.size() >= (index + 1) * 4
                                                   ? reinterpret_cast<const float*>(tensor.bytes.data())[index]
                                                   : 0;
            case ElementType::Int32: return reinterpret_cast<const std::int32_t*>(tensor.bytes.data())[index];
            case ElementType::Int64: return reinterpret_cast<const std::int64_t*>(tensor.bytes.data())[index];
            case ElementType::Bool: return std::to_integer<std::uint8_t>(tensor.bytes[index]) != 0;
            }
            return 0;
        }

        void write_number(std::span<std::byte> bytes,
                          const ElementType type,
                          const std::size_t index,
                          const long double value) {
            switch (type) {
            case ElementType::Float32: reinterpret_cast<float*>(bytes.data())[index] = static_cast<float>(value); break;
            case ElementType::Int32: reinterpret_cast<std::int32_t*>(bytes.data())[index] = static_cast<std::int32_t>(value); break;
            case ElementType::Int64: reinterpret_cast<std::int64_t*>(bytes.data())[index] = static_cast<std::int64_t>(value); break;
            case ElementType::Bool: bytes[index] = static_cast<std::byte>(value != 0); break;
            }
        }

        [[nodiscard]] std::expected<std::vector<std::int64_t>, Error>
        integer_values(const HostTensor& tensor, const Node* node = nullptr) {
            if (tensor.type != ElementType::Int64 && tensor.type != ElementType::Int32)
                return std::unexpected(execution_error("expected an int64/int32 control tensor", node));
            auto count = element_count(tensor.shape);
            if (!count)
                return std::unexpected(count.error());
            std::vector<std::int64_t> result(*count);
            // Keep shape controls in the integer domain. On MSVC, long double is
            // only 64 bits, so routing INT64_MAX Slice sentinels through
            // read_number() rounds them to the out-of-range value 2^63.
            for (std::size_t index = 0; index < *count; ++index) {
                if (tensor.type == ElementType::Int64) {
                    std::memcpy(&result[index], tensor.bytes.data() + index * sizeof(std::int64_t),
                                sizeof(std::int64_t));
                } else {
                    std::int32_t value = 0;
                    std::memcpy(&value, tensor.bytes.data() + index * sizeof(value), sizeof(value));
                    result[index] = value;
                }
            }
            return result;
        }

        [[nodiscard]] std::int64_t normalize_axis(std::int64_t axis, const std::size_t rank) {
            if (axis < 0)
                axis += static_cast<std::int64_t>(rank);
            return axis;
        }

        [[nodiscard]] std::optional<std::int64_t>
        int_attribute(const Node& node, const std::string_view name) {
            if (const auto* attr = find_attribute(node, name))
                return std::get<std::int64_t>(attr->value);
            return std::nullopt;
        }

        [[nodiscard]] std::vector<std::int64_t>
        ints_attribute(const Node& node, const std::string_view name) {
            if (const auto* attr = find_attribute(node, name))
                return std::get<std::vector<std::int64_t>>(attr->value);
            return {};
        }

        [[nodiscard]] std::string
        string_attribute(const Node& node, const std::string_view name, std::string fallback = {}) {
            if (const auto* attr = find_attribute(node, name))
                return std::get<std::string>(attr->value);
            return fallback;
        }

        [[nodiscard]] float float_attribute(const Node& node,
                                            const std::string_view name,
                                            const float fallback) {
            if (const auto* attr = find_attribute(node, name))
                return std::get<float>(attr->value);
            return fallback;
        }

        [[nodiscard]] std::expected<ElementType, Error>
        element_type_from_onnx(const std::int64_t type, const Node& node) {
            switch (type) {
            case 1: return ElementType::Float32;
            case 6: return ElementType::Int32;
            case 7: return ElementType::Int64;
            case 9: return ElementType::Bool;
            default: return std::unexpected(execution_error("unsupported Cast target type " + std::to_string(type), &node));
            }
        }

        [[nodiscard]] std::expected<std::vector<std::int64_t>, Error>
        broadcast_shape(const std::span<const std::int64_t> lhs,
                        const std::span<const std::int64_t> rhs,
                        const Node* node = nullptr) {
            const auto rank = std::max(lhs.size(), rhs.size());
            std::vector<std::int64_t> output(rank, 1);
            for (std::size_t trailing = 0; trailing < rank; ++trailing) {
                const auto l = trailing < lhs.size() ? lhs[lhs.size() - 1 - trailing] : 1;
                const auto r = trailing < rhs.size() ? rhs[rhs.size() - 1 - trailing] : 1;
                if (l != r && l != 1 && r != 1) {
                    std::string detail = "incompatible broadcast dimensions [";
                    for (const auto value : lhs) detail += std::to_string(value) + ",";
                    detail += "] and [";
                    for (const auto value : rhs) detail += std::to_string(value) + ",";
                    detail += "]";
                    return std::unexpected(execution_error(std::move(detail), node));
                }
                output[rank - 1 - trailing] = std::max(l, r);
            }
            return output;
        }

        [[nodiscard]] std::size_t broadcast_index(std::size_t output_index,
                                                  const std::span<const std::int64_t> output_shape,
                                                  const std::span<const std::int64_t> input_shape) {
            const auto input_strides = contiguous_strides(input_shape);
            std::size_t index = 0;
            for (std::size_t axis = output_shape.size(); axis-- > 0;) {
                const auto coord = output_shape[axis] == 0 ? 0 : output_index % output_shape[axis];
                output_index = output_shape[axis] == 0 ? 0 : output_index / output_shape[axis];
                if (axis + input_shape.size() >= output_shape.size()) {
                    const auto input_axis = axis + input_shape.size() - output_shape.size();
                    if (input_shape[input_axis] != 1)
                        index += coord * input_strides[input_axis];
                }
            }
            return index;
        }

        struct ValueSpec {
            ElementType type = ElementType::Float32;
            std::vector<std::int64_t> shape;
        };

        enum class RefKind : std::uint8_t { None, Weight, Arena, Literal };
        struct DeviceRef {
            RefKind kind = RefKind::None;
            BufferBinding direct{};
            VkDeviceSize offset = 0;
            VkDeviceSize range = 0;
            std::size_t slice = std::numeric_limits<std::size_t>::max();
        };

        struct Value {
            ElementType type = ElementType::Float32;
            std::vector<std::int64_t> shape;
            std::vector<std::int64_t> strides;
            std::int64_t element_offset = 0;
            std::optional<HostTensor> host;
            DeviceRef device;
            std::optional<std::size_t> producer_dispatch;
        };

        [[nodiscard]] std::expected<HostTensor, Error> constant_value(const Node& node) {
            for (const auto& attribute : node.attributes) {
                if (attribute.name == "value")
                    return host_tensor(std::get<TensorData>(attribute.value));
                if (attribute.name == "value_float") {
                    std::vector<std::byte> bytes(4);
                    write_number(bytes, ElementType::Float32, 0, std::get<float>(attribute.value));
                    return owned_host(ElementType::Float32, {}, std::move(bytes));
                }
                if (attribute.name == "value_int") {
                    std::vector<std::byte> bytes(8);
                    write_number(bytes, ElementType::Int64, 0, std::get<std::int64_t>(attribute.value));
                    return owned_host(ElementType::Int64, {}, std::move(bytes));
                }
                if (attribute.name == "value_floats") {
                    const auto& values = std::get<std::vector<float>>(attribute.value);
                    std::vector<std::byte> bytes(values.size() * 4);
                    std::memcpy(bytes.data(), values.data(), bytes.size());
                    return owned_host(ElementType::Float32, {static_cast<std::int64_t>(values.size())}, std::move(bytes));
                }
                if (attribute.name == "value_ints") {
                    const auto& values = std::get<std::vector<std::int64_t>>(attribute.value);
                    std::vector<std::byte> bytes(values.size() * 8);
                    std::memcpy(bytes.data(), values.data(), bytes.size());
                    return owned_host(ElementType::Int64, {static_cast<std::int64_t>(values.size())}, std::move(bytes));
                }
            }
            return std::unexpected(execution_error("Constant has no supported value", &node));
        }

        [[nodiscard]] std::expected<std::vector<std::int64_t>, Error>
        axes_from(const Node& node,
                  const std::span<const Value* const> inputs,
                  const std::size_t input_index,
                  const std::size_t rank,
                  const bool default_all) {
            std::vector<std::int64_t> axes;
            if (input_index < inputs.size() && inputs[input_index] && inputs[input_index]->host) {
                auto values = integer_values(*inputs[input_index]->host, &node);
                if (!values)
                    return std::unexpected(values.error());
                axes = std::move(*values);
            } else {
                axes = ints_attribute(node, "axes");
            }
            if (axes.empty() && default_all) {
                axes.resize(rank);
                std::iota(axes.begin(), axes.end(), 0);
            }
            for (auto& axis : axes) {
                axis = normalize_axis(axis, rank);
                if (axis < 0 || axis >= static_cast<std::int64_t>(rank))
                    return std::unexpected(execution_error("axis is out of range", &node));
            }
            std::ranges::sort(axes);
            if (std::ranges::adjacent_find(axes) != axes.end())
                return std::unexpected(execution_error("axes contain duplicates", &node));
            return axes;
        }

        [[nodiscard]] std::expected<std::vector<ValueSpec>, Error>
        infer_outputs(const Node& node, const std::span<const Value* const> in) {
            const auto unary = [&]() -> std::expected<std::vector<ValueSpec>, Error> {
                if (in.empty() || !in[0]) return std::unexpected(execution_error("missing required input", &node));
                return std::vector<ValueSpec>{{in[0]->type, in[0]->shape}};
            };
            if (node.op_type == "Constant") {
                auto value = constant_value(node);
                if (!value) return std::unexpected(value.error());
                return std::vector<ValueSpec>{{value->type, value->shape}};
            }
            if (node.op_type == "Shape") {
                auto start = int_attribute(node, "start").value_or(0);
                auto end = int_attribute(node, "end").value_or(static_cast<std::int64_t>(in[0]->shape.size()));
                start = std::clamp(normalize_axis(start, in[0]->shape.size()), std::int64_t{0}, static_cast<std::int64_t>(in[0]->shape.size()));
                end = std::clamp(normalize_axis(end, in[0]->shape.size()), std::int64_t{0}, static_cast<std::int64_t>(in[0]->shape.size()));
                return std::vector<ValueSpec>{{ElementType::Int64, {std::max<std::int64_t>(0, end - start)}}};
            }
            if (node.op_type == "Cast") {
                auto type = element_type_from_onnx(*int_attribute(node, "to"), node);
                if (!type) return std::unexpected(type.error());
                return std::vector<ValueSpec>{{*type, in[0]->shape}};
            }
            if (node.op_type == "Equal") {
                auto shape = broadcast_shape(in[0]->shape, in[1]->shape, &node);
                if (!shape) return std::unexpected(shape.error());
                return std::vector<ValueSpec>{{ElementType::Bool, std::move(*shape)}};
            }
            static const std::unordered_set<std::string> binary{
                "Add", "Sub", "Mul", "Div", "Pow", "Mod"};
            if (binary.contains(node.op_type)) {
                auto shape = broadcast_shape(in[0]->shape, in[1]->shape, &node);
                if (!shape) return std::unexpected(shape.error());
                return std::vector<ValueSpec>{{in[0]->type, std::move(*shape)}};
            }
            if (node.op_type == "Where") {
                auto xy = broadcast_shape(in[1]->shape, in[2]->shape, &node);
                if (!xy) return std::unexpected(xy.error());
                auto all = broadcast_shape(in[0]->shape, *xy, &node);
                if (!all) return std::unexpected(all.error());
                return std::vector<ValueSpec>{{in[1]->type, std::move(*all)}};
            }
            if (node.op_type == "Concat") {
                const auto rank = in[0]->shape.size();
                const auto axis = normalize_axis(*int_attribute(node, "axis"), rank);
                if (axis < 0 || axis >= static_cast<std::int64_t>(rank))
                    return std::unexpected(execution_error("Concat axis is out of range", &node));
                auto shape = in[0]->shape;
                for (std::size_t i = 1; i < in.size(); ++i) {
                    if (!in[i] || in[i]->shape.size() != rank)
                        return std::unexpected(execution_error("Concat ranks do not match", &node));
                    for (std::size_t d = 0; d < rank; ++d)
                        if (d != static_cast<std::size_t>(axis) && in[i]->shape[d] != shape[d])
                            return std::unexpected(execution_error("Concat dimensions do not match", &node));
                    shape[axis] += in[i]->shape[axis];
                }
                return std::vector<ValueSpec>{{in[0]->type, std::move(shape)}};
            }
            if (node.op_type == "ConstantOfShape") {
                if (!in[0]->host) return std::unexpected(execution_error("ConstantOfShape shape is not host-visible", &node));
                auto shape = integer_values(*in[0]->host, &node);
                if (!shape) return std::unexpected(shape.error());
                ElementType type = ElementType::Float32;
                if (const auto* attr = find_attribute(node, "value"))
                    type = std::get<TensorData>(attr->value).type;
                return std::vector<ValueSpec>{{type, std::move(*shape)}};
            }
            if (node.op_type == "Expand") {
                if (!in[1]->host) return std::unexpected(execution_error("Expand shape is not host-visible", &node));
                auto shape = integer_values(*in[1]->host, &node);
                if (!shape) return std::unexpected(shape.error());
                auto compatible = broadcast_shape(in[0]->shape, *shape, &node);
                if (!compatible)
                    return std::unexpected(execution_error("invalid Expand shape", &node));
                return std::vector<ValueSpec>{{in[0]->type, std::move(*compatible)}};
            }
            if (node.op_type == "Gather") {
                const auto axis = normalize_axis(int_attribute(node, "axis").value_or(0), in[0]->shape.size());
                if (axis < 0 || axis >= static_cast<std::int64_t>(in[0]->shape.size()))
                    return std::unexpected(execution_error("Gather axis is out of range", &node));
                std::vector<std::int64_t> shape(in[0]->shape.begin(), in[0]->shape.begin() + axis);
                shape.insert(shape.end(), in[1]->shape.begin(), in[1]->shape.end());
                shape.insert(shape.end(), in[0]->shape.begin() + axis + 1, in[0]->shape.end());
                return std::vector<ValueSpec>{{in[0]->type, std::move(shape)}};
            }
            if (node.op_type == "Reshape") {
                if (!in[1]->host) return std::unexpected(execution_error("Reshape shape is not host-visible", &node));
                auto shape = integer_values(*in[1]->host, &node);
                if (!shape) return std::unexpected(shape.error());
                const bool allowzero = int_attribute(node, "allowzero").value_or(0) != 0;
                std::int64_t inferred_axis = -1;
                std::size_t known = 1;
                for (std::size_t i = 0; i < shape->size(); ++i) {
                    if ((*shape)[i] == 0 && !allowzero) {
                        if (i >= in[0]->shape.size()) return std::unexpected(execution_error("Reshape zero axis is invalid", &node));
                        (*shape)[i] = in[0]->shape[i];
                    } else if ((*shape)[i] == -1) {
                        if (inferred_axis >= 0) return std::unexpected(execution_error("Reshape has multiple -1 dimensions", &node));
                        inferred_axis = static_cast<std::int64_t>(i);
                        continue;
                    } else if ((*shape)[i] < 0) {
                        return std::unexpected(execution_error("Reshape has an invalid dimension", &node));
                    }
                    known *= static_cast<std::size_t>((*shape)[i]);
                }
                auto input_count = element_count(in[0]->shape);
                if (!input_count) return std::unexpected(input_count.error());
                if (inferred_axis >= 0) {
                    if (known == 0 || *input_count % known != 0) return std::unexpected(execution_error("Reshape cannot infer dimension", &node));
                    (*shape)[inferred_axis] = static_cast<std::int64_t>(*input_count / known);
                }
                auto output_count = element_count(*shape);
                if (!output_count || *output_count != *input_count) return std::unexpected(execution_error("Reshape element count changes", &node));
                return std::vector<ValueSpec>{{in[0]->type, std::move(*shape)}};
            }
            if (node.op_type == "Squeeze" || node.op_type == "Unsqueeze") {
                if (node.op_type == "Squeeze") {
                    auto axes = axes_from(node, in, 1, in[0]->shape.size(), false);
                    if (!axes) return std::unexpected(axes.error());
                    std::vector<std::int64_t> shape;
                    for (std::size_t axis = 0; axis < in[0]->shape.size(); ++axis) {
                        const bool remove = axes->empty() ? in[0]->shape[axis] == 1 : std::ranges::binary_search(*axes, static_cast<std::int64_t>(axis));
                        if (remove && in[0]->shape[axis] != 1) return std::unexpected(execution_error("Squeeze axis is not one", &node));
                        if (!remove) shape.push_back(in[0]->shape[axis]);
                    }
                    return std::vector<ValueSpec>{{in[0]->type, std::move(shape)}};
                }
                if (!in[1]->host) return std::unexpected(execution_error("Unsqueeze axes are not host-visible", &node));
                auto raw_axes = integer_values(*in[1]->host, &node);
                if (!raw_axes) return std::unexpected(raw_axes.error());
                const auto output_rank = in[0]->shape.size() + raw_axes->size();
                std::vector<std::int64_t> axes;
                for (auto axis : *raw_axes) {
                    axis = normalize_axis(axis, output_rank);
                    if (axis < 0 || axis >= static_cast<std::int64_t>(output_rank)) return std::unexpected(execution_error("Unsqueeze axis is out of range", &node));
                    axes.push_back(axis);
                }
                std::ranges::sort(axes);
                std::vector<std::int64_t> shape;
                std::size_t source = 0;
                for (std::size_t axis = 0; axis < output_rank; ++axis)
                    shape.push_back(std::ranges::binary_search(axes, static_cast<std::int64_t>(axis)) ? 1 : in[0]->shape[source++]);
                return std::vector<ValueSpec>{{in[0]->type, std::move(shape)}};
            }
            if (node.op_type == "Transpose") {
                auto perm = ints_attribute(node, "perm");
                if (perm.empty()) {
                    perm.resize(in[0]->shape.size());
                    std::iota(perm.rbegin(), perm.rend(), 0);
                }
                if (perm.size() != in[0]->shape.size()) return std::unexpected(execution_error("Transpose perm rank mismatch", &node));
                std::vector<std::int64_t> shape(perm.size());
                for (std::size_t i = 0; i < perm.size(); ++i) shape[i] = in[0]->shape[perm[i]];
                return std::vector<ValueSpec>{{in[0]->type, std::move(shape)}};
            }
            if (node.op_type == "Slice") {
                if (!in[1]->host || !in[2]->host) return std::unexpected(execution_error("Slice bounds are not host-visible", &node));
                auto starts = integer_values(*in[1]->host, &node); auto ends = integer_values(*in[2]->host, &node);
                if (!starts || !ends) return std::unexpected(starts ? ends.error() : starts.error());
                std::vector<std::int64_t> axes(starts->size()), steps(starts->size(), 1);
                std::iota(axes.begin(), axes.end(), 0);
                if (in.size() > 3 && in[3] && in[3]->host) { auto v = integer_values(*in[3]->host, &node); if (!v) return std::unexpected(v.error()); axes = std::move(*v); }
                if (in.size() > 4 && in[4] && in[4]->host) { auto v = integer_values(*in[4]->host, &node); if (!v) return std::unexpected(v.error()); steps = std::move(*v); }
                auto shape = in[0]->shape;
                for (std::size_t i = 0; i < starts->size(); ++i) {
                    auto axis = normalize_axis(axes[i], shape.size());
                    const auto extent = shape[axis]; const auto step = steps[i];
                    if (step == 0) return std::unexpected(execution_error("Slice step is zero", &node));
                    auto start = (*starts)[i], end = (*ends)[i];
                    if (step > 0) { if (start < 0) start += extent; if (end < 0) end += extent; start = std::clamp(start, std::int64_t{0}, extent); end = std::clamp(end, std::int64_t{0}, extent); shape[axis] = std::max<std::int64_t>(0, (end - start + step - 1) / step); }
                    else { if (start < 0) start += extent; if (end < 0 && end != std::numeric_limits<std::int64_t>::min()) end += extent; start = std::clamp(start, std::int64_t{-1}, extent - 1); end = std::clamp(end, std::int64_t{-1}, extent - 1); shape[axis] = std::max<std::int64_t>(0, (start - end - step - 1) / (-step)); }
                }
                return std::vector<ValueSpec>{{in[0]->type, std::move(shape)}};
            }
            if (node.op_type == "Split") {
                const auto axis = normalize_axis(int_attribute(node, "axis").value_or(0), in[0]->shape.size());
                std::vector<std::int64_t> splits;
                if (in.size() > 1 && in[1] && in[1]->host) { auto v = integer_values(*in[1]->host, &node); if (!v) return std::unexpected(v.error()); splits = std::move(*v); }
                else splits = ints_attribute(node, "split");
                if (splits.empty()) splits.assign(node.outputs.size(), in[0]->shape[axis] / static_cast<std::int64_t>(node.outputs.size()));
                if (splits.size() != node.outputs.size() || std::accumulate(splits.begin(), splits.end(), std::int64_t{0}) != in[0]->shape[axis]) return std::unexpected(execution_error("invalid Split sizes", &node));
                std::vector<ValueSpec> output;
                for (const auto size : splits) { auto shape = in[0]->shape; shape[axis] = size; output.push_back({in[0]->type, std::move(shape)}); }
                return output;
            }
            if (node.op_type == "Pad") {
                if (!in[1]->host) return std::unexpected(execution_error("Pad pads are not host-visible", &node));
                auto pads = integer_values(*in[1]->host, &node); if (!pads) return std::unexpected(pads.error());
                if (pads->size() != in[0]->shape.size() * 2) return std::unexpected(execution_error("Pad vector has wrong length", &node));
                auto shape = in[0]->shape; for (std::size_t i = 0; i < shape.size(); ++i) { shape[i] += (*pads)[i] + (*pads)[i + shape.size()]; if (shape[i] < 0) return std::unexpected(execution_error("Pad produces a negative extent", &node)); }
                return std::vector<ValueSpec>{{in[0]->type, std::move(shape)}};
            }
            if (node.op_type == "Range") {
                if (!in[0]->host || !in[1]->host || !in[2]->host) return std::unexpected(execution_error("Range inputs are not host-visible", &node));
                const long double start = read_number(*in[0]->host, 0), limit = read_number(*in[1]->host, 0), delta = read_number(*in[2]->host, 0);
                if (delta == 0) return std::unexpected(execution_error("Range delta is zero", &node));
                const auto count = static_cast<std::int64_t>(std::max<long double>(0, std::ceil((limit - start) / delta)));
                return std::vector<ValueSpec>{{in[0]->type, {count}}};
            }
            if (node.op_type == "Conv" || node.op_type == "ConvTranspose") {
                if (in[0]->shape.size() != 4 || in[1]->shape.size() != 4) return std::unexpected(execution_error("only 2D NCHW convolution is supported", &node));
                auto strides = ints_attribute(node, "strides"); if (strides.empty()) strides = {1, 1};
                auto dilations = ints_attribute(node, "dilations"); if (dilations.empty()) dilations = {1, 1};
                auto pads = ints_attribute(node, "pads"); if (pads.empty()) pads = {0, 0, 0, 0};
                auto kernel = ints_attribute(node, "kernel_shape"); if (kernel.empty()) kernel = {in[1]->shape[2], in[1]->shape[3]};
                std::vector<std::int64_t> shape{in[0]->shape[0], node.op_type == "Conv" ? in[1]->shape[0] : in[1]->shape[1] * int_attribute(node, "group").value_or(1), 0, 0};
                for (std::size_t d = 0; d < 2; ++d) {
                    if (node.op_type == "Conv") shape[2 + d] = (in[0]->shape[2 + d] + pads[d] + pads[d + 2] - dilations[d] * (kernel[d] - 1) - 1) / strides[d] + 1;
                    else { auto output_padding = ints_attribute(node, "output_padding"); if (output_padding.empty()) output_padding = {0, 0}; shape[2 + d] = strides[d] * (in[0]->shape[2 + d] - 1) + output_padding[d] + (kernel[d] - 1) * dilations[d] + 1 - pads[d] - pads[d + 2]; }
                }
                return std::vector<ValueSpec>{{ElementType::Float32, std::move(shape)}};
            }
            if (node.op_type == "MatMul") {
                auto a = in[0]->shape, b = in[1]->shape; const bool a1 = a.size() == 1, b1 = b.size() == 1;
                if (a1) a.insert(a.begin(), 1); if (b1) b.push_back(1);
                if (a.back() != b[b.size() - 2]) return std::unexpected(execution_error("MatMul K dimensions differ", &node));
                auto batch = broadcast_shape(std::span(a).first(a.size() - 2), std::span(b).first(b.size() - 2), &node); if (!batch) return std::unexpected(batch.error());
                auto shape = std::move(*batch); shape.push_back(a[a.size() - 2]); shape.push_back(b.back());
                if (a1) shape.erase(shape.end() - 2); if (b1) shape.pop_back();
                return std::vector<ValueSpec>{{ElementType::Float32, std::move(shape)}};
            }
            if (node.op_type == "Gemm") {
                const bool ta = int_attribute(node, "transA").value_or(0), tb = int_attribute(node, "transB").value_or(0);
                const auto m = in[0]->shape[ta ? 1 : 0], k = in[0]->shape[ta ? 0 : 1], bk = in[1]->shape[tb ? 1 : 0], n = in[1]->shape[tb ? 0 : 1];
                if (k != bk) return std::unexpected(execution_error("Gemm K dimensions differ", &node));
                return std::vector<ValueSpec>{{ElementType::Float32, {m, n}}};
            }
            if (node.op_type.starts_with("Reduce")) {
                auto axes = axes_from(node, in, 1, in[0]->shape.size(), true); if (!axes) return std::unexpected(axes.error());
                const bool keep = int_attribute(node, "keepdims").value_or(1) != 0; std::vector<std::int64_t> shape;
                for (std::size_t axis = 0; axis < in[0]->shape.size(); ++axis) { const bool reduced = std::ranges::binary_search(*axes, static_cast<std::int64_t>(axis)); if (!reduced) shape.push_back(in[0]->shape[axis]); else if (keep) shape.push_back(1); }
                return std::vector<ValueSpec>{{in[0]->type, std::move(shape)}};
            }
            if (node.op_type == "Resize") {
                std::vector<std::int64_t> shape;
                if (in.size() > 3 && in[3] && in[3]->host) { auto sizes = integer_values(*in[3]->host, &node); if (!sizes) return std::unexpected(sizes.error()); shape = std::move(*sizes); }
                else if (in.size() > 2 && in[2] && in[2]->host) { auto count = element_count(in[2]->host->shape); if (!count) return std::unexpected(count.error()); for (std::size_t i = 0; i < *count; ++i) shape.push_back(static_cast<std::int64_t>(std::floor(in[0]->shape[i] * read_number(*in[2]->host, i)))); }
                else return std::unexpected(execution_error("Resize has neither host-visible sizes nor scales", &node));
                return std::vector<ValueSpec>{{in[0]->type, std::move(shape)}};
            }
            if (node.op_type == "If") {
                const auto* branch = std::get<std::shared_ptr<Graph>>(find_attribute(node, "then_branch")->value).get();
                std::vector<ValueSpec> result; for (const auto& output : branch->outputs) result.push_back({output.type, output.shape}); return result;
            }
            return unary();
        }

        [[nodiscard]] std::expected<std::vector<HostTensor>, Error>
        evaluate_host(const Node& node,
                      const std::span<const Value* const> inputs,
                      const std::span<const ValueSpec> specs) {
            if (node.op_type == "Constant") { auto v = constant_value(node); if (!v) return std::unexpected(v.error()); return std::vector<HostTensor>{std::move(*v)}; }
            std::vector<HostTensor> outputs;
            for (const auto& spec : specs) {
                auto count = element_count(spec.shape); if (!count) return std::unexpected(count.error());
                outputs.push_back(owned_host(spec.type, spec.shape, std::vector<std::byte>(*count * element_size(spec.type))));
            }
            if (node.op_type == "Shape") {
                auto start = int_attribute(node, "start").value_or(0); if (start < 0) start += inputs[0]->shape.size();
                for (std::size_t i = 0; i < outputs[0].shape[0]; ++i) write_number(*outputs[0].owner, outputs[0].type, i, inputs[0]->shape[start + i]);
                return outputs;
            }
            if (node.op_type == "Range") {
                const auto count = static_cast<std::size_t>(outputs[0].shape[0]); const auto start = read_number(*inputs[0]->host, 0), delta = read_number(*inputs[2]->host, 0);
                for (std::size_t i = 0; i < count; ++i) write_number(*outputs[0].owner, outputs[0].type, i, start + i * delta); return outputs;
            }
            if (node.op_type == "ConstantOfShape") {
                long double value = 0; if (const auto* attr = find_attribute(node, "value")) value = read_number(host_tensor(std::get<TensorData>(attr->value)), 0);
                auto count = element_count(outputs[0].shape); for (std::size_t i = 0; i < *count; ++i) write_number(*outputs[0].owner, outputs[0].type, i, value); return outputs;
            }
            if (node.op_type == "Cast") {
                auto count = element_count(outputs[0].shape); for (std::size_t i = 0; i < *count; ++i) write_number(*outputs[0].owner, outputs[0].type, i, read_number(*inputs[0]->host, i)); return outputs;
            }
            static const std::unordered_set<std::string> elementwise{"Add","Sub","Mul","Div","Pow","Mod","Equal","Where","Neg","Round","Sqrt","Identity","Reshape","Squeeze","Unsqueeze","Expand"};
            if (elementwise.contains(node.op_type)) {
                auto count = element_count(outputs[0].shape);
                const bool linear_view = node.op_type == "Identity" || node.op_type == "Reshape" ||
                                         node.op_type == "Squeeze" || node.op_type == "Unsqueeze";
                if (linear_view) {
                    std::memcpy(outputs[0].owner->data(), inputs[0]->host->bytes.data(),
                                outputs[0].owner->size());
                    return outputs;
                }
                for (std::size_t i = 0; i < *count; ++i) {
                    const auto ai = broadcast_index(i, outputs[0].shape, inputs[0]->shape);
                    long double x = read_number(*inputs[0]->host, ai), value = x;
                    long double y = 0; if (inputs.size() > 1 && inputs[1] && inputs[1]->host) y = read_number(*inputs[1]->host, broadcast_index(i, outputs[0].shape, inputs[1]->shape));
                    if (node.op_type == "Add") value = x + y; else if (node.op_type == "Sub") value = x - y; else if (node.op_type == "Mul") value = x * y; else if (node.op_type == "Div") value = x / y; else if (node.op_type == "Pow") value = std::pow(x, y); else if (node.op_type == "Mod") value = std::fmod(x, y); else if (node.op_type == "Equal") value = x == y; else if (node.op_type == "Neg") value = -x; else if (node.op_type == "Round") value = std::nearbyint(x); else if (node.op_type == "Sqrt") value = std::sqrt(x); else if (node.op_type == "Where") { const bool condition = read_number(*inputs[0]->host, broadcast_index(i, outputs[0].shape, inputs[0]->shape)) != 0; value = read_number(*(condition ? inputs[1] : inputs[2])->host, broadcast_index(i, outputs[0].shape, (condition ? inputs[1] : inputs[2])->shape)); }
                    write_number(*outputs[0].owner, outputs[0].type, i, value);
                }
                return outputs;
            }
            if (node.op_type == "Concat") {
                const auto axis = normalize_axis(*int_attribute(node, "axis"), outputs[0].shape.size()); const auto inner = std::accumulate(outputs[0].shape.begin() + axis + 1, outputs[0].shape.end(), std::size_t{1}, std::multiplies<>()); const auto outer = std::accumulate(outputs[0].shape.begin(), outputs[0].shape.begin() + axis, std::size_t{1}, std::multiplies<>()); const auto width = element_size(outputs[0].type);
                std::size_t axis_offset = 0; for (const auto* input : inputs) { for (std::size_t outer_i = 0; outer_i < outer; ++outer_i) { const auto count = static_cast<std::size_t>(input->shape[axis]) * inner; const auto src = outer_i * count; const auto dst = (outer_i * outputs[0].shape[axis] + axis_offset) * inner; std::memcpy(outputs[0].owner->data() + dst * width, input->host->bytes.data() + src * width, count * width); } axis_offset += input->shape[axis]; } return outputs;
            }
            if (node.op_type == "Gather") {
                const auto axis = normalize_axis(int_attribute(node, "axis").value_or(0), inputs[0]->shape.size()); const auto inner = std::accumulate(inputs[0]->shape.begin() + axis + 1, inputs[0]->shape.end(), std::size_t{1}, std::multiplies<>()); const auto outer = std::accumulate(inputs[0]->shape.begin(), inputs[0]->shape.begin() + axis, std::size_t{1}, std::multiplies<>()); auto indices = integer_values(*inputs[1]->host, &node); if (!indices) return std::unexpected(indices.error()); const auto width = element_size(outputs[0].type);
                for (std::size_t outer_i = 0; outer_i < outer; ++outer_i) for (std::size_t j = 0; j < indices->size(); ++j) { auto index = (*indices)[j]; if (index < 0) index += inputs[0]->shape[axis]; const auto src = (outer_i * inputs[0]->shape[axis] + index) * inner; const auto dst = (outer_i * indices->size() + j) * inner; std::memcpy(outputs[0].owner->data() + dst * width, inputs[0]->host->bytes.data() + src * width, inner * width); } return outputs;
            }
            if (node.op_type == "Slice" || node.op_type == "Transpose") {
                // Control tensors using these operators are small; use a generic coordinate map.
                auto out_count = element_count(outputs[0].shape); auto in_strides = contiguous_strides(inputs[0]->shape); std::vector<std::int64_t> base(inputs[0]->shape.size()), step(inputs[0]->shape.size(), 1), perm(inputs[0]->shape.size()); std::iota(perm.begin(), perm.end(), 0);
                if (node.op_type == "Transpose") { auto p = ints_attribute(node, "perm"); if (!p.empty()) perm = std::move(p); else std::reverse(perm.begin(), perm.end()); }
                else { auto starts = integer_values(*inputs[1]->host, &node); auto axes = inputs.size() > 3 && inputs[3] && inputs[3]->host ? integer_values(*inputs[3]->host, &node) : std::expected<std::vector<std::int64_t>, Error>(std::vector<std::int64_t>(starts->size())); if (!starts || !axes) return std::unexpected(starts ? axes.error() : starts.error()); if (!(inputs.size() > 3 && inputs[3] && inputs[3]->host)) std::iota(axes->begin(), axes->end(), 0); std::vector<std::int64_t> steps(starts->size(),1); if (inputs.size()>4 && inputs[4]&&inputs[4]->host) { auto s=integer_values(*inputs[4]->host,&node); if(!s)return std::unexpected(s.error()); steps=std::move(*s);} for(size_t i=0;i<starts->size();++i){auto ax=normalize_axis((*axes)[i],inputs[0]->shape.size()); auto st=(*starts)[i]; if(st<0)st+=inputs[0]->shape[ax]; base[ax]=st; step[ax]=steps[i];} }
                const auto width=element_size(outputs[0].type); for(std::size_t id=0;id<*out_count;++id){auto cursor=id; std::vector<std::int64_t> coord(outputs[0].shape.size()); for(size_t a=coord.size();a-->0;){coord[a]=cursor%outputs[0].shape[a];cursor/=outputs[0].shape[a];} std::int64_t src=0; for(size_t a=0;a<coord.size();++a){auto input_axis=node.op_type=="Transpose"?perm[a]:a; src+=(base[input_axis]+coord[a]*step[input_axis])*in_strides[input_axis];} std::memcpy(outputs[0].owner->data()+id*width,inputs[0]->host->bytes.data()+src*width,width);} return outputs;
            }
            if (node.op_type == "Split") {
                const auto axis=normalize_axis(int_attribute(node,"axis").value_or(0),inputs[0]->shape.size()); const auto inner=std::accumulate(inputs[0]->shape.begin()+axis+1,inputs[0]->shape.end(),size_t{1},std::multiplies<>()); const auto outer=std::accumulate(inputs[0]->shape.begin(),inputs[0]->shape.begin()+axis,size_t{1},std::multiplies<>()); const auto width=element_size(inputs[0]->type); size_t axis_offset=0; for(size_t out=0;out<outputs.size();++out){for(size_t oi=0;oi<outer;++oi){auto count=outputs[out].shape[axis]*inner; auto src=(oi*inputs[0]->shape[axis]+axis_offset)*inner; auto dst=oi*count;std::memcpy(outputs[out].owner->data()+dst*width,inputs[0]->host->bytes.data()+src*width,count*width);}axis_offset+=outputs[out].shape[axis];}return outputs;
            }
            return std::unexpected(execution_error("host constant folding is not implemented", &node));
        }

        [[nodiscard]] std::uint32_t u32(const std::int64_t value) { return static_cast<std::uint32_t>(value); }
        [[nodiscard]] std::uint32_t f32(const float value) { return std::bit_cast<std::uint32_t>(value); }

        void put_shape(std::array<std::uint32_t, kParameterWords>& p,
                       const std::size_t base,
                       const std::span<const std::int64_t> values) {
            for (std::size_t i = 0; i < values.size() && i < kKernelRank; ++i) p[base + i] = u32(values[i]);
        }

        struct Dispatch {
            Kernel kernel = Kernel::Elementwise;
            std::uint64_t count = 0;
            std::array<std::uint32_t, 3> workgroups{};
            std::array<DeviceRef, 4> values;
            std::optional<DeviceRef> packed_b;
            std::array<std::uint32_t, kParameterWords> parameters{};
        };

        struct ArenaSlice { VkDeviceSize offset=0,size=0; std::size_t release_at=0; bool pinned=false; };
        struct InputSlot { std::string name; ElementType type; std::vector<std::int64_t> shape; VkDeviceSize staging_offset=0; DeviceRef device; };
        struct OutputSlot { std::string name; ElementType type; std::vector<std::int64_t> shape; VkDeviceSize readback_offset=0; VkDeviceSize bytes=0; };
    } // namespace

    struct ExecutionPlan::Impl {
        const Model* model = nullptr;
        const WeightStore* weights = nullptr;
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VkCommandBuffer command = VK_NULL_HANDLE;
        Buffer arena;
        Buffer literals;
        Buffer metadata;
        Buffer staging;
        Buffer readback;
        Buffer dummy;
        std::vector<Dispatch> dispatches;
        std::vector<InputSlot> input_slots;
        std::vector<OutputSlot> output_slots;
        std::vector<ArenaSlice> slices;
        VkDeviceSize arena_size = 0;
        std::vector<std::byte> literal_bytes;
        std::unordered_map<std::string, Value> values;
        std::unordered_map<std::string, std::size_t> last_use;
        std::unordered_set<std::string> requested;
        std::unordered_set<std::size_t> live_nodes;
        std::unordered_map<std::string, const Node*> producers;
        std::unordered_map<std::string, std::vector<const Node*>> consumers;
        std::unordered_map<const Node*, std::size_t> node_indices;
        std::unordered_set<std::string> nested_captures;
        std::size_t step = 0;
        std::size_t nested_depth = 0;
        VkDeviceSize alignment = 1;

        [[nodiscard]] DeviceRef allocate_bytes(VkDeviceSize bytes,const std::size_t release_at,const bool pinned){bytes=align_up(std::max<VkDeviceSize>(4,bytes),alignment);const bool keep_pinned=pinned||nested_depth!=0;if(!keep_pinned)for(std::size_t i=0;i<slices.size();++i)if(!slices[i].pinned&&slices[i].release_at<step&&slices[i].size>=bytes){slices[i].release_at=release_at;slices[i].pinned=false;return {RefKind::Arena,{},slices[i].offset,bytes,i};}const auto offset=align_up(arena_size,alignment);slices.push_back({offset,bytes,keep_pinned?std::numeric_limits<std::size_t>::max():release_at,keep_pinned});arena_size=offset+bytes;return {RefKind::Arena,{},offset,bytes,slices.size()-1};}

        [[nodiscard]] DeviceRef allocate_arena(const ValueSpec& spec, const std::size_t release_at, const bool pinned) {
            return allocate_bytes(element_count(spec.shape).value_or(0)*4,release_at,pinned);
        }

        void extend_lifetime(DeviceRef& ref,const std::size_t until){if(ref.kind==RefKind::Arena&&ref.slice<slices.size())slices[ref.slice].release_at=std::max(slices[ref.slice].release_at,until);}

        [[nodiscard]] DeviceRef literal(const HostTensor& host) {
            const auto offset=align_up(literal_bytes.size(),alignment); literal_bytes.resize(offset);
            auto count=element_count(host.shape).value_or(0); const auto internal_width=(host.type==ElementType::Bool||host.type==ElementType::Int64)?4:element_size(host.type); const auto bytes=std::max<std::size_t>(4,count*internal_width); literal_bytes.resize(offset+bytes);
            if(host.type==ElementType::Bool){for(size_t i=0;i<count;++i)reinterpret_cast<std::uint32_t*>(literal_bytes.data()+offset)[i]=read_number(host,i)!=0;}
            else if(host.type==ElementType::Int64){for(size_t i=0;i<count;++i)reinterpret_cast<std::int32_t*>(literal_bytes.data()+offset)[i]=static_cast<std::int32_t>(read_number(host,i));}
            else if(!host.bytes.empty())std::memcpy(literal_bytes.data()+offset,host.bytes.data(),host.bytes.size());
            return {RefKind::Literal,{},offset,bytes,{}};
        }

        [[nodiscard]] DeviceRef ensure_device(Value& value) {
            if(value.device.kind!=RefKind::None)return value.device;
            value.device=literal(*value.host);return value.device;
        }

        [[nodiscard]] const Node* producer_of(const std::string& value) const {
            const auto found = producers.find(value);
            return found == producers.end() ? nullptr : found->second;
        }

        [[nodiscard]] const Node* sole_consumer_of(const std::string& value) const {
            const auto found = consumers.find(value);
            return found != consumers.end() && found->second.size() == 1 ? found->second.front() : nullptr;
        }

        [[nodiscard]] std::size_t node_step(const Node* node) const {
            const auto found = node_indices.find(node);
            return found == node_indices.end() ? 0 : found->second + 1;
        }

        void base_params(std::array<std::uint32_t,kParameterWords>& p,const Value& output){p[1]=u32(element_count(output.shape).value_or(0));p[2]=u32(output.shape.size());put_shape(p,16,output.shape);}
        void input_params(std::array<std::uint32_t,kParameterWords>& p,const Value& value,const int slot){const size_t dims=slot==0?24:slot==1?40:56;const size_t strides=slot==0?32:slot==1?48:64;p[3+slot]=u32(value.shape.size());p[6+slot]=u32(static_cast<int>(value.type));put_shape(p,dims,value.shape);put_shape(p,strides,value.strides);p[80+slot]=u32(value.element_offset);}

        [[nodiscard]] std::expected<void,Error> compile_node(const Node& node,const std::size_t node_index);
        [[nodiscard]] std::expected<void,Error> compile_graph(const Graph& graph);
        void tune_matmuls(VulkanRuntime& runtime);
        [[nodiscard]] std::expected<void,Error> finalize(VulkanRuntime& runtime);
    };

    // The GPU lowering is deliberately correctness-first. Views remain strided;
    // every arithmetic node materializes a contiguous result.
    std::expected<void,Error> ExecutionPlan::Impl::compile_node(const Node& node,const std::size_t node_index){
        std::vector<Value*> input_values; std::vector<const Value*> input_const;
        for(const auto& name:node.inputs){if(name.empty()){input_values.push_back(nullptr);input_const.push_back(nullptr);continue;}auto it=values.find(name);if(it==values.end())return std::unexpected(execution_error("input '"+name+"' is unavailable",&node));input_values.push_back(&it->second);input_const.push_back(&it->second);}
        auto specs=infer_outputs(node,input_const);if(!specs)return std::unexpected(specs.error());
        // Fusions are recognized at their final consumer. Keep their original
        // sources alive until then so the arena allocator cannot recycle them
        // for the intermediate dispatches that will be removed.
        if(node.op_type=="Relu"&&!node.outputs.empty()&&!input_values.empty()&&input_values[0]){
            const auto* pad=sole_consumer_of(node.outputs[0]);
            const auto* conv=pad&&!pad->outputs.empty()?sole_consumer_of(pad->outputs[0]):nullptr;
            if(pad&&pad->op_type=="Pad"&&string_attribute(*pad,"mode","constant")=="edge"&&
               conv&&conv->op_type=="Conv"&&int_attribute(*conv,"group").value_or(1)==1)
                extend_lifetime(input_values[0]->device,node_step(conv));
        }
        if(node.op_type=="Mul"&&node.outputs.size()==1){
            const auto* qk=sole_consumer_of(node.outputs[0]);
            const auto* softmax=qk&&qk->outputs.size()==1?sole_consumer_of(qk->outputs[0]):nullptr;
            const auto* attention=softmax&&softmax->outputs.size()==1?sole_consumer_of(softmax->outputs[0]):nullptr;
            if(qk&&qk->op_type=="MatMul"&&softmax&&softmax->op_type=="Softmax"&&
               attention&&attention->op_type=="MatMul"){
                for(auto* input:input_values)
                    if(input&&!input->host)
                        extend_lifetime(input->device,node_step(attention));
            }
        }
        if(node.op_type=="Constant"){auto folded=evaluate_host(node,input_const,*specs);if(!folded)return std::unexpected(folded.error());values[node.outputs[0]]={folded->at(0).type,folded->at(0).shape,contiguous_strides(folded->at(0).shape),0,std::move(folded->at(0)),{}};return {};}
        if(node.op_type=="If"){
            if(!input_values[0]->host)return std::unexpected(execution_error("If condition is not host-visible",&node));const bool condition=read_number(*input_values[0]->host,0)!=0;const auto* attr=find_attribute(node,condition?"then_branch":"else_branch");auto branch=std::get<std::shared_ptr<Graph>>(attr->value);auto compiled=compile_graph(*branch);if(!compiled)return compiled;for(size_t i=0;i<node.outputs.size();++i){auto it=values.find(branch->outputs[i].name);if(it==values.end())return std::unexpected(execution_error("selected If branch output is missing",&node));values[node.outputs[i]]=it->second;extend_lifetime(values[node.outputs[i]].device,last_use[node.outputs[i]]);}return {};}
        if(node.op_type=="Shape"){auto folded=evaluate_host(node,input_const,*specs);if(!folded)return std::unexpected(folded.error());values[node.outputs[0]]={folded->at(0).type,folded->at(0).shape,contiguous_strides(folded->at(0).shape),0,std::move(folded->at(0)),{}};return {};}
        bool all_host=true;for(auto* value:input_values)if(value&&!value->host)all_host=false;size_t folded_count=0;for(const auto& spec:*specs)folded_count+=element_count(spec.shape).value_or(4097);
        static const std::unordered_set<std::string> host_ops{"Shape","Cast","Equal","Add","Sub","Mul","Div","Pow","Mod","Where","Concat","ConstantOfShape","Expand","Gather","Reshape","Squeeze","Unsqueeze","Transpose","Slice","Split","Range","Identity","Round","Neg","Sqrt"};
        if(all_host&&folded_count<=4096&&host_ops.contains(node.op_type)){auto folded=evaluate_host(node,input_const,*specs);if(!folded)return std::unexpected(folded.error());for(size_t i=0;i<node.outputs.size();++i)values[node.outputs[i]]={(*folded)[i].type,(*folded)[i].shape,contiguous_strides((*folded)[i].shape),0,std::move((*folded)[i]),{}};return {};}

        const auto make_output=[&](size_t i){Value value{(*specs)[i].type,(*specs)[i].shape,contiguous_strides((*specs)[i].shape)};const auto release=requested.contains(node.outputs[i])?std::numeric_limits<size_t>::max():last_use[node.outputs[i]];value.device=allocate_arena((*specs)[i],release,requested.contains(node.outputs[i]));return value;};
        static const std::unordered_set<std::string> view_ops{"Identity","Reshape","Squeeze","Unsqueeze","Transpose","Expand","Slice","Split"};
        if(node.op_type=="Squeeze"&&node.inputs.size()>=1&&node.outputs.size()==1){
            const auto* split=producer_of(node.inputs[0]);
            if(split&&split->op_type=="Split"&&split->outputs.size()==3){
                std::array<const Node*,3> squeezes{};
                bool exact=true;
                for(size_t i=0;i<squeezes.size();++i){
                    squeezes[i]=sole_consumer_of(split->outputs[i]);
                    exact=exact&&squeezes[i]&&squeezes[i]->op_type=="Squeeze"&&
                          squeezes[i]->inputs.size()>=1&&squeezes[i]->inputs[0]==split->outputs[i]&&
                          squeezes[i]->outputs.size()==1;
                }
                const auto latest=exact?*std::ranges::max_element(squeezes,[&](const Node* lhs,const Node* rhs){return node_indices[lhs]<node_indices[rhs];}):nullptr;
                if(exact&&latest==&node&&specs->size()==1&&(*specs)[0].type==ElementType::Float32&&
                   (*specs)[0].shape.size()==4){
                    std::array<Value*,3> source_views{};
                    std::array<Value*,3> packed_outputs{};
                    Value current{(*specs)[0].type,(*specs)[0].shape,contiguous_strides((*specs)[0].shape)};
                    current.device=allocate_arena((*specs)[0],last_use[node.outputs[0]],requested.contains(node.outputs[0]));
                    size_t current_slot=0;
                    for(size_t i=0;i<3&&exact;++i){
                        const auto source=values.find(split->outputs[i]);
                        exact=source!=values.end()&&source->second.type==ElementType::Float32&&
                              source->second.shape.size()==5&&source->second.shape[0]==1&&
                              std::ranges::equal(std::span(source->second.shape).subspan(1),(*specs)[0].shape)&&
                              source->second.strides.size()==5;
                        if(!exact)break;
                        source_views[i]=&source->second;
                        if(squeezes[i]==&node){packed_outputs[i]=&current;current_slot=i;}
                        else{
                            const auto output=values.find(squeezes[i]->outputs[0]);
                            exact=output!=values.end()&&output->second.shape==(*specs)[0].shape&&
                                  output->second.strides==contiguous_strides(output->second.shape)&&
                                  output->second.element_offset==0&&output->second.producer_dispatch;
                            if(exact)packed_outputs[i]=&output->second;
                        }
                    }
                    std::array<size_t,2> old_dispatches{};size_t old_count=0;
                    for(size_t i=0;i<3&&exact;++i)if(i!=current_slot)
                        old_dispatches[old_count++]=*packed_outputs[i]->producer_dispatch;
                    std::ranges::sort(old_dispatches);
                    exact=exact&&old_count==2&&dispatches.size()>=2&&
                          old_dispatches[0]==dispatches.size()-2&&old_dispatches[1]==dispatches.size()-1&&
                          dispatches[old_dispatches[0]].kernel==Kernel::Transform&&
                          dispatches[old_dispatches[1]].kernel==Kernel::Transform;
                    for(size_t i=1;i<3&&exact;++i)
                        exact=source_views[i]->device.kind==source_views[0]->device.kind&&
                              source_views[i]->device.slice==source_views[0]->device.slice&&
                              source_views[i]->device.offset==source_views[0]->device.offset&&
                              source_views[i]->strides==source_views[0]->strides;
                    if(exact){
                        dispatches.resize(dispatches.size()-2);
                        Dispatch pack;pack.kernel=Kernel::QkvPack;
                        pack.count=element_count((*specs)[0].shape).value_or(0);
                        pack.parameters[1]=u32(pack.count);
                        for(size_t axis=0;axis<4;++axis)pack.parameters[80+axis]=u32((*specs)[0].shape[axis]);
                        for(size_t i=0;i<3;++i){pack.parameters[84+i]=u32(source_views[i]->element_offset);pack.parameters[91+i]=u32(packed_outputs[i]->element_offset);}
                        for(size_t axis=0;axis<4;++axis)pack.parameters[87+axis]=u32(source_views[0]->strides[axis+1]);
                        pack.values={source_views[0]->device,packed_outputs[0]->device,
                                     packed_outputs[1]->device,packed_outputs[2]->device};
                        const auto producer=dispatches.size();dispatches.push_back(std::move(pack));
                        for(auto* output:packed_outputs)output->producer_dispatch=producer;
                        values[node.outputs[0]]=std::move(current);
                        return {};
                    }
                }
            }
        }
        if(view_ops.contains(node.op_type)){
            if(input_values.empty()||!input_values[0])return std::unexpected(execution_error("view has no data input",&node));
            if(input_values[0]->host){auto folded=evaluate_host(node,input_const,*specs);if(!folded)return std::unexpected(folded.error());for(size_t i=0;i<node.outputs.size();++i)values[node.outputs[i]]={(*folded)[i].type,(*folded)[i].shape,contiguous_strides((*folded)[i].shape),0,std::move((*folded)[i]),{}};return {};}
            auto base=*input_values[0];
            if(node.op_type=="Identity"){values[node.outputs[0]]=base;extend_lifetime(values[node.outputs[0]].device,last_use[node.outputs[0]]);return {};}
            if(node.op_type=="Reshape"||node.op_type=="Squeeze"||node.op_type=="Unsqueeze"){
                std::optional<std::size_t> material_dispatch;
                if(base.element_offset!=0||base.strides!=contiguous_strides(base.shape)){
                    ValueSpec material_spec{base.type,base.shape};
                    Value material{base.type,base.shape,contiguous_strides(base.shape)};
                    material.device=allocate_arena(material_spec,last_use[node.outputs[0]],requested.contains(node.outputs[0]));
                    Dispatch copy;copy.kernel=Kernel::Transform;copy.count=element_count(base.shape).value_or(0);copy.parameters[0]=0;copy.parameters[1]=u32(copy.count);copy.parameters[2]=u32(base.shape.size());put_shape(copy.parameters,16,base.shape);put_shape(copy.parameters,24,base.strides);copy.parameters[80]=u32(base.element_offset);copy.values[0]=base.device;copy.values[3]=material.device;material_dispatch=dispatches.size();dispatches.push_back(std::move(copy));base=std::move(material);
                }
                base.shape=(*specs)[0].shape;base.strides=contiguous_strides(base.shape);base.element_offset=0;base.producer_dispatch=material_dispatch;values[node.outputs[0]]=base;extend_lifetime(values[node.outputs[0]].device,last_use[node.outputs[0]]);return {};
            }
            if(node.op_type=="Transpose"){auto perm=ints_attribute(node,"perm");if(perm.empty()){perm.resize(base.shape.size());std::iota(perm.rbegin(),perm.rend(),0);}Value out=base;out.producer_dispatch.reset();out.shape=(*specs)[0].shape;for(size_t i=0;i<perm.size();++i)out.strides[i]=base.strides[perm[i]];values[node.outputs[0]]=out;extend_lifetime(values[node.outputs[0]].device,last_use[node.outputs[0]]);return {};}
            if(node.op_type=="Expand"){Value out=base;out.producer_dispatch.reset();const auto target=(*specs)[0].shape;std::vector<int64_t> strides(target.size(),0);for(size_t t=0;t<base.shape.size();++t){auto dst=target.size()-base.shape.size()+t;if(base.shape[t]!=1)strides[dst]=base.strides[t];}out.shape=target;out.strides=std::move(strides);values[node.outputs[0]]=out;extend_lifetime(values[node.outputs[0]].device,last_use[node.outputs[0]]);return {};}
            if(node.op_type=="Slice"){auto starts=integer_values(*input_values[1]->host,&node);auto axes=input_values.size()>3&&input_values[3]&&input_values[3]->host?integer_values(*input_values[3]->host,&node):std::expected<std::vector<int64_t>,Error>(std::vector<int64_t>(starts->size()));if(!(input_values.size()>3&&input_values[3]&&input_values[3]->host))std::iota(axes->begin(),axes->end(),0);std::vector<int64_t> steps(starts->size(),1);if(input_values.size()>4&&input_values[4]&&input_values[4]->host)steps=*integer_values(*input_values[4]->host,&node);Value out=base;out.producer_dispatch.reset();out.shape=(*specs)[0].shape;for(size_t i=0;i<starts->size();++i){auto axis=normalize_axis((*axes)[i],base.shape.size());auto start=(*starts)[i];if(start<0)start+=base.shape[axis];start=std::clamp(start,int64_t{0},base.shape[axis]-1);out.element_offset+=start*base.strides[axis];out.strides[axis]*=steps[i];}values[node.outputs[0]]=out;extend_lifetime(values[node.outputs[0]].device,last_use[node.outputs[0]]);return {};}
            if(node.op_type=="Split"){auto axis=normalize_axis(int_attribute(node,"axis").value_or(0),base.shape.size());int64_t offset=0;for(size_t i=0;i<node.outputs.size();++i){Value out=base;out.producer_dispatch.reset();out.shape=(*specs)[i].shape;out.element_offset+=offset*base.strides[axis];offset+=out.shape[axis];values[node.outputs[i]]=out;extend_lifetime(values[node.outputs[i]].device,last_use[node.outputs[i]]);}return {};}
        }

        std::vector<Value> outputs;for(size_t i=0;i<specs->size();++i)outputs.push_back(make_output(i));
        Dispatch dispatch;dispatch.count=element_count(outputs[0].shape).value_or(0);base_params(dispatch.parameters,outputs[0]);dispatch.parameters[9]=u32(static_cast<int>(outputs[0].type));
        const auto ref=[&](size_t i)->DeviceRef{if(i>=input_values.size()||!input_values[i])return {};return ensure_device(*input_values[i]);};
        const auto finish=[&](Dispatch d){const auto producer=dispatches.size();d.values[3]=outputs[0].device;dispatches.push_back(std::move(d));outputs[0].producer_dispatch=producer;for(size_t i=0;i<outputs.size();++i)values[node.outputs[i]]=std::move(outputs[i]);};
        static const std::unordered_map<std::string,uint32_t> element_ops{{"Add",1},{"Sub",2},{"Mul",3},{"Div",4},{"Pow",5},{"Mod",6},{"Equal",7},{"Where",8},{"Neg",9},{"Exp",10},{"Erf",11},{"Reciprocal",12},{"Relu",13},{"Sqrt",14},{"Sigmoid",15},{"Round",16},{"Clip",17},{"Cast",18}};
        if(node.op_type=="Add"&&node.inputs.size()==2){
            const auto producer_of=[&](const std::string& value)->const Node*{
                const auto it=std::ranges::find_if(model->graph.nodes,[&](const Node& candidate){return std::ranges::find(candidate.outputs,value)!=candidate.outputs.end();});
                return it==model->graph.nodes.end()?nullptr:&*it;
            };
            for(size_t mul_slot=0;mul_slot<2;++mul_slot){
                const auto* mul=producer_of(node.inputs[mul_slot]);
                if(!mul||mul->op_type!="Mul"||mul->inputs.size()!=2)continue;
                const auto bias_it=values.find(node.inputs[1-mul_slot]);
                if(bias_it==values.end())continue;
                for(size_t div_slot=0;div_slot<2;++div_slot){
                    const auto* div=producer_of(mul->inputs[div_slot]);
                    if(!div||div->op_type!="Div"||div->inputs.size()!=2)continue;
                    const auto scale_it=values.find(mul->inputs[1-div_slot]);
                    const auto* sub=producer_of(div->inputs[0]);
                    const auto* sqrt=producer_of(div->inputs[1]);
                    if(scale_it==values.end()||!sub||sub->op_type!="Sub"||sub->inputs.size()!=2||
                       !sqrt||sqrt->op_type!="Sqrt"||sqrt->inputs.size()!=1)continue;
                    const auto* mean1=producer_of(sub->inputs[1]);
                    const auto* epsilon_add=producer_of(sqrt->inputs[0]);
                    if(!mean1||mean1->op_type!="ReduceMean"||mean1->inputs.empty()||mean1->inputs[0]!=sub->inputs[0]||
                       !epsilon_add||epsilon_add->op_type!="Add"||epsilon_add->inputs.size()!=2)continue;
                    const Node* mean2=nullptr;const Value* epsilon=nullptr;
                    for(size_t mean_slot=0;mean_slot<2;++mean_slot){
                        const auto* candidate=producer_of(epsilon_add->inputs[mean_slot]);
                        if(candidate&&candidate->op_type=="ReduceMean"){
                            mean2=candidate;
                            const auto epsilon_it=values.find(epsilon_add->inputs[1-mean_slot]);
                            if(epsilon_it!=values.end())epsilon=&epsilon_it->second;
                            break;
                        }
                    }
                    if(!mean2||mean2->inputs.empty()||!epsilon||!epsilon->host)continue;
                    const auto* pow=producer_of(mean2->inputs[0]);
                    if(!pow||pow->op_type!="Pow"||pow->inputs.size()!=2||pow->inputs[0]!=sub->outputs[0])continue;
                    const auto exponent_it=values.find(pow->inputs[1]);
                    const auto source_it=values.find(sub->inputs[0]);
                    if(exponent_it==values.end()||!exponent_it->second.host||source_it==values.end())continue;
                    auto& source=source_it->second;auto& scale=scale_it->second;auto& bias=bias_it->second;
                    if(source.type!=ElementType::Float32||scale.type!=ElementType::Float32||bias.type!=ElementType::Float32||
                       source.shape.empty()||source.shape!=outputs[0].shape||source.element_offset<0||
                       source.strides!=contiguous_strides(source.shape)||
                       element_count(scale.shape).value_or(0)!=static_cast<size_t>(source.shape.back())||
                       element_count(bias.shape).value_or(0)!=static_cast<size_t>(source.shape.back())||
                       read_number(*exponent_it->second.host,0)!=2.0)continue;
                    const auto axes1=ints_attribute(*mean1,"axes");const auto axes2=ints_attribute(*mean2,"axes");
                    if(axes1.size()!=1||axes2.size()!=1||normalize_axis(axes1[0],source.shape.size())!=source.shape.size()-1||
                       normalize_axis(axes2[0],source.shape.size())!=source.shape.size()-1)continue;
                    const std::array<const Node*,8> chain{mean1,sub,pow,mean2,epsilon_add,sqrt,div,mul};
                    if(dispatches.size()<chain.size())continue;
                    bool consecutive=true;
                    for(size_t i=0;i<chain.size();++i){
                        const auto value_it=values.find(chain[i]->outputs[0]);
                        consecutive=consecutive&&value_it!=values.end()&&value_it->second.producer_dispatch&&
                                    *value_it->second.producer_dispatch==dispatches.size()-chain.size()+i&&
                                    last_use[chain[i]->outputs[0]]<=step;
                    }
                    if(!consecutive)continue;
                    dispatches.resize(dispatches.size()-chain.size());
                    Dispatch fused;fused.kernel=Kernel::LayerNorm;
                    const auto width=u32(source.shape.back());
                    fused.count=element_count(source.shape).value_or(0)/width;
                    fused.parameters[1]=u32(fused.count);fused.parameters[80]=width;
                    fused.parameters[81]=u32(source.element_offset);fused.parameters[82]=u32(scale.element_offset);
                    fused.parameters[83]=u32(bias.element_offset);
                    fused.parameters[84]=f32(static_cast<float>(read_number(*epsilon->host,0)));
                    fused.values={ensure_device(source),ensure_device(scale),ensure_device(bias),outputs[0].device};
                    outputs[0].producer_dispatch=dispatches.size();dispatches.push_back(std::move(fused));
                    values[node.outputs[0]]=std::move(outputs[0]);return {};
                }
            }
        }
        if(node.op_type=="Add"&&input_values.size()==2){
            for(size_t dynamic_slot=0;dynamic_slot<2;++dynamic_slot){
                auto* dynamic=input_values[dynamic_slot];auto* bias=input_values[1-dynamic_slot];
                if(!dynamic||!bias||!dynamic->producer_dispatch||dynamic->element_offset!=0||
                   dynamic->strides!=contiguous_strides(dynamic->shape)||dynamic->shape!=outputs[0].shape||
                   dynamic->shape.empty()||bias->shape.empty()||bias->type!=ElementType::Float32||
                   element_count(bias->shape).value_or(0)!=static_cast<size_t>(dynamic->shape.back())||
                   last_use[node.inputs[dynamic_slot]]!=step)continue;
                auto& producer=dispatches[*dynamic->producer_dispatch];
                if((producer.kernel!=Kernel::MatMul&&producer.kernel!=Kernel::MatMulTiled&&
                    producer.kernel!=Kernel::MatMulSmallK&&producer.kernel!=Kernel::MatMulLargeN&&
                    producer.kernel!=Kernel::MatMulPackedB)||producer.parameters[89]!=0)continue;
                producer.values[2]=ensure_device(*bias);producer.parameters[88]=f32(1.0f);producer.parameters[89]=1;
                producer.parameters[90]=u32(bias->element_offset);producer.parameters[91]=1;
                producer.parameters[92]=u32(bias->strides.back());
                Value fused=*dynamic;fused.shape=outputs[0].shape;fused.strides=outputs[0].strides;
                extend_lifetime(fused.device,last_use[node.outputs[0]]);values[node.outputs[0]]=std::move(fused);return {};
            }
        }
        if(auto it=element_ops.find(node.op_type);it!=element_ops.end()){dispatch.kernel=Kernel::Elementwise;dispatch.parameters[0]=it->second;const auto set_index_mode=[&](const Value& value,const size_t slot){uint32_t mode=0;if(element_count(value.shape).value_or(0)==1)mode=2;else if(value.shape==outputs[0].shape&&value.strides==contiguous_strides(value.shape))mode=1;dispatch.parameters[10]|=mode<<u32(slot*2);};if(node.op_type=="Where"){input_params(dispatch.parameters,*input_values[1],0);input_params(dispatch.parameters,*input_values[2],1);input_params(dispatch.parameters,*input_values[0],2);set_index_mode(*input_values[1],0);set_index_mode(*input_values[2],1);set_index_mode(*input_values[0],2);}else for(size_t i=0;i<3;++i)if(i<input_values.size()&&input_values[i]){input_params(dispatch.parameters,*input_values[i],i);set_index_mode(*input_values[i],i);}dispatch.values={ref(node.op_type=="Where"?1:0),ref(node.op_type=="Where"?2:1),ref(node.op_type=="Where"?0:2),{}};if(node.op_type=="Mod")dispatch.parameters[83]=u32(int_attribute(node,"fmod").value_or(0));if(node.op_type=="Clip"){dispatch.parameters[83]=input_values.size()>1&&input_values[1];dispatch.parameters[84]=input_values.size()>2&&input_values[2];}finish(std::move(dispatch));return {};}
        if(node.op_type=="ConstantOfShape"){dispatch.kernel=Kernel::Transform;dispatch.parameters[0]=1;float scalar=0;if(const auto* a=find_attribute(node,"value"))scalar=static_cast<float>(read_number(host_tensor(std::get<TensorData>(a->value)),0));dispatch.parameters[80]=f32(scalar);finish(std::move(dispatch));return {};}
        if(node.op_type=="Concat"){dispatch.kernel=Kernel::Transform;dispatch.parameters[0]=2;auto axis=normalize_axis(*int_attribute(node,"axis"),outputs[0].shape.size());auto out_strides=outputs[0].strides;int64_t axis_offset=0;for(auto* input:input_values){Dispatch part=dispatch;part.count=element_count(input->shape).value_or(0);part.parameters[1]=u32(part.count);part.parameters[2]=u32(input->shape.size());put_shape(part.parameters,16,input->shape);put_shape(part.parameters,24,input->strides);put_shape(part.parameters,32,out_strides);part.parameters[80]=u32(input->element_offset);part.parameters[81]=u32(axis);part.parameters[82]=u32(axis_offset);part.values[0]=ensure_device(*input);part.values[3]=outputs[0].device;dispatches.push_back(std::move(part));axis_offset+=input->shape[axis];}values[node.outputs[0]]=std::move(outputs[0]);return {};}
        if(node.op_type=="Pad"){dispatch.kernel=Kernel::Transform;dispatch.parameters[0]=input_values[0]->shape.size()==4?7:3;auto pads=*integer_values(*input_values[1]->host,&node);put_shape(dispatch.parameters,24,input_values[0]->strides);for(size_t i=0;i<input_values[0]->shape.size();++i){dispatch.parameters[40+i]=u32(pads[i]);dispatch.parameters[48+i]=u32(input_values[0]->shape[i]);}float pad=0;if(input_values.size()>2&&input_values[2]&&input_values[2]->host)pad=static_cast<float>(read_number(*input_values[2]->host,0));dispatch.parameters[80]=f32(pad);auto mode=string_attribute(node,"mode","constant");dispatch.parameters[81]=mode=="reflect"?1:mode=="edge"?2:0;dispatch.parameters[82]=u32(input_values[0]->element_offset);dispatch.values[0]=ref(0);finish(std::move(dispatch));return {};}
        if(node.op_type=="Gather"){dispatch.kernel=Kernel::Transform;dispatch.parameters[0]=4;auto axis=normalize_axis(int_attribute(node,"axis").value_or(0),input_values[0]->shape.size());dispatch.parameters[3]=u32(input_values[0]->shape.size());dispatch.parameters[4]=u32(input_values[1]->shape.size());put_shape(dispatch.parameters,24,input_values[0]->strides);put_shape(dispatch.parameters,40,input_values[1]->strides);put_shape(dispatch.parameters,56,input_values[0]->shape);dispatch.parameters[80]=u32(axis);dispatch.parameters[81]=u32(input_values[0]->element_offset);dispatch.parameters[82]=u32(input_values[1]->element_offset);dispatch.values[0]=ref(0);dispatch.values[1]=ref(1);finish(std::move(dispatch));return {};}
        if(node.op_type=="Resize"){dispatch.kernel=Kernel::Transform;auto mode=string_attribute(node,"mode","nearest");const bool nchw_linear=mode=="linear"&&input_values[0]->shape.size()==4&&outputs[0].shape.size()==4&&input_values[0]->shape[0]==outputs[0].shape[0]&&input_values[0]->shape[1]==outputs[0].shape[1];dispatch.parameters[0]=nchw_linear?6:5;put_shape(dispatch.parameters,24,input_values[0]->strides);put_shape(dispatch.parameters,48,input_values[0]->shape);for(size_t i=0;i<outputs[0].shape.size();++i)dispatch.parameters[40+i]=f32(static_cast<float>(outputs[0].shape[i])/input_values[0]->shape[i]);dispatch.parameters[80]=mode=="linear"?1:mode=="cubic"?2:0;auto transform=string_attribute(node,"coordinate_transformation_mode","half_pixel");dispatch.parameters[81]=transform=="half_pixel"?1:transform=="align_corners"?2:transform=="pytorch_half_pixel"?3:0;auto nearest=string_attribute(node,"nearest_mode","round_prefer_floor");dispatch.parameters[82]=nearest=="floor"?0:nearest=="ceil"?1:nearest=="round_prefer_floor"?2:3;dispatch.parameters[83]=u32(input_values[0]->element_offset);dispatch.parameters[84]=f32(float_attribute(node,"cubic_coeff_a",-0.75f));dispatch.parameters[85]=u32(int_attribute(node,"exclude_outside").value_or(0));dispatch.values[0]=ref(0);finish(std::move(dispatch));return {};}
        if(node.op_type=="MatMul"||node.op_type=="Gemm"){
            if(node.op_type=="MatMul"&&node.inputs.size()==2){
                const auto* softmax=producer_of(node.inputs[0]);
                const auto* qk=softmax&&!softmax->inputs.empty()?producer_of(softmax->inputs[0]):nullptr;
                const auto scalar_mul=[&](const Node* mul)->std::optional<std::pair<Value*,float>>{
                    if(!mul||mul->op_type!="Mul"||mul->inputs.size()!=2)return std::nullopt;
                    for(size_t scalar_slot=0;scalar_slot<2;++scalar_slot){
                        const auto scalar=values.find(mul->inputs[scalar_slot]);
                        const auto dynamic=values.find(mul->inputs[1-scalar_slot]);
                        if(scalar==values.end()||dynamic==values.end()||!scalar->second.host||
                           element_count(scalar->second.shape).value_or(0)!=1||dynamic->second.host)continue;
                        return std::pair<Value*,float>{&dynamic->second,
                            static_cast<float>(read_number(*scalar->second.host,0))};
                    }
                    return std::nullopt;
                };
                if(softmax&&softmax->op_type=="Softmax"&&softmax->outputs.size()==1&&
                   sole_consumer_of(softmax->outputs[0])==&node&&qk&&qk->op_type=="MatMul"&&
                   qk->inputs.size()==2&&qk->outputs.size()==1&&sole_consumer_of(qk->outputs[0])==softmax){
                    const auto* q_mul=producer_of(qk->inputs[0]);
                    const auto* k_mul=producer_of(qk->inputs[1]);
                    auto q_scaled=scalar_mul(q_mul);auto k_scaled=scalar_mul(k_mul);
                    // Locate the dynamic K input directly; the scalar may be in either Mul slot.
                    std::string k_dynamic_name;
                    if(k_mul)for(const auto& name:k_mul->inputs){const auto found=values.find(name);if(found!=values.end()&&!found->second.host)k_dynamic_name=name;}
                    const auto* k_transpose=k_dynamic_name.empty()?nullptr:producer_of(k_dynamic_name);
                    auto perm=k_transpose&&k_transpose->op_type=="Transpose"?ints_attribute(*k_transpose,"perm"):std::vector<int64_t>{};
                    auto k_source=k_transpose&&!k_transpose->inputs.empty()?values.find(k_transpose->inputs[0]):values.end();
                    auto v_source=values.find(node.inputs[1]);
                    const auto softmax_axis=normalize_axis(int_attribute(*softmax,"axis").value_or(-1),
                                                           input_values[0]->shape.size());
                    const auto attention_scale=q_scaled&&k_scaled?q_scaled->second*k_scaled->second:0.0f;
                    auto nonnegative_strides=[](const Value& value){return value.strides.size()==4&&
                        std::ranges::all_of(value.strides,[](const auto stride){return stride>=0;});};
                    if(q_scaled&&k_scaled&&k_transpose&&perm==std::vector<int64_t>{0,1,3,2}&&
                       k_source!=values.end()&&v_source!=values.end()&&softmax_axis==3&&
                       q_scaled->first->type==ElementType::Float32&&k_source->second.type==ElementType::Float32&&
                       v_source->second.type==ElementType::Float32&&outputs[0].type==ElementType::Float32&&
                       q_scaled->first->shape.size()==4&&q_scaled->first->shape==k_source->second.shape&&
                       q_scaled->first->shape==v_source->second.shape&&q_scaled->first->shape==outputs[0].shape&&
                       // The per-row kernel is bounded to short sequences;
                       // synchronization dominates MoGe's long sequences.
                       outputs[0].shape[2]<=kMaximumPortableFusedAttentionTokens&&
                       nonnegative_strides(*q_scaled->first)&&nonnegative_strides(k_source->second)&&
                       nonnegative_strides(v_source->second)&&
                       std::ranges::all_of(outputs[0].shape,[](const auto extent){return extent>0;})&&
                       outputs[0].shape[3]<=64&&
                       std::isfinite(q_scaled->second)&&std::isfinite(k_scaled->second)&&
                       std::isfinite(attention_scale)&&
                       q_mul->outputs.size()==1&&k_mul->outputs.size()==1&&
                       sole_consumer_of(q_mul->outputs[0])==qk&&sole_consumer_of(k_mul->outputs[0])==qk&&
                       last_use[q_mul->outputs[0]]==node_step(qk)&&
                       last_use[k_mul->outputs[0]]==node_step(qk)&&
                       last_use[qk->outputs[0]]==node_step(softmax)&&
                       last_use[softmax->outputs[0]]==node_step(&node)&&
                       !nested_captures.contains(q_mul->outputs[0])&&
                       !nested_captures.contains(k_mul->outputs[0])&&
                       !nested_captures.contains(qk->outputs[0])&&
                       !nested_captures.contains(softmax->outputs[0])){
                        const auto q_value=values.find(q_mul->outputs[0]);const auto k_value=values.find(k_mul->outputs[0]);
                        const auto score=values.find(qk->outputs[0]);const auto probabilities=values.find(softmax->outputs[0]);
                        if(q_value!=values.end()&&k_value!=values.end()&&score!=values.end()&&probabilities!=values.end()&&
                           q_value->second.producer_dispatch&&k_value->second.producer_dispatch&&
                           score->second.producer_dispatch&&probabilities->second.producer_dispatch&&dispatches.size()>=4){
                            std::array<size_t,2> scales{*q_value->second.producer_dispatch,*k_value->second.producer_dispatch};
                            std::ranges::sort(scales);
                            if(scales[0]==dispatches.size()-4&&scales[1]==dispatches.size()-3&&
                               *score->second.producer_dispatch==dispatches.size()-2&&
                               *probabilities->second.producer_dispatch==dispatches.size()-1){
                                dispatches.resize(dispatches.size()-4);
                                Dispatch attention;attention.kernel=Kernel::Attention;
                                const auto& shape=outputs[0].shape;
                                attention.count=static_cast<uint64_t>(shape[0])*shape[1]*shape[2];
                                attention.parameters[1]=u32(attention.count);
                                for(size_t axis=0;axis<4;++axis)attention.parameters[80+axis]=u32(shape[axis]);
                                attention.parameters[84]=f32(attention_scale);
                                attention.parameters[85]=u32(q_scaled->first->element_offset);
                                attention.parameters[86]=u32(k_source->second.element_offset);
                                attention.parameters[87]=u32(v_source->second.element_offset);
                                attention.parameters[88]=u32(outputs[0].element_offset);
                                for(size_t axis=0;axis<4;++axis){
                                    attention.parameters[89+axis]=u32(q_scaled->first->strides[axis]);
                                    attention.parameters[93+axis]=u32(k_source->second.strides[axis]);
                                    attention.parameters[97+axis]=u32(v_source->second.strides[axis]);
                                    attention.parameters[101+axis]=u32(outputs[0].strides[axis]);
                                }
                                attention.values={q_scaled->first->device,k_source->second.device,
                                                  ensure_device(v_source->second),{}};
                                finish(std::move(attention));return {};
                            }
                        }
                    }
                }
            }
            auto a=input_values[0],b=input_values[1];
            dispatch.parameters[3]=u32(a->shape.size());dispatch.parameters[4]=u32(b->shape.size());
            put_shape(dispatch.parameters,24,a->shape);put_shape(dispatch.parameters,32,a->strides);
            put_shape(dispatch.parameters,40,b->shape);put_shape(dispatch.parameters,48,b->strides);
            bool ta=node.op_type=="Gemm"&&int_attribute(node,"transA").value_or(0);
            bool tb=node.op_type=="Gemm"&&int_attribute(node,"transB").value_or(0);
            auto ashape=a->shape,bshape=b->shape;
            if(ashape.size()==1)ashape.insert(ashape.begin(),1);
            if(bshape.size()==1)bshape.push_back(1);
            const auto n_size=u32(bshape.back());
            const auto m_size=u32(ashape[ashape.size()-2]);
            dispatch.parameters[80]=n_size;dispatch.parameters[81]=m_size;
            const auto k_size=u32(ta?ashape[ashape.size()-2]:ashape.back());
            dispatch.parameters[82]=k_size;
            dispatch.parameters[83]=u32(a->element_offset);dispatch.parameters[84]=u32(b->element_offset);
            dispatch.parameters[85]=ta;dispatch.parameters[86]=tb;
            dispatch.parameters[87]=f32(node.op_type=="Gemm"?float_attribute(node,"alpha",1):1);
            dispatch.parameters[88]=f32(node.op_type=="Gemm"?float_attribute(node,"beta",1):0);
            if(node.op_type=="Gemm"&&input_values.size()>2&&input_values[2]){
                dispatch.parameters[89]=1;dispatch.parameters[90]=u32(input_values[2]->element_offset);
                dispatch.parameters[91]=input_values[2]->shape.size()==1;
                dispatch.parameters[92]=u32(input_values[2]->strides[0]);
                dispatch.parameters[93]=u32(input_values[2]->shape.size()>1?input_values[2]->strides[1]:0);
            }
            const bool tiled=a->shape.size()>=2&&b->shape.size()>=2&&outputs[0].shape.size()>=2;
            // Small-K products are accuracy-sensitive and conversion-bound. Keep
            // them on the exact FP32 path.
            const bool small_k=tiled&&k_size<=64u;
            dispatch.kernel=tiled?(small_k?Kernel::MatMulSmallK:Kernel::MatMulTiled):Kernel::MatMul;
            if(tiled){
                const auto batches=dispatch.count/(static_cast<uint64_t>(m_size)*n_size);
                const auto tile_y=small_k?128u:64u;
                dispatch.workgroups={(n_size+63u)/64u,(m_size+tile_y-1u)/tile_y,u32(batches)};
                const bool contiguous_a=a->shape.size()==outputs[0].shape.size()&&!ta&&
                    a->strides[a->strides.size()-2]==static_cast<int64_t>(k_size)&&a->strides.back()==1;
                const auto packed=weights->packed_tensors.find(node.inputs[1]);
                if(!small_k&&node.op_type=="MatMul"&&contiguous_a&&b->shape.size()==2&&!tb&&
                   m_size>=64u&&n_size>=128u&&k_size>64u&&packed!=weights->packed_tensors.end()){
                    DeviceRef packed_ref;packed_ref.kind=RefKind::Weight;packed_ref.direct=packed->second.binding;
                    packed_ref.range=packed->second.binding.range;dispatch.packed_b=packed_ref;
                    dispatch.parameters[94]=(n_size+63u)/64u;
                    dispatch.parameters[95]=(k_size+15u)/16u;
                }
            }
            dispatch.values={ref(0),ref(1),ref(2),{}};
            finish(std::move(dispatch));return {};
        }
        if(node.op_type=="Conv"||node.op_type=="ConvTranspose"){
            auto strides=ints_attribute(node,"strides");if(strides.empty())strides={1,1};
            auto dil=ints_attribute(node,"dilations");if(dil.empty())dil={1,1};
            auto pads=ints_attribute(node,"pads");if(pads.empty())pads={0,0,0,0};
            const auto groups=u32(int_attribute(node,"group").value_or(1));
            Value* convolution_input=input_values[0];
            bool fused_edge_relu=false;
            if(node.op_type=="Conv"&&groups==1&&input_values[1]->shape.size()==4&&
               input_values[1]->shape[2]==3&&input_values[1]->shape[3]==3&&
               strides==std::vector<int64_t>{1,1}&&dil==std::vector<int64_t>{1,1}&&
               std::ranges::all_of(pads,[](const auto value){return value==0;})){
                const auto* pad_node=producer_of(node.inputs[0]);
                const auto* relu_node=pad_node&&!pad_node->inputs.empty()?producer_of(pad_node->inputs[0]):nullptr;
                auto source=relu_node&&!relu_node->inputs.empty()?values.find(relu_node->inputs[0]):values.end();
                auto pad_control=pad_node&&pad_node->inputs.size()>1?values.find(pad_node->inputs[1]):values.end();
                if(pad_node&&pad_node->op_type=="Pad"&&string_attribute(*pad_node,"mode","constant")=="edge"&&
                   pad_node->outputs.size()==1&&sole_consumer_of(pad_node->outputs[0])==&node&&
                   relu_node&&relu_node->op_type=="Relu"&&relu_node->outputs.size()==1&&
                   sole_consumer_of(relu_node->outputs[0])==pad_node&&source!=values.end()&&
                   source->second.type==ElementType::Float32&&source->second.shape.size()==4&&
                   pad_control!=values.end()&&pad_control->second.host){
                    auto edge_pads=integer_values(*pad_control->second.host,&node);
                    const std::vector<int64_t> expected_pads{0,0,1,1,0,0,1,1};
                    const auto relu_value=values.find(relu_node->outputs[0]);
                    const auto pad_value=values.find(pad_node->outputs[0]);
                    if(edge_pads&&*edge_pads==expected_pads&&
                       source->second.shape[0]==outputs[0].shape[0]&&
                       source->second.shape[1]==input_values[1]->shape[1]&&
                       source->second.shape[2]==outputs[0].shape[2]&&
                       source->second.shape[3]==outputs[0].shape[3]&&
                       last_use[relu_node->outputs[0]]==node_step(pad_node)&&
                       last_use[pad_node->outputs[0]]==node_step(&node)&&
                       !nested_captures.contains(relu_node->outputs[0])&&
                       !nested_captures.contains(pad_node->outputs[0])&&
                       relu_value!=values.end()&&pad_value!=values.end()&&
                       relu_value->second.producer_dispatch&&pad_value->second.producer_dispatch&&
                       dispatches.size()>=2&&*relu_value->second.producer_dispatch==dispatches.size()-2&&
                       *pad_value->second.producer_dispatch==dispatches.size()-1){
                        dispatches.resize(dispatches.size()-2);
                        convolution_input=&source->second;
                        fused_edge_relu=true;
                    }
                }
            }
            const auto ow=u32(outputs[0].shape[3]),oh=u32(outputs[0].shape[2]),oc=u32(outputs[0].shape[1]);
            dispatch.parameters[80]=ow;dispatch.parameters[81]=oh;dispatch.parameters[82]=oc;
            dispatch.parameters[83]=groups;dispatch.parameters[84]=u32(convolution_input->shape[1]);
            dispatch.parameters[85]=u32(input_values[1]->shape[2]);dispatch.parameters[86]=u32(input_values[1]->shape[3]);
            dispatch.parameters[87]=u32(strides[0]);dispatch.parameters[88]=u32(strides[1]);
            dispatch.parameters[89]=u32(dil[0]);dispatch.parameters[90]=u32(dil[1]);
            dispatch.parameters[91]=u32(pads[0]);dispatch.parameters[92]=u32(pads[1]);
            dispatch.parameters[93]=u32(convolution_input->shape[2]);dispatch.parameters[94]=u32(convolution_input->shape[3]);
            dispatch.parameters[95]=input_values.size()>2&&input_values[2];
            dispatch.parameters[96]=input_values.size()>2&&input_values[2]?u32(input_values[2]->element_offset):0;
            dispatch.parameters[97]=input_values.size()>2&&input_values[2]?u32(input_values[2]->strides[0]):0;
            dispatch.parameters[98]=u32(convolution_input->element_offset);dispatch.parameters[99]=u32(input_values[1]->element_offset);
            put_shape(dispatch.parameters,32,convolution_input->strides);put_shape(dispatch.parameters,48,input_values[1]->strides);
            const bool conv1x1=node.op_type=="Conv"&&groups==1&&dispatch.parameters[85]==1&&dispatch.parameters[86]==1&&
                strides[0]==1&&strides[1]==1&&dil[0]==1&&dil[1]==1&&
                std::ranges::all_of(pads,[](auto value){return value==0;})&&
                convolution_input->shape[2]==outputs[0].shape[2]&&convolution_input->shape[3]==outputs[0].shape[3];
            const bool tiled=groups==1;
            dispatch.parameters[100]=node.op_type=="ConvTranspose";
            const auto output_padding=ints_attribute(node,"output_padding");
            const bool transpose2x2=node.op_type=="ConvTranspose"&&tiled&&
                dispatch.parameters[85]==2&&dispatch.parameters[86]==2&&
                strides==std::vector<int64_t>{2,2}&&dil==std::vector<int64_t>{1,1}&&
                std::ranges::all_of(pads,[](const auto value){return value==0;})&&
                (output_padding.empty()||std::ranges::all_of(output_padding,[](const auto value){return value==0;}))&&
                outputs[0].shape[2]==convolution_input->shape[2]*2&&
                outputs[0].shape[3]==convolution_input->shape[3]*2;
            if(fused_edge_relu){
                dispatch.kernel=Kernel::Conv3x3Edge;dispatch.parameters[100]=1;
            }else if(transpose2x2){
                dispatch.kernel=Kernel::ConvTranspose2x2;
            }else if(node.op_type=="ConvTranspose"){
                dispatch.kernel=tiled?Kernel::ConvTransposeTiled:Kernel::ConvTranspose;
            }else if(conv1x1){
                dispatch.kernel=Kernel::Conv1x1;
            }else if(tiled){
                dispatch.kernel=Kernel::ConvTiled;
            }else{
                dispatch.kernel=Kernel::Conv;
            }
            if(tiled){
                const auto spatial=transpose2x2?u32(convolution_input->shape[2]*convolution_input->shape[3]):ow*oh;
                dispatch.workgroups={(spatial+63u)/64u,(oc+63u)/64u,
                                     u32(outputs[0].shape[0])*(transpose2x2?4u:1u)};
            }
            dispatch.values={ensure_device(*convolution_input),ref(1),ref(2),{}};
            finish(std::move(dispatch));return {};
        }
        if(node.op_type.starts_with("Reduce")){dispatch.parameters[0]=node.op_type=="ReduceMean"?1:node.op_type=="ReduceL2"?2:0;auto axes=*axes_from(node,input_const,1,input_values[0]->shape.size(),true);uint32_t mask=0;uint64_t reduced=1;for(auto axis:axes){mask|=1u<<axis;reduced*=input_values[0]->shape[axis];}dispatch.kernel=reduced>=64?Kernel::Reduce:Kernel::ReduceSerial;dispatch.parameters[80]=mask;dispatch.parameters[81]=u32(reduced);dispatch.parameters[82]=u32(input_values[0]->element_offset);dispatch.parameters[83]=u32(input_values[0]->shape.size());dispatch.parameters[84]=u32(int_attribute(node,"keepdims").value_or(1));put_shape(dispatch.parameters,24,input_values[0]->shape);put_shape(dispatch.parameters,32,input_values[0]->strides);dispatch.values[0]=ref(0);finish(std::move(dispatch));return {};}
        if(node.op_type=="Softmax"){dispatch.kernel=Kernel::Softmax;auto axis=normalize_axis(int_attribute(node,"axis").value_or(-1),input_values[0]->shape.size());dispatch.count=element_count(input_values[0]->shape).value_or(0)/input_values[0]->shape[axis];dispatch.parameters[1]=u32(dispatch.count);dispatch.parameters[80]=u32(axis);dispatch.parameters[81]=u32(input_values[0]->shape[axis]);dispatch.parameters[82]=u32(input_values[0]->element_offset);put_shape(dispatch.parameters,24,input_values[0]->strides);put_shape(dispatch.parameters,32,outputs[0].strides);dispatch.values[0]=ref(0);finish(std::move(dispatch));return {};}
        return std::unexpected(execution_error("GPU lowering is not implemented",&node));
    }

    std::expected<void,Error> ExecutionPlan::Impl::compile_graph(const Graph& graph){const bool top_level=&graph==&model->graph;if(!top_level)++nested_depth;for(size_t i=0;i<graph.nodes.size();++i){if(top_level){++step;if(!live_nodes.contains(i))continue;}auto result=compile_node(graph.nodes[i],i);if(!result){if(!top_level)--nested_depth;return result;}}if(!top_level)--nested_depth;return {};}

    void ExecutionPlan::Impl::tune_matmuls(VulkanRuntime& runtime){
        auto metadata_buffer=runtime.create_buffer(kParameterWords*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if(!metadata_buffer)return;
        auto pool=runtime.create_descriptor_pool(1);if(!pool)return;
        auto set=runtime.allocate_descriptor_set(*pool);
        if(!set){vkDestroyDescriptorPool(runtime.device(),*pool,nullptr);return;}
        const auto resolve=[&](const DeviceRef& ref)->BufferBinding{if(ref.kind==RefKind::Weight)return ref.direct;if(ref.kind==RefKind::Arena)return {arena.buffer,ref.offset,ref.range};if(ref.kind==RefKind::Literal)return {literals.buffer,ref.offset,ref.range};return {dummy.buffer,0,4};};
        struct Choice{Kernel kernel;DeviceRef b;std::array<uint32_t,3> groups;uint32_t packed;};
        for(auto& dispatch:dispatches){
            if(dispatch.kernel!=Kernel::MatMulTiled||dispatch.parameters[81]<64u||
               dispatch.parameters[80]<128u||dispatch.parameters[82]<=64u)continue;
            const auto m=dispatch.parameters[81],n=dispatch.parameters[80],k=dispatch.parameters[82];
            const auto output_rank=dispatch.parameters[2],a_rank=dispatch.parameters[3];
            const auto batch_rank=output_rank-2u;
            const bool contiguous_linear=output_rank>=2u&&output_rank<=8u&&a_rank==output_rank&&
                dispatch.parameters[4]==2u&&dispatch.parameters[85]==0u&&dispatch.parameters[86]==0u&&
                dispatch.parameters[87]==f32(1.0f)&&
                dispatch.parameters[24+batch_rank]==m&&dispatch.parameters[25+batch_rank]==k&&
                dispatch.parameters[32+batch_rank]==k&&dispatch.parameters[33+batch_rank]==1u&&
                dispatch.parameters[40]==k&&dispatch.parameters[41]==n&&
                dispatch.parameters[48]==n&&dispatch.parameters[49]==1u&&
                (dispatch.parameters[89]==0u||dispatch.parameters[91]==1u);
            if(!contiguous_linear)continue;
            const auto batches=dispatch.workgroups[2];
            const auto key=std::to_string(n)+":"+std::to_string(k)+":"+
                std::to_string(dispatch.parameters[89])+":"+
                std::to_string(dispatch.parameters[92])+":"+
                (dispatch.packed_b?"p":"r");
            if(const auto cached=runtime.matmul_tuning_choice(key)){
                const bool packed=cached->packed&&dispatch.packed_b.has_value();
                dispatch.kernel=packed?Kernel::MatMulPackedB:cached->kernel;
                if(packed)dispatch.values[1]=*dispatch.packed_b;
                const auto tile_x=dispatch.kernel==Kernel::MatMulTiled?64u:128u;
                dispatch.workgroups={(n+tile_x-1u)/tile_x,(m+63u)/64u,batches};
                dispatch.parameters[96]=packed;continue;
            }
            std::vector<Choice> choices{
                {Kernel::MatMulTiled,dispatch.values[1],{(n+63u)/64u,(m+63u)/64u,batches},0},
                {Kernel::MatMulLargeN,dispatch.values[1],{(n+127u)/128u,(m+63u)/64u,batches},0},
            };
            if(dispatch.packed_b)choices.push_back({Kernel::MatMulPackedB,*dispatch.packed_b,
                {(n+127u)/128u,(m+63u)/64u,batches},1});
            Choice best=choices.front();auto best_time=std::chrono::nanoseconds::max();
            for(const auto& choice:choices){
                if(choice.groups[0]>runtime.maximum_group_count_x()||choice.groups[1]>runtime.maximum_group_count_y()||
                   choice.groups[2]>runtime.maximum_group_count_z())continue;
                auto parameters=dispatch.parameters;parameters[96]=choice.packed;
                std::memcpy(metadata_buffer->mapped,parameters.data(),kParameterWords*4);
                std::array<BufferBinding,5> bindings{resolve(dispatch.values[0]),resolve(choice.b),
                    resolve(dispatch.values[2]),resolve(dispatch.values[3]),
                    BufferBinding{metadata_buffer->buffer,0,kParameterWords*4}};
                runtime.update_descriptor_set(*set,bindings);
                const auto submit=[&](const uint32_t repeats){return runtime.immediate_submit([&](const VkCommandBuffer command){
                    for(uint32_t repeat=0;repeat<repeats;++repeat){
                        runtime.bind_and_dispatch(command,choice.kernel,*set,dispatch.count,choice.groups);
                        if(repeat+1<repeats){VkMemoryBarrier barrier{.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                            .srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_WRITE_BIT};
                            vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,nullptr,0,nullptr);}
                    }
                });};
                if(!submit(1))continue;
                const auto begin=std::chrono::steady_clock::now();if(!submit(5))continue;
                const auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now()-begin);
                if(best_time==std::chrono::nanoseconds::max()||elapsed*100<best_time*97){best=choice;best_time=elapsed;}
            }
            dispatch.kernel=best.kernel;dispatch.values[1]=best.b;dispatch.workgroups=best.groups;
            dispatch.parameters[96]=best.packed;
            runtime.cache_matmul_tuning_choice(key,{best.kernel,best.packed!=0u});
        }
        vkDestroyDescriptorPool(runtime.device(),*pool,nullptr);
    }

    std::expected<void,Error> ExecutionPlan::Impl::finalize(VulkanRuntime& runtime){
        auto arena_buffer=runtime.create_buffer(arena_size,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);if(!arena_buffer)return std::unexpected(arena_buffer.error());arena=std::move(*arena_buffer);
        auto literal_buffer=runtime.create_buffer(literal_bytes.size(),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);if(!literal_buffer)return std::unexpected(literal_buffer.error());literals=std::move(*literal_buffer);if(!literal_bytes.empty())std::memcpy(literals.mapped,literal_bytes.data(),literal_bytes.size());
        auto dummy_buffer=runtime.create_buffer(4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);if(!dummy_buffer)return std::unexpected(dummy_buffer.error());dummy=std::move(*dummy_buffer);
        tune_matmuls(runtime);
        const auto meta_stride=align_up(kParameterWords*4,runtime.storage_alignment());auto meta_buffer=runtime.create_buffer(meta_stride*dispatches.size(),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);if(!meta_buffer)return std::unexpected(meta_buffer.error());metadata=std::move(*meta_buffer);for(size_t i=0;i<dispatches.size();++i)std::memcpy(static_cast<std::byte*>(metadata.mapped)+i*meta_stride,dispatches[i].parameters.data(),kParameterWords*4);
        VkDeviceSize staging_size=0;for(auto& slot:input_slots){slot.staging_offset=align_up(staging_size,alignment);auto count=element_count(slot.shape).value_or(0);staging_size=slot.staging_offset+count*element_size(slot.type);}auto staging_buffer=runtime.create_buffer(staging_size,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);if(!staging_buffer)return std::unexpected(staging_buffer.error());staging=std::move(*staging_buffer);
        VkDeviceSize readback_size=0;for(auto& slot:output_slots){slot.readback_offset=align_up(readback_size,alignment);slot.bytes=element_count(slot.shape).value_or(0)*element_size(slot.type);readback_size=slot.readback_offset+slot.bytes;}auto readback_buffer=runtime.create_buffer(readback_size,VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,VK_MEMORY_PROPERTY_HOST_CACHED_BIT);if(!readback_buffer)return std::unexpected(readback_buffer.error());readback=std::move(*readback_buffer);
        auto pool=runtime.create_descriptor_pool(static_cast<uint32_t>(dispatches.size()));if(!pool)return std::unexpected(pool.error());descriptor_pool=*pool;
        std::vector<VkDescriptorSet> sets;sets.reserve(dispatches.size());const auto resolve=[&](const DeviceRef& ref)->BufferBinding{if(ref.kind==RefKind::Weight)return ref.direct;if(ref.kind==RefKind::Arena)return {arena.buffer,ref.offset,ref.range};if(ref.kind==RefKind::Literal)return {literals.buffer,ref.offset,ref.range};return {dummy.buffer,0,4};};
        for(size_t i=0;i<dispatches.size();++i){auto set=runtime.allocate_descriptor_set(descriptor_pool);if(!set)return std::unexpected(set.error());sets.push_back(*set);std::array<BufferBinding,5> bindings;for(size_t b=0;b<4;++b)bindings[b]=resolve(dispatches[i].values[b]);bindings[4]={metadata.buffer,i*meta_stride,kParameterWords*4};runtime.update_descriptor_set(*set,bindings);}
        const VkCommandBufferAllocateInfo alloc{.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=runtime.command_pool(),.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};if(auto result=vkAllocateCommandBuffers(runtime.device(),&alloc,&command);result!=VK_SUCCESS)return std::unexpected(execution_error("vkAllocateCommandBuffers failed"));const VkCommandBufferBeginInfo begin{.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};if(vkBeginCommandBuffer(command,&begin)!=VK_SUCCESS)return std::unexpected(execution_error("vkBeginCommandBuffer failed"));
        for(const auto& slot:input_slots){VkBufferCopy copy{slot.staging_offset,slot.device.offset,element_count(slot.shape).value_or(0)*4};vkCmdCopyBuffer(command,staging.buffer,arena.buffer,1,&copy);}VkMemoryBarrier barrier{.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT};vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&barrier,0,nullptr,0,nullptr);
        for(size_t i=0;i<dispatches.size();++i){
            auto& dispatch=dispatches[i];
            if(dispatch.kernel==Kernel::ConvTranspose2x2&&
               dispatch.workgroups[2]>runtime.maximum_group_count_z()){
                dispatch.kernel=Kernel::ConvTransposeTiled;
                const auto spatial=static_cast<uint64_t>(dispatch.parameters[80])*dispatch.parameters[81];
                dispatch.workgroups={u32((spatial+63u)/64u),(dispatch.parameters[82]+63u)/64u,
                                     dispatch.parameters[16]};
            }
            if(dispatch.workgroups[0]==0&&(dispatch.kernel==Kernel::Reduce||dispatch.kernel==Kernel::Softmax||
                dispatch.kernel==Kernel::LayerNorm||dispatch.kernel==Kernel::Attention)){
                const auto groups_x=u32(std::min<uint64_t>(dispatch.count,runtime.maximum_group_count_x()));
                const auto groups_y=u32((dispatch.count+groups_x-1)/groups_x);
                if(groups_y>runtime.maximum_group_count_y())return std::unexpected(execution_error("workgroup grid exceeds Vulkan device limits"));
                dispatch.workgroups={groups_x,groups_y,1};
            }
            if(dispatch.workgroups[0]!=0&&(dispatch.workgroups[0]>runtime.maximum_group_count_x()||
                dispatch.workgroups[1]>runtime.maximum_group_count_y()||dispatch.workgroups[2]>runtime.maximum_group_count_z()))
                return std::unexpected(execution_error("specialized workgroup grid exceeds Vulkan device limits"));
            runtime.bind_and_dispatch(command,dispatch.kernel,sets[i],dispatch.count,dispatch.workgroups);
            VkMemoryBarrier compute{.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT};vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&compute,0,nullptr,0,nullptr);
        }
        barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&barrier,0,nullptr,0,nullptr);for(const auto& slot:output_slots){auto& value=values.at(slot.name);if(value.strides!=contiguous_strides(value.shape)||value.element_offset!=0)return std::unexpected(execution_error("requested strided output requires materialization"));auto source=resolve(value.device);VkBufferCopy copy{source.offset,slot.readback_offset,slot.bytes};vkCmdCopyBuffer(command,source.buffer,readback.buffer,1,&copy);}if(vkEndCommandBuffer(command)!=VK_SUCCESS)return std::unexpected(execution_error("vkEndCommandBuffer failed"));return {};
    }

    ExecutionPlan::ExecutionPlan(VulkanRuntime& runtime) : runtime_(runtime) {}
    ExecutionPlan::~ExecutionPlan(){if(!impl_)return;if(impl_->command)vkFreeCommandBuffers(runtime_.device(),runtime_.command_pool(),1,&impl_->command);if(impl_->descriptor_pool)vkDestroyDescriptorPool(runtime_.device(),impl_->descriptor_pool,nullptr);}

    std::expected<std::unique_ptr<ExecutionPlan>,Error> ExecutionPlan::create(const Model& model,const WeightStore& weights,VulkanRuntime& runtime,std::span<const NamedTensorView> inputs,std::span<const std::string_view> requested_outputs){auto plan=std::unique_ptr<ExecutionPlan>(new ExecutionPlan(runtime));plan->impl_=std::make_unique<Impl>();auto& p=*plan->impl_;p.model=&model;p.weights=&weights;p.alignment=runtime.storage_alignment();for(auto output:requested_outputs)p.requested.emplace(output);if(p.requested.empty())for(const auto& output:model.graph.outputs)p.requested.emplace(output.name);
        for(size_t i=0;i<model.graph.nodes.size();++i){const auto& node=model.graph.nodes[i];p.node_indices.emplace(&node,i);for(const auto& output:node.outputs)if(!output.empty())p.producers.emplace(output,&node);for(const auto& input:node.inputs)if(!input.empty())p.consumers[input].push_back(&node);for(const auto& attribute:node.attributes)if(const auto* branch=std::get_if<std::shared_ptr<Graph>>(&attribute.value);branch&&*branch)collect_captures(**branch,p.nested_captures);}
        std::unordered_set<std::string> needed=p.requested;for(size_t i=model.graph.nodes.size();i-->0;){const auto& node=model.graph.nodes[i];bool live=false;for(const auto& output:node.outputs)live|=needed.contains(output);if(live){p.live_nodes.emplace(i);for(const auto& input:node.inputs)if(!input.empty())needed.emplace(input);for(const auto& attribute:node.attributes)if(const auto* branch=std::get_if<std::shared_ptr<Graph>>(&attribute.value);branch&&*branch){std::unordered_set<std::string> captures;collect_captures(**branch,captures);needed.insert(captures.begin(),captures.end());}}}
        for(size_t i=0;i<model.graph.nodes.size();++i)if(p.live_nodes.contains(i)){for(const auto& input:model.graph.nodes[i].inputs)if(!input.empty())p.last_use[input]=i+1;for(const auto& attribute:model.graph.nodes[i].attributes)if(const auto* branch=std::get_if<std::shared_ptr<Graph>>(&attribute.value);branch&&*branch){std::unordered_set<std::string> captures;collect_captures(**branch,captures);for(const auto& capture:captures)p.last_use[capture]=std::max(p.last_use[capture],i+1);}}for(const auto& output:p.requested)p.last_use[output]=std::numeric_limits<size_t>::max();
        for(const auto& initializer:model.graph.initializers){Value value{initializer.type,initializer.shape,contiguous_strides(initializer.shape),0,host_tensor(initializer),{}};if(auto it=weights.tensors.find(initializer.name);it!=weights.tensors.end()){value.device.kind=RefKind::Weight;value.device.direct=it->second.binding;value.device.range=it->second.binding.range;}p.values.emplace(initializer.name,std::move(value));}
        for(const auto& info:model.graph.inputs){auto it=std::ranges::find(inputs,info.name,&NamedTensorView::name);if(it==inputs.end())return std::unexpected(Error{ErrorCode::InvalidInput,"missing model input '"+info.name+"'"});Value value{it->tensor.type,std::vector<int64_t>(it->tensor.shape.begin(),it->tensor.shape.end()),contiguous_strides(it->tensor.shape)};if(value.type==ElementType::Float32){ValueSpec spec{value.type,value.shape};value.device=p.allocate_arena(spec,std::numeric_limits<size_t>::max(),true);p.input_slots.push_back({info.name,value.type,value.shape,0,value.device});}else{std::vector<std::byte> bytes(it->tensor.bytes.begin(),it->tensor.bytes.end());value.host=owned_host(value.type,value.shape,std::move(bytes));}p.values.emplace(info.name,std::move(value));}
        auto compiled=p.compile_graph(model.graph);if(!compiled)return std::unexpected(compiled.error());
        if(requested_outputs.empty()){
            for(const auto& output:model.graph.outputs){auto it=p.values.find(output.name);if(it==p.values.end())return std::unexpected(execution_error("requested output '"+output.name+"' was pruned"));p.output_slots.push_back({output.name,it->second.type,it->second.shape});}
        }else{
            for(const auto output:requested_outputs){auto it=p.values.find(std::string(output));if(it==p.values.end())return std::unexpected(execution_error("requested output '"+std::string(output)+"' was pruned"));p.output_slots.push_back({std::string(output),it->second.type,it->second.shape});}
        }
        auto finalized=p.finalize(runtime);if(!finalized)return std::unexpected(finalized.error());return plan;}

    std::expected<std::vector<NamedTensor>,Error> ExecutionPlan::run(std::span<const NamedTensorView> inputs){
        for(const auto& slot:impl_->input_slots){
            auto it=std::ranges::find(inputs,slot.name,&NamedTensorView::name);
            if(it==inputs.end()||!std::ranges::equal(it->tensor.shape,slot.shape)||it->tensor.type!=slot.type)
                return std::unexpected(Error{ErrorCode::InvalidInput,"input does not match cached execution plan: "+slot.name});
            std::memcpy(static_cast<std::byte*>(impl_->staging.mapped)+slot.staging_offset,it->tensor.bytes.data(),it->tensor.bytes.size());
        }
        const VkSubmitInfo submit{.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&impl_->command};
        auto status=vkQueueSubmit(runtime_.queue(),1,&submit,VK_NULL_HANDLE);
        if(status==VK_SUCCESS)status=vkQueueWaitIdle(runtime_.queue());
        if(status!=VK_SUCCESS)return std::unexpected(Error{ErrorCode::VulkanFailure,"Vulkan inference submission failed with VkResult "+std::to_string(status)});
        std::vector<NamedTensor> result;
        for(const auto& slot:impl_->output_slots){
            std::vector<std::byte> bytes(slot.bytes);
            std::memcpy(bytes.data(),static_cast<const std::byte*>(impl_->readback.mapped)+slot.readback_offset,slot.bytes);
            result.push_back({slot.name,Tensor(slot.type,slot.shape,std::move(bytes))});
        }
        return result;
    }

    std::string plan_signature(std::span<const NamedTensorView> inputs,std::span<const std::string_view> requested_outputs){std::string result;for(const auto& input:inputs){result.append(input.name).push_back(':');result.append(std::to_string(static_cast<int>(input.tensor.type))).push_back('[');for(auto dim:input.tensor.shape)result.append(std::to_string(dim)).push_back(',');result.push_back(']');if(input.tensor.type!=ElementType::Float32)result.append(reinterpret_cast<const char*>(input.tensor.bytes.data()),input.tensor.bytes.size());result.push_back(';');}result.push_back('|');for(auto output:requested_outputs)result.append(output).push_back(',');return result;}

} // namespace lfs::onnx_vulkan::detail
