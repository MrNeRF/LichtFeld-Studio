/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs_onnx_vulkan/onnx_vulkan.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace lfs::onnx_vulkan::detail {

    struct Graph;

    struct TensorData {
        std::string name;
        ElementType type = ElementType::Float32;
        std::vector<std::int64_t> shape;
        std::span<const std::byte> bytes;
        std::shared_ptr<std::vector<std::byte>> external_owner;
    };

    enum class AttributeType : std::uint8_t {
        Undefined = 0,
        Float = 1,
        Int = 2,
        String = 3,
        Tensor = 4,
        Graph = 5,
        Floats = 6,
        Ints = 7,
        Strings = 8,
        Tensors = 9,
        Graphs = 10,
    };

    using AttributeValue = std::variant<std::monostate,
                                        float,
                                        std::int64_t,
                                        std::string,
                                        TensorData,
                                        std::shared_ptr<Graph>,
                                        std::vector<float>,
                                        std::vector<std::int64_t>,
                                        std::vector<std::string>,
                                        std::vector<TensorData>,
                                        std::vector<std::shared_ptr<Graph>>>;

    struct Attribute {
        std::string name;
        AttributeType type = AttributeType::Undefined;
        AttributeValue value;
    };

    struct Node {
        std::string name;
        std::string domain;
        std::string op_type;
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
        std::vector<Attribute> attributes;
    };

    struct Graph {
        std::string name;
        std::vector<ValueInfo> inputs;
        std::vector<ValueInfo> outputs;
        std::vector<ValueInfo> value_info;
        std::vector<TensorData> initializers;
        std::vector<Node> nodes;
    };

    struct Model {
        std::int64_t ir_version = 0;
        std::unordered_map<std::string, std::int64_t> opsets;
        Graph graph;
        std::shared_ptr<std::vector<std::byte>> model_bytes;
        std::filesystem::path path;
    };

    [[nodiscard]] std::expected<Model, Error>
    parse_model(const std::filesystem::path& path, const SessionOptions& options);

    [[nodiscard]] std::expected<void, Error> validate_model(const Model& model);

    [[nodiscard]] const Attribute* find_attribute(const Node& node, std::string_view name) noexcept;

} // namespace lfs::onnx_vulkan::detail
