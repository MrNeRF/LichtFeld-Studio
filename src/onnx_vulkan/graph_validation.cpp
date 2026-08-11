/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "model.hpp"
#include "operator_registry.hpp"

#include <algorithm>
#include <deque>
#include <unordered_map>
#include <unordered_set>

namespace lfs::onnx_vulkan::detail {
    namespace {
        [[nodiscard]] Error graph_error(std::string message) {
            return {ErrorCode::MalformedModel, std::move(message), {}, {}};
        }

        [[nodiscard]] Error node_unsupported(const Node& node,
                                             std::string message,
                                             std::string capability) {
            if (!node.name.empty())
                message = "node '" + node.name + "' (" + node.op_type + "): " + message;
            else
                message = "node <unnamed> (" + node.op_type + "): " + message;
            return {ErrorCode::UnsupportedModel, std::move(message), node.name, std::move(capability)};
        }

        [[nodiscard]] std::expected<void, Error>
        validate_node_schema(const Node& node, const std::int64_t opset) {
            if (!node.domain.empty() && node.domain != "ai.onnx")
                return std::unexpected(node_unsupported(node,
                                                        "unsupported operator domain '" + node.domain + "'",
                                                        "domain " + node.domain));
            const auto* schema = find_operator_schema(node.op_type, opset);
            if (!schema)
                return std::unexpected(node_unsupported(node,
                                                        "operator is not implemented for ai.onnx opset " +
                                                            std::to_string(opset),
                                                        "ai.onnx::" + node.op_type + "@" + std::to_string(opset)));
            if (node.inputs.size() < schema->minimum_inputs || node.inputs.size() > schema->maximum_inputs)
                return std::unexpected(graph_error("node '" + node.name + "' (" + node.op_type +
                                                   ") has an invalid input count"));
            if (node.outputs.size() < schema->minimum_outputs || node.outputs.size() > schema->maximum_outputs)
                return std::unexpected(graph_error("node '" + node.name + "' (" + node.op_type +
                                                   ") has an invalid output count"));
            for (const auto& attribute : node.attributes) {
                const auto it = std::ranges::find(schema->attributes, attribute.name, &AttributeSchema::name);
                if (it == schema->attributes.end())
                    return std::unexpected(node_unsupported(node,
                                                            "attribute '" + attribute.name + "' is not supported",
                                                            node.op_type + " attribute " + attribute.name));
                if (it->type != AttributeType::Undefined && it->type != attribute.type)
                    return std::unexpected(graph_error("node '" + node.name + "' attribute '" + attribute.name +
                                                       "' has the wrong type"));
            }
            for (const auto& attribute : schema->attributes) {
                if (attribute.required && !find_attribute(node, attribute.name))
                    return std::unexpected(graph_error("node '" + node.name + "' (" + node.op_type +
                                                       ") is missing required attribute '" +
                                                       std::string(attribute.name) + "'"));
            }

            if (node.op_type == "Constant") {
                const auto count = std::ranges::count_if(node.attributes, [](const Attribute& attribute) {
                    return attribute.name.starts_with("value");
                });
                if (count != 1)
                    return std::unexpected(graph_error("Constant node '" + node.name +
                                                       "' must have exactly one value attribute"));
            }
            if (node.op_type == "If") {
                const auto* then_attr = find_attribute(node, "then_branch");
                const auto* else_attr = find_attribute(node, "else_branch");
                if (!then_attr || !else_attr)
                    return std::unexpected(graph_error("If node '" + node.name + "' is missing a branch graph"));
            }
            return {};
        }

        void add_value_info_names(const std::span<const ValueInfo> infos,
                                  std::unordered_set<std::string>& names) {
            for (const auto& info : infos)
                names.emplace(info.name);
        }

        [[nodiscard]] std::expected<void, Error>
        validate_graph(const Graph& graph,
                       const std::int64_t opset,
                       const std::unordered_set<std::string>& outer_scope) {
            std::unordered_set<std::string> graph_inputs;
            for (const auto& input : graph.inputs) {
                if (!graph_inputs.emplace(input.name).second)
                    return std::unexpected(graph_error("graph has duplicate input '" + input.name + "'"));
            }

            std::unordered_set<std::string> initializer_names;
            for (const auto& initializer : graph.initializers) {
                if (initializer.name.empty())
                    return std::unexpected(graph_error("graph has an initializer with an empty name"));
                if (!initializer_names.emplace(initializer.name).second)
                    return std::unexpected(graph_error("graph has duplicate initializer '" + initializer.name + "'"));
            }

            std::unordered_set<std::string> node_names;
            std::unordered_map<std::string, std::size_t> producer;
            for (std::size_t index = 0; index < graph.nodes.size(); ++index) {
                const auto& node = graph.nodes[index];
                if (!node.name.empty() && !node_names.emplace(node.name).second)
                    return std::unexpected(graph_error("graph has duplicate node name '" + node.name + "'"));
                bool has_output = false;
                for (const auto& output : node.outputs) {
                    if (output.empty())
                        continue;
                    has_output = true;
                    if (graph_inputs.contains(output) || initializer_names.contains(output) ||
                        !producer.emplace(output, index).second)
                        return std::unexpected(graph_error("graph has duplicate value producer for '" + output + "'"));
                }
                if (!has_output)
                    return std::unexpected(graph_error("node '" + node.name + "' has no named output"));
                if (auto valid = validate_node_schema(node, opset); !valid)
                    return std::unexpected(valid.error());
            }

            std::vector<std::vector<std::size_t>> consumers(graph.nodes.size());
            std::vector<std::size_t> indegree(graph.nodes.size());
            bool topologically_ordered = true;
            for (std::size_t index = 0; index < graph.nodes.size(); ++index) {
                const auto& node = graph.nodes[index];
                std::unordered_set<std::size_t> dependencies;
                for (const auto& input : node.inputs) {
                    if (input.empty())
                        continue;
                    if (const auto it = producer.find(input); it != producer.end()) {
                        dependencies.emplace(it->second);
                        topologically_ordered &= it->second < index;
                    } else if (!graph_inputs.contains(input) && !initializer_names.contains(input) &&
                               !outer_scope.contains(input)) {
                        return std::unexpected(graph_error("node '" + node.name + "' references unknown input '" +
                                                           input + "'"));
                    }
                }
                for (const auto dependency : dependencies) {
                    consumers[dependency].push_back(index);
                    ++indegree[index];
                }
            }

            std::deque<std::size_t> ready;
            for (std::size_t index = 0; index < indegree.size(); ++index)
                if (indegree[index] == 0)
                    ready.push_back(index);
            std::size_t visited = 0;
            while (!ready.empty()) {
                const auto index = ready.front();
                ready.pop_front();
                ++visited;
                for (const auto consumer : consumers[index])
                    if (--indegree[consumer] == 0)
                        ready.push_back(consumer);
            }
            if (visited != graph.nodes.size())
                return std::unexpected(graph_error("graph contains a cycle"));
            if (!topologically_ordered)
                return std::unexpected(graph_error("graph nodes are not topologically ordered"));

            std::unordered_set<std::string> visible = outer_scope;
            visible.insert(graph_inputs.begin(), graph_inputs.end());
            visible.insert(initializer_names.begin(), initializer_names.end());
            for (const auto& [name, index] : producer) {
                static_cast<void>(index);
                visible.emplace(name);
            }
            std::unordered_set<std::string> output_names;
            for (const auto& output : graph.outputs) {
                if (!output_names.emplace(output.name).second)
                    return std::unexpected(graph_error("graph has duplicate output '" + output.name + "'"));
                if (!visible.contains(output.name))
                    return std::unexpected(graph_error("graph output '" + output.name + "' has no producer"));
            }

            std::unordered_set<std::string> info_names;
            for (const auto& info : graph.value_info) {
                if (!info_names.emplace(info.name).second)
                    return std::unexpected(graph_error("graph has duplicate value_info '" + info.name + "'"));
                if (!visible.contains(info.name))
                    return std::unexpected(graph_error("value_info references unknown value '" + info.name + "'"));
            }

            for (auto& node : graph.nodes) {
                if (node.op_type != "If")
                    continue;
                const auto* then_attr = find_attribute(node, "then_branch");
                const auto* else_attr = find_attribute(node, "else_branch");
                auto then_graph = std::get<std::shared_ptr<Graph>>(then_attr->value);
                auto else_graph = std::get<std::shared_ptr<Graph>>(else_attr->value);
                if (then_graph->outputs.size() != node.outputs.size() ||
                    else_graph->outputs.size() != node.outputs.size())
                    return std::unexpected(graph_error("If node '" + node.name +
                                                       "' branch output counts do not match the node"));
                if (auto valid = validate_graph(*then_graph, opset, visible); !valid)
                    return std::unexpected(valid.error());
                if (auto valid = validate_graph(*else_graph, opset, visible); !valid)
                    return std::unexpected(valid.error());
            }
            return {};
        }
    } // namespace

    std::expected<void, Error> validate_model(const Model& model) {
        const auto opset = model.opsets.find("");
        if (opset == model.opsets.end())
            return std::unexpected(graph_error("model has no ai.onnx opset"));
        if (model.graph.inputs.empty())
            return std::unexpected(graph_error("model graph has no inputs"));
        return validate_graph(model.graph, opset->second, {});
    }

} // namespace lfs::onnx_vulkan::detail
