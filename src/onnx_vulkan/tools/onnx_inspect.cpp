/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "model.hpp"

#include <algorithm>
#include <iostream>
#include <map>

namespace {
    void count_graph(const lfs::onnx_vulkan::detail::Graph& graph,
                     std::map<std::string, std::size_t>& counts,
                     std::size_t& initializer_bytes) {
        for (const auto& initializer : graph.initializers)
            initializer_bytes += initializer.bytes.size();
        for (const auto& node : graph.nodes) {
            ++counts[node.op_type];
            for (const auto& attribute : node.attributes) {
                if (const auto* nested = std::get_if<std::shared_ptr<lfs::onnx_vulkan::detail::Graph>>(&attribute.value);
                    nested && *nested)
                    count_graph(**nested, counts, initializer_bytes);
            }
        }
    }
}

int main(const int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: lfs_onnx_inspect <model.onnx> [node-filter]\n";
        return 2;
    }
    auto model = lfs::onnx_vulkan::detail::parse_model(argv[1], {});
    if (!model) {
        std::cerr << model.error().message << '\n';
        return 1;
    }
    std::map<std::string, std::size_t> counts;
    std::size_t initializer_bytes = 0;
    count_graph(model->graph, counts, initializer_bytes);
    std::cout << "IR " << model->ir_version << " opset " << model->opsets.at("")
              << " nodes " << model->graph.nodes.size()
              << " initializers " << model->graph.initializers.size()
              << " value_info " << model->graph.value_info.size()
              << " initializer_bytes " << initializer_bytes << '\n';
    std::cout << "inputs";
    for (const auto& input : model->graph.inputs)
        std::cout << ' ' << input.name;
    std::cout << "\noutputs";
    for (const auto& output : model->graph.outputs)
        std::cout << ' ' << output.name;
    std::cout << '\n';
    for (const auto& [name, count] : counts)
        std::cout << name << ' ' << count << '\n';
    if (argc == 3) {
        std::string_view filter = argv[2];
        const bool exact = filter.starts_with('=');
        if (exact)
            filter.remove_prefix(1);
        for (std::size_t node_index = 0; node_index < model->graph.nodes.size(); ++node_index) {
            const auto& node = model->graph.nodes[node_index];
            const bool named_value = std::ranges::find(node.inputs, filter) != node.inputs.end() ||
                                     std::ranges::find(node.outputs, filter) != node.outputs.end();
            const bool named_node = exact ? node.name == filter : node.name.contains(filter);
            if (!named_node && !named_value)
                continue;
            std::cout << "NODE[" << node_index << "] " << node.name << " " << node.op_type << " IN";
            for (const auto& input : node.inputs) std::cout << ' ' << input;
            std::cout << " OUT";
            for (const auto& output : node.outputs) std::cout << ' ' << output;
            std::cout << " ATTR";
            for (const auto& attribute : node.attributes) {
                std::cout << ' ' << attribute.name << '=';
                if (const auto* value = std::get_if<std::string>(&attribute.value))
                    std::cout << *value;
                else if (const auto* value = std::get_if<std::int64_t>(&attribute.value))
                    std::cout << *value;
                else if (const auto* value = std::get_if<float>(&attribute.value))
                    std::cout << *value;
                else if (const auto* values = std::get_if<std::vector<std::int64_t>>(&attribute.value)) {
                    std::cout << '[';
                    for (const auto value : *values) std::cout << value << ',';
                    std::cout << ']';
                } else
                    std::cout << '<' << static_cast<int>(attribute.type) << '>';
            }
            std::cout << '\n';
        }
    }
    if (auto valid = lfs::onnx_vulkan::detail::validate_model(*model); !valid) {
        std::cerr << valid.error().message << '\n';
        return 1;
    }
    return 0;
}
