/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/nodes/types.hpp"
#include "gui/node_canvas_interaction.hpp"

#include <algorithm>
#include <numeric>
#include <queue>
#include <ranges>

namespace lfs::vis::gui {
    std::unordered_map<std::string, CanvasPoint> NodeCanvasInteraction::arrangedPositions(
        const std::string_view input, const std::string_view output,
        const std::unordered_set<std::string>& subset) const {
        if (!subset.empty()) {
            std::vector<CanvasNode> nodes;
            std::vector<CanvasLink> links;
            for (const auto& node : nodes_)
                if (subset.contains(node.id))
                    nodes.push_back(node);
            if (nodes.empty())
                return {};
            for (const auto& link : links_)
                if (subset.contains(link.from.node) && subset.contains(link.to.node))
                    links.push_back(link);
            NodeCanvasInteraction selection;
            selection.setDpRatio(dp_ratio_);
            selection.setGraph(nodes, std::move(links));
            auto positions = selection.arrangedPositions(input, output);
            const auto centre = [&](const bool arranged) {
                CanvasPoint low{nodes.front().bounds.x, nodes.front().bounds.y};
                if (arranged)
                    low = positions.at(nodes.front().id);
                CanvasPoint high = low;
                for (const auto& node : nodes) {
                    const auto position = arranged ? positions.at(node.id) : CanvasPoint{node.bounds.x, node.bounds.y};
                    low.x = std::min(low.x, position.x);
                    low.y = std::min(low.y, position.y);
                    high.x = std::max(high.x, position.x + node.bounds.width);
                    high.y = std::max(high.y, position.y + node.bounds.height);
                }
                return (low + high) * 0.5f;
            };
            const auto offset = centre(false) - centre(true);
            for (auto& [name, position] : positions)
                position = position + offset;
            return positions;
        }
        const auto count = nodes_.size();
        std::unordered_map<std::string, std::size_t> indices;
        for (std::size_t i = 0; i < count; ++i)
            indices.emplace(nodes_[i].id, i);
        std::vector<std::vector<std::size_t>> parents(count), children(count);
        std::vector<std::size_t> degree(count), depth(count);
        for (const auto& link : links_) {
            const auto from = indices.find(link.from.node);
            const auto to = indices.find(link.to.node);
            if (from == indices.end() || to == indices.end())
                continue;
            parents[to->second].push_back(from->second);
            children[from->second].push_back(to->second);
            ++degree[to->second];
        }
        std::queue<std::size_t> ready;
        for (std::size_t i = 0; i < count; ++i)
            if (degree[i] == 0)
                ready.push(i);
        std::vector<std::size_t> order;
        while (!ready.empty()) {
            const auto node = ready.front();
            ready.pop();
            order.push_back(node);
            for (const auto child : children[node]) {
                depth[child] = std::max(depth[child], depth[node] + 1);
                if (--degree[child] == 0)
                    ready.push(child);
            }
        }
        // Invalid cyclic graphs remain arrangeable, with cycle members in a
        // separate column; evaluation still reports the actual validation error.
        auto maximum = depth.empty() ? 0 : *std::ranges::max_element(depth);
        for (std::size_t i = 0; i < count; ++i)
            if (degree[i] != 0)
                depth[i] = maximum + 1;
        for (const auto i : order | std::views::reverse) {
            const bool helper = !nodes_[i].sockets.empty() &&
                                std::ranges::none_of(nodes_[i].sockets, [](const auto& socket) {
                                    return socket.direction == CanvasSocketDirection::Output && socket.type == lfs::nodes::GEOMETRY_SOCKET;
                                });
            if (helper && !children[i].empty()) {
                const auto nearest = *std::ranges::min_element(children[i], {}, [&](const auto child) { return depth[child]; });
                if (depth[nearest] > 0)
                    depth[i] = depth[nearest] - 1;
            }
        }
        if (const auto found = indices.find(std::string(input)); found != indices.end())
            depth[found->second] = 0;
        maximum = depth.empty() ? 0 : *std::ranges::max_element(depth);
        if (const auto found = indices.find(std::string(output)); found != indices.end())
            depth[found->second] = ++maximum;
        std::vector<std::vector<std::size_t>> layers(maximum + 1);
        std::vector<float> rank(count);
        for (std::size_t i = 0; i < count; ++i) {
            rank[i] = static_cast<float>(layers[depth[i]].size());
            layers[depth[i]].push_back(i);
        }
        for (int sweep = 0; sweep < 6; ++sweep) {
            const bool forward = sweep % 2 == 0;
            for (std::size_t step = 0; step < layers.size(); ++step) {
                auto& layer = layers[forward ? step : layers.size() - 1 - step];
                const auto centre = [&](const std::size_t node) {
                    const auto& neighbours = forward ? parents[node] : children[node];
                    if (neighbours.empty())
                        return rank[node];
                    float sum = 0.0f;
                    for (const auto neighbour : neighbours)
                        sum += rank[neighbour];
                    return sum / neighbours.size();
                };
                std::ranges::stable_sort(layer, [&](const auto a, const auto b) { return centre(a) < centre(b); });
                for (std::size_t i = 0; i < layer.size(); ++i)
                    rank[layer[i]] = static_cast<float>(i);
            }
        }
        std::unordered_map<std::string, CanvasPoint> result;
        float x = 24.0f * dp_ratio_;
        for (const auto& layer : layers) {
            float width = 0.0f;
            float y = 24.0f * dp_ratio_;
            for (const auto index : layer) {
                const auto& node = nodes_[index];
                result.emplace(node.id, CanvasPoint{x, y});
                width = std::max(width, node.bounds.width);
                // Leave room for the minimum readable title at the lowest LOD.
                y += node.bounds.height + 40.0f * dp_ratio_;
            }
            if (!layer.empty())
                x += width + 64.0f * dp_ratio_;
        }
        return result;
    }
} // namespace lfs::vis::gui
