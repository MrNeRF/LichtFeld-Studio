/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "app/mcp_app_utils.hpp"
#include "mcp/mcp_tools.hpp"
#include "visualizer/nodes/modifier_manager.hpp"
#include "visualizer/scene/scene_manager.hpp"
#include "visualizer/visualizer_impl.hpp"

namespace lfs::app::node_mcp {
    using json = nlohmann::json;
    using Handler = std::function<json(vis::VisualizerImpl&, const json&)>;
    struct ArgumentError {
        std::string message;
    };
    json types(vis::ModifierManager& manager);
    json stacks(vis::SceneManager& scene);
    json stack(vis::SceneManager& scene, const core::Uuid& target);
    json editor(vis::VisualizerImpl& viewer);
    json progress(const vis::ModifierWorkerProgress& value);
    std::expected<lfs::nodes::Value, ArgumentError> value(const json& input, const lfs::nodes::SocketDecl& socket);
    std::expected<void, ArgumentError> property(const json& input, const lfs::nodes::PropertyDecl& declaration);
    std::optional<core::Uuid> target(vis::SceneManager& scene, const json& args);
    lfs::nodes::NodeTree* tree(vis::ModifierManager& manager, const json& args);
    vis::Modifier* modifier(vis::ModifierManager& manager, const core::Uuid& host, const json& args);
    void add(mcp::ToolRegistry& registry, vis::VisualizerImpl* viewer, std::string name,
             std::string description, json properties, std::vector<std::string> required, Handler handler,
             bool read_only = false, bool destructive = false);
    inline json stringSchema() { return {{"type", "string"}}; }
    inline json boolSchema() { return {{"type", "boolean"}}; }
    inline json pointSchema() { return {{"type", "array"}, {"items", {{"type", "number"}}}, {"minItems", 2}, {"maxItems", 2}}; }
    inline json failure(const std::string& message, const std::string& argument) {
        return mcp::invalid_argument_result(message, argument);
    }
} // namespace lfs::app::node_mcp
