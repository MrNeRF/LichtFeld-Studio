/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "mcp/mcp_tools.hpp"
#include "visualizer/operator/operator_id.hpp"

#include <expected>
#include <functional>
#include <string>
#include <vector>

namespace lfs::vis {
    class Visualizer;
}

namespace lfs::vis::op {
    class OperatorProperties;
    struct OperatorReturnValue;
} // namespace lfs::vis::op

namespace lfs::app {

    using json = nlohmann::json;

    struct GuiOperatorToolBinding {
        std::string tool_name;
        vis::op::BuiltinOp operator_id;
        std::string category;
        std::string description;
        std::vector<std::string> required;
        bool destructive = false;
        std::function<std::expected<void, std::string>(vis::Visualizer& viewer, const json& args,
                                                       vis::op::OperatorProperties& props)>
            prepare;
        std::function<json(vis::Visualizer& viewer, const json& args,
                           const vis::op::OperatorProperties& props,
                           const vis::op::OperatorReturnValue& result)>
            on_success;
    };

    void register_gui_operator_tool(mcp::ToolRegistry& registry,
                                    vis::Visualizer* viewer,
                                    GuiOperatorToolBinding binding);

    void register_generic_gui_operator_tools(mcp::ToolRegistry& registry,
                                             vis::Visualizer* viewer);

    void register_generic_gui_operator_resources(mcp::ResourceRegistry& registry,
                                                 vis::Visualizer* viewer);

} // namespace lfs::app
