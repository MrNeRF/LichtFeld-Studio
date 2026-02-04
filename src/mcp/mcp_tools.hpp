/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "mcp_protocol.hpp"
#include "training/control/command_api.hpp"

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::mcp {

    class LFS_MCP_API ToolRegistry {
    public:
        using ToolHandler = std::function<json(const json& params)>;

        static ToolRegistry& instance();

        void register_tool(McpTool tool, ToolHandler handler);
        void unregister_tool(const std::string& name);

        std::vector<McpTool> list_tools() const;
        json call_tool(const std::string& name, const json& arguments);

        void generate_from_command_center();

    private:
        ToolRegistry() = default;

        McpTool operation_to_tool(const training::OperationInfo& op) const;
        json arg_type_to_json_schema(training::ArgType type) const;
        ToolHandler create_command_handler(const training::OperationInfo& op) const;

        training::Command json_to_command(
            const training::OperationInfo& op,
            const json& arguments) const;

        struct RegisteredTool {
            McpTool tool;
            ToolHandler handler;
        };

        std::unordered_map<std::string, RegisteredTool> tools_;
        mutable std::mutex mutex_;
    };

    void register_builtin_tools();

} // namespace lfs::mcp
