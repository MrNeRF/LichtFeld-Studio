/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "mcp_protocol.hpp"
#include "training/control/command_api.hpp"

#include <expected>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfs::mcp {

    class LFS_MCP_API ToolRegistry {
    public:
        using ToolHandler = std::function<json(const json& params)>;

        ToolRegistry() = default;

        static ToolRegistry& instance();

        void register_tool(McpTool tool, ToolHandler handler);
        void unregister_tool(const std::string& name);

        std::vector<McpTool> list_tools() const;
        json call_tool(const std::string& name, const json& arguments,
                       lfs::OperationId operation_id = {});

        void generate_from_command_center();
        void set_lazy_initializer(std::function<void()> initializer);

    private:
        void ensure_initialized() const;

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
        mutable std::once_flag initialization_once_;
        std::function<void()> lazy_initializer_;
    };

    class LFS_MCP_API ResourceRegistry {
    public:
        using ResourceHandler =
            std::function<std::expected<std::vector<McpResourceContent>, std::string>(const std::string& uri)>;

        static ResourceRegistry& instance();

        void register_resource(McpResource resource, ResourceHandler handler);
        void register_resource_prefix(std::string uri_prefix, ResourceHandler handler);
        void unregister_resource_prefix(const std::string& uri_prefix);

        std::vector<McpResource> list_resources() const;
        std::expected<std::vector<McpResourceContent>, std::string> read_resource(const std::string& uri) const;
        void set_lazy_initializer(std::function<void()> initializer);

    private:
        ResourceRegistry() = default;

        void ensure_initialized() const;

        struct RegisteredResource {
            McpResource resource;
            ResourceHandler handler;
        };

        std::unordered_map<std::string, RegisteredResource> resources_;
        std::unordered_map<std::string, ResourceHandler> prefix_handlers_;
        mutable std::mutex mutex_;
        mutable std::once_flag initialization_once_;
        std::function<void()> lazy_initializer_;
    };

    LFS_MCP_API void register_core_tools();
    LFS_MCP_API void register_core_resources();
} // namespace lfs::mcp
