/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "app/mcp_shader_tools.hpp"
#include "app/include/app/mcp_app_utils.hpp"
#include "core/error_envelope.hpp"
#include "mcp/mcp_tools.hpp"
#include "shaders/shader_api.hpp"

namespace lfs::app {
    namespace {
        using json = nlohmann::json;
        using namespace vis::shaders;
        Result<json> invoke(const ShaderBackend& backend, const std::string& name, const json& args) {
            auto* service = backend();
            if (!service)
                return make_error(ErrorInit{.code = ErrorCode::Unavailable,
                                            .domain = ErrorDomain::MCP,
                                            .user_message = "Shader service is unavailable.",
                                            .detection = LFS_SOURCE_SITE_CURRENT()});
            return shaderOperation(*service, name, args);
        }
    } // namespace
    void register_shader_tools(mcp::ToolRegistry& registry, vis::Visualizer* viewer, ShaderBackend backend) {
        for (const auto& tool : shaderTools()) {
            registry.register_tool(
                mcp::McpTool{
                    .name = "shader." + tool.name,
                    .description = tool.description,
                    .input_schema = {.type = "object", .properties = tool.properties, .required = tool.required},
                    .metadata = {.category = "shader",
                                 .kind = tool.query ? "query" : "mutation",
                                 .runtime = "gui",
                                 .thread_affinity = "gui_thread",
                                 .destructive = tool.name == "remove",
                                 .long_running = tool.name == "bind"}},
                [viewer, backend, name = tool.name](const json& args) -> json {
                    auto result = post_and_wait(viewer, [backend, name, args] { return invoke(backend, name, args); });
                    return result ? *result : json{{"error", core::to_wire_envelope(result.error())}};
                });
        }
    }

    void register_shader_resources(mcp::ResourceRegistry& registry, vis::Visualizer* viewer, ShaderBackend backend) {
        using ResourceResult = mcp::ResourceRegistry::ResourceHandler::result_type;
        const auto read = [viewer, backend](const std::string& uri) -> ResourceResult {
            auto result = post_and_wait(viewer, [backend, uri]() -> Result<json> {
                if (uri == "lichtfeld://shaders/catalog")
                    return shaderCatalog();
                if (uri == "lichtfeld://shaders/state") {
                    if (auto* service = backend())
                        return shaderState(service->inspect(false));
                    return invoke(backend, "list", json::object());
                }
                return invoke(backend, "get",
                              {{"shader_id", uri.substr(std::string("lichtfeld://shaders/assets/").size())}});
            });
            if (!result)
                return std::unexpected(std::string(result.error().user_message()));
            return std::vector<mcp::McpResourceContent>{{uri, "application/json", result->dump()}};
        };
        for (const auto* name : {"catalog", "state"})
            registry.register_resource({"lichtfeld://shaders/" + std::string(name), "Shaders " + std::string(name),
                                        "Copied shader service observation", "application/json"},
                                       read);
        registry.register_resource_prefix("lichtfeld://shaders/assets/", read);
    }
} // namespace lfs::app
