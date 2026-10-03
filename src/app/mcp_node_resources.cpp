/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "app/mcp_node_tools.hpp"
#include "mcp_node_utils.hpp"
#include "visualizer/gui/gui_manager.hpp"
#include "visualizer/gui/rmlui/elements/node_canvas_element.hpp"
#include "visualizer/gui/screen_host.hpp"

#include <algorithm>
#include <cmath>

namespace lfs::app::node_mcp {
    namespace {
        json socket(const lfs::nodes::SocketDecl& value) {
            json result = {{"identifier", value.identifier}, {"label", value.label}, {"type", value.type}, {"default", value.default_value}, {"field", value.field}, {"multi_input", value.multi_input}, {"hide_value", value.hide_value}};
            const auto number = [&](const char* name, const std::optional<double>& v) {
                result[name] = v ? json(*v) : json(nullptr);
            };
            number("min", value.min);
            number("max", value.max);
            number("step", value.step);
            number("soft_min", value.soft_min);
            number("soft_max", value.soft_max);
            return result;
        }
        json evaluation(const vis::ModifierEvaluation* value, const vis::ModifierStack* stack) {
            if (!value)
                return nullptr;
            json nodes = json::object();
            for (const auto& [name, result] : value->nodes) {
                json errors = json::array();
                if (stack) {
                    const auto separator = name.find('/');
                    const auto modifier = std::ranges::find(stack->modifiers, name.substr(0, separator), &vis::Modifier::uuid);
                    if (separator != std::string::npos && modifier != stack->modifiers.end()) {
                        const auto error = value->errors.find(modifier->name + name.substr(separator));
                        if (error != value->errors.end())
                            errors.push_back(error->second);
                    }
                }
                nodes[name] = {{"ok", errors.empty()}, {"errors", std::move(errors)}, {"time_ms", result.time_ms}, {"cached", result.cached}, {"element_count", result.element_count ? json(*result.element_count) : json(nullptr)}, {"selected_share", result.selected_share ? json(*result.selected_share) : json(nullptr)}};
            }
            return {{"ok", value->ok}, {"unchanged", value->unchanged}, {"errors", value->errors}, {"time_ms", value->total_time_ms}, {"nodes", nodes}};
        }
    } // namespace

    json types(vis::ModifierManager& manager) {
        json result = json::array();
        constexpr const char* kinds[] = {"enum", "string", "int", "float", "bool", "data"};
        for (const auto& descriptor : manager.registry().list()) {
            const auto& type = *descriptor;
            json inputs = json::array();
            json outputs = json::array();
            json properties = json::array();
            for (const auto& input : type.inputs)
                inputs.push_back(socket(input));
            for (const auto& output : type.outputs)
                outputs.push_back(socket(output));
            for (const auto& prop : type.properties)
                properties.push_back({{"identifier", prop.identifier}, {"label", prop.label}, {"kind", kinds[static_cast<int>(prop.kind)]}, {"default", prop.default_value}, {"items", prop.items}, {"min", prop.min ? json(*prop.min) : json(nullptr)}, {"max", prop.max ? json(*prop.max) : json(nullptr)}});
            result.push_back({{"id", type.id}, {"label", type.label}, {"category", type.category}, {"description", type.description}, {"version", type.version}, {"tree_types", type.tree_types}, {"inputs", inputs}, {"outputs", outputs}, {"properties", properties}});
        }
        return result;
    }

    json progress(const vis::ModifierWorkerProgress& value) {
        return {{"busy", value.busy}, {"target", value.host.to_string()}, {"modifier", value.modifier}, {"node", value.node}, {"label", value.label}, {"completed", value.completed}, {"total", value.total}, {"current", value.busy ? std::min(value.completed + 1, value.total) : value.completed}, {"generation", value.generation}};
    }

    json stack(vis::SceneManager& scene, const core::Uuid& target) {
        auto& manager = scene.modifierManager();
        const auto* node = scene.getScene().getNodeByUuid(target);
        if (!node)
            return failure("Unknown scene node UUID; read lichtfeld://scene/nodes", "target");
        const auto* value = manager.stack(target);
        return {{"success", true}, {"target", target.to_string()}, {"name", node->name}, {"modifiers", value ? json(value->modifiers) : json::array()}, {"evaluation", evaluation(manager.lastResult(target), value)}, {"progress", progress(manager.progress())}};
    }

    json stacks(vis::SceneManager& scene) {
        json result = json::array();
        for (const auto* node : scene.getScene().getNodes())
            if (scene.modifierManager().stack(node->uuid))
                result.push_back(stack(scene, node->uuid));
        return result;
    }

    json editor(vis::VisualizerImpl& viewer) {
        bool open = false;
        for (const auto id : viewer.screens().screen().areas())
            open |= viewer.screens().screen().area(id)->editor == "node_editor";
        json result = {{"success", true}, {"open", open}};
        if (open)
            if (auto* canvas = viewer.getGuiManager()->screenHost().nodeCanvas()) {
                canvas->refresh();
                result.update(canvas->viewState());
            }
        result["progress"] = progress(viewer.getSceneManager()->modifierManager().progress());
        return result;
    }

    std::optional<core::Uuid> target(vis::SceneManager& scene, const json& args) {
        const auto id = core::Uuid::from_string(args.value("target", ""));
        const auto* node = id ? scene.getScene().getNodeByUuid(*id) : nullptr;
        if (!node || (node->type != core::NodeType::SPLAT && node->type != core::NodeType::MESH && node->type != core::NodeType::POINTCLOUD))
            return std::nullopt;
        return id;
    }

    lfs::nodes::NodeTree* tree(vis::ModifierManager& manager, const json& args) {
        const auto id = args.value("tree", "");
        auto* value = manager.tree(id);
        return value && value->uuid == id ? value : nullptr;
    }

    vis::Modifier* modifier(vis::ModifierManager& manager, const core::Uuid& host, const json& args) {
        auto* value = manager.stack(host);
        if (!value)
            return nullptr;
        const auto found = std::ranges::find(value->modifiers, args.value("modifier", ""), &vis::Modifier::uuid);
        return found == value->modifiers.end() ? nullptr : &*found;
    }

    std::expected<lfs::nodes::Value, ArgumentError> value(const json& input, const lfs::nodes::SocketDecl& socket) {
        using namespace lfs::nodes;
        const auto invalid = [&]() { return std::unexpected(ArgumentError{"Expected " + socket.type + " value for " + socket.identifier}); };
        if (socket.type == BOOL_SOCKET && input.is_boolean())
            return Value(input.get<bool>());
        if (socket.type == STRING_SOCKET && input.is_string())
            return Value(input.get<std::string>());
        if ((socket.type == FLOAT_SOCKET || socket.type == INT_SOCKET) && input.is_number()) {
            const double number = input.get<double>();
            if (!std::isfinite(number) || (socket.min && number < *socket.min) || (socket.max && number > *socket.max))
                return std::unexpected(ArgumentError{"Value is non-finite or outside the socket range; read lichtfeld://nodes/types"});
            if (socket.type == FLOAT_SOCKET)
                return Value(static_cast<float>(number));
            if (input.is_number_integer())
                return Value(input.get<std::int64_t>());
        }
        if ((socket.type == VECTOR_SOCKET || socket.type == COLOUR_SOCKET) && input.is_array() &&
            (input.size() == 3 || (socket.type == COLOUR_SOCKET && input.size() == 4))) {
            for (const auto& number : input)
                if (!number.is_number() || !std::isfinite(number.get<float>()))
                    return invalid();
            if (socket.type == VECTOR_SOCKET)
                return Value(glm::vec3(input[0].get<float>(), input[1].get<float>(), input[2].get<float>()));
            return Value(glm::vec4(input[0].get<float>(), input[1].get<float>(), input[2].get<float>(), input.size() == 4 ? input[3].get<float>() : 1.0f));
        }
        return invalid();
    }

    std::expected<void, ArgumentError> property(const json& input, const lfs::nodes::PropertyDecl& declaration) {
        using lfs::nodes::PropertyKind;
        bool valid = false;
        switch (declaration.kind) {
        case PropertyKind::Enum:
            valid = input.is_string() && std::ranges::find(declaration.items, input.get<std::string>()) != declaration.items.end();
            break;
        case PropertyKind::String: valid = input.is_string(); break;
        case PropertyKind::Bool: valid = input.is_boolean(); break;
        case PropertyKind::Data: valid = true; break;
        case PropertyKind::Int: valid = input.is_number_integer(); break;
        case PropertyKind::Float: valid = input.is_number(); break;
        }
        if (valid && input.is_number()) {
            const double number = input.get<double>();
            valid = std::isfinite(number) && (!declaration.min || number >= *declaration.min) && (!declaration.max || number <= *declaration.max);
        }
        if (!valid)
            return std::unexpected(ArgumentError{"Invalid property value; use the kind, range and enum items in lichtfeld://nodes/types"});
        return {};
    }

    void add(mcp::ToolRegistry& registry, vis::VisualizerImpl* viewer, std::string name,
             std::string description, json properties, std::vector<std::string> required, Handler handler,
             const bool read_only, const bool destructive) {
        registry.register_tool(mcp::McpTool{
                                   .name = std::move(name),
                                   .description = std::move(description),
                                   .input_schema = {.type = "object", .properties = std::move(properties), .required = std::move(required)},
                                   .metadata = mcp::McpToolMetadata{.category = "nodes", .kind = read_only ? "query" : "command", .runtime = "gui", .thread_affinity = "gui_thread", .destructive = destructive}},
                               [viewer, handler = std::move(handler)](const json& args) {
                                   return post_and_wait(viewer, [&]() { return handler(*viewer, args); });
                               });
    }
} // namespace lfs::app::node_mcp

namespace lfs::app {
    void register_gui_node_resources(mcp::ResourceRegistry& registry, vis::Visualizer* viewer) {
        auto* impl = dynamic_cast<vis::VisualizerImpl*>(viewer);
        if (!impl)
            return;
        // ResourceRegistry owns the legacy resource-boundary result contract.
        using ResourceResult = mcp::ResourceRegistry::ResourceHandler::result_type;
        const auto read = [impl](const std::string& uri) -> ResourceResult {
            return post_and_wait(impl, [impl, uri]() -> ResourceResult {
                auto& scene = *impl->getSceneManager();
                auto& manager = scene.modifierManager();
                nlohmann::json result;
                if (uri == "lichtfeld://nodes/types")
                    result = node_mcp::types(manager);
                else if (uri == "lichtfeld://nodes/trees") {
                    result = nlohmann::json::array();
                    for (const auto* tree : manager.trees())
                        result.push_back({{"uuid", tree->uuid}, {"name", tree->name}, {"nodes", tree->nodes.size()}});
                } else if (uri.starts_with("lichtfeld://nodes/trees/")) {
                    const auto* tree = node_mcp::tree(manager, {{"tree", uri.substr(std::string_view("lichtfeld://nodes/trees/").size())}});
                    if (!tree)
                        return std::unexpected("Unknown graph UUID; read lichtfeld://nodes/trees");
                    result = tree->to_json();
                } else if (uri == "lichtfeld://nodes/stacks")
                    result = node_mcp::stacks(scene);
                else if (uri.starts_with("lichtfeld://nodes/stacks/")) {
                    const auto id = core::Uuid::from_string(uri.substr(std::string_view("lichtfeld://nodes/stacks/").size()));
                    if (!id)
                        return std::unexpected("Invalid scene node UUID");
                    result = node_mcp::stack(scene, *id);
                } else if (uri == "lichtfeld://nodes/editor")
                    result = node_mcp::editor(*impl);
                else
                    return std::unexpected("Unknown node resource URI: " + uri);
                return single_json_resource(uri, result);
            });
        };
        for (const auto& name : {"types", "trees", "stacks", "editor"})
            registry.register_resource(mcp::McpResource{.uri = "lichtfeld://nodes/" + std::string(name), .name = "Nodes: " + std::string(name), .description = "Node system state and descriptors", .mime_type = "application/json"}, read);
        registry.register_resource_prefix("lichtfeld://nodes/", read);
    }
} // namespace lfs::app
