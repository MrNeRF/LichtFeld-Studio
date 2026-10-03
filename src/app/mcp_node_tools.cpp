/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "app/mcp_node_tools.hpp"
#include "core/logger.hpp"
#include "mcp_node_utils.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace lfs::app {
    namespace {
        using namespace node_mcp;

        json treeState(const lfs::nodes::NodeTree& tree) {
            return {{"success", true}, {"tree", tree.to_json()}};
        }

        bool finitePoint(const json& value) {
            return value.is_array() && value.size() == 2 &&
                   std::ranges::all_of(value, [](const auto& v) { return v.is_number() && std::isfinite(v.template get<float>()); });
        }

        // Suggestions only: identifiers are always resolved by exact lookup.
        json typeSuggestions(const vis::ModifierManager& manager, const std::string_view unknown) {
            std::vector<std::pair<std::size_t, std::string>> ranked;
            for (const auto& type : manager.registry().list()) {
                std::vector<std::size_t> previous(type->id.size() + 1), current(previous.size());
                std::iota(previous.begin(), previous.end(), 0);
                for (std::size_t i = 0; i < unknown.size(); ++i) {
                    current[0] = i + 1;
                    for (std::size_t j = 0; j < type->id.size(); ++j)
                        current[j + 1] = std::min({current[j] + 1, previous[j + 1] + 1,
                                                   previous[j] + (unknown[i] != type->id[j])});
                    previous.swap(current);
                }
                ranked.emplace_back(previous.back(), type->id);
            }
            std::ranges::sort(ranked);
            json result = json::array();
            for (std::size_t i = 0; i < std::min<std::size_t>(3, ranked.size()); ++i)
                result.push_back(ranked[i].second);
            return result;
        }

        json editNode(vis::VisualizerImpl& viewer, const json& args, const std::string_view operation) {
            auto& manager = viewer.getSceneManager()->modifierManager();
            auto* graph = tree(manager, args);
            if (!graph)
                return failure("Unknown graph UUID; read lichtfeld://nodes/trees", "tree");
            auto* node = graph->find_node(args.value("node", ""));
            if (!node)
                return failure("Unknown node name; read lichtfeld://nodes/trees/" + graph->uuid, "node");
            if (operation == "set_input") {
                const auto type = manager.registry().find(node->type_id);
                if (!type)
                    return failure("Node type is not registered", "node");
                const auto input = std::ranges::find(type->inputs, args.value("input", ""), &lfs::nodes::SocketDecl::identifier);
                if (input == type->inputs.end())
                    return failure("Unknown input identifier; read lichtfeld://nodes/types", "input");
                auto converted = value(args.at("value"), *input);
                if (!converted)
                    return failure(converted.error().message, "value");
                const auto result = manager.setNodeInput(graph->uuid, node->name, input->identifier, std::move(*converted));
                if (!result)
                    return failure(result.error().message, "value");
                return treeState(*graph);
            }
            const auto before = graph->to_json();
            if (operation == "remove") {
                if (!graph->remove_node(node->name))
                    return failure("Group Input and Group Output cannot be removed", "node");
            } else if (operation == "move") {
                if (!finitePoint(args.at("location")))
                    return failure("location must contain two finite coordinates", "location");
                node->location = args.at("location").get<std::array<float, 2>>();
            } else if (operation == "mute") {
                node->muted = args.at("muted").get<bool>();
            } else if (operation == "set_property") {
                const auto type = manager.registry().find(node->type_id);
                if (!type)
                    return failure("Node type is not registered", "node");
                const auto prop = std::ranges::find(type->properties, args.value("property", ""), &lfs::nodes::PropertyDecl::identifier);
                if (prop == type->properties.end())
                    return failure("Unknown property identifier; read lichtfeld://nodes/types", "property");
                const auto valid = property(args.at("value"), *prop);
                if (!valid)
                    return failure(valid.error().message, "value");
                node->properties[prop->identifier] = args.at("value");
            }
            manager.recordTreeEdit(graph->uuid, before);
            return treeState(*graph);
        }

        json editLink(vis::VisualizerImpl& viewer, const json& args, const bool remove) {
            auto& manager = viewer.getSceneManager()->modifierManager();
            auto* graph = tree(manager, args);
            if (!graph)
                return failure("Unknown graph UUID; read lichtfeld://nodes/trees", "tree");
            const lfs::nodes::Link link{args.at("from_node"), args.at("from_socket"), args.at("to_node"), args.at("to_socket")};
            const auto before = graph->to_json();
            std::string error;
            if (remove) {
                if (!graph->remove_link(link))
                    return failure("Link does not exist; read the graph's links", "from_socket");
            } else if (!graph->add_link(link, &error)) {
                const auto typeOf = [&](const std::string& name, const std::string& id, bool output) {
                    const auto* node = graph->find_node(name);
                    const auto type = node ? manager.registry().find(node->type_id) : nullptr;
                    if (type) {
                        const auto& sockets = output ? type->outputs : type->inputs;
                        const auto found = std::ranges::find(sockets, id, &lfs::nodes::SocketDecl::identifier);
                        if (found != sockets.end())
                            return found->type;
                    }
                    return std::string("unknown");
                };
                auto result = failure(error, "to_socket");
                result["from_type"] = typeOf(link.from_node, link.from_socket, true);
                result["to_type"] = typeOf(link.to_node, link.to_socket, false);
                return result;
            }
            manager.recordTreeEdit(graph->uuid, before);
            return treeState(*graph);
        }

        json editModifier(vis::VisualizerImpl& viewer, const json& args, const std::string_view operation) {
            auto& scene = *viewer.getSceneManager();
            auto& manager = scene.modifierManager();
            const auto host = target(scene, args);
            if (!host)
                return failure("target must be the UUID of a splat, mesh or point cloud; read lichtfeld://scene/nodes", "target");
            if (operation == "add") {
                const auto* graph = tree(manager, args);
                if (!graph)
                    return failure("Unknown graph UUID; create a graph first", "tree");
                const auto id = manager.addModifier(*host, graph->uuid, args.value("name", "")).uuid;
                auto result = stack(scene, *host);
                result["modifier"] = id;
                return result;
            }
            auto* item = modifier(manager, *host, args);
            if (!item)
                return failure("Unknown modifier UUID; read lichtfeld://nodes/stacks/" + host->to_string(), "modifier");
            const std::string name = item->name;
            if (operation == "remove")
                manager.removeModifier(*host, name);
            else if (operation == "move") {
                const int index = args.at("index").get<int>();
                if (index < 0 || static_cast<std::size_t>(index) >= manager.stack(*host)->modifiers.size())
                    return failure("index must be within the modifier stack", "index");
                manager.moveModifier(*host, name, static_cast<std::size_t>(index));
            } else if (operation == "apply" || operation == "capture_selection") {
                const auto result = operation == "apply" ? manager.applyModifier(*host, name)
                                                         : manager.captureSelection(*host, name, args.at("node").get<std::string>());
                if (!result)
                    return failure(result.error().message, operation == "apply" ? "modifier" : "node");
            } else if (operation == "set") {
                auto replacement = *item;
                if (args.contains("enabled"))
                    replacement.enabled = args.at("enabled").get<bool>();
                if (args.contains("show_viewport"))
                    replacement.show_viewport = args.at("show_viewport").get<bool>();
                if (args.contains("name")) {
                    const auto new_name = args.at("name").get<std::string>();
                    if (new_name.empty() || std::ranges::any_of(manager.stack(*host)->modifiers, [&](const auto& other) { return other.uuid != item->uuid && other.name == new_name; }))
                        return failure("Modifier name must be non-empty and unique in its stack", "name");
                    replacement.name = new_name;
                }
                if (args.contains("input_overrides")) {
                    const auto* graph = manager.tree(item->tree_uuid);
                    if (!graph)
                        return failure("The modifier's graph was deleted", "modifier");
                    for (const auto& [id, raw] : args.at("input_overrides").items()) {
                        const auto socket = std::ranges::find(graph->interface.inputs, id, &lfs::nodes::InterfaceSocket::identifier);
                        if (socket == graph->interface.inputs.end())
                            return failure("Unknown graph interface input: " + id, "input_overrides");
                        if (raw.is_null()) {
                            replacement.input_overrides.erase(id);
                            continue;
                        }
                        const auto converted = value(raw, {.identifier = socket->identifier, .type = socket->type, .min = socket->min, .max = socket->max});
                        if (!converted)
                            return failure(converted.error().message, "input_overrides");
                        replacement.input_overrides[id] = *converted;
                    }
                }
                const json before = manager.stack(*host)->modifiers;
                if (json(*item) != json(replacement)) {
                    *item = std::move(replacement);
                    manager.recordStackEdit(*host, before);
                }
            }
            return stack(scene, *host);
        }
    } // namespace

    void register_gui_node_tools(mcp::ToolRegistry& registry, vis::Visualizer* viewer) {
        using namespace node_mcp;
        auto* impl = dynamic_cast<vis::VisualizerImpl*>(viewer);
        if (!impl)
            return;
        add(registry, impl, "nodes.tree_create", "Create an undoable node graph with linked Group Input and Group Output", {{"name", stringSchema()}}, {},
            [](auto& viewer, const json& args) { return treeState(viewer.getSceneManager()->modifierManager().newTree(args.value("name", "Node Graph"))); });
        add(registry, impl, "nodes.tree_import_json", "Import a graph JSON object as a new graph with a fresh UUID", {{"json", {{"type", "object"}}}}, {"json"},
            [](auto& viewer, const json& args) {
                auto& manager = viewer.getSceneManager()->modifierManager();
                auto data = args.at("json");
                data["uuid"] = core::generate_uuid_v4().to_string();
                try {
                    if (!data.contains("nodes") || !data["nodes"].is_array())
                        return failure("Graph JSON needs a nodes array with Group Input and Group Output", "json");
                    std::unordered_set<std::string> names;
                    std::size_t inputs = 0;
                    std::size_t outputs = 0;
                    for (const auto& node : data["nodes"]) {
                        const auto name = node.value("name", "");
                        if (name.empty() || !names.insert(name).second)
                            return failure("Imported node names must be non-empty and unique", "json");
                        inputs += node.value("type_id", "") == "lfs.group_input";
                        outputs += node.value("type_id", "") == "lfs.group_output";
                    }
                    if (inputs != 1 || outputs != 1)
                        return failure("Graph JSON must contain exactly one Group Input and one Group Output", "json");
                    return treeState(manager.loadTree(data));
                } catch (const std::exception& error) {
                    LOG_WARN("MCP graph import rejected: {}", error.what());
                    return failure(std::string("Invalid graph JSON: ") + error.what(), "json");
                }
            });
        for (const std::string operation : {"delete", "rename", "export_json"}) {
            json properties = {{"tree", stringSchema()}};
            std::vector<std::string> required = {"tree"};
            if (operation == "rename") {
                properties["name"] = stringSchema();
                required.push_back("name");
            }
            add(registry, impl, "nodes.tree_" + operation, operation + " a node graph identified by its exact UUID", properties, required, [operation](auto& viewer, const json& args) {
                    auto& manager = viewer.getSceneManager()->modifierManager();
                    auto* graph = tree(manager, args);
                    if (!graph)
                        return failure("Unknown graph UUID; read lichtfeld://nodes/trees", "tree");
                    if (operation == "delete") {
                        manager.removeTree(graph->uuid);
                        return json{{"success", true}, {"trees", manager.toJson()["trees"]}, {"stacks", stacks(*viewer.getSceneManager())}};
                    }
                    if (operation == "rename") {
                        const auto name = args.at("name").get<std::string>();
                        if (name.empty())
                            return failure("Graph name cannot be empty", "name");
                        const auto before = graph->to_json();
                        graph->name = name;
                        manager.recordTreeEdit(graph->uuid, before);
                    }
                    return treeState(*graph); }, operation == "export_json", operation == "delete");
        }
        add(registry, impl, "nodes.node_add", "Add a registered node type; omit location for placement beside existing nodes", {{"tree", stringSchema()}, {"type_id", stringSchema()}, {"name", stringSchema()}, {"location", pointSchema()}}, {"tree", "type_id"},
            [](auto& viewer, const json& args) {
                auto& manager = viewer.getSceneManager()->modifierManager();
                auto* graph = tree(manager, args);
                if (!graph)
                    return failure("Unknown graph UUID; read lichtfeld://nodes/trees", "tree");
                const auto id = args.at("type_id").get<std::string>();
                if (!manager.registry().find(id)) {
                    auto result = failure("Unknown type_id; use an exact id from lichtfeld://nodes/types", "type_id");
                    result["suggestions"] = typeSuggestions(manager, id);
                    result["available_types"] = json::array();
                    for (const auto& type : manager.registry().list())
                        result["available_types"].push_back(type->id);
                    return result;
                }
                if (args.contains("location") && !finitePoint(args.at("location")))
                    return failure("location must contain two finite coordinates", "location");
                if (args.contains("name") && graph->find_node(args.at("name").get<std::string>()))
                    return failure("Node name already exists in this graph", "name");
                std::array<float, 2> location = {0.0f, 0.0f};
                for (const auto& node : graph->nodes)
                    location[0] = std::max(location[0], node.location[0] + 288.0f);
                if (args.contains("location"))
                    location = args.at("location").get<std::array<float, 2>>();
                const auto before = graph->to_json();
                auto& node = graph->add_node(id, args.value("name", ""));
                node.location = location;
                const auto name = node.name;
                manager.recordTreeEdit(graph->uuid, before);
                auto result = treeState(*graph);
                result["node"] = name;
                return result;
            });
        for (const std::string operation : {"remove", "set_input", "set_property", "mute", "move"}) {
            json properties = {{"tree", stringSchema()}, {"node", stringSchema()}};
            std::vector<std::string> required = {"tree", "node"};
            if (operation == "set_input" || operation == "set_property") {
                const std::string key = operation == "set_input" ? "input" : "property";
                properties[key] = stringSchema();
                properties["value"] = json::object();
                required.insert(required.end(), {key, "value"});
            } else if (operation == "mute" || operation == "move") {
                const std::string key = operation == "mute" ? "muted" : "location";
                properties[key] = operation == "mute" ? boolSchema() : pointSchema();
                required.push_back(key);
            }
            add(registry, impl, "nodes.node_" + operation, operation + " a graph node (undoable); exact node name required", properties, required, [operation](auto& viewer, const json& args) { return editNode(viewer, args, operation); }, false, operation == "remove");
        }
        for (const bool remove : {false, true})
            add(registry, impl, remove ? "nodes.unlink" : "nodes.link", remove ? "Remove an exact graph link" : "Connect compatible sockets; supports multi-input sockets", {{"tree", stringSchema()}, {"from_node", stringSchema()}, {"from_socket", stringSchema()}, {"to_node", stringSchema()}, {"to_socket", stringSchema()}}, {"tree", "from_node", "from_socket", "to_node", "to_socket"}, [remove](auto& viewer, const json& args) { return editLink(viewer, args, remove); }, false, remove);
        for (const std::string operation : {"add", "remove", "move", "set", "apply", "capture_selection"}) {
            json properties = {{"target", stringSchema()}};
            std::vector<std::string> required = {"target"};
            const std::string key = operation == "add" ? "tree" : "modifier";
            properties[key] = stringSchema();
            required.push_back(key);
            if (operation == "add" || operation == "set")
                properties["name"] = stringSchema();
            if (operation == "set") {
                properties["enabled"] = boolSchema();
                properties["show_viewport"] = boolSchema();
                properties["input_overrides"] = {{"type", "object"}};
            } else if (operation == "move") {
                properties["index"] = {{"type", "integer"}, {"minimum", 0}};
                required.push_back("index");
            } else if (operation == "capture_selection") {
                properties["node"] = stringSchema();
                required.push_back("node");
            }
            add(registry, impl, "nodes.modifier_" + operation, operation + " a scene node's modifier (undoable)", properties, required, [operation](auto& viewer, const json& args) { return editModifier(viewer, args, operation); }, false, operation == "remove" || operation == "apply");
        }
        register_gui_node_editor_tools(registry, viewer);
    }
} // namespace lfs::app
