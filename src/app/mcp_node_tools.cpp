/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "app/mcp_node_tools.hpp"
#include "core/logger.hpp"
#include "mcp_node_utils.hpp"

#include <SDL3/SDL_clipboard.h>

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
                const lfs::nodes::TreeResolver resolver = [&](const std::string_view uuid) {
                    return manager.tree(uuid);
                };
                const auto inputs = lfs::nodes::effective_inputs(*graph, *node, resolver);
                const auto input = std::ranges::find(inputs, args.value("input", ""), &lfs::nodes::SocketDecl::identifier);
                if (input == inputs.end())
                    return failure("Unknown input identifier; read the graph resource for resolved sockets", "input");
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
                if (node->type_id == "lfs.group" && prop->identifier == "tree") {
                    const auto result = manager.setGroupGraph(graph->uuid, node->name,
                                                              args.at("value").get<std::string>());
                    if (!result)
                        return failure(result.error().message, "value");
                    return treeState(*graph);
                }
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
            } else if (!graph->add_link(link, &error, [&](const std::string_view uuid) {
                           return manager.tree(uuid);
                       })) {
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
                        const auto socket = std::ranges::find(graph->group_interface.inputs, id, &lfs::nodes::InterfaceSocket::identifier);
                        if (socket == graph->group_interface.inputs.end())
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
        for (const bool remove : {false, true}) {
            add(registry, impl, remove ? "nodes.keyframe_remove" : "nodes.keyframe_set",
                remove ? "Remove an input keyframe at the sequencer playhead or supplied time" : "Keyframe an unlinked node input using the sequencer animation track",
                {{"tree", stringSchema()}, {"node", stringSchema()}, {"input", stringSchema()}, {"time", {{"type", "number"}, {"minimum", 0}}}, {"value", json::object()}, {"easing", {{"type", "integer"}, {"minimum", 0}, {"maximum", 3}}}},
                {"tree", "node", "input"}, [remove](auto& viewer, const json& args) {
                    auto& manager = viewer.getSceneManager()->modifierManager();
                    auto* graph = tree(manager, args);
                    auto* node = graph ? graph->find_node(args.value("node", "")) : nullptr;
                    if (!node)
                        return failure("Unknown graph or node", "node");
                    const auto inputs = lfs::nodes::effective_inputs(*graph, *node, [&](std::string_view id) { return manager.tree(id); });
                    const auto socket = std::ranges::find(inputs, args.value("input", ""), &lfs::nodes::SocketDecl::identifier);
                    if (socket == inputs.end())
                        return failure("Unknown input identifier", "input");
                    std::optional<lfs::nodes::Value> supplied;
                    if (args.contains("value")) {
                        auto converted = value(args.at("value"), *socket);
                        if (!converted)
                            return failure(converted.error().message, "value");
                        supplied = std::move(*converted);
                    }
                    const auto time = args.contains("time") ? std::optional{args.at("time").get<float>()} : std::nullopt;
                    const auto result = remove ? manager.keyframeRemove(graph->uuid, node->name, socket->identifier, time)
                                               : manager.keyframeSet(graph->uuid, node->name, socket->identifier, time, supplied, args.value("easing", 0));
                    if (!result)
                        return failure(result.error().message, "input");
                    return json{{"success", true}, {"animation", manager.animationJson()}};
                });
        }
        add(registry, impl, "nodes.node_rename", "Rename a node and update links and sequencer target paths atomically",
            {{"tree", stringSchema()}, {"node", stringSchema()}, {"name", stringSchema()}}, {"tree", "node", "name"},
            [](auto& viewer, const json& args) {
                auto& manager = viewer.getSceneManager()->modifierManager();
                const auto result = manager.renameNode(args.at("tree").get<std::string>(), args.at("node").get<std::string>(), args.at("name").get<std::string>());
                return result ? json{{"success", true}} : failure(result.error().message, "name");
            });
        add(registry, impl, "nodes.tree_create", "Create an undoable node graph with linked Group Input and Group Output", {{"name", stringSchema()}}, {},
            [](auto& viewer, const json& args) { return treeState(viewer.getSceneManager()->modifierManager().newTree(args.value("name", "Node Graph"))); });
        add(registry, impl, "nodes.tree_import_json", "Import a graph JSON object as a new graph with a fresh UUID", {{"json", {{"type", "object"}}}}, {"json"},
            [](auto& viewer, const json& args) {
                auto& manager = viewer.getSceneManager()->modifierManager();
                auto data = args.at("json");
                if (data.contains("animation") && !data.contains("animation_tree"))
                    data["animation_tree"] = data.value("uuid", "");
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
            add(registry, impl, "nodes.tree_" + operation, operation + " a node graph identified by UUID or unique exact name", properties, required, [operation](auto& viewer, const json& args) {
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
                    auto result = treeState(*graph);
                    if (operation == "export_json") {
                        result["tree"]["animation_tree"] = graph->uuid;
                        result["tree"]["animation"] = manager.animationJson();
                    }
                    return result; }, operation == "export_json", operation == "delete");
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

        add(registry, impl, "nodes.copy", "Serialize nodes and their internal links to the portable LichtFeld node clipboard format", {{"tree", stringSchema()}, {"nodes", {{"type", "array"}, {"items", stringSchema()}}}}, {"tree", "nodes"}, [](auto& viewer, const json& args) {
                auto& manager = viewer.getSceneManager()->modifierManager();
                const auto result = manager.copyNodes(args.at("tree").get<std::string>(), args.at("nodes").get<std::vector<std::string>>());
                if (!result)
                    return failure(result.error().message, "nodes");
                SDL_SetClipboardText(result->c_str());
                return json{{"success", true}, {"clipboard", *result}}; }, true);
        add(registry, impl, "nodes.paste", "Paste a portable LichtFeld node clipboard into a graph as one undo step",
            {{"tree", stringSchema()}, {"clipboard", stringSchema()}, {"location", pointSchema()}},
            {"tree", "clipboard"}, [](auto& viewer, const json& args) {
                auto& manager = viewer.getSceneManager()->modifierManager();
                std::optional<std::array<float, 2>> location;
                if (args.contains("location")) {
                    if (!finitePoint(args["location"]))
                        return failure("location must contain two finite coordinates", "location");
                    location = args["location"].get<std::array<float, 2>>();
                }
                const auto result = manager.pasteNodes(args.at("tree").get<std::string>(), args.at("clipboard").get<std::string>(), location);
                if (!result)
                    return failure(result.error().message, "clipboard");
                return json{{"success", true}, {"nodes", result->nodes}, {"dropped_links", result->dropped_links}};
            });
        add(registry, impl, "nodes.group_make", "Move selected nodes into a reusable graph and replace them with one Group node",
            {{"tree", stringSchema()}, {"nodes", {{"type", "array"}, {"items", stringSchema()}}}, {"name", stringSchema()}},
            {"tree", "nodes"}, [](auto& viewer, const json& args) {
                auto& manager = viewer.getSceneManager()->modifierManager();
                const auto result = manager.makeGroup(args.at("tree").get<std::string>(), args.at("nodes").get<std::vector<std::string>>(), args.value("name", "Group"));
                if (!result)
                    return failure(result.error().message, "nodes");
                return json{{"success", true}, {"group_node", result->group_node}, {"graph", result->graph}};
            });
        add(registry, impl, "nodes.group_ungroup", "Inline one Group node while keeping its referenced graph",
            {{"tree", stringSchema()}, {"node", stringSchema()}}, {"tree", "node"},
            [](auto& viewer, const json& args) {
                auto& manager = viewer.getSceneManager()->modifierManager();
                const auto result = manager.ungroup(args.at("tree").get<std::string>(), args.at("node").get<std::string>());
                if (!result)
                    return failure(result.error().message, "node");
                return json{{"success", true}, {"nodes", *result}};
            });
        add(registry, impl, "nodes.group_set_graph", "Assign a same-type acyclic graph to a Group node",
            {{"tree", stringSchema()}, {"node", stringSchema()}, {"graph", stringSchema()}},
            {"tree", "node", "graph"}, [](auto& viewer, const json& args) {
                auto& manager = viewer.getSceneManager()->modifierManager();
                const auto result = manager.setGroupGraph(args.at("tree").get<std::string>(), args.at("node").get<std::string>(), args.at("graph").get<std::string>());
                if (!result)
                    return failure(result.error().message, "graph");
                return treeState(*manager.tree(args.at("tree").get<std::string>()));
            });
        add(registry, impl, "nodes.group_make_single_user", "Copy a Group node's graph and assign the copy only to that node",
            {{"tree", stringSchema()}, {"node", stringSchema()}}, {"tree", "node"},
            [](auto& viewer, const json& args) {
                auto& manager = viewer.getSceneManager()->modifierManager();
                const auto result = manager.makeGroupSingleUser(args.at("tree").get<std::string>(), args.at("node").get<std::string>());
                if (!result)
                    return failure(result.error().message, "node");
                return treeState(*manager.tree(args.at("tree").get<std::string>()));
            });

        const auto interface_properties = json{{"tree", stringSchema()},
                                               {"side", {{"type", "string"}, {"enum", {"input", "output"}}}},
                                               {"identifier", stringSchema()},
                                               {"type", stringSchema()},
                                               {"label", stringSchema()},
                                               {"default", json::object()},
                                               {"min", {{"type", {"number", "null"}}}},
                                               {"max", {{"type", {"number", "null"}}}},
                                               {"step", {{"type", {"number", "null"}}}},
                                               {"index", {{"type", "integer"}, {"minimum", 0}}}};
        add(registry, impl, "nodes.interface_add", "Add one stable-identifier graph interface socket",
            interface_properties, {"tree", "side", "type", "label"},
            [](auto& viewer, const json& args) {
                auto& manager = viewer.getSceneManager()->modifierManager();
                std::string type = args.at("type");
                if (!type.starts_with("lfs."))
                    type = "lfs." + type;
                lfs::nodes::Value initial;
                if (args.contains("default")) {
                    const auto converted = value(args["default"], {.identifier = "default", .type = type});
                    if (!converted)
                        return failure(converted.error().message, "default");
                    initial = *converted;
                }
                const auto number = [&](const char* key) -> std::optional<double> {
                    return args.contains(key) && args[key].is_number()
                               ? std::optional<double>{args[key].get<double>()}
                               : std::nullopt;
                };
                const auto result = manager.interfaceAdd(args.at("tree").get<std::string>(), args.at("side") == "output",
                                                         type, args.at("label").get<std::string>(), initial,
                                                         number("min"), number("max"), number("step"));
                if (!result)
                    return failure(result.error().message, "type");
                return json{{"success", true}, {"identifier", *result}};
            });
        for (const std::string operation : {"remove", "update", "move"})
            add(registry, impl, "nodes.interface_" + operation, operation + " one graph interface socket", interface_properties, operation == "move" ? std::vector<std::string>{"tree", "side", "identifier", "index"} : std::vector<std::string>{"tree", "side", "identifier"}, [operation](auto& viewer, const json& args) {
                    auto& manager = viewer.getSceneManager()->modifierManager();
                    const bool output = args.at("side") == "output";
                    vis::ModifierResult result;
                    if (operation == "remove")
                        result = manager.interfaceRemove(args.at("tree").get<std::string>(), output, args.at("identifier").get<std::string>());
                    else if (operation == "move")
                        result = manager.interfaceMove(args.at("tree").get<std::string>(), output, args.at("identifier").get<std::string>(), args.at("index").get<std::size_t>());
                    else {
                        auto* graph = manager.tree(args.at("tree").get<std::string>());
                        if (!graph)
                            return failure("Node graph does not exist", "tree");
                        auto& sockets = output ? graph->group_interface.outputs : graph->group_interface.inputs;
                        const auto socket = std::ranges::find(sockets, args.at("identifier").get<std::string>(), &lfs::nodes::InterfaceSocket::identifier);
                        if (socket == sockets.end())
                            return failure("Interface socket does not exist", "identifier");
                        json changes;
                        for (const auto* key : {"label", "min", "max", "step"})
                            if (args.contains(key))
                                changes[key] = args[key];
                        if (args.contains("default")) {
                            const auto converted = value(args["default"], {.identifier = socket->identifier, .type = socket->type, .min = socket->min, .max = socket->max});
                            if (!converted)
                                return failure(converted.error().message, "default");
                            changes["default"] = *converted;
                        }
                        result = manager.interfaceUpdate(args.at("tree").get<std::string>(), output, args.at("identifier").get<std::string>(), changes);
                    }
                    if (!result)
                        return failure(result.error().message, "identifier");
                    return treeState(*manager.tree(args.at("tree").get<std::string>())); }, false, operation == "remove");

        add(registry, impl, "nodes.frame_wrap", "Wrap nodes in a new explicit-membership Frame",
            {{"tree", stringSchema()}, {"nodes", {{"type", "array"}, {"items", stringSchema()}}}, {"label", stringSchema()}},
            {"tree", "nodes"}, [](auto& viewer, const json& args) {
                auto& manager = viewer.getSceneManager()->modifierManager();
                const auto result = manager.frameWrap(args.at("tree").get<std::string>(), args.at("nodes").get<std::vector<std::string>>(), args.value("label", "Frame"));
                if (!result)
                    return failure(result.error().message, "nodes");
                return json{{"success", true}, {"frame", *result}};
            });
        add(registry, impl, "nodes.frame_set_members", "Replace a Frame's explicit member list",
            {{"tree", stringSchema()}, {"frame", stringSchema()}, {"nodes", {{"type", "array"}, {"items", stringSchema()}}}},
            {"tree", "frame", "nodes"}, [](auto& viewer, const json& args) {
                auto& manager = viewer.getSceneManager()->modifierManager();
                const auto result = manager.frameSetMembers(args.at("tree").get<std::string>(), args.at("frame").get<std::string>(), args.at("nodes").get<std::vector<std::string>>());
                if (!result)
                    return failure(result.error().message, "nodes");
                return treeState(*manager.tree(args.at("tree").get<std::string>()));
            });
        add(registry, impl, "nodes.reroute_insert", "Split an exact link with a typed pass-through Reroute",
            {{"tree", stringSchema()}, {"link", {{"type", "object"}}}, {"location", pointSchema()}},
            {"tree", "link"}, [](auto& viewer, const json& args) {
                auto& manager = viewer.getSceneManager()->modifierManager();
                const auto& item = args.at("link");
                const lfs::nodes::Link link{item.value("from_node", ""), item.value("from_socket", ""), item.value("to_node", ""), item.value("to_socket", "")};
                std::optional<std::array<float, 2>> location;
                if (args.contains("location"))
                    location = args["location"].get<std::array<float, 2>>();
                const auto result = manager.rerouteInsert(args.at("tree").get<std::string>(), link, location);
                if (!result)
                    return failure(result.error().message, "link");
                return json{{"success", true}, {"node", *result}};
            });
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
