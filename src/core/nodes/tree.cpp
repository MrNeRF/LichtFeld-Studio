/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/nodes/tree.hpp"

#include "core/uuid.hpp"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <unordered_set>

namespace lfs::nodes {
    namespace {

        SocketDecl interface_decl(const InterfaceSocket& socket) {
            SocketDecl result{socket.identifier, socket.label, socket.type, socket.default_value};
            result.min = socket.min;
            result.max = socket.max;
            result.step = socket.step;
            // Interface values may carry lazy fields across a group boundary. Geometry
            // and strings are concrete values; numeric/vector values are field-capable.
            result.field = socket.type != GEOMETRY_SOCKET && socket.type != STRING_SOCKET;
            return result;
        }

        std::vector<SocketDecl> effective_sockets_impl(const NodeTree& tree, const Node& node,
                                                       const bool output, const TreeResolver& resolver,
                                                       std::unordered_set<std::string>& visiting) {
            if (node.type_id == "lfs.group_input") {
                if (!output)
                    return {};
                std::vector<SocketDecl> result;
                result.reserve(tree.interface.inputs.size());
                for (const auto& item : tree.interface.inputs)
                    result.push_back(interface_decl(item));
                return result;
            }
            if (node.type_id == "lfs.group_output") {
                if (output)
                    return {};
                std::vector<SocketDecl> result;
                result.reserve(tree.interface.outputs.size());
                for (const auto& item : tree.interface.outputs)
                    result.push_back(interface_decl(item));
                return result;
            }
            if (node.type_id == "lfs.group") {
                const auto graph = node.properties.find("tree");
                if (!resolver || graph == node.properties.end() || !graph->is_string())
                    return {};
                const NodeTree* referenced = resolver(graph->get_ref<const std::string&>());
                if (!referenced || referenced->tree_type != tree.tree_type)
                    return {};
                const auto& sockets = output ? referenced->interface.outputs : referenced->interface.inputs;
                std::vector<SocketDecl> result;
                result.reserve(sockets.size());
                for (const auto& item : sockets)
                    result.push_back(interface_decl(item));
                return result;
            }
            if (node.type_id == "lfs.reroute") {
                std::string type(ANY_SOCKET);
                if (visiting.insert(node.name).second) {
                    const auto incoming = std::ranges::find(tree.links, node.name, &Link::to_node);
                    if (incoming != tree.links.end() && incoming->to_socket == "Input") {
                        if (const auto* upstream = tree.find_node(incoming->from_node)) {
                            const auto outputs = effective_sockets_impl(tree, *upstream, true, resolver, visiting);
                            const auto found = std::ranges::find(outputs, incoming->from_socket,
                                                                 &SocketDecl::identifier);
                            if (found != outputs.end())
                                type = found->type;
                        }
                    }
                    visiting.erase(node.name);
                }
                SocketDecl result{output ? "Output" : "Input", "", type, {}};
                result.field = type != GEOMETRY_SOCKET && type != STRING_SOCKET;
                return {std::move(result)};
            }
            const auto type = tree.registry().find(node.type_id);
            return type ? (output ? type->outputs : type->inputs) : std::vector<SocketDecl>{};
        }

        std::string unique_name(const std::vector<Node>& nodes, std::string base) {
            if (base.empty())
                base = "Node";
            const auto exists = [&](std::string_view candidate) {
                return std::ranges::any_of(nodes, [&](const Node& node) {
                    return node.name == candidate;
                });
            };
            if (!exists(base))
                return base;
            for (int suffix = 2;; ++suffix) {
                std::string candidate = base + " " + std::to_string(suffix);
                if (!exists(candidate))
                    return candidate;
            }
        }

        void apply_defaults(Node& node, const NodeTypeInfo& type) {
            for (const auto& input : type.inputs)
                if (!node.input_values.contains(input.identifier))
                    node.input_values.emplace(input.identifier, input.default_value);
            if (!node.properties.is_object())
                node.properties = nlohmann::json::object();
            nlohmann::json properties = nlohmann::json::object();
            for (const auto& property : type.properties) {
                if (node.properties.contains(property.identifier))
                    properties[property.identifier] = node.properties[property.identifier];
                else
                    properties[property.identifier] = property.default_value;
            }
            node.properties = std::move(properties);
        }

    } // namespace

    std::vector<SocketDecl> effective_inputs(const NodeTree& tree, const Node& node,
                                             const TreeResolver& resolver) {
        std::unordered_set<std::string> visiting;
        return effective_sockets_impl(tree, node, false, resolver, visiting);
    }

    std::vector<SocketDecl> effective_outputs(const NodeTree& tree, const Node& node,
                                              const TreeResolver& resolver) {
        std::unordered_set<std::string> visiting;
        return effective_sockets_impl(tree, node, true, resolver, visiting);
    }

    bool group_reference_would_cycle(const NodeTree& owner, const std::string_view referenced_uuid,
                                     const TreeResolver& resolver, std::string* cycle) {
        if (!resolver || referenced_uuid.empty())
            return false;
        std::vector<const NodeTree*> path{&owner};
        std::unordered_set<std::string> active{owner.uuid};
        std::function<bool(const NodeTree*)> visit = [&](const NodeTree* tree) {
            if (!tree)
                return false;
            if (tree->uuid == owner.uuid) {
                path.push_back(tree);
                return true;
            }
            if (!active.insert(tree->uuid).second)
                return false;
            path.push_back(tree);
            for (const auto& node : tree->nodes) {
                if (node.type_id != "lfs.group")
                    continue;
                const auto property = node.properties.find("tree");
                if (property != node.properties.end() && property->is_string() &&
                    visit(resolver(property->get_ref<const std::string&>())))
                    return true;
            }
            path.pop_back();
            active.erase(tree->uuid);
            return false;
        };
        if (!visit(resolver(referenced_uuid)))
            return false;
        if (cycle) {
            cycle->clear();
            for (const auto* tree : path) {
                if (!cycle->empty())
                    *cycle += " → ";
                *cycle += tree->name;
            }
        }
        return true;
    }

    bool group_selection_would_cycle(const NodeTree& tree,
                                     const std::unordered_set<std::string>& selected) {
        constexpr std::string_view GROUP = "\x1fgroup";
        std::unordered_map<std::string, std::vector<std::string>> edges;
        for (const auto& link : tree.links) {
            const std::string from = selected.contains(link.from_node) ? std::string(GROUP) : link.from_node;
            const std::string to = selected.contains(link.to_node) ? std::string(GROUP) : link.to_node;
            if (from != to)
                edges[from].push_back(to);
        }
        std::unordered_map<std::string, int> state;
        std::function<bool(const std::string&)> visit = [&](const std::string& node) {
            if (state[node] == 1)
                return true;
            if (state[node] == 2)
                return false;
            state[node] = 1;
            if (const auto found = edges.find(node); found != edges.end())
                for (const auto& target : found->second)
                    if (visit(target))
                        return true;
            state[node] = 2;
            return false;
        };
        for (const auto& [node, _] : edges)
            if (visit(node))
                return true;
        return false;
    }

    void to_json(nlohmann::json& json, const Value& value) {
        json = nlohmann::json::object();
        if (std::holds_alternative<std::monostate>(value.data)) {
            json["type"] = "none";
        } else if (const auto* v = value.get_if<float>()) {
            json = {{"type", "float"}, {"value", *v}};
        } else if (const auto* v = value.get_if<std::int64_t>()) {
            json = {{"type", "int"}, {"value", *v}};
        } else if (const auto* v = value.get_if<bool>()) {
            json = {{"type", "bool"}, {"value", *v}};
        } else if (const auto* v = value.get_if<glm::vec3>()) {
            json = {{"type", "vec3"}, {"value", {*v == *v ? v->x : 0.0f, v->y, v->z}}};
        } else if (const auto* v = value.get_if<glm::vec4>()) {
            json = {{"type", "vec4"}, {"value", {v->x, v->y, v->z, v->w}}};
        } else if (const auto* v = value.get_if<std::string>()) {
            json = {{"type", "string"}, {"value", *v}};
        } else if (std::holds_alternative<Geometry>(value.data)) {
            json["type"] = "geometry";
        } else {
            json["type"] = "runtime";
        }
    }

    void from_json(const nlohmann::json& json, Value& value) {
        if (!json.is_object()) {
            value = {};
            return;
        }
        const std::string type = json.value("type", "none");
        const auto& payload = json.contains("value") ? json["value"] : nlohmann::json();
        if (type == "float" && payload.is_number())
            value = payload.get<float>();
        else if (type == "int" && payload.is_number_integer())
            value = payload.get<std::int64_t>();
        else if (type == "bool" && payload.is_boolean())
            value = payload.get<bool>();
        else if (type == "string" && payload.is_string())
            value = payload.get<std::string>();
        else if (type == "vec3" && payload.is_array() && payload.size() >= 3)
            value = glm::vec3(payload[0].get<float>(), payload[1].get<float>(), payload[2].get<float>());
        else if (type == "vec4" && payload.is_array() && payload.size() >= 4)
            value = glm::vec4(payload[0].get<float>(), payload[1].get<float>(), payload[2].get<float>(),
                              payload[3].get<float>());
        else if (type == "geometry")
            value = Geometry{};
        else
            value = {};
    }

    NodeTree::NodeTree(const NodeTypeRegistry& registry, std::string tree_name, std::string type)
        : NodeTree(registry, EmptyTag{}) {
        uuid = core::generate_uuid_v4().to_string();
        name = std::move(tree_name);
        tree_type = std::move(type);
        interface.inputs.push_back({"Geometry", "Geometry", std::string(GEOMETRY_SOCKET), Geometry{}});
        interface.outputs.push_back({"Geometry", "Geometry", std::string(GEOMETRY_SOCKET), Geometry{}});
        add_node("lfs.group_input", "Group Input").location = {-200.0f, 0.0f};
        add_node("lfs.group_output", "Group Output").location = {200.0f, 0.0f};
        add_link({"Group Input", "Geometry", "Group Output", "Geometry"});
    }

    NodeTree::NodeTree(const NodeTypeRegistry& registry, EmptyTag) : registry_(&registry) {}

    Node* NodeTree::find_node(std::string_view node_name) {
        const auto found = std::ranges::find(nodes, node_name, &Node::name);
        return found == nodes.end() ? nullptr : &*found;
    }

    const Node* NodeTree::find_node(std::string_view node_name) const {
        const auto found = std::ranges::find(nodes, node_name, &Node::name);
        return found == nodes.end() ? nullptr : &*found;
    }

    Node& NodeTree::add_node(std::string type_id, std::string node_name) {
        const auto type = registry_->find(type_id);
        if (node_name.empty())
            node_name = type ? type->label : type_id;
        Node node;
        node.name = unique_name(nodes, std::move(node_name));
        node.type_id = std::move(type_id);
        if (type) {
            node.version = type->version;
            apply_defaults(node, *type);
        }
        nodes.push_back(std::move(node));
        return nodes.back();
    }

    bool NodeTree::remove_node(std::string_view node_name) {
        const auto found = std::ranges::find(nodes, node_name, &Node::name);
        if (found == nodes.end() || found->type_id == "lfs.group_input" ||
            found->type_id == "lfs.group_output")
            return false;
        if (found->type_id == "lfs.frame")
            for (auto& node : nodes)
                if (node.ui.value("frame", "") == node_name)
                    node.ui.erase("frame");
        links.erase(std::remove_if(links.begin(), links.end(),
                                   [&](const Link& link) {
                                       return link.from_node == node_name || link.to_node == node_name;
                                   }),
                    links.end());
        nodes.erase(found);
        return true;
    }

    bool NodeTree::add_link(Link link, std::string* error, const TreeResolver& resolver) {
        const Node* from = find_node(link.from_node);
        const Node* to = find_node(link.to_node);
        if (!from || !to) {
            if (error)
                *error = "Link endpoint does not exist";
            return false;
        }
        const std::vector<Link> previous_links = links;
        const auto outputs = effective_outputs(*this, *from, resolver);
        const auto inputs = effective_inputs(*this, *to, resolver);
        const auto output = std::ranges::find(outputs, link.from_socket, &SocketDecl::identifier);
        const auto input = std::ranges::find(inputs, link.to_socket, &SocketDecl::identifier);
        if (output == outputs.end() || input == inputs.end()) {
            if (error)
                *error = "Link socket does not exist";
            return false;
        }
        if (!can_convert_socket(output->type, input->type)) {
            if (error)
                *error = "Link socket types are incompatible";
            return false;
        }
        if (!input->multi_input) {
                links.erase(std::remove_if(links.begin(), links.end(),
                                           [&](const Link& old) {
                                               return old.to_node == link.to_node &&
                                                      old.to_socket == link.to_socket;
                                           }),
                            links.end());
        }
        links.push_back(std::move(link));
        const auto issues = validate(resolver);
        const auto invalid = std::ranges::find_if(issues, [](const ValidationIssue& issue) {
            return issue.message == "Node graph contains a cycle" ||
                   issue.message == "Link socket types are incompatible";
        });
        if (invalid != issues.end()) {
            links = previous_links;
            if (error)
                *error = invalid->message;
            return false;
        }
        return true;
    }

    bool NodeTree::remove_link(const Link& link) {
        const auto found = std::ranges::find(links, link);
        if (found == links.end())
            return false;
        links.erase(found);
        return true;
    }

    std::vector<ValidationIssue> NodeTree::validate(const TreeResolver& resolver) const {
        std::vector<ValidationIssue> issues;
        const auto input_count = std::ranges::count(nodes, std::string("lfs.group_input"), &Node::type_id);
        const auto output_count = std::ranges::count(nodes, std::string("lfs.group_output"), &Node::type_id);
        if (tree_type == "lfs.geometry" && (input_count != 1 || output_count != 1))
            issues.push_back({{}, "Geometry tree must have exactly one Group Input and Group Output"});
        std::unordered_map<std::string, std::vector<std::string>> adjacency;
        std::unordered_set<std::string> linked_inputs;
        for (const auto& link : links) {
            const Node* from = find_node(link.from_node);
            const Node* to = find_node(link.to_node);
            if (!from || !to) {
                issues.push_back({to ? to->name : std::string{}, "Link endpoint does not exist"});
                continue;
            }
            if (from->type_id != "lfs.frame" && from->type_id != "lfs.note" &&
                to->type_id != "lfs.frame" && to->type_id != "lfs.note")
                adjacency[from->name].push_back(to->name);
            const auto outputs = effective_outputs(*this, *from, resolver);
            const auto inputs = effective_inputs(*this, *to, resolver);
            const auto output = std::ranges::find(outputs, link.from_socket, &SocketDecl::identifier);
            const auto input = std::ranges::find(inputs, link.to_socket, &SocketDecl::identifier);
            {
                if (output == outputs.end() || input == inputs.end())
                    issues.push_back({to->name, "Link socket does not exist"});
                else if (!can_convert_socket(output->type, input->type))
                    issues.push_back({to->name, "Link socket types are incompatible"});
                else if (!input->multi_input &&
                         !linked_inputs.insert(to->name + "\n" + input->identifier).second)
                    issues.push_back({to->name, "Input socket has more than one link"});
            }
        }
        std::unordered_map<std::string, int> state;
        std::function<bool(const std::string&)> visit = [&](const std::string& name) {
            if (state[name] == 1)
                return true;
            if (state[name] == 2)
                return false;
            state[name] = 1;
            for (const auto& next : adjacency[name])
                if (visit(next))
                    return true;
            state[name] = 2;
            return false;
        };
        for (const auto& node : nodes) {
            if (visit(node.name)) {
                issues.push_back({node.name, "Node graph contains a cycle"});
                break;
            }
        }
        for (const auto& node : nodes) {
            if (node.type_id != "lfs.group")
                continue;
            const auto graph = node.properties.find("tree");
            if (graph == node.properties.end() || !graph->is_string() || !resolver ||
                !resolver(graph->get_ref<const std::string&>())) {
                issues.push_back({node.name, "Missing graph"});
                continue;
            }
            std::string cycle;
            if (group_reference_would_cycle(*this, graph->get_ref<const std::string&>(), resolver,
                                            &cycle))
                issues.push_back({node.name, "Group cycle: " + cycle});
        }
        return issues;
    }

    nlohmann::json NodeTree::to_json() const {
        nlohmann::json result = {{"schema_version", 1},
                                 {"uuid", uuid},
                                 {"name", name},
                                 {"tree_type", tree_type},
                                 {"nodes", nlohmann::json::array()},
                                 {"links", nlohmann::json::array()}};
        for (const auto& node : nodes) {
            nlohmann::json item = node.preserved.is_object() ? node.preserved : nlohmann::json::object();
            item["name"] = node.name;
            item["type_id"] = node.type_id;
            item["location"] = {node.location[0], node.location[1]};
            item["muted"] = node.muted;
            item["version"] = node.version;
            item["input_values"] = node.input_values;
            item["properties"] = node.properties;
            item["ui"] = node.ui;
            result["nodes"].push_back(std::move(item));
        }
        for (const auto& link : links)
            result["links"].push_back({{"from_node", link.from_node},
                                       {"from_socket", link.from_socket},
                                       {"to_node", link.to_node},
                                       {"to_socket", link.to_socket}});
        auto encode_interface = [](const std::vector<InterfaceSocket>& sockets) {
            nlohmann::json result = nlohmann::json::array();
            for (const auto& socket_value : sockets) {
                nlohmann::json item = {{"identifier", socket_value.identifier},
                                       {"label", socket_value.label},
                                       {"type", socket_value.type},
                                       {"default", socket_value.default_value}};
                if (socket_value.min)
                    item["min"] = *socket_value.min;
                if (socket_value.max)
                    item["max"] = *socket_value.max;
                if (socket_value.step)
                    item["step"] = *socket_value.step;
                result.push_back(std::move(item));
            }
            return result;
        };
        result["interface"] = {{"inputs", encode_interface(interface.inputs)},
                               {"outputs", encode_interface(interface.outputs)}};
        return result;
    }

    NodeTree NodeTree::from_json(const nlohmann::json& json, const NodeTypeRegistry& registry) {
        NodeTree tree(registry, EmptyTag{});
        tree.uuid = json.value("uuid", core::generate_uuid_v4().to_string());
        tree.name = json.value("name", "Node Graph");
        tree.tree_type = json.value("tree_type", "lfs.geometry");
        if (const auto found = json.find("nodes"); found != json.end() && found->is_array()) {
            for (const auto& item : *found) {
                if (!item.is_object())
                    continue;
                Node node;
                node.name = unique_name(tree.nodes, item.value("name", "Node"));
                node.type_id = item.value("type_id", "");
                node.muted = item.value("muted", false);
                node.version = item.value("version", 1);
                if (item.contains("location") && item["location"].is_array() && item["location"].size() >= 2)
                    node.location = {item["location"][0].get<float>(), item["location"][1].get<float>()};
                if (item.contains("input_values") && item["input_values"].is_object())
                    node.input_values = item["input_values"].get<std::unordered_map<std::string, Value>>();
                if (item.contains("properties") && item["properties"].is_object())
                    node.properties = item["properties"];
                if (item.contains("ui") && item["ui"].is_object())
                    node.ui = item["ui"];
                if (const auto type = registry.find(node.type_id)) {
                    if (node.version < type->version && type->upgrade)
                        node.properties = type->upgrade(std::move(node.properties), node.version);
                    node.version = type->version;
                    if (node.type_id != "lfs.group" && node.type_id != "lfs.reroute") {
                        std::unordered_map<std::string, Value> inputs;
                        for (const auto& declaration : type->inputs) {
                            if (const auto value = node.input_values.find(declaration.identifier);
                                value != node.input_values.end())
                                inputs.emplace(declaration.identifier, value->second);
                            else
                                inputs.emplace(declaration.identifier, declaration.default_value);
                        }
                        node.input_values = std::move(inputs);
                    }
                    apply_defaults(node, *type);
                } else {
                    node.preserved = item;
                }
                tree.nodes.push_back(std::move(node));
            }
        }
        if (const auto found = json.find("links"); found != json.end() && found->is_array()) {
            for (const auto& item : *found) {
                if (!item.is_object())
                    continue;
                tree.links.push_back({item.value("from_node", ""), item.value("from_socket", ""),
                                      item.value("to_node", ""), item.value("to_socket", "")});
            }
        }
        const auto decode_interface = [](const nlohmann::json& items) {
            std::vector<InterfaceSocket> result;
            if (!items.is_array())
                return result;
            for (const auto& item : items) {
                if (!item.is_object())
                    continue;
                InterfaceSocket socket_value;
                socket_value.identifier = item.value("identifier", "");
                socket_value.label = item.value("label", socket_value.identifier);
                socket_value.type = item.value("type", std::string(FLOAT_SOCKET));
                if (item.contains("default"))
                    socket_value.default_value = item["default"].get<Value>();
                if (item.contains("min") && item["min"].is_number())
                    socket_value.min = item["min"].get<double>();
                if (item.contains("max") && item["max"].is_number())
                    socket_value.max = item["max"].get<double>();
                if (item.contains("step") && item["step"].is_number())
                    socket_value.step = item["step"].get<double>();
                result.push_back(std::move(socket_value));
            }
            return result;
        };
        if (json.contains("interface") && json["interface"].is_object()) {
            tree.interface.inputs =
                decode_interface(json["interface"].value("inputs", nlohmann::json::array()));
            tree.interface.outputs =
                decode_interface(json["interface"].value("outputs", nlohmann::json::array()));
        }
        if (tree.interface.inputs.empty())
            tree.interface.inputs.push_back(
                {"Geometry", "Geometry", std::string(GEOMETRY_SOCKET), Geometry{}});
        if (tree.interface.outputs.empty())
            tree.interface.outputs.push_back(
                {"Geometry", "Geometry", std::string(GEOMETRY_SOCKET), Geometry{}});
        return tree;
    }

    const Node& NodeTree::input_node() const {
        const auto found = std::ranges::find(nodes, std::string("lfs.group_input"), &Node::type_id);
        if (found == nodes.end())
            throw std::runtime_error("Geometry tree has no Group Input node");
        return *found;
    }

    const Node& NodeTree::output_node() const {
        const auto found = std::ranges::find(nodes, std::string("lfs.group_output"), &Node::type_id);
        if (found == nodes.end())
            throw std::runtime_error("Geometry tree has no Group Output node");
        return *found;
    }

} // namespace lfs::nodes
