/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_nodes.hpp"

#include "core/logger.hpp"
#include "core/nodes/nodes.hpp"
#include "core/path_utils.hpp"
#include "py_tensor.hpp"
#include "py_ui.hpp"
#include "py_viewer_dispatch.hpp"
#include "visualizer/nodes/modifier_manager.hpp"
#include "visualizer/nodes/node_animation.hpp"
#include "visualizer/scene/scene_manager.hpp"
#include "visualizer/sequencer/sequencer_controller.hpp"

#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace nb = nanobind;

namespace lfs::python {
    namespace {
        using namespace lfs::nodes;

        struct StandaloneLibrary {
            vis::SequencerController sequencer;
            NodeTypeRegistry registry;
            std::unordered_map<std::string, std::unique_ptr<NodeTree>> trees;
            std::unordered_map<std::string, EvalResult> evaluations;

            StandaloneLibrary() {
                register_builtin_nodes(registry);
            }
        };

        StandaloneLibrary& standalone() {
            static auto* value = new StandaloneLibrary;
            return *value;
        }

        std::string standalone_tree_name(std::string name, std::string_view except = {}) {
            if (name.empty())
                name = "Node Graph";
            std::unordered_set<std::string_view> names;
            for (const auto& [uuid, tree] : standalone().trees)
                if (uuid != except)
                    names.insert(tree->name);
            const auto base = name;
            for (int suffix = 2; names.contains(name); ++suffix)
                name = base + " " + std::to_string(suffix);
            return name;
        }

        std::string interface_name(const TreeInterface& sockets, std::string label) {
            if (label.empty())
                label = "Socket";
            const auto used = [&](const std::string_view candidate) {
                return std::ranges::any_of(sockets.inputs, [&](const auto& socket) { return socket.identifier == candidate; }) ||
                       std::ranges::any_of(sockets.outputs, [&](const auto& socket) { return socket.identifier == candidate; });
            };
            if (!used(label))
                return label;
            const auto base = label;
            for (int suffix = 2;; ++suffix) {
                label = base + " " + std::to_string(suffix);
                if (!used(label))
                    return label;
            }
        }

        vis::ModifierManager* live_manager() {
            auto* scene_manager = get_scene_manager();
            return scene_manager ? &scene_manager->modifierManager() : nullptr;
        }

        NodeTypeRegistry& active_registry() {
            if (auto* manager = live_manager())
                return manager->registry();
            return standalone().registry;
        }

        NodeTree* active_tree(const std::string& id) {
            if (auto* manager = live_manager())
                return manager->tree(id);
            const auto found = standalone().trees.find(id);
            return found == standalone().trees.end() ? nullptr : found->second.get();
        }

        TreeResolver active_tree_resolver() {
            return [](const std::string_view id) { return active_tree(std::string(id)); };
        }

        nb::object value_to_python(const Value& value);

        Value python_to_value(const nb::handle value) {
            if (value.is_none())
                return {};
            if (nb::isinstance<nb::bool_>(value))
                return nb::cast<bool>(value);
            if (nb::isinstance<nb::int_>(value))
                return nb::cast<std::int64_t>(value);
            if (nb::isinstance<nb::float_>(value))
                return nb::cast<float>(value);
            if (nb::isinstance<nb::str>(value))
                return nb::cast<std::string>(value);
            if (nb::isinstance<nb::sequence>(value)) {
                const auto sequence = nb::cast<nb::sequence>(value);
                if (nb::len(sequence) == 3) {
                    return glm::vec3(nb::cast<float>(sequence[0]),
                                     nb::cast<float>(sequence[1]),
                                     nb::cast<float>(sequence[2]));
                }
                if (nb::len(sequence) == 4) {
                    return glm::vec4(nb::cast<float>(sequence[0]),
                                     nb::cast<float>(sequence[1]),
                                     nb::cast<float>(sequence[2]),
                                     nb::cast<float>(sequence[3]));
                }
            }
            throw nb::type_error("Unsupported node value");
        }

        nlohmann::json python_to_json(const nb::handle value) {
            if (value.is_none())
                return nullptr;
            if (nb::isinstance<nb::bool_>(value))
                return nb::cast<bool>(value);
            if (nb::isinstance<nb::int_>(value))
                return nb::cast<std::int64_t>(value);
            if (nb::isinstance<nb::float_>(value))
                return nb::cast<double>(value);
            if (nb::isinstance<nb::str>(value))
                return nb::cast<std::string>(value);
            if (nb::isinstance<nb::dict>(value)) {
                nlohmann::json result = nlohmann::json::object();
                for (const auto [key, item] : nb::cast<nb::dict>(value))
                    result[nb::cast<std::string>(key)] = python_to_json(item);
                return result;
            }
            if (nb::isinstance<nb::sequence>(value)) {
                nlohmann::json result = nlohmann::json::array();
                for (const auto item : nb::cast<nb::sequence>(value))
                    result.push_back(python_to_json(item));
                return result;
            }
            throw nb::type_error("Unsupported node property value");
        }

        nb::object json_to_python(const nlohmann::json& value) {
            if (value.is_null())
                return nb::none();
            if (value.is_boolean())
                return nb::bool_(value.get<bool>());
            if (value.is_number_integer())
                return nb::int_(value.get<std::int64_t>());
            if (value.is_number())
                return nb::float_(value.get<double>());
            if (value.is_string())
                return nb::str(value.get<std::string>().c_str());
            if (value.is_array()) {
                nb::list result;
                for (const auto& item : value)
                    result.append(json_to_python(item));
                return std::move(result);
            }
            nb::dict result;
            for (const auto& [key, item] : value.items())
                result[nb::str(key.c_str())] = json_to_python(item);
            return std::move(result);
        }

        struct PySplats {
            SplatsComponent value;

            PySplats replace(const nb::kwargs& kwargs) const {
                PySplats result = *this;
                const auto set = [&](const char* key, core::Tensor& target) {
                    if (kwargs.contains(key))
                        target = nb::cast<PyTensor>(kwargs[key]).tensor();
                };
                set("means", result.value.means);
                set("sh0", result.value.sh0);
                set("shN", result.value.shN);
                set("scaling", result.value.scaling);
                set("rotation", result.value.rotation);
                set("opacity", result.value.opacity);
                return result;
            }
        };

        struct PyPoints {
            PointsComponent value;

            PyPoints replace(const nb::kwargs& kwargs) const {
                PyPoints result = *this;
                if (kwargs.contains("positions"))
                    result.value.positions = nb::cast<PyTensor>(kwargs["positions"]).tensor();
                if (kwargs.contains("colors"))
                    result.value.colors = nb::cast<PyTensor>(kwargs["colors"]).tensor();
                return result;
            }
        };

        struct PyMesh {
            MeshComponent value;

            PyMesh replace(const nb::kwargs& kwargs) const {
                if (!value.mesh)
                    return *this;
                const auto& source = *value.mesh;
                auto result = std::make_shared<core::MeshData>();
                result->vertices = kwargs.contains("vertices")
                                       ? nb::cast<PyTensor>(kwargs["vertices"]).tensor()
                                       : source.vertices;
                result->indices = kwargs.contains("indices")
                                      ? nb::cast<PyTensor>(kwargs["indices"]).tensor()
                                      : source.indices;
                result->normals = source.normals;
                result->tangents = source.tangents;
                result->texcoords = source.texcoords;
                result->colors = source.colors;
                result->materials = source.materials;
                result->submeshes = source.submeshes;
                result->texture_images = source.texture_images;
                return PyMesh{MeshComponent{std::move(result)}};
            }
        };

        struct PyGeometry {
            Geometry value;

            PyGeometry replace(const nb::kwargs& kwargs) const {
                PyGeometry result = *this;
                if (kwargs.contains("splats")) {
                    result.value.splats = kwargs["splats"].is_none()
                                              ? std::optional<SplatsComponent>{}
                                              : nb::cast<PySplats>(kwargs["splats"]).value;
                }
                if (kwargs.contains("points")) {
                    result.value.points = kwargs["points"].is_none()
                                              ? std::optional<PointsComponent>{}
                                              : nb::cast<PyPoints>(kwargs["points"]).value;
                }
                if (kwargs.contains("mesh")) {
                    result.value.mesh = kwargs["mesh"].is_none()
                                            ? std::optional<MeshComponent>{}
                                            : nb::cast<PyMesh>(kwargs["mesh"]).value;
                }
                return result;
            }
        };

        nb::object value_to_python(const Value& value) {
            if (const auto* geometry = value.get_if<Geometry>())
                return nb::cast(PyGeometry{*geometry});
            if (const auto* number = value.get_if<float>())
                return nb::float_(*number);
            if (const auto* integer = value.get_if<std::int64_t>())
                return nb::int_(*integer);
            if (const auto* boolean = value.get_if<bool>())
                return nb::bool_(*boolean);
            if (const auto* vector = value.get_if<glm::vec3>())
                return nb::make_tuple(vector->x, vector->y, vector->z);
            if (const auto* color = value.get_if<glm::vec4>())
                return nb::make_tuple(color->x, color->y, color->z, color->w);
            if (const auto* text = value.get_if<std::string>())
                return nb::str(text->c_str());
            return nb::none();
        }

        struct PyTree {
            std::string uuid;
        };

        struct PyInterface {
            std::string tree_uuid;
        };

        struct PyNode {
            std::string tree_uuid;
            std::string name;
        };

        NodeTree& require_tree(const PyTree& handle) {
            auto* tree = active_tree(handle.uuid);
            if (!tree)
                throw nb::key_error("Node tree no longer exists");
            return *tree;
        }

        Node& require_node(const PyNode& handle) {
            auto* tree = active_tree(handle.tree_uuid);
            auto* node = tree ? tree->find_node(handle.name) : nullptr;
            if (!node)
                throw nb::key_error("Node no longer exists");
            return *node;
        }

        void rewrite_group_references(nlohmann::json& tree_json,
                                      const std::unordered_map<std::string, std::string>& remap) {
            for (auto& node : tree_json["nodes"]) {
                if (node.value("type_id", "") != "lfs.group")
                    continue;
                auto& properties = node["properties"];
                const auto found = remap.find(properties.value("tree", ""));
                if (found != remap.end())
                    properties["tree"] = found->second;
            }
        }

        std::string standalone_copy(const PyTree& handle, const std::vector<std::string>& names) {
            const auto& tree = require_tree(handle);
            std::unordered_set<std::string> selected(names.begin(), names.end());
            selected.erase(tree.input_node().name);
            selected.erase(tree.output_node().name);
            const auto source = tree.to_json();
            nlohmann::json result{{"format", "lfs.node-clipboard"},
                                  {"version", 1},
                                  {"nodes", nlohmann::json::array()},
                                  {"links", nlohmann::json::array()},
                                  {"trees", nlohmann::json::object()}};
            for (const auto& node : source["nodes"])
                if (selected.contains(node.value("name", "")))
                    result["nodes"].push_back(node);
            for (const auto& link : source["links"])
                if (selected.contains(link.value("from_node", "")) &&
                    selected.contains(link.value("to_node", "")))
                    result["links"].push_back(link);
            std::unordered_set<std::string> visited;
            std::function<void(const nlohmann::json&)> include_groups =
                [&](const nlohmann::json& nodes) {
                    for (const auto& node : nodes) {
                        if (node.value("type_id", "") != "lfs.group")
                            continue;
                        const auto uuid = node.value("properties", nlohmann::json::object())
                                              .value("tree", "");
                        const auto* referenced = active_tree(uuid);
                        if (!referenced || !visited.insert(uuid).second)
                            continue;
                        const auto json = referenced->to_json();
                        result["trees"][uuid] = json;
                        include_groups(json["nodes"]);
                    }
                };
            include_groups(result["nodes"]);
            result["source_tree"] = tree.uuid;
            result["animation"] = vis::nodeAnimationJson(standalone().sequencer.timeline().animationClip());
            return result.dump();
        }

        std::vector<PyNode> standalone_paste(const PyTree& handle, const std::string& text,
                                             const std::optional<std::array<float, 2>> location) {
            auto& destination = require_tree(handle);
            nlohmann::json clipboard;
            try {
                clipboard = nlohmann::json::parse(text);
            } catch (const std::exception&) {
                throw nb::value_error("Clipboard does not contain LichtFeld nodes");
            }
            if (!clipboard.is_object() || clipboard["format"] != "lfs.node-clipboard" ||
                clipboard["version"] != 1 || !clipboard["nodes"].is_array() ||
                !clipboard["links"].is_array() || !clipboard["trees"].is_object())
                throw nb::value_error("Clipboard does not contain LichtFeld nodes");

            std::unordered_map<std::string, std::string> graph_remap;
            const auto imported = clipboard.value("trees", nlohmann::json::object());
            for (const auto& [uuid, json] : imported.items()) {
                const auto* existing = active_tree(uuid);
                graph_remap[uuid] = !existing || existing->to_json() == json
                                        ? uuid
                                        : core::generate_uuid_v4().to_string();
            }
            // A reused parent must also reference the reused version of every child.
            bool changed;
            do {
                changed = false;
                for (const auto& [uuid, json] : imported.items()) {
                    if (graph_remap[uuid] != uuid)
                        continue;
                    auto rewritten = json;
                    rewrite_group_references(rewritten, graph_remap);
                    if (rewritten != json) {
                        graph_remap[uuid] = core::generate_uuid_v4().to_string();
                        changed = true;
                    }
                }
            } while (changed);
            for (const auto& [uuid, source] : imported.items()) {
                if (active_tree(uuid) && graph_remap[uuid] == uuid)
                    continue;
                auto json = source;
                json["uuid"] = graph_remap[uuid];
                rewrite_group_references(json, graph_remap);
                auto tree = std::make_unique<NodeTree>(NodeTree::from_json(json, standalone().registry));
                tree->name = standalone_tree_name(tree->name);
                standalone().trees[tree->uuid] = std::move(tree);
            }

            nlohmann::json temporary{{"uuid", core::generate_uuid_v4().to_string()},
                                     {"name", "Clipboard"},
                                     {"tree_type", destination.tree_type},
                                     {"nodes", clipboard["nodes"]},
                                     {"links", nlohmann::json::array()},
                                     {"interface", {{"inputs", nlohmann::json::array()}, {"outputs", nlohmann::json::array()}}}};
            rewrite_group_references(temporary, graph_remap);
            auto decoded = NodeTree::from_json(temporary, standalone().registry);
            float centre_x = 0.0f;
            float centre_y = 0.0f;
            if (!decoded.nodes.empty()) {
                auto min_x = decoded.nodes.front().location[0];
                auto max_x = min_x;
                auto min_y = decoded.nodes.front().location[1];
                auto max_y = min_y;
                for (const auto& node : decoded.nodes) {
                    min_x = std::min(min_x, node.location[0]);
                    max_x = std::max(max_x, node.location[0]);
                    min_y = std::min(min_y, node.location[1]);
                    max_y = std::max(max_y, node.location[1]);
                }
                centre_x = (min_x + max_x) * 0.5f;
                centre_y = (min_y + max_y) * 0.5f;
            }
            const auto offset_x = location ? (*location)[0] - centre_x : 30.0f;
            const auto offset_y = location ? (*location)[1] - centre_y : 30.0f;
            std::unordered_map<std::string, std::string> remap;
            std::vector<PyNode> result;
            for (const auto& source : decoded.nodes) {
                if (source.type_id == "lfs.group_input" || source.type_id == "lfs.group_output")
                    continue;
                auto& copy = destination.add_node(source.type_id, source.name);
                const auto unique = copy.name;
                copy = source;
                copy.name = unique;
                copy.location = {source.location[0] + offset_x, source.location[1] + offset_y};
                remap[source.name] = unique;
                result.push_back({destination.uuid, unique});
            }
            for (const auto& source : decoded.nodes) {
                auto* copy = remap.contains(source.name) ? destination.find_node(remap[source.name]) : nullptr;
                const auto frame = source.ui.find("frame");
                if (!copy || frame == source.ui.end() || !frame->is_string())
                    continue;
                const auto mapped = remap.find(frame->get_ref<const std::string&>());
                if (mapped == remap.end())
                    copy->ui.erase("frame");
                else
                    copy->ui["frame"] = mapped->second;
            }
            for (const auto& item : clipboard.value("links", nlohmann::json::array())) {
                const auto from = remap.find(item.value("from_node", ""));
                const auto to = remap.find(item.value("to_node", ""));
                if (from != remap.end() && to != remap.end())
                    destination.add_link({from->second, item.value("from_socket", ""),
                                          to->second, item.value("to_socket", "")},
                                         nullptr, active_tree_resolver());
            }
            if (clipboard.contains("animation")) {
                vis::copyNodeAnimation(standalone().sequencer, clipboard["animation"], clipboard.value("source_tree", ""), destination.uuid, remap);
                for (const auto& [source, target] : graph_remap) {
                    std::unordered_map<std::string, std::string> names;
                    for (const auto& node : active_tree(target)->nodes)
                        names[node.name] = node.name;
                    vis::copyNodeAnimation(standalone().sequencer, clipboard["animation"], source, target, names);
                }
            }
            return result;
        }

        PyNode standalone_make_group(const PyTree& handle, const std::vector<PyNode>& handles,
                                     std::string name) {
            auto& outer = require_tree(handle);
            std::unordered_set<std::string> selected;
            for (const auto& item : handles)
                if (const auto* node = outer.find_node(item.name);
                    node && node->type_id != "lfs.group_input" && node->type_id != "lfs.group_output")
                    selected.insert(node->name);
            if (selected.empty())
                throw std::invalid_argument("Select at least one node");
            if (group_selection_would_cycle(outer, selected))
                throw std::invalid_argument(
                    "Selection cannot be grouped because dependencies leave and re-enter it");
            const auto original_nodes = outer.nodes;
            const auto original_links = outer.links;
            auto nested = std::make_unique<NodeTree>(standalone().registry, standalone_tree_name(std::move(name)));
            nested->group_interface.inputs.clear();
            nested->group_interface.outputs.clear();
            nested->links.clear();
            const auto input_name = nested->input_node().name;
            const auto output_name = nested->output_node().name;
            const TreeResolver resolver = [nested_ptr = nested.get()](const std::string_view uuid) {
                return uuid == nested_ptr->uuid ? nested_ptr : active_tree(std::string(uuid));
            };
            std::unordered_map<std::string, std::string> remap;
            float x = 0.0f;
            float y = 0.0f;
            for (const auto& source : original_nodes) {
                if (!selected.contains(source.name))
                    continue;
                auto& copy = nested->add_node(source.type_id, source.name);
                const auto unique = copy.name;
                copy = source;
                copy.name = unique;
                remap[source.name] = unique;
                x += source.location[0];
                y += source.location[1];
            }
            x /= static_cast<float>(selected.size());
            y /= static_cast<float>(selected.size());
            for (auto& node : nested->nodes)
                if (node.type_id != "lfs.group_input" && node.type_id != "lfs.group_output") {
                    node.location[0] -= x;
                    node.location[1] -= y;
                    const auto frame = node.ui.find("frame");
                    if (frame != node.ui.end() && frame->is_string()) {
                        const auto mapped = remap.find(frame->get_ref<const std::string&>());
                        if (mapped == remap.end())
                            node.ui.erase("frame");
                        else
                            node.ui["frame"] = mapped->second;
                    }
                }
            std::unordered_map<std::string, std::string> input_sockets;
            std::unordered_map<std::string, std::string> output_sockets;
            for (const auto& link : original_links) {
                const bool from_inside = selected.contains(link.from_node);
                const bool to_inside = selected.contains(link.to_node);
                if (from_inside && to_inside) {
                    nested->add_link({remap[link.from_node], link.from_socket, remap[link.to_node], link.to_socket}, nullptr, resolver);
                } else if (!from_inside && to_inside) {
                    const auto key = link.from_node + "\n" + link.from_socket;
                    if (!input_sockets.contains(key)) {
                        const auto* source = outer.find_node(link.from_node);
                        const auto outputs = source ? effective_outputs(outer, *source, resolver) : std::vector<SocketDecl>{};
                        const auto declaration = std::ranges::find(outputs, link.from_socket, &SocketDecl::identifier);
                        if (declaration == outputs.end())
                            continue;
                        Value initial = declaration->default_value;
                        if (const auto* target = outer.find_node(link.to_node))
                            if (const auto found = target->input_values.find(link.to_socket); found != target->input_values.end())
                                initial = found->second;
                        const auto identifier = interface_name(nested->group_interface,
                                                               declaration->label.empty() ? declaration->identifier : declaration->label);
                        nested->group_interface.inputs.push_back({identifier, declaration->label.empty() ? declaration->identifier : declaration->label,
                                                                  declaration->type, initial, declaration->min, declaration->max, declaration->step});
                        input_sockets[key] = identifier;
                    }
                    nested->add_link({input_name, input_sockets[key], remap[link.to_node], link.to_socket}, nullptr, resolver);
                } else if (from_inside && !to_inside) {
                    const auto key = link.from_node + "\n" + link.from_socket;
                    if (!output_sockets.contains(key)) {
                        const auto* source = outer.find_node(link.from_node);
                        const auto outputs = source ? effective_outputs(outer, *source, resolver) : std::vector<SocketDecl>{};
                        const auto declaration = std::ranges::find(outputs, link.from_socket, &SocketDecl::identifier);
                        if (declaration == outputs.end())
                            continue;
                        const auto identifier = interface_name(nested->group_interface,
                                                               declaration->label.empty() ? declaration->identifier : declaration->label);
                        nested->group_interface.outputs.push_back({identifier, declaration->label.empty() ? declaration->identifier : declaration->label,
                                                                   declaration->type, declaration->default_value, declaration->min, declaration->max, declaration->step});
                        output_sockets[key] = identifier;
                        nested->add_link({remap[link.from_node], link.from_socket, output_name, identifier}, nullptr, resolver);
                    }
                }
            }
            nested->find_node(input_name)->location = {-240.0f, 0.0f};
            nested->find_node(output_name)->location = {240.0f, 0.0f};
            const auto nested_uuid = nested->uuid;
            const auto nested_name = nested->name;
            standalone().trees[nested_uuid] = std::move(nested);
            for (const auto& item : selected)
                outer.remove_node(item);
            auto& group = outer.add_node("lfs.group", nested_name);
            group.location = {x, y};
            group.properties["tree"] = nested_uuid;
            for (const auto& socket : active_tree(nested_uuid)->group_interface.inputs)
                group.input_values[socket.identifier] = socket.default_value;
            outer.links.clear();
            std::unordered_set<std::string> linked_inputs;
            for (const auto& link : original_links) {
                const bool from_inside = selected.contains(link.from_node);
                const bool to_inside = selected.contains(link.to_node);
                if (!from_inside && !to_inside)
                    outer.add_link(link, nullptr, active_tree_resolver());
                else if (!from_inside && to_inside) {
                    const auto key = link.from_node + "\n" + link.from_socket;
                    if (linked_inputs.insert(key).second)
                        outer.add_link({link.from_node, link.from_socket, group.name, input_sockets.at(key)}, nullptr, active_tree_resolver());
                } else if (from_inside && !to_inside) {
                    const auto key = link.from_node + "\n" + link.from_socket;
                    outer.add_link({group.name, output_sockets.at(key), link.to_node, link.to_socket}, nullptr, active_tree_resolver());
                }
            }
            vis::copyNodeAnimation(standalone().sequencer, vis::nodeAnimationJson(standalone().sequencer.timeline().animationClip()),
                                   outer.uuid, nested_uuid, remap, true);
            return {outer.uuid, group.name};
        }

        void standalone_ungroup(const PyTree& handle, const PyNode& group_handle) {
            auto& outer = require_tree(handle);
            const auto* group = outer.find_node(group_handle.name);
            const auto* nested = group && group->type_id == "lfs.group"
                                     ? active_tree(group->properties.value("tree", std::string{}))
                                     : nullptr;
            if (!group || !nested)
                throw std::invalid_argument("Group node or graph does not exist");
            const auto links = outer.links;
            const auto values = group->input_values;
            const auto location = group->location;
            std::unordered_map<std::string, Link> incoming;
            for (const auto& link : links) {
                if (link.to_node == group->name)
                    incoming[link.to_socket] = link;
            }
            std::unordered_map<std::string, std::string> remap;
            for (const auto& source : nested->nodes) {
                if (source.type_id == "lfs.group_input" || source.type_id == "lfs.group_output")
                    continue;
                auto& copy = outer.add_node(source.type_id, source.name);
                const auto unique = copy.name;
                copy = source;
                copy.name = unique;
                copy.location = {source.location[0] + location[0], source.location[1] + location[1]};
                remap[source.name] = unique;
            }
            for (const auto& source : nested->nodes) {
                if (!remap.contains(source.name))
                    continue;
                auto* copy = outer.find_node(remap[source.name]);
                const auto frame = source.ui.find("frame");
                if (copy && frame != source.ui.end() && frame->is_string()) {
                    const auto mapped = remap.find(frame->get_ref<const std::string&>());
                    if (mapped == remap.end())
                        copy->ui.erase("frame");
                    else
                        copy->ui["frame"] = mapped->second;
                }
            }
            const auto input_node = nested->input_node().name;
            const auto output_node = nested->output_node().name;
            const auto nested_links = nested->links;
            outer.remove_node(group_handle.name);
            outer.links.clear();
            for (const auto& link : links) {
                if (link.to_node == group_handle.name)
                    continue;
                if (link.from_node != group_handle.name) {
                    outer.add_link(link, nullptr, active_tree_resolver());
                    continue;
                }
                for (const auto& inner : nested_links) {
                    if (inner.to_node != output_node || inner.to_socket != link.from_socket)
                        continue;
                    if (inner.from_node == input_node) {
                        if (incoming.contains(inner.from_socket)) {
                            const auto& source = incoming.at(inner.from_socket);
                            outer.add_link({source.from_node, source.from_socket, link.to_node,
                                            link.to_socket},
                                           nullptr, active_tree_resolver());
                        }
                    } else {
                        outer.add_link({remap[inner.from_node], inner.from_socket, link.to_node,
                                        link.to_socket},
                                       nullptr, active_tree_resolver());
                    }
                }
            }
            for (const auto& link : nested_links) {
                const bool from_input = link.from_node == input_node;
                const bool to_output = link.to_node == output_node;
                if (to_output)
                    continue;
                if (!from_input)
                    outer.add_link({remap[link.from_node], link.from_socket, remap[link.to_node], link.to_socket}, nullptr, active_tree_resolver());
                else {
                    if (incoming.contains(link.from_socket)) {
                        const auto& source = incoming.at(link.from_socket);
                        outer.add_link({source.from_node, source.from_socket, remap[link.to_node], link.to_socket}, nullptr, active_tree_resolver());
                    } else if (auto* target = outer.find_node(remap[link.to_node]); values.contains(link.from_socket))
                        target->input_values[link.to_socket] = values.at(link.from_socket);
                }
            }
            vis::copyNodeAnimation(standalone().sequencer,
                                   vis::nodeAnimationJson(standalone().sequencer.timeline().animationClip()), nested->uuid, outer.uuid, remap);
        }

        void record_tree_mutation(NodeTree& tree, nlohmann::json before,
                                  std::string merge_key = {}) {
            if (auto* manager = live_manager())
                manager->recordTreeEdit(tree.uuid, std::move(before), std::move(merge_key));
            else {
                if (tree.name != before.value("name", ""))
                    tree.name = standalone_tree_name(tree.name, tree.uuid);
                for (const auto& node : before.at("nodes"))
                    if (!tree.find_node(node.at("name").get<std::string>()))
                        vis::removeNodeAnimation(standalone().sequencer, tree.uuid, node.at("name").get<std::string>());
            }
        }

        nlohmann::json modifier_stack_json(const core::Uuid& node_uuid) {
            const auto* manager = live_manager();
            const auto* stack = manager ? manager->stack(node_uuid) : nullptr;
            return stack ? nlohmann::json(stack->modifiers) : nlohmann::json::array();
        }

        struct PyInputDecl {
            std::string identifier;
            std::string type;
            Value default_value;
            std::optional<double> min;
            std::optional<double> max;
            bool field = false;
            std::string description;
        };

        struct PyOutputDecl {
            std::string identifier;
            std::string type;
            std::string description;
        };

        struct PyPropertyDecl {
            std::string identifier;
            std::string type;
            nlohmann::json default_value;
            std::vector<std::string> items;
            std::string description;
        };

        struct PyNodeBase {};

        using SafeClass = std::shared_ptr<nb::object>;
        std::mutex& python_types_mutex() {
            static auto* value = new std::mutex;
            return *value;
        }

        std::unordered_map<std::string, SafeClass>& python_types() {
            static auto* value = new std::unordered_map<std::string, SafeClass>;
            return *value;
        }

        std::unordered_map<std::string, std::string>& python_type_modules() {
            static auto* value = new std::unordered_map<std::string, std::string>;
            return *value;
        }

        SafeClass safe_class(nb::object type) {
            return {new nb::object(std::move(type)), [](nb::object* object) {
                        if (!Py_IsInitialized())
                            return;
                        if (PyGILState_Check()) {
                            delete object;
                        } else {
                            nb::gil_scoped_acquire gil;
                            delete object;
                        }
                    }};
        }

        struct PyNodeContext {
            // Cleared when execute() returns, so a context kept by Python raises instead of reading freed memory.
            std::shared_ptr<NodeContext*> guard;
            std::shared_ptr<const std::unordered_map<std::string, std::string>> input_types;

            NodeContext& live() const {
                if (!guard || !*guard)
                    throw nb::value_error("This node context is only valid while execute() runs");
                return **guard;
            }

            nb::object input(const std::string& name) const {
                return value_to_python(live().input(name));
            }

            PyTensor field(const std::string& name, const PyGeometry& geometry) const {
                const auto context = lfs::nodes::field_context(geometry.value);
                if (!context)
                    throw nb::value_error("Geometry has no component for field evaluation");
                // Python components are temporary copies, so a memo shared with the evaluator would
                // rarely hit; keep memoization within this field evaluation.
                FieldMemo memo;
                std::string type(FLOAT_SOCKET);
                if (input_types)
                    if (const auto declared = input_types->find(name); declared != input_types->end())
                        type = declared->second;
                return PyTensor(live().field(name, type).evaluate(*context, memo));
            }

            nb::object prop(const std::string& name) const {
                const auto& properties = live().properties();
                const auto found = properties.find(name);
                return found == properties.end() ? nb::none()
                                                 : json_to_python(*found);
            }

            void output(const std::string& name, const nb::handle value) const {
                if (nb::isinstance<PyGeometry>(value))
                    live().set_output(name, nb::cast<PyGeometry>(value).value);
                else
                    live().set_output(name, python_to_value(value));
            }
        };

        std::string socket_type(std::string value) {
            if (value.starts_with("lfs."))
                return value;
            std::ranges::transform(value, value.begin(), [](const unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            return "lfs." + value;
        }

        NodeTypeInfo node_type_from_python(const nb::object& cls) {
            NodeTypeInfo info;
            info.id = nb::cast<std::string>(cls.attr("id"));
            info.label = nb::hasattr(cls, "label") ? nb::cast<std::string>(cls.attr("label")) : info.id;
            info.category = nb::hasattr(cls, "category")
                                ? nb::cast<std::string>(cls.attr("category"))
                                : "Utilities";
            info.description = nb::hasattr(cls, "description")
                                   ? nb::cast<std::string>(cls.attr("description"))
                                   : std::string{};
            info.help = nb::hasattr(cls, "help") ? nb::cast<std::string>(cls.attr("help")) : std::string{};
            std::unordered_set<std::string> declared;
            if (!nb::hasattr(cls, "execute") || !PyCallable_Check(cls.attr("execute").ptr()))
                throw nb::type_error("Node classes must implement execute(self, ctx)");
            const auto append = [&](const nb::handle value, std::string_view collection) {
                if (nb::isinstance<PyInputDecl>(value)) {
                    const auto& declaration = nb::cast<const PyInputDecl&>(value);
                    const auto& identifier = declaration.identifier;
                    if (collection != "inputs" || identifier.empty() || !declared.insert("input:" + identifier).second)
                        throw nb::value_error("inputs must contain Input declarations with unique, non-empty identifiers");
                    info.inputs.push_back({identifier, identifier, declaration.type,
                                           declaration.default_value, declaration.min,
                                           declaration.max, std::nullopt, declaration.field});
                    info.inputs.back().description = declaration.description;
                } else if (nb::isinstance<PyOutputDecl>(value)) {
                    const auto& declaration = nb::cast<const PyOutputDecl&>(value);
                    const auto& identifier = declaration.identifier;
                    if (collection != "outputs" || identifier.empty() || !declared.insert("output:" + identifier).second)
                        throw nb::value_error("outputs must contain Output declarations with unique, non-empty identifiers");
                    info.outputs.push_back({identifier, identifier, declaration.type});
                    info.outputs.back().description = declaration.description;
                } else if (nb::isinstance<PyPropertyDecl>(value)) {
                    const auto& declaration = nb::cast<const PyPropertyDecl&>(value);
                    const auto& identifier = declaration.identifier;
                    if (collection != "properties" || identifier.empty() || !declared.insert("property:" + identifier).second)
                        throw nb::value_error("properties must contain Property declarations with unique, non-empty identifiers");
                    PropertyKind kind = PropertyKind::String;
                    if (declaration.type == "bool")
                        kind = PropertyKind::Bool;
                    else if (declaration.type == "int")
                        kind = PropertyKind::Int;
                    else if (declaration.type == "float")
                        kind = PropertyKind::Float;
                    else if (declaration.type == "enum")
                        kind = PropertyKind::Enum;
                    info.properties.push_back({identifier, identifier, kind,
                                               declaration.default_value, declaration.items});
                    info.properties.back().description = declaration.description;
                } else {
                    throw nb::type_error("Node declarations must be Input, Output or Property objects in their corresponding lists");
                }
            };
            for (const auto* collection : {"inputs", "outputs", "properties"}) {
                if (!nb::hasattr(cls, collection))
                    continue;
                if (!nb::isinstance<nb::list>(cls.attr(collection)))
                    throw nb::type_error("Node inputs, outputs and properties must be lists");
                for (const auto value : nb::cast<nb::list>(cls.attr(collection)))
                    append(value, collection);
            }
            for (const auto base : nb::cast<nb::tuple>(cls.attr("__mro__"))) {
                const auto attributes = nb::cast<nb::dict>(nb::module_::import_("builtins").attr("dict")(base.attr("__dict__")));
                for (const auto [key, value] : attributes) {
                    if (nb::isinstance<PyInputDecl>(value) || nb::isinstance<PyOutputDecl>(value) || nb::isinstance<PyPropertyDecl>(value))
                        throw nb::type_error("Attribute-style node declarations are unsupported; use inputs, outputs and properties lists");
                }
            }
            const std::string type_id = info.id;
            if (nb::cast<std::string>(cls.attr("__module__")) == "lfs_plugins.node_posterize" && info.id == "lfs.posterize")
                set_builtin_node_text(info);
            auto input_types = std::make_shared<std::unordered_map<std::string, std::string>>();
            for (const auto& input : info.inputs)
                input_types->emplace(input.identifier, input.type);
            info.evaluate = [type_id, input_types = std::shared_ptr<const std::unordered_map<std::string, std::string>>(std::move(input_types))](NodeContext& context) {
                SafeClass type;
                {
                    std::lock_guard lock(python_types_mutex());
                    const auto found = python_types().find(type_id);
                    if (found == python_types().end())
                        throw NodeError("Python node type is not registered");
                    type = found->second;
                }
                nb::gil_scoped_acquire gil;
                const auto guard = std::make_shared<NodeContext*>(&context);
                struct Expire {
                    const std::shared_ptr<NodeContext*>& guard;
                    bool outer;
                    ~Expire() {
                        *guard = nullptr;
                        g_in_node_execute = outer;
                    }
                } expire{guard, g_in_node_execute};
                g_in_node_execute = true;
                try {
                    auto instance = (**type)();
                    const auto callback = instance.attr("execute");
                    const PyNodeContext python_context{guard, input_types};
                    auto result = callback(python_context);
                    if (!result.is_none()) {
                        if (nb::isinstance<PyGeometry>(result))
                            context.set_output("Geometry", nb::cast<PyGeometry>(result).value);
                        else if (nb::isinstance<nb::dict>(result))
                            for (const auto [key, value] : nb::cast<nb::dict>(result))
                                python_context.output(nb::cast<std::string>(key), value);
                    }
                } catch (const nb::python_error& error) {
                    LOG_WARN("Python node {} failed:\n{}", type_id, error.what());
                    const auto name = nb::cast<std::string>(error.type().attr("__name__"));
                    throw NodeError("Python node '" + type_id + "' failed (" + name + "). See the log for details.");
                }
            };
            return info;
        }

        void install_python_type(const NodeTypeInfo& info) {
            standalone().registry.unregister_type(info.id);
            standalone().registry.register_type(info);
            invoke_on_viewer([info] {
                if (auto* manager = live_manager()) {
                    manager->registry().unregister_type(info.id);
                    manager->registry().register_type(info);
                    manager->markDirty();
                }
            });
        }

        bool unregister_python_type(const std::string& id) {
            standalone().registry.unregister_type(id);
            invoke_on_viewer([id] {
                if (auto* manager = live_manager()) {
                    manager->registry().unregister_type(id);
                    manager->markDirty();
                }
            });
            std::lock_guard lock(python_types_mutex());
            python_type_modules().erase(id);
            return python_types().erase(id) != 0;
        }

        size_t unregister_python_types_for_module(const std::string& prefix) {
            if (prefix.empty())
                return 0;
            std::vector<std::string> ids;
            {
                std::lock_guard lock(python_types_mutex());
                for (const auto& [id, module_name] : python_type_modules()) {
                    if (module_name == prefix || module_name.starts_with(prefix + "."))
                        ids.push_back(id);
                }
            }
            for (const auto& id : ids)
                unregister_python_type(id);
            return ids.size();
        }

        nb::dict evaluation_dict(const EvalResult& result) {
            nb::dict errors;
            for (const auto& [node, message] : result.errors)
                errors[nb::str(node.c_str())] = message;
            double total = 0.0;
            for (const auto& [_, time] : result.time_ms)
                total += time;
            nb::dict nodes;
            for (const auto& [node, time] : result.time_ms)
                nodes[nb::str(node.c_str())] = time;
            nb::dict value;
            value["ok"] = result.ok;
            value["errors"] = std::move(errors);
            value["time_ms"] = total;
            value["node_time_ms"] = std::move(nodes);
            return value;
        }

        std::optional<core::Uuid> scene_node_uuid(const std::string& name) {
            auto* scene_manager = get_scene_manager();
            if (!scene_manager)
                return std::nullopt;
            const auto* node = scene_manager->getScene().getNode(name);
            return node ? std::optional{node->uuid} : std::nullopt;
        }

        struct PyModifier {
            core::Uuid node_uuid;
            std::string uuid;
        };

        vis::Modifier& require_modifier(const PyModifier& handle) {
            auto* manager = live_manager();
            auto* stack = manager ? manager->stack(handle.node_uuid) : nullptr;
            if (stack) {
                const auto found = std::ranges::find(stack->modifiers, handle.uuid,
                                                     &vis::Modifier::uuid);
                if (found != stack->modifiers.end())
                    return *found;
            }
            throw nb::key_error("Modifier no longer exists");
        }
    } // namespace

    void register_nodes(nb::module_& module) {
        nb::class_<PySplats>(module, "Splats")
            .def("__init__", [](PySplats* self, PyTensor means, PyTensor sh0, PyTensor shN, PyTensor scaling, PyTensor rotation, PyTensor opacity, int sh_degree, float scene_scale) { new (self) PySplats{{means.tensor(), sh0.tensor(), shN.tensor(), scaling.tensor(), rotation.tensor(), opacity.tensor(), sh_degree, scene_scale, {}}}; }, nb::arg("means"), nb::arg("sh0"), nb::arg("shN"), nb::arg("scaling"), nb::arg("rotation"), nb::arg("opacity"), nb::arg("sh_degree") = 0, nb::arg("scene_scale") = 1.0f)
            .def_prop_ro("means", [](const PySplats& value) { return PyTensor(value.value.means); })
            .def_prop_ro("sh0", [](const PySplats& value) { return PyTensor(value.value.sh0); })
            .def_prop_ro("shN", [](const PySplats& value) { return PyTensor(value.value.shN); })
            .def_prop_ro("scaling", [](const PySplats& value) { return PyTensor(value.value.scaling); })
            .def_prop_ro("rotation", [](const PySplats& value) { return PyTensor(value.value.rotation); })
            .def_prop_ro("opacity", [](const PySplats& value) { return PyTensor(value.value.opacity); })
            .def("replace", &PySplats::replace);

        nb::class_<PyPoints>(module, "Points")
            .def("__init__", [](PyPoints* self, PyTensor positions, PyTensor colors) { new (self) PyPoints{{positions.tensor(), colors.tensor(), {}}}; }, nb::arg("positions"), nb::arg("colors"))
            .def_prop_ro("positions", [](const PyPoints& value) { return PyTensor(value.value.positions); })
            .def_prop_ro("colors", [](const PyPoints& value) { return PyTensor(value.value.colors); })
            .def("replace", &PyPoints::replace);

        nb::class_<PyMesh>(module, "Mesh")
            .def_prop_ro("vertices", [](const PyMesh& value) {
                return PyTensor(value.value.mesh ? value.value.mesh->vertices : core::Tensor{});
            })
            .def_prop_ro("indices", [](const PyMesh& value) {
                return PyTensor(value.value.mesh ? value.value.mesh->indices : core::Tensor{});
            })
            .def("replace", &PyMesh::replace);

        nb::class_<PyGeometry>(module, "Geometry")
            .def(nb::init<>())
            .def("__init__", [](PyGeometry* self, std::optional<PySplats> splats, std::optional<PyPoints> points, std::optional<PyMesh> mesh) { new (self) PyGeometry{{splats ? std::optional{splats->value} : std::nullopt,
                                                                                                                                                                       points ? std::optional{points->value} : std::nullopt,
                                                                                                                                                                       mesh ? std::optional{mesh->value} : std::nullopt}}; }, nb::arg("splats") = nb::none(), nb::arg("points") = nb::none(), nb::arg("mesh") = nb::none())
            .def_prop_ro("splats", [](const PyGeometry& value) -> std::optional<PySplats> { return value.value.splats ? std::optional{PySplats{*value.value.splats}} : std::nullopt; })
            .def_prop_ro("points", [](const PyGeometry& value) -> std::optional<PyPoints> { return value.value.points ? std::optional{PyPoints{*value.value.points}} : std::nullopt; })
            .def_prop_ro("mesh", [](const PyGeometry& value) -> std::optional<PyMesh> { return value.value.mesh ? std::optional{PyMesh{*value.value.mesh}} : std::nullopt; })
            .def_prop_ro("empty", [](const PyGeometry& value) { return value.value.empty(); })
            .def("replace", &PyGeometry::replace);

        nb::class_<PyInputDecl>(module, "Input")
            .def("__init__", [](PyInputDecl* self, std::string identifier, std::string type, nb::object default_value, std::optional<double> min, std::optional<double> max, bool field, std::string description) { new (self) PyInputDecl{std::move(identifier), socket_type(std::move(type)), python_to_value(default_value), min, max, field, std::move(description)}; }, nb::arg("identifier"), nb::arg("type"), nb::arg("default") = nb::none(), nb::arg("min") = nb::none(), nb::arg("max") = nb::none(), nb::arg("field") = false, nb::arg("description") = "");
        nb::class_<PyOutputDecl>(module, "Output")
            .def("__init__", [](PyOutputDecl* self, std::string identifier, std::string type, std::string description) { new (self) PyOutputDecl{std::move(identifier), socket_type(std::move(type)), std::move(description)}; }, nb::arg("identifier"), nb::arg("type"), nb::arg("description") = "");
        nb::class_<PyPropertyDecl>(module, "Property")
            .def("__init__", [](PyPropertyDecl* self, std::string identifier, std::string type, nb::object default_value, std::vector<std::string> items, std::string description) { new (self) PyPropertyDecl{std::move(identifier), std::move(type), python_to_json(default_value), std::move(items), std::move(description)}; }, nb::arg("identifier"), nb::arg("type"), nb::arg("default") = nb::none(), nb::arg("items") = std::vector<std::string>{}, nb::arg("description") = "");
        nb::class_<PyNodeBase>(module, "Node", nb::dynamic_attr()).def(nb::init<>());
        nb::class_<PyNodeContext>(module, "NodeContext")
            .def("input", &PyNodeContext::input)
            .def("field", &PyNodeContext::field)
            .def("field", [](const PyNodeContext& context, const std::string& name,
                             const PySplats& component) {
                return context.field(name, PyGeometry{Geometry{.splats = component.value}});
            })
            .def("field", [](const PyNodeContext& context, const std::string& name, const PyPoints& component) {
                return context.field(name, PyGeometry{Geometry{.points = component.value}});
            })
            .def("field", [](const PyNodeContext& context, const std::string& name, const PyMesh& component) {
                return context.field(name, PyGeometry{Geometry{.mesh = component.value}});
            })
            .def("prop", &PyNodeContext::prop)
            .def("output", &PyNodeContext::output);

        nb::class_<PyNode>(module, "NodeHandle")
            .def_prop_rw("name", [](const PyNode& value) { return value.name; }, [](PyNode& value, const std::string& name) { invoke_on_viewer([&value, name] {
                                                                                                                                  if (auto* manager = live_manager()) {
                                                                                                                                      const auto result = manager->renameNode(value.tree_uuid, value.name, name);
                                                                                                                                      if (!result)
                                                                                                                                          throw std::invalid_argument(result.error().message);
                                                                                                                                  } else {
                                                                                                                                      auto& graph = *active_tree(value.tree_uuid);
                                                                                                                                      const auto animation = vis::nodeAnimationJson(standalone().sequencer.timeline().animationClip());
                                                                                                                                      if (!graph.rename_node(value.name, name))
                                                                                                                                          throw std::invalid_argument("Node name must be unique and non-empty");
                                                                                                                                      vis::copyNodeAnimation(standalone().sequencer, animation, graph.uuid, graph.uuid, {{value.name, name}}, true);
                                                                                                                                  }
                                                                                                                                  value.name = name;
                                                                                                                              }); })
            .def("keyframe_insert", [](const PyNode& node, const std::string& input, std::optional<float> time, nb::object value, int easing) {
                const auto converted = value.is_none() ? std::optional<Value>{} : std::optional{python_to_value(value)};
                invoke_on_viewer([node, input, time, converted, easing] {
                    if (auto* manager = live_manager()) {
                        const auto result = manager->keyframeSet(node.tree_uuid, node.name, input, time, converted, easing);
                        if (!result) throw std::invalid_argument(result.error().message);
                    } else {
                        const auto result = vis::setNodeKeyframe(*active_tree(node.tree_uuid), node.name, input,
                            standalone().sequencer, time, converted, static_cast<sequencer::EasingType>(easing), active_tree_resolver());
                        if (!result) throw std::invalid_argument(result.error().message);
                    }
                }); }, nb::arg("input"), nb::arg("time") = nb::none(), nb::arg("value") = nb::none(), nb::arg("easing") = 0)
            .def("keyframe_remove", [](const PyNode& node, const std::string& input, std::optional<float> time) { return invoke_on_viewer([node, input, time] {
                                                                                                                      if (auto* manager = live_manager())
                                                                                                                          return bool(manager->keyframeRemove(node.tree_uuid, node.name, input, time));
                                                                                                                      return vis::removeNodeKeyframe(standalone().sequencer, vis::nodeInputTrackPath(node.tree_uuid, node.name, input), time);
                                                                                                                  },
                                                                                                                                          false); }, nb::arg("input"), nb::arg("time") = nb::none())
            .def_prop_ro("type_id", [](const PyNode& value) { return require_node(value).type_id; })
            .def_prop_rw("location", [](const PyNode& value) { return require_node(value).location; }, [](const PyNode& value, std::array<float, 2> location) { invoke_on_viewer([value, location] {
                                                                                                                                                                    auto& tree = *active_tree(value.tree_uuid);
                                                                                                                                                                    auto before = tree.to_json();
                                                                                                                                                                    require_node(value).location = location;
                                                                                                                                                                    record_tree_mutation(tree, std::move(before),
                                                                                                                                                                                         value.name + ":location");
                                                                                                                                                                }); })
            .def_prop_rw("muted", [](const PyNode& value) { return require_node(value).muted; }, [](const PyNode& value, bool muted) { invoke_on_viewer([value, muted] {
                                                                                                                                           auto& tree = *active_tree(value.tree_uuid);
                                                                                                                                           auto before = tree.to_json();
                                                                                                                                           require_node(value).muted = muted;
                                                                                                                                           record_tree_mutation(tree, std::move(before),
                                                                                                                                                                value.name + ":muted");
                                                                                                                                       }); })
            .def_prop_rw("graph", [](const PyNode& value) -> std::optional<PyTree> {
                const auto& node = require_node(value);
                if (node.type_id != "lfs.group")
                    return std::nullopt;
                const auto found = node.properties.find("tree");
                return found != node.properties.end() && found->is_string()
                           ? std::optional{PyTree{found->get<std::string>()}}
                           : std::nullopt; }, [](const PyNode& value, const PyTree& graph) { invoke_on_viewer([value, graph] {
                                                                                                                                           if (auto* manager = live_manager()) {
                                                                                                                                               const auto result = manager->setGroupGraph(value.tree_uuid, value.name, graph.uuid);
                                                                                                                                               if (!result)
                                                                                                                                                   throw std::invalid_argument(result.error().message);
                                                                                                                                               return;
                                                                                                                                           }
                                                                                                                                           auto& owner = *active_tree(value.tree_uuid);
                                                                                                                                           auto* referenced = active_tree(graph.uuid);
                                                                                                                                           std::string cycle;
                                                                                                                                           if (!referenced || group_reference_would_cycle(owner, graph.uuid,
                                                                                                                                                                                          active_tree_resolver(), &cycle))
                                                                                                                                               throw std::invalid_argument(referenced ? "Group cycle: " + cycle
                                                                                                                                                                                      : "Missing graph");
                                                                                                                                           auto& node = require_node(value);
                                                                                                                                           node.properties["tree"] = graph.uuid;
                                                                                                                                           node.input_values.clear();
                                                                                                                                           for (const auto& socket : referenced->group_interface.inputs)
                                                                                                                                               node.input_values[socket.identifier] = socket.default_value;
                                                                                                                                           const auto inputs = effective_inputs(owner, node, active_tree_resolver());
                                                                                                                                           const auto outputs = effective_outputs(owner, node, active_tree_resolver());
                                                                                                                                           std::erase_if(owner.links, [&](const auto& link) {
                                                                                                                                               if (link.from_node == node.name)
                                                                                                                                                   return std::ranges::find(outputs, link.from_socket, &SocketDecl::identifier) == outputs.end();
                                                                                                                                               if (link.to_node == node.name)
                                                                                                                                                   return std::ranges::find(inputs, link.to_socket, &SocketDecl::identifier) == inputs.end();
                                                                                                                                               return false;
                                                                                                                                           });
                                                                                                                                       }); })
            .def("set_input", [](const PyNode& node, const std::string& name, nb::object value) {
                auto converted = python_to_value(value);
                invoke_on_viewer([node, name, converted = std::move(converted)]() mutable {
                    if (auto* manager = live_manager()) {
                        const auto result = manager->setNodeInput(node.tree_uuid, node.name, name, std::move(converted));
                        if (!result)
                            throw std::invalid_argument(result.error().message);
                        return;
                    }
                    auto& tree = *active_tree(node.tree_uuid);
                    auto before = tree.to_json();
                    require_node(node).input_values[name] = std::move(converted);
                    record_tree_mutation(tree, std::move(before), node.name + ":input:" + name);
                }); })
            .def("get_input", [](const PyNode& node, const std::string& name) {
                const auto& values = require_node(node).input_values;
                const auto found = values.find(name);
                return found == values.end() ? nb::none() : value_to_python(found->second); })
            .def("set_property", [](const PyNode& node, const std::string& name, nb::object value) {
                auto converted = python_to_json(value);
                invoke_on_viewer([node, name, converted = std::move(converted)]() mutable {
                    auto& tree = *active_tree(node.tree_uuid);
                    auto before = tree.to_json();
                    require_node(node).properties[name] = std::move(converted);
                    record_tree_mutation(tree, std::move(before),
                                         node.name + ":property:" + name);
                }); })
            .def("get_property", [](const PyNode& node, const std::string& name) {
                const auto& properties = require_node(node).properties;
                const auto found = properties.find(name);
                return found == properties.end() ? nb::none() : json_to_python(*found); })
            .def_prop_ro("error", [](const PyNode& node) -> std::optional<std::string> {
                const auto found = standalone().evaluations.find(node.tree_uuid);
                if (found == standalone().evaluations.end())
                    return std::nullopt;
                const auto error = found->second.errors.find(node.name);
                return error == found->second.errors.end() ? std::nullopt
                                                           : std::optional{error->second}; })
            .def_prop_ro("eval_time_ms", [](const PyNode& node) {
                const auto found = standalone().evaluations.find(node.tree_uuid);
                if (found == standalone().evaluations.end())
                    return 0.0;
                const auto time = found->second.time_ms.find(node.name);
                return time == found->second.time_ms.end() ? 0.0 : time->second; });

        nb::class_<PyInterface>(module, "TreeInterface")
            .def("add_input", [](const PyInterface& value, std::string type, std::string label, nb::object initial, std::optional<double> min, std::optional<double> max, std::optional<double> step) {
                auto converted = python_to_value(initial);
                return invoke_on_viewer([value, type = socket_type(std::move(type)), label = std::move(label), converted = std::move(converted), min, max, step]() mutable {
                    if (auto* manager = live_manager()) {
                        auto result = manager->interfaceAdd(value.tree_uuid, false, std::move(type), std::move(label), std::move(converted), min, max, step);
                        if (!result)
                            throw std::invalid_argument(result.error().message);
                        return *result;
                    }
                    auto& tree = *active_tree(value.tree_uuid);
                    const auto identifier = interface_name(tree.group_interface, label);
                    tree.group_interface.inputs.push_back({identifier, label, std::move(type), std::move(converted), min, max, step});
                    for (auto& [_, other] : standalone().trees)
                        for (auto& node : other->nodes)
                            if (node.type_id == "lfs.group" && node.properties.value("tree", "") == tree.uuid)
                                node.input_values[identifier] = tree.group_interface.inputs.back().default_value;
                    return identifier;
                }, std::string{}); }, nb::arg("type"), nb::arg("label"), nb::arg("default") = nb::none(), nb::arg("min") = nb::none(), nb::arg("max") = nb::none(), nb::arg("step") = nb::none())
            .def("add_output", [](const PyInterface& value, std::string type, std::string label, nb::object initial, std::optional<double> min, std::optional<double> max, std::optional<double> step) {
                auto converted = python_to_value(initial);
                return invoke_on_viewer([value, type = socket_type(std::move(type)), label = std::move(label), converted = std::move(converted), min, max, step]() mutable {
                    if (auto* manager = live_manager()) {
                        auto result = manager->interfaceAdd(value.tree_uuid, true, std::move(type), std::move(label), std::move(converted), min, max, step);
                        if (!result)
                            throw std::invalid_argument(result.error().message);
                        return *result;
                    }
                    auto& tree = *active_tree(value.tree_uuid);
                    const auto identifier = interface_name(tree.group_interface, label);
                    tree.group_interface.outputs.push_back({identifier, label, std::move(type), std::move(converted), min, max, step});
                    return identifier;
                }, std::string{}); }, nb::arg("type"), nb::arg("label"), nb::arg("default") = nb::none(), nb::arg("min") = nb::none(), nb::arg("max") = nb::none(), nb::arg("step") = nb::none())
            .def("remove", [](const PyInterface& value, const std::string& side, const std::string& identifier) { invoke_on_viewer([value, side, identifier] {
                                                                                                                      if (auto* manager = live_manager()) {
                                                                                                                          const auto result = manager->interfaceRemove(value.tree_uuid, side == "output", identifier);
                                                                                                                          if (!result)
                                                                                                                              throw std::invalid_argument(result.error().message);
                                                                                                                          return;
                                                                                                                      }
                                                                                                                      auto& tree = *active_tree(value.tree_uuid);
                                                                                                                      const bool output = side == "output";
                                                                                                                      auto& sockets = output ? tree.group_interface.outputs : tree.group_interface.inputs;
                                                                                                                      if (std::erase_if(sockets, [&](const auto& socket) { return socket.identifier == identifier; }) == 0)
                                                                                                                          throw std::invalid_argument("Interface socket does not exist");
                                                                                                                      const auto endpoint = output ? tree.output_node().name : tree.input_node().name;
                                                                                                                      std::erase_if(tree.links, [&](const auto& link) {
                                                                                                                          return output ? link.to_node == endpoint && link.to_socket == identifier
                                                                                                                                        : link.from_node == endpoint && link.from_socket == identifier;
                                                                                                                      });
                                                                                                                      for (auto& [_, other] : standalone().trees) {
                                                                                                                          for (auto& node : other->nodes)
                                                                                                                              if (!output && node.type_id == "lfs.group" && node.properties.value("tree", "") == tree.uuid)
                                                                                                                                  node.input_values.erase(identifier);
                                                                                                                          std::erase_if(other->links, [&](const auto& link) {
                                                                                                                              const auto* from = other->find_node(link.from_node);
                                                                                                                              const auto* to = other->find_node(link.to_node);
                                                                                                                              return output ? from && from->type_id == "lfs.group" && from->properties.value("tree", "") == tree.uuid && link.from_socket == identifier
                                                                                                                                            : to && to->type_id == "lfs.group" && to->properties.value("tree", "") == tree.uuid && link.to_socket == identifier;
                                                                                                                          });
                                                                                                                      }
                                                                                                                  }); })
            .def("move", [](const PyInterface& value, const std::string& side, const std::string& identifier, std::size_t index) { invoke_on_viewer([value, side, identifier, index] {
                                                                                                                                       if (auto* manager = live_manager()) {
                                                                                                                                           const auto result = manager->interfaceMove(value.tree_uuid, side == "output", identifier, index);
                                                                                                                                           if (!result)
                                                                                                                                               throw std::invalid_argument(result.error().message);
                                                                                                                                           return;
                                                                                                                                       }
                                                                                                                                       auto& tree = *active_tree(value.tree_uuid);
                                                                                                                                       auto& sockets = side == "output" ? tree.group_interface.outputs : tree.group_interface.inputs;
                                                                                                                                       const auto found = std::ranges::find(sockets, identifier, &InterfaceSocket::identifier);
                                                                                                                                       if (found == sockets.end())
                                                                                                                                           throw std::invalid_argument("Interface socket does not exist");
                                                                                                                                       auto socket = std::move(*found);
                                                                                                                                       sockets.erase(found);
                                                                                                                                       sockets.insert(sockets.begin() + std::min(index, sockets.size()), std::move(socket));
                                                                                                                                   }); });

        nb::class_<PyTree>(module, "NodeTree")
            .def_prop_ro("uuid", [](const PyTree& value) { return value.uuid; })
            .def_prop_rw("name", [](const PyTree& value) { return require_tree(value).name; }, [](const PyTree& value, std::string name) { invoke_on_viewer([value, name = std::move(name)]() mutable {
                                                                                                                                               auto& tree = require_tree(value);
                                                                                                                                               auto before = tree.to_json();
                                                                                                                                               tree.name = std::move(name);
                                                                                                                                               record_tree_mutation(tree, std::move(before), "tree:name");
                                                                                                                                           }); })
            .def_prop_ro("nodes", [](const PyTree& value) {
                std::vector<PyNode> result;
                for (const auto& node : require_tree(value).nodes)
                    result.push_back({value.uuid, node.name});
                return result; })
            .def_prop_ro("input_node", [](const PyTree& value) { return PyNode{value.uuid, require_tree(value).input_node().name}; })
            .def_prop_ro("output_node", [](const PyTree& value) { return PyNode{value.uuid, require_tree(value).output_node().name}; })
            .def_prop_ro("interface", [](const PyTree& value) { return PyInterface{value.uuid}; })
            .def("add_node", [](const PyTree& value, std::string type_id, std::string name, std::array<float, 2> location) {
                const auto node_name = invoke_on_viewer(
                    [value, type_id = std::move(type_id), name = std::move(name), location]() mutable {
                        auto& tree = require_tree(value);
                        auto before = tree.to_json();
                        auto& node = tree.add_node(std::move(type_id), std::move(name));
                        node.location = location;
                        const auto result = node.name;
                        record_tree_mutation(tree, std::move(before));
                        return result;
                    }, std::string{});
                return PyNode{value.uuid, node_name}; }, nb::arg("type_id"), nb::arg("name") = "", nb::arg("location") = std::array<float, 2>{0.0f, 0.0f})
            .def("remove_node", [](const PyTree& value, const std::string& name) { return invoke_on_viewer([value, name] {
                                                                                       auto& tree = require_tree(value);
                                                                                       auto before = tree.to_json();
                                                                                       const bool removed = tree.remove_node(name);
                                                                                       if (removed)
                                                                                           record_tree_mutation(tree, std::move(before));
                                                                                       return removed;
                                                                                   },
                                                                                                           false); })
            .def("link", [](const PyTree& value, const PyNode& from, const std::string& output, const PyNode& to, const std::string& input) {
                const auto error = invoke_on_viewer([value, from, output, to, input] {
                    auto& tree = require_tree(value);
                    auto before = tree.to_json();
                    std::string message;
                    if (tree.add_link({from.name, output, to.name, input}, &message,
                                      active_tree_resolver()))
                        record_tree_mutation(tree, std::move(before));
                    return message;
                }, std::string{"Viewer is unavailable"});
                if (!error.empty())
                    throw nb::value_error(error.c_str()); })
            .def("unlink", [](const PyTree& value, const PyNode& from, const std::string& output, const PyNode& to, const std::string& input) { return invoke_on_viewer([value, from, output, to, input] {
                                                                                                                                                    auto& tree = require_tree(value);
                                                                                                                                                    auto before = tree.to_json();
                                                                                                                                                    const bool removed = tree.remove_link({from.name, output, to.name, input});
                                                                                                                                                    if (removed)
                                                                                                                                                        record_tree_mutation(tree, std::move(before));
                                                                                                                                                    return removed;
                                                                                                                                                },
                                                                                                                                                                        false); })
            .def("copy", [](const PyTree& value, const std::vector<PyNode>& nodes) {
                std::vector<std::string> names;
                for (const auto& node : nodes)
                    names.push_back(node.name);
                if (auto* manager = live_manager()) {
                    const auto result = manager->copyNodes(value.uuid, names);
                    if (!result)
                        throw nb::value_error(result.error().message.c_str());
                    return *result;
                }
                return standalone_copy(value, names); })
            .def("paste", [](const PyTree& value, const std::string& text, std::optional<std::array<float, 2>> location) {
                if (auto* manager = live_manager()) {
                    const auto result = manager->pasteNodes(value.uuid, text, location);
                    if (!result)
                        throw nb::value_error(result.error().message.c_str());
                    std::vector<PyNode> nodes;
                    for (const auto& name : result->nodes)
                        nodes.push_back({value.uuid, name});
                    return nodes;
                }
                return standalone_paste(value, text, location); }, nb::arg("text"), nb::arg("location") = nb::none())
            .def("make_group", [](const PyTree& value, const std::vector<PyNode>& nodes, std::optional<std::string> name) {
                auto* manager = live_manager();
                if (!manager)
                    return standalone_make_group(value, nodes, name.value_or("Group"));
                std::vector<std::string> names;
                for (const auto& node : nodes)
                    names.push_back(node.name);
                const auto result = manager->makeGroup(value.uuid, names, name.value_or("Group"));
                if (!result)
                    throw nb::value_error(result.error().message.c_str());
                return PyNode{value.uuid, result->group_node}; }, nb::arg("nodes"), nb::arg("name") = nb::none())
            .def("ungroup", [](const PyTree& value, const PyNode& node) {
                auto* manager = live_manager();
                if (!manager) {
                    standalone_ungroup(value, node);
                    return;
                }
                const auto result = manager->ungroup(value.uuid, node.name);
                if (!result)
                    throw nb::value_error(result.error().message.c_str()); })
            .def("add_input", [](const PyTree& value, std::string identifier, std::string type, nb::object default_value, std::optional<double> min, std::optional<double> max) {
                auto converted = python_to_value(default_value);
                invoke_on_viewer([value, identifier = std::move(identifier),
                                  type = socket_type(std::move(type)),
                                  converted = std::move(converted), min, max]() mutable {
                    auto& tree = require_tree(value);
                    auto before = tree.to_json();
                    tree.group_interface.inputs.push_back(
                        {identifier, identifier, std::move(type), std::move(converted),
                         min, max, std::nullopt});
                    record_tree_mutation(tree, std::move(before));
                }); }, nb::arg("identifier"), nb::arg("type"), nb::arg("default") = nb::none(), nb::arg("min") = nb::none(), nb::arg("max") = nb::none())
            .def("to_json", [](const PyTree& value) {
                auto json = require_tree(value).to_json();
                auto* manager = live_manager();
                json["animation_tree"] = value.uuid;
                json["animation"] = manager ? manager->animationJson() : vis::nodeAnimationJson(standalone().sequencer.timeline().animationClip());
                return json.dump(2); });

        nb::class_<PyModifier>(module, "Modifier")
            .def_prop_ro("name", [](const PyModifier& value) { return require_modifier(value).name; })
            .def_prop_rw("enabled", [](const PyModifier& value) { return require_modifier(value).enabled; }, [](const PyModifier& value, bool enabled) { invoke_on_viewer([value, enabled] {
                                                                                                                                                             auto before = modifier_stack_json(value.node_uuid);
                                                                                                                                                             require_modifier(value).enabled = enabled;
                                                                                                                                                             live_manager()->recordStackEdit(
                                                                                                                                                                 value.node_uuid, std::move(before),
                                                                                                                                                                 value.uuid + ":enabled");
                                                                                                                                                         }); })
            .def_prop_rw("show_viewport", [](const PyModifier& value) { return require_modifier(value).show_viewport; }, [](const PyModifier& value, bool visible) { invoke_on_viewer([value, visible] {
                                                                                                                                                                         auto before = modifier_stack_json(value.node_uuid);
                                                                                                                                                                         require_modifier(value).show_viewport = visible;
                                                                                                                                                                         live_manager()->recordStackEdit(
                                                                                                                                                                             value.node_uuid, std::move(before),
                                                                                                                                                                             value.uuid + ":show_viewport");
                                                                                                                                                                     }); })
            .def("set_input", [](const PyModifier& value, const std::string& name, nb::object input) {
                auto converted = python_to_value(input);
                invoke_on_viewer([value, name, converted = std::move(converted)]() mutable {
                    auto before = modifier_stack_json(value.node_uuid);
                    require_modifier(value).input_overrides[name] = std::move(converted);
                    live_manager()->recordStackEdit(
                        value.node_uuid, std::move(before),
                        value.uuid + ":input:" + name);
                }); })
            .def("capture_selection", [](const PyModifier& value, const std::string& node_name) {
                const auto error = invoke_on_viewer([value, node_name] {
                    auto before = modifier_stack_json(value.node_uuid);
                    const auto result = live_manager()->captureSelection(
                        value.node_uuid, require_modifier(value).name, node_name);
                    if (!result)
                        return result.error().message;
                    live_manager()->recordStackEdit(value.node_uuid, std::move(before));
                    return std::string{};
                }, std::string{"Viewer is unavailable"});
                if (!error.empty())
                    throw nb::value_error(error.c_str()); });

        module.def("node_types", [] {
            nb::list result;
            for (const auto& type : active_registry().list_localized()) {
                nb::dict item;
                item["id"] = type->id;
                item["label"] = type->label;
                item["category"] = type->category;
                item["description"] = type->description;
                item["help"] = type->help;
                item["version"] = type->version;
                nb::list inputs;
                for (const auto& socket : type->inputs) {
                    nb::dict descriptor;
                    descriptor["identifier"] = socket.identifier;
                    descriptor["label"] = socket.label;
                    descriptor["type"] = socket.type;
                    descriptor["default"] = value_to_python(socket.default_value);
                    descriptor["min"] = socket.min ? nb::cast(*socket.min) : nb::none();
                    descriptor["max"] = socket.max ? nb::cast(*socket.max) : nb::none();
                    descriptor["soft_min"] = socket.soft_min ? nb::cast(*socket.soft_min) : nb::none();
                    descriptor["soft_max"] = socket.soft_max ? nb::cast(*socket.soft_max) : nb::none();
                    descriptor["step"] = socket.step ? nb::cast(*socket.step) : nb::none();
                    descriptor["field"] = socket.field;
                    descriptor["multi_input"] = socket.multi_input;
                    descriptor["hide_value"] = socket.hide_value;
                    descriptor["description"] = socket.description;
                    inputs.append(std::move(descriptor));
                }
                item["inputs"] = std::move(inputs);
                nb::list outputs;
                for (const auto& socket : type->outputs) {
                    nb::dict descriptor;
                    descriptor["identifier"] = socket.identifier;
                    descriptor["label"] = socket.label;
                    descriptor["type"] = socket.type;
                    descriptor["multi_input"] = socket.multi_input;
                    descriptor["description"] = socket.description;
                    outputs.append(std::move(descriptor));
                }
                item["outputs"] = std::move(outputs);
                nb::list properties;
                for (const auto& property : type->properties) {
                    nb::dict descriptor;
                    descriptor["identifier"] = property.identifier;
                    descriptor["label"] = property.label;
                    descriptor["description"] = property.description;
                    descriptor["kind"] = static_cast<int>(property.kind);
                    descriptor["default"] = json_to_python(property.default_value);
                    descriptor["items"] = property.items;
                    descriptor["min"] = property.min ? nb::cast(*property.min) : nb::none();
                    descriptor["max"] = property.max ? nb::cast(*property.max) : nb::none();
                    properties.append(std::move(descriptor));
                }
                item["properties"] = std::move(properties);
                result.append(std::move(item));
            }
            return result;
        });
        module.def("new_tree", [](std::string name) {
            const auto uuid = invoke_on_viewer([name = std::move(name)]() mutable {
                if (auto* manager = live_manager())
                    return manager->newTree(std::move(name)).uuid;
                auto tree = std::make_unique<NodeTree>(standalone().registry, standalone_tree_name(std::move(name)));
                const auto result = tree->uuid;
                standalone().trees[result] = std::move(tree);
                return result;
            }, std::string{});
            return PyTree{uuid}; }, nb::arg("name") = "Node Graph");
        module.def("get_tree", [](const std::string& name) -> std::optional<PyTree> {
            if (auto* manager = live_manager()) {
                const auto* tree = manager->tree(name);
                return tree ? std::optional{PyTree{tree->uuid}} : std::nullopt;
            }
            if (const auto found = standalone().trees.find(name); found != standalone().trees.end())
                return PyTree{found->first};
            for (const auto& [uuid, tree] : standalone().trees)
                if (tree->name == name)
                    return PyTree{uuid};
            return std::nullopt;
        });
        module.def("trees", [] {
            std::vector<PyTree> result;
            if (auto* manager = live_manager()) {
                for (const auto* tree : manager->trees())
                    result.push_back({tree->uuid});
            } else {
                for (const auto& [uuid, _] : standalone().trees)
                    result.push_back({uuid});
            }
            return result;
        });
        module.def("remove_tree", [](const std::string& name) {
            return invoke_on_viewer([name] {
                if (auto* manager = live_manager())
                    return manager->removeTree(name);
                const auto* tree = active_tree(name);
                if (!tree)
                    return false;
                const auto uuid = tree->uuid;
                vis::removeNodeAnimation(standalone().sequencer, uuid);
                return standalone().trees.erase(uuid) != 0;
            },
                                    false);
        });
        module.def("load_tree", [](const std::string& text) {
            const auto json = nlohmann::json::parse(text);
            const auto uuid = invoke_on_viewer([json] {
                if (auto* manager = live_manager())
                    return manager->loadTree(json).uuid;
                auto tree = std::make_unique<NodeTree>(
                    NodeTree::from_json(json, standalone().registry));
                tree->name = standalone_tree_name(tree->name, tree->uuid);
                const auto result = tree->uuid;
                if (json.contains("animation")) {
                    std::unordered_map<std::string, std::string> names;
                    for (const auto& node : tree->nodes)
                        names[node.name] = node.name;
                    vis::copyNodeAnimation(standalone().sequencer, json.at("animation"),
                                           json.value("animation_tree", result), result, names);
                }
                standalone().trees[result] = std::move(tree);
                return result;
            },
                                               std::string{});
            return PyTree{uuid};
        });
        module.def("link", [](const PyTree& tree, const PyNode& from, const std::string& output,
                              const PyNode& to, const std::string& input) {
            const auto error = invoke_on_viewer([tree, from, output, to, input] {
                auto& value = require_tree(tree);
                auto before = value.to_json();
                std::string message;
                if (value.add_link({from.name, output, to.name, input}, &message,
                                   active_tree_resolver()))
                    record_tree_mutation(value, std::move(before));
                return message;
            },
                                                std::string{"Viewer is unavailable"});
            if (!error.empty())
                throw nb::value_error(error.c_str());
        });
        module.def("unlink", [](const PyTree& tree, const PyNode& from, const std::string& output,
                                const PyNode& to, const std::string& input) {
            return invoke_on_viewer([tree, from, output, to, input] {
                auto& value = require_tree(tree);
                auto before = value.to_json();
                const bool removed = value.remove_link({from.name, output, to.name, input});
                if (removed)
                    record_tree_mutation(value, std::move(before));
                return removed;
            },
                                    false);
        });
        module.def("evaluate_tree", [](const PyTree& tree, const PyGeometry& geometry, std::optional<float> time, std::optional<std::string> device) {
            if (device && *device != "cpu" && *device != "gpu")
                throw std::invalid_argument("Evaluation device must be cpu or gpu");
            auto graph = require_tree(tree);
            auto* manager = live_manager();
            const auto* controller = manager ? manager->sequencer() : &standalone().sequencer;
            const float seconds = time.value_or(controller ? controller->playhead() : 0.0f);
            std::unordered_map<std::string, std::unique_ptr<NodeTree>> animated_groups;
            const auto* clip = controller ? controller->timeline().animationClip() : nullptr;
            vis::applyNodeAnimation(graph, clip, seconds);
            auto result = evaluate(graph,
                                   {.geometry = geometry.value,
                                    .device = device ? std::optional{*device == "cpu" ? core::Device::CPU : core::Device::GPU} : std::nullopt,
                                    .tree_resolver = [&](std::string_view id) -> const NodeTree* {
                                        auto& copy = animated_groups[std::string(id)];
                                        if (!copy) {
                                            const auto* original = active_tree(std::string(id));
                                            if (!original) return nullptr;
                                            copy = std::make_unique<NodeTree>(*original);
                                            vis::applyNodeAnimation(*copy, clip, seconds);
                                        }
                                        return copy.get();
                                    },
                                    .seconds = seconds,
                                    .frames_per_second = controller ? controller->framesPerSecond() : 24.0f});
            standalone().evaluations[tree.uuid] = result;
            if (!result.ok) {
                std::string messages;
                for (const auto& [node, message] : result.errors) {
                    if (!messages.empty())
                        messages += "; ";
                    messages += node + ": " + message;
                }
                throw nb::value_error(messages.c_str());
            }
            return PyGeometry{std::move(result.geometry)}; }, nb::arg("tree"), nb::arg("geometry"), nb::arg("time") = nb::none(), nb::arg("device") = nb::none());
        module.def("register_node", [](nb::object cls) {
            auto info = node_type_from_python(cls);
            const auto module_name = nb::cast<std::string>(cls.attr("__module__"));
            {
                std::lock_guard lock(python_types_mutex());
                python_types()[info.id] = safe_class(std::move(cls));
                python_type_modules()[info.id] = module_name;
            }
            install_python_type(info);
            return info.id;
        });
        module.def("unregister_node", &unregister_python_type);
        module.def("unregister_nodes_for_module", &unregister_python_types_for_module);

        module.def("add_modifier", [](const std::string& node_name, const PyTree& tree, std::string name) {
            const auto result = invoke_on_viewer(
                [node_name, tree, name = std::move(name)]() mutable
                    -> std::optional<PyModifier> {
                    auto uuid = scene_node_uuid(node_name);
                    if (!uuid || !live_manager())
                        return std::nullopt;
                    auto& modifier = live_manager()->addModifier(
                        *uuid, tree.uuid, std::move(name));
                    return PyModifier{*uuid, modifier.uuid};
                }, std::optional<PyModifier>{});
            if (!result)
                throw std::runtime_error("Scene node or scene manager is unavailable");
            return *result; }, nb::arg("node_name"), nb::arg("tree"), nb::arg("name") = "");
        module.def("templates", [] {
            const auto result = invoke_on_viewer([] {
                nlohmann::json values = nlohmann::json::array();
                if (auto* manager = live_manager())
                    for (const auto& value : manager->templates())
                        values.push_back({{"id", value.id}, {"name", value.name}, {"description", value.description}, {"category", value.category}, {"scene_kinds", value.scene_kinds}, {"adjust", value.adjust}, {"builtin", value.builtin}});
                return values;
            },
                                                 nlohmann::json::array());
            return json_to_python(result);
        });
        module.def("apply_template", [](const std::string& node_name, const std::string& template_id, std::string name) {
            const auto result = invoke_on_viewer(
                [node_name, template_id, name = std::move(name)]() mutable
                    -> std::optional<PyModifier> {
                    const auto uuid = scene_node_uuid(node_name);
                    if (!uuid || !live_manager())
                        return std::nullopt;
                    const auto applied = live_manager()->applyTemplate(*uuid, template_id,
                                                                        std::move(name));
                    if (!applied)
                        throw std::invalid_argument(applied.error().message);
                    return PyModifier{*uuid, (*applied)->uuid};
                }, std::optional<PyModifier>{});
            if (!result)
                throw std::runtime_error("Scene node or scene manager is unavailable");
            return *result; }, nb::arg("target"), nb::arg("id"), nb::arg("name") = "");
        module.def("save_template", [](const PyTree& tree, std::string name, std::string description, std::string category) {
            const auto result = invoke_on_viewer(
                [tree, name = std::move(name), description = std::move(description),
                 category = std::move(category)]() mutable {
                    if (!live_manager())
                        throw std::runtime_error("User templates require a running viewer");
                    auto saved = live_manager()->saveTemplate(tree.uuid, std::move(name),
                                                               std::move(description),
                                                               std::move(category));
                    if (!saved)
                        throw std::invalid_argument(saved.error().message);
                    return saved->id;
                }, std::string{});
            return result; }, nb::arg("tree"), nb::arg("name"), nb::arg("description"), nb::arg("category"));
        module.def("delete_template", [](const std::string& id) { return invoke_on_viewer([id] {
                                                                      if (!live_manager())
                                                                          return false;
                                                                      const auto result = live_manager()->deleteTemplate(id);
                                                                      if (!result)
                                                                          throw std::invalid_argument(result.error().message);
                                                                      return true;
                                                                  },
                                                                                          false); }, nb::arg("id"));
        module.def("rename_template", [](const std::string& id, const std::string& name) { return invoke_on_viewer([id, name] {
                                                                                               if (!live_manager())
                                                                                                   throw std::runtime_error("User templates require a running viewer");
                                                                                               const auto result = live_manager()->renameTemplate(id, name);
                                                                                               if (!result)
                                                                                                   throw std::invalid_argument(result.error().message);
                                                                                               return true;
                                                                                           },
                                                                                                                   false); }, nb::arg("id"), nb::arg("name"));
        module.def("import_template", [](const std::string& path) { return invoke_on_viewer([path] {
                                                                        if (!live_manager())
                                                                            throw std::runtime_error("User templates require a running viewer");
                                                                        const auto result = live_manager()->importTemplate(core::utf8_to_path(path));
                                                                        if (!result)
                                                                            throw std::invalid_argument(result.error().message);
                                                                        return result->id;
                                                                    },
                                                                                            std::string{}); }, nb::arg("path"));
        module.def("export_template", [](const std::string& id, const std::string& path) { return invoke_on_viewer([id, path] {
                                                                                               if (!live_manager())
                                                                                                   throw std::runtime_error("User templates require a running viewer");
                                                                                               const auto result = live_manager()->exportTemplate(id, core::utf8_to_path(path));
                                                                                               if (!result)
                                                                                                   throw std::invalid_argument(result.error().message);
                                                                                               return true;
                                                                                           },
                                                                                                                   false); }, nb::arg("id"), nb::arg("path"));
        module.def("preview", [](const std::string& target, const std::string& node, std::optional<std::string> socket) {
            const auto error = invoke_on_viewer([target, node, socket = std::move(socket)] {
                const auto uuid = scene_node_uuid(target);
                if (!uuid || !live_manager())
                    return std::string{"Scene node or scene manager is unavailable"};
                const auto result = live_manager()->previewSet(*uuid, node, socket);
                return result ? std::string{} : result.error().message;
            }, std::string{"Viewer is unavailable"});
            if (!error.empty())
                throw std::invalid_argument(error); }, nb::arg("target"), nb::arg("node"), nb::arg("socket") = nb::none());
        module.def("clear_preview", [] {
            invoke_on_viewer([] {
                if (live_manager())
                    live_manager()->previewClear();
            });
        });
        module.def("modifiers", [](const std::string& node_name) {
            std::vector<PyModifier> result;
            const auto uuid = scene_node_uuid(node_name);
            const auto* stack = uuid && live_manager() ? live_manager()->stack(*uuid) : nullptr;
            if (stack)
                for (const auto& modifier : stack->modifiers)
                    result.push_back({*uuid, modifier.uuid});
            return result;
        });
        module.def("remove_modifier", [](const std::string& node_name, const std::string& name) {
            return invoke_on_viewer([node_name, name] {
                const auto uuid = scene_node_uuid(node_name);
                return uuid && live_manager() && live_manager()->removeModifier(*uuid, name);
            },
                                    false);
        });
        module.def("move_modifier", [](const std::string& node_name, const std::string& name,
                                       size_t index) {
            return invoke_on_viewer([node_name, name, index] {
                const auto uuid = scene_node_uuid(node_name);
                return uuid && live_manager() && live_manager()->moveModifier(*uuid, name, index);
            },
                                    false);
        });
        module.def("performance", [](const bool reset) {
            const auto result = invoke_on_viewer([reset] {
                return live_manager() ? live_manager()->performance(reset) : nlohmann::json::object();
            }, nlohmann::json::object());
            return json_to_python(result); }, nb::arg("reset") = false);
        module.def(
            "profile", [](const bool enabled) {
                invoke_on_viewer([enabled] {
                    if (auto* manager = live_manager())
                        manager->setProfiling(enabled);
                    return 0; }, 0);
            },
            nb::arg("enabled"), "Include each node's device work in its time; each node then waits for the GPU.");
        module.def("evaluate", [](const std::string& node_name) {
            const auto result = invoke_on_viewer([node_name] {
                std::optional<nb::gil_scoped_release> release;
                if (PyGILState_Check())
                    release.emplace();
                const auto uuid = scene_node_uuid(node_name);
                return uuid && live_manager()
                           ? std::optional{live_manager()->evaluate(*uuid)}
                           : std::optional<vis::ModifierEvaluation>{};
            },
                                                 std::optional<vis::ModifierEvaluation>{});
            if (!result)
                throw std::runtime_error("Scene node or scene manager is unavailable");
            EvalResult core_result{.geometry = result->geometry, .ok = result->ok, .errors = result->errors, .time_ms = result->time_ms};
            return evaluation_dict(core_result);
        });
        module.def("evaluated", [](const std::string& node_name) -> std::optional<PyGeometry> {
            const auto geometry = invoke_on_viewer([node_name]() -> std::optional<Geometry> {
                const auto uuid = scene_node_uuid(node_name);
                return uuid && live_manager() ? live_manager()->evaluated(*uuid) : std::nullopt;
            },
                                                   std::optional<Geometry>{});
            return geometry ? std::optional{PyGeometry{*geometry}} : std::nullopt;
        });
        module.def("apply_modifier", [](const std::string& node_name, const std::string& name) {
            const auto error = invoke_on_viewer([node_name, name] {
                std::optional<nb::gil_scoped_release> release;
                if (PyGILState_Check())
                    release.emplace();
                const auto uuid = scene_node_uuid(node_name);
                if (!uuid || !live_manager())
                    return std::string{"Scene node or scene manager is unavailable"};
                const auto result = live_manager()->applyModifier(*uuid, name);
                return result ? std::string{} : result.error().message;
            },
                                                std::string{"Viewer is unavailable"});
            if (!error.empty())
                throw std::runtime_error(error);
        });
        // Release callback classes before Python tears down nanobind's types.
        // A module capsule runs too late and retains their declaration objects.
        nb::module_::import_("atexit").attr("register")(nb::cpp_function([] {
            std::lock_guard lock(python_types_mutex());
            python_types().clear();
            python_type_modules().clear();
        }));
    }
} // namespace lfs::python
