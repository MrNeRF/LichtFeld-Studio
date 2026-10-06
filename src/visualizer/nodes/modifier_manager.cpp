/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/nodes/modifier_manager.hpp"
#include "core/logger.hpp"
#include "ipc/view_context.hpp"
#include "modifier_evaluation_worker.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "sequencer/sequencer_controller.hpp"
#include "visualizer/nodes/camera_nodes.hpp"
#include "visualizer/nodes/node_animation.hpp"
#include "visualizer/nodes/viewport_coordinates.hpp"

#include "core/nodes/builtin.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_completion.hpp"
#include "lfs/training/live_model_mutation_guard.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "scene/scene_manager.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/trainer.hpp"
#include "visualizer/core/training_manager.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <glm/gtc/matrix_transform.hpp>
#include <numeric>
#include <ranges>

namespace lfs::vis {
    namespace {
        using lfs::nodes::Geometry;
        using lfs::nodes::NodeContext;
        using lfs::nodes::NodeError;
        using lfs::nodes::NodeTypeInfo;
        using lfs::nodes::PropertyDecl;
        using lfs::nodes::PropertyKind;
        using lfs::nodes::SocketDecl;

        const lfs::nodes::Node* previewNode(const ModifierManager& manager,
                                            const lfs::nodes::NodeTree*& graph, std::string_view path) {
            while (graph) {
                if (const auto* node = graph->find_node(path))
                    return node;
                const auto slash = path.find('/');
                if (slash == std::string_view::npos)
                    return nullptr;
                const auto* instance = graph->find_node(path.substr(0, slash));
                if (!instance || instance->type_id != "lfs.group")
                    return nullptr;
                graph = manager.tree(instance->properties.value("tree", std::string{}));
                path.remove_prefix(slash + 1);
            }
            return nullptr;
        }

        std::string new_uuid() {
            return core::generate_uuid_v4().to_string();
        }

        nlohmann::json stack_json(const ModifierStack& stack) {
            return stack.modifiers;
        }

        nlohmann::json evaluation_tree(nlohmann::json tree) {
            tree.erase("name");
            for (auto& node : tree["nodes"]) {
                node.erase("location");
                node.erase("ui");
            }
            return tree;
        }

        nlohmann::json evaluation_state(const nlohmann::json& state) {
            nlohmann::json result = state;
            result["trees"] = nlohmann::json::object();
            for (const auto& tree : state["trees"])
                result["trees"][tree.at("uuid").get<std::string>()] = evaluation_tree(tree);
            for (auto& stack : result["stacks"])
                for (auto& modifier : stack)
                    modifier.erase("name");
            return result;
        }

        class ModifierStateUndoEntry final : public op::UndoEntry {
        public:
            ModifierStateUndoEntry(ModifierManager& manager, nlohmann::json before,
                                   nlohmann::json after, std::string merge_key)
                : manager_(&manager),
                  before_(std::move(before)),
                  after_(std::move(after)),
                  merge_key_(std::move(merge_key)),
                  estimated_bytes_(before_.dump().size() + after_.dump().size()) {}

            void undo() override {
                if (const auto restored = manager_->restoreJson(before_); !restored)
                    throw op::HistoryCorruptionError(restored.error().message);
            }

            void redo() override {
                if (const auto restored = manager_->restoreJson(after_); !restored)
                    throw op::HistoryCorruptionError(restored.error().message);
            }

            [[nodiscard]] std::string name() const override {
                return "node.modifier_edit";
            }

            [[nodiscard]] size_t estimatedBytes() const override {
                return estimated_bytes_;
            }

            bool tryMerge(const UndoEntry& incoming) override {
                const auto* other = dynamic_cast<const ModifierStateUndoEntry*>(&incoming);
                if (!other || merge_key_.empty() || other->merge_key_ != merge_key_ ||
                    other->manager_ != manager_)
                    return false;
                after_ = other->after_;
                estimated_bytes_ = before_.dump().size() + after_.dump().size();
                return true;
            }

        private:
            ModifierManager* manager_ = nullptr;
            nlohmann::json before_;
            nlohmann::json after_;
            std::string merge_key_;
            size_t estimated_bytes_ = 0;
        };

        class NodeInputUndoEntry final : public op::UndoEntry {
        public:
            NodeInputUndoEntry(ModifierManager& manager, std::string tree, std::string node,
                               std::string input, lfs::nodes::Value before, lfs::nodes::Value after)
                : manager_(&manager), tree_(std::move(tree)), node_(std::move(node)), input_(std::move(input)), before_(std::move(before)), after_(std::move(after)), edited_at_(std::chrono::steady_clock::now()) {
                bytes_ = nlohmann::json(before_).dump().size() + nlohmann::json(after_).dump().size();
            }

            void undo() override { restore(before_); }
            void redo() override { restore(after_); }
            std::string name() const override { return "node.modifier_edit"; }
            std::size_t estimatedBytes() const override { return bytes_; }

            bool tryMerge(const UndoEntry& incoming) override {
                const auto* next = dynamic_cast<const NodeInputUndoEntry*>(&incoming);
                if (!next || next->manager_ != manager_ || next->tree_ != tree_ ||
                    next->node_ != node_ || next->input_ != input_ ||
                    next->edited_at_ - edited_at_ > std::chrono::milliseconds(500))
                    return false;
                after_ = next->after_;
                edited_at_ = next->edited_at_;
                bytes_ = nlohmann::json(before_).dump().size() + nlohmann::json(after_).dump().size();
                return true;
            }

        private:
            void restore(const lfs::nodes::Value& value) {
                auto* tree = manager_->tree(tree_);
                auto* node = tree ? tree->find_node(node_) : nullptr;
                if (!node)
                    throw op::HistoryCorruptionError(std::format("Cannot restore input '{}': node '{}' in graph '{}' is missing", input_, node_, tree_));
                node->input_values[input_] = value;
                manager_->markDirty();
            }

            ModifierManager* manager_;
            std::string tree_;
            std::string node_;
            std::string input_;
            lfs::nodes::Value before_;
            lfs::nodes::Value after_;
            std::chrono::steady_clock::time_point edited_at_;
            std::size_t bytes_ = 0;
        };

        nlohmann::json replace_tree_state(nlohmann::json state, std::string_view uuid,
                                          nlohmann::json tree) {
            auto& trees = state["trees"];
            for (auto& item : trees) {
                if (item.value("uuid", "") == uuid) {
                    item = std::move(tree);
                    return state;
                }
            }
            trees.push_back(std::move(tree));
            return state;
        }

        glm::vec3 vector_value(const lfs::nodes::Node& node, const std::string_view input,
                               const glm::vec3 fallback = glm::vec3(0.0f)) {
            const auto found = node.input_values.find(std::string(input));
            if (found == node.input_values.end())
                return fallback;
            if (const auto* value = found->second.get_if<glm::vec3>())
                return *value;
            if (const auto* value = found->second.get_if<glm::vec4>())
                return glm::vec3(*value);
            return fallback;
        }

        float float_value(const lfs::nodes::Node& node, const std::string_view input,
                          const float fallback = 0.0f) {
            const auto found = node.input_values.find(std::string(input));
            if (found == node.input_values.end())
                return fallback;
            if (const auto* value = found->second.get_if<float>())
                return *value;
            if (const auto* value = found->second.get_if<std::int64_t>())
                return static_cast<float>(*value);
            return fallback;
        }

        bool input_linked(const lfs::nodes::NodeTree& tree, const std::string_view node,
                          const std::string_view input) {
            return std::ranges::any_of(tree.links, [&](const auto& link) {
                return link.to_node == node && link.to_socket == input;
            });
        }

        lfs::nodes::Value default_for_socket(const std::string_view type) {
            using namespace lfs::nodes;
            if (type == GEOMETRY_SOCKET)
                return Geometry{};
            if (type == FLOAT_SOCKET)
                return 0.0f;
            if (type == INT_SOCKET)
                return std::int64_t(0);
            if (type == BOOL_SOCKET)
                return false;
            if (type == VECTOR_SOCKET || type == COLOUR_SOCKET)
                return glm::vec3(0.0f);
            if (type == STRING_SOCKET)
                return std::string{};
            return {};
        }

        std::string interface_identifier(const lfs::nodes::TreeInterface& sockets,
                                         std::string label) {
            if (label.empty())
                label = "Socket";
            for (auto& character : label)
                if (character == '\n' || character == '\r')
                    character = ' ';
            const auto exists = [&](const std::string_view value) {
                return std::ranges::any_of(sockets.inputs, [&](const auto& item) {
                           return item.identifier == value;
                       }) ||
                       std::ranges::any_of(sockets.outputs, [&](const auto& item) {
                           return item.identifier == value;
                       });
            };
            if (!exists(label))
                return label;
            const auto base = label;
            for (std::size_t suffix = 2;; ++suffix) {
                label = base + " " + std::to_string(suffix);
                if (!exists(label))
                    return label;
            }
        }

        void rewrite_group_references(nlohmann::json& tree,
                                      const std::unordered_map<std::string, std::string>& remap) {
            for (auto& node : tree["nodes"]) {
                if (node.value("type_id", "") != "lfs.group")
                    continue;
                auto& graph = node["properties"]["tree"];
                if (graph.is_string())
                    if (const auto found = remap.find(graph.get<std::string>()); found != remap.end())
                        graph = found->second;
            }
        }
    } // namespace

    HsvPickBands centreHsvPickBands(const glm::vec3 rgb, const float saturation_width,
                                    const float value_width) {
        const glm::vec3 value = glm::clamp(rgb, glm::vec3(0.0f), glm::vec3(1.0f));
        const float maximum = std::max({value.r, value.g, value.b});
        const float minimum = std::min({value.r, value.g, value.b});
        const float delta = maximum - minimum;
        float hue = 0.0f;
        if (delta > 1e-8f) {
            if (maximum == value.r)
                hue = std::fmod((value.g - value.b) / delta, 6.0f) / 6.0f;
            else if (maximum == value.g)
                hue = ((value.b - value.r) / delta + 2.0f) / 6.0f;
            else
                hue = ((value.r - value.g) / delta + 4.0f) / 6.0f;
            if (hue < 0.0f)
                hue += 1.0f;
        }
        const float saturation = maximum <= 1e-8f ? 0.0f : delta / maximum;
        const auto centred_band = [](const float centre, const float requested_width) {
            const float width = std::clamp(requested_width, 0.0f, 1.0f);
            float minimum = centre - width * 0.5f;
            float maximum = centre + width * 0.5f;
            if (minimum < 0.0f) {
                maximum -= minimum;
                minimum = 0.0f;
            }
            if (maximum > 1.0f) {
                minimum -= maximum - 1.0f;
                maximum = 1.0f;
            }
            return std::pair{std::max(0.0f, minimum), std::min(1.0f, maximum)};
        };
        const auto saturation_band = centred_band(saturation, saturation_width);
        const auto value_band = centred_band(maximum, value_width);
        return {
            .hue = hue,
            .saturation_min = saturation_band.first,
            .saturation_max = saturation_band.second,
            .value_min = value_band.first,
            .value_max = value_band.second,
        };
    }

    void to_json(nlohmann::json& json, const Modifier& modifier) {
        json = {{"uuid", modifier.uuid},
                {"name", modifier.name},
                {"tree_uuid", modifier.tree_uuid},
                {"enabled", modifier.enabled},
                {"show_viewport", modifier.show_viewport},
                {"input_overrides", modifier.input_overrides},
                {"stored_selections", modifier.stored_selections}};
    }

    void from_json(const nlohmann::json& json, Modifier& modifier) {
        modifier.uuid = json.value("uuid", new_uuid());
        modifier.name = json.value("name", "Node Graph");
        modifier.tree_uuid = json.value("tree_uuid", "");
        modifier.enabled = json.value("enabled", true);
        modifier.show_viewport = json.value("show_viewport", true);
        if (json.contains("input_overrides") && json["input_overrides"].is_object())
            modifier.input_overrides =
                json["input_overrides"].get<std::unordered_map<std::string, lfs::nodes::Value>>();
        if (json.contains("stored_selections") && json["stored_selections"].is_object())
            modifier.stored_selections =
                json["stored_selections"].get<std::unordered_map<std::string, nlohmann::json>>();
    }

    ModifierManager::ModifierManager(SceneManager& scene_manager)
        : scene_manager_(&scene_manager), viewer_thread_(std::this_thread::get_id()) {
        lfs::nodes::register_builtin_nodes(registry_);
        registerVisualizerNodes();
        worker_ = std::make_unique<ModifierEvaluationWorker>(registry_);
        last_scene_generation_ = scene_manager_->getScene().renderGeneration();
    }

    ModifierManager::~ModifierManager() = default;

    lfs::nodes::NodeTypeRegistry& ModifierManager::registry() noexcept {
        return registry_;
    }

    const lfs::nodes::NodeTypeRegistry& ModifierManager::registry() const noexcept {
        return registry_;
    }

    ModifierResult ModifierManager::previewSet(const core::Uuid& node_uuid,
                                               const std::string_view node_name,
                                               std::optional<std::string> socket) {
        const auto* modifiers = stack(node_uuid);
        if (!modifiers)
            return std::unexpected(ModifierError{"Target has no node modifiers"});
        const Modifier* owner = nullptr;
        const lfs::nodes::Node* node = nullptr;
        const lfs::nodes::NodeTree* graph = nullptr;
        std::string node_path;
        for (const auto& modifier : modifiers->modifiers) {
            const auto* candidate_graph = tree(modifier.tree_uuid);
            auto local_name = node_name;
            if (local_name.starts_with(modifier.uuid + "/"))
                local_name.remove_prefix(modifier.uuid.size() + 1);
            const auto* candidate = previewNode(*this, candidate_graph, local_name);
            if (!candidate)
                continue;
            if (node)
                return std::unexpected(ModifierError{"Node name is ambiguous across the modifier stack"});
            owner = &modifier;
            graph = candidate_graph;
            node = candidate;
            node_path = local_name;
        }
        if (!owner || !graph || !node)
            return std::unexpected(ModifierError{"Node does not exist in the target's modifier stack"});
        const auto outputs = lfs::nodes::effective_outputs(
            *graph, *node, [&](const std::string_view uuid) { return tree(uuid); });
        auto output = outputs.end();
        if (socket)
            output = std::ranges::find(outputs, *socket, &lfs::nodes::SocketDecl::identifier);
        else {
            output = std::ranges::find(outputs, std::string(lfs::nodes::GEOMETRY_SOCKET),
                                       &lfs::nodes::SocketDecl::type);
            if (output == outputs.end() && !outputs.empty())
                output = outputs.begin();
        }
        if (output == outputs.end())
            return std::unexpected(ModifierError{"Preview socket is not an output of this node"});
        static const std::unordered_set<std::string_view> supported{
            lfs::nodes::GEOMETRY_SOCKET, lfs::nodes::FLOAT_SOCKET, lfs::nodes::INT_SOCKET,
            lfs::nodes::BOOL_SOCKET, lfs::nodes::VECTOR_SOCKET, lfs::nodes::COLOUR_SOCKET};
        if (!supported.contains(output->type))
            return std::unexpected(ModifierError{"This output type cannot be previewed"});
        const auto previous = preview_ ? preview_->target : core::Uuid{};
        preview_ = NodePreviewState{.target = node_uuid, .modifier_uuid = owner->uuid, .tree_uuid = owner->tree_uuid, .node = node_path, .socket = output->identifier, .socket_type = output->type, .label = node->name};
        if (previous != core::Uuid{} && previous != node_uuid)
            markDirty(previous);
        markDirty(node_uuid);
        return {};
    }

    void ModifierManager::previewClear() {
        if (!preview_)
            return;
        const auto target = preview_->target;
        preview_.reset();
        markDirty(target);
    }

    lfs::nodes::NodeTree& ModifierManager::newTree(std::string name) {
        const auto before = restoring_ ? nlohmann::json{} : toJson(false);
        auto value = std::make_unique<lfs::nodes::NodeTree>(registry_, uniqueTreeName(std::move(name)));
        auto* result = value.get();
        trees_[result->uuid] = std::move(value);
        ++generation_;
        if (!restoring_)
            op::undoHistory().push(std::make_unique<ModifierStateUndoEntry>(
                *this, before, toJson(false), std::string{}));
        return *result;
    }

    lfs::nodes::NodeTree& ModifierManager::loadTree(const nlohmann::json& json) {
        const auto before = restoring_ ? nlohmann::json{} : toJson(false);
        auto value = std::make_unique<lfs::nodes::NodeTree>(
            lfs::nodes::NodeTree::from_json(json, registry_));
        value->name = uniqueTreeName(value->name, value->uuid);
        auto* result = value.get();
        trees_[result->uuid] = std::move(value);
        ++generation_;
        markDirty();
        if (auto* controller = sequencer(); controller && json.contains("animation")) {
            std::unordered_map<std::string, std::string> names;
            for (const auto& node : result->nodes)
                names[node.name] = node.name;
            copyNodeAnimation(*controller, json.at("animation"), json.value("animation_tree", result->uuid), result->uuid, names);
        }
        if (!restoring_)
            op::undoHistory().push(std::make_unique<ModifierStateUndoEntry>(
                *this, before, toJson(false), std::string{}));
        return *result;
    }

    lfs::nodes::NodeTree* ModifierManager::tree(std::string_view uuid_or_name) {
        if (const auto found = trees_.find(std::string(uuid_or_name)); found != trees_.end())
            return found->second.get();
        lfs::nodes::NodeTree* result = nullptr;
        for (const auto& [_, value] : trees_) {
            if (value->name != uuid_or_name)
                continue;
            if (result)
                return nullptr;
            result = value.get();
        }
        return result;
    }

    std::string ModifierManager::uniqueTreeName(std::string name, const std::string_view except_uuid) const {
        if (name.empty())
            name = "Node Graph";
        std::unordered_set<std::string_view> used;
        for (const auto& [uuid, value] : trees_)
            if (uuid != except_uuid)
                used.insert(value->name);
        auto candidate = name;
        for (size_t suffix = 2; used.contains(candidate); ++suffix)
            candidate = std::format("{} {}", name, suffix);
        return candidate;
    }

    const lfs::nodes::NodeTree* ModifierManager::tree(std::string_view uuid_or_name) const {
        return const_cast<ModifierManager*>(this)->tree(uuid_or_name);
    }

    std::vector<lfs::nodes::NodeTree*> ModifierManager::trees() {
        std::vector<lfs::nodes::NodeTree*> result;
        result.reserve(trees_.size());
        for (auto& [_, value] : trees_)
            result.push_back(value.get());
        std::ranges::sort(result, {}, &lfs::nodes::NodeTree::name);
        return result;
    }

    std::expected<std::string, ModifierError>
    ModifierManager::copyNodes(const std::string_view tree_uuid,
                               const std::vector<std::string>& names) const {
        const auto* graph = tree(tree_uuid);
        if (!graph)
            return std::unexpected(ModifierError{"Node graph does not exist"});
        std::unordered_set<std::string> selected(names.begin(), names.end());
        for (const auto& node : graph->nodes)
            if ((node.type_id == "lfs.group_input" || node.type_id == "lfs.group_output") &&
                selected.contains(node.name))
                selected.erase(node.name);
        nlohmann::json clipboard{{"format", "lfs.node-clipboard"},
                                 {"version", 1},
                                 {"nodes", nlohmann::json::array()},
                                 {"links", nlohmann::json::array()},
                                 {"trees", nlohmann::json::object()}};
        const auto source = graph->to_json();
        for (const auto& node : source["nodes"])
            if (selected.contains(node.value("name", "")))
                clipboard["nodes"].push_back(node);
        for (const auto& link : source["links"])
            if (selected.contains(link.value("from_node", "")) &&
                selected.contains(link.value("to_node", "")))
                clipboard["links"].push_back(link);

        std::unordered_set<std::string> visited;
        std::function<void(const nlohmann::json&)> include_groups = [&](const nlohmann::json& nodes) {
            for (const auto& node : nodes) {
                if (node.value("type_id", "") != "lfs.group")
                    continue;
                const std::string uuid = node.value("properties", nlohmann::json::object())
                                             .value("tree", "");
                const auto* referenced = tree(uuid);
                if (!referenced || !visited.insert(uuid).second)
                    continue;
                auto json = referenced->to_json();
                clipboard["trees"][uuid] = json;
                include_groups(json["nodes"]);
            }
        };
        include_groups(clipboard["nodes"]);
        clipboard["source_tree"] = graph->uuid;
        clipboard["animation"] = animationJson();
        return clipboard.dump();
    }

    std::expected<PasteNodesResult, ModifierError>
    ModifierManager::pasteNodes(const std::string_view tree_uuid, const std::string_view text,
                                const std::optional<std::array<float, 2>> location) {
        auto* destination = tree(tree_uuid);
        if (!destination)
            return std::unexpected(ModifierError{"Node graph does not exist"});
        nlohmann::json clipboard;
        try {
            clipboard = nlohmann::json::parse(text);
        } catch (const std::exception&) {
            return std::unexpected(ModifierError{"Clipboard does not contain LichtFeld nodes"});
        }
        if (!clipboard.is_object() || clipboard["format"] != "lfs.node-clipboard" ||
            clipboard["version"] != 1 || !clipboard["nodes"].is_array() ||
            !clipboard["links"].is_array() || !clipboard["trees"].is_object())
            return std::unexpected(ModifierError{"Clipboard does not contain LichtFeld nodes"});

        const auto before = toJson(false);
        std::unordered_map<std::string, std::string> graph_remap;
        const auto imported = clipboard.value("trees", nlohmann::json::object());
        for (const auto& [uuid, json] : imported.items()) {
            const auto* existing = tree(uuid);
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
        for (const auto& [uuid, source_json] : imported.items()) {
            if (tree(uuid) && graph_remap[uuid] == uuid)
                continue;
            auto json = source_json;
            json["uuid"] = graph_remap[uuid];
            rewrite_group_references(json, graph_remap);
            auto value = std::make_unique<lfs::nodes::NodeTree>(
                lfs::nodes::NodeTree::from_json(json, registry_));
            value->name = uniqueTreeName(value->name);
            trees_[value->uuid] = std::move(value);
        }

        auto nodes_json = clipboard["nodes"];
        nlohmann::json temporary{{"uuid", core::generate_uuid_v4().to_string()},
                                 {"name", "Clipboard"},
                                 {"tree_type", destination->tree_type},
                                 {"nodes", nodes_json},
                                 {"links", nlohmann::json::array()},
                                 {"interface", {{"inputs", nlohmann::json::array()}, {"outputs", nlohmann::json::array()}}}};
        rewrite_group_references(temporary, graph_remap);
        auto decoded = lfs::nodes::NodeTree::from_json(temporary, registry_);
        float centre_x = 0.0f;
        float centre_y = 0.0f;
        if (!decoded.nodes.empty()) {
            float min_x = decoded.nodes.front().location[0];
            float max_x = min_x;
            float min_y = decoded.nodes.front().location[1];
            float max_y = min_y;
            for (const auto& node : decoded.nodes) {
                min_x = std::min(min_x, node.location[0]);
                max_x = std::max(max_x, node.location[0]);
                min_y = std::min(min_y, node.location[1]);
                max_y = std::max(max_y, node.location[1]);
            }
            centre_x = (min_x + max_x) * 0.5f;
            centre_y = (min_y + max_y) * 0.5f;
        }
        const float offset_x = location ? (*location)[0] - centre_x : 30.0f;
        const float offset_y = location ? (*location)[1] - centre_y : 30.0f;
        std::unordered_map<std::string, std::string> node_remap;
        PasteNodesResult result;
        for (const auto& source : decoded.nodes) {
            if (source.type_id == "lfs.group_input" || source.type_id == "lfs.group_output")
                continue;
            auto& copy = destination->add_node(source.type_id, source.name);
            const auto unique = copy.name;
            copy = source;
            copy.name = unique;
            copy.location = {source.location[0] + offset_x, source.location[1] + offset_y};
            node_remap[source.name] = copy.name;
            result.nodes.push_back(copy.name);
        }
        for (const auto& source : decoded.nodes) {
            const auto pasted = node_remap.find(source.name);
            if (pasted == node_remap.end())
                continue;
            auto* copy = destination->find_node(pasted->second);
            const auto frame = source.ui.find("frame");
            if (!copy || frame == source.ui.end() || !frame->is_string())
                continue;
            const auto mapped = node_remap.find(frame->get_ref<const std::string&>());
            if (mapped == node_remap.end())
                copy->ui.erase("frame");
            else
                copy->ui["frame"] = mapped->second;
        }
        const lfs::nodes::TreeResolver resolver = [this](const std::string_view uuid) {
            return tree(uuid);
        };
        for (const auto& item : clipboard.value("links", nlohmann::json::array())) {
            lfs::nodes::Link link{node_remap[item.value("from_node", "")],
                                  item.value("from_socket", ""),
                                  node_remap[item.value("to_node", "")],
                                  item.value("to_socket", "")};
            if (link.from_node.empty() || link.to_node.empty() ||
                !destination->add_link(std::move(link), nullptr, resolver))
                ++result.dropped_links;
        }
        if (auto* controller = sequencer(); controller && clipboard.contains("animation")) {
            copyNodeAnimation(*controller, clipboard["animation"], clipboard.value("source_tree", ""),
                              destination->uuid, node_remap);
            for (const auto& [source, target] : graph_remap) {
                std::unordered_map<std::string, std::string> names;
                for (const auto& node : tree(target)->nodes)
                    names[node.name] = node.name;
                copyNodeAnimation(*controller, clipboard["animation"], source, target, names);
            }
        }
        recordLibraryEdit(before);
        return result;
    }

    std::expected<MakeGroupResult, ModifierError>
    ModifierManager::makeGroup(const std::string_view tree_uuid,
                               const std::vector<std::string>& names, std::string group_name) {
        auto* outer = tree(tree_uuid);
        if (!outer)
            return std::unexpected(ModifierError{"Node graph does not exist"});
        std::unordered_set<std::string> selected;
        for (const auto& name : names)
            if (const auto* node = outer->find_node(name);
                node && node->type_id != "lfs.group_input" && node->type_id != "lfs.group_output")
                selected.insert(name);
        if (selected.empty())
            return std::unexpected(ModifierError{"Select at least one node"});
        if (lfs::nodes::group_selection_would_cycle(*outer, selected))
            return std::unexpected(ModifierError{
                "Selection cannot be grouped because dependencies leave and re-enter it"});

        const auto before = toJson(false);
        const auto original_nodes = outer->nodes;
        const auto original_links = outer->links;
        auto nested = std::make_unique<lfs::nodes::NodeTree>(registry_,
                                                             uniqueTreeName(std::move(group_name)));
        nested->group_interface.inputs.clear();
        nested->group_interface.outputs.clear();
        nested->links.clear();
        const auto input_name = nested->input_node().name;
        const auto output_name = nested->output_node().name;
        const lfs::nodes::TreeResolver resolver = [this, nested_ptr = nested.get()](const std::string_view uuid) {
            return uuid == nested_ptr->uuid ? nested_ptr : tree(uuid);
        };
        std::unordered_map<std::string, std::string> node_remap;
        float centre_x = 0.0f;
        float centre_y = 0.0f;
        for (const auto& source : original_nodes) {
            if (!selected.contains(source.name))
                continue;
            auto& copy = nested->add_node(source.type_id, source.name);
            const auto unique = copy.name;
            copy = source;
            copy.name = unique;
            node_remap[source.name] = unique;
            centre_x += source.location[0];
            centre_y += source.location[1];
        }
        for (const auto& source : original_nodes) {
            const auto moved = node_remap.find(source.name);
            if (moved == node_remap.end())
                continue;
            auto* copy = nested->find_node(moved->second);
            const auto frame = source.ui.find("frame");
            if (!copy || frame == source.ui.end() || !frame->is_string())
                continue;
            const auto mapped = node_remap.find(frame->get_ref<const std::string&>());
            if (mapped == node_remap.end())
                copy->ui.erase("frame");
            else
                copy->ui["frame"] = mapped->second;
        }
        centre_x /= static_cast<float>(selected.size());
        centre_y /= static_cast<float>(selected.size());
        for (auto& node : nested->nodes)
            if (node.type_id != "lfs.group_input" && node.type_id != "lfs.group_output") {
                node.location[0] -= centre_x;
                node.location[1] -= centre_y;
            }

        std::unordered_map<std::string, std::string> incoming_sockets;
        std::unordered_map<std::string, std::string> outgoing_sockets;
        for (const auto& link : original_links) {
            const bool from_inside = selected.contains(link.from_node);
            const bool to_inside = selected.contains(link.to_node);
            if (from_inside && to_inside) {
                nested->add_link({node_remap[link.from_node], link.from_socket,
                                  node_remap[link.to_node], link.to_socket},
                                 nullptr, resolver);
                continue;
            }
            if (!from_inside && to_inside) {
                const std::string key = link.from_node + "\n" + link.from_socket;
                auto socket = incoming_sockets.find(key);
                if (socket == incoming_sockets.end()) {
                    const auto* source = outer->find_node(link.from_node);
                    const auto outputs = source ? lfs::nodes::effective_outputs(*outer, *source, resolver)
                                                : std::vector<lfs::nodes::SocketDecl>{};
                    const auto declaration = std::ranges::find(outputs, link.from_socket,
                                                               &lfs::nodes::SocketDecl::identifier);
                    if (declaration == outputs.end())
                        continue;
                    const auto* target = outer->find_node(link.to_node);
                    lfs::nodes::Value value = declaration->default_value;
                    if (target)
                        if (const auto own = target->input_values.find(link.to_socket);
                            own != target->input_values.end())
                            value = own->second;
                    const auto identifier = interface_identifier(
                        nested->group_interface,
                        declaration->label.empty() ? declaration->identifier : declaration->label);
                    nested->group_interface.inputs.push_back(
                        {identifier, declaration->label.empty() ? declaration->identifier : declaration->label,
                         declaration->type, value, declaration->min, declaration->max, declaration->step});
                    socket = incoming_sockets.emplace(key, identifier).first;
                }
                nested->add_link({input_name, socket->second, node_remap[link.to_node], link.to_socket},
                                 nullptr, resolver);
            } else if (from_inside && !to_inside) {
                const std::string key = link.from_node + "\n" + link.from_socket;
                auto socket = outgoing_sockets.find(key);
                if (socket == outgoing_sockets.end()) {
                    const auto* source = outer->find_node(link.from_node);
                    const auto outputs = source ? lfs::nodes::effective_outputs(*outer, *source, resolver)
                                                : std::vector<lfs::nodes::SocketDecl>{};
                    const auto declaration = std::ranges::find(outputs, link.from_socket,
                                                               &lfs::nodes::SocketDecl::identifier);
                    if (declaration == outputs.end())
                        continue;
                    const auto identifier = interface_identifier(
                        nested->group_interface,
                        declaration->label.empty() ? declaration->identifier : declaration->label);
                    nested->group_interface.outputs.push_back(
                        {identifier, declaration->label.empty() ? declaration->identifier : declaration->label,
                         declaration->type, declaration->default_value, declaration->min,
                         declaration->max, declaration->step});
                    socket = outgoing_sockets.emplace(key, identifier).first;
                    nested->add_link({node_remap[link.from_node], link.from_socket, output_name,
                                      identifier},
                                     nullptr, resolver);
                }
            }
        }
        nested->find_node(input_name)->location = {-240.0f, 0.0f};
        nested->find_node(output_name)->location = {240.0f, 0.0f};
        const std::string nested_uuid = nested->uuid;
        const std::string nested_name = nested->name;
        trees_[nested_uuid] = std::move(nested);
        for (const auto& name : selected)
            outer->remove_node(name);
        auto& group = outer->add_node("lfs.group", nested_name);
        group.location = {centre_x, centre_y};
        group.properties["tree"] = nested_uuid;
        for (const auto& socket : tree(nested_uuid)->group_interface.inputs)
            group.input_values[socket.identifier] = socket.default_value;
        const lfs::nodes::TreeResolver outer_resolver = [this](const std::string_view uuid) {
            return tree(uuid);
        };
        outer->links.clear();
        std::unordered_set<std::string> linked_inputs;
        for (const auto& link : original_links) {
            const bool from_inside = selected.contains(link.from_node);
            const bool to_inside = selected.contains(link.to_node);
            if (!from_inside && !to_inside) {
                outer->add_link(link, nullptr, outer_resolver);
            } else if (!from_inside && to_inside) {
                const auto key = link.from_node + "\n" + link.from_socket;
                if (linked_inputs.insert(key).second)
                    outer->add_link({link.from_node, link.from_socket, group.name,
                                     incoming_sockets.at(key)},
                                    nullptr, outer_resolver);
            } else if (from_inside && !to_inside) {
                const auto key = link.from_node + "\n" + link.from_socket;
                outer->add_link({group.name, outgoing_sockets.at(key), link.to_node,
                                 link.to_socket},
                                nullptr, outer_resolver);
            }
        }
        const auto result = MakeGroupResult{group.name, nested_uuid};
        if (auto* controller = sequencer())
            copyNodeAnimation(*controller, before.at("animation"), outer->uuid, nested_uuid, node_remap, true);
        recordLibraryEdit(before);
        return result;
    }

    std::expected<std::vector<std::string>, ModifierError>
    ModifierManager::ungroup(const std::string_view tree_uuid, const std::string_view node_name) {
        auto* outer = tree(tree_uuid);
        auto* group = outer ? outer->find_node(node_name) : nullptr;
        if (!group || group->type_id != "lfs.group")
            return std::unexpected(ModifierError{"Group node does not exist"});
        const auto graph_property = group->properties.find("tree");
        auto* nested = graph_property != group->properties.end() && graph_property->is_string()
                           ? tree(graph_property->get_ref<const std::string&>())
                           : nullptr;
        if (!nested)
            return std::unexpected(ModifierError{"Missing graph"});
        const auto before = toJson(false);
        const auto outer_links = outer->links;
        const auto group_values = group->input_values;
        const auto group_location = group->location;
        std::unordered_map<std::string, lfs::nodes::Link> incoming;
        for (const auto& link : outer_links) {
            if (link.to_node == node_name)
                incoming[link.to_socket] = link;
        }
        std::unordered_map<std::string, std::string> remap;
        std::vector<std::string> created;
        for (const auto& source : nested->nodes) {
            if (source.type_id == "lfs.group_input" || source.type_id == "lfs.group_output")
                continue;
            auto& copy = outer->add_node(source.type_id, source.name);
            const auto unique = copy.name;
            copy = source;
            copy.name = unique;
            copy.location = {source.location[0] + group_location[0],
                             source.location[1] + group_location[1]};
            remap[source.name] = unique;
            created.push_back(unique);
        }
        for (const auto& source : nested->nodes) {
            const auto pasted = remap.find(source.name);
            if (pasted == remap.end())
                continue;
            auto* copy = outer->find_node(pasted->second);
            const auto frame = source.ui.find("frame");
            if (!copy || frame == source.ui.end() || !frame->is_string())
                continue;
            const auto mapped = remap.find(frame->get_ref<const std::string&>());
            if (mapped == remap.end())
                copy->ui.erase("frame");
            else
                copy->ui["frame"] = mapped->second;
        }
        const auto input_node = nested->input_node().name;
        const auto output_node = nested->output_node().name;
        const auto nested_links = nested->links;
        outer->remove_node(node_name);
        outer->links.clear();
        const lfs::nodes::TreeResolver resolver = [this](const std::string_view uuid) {
            return tree(uuid);
        };
        // Rebuild the outer links in their original order. This is observable for
        // multi-input sockets, so a group output must occupy the same slot that the
        // group link occupied.
        for (const auto& link : outer_links) {
            if (link.to_node == node_name)
                continue;
            if (link.from_node != node_name) {
                outer->add_link(link, nullptr, resolver);
                continue;
            }
            for (const auto& inner : nested_links) {
                if (inner.to_node != output_node || inner.to_socket != link.from_socket)
                    continue;
                if (inner.from_node == input_node) {
                    if (const auto source = incoming.find(inner.from_socket); source != incoming.end())
                        outer->add_link({source->second.from_node, source->second.from_socket,
                                         link.to_node, link.to_socket},
                                        nullptr, resolver);
                } else {
                    outer->add_link({remap[inner.from_node], inner.from_socket,
                                     link.to_node, link.to_socket},
                                    nullptr, resolver);
                }
            }
        }
        // Links whose target is inside the group have no peers in the outer graph;
        // preserve their order exactly as stored by the nested graph.
        for (const auto& link : nested_links) {
            const bool from_input = link.from_node == input_node;
            const bool to_output = link.to_node == output_node;
            if (to_output)
                continue;
            if (!from_input) {
                outer->add_link({remap[link.from_node], link.from_socket, remap[link.to_node],
                                 link.to_socket},
                                nullptr, resolver);
            } else {
                if (const auto source = incoming.find(link.from_socket); source != incoming.end())
                    outer->add_link({source->second.from_node, source->second.from_socket,
                                     remap[link.to_node], link.to_socket},
                                    nullptr, resolver);
                else if (auto* target = outer->find_node(remap[link.to_node]))
                    if (const auto value = group_values.find(link.from_socket);
                        value != group_values.end())
                        target->input_values[link.to_socket] = value->second;
            }
        }
        if (auto* controller = sequencer())
            copyNodeAnimation(*controller, before.at("animation"), nested->uuid, outer->uuid, remap);
        recordLibraryEdit(before);
        return created;
    }

    ModifierResult ModifierManager::setGroupGraph(const std::string_view tree_uuid,
                                                  const std::string_view node_name,
                                                  const std::string_view graph_uuid) {
        auto* owner = tree(tree_uuid);
        auto* node = owner ? owner->find_node(node_name) : nullptr;
        const auto* referenced = tree(graph_uuid);
        if (!owner || !node || node->type_id != "lfs.group" || !referenced)
            return std::unexpected(ModifierError{"Group node or graph does not exist"});
        if (owner->tree_type != referenced->tree_type)
            return std::unexpected(ModifierError{"Group graph has a different tree type"});
        const lfs::nodes::TreeResolver resolver = [this](const std::string_view uuid) {
            return tree(uuid);
        };
        std::string cycle;
        if (lfs::nodes::group_reference_would_cycle(*owner, referenced->uuid, resolver, &cycle))
            return std::unexpected(ModifierError{"Group cycle: " + cycle});
        const auto before = owner->to_json();
        node->properties["tree"] = referenced->uuid;
        std::unordered_map<std::string, lfs::nodes::Value> values;
        for (const auto& socket : referenced->group_interface.inputs) {
            const auto existing = node->input_values.find(socket.identifier);
            values[socket.identifier] = existing == node->input_values.end()
                                            ? socket.default_value
                                            : existing->second;
        }
        node->input_values = std::move(values);
        const auto inputs = lfs::nodes::effective_inputs(*owner, *node, resolver);
        const auto outputs = lfs::nodes::effective_outputs(*owner, *node, resolver);
        std::erase_if(owner->links, [&](const auto& link) {
            if (link.from_node == node_name)
                return std::ranges::find(outputs, link.from_socket,
                                         &lfs::nodes::SocketDecl::identifier) == outputs.end();
            if (link.to_node == node_name)
                return std::ranges::find(inputs, link.to_socket,
                                         &lfs::nodes::SocketDecl::identifier) == inputs.end();
            return false;
        });
        recordTreeEdit(owner->uuid, before);
        return {};
    }

    ModifierResult ModifierManager::makeGroupSingleUser(const std::string_view tree_uuid,
                                                        const std::string_view node_name) {
        auto* owner = tree(tree_uuid);
        auto* node = owner ? owner->find_node(node_name) : nullptr;
        lfs::nodes::NodeTree* source = nullptr;
        if (node) {
            const auto property = node->properties.find("tree");
            if (property != node->properties.end() && property->is_string())
                source = tree(property->get_ref<const std::string&>());
        }
        if (!source)
            return std::unexpected(ModifierError{"Group node or graph does not exist"});
        const auto before = toJson(false);
        auto json = source->to_json();
        const auto uuid = core::generate_uuid_v4().to_string();
        json["uuid"] = uuid;
        json["name"] = uniqueTreeName(source->name);
        auto copy = std::make_unique<lfs::nodes::NodeTree>(
            lfs::nodes::NodeTree::from_json(json, registry_));
        trees_[uuid] = std::move(copy);
        node->properties["tree"] = uuid;
        if (auto* controller = sequencer()) {
            std::unordered_map<std::string, std::string> names;
            for (const auto& original : source->nodes)
                names[original.name] = original.name;
            copyNodeAnimation(*controller, before.at("animation"), source->uuid, uuid, names);
        }
        recordLibraryEdit(before);
        return {};
    }

    std::expected<std::string, ModifierError>
    ModifierManager::interfaceAdd(const std::string_view tree_uuid, const bool output,
                                  std::string type, std::string label,
                                  lfs::nodes::Value default_value,
                                  const std::optional<double> min,
                                  const std::optional<double> max,
                                  const std::optional<double> step) {
        auto* graph = tree(tree_uuid);
        if (!graph)
            return std::unexpected(ModifierError{"Node graph does not exist"});
        static const std::unordered_set<std::string> allowed{
            std::string(lfs::nodes::GEOMETRY_SOCKET), std::string(lfs::nodes::FLOAT_SOCKET),
            std::string(lfs::nodes::INT_SOCKET), std::string(lfs::nodes::BOOL_SOCKET),
            std::string(lfs::nodes::VECTOR_SOCKET), std::string(lfs::nodes::COLOUR_SOCKET),
            std::string(lfs::nodes::STRING_SOCKET)};
        if (!allowed.contains(type))
            return std::unexpected(ModifierError{"Unknown interface socket type"});
        const auto before = toJson(false);
        const auto identifier = interface_identifier(graph->group_interface, label);
        if (std::holds_alternative<std::monostate>(default_value.data))
            default_value = default_for_socket(type);
        lfs::nodes::InterfaceSocket socket{identifier, label.empty() ? identifier : std::move(label),
                                           std::move(type), std::move(default_value), min, max, step};
        auto& sockets = output ? graph->group_interface.outputs : graph->group_interface.inputs;
        sockets.push_back(socket);
        if (!output)
            for (auto& [_, other] : trees_)
                for (auto& node : other->nodes)
                    if (node.type_id == "lfs.group" && node.properties.value("tree", "") == graph->uuid)
                        node.input_values[identifier] = socket.default_value;
        recordLibraryEdit(before);
        return identifier;
    }

    ModifierResult ModifierManager::interfaceRemove(const std::string_view tree_uuid,
                                                    const bool output,
                                                    const std::string_view identifier) {
        auto* graph = tree(tree_uuid);
        if (!graph)
            return std::unexpected(ModifierError{"Node graph does not exist"});
        auto& sockets = output ? graph->group_interface.outputs : graph->group_interface.inputs;
        const auto found = std::ranges::find(sockets, identifier,
                                             &lfs::nodes::InterfaceSocket::identifier);
        if (found == sockets.end())
            return std::unexpected(ModifierError{"Interface socket does not exist"});
        const auto before = toJson(false);
        sockets.erase(found);
        const auto endpoint = output ? graph->output_node().name : graph->input_node().name;
        std::erase_if(graph->links, [&](const auto& link) {
            return output ? link.to_node == endpoint && link.to_socket == identifier
                          : link.from_node == endpoint && link.from_socket == identifier;
        });
        if (!output) {
            for (auto& [_, stack] : stacks_)
                for (auto& modifier : stack.modifiers)
                    if (modifier.tree_uuid == graph->uuid)
                        modifier.input_overrides.erase(std::string(identifier));
            for (auto& [_, other] : trees_)
                for (auto& node : other->nodes)
                    if (node.type_id == "lfs.group" && node.properties.value("tree", "") == graph->uuid)
                        node.input_values.erase(std::string(identifier));
        }
        for (auto& [_, other] : trees_)
            std::erase_if(other->links, [&](const auto& link) {
                const auto* from = other->find_node(link.from_node);
                const auto* to = other->find_node(link.to_node);
                return output ? from && from->type_id == "lfs.group" &&
                                    from->properties.value("tree", "") == graph->uuid &&
                                    link.from_socket == identifier
                              : to && to->type_id == "lfs.group" &&
                                    to->properties.value("tree", "") == graph->uuid &&
                                    link.to_socket == identifier;
            });
        recordLibraryEdit(before);
        return {};
    }

    ModifierResult ModifierManager::interfaceUpdate(const std::string_view tree_uuid,
                                                    const bool output,
                                                    const std::string_view identifier,
                                                    const nlohmann::json& changes) {
        auto* graph = tree(tree_uuid);
        if (!graph)
            return std::unexpected(ModifierError{"Node graph does not exist"});
        auto& sockets = output ? graph->group_interface.outputs : graph->group_interface.inputs;
        const auto found = std::ranges::find(sockets, identifier,
                                             &lfs::nodes::InterfaceSocket::identifier);
        if (found == sockets.end())
            return std::unexpected(ModifierError{"Interface socket does not exist"});
        const auto before = toJson(false);
        if (changes.contains("label") && changes["label"].is_string())
            found->label = changes["label"];
        if (changes.contains("default"))
            found->default_value = changes["default"].get<lfs::nodes::Value>();
        const auto number = [&](const char* key, std::optional<double>& value) {
            if (!changes.contains(key))
                return;
            value = changes[key].is_null() ? std::optional<double>{}
                                           : std::optional<double>{changes[key].get<double>()};
        };
        number("min", found->min);
        number("max", found->max);
        number("step", found->step);
        recordLibraryEdit(before);
        return {};
    }

    ModifierResult ModifierManager::interfaceMove(const std::string_view tree_uuid,
                                                  const bool output,
                                                  const std::string_view identifier,
                                                  std::size_t index) {
        auto* graph = tree(tree_uuid);
        if (!graph)
            return std::unexpected(ModifierError{"Node graph does not exist"});
        auto& sockets = output ? graph->group_interface.outputs : graph->group_interface.inputs;
        const auto found = std::ranges::find(sockets, identifier,
                                             &lfs::nodes::InterfaceSocket::identifier);
        if (found == sockets.end())
            return std::unexpected(ModifierError{"Interface socket does not exist"});
        const auto before = toJson(false);
        auto value = std::move(*found);
        const auto old_index = static_cast<std::size_t>(std::distance(sockets.begin(), found));
        sockets.erase(sockets.begin() + static_cast<std::ptrdiff_t>(old_index));
        index = std::min(index, sockets.size());
        sockets.insert(sockets.begin() + static_cast<std::ptrdiff_t>(index), std::move(value));
        recordLibraryEdit(before);
        return {};
    }

    std::expected<std::string, ModifierError>
    ModifierManager::frameWrap(const std::string_view tree_uuid,
                               const std::vector<std::string>& names, std::string label) {
        auto* graph = tree(tree_uuid);
        if (!graph)
            return std::unexpected(ModifierError{"Node graph does not exist"});
        std::vector<std::string> members;
        for (const auto& name : names)
            if (auto* node = graph->find_node(name); node && node->type_id != "lfs.frame")
                members.push_back(node->name);
        if (members.empty())
            return std::unexpected(ModifierError{"Select at least one node"});
        const auto before = graph->to_json();
        float x = 0.0f;
        float y = 0.0f;
        for (const auto& member : members) {
            const auto* node = graph->find_node(member);
            x += node->location[0];
            y += node->location[1];
        }
        auto& frame = graph->add_node("lfs.frame", label.empty() ? "Frame" : label);
        frame.location = {x / members.size(), y / members.size()};
        frame.properties["label"] = label.empty() ? frame.name : label;
        for (const auto& member : members)
            graph->find_node(member)->ui["frame"] = frame.name;
        const auto result = frame.name;
        recordTreeEdit(graph->uuid, before, {}, false);
        return result;
    }

    ModifierResult ModifierManager::frameSetMembers(const std::string_view tree_uuid,
                                                    const std::string_view frame_name,
                                                    const std::vector<std::string>& names) {
        auto* graph = tree(tree_uuid);
        auto* frame = graph ? graph->find_node(frame_name) : nullptr;
        if (!frame || frame->type_id != "lfs.frame")
            return std::unexpected(ModifierError{"Frame does not exist"});
        for (const auto& name : names)
            if (!graph->find_node(name))
                return std::unexpected(ModifierError{"Frame member does not exist"});
        const auto before = graph->to_json();
        const std::unordered_set<std::string> members(names.begin(), names.end());
        for (auto& node : graph->nodes) {
            if (node.ui.value("frame", "") == frame_name)
                node.ui.erase("frame");
            if (members.contains(node.name) && node.name != frame_name)
                node.ui["frame"] = frame_name;
        }
        recordTreeEdit(graph->uuid, before, {}, false);
        return {};
    }

    std::expected<std::string, ModifierError>
    ModifierManager::rerouteInsert(const std::string_view tree_uuid,
                                   const lfs::nodes::Link& link,
                                   const std::optional<std::array<float, 2>> location) {
        auto* graph = tree(tree_uuid);
        if (!graph || std::ranges::find(graph->links, link) == graph->links.end())
            return std::unexpected(ModifierError{"Link does not exist"});
        // MCP/UI callers commonly pass a reference into graph->links. Erasing that
        // vector element would invalidate the reference before the replacement links
        // are built, so retain an owning copy first.
        const auto original_link = link;
        const auto before = graph->to_json();
        const auto original = std::ranges::find(graph->links, original_link);
        const auto original_index = static_cast<std::size_t>(std::distance(graph->links.begin(), original));
        const auto* from = graph->find_node(link.from_node);
        const auto* to = graph->find_node(link.to_node);
        const std::array<float, 2> midpoint{
            from && to ? (from->location[0] + to->location[0]) * 0.5f : 0.0f,
            from && to ? (from->location[1] + to->location[1]) * 0.5f : 0.0f};
        auto& reroute = graph->add_node("lfs.reroute");
        reroute.location = location.value_or(midpoint);
        graph->remove_link(original_link);
        const lfs::nodes::TreeResolver resolver = [this](const std::string_view uuid) {
            return tree(uuid);
        };
        std::string error;
        if (!graph->add_link({original_link.from_node, original_link.from_socket, reroute.name, "Input"}, &error,
                             resolver) ||
            !graph->add_link({reroute.name, "Output", original_link.to_node, original_link.to_socket}, &error,
                             resolver)) {
            *graph = lfs::nodes::NodeTree::from_json(before, registry_);
            return std::unexpected(ModifierError{error});
        }
        const lfs::nodes::Link downstream{reroute.name, "Output", original_link.to_node, original_link.to_socket};
        const auto inserted = std::ranges::find(graph->links, downstream);
        if (inserted != graph->links.end()) {
            auto value = std::move(*inserted);
            graph->links.erase(inserted);
            graph->links.insert(graph->links.begin() +
                                    static_cast<std::ptrdiff_t>(std::min(original_index, graph->links.size())),
                                std::move(value));
        }
        const auto result = reroute.name;
        recordTreeEdit(graph->uuid, before);
        return result;
    }

    bool ModifierManager::removeTree(std::string_view uuid_or_name) {
        const auto* value = tree(uuid_or_name);
        if (!value)
            return false;
        const std::string uuid = value->uuid;
        const auto before = toJson(false);
        if (auto* controller = sequencer())
            removeNodeAnimation(*controller, uuid);
        trees_.erase(uuid);
        for (auto& [_, stack] : stacks_)
            std::erase_if(stack.modifiers, [&](const Modifier& modifier) { return modifier.tree_uuid == uuid; });
        ++generation_;
        markDirty();
        if (!restoring_)
            op::undoHistory().push(std::make_unique<ModifierStateUndoEntry>(
                *this, before, toJson(false), std::string{}));
        return true;
    }

    Modifier& ModifierManager::addModifier(const core::Uuid& node_uuid, std::string tree_uuid,
                                           std::string name) {
        const auto* node_tree = tree(tree_uuid);
        if (!node_tree)
            throw std::invalid_argument("Node tree does not exist");
        tree_uuid = node_tree->uuid;
        if (name.empty())
            name = node_tree->name;
        const auto before = stack(node_uuid) ? stack_json(*stack(node_uuid))
                                             : nlohmann::json::array();
        auto& modifiers = stacks_[node_uuid].modifiers;
        const std::string base = name;
        for (int suffix = 2; std::ranges::any_of(modifiers, [&](const Modifier& item) {
                 return item.name == name;
             });
             ++suffix)
            name = std::format("{} {}", base, suffix);
        modifiers.push_back({.uuid = new_uuid(), .name = std::move(name), .tree_uuid = std::move(tree_uuid)});
        markDirty(node_uuid);
        recordStackEdit(node_uuid, before);
        return modifiers.back();
    }

    Modifier* ModifierManager::findModifier(const core::Uuid& node_uuid, std::string_view name) {
        auto* value = stack(node_uuid);
        if (!value)
            return nullptr;
        const auto found = std::ranges::find(value->modifiers, name, &Modifier::name);
        return found == value->modifiers.end() ? nullptr : &*found;
    }

    bool ModifierManager::renameModifier(const core::Uuid& node_uuid, std::string_view name,
                                         std::string new_name) {
        auto* value = stack(node_uuid);
        if (!value || new_name.empty())
            return false;
        const auto found = std::ranges::find(value->modifiers, name, &Modifier::name);
        if (found == value->modifiers.end() ||
            std::ranges::any_of(value->modifiers, [&](const Modifier& modifier) {
                return &modifier != &*found && modifier.name == new_name;
            }))
            return false;
        if (found->name == new_name)
            return true;
        const auto before = stack_json(*value);
        found->name = std::move(new_name);
        markDirty(node_uuid);
        recordStackEdit(node_uuid, before);
        return true;
    }

    bool ModifierManager::removeModifier(const core::Uuid& node_uuid, std::string_view name) {
        auto* value = stack(node_uuid);
        if (!value)
            return false;
        const auto before = stack_json(*value);
        const auto found = std::ranges::find(value->modifiers, name, &Modifier::name);
        if (found == value->modifiers.end())
            return false;
        value->modifiers.erase(found);
        markDirty(node_uuid);
        recordStackEdit(node_uuid, before);
        return true;
    }

    bool ModifierManager::moveModifier(const core::Uuid& node_uuid, std::string_view name, size_t index) {
        auto* value = stack(node_uuid);
        if (!value || value->modifiers.empty())
            return false;
        const auto before = stack_json(*value);
        const auto found = std::ranges::find(value->modifiers, name, &Modifier::name);
        if (found == value->modifiers.end())
            return false;
        Modifier modifier = std::move(*found);
        value->modifiers.erase(found);
        index = std::min(index, value->modifiers.size());
        value->modifiers.insert(value->modifiers.begin() + static_cast<std::ptrdiff_t>(index),
                                std::move(modifier));
        markDirty(node_uuid);
        recordStackEdit(node_uuid, before);
        return true;
    }

    ModifierStack* ModifierManager::stack(const core::Uuid& node_uuid) {
        const auto found = stacks_.find(node_uuid);
        return found == stacks_.end() ? nullptr : &found->second;
    }

    const ModifierStack* ModifierManager::stack(const core::Uuid& node_uuid) const {
        return const_cast<ModifierManager*>(this)->stack(node_uuid);
    }

    ModifierResult ModifierManager::setNodeInput(const std::string_view tree_uuid,
                                                 const std::string_view node_name,
                                                 const std::string_view input,
                                                 lfs::nodes::Value value) {
        auto* graph = tree(tree_uuid);
        auto* node = graph ? graph->find_node(node_name) : nullptr;
        if (!node)
            return std::unexpected(ModifierError{std::format("Node '{}' does not exist in graph '{}'", node_name, tree_uuid)});
        const lfs::nodes::TreeResolver resolver = [this](const std::string_view uuid) {
            return tree(uuid);
        };
        const auto sockets = lfs::nodes::effective_inputs(*graph, *node, resolver);
        const auto socket = std::ranges::find(sockets, input, &lfs::nodes::SocketDecl::identifier);
        if (socket == sockets.end())
            return std::unexpected(ModifierError{std::format("Input '{}' does not exist on node '{}' ({})", input, node_name, node->type_id)});
        if (const auto* controller = sequencer(); controller && animatableNodeInput(*graph, *node, *socket))
            if (const auto* clip = controller->timeline().animationClip();
                clip && clip->getTrackByPath(nodeInputTrackPath(graph->uuid, node->name, input)))
                return keyframeSet(graph->uuid, node->name, input, {}, std::move(value));
        const auto found = node->input_values.find(std::string(input));
        auto before = found == node->input_values.end() ? socket->default_value : found->second;
        if (nlohmann::json(before) == nlohmann::json(value))
            return {};
        node->input_values[std::string(input)] = value;
        op::undoHistory().push(std::make_unique<NodeInputUndoEntry>(
            *this, graph->uuid, node->name, std::string(input), std::move(before), std::move(value)));
        markDirty();
        return {};
    }

    void ModifierManager::markDirty(const core::Uuid& node_uuid) {
        animation_only_request_ = false;
        if (preview_ && (node_uuid.is_nil() || node_uuid == preview_->target)) {
            const auto* graph = tree(preview_->tree_uuid);
            const auto* modifiers = stack(preview_->target);
            const bool has_modifier = modifiers && std::ranges::any_of(
                                                       modifiers->modifiers, [&](const Modifier& item) {
                                                           return item.uuid == preview_->modifier_uuid && item.tree_uuid == preview_->tree_uuid;
                                                       });
            if (!graph || !has_modifier || !previewNode(*this, graph, preview_->node))
                preview_.reset();
        }
        if (auto* controller = sequencer(); controller && !controller->timeline().hasAnimationClip() && timeDependent()) {
            controller->timeline().ensureAnimationClip();
            controller->animationTracksChanged();
        }
        ++generation_;
        worker_->invalidate(++output_generation_);
        if (node_uuid.is_nil()) {
            for (auto& [_, state] : runtime_)
                state.dirty = true;
            for (const auto& [uuid, _] : stacks_)
                runtime_[uuid].dirty = true;
            return;
        }
        runtime_[node_uuid].dirty = true;
    }

    ModifierResult
    ModifierManager::captureSelection(const core::Uuid& node_uuid, std::string_view modifier_name,
                                      std::string_view stored_selection_node) {
        auto& scene = scene_manager_->getScene();
        const auto* node = scene.getNodeByUuid(node_uuid);
        auto* modifier = findModifier(node_uuid, modifier_name);
        if (!node || !modifier)
            return std::unexpected(ModifierError{"Scene node or modifier does not exist"});
        const auto* node_tree = tree(modifier->tree_uuid);
        const auto* stored_node = node_tree ? node_tree->find_node(stored_selection_node) : nullptr;
        if (!stored_node || stored_node->type_id != "lfs.stored_selection")
            return std::unexpected(ModifierError{"Stored Selection node does not exist"});
        if (modifier->enabled && modifier->show_viewport && node->evaluated_model && node->model &&
            node->evaluated_model->size() != node->model->size())
            return std::unexpected(
                ModifierError{"Hide the node modifiers before capturing a stored selection"});
        auto selection = scene.selectionMaskSliceForNode(node->id);
        if (!selection)
            selection = std::make_shared<core::Tensor>(core::Tensor::zeros_bool(
                {node->model ? static_cast<size_t>(node->model->size()) : 0}, core::Device::CPU));
        lfs::nodes::Node encoded;
        lfs::nodes::set_stored_selection(encoded, selection->to(core::DataType::Bool));
        const nlohmann::json before = stack(node_uuid)->modifiers;
        modifier->stored_selections[std::string(stored_selection_node)] = encoded.properties;
        recordStackEdit(node_uuid, before);
        return {};
    }

    ModifierResult
    ModifierManager::applyModifier(const core::Uuid& node_uuid, std::string_view modifier_name) {
        auto& scene = scene_manager_->getScene();
        auto* node = scene.getNodeByUuid(node_uuid);
        auto* modifiers = stack(node_uuid);
        if (!node || !modifiers)
            return std::unexpected(ModifierError{"Scene node or modifier does not exist"});
        const auto found = std::ranges::find(modifiers->modifiers, modifier_name, &Modifier::name);
        if (found == modifiers->modifiers.end())
            return std::unexpected(ModifierError{"Modifier does not exist"});
        const bool training_model = scene_manager_->getContentType() == SceneManager::ContentType::Dataset &&
                                    node_uuid == scene.getTrainingModelNodeUuid();
        auto* trainer_manager = scene_manager_->getTrainerManager();
        if (training_model)
            waitForTrainingBoundary();
        if (training_model && trainer_manager && trainer_manager->isModelChanging())
            return std::unexpected(ModifierError{"Modifiers are paused while training"});
        const size_t index = static_cast<size_t>(std::distance(modifiers->modifiers.begin(), found));
        auto prepared = evaluateForApply(node_uuid, index);
        const auto& result = prepared.evaluation;
        if (!result.ok)
            return std::unexpected(ModifierError{
                result.errors.empty() ? "Modifier evaluation failed" : result.errors.begin()->second});
        if ((node->type == core::NodeType::SPLAT && !result.geometry.splats) ||
            (node->type == core::NodeType::POINTCLOUD && !result.geometry.points) ||
            (node->type == core::NodeType::MESH && !result.geometry.mesh))
            return std::unexpected(
                ModifierError{"The modifier result does not match the host geometry type"});
        if (training_model && (!prepared.splats || !node->model ||
                               prepared.splats->size() != node->model->size()))
            return std::unexpected(
                ModifierError{"Export the result or stop training first"});

        const auto manager_before = toJson(false);
        op::SceneGraphCaptureOptions options;
        options.mode = op::SceneGraphCaptureMode::FULL;
        options.include_selected_nodes = true;
        options.include_scene_context = true;
        options.payload_uuids = {node_uuid};
        auto scene_before = op::SceneGraphPatchEntry::captureStateByIds(
            *scene_manager_, {node->id}, options);
        op::TransactionGuard transaction("node.apply_modifier");
        if (training_model) {
            auto& destination = *node->model;
            auto* trainer = trainer_manager ? trainer_manager->getTrainer() : nullptr;
            std::unique_lock<std::shared_mutex> render_lock;
            std::unique_lock<std::shared_mutex> model_lock;
            if (trainer) {
                render_lock = std::unique_lock<std::shared_mutex>(trainer->getRenderMutex());
                model_lock = std::unique_lock<std::shared_mutex>(trainer->getModelAccessMutex());
            }
            lfs::training::LiveModelMutationGuard mutation_guard("ModifierManager::applyModifier");
            destination.means().copy_(prepared.splats->means_raw());
            destination.sh0().copy_(prepared.splats->sh0_raw());
            destination.scaling_raw().copy_(prepared.splats->scaling_raw());
            destination.rotation_raw().copy_(prepared.splats->rotation_raw());
            destination.opacity_raw().copy_(prepared.splats->opacity_raw());
            if (destination.max_sh_coeffs_rest() > 0) {
                auto canonical = destination.shN_canonical();
                const auto source = prepared.splats->shN_canonical();
                if (source.is_valid() && source.numel() > 0) {
                    const int coefficients = static_cast<int>(std::min(canonical.size(1), source.size(1)));
                    canonical.slice(1, 0, coefficients).copy_(source.slice(1, 0, coefficients));
                }
                std::vector<int> rows(static_cast<size_t>(destination.size()));
                std::iota(rows.begin(), rows.end(), 0);
                const auto indices = core::Tensor::from_vector(
                                         rows, {rows.size()}, core::Device::CPU)
                                         .to(destination.means_raw().device());
                lfs::training::sh_value::scatter_canonical_into_shN(destination, indices, canonical);
            }
            destination.set_active_sh_degree(prepared.splats->get_active_sh_degree());
            if (trainer && trainer->isInitialized())
                trainer->get_strategy_mutable().get_optimizer().reset_all_states();
            core::TensorCompletion completion;
            completion.include_current_gpu();
            completion.wait();
            scene.clearNodeEvaluatedPayload(node->id);
            scene.markPayloadDiverged(node->id);
            scene.notifyMutation(core::Scene::MutationType::MODEL_CHANGED);
        } else if (node->type == core::NodeType::SPLAT) {
            auto baked = std::make_unique<core::SplatData>(prepared.splats->readOnlySnapshot());
            // Later edits that rebuild tensors must stay in renderer storage.
            if (auto allocator = scene_manager_->makeExternalSplatAllocator())
                baked->set_tensor_allocator(std::move(allocator));
            scene.replaceNodeModel(node->name, std::move(baked));
        } else if (node->type == core::NodeType::POINTCLOUD) {
            scene.replaceNodePointCloud(
                node->name, std::move(prepared.points));
        } else {
            scene.replaceNodeMesh(
                node->name,
                std::move(prepared.mesh));
        }
        modifiers->modifiers.erase(modifiers->modifiers.begin(),
                                   modifiers->modifiers.begin() + static_cast<std::ptrdiff_t>(index + 1));
        markDirty(node_uuid);
        tick();
        auto scene_after = op::SceneGraphPatchEntry::captureStateByIds(
            *scene_manager_, {node->id}, options);
        op::undoHistory().push(std::make_unique<op::SceneGraphPatchEntry>(
            *scene_manager_, "node.apply_modifier", std::move(scene_before), std::move(scene_after)));
        op::undoHistory().push(std::make_unique<ModifierStateUndoEntry>(
            *this, manager_before, toJson(false), std::string{}));
        transaction.commit();
        return {};
    }

    nlohmann::json ModifierManager::toJson(bool existing_nodes_only) const {
        nlohmann::json result = {{"schema_version", 1}, {"trees", nlohmann::json::array()}, {"stacks", nlohmann::json::object()}};
        // Project persistence uses the sequencer chapter. Only undo snapshots include
        // the node-track subset so graph and target-path edits undo atomically.
        if (!existing_nodes_only)
            result["animation"] = animationJson();
        for (const auto& [_, value] : trees_)
            result["trees"].push_back(value->to_json());
        std::sort(result["trees"].begin(), result["trees"].end(), [](const auto& a, const auto& b) {
            return a.at("uuid").template get<std::string>() < b.at("uuid").template get<std::string>();
        });
        for (const auto& [uuid, value] : stacks_) {
            if (existing_nodes_only && !scene_manager_->getScene().getNodeByUuid(uuid))
                continue;
            result["stacks"][uuid.to_string()] = value.modifiers;
        }
        return result;
    }

    ModifierResult ModifierManager::restoreJson(const nlohmann::json& json) {
        if (!json.is_object() || json.value("schema_version", 0) != 1 ||
            !json.contains("trees") || !json["trees"].is_array() ||
            !json.contains("stacks") || !json["stacks"].is_object())
            return std::unexpected(ModifierError{"Invalid NODE chapter"});
        decltype(trees_) trees;
        decltype(stacks_) stacks;
        try {
            for (const auto& item : json["trees"]) {
                auto tree = std::make_unique<lfs::nodes::NodeTree>(lfs::nodes::NodeTree::from_json(item, registry_));
                trees[tree->uuid] = std::move(tree);
            }
            for (const auto& [uuid_text, items] : json["stacks"].items()) {
                const auto uuid = core::Uuid::from_string(uuid_text);
                if (!uuid || !items.is_array())
                    continue;
                stacks[*uuid].modifiers = items.get<std::vector<Modifier>>();
            }
        } catch (const std::exception& error) {
            return std::unexpected(ModifierError{error.what()});
        }
        const bool output_changed = evaluation_state(toJson(false)) != evaluation_state(json);
        if (json.contains("animation"))
            if (auto* controller = sequencer())
                restoreNodeAnimation(*controller, json.at("animation"));
        trees_ = std::move(trees);
        stacks_ = std::move(stacks);
        std::erase_if(runtime_, [&](const auto& entry) {
            if (stacks_.contains(entry.first))
                return false;
            scene_manager_->getScene().clearNodeEvaluatedPayload(entry.first);
            return true;
        });
        if (output_changed)
            markDirty();
        else
            ++generation_;
        return {};
    }

    void ModifierManager::clearEvaluatedPayloads() {
        auto& scene = scene_manager_->getScene();
        for (const auto* node : scene.getNodes())
            if (node && scene.hasEvaluatedPayload(node->id))
                scene.clearNodeEvaluatedPayload(node->id);
    }

    void ModifierManager::clear() {
        cancelViewportMode();
        viewport_selection_.reset();
        preview_.reset();
        gizmo_before_.reset();
        worker_->invalidate(++output_generation_);
        ++source_generation_;
        clearEvaluatedPayloads();
        trees_.clear();
        stacks_.clear();
        runtime_.clear();
        ++generation_;
    }

    void ModifierManager::recordTreeEdit(std::string_view tree_uuid, nlohmann::json before,
                                         std::string merge_key, const bool reevaluate) {
        if (restoring_)
            return;
        auto* edited = tree(tree_uuid);
        if (!edited)
            return;
        if (before.value("name", "") != edited->name)
            edited->name = uniqueTreeName(edited->name, edited->uuid);
        const auto after_tree = edited->to_json();
        if (after_tree == before)
            return;
        const auto before_animation = animationJson();
        if (auto* controller = sequencer()) {
            for (const auto& node : before.at("nodes"))
                if (!edited->find_node(node.at("name").get<std::string>()))
                    removeNodeAnimation(*controller, edited->uuid, node.at("name").get<std::string>());
            const auto* clip = controller->timeline().animationClip();
            for (const auto& node : edited->nodes) {
                const auto previous = std::ranges::find_if(before.at("nodes"), [&](const auto& item) {
                    return item.value("name", "") == node.name;
                });
                if (previous == before.at("nodes").end())
                    continue;
                for (const auto& [input, value] : node.input_values) {
                    if (!clip || !clip->getTrackByPath(nodeInputTrackPath(edited->uuid, node.name, input)))
                        continue;
                    const auto old_values = previous->value("input_values", nlohmann::json::object());
                    const nlohmann::json encoded = value;
                    if (old_values.contains(input) && old_values.at(input) == encoded)
                        continue;
                    const auto result = setNodeKeyframe(*edited, node.name, input, *controller, {}, value,
                                                        sequencer::EasingType::LINEAR, [this](std::string_view id) { return tree(id); });
                    if (!result)
                        LOG_DEBUG("Node input animation unchanged: {}", result.error().message);
                }
            }
        }
        const bool output_changed = evaluation_tree(after_tree) != evaluation_tree(before);
        auto after = toJson(false);
        auto before_state = replace_tree_state(after, tree_uuid, std::move(before));
        before_state["animation"] = before_animation;
        op::undoHistory().push(std::make_unique<ModifierStateUndoEntry>(
            *this, std::move(before_state), std::move(after), std::move(merge_key)));
        if (output_changed && reevaluate)
            markDirty();
        else
            ++generation_;
    }

    void ModifierManager::recordStackEdit(const core::Uuid& node_uuid, nlohmann::json before,
                                          std::string merge_key) {
        if (restoring_)
            return;
        auto after = toJson(false);
        auto before_state = after;
        before_state["stacks"][node_uuid.to_string()] = std::move(before);
        op::undoHistory().push(std::make_unique<ModifierStateUndoEntry>(
            *this, std::move(before_state), std::move(after), std::move(merge_key)));
        if (const auto* modifiers = stack(node_uuid);
            modifiers && std::ranges::none_of(modifiers->modifiers, [](const Modifier& modifier) {
                return modifier.enabled && modifier.show_viewport;
            }))
            scene_manager_->getScene().clearNodeEvaluatedPayload(node_uuid);
        markDirty(node_uuid);
    }

    void ModifierManager::recordLibraryEdit(nlohmann::json before, std::string merge_key) {
        if (restoring_)
            return;
        auto after = toJson(false);
        if (before == after)
            return;
        op::undoHistory().push(std::make_unique<ModifierStateUndoEntry>(
            *this, std::move(before), std::move(after), std::move(merge_key)));
        markDirty();
    }

    std::uint64_t ModifierManager::generation() const {
        return generation_;
    }

    void ModifierManager::setViewportNodeSelection(const core::Uuid& host, std::string tree_uuid,
                                                   std::string node_name, const bool editor_visible) {
        const bool changed = !viewport_selection_ || viewport_selection_->host != host ||
                             viewport_selection_->tree_uuid != tree_uuid ||
                             viewport_selection_->node != node_name;
        if (changed) {
            if (paint_before_)
                endPaintStroke(true);
            paint_mode_ = false;
            colour_pick_.reset();
            gizmo_before_.reset();
        }
        viewport_selection_ = ViewportSelection{host, std::move(tree_uuid), std::move(node_name),
                                                editor_visible};
    }

    void ModifierManager::setViewportEditorVisible(const bool visible) {
        if (!viewport_selection_)
            return;
        viewport_selection_->editor_visible = visible;
        if (!visible)
            cancelViewportMode();
    }

    void ModifierManager::clearViewportNodeSelection() {
        if (paint_before_)
            endPaintStroke(true);
        viewport_selection_.reset();
        paint_mode_ = false;
        colour_pick_.reset();
        gizmo_before_.reset();
    }

    std::optional<NodeViewportGizmo> ModifierManager::viewportNodeGizmo() const {
        if (!viewport_selection_ || !viewport_selection_->editor_visible)
            return std::nullopt;
        const auto* graph = tree(viewport_selection_->tree_uuid);
        const auto* node = graph ? graph->find_node(viewport_selection_->node) : nullptr;
        const auto* host = scene_manager_->getScene().getNodeByUuid(viewport_selection_->host);
        if (!node || !host)
            return std::nullopt;

        NodeViewportGizmo result;
        result.host = host->uuid;
        result.host_id = host->id;
        result.tree_uuid = graph->uuid;
        result.node = node->name;
        nodes::NodeViewportTransform transform;
        std::vector<std::string_view> editable_inputs;
        if (node->type_id == "lfs.box_selection") {
            result.kind = NodeViewportGizmoKind::Box;
            transform.translation = vector_value(*node, "Centre");
            transform.rotation_degrees = vector_value(*node, "Rotation");
            transform.scale = glm::max(vector_value(*node, "Size", glm::vec3(1.0f)), glm::vec3(1e-6f));
            result.falloff = std::max(0.0f, float_value(*node, "Falloff"));
            editable_inputs = {"Centre", "Rotation", "Size"};
        } else if (node->type_id == "lfs.ellipsoid_selection") {
            result.kind = NodeViewportGizmoKind::Ellipsoid;
            transform.translation = vector_value(*node, "Centre");
            transform.rotation_degrees = vector_value(*node, "Rotation");
            transform.scale = glm::max(vector_value(*node, "Radii", glm::vec3(1.0f)), glm::vec3(1e-6f));
            result.falloff = std::max(0.0f, float_value(*node, "Falloff"));
            editable_inputs = {"Centre", "Rotation", "Radii"};
        } else if (node->type_id == "lfs.transform_geometry") {
            result.kind = NodeViewportGizmoKind::Transform;
            transform.translation = vector_value(*node, "Translation");
            transform.rotation_degrees = vector_value(*node, "Rotation");
            transform.scale = glm::vec3(std::max(1e-6f, float_value(*node, "Scale", 1.0f)));
            editable_inputs = {"Translation", "Rotation", "Scale"};
        } else {
            return std::nullopt;
        }
        result.editable = std::ranges::none_of(editable_inputs, [&](const auto input) {
            return input_linked(*graph, node->name, input);
        });
        result.local_translation = transform.translation;
        result.local_rotation = transform.rotation_degrees;
        result.local_scale = transform.scale;
        result.local_transform = nodes::ViewportCoordinates::composeLocal(transform);
        const nodes::ViewportCoordinates coordinates(scene_manager_->getScene(), host->id);
        if (!coordinates.valid())
            return std::nullopt;
        result.world_transform = coordinates.transformToWorld(transform);
        return result;
    }

    bool ModifierManager::beginViewportNodeGizmoDrag() {
        const auto state = viewportNodeGizmo();
        auto* graph = state ? tree(state->tree_uuid) : nullptr;
        if (!state || !state->editable || !graph || gizmo_before_)
            return false;
        gizmo_before_ = graph->to_json();
        return true;
    }

    bool ModifierManager::updateViewportNodeGizmo(const glm::mat4& world_transform) {
        const auto state = viewportNodeGizmo();
        auto* graph = state ? tree(state->tree_uuid) : nullptr;
        auto* node = graph ? graph->find_node(state->node) : nullptr;
        if (!state || !state->editable || !node)
            return false;
        const nodes::ViewportCoordinates coordinates(scene_manager_->getScene(), state->host_id);
        if (!coordinates.valid())
            return false;
        const auto local = coordinates.transformToLocal(world_transform);
        if (state->kind == NodeViewportGizmoKind::Transform) {
            node->input_values["Translation"] = local.translation;
            node->input_values["Rotation"] = local.rotation_degrees;
            node->input_values["Scale"] = std::max(1e-6f, (std::abs(local.scale.x) +
                                                           std::abs(local.scale.y) +
                                                           std::abs(local.scale.z)) /
                                                              3.0f);
        } else {
            node->input_values["Centre"] = local.translation;
            node->input_values["Rotation"] = local.rotation_degrees;
            node->input_values[state->kind == NodeViewportGizmoKind::Box ? "Size" : "Radii"] =
                glm::max(glm::abs(local.scale), glm::vec3(1e-6f));
        }
        markDirty(state->host);
        return true;
    }

    void ModifierManager::endViewportNodeGizmoDrag(const bool cancel) {
        if (!gizmo_before_ || !viewport_selection_)
            return;
        auto* graph = tree(viewport_selection_->tree_uuid);
        if (!graph) {
            gizmo_before_.reset();
            return;
        }
        auto before = std::move(*gizmo_before_);
        gizmo_before_.reset();
        if (cancel) {
            try {
                const auto restored = lfs::nodes::NodeTree::from_json(before, registry_);
                const auto* source = restored.find_node(viewport_selection_->node);
                auto* destination = graph->find_node(viewport_selection_->node);
                if (source && destination)
                    destination->input_values = source->input_values;
                markDirty(viewport_selection_->host);
            } catch (const std::exception&) {
                // LFS-CENSUS-OK(empty-catch): cancellation is best-effort; the live graph stays valid.
            }
            return;
        }
        recordTreeEdit(graph->uuid, std::move(before));
    }

    bool ModifierManager::setPaintMode(const bool enabled) {
        if (!enabled) {
            if (paint_before_)
                endPaintStroke(false);
            paint_mode_ = false;
            return true;
        }
        if (!viewport_selection_)
            return false;
        const auto* graph = tree(viewport_selection_->tree_uuid);
        const auto* node = graph ? graph->find_node(viewport_selection_->node) : nullptr;
        if (!node || node->type_id != "lfs.paint_selection")
            return false;
        colour_pick_.reset();
        paint_mode_ = true;
        return true;
    }

    void ModifierManager::adjustPaintRadius(const float factor) {
        if (std::isfinite(factor) && factor > 0.0f)
            paint_radius_ = std::clamp(paint_radius_ * factor, 2.0f, 256.0f);
    }

    bool ModifierManager::beginPaintStroke() {
        if (!paint_mode_ || paint_before_ || !viewport_selection_)
            return false;
        auto* graph = tree(viewport_selection_->tree_uuid);
        auto* node = graph ? graph->find_node(viewport_selection_->node) : nullptr;
        if (!node || node->type_id != "lfs.paint_selection")
            return false;
        paint_before_ = graph->to_json();
        if (!node->properties.contains("data") || !node->properties["data"].is_array())
            node->properties["data"] = nlohmann::json::array();
        node->properties["data"].push_back(nlohmann::json::array());
        return true;
    }

    bool ModifierManager::appendPaintSample(const PaintStrokeSample& sample,
                                            const bool position_is_world) {
        if (!paint_before_ || !viewport_selection_ || !std::isfinite(sample.radius) ||
            sample.radius <= 0.0f)
            return false;
        auto* graph = tree(viewport_selection_->tree_uuid);
        auto* node = graph ? graph->find_node(viewport_selection_->node) : nullptr;
        const auto* host = scene_manager_->getScene().getNodeByUuid(viewport_selection_->host);
        if (!node || !host || node->type_id != "lfs.paint_selection")
            return false;
        glm::vec3 position = sample.position;
        float radius = sample.radius;
        if (position_is_world) {
            const nodes::ViewportCoordinates coordinates(scene_manager_->getScene(), host->id);
            if (!coordinates.valid())
                return false;
            position = coordinates.pointToLocal(position);
            radius = coordinates.radiusToLocal(radius);
        }
        auto& data = node->properties["data"];
        if (!data.is_array() || data.empty() || !data.back().is_array())
            return false;
        data.back().push_back({position.x, position.y, position.z, radius,
                               std::clamp(sample.value, 0.0f, 1.0f)});
        markDirty(viewport_selection_->host);
        return true;
    }

    void ModifierManager::endPaintStroke(const bool cancel) {
        if (!paint_before_ || !viewport_selection_)
            return;
        auto* graph = tree(viewport_selection_->tree_uuid);
        if (!graph) {
            paint_before_.reset();
            return;
        }
        auto before = std::move(*paint_before_);
        paint_before_.reset();
        if (cancel) {
            try {
                const auto restored = lfs::nodes::NodeTree::from_json(before, registry_);
                const auto* source = restored.find_node(viewport_selection_->node);
                auto* destination = graph->find_node(viewport_selection_->node);
                if (source && destination)
                    destination->properties = source->properties;
                markDirty(viewport_selection_->host);
            } catch (const std::exception&) {
                // LFS-CENSUS-OK(empty-catch): cancellation is best-effort and the live graph remains valid.
            }
            return;
        }
        auto* node = graph->find_node(viewport_selection_->node);
        if (node && node->properties.contains("data") && node->properties["data"].is_array() &&
            !node->properties["data"].empty() && node->properties["data"].back().empty())
            node->properties["data"].erase(node->properties["data"].end() - 1);
        recordTreeEdit(graph->uuid, std::move(before));
    }

    ModifierResult ModifierManager::addPaintStroke(const std::vector<PaintStrokeSample>& samples,
                                                   const bool positions_are_world) {
        if (!viewport_selection_)
            return std::unexpected(ModifierError{"No selected Paint Selection node"});
        auto* graph = tree(viewport_selection_->tree_uuid);
        auto* node = graph ? graph->find_node(viewport_selection_->node) : nullptr;
        const auto* host = scene_manager_->getScene().getNodeByUuid(viewport_selection_->host);
        if (!node || node->type_id != "lfs.paint_selection" || !host)
            return std::unexpected(ModifierError{"The selected node is not Paint Selection"});
        const auto before = graph->to_json();
        nlohmann::json stroke = nlohmann::json::array();
        std::optional<nodes::ViewportCoordinates> coordinates;
        float radius_scale = 1.0f;
        if (positions_are_world) {
            coordinates.emplace(scene_manager_->getScene(), host->id);
            if (!coordinates->valid())
                return std::unexpected(ModifierError{"The host transform is not invertible"});
            radius_scale = coordinates->radiusToLocal(1.0f);
        }
        for (const auto& sample : samples) {
            if (!std::isfinite(sample.position.x) || !std::isfinite(sample.position.y) ||
                !std::isfinite(sample.position.z) || !std::isfinite(sample.radius) ||
                !std::isfinite(sample.value) || sample.radius <= 0.0f)
                continue;
            const glm::vec3 position = coordinates ? coordinates->pointToLocal(sample.position) : sample.position;
            stroke.push_back({position.x, position.y, position.z, sample.radius * radius_scale,
                              std::clamp(sample.value, 0.0f, 1.0f)});
        }
        if (stroke.empty())
            return std::unexpected(ModifierError{"A stroke needs at least one finite sample with a positive radius"});
        if (!node->properties.contains("data") || !node->properties["data"].is_array())
            node->properties["data"] = nlohmann::json::array();
        node->properties["data"].push_back(std::move(stroke));
        recordTreeEdit(graph->uuid, before);
        return {};
    }

    ModifierResult ModifierManager::clearPaintStrokes() {
        if (!viewport_selection_)
            return std::unexpected(ModifierError{"No selected Paint Selection node"});
        auto* graph = tree(viewport_selection_->tree_uuid);
        auto* node = graph ? graph->find_node(viewport_selection_->node) : nullptr;
        if (!node || node->type_id != "lfs.paint_selection")
            return std::unexpected(ModifierError{"The selected node is not Paint Selection"});
        const auto before = graph->to_json();
        node->properties["data"] = nlohmann::json::array();
        recordTreeEdit(graph->uuid, before);
        return {};
    }

    bool ModifierManager::beginColourPick(std::string node, std::string input, const bool widen_hue) {
        if (!viewport_selection_ || viewport_selection_->node != node)
            return false;
        auto* graph = tree(viewport_selection_->tree_uuid);
        auto* selected = graph ? graph->find_node(node) : nullptr;
        if (!selected)
            return false;
        const bool hsv = selected->type_id == "lfs.hsv_range" && input == "Hue";
        const bool colour =
            (((selected->type_id == "lfs.colour_key" || selected->type_id == "lfs.recolour" ||
               selected->type_id == "lfs.set_colour" || selected->type_id == "lfs.colour") &&
              input == "Colour") ||
             (selected->type_id == "lfs.mix_colour" && (input == "A" || input == "B")));
        if (!hsv && !colour)
            return false;
        const auto type = registry_.find(selected->type_id);
        if (hsv && input_linked(*graph, node, input))
            return false;
        if (!hsv) {
            if (!type)
                return false;
            const auto socket = std::ranges::find(type->inputs, input,
                                                  &lfs::nodes::SocketDecl::identifier);
            if (socket == type->inputs.end() || socket->type != lfs::nodes::COLOUR_SOCKET ||
                input_linked(*graph, node, input))
                return false;
        }
        paint_mode_ = false;
        colour_pick_ = ColourPick{std::move(node), std::move(input), widen_hue};
        return true;
    }

    void ModifierManager::cancelViewportMode() {
        if (paint_before_)
            endPaintStroke(true);
        paint_mode_ = false;
        colour_pick_.reset();
    }

    ModifierResult ModifierManager::applyPickedColour(const glm::vec3 colour, const bool widen_hue) {
        if (!viewport_selection_ || !colour_pick_)
            return std::unexpected(ModifierError{"Colour pick mode is not active"});
        auto* graph = tree(viewport_selection_->tree_uuid);
        auto* node = graph ? graph->find_node(colour_pick_->node) : nullptr;
        if (!node)
            return std::unexpected(ModifierError{"The colour target node no longer exists"});
        const auto before = graph->to_json();
        if (node->type_id == "lfs.hsv_range" && colour_pick_->input == "Hue") {
            const float saturation_width = float_value(*node, "Saturation Max", 1.0f) -
                                           float_value(*node, "Saturation Min", 0.0f);
            const float value_width = float_value(*node, "Value Max", 1.0f) -
                                      float_value(*node, "Value Min", 0.0f);
            const auto picked = centreHsvPickBands(colour, saturation_width, value_width);
            if (widen_hue || colour_pick_->widen_hue) {
                const float current = float_value(*node, "Hue");
                const float difference = std::abs(current - picked.hue);
                node->input_values["Hue Range"] =
                    std::max(float_value(*node, "Hue Range"), std::min(difference, 1.0f - difference));
            } else {
                node->input_values["Hue"] = picked.hue;
                node->input_values["Saturation Min"] = picked.saturation_min;
                node->input_values["Saturation Max"] = picked.saturation_max;
                node->input_values["Value Min"] = picked.value_min;
                node->input_values["Value Max"] = picked.value_max;
            }
        } else {
            node->input_values[colour_pick_->input] = glm::vec4(glm::clamp(colour, glm::vec3(0), glm::vec3(1)), 1.0f);
        }
        colour_pick_.reset();
        recordTreeEdit(graph->uuid, before);
        return {};
    }

    ModifierResult ModifierManager::captureViewportCamera(std::string_view tree_uuid, std::string_view node_name) {
        auto* graph = tree(tree_uuid);
        auto* node = graph ? graph->find_node(node_name) : nullptr;
        const auto view = get_current_view_info();
        if (!node || node->type_id != "lfs.view_distance")
            return std::unexpected(ModifierError{"View Distance node no longer exists"});
        if (!view)
            return std::unexpected(ModifierError{"No viewport camera is available"});
        const auto before = graph->to_json();
        glm::mat3 rotation(1);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                rotation[c][r] = view->rotation[3 * r + c];
        const glm::vec3 eye(view->translation[0], view->translation[1], view->translation[2]);
        const auto pose = glm::inverse(rendering::dataWorldToCameraFromVisualizerPose(rotation, eye));
        auto data = nlohmann::json::array();
        for (int c = 0; c < 4; ++c)
            for (int r = 0; r < 4; ++r)
                data.push_back(pose[c][r]);
        node->properties["captured_transform"] = std::move(data);
        node->properties["source"] = "captured";
        recordTreeEdit(graph->uuid, before);
        return {};
    }

    void ModifierManager::registerVisualizerNodes() {
        registerCameraNodes(registry_);
        NodeTypeInfo object_info;
        object_info.id = "lfs.object_info";
        object_info.category = "Input";
        object_info.outputs = {SocketDecl{"Geometry", "Geometry", std::string(lfs::nodes::GEOMETRY_SOCKET)}};
        object_info.properties = {
            PropertyDecl{"object", "Object", PropertyKind::String, ""},
            PropertyDecl{"transform_space", "Transform Space", PropertyKind::Enum, "original", {"original", "relative"}}};
        object_info.uses_host = true;
        object_info.evaluate = [](NodeContext& context) {
            auto* host = context.host();
            if (!host)
                throw NodeError("Object Info requires a scene evaluation host");
            const auto object = context.properties().value("object", "");
            const auto transform_space = context.properties().value("transform_space", "original") == "relative"
                                             ? lfs::nodes::TransformSpace::Relative
                                             : lfs::nodes::TransformSpace::Original;
            auto geometry = host->object_geometry(object, transform_space);
            if (!geometry)
                throw NodeError("Object Info target produced no geometry");
            context.set_output("Geometry", std::move(*geometry));
        };
        lfs::nodes::set_builtin_node_text(object_info);
        registry_.register_type(std::move(object_info));
    }

} // namespace lfs::vis
