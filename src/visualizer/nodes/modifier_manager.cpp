/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/nodes/modifier_manager.hpp"
#include "modifier_evaluation_worker.hpp"

#include "core/nodes/builtin.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor_backend.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "scene/scene_manager.hpp"

#include <algorithm>
#include <chrono>
#include <format>
#include <glm/gtc/matrix_transform.hpp>
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
    } // namespace

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

    bool ModifierManager::removeTree(std::string_view uuid_or_name) {
        const auto* value = tree(uuid_or_name);
        if (!value)
            return false;
        const std::string uuid = value->uuid;
        const auto before = toJson(false);
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
        const auto type = registry_.find(node->type_id);
        if (!type)
            return std::unexpected(ModifierError{std::format("Node type '{}' is not registered", node->type_id)});
        const auto socket = std::ranges::find(type->inputs, input, &lfs::nodes::SocketDecl::identifier);
        if (socket == type->inputs.end())
            return std::unexpected(ModifierError{std::format("Input '{}' does not exist on node '{}' ({})", input, node_name, node->type_id)});
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
        if (node->evaluated_model && node->model && node->evaluated_model->size() != node->model->size())
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

        const auto manager_before = toJson(false);
        op::SceneGraphCaptureOptions options;
        options.mode = op::SceneGraphCaptureMode::FULL;
        options.include_selected_nodes = true;
        options.include_scene_context = true;
        options.payload_uuids = {node_uuid};
        auto scene_before = op::SceneGraphPatchEntry::captureStateByIds(
            *scene_manager_, {node->id}, options);
        op::TransactionGuard transaction("node.apply_modifier");
        if (node->type == core::NodeType::SPLAT) {
            scene.replaceNodeModel(node->name,
                                   std::make_unique<core::SplatData>(prepared.splats->readOnlySnapshot()));
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
        for (const auto& [_, value] : trees_)
            result["trees"].push_back(value->to_json());
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
        const bool output_changed = evaluation_tree(after_tree) != evaluation_tree(before);
        auto after = toJson(false);
        auto before_state = replace_tree_state(after, tree_uuid, std::move(before));
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
        markDirty(node_uuid);
    }

    std::uint64_t ModifierManager::generation() const {
        return generation_;
    }

    void ModifierManager::registerVisualizerNodes() {
        NodeTypeInfo object_info;
        object_info.id = "lfs.object_info";
        object_info.label = "Object Info";
        object_info.category = "Input";
        object_info.description = "Read effective geometry from another scene node.";
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
        registry_.register_type(std::move(object_info));
    }

} // namespace lfs::vis
