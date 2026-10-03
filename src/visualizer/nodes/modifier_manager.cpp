/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/nodes/modifier_manager.hpp"
#include "modifier_evaluation_worker.hpp"
#include "visualizer/nodes/viewport_coordinates.hpp"

#include "core/nodes/builtin.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor_backend.hpp"
#include "operation/undo_entry.hpp"
#include "operation/undo_history.hpp"
#include "scene/scene_manager.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
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
        cancelViewportMode();
        viewport_selection_.reset();
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
        if (const auto* modifiers = stack(node_uuid);
            modifiers && std::ranges::none_of(modifiers->modifiers, [](const Modifier& modifier) {
                return modifier.enabled && modifier.show_viewport;
            }))
            scene_manager_->getScene().clearNodeEvaluatedPayload(node_uuid);
        markDirty(node_uuid);
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

    void ModifierManager::registerVisualizerNodes() {
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
