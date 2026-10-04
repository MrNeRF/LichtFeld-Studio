/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/services.hpp"
#include "gui/gui_manager.hpp"
#include "scene/scene_manager.hpp"
#include "sequencer/sequencer_controller.hpp"
#include "visualizer/nodes/modifier_manager.hpp"
#include "visualizer/nodes/node_animation.hpp"
#include <cmath>

namespace lfs::vis {
    SequencerController* ModifierManager::sequencer() const {
        if (sequencer_)
            return sequencer_;
        auto* gui = services().guiOrNull();
        return gui ? &gui->sequencer() : nullptr;
    }

    float ModifierManager::animationTime() const {
        const auto* controller = sequencer();
        return export_time_.value_or(controller ? controller->playhead() : 0.0f);
    }

    nlohmann::json ModifierManager::animationJson() const {
        const auto* controller = sequencer();
        return nodeAnimationJson(controller ? controller->timeline().animationClip() : nullptr);
    }

    ModifierResult ModifierManager::keyframeSet(std::string_view uuid, std::string_view node,
                                                std::string_view input, std::optional<float> time,
                                                std::optional<lfs::nodes::Value> value, int easing) {
        const auto* graph = tree(uuid);
        auto* controller = sequencer();
        if (!graph || !controller)
            return std::unexpected(ModifierError{"Node graph or sequencer is unavailable"});
        const auto before = toJson(false);
        const auto result = setNodeKeyframe(*graph, node, input, *controller, time, value,
                                            static_cast<sequencer::EasingType>(easing),
                                            [this](std::string_view id) { return tree(id); });
        if (!result)
            return result;
        recordLibraryEdit(before);
        return {};
    }

    ModifierResult ModifierManager::keyframeRemove(std::string_view uuid, std::string_view node,
                                                   std::string_view input, std::optional<float> time) {
        const auto* graph = tree(uuid);
        auto* controller = sequencer();
        if (!graph || !controller)
            return std::unexpected(ModifierError{"Node graph or sequencer is unavailable"});
        const auto before = toJson(false);
        if (!removeNodeKeyframe(*controller, nodeInputTrackPath(graph->uuid, node, input), time))
            return std::unexpected(ModifierError{"No keyframe exists at this time"});
        recordLibraryEdit(before);
        return {};
    }

    ModifierResult ModifierManager::renameNode(std::string_view uuid, std::string_view node, std::string name) {
        auto* graph = tree(uuid);
        if (!graph)
            return std::unexpected(ModifierError{"Node graph does not exist"});
        const auto before = toJson(false);
        const std::string old_name(node);
        if (!graph->rename_node(old_name, name))
            return std::unexpected(ModifierError{"Node name must be non-empty and unique"});
        if (auto* controller = sequencer())
            copyNodeAnimation(*controller, before.at("animation"), graph->uuid, graph->uuid,
                              {{old_name, name}}, true);
        for (auto& [_, stack] : stacks_)
            for (auto& modifier : stack.modifiers)
                if (modifier.tree_uuid == graph->uuid)
                    if (auto selection = modifier.stored_selections.extract(old_name); !selection.empty()) {
                        selection.key() = name;
                        modifier.stored_selections.insert(std::move(selection));
                    }
        recordLibraryEdit(before);
        return {};
    }

    bool ModifierManager::timeDependent(const core::Uuid& host) const {
        const auto* controller = sequencer();
        const auto* clip = controller ? controller->timeline().animationClip() : nullptr;
        std::unordered_set<std::string> visiting;
        std::function<bool(const lfs::nodes::NodeTree&, const lfs::nodes::Node&)> visit;
        std::function<bool(const core::Uuid&)> visit_stack;
        visit = [&](const auto& graph, const auto& node) {
            if (!visiting.insert(graph.uuid + "/" + node.name).second)
                return false;
            if (!node.muted && node.type_id == "lfs.scene_time")
                return true;
            if (!node.muted && clip)
                for (const auto& [input, _] : node.input_values)
                    if (const auto* track = clip->getTrackByPath(nodeInputTrackPath(graph.uuid, node.name, input));
                        track && track->keyframeCount() &&
                        std::ranges::none_of(graph.links, [&](const auto& link) {
                            return link.to_node == node.name && link.to_socket == input;
                        }))
                        return true;
            if (!node.muted && node.type_id == "lfs.group")
                if (const auto* nested = tree(node.properties.value("tree", "")); nested && visit(*nested, nested->output_node()))
                    return true;
            if (!node.muted && node.type_id == "lfs.object_info") {
                const auto& scene = scene_manager_->getScene();
                const auto id = scene.getNodeIdByName(node.properties.value("object", ""));
                if (const auto* source = scene.getNodeById(id); source && visit_stack(source->uuid))
                    return true;
            }
            for (const auto& link : graph.links)
                if (link.to_node == node.name)
                    if (const auto* source = graph.find_node(link.from_node); source && visit(graph, *source))
                        return true;
            return false;
        };
        visit_stack = [&](const core::Uuid& uuid) {
            const auto* value = stack(uuid);
            if (!value)
                return false;
            for (const auto& modifier : value->modifiers)
                if (modifier.enabled && modifier.show_viewport)
                    if (const auto* graph = tree(modifier.tree_uuid); graph && visit(*graph, graph->output_node()))
                        return true;
            return false;
        };
        if (!host.is_nil())
            return visit_stack(host);
        for (const auto& [uuid, _] : stacks_)
            if (visit_stack(uuid))
                return true;
        return false;
    }

    void ModifierManager::updateAnimationTime() {
        const auto* controller = sequencer();
        const float time = animationTime();
        const float fps = export_time_ ? export_fps_ : controller ? controller->framesPerSecond()
                                                                  : 24.0f;
        const auto revision = controller ? controller->timelineRevision() : 0;
        if (time == last_animation_time_ && fps == last_animation_fps_ && revision == last_animation_revision_)
            return;
        last_animation_time_ = time;
        last_animation_fps_ = fps;
        last_animation_revision_ = revision;
        const bool animated = timeDependent();
        if (animated) {
            // A scrub may replace pending animated work, but must not discard
            // pending ordinary graph/scene edits on unrelated static hosts.
            const bool animation_only = animation_only_request_ ||
                (requested_generation_ == last_installed_generation_ &&
                 std::ranges::none_of(runtime_, [](const auto& item) { return item.second.dirty; }));
            markDirty();
            animation_only_request_ = animation_only;
        } else if (last_time_dependent_) {
            // Clearing the sequencer's last track restores the base inputs.
            markDirty();
        } else
            ++generation_; // Refresh displayed values without submitting tensor work.
        last_time_dependent_ = animated;
    }

    ModifierResult ModifierManager::evaluateAtTime(const float seconds, const float fps) {
        if (!std::isfinite(seconds) || !std::isfinite(fps) || fps <= 0)
            return std::unexpected(ModifierError{"Export time and frame rate must be finite and valid"});
        export_time_ = seconds;
        export_fps_ = fps;
        updateAnimationTime();
        for (const auto& [host, _] : stacks_) {
            const auto result = evaluate(host);
            if (!result.ok) {
                export_time_.reset();
                return std::unexpected(ModifierError{"Node evaluation failed at export time " + std::to_string(seconds) +
                                                     (result.errors.empty() ? "" : ": " + result.errors.begin()->second)});
            }
        }
        export_time_.reset();
        return {};
    }
} // namespace lfs::vis
