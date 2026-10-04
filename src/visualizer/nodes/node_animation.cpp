/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "visualizer/nodes/node_animation.hpp"
#include "sequencer/sequencer_controller.hpp"
#include "visualizer/nodes/modifier_manager.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace lfs::vis {
    using namespace lfs::nodes;

    std::string nodeInputTrackPath(std::string_view tree, std::string_view node, std::string_view input) {
        return "nodes/" + std::string(tree) + "/" + std::string(node) + "/" + std::string(input);
    }

    bool animatableNodeInput(const NodeTree& tree, const Node& node, const SocketDecl& socket) {
        return (socket.type == FLOAT_SOCKET || socket.type == INT_SOCKET ||
                socket.type == VECTOR_SOCKET || socket.type == COLOUR_SOCKET) &&
               std::ranges::none_of(tree.links, [&](const Link& link) {
                   return link.to_node == node.name && link.to_socket == socket.identifier;
               });
    }

    Value nodeAnimationValue(const sequencer::AnimationValue& value) {
        return std::visit([](const auto& typed) -> Value {
            using T = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<T, int>)
                return std::int64_t(typed);
            else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, glm::vec3> ||
                               std::is_same_v<T, glm::vec4> || std::is_same_v<T, bool>)
                return typed;
            else
                return {};
        },
                          value);
    }

    ModifierResult setNodeKeyframe(
        const NodeTree& tree, std::string_view name, std::string_view input,
        SequencerController& controller, std::optional<float> time, std::optional<Value> value,
        sequencer::EasingType easing, const TreeResolver& resolver) {
        const auto* node = tree.find_node(name);
        if (!node)
            return std::unexpected(ModifierError{"Node does not exist"});
        const auto sockets = effective_inputs(tree, *node, resolver);
        const auto socket = std::ranges::find(sockets, input, &SocketDecl::identifier);
        if (socket == sockets.end() || !animatableNodeInput(tree, *node, *socket))
            return std::unexpected(ModifierError{"Keyframes require an unlinked number, vector or colour input"});
        const float seconds = time.value_or(controller.playhead());
        if (!std::isfinite(seconds) || seconds < 0 || seconds > sequencer::MAX_SEQUENCER_TIME_SECONDS ||
            !sequencer::isValidEasingType(easing))
            return std::unexpected(ModifierError{"Invalid keyframe time or easing"});
        auto& clip = controller.timeline().ensureAnimationClip();
        const auto path = nodeInputTrackPath(tree.uuid, name, input);
        auto* track = clip.getTrackByPath(path);
        if (!value) {
            if (track) {
                if (const auto evaluated = track->evaluate(seconds))
                    value = nodeAnimationValue(*evaluated);
            }
            if (!value) {
                const auto own = node->input_values.find(std::string(input));
                value = own == node->input_values.end() ? socket->default_value : own->second;
            }
        }
        std::optional<sequencer::AnimationValue> converted;
        if (socket->type == VECTOR_SOCKET || socket->type == COLOUR_SOCKET) {
            if (const auto* v = value->get_if<glm::vec4>())
                converted = *v;
            else if (const auto* v = value->get_if<glm::vec3>())
                converted = *v;
            else if (const auto* v = value->get_if<float>())
                converted = glm::vec3(*v);
            else if (const auto* v = value->get_if<std::int64_t>())
                converted = glm::vec3(*v);
        } else {
            std::optional<double> number;
            if (const auto* v = value->get_if<float>())
                number = *v;
            else if (const auto* v = value->get_if<std::int64_t>())
                number = *v;
            if (number && std::isfinite(*number) && std::abs(*number) <= sequencer::MAX_ANIMATION_VALUE_MAGNITUDE) {
                if (socket->type == INT_SOCKET) {
                    if (*number >= std::numeric_limits<int>::min() && *number <= std::numeric_limits<int>::max())
                        converted = static_cast<int>(*number);
                } else
                    converted = static_cast<float>(*number);
            }
        }
        if (!converted)
            return std::unexpected(ModifierError{"Keyframe value does not match the input type"});
        if (socket->type == COLOUR_SOCKET)
            if (const auto* v = std::get_if<glm::vec3>(&*converted))
                converted = glm::vec4(*v, 1.0f);
        if (socket->type == VECTOR_SOCKET)
            if (const auto* v = std::get_if<glm::vec4>(&*converted))
                converted = glm::vec3(*v);
        if (const auto* v = std::get_if<glm::vec3>(&*converted))
            for (int axis = 0; axis < 3; ++axis)
                if (!std::isfinite((*v)[axis]) || std::abs((*v)[axis]) > sequencer::MAX_ANIMATION_VALUE_MAGNITUDE)
                    return std::unexpected(ModifierError{"Keyframe components must be bounded and finite"});
        if (const auto* v = std::get_if<glm::vec4>(&*converted))
            for (int axis = 0; axis < 4; ++axis)
                if (!std::isfinite((*v)[axis]) || std::abs((*v)[axis]) > sequencer::MAX_ANIMATION_VALUE_MAGNITUDE)
                    return std::unexpected(ModifierError{"Keyframe components must be bounded and finite"});
        if (track && track->valueType() != sequencer::getValueType(*converted))
            return std::unexpected(ModifierError{"Animation track type differs from the input type"});
        if (!track)
            track = clip.getTrack(clip.addTrack(sequencer::getValueType(*converted), path));
        track->addKeyframe(seconds, *converted, easing);
        controller.animationTracksChanged();
        return {};
    }

    bool removeNodeKeyframe(SequencerController& controller, std::string_view path,
                            std::optional<float> time) {
        auto* clip = controller.timeline().animationClip();
        auto* track = clip ? clip->getTrackByPath(std::string(path)) : nullptr;
        if (!track)
            return false;
        const float seconds = time.value_or(controller.playhead());
        for (size_t i = 0; i < track->keyframeCount(); ++i) {
            if (std::abs(track->keyframe(i).time - seconds) >= 1e-4f)
                continue;
            track->removeKeyframe(i);
            if (!track->keyframeCount())
                clip->removeTrack(track->id());
            controller.animationTracksChanged();
            return true;
        }
        return false;
    }

    void removeNodeAnimation(SequencerController& controller, std::string_view tree,
                             std::optional<std::string_view> node) {
        auto* clip = controller.timeline().animationClip();
        if (!clip)
            return;
        const auto prefix = node ? nodeInputTrackPath(tree, *node, "") : "nodes/" + std::string(tree) + "/";
        bool changed = false;
        for (const auto id : clip->trackIds())
            if (clip->getTrack(id)->targetPath().starts_with(prefix)) {
                clip->removeTrack(id);
                changed = true;
            }
        if (changed)
            controller.animationTracksChanged();
    }

    void applyNodeAnimation(NodeTree& tree, const sequencer::AnimationClip* clip, const float time) {
        if (!clip)
            return;
        for (auto& node : tree.nodes) {
            for (auto& [input, value] : node.input_values) {
                if (std::ranges::any_of(tree.links, [&](const Link& link) {
                        return link.to_node == node.name && link.to_socket == input;
                    }))
                    continue;
                const auto* track = clip->getTrackByPath(nodeInputTrackPath(tree.uuid, node.name, input));
                if (track)
                    if (auto evaluated = track->evaluate(time))
                        value = nodeAnimationValue(*evaluated);
            }
        }
    }

    nlohmann::json nodeAnimationJson(const sequencer::AnimationClip* clip) {
        auto json = clip ? clip->toJson() : sequencer::AnimationClip{}.toJson();
        auto& tracks = json["tracks"];
        tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [](const auto& track) {
                         return !track.at("target").template get<std::string>().starts_with("nodes/");
                     }),
                     tracks.end());
        return json;
    }

    void restoreNodeAnimation(SequencerController& controller, const nlohmann::json& json) {
        const auto source = sequencer::AnimationClip::fromJson(json);
        auto& clip = controller.timeline().ensureAnimationClip();
        for (const auto id : clip.trackIds())
            if (clip.getTrack(id)->targetPath().starts_with("nodes/"))
                clip.removeTrack(id);
        for (const auto id : source.trackIds()) {
            const auto* track = source.getTrack(id);
            if (!track->targetPath().starts_with("nodes/"))
                continue;
            auto* target = clip.getTrack(clip.addTrack(track->valueType(), track->targetPath()));
            for (const auto& key : track->keyframes())
                target->addKeyframe(key.time, key.value, key.easing);
        }
        controller.animationTracksChanged();
    }

    void copyNodeAnimation(SequencerController& controller, const nlohmann::json& json,
                           std::string_view source_tree, std::string_view target_tree,
                           const std::unordered_map<std::string, std::string>& nodes, const bool move) {
        const auto source = sequencer::AnimationClip::fromJson(json);
        auto& clip = controller.timeline().ensureAnimationClip();
        bool changed = false;
        for (const auto id : source.trackIds()) {
            const auto* track = source.getTrack(id);
            for (const auto& [old_name, new_name] : nodes) {
                const auto prefix = nodeInputTrackPath(source_tree, old_name, "");
                if (!track->targetPath().starts_with(prefix))
                    continue;
                const auto path = nodeInputTrackPath(target_tree, new_name, track->targetPath().substr(prefix.size()));
                if (path == track->targetPath() && clip.getTrackByPath(path))
                    break;
                if (move) {
                    if (const auto* current = clip.getTrackByPath(track->targetPath()))
                        clip.renameTrack(current->id(), path);
                } else {
                    auto* target = clip.getTrack(clip.addTrack(track->valueType(), path));
                    for (const auto& key : track->keyframes())
                        target->addKeyframe(key.time, key.value, key.easing);
                }
                changed = true;
                break;
            }
        }
        if (changed)
            controller.animationTracksChanged();
    }
} // namespace lfs::vis
