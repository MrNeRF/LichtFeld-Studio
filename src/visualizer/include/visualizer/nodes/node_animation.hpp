/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/nodes/nodes.hpp"
#include "sequencer/animation_clip.hpp"
#include "visualizer/nodes/modifier_error.hpp"
#include <expected>

namespace lfs::vis {
    class SequencerController;

    LFS_VIS_API std::string nodeInputTrackPath(std::string_view tree, std::string_view node,
                                               std::string_view input);
    LFS_VIS_API bool animatableNodeInput(const lfs::nodes::NodeTree& tree,
                                         const lfs::nodes::Node& node,
                                         const lfs::nodes::SocketDecl& socket);
    LFS_VIS_API lfs::nodes::Value nodeAnimationValue(const sequencer::AnimationValue& value);
    LFS_VIS_API ModifierResult setNodeKeyframe(
        const lfs::nodes::NodeTree& tree, std::string_view node, std::string_view input,
        SequencerController& controller, std::optional<float> time,
        std::optional<lfs::nodes::Value> value,
        sequencer::EasingType easing = sequencer::EasingType::LINEAR,
        const lfs::nodes::TreeResolver& resolver = {});
    LFS_VIS_API bool removeNodeKeyframe(SequencerController& controller, std::string_view path,
                                        std::optional<float> time);
    LFS_VIS_API void removeNodeAnimation(SequencerController& controller, std::string_view tree,
                                         std::optional<std::string_view> node = {});
    LFS_VIS_API void applyNodeAnimation(lfs::nodes::NodeTree& tree,
                                        const sequencer::AnimationClip* clip, float time);
    // Track payloads remain AnimationClip JSON. These subsets are for clipboard/undo,
    // not a second persistence or interpolation system.
    LFS_VIS_API nlohmann::json nodeAnimationJson(const sequencer::AnimationClip* clip);
    LFS_VIS_API void restoreNodeAnimation(SequencerController& controller, const nlohmann::json& json);
    LFS_VIS_API void copyNodeAnimation(SequencerController& controller, const nlohmann::json& json,
                                       std::string_view source_tree, std::string_view target_tree,
                                       const std::unordered_map<std::string, std::string>& nodes,
                                       bool move = false);
} // namespace lfs::vis
