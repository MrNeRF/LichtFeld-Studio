/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "undo_entry.hpp"
#include "core/logger.hpp"
#include "core/scene.hpp"
#include "scene/scene_manager.hpp"

namespace lfs::vis::op {

    SceneSnapshot::SceneSnapshot(SceneManager& scene, std::string name)
        : scene_(scene),
          name_(std::move(name)) {}

    void SceneSnapshot::captureSelection() {
        auto mask = scene_.getScene().getSelectionMask();
        if (mask) {
            selection_before_ = std::make_shared<lfs::core::Tensor>(mask->clone());
        }
        captured_ = captured_ | ModifiesFlag::SELECTION;
    }

    void SceneSnapshot::captureTransforms(const std::vector<std::string>& nodes) {
        for (const auto& node_name : nodes) {
            transforms_before_[node_name] = scene_.getNodeTransform(node_name);
        }
        captured_ = captured_ | ModifiesFlag::TRANSFORMS;
    }

    void SceneSnapshot::captureTopology() {
        captured_ = captured_ | ModifiesFlag::TOPOLOGY;
    }

    void SceneSnapshot::captureAfter() {
        if (hasFlag(captured_, ModifiesFlag::SELECTION)) {
            auto mask = scene_.getScene().getSelectionMask();
            if (mask) {
                selection_after_ = std::make_shared<lfs::core::Tensor>(mask->clone());
            }
        }

        if (hasFlag(captured_, ModifiesFlag::TRANSFORMS)) {
            for (const auto& [node_name, _] : transforms_before_) {
                transforms_after_[node_name] = scene_.getNodeTransform(node_name);
            }
        }
    }

    void SceneSnapshot::undo() {
        if (hasFlag(captured_, ModifiesFlag::SELECTION) && selection_before_) {
            auto mask = std::make_shared<lfs::core::Tensor>(selection_before_->clone());
            scene_.getScene().setSelectionMask(mask);
        }

        if (hasFlag(captured_, ModifiesFlag::TRANSFORMS)) {
            for (const auto& [node_name, transform] : transforms_before_) {
                scene_.setNodeTransform(node_name, transform);
            }
        }
    }

    void SceneSnapshot::redo() {
        if (hasFlag(captured_, ModifiesFlag::SELECTION) && selection_after_) {
            auto mask = std::make_shared<lfs::core::Tensor>(selection_after_->clone());
            scene_.getScene().setSelectionMask(mask);
        }

        if (hasFlag(captured_, ModifiesFlag::TRANSFORMS)) {
            for (const auto& [node_name, transform] : transforms_after_) {
                scene_.setNodeTransform(node_name, transform);
            }
        }
    }

} // namespace lfs::vis::op
