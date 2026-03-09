/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "undo_entry.hpp"
#include "core/logger.hpp"
#include "core/scene.hpp"
#include "scene/scene_manager.hpp"

namespace lfs::vis::op {

    namespace {
        bool cropBoxesEqual(const lfs::core::CropBoxData& lhs, const lfs::core::CropBoxData& rhs) {
            return lhs.min == rhs.min &&
                   lhs.max == rhs.max &&
                   lhs.inverse == rhs.inverse &&
                   lhs.enabled == rhs.enabled &&
                   lhs.color == rhs.color &&
                   lhs.line_width == rhs.line_width &&
                   lhs.flash_intensity == rhs.flash_intensity;
        }

        bool ellipsoidsEqual(const lfs::core::EllipsoidData& lhs, const lfs::core::EllipsoidData& rhs) {
            return lhs.radii == rhs.radii &&
                   lhs.inverse == rhs.inverse &&
                   lhs.enabled == rhs.enabled &&
                   lhs.color == rhs.color &&
                   lhs.line_width == rhs.line_width &&
                   lhs.flash_intensity == rhs.flash_intensity;
        }
    } // namespace

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

    bool SceneSnapshot::captureTransformsBefore(const std::vector<std::string>& nodes,
                                                const std::vector<glm::mat4>& transforms) {
        if (nodes.size() != transforms.size()) {
            LOG_ERROR("Cannot capture transform snapshot: {} node names but {} transforms",
                      nodes.size(), transforms.size());
            return false;
        }
        for (size_t i = 0; i < nodes.size(); ++i) {
            transforms_before_[nodes[i]] = transforms[i];
        }
        captured_ = captured_ | ModifiesFlag::TRANSFORMS;
        return true;
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
        if (hasFlag(captured_, ModifiesFlag::SELECTION)) {
            if (selection_before_) {
                auto mask = std::make_shared<lfs::core::Tensor>(selection_before_->clone());
                scene_.getScene().setSelectionMask(mask);
            } else {
                scene_.getScene().clearSelection();
            }
        }

        if (hasFlag(captured_, ModifiesFlag::TRANSFORMS)) {
            for (const auto& [node_name, transform] : transforms_before_) {
                scene_.setNodeTransform(node_name, transform);
            }
        }
    }

    void SceneSnapshot::redo() {
        if (hasFlag(captured_, ModifiesFlag::SELECTION)) {
            if (selection_after_) {
                auto mask = std::make_shared<lfs::core::Tensor>(selection_after_->clone());
                scene_.getScene().setSelectionMask(mask);
            } else {
                scene_.getScene().clearSelection();
            }
        }

        if (hasFlag(captured_, ModifiesFlag::TRANSFORMS)) {
            for (const auto& [node_name, transform] : transforms_after_) {
                scene_.setNodeTransform(node_name, transform);
            }
        }
    }

    CropBoxUndoEntry::CropBoxUndoEntry(SceneManager& scene, std::string node_name,
                                       lfs::core::CropBoxData before, glm::mat4 transform_before)
        : scene_(scene),
          node_name_(std::move(node_name)),
          before_(std::move(before)),
          transform_before_(transform_before) {
        captureAfter();
    }

    void CropBoxUndoEntry::captureAfter() {
        const auto* node = scene_.getScene().getNode(node_name_);
        assert(node && node->cropbox);
        after_ = *node->cropbox;
        transform_after_ = scene_.getNodeTransform(node_name_);
    }

    void CropBoxUndoEntry::undo() {
        auto* node = scene_.getScene().getMutableNode(node_name_);
        if (node && node->cropbox) {
            *node->cropbox = before_;
            scene_.setNodeTransform(node_name_, transform_before_);
        }
    }

    void CropBoxUndoEntry::redo() {
        auto* node = scene_.getScene().getMutableNode(node_name_);
        if (node && node->cropbox) {
            *node->cropbox = after_;
            scene_.setNodeTransform(node_name_, transform_after_);
        }
    }

    bool CropBoxUndoEntry::hasChanges() const {
        return !cropBoxesEqual(before_, after_) || transform_before_ != transform_after_;
    }

    EllipsoidUndoEntry::EllipsoidUndoEntry(SceneManager& scene, std::string node_name,
                                           lfs::core::EllipsoidData before, glm::mat4 transform_before)
        : scene_(scene),
          node_name_(std::move(node_name)),
          before_(std::move(before)),
          transform_before_(transform_before) {
        captureAfter();
    }

    void EllipsoidUndoEntry::captureAfter() {
        const auto* node = scene_.getScene().getNode(node_name_);
        assert(node && node->ellipsoid);
        after_ = *node->ellipsoid;
        transform_after_ = scene_.getNodeTransform(node_name_);
    }

    void EllipsoidUndoEntry::undo() {
        auto* node = scene_.getScene().getMutableNode(node_name_);
        if (node && node->ellipsoid) {
            *node->ellipsoid = before_;
            scene_.setNodeTransform(node_name_, transform_before_);
        }
    }

    void EllipsoidUndoEntry::redo() {
        auto* node = scene_.getScene().getMutableNode(node_name_);
        if (node && node->ellipsoid) {
            *node->ellipsoid = after_;
            scene_.setNodeTransform(node_name_, transform_after_);
        }
    }

    bool EllipsoidUndoEntry::hasChanges() const {
        return !ellipsoidsEqual(before_, after_) || transform_before_ != transform_after_;
    }

} // namespace lfs::vis::op
