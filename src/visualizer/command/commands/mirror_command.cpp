/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mirror_command.hpp"
#include "core/splat_data.hpp"
#include "scene/scene_manager.hpp"

namespace lfs::vis::command {

    MirrorCommand::MirrorCommand(SceneManager* scene_manager,
                                 std::string node_name,
                                 const lfs::core::MirrorAxis axis,
                                 const glm::vec3 center,
                                 std::shared_ptr<lfs::core::Tensor> selection_mask,
                                 std::shared_ptr<lfs::core::Tensor> old_means,
                                 std::shared_ptr<lfs::core::Tensor> old_rotation,
                                 std::shared_ptr<lfs::core::Tensor> old_shN)
        : scene_manager_(scene_manager),
          node_name_(std::move(node_name)),
          axis_(axis),
          center_(center),
          selection_mask_(std::move(selection_mask)),
          old_means_(std::move(old_means)),
          old_rotation_(std::move(old_rotation)),
          old_shN_(std::move(old_shN)) {}

    void MirrorCommand::restoreState() {
        if (!scene_manager_)
            return;

        auto* node = scene_manager_->getScene().getMutableNode(node_name_);
        if (!node || !node->model)
            return;

        auto& model = *node->model;

        if (old_means_ && old_means_->is_valid()) {
            model.means().copy_from(*old_means_);
        }
        if (old_rotation_ && old_rotation_->is_valid()) {
            model.rotation_raw().copy_from(*old_rotation_);
        }
        if (old_shN_ && old_shN_->is_valid() && model.shN().is_valid()) {
            model.shN().copy_from(*old_shN_);
        }
    }

    void MirrorCommand::applyMirror() {
        if (!scene_manager_ || !selection_mask_ || !selection_mask_->is_valid())
            return;

        auto* node = scene_manager_->getScene().getMutableNode(node_name_);
        if (!node || !node->model)
            return;

        lfs::core::mirror_gaussians(*node->model, *selection_mask_, axis_, center_);
    }

    void MirrorCommand::undo() { restoreState(); }
    void MirrorCommand::redo() { applyMirror(); }

    std::string MirrorCommand::getName() const {
        switch (axis_) {
        case lfs::core::MirrorAxis::X: return "Mirror X";
        case lfs::core::MirrorAxis::Y: return "Mirror Y";
        case lfs::core::MirrorAxis::Z: return "Mirror Z";
        }
        return "Mirror";
    }

} // namespace lfs::vis::command
