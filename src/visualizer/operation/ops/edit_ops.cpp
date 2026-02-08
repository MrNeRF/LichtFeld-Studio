/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "edit_ops.hpp"
#include "core/logger.hpp"
#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include "scene/scene_manager.hpp"

namespace lfs::vis::op {

    OperationResult EditDelete::execute(SceneManager& scene,
                                        const OperatorProperties& /*props*/,
                                        const std::any& /*input*/) {
        if (!scene.getScene().hasSelection()) {
            return OperationResult::failure("Nothing selected");
        }

        auto selection = scene.getScene().getSelectionMask();
        if (!selection) {
            return OperationResult::failure("No selection mask");
        }

        auto& sc = scene.getScene();
        auto nodes = sc.getVisibleNodes();
        if (nodes.empty()) {
            return OperationResult::failure("No visible nodes");
        }

        size_t offset = 0;
        bool any_deleted = false;

        for (const auto* node : nodes) {
            if (!node || !node->model) {
                continue;
            }

            const size_t node_size = node->model->size();
            if (node_size == 0) {
                continue;
            }

            auto node_selection = selection->slice(0, offset, offset + node_size);
            auto bool_mask = node_selection.to(lfs::core::DataType::Bool);

            auto* mutable_node = sc.getMutableNode(node->name);
            if (mutable_node && mutable_node->model) {
                mutable_node->model->soft_delete(bool_mask);
                any_deleted = true;
            }

            offset += node_size;
        }

        if (any_deleted) {
            sc.clearSelection();
            sc.invalidateCache();
        }

        return OperationResult::success();
    }

    bool EditDelete::poll(SceneManager& scene) const {
        return scene.getScene().hasSelection();
    }

    OperationResult EditDuplicate::execute(SceneManager& scene,
                                           const OperatorProperties& /*props*/,
                                           const std::any& /*input*/) {
        if (!scene.getScene().hasSelection()) {
            return OperationResult::failure("Nothing selected");
        }

        LOG_INFO("EditDuplicate: duplicating selected gaussians");

        return OperationResult::success();
    }

    bool EditDuplicate::poll(SceneManager& scene) const {
        return scene.getScene().hasSelection();
    }

} // namespace lfs::vis::op
