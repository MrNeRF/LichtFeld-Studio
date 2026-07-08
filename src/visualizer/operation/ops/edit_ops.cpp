/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "edit_ops.hpp"
#include "core/logger.hpp"
#include "core/selection_domain.hpp"
#include "scene/scene_manager.hpp"
#include <format>

namespace lfs::vis::op {

    OperationResult EditDelete::execute(SceneManager& scene,
                                        const OperatorProperties& /*props*/,
                                        const std::any& /*input*/) {
        std::expected<void, std::string> result;
        switch (resolveSelectionDomain(scene)) {
        case SelectionDomain::Gaussians:
            result = scene.deleteSelectedGaussiansWithHistory();
            break;
        case SelectionDomain::PointCloud:
            result = scene.deleteSelectedPointsWithHistory();
            break;
        case SelectionDomain::Cameras: {
            const auto selected_names = scene.getSelectedNodeNames();
            auto camera_result = scene.setCamerasTrainingEnabledWithHistory(selected_names, false);
            if (!camera_result) {
                return OperationResult::failure(camera_result.error());
            }
            const auto message = std::format("Disabled {} cameras for training", *camera_result);
            LOG_INFO("{}", message);
            return OperationResult::success(message);
        }
        }

        if (!result) {
            return OperationResult::failure(result.error());
        }

        return OperationResult::success();
    }

    bool EditDelete::poll(SceneManager& scene) const {
        switch (resolveSelectionDomain(scene)) {
        case SelectionDomain::Gaussians:
            return scene.getScene().hasSelection();
        case SelectionDomain::PointCloud:
            return scene.getActivePointSelectionCount() > 0;
        case SelectionDomain::Cameras:
            for (const auto& name : scene.getSelectedNodeNames()) {
                const auto* const node = scene.getScene().getNode(name);
                if (node && (node->type == lfs::core::NodeType::CAMERA ||
                             node->type == lfs::core::NodeType::CAMERA_GROUP)) {
                    return true;
                }
            }
            return false;
        }
        return false;
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
