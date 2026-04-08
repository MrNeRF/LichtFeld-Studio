/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "edit_ops.hpp"
#include "core/logger.hpp"
#include "scene/scene_manager.hpp"

namespace lfs::vis::op {

    OperationResult EditDelete::execute(SceneManager& scene,
                                        const OperatorProperties& /*props*/,
                                        const std::any& /*input*/) {
        if (const auto result = scene.softDeleteSelectedGaussians(); !result) {
            return OperationResult::failure(result.error());
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
