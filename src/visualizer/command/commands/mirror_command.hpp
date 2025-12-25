/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "command/command.hpp"
#include "core/splat_data_mirror.hpp"
#include "core/tensor_fwd.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <string>

namespace lfs::vis {
    class SceneManager;
}

namespace lfs::vis::command {

    /// Command for mirroring selected gaussians with undo/redo support
    class MirrorCommand final : public Command {
    public:
        MirrorCommand(SceneManager* scene_manager,
                      std::string node_name,
                      lfs::core::MirrorAxis axis,
                      glm::vec3 center,
                      std::shared_ptr<lfs::core::Tensor> selection_mask,
                      std::shared_ptr<lfs::core::Tensor> old_means,
                      std::shared_ptr<lfs::core::Tensor> old_rotation,
                      std::shared_ptr<lfs::core::Tensor> old_shN);

        void undo() override;
        void redo() override;
        [[nodiscard]] std::string getName() const override;

    private:
        void restoreState();
        void applyMirror();

        SceneManager* const scene_manager_;
        const std::string node_name_;
        const lfs::core::MirrorAxis axis_;
        const glm::vec3 center_;
        const std::shared_ptr<lfs::core::Tensor> selection_mask_;
        const std::shared_ptr<lfs::core::Tensor> old_means_;
        const std::shared_ptr<lfs::core::Tensor> old_rotation_;
        const std::shared_ptr<lfs::core::Tensor> old_shN_;
    };

} // namespace lfs::vis::command
