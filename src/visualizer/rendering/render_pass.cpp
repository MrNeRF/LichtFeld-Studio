/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "render_pass.hpp"
#include "scene/scene_manager.hpp"
#include "training/training_manager.hpp"

namespace lfs::vis {

    std::optional<std::shared_lock<std::shared_mutex>> acquireRenderLock(const FrameContext& ctx) {
        std::optional<std::shared_lock<std::shared_mutex>> lock;
        if (const auto* tm = ctx.scene_manager ? ctx.scene_manager->getTrainerManager() : nullptr)
            if (const auto* trainer = tm->getTrainer())
                lock.emplace(trainer->getRenderMutex());
        return lock;
    }

} // namespace lfs::vis
