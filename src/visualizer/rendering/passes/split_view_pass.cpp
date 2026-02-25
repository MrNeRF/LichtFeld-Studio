/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "split_view_pass.hpp"
#include "../rendering_manager.hpp"
#include "core/logger.hpp"
#include "scene/scene_manager.hpp"
#include "training/training_manager.hpp"
#include <shared_mutex>

namespace lfs::vis {

    bool SplitViewPass::shouldExecute(DirtyMask frame_dirty, const FrameContext& ctx) const {
        if (ctx.settings.split_view_mode == SplitViewMode::Disabled)
            return false;
        return (frame_dirty & sensitivity()) != 0;
    }

    void SplitViewPass::execute(lfs::rendering::RenderingEngine& engine,
                                const FrameContext& ctx,
                                FrameResources& res) {
        auto split_request = mgr_.createSplitViewRequest(ctx.ctx, ctx.scene_manager);
        if (!split_request) {
            res.split_view_executed = false;
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mgr_.split_info_mutex_);
            mgr_.current_split_info_.enabled = true;
            if (split_request->panels.size() >= 2) {
                mgr_.current_split_info_.left_name = split_request->panels[0].label;
                mgr_.current_split_info_.right_name = split_request->panels[1].label;
            }
        }

        std::optional<std::shared_lock<std::shared_mutex>> render_lock;
        if (const auto* tm = ctx.scene_manager ? ctx.scene_manager->getTrainerManager() : nullptr) {
            if (const auto* trainer = tm->getTrainer()) {
                render_lock.emplace(trainer->getRenderMutex());
            }
        }

        auto result = engine.renderSplitView(*split_request);
        render_lock.reset();

        if (result) {
            res.cached_result = *result;
            res.cached_result_size = ctx.render_size;
        } else {
            LOG_ERROR("Failed to render split view: {}", result.error());
            res.cached_result_size = {0, 0};
        }

        res.split_view_executed = true;
    }

} // namespace lfs::vis
