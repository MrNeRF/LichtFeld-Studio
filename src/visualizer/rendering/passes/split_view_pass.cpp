/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "split_view_pass.hpp"
#include "../split_view_composition.hpp"
#include "core/logger.hpp"

namespace lfs::vis {

    bool SplitViewPass::shouldExecute(DirtyMask frame_dirty, const FrameContext& ctx) const {
        if (!splitViewEnabled(ctx.settings.split_view_mode))
            return false;
        return (frame_dirty & sensitivity()) != 0;
    }

    void SplitViewPass::execute(lfs::rendering::RenderingEngine& engine,
                                const FrameContext& ctx,
                                FrameResources& res) {
        auto composition = buildSplitViewCompositionPlan(ctx, res);
        if (!composition) {
            res.split_view_executed = false;
            return;
        }

        res.split_info = composition->toInfo();

        auto render_lock = acquireRenderLock(ctx);

        auto result = engine.renderSplitViewGpuFrame(composition->toRequest());
        render_lock.reset();

        if (result) {
            res.cached_metadata = makeCachedRenderMetadata(result->metadata);
            res.cached_gpu_frame = result->frame;
            res.cached_result_size = ctx.render_size;
            res.split_view_executed = true;
        } else {
            LOG_ERROR("Failed to render split view: {}", result.error());
            res.cached_metadata = {};
            res.cached_gpu_frame.reset();
            res.cached_result_size = {0, 0};
        }
    }

} // namespace lfs::vis
