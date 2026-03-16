/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "present_pass.hpp"
#include "core/logger.hpp"
#include <glad/glad.h>

namespace lfs::vis {

    void PresentPass::execute(lfs::rendering::RenderingEngine& engine,
                              const FrameContext& ctx,
                              FrameResources& res) {
        const bool has_gpu_frame = res.cached_gpu_frame && res.cached_gpu_frame->valid();
        const bool has_legacy_result = res.cached_result.image &&
                                       res.cached_result_size.x > 0 &&
                                       res.cached_result_size.y > 0;

        if (res.split_view_executed && !has_gpu_frame) {
            return;
        }

        if (!has_gpu_frame && !has_legacy_result) {
            return;
        }

        if (res.splats_presented)
            return;

        glViewport(ctx.viewport_pos.x, ctx.viewport_pos.y, ctx.render_size.x, ctx.render_size.y);
        glClearColor(ctx.settings.background_color.r, ctx.settings.background_color.g,
                     ctx.settings.background_color.b, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        auto present_result = has_gpu_frame
                                  ? engine.presentGpuFrame(*res.cached_gpu_frame,
                                                           ctx.viewport_pos,
                                                           res.cached_result_size)
                                  : engine.presentToScreen(res.cached_result,
                                                           ctx.viewport_pos,
                                                           res.cached_result_size);
        if (present_result) {
            res.splats_presented = true;
        } else {
            LOG_ERROR("Failed to present render result: {}", present_result.error());
        }
    }

} // namespace lfs::vis
