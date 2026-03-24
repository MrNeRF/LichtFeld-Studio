/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "../render_pass.hpp"
#include "core/export.hpp"
#include <glm/glm.hpp>

namespace lfs::vis {

    class LFS_VIS_API SplatRasterPass final : public RenderPass {
    public:
        SplatRasterPass() = default;

        [[nodiscard]] const char* name() const override { return "SplatRasterPass"; }
        [[nodiscard]] DirtyMask sensitivity() const override {
            return DirtyFlag::SPLATS | DirtyFlag::SELECTION | DirtyFlag::CAMERA |
                   DirtyFlag::SPLIT_VIEW |
                   DirtyFlag::VIEWPORT | DirtyFlag::BACKGROUND | DirtyFlag::PPISP;
        }

        [[nodiscard]] bool shouldExecute(DirtyMask frame_dirty, const FrameContext& ctx) const override;

        void execute(lfs::rendering::RenderingEngine& engine,
                     const FrameContext& ctx,
                     FrameResources& res) override;

    private:
        void renderToTexture(lfs::rendering::RenderingEngine& engine,
                             const FrameContext& ctx, FrameResources& res);
    };

} // namespace lfs::vis
