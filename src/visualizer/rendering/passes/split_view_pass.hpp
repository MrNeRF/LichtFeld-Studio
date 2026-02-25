/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "../render_pass.hpp"

namespace lfs::vis {

    class SplitViewPass final : public RenderPass {
    public:
        SplitViewPass() = default;

        [[nodiscard]] const char* name() const override { return "SplitViewPass"; }
        [[nodiscard]] DirtyMask sensitivity() const override {
            return DirtyFlag::SPLIT_VIEW | DirtyFlag::SPLATS | DirtyFlag::CAMERA | DirtyFlag::VIEWPORT;
        }

        [[nodiscard]] bool shouldExecute(DirtyMask frame_dirty, const FrameContext& ctx) const override;

        void execute(lfs::rendering::RenderingEngine& engine,
                     const FrameContext& ctx,
                     FrameResources& res) override;

    private:
        [[nodiscard]] std::optional<lfs::rendering::SplitViewRequest>
        buildSplitViewRequest(const FrameContext& ctx, const FrameResources& res);
    };

} // namespace lfs::vis
