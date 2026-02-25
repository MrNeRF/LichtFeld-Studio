/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "../render_pass.hpp"

namespace lfs::vis {

    class RenderingManager;

    class PointCloudPass final : public RenderPass {
    public:
        explicit PointCloudPass(RenderingManager& mgr) : mgr_(mgr) {}

        [[nodiscard]] const char* name() const override { return "PointCloudPass"; }
        [[nodiscard]] DirtyMask sensitivity() const override {
            return DirtyFlag::SPLATS | DirtyFlag::CAMERA | DirtyFlag::VIEWPORT;
        }

        [[nodiscard]] bool shouldExecute(DirtyMask frame_dirty, const FrameContext& ctx) const override;

        void execute(lfs::rendering::RenderingEngine& engine,
                     const FrameContext& ctx,
                     FrameResources& res) override;

    private:
        RenderingManager& mgr_;
    };

} // namespace lfs::vis
