/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "../render_pass.hpp"

namespace lfs::vis {

    class OverlayPass final : public RenderPass {
    public:
        OverlayPass() = default;

        [[nodiscard]] const char* name() const override { return "OverlayPass"; }
        [[nodiscard]] DirtyMask sensitivity() const override {
            return DirtyFlag::OVERLAY | DirtyFlag::CAMERA | DirtyFlag::VIEWPORT | DirtyFlag::SPLATS;
        }

        // Must run every frame — GL double buffering requires overlay redraw
        [[nodiscard]] bool shouldExecute(DirtyMask, const FrameContext&) const override {
            return true;
        }

        void execute(lfs::rendering::RenderingEngine& engine,
                     const FrameContext& ctx,
                     FrameResources& res) override;
    };

} // namespace lfs::vis
