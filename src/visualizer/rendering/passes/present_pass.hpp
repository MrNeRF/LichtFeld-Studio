/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "../render_pass.hpp"

namespace lfs::vis {

    class PresentPass final : public RenderPass {
    public:
        PresentPass() = default;

        [[nodiscard]] const char* name() const override { return "PresentPass"; }
        [[nodiscard]] DirtyMask sensitivity() const override { return DirtyFlag::ALL; }
        [[nodiscard]] bool shouldExecute(DirtyMask, const FrameContext&) const override { return true; }

        void execute(lfs::rendering::RenderingEngine& engine,
                     const FrameContext& ctx,
                     FrameResources& res) override;
    };

} // namespace lfs::vis
