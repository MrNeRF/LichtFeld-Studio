/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "../render_pass.hpp"

namespace lfs::vis {

    class RenderingManager;

    class SplatRasterPass final : public RenderPass {
    public:
        explicit SplatRasterPass(RenderingManager& mgr) : mgr_(mgr) {}

        [[nodiscard]] const char* name() const override { return "SplatRasterPass"; }
        [[nodiscard]] DirtyMask sensitivity() const override {
            return DirtyFlag::SPLATS | DirtyFlag::SELECTION | DirtyFlag::CAMERA |
                   DirtyFlag::VIEWPORT | DirtyFlag::BACKGROUND;
        }

        [[nodiscard]] bool shouldExecute(DirtyMask frame_dirty, const FrameContext& ctx) const override;

        void execute(lfs::rendering::RenderingEngine& engine,
                     const FrameContext& ctx,
                     FrameResources& res) override;

        void renderToTexture(const RenderingManager::RenderContext& context,
                             SceneManager* scene_manager,
                             const lfs::core::SplatData* model);

    private:
        RenderingManager& mgr_;
    };

} // namespace lfs::vis
