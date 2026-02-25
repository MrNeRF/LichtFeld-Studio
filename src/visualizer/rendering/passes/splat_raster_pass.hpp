/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "../render_pass.hpp"
#include <glm/glm.hpp>

namespace lfs::vis {

    class SplatRasterPass final : public RenderPass {
    public:
        SplatRasterPass() = default;
        ~SplatRasterPass() override;

        [[nodiscard]] const char* name() const override { return "SplatRasterPass"; }
        [[nodiscard]] DirtyMask sensitivity() const override {
            return DirtyFlag::SPLATS | DirtyFlag::SELECTION | DirtyFlag::CAMERA |
                   DirtyFlag::VIEWPORT | DirtyFlag::BACKGROUND | DirtyFlag::PPISP;
        }

        [[nodiscard]] bool shouldExecute(DirtyMask frame_dirty, const FrameContext& ctx) const override;

        void execute(lfs::rendering::RenderingEngine& engine,
                     const FrameContext& ctx,
                     FrameResources& res) override;

    private:
        void renderToTexture(lfs::rendering::RenderingEngine& engine,
                             const FrameContext& ctx, FrameResources& res);

        unsigned int render_fbo_ = 0;
        unsigned int render_depth_rbo_ = 0;
        glm::ivec2 texture_size_{0, 0};
        glm::ivec2 depth_buffer_size_{0, 0};

        unsigned long long* d_hovered_depth_id_ = nullptr;

        // Async hover readback (1-frame latency)
        unsigned long long* h_hovered_depth_id_ = nullptr;
        unsigned long long last_hovered_result_ = 0xFFFFFFFFFFFFFFFFULL;
        cudaEvent_t readback_event_ = nullptr;
        bool readback_pending_ = false;
    };

} // namespace lfs::vis
