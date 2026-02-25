/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "../render_pass.hpp"

namespace lfs::vis {

    class MeshPass final : public RenderPass {
    public:
        MeshPass() = default;

        [[nodiscard]] const char* name() const override { return "MeshPass"; }

        static constexpr DirtyMask MESH_GEOMETRY_MASK =
            DirtyFlag::MESH | DirtyFlag::CAMERA | DirtyFlag::VIEWPORT;

        [[nodiscard]] DirtyMask sensitivity() const override { return DirtyFlag::ALL; }
        [[nodiscard]] bool shouldExecute(DirtyMask frame_dirty, const FrameContext& ctx) const override;

        void execute(lfs::rendering::RenderingEngine& engine,
                     const FrameContext& ctx,
                     FrameResources& res) override;
    };

} // namespace lfs::vis
