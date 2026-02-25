/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "../render_pass.hpp"
#include "core/point_cloud.hpp"
#include <glm/glm.hpp>
#include <memory>

namespace lfs::vis {

    class PointCloudPass final : public RenderPass {
    public:
        PointCloudPass() = default;

        [[nodiscard]] const char* name() const override { return "PointCloudPass"; }
        [[nodiscard]] DirtyMask sensitivity() const override {
            return DirtyFlag::SPLATS | DirtyFlag::CAMERA | DirtyFlag::VIEWPORT;
        }

        [[nodiscard]] bool shouldExecute(DirtyMask frame_dirty, const FrameContext& ctx) const override;

        void execute(lfs::rendering::RenderingEngine& engine,
                     const FrameContext& ctx,
                     FrameResources& res) override;

        void resetCache();

    private:
        std::unique_ptr<lfs::core::PointCloud> cached_filtered_point_cloud_;
        const lfs::core::PointCloud* cached_source_point_cloud_ = nullptr;
        glm::mat4 cached_cropbox_transform_{1.0f};
        glm::vec3 cached_cropbox_min_{0.0f};
        glm::vec3 cached_cropbox_max_{0.0f};
        bool cached_cropbox_inverse_ = false;
    };

} // namespace lfs::vis
