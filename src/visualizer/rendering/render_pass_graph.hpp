/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <memory>
#include <vector>

namespace lfs::vis {

    class RenderPass;
    class SplatRasterPass;
    class PointCloudPass;

    class RenderPassGraph {
    public:
        RenderPassGraph();
        ~RenderPassGraph();

        RenderPassGraph(const RenderPassGraph&) = delete;
        RenderPassGraph& operator=(const RenderPassGraph&) = delete;

        [[nodiscard]] const std::vector<std::unique_ptr<RenderPass>>& passes() const { return passes_; }
        [[nodiscard]] std::vector<std::unique_ptr<RenderPass>>& passes() { return passes_; }

        [[nodiscard]] SplatRasterPass* splatRasterPass() const { return splat_raster_pass_; }
        [[nodiscard]] PointCloudPass* pointCloudPass() const { return point_cloud_pass_; }

        void resetPointCloudCache() const;

    private:
        std::vector<std::unique_ptr<RenderPass>> passes_;
        SplatRasterPass* splat_raster_pass_ = nullptr;
        PointCloudPass* point_cloud_pass_ = nullptr;
    };

} // namespace lfs::vis
