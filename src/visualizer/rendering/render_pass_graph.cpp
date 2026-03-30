/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "render_pass_graph.hpp"
#include "passes/mesh_pass.hpp"
#include "passes/overlay_pass.hpp"
#include "passes/point_cloud_pass.hpp"
#include "passes/present_pass.hpp"
#include "passes/splat_raster_pass.hpp"
#include "passes/split_view_pass.hpp"
#include "render_pass.hpp"

namespace lfs::vis {

    RenderPassGraph::RenderPassGraph() {
        passes_.push_back(std::make_unique<SplitViewPass>());
        passes_.push_back(std::make_unique<SplatRasterPass>());
        splat_raster_pass_ = static_cast<SplatRasterPass*>(passes_.back().get());
        passes_.push_back(std::make_unique<PointCloudPass>());
        point_cloud_pass_ = static_cast<PointCloudPass*>(passes_.back().get());
        passes_.push_back(std::make_unique<PresentPass>());
        passes_.push_back(std::make_unique<MeshPass>());
        passes_.push_back(std::make_unique<OverlayPass>());
    }

    RenderPassGraph::~RenderPassGraph() = default;

    void RenderPassGraph::resetPointCloudCache() const {
        if (point_cloud_pass_) {
            point_cloud_pass_->resetCache();
        }
    }

} // namespace lfs::vis
