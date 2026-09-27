/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/error.hpp"
#include "core/splat_data.hpp"
#include "lfs/training/ops/gsplat.hpp"
#include "optimizer/render_output.hpp"

namespace lfs::training {
    class AdamOptimizer;
    lfs::gpu_ops::GsplatGradients gsplat_gradients(AdamOptimizer&);

    using GsplatRenderMode = lfs::gpu_ops::GsplatRenderMode;
    [[nodiscard]] lfs::Error gsplat_raster_error(const lfs::gpu_ops::RasterResult&);

    lfs::gpu_ops::RasterResult gsplat_render(
        const lfs::gpu_ops::GsplatRasterOps&, lfs::gpu_ops::GsplatSaved&,
        const core::Camera&, core::SplatData&, const core::Tensor& bg_color,
        int tile_x, int tile_y, int tile_w, int tile_h,
        float scaling_modifier, bool antialiased, GsplatRenderMode,
        const core::Tensor& bg_image, RenderOutput&);

    RenderOutput gsplat_infer(
        const lfs::gpu_ops::GsplatRasterOps&, lfs::gpu_ops::GsplatSaved&,
        const core::Camera&, core::SplatData&, const core::Tensor& bg_color,
        float scaling_modifier = 1.f, bool antialiased = false,
        GsplatRenderMode = GsplatRenderMode::RGB);

    void gsplat_record_vram(const lfs::gpu_ops::GsplatSaved&, const RenderOutput&,
                            const core::Tensor& gt_tile, const core::Tensor& bg_tile,
                            const core::Tensor& error_map);
    bool gsplat_release_caches(lfs::gpu_ops::GsplatSaved&) noexcept;
} // namespace lfs::training
