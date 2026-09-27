/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "cuda_backend_test.hpp"
#include "optimizer/adam_optimizer.hpp"
#include "training/rasterization/gsplat_rasterizer.hpp"
#include <expected>

namespace lfs::test {
    struct GsplatTestContext : training::GsplatRasterizeContext {
        gpu_ops::GsplatSaved* saved = nullptr;
        void release() { training::cuda_gsplat_ops().release(*saved); }
    };

    // Test camera binding; the fixture retains its renderer between forwards.
    class GsplatTestRenderer {
    public:
        gpu_ops::GsplatSaved gsplat_saved;

        std::expected<std::pair<training::RenderOutput, GsplatTestContext>, std::string>
        gsplat_rasterize_forward(core::Camera& camera, core::SplatData& model, core::Tensor& bg,
                                 int x = 0, int y = 0, int w = 0, int h = 0, float scale = 1.f,
                                 bool aa = false, training::GsplatRenderMode mode = training::GsplatRenderMode::RGB,
                                 bool = false, const core::Tensor& bg_image = {}) {
            const auto& ops = training::cuda_gsplat_ops();
            if (!gsplat_saved.backend)
                gsplat_saved.backend = ops.create();
            training::RenderOutput output;
            const auto result = training::gsplat_render(ops, gsplat_saved, camera, model, bg,
                                                        x, y, w, h, scale, aa, mode, bg_image, output);
            if (result.code != gpu_ops::RasterResult::Code::Success)
                return std::unexpected(std::string(result.message));
            GsplatTestContext ctx;
            static_cast<training::GsplatRasterizeContext&>(ctx) = training::cuda_gsplat_frame(gsplat_saved);
            ctx.saved = &gsplat_saved;
            return std::pair{std::move(output), std::move(ctx)};
        }

        void gsplat_rasterize_backward(GsplatTestContext& ctx, const core::Tensor& image,
                                       const core::Tensor& alpha, core::SplatData& model, training::AdamOptimizer& optimizer,
                                       const core::Tensor& error = {}, const core::Tensor& edge = {}, core::Tensor scores = {}) {
            core::Tensor unused;
            training::cuda_gsplat_ops().backward(*ctx.saved, image, alpha,
                                                 training::gsplat_gradients(optimizer),
                                                 model._densification_info, error, edge, scores,
                                                 optimizer.collect_projected_screen_share() ? model._max_screen_share : unused);
        }

        training::RenderOutput gsplat_rasterize(const core::Camera& camera, core::SplatData& model,
                                                core::Tensor& bg, float scale = 1.f, bool aa = false,
                                                training::GsplatRenderMode mode = training::GsplatRenderMode::RGB, bool = false) {
            const auto& ops = training::cuda_gsplat_ops();
            if (!gsplat_saved.backend)
                gsplat_saved.backend = ops.create();
            return training::gsplat_infer(ops, gsplat_saved, camera, model, bg, scale, aa, mode);
        }

        bool release_gsplat_caches() {
            training::cuda_gsplat_ops().release(gsplat_saved);
            return training::gsplat_release_caches(gsplat_saved);
        }
    };

    class GsplatBackendTest : public CudaBackendTest, public GsplatTestRenderer {};
} // namespace lfs::test
