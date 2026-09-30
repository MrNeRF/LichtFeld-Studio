/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#ifdef __APPLE__
#include "vksplat_viewport_renderer.hpp"

namespace lfs::vis {
    // Native rasterization with imported textures for the existing desktop compositor.
    // This boundary contains no Objective-C types, keeping Apple headers out of the
    // cross-platform viewport and Python bindings.
    class LFS_VIS_API MetalViewportRenderer {
    public:
        MetalViewportRenderer();
        ~MetalViewportRenderer();
        static bool supports(const core::SplatData&, const rendering::ViewportRenderRequest&);
        std::expected<VksplatViewportRenderer::RenderResult, std::string> render(
            VulkanContext&, const core::SplatData&, const rendering::ViewportRenderRequest&,
            VksplatViewportRenderer::OutputSlot);
        glm::ivec2 size(VksplatViewportRenderer::OutputSlot) const;
        // Explicit validation/readback boundary: waits for the native command and
        // distinguishes a complete image from capacity-overflow fallback output.
        std::expected<bool, std::string> outputComplete(VksplatViewportRenderer::OutputSlot) const;
        std::expected<void, std::string> readColor(VksplatViewportRenderer::OutputSlot,
                                                   core::Tensor&, int x, int y) const;
        std::expected<float, std::string> readDepth(const VksplatViewportRenderer::DepthSampleRequest&) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::vis
#endif
