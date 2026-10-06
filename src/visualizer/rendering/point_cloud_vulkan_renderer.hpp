/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "scene_renderer.hpp"

#include "core/export.hpp"
#include "core/tensor.hpp"
#include "render_target_id.hpp"
#include "rendering/rendering.hpp"
#include "window/vulkan_context.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <glm/glm.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace lfs::vis {

    // Vulkan-native point cloud rasterizer. Renders points as disk-splats via a
    // graphics pipeline with hardware depth test (equivalent to the CUDA
    // atomicMin path), then exposes the color + depth VkImages so the rendering
    // manager can route them through the same external-image plumbing as
    // VkSplat (no CUDA tensor staging on the display path).
    class LFS_VIS_API PointCloudVulkanRenderer {
        friend struct PointCloudOutputOwnershipTestAccess;

    public:
        using RenderResult = PointSceneRenderer::RenderResult;
        using RenderRequest = PointSceneRenderer::RenderRequest;
        using CropBox = PointSceneRenderer::CropBox;
        using CropEllipsoid = PointSceneRenderer::CropEllipsoid;

        PointCloudVulkanRenderer();
        ~PointCloudVulkanRenderer();

        PointCloudVulkanRenderer(const PointCloudVulkanRenderer&) = delete;
        PointCloudVulkanRenderer& operator=(const PointCloudVulkanRenderer&) = delete;

        [[nodiscard]] std::expected<RenderResult, std::string> render(
            VulkanContext& context,
            const RenderRequest& request,
            RenderTargetId target);
        [[nodiscard]] std::expected<std::shared_ptr<lfs::core::Tensor>, std::string> readOutputImage(
            VulkanContext& context,
            RenderTargetId target);

        [[nodiscard]] std::expected<float, std::string> sampleDepthAtPixel(
            VulkanContext& context, const SceneRenderer::DepthSampleRequest& request);

        [[nodiscard]] bool takeRefinementRequest();
        [[nodiscard]] bool hasRenderTarget(RenderTargetId target) const;
        [[nodiscard]] bool releaseRenderTarget(RenderTargetId target);
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    struct LFS_VIS_API PointCloudOutputOwnershipTestAccess {
        static const void* createEmptyOutput(PointCloudVulkanRenderer& renderer, RenderTargetId target);
        static const void* outputIdentity(const PointCloudVulkanRenderer& renderer, RenderTargetId target);
    };

} // namespace lfs::vis
