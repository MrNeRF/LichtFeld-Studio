/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "render_pass.hpp"
#include <memory>
#include <optional>

namespace lfs::rendering {
    class RenderingEngine;
}

namespace lfs::vis {

    class LFS_VIS_API ViewportArtifactService {
    public:
        ViewportArtifactService() = default;
        ~ViewportArtifactService();

        ViewportArtifactService(const ViewportArtifactService&) = delete;
        ViewportArtifactService& operator=(const ViewportArtifactService&) = delete;

        [[nodiscard]] bool hasGpuFrame() const;
        [[nodiscard]] bool hasViewportOutput() const;
        [[nodiscard]] bool hasOutputArtifacts() const;

        [[nodiscard]] const CachedRenderMetadata& cachedMetadata() const { return metadata_; }
        [[nodiscard]] const std::optional<lfs::rendering::GpuFrame>& gpuFrame() const { return gpu_frame_; }
        [[nodiscard]] glm::ivec2 renderedSize() const { return rendered_size_; }
        [[nodiscard]] uint64_t artifactGeneration() const { return artifact_generation_; }

        [[nodiscard]] std::shared_ptr<lfs::core::Tensor> getCapturedImageIfCurrent() const;

        void clearViewportOutput();
        void updateFromFrameResources(const FrameResources& resources, bool viewport_output_updated);
        void storeCapturedImage(std::shared_ptr<lfs::core::Tensor> image);

        [[nodiscard]] float sampleLinearDepthAt(int x,
                                                int y,
                                                const glm::ivec2& fallback_viewport_size,
                                                const lfs::rendering::RenderingEngine* engine,
                                                std::optional<SplitViewPanelId> panel = std::nullopt) const;

    private:
        void invalidateCapture();
        [[nodiscard]] float readLinearDepth(const lfs::rendering::GpuFrame& frame,
                                            int x,
                                            int y,
                                            int viewport_height) const;

        CachedRenderMetadata metadata_;
        std::optional<lfs::rendering::GpuFrame> gpu_frame_;
        glm::ivec2 rendered_size_{0};
        std::shared_ptr<lfs::core::Tensor> captured_image_;
        uint64_t artifact_generation_ = 1;
        uint64_t captured_artifact_generation_ = 0;
        mutable unsigned int depth_readback_fbo_ = 0;
    };

} // namespace lfs::vis
