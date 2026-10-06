/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "render_pass.hpp"
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

namespace lfs::vis {

    class LFS_VIS_API ViewportArtifactService {
    public:
        ViewportArtifactService() = default;
        ~ViewportArtifactService();

        ViewportArtifactService(ViewportArtifactService&&) noexcept = default;
        ViewportArtifactService& operator=(ViewportArtifactService&&) noexcept = default;
        ViewportArtifactService(const ViewportArtifactService&) = delete;
        ViewportArtifactService& operator=(const ViewportArtifactService&) = delete;

        [[nodiscard]] std::optional<lfs::rendering::ViewerBackend> viewerBackend() const { return metadata_.viewer_backend; }

        [[nodiscard]] glm::ivec2 renderedSize() const { return rendered_size_; }
        [[nodiscard]] uint64_t artifactGeneration() const { return artifact_generation_; }

        [[nodiscard]] std::shared_ptr<lfs::core::Tensor> getCapturedImageIfCurrent() const;

        void clearViewportOutput();
        void invalidateCapturedImage();
        void updateFromImageOutput(std::shared_ptr<lfs::core::Tensor> image,
                                   const lfs::rendering::FrameMetadata& metadata,
                                   const glm::ivec2& rendered_size,
                                   bool viewport_output_updated);
        void storeCapturedImage(std::shared_ptr<lfs::core::Tensor> image);

        // For split-view frames: the live viewport composes on the GPU and never produces
        // a single CPU/CUDA tensor. Capture/screenshot paths still need one, so the caller
        // hands us a closure that does the CPU compose on demand. The first capture
        // request runs it, caches the result separately from the display-facing
        // capture, and subsequent captures reuse it until the next frame invalidates it.
        using LazyCaptureFn = std::function<std::shared_ptr<lfs::core::Tensor>()>;
        void setLazyCapture(LazyCaptureFn fn,
                            const lfs::rendering::FrameMetadata& metadata,
                            const glm::ivec2& rendered_size);
        void setLazyCaptureForCurrentOutput(LazyCaptureFn fn,
                                            const lfs::rendering::FrameMetadata& metadata,
                                            const glm::ivec2& rendered_size);
        [[nodiscard]] std::shared_ptr<lfs::core::Tensor> resolveLazyCapture();
        [[nodiscard]] bool hasLazyCapture() const { return static_cast<bool>(lazy_capture_); }

        [[nodiscard]] float sampleLinearDepthAt(int x,
                                                int y,
                                                const glm::ivec2& fallback_viewport_size,
                                                std::optional<SplitViewPanelId> panel = std::nullopt) const;

    private:
        void invalidateCapture();
        void setMetadata(const CachedRenderMetadata& metadata);
        CachedRenderMetadata metadata_;
        glm::ivec2 rendered_size_{0};
        std::shared_ptr<lfs::core::Tensor> captured_image_;
        std::shared_ptr<lfs::core::Tensor> lazy_captured_image_;
        uint64_t artifact_generation_ = 1;
        uint64_t captured_artifact_generation_ = 0;
        uint64_t lazy_captured_artifact_generation_ = 0;
        LazyCaptureFn lazy_capture_;
    };

} // namespace lfs::vis
