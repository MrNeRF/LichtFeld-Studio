/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/viewport_artifact_service.hpp"
#include <gtest/gtest.h>

namespace lfs::vis {
    TEST(ViewerBackendMetadata, TracksPublishedOutputAndClearsOnSceneClose) {
        ViewportArtifactService artifacts;
        EXPECT_EQ(artifacts.viewerBackendMask(), 0u);
        rendering::FrameMetadata frame;
        frame.valid = true;
        frame.viewer_backend_mask = rendering::viewerBackendBit(rendering::ViewerBackend::Metal);
        artifacts.setLazyCapture([] { return std::shared_ptr<core::Tensor>{}; }, frame, {64, 48});
        EXPECT_EQ(artifacts.viewerBackendMask(), 2u);
        frame.viewer_backend_mask = rendering::viewerBackendBit(rendering::ViewerBackend::Vulkan);
        artifacts.setLazyCaptureForCurrentOutput([] { return std::shared_ptr<core::Tensor>{}; }, frame, {64, 48});
        EXPECT_EQ(artifacts.viewerBackendMask(), 1u);
        artifacts.clearViewportOutput();
        EXPECT_EQ(artifacts.viewerBackendMask(), 0u);
    }

    TEST(ViewerBackendMetadata, PreservesSoftwarePointCloudPanelBackend) {
        ViewportArtifactService artifacts;
        rendering::FrameMetadata frame;
        frame.valid = true;
        frame.viewer_backend_mask = rendering::softwareViewerBackendBit;
        artifacts.updateFromImageOutput({}, frame, {64, 48}, true);
        EXPECT_EQ(artifacts.viewerBackendMask(), 4u);
    }

    TEST(ViewerBackendMetadata, PreservesMixedSplitAndCachedFrameWithoutCapturingPixels) {
        ViewportArtifactService artifacts;
        rendering::FrameMetadata frame;
        frame.valid = true;
        frame.viewer_backend_mask = rendering::viewerBackendBit(rendering::ViewerBackend::Metal) |
                                    rendering::viewerBackendBit(rendering::ViewerBackend::Vulkan);
        FrameResources resources;
        resources.cached_metadata = makeCachedRenderMetadata(frame);
        resources.cached_result_size = {128, 48};
        artifacts.updateFromFrameResources(resources, true);
        EXPECT_EQ(artifacts.viewerBackendMask(), 3u);
        EXPECT_FALSE(artifacts.getCapturedImageIfCurrent());
        artifacts.updateFromImageOutput({}, frame, {128, 48}, true);
        EXPECT_EQ(artifacts.viewerBackendMask(), 3u);
    }
} // namespace lfs::vis
