/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/viewport_artifact_service.hpp"
#include <gtest/gtest.h>

namespace lfs::vis {
    TEST(ViewerBackendMetadata, TracksPublishedOutputAndClearsOnSceneClose) {
        ViewportArtifactService artifacts;
        EXPECT_FALSE(artifacts.viewerBackend());
        rendering::FrameMetadata frame;
        frame.valid = true;
        frame.viewer_backend = rendering::ViewerBackend::Metal;
        artifacts.setLazyCapture([] { return std::shared_ptr<core::Tensor>{}; }, frame, {64, 48});
        EXPECT_EQ(artifacts.viewerBackend(), rendering::ViewerBackend::Metal);
        frame.viewer_backend = rendering::ViewerBackend::Vulkan;
        artifacts.setLazyCaptureForCurrentOutput([] { return std::shared_ptr<core::Tensor>{}; }, frame, {64, 48});
        EXPECT_EQ(artifacts.viewerBackend(), rendering::ViewerBackend::Vulkan);
        artifacts.clearViewportOutput();
        EXPECT_FALSE(artifacts.viewerBackend());
    }

    TEST(ViewerBackendMetadata, PreservesCudaPointCloudPanelBackend) {
        ViewportArtifactService artifacts;
        rendering::FrameMetadata frame;
        frame.valid = true;
        frame.viewer_backend = rendering::ViewerBackend::Cuda;
        artifacts.updateFromImageOutput({}, frame, {64, 48}, true);
        EXPECT_EQ(artifacts.viewerBackend(), rendering::ViewerBackend::Cuda);
    }

} // namespace lfs::vis
