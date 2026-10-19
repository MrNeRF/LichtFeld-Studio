/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/point_cloud.hpp"
#include "core/tensor.hpp"
#include "rendering/rendering.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

TEST(PointCloudSelectionOverlay, CommittedAndTransientColorsReachRasterImage) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP() << "requires CUDA";
    using namespace lfs::core;
    using namespace lfs::rendering;
    auto engine = RenderingEngine::create();
    lfs::core::PointCloud cloud;
    cloud.means = Tensor::from_vector(std::vector<float>{0.f, 0.f, -2.f}, {size_t{1}, size_t{3}}, Device::CPU);
    cloud.colors = Tensor::zeros({size_t{1}, size_t{3}}, Device::CPU);
    PointCloudRenderRequest request;
    request.frame_view.size = {64, 64};
    request.overlay.selection_colors[3] = {1.f, 0.f, 0.f, 1.f};
    request.overlay.selection_colors[kSelectionPreviewColorIndex] = {0.f, 1.f, 0.f, 1.f};
    request.overlay.selection_mask = std::make_shared<Tensor>(
        Tensor::from_vector(std::vector<int>{3}, {size_t{1}}, Device::CPU).to(DataType::UInt8));
    const auto image_max = [](const auto& result, size_t channel) {
        return result.image->slice(0, channel, channel + 1).max_scalar();
    };
    auto committed = engine->renderPointCloudImage(cloud, request);
    ASSERT_TRUE(committed) << committed.error();
    EXPECT_NEAR(image_max(*committed, 0), .75f, .01f);
    EXPECT_LT(image_max(*committed, 1), .01f);
    Tensor preview = Tensor::ones({size_t{1}}, Device::CPU, DataType::UInt8);
    request.overlay.transient_mask.mask = &preview;
    request.overlay.transient_mask.additive = false;
    auto subtract = engine->renderPointCloudImage(cloud, request);
    ASSERT_TRUE(subtract) << subtract.error();
    EXPECT_NEAR(image_max(*subtract, 1), .9f, .01f);
    request.overlay.selection_mask.reset();
    request.overlay.transient_mask.additive = true;
    auto add = engine->renderPointCloudImage(cloud, request);
    ASSERT_TRUE(add) << add.error();
    EXPECT_NEAR(image_max(*add, 1), .9f, .01f);
    request.overlay.transient_mask = {};
    auto plain = engine->renderPointCloudImage(cloud, request);
    ASSERT_TRUE(plain) << plain.error();
    EXPECT_LT(image_max(*plain, 0), .01f);
    EXPECT_LT(image_max(*plain, 1), .01f);
}
