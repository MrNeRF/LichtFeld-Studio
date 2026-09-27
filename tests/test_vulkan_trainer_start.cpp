/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/parameters.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "training/trainer.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>

namespace {

    TEST(VulkanTrainerStart, InitializeReportsMissingFamilies) {
        if (!lfs::core::gpu_backend_available(lfs::core::GpuBackend::Vulkan))
            GTEST_SKIP() << "Vulkan is unavailable";
        if (lfs::core::default_gpu_backend() != lfs::core::GpuBackend::Vulkan)
            GTEST_SKIP() << "Vulkan is not the process backend";

        lfs::core::Scene scene;
        const auto cameras = scene.addGroup("Cameras");
        scene.addCamera("camera.png", cameras,
                        std::make_shared<lfs::core::Camera>(
                            lfs::core::Tensor::eye(3, lfs::core::Device::CPU),
                            lfs::core::Tensor::zeros({3}, lfs::core::Device::CPU),
                            100.0f, 100.0f, 32.0f, 32.0f,
                            lfs::core::Tensor(), lfs::core::Tensor(),
                            lfs::core::CameraModelType::PINHOLE,
                            "camera.png", std::filesystem::path{}, std::filesystem::path{},
                            64, 64, 0));
        lfs::training::Trainer trainer(scene);
        const auto result = trainer.initialize(lfs::core::param::TrainingParameters{});
        ASSERT_FALSE(result.has_value());
        EXPECT_NE(result.error().find("Vulkan training is unavailable"), std::string::npos);
        EXPECT_NE(result.error().find("Missing families:"), std::string::npos);
        EXPECT_NE(result.error().find("Photometric"), std::string::npos);
        EXPECT_NE(result.error().find("Fast"), std::string::npos);
    }

} // namespace
