/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/parameters.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "lfs/training/ops/registry.hpp"
#include "training/trainer.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string>

namespace {

    TEST(VulkanTrainerStart, ThreeDGUTReachesModelValidation) {
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
        // ThreeDGUT passes the family check and reaches scene validation.
        lfs::core::param::TrainingParameters params;
        params.optimization.set_raster_backend(lfs::core::param::RasterBackendId::ThreeDGUT);
        const auto result = trainer.initialize(params);
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error(), "Scene has no training model set");
    }

    TEST(TrainingOpsSessionBackend, MortonAndSessionFollowTheDefaultBackend) {
        const auto backend = lfs::core::default_gpu_backend();
        if (lfs::training::training_ops(backend).morton == nullptr) {
            EXPECT_THROW(lfs::training::training_morton_ops(), std::runtime_error);
        } else {
            EXPECT_EQ(&lfs::training::training_morton_ops(), lfs::training::training_ops(backend).morton);
        }
        if (lfs::training::training_ops(backend).session == nullptr) {
            try {
                lfs::training::training_session_ops();
                FAIL() << "missing session ops did not throw";
            } catch (const std::runtime_error& error) {
                EXPECT_NE(std::string(error.what()).find("Session"), std::string::npos);
            }
        } else {
            EXPECT_EQ(&lfs::training::training_session_ops(), lfs::training::training_ops(backend).session);
        }
    }

} // namespace
