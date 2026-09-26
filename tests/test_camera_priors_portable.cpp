/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Depth and normal priors and lens undistortion without CUDA: the camera loads
// them through the portable image ops on the Vulkan and Metal backends.

#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_image.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {
    using namespace lfs::core;

    constexpr int kWidth = 32, kHeight = 24;

    Tensor opencv_radial() { return Tensor::from_vector({-0.2f, 0.05f}, {2}, Device::CPU); }

    Tensor no_distortion() { return Tensor::zeros({0}, Device::CPU); }

    class CameraPriorsPortable : public testing::TestWithParam<GpuBackend> {
    protected:
        void SetUp() override {
            if (!gpu_backend_available(GetParam()))
                GTEST_SKIP() << "Backend unavailable";
            directory_ = std::filesystem::temp_directory_path() /
                         ("lfs_camera_priors_" + std::to_string(static_cast<int>(GetParam())));
            std::filesystem::create_directories(directory_);
            // Depth rises left to right; normals all face the camera.
            std::vector<uint16_t> depth(kWidth * kHeight);
            for (int y = 0; y < kHeight; ++y)
                for (int x = 0; x < kWidth; ++x)
                    depth[y * kWidth + x] = static_cast<uint16_t>(1000 + 1000 * x);
            ASSERT_TRUE(save_png(depth_path(), depth.data(), kWidth, kHeight, 1, 16, 1));
            // 16-bit, like the depth, so both decode without the app's image loader.
            std::vector<uint16_t> normal(kWidth * kHeight * 3);
            for (size_t i = 0; i < normal.size(); i += 3) {
                normal[i] = 32768;
                normal[i + 1] = 32768;
                normal[i + 2] = 65535;
            }
            ASSERT_TRUE(save_png(normal_path(), normal.data(), kWidth, kHeight, 3, 16, 1));
        }

        void TearDown() override {
            std::error_code ignored;
            std::filesystem::remove_all(directory_, ignored);
        }

        std::filesystem::path depth_path() const { return directory_ / "depth.png"; }
        std::filesystem::path normal_path() const { return directory_ / "normal.png"; }

        std::unique_ptr<Camera> make_camera(Tensor radial) const {
            auto camera = std::make_unique<Camera>(
                Tensor::eye(3, Device::CPU), Tensor::zeros({3}, Device::CPU), 30.0f, 30.0f, kWidth / 2.0f,
                kHeight / 2.0f, std::move(radial), no_distortion(), CameraModelType::PINHOLE, "view", "", "",
                kWidth, kHeight, 0, 0, depth_path(), normal_path());
            camera->set_image_dimensions(kWidth, kHeight);
            return camera;
        }

        std::filesystem::path directory_;
    };

    TEST(UndistortParamsPortable, SolveAndRescaleTheUndistortedCamera) {
        const auto params = compute_undistort_params(30.0f, 30.0f, 16.0f, 12.0f, kWidth, kHeight, opencv_radial(),
                                                     no_distortion(), CameraModelType::PINHOLE);
        EXPECT_EQ(params.src_width, kWidth);
        EXPECT_EQ(params.num_distortion, 2);
        EXPECT_GT(params.dst_width, 0);
        EXPECT_GT(params.dst_height, 0);
        EXPECT_TRUE(std::isfinite(params.dst_fx) && params.dst_fx > 0.0f);
        const auto half = scale_undistort_params(params, kWidth / 2, kHeight / 2, 0);
        EXPECT_FLOAT_EQ(half.src_fx, params.src_fx / 2.0f);
        EXPECT_FLOAT_EQ(half.dst_fx, params.dst_fx / 2.0f);
    }

    TEST_P(CameraPriorsPortable, LoadsDepthAndNormalPriors) {
        GpuBackendScope scope(GetParam());
        const auto camera = make_camera(no_distortion());
        const Tensor depth = camera->load_and_get_depth(-1, 0);
        ASSERT_TRUE(depth.is_valid());
        EXPECT_EQ(gpu_backend_of(depth), GetParam());
        ASSERT_EQ(depth.shape(), TensorShape({kHeight, kWidth}));
        const auto depth_values = depth.cpu().to_vector();
        for (int x = 0; x < kWidth; ++x)
            EXPECT_NEAR(depth_values[5 * kWidth + x], (1000.0f + 1000.0f * x) / 65535.0f, 1.0e-5f) << "x=" << x;

        const Tensor normal = camera->load_and_get_normal(-1, 0, {});
        ASSERT_TRUE(normal.is_valid());
        ASSERT_EQ(normal.shape(), TensorShape({3, kHeight, kWidth}));
        const auto normal_values = normal.cpu().to_vector();
        const size_t plane = kWidth * kHeight, center = 12 * kWidth + 16;
        EXPECT_NEAR(normal_values[2 * plane + center], 1.0f, 1.0e-3f);
        EXPECT_NEAR(normal_values[center], 0.0f, 1.0e-2f);
    }

    TEST_P(CameraPriorsPortable, UndistortsPriorsOfDistortedCameras) {
        GpuBackendScope scope(GetParam());
        const auto camera = make_camera(opencv_radial());
        camera->prepare_undistortion();
        ASSERT_TRUE(camera->is_undistort_prepared());
        const auto scaled = scale_undistort_params(camera->undistort_params(), kWidth, kHeight, 0);
        const Tensor depth = camera->load_and_get_depth(-1, 0);
        ASSERT_TRUE(depth.is_valid());
        ASSERT_EQ(depth.shape(), TensorShape({static_cast<size_t>(scaled.dst_height), static_cast<size_t>(scaled.dst_width)}));
        // The principal point maps onto itself, so the center keeps its depth.
        const auto values = depth.cpu().to_vector();
        const float center = values[(scaled.dst_height / 2) * scaled.dst_width + scaled.dst_width / 2];
        EXPECT_GT(center, 1000.0f / 65535.0f);
        EXPECT_LT(center, (1000.0f + 1000.0f * kWidth) / 65535.0f);
        const Tensor normal = camera->load_and_get_normal(-1, 0, {});
        ASSERT_TRUE(normal.is_valid());
        EXPECT_EQ(normal.shape()[1], static_cast<size_t>(scaled.dst_height));
    }

    INSTANTIATE_TEST_SUITE_P(Backends, CameraPriorsPortable, testing::Values(GpuBackend::Vulkan, GpuBackend::Metal),
                             [](const auto& info) { return info.param == GpuBackend::Metal ? "Metal" : "Vulkan"; });
} // namespace
