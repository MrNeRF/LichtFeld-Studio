/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "training/rasterization/gsplat_rasterizer.hpp"
#include <gtest/gtest.h>

using namespace lfs::training;
using namespace lfs::core;

class GsplatRasterizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create minimal test data
        const size_t N = 100; // Number of Gaussians
        const int sh_degree = 0;

        // Create random Gaussian parameters
        means_ = Tensor::randn({N, 3}, Device::CUDA, DataType::Float32);
        sh0_ = Tensor::randn({N, 1, 3}, Device::CUDA, DataType::Float32);            // sh0 is [N, 1, 3]
        shN_ = Tensor::zeros({N, 0, 3}, Device::CUDA, DataType::Float32);            // No higher SH for degree 0
        scaling_ = Tensor::randn({N, 3}, Device::CUDA, DataType::Float32).mul(0.1f); // Small scales
        rotation_ = Tensor::randn({N, 4}, Device::CUDA, DataType::Float32);
        opacity_ = Tensor::randn({N}, Device::CUDA, DataType::Float32);

        // Create SplatData
        splat_data_ = std::make_unique<SplatData>(
            sh_degree,
            means_,
            sh0_,
            shN_,
            scaling_,
            rotation_,
            opacity_,
            1.0f // scene_scale
        );

        // Create camera
        auto R = Tensor::eye(3, Device::CUDA);
        auto T = Tensor::zeros({3}, Device::CUDA, DataType::Float32);

        // Set camera at z=3 looking at origin
        std::vector<float> T_data = {0.0f, 0.0f, 3.0f};
        T = Tensor::from_blob(T_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);

        camera_ = std::make_unique<Camera>(
            R, T,
            500.0f, 500.0f, // focal_x, focal_y
            320.0f, 240.0f, // center_x, center_y
            Tensor(),       // radial_distortion
            Tensor(),       // tangential_distortion
            lfs::core::CameraModelType::PINHOLE,
            "test_image",
            "",
            std::filesystem::path{}, // mask_path
            640, 480,                // camera_width, camera_height (constructor sets image_width/height too)
            0                        // uid
        );

        // Background color
        bg_color_ = Tensor::zeros({3}, Device::CUDA, DataType::Float32);
        bg_color_.fill_(0.5f); // Gray background
    }

    std::unique_ptr<SplatData> splat_data_;
    std::unique_ptr<Camera> camera_;
    Tensor means_, sh0_, shN_, scaling_, rotation_, opacity_;
    Tensor bg_color_;
};

TEST_F(GsplatRasterizerTest, ForwardPassBasic) {
    // Just test that forward pass doesn't crash
    auto result = gsplat_rasterize_forward(
        *camera_, *splat_data_, bg_color_,
        0, 0, 0, 0, 1.0f, false, GsplatRenderMode::RGB);

    ASSERT_TRUE(result.has_value()) << "Forward pass failed: " << result.error();

    auto& [render_output, ctx] = result.value();

    // Check output dimensions
    EXPECT_EQ(render_output.width, 640);
    EXPECT_EQ(render_output.height, 480);
    EXPECT_TRUE(render_output.image.is_valid());
    EXPECT_EQ(render_output.image.shape()[0], 3); // CHW format
    EXPECT_EQ(render_output.image.shape()[1], 480);
    EXPECT_EQ(render_output.image.shape()[2], 640);

    // Check alpha
    EXPECT_TRUE(render_output.alpha.is_valid());
    EXPECT_EQ(render_output.alpha.shape()[0], 1);
    EXPECT_EQ(render_output.alpha.shape()[1], 480);
    EXPECT_EQ(render_output.alpha.shape()[2], 640);

    std::cout << "Forward pass succeeded!" << std::endl;
    std::cout << "  Image shape: [" << render_output.image.shape()[0] << ", "
              << render_output.image.shape()[1] << ", "
              << render_output.image.shape()[2] << "]" << std::endl;
}

TEST_F(GsplatRasterizerTest, InferenceWrapper) {
    // Test the convenience wrapper
    EXPECT_NO_THROW({
        auto output = gsplat_rasterize(*camera_, *splat_data_, bg_color_);
        EXPECT_TRUE(output.image.is_valid());
    });
}
