/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "core/camera.hpp"
#include "core/splat_data.hpp"
#include "rasterization/rasterizer.hpp"

constexpr int RANDOM_SEED = 42;
constexpr float TOLERANCE = 1e-3f; // Slightly relaxed tolerance for numerical differences

class RenderModeTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(RANDOM_SEED);
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA is not available";
        setupTestData();
    }

    void setupTestData() {
        // Camera parameters
        image_width_ = 128; // Smaller for faster tests
        image_height_ = 128;
        fx_ = 100.0f;
        fy_ = 100.0f;
        cx_ = 64.0f;
        cy_ = 64.0f;

        // Number of Gaussians
        num_gaussians_ = 50; // Smaller for faster tests

        auto device = torch::kCUDA;

        // Create random Gaussian parameters
        means_ = torch::randn({num_gaussians_, 3}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
        means_.index_put_({torch::indexing::Slice(), 2}, means_.index({torch::indexing::Slice(), 2}).abs() + 2.0f);

        scales_ = torch::randn({num_gaussians_, 3}, torch::TensorOptions().device(device).dtype(torch::kFloat32)) * 0.1f + 0.3f;
        scales_ = scales_.abs();

        rotations_ = torch::randn({num_gaussians_, 4}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
        rotations_ = torch::nn::functional::normalize(rotations_, torch::nn::functional::NormalizeFuncOptions().dim(1));

        opacities_ = torch::rand({num_gaussians_, 1}, torch::TensorOptions().device(device).dtype(torch::kFloat32)) * 0.8f + 0.2f;

        // SH coefficients (degree 3)
        sh_degree_ = 3;
        int num_sh_bases = (sh_degree_ + 1) * (sh_degree_ + 1);
        sh_coeffs_ = torch::randn({num_gaussians_, num_sh_bases, 3}, torch::TensorOptions().device(device).dtype(torch::kFloat32)) * 0.1f;

        // Background color
        bg_color_ = torch::tensor({0.5f, 0.5f, 0.5f}, torch::TensorOptions().device(device).dtype(torch::kFloat32));

        // Create camera
        auto R = torch::eye(3, torch::TensorOptions().device(device).dtype(torch::kFloat32));
        auto T = torch::tensor({0.0f, 0.0f, 3.0f}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
        auto radial_dist = torch::zeros({4}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
        auto tangential_dist = torch::zeros({2}, torch::TensorOptions().device(device).dtype(torch::kFloat32));

        camera_ = std::make_unique<gs::Camera>(
            R, T, fx_, fy_, cx_, cy_,
            radial_dist, tangential_dist,
            gsplat::CameraModelType::PINHOLE,
            "test_camera", "",
            image_width_, image_height_, 0);
    }

    bool tensorsAlmostEqual(const torch::Tensor& a, const torch::Tensor& b, float tolerance = TOLERANCE) {
        if (!a.defined() || !b.defined()) {
            return a.defined() == b.defined();
        }
        if (a.sizes() != b.sizes()) {
            std::cout << "Size mismatch: " << a.sizes() << " vs " << b.sizes() << std::endl;
            return false;
        }
        auto diff = (a - b).abs();
        auto max_diff = diff.max().item<float>();
        auto mean_diff = diff.mean().item<float>();

        if (max_diff > tolerance) {
            std::cout << "Max difference: " << max_diff << ", Mean difference: " << mean_diff << std::endl;
            return false;
        }
        return true;
    }

    int image_width_;
    int image_height_;
    float fx_, fy_, cx_, cy_;
    int num_gaussians_;
    int sh_degree_;

    torch::Tensor means_;
    torch::Tensor scales_;
    torch::Tensor rotations_;
    torch::Tensor opacities_;
    torch::Tensor sh_coeffs_;
    torch::Tensor bg_color_;

    std::unique_ptr<gs::Camera> camera_;
};

// Test RGB mode
TEST_F(RenderModeTest, RGBMode) {
    auto sh0 = sh_coeffs_.index({torch::indexing::Slice(), 0, torch::indexing::Slice()}).unsqueeze(1);
    auto shN = sh_coeffs_.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None), torch::indexing::Slice()});

    gs::SplatData splat_data(sh_degree_, means_, sh0, shN, scales_, rotations_, opacities_, 1.0f);
    splat_data.set_active_sh_degree(sh_degree_);

    auto output = gs::training::rasterize(
        *camera_, splat_data, bg_color_,
        1.0f, false, false,
        gs::training::RenderMode::RGB,
        nullptr);

    ASSERT_TRUE(output.image.defined()) << "RGB image should be defined";
    ASSERT_EQ(output.image.size(0), 3) << "RGB should have 3 channels";
    ASSERT_EQ(output.image.size(1), image_height_);
    ASSERT_EQ(output.image.size(2), image_width_);
    ASSERT_FALSE(output.depth.defined()) << "Depth should not be defined in RGB mode";
}

// Test D mode (accumulated depth)
TEST_F(RenderModeTest, DMode) {
    auto sh0 = sh_coeffs_.index({torch::indexing::Slice(), 0, torch::indexing::Slice()}).unsqueeze(1);
    auto shN = sh_coeffs_.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None), torch::indexing::Slice()});

    gs::SplatData splat_data(sh_degree_, means_, sh0, shN, scales_, rotations_, opacities_, 1.0f);
    splat_data.set_active_sh_degree(sh_degree_);

    auto output = gs::training::rasterize(
        *camera_, splat_data, bg_color_,
        1.0f, false, false,
        gs::training::RenderMode::D,
        nullptr);

    ASSERT_FALSE(output.image.defined()) << "RGB image should not be defined in D mode";
    ASSERT_TRUE(output.depth.defined()) << "Depth should be defined in D mode";
    ASSERT_EQ(output.depth.size(0), 1) << "Depth should have 1 channel";
    ASSERT_EQ(output.depth.size(1), image_height_);
    ASSERT_EQ(output.depth.size(2), image_width_);
}

// Test ED mode (expected depth)
TEST_F(RenderModeTest, EDMode) {
    auto sh0 = sh_coeffs_.index({torch::indexing::Slice(), 0, torch::indexing::Slice()}).unsqueeze(1);
    auto shN = sh_coeffs_.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None), torch::indexing::Slice()});

    gs::SplatData splat_data(sh_degree_, means_, sh0, shN, scales_, rotations_, opacities_, 1.0f);
    splat_data.set_active_sh_degree(sh_degree_);

    auto output = gs::training::rasterize(
        *camera_, splat_data, bg_color_,
        1.0f, false, false,
        gs::training::RenderMode::ED,
        nullptr);

    ASSERT_FALSE(output.image.defined()) << "RGB image should not be defined in ED mode";
    ASSERT_TRUE(output.depth.defined()) << "Depth should be defined in ED mode";
    ASSERT_EQ(output.depth.size(0), 1) << "Depth should have 1 channel";
}

// Test RGB_D mode
TEST_F(RenderModeTest, RGB_DMode) {
    auto sh0 = sh_coeffs_.index({torch::indexing::Slice(), 0, torch::indexing::Slice()}).unsqueeze(1);
    auto shN = sh_coeffs_.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None), torch::indexing::Slice()});

    gs::SplatData splat_data(sh_degree_, means_, sh0, shN, scales_, rotations_, opacities_, 1.0f);
    splat_data.set_active_sh_degree(sh_degree_);

    auto output = gs::training::rasterize(
        *camera_, splat_data, bg_color_,
        1.0f, false, false,
        gs::training::RenderMode::RGB_D,
        nullptr);

    ASSERT_TRUE(output.image.defined()) << "RGB image should be defined in RGB_D mode";
    ASSERT_TRUE(output.depth.defined()) << "Depth should be defined in RGB_D mode";
    ASSERT_EQ(output.image.size(0), 3) << "RGB should have 3 channels";
    ASSERT_EQ(output.depth.size(0), 1) << "Depth should have 1 channel";
}

// Test RGB_ED mode
TEST_F(RenderModeTest, RGB_EDMode) {
    auto sh0 = sh_coeffs_.index({torch::indexing::Slice(), 0, torch::indexing::Slice()}).unsqueeze(1);
    auto shN = sh_coeffs_.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None), torch::indexing::Slice()});

    gs::SplatData splat_data(sh_degree_, means_, sh0, shN, scales_, rotations_, opacities_, 1.0f);
    splat_data.set_active_sh_degree(sh_degree_);

    auto output = gs::training::rasterize(
        *camera_, splat_data, bg_color_,
        1.0f, false, false,
        gs::training::RenderMode::RGB_ED,
        nullptr);

    ASSERT_TRUE(output.image.defined()) << "RGB image should be defined in RGB_ED mode";
    ASSERT_TRUE(output.depth.defined()) << "Depth should be defined in RGB_ED mode";
    ASSERT_EQ(output.image.size(0), 3) << "RGB should have 3 channels";
    ASSERT_EQ(output.depth.size(0), 1) << "Depth should have 1 channel";
}
