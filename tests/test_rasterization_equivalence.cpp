/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <random>

#include "rasterization/rasterizer.hpp"
#include "core/camera.hpp"
#include "core/splat_data.hpp"

constexpr int RANDOM_SEED = 42;
constexpr float TOLERANCE = 1e-4f;  // Tolerance for floating point comparison

class RasterizationEquivalenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set random seed for reproducibility
        torch::manual_seed(RANDOM_SEED);

        // Ensure CUDA is available
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA is not available";

        // Create test data
        setupTestData();
    }

    void setupTestData() {
        // Camera parameters
        image_width_ = 256;
        image_height_ = 256;
        fx_ = 200.0f;
        fy_ = 200.0f;
        cx_ = 128.0f;
        cy_ = 128.0f;

        // Number of Gaussians
        num_gaussians_ = 100;

        // Use CUDA device
        auto device = torch::kCUDA;

        // Create random Gaussian parameters
        means_ = torch::randn({num_gaussians_, 3}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
        means_.index_put_({torch::indexing::Slice(), 2}, means_.index({torch::indexing::Slice(), 2}).abs() + 2.0f); // Ensure positive z

        scales_ = torch::randn({num_gaussians_, 3}, torch::TensorOptions().device(device).dtype(torch::kFloat32)) * 0.1f + 0.5f;
        scales_ = scales_.abs(); // Ensure positive scales

        rotations_ = torch::randn({num_gaussians_, 4}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
        rotations_ = torch::nn::functional::normalize(rotations_, torch::nn::functional::NormalizeFuncOptions().dim(1)); // Normalize quaternions

        opacities_ = torch::rand({num_gaussians_, 1}, torch::TensorOptions().device(device).dtype(torch::kFloat32)) * 0.8f + 0.2f;

        // SH coefficients (degree 3 = 16 bases)
        sh_degree_ = 3;
        int num_sh_bases = (sh_degree_ + 1) * (sh_degree_ + 1);
        sh_coeffs_ = torch::randn({num_gaussians_, num_sh_bases, 3}, torch::TensorOptions().device(device).dtype(torch::kFloat32)) * 0.1f;

        // Background color
        bg_color_ = torch::tensor({0.5f, 0.5f, 0.5f}, torch::TensorOptions().device(device).dtype(torch::kFloat32));

        // Create camera - R and T from identity view matrix moved back 5 units in z
        auto R = torch::eye(3, torch::TensorOptions().device(device).dtype(torch::kFloat32));
        auto T = torch::tensor({0.0f, 0.0f, 5.0f}, torch::TensorOptions().device(device).dtype(torch::kFloat32));

        auto radial_dist = torch::zeros({4}, torch::TensorOptions().device(device).dtype(torch::kFloat32));
        auto tangential_dist = torch::zeros({2}, torch::TensorOptions().device(device).dtype(torch::kFloat32));

        camera_ = std::make_unique<gs::Camera>(
            R, T,
            fx_, fy_,
            cx_, cy_,
            radial_dist,
            tangential_dist,
            gsplat::CameraModelType::PINHOLE,
            "test_camera",
            "",
            image_width_,
            image_height_,
            0  // uid
        );
    }

    // Helper to check tensor equality with tolerance
    bool tensorsAlmostEqual(const torch::Tensor& a, const torch::Tensor& b, float tolerance = TOLERANCE) {
        if (!a.defined() || !b.defined()) {
            return a.defined() == b.defined();
        }
        if (a.sizes() != b.sizes()) {
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

// Test that the current fused rasterization produces consistent results
TEST_F(RasterizationEquivalenceTest, FusedRasterizationIsConsistent) {
    // Split sh_coeffs into sh0 and shN
    auto sh0 = sh_coeffs_.index({torch::indexing::Slice(), 0, torch::indexing::Slice()}).unsqueeze(1); // [N, 1, 3]
    auto shN = sh_coeffs_.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None), torch::indexing::Slice()}); // [N, K-1, 3]

    // Create SplatData with proper constructor
    gs::SplatData splat_data(
        sh_degree_,
        means_,
        sh0,
        shN,
        scales_,
        rotations_,
        opacities_,
        1.0f  // scene_scale
    );

    splat_data.set_active_sh_degree(sh_degree_);

    // Run rasterization twice and check consistency
    auto output1 = gs::training::rasterize(
        *camera_,
        splat_data,
        bg_color_,
        1.0f, // scaling_modifier
        false, // packed
        false, // antialiased
        gs::training::RenderMode::RGB,
        nullptr // bounding_box
    );

    auto output2 = gs::training::rasterize(
        *camera_,
        splat_data,
        bg_color_,
        1.0f,
        false,
        false,
        gs::training::RenderMode::RGB,
        nullptr
    );

    ASSERT_TRUE(tensorsAlmostEqual(output1.image, output2.image))
        << "Rasterization is not deterministic";
    ASSERT_TRUE(tensorsAlmostEqual(output1.alpha, output2.alpha))
        << "Alpha is not deterministic";
}

// Test forward pass equivalence (will be enabled after implementing fused version)
TEST_F(RasterizationEquivalenceTest, DISABLED_FusedVsMultiStepForward) {
    // Split sh_coeffs into sh0 and shN
    auto sh0 = sh_coeffs_.index({torch::indexing::Slice(), 0, torch::indexing::Slice()}).unsqueeze(1); // [N, 1, 3]
    auto shN = sh_coeffs_.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None), torch::indexing::Slice()}); // [N, K-1, 3]

    // Create SplatData with proper constructor
    gs::SplatData splat_data(
        sh_degree_,
        means_,
        sh0,
        shN,
        scales_,
        rotations_,
        opacities_,
        1.0f  // scene_scale
    );

    splat_data.set_active_sh_degree(sh_degree_);

    // Current multi-step approach
    auto output_multistep = gs::training::rasterize(
        *camera_,
        splat_data,
        bg_color_,
        1.0f,
        false,
        false,
        gs::training::RenderMode::RGB,
        nullptr
    );

    // TODO: Call new fused version here
    // auto output_fused = gs::training::rasterize_fused(...);

    // Compare outputs
    // ASSERT_TRUE(tensorsAlmostEqual(output_multistep.image, output_fused.image))
    //     << "Fused and multi-step forward passes produce different images";
    // ASSERT_TRUE(tensorsAlmostEqual(output_multistep.alpha, output_fused.alpha))
    //     << "Fused and multi-step forward passes produce different alpha";
}

// Test backward pass equivalence
TEST_F(RasterizationEquivalenceTest, DISABLED_FusedVsMultiStepBackward) {
    // Enable gradient tracking
    means_.requires_grad_(true);
    scales_.requires_grad_(true);
    rotations_.requires_grad_(true);
    opacities_.requires_grad_(true);
    sh_coeffs_.requires_grad_(true);

    // Split sh_coeffs into sh0 and shN
    auto sh0 = sh_coeffs_.index({torch::indexing::Slice(), 0, torch::indexing::Slice()}).unsqueeze(1); // [N, 1, 3]
    auto shN = sh_coeffs_.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None), torch::indexing::Slice()}); // [N, K-1, 3]

    // Create SplatData with proper constructor
    gs::SplatData splat_data(
        sh_degree_,
        means_,
        sh0,
        shN,
        scales_,
        rotations_,
        opacities_,
        1.0f  // scene_scale
    );

    splat_data.set_active_sh_degree(sh_degree_);

    // Multi-step approach
    auto output_multistep = gs::training::rasterize(
        *camera_,
        splat_data,
        bg_color_,
        1.0f,
        false,
        false,
        gs::training::RenderMode::RGB,
        nullptr
    );

    auto loss_multistep = output_multistep.image.sum();
    loss_multistep.backward();

    auto grad_means_multistep = means_.grad().clone();
    auto grad_scales_multistep = scales_.grad().clone();
    auto grad_rotations_multistep = rotations_.grad().clone();
    auto grad_opacities_multistep = opacities_.grad().clone();
    auto grad_sh_multistep = sh_coeffs_.grad().clone();

    // Clear gradients
    means_.grad().zero_();
    scales_.grad().zero_();
    rotations_.grad().zero_();
    opacities_.grad().zero_();
    sh_coeffs_.grad().zero_();

    // TODO: Call fused version and compare gradients
    // auto output_fused = gs::training::rasterize_fused(...);
    // auto loss_fused = output_fused.image.sum();
    // loss_fused.backward();

    // ASSERT_TRUE(tensorsAlmostEqual(means_.grad(), grad_means_multistep))
    //     << "Gradients for means differ between fused and multi-step";
    // ASSERT_TRUE(tensorsAlmostEqual(scales_.grad(), grad_scales_multistep))
    //     << "Gradients for scales differ between fused and multi-step";
    // ASSERT_TRUE(tensorsAlmostEqual(rotations_.grad(), grad_rotations_multistep))
    //     << "Gradients for rotations differ between fused and multi-step";
    // ASSERT_TRUE(tensorsAlmostEqual(opacities_.grad(), grad_opacities_multistep))
    //     << "Gradients for opacities differ between fused and multi-step";
    // ASSERT_TRUE(tensorsAlmostEqual(sh_coeffs_.grad(), grad_sh_multistep))
    //     << "Gradients for SH coefficients differ between fused and multi-step";
}

// Test different render modes
TEST_F(RasterizationEquivalenceTest, DISABLED_FusedVsMultiStepDifferentModes) {
    // Split sh_coeffs into sh0 and shN
    auto sh0 = sh_coeffs_.index({torch::indexing::Slice(), 0, torch::indexing::Slice()}).unsqueeze(1); // [N, 1, 3]
    auto shN = sh_coeffs_.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None), torch::indexing::Slice()}); // [N, K-1, 3]

    // Create SplatData with proper constructor
    gs::SplatData splat_data(
        sh_degree_,
        means_,
        sh0,
        shN,
        scales_,
        rotations_,
        opacities_,
        1.0f  // scene_scale
    );

    splat_data.set_active_sh_degree(sh_degree_);

    // Test different render modes
    std::vector<gs::training::RenderMode> modes = {
        gs::training::RenderMode::RGB,
        gs::training::RenderMode::D,
        gs::training::RenderMode::ED,
        gs::training::RenderMode::RGB_D,
        gs::training::RenderMode::RGB_ED
    };

    for (auto mode : modes) {
        auto output_multistep = gs::training::rasterize(
            *camera_,
            splat_data,
            bg_color_,
            1.0f,
            false,
            false,
            mode,
            nullptr
        );

        // TODO: Call fused version and compare
        // auto output_fused = gs::training::rasterize_fused(..., mode);
        // ASSERT_TRUE(tensorsAlmostEqual(output_multistep.image, output_fused.image))
        //     << "Outputs differ for render mode " << static_cast<int>(mode);
    }
}

// Performance benchmark test
TEST_F(RasterizationEquivalenceTest, DISABLED_PerformanceBenchmark) {
    // Split sh_coeffs into sh0 and shN
    auto sh0 = sh_coeffs_.index({torch::indexing::Slice(), 0, torch::indexing::Slice()}).unsqueeze(1); // [N, 1, 3]
    auto shN = sh_coeffs_.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None), torch::indexing::Slice()}); // [N, K-1, 3]

    // Create SplatData with proper constructor
    gs::SplatData splat_data(
        sh_degree_,
        means_,
        sh0,
        shN,
        scales_,
        rotations_,
        opacities_,
        1.0f  // scene_scale
    );

    splat_data.set_active_sh_degree(sh_degree_);

    const int num_iterations = 100;

    // Warm-up
    for (int i = 0; i < 10; ++i) {
        gs::training::rasterize(
            *camera_,
            splat_data,
            bg_color_,
            1.0f,
            false,
            false,
            gs::training::RenderMode::RGB,
            nullptr
        );
    }
    torch::cuda::synchronize();

    // Benchmark multi-step
    auto start_multistep = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        gs::training::rasterize(
            *camera_,
            splat_data,
            bg_color_,
            1.0f,
            false,
            false,
            gs::training::RenderMode::RGB,
            nullptr
        );
    }
    torch::cuda::synchronize();
    auto end_multistep = std::chrono::high_resolution_clock::now();
    auto duration_multistep = std::chrono::duration_cast<std::chrono::microseconds>(end_multistep - start_multistep).count();

    std::cout << "Multi-step rasterization: " << duration_multistep / num_iterations << " µs per iteration" << std::endl;

    // TODO: Benchmark fused version
    // auto start_fused = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < num_iterations; ++i) {
    //     rasterize_fused(...);
    // }
    // torch::cuda::synchronize();
    // auto end_fused = std::chrono::high_resolution_clock::now();
    // auto duration_fused = std::chrono::duration_cast<std::chrono::microseconds>(end_fused - start_fused).count();
    //
    // std::cout << "Fused rasterization: " << duration_fused / num_iterations << " µs per iteration" << std::endl;
    // std::cout << "Speedup: " << static_cast<double>(duration_multistep) / duration_fused << "x" << std::endl;
}
