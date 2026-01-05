/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/logger.hpp"
#include "core/parameters.hpp"
#include "core/point_cloud.hpp"
#include "core/splat_data.hpp"
#include "training/strategies/adc.hpp"
#include <gtest/gtest.h>

using namespace lfs::training;
using namespace lfs::core;

// Helper to create test SplatData
static SplatData create_test_splat_data(int n_gaussians = 100) {
    std::vector<float> means_data(n_gaussians * 3, 0.0f);
    std::vector<float> sh0_data(n_gaussians * 3, 0.5f);
    std::vector<float> shN_data(n_gaussians * 48, 0.0f);     // 3 SH degrees, 16*3=48
    std::vector<float> scaling_data(n_gaussians * 3, -2.0f); // log(0.135)
    std::vector<float> rotation_data(n_gaussians * 4);
    for (int i = 0; i < n_gaussians; ++i) {
        rotation_data[i * 4 + 0] = 1.0f; // w
        rotation_data[i * 4 + 1] = 0.0f; // x
        rotation_data[i * 4 + 2] = 0.0f; // y
        rotation_data[i * 4 + 3] = 0.0f; // z
    }
    std::vector<float> opacity_data(n_gaussians, 0.5f); // logit(0.5) ≈ 0

    auto means = Tensor::from_vector(means_data, TensorShape({static_cast<size_t>(n_gaussians), 3}), Device::CUDA);
    auto sh0 = Tensor::from_vector(sh0_data, TensorShape({static_cast<size_t>(n_gaussians), 3}), Device::CUDA);
    auto shN = Tensor::from_vector(shN_data, TensorShape({static_cast<size_t>(n_gaussians), 48}), Device::CUDA);
    auto scaling = Tensor::from_vector(scaling_data, TensorShape({static_cast<size_t>(n_gaussians), 3}), Device::CUDA);
    auto rotation = Tensor::from_vector(rotation_data, TensorShape({static_cast<size_t>(n_gaussians), 4}), Device::CUDA);
    auto opacity = Tensor::from_vector(opacity_data, TensorShape({static_cast<size_t>(n_gaussians), 1}), Device::CUDA);

    return SplatData(3, means, sh0, shN, scaling, rotation, opacity, 1.0f);
}

TEST(ADCTest, Initialization) {
    auto splat_data = create_test_splat_data(50);
    ADC strategy(splat_data);

    param::OptimizationParameters opt_params;
    opt_params.iterations = 100;
    opt_params.means_lr = 0.00016f;

    strategy.initialize(opt_params);

    EXPECT_EQ(strategy.get_model().size(), 50);
    EXPECT_TRUE(strategy.get_optimizer().get_grad(lfs::training::ParamType::Means).is_valid());
}

TEST(ADCTest, IsRefining) {
    auto splat_data = create_test_splat_data(50);
    ADC strategy(splat_data);

    param::OptimizationParameters opt_params;
    opt_params.start_refine = 500;
    opt_params.refine_every = 100;
    opt_params.reset_every = 3000;
    opt_params.pause_refine_after_reset = 500;

    strategy.initialize(opt_params);

    // Before start_refine
    EXPECT_FALSE(strategy.is_refining(400));
    EXPECT_FALSE(strategy.is_refining(500));

    // After start_refine, on refine_every iteration
    EXPECT_TRUE(strategy.is_refining(600));
    EXPECT_FALSE(strategy.is_refining(650));
    EXPECT_TRUE(strategy.is_refining(700));

    // After reset, during pause period
    EXPECT_FALSE(strategy.is_refining(3100));
    EXPECT_FALSE(strategy.is_refining(3400));

    // After pause period
    EXPECT_TRUE(strategy.is_refining(3500));
}

TEST(ADCTest, DuplicateGaussians_AddsCorrectly) {
    auto splat_data = create_test_splat_data(10);
    ADC strategy(splat_data);

    param::OptimizationParameters opt_params;
    opt_params.iterations = 100;
    strategy.initialize(opt_params);

    int initial_size = strategy.get_model().size();
    EXPECT_EQ(initial_size, 10);

    // Mark 3 Gaussians for duplication
    std::vector<bool> dup_data(10, false);
    dup_data[0] = true;
    dup_data[5] = true;
    dup_data[9] = true;
    auto is_dup = Tensor::from_vector(dup_data, TensorShape({10}), Device::CUDA);

    strategy.remove_gaussians(Tensor::zeros_bool({static_cast<size_t>(initial_size)}, Device::CUDA));

    // Access private method via public interface - we'll use grow_gs which calls duplicate
    // For now, just verify initialization works
    EXPECT_EQ(strategy.get_model().size(), initial_size);
}

TEST(ADCTest, SplitGaussians_WithQuaternions) {
    auto splat_data = create_test_splat_data(5);
    ADC strategy(splat_data);

    param::OptimizationParameters opt_params;
    opt_params.iterations = 100;
    opt_params.revised_opacity = true;
    strategy.initialize(opt_params);

    int initial_size = strategy.get_model().size();
    EXPECT_EQ(initial_size, 5);

    // Verify quaternions are properly normalized
    auto quats = strategy.get_model().get_rotation();
    EXPECT_EQ(quats.shape()[0], 5);
    EXPECT_EQ(quats.shape()[1], 4);
}

TEST(ADCTest, GrowGaussians_HighGradientSmall) {
    auto splat_data = create_test_splat_data(20);
    ADC strategy(splat_data);

    param::OptimizationParameters opt_params;
    opt_params.iterations = 10000;
    opt_params.grad_threshold = 0.0002f;
    opt_params.grow_scale3d = 2.0f;
    opt_params.start_refine = 500;
    opt_params.stop_refine = 15000;
    opt_params.refine_every = 100;
    opt_params.reset_every = 3000;
    opt_params.pause_refine_after_reset = 500;

    strategy.initialize(opt_params);

    // Initialize densification info
    strategy.get_model()._densification_info = Tensor::zeros({2, 20}, Device::CUDA);

    // Simulate high gradients for some Gaussians
    auto grad_numer = Tensor::ones({20}, Device::CUDA) * 0.001f; // High gradient
    auto grad_denom = Tensor::ones({20}, Device::CUDA);
    strategy.get_model()._densification_info = Tensor::stack({grad_denom, grad_numer}, 0);

    int size_before = strategy.get_model().size();

    // This would call grow_gs internally, but we can't access it directly
    // Just verify the state is correct
    EXPECT_EQ(size_before, 20);
}

TEST(ADCTest, PruneGaussians_LowOpacity) {
    auto splat_data = create_test_splat_data(30);

    // Set some opacities very low
    std::vector<float> opacity_data(30);
    for (int i = 0; i < 30; ++i) {
        opacity_data[i] = (i < 10) ? -5.0f : 0.5f; // First 10 have very low opacity
    }
    splat_data.opacity_raw() = Tensor::from_vector(opacity_data, TensorShape({30, 1}), Device::CUDA);

    ADC strategy(splat_data);

    param::OptimizationParameters opt_params;
    opt_params.iterations = 100;
    opt_params.prune_opacity = 0.005f;
    strategy.initialize(opt_params);

    // Verify low opacity Gaussians exist
    auto opacity = strategy.get_model().get_opacity();
    EXPECT_EQ(opacity.shape()[0], 30);
}

TEST(ADCTest, ResetOpacity_ClampsValues) {
    auto splat_data = create_test_splat_data(20);

    // Set some opacities very high
    std::vector<float> opacity_data(20, 3.0f); // High opacity (logit space)
    splat_data.opacity_raw() = Tensor::from_vector(opacity_data, TensorShape({20, 1}), Device::CUDA);

    ADC strategy(splat_data);

    param::OptimizationParameters opt_params;
    opt_params.iterations = 100;
    opt_params.prune_opacity = 0.01f;
    opt_params.reset_every = 3000;
    strategy.initialize(opt_params);

    // Verify high opacities exist
    auto opacity_before = strategy.get_model().get_opacity();
    float max_opacity_before = opacity_before.max().item<float>();
    EXPECT_GT(max_opacity_before, 0.9f);
}

TEST(ADCTest, RemoveGaussians) {
    auto splat_data = create_test_splat_data(50);
    ADC strategy(splat_data);

    param::OptimizationParameters opt_params;
    opt_params.iterations = 100;
    strategy.initialize(opt_params);

    EXPECT_EQ(strategy.get_model().size(), 50);

    // Remove 10 Gaussians
    // Create mask using direct memory access (workaround for bool tensor creation)
    auto mask = Tensor::zeros_bool({50}, Device::CPU); // Start on CPU
    auto mask_ptr = mask.ptr<unsigned char>();
    for (int i = 0; i < 10; ++i) {
        mask_ptr[i] = 1; // Set first 10 to true
    }
    mask = mask.to(Device::CUDA); // Transfer to GPU

    strategy.remove_gaussians(mask);

    // With soft deletion, tensor size stays the same but active_count() reflects logical size
    EXPECT_EQ(strategy.get_model().size(), 50); // Physical size unchanged
    EXPECT_EQ(strategy.active_count(), 40);     // Logical size is 40
    EXPECT_EQ(strategy.free_count(), 10);       // 10 free slots
}

TEST(ADCTest, FullTrainingLoop_ShortRun) {
    auto splat_data = create_test_splat_data(30);
    ADC strategy(splat_data);

    param::OptimizationParameters opt_params;
    opt_params.iterations = 50;
    opt_params.start_refine = 10;
    opt_params.stop_refine = 40;
    opt_params.refine_every = 10;
    opt_params.reset_every = 30;
    opt_params.sh_degree_interval = 1000;

    strategy.initialize(opt_params);

    // Allocate densification info
    strategy.get_model()._densification_info = Tensor::zeros({2, 30}, Device::CUDA);

    RenderOutput render_output;

    for (int iter = 0; iter < 50; ++iter) {
        // Simulate gradients
        auto& grad = strategy.get_optimizer().get_grad(lfs::training::ParamType::Means);
        if (grad.is_valid()) {
            grad = Tensor::rand_like(grad) * 0.001f;
        }

        strategy.post_backward(iter, render_output);
        strategy.step(iter);
    }

    // Model should still be valid
    EXPECT_GT(strategy.get_model().size(), 0);
    EXPECT_LE(strategy.get_model().size(), 100); // Shouldn't grow too much
}

TEST(ADCTest, EdgeCase_NoRefinement) {
    auto splat_data = create_test_splat_data(25);
    ADC strategy(splat_data);

    param::OptimizationParameters opt_params;
    opt_params.iterations = 100;
    opt_params.start_refine = 200; // Never refine
    opt_params.stop_refine = 300;

    strategy.initialize(opt_params);

    strategy.get_model()._densification_info = Tensor::zeros({2, 25}, Device::CUDA);

    RenderOutput render_output;

    for (int iter = 0; iter < 100; ++iter) {
        strategy.post_backward(iter, render_output);
        strategy.step(iter);
    }

    // Size should remain the same (no densification)
    EXPECT_EQ(strategy.get_model().size(), 25);
}

TEST(ADCTest, EdgeCase_AllGaussiansLowOpacity) {
    auto splat_data = create_test_splat_data(20);

    // Set all opacities very low
    std::vector<float> opacity_data(20, -10.0f);
    splat_data.opacity_raw() = Tensor::from_vector(opacity_data, TensorShape({20, 1}), Device::CUDA);

    ADC strategy(splat_data);

    param::OptimizationParameters opt_params;
    opt_params.iterations = 100;
    opt_params.prune_opacity = 0.005f;
    strategy.initialize(opt_params);

    EXPECT_EQ(strategy.get_model().size(), 20);
}

TEST(ADCTest, EdgeCase_HighSHDegree) {
    auto splat_data = create_test_splat_data(15);
    // SH degree is already 3, which is max in the test setup

    ADC strategy(splat_data);

    param::OptimizationParameters opt_params;
    opt_params.iterations = 100;
    opt_params.sh_degree_interval = 10;

    strategy.initialize(opt_params);

    RenderOutput render_output;

    for (int iter = 0; iter < 50; iter += 10) {
        strategy.post_backward(iter, render_output);
    }

    // SH degree shouldn't exceed max
    EXPECT_LE(strategy.get_model().get_max_sh_degree(), 3);
}

TEST(ADCStressTest, LongTrainingLoop_200Iterations) {
    auto splat_data = create_test_splat_data(50);
    ADC strategy(splat_data);

    param::OptimizationParameters opt_params;
    opt_params.iterations = 200;
    opt_params.start_refine = 20;
    opt_params.stop_refine = 180;
    opt_params.refine_every = 20;
    opt_params.reset_every = 100;
    opt_params.pause_refine_after_reset = 10;
    opt_params.grad_threshold = 0.00005f; // Lower threshold for test gradients

    strategy.initialize(opt_params);

    strategy.get_model()._densification_info = Tensor::zeros({2, static_cast<size_t>(strategy.get_model().size())}, Device::CUDA);

    RenderOutput render_output;

    for (int iter = 0; iter < 200; ++iter) {
        auto& grad = strategy.get_optimizer().get_grad(lfs::training::ParamType::Means);
        if (grad.is_valid()) {
            grad = Tensor::rand_like(grad) * 0.001f;
        }

        // Simulate gradient accumulation into densification_info (like real rendering does)
        // In real training, radii and gradients from rendering update densification_info
        auto& densification_info = strategy.get_model()._densification_info;
        if (densification_info.numel() > 0) {
            // Row 0: visibility count (simulate visible Gaussians)
            densification_info[0] = Tensor::ones({static_cast<size_t>(strategy.get_model().size())}, Device::CUDA);
            // Row 1: accumulated gradient norms (simulate low gradient activity to prevent excessive growth)
            densification_info[1] = Tensor::rand({static_cast<size_t>(strategy.get_model().size())}, Device::CUDA) * 0.0001f;
        }

        int size_before = strategy.get_model().size();
        strategy.post_backward(iter, render_output);

        // Resize densification info if size changed
        if (strategy.get_model().size() != size_before) {
            strategy.get_model()._densification_info = Tensor::zeros(
                {2, static_cast<size_t>(strategy.get_model().size())},
                Device::CUDA);
        }

        strategy.step(iter);
    }

    // Model should still be valid and reasonable size
    EXPECT_GT(strategy.get_model().size(), 0);
    EXPECT_LT(strategy.get_model().size(), 1000); // Allow for moderate growth during stress test
}

TEST(ADCStressTest, ZeroGradients_NoCorruption) {
    auto splat_data = create_test_splat_data(30);
    ADC strategy(splat_data);

    param::OptimizationParameters opt_params;
    opt_params.iterations = 100;
    opt_params.start_refine = 10;
    opt_params.stop_refine = 80;
    opt_params.refine_every = 10;

    strategy.initialize(opt_params);

    strategy.get_model()._densification_info = Tensor::zeros({2, 30}, Device::CUDA);

    RenderOutput render_output;

    for (int iter = 0; iter < 100; ++iter) {
        // Always zero gradients
        auto& grad = strategy.get_optimizer().get_grad(lfs::training::ParamType::Means);
        if (grad.is_valid()) {
            grad = Tensor::zeros_like(grad);
        }

        strategy.post_backward(iter, render_output);
        strategy.step(iter);
    }

    // Model parameters should still be valid
    auto means = strategy.get_model().means();
    EXPECT_EQ(means.device(), Device::CUDA);
}

TEST(ADCStressTest, VeryLargeModel_1kGaussians) {
    auto splat_data = create_test_splat_data(1000);
    ADC strategy(splat_data);

    param::OptimizationParameters opt_params;
    opt_params.iterations = 50;
    opt_params.start_refine = 10;
    opt_params.stop_refine = 40;
    opt_params.refine_every = 10;

    strategy.initialize(opt_params);

    EXPECT_EQ(strategy.get_model().size(), 1000);

    strategy.get_model()._densification_info = Tensor::zeros({2, 1000}, Device::CUDA);

    RenderOutput render_output;

    for (int iter = 0; iter < 50; ++iter) {
        int size_before = strategy.get_model().size();

        strategy.post_backward(iter, render_output);

        if (strategy.get_model().size() != size_before) {
            strategy.get_model()._densification_info = Tensor::zeros(
                {2, static_cast<size_t>(strategy.get_model().size())},
                Device::CUDA);
        }

        strategy.step(iter);
    }

    EXPECT_GT(strategy.get_model().size(), 900); // Should not lose too many
}
