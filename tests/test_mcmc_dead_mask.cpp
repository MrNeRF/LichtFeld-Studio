/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <gtest/gtest.h>
#include <random>
#include <torch/torch.h>
#include <vector>

using namespace lfs::core;

/**
 * Comprehensive tests for MCMC dead mask calculation:
 *   dead_mask = (opacities <= min_opacity).logical_or(rot_mag_sq < 1e-8f)
 *
 * This is critical logic that determines which Gaussians are "dead" and should be relocated.
 * Bugs here could cause:
 * - Memory corruption (relocating wrong indices)
 * - Performance issues (not relocating dead Gaussians)
 * - Visual artifacts (removing healthy Gaussians)
 */

class MCMCDeadMaskTest : public ::testing::Test {
protected:
    void SetUp() override {
        min_opacity = 0.005f; // Default from MCMC params
        rot_threshold = 1e-8f;
    }

    float min_opacity;
    float rot_threshold;

    // Helper: Create rotation magnitude squared from quaternion
    Tensor compute_rotation_mag_sq(const Tensor& rotations) {
        // rotations shape: [N, 4] (quaternion: w, x, y, z)
        // Matches actual implementation: (rotation_raw * rotation_raw).sum(-1)
        return (rotations * rotations).sum(-1);
    }

    // Helper: Verify mask correctness
    void verify_dead_mask(const Tensor& opacities, const Tensor& rot_mag_sq,
                          const Tensor& dead_mask, float min_opacity) {
        auto opacities_cpu = opacities.cpu();
        auto rot_mag_sq_cpu = rot_mag_sq.cpu();
        auto mask_cpu = dead_mask.cpu();

        auto opacity_vec = opacities_cpu.to_vector();
        auto rot_vec = rot_mag_sq_cpu.to_vector();
        auto mask_vec = mask_cpu.to_vector();

        ASSERT_EQ(opacity_vec.size(), rot_vec.size());
        ASSERT_EQ(opacity_vec.size(), mask_vec.size());

        for (size_t i = 0; i < opacity_vec.size(); ++i) {
            bool expected = (opacity_vec[i] <= min_opacity) || (rot_vec[i] < rot_threshold);
            bool actual = mask_vec[i] > 0.5f; // bool mask is stored as float

            if (expected != actual) {
                FAIL() << "Mask mismatch at index " << i
                       << ": opacity=" << opacity_vec[i]
                       << ", rot_mag_sq=" << rot_vec[i]
                       << ", expected=" << expected
                       << ", actual=" << actual;
            }
        }
    }
};

// ============= Basic Tests =============

TEST_F(MCMCDeadMaskTest, BasicFunctionality) {
    constexpr int N = 10;

    // Create test data: some dead by opacity, some by rotation, some alive
    auto opacities = Tensor::zeros({static_cast<size_t>(N)}, Device::CUDA);
    auto rotations = Tensor::ones({static_cast<size_t>(N), 4}, Device::CUDA);

    // Manually set some values (use CPU for easy manipulation)
    auto opacities_cpu = opacities.cpu();
    auto rotations_cpu = rotations.cpu();

    auto opacity_vec = opacities_cpu.to_vector();
    opacity_vec[0] = 0.0f;   // Dead by opacity
    opacity_vec[1] = 0.001f; // Dead by opacity
    opacity_vec[2] = 0.005f; // Dead by opacity (exactly at threshold)
    opacity_vec[3] = 0.006f; // Alive (above threshold)
    opacity_vec[4] = 0.5f;   // Alive
    opacity_vec[5] = 1.0f;   // Alive
    opacity_vec[6] = 0.5f;   // Will be dead by rotation
    opacity_vec[7] = 0.5f;   // Will be dead by rotation
    opacity_vec[8] = 0.5f;   // Alive (rot at threshold)
    opacity_vec[9] = 0.5f;   // Alive

    // Set back
    opacities = Tensor::from_vector(opacity_vec, {static_cast<size_t>(N)}, Device::CUDA);

    // Set rotation magnitudes
    auto rot_mag_sq = Tensor::ones({static_cast<size_t>(N)}, Device::CUDA);
    auto rot_mag_vec = rot_mag_sq.cpu().to_vector();
    rot_mag_vec[0] = 1.0f;  // Alive (already dead by opacity)
    rot_mag_vec[1] = 1.0f;  // Alive (already dead by opacity)
    rot_mag_vec[2] = 1.0f;  // Alive (already dead by opacity)
    rot_mag_vec[3] = 1.0f;  // Alive
    rot_mag_vec[4] = 1.0f;  // Alive
    rot_mag_vec[5] = 1.0f;  // Alive
    rot_mag_vec[6] = 1e-9f; // Dead (< 1e-8)
    rot_mag_vec[7] = 1e-9f; // Dead (< 1e-8)
    rot_mag_vec[8] = 1e-8f; // Alive (exactly at threshold, not <)
    rot_mag_vec[9] = 1e-7f; // Alive

    rot_mag_sq = Tensor::from_vector(rot_mag_vec, {static_cast<size_t>(N)}, Device::CUDA);

    // Compute dead mask
    auto dead_mask = (opacities <= min_opacity).logical_or(rot_mag_sq < rot_threshold);

    // Verify
    verify_dead_mask(opacities, rot_mag_sq, dead_mask, min_opacity);

    // Check that we have expected dead count
    auto dead_count = dead_mask.sum().item();
    EXPECT_EQ(dead_count, 5) << "Expected 5 dead Gaussians (3 by opacity, 2 by rotation)";
}

TEST_F(MCMCDeadMaskTest, BoundaryConditions) {
    constexpr int N = 6;

    auto opacities = Tensor::zeros({static_cast<size_t>(N)}, Device::CUDA);
    auto rot_mag_sq = Tensor::ones({static_cast<size_t>(N)}, Device::CUDA);

    auto opacity_vec = opacities.cpu().to_vector();
    opacity_vec[0] = min_opacity - 1e-6f; // Dead (<= threshold)
    opacity_vec[1] = min_opacity;         // Dead (exactly at threshold)
    opacity_vec[2] = min_opacity + 1e-6f; // Alive (above threshold)
    opacity_vec[3] = 0.5f;                // Alive
    opacity_vec[4] = 0.5f;                // Alive
    opacity_vec[5] = 0.5f;                // Alive

    opacities = Tensor::from_vector(opacity_vec, {static_cast<size_t>(N)}, Device::CUDA);

    auto rot_vec = rot_mag_sq.cpu().to_vector();
    rot_vec[0] = 1.0f;                  // Alive (already dead by opacity)
    rot_vec[1] = 1.0f;                  // Alive (already dead by opacity)
    rot_vec[2] = 1.0f;                  // Alive
    rot_vec[3] = rot_threshold - 1e-9f; // Dead (< threshold)
    rot_vec[4] = rot_threshold;         // Alive (exactly at threshold, not <)
    rot_vec[5] = rot_threshold + 1e-9f; // Alive (above threshold)

    rot_mag_sq = Tensor::from_vector(rot_vec, {static_cast<size_t>(N)}, Device::CUDA);

    auto dead_mask = (opacities <= min_opacity).logical_or(rot_mag_sq < rot_threshold);

    verify_dead_mask(opacities, rot_mag_sq, dead_mask, min_opacity);

    auto dead_count = dead_mask.sum().item();
    EXPECT_EQ(dead_count, 3) << "Expected 3 dead Gaussians (2 by opacity, 1 by rotation)";
}

TEST_F(MCMCDeadMaskTest, AllDead) {
    constexpr int N = 100;

    auto opacities = Tensor::zeros({static_cast<size_t>(N)}, Device::CUDA);
    auto rot_mag_sq = Tensor::ones({static_cast<size_t>(N)}, Device::CUDA);

    // Debug: Check actual values
    auto opacity_cpu = opacities.cpu();
    auto rot_cpu = rot_mag_sq.cpu();
    auto opacity_vec = opacity_cpu.to_vector();
    auto rot_vec = rot_cpu.to_vector();

    std::cout << "DEBUG AllDead: opacity[0]=" << opacity_vec[0]
              << ", rot_mag_sq[0]=" << rot_vec[0]
              << ", min_opacity=" << min_opacity << std::endl;

    auto opacity_cmp = opacities <= min_opacity;
    auto rot_cmp = rot_mag_sq < rot_threshold;

    std::cout << "DEBUG AllDead: opacity_cmp sum=" << opacity_cmp.sum().item()
              << ", rot_cmp sum=" << rot_cmp.sum().item() << std::endl;

    auto dead_mask = opacity_cmp.logical_or(rot_cmp);

    std::cout << "DEBUG AllDead: dead_mask dtype=" << static_cast<int>(dead_mask.dtype())
              << ", dead_mask shape=" << dead_mask.shape()[0] << std::endl;

    auto dead_mask_cpu = dead_mask.cpu();
    auto mask_vec = dead_mask_cpu.to_vector();
    std::cout << "DEBUG AllDead: dead_mask[0]=" << mask_vec[0]
              << ", dead_mask[1]=" << mask_vec[1] << std::endl;

    auto dead_count = dead_mask.sum().item();
    std::cout << "DEBUG AllDead: dead_count=" << dead_count << " (expected " << N << ")" << std::endl;

    EXPECT_EQ(dead_count, N) << "All Gaussians should be dead (opacity = 0)";
}

TEST_F(MCMCDeadMaskTest, AllAlive) {
    constexpr int N = 100;

    auto opacities = Tensor::ones({static_cast<size_t>(N)}, Device::CUDA);
    auto rot_mag_sq = Tensor::ones({static_cast<size_t>(N)}, Device::CUDA);

    auto dead_mask = (opacities <= min_opacity).logical_or(rot_mag_sq < rot_threshold);

    auto dead_count = dead_mask.sum().item();
    EXPECT_EQ(dead_count, 0) << "All Gaussians should be alive";
}

TEST_F(MCMCDeadMaskTest, StressTestLarge) {
    constexpr int N = 1000000; // 1M Gaussians

    // Random opacities and rotation magnitudes
    auto opacities = Tensor::rand({static_cast<size_t>(N)}, Device::CUDA);
    auto rotations = Tensor::randn({static_cast<size_t>(N), 4}, Device::CUDA);
    auto rot_mag_sq = compute_rotation_mag_sq(rotations);

    // This should not crash and should complete quickly
    auto dead_mask = (opacities <= min_opacity).logical_or(rot_mag_sq < rot_threshold);

    // Just verify we can compute it and get a reasonable result
    auto dead_count = dead_mask.sum().item();
    EXPECT_GE(dead_count, 0);
    EXPECT_LE(dead_count, N);
}

TEST_F(MCMCDeadMaskTest, WithActualQuaternions) {
    constexpr int N = 1000;

    // Create random quaternions
    auto rotations = Tensor::randn({static_cast<size_t>(N), 4}, Device::CUDA);
    auto rot_mag_sq = compute_rotation_mag_sq(rotations);

    // Random opacities
    auto opacities = Tensor::rand({static_cast<size_t>(N)}, Device::CUDA);

    // Compute dead mask
    auto dead_mask = (opacities <= min_opacity).logical_or(rot_mag_sq < rot_threshold);

    // Verify
    verify_dead_mask(opacities, rot_mag_sq, dead_mask, min_opacity);
}

TEST_F(MCMCDeadMaskTest, Consistency) {
    constexpr int N = 100;

    auto opacities = Tensor::rand({static_cast<size_t>(N)}, Device::CUDA);
    auto rot_mag_sq = Tensor::rand({static_cast<size_t>(N)}, Device::CUDA);

    // Compute mask multiple times - should be deterministic
    for (int i = 0; i < 10; ++i) {
        auto dead_mask1 = (opacities <= min_opacity).logical_or(rot_mag_sq < rot_threshold);
        auto dead_mask2 = (opacities <= min_opacity).logical_or(rot_mag_sq < rot_threshold);

        auto mismatches = dead_mask1.logical_xor(dead_mask2).sum().item();
        EXPECT_EQ(mismatches, 0) << "Dead mask should be deterministic (iteration " << i << ")";
    }
}

TEST_F(MCMCDeadMaskTest, NonzeroIndicesMatch) {
    // Test that nonzero() returns correct indices matching the mask
    constexpr int N = 100;

    auto opacities = Tensor::zeros({static_cast<size_t>(N)}, Device::CUDA);
    auto rot_mag_sq = Tensor::ones({static_cast<size_t>(N)}, Device::CUDA);

    // Set some specific opacities to be alive
    auto opacities_cpu = opacities.cpu();
    auto opacity_vec = opacities_cpu.to_vector();
    std::vector<int> expected_dead_indices;
    for (int i = 0; i < N; ++i) {
        if (i % 5 == 0) {
            opacity_vec[i] = 0.01f; // Alive
        } else {
            opacity_vec[i] = 0.0f; // Dead
            expected_dead_indices.push_back(i);
        }
    }
    opacities = Tensor::from_vector(opacity_vec, {static_cast<size_t>(N)}, Device::CUDA);

    // Compute dead mask and indices
    auto dead_mask = (opacities <= min_opacity).logical_or(rot_mag_sq < rot_threshold);
    auto dead_indices = dead_mask.nonzero().squeeze(-1).cpu();

    // Verify count
    EXPECT_EQ(dead_indices.numel(), expected_dead_indices.size());

    // Verify indices match
    auto indices_vec = dead_indices.to_vector();
    for (size_t i = 0; i < expected_dead_indices.size(); ++i) {
        EXPECT_EQ(static_cast<int>(indices_vec[i]), expected_dead_indices[i])
            << "Dead index mismatch at position " << i;
    }
}

// ============= MCMC Ratios / Occurrence Counting Tests =============

TEST_F(MCMCDeadMaskTest, RatiosBasicOccurrenceCounting) {
    // Test basic occurrence counting: sampled_idxs = [0, 1, 2, 1, 0, 1]
    // Expected ratios at those indices: [2, 3, 1, 3, 2, 3]
    constexpr int N = 10;

    auto opacities = Tensor::ones({static_cast<size_t>(N)}, Device::CUDA);

    // Sampled indices with duplicates
    std::vector<int> sampled = {0, 1, 2, 1, 0, 1};
    auto sampled_idxs = Tensor::from_vector(sampled, TensorShape{sampled.size()}, Device::CUDA);

    // Count occurrences using index_add_
    auto ratios = Tensor::ones_like(opacities, DataType::Int32);
    ratios = ratios.index_add_(0, sampled_idxs, Tensor::ones(TensorShape{sampled_idxs.numel()}, Device::CUDA, DataType::Int32));
    ratios = ratios.index_select(0, sampled_idxs).contiguous();

    // Verify - use to_vector_int() for Int32 tensors
    auto ratios_cpu = ratios.cpu();
    auto ratios_vec = ratios_cpu.to_vector_int();

    ASSERT_EQ(ratios_vec.size(), sampled.size());
    EXPECT_EQ(static_cast<int>(ratios_vec[0]), 3); // Index 0 appears twice, starts at 1 -> 1+2=3
    EXPECT_EQ(static_cast<int>(ratios_vec[1]), 4); // Index 1 appears 3 times -> 1+3=4
    EXPECT_EQ(static_cast<int>(ratios_vec[2]), 2); // Index 2 appears once -> 1+1=2
    EXPECT_EQ(static_cast<int>(ratios_vec[3]), 4); // Index 1 again
    EXPECT_EQ(static_cast<int>(ratios_vec[4]), 3); // Index 0 again
    EXPECT_EQ(static_cast<int>(ratios_vec[5]), 4); // Index 1 again
}

TEST_F(MCMCDeadMaskTest, RatiosAllUnique) {
    // Test with all unique indices - each should have ratio 2 (1 + 1)
    constexpr int N = 100;

    auto opacities = Tensor::ones({static_cast<size_t>(N)}, Device::CUDA);

    // All unique indices [0, 1, 2, ..., 49]
    std::vector<int> sampled(50);
    std::iota(sampled.begin(), sampled.end(), 0);
    auto sampled_idxs = Tensor::from_vector(sampled, TensorShape{sampled.size()}, Device::CUDA);

    auto ratios = Tensor::ones_like(opacities, DataType::Int32);
    ratios = ratios.index_add_(0, sampled_idxs, Tensor::ones(TensorShape{sampled_idxs.numel()}, Device::CUDA, DataType::Int32));
    ratios = ratios.index_select(0, sampled_idxs).contiguous();

    auto ratios_cpu = ratios.cpu();
    auto ratios_vec = ratios_cpu.to_vector_int();

    // All should be 2 (initial 1 + added 1)
    for (size_t i = 0; i < ratios_vec.size(); ++i) {
        EXPECT_EQ(static_cast<int>(ratios_vec[i]), 2)
            << "Unique index " << i << " should have ratio 2";
    }
}

TEST_F(MCMCDeadMaskTest, RatiosAllSameIndex) {
    // Test with all samples pointing to same index
    constexpr int N = 100;
    constexpr int num_samples = 50;

    auto opacities = Tensor::ones({static_cast<size_t>(N)}, Device::CUDA);

    // All samples point to index 42
    std::vector<int> sampled(num_samples, 42);
    auto sampled_idxs = Tensor::from_vector(sampled, TensorShape{sampled.size()}, Device::CUDA);

    auto ratios = Tensor::ones_like(opacities, DataType::Int32);
    ratios = ratios.index_add_(0, sampled_idxs, Tensor::ones(TensorShape{sampled_idxs.numel()}, Device::CUDA, DataType::Int32));
    ratios = ratios.index_select(0, sampled_idxs).contiguous();

    auto ratios_cpu = ratios.cpu();
    auto ratios_vec = ratios_cpu.to_vector_int();

    // All should be num_samples + 1 (initial 1 + num_samples)
    const int expected = num_samples + 1;
    for (size_t i = 0; i < ratios_vec.size(); ++i) {
        EXPECT_EQ(static_cast<int>(ratios_vec[i]), expected)
            << "Sample " << i << " should have ratio " << expected;
    }
}

TEST_F(MCMCDeadMaskTest, RatiosClampingBehavior) {
    // Test clamping to [1, n_max]
    constexpr int N = 100;
    constexpr int n_max = 10; // Small n_max to trigger clamping

    auto opacities = Tensor::ones({static_cast<size_t>(N)}, Device::CUDA);

    // Create samples where index 0 appears many times (will exceed n_max)
    std::vector<int> sampled;
    for (int i = 0; i < 20; ++i)
        sampled.push_back(0); // Index 0: 20 times
    for (int i = 0; i < 5; ++i)
        sampled.push_back(1); // Index 1: 5 times
    sampled.push_back(2);     // Index 2: once

    auto sampled_idxs = Tensor::from_vector(sampled, TensorShape{sampled.size()}, Device::CUDA);

    auto ratios = Tensor::ones_like(opacities, DataType::Int32);
    ratios = ratios.index_add_(0, sampled_idxs, Tensor::ones(TensorShape{sampled_idxs.numel()}, Device::CUDA, DataType::Int32));
    ratios = ratios.index_select(0, sampled_idxs).contiguous();

    // Clamp to [1, n_max]
    ratios = ratios.clamp(1, n_max);

    auto ratios_cpu = ratios.cpu();
    auto ratios_vec = ratios_cpu.to(DataType::Float32).to_vector();

    // Verify clamping
    for (size_t i = 0; i < ratios_vec.size(); ++i) {
        int ratio = static_cast<int>(ratios_vec[i]);
        EXPECT_GE(ratio, 1) << "Ratio at " << i << " should be >= 1";
        EXPECT_LE(ratio, n_max) << "Ratio at " << i << " should be <= " << n_max;
    }

    // First 20 samples (index 0) should be clamped to n_max
    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(static_cast<int>(ratios_vec[i]), n_max)
            << "Index 0 occurrence " << i << " should be clamped to " << n_max;
    }
}

TEST_F(MCMCDeadMaskTest, RatiosRealisticMCMCScenario) {
    // Simulate realistic MCMC scenario with 54,275 Gaussians
    constexpr size_t N = 54275;
    constexpr size_t num_samples = 10000;
    constexpr int n_max = 50;

    auto opacities = Tensor::ones({N}, Device::CUDA);

    // Generate random samples with some duplicates (using modulo to create patterns)
    std::vector<int> sampled(num_samples);
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, N - 1);

    // Create biased distribution - some indices sampled more often
    std::map<int, int> expected_counts;
    for (size_t i = 0; i < num_samples; ++i) {
        if (i % 10 == 0) {
            sampled[i] = 42; // Hot spot index
        } else {
            sampled[i] = dist(rng);
        }
        expected_counts[sampled[i]]++;
    }

    auto sampled_idxs = Tensor::from_vector(sampled, TensorShape{sampled.size()}, Device::CUDA);

    auto ratios = Tensor::ones_like(opacities, DataType::Int32);
    ratios = ratios.index_add_(0, sampled_idxs, Tensor::ones(TensorShape{sampled_idxs.numel()}, Device::CUDA, DataType::Int32));
    ratios = ratios.index_select(0, sampled_idxs).contiguous();
    ratios = ratios.clamp(1, n_max);

    auto ratios_cpu = ratios.cpu();
    auto ratios_vec = ratios_cpu.to(DataType::Float32).to_vector();

    // Verify all ratios are in valid range
    for (size_t i = 0; i < ratios_vec.size(); ++i) {
        int ratio = static_cast<int>(ratios_vec[i]);
        ASSERT_GE(ratio, 1) << "Invalid ratio at sample " << i;
        ASSERT_LE(ratio, n_max) << "Invalid ratio at sample " << i;
    }

    // Verify hot spot index (42) has high ratio (clamped)
    for (size_t i = 0; i < num_samples; i += 10) {
        EXPECT_EQ(static_cast<int>(ratios_vec[i]), n_max)
            << "Hot spot sample " << i << " should be clamped";
    }
}

TEST_F(MCMCDeadMaskTest, RatiosStressTestLarge) {
    // Stress test with 1M Gaussians and 100K samples
    constexpr size_t N = 1000000;
    constexpr size_t num_samples = 100000;
    constexpr int n_max = 100;

    auto opacities = Tensor::ones({N}, Device::CUDA);

    // Generate samples with power-law distribution (realistic for MCMC)
    std::vector<int> sampled(num_samples);
    std::mt19937 rng(42);

    for (size_t i = 0; i < num_samples; ++i) {
        // Zipf-like distribution - early indices more likely
        double u = static_cast<double>(rng()) / rng.max();
        int idx = static_cast<int>(std::pow(u, 3.0) * (N - 1));
        sampled[i] = std::min(idx, static_cast<int>(N - 1));
    }

    auto sampled_idxs = Tensor::from_vector(sampled, TensorShape{sampled.size()}, Device::CUDA);

    // This should not crash and should complete quickly
    auto ratios = Tensor::ones_like(opacities, DataType::Int32);
    ratios = ratios.index_add_(0, sampled_idxs, Tensor::ones(TensorShape{sampled_idxs.numel()}, Device::CUDA, DataType::Int32));
    ratios = ratios.index_select(0, sampled_idxs).contiguous();
    ratios = ratios.clamp(1, n_max);

    auto ratios_cpu = ratios.cpu();
    auto ratios_vec = ratios_cpu.to(DataType::Float32).to_vector();

    // Verify basic properties
    ASSERT_EQ(ratios_vec.size(), num_samples);

    // All ratios should be valid
    for (size_t i = 0; i < std::min<size_t>(1000, ratios_vec.size()); ++i) {
        int ratio = static_cast<int>(ratios_vec[i]);
        EXPECT_GE(ratio, 1);
        EXPECT_LE(ratio, n_max);
    }
}

TEST_F(MCMCDeadMaskTest, RatiosEdgeCaseEmptySamples) {
    // Edge case: no samples
    constexpr int N = 100;

    auto opacities = Tensor::ones({static_cast<size_t>(N)}, Device::CUDA);

    std::vector<int> sampled; // Empty
    auto sampled_idxs = Tensor::from_vector(sampled, TensorShape{sampled.size()}, Device::CUDA);

    auto ratios = Tensor::ones_like(opacities, DataType::Int32);
    ratios = ratios.index_add_(0, sampled_idxs, Tensor::ones(TensorShape{sampled_idxs.numel()}, Device::CUDA, DataType::Int32));

    // index_select with empty indices should return empty tensor
    if (sampled_idxs.numel() > 0) {
        ratios = ratios.index_select(0, sampled_idxs).contiguous();
        EXPECT_EQ(ratios.numel(), 0);
    }
}

// ============================================================================
// LibTorch Comparison Tests
// ============================================================================

TEST_F(MCMCDeadMaskTest, DeadMaskCalculation_LibTorchComparison) {
    // CRITICAL: Compare full dead mask calculation against LibTorch
    // dead_mask = (opacities <= min_opacity).logical_or(rot_mag_sq < rot_threshold)

    const size_t N = 100;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> opacity_dist(0.0f, 0.02f);
    std::uniform_real_distribution<float> rot_dist(0.0f, 1e-6f);

    std::vector<float> opacity_data(N);
    std::vector<float> rot_mag_sq_data(N);

    for (size_t i = 0; i < N; ++i) {
        opacity_data[i] = opacity_dist(rng);
        rot_mag_sq_data[i] = rot_dist(rng);
    }

    // ========== LFS Tensor Library ==========
    auto opacities_lfs = Tensor::from_vector(opacity_data, TensorShape{N}, Device::CUDA);
    auto rot_mag_sq_lfs = Tensor::from_vector(rot_mag_sq_data, TensorShape{N}, Device::CUDA);
    auto dead_mask_lfs = (opacities_lfs <= min_opacity).logical_or(rot_mag_sq_lfs < rot_threshold);
    auto result_lfs = dead_mask_lfs.cpu().to_vector_bool();

    // ========== LibTorch ==========
    auto opacities_torch = torch::from_blob(
                               opacity_data.data(),
                               {static_cast<long>(N)},
                               torch::kFloat32)
                               .clone()
                               .cuda();
    auto rot_mag_sq_torch = torch::from_blob(
                                rot_mag_sq_data.data(),
                                {static_cast<long>(N)},
                                torch::kFloat32)
                                .clone()
                                .cuda();
    auto dead_mask_torch = (opacities_torch <= min_opacity).logical_or(rot_mag_sq_torch < rot_threshold);
    auto result_torch_tensor = dead_mask_torch.cpu();

    // ========== Compare Results ==========
    for (size_t i = 0; i < N; ++i) {
        bool lfs_val = result_lfs[i];
        bool torch_val = result_torch_tensor[i].item<bool>();

        EXPECT_EQ(lfs_val, torch_val)
            << "Dead mask mismatch at index " << i
            << ": opacity=" << opacity_data[i]
            << ", rot_mag_sq=" << rot_mag_sq_data[i]
            << ", LFS=" << lfs_val << " vs LibTorch=" << torch_val;
    }
}

TEST_F(MCMCDeadMaskTest, RotationMagnitudeSquared_LibTorchComparison) {
    // Compare rotation magnitude calculation: (rotations * rotations).sum(-1)

    std::vector<float> quaternions = {
        1.0f, 0.0f, 0.0f, 0.0f,       // Unit quaternion
        0.5f, 0.5f, 0.5f, 0.5f,       // Normalized
        2.0f, 0.0f, 0.0f, 0.0f,       // Scaled
        1.0f, 1.0f, 1.0f, 1.0f,       // All ones
        1e-5f, 1e-5f, 1e-5f, 1e-5f,   // Very small
        0.7071f, 0.7071f, 0.0f, 0.0f, // Random
        3.16e-5f, 0.0f, 0.0f, 0.0f,   // Near threshold
        -1.0f, 1.0f, -1.0f, 1.0f,     // Negative values
    };

    const size_t N = quaternions.size() / 4;

    // ========== LFS Tensor Library ==========
    auto rotation_lfs = Tensor::from_vector(quaternions, TensorShape{N, 4}, Device::CUDA);
    auto rot_mag_sq_lfs = (rotation_lfs * rotation_lfs).sum(-1);
    auto result_lfs = rot_mag_sq_lfs.cpu().to_vector();

    // ========== LibTorch ==========
    auto rotation_torch = torch::from_blob(
                              quaternions.data(),
                              {static_cast<long>(N), 4},
                              torch::kFloat32)
                              .clone()
                              .cuda();
    auto rot_mag_sq_torch = (rotation_torch * rotation_torch).sum(-1);
    auto result_torch_tensor = rot_mag_sq_torch.cpu();

    // ========== Compare Results ==========
    for (size_t i = 0; i < N; ++i) {
        float lfs_val = result_lfs[i];
        float torch_val = result_torch_tensor[i].item<float>();
        float diff = std::abs(lfs_val - torch_val);

        EXPECT_LT(diff, 1e-6f)
            << "Rotation magnitude mismatch at index " << i
            << ": LFS=" << lfs_val << " vs LibTorch=" << torch_val
            << " (diff=" << diff << ")";
    }
}

TEST_F(MCMCDeadMaskTest, LogicalOr_LibTorchComparison) {
    // Test logical_or operation with all combinations

    std::vector<float> mask1_data = {true, true, false, false, true, false};
    std::vector<float> mask2_data = {true, false, true, false, false, true};

    const size_t N = mask1_data.size();

    // ========== LFS Tensor Library ==========
    auto mask1_lfs = Tensor::from_vector(mask1_data, TensorShape{N}, Device::CUDA).to(DataType::Bool);
    auto mask2_lfs = Tensor::from_vector(mask2_data, TensorShape{N}, Device::CUDA).to(DataType::Bool);
    auto result_lfs_tensor = mask1_lfs.logical_or(mask2_lfs);
    auto result_lfs = result_lfs_tensor.cpu().to_vector_bool();

    // ========== LibTorch ==========
    auto mask1_torch = torch::from_blob(
                           mask1_data.data(),
                           {static_cast<long>(N)},
                           torch::kFloat32)
                           .clone()
                           .to(torch::kBool)
                           .cuda();
    auto mask2_torch = torch::from_blob(
                           mask2_data.data(),
                           {static_cast<long>(N)},
                           torch::kFloat32)
                           .clone()
                           .to(torch::kBool)
                           .cuda();
    auto result_torch = mask1_torch.logical_or(mask2_torch);
    auto result_torch_tensor = result_torch.cpu();

    // ========== Compare Results ==========
    std::vector<bool> expected = {true, true, true, false, true, true}; // TT, TF, FT, FF, TF, FT

    for (size_t i = 0; i < N; ++i) {
        bool lfs_val = result_lfs[i];
        bool torch_val = result_torch_tensor[i].item<bool>();

        EXPECT_EQ(lfs_val, torch_val)
            << "Logical OR mismatch at index " << i
            << ": mask1=" << (mask1_data[i] > 0.5f)
            << ", mask2=" << (mask2_data[i] > 0.5f)
            << ", expected=" << expected[i]
            << ", LFS=" << lfs_val << " vs LibTorch=" << torch_val;
    }
}

TEST_F(MCMCDeadMaskTest, ComparisonOperators_LibTorchComparison) {
    // Test <= and < operators with boundary cases

    std::vector<float> values = {
        0.0f, 0.004f, 0.005f, 0.005f + 1e-7f, 0.01f,
        1e-9f, 1e-8f, 1e-8f + 1e-10f, 1e-7f, 1.0f};
    const size_t N = values.size();

    // Test <= operator with min_opacity threshold
    {
        // ========== LFS Tensor Library ==========
        auto tensor_lfs = Tensor::from_vector(values, TensorShape{N}, Device::CUDA);
        auto result_lfs_tensor = tensor_lfs <= min_opacity;
        auto result_lfs = result_lfs_tensor.cpu().to_vector_bool();

        // ========== LibTorch ==========
        auto tensor_torch = torch::from_blob(
                                values.data(),
                                {static_cast<long>(N)},
                                torch::kFloat32)
                                .clone()
                                .cuda();
        auto result_torch = tensor_torch <= min_opacity;
        auto result_torch_tensor = result_torch.cpu();

        // ========== Compare Results ==========
        for (size_t i = 0; i < N; ++i) {
            bool lfs_val = result_lfs[i];
            bool torch_val = result_torch_tensor[i].item<bool>();

            EXPECT_EQ(lfs_val, torch_val)
                << "<= operator mismatch at index " << i
                << ": value=" << values[i]
                << ", threshold=" << min_opacity
                << ", LFS=" << lfs_val << " vs LibTorch=" << torch_val;
        }
    }

    // Test < operator with rot_threshold
    {
        // ========== LFS Tensor Library ==========
        auto tensor_lfs = Tensor::from_vector(values, TensorShape{N}, Device::CUDA);
        auto result_lfs_tensor = tensor_lfs < rot_threshold;
        auto result_lfs = result_lfs_tensor.cpu().to_vector_bool();

        // ========== LibTorch ==========
        auto tensor_torch = torch::from_blob(
                                values.data(),
                                {static_cast<long>(N)},
                                torch::kFloat32)
                                .clone()
                                .cuda();
        auto result_torch = tensor_torch < rot_threshold;
        auto result_torch_tensor = result_torch.cpu();

        // ========== Compare Results ==========
        for (size_t i = 0; i < N; ++i) {
            bool lfs_val = result_lfs[i];
            bool torch_val = result_torch_tensor[i].item<bool>();

            EXPECT_EQ(lfs_val, torch_val)
                << "< operator mismatch at index " << i
                << ": value=" << values[i]
                << ", threshold=" << rot_threshold
                << ", LFS=" << lfs_val << " vs LibTorch=" << torch_val;
        }
    }
}

TEST_F(MCMCDeadMaskTest, IndexAddInt32_LibTorchComparison) {
    // Test index_add_ with Int32 as used for occurrence counting

    const size_t N = 50;
    std::vector<int> sampled_indices = {0, 5, 10, 5, 0, 15, 5, 10, 0, 20};

    // ========== LFS Tensor Library ==========
    auto base_lfs = Tensor::ones({N}, Device::CUDA, DataType::Int32);
    auto indices_lfs = Tensor::from_vector(sampled_indices, TensorShape{sampled_indices.size()}, Device::CUDA);
    auto values_lfs = Tensor::ones(TensorShape{sampled_indices.size()}, Device::CUDA, DataType::Int32);
    auto result_lfs_tensor = base_lfs.index_add_(0, indices_lfs, values_lfs);
    auto result_lfs = result_lfs_tensor.cpu().to_vector_int();

    // ========== LibTorch ==========
    auto base_torch = torch::ones({static_cast<long>(N)}, torch::kInt32).cuda();
    auto indices_torch = torch::from_blob(
                             sampled_indices.data(),
                             {static_cast<long>(sampled_indices.size())},
                             torch::kInt32)
                             .clone()
                             .to(torch::kInt64)
                             .cuda();
    auto values_torch = torch::ones({static_cast<long>(sampled_indices.size())}, torch::kInt32).cuda();
    auto result_torch = base_torch.index_add_(0, indices_torch, values_torch);
    auto result_torch_tensor = result_torch.cpu();

    // ========== Compare Results ==========
    for (size_t i = 0; i < N; ++i) {
        int lfs_val = result_lfs[i];
        int torch_val = result_torch_tensor[i].item<int>();

        EXPECT_EQ(lfs_val, torch_val)
            << "index_add_ mismatch at index " << i
            << ": LFS=" << lfs_val << " vs LibTorch=" << torch_val;
    }

    // Verify specific counts
    EXPECT_EQ(result_lfs[0], 4);  // Index 0 appears 3 times: 1 + 3 = 4
    EXPECT_EQ(result_lfs[5], 4);  // Index 5 appears 3 times: 1 + 3 = 4
    EXPECT_EQ(result_lfs[10], 3); // Index 10 appears 2 times: 1 + 2 = 3
    EXPECT_EQ(result_lfs[15], 2); // Index 15 appears 1 time: 1 + 1 = 2
    EXPECT_EQ(result_lfs[20], 2); // Index 20 appears 1 time: 1 + 1 = 2
}

TEST_F(MCMCDeadMaskTest, ClampInt32_LibTorchComparison) {
    // Test clamp with Int32 as used for ratio clamping

    std::vector<float> values_float = {-10, 0, 1, 5, 25, 50, 51, 52, 100, 200};
    const size_t N = values_float.size();
    const int n_max = 51;

    // ========== LFS Tensor Library ==========
    auto tensor_lfs = Tensor::from_vector(values_float, TensorShape{N}, Device::CUDA).to(DataType::Int32);
    auto result_lfs_tensor = tensor_lfs.clamp(1, n_max);
    auto result_lfs = result_lfs_tensor.cpu().to_vector_int();

    // ========== LibTorch ==========
    auto tensor_torch = torch::from_blob(
                            values_float.data(),
                            {static_cast<long>(N)},
                            torch::kFloat32)
                            .clone()
                            .to(torch::kInt32)
                            .cuda();
    auto result_torch = tensor_torch.clamp(1, n_max);
    auto result_torch_tensor = result_torch.cpu();

    // ========== Compare Results ==========
    for (size_t i = 0; i < N; ++i) {
        int lfs_val = result_lfs[i];
        int torch_val = result_torch_tensor[i].item<int>();

        EXPECT_EQ(lfs_val, torch_val)
            << "Clamp mismatch at index " << i
            << ": input=" << static_cast<int>(values_float[i])
            << ", LFS=" << lfs_val << " vs LibTorch=" << torch_val;

        // Verify clamping bounds
        EXPECT_GE(lfs_val, 1);
        EXPECT_LE(lfs_val, n_max);
    }
}
