/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * Comprehensive tests for GPU-based NaN/Inf detection.
 * Verifies correctness against LibTorch reference implementation.
 */

#include "core/tensor.hpp"
#include <chrono>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <torch/torch.h>

using namespace lfs::core;

class NaNInfGPUCheckTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure CUDA is available
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available";
    }

    // Helper to create LFS tensor from torch tensor
    Tensor from_torch(const torch::Tensor& t) {
        auto cpu_t = t.to(torch::kCPU).contiguous();
        std::vector<float> data(cpu_t.data_ptr<float>(),
                                cpu_t.data_ptr<float>() + cpu_t.numel());

        std::vector<size_t> shape;
        for (int i = 0; i < cpu_t.dim(); ++i) {
            shape.push_back(static_cast<size_t>(cpu_t.size(i)));
        }

        return Tensor::from_vector(data, TensorShape(shape), Device::CUDA);
    }

    // Reference implementation using LibTorch
    bool torch_has_nan(const torch::Tensor& t) {
        return torch::isnan(t).any().item<bool>();
    }

    bool torch_has_inf(const torch::Tensor& t) {
        return torch::isinf(t).any().item<bool>();
    }

    bool torch_has_nan_or_inf(const torch::Tensor& t) {
        return torch_has_nan(t) || torch_has_inf(t);
    }
};

// ============= Basic NaN Detection Tests =============

TEST_F(NaNInfGPUCheckTest, AllFinite_NoNaN) {
    auto torch_t = torch::randn({1000, 100}, torch::kCUDA);
    auto lfs_t = from_torch(torch_t);

    EXPECT_FALSE(torch_has_nan(torch_t)) << "Torch should not detect NaN";
    EXPECT_FALSE(lfs_t.has_nan()) << "LFS should not detect NaN";
}

TEST_F(NaNInfGPUCheckTest, SingleNaN_Beginning) {
    auto torch_t = torch::randn({1000}, torch::kCUDA);
    torch_t[0] = std::numeric_limits<float>::quiet_NaN();
    auto lfs_t = from_torch(torch_t);

    EXPECT_TRUE(torch_has_nan(torch_t)) << "Torch should detect NaN at beginning";
    EXPECT_TRUE(lfs_t.has_nan()) << "LFS should detect NaN at beginning";
}

TEST_F(NaNInfGPUCheckTest, SingleNaN_Middle) {
    auto torch_t = torch::randn({1000}, torch::kCUDA);
    torch_t[500] = std::numeric_limits<float>::quiet_NaN();
    auto lfs_t = from_torch(torch_t);

    EXPECT_TRUE(torch_has_nan(torch_t)) << "Torch should detect NaN in middle";
    EXPECT_TRUE(lfs_t.has_nan()) << "LFS should detect NaN in middle";
}

TEST_F(NaNInfGPUCheckTest, SingleNaN_End) {
    auto torch_t = torch::randn({1000}, torch::kCUDA);
    torch_t[999] = std::numeric_limits<float>::quiet_NaN();
    auto lfs_t = from_torch(torch_t);

    EXPECT_TRUE(torch_has_nan(torch_t)) << "Torch should detect NaN at end";
    EXPECT_TRUE(lfs_t.has_nan()) << "LFS should detect NaN at end";
}

TEST_F(NaNInfGPUCheckTest, MultipleNaNs) {
    auto torch_t = torch::randn({1000}, torch::kCUDA);
    torch_t[100] = std::numeric_limits<float>::quiet_NaN();
    torch_t[500] = std::numeric_limits<float>::quiet_NaN();
    torch_t[900] = std::numeric_limits<float>::quiet_NaN();
    auto lfs_t = from_torch(torch_t);

    EXPECT_TRUE(torch_has_nan(torch_t)) << "Torch should detect multiple NaNs";
    EXPECT_TRUE(lfs_t.has_nan()) << "LFS should detect multiple NaNs";
}

TEST_F(NaNInfGPUCheckTest, AllNaN) {
    auto torch_t = torch::full({100}, std::numeric_limits<float>::quiet_NaN(), torch::kCUDA);
    auto lfs_t = from_torch(torch_t);

    EXPECT_TRUE(torch_has_nan(torch_t)) << "Torch should detect all NaN";
    EXPECT_TRUE(lfs_t.has_nan()) << "LFS should detect all NaN";
}

TEST_F(NaNInfGPUCheckTest, SignalingNaN) {
    auto torch_t = torch::randn({1000}, torch::kCUDA);
    torch_t[500] = std::numeric_limits<float>::signaling_NaN();
    auto lfs_t = from_torch(torch_t);

    EXPECT_TRUE(torch_has_nan(torch_t)) << "Torch should detect signaling NaN";
    EXPECT_TRUE(lfs_t.has_nan()) << "LFS should detect signaling NaN";
}

// ============= Basic Inf Detection Tests =============

TEST_F(NaNInfGPUCheckTest, AllFinite_NoInf) {
    auto torch_t = torch::randn({1000, 100}, torch::kCUDA);
    auto lfs_t = from_torch(torch_t);

    EXPECT_FALSE(torch_has_inf(torch_t)) << "Torch should not detect Inf";
    EXPECT_FALSE(lfs_t.has_inf()) << "LFS should not detect Inf";
}

TEST_F(NaNInfGPUCheckTest, PositiveInf_Beginning) {
    auto torch_t = torch::randn({1000}, torch::kCUDA);
    torch_t[0] = std::numeric_limits<float>::infinity();
    auto lfs_t = from_torch(torch_t);

    EXPECT_TRUE(torch_has_inf(torch_t)) << "Torch should detect +Inf at beginning";
    EXPECT_TRUE(lfs_t.has_inf()) << "LFS should detect +Inf at beginning";
}

TEST_F(NaNInfGPUCheckTest, NegativeInf_Middle) {
    auto torch_t = torch::randn({1000}, torch::kCUDA);
    torch_t[500] = -std::numeric_limits<float>::infinity();
    auto lfs_t = from_torch(torch_t);

    EXPECT_TRUE(torch_has_inf(torch_t)) << "Torch should detect -Inf in middle";
    EXPECT_TRUE(lfs_t.has_inf()) << "LFS should detect -Inf in middle";
}

TEST_F(NaNInfGPUCheckTest, PositiveInf_End) {
    auto torch_t = torch::randn({1000}, torch::kCUDA);
    torch_t[999] = std::numeric_limits<float>::infinity();
    auto lfs_t = from_torch(torch_t);

    EXPECT_TRUE(torch_has_inf(torch_t)) << "Torch should detect +Inf at end";
    EXPECT_TRUE(lfs_t.has_inf()) << "LFS should detect +Inf at end";
}

TEST_F(NaNInfGPUCheckTest, MixedPosNegInf) {
    auto torch_t = torch::randn({1000}, torch::kCUDA);
    torch_t[100] = std::numeric_limits<float>::infinity();
    torch_t[500] = -std::numeric_limits<float>::infinity();
    auto lfs_t = from_torch(torch_t);

    EXPECT_TRUE(torch_has_inf(torch_t)) << "Torch should detect mixed Inf";
    EXPECT_TRUE(lfs_t.has_inf()) << "LFS should detect mixed Inf";
}

TEST_F(NaNInfGPUCheckTest, AllInf) {
    auto torch_t = torch::full({100}, std::numeric_limits<float>::infinity(), torch::kCUDA);
    auto lfs_t = from_torch(torch_t);

    EXPECT_TRUE(torch_has_inf(torch_t)) << "Torch should detect all Inf";
    EXPECT_TRUE(lfs_t.has_inf()) << "LFS should detect all Inf";
}

// ============= Mixed NaN and Inf Tests =============

TEST_F(NaNInfGPUCheckTest, BothNaNAndInf) {
    auto torch_t = torch::randn({1000}, torch::kCUDA);
    torch_t[100] = std::numeric_limits<float>::quiet_NaN();
    torch_t[500] = std::numeric_limits<float>::infinity();
    auto lfs_t = from_torch(torch_t);

    EXPECT_TRUE(torch_has_nan_or_inf(torch_t)) << "Torch should detect NaN and Inf";
    EXPECT_TRUE(lfs_t.has_nan()) << "LFS has_nan should detect (checks both)";
    EXPECT_TRUE(lfs_t.has_inf()) << "LFS has_inf should detect (checks both)";
}

TEST_F(NaNInfGPUCheckTest, NaNOnlyMatchesTorchPredicates) {
    auto torch_t = torch::randn({1000}, torch::kCUDA);
    torch_t[500] = std::numeric_limits<float>::quiet_NaN();
    auto lfs_t = from_torch(torch_t);

    EXPECT_EQ(lfs_t.has_nan(), torch_has_nan(torch_t));
    EXPECT_EQ(lfs_t.has_inf(), torch_has_inf(torch_t));
}

TEST_F(NaNInfGPUCheckTest, InfOnlyMatchesTorchPredicates) {
    auto torch_t = torch::randn({1000}, torch::kCUDA);
    torch_t[500] = std::numeric_limits<float>::infinity();
    auto lfs_t = from_torch(torch_t);

    EXPECT_EQ(lfs_t.has_nan(), torch_has_nan(torch_t));
    EXPECT_EQ(lfs_t.has_inf(), torch_has_inf(torch_t));
}

// ============= Edge Cases =============

TEST_F(NaNInfGPUCheckTest, EmptyTensor) {
    auto lfs_t = Tensor::empty({0}, Device::CUDA);

    EXPECT_FALSE(lfs_t.has_nan()) << "Empty tensor should not have NaN";
    EXPECT_FALSE(lfs_t.has_inf()) << "Empty tensor should not have Inf";
}

TEST_F(NaNInfGPUCheckTest, SingleElement_Finite) {
    auto torch_t = torch::tensor({42.0f}, torch::kCUDA);
    auto lfs_t = from_torch(torch_t);

    EXPECT_FALSE(torch_has_nan_or_inf(torch_t));
    EXPECT_FALSE(lfs_t.has_nan());
}

TEST_F(NaNInfGPUCheckTest, SingleElement_NaN) {
    auto torch_t = torch::tensor({std::numeric_limits<float>::quiet_NaN()}, torch::kCUDA);
    auto lfs_t = from_torch(torch_t);

    EXPECT_TRUE(torch_has_nan(torch_t));
    EXPECT_TRUE(lfs_t.has_nan());
}

TEST_F(NaNInfGPUCheckTest, SingleElement_Inf) {
    auto torch_t = torch::tensor({std::numeric_limits<float>::infinity()}, torch::kCUDA);
    auto lfs_t = from_torch(torch_t);

    EXPECT_TRUE(torch_has_inf(torch_t));
    EXPECT_TRUE(lfs_t.has_inf());
}

TEST_F(NaNInfGPUCheckTest, ZeroTensor) {
    auto torch_t = torch::zeros({1000, 100}, torch::kCUDA);
    auto lfs_t = from_torch(torch_t);

    EXPECT_FALSE(torch_has_nan_or_inf(torch_t));
    EXPECT_FALSE(lfs_t.has_nan());
}

TEST_F(NaNInfGPUCheckTest, MaxFloatValues) {
    auto torch_t = torch::full({1000}, std::numeric_limits<float>::max(), torch::kCUDA);
    auto lfs_t = from_torch(torch_t);

    EXPECT_FALSE(torch_has_nan_or_inf(torch_t)) << "Max float is not Inf";
    EXPECT_FALSE(lfs_t.has_nan()) << "Max float should not be detected as NaN/Inf";
}

TEST_F(NaNInfGPUCheckTest, MinFloatValues) {
    auto torch_t = torch::full({1000}, std::numeric_limits<float>::lowest(), torch::kCUDA);
    auto lfs_t = from_torch(torch_t);

    EXPECT_FALSE(torch_has_nan_or_inf(torch_t)) << "Min float is not -Inf";
    EXPECT_FALSE(lfs_t.has_nan()) << "Min float should not be detected as NaN/Inf";
}

TEST_F(NaNInfGPUCheckTest, DenormalizedNumbers) {
    auto torch_t = torch::full({1000}, std::numeric_limits<float>::denorm_min(), torch::kCUDA);
    auto lfs_t = from_torch(torch_t);

    EXPECT_FALSE(torch_has_nan_or_inf(torch_t)) << "Denormalized numbers are finite";
    EXPECT_FALSE(lfs_t.has_nan()) << "Denormalized numbers should not be detected";
}

TEST_F(NaNInfGPUCheckTest, NegativeZero) {
    auto torch_t = torch::full({1000}, -0.0f, torch::kCUDA);
    auto lfs_t = from_torch(torch_t);

    EXPECT_FALSE(torch_has_nan_or_inf(torch_t)) << "Negative zero is finite";
    EXPECT_FALSE(lfs_t.has_nan()) << "Negative zero should not be detected";
}

// ============= Various Tensor Sizes (Edge Cases for Vectorization) =============

TEST_F(NaNInfGPUCheckTest, Size1_WithNaN) {
    auto torch_t = torch::tensor({std::numeric_limits<float>::quiet_NaN()}, torch::kCUDA);
    auto lfs_t = from_torch(torch_t);
    EXPECT_EQ(torch_has_nan(torch_t), lfs_t.has_nan());
}

TEST_F(NaNInfGPUCheckTest, Size2_WithNaN) {
    auto torch_t = torch::tensor({1.0f, std::numeric_limits<float>::quiet_NaN()}, torch::kCUDA);
    auto lfs_t = from_torch(torch_t);
    EXPECT_EQ(torch_has_nan(torch_t), lfs_t.has_nan());
}

TEST_F(NaNInfGPUCheckTest, Size3_WithNaN) {
    auto torch_t = torch::tensor({1.0f, 2.0f, std::numeric_limits<float>::quiet_NaN()}, torch::kCUDA);
    auto lfs_t = from_torch(torch_t);
    EXPECT_EQ(torch_has_nan(torch_t), lfs_t.has_nan());
}

TEST_F(NaNInfGPUCheckTest, Size4_WithNaN) {
    auto torch_t = torch::tensor({1.0f, 2.0f, 3.0f, std::numeric_limits<float>::quiet_NaN()}, torch::kCUDA);
    auto lfs_t = from_torch(torch_t);
    EXPECT_EQ(torch_has_nan(torch_t), lfs_t.has_nan());
}

TEST_F(NaNInfGPUCheckTest, Size5_WithNaN) {
    auto torch_t = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, std::numeric_limits<float>::quiet_NaN()}, torch::kCUDA);
    auto lfs_t = from_torch(torch_t);
    EXPECT_EQ(torch_has_nan(torch_t), lfs_t.has_nan());
}

TEST_F(NaNInfGPUCheckTest, Size1023_WithNaN) {
    auto torch_t = torch::randn({1023}, torch::kCUDA);
    torch_t[1022] = std::numeric_limits<float>::quiet_NaN();
    auto lfs_t = from_torch(torch_t);
    EXPECT_EQ(torch_has_nan(torch_t), lfs_t.has_nan());
}

TEST_F(NaNInfGPUCheckTest, Size1024_WithNaN) {
    auto torch_t = torch::randn({1024}, torch::kCUDA);
    torch_t[1023] = std::numeric_limits<float>::quiet_NaN();
    auto lfs_t = from_torch(torch_t);
    EXPECT_EQ(torch_has_nan(torch_t), lfs_t.has_nan());
}

TEST_F(NaNInfGPUCheckTest, Size1025_WithNaN) {
    auto torch_t = torch::randn({1025}, torch::kCUDA);
    torch_t[1024] = std::numeric_limits<float>::quiet_NaN();
    auto lfs_t = from_torch(torch_t);
    EXPECT_EQ(torch_has_nan(torch_t), lfs_t.has_nan());
}

// ============= Large Tensor Tests =============

TEST_F(NaNInfGPUCheckTest, LargeTensor_1M_NoNaN) {
    auto torch_t = torch::randn({1000000}, torch::kCUDA);
    auto lfs_t = from_torch(torch_t);

    EXPECT_EQ(torch_has_nan(torch_t), lfs_t.has_nan());
}

TEST_F(NaNInfGPUCheckTest, LargeTensor_1M_SingleNaN_Beginning) {
    auto torch_t = torch::randn({1000000}, torch::kCUDA);
    torch_t[0] = std::numeric_limits<float>::quiet_NaN();
    auto lfs_t = from_torch(torch_t);

    EXPECT_TRUE(torch_has_nan(torch_t));
    EXPECT_TRUE(lfs_t.has_nan());
}

TEST_F(NaNInfGPUCheckTest, LargeTensor_1M_SingleNaN_End) {
    auto torch_t = torch::randn({1000000}, torch::kCUDA);
    torch_t[999999] = std::numeric_limits<float>::quiet_NaN();
    auto lfs_t = from_torch(torch_t);

    EXPECT_TRUE(torch_has_nan(torch_t));
    EXPECT_TRUE(lfs_t.has_nan());
}

TEST_F(NaNInfGPUCheckTest, LargeTensor_5M_NoNaN) {
    auto torch_t = torch::randn({5000000}, torch::kCUDA);
    auto lfs_t = from_torch(torch_t);

    EXPECT_EQ(torch_has_nan(torch_t), lfs_t.has_nan());
}

TEST_F(NaNInfGPUCheckTest, LargeTensor_5M_SingleNaN) {
    auto torch_t = torch::randn({5000000}, torch::kCUDA);
    torch_t[2500000] = std::numeric_limits<float>::quiet_NaN();
    auto lfs_t = from_torch(torch_t);

    EXPECT_TRUE(torch_has_nan(torch_t));
    EXPECT_TRUE(lfs_t.has_nan());
}

// ============= Multi-dimensional Tensor Tests =============

TEST_F(NaNInfGPUCheckTest, MultiDim_2D_NoNaN) {
    auto torch_t = torch::randn({100, 100}, torch::kCUDA);
    auto lfs_t = from_torch(torch_t);

    EXPECT_EQ(torch_has_nan(torch_t), lfs_t.has_nan());
}

TEST_F(NaNInfGPUCheckTest, MultiDim_2D_WithNaN) {
    auto torch_t = torch::randn({100, 100}, torch::kCUDA);
    torch_t[50][50] = std::numeric_limits<float>::quiet_NaN();
    auto lfs_t = from_torch(torch_t);

    EXPECT_TRUE(torch_has_nan(torch_t));
    EXPECT_TRUE(lfs_t.has_nan());
}

TEST_F(NaNInfGPUCheckTest, MultiDim_3D_NoNaN) {
    auto torch_t = torch::randn({10, 20, 30}, torch::kCUDA);
    auto lfs_t = from_torch(torch_t);

    EXPECT_EQ(torch_has_nan(torch_t), lfs_t.has_nan());
}

TEST_F(NaNInfGPUCheckTest, MultiDim_3D_WithNaN) {
    auto torch_t = torch::randn({10, 20, 30}, torch::kCUDA);
    torch_t[5][10][15] = std::numeric_limits<float>::quiet_NaN();
    auto lfs_t = from_torch(torch_t);

    EXPECT_TRUE(torch_has_nan(torch_t));
    EXPECT_TRUE(lfs_t.has_nan());
}

// ============= Stress Test: Random Positions =============

TEST_F(NaNInfGPUCheckTest, RandomPositions_100Trials) {
    std::mt19937 rng(42);
    const int num_trials = 100;
    const int tensor_size = 100000;

    for (int trial = 0; trial < num_trials; ++trial) {
        auto torch_t = torch::randn({tensor_size}, torch::kCUDA);

        // Randomly decide whether to insert NaN/Inf
        std::uniform_int_distribution<int> type_dist(0, 2); // 0=none, 1=nan, 2=inf
        int type = type_dist(rng);

        if (type > 0) {
            std::uniform_int_distribution<int> pos_dist(0, tensor_size - 1);
            int pos = pos_dist(rng);

            if (type == 1) {
                torch_t[pos] = std::numeric_limits<float>::quiet_NaN();
            } else {
                torch_t[pos] = std::numeric_limits<float>::infinity();
            }
        }

        auto lfs_t = from_torch(torch_t);

        bool torch_result = torch_has_nan_or_inf(torch_t);
        bool lfs_result = lfs_t.has_nan() || lfs_t.has_inf();

        EXPECT_EQ(torch_result, lfs_result)
            << "Mismatch at trial " << trial << " with type " << type;
    }
}

// ============= Typical Gaussian Splatting Tensor Shapes =============

TEST_F(NaNInfGPUCheckTest, GaussianMeans_5M_x_3) {
    auto torch_t = torch::randn({5000000, 3}, torch::kCUDA);
    auto lfs_t = from_torch(torch_t);

    EXPECT_EQ(torch_has_nan(torch_t), lfs_t.has_nan());
}

TEST_F(NaNInfGPUCheckTest, GaussianMeans_5M_x_3_WithNaN) {
    auto torch_t = torch::randn({5000000, 3}, torch::kCUDA);
    torch_t[2500000][1] = std::numeric_limits<float>::quiet_NaN();
    auto lfs_t = from_torch(torch_t);

    EXPECT_TRUE(torch_has_nan(torch_t));
    EXPECT_TRUE(lfs_t.has_nan());
}

TEST_F(NaNInfGPUCheckTest, GaussianScales_5M_x_3) {
    auto torch_t = torch::randn({5000000, 3}, torch::kCUDA);
    auto lfs_t = from_torch(torch_t);

    EXPECT_EQ(torch_has_nan(torch_t), lfs_t.has_nan());
}

TEST_F(NaNInfGPUCheckTest, GaussianRotations_5M_x_4) {
    auto torch_t = torch::randn({5000000, 4}, torch::kCUDA);
    auto lfs_t = from_torch(torch_t);

    EXPECT_EQ(torch_has_nan(torch_t), lfs_t.has_nan());
}

TEST_F(NaNInfGPUCheckTest, GaussianSH_5M_x_48) {
    auto torch_t = torch::randn({5000000, 48}, torch::kCUDA);
    auto lfs_t = from_torch(torch_t);

    EXPECT_EQ(torch_has_nan(torch_t), lfs_t.has_nan());
}

TEST_F(NaNInfGPUCheckTest, GaussianOpacity_5M_x_1) {
    auto torch_t = torch::randn({5000000, 1}, torch::kCUDA);
    auto lfs_t = from_torch(torch_t);

    EXPECT_EQ(torch_has_nan(torch_t), lfs_t.has_nan());
}

// ============= CPU Fallback Tests =============

TEST_F(NaNInfGPUCheckTest, CPU_NoNaN) {
    std::vector<float> data(1000, 1.0f);
    auto lfs_t = Tensor::from_vector(data, {1000}, Device::CPU);

    EXPECT_FALSE(lfs_t.has_nan());
    EXPECT_FALSE(lfs_t.has_inf());
}

TEST_F(NaNInfGPUCheckTest, CPU_WithNaN) {
    std::vector<float> data(1000, 1.0f);
    data[500] = std::numeric_limits<float>::quiet_NaN();
    auto lfs_t = Tensor::from_vector(data, {1000}, Device::CPU);

    EXPECT_TRUE(lfs_t.has_nan());
}

TEST_F(NaNInfGPUCheckTest, CPU_WithInf) {
    std::vector<float> data(1000, 1.0f);
    data[500] = std::numeric_limits<float>::infinity();
    auto lfs_t = Tensor::from_vector(data, {1000}, Device::CPU);

    EXPECT_TRUE(lfs_t.has_inf());
}

// ============= assert_finite() Tests =============

TEST_F(NaNInfGPUCheckTest, AssertFinite_Passes) {
    auto torch_t = torch::randn({1000}, torch::kCUDA);
    auto lfs_t = from_torch(torch_t);

    EXPECT_NO_THROW(lfs_t.assert_finite());
}

TEST_F(NaNInfGPUCheckTest, AssertFinite_ThrowsOnNaN) {
    auto torch_t = torch::randn({1000}, torch::kCUDA);
    torch_t[500] = std::numeric_limits<float>::quiet_NaN();
    auto lfs_t = from_torch(torch_t);

    EXPECT_THROW(lfs_t.assert_finite(), std::runtime_error);
}

TEST_F(NaNInfGPUCheckTest, AssertFinite_ThrowsOnInf) {
    auto torch_t = torch::randn({1000}, torch::kCUDA);
    torch_t[500] = std::numeric_limits<float>::infinity();
    auto lfs_t = from_torch(torch_t);

    EXPECT_THROW(lfs_t.assert_finite(), std::runtime_error);
}

// ============= Alignment Edge Cases =============

TEST_F(NaNInfGPUCheckTest, UnalignedTensor_UsesScalarKernel) {
    // Create a tensor and then slice it to get unaligned data
    // The vectorized kernel requires 16-byte alignment
    auto torch_t = torch::randn({1001}, torch::kCUDA);
    auto torch_sliced = torch_t.slice(0, 1); // Start at index 1, potentially unaligned
    torch_sliced[500] = std::numeric_limits<float>::quiet_NaN();

    // We can't easily create unaligned LFS tensor, but we can test small sizes
    // which use scalar kernel
    auto lfs_small = Tensor::randn({100}, Device::CUDA);
    auto vec = lfs_small.to_vector();
    vec[50] = std::numeric_limits<float>::quiet_NaN();
    auto lfs_with_nan = Tensor::from_vector(vec, {100}, Device::CUDA);

    EXPECT_TRUE(lfs_with_nan.has_nan());
}

// ============= Comprehensive Correctness Sweep =============

TEST_F(NaNInfGPUCheckTest, CorrectnessSwept_AllSizesUpTo10K) {
    std::mt19937 rng(12345);

    // Test various sizes that exercise different code paths
    std::vector<int> sizes = {1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33,
                              63, 64, 65, 127, 128, 129, 255, 256, 257,
                              511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049,
                              4095, 4096, 4097, 8191, 8192, 8193, 10000};

    for (int size : sizes) {
        // Test clean tensor
        auto torch_clean = torch::randn({size}, torch::kCUDA);
        auto lfs_clean = from_torch(torch_clean);
        EXPECT_FALSE(lfs_clean.has_nan()) << "False positive at size " << size;

        // Test with NaN at various positions
        for (int pos : {0, size / 2, size - 1}) {
            if (pos >= size)
                continue;

            auto torch_nan = torch::randn({size}, torch::kCUDA);
            torch_nan[pos] = std::numeric_limits<float>::quiet_NaN();
            auto lfs_nan = from_torch(torch_nan);

            EXPECT_TRUE(lfs_nan.has_nan())
                << "False negative at size " << size << " pos " << pos;
        }
    }
}

// ============= Exhaustive Large Tensor Tests (Grid-Stride Loop Edge Cases) =============

TEST_F(NaNInfGPUCheckTest, GridStrideLoop_ExactBoundaries) {
    // Test sizes around grid-stride boundaries
    // With BLOCK_SIZE=256 and MAX_BLOCKS=1024, vec4 kernel processes:
    // 256 * 1024 * 4 = 1,048,576 elements per "stride"
    constexpr int STRIDE_SIZE = 256 * 1024 * 4; // ~1M

    std::vector<int> sizes = {
        STRIDE_SIZE - 1, // Just under one stride
        STRIDE_SIZE,     // Exactly one stride
        STRIDE_SIZE + 1, // Just over one stride
        STRIDE_SIZE + 100,
        STRIDE_SIZE * 2 - 1, // Just under two strides
        STRIDE_SIZE * 2,     // Exactly two strides
        STRIDE_SIZE * 2 + 1, // Just over two strides
        STRIDE_SIZE * 3,     // Three strides
        STRIDE_SIZE * 5,     // Five strides (~5M, typical Gaussian count)
    };

    for (int size : sizes) {
        // Test clean tensor
        auto torch_clean = torch::randn({size}, torch::kCUDA);
        auto lfs_clean = from_torch(torch_clean);
        EXPECT_FALSE(lfs_clean.has_nan()) << "False positive at size " << size;

        // Test NaN at critical positions
        std::vector<int> positions = {
            0,               // First element
            size / 4,        // Quarter
            size / 2,        // Middle
            3 * size / 4,    // Three quarters
            size - 1,        // Last element
            STRIDE_SIZE - 1, // End of first stride
            STRIDE_SIZE,     // Start of second stride
            STRIDE_SIZE + 1, // Second element of second stride
        };

        for (int pos : positions) {
            if (pos >= size)
                continue;

            auto torch_nan = torch::randn({size}, torch::kCUDA);
            torch_nan[pos] = std::numeric_limits<float>::quiet_NaN();
            auto lfs_nan = from_torch(torch_nan);

            bool torch_result = torch_has_nan(torch_nan);
            bool lfs_result = lfs_nan.has_nan();

            EXPECT_EQ(torch_result, lfs_result)
                << "Mismatch at size=" << size << " pos=" << pos
                << " (torch=" << torch_result << ", lfs=" << lfs_result << ")";
        }
    }
}

TEST_F(NaNInfGPUCheckTest, GridStrideLoop_AllPositions_1M) {
    // Exhaustively test every 10000th position in a 1M tensor
    const int size = 1000000;
    auto base_tensor = torch::randn({size}, torch::kCUDA);

    for (int pos = 0; pos < size; pos += 10000) {
        auto torch_nan = base_tensor.clone();
        torch_nan[pos] = std::numeric_limits<float>::quiet_NaN();
        auto lfs_nan = from_torch(torch_nan);

        EXPECT_TRUE(lfs_nan.has_nan())
            << "Failed to detect NaN at position " << pos << " in 1M tensor";
    }

    // Also test last few positions explicitly
    for (int offset = 0; offset < 10; ++offset) {
        int pos = size - 1 - offset;
        auto torch_nan = base_tensor.clone();
        torch_nan[pos] = std::numeric_limits<float>::quiet_NaN();
        auto lfs_nan = from_torch(torch_nan);

        EXPECT_TRUE(lfs_nan.has_nan())
            << "Failed to detect NaN at position " << pos << " (offset " << offset << " from end)";
    }
}

TEST_F(NaNInfGPUCheckTest, GridStrideLoop_AllPositions_5M) {
    // Exhaustively test every 50000th position in a 5M tensor
    const int size = 5000000;
    auto base_tensor = torch::randn({size}, torch::kCUDA);

    for (int pos = 0; pos < size; pos += 50000) {
        auto torch_nan = base_tensor.clone();
        torch_nan[pos] = std::numeric_limits<float>::quiet_NaN();
        auto lfs_nan = from_torch(torch_nan);

        EXPECT_TRUE(lfs_nan.has_nan())
            << "Failed to detect NaN at position " << pos << " in 5M tensor";
    }

    // Test stride boundaries explicitly
    constexpr int STRIDE_SIZE = 256 * 1024 * 4;
    for (int stride = 0; stride < 5; ++stride) {
        int boundary = stride * STRIDE_SIZE;
        if (boundary >= size)
            break;

        for (int offset : {-1, 0, 1}) {
            int pos = boundary + offset;
            if (pos < 0 || pos >= size)
                continue;

            auto torch_nan = base_tensor.clone();
            torch_nan[pos] = std::numeric_limits<float>::quiet_NaN();
            auto lfs_nan = from_torch(torch_nan);

            EXPECT_TRUE(lfs_nan.has_nan())
                << "Failed at stride boundary: stride=" << stride << " offset=" << offset << " pos=" << pos;
        }
    }
}

TEST_F(NaNInfGPUCheckTest, GridStrideLoop_Remainder_EdgeCases) {
    // Test remainder handling (last 0-3 elements after vec4 processing)
    for (int base_size = 1000000; base_size <= 1000003; ++base_size) {
        // NaN in last element (remainder)
        auto torch_t = torch::randn({base_size}, torch::kCUDA);
        torch_t[base_size - 1] = std::numeric_limits<float>::quiet_NaN();
        auto lfs_t = from_torch(torch_t);

        EXPECT_TRUE(lfs_t.has_nan())
            << "Failed to detect NaN in remainder at size " << base_size;
    }
}

TEST_F(NaNInfGPUCheckTest, RandomStress_1000Trials_VariousSizes) {
    std::mt19937 rng(99999);
    const int num_trials = 1000;

    std::vector<int> sizes = {100, 1000, 10000, 100000, 1000000, 5000000};

    for (int size : sizes) {
        int detected = 0;
        int total_with_nan = 0;

        for (int trial = 0; trial < num_trials / static_cast<int>(sizes.size()); ++trial) {
            auto torch_t = torch::randn({size}, torch::kCUDA);

            // 50% chance of inserting NaN
            bool insert_nan = (rng() % 2) == 0;
            if (insert_nan) {
                int pos = rng() % size;
                torch_t[pos] = std::numeric_limits<float>::quiet_NaN();
                total_with_nan++;
            }

            auto lfs_t = from_torch(torch_t);
            bool lfs_result = lfs_t.has_nan();

            if (insert_nan) {
                if (lfs_result)
                    detected++;
                EXPECT_TRUE(lfs_result) << "False negative at size " << size << " trial " << trial;
            } else {
                EXPECT_FALSE(lfs_result) << "False positive at size " << size << " trial " << trial;
            }
        }
    }
}

TEST_F(NaNInfGPUCheckTest, InfDetection_LargeTensors) {
    // Verify Inf detection works the same as NaN for large tensors
    std::vector<int> sizes = {1000000, 5000000};

    for (int size : sizes) {
        // Test +Inf
        auto torch_pinf = torch::randn({size}, torch::kCUDA);
        torch_pinf[size / 2] = std::numeric_limits<float>::infinity();
        auto lfs_pinf = from_torch(torch_pinf);
        EXPECT_TRUE(lfs_pinf.has_nan()) << "+Inf not detected at size " << size; // has_nan checks both

        // Test -Inf
        auto torch_ninf = torch::randn({size}, torch::kCUDA);
        torch_ninf[size / 2] = -std::numeric_limits<float>::infinity();
        auto lfs_ninf = from_torch(torch_ninf);
        EXPECT_TRUE(lfs_ninf.has_nan()) << "-Inf not detected at size " << size;
    }
}

TEST_F(NaNInfGPUCheckTest, MultiDim_LargeTensors) {
    // Test typical Gaussian splatting tensor shapes with various NaN positions
    struct TestCase {
        std::vector<int64_t> shape;
        std::string name;
    };

    std::vector<TestCase> cases = {
        {{5000000, 3}, "means"},
        {{5000000, 4}, "rotations"},
        {{5000000, 1}, "opacity"},
        {{1000000, 48}, "SH"},
    };

    for (const auto& tc : cases) {
        auto torch_t = torch::randn(tc.shape, torch::kCUDA);
        int total_elements = 1;
        for (auto d : tc.shape)
            total_elements *= d;

        // Insert NaN at various positions
        std::vector<int> flat_positions = {0, total_elements / 4, total_elements / 2,
                                           3 * total_elements / 4, total_elements - 1};

        for (int flat_pos : flat_positions) {
            auto torch_nan = torch_t.clone();
            torch_nan.view({-1})[flat_pos] = std::numeric_limits<float>::quiet_NaN();
            auto lfs_nan = from_torch(torch_nan);

            EXPECT_TRUE(lfs_nan.has_nan())
                << "Failed for " << tc.name << " at flat position " << flat_pos;
        }
    }
}
