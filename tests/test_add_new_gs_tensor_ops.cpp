/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <vector>

using namespace lfs::core;

/**
 * Test suite for tensor operations used in MCMC::add_new_gs()
 *
 * All operations are verified against LibTorch as the bug-free reference.
 * These tests cover the exact operation combinations used in add_new_gs:
 *
 * 1. Get opacities and handle shape (squeeze)
 * 2. Flatten for multinomial sampling
 * 3. Index selection to gather parameters
 * 4. Count occurrences: zeros -> index_add_ -> index_select -> add -> clamp -> to(Int32)
 * 5. Clamp opacities and compute logit: clamp -> division -> log
 * 6. Unsqueeze for dimension matching
 * 7. Log for scaling conversion
 */
class AddNewGsTensorOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA if needed
    }
};

// Test 1: zeros() with different shapes and dtypes
TEST_F(AddNewGsTensorOpsTest, Zeros_LibTorchComparison) {
    // Test Float32
    {
        const size_t N = 100;
        auto zeros_lfs = Tensor::zeros({N}, Device::GPU, DataType::Float32);
        auto result_lfs = zeros_lfs.cpu().to_vector();

        auto zeros_torch = torch::zeros({static_cast<long>(N)}, torch::kFloat32).cuda();
        auto result_torch = zeros_torch.cpu();

        for (size_t i = 0; i < N; ++i) {
            float lfs_val = result_lfs[i];
            float torch_val = result_torch[i].item<float>();
            EXPECT_EQ(lfs_val, torch_val) << "Mismatch at index " << i;
        }
    }

    // Test Int32
    {
        const size_t N = 100;
        auto zeros_lfs = Tensor::zeros({N}, Device::GPU, DataType::Int32);
        auto result_lfs = zeros_lfs.cpu().to_vector_int();

        auto zeros_torch = torch::zeros({static_cast<long>(N)}, torch::kInt32).cuda();
        auto result_torch = zeros_torch.cpu();

        for (size_t i = 0; i < N; ++i) {
            int32_t lfs_val = result_lfs[i];
            int32_t torch_val = result_torch[i].item<int32_t>();
            EXPECT_EQ(lfs_val, torch_val) << "Mismatch at index " << i;
        }
    }
}

// Test 2: index_add_ for counting occurrences (Float32)
TEST_F(AddNewGsTensorOpsTest, IndexAddFloat32_LibTorchComparison) {
    const size_t N = 20;

    // Base tensor: zeros
    std::vector<float> base_data(N, 0.0f);

    // Indices to add to (with duplicates to test counting)
    std::vector<int32_t> indices = {0, 5, 10, 5, 15, 10, 10, 0, 19, 5}; // 10 samples

    // Values to add (all ones for counting)
    std::vector<float> values(10, 1.0f);

    // LFS
    auto base_lfs = Tensor::from_vector(base_data, TensorShape{N}, Device::GPU);
    auto indices_lfs = Tensor::from_vector(indices, TensorShape{10}, Device::GPU);
    auto values_lfs = Tensor::from_vector(values, TensorShape{10}, Device::GPU);

    auto result_lfs_tensor = base_lfs.index_add_(0, indices_lfs, values_lfs);
    auto result_lfs = result_lfs_tensor.cpu().to_vector();

    // LibTorch (convert indices to int64)
    std::vector<int64_t> indices_i64(indices.begin(), indices.end());
    auto base_torch = torch::zeros({static_cast<long>(N)}, torch::kFloat32).cuda();
    auto indices_torch = torch::from_blob(
                             const_cast<int64_t*>(indices_i64.data()),
                             {10},
                             torch::kInt64)
                             .clone()
                             .cuda();
    auto values_torch = torch::ones({10}, torch::kFloat32).cuda();

    auto result_torch = base_torch.index_add_(0, indices_torch, values_torch).cpu();

    // Compare
    for (size_t i = 0; i < N; ++i) {
        float lfs_val = result_lfs[i];
        float torch_val = result_torch[i].item<float>();
        float diff = std::abs(lfs_val - torch_val);

        EXPECT_LT(diff, 1e-6f)
            << "Mismatch at index " << i
            << ": LFS=" << lfs_val << " vs LibTorch=" << torch_val;
    }

    // Expected: index 0 -> 2, index 5 -> 3, index 10 -> 3, index 15 -> 1, index 19 -> 1
    EXPECT_NEAR(result_lfs[0], 2.0f, 1e-6f);
    EXPECT_NEAR(result_lfs[5], 3.0f, 1e-6f);
    EXPECT_NEAR(result_lfs[10], 3.0f, 1e-6f);
    EXPECT_NEAR(result_lfs[15], 1.0f, 1e-6f);
    EXPECT_NEAR(result_lfs[19], 1.0f, 1e-6f);
}

// Test 3: Chained operation - count occurrences pattern from add_new_gs
// zeros -> index_add_ -> index_select -> add ones -> clamp -> to(Int32)
TEST_F(AddNewGsTensorOpsTest, CountOccurrencesPattern_LibTorchComparison) {
    const size_t N = 50;
    const int n_samples = 20;
    const int n_max = 10;

    // Sampled indices (with duplicates)
    std::vector<int32_t> sampled_indices = {
        0, 5, 10, 5, 15, 10, 10, 0, 19, 5,     // first 10
        25, 30, 25, 35, 40, 25, 30, 45, 49, 25 // next 10
    };

    // LFS: zeros -> index_add_ -> index_select -> add ones -> clamp -> to(Int32)
    auto ratios_lfs = Tensor::zeros({N}, Device::GPU, DataType::Float32);
    auto sampled_idxs_lfs = Tensor::from_vector(sampled_indices, TensorShape{static_cast<size_t>(n_samples)}, Device::GPU);
    auto ones_lfs = Tensor::ones({static_cast<size_t>(n_samples)}, Device::GPU, DataType::Float32);

    ratios_lfs = ratios_lfs.index_add_(0, sampled_idxs_lfs, ones_lfs);
    ratios_lfs = ratios_lfs.index_select(0, sampled_idxs_lfs);
    ratios_lfs = ratios_lfs + Tensor::ones_like(ratios_lfs); // Add 1 more
    ratios_lfs = ratios_lfs.clamp(1.0f, static_cast<float>(n_max));
    ratios_lfs = ratios_lfs.to(DataType::Int32).contiguous();

    auto result_lfs = ratios_lfs.cpu().to_vector_int();

    // LibTorch: same operations (convert indices to int64)
    std::vector<int64_t> sampled_indices_i64(sampled_indices.begin(), sampled_indices.end());
    auto ratios_torch = torch::zeros({static_cast<long>(N)}, torch::kFloat32).cuda();
    auto sampled_idxs_torch = torch::from_blob(
                                  const_cast<int64_t*>(sampled_indices_i64.data()),
                                  {n_samples},
                                  torch::kInt64)
                                  .clone()
                                  .cuda();
    auto ones_torch = torch::ones({n_samples}, torch::kFloat32).cuda();

    ratios_torch = ratios_torch.index_add_(0, sampled_idxs_torch, ones_torch);
    ratios_torch = ratios_torch.index_select(0, sampled_idxs_torch);
    ratios_torch = ratios_torch + torch::ones_like(ratios_torch);
    ratios_torch = ratios_torch.clamp(1.0f, static_cast<float>(n_max));
    ratios_torch = ratios_torch.to(torch::kInt32).contiguous();

    auto result_torch = ratios_torch.cpu();

    // Compare
    for (int i = 0; i < n_samples; ++i) {
        int32_t lfs_val = result_lfs[i];
        int32_t torch_val = result_torch[i].item<int32_t>();

        EXPECT_EQ(lfs_val, torch_val)
            << "Mismatch at sample " << i
            << " (index=" << sampled_indices[i] << ")"
            << ": LFS=" << lfs_val << " vs LibTorch=" << torch_val;
    }

    // Expected counts (clamped to n_max=10):
    // index 0 -> 2+1=3, index 5 -> 3+1=4, index 10 -> 3+1=4
    // index 25 -> 4+1=5, index 30 -> 2+1=3
    EXPECT_EQ(result_lfs[0], 3); // sampled_indices[0] = 0, count=2, +1 = 3
    EXPECT_EQ(result_lfs[1], 4); // sampled_indices[1] = 5, count=3, +1 = 4
}

// Test 4: Opacity clamping and logit calculation
// clamp(min, max) -> division: p / (1 - p) -> log()
TEST_F(AddNewGsTensorOpsTest, OpacityLogitCalculation_LibTorchComparison) {
    const float min_opacity = 0.005f;
    const float max_opacity = 1.0f - 1e-7f;

    // Test values including edge cases
    std::vector<float> opacities = {
        0.0f,   // Below min -> clamps to min
        0.001f, // Below min -> clamps to min
        0.005f, // At min
        0.01f,  // Normal
        0.1f,   // Normal
        0.5f,   // Normal
        0.9f,   // Normal
        0.99f,  // High
        1.0f,   // At max -> clamps to max_opacity
        1.1f    // Above max -> clamps to max_opacity
    };

    // LFS: clamp -> logit
    auto opacities_lfs = Tensor::from_vector(opacities, TensorShape{10}, Device::GPU);
    auto clamped_lfs = opacities_lfs.clamp(min_opacity, max_opacity);
    auto logit_lfs = (clamped_lfs / (Tensor::ones_like(clamped_lfs) - clamped_lfs)).log();
    auto result_lfs = logit_lfs.cpu().to_vector();

    // LibTorch: same operations
    auto opacities_torch = torch::from_blob(
                               const_cast<float*>(opacities.data()),
                               {10},
                               torch::kFloat32)
                               .clone()
                               .cuda();
    auto clamped_torch = opacities_torch.clamp(min_opacity, max_opacity);
    auto logit_torch = (clamped_torch / (torch::ones_like(clamped_torch) - clamped_torch)).log();
    auto result_torch = logit_torch.cpu();

    // Compare
    for (size_t i = 0; i < 10; ++i) {
        float lfs_val = result_lfs[i];
        float torch_val = result_torch[i].item<float>();
        float diff = std::abs(lfs_val - torch_val);

        EXPECT_LT(diff, 1e-5f)
            << "Mismatch at index " << i
            << " (opacity=" << opacities[i] << ")"
            << ": LFS=" << lfs_val << " vs LibTorch=" << torch_val;
    }
}

// Test 5: Scale log conversion
TEST_F(AddNewGsTensorOpsTest, ScaleLogConversion_LibTorchComparison) {
    // Test various scale values
    std::vector<float> scales = {
        0.001f, 0.01f, 0.1f, 0.5f, 1.0f, 2.0f, 5.0f, 10.0f, 100.0f};

    // LFS
    auto scales_lfs = Tensor::from_vector(scales, TensorShape{9}, Device::GPU);
    auto log_scales_lfs = scales_lfs.log();
    auto result_lfs = log_scales_lfs.cpu().to_vector();

    // LibTorch
    auto scales_torch = torch::from_blob(
                            const_cast<float*>(scales.data()),
                            {9},
                            torch::kFloat32)
                            .clone()
                            .cuda();
    auto log_scales_torch = scales_torch.log();
    auto result_torch = log_scales_torch.cpu();

    // Compare
    for (size_t i = 0; i < 9; ++i) {
        float lfs_val = result_lfs[i];
        float torch_val = result_torch[i].item<float>();
        float diff = std::abs(lfs_val - torch_val);

        EXPECT_LT(diff, 1e-6f)
            << "Mismatch at index " << i
            << " (scale=" << scales[i] << ")"
            << ": LFS=" << lfs_val << " vs LibTorch=" << torch_val;
    }
}

// Test 6: Unsqueeze/Squeeze for dimension matching
TEST_F(AddNewGsTensorOpsTest, UnsqueezeSqueeze_LibTorchComparison) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    // Test unsqueeze(-1): [5] -> [5, 1]
    {
        auto tensor_lfs = Tensor::from_vector(data, TensorShape{5}, Device::GPU);
        auto unsqueezed_lfs = tensor_lfs.unsqueeze(-1);

        EXPECT_EQ(unsqueezed_lfs.ndim(), 2);
        EXPECT_EQ(unsqueezed_lfs.shape()[0], 5);
        EXPECT_EQ(unsqueezed_lfs.shape()[1], 1);

        auto result_lfs = unsqueezed_lfs.cpu().to_vector();

        auto tensor_torch = torch::from_blob(
                                const_cast<float*>(data.data()),
                                {5},
                                torch::kFloat32)
                                .clone()
                                .cuda();
        auto unsqueezed_torch = tensor_torch.unsqueeze(-1);
        auto result_torch = unsqueezed_torch.cpu();

        for (size_t i = 0; i < 5; ++i) {
            float lfs_val = result_lfs[i];
            float torch_val = result_torch[i][0].item<float>();
            EXPECT_NEAR(lfs_val, torch_val, 1e-6f);
        }
    }

    // Test squeeze(-1): [5, 1] -> [5]
    {
        auto tensor_lfs = Tensor::from_vector(data, TensorShape{5, 1}, Device::GPU);
        auto squeezed_lfs = tensor_lfs.squeeze(-1);

        EXPECT_EQ(squeezed_lfs.ndim(), 1);
        EXPECT_EQ(squeezed_lfs.shape()[0], 5);

        auto result_lfs = squeezed_lfs.cpu().to_vector();

        auto tensor_torch = torch::from_blob(
                                const_cast<float*>(data.data()),
                                {5, 1},
                                torch::kFloat32)
                                .clone()
                                .cuda();
        auto squeezed_torch = tensor_torch.squeeze(-1);
        auto result_torch = squeezed_torch.cpu();

        for (size_t i = 0; i < 5; ++i) {
            float lfs_val = result_lfs[i];
            float torch_val = result_torch[i].item<float>();
            EXPECT_NEAR(lfs_val, torch_val, 1e-6f);
        }
    }
}

// Test 7: Index selection on multiple tensors (gathering parameters)
TEST_F(AddNewGsTensorOpsTest, MultipleIndexSelect_LibTorchComparison) {
    const size_t N = 100;
    const size_t n_samples = 10;

    // Create test data
    std::vector<float> opacities(N);
    std::vector<float> scales(N * 3);
    for (size_t i = 0; i < N; ++i) {
        opacities[i] = 0.01f * (i + 1);
        scales[i * 3 + 0] = 0.1f * (i + 1);
        scales[i * 3 + 1] = 0.2f * (i + 1);
        scales[i * 3 + 2] = 0.3f * (i + 1);
    }

    // Indices to sample
    std::vector<int32_t> indices = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90};

    // LFS
    auto opacities_lfs = Tensor::from_vector(opacities, TensorShape{N}, Device::GPU);
    auto scales_lfs = Tensor::from_vector(scales, TensorShape{N, 3}, Device::GPU);
    auto indices_lfs = Tensor::from_vector(indices, TensorShape{n_samples}, Device::GPU);

    auto sampled_opacities_lfs = opacities_lfs.index_select(0, indices_lfs);
    auto sampled_scales_lfs = scales_lfs.index_select(0, indices_lfs);

    auto result_opacities_lfs = sampled_opacities_lfs.cpu().to_vector();
    auto result_scales_lfs = sampled_scales_lfs.cpu().to_vector();

    // LibTorch (convert indices to int64)
    std::vector<int64_t> indices_i64(indices.begin(), indices.end());
    auto opacities_torch = torch::from_blob(
                               const_cast<float*>(opacities.data()),
                               {static_cast<long>(N)},
                               torch::kFloat32)
                               .clone()
                               .cuda();
    auto scales_torch = torch::from_blob(
                            const_cast<float*>(scales.data()),
                            {static_cast<long>(N), 3},
                            torch::kFloat32)
                            .clone()
                            .cuda();
    auto indices_torch = torch::from_blob(
                             const_cast<int64_t*>(indices_i64.data()),
                             {static_cast<long>(n_samples)},
                             torch::kInt64)
                             .clone()
                             .cuda();

    auto sampled_opacities_torch = opacities_torch.index_select(0, indices_torch);
    auto sampled_scales_torch = scales_torch.index_select(0, indices_torch);

    auto result_opacities_torch = sampled_opacities_torch.cpu();
    auto result_scales_torch = sampled_scales_torch.cpu();

    // Compare opacities
    for (size_t i = 0; i < n_samples; ++i) {
        float lfs_val = result_opacities_lfs[i];
        float torch_val = result_opacities_torch[i].item<float>();
        float diff = std::abs(lfs_val - torch_val);

        EXPECT_LT(diff, 1e-6f)
            << "Opacity mismatch at sample " << i
            << " (index=" << indices[i] << ")"
            << ": LFS=" << lfs_val << " vs LibTorch=" << torch_val;
    }

    // Compare scales
    for (size_t i = 0; i < n_samples; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            float lfs_val = result_scales_lfs[i * 3 + j];
            float torch_val = result_scales_torch[i][j].item<float>();
            float diff = std::abs(lfs_val - torch_val);

            EXPECT_LT(diff, 1e-6f)
                << "Scale mismatch at sample " << i << ", dim " << j
                << " (index=" << indices[i] << ")"
                << ": LFS=" << lfs_val << " vs LibTorch=" << torch_val;
        }
    }
}

// Test 8: Multinomial sampling (THE KEY OPERATION in add_new_gs!)
TEST_F(AddNewGsTensorOpsTest, MultinomialSampling_LibTorchComparison) {
    const size_t N = 100;
    const int n_samples = 20;

    // Create probability weights (simulating opacities)
    std::vector<float> probs(N);
    for (size_t i = 0; i < N; ++i) {
        probs[i] = 0.01f * (i % 10 + 1); // Values 0.01 to 0.10
    }

    // LFS
    auto probs_lfs = Tensor::from_vector(probs, TensorShape{N}, Device::GPU);
    auto sampled_lfs = Tensor::multinomial(probs_lfs, n_samples, true); // with replacement
    auto result_lfs = sampled_lfs.cpu().to_vector_int();

    // LibTorch
    auto probs_torch = torch::from_blob(
                           const_cast<float*>(probs.data()),
                           {static_cast<long>(N)},
                           torch::kFloat32)
                           .clone()
                           .cuda();
    auto sampled_torch = probs_torch.multinomial(n_samples, true); // with replacement
    auto result_torch = sampled_torch.cpu();

    // Note: We can't compare exact indices (multinomial is random),
    // but we can verify the output properties:

    // 1. Correct number of samples
    EXPECT_EQ(result_lfs.size(), static_cast<size_t>(n_samples));
    EXPECT_EQ(result_torch.size(0), n_samples);

    // 2. All indices are valid (within range [0, N))
    for (size_t i = 0; i < result_lfs.size(); ++i) {
        EXPECT_GE(result_lfs[i], 0) << "LFS: Index out of range at position " << i;
        EXPECT_LT(result_lfs[i], static_cast<int>(N)) << "LFS: Index out of range at position " << i;
    }

    for (int i = 0; i < n_samples; ++i) {
        int idx = result_torch[i].item<int>();
        EXPECT_GE(idx, 0) << "Torch: Index out of range at position " << i;
        EXPECT_LT(idx, static_cast<int>(N)) << "Torch: Index out of range at position " << i;
    }

    // 3. Test without replacement (should have no duplicates)
    auto sampled_no_replace_lfs = Tensor::multinomial(probs_lfs, 10, false);
    auto result_no_replace_lfs = sampled_no_replace_lfs.cpu().to_vector_int();

    // All indices valid
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_GE(result_no_replace_lfs[i], 0);
        EXPECT_LT(result_no_replace_lfs[i], static_cast<int>(N));
    }

    // No duplicates when replacement=false
    std::set<int> unique_no_replace(result_no_replace_lfs.begin(), result_no_replace_lfs.end());
    EXPECT_EQ(unique_no_replace.size(), 10u) << "LFS: Without replacement should have no duplicates";
}

// Test 9: Flatten operation (used for multinomial sampling)
TEST_F(AddNewGsTensorOpsTest, Flatten_LibTorchComparison) {
    // Test with 1D tensor (already flat)
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

        auto tensor_lfs = Tensor::from_vector(data, TensorShape{5}, Device::GPU);
        auto flattened_lfs = tensor_lfs.flatten();

        EXPECT_EQ(flattened_lfs.ndim(), 1);
        EXPECT_EQ(flattened_lfs.shape()[0], 5);

        auto result_lfs = flattened_lfs.cpu().to_vector();

        auto tensor_torch = torch::from_blob(
                                const_cast<float*>(data.data()),
                                {5},
                                torch::kFloat32)
                                .clone()
                                .cuda();
        auto flattened_torch = tensor_torch.flatten();
        auto result_torch = flattened_torch.cpu();

        for (size_t i = 0; i < 5; ++i) {
            float lfs_val = result_lfs[i];
            float torch_val = result_torch[i].item<float>();
            EXPECT_NEAR(lfs_val, torch_val, 1e-6f);
        }
    }

    // Test with 2D tensor [N, 1]
    {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

        auto tensor_lfs = Tensor::from_vector(data, TensorShape{5, 1}, Device::GPU);
        auto flattened_lfs = tensor_lfs.flatten();

        EXPECT_EQ(flattened_lfs.ndim(), 1);
        EXPECT_EQ(flattened_lfs.shape()[0], 5);

        auto result_lfs = flattened_lfs.cpu().to_vector();

        auto tensor_torch = torch::from_blob(
                                const_cast<float*>(data.data()),
                                {5, 1},
                                torch::kFloat32)
                                .clone()
                                .cuda();
        auto flattened_torch = tensor_torch.flatten();
        auto result_torch = flattened_torch.cpu();

        for (size_t i = 0; i < 5; ++i) {
            float lfs_val = result_lfs[i];
            float torch_val = result_torch[i].item<float>();
            EXPECT_NEAR(lfs_val, torch_val, 1e-6f);
        }
    }
}

// Test 9: Add operation (tensor + tensor)
TEST_F(AddNewGsTensorOpsTest, TensorAddition_LibTorchComparison) {
    std::vector<float> data1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> data2 = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f};

    // LFS
    auto tensor1_lfs = Tensor::from_vector(data1, TensorShape{5}, Device::GPU);
    auto tensor2_lfs = Tensor::from_vector(data2, TensorShape{5}, Device::GPU);
    auto result_lfs_tensor = tensor1_lfs + tensor2_lfs;
    auto result_lfs = result_lfs_tensor.cpu().to_vector();

    // LibTorch
    auto tensor1_torch = torch::from_blob(
                             const_cast<float*>(data1.data()),
                             {5},
                             torch::kFloat32)
                             .clone()
                             .cuda();
    auto tensor2_torch = torch::from_blob(
                             const_cast<float*>(data2.data()),
                             {5},
                             torch::kFloat32)
                             .clone()
                             .cuda();
    auto result_torch = (tensor1_torch + tensor2_torch).cpu();

    // Compare
    for (size_t i = 0; i < 5; ++i) {
        float lfs_val = result_lfs[i];
        float torch_val = result_torch[i].item<float>();
        float diff = std::abs(lfs_val - torch_val);

        EXPECT_LT(diff, 1e-6f)
            << "Mismatch at index " << i
            << ": LFS=" << lfs_val << " vs LibTorch=" << torch_val;
    }
}

// Test 10: Full add_new_gs workflow simulation
TEST_F(AddNewGsTensorOpsTest, FullAddNewWorkflow_LibTorchComparison) {
    const size_t N = 50;
    const int n_new = 10;
    const int n_max = 20;
    const float min_opacity = 0.005f;
    const float max_opacity = 1.0f - 1e-7f;

    // Initial opacities
    std::vector<float> opacities(N);
    for (size_t i = 0; i < N; ++i) {
        opacities[i] = 0.01f * (i % 10 + 1); // Values 0.01 to 0.10
    }

    // Sampled indices
    std::vector<int32_t> sampled_indices = {5, 10, 15, 5, 20, 25, 10, 30, 35, 5};

    // Initial scales [N, 3]
    std::vector<float> scales(N * 3);
    for (size_t i = 0; i < N; ++i) {
        scales[i * 3 + 0] = 0.1f * (i % 5 + 1);
        scales[i * 3 + 1] = 0.2f * (i % 5 + 1);
        scales[i * 3 + 2] = 0.3f * (i % 5 + 1);
    }

    // === LFS Workflow ===
    auto opacities_lfs = Tensor::from_vector(opacities, TensorShape{N}, Device::GPU);
    auto scales_lfs = Tensor::from_vector(scales, TensorShape{N, 3}, Device::GPU);
    auto sampled_idxs_lfs = Tensor::from_vector(sampled_indices, TensorShape{static_cast<size_t>(n_new)}, Device::GPU);

    // 1. Get sampled opacities and scales
    auto sampled_opacities_lfs = opacities_lfs.index_select(0, sampled_idxs_lfs);
    auto sampled_scales_lfs = scales_lfs.index_select(0, sampled_idxs_lfs);

    // 2. Count occurrences
    auto ratios_lfs = Tensor::zeros({N}, Device::GPU, DataType::Float32);
    ratios_lfs = ratios_lfs.index_add_(0, sampled_idxs_lfs, Tensor::ones({static_cast<size_t>(n_new)}, Device::GPU));
    ratios_lfs = ratios_lfs.index_select(0, sampled_idxs_lfs) + Tensor::ones_like(ratios_lfs.index_select(0, sampled_idxs_lfs));
    ratios_lfs = ratios_lfs.clamp(1.0f, static_cast<float>(n_max));

    // 3. Simulate relocation: divide by ratio (simplified - real code uses binomial kernel)
    auto new_opacities_lfs = sampled_opacities_lfs / ratios_lfs;
    auto new_scales_lfs = sampled_scales_lfs / ratios_lfs.unsqueeze(-1);

    // 4. Clamp and compute logit
    new_opacities_lfs = new_opacities_lfs.clamp(min_opacity, max_opacity);
    auto new_opacity_raw_lfs = (new_opacities_lfs / (Tensor::ones_like(new_opacities_lfs) - new_opacities_lfs)).log();

    // 5. Compute log scales
    auto new_scaling_raw_lfs = new_scales_lfs.log();

    auto final_opacity_lfs = new_opacity_raw_lfs.cpu().to_vector();
    auto final_scales_lfs = new_scaling_raw_lfs.cpu().to_vector();

    // === LibTorch Workflow ===
    std::vector<int64_t> sampled_indices_i64(sampled_indices.begin(), sampled_indices.end());
    auto opacities_torch = torch::from_blob(
                               const_cast<float*>(opacities.data()),
                               {static_cast<long>(N)},
                               torch::kFloat32)
                               .clone()
                               .cuda();
    auto scales_torch = torch::from_blob(
                            const_cast<float*>(scales.data()),
                            {static_cast<long>(N), 3},
                            torch::kFloat32)
                            .clone()
                            .cuda();
    auto sampled_idxs_torch = torch::from_blob(
                                  const_cast<int64_t*>(sampled_indices_i64.data()),
                                  {n_new},
                                  torch::kInt64)
                                  .clone()
                                  .cuda();

    // 1. Get sampled opacities and scales
    auto sampled_opacities_torch = opacities_torch.index_select(0, sampled_idxs_torch);
    auto sampled_scales_torch = scales_torch.index_select(0, sampled_idxs_torch);

    // 2. Count occurrences
    auto ratios_torch = torch::zeros({static_cast<long>(N)}, torch::kFloat32).cuda();
    ratios_torch = ratios_torch.index_add_(0, sampled_idxs_torch, torch::ones({n_new}, torch::kFloat32).cuda());
    ratios_torch = ratios_torch.index_select(0, sampled_idxs_torch) + torch::ones_like(ratios_torch.index_select(0, sampled_idxs_torch));
    ratios_torch = ratios_torch.clamp(1.0f, static_cast<float>(n_max));

    // 3. Simulate relocation
    auto new_opacities_torch = sampled_opacities_torch / ratios_torch;
    auto new_scales_torch = sampled_scales_torch / ratios_torch.unsqueeze(-1);

    // 4. Clamp and compute logit
    new_opacities_torch = new_opacities_torch.clamp(min_opacity, max_opacity);
    auto new_opacity_raw_torch = (new_opacities_torch / (torch::ones_like(new_opacities_torch) - new_opacities_torch)).log();

    // 5. Compute log scales
    auto new_scaling_raw_torch = new_scales_torch.log();

    auto final_opacity_torch = new_opacity_raw_torch.cpu();
    auto final_scales_torch = new_scaling_raw_torch.cpu();

    // === Compare Results ===
    // Compare opacities
    for (int i = 0; i < n_new; ++i) {
        float lfs_val = final_opacity_lfs[i];
        float torch_val = final_opacity_torch[i].item<float>();
        float diff = std::abs(lfs_val - torch_val);

        EXPECT_LT(diff, 1e-5f)
            << "Opacity mismatch at sample " << i
            << " (index=" << sampled_indices[i] << ")"
            << ": LFS=" << lfs_val << " vs LibTorch=" << torch_val;
    }

    // Compare scales
    for (int i = 0; i < n_new; ++i) {
        for (int j = 0; j < 3; ++j) {
            float lfs_val = final_scales_lfs[i * 3 + j];
            float torch_val = final_scales_torch[i][j].item<float>();
            float diff = std::abs(lfs_val - torch_val);

            EXPECT_LT(diff, 1e-5f)
                << "Scale mismatch at sample " << i << ", dim " << j
                << " (index=" << sampled_indices[i] << ")"
                << ": LFS=" << lfs_val << " vs LibTorch=" << torch_val;
        }
    }
}

// Test 11: ones_like with different dtypes (used in ratio counting)
TEST_F(AddNewGsTensorOpsTest, OnesLikeDifferentDtypes_LibTorchComparison) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    // Float32
    {
        auto tensor_lfs = Tensor::from_vector(data, TensorShape{5}, Device::GPU);
        auto ones_lfs = Tensor::ones_like(tensor_lfs);
        auto result_lfs = ones_lfs.cpu().to_vector();

        auto tensor_torch = torch::from_blob(
                                const_cast<float*>(data.data()),
                                {5},
                                torch::kFloat32)
                                .clone()
                                .cuda();
        auto ones_torch = torch::ones_like(tensor_torch);
        auto result_torch = ones_torch.cpu();

        for (size_t i = 0; i < 5; ++i) {
            float lfs_val = result_lfs[i];
            float torch_val = result_torch[i].item<float>();
            EXPECT_NEAR(lfs_val, torch_val, 1e-6f);
            EXPECT_NEAR(lfs_val, 1.0f, 1e-6f);
        }
    }

    // Int32
    {
        std::vector<int32_t> int_data = {1, 2, 3, 4, 5};
        auto tensor_lfs = Tensor::from_vector(int_data, TensorShape{5}, Device::GPU);
        auto ones_lfs = Tensor::ones_like(tensor_lfs);
        auto result_lfs = ones_lfs.cpu().to_vector_int();

        auto tensor_torch = torch::from_blob(
                                const_cast<int32_t*>(int_data.data()),
                                {5},
                                torch::kInt32)
                                .clone()
                                .cuda();
        auto ones_torch = torch::ones_like(tensor_torch);
        auto result_torch = ones_torch.cpu();

        for (size_t i = 0; i < 5; ++i) {
            int32_t lfs_val = result_lfs[i];
            int32_t torch_val = result_torch[i].item<int32_t>();
            EXPECT_EQ(lfs_val, torch_val);
            EXPECT_EQ(lfs_val, 1);
        }
    }
}

// Test 12: Contiguous operation (ensures tensor is contiguous before CUDA kernels)
TEST_F(AddNewGsTensorOpsTest, Contiguous_LibTorchComparison) {
    std::vector<float> data(20);
    for (size_t i = 0; i < 20; ++i) {
        data[i] = static_cast<float>(i);
    }

    // Create tensor and make it non-contiguous via transpose
    auto tensor_lfs = Tensor::from_vector(data, TensorShape{4, 5}, Device::GPU);
    auto transposed_lfs = tensor_lfs.transpose(0, 1); // [5, 4] - non-contiguous
    auto contiguous_lfs = transposed_lfs.contiguous();

    EXPECT_TRUE(contiguous_lfs.is_contiguous());

    auto result_lfs = contiguous_lfs.cpu().to_vector();

    // LibTorch
    auto tensor_torch = torch::from_blob(
                            const_cast<float*>(data.data()),
                            {4, 5},
                            torch::kFloat32)
                            .clone()
                            .cuda();
    auto transposed_torch = tensor_torch.transpose(0, 1);
    auto contiguous_torch = transposed_torch.contiguous();

    auto result_torch = contiguous_torch.cpu();

    // Compare
    for (size_t i = 0; i < 20; ++i) {
        float lfs_val = result_lfs[i];
        float torch_val = result_torch.view({-1})[i].item<float>();
        EXPECT_NEAR(lfs_val, torch_val, 1e-6f);
    }
}
