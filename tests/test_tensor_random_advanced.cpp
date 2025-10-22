/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <gtest/gtest.h>
#include <map>
#include <set>
#include <torch/torch.h>

using namespace lfs::core;

// ============= Helper Functions =============

namespace {

    void compare_int_tensors(const Tensor& custom, const torch::Tensor& reference,
                             const std::string& msg = "") {
        auto ref_cpu = reference.to(torch::kCPU).contiguous().flatten();
        auto custom_cpu = custom.cpu();

        ASSERT_EQ(custom_cpu.ndim(), reference.dim()) << msg << ": Rank mismatch";

        for (size_t i = 0; i < custom_cpu.ndim(); ++i) {
            ASSERT_EQ(custom_cpu.size(i), static_cast<size_t>(reference.size(i)))
                << msg << ": Shape mismatch at dim " << i;
        }

        ASSERT_EQ(custom_cpu.numel(), static_cast<size_t>(ref_cpu.numel()))
            << msg << ": Element count mismatch";

        auto custom_vec = custom_cpu.to_vector_int();
        auto ref_accessor = ref_cpu.accessor<int, 1>();

        for (size_t i = 0; i < custom_vec.size(); ++i) {
            EXPECT_EQ(custom_vec[i], ref_accessor[i])
                << msg << ": Mismatch at index " << i
                << " (custom=" << custom_vec[i] << ", ref=" << ref_accessor[i] << ")";
        }
    }

    void compare_tensors(const Tensor& custom, const torch::Tensor& reference,
                         float rtol = 1e-4f, float atol = 1e-5f, const std::string& msg = "") {
        if (reference.dtype() == torch::kInt32 || reference.dtype() == torch::kInt64) {
            compare_int_tensors(custom, reference, msg);
            return;
        }

        auto ref_cpu = reference.to(torch::kCPU).contiguous().flatten();
        auto custom_cpu = custom.cpu();

        ASSERT_EQ(custom_cpu.ndim(), reference.dim()) << msg << ": Rank mismatch";

        for (size_t i = 0; i < custom_cpu.ndim(); ++i) {
            ASSERT_EQ(custom_cpu.size(i), static_cast<size_t>(reference.size(i)))
                << msg << ": Shape mismatch at dim " << i;
        }

        ASSERT_EQ(custom_cpu.numel(), static_cast<size_t>(ref_cpu.numel()))
            << msg << ": Element count mismatch";

        auto custom_vec = custom_cpu.to_vector();
        auto ref_accessor = ref_cpu.accessor<float, 1>();

        for (size_t i = 0; i < custom_vec.size(); ++i) {
            float ref_val = ref_accessor[i];
            float custom_val = custom_vec[i];

            if (std::isnan(ref_val)) {
                EXPECT_TRUE(std::isnan(custom_val)) << msg << ": Expected NaN at index " << i;
            } else if (std::isinf(ref_val)) {
                EXPECT_TRUE(std::isinf(custom_val)) << msg << ": Expected Inf at index " << i;
            } else {
                float diff = std::abs(custom_val - ref_val);
                float threshold = atol + rtol * std::abs(ref_val);
                EXPECT_LE(diff, threshold)
                    << msg << ": Mismatch at index " << i
                    << " (custom=" << custom_val << ", ref=" << ref_val << ")";
            }
        }
    }

} // anonymous namespace

class TensorRandomAdvancedTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA is not available for testing";
        torch::manual_seed(42);
        Tensor::manual_seed(42);
    }
};

// ============= Multinomial Tests =============

TEST_F(TensorRandomAdvancedTest, MultinomialBasicCPU) {
    Tensor::manual_seed(123);
    torch::manual_seed(123);

    std::vector<float> weights_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    auto custom_weights = Tensor::from_vector(weights_data, {5}, Device::CPU);
    auto torch_weights = torch::tensor(weights_data, torch::TensorOptions().device(torch::kCPU));

    auto custom_samples = Tensor::multinomial(custom_weights, 10, true);
    auto torch_samples = torch::multinomial(torch_weights, 10, true);

    ASSERT_TRUE(custom_samples.is_valid());
    EXPECT_EQ(custom_samples.dtype(), DataType::Int32);
    EXPECT_EQ(custom_samples.shape(), TensorShape({10}));
    EXPECT_EQ(custom_samples.device(), Device::CPU);

    // Check all samples are in valid range [0, 5)
    auto custom_values = custom_samples.to_vector_int();
    for (int v : custom_values) {
        EXPECT_GE(v, 0);
        EXPECT_LT(v, 5);
    }

    // PyTorch samples should also be valid
    auto torch_values = torch_samples.to(torch::kCPU);
    for (int64_t i = 0; i < torch_samples.numel(); ++i) {
        int64_t v = torch_values[i].item<int64_t>();
        EXPECT_GE(v, 0);
        EXPECT_LT(v, 5);
    }
}

TEST_F(TensorRandomAdvancedTest, MultinomialBasicCUDA) {
    Tensor::manual_seed(456);
    torch::manual_seed(456);

    std::vector<float> weights_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    auto custom_weights = Tensor::from_vector(weights_data, {5}, Device::CUDA);
    auto torch_weights = torch::tensor(weights_data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_samples = Tensor::multinomial(custom_weights, 10, true);
    auto torch_samples = torch::multinomial(torch_weights, 10, true);

    ASSERT_TRUE(custom_samples.is_valid());
    EXPECT_EQ(custom_samples.dtype(), DataType::Int32);
    EXPECT_EQ(custom_samples.shape(), TensorShape({10}));
    EXPECT_EQ(custom_samples.device(), Device::CUDA);

    auto custom_values = custom_samples.to_vector_int();
    for (int v : custom_values) {
        EXPECT_GE(v, 0);
        EXPECT_LT(v, 5);
    }
}

TEST_F(TensorRandomAdvancedTest, MultinomialWithReplacement) {
    Tensor::manual_seed(789);

    std::vector<float> weights_data = {1.0f, 1.0f, 1.0f};
    auto custom_weights = Tensor::from_vector(weights_data, {3}, Device::CPU);

    auto custom_samples = Tensor::multinomial(custom_weights, 100, true);

    ASSERT_TRUE(custom_samples.is_valid());
    EXPECT_EQ(custom_samples.numel(), 100);

    // With replacement, count frequency of each index
    auto values = custom_samples.to_vector_int();
    std::map<int, int> freq;
    for (int v : values) {
        EXPECT_GE(v, 0);
        EXPECT_LT(v, 3);
        freq[v]++;
    }

    // All 3 indices should appear (with high probability)
    EXPECT_EQ(freq.size(), 3);

    // Each should appear roughly 33 times (allow variance)
    for (const auto& [idx, count] : freq) {
        EXPECT_GT(count, 10); // At least 10% of samples
        EXPECT_LT(count, 60); // At most 60% of samples
    }
}

TEST_F(TensorRandomAdvancedTest, MultinomialWithoutReplacement) {
    Tensor::manual_seed(999);
    torch::manual_seed(999);

    std::vector<float> weights_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    auto custom_weights = Tensor::from_vector(weights_data, {5}, Device::CPU);
    auto torch_weights = torch::tensor(weights_data, torch::TensorOptions().device(torch::kCPU));

    auto custom_samples = Tensor::multinomial(custom_weights, 5, false);
    auto torch_samples = torch::multinomial(torch_weights, 5, false);

    ASSERT_TRUE(custom_samples.is_valid());
    EXPECT_EQ(custom_samples.numel(), 5);

    // Without replacement, each index should appear exactly once
    auto custom_values = custom_samples.to_vector_int();
    std::set<int> custom_unique(custom_values.begin(), custom_values.end());
    EXPECT_EQ(custom_unique.size(), 5);

    // All values should be in [0, 5)
    for (int v : custom_values) {
        EXPECT_GE(v, 0);
        EXPECT_LT(v, 5);
    }

    // PyTorch should also have unique values
    auto torch_cpu = torch_samples.to(torch::kCPU);
    std::set<int64_t> torch_unique;
    for (int64_t i = 0; i < torch_samples.numel(); ++i) {
        int64_t v = torch_cpu[i].item<int64_t>();
        torch_unique.insert(v);
        EXPECT_GE(v, 0);
        EXPECT_LT(v, 5);
    }
    EXPECT_EQ(torch_unique.size(), 5);
}

TEST_F(TensorRandomAdvancedTest, MultinomialPartialSamplingWithoutReplacement) {
    Tensor::manual_seed(111);

    std::vector<float> weights_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    auto custom_weights = Tensor::from_vector(weights_data, {7}, Device::CPU);

    auto custom_samples = Tensor::multinomial(custom_weights, 3, false);

    ASSERT_TRUE(custom_samples.is_valid());
    EXPECT_EQ(custom_samples.numel(), 3);

    // Should get 3 unique values from [0, 7)
    auto values = custom_samples.to_vector_int();
    std::set<int> unique(values.begin(), values.end());
    EXPECT_EQ(unique.size(), 3);

    for (int v : values) {
        EXPECT_GE(v, 0);
        EXPECT_LT(v, 7);
    }
}

TEST_F(TensorRandomAdvancedTest, MultinomialBiasedWeights) {
    Tensor::manual_seed(222);

    // Heavily biased weights: index 0 has weight 1000, others have weight 1
    std::vector<float> weights_data = {1000.0f, 1.0f, 1.0f, 1.0f};
    auto custom_weights = Tensor::from_vector(weights_data, {4}, Device::CPU);

    auto custom_samples = Tensor::multinomial(custom_weights, 1000, true);

    auto values = custom_samples.to_vector_int();
    std::map<int, int> freq;
    for (int v : values) {
        EXPECT_GE(v, 0);
        EXPECT_LT(v, 4);
        freq[v]++;
    }

    // Index 0 should dominate (expect at least 95% of samples)
    EXPECT_GT(freq[0], 950);

    // Other indices should be rare
    int others = freq[1] + freq[2] + freq[3];
    EXPECT_LT(others, 50);
}

TEST_F(TensorRandomAdvancedTest, MultinomialZeroWeight) {
    Tensor::manual_seed(333);

    // Some weights are zero
    std::vector<float> weights_data = {1.0f, 0.0f, 1.0f, 0.0f};
    auto custom_weights = Tensor::from_vector(weights_data, {4}, Device::CPU);

    auto custom_samples = Tensor::multinomial(custom_weights, 100, true);

    auto values = custom_samples.to_vector_int();

    // Should never sample indices with zero weight
    for (int v : values) {
        EXPECT_TRUE(v == 0 || v == 2) << "Got invalid index: " << v;
        EXPECT_FALSE(v == 1 || v == 3) << "Sampled zero-weight index: " << v;
    }

    // Both indices should appear
    std::set<int> unique(values.begin(), values.end());
    EXPECT_EQ(unique.size(), 2);
}

TEST_F(TensorRandomAdvancedTest, MultinomialSingleSample) {
    Tensor::manual_seed(444);
    torch::manual_seed(444);

    std::vector<float> weights_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                       1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    auto custom_weights = Tensor::from_vector(weights_data, {10}, Device::CPU);
    auto torch_weights = torch::tensor(weights_data, torch::TensorOptions().device(torch::kCPU));

    auto custom_sample = Tensor::multinomial(custom_weights, 1, true);
    auto torch_sample = torch::multinomial(torch_weights, 1, true);

    ASSERT_TRUE(custom_sample.is_valid());
    EXPECT_EQ(custom_sample.numel(), 1);

    int custom_value = custom_sample.to_vector_int()[0];
    EXPECT_GE(custom_value, 0);
    EXPECT_LT(custom_value, 10);

    int64_t torch_value = torch_sample[0].item<int64_t>();
    EXPECT_GE(torch_value, 0);
    EXPECT_LT(torch_value, 10);
}

TEST_F(TensorRandomAdvancedTest, MultinomialInvalidInputs) {
    // Non-1D weights
    auto weights_2d = Tensor::ones({3, 3}, Device::CPU);
    auto result1 = Tensor::multinomial(weights_2d, 5, true);
    EXPECT_FALSE(result1.is_valid());

    // Zero total weight
    auto zero_weights = Tensor::zeros({5}, Device::CPU);
    auto result2 = Tensor::multinomial(zero_weights, 5, true);
    EXPECT_FALSE(result2.is_valid());

    // Negative weights - should handle gracefully
    auto negative_weights = Tensor::full({5}, -1.0f, Device::CPU);
    auto result3 = Tensor::multinomial(negative_weights, 5, true);
    // Either invalid or handled by clamping to zero
    // Just ensure it doesn't crash
}

TEST_F(TensorRandomAdvancedTest, MultinomialTooManySamplesWithoutReplacement) {
    Tensor::manual_seed(555);

    std::vector<float> weights_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    auto custom_weights = Tensor::from_vector(weights_data, {5}, Device::CPU);

    // Try to sample more than available without replacement
    auto custom_samples = Tensor::multinomial(custom_weights, 10, false);

    // Implementation should either fail or cap at num_weights
    if (custom_samples.is_valid()) {
        auto values = custom_samples.to_vector_int();
        std::set<int> unique(values.begin(), values.end());
        EXPECT_LE(unique.size(), 5) << "Cannot have more unique samples than weights";
        EXPECT_LE(custom_samples.numel(), 5) << "Should cap at number of weights";
    }
}

// ============= Reproducibility Tests =============

TEST_F(TensorRandomAdvancedTest, MultinomialReproducibilityCPU) {
    std::vector<float> weights_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                       6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    auto weights = Tensor::from_vector(weights_data, {10}, Device::CPU);

    Tensor::manual_seed(12345);
    auto samples1 = Tensor::multinomial(weights, 20, true);

    Tensor::manual_seed(12345);
    auto samples2 = Tensor::multinomial(weights, 20, true);

    auto values1 = samples1.to_vector_int();
    auto values2 = samples2.to_vector_int();

    EXPECT_EQ(values1, values2) << "Same seed should produce same results";
}

TEST_F(TensorRandomAdvancedTest, MultinomialReproducibilityCUDA) {
    std::vector<float> weights_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                       6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    auto weights = Tensor::from_vector(weights_data, {10}, Device::CUDA);

    Tensor::manual_seed(12345);
    auto samples1 = Tensor::multinomial(weights, 20, true);

    Tensor::manual_seed(12345);
    auto samples2 = Tensor::multinomial(weights, 20, true);

    auto values1 = samples1.to_vector_int();
    auto values2 = samples2.to_vector_int();

    EXPECT_EQ(values1, values2) << "Same seed should produce same results";
}

TEST_F(TensorRandomAdvancedTest, MultinomialWithoutReplacementReproducibility) {
    std::vector<float> weights_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    auto weights = Tensor::from_vector(weights_data, {5}, Device::CPU);

    Tensor::manual_seed(67890);
    auto samples1 = Tensor::multinomial(weights, 3, false);

    Tensor::manual_seed(67890);
    auto samples2 = Tensor::multinomial(weights, 3, false);

    auto values1 = samples1.to_vector_int();
    auto values2 = samples2.to_vector_int();

    EXPECT_EQ(values1, values2) << "Same seed should produce same results";
}

// ============= Distribution Tests =============

TEST_F(TensorRandomAdvancedTest, MultinomialUniformDistribution) {
    Tensor::manual_seed(666);

    // Uniform weights
    std::vector<float> weights_data(10, 1.0f);
    auto weights = Tensor::from_vector(weights_data, {10}, Device::CUDA);

    auto samples = Tensor::multinomial(weights, 10000, true);

    auto values = samples.to_vector_int();
    std::map<int, int> freq;
    for (int v : values) {
        freq[v]++;
    }

    // Each index should appear roughly 1000 times
    for (const auto& [idx, count] : freq) {
        EXPECT_GT(count, 700);  // At least 70% of expected
        EXPECT_LT(count, 1300); // At most 130% of expected
    }
}

TEST_F(TensorRandomAdvancedTest, MultinomialSkewedDistribution) {
    Tensor::manual_seed(777);

    // Skewed weights: linearly increasing
    std::vector<float> weights_data;
    for (int i = 1; i <= 10; ++i) {
        weights_data.push_back(static_cast<float>(i));
    }
    auto weights = Tensor::from_vector(weights_data, {10}, Device::CUDA);

    auto samples = Tensor::multinomial(weights, 5000, true);

    auto values = samples.to_vector_int();
    std::map<int, int> freq;
    for (int v : values) {
        freq[v]++;
    }

    // Higher indices should appear more frequently
    // Index 9 (weight 10) should appear ~2x as often as index 4 (weight 5)
    if (freq.count(9) && freq.count(4)) {
        float ratio = static_cast<float>(freq[9]) / static_cast<float>(freq[4]);
        EXPECT_GT(ratio, 1.5f) << "Higher weight index should appear more often";
        EXPECT_LT(ratio, 2.5f) << "But within statistical variance";
    }
}

// ============= Performance Test =============

TEST_F(TensorRandomAdvancedTest, MultinomialLargeScaleCUDA) {
    Tensor::manual_seed(888);

    auto weights = Tensor::ones({1000}, Device::CUDA);
    auto samples = Tensor::multinomial(weights, 10000, true);

    ASSERT_TRUE(samples.is_valid());
    EXPECT_EQ(samples.numel(), 10000);
    EXPECT_EQ(samples.device(), Device::CUDA);

    // Verify all samples are in valid range
    auto values = samples.to_vector_int();
    for (int v : values) {
        EXPECT_GE(v, 0);
        EXPECT_LT(v, 1000);
    }

    // Check distribution is roughly uniform
    std::map<int, int> freq;
    for (int v : values) {
        freq[v]++;
    }

    // With 10000 samples over 1000 bins, expect ~10 per bin on average
    // Most bins should have counts between 1 and 30
    for (const auto& [idx, count] : freq) {
        EXPECT_GT(count, 0);
        EXPECT_LT(count, 50); // Allow variance
    }
}

TEST_F(TensorRandomAdvancedTest, MultinomialLargeWeightsArray) {
    Tensor::manual_seed(999);

    // 10,000 uniform weights
    auto weights = Tensor::ones({10000}, Device::CUDA);
    auto samples = Tensor::multinomial(weights, 100, true);

    ASSERT_TRUE(samples.is_valid());
    EXPECT_EQ(samples.numel(), 100);

    auto values = samples.to_vector_int();
    for (int v : values) {
        EXPECT_GE(v, 0);
        EXPECT_LT(v, 10000);
    }
}

// ============= Integration with Other Ops =============

TEST_F(TensorRandomAdvancedTest, MultinomialAsIndices) {
    Tensor::manual_seed(1111);

    std::vector<float> weights_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    auto weights = Tensor::from_vector(weights_data, {5}, Device::CPU);
    auto indices = Tensor::multinomial(weights, 10, true);

    // Use multinomial result as indices for index_select
    auto data = Tensor::arange(0.0f, 5.0f);
    auto selected = data.index_select(0, indices);

    ASSERT_TRUE(selected.is_valid());
    EXPECT_EQ(selected.shape(), TensorShape({10}));

    // All selected values should be in [0, 5)
    auto values = selected.to_vector();
    for (float v : values) {
        EXPECT_GE(v, 0.0f);
        EXPECT_LT(v, 5.0f);
        // Should be integers
        EXPECT_FLOAT_EQ(v, std::floor(v));
    }
}

TEST_F(TensorRandomAdvancedTest, MultinomialGatherPattern) {
    Tensor::manual_seed(2222);

    // Create a simple embedding-like tensor
    auto embeddings = Tensor::arange(0.0f, 20.0f).reshape({5, 4});

    std::vector<float> weights_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto weights = Tensor::from_vector(weights_data, {5}, Device::CUDA);
    auto indices = Tensor::multinomial(weights, 3, false).cuda();

    // Gather rows based on multinomial indices
    auto gathered = embeddings.cuda().index_select(0, indices);

    EXPECT_EQ(gathered.shape(), TensorShape({3, 4}));
    EXPECT_TRUE(gathered.is_valid());
}

TEST_F(TensorRandomAdvancedTest, MultinomialBatchProcessing) {
    Tensor::manual_seed(3333);

    std::vector<float> weights_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto weights = Tensor::from_vector(weights_data, {4}, Device::CUDA);

    std::vector<Tensor> batch_samples;
    for (int i = 0; i < 5; ++i) {
        // Use emplace_back to construct directly in the vector
        batch_samples.emplace_back(Tensor::multinomial(weights, 10, true));
        EXPECT_TRUE(batch_samples.back().is_valid());
        EXPECT_EQ(batch_samples.back().numel(), 10);
    }

    // All batches should be valid and have correct shape
    EXPECT_EQ(batch_samples.size(), 5);

    // Each batch should be different (unless by extreme coincidence)
    bool all_different = false;
    for (size_t i = 0; i < batch_samples.size() - 1; ++i) {
        auto vals1 = batch_samples[i].to_vector_int();
        auto vals2 = batch_samples[i + 1].to_vector_int();
        if (vals1 != vals2) {
            all_different = true;
            break;
        }
    }
    EXPECT_TRUE(all_different) << "Successive calls should produce different results";
}
// ============= Edge Cases =============

TEST_F(TensorRandomAdvancedTest, MultinomialSingleWeight) {
    Tensor::manual_seed(4444);

    std::vector<float> weights_data = {1.0f};
    auto weights = Tensor::from_vector(weights_data, {1}, Device::CPU);

    auto samples = Tensor::multinomial(weights, 5, true);

    ASSERT_TRUE(samples.is_valid());
    EXPECT_EQ(samples.numel(), 5);

    // All samples must be index 0
    auto values = samples.to_vector_int();
    for (int v : values) {
        EXPECT_EQ(v, 0);
    }
}

TEST_F(TensorRandomAdvancedTest, MultinomialVerySmallWeights) {
    Tensor::manual_seed(5555);

    std::vector<float> weights_data = {1e-10f, 1e-10f, 1e-10f, 1e-10f};
    auto weights = Tensor::from_vector(weights_data, {4}, Device::CPU);

    auto samples = Tensor::multinomial(weights, 20, true);

    ASSERT_TRUE(samples.is_valid());
    EXPECT_EQ(samples.numel(), 20);

    // Should still work with very small weights
    auto values = samples.to_vector_int();
    for (int v : values) {
        EXPECT_GE(v, 0);
        EXPECT_LT(v, 4);
    }
}

TEST_F(TensorRandomAdvancedTest, MultinomialVeryLargeWeights) {
    Tensor::manual_seed(6666);

    std::vector<float> weights_data = {1e6f, 1e6f, 1e6f, 1e6f};
    auto weights = Tensor::from_vector(weights_data, {4}, Device::CPU);

    auto samples = Tensor::multinomial(weights, 20, true);

    ASSERT_TRUE(samples.is_valid());
    EXPECT_EQ(samples.numel(), 20);

    auto values = samples.to_vector_int();
    for (int v : values) {
        EXPECT_GE(v, 0);
        EXPECT_LT(v, 4);
    }
}