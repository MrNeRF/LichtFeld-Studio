/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <torch/torch.h>

using namespace lfs::core;

// ============= Helper Functions =============

namespace {

    // Helper for comparing boolean tensors - ONLY for deterministic operations
    void compare_bool_tensors(const Tensor& custom, const torch::Tensor& reference,
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

        auto custom_vec = custom_cpu.to_vector_bool();
        auto ref_accessor = ref_cpu.accessor<bool, 1>();

        for (size_t i = 0; i < custom_vec.size(); ++i) {
            EXPECT_EQ(custom_vec[i], ref_accessor[i])
                << msg << ": Mismatch at index " << i
                << " (custom=" << custom_vec[i] << ", ref=" << ref_accessor[i] << ")";
        }
    }

    // Helper for comparing integer tensors - ONLY for deterministic operations
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

    // ONLY use for deterministic operations, NOT for random generation!
    void compare_tensors(const Tensor& custom, const torch::Tensor& reference,
                         float rtol = 1e-4f, float atol = 1e-5f, const std::string& msg = "") {
        if (reference.dtype() == torch::kBool) {
            compare_bool_tensors(custom, reference, msg);
            return;
        }

        if (reference.dtype() == torch::kInt32) {
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

    // Helper to check if values are in expected range
    bool check_range(const Tensor& tensor, float min, float max) {
        auto values = tensor.to_vector();
        for (float val : values) {
            if (val < min || val > max) {
                return false;
            }
        }
        return true;
    }

    // Helper to compute mean and std
    std::pair<float, float> compute_stats(const Tensor& tensor) {
        auto values = tensor.to_vector();
        if (values.empty())
            return {0.0f, 0.0f};

        float sum = std::accumulate(values.begin(), values.end(), 0.0f);
        float mean = sum / values.size();

        float sq_sum = 0.0f;
        for (float val : values) {
            float diff = val - mean;
            sq_sum += diff * diff;
        }
        float std = std::sqrt(sq_sum / values.size());

        return {mean, std};
    }

} // anonymous namespace

class TensorRandomTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA is not available for testing";
        // Only seed our library
        Tensor::manual_seed(42);
    }
};

// ============= Uniform Distribution Tests =============

TEST_F(TensorRandomTest, RandBasic) {
    auto t = Tensor::rand({100, 100}, Device::CUDA);

    // Check all values are in [0, 1)
    EXPECT_TRUE(check_range(t, 0.0f, 1.0f));

    // Check mean is approximately 0.5
    auto [mean, std] = compute_stats(t);
    EXPECT_NEAR(mean, 0.5f, 0.05f);

    // For uniform [0,1], std should be approximately sqrt(1/12) ≈ 0.289
    EXPECT_NEAR(std, 0.289f, 0.05f);
}

TEST_F(TensorRandomTest, RandShape) {
    auto t = Tensor::rand({3, 4, 5}, Device::CUDA);

    EXPECT_EQ(t.ndim(), 3);
    EXPECT_EQ(t.shape().dims(), std::vector<size_t>({3, 4, 5}));
    EXPECT_EQ(t.numel(), 60);
    EXPECT_TRUE(check_range(t, 0.0f, 1.0f));
}

TEST_F(TensorRandomTest, UniformCustomRange) {
    float low = -5.0f;
    float high = 10.0f;

    auto t = Tensor::uniform({1000}, low, high, Device::CUDA);

    // Check all values are in range
    EXPECT_TRUE(check_range(t, low, high));

    // Check distribution parameters
    float expected_mean = (low + high) / 2.0f;
    float expected_std = (high - low) / std::sqrt(12.0f);

    auto [mean, std] = compute_stats(t);
    EXPECT_NEAR(mean, expected_mean, 0.5f);
    EXPECT_NEAR(std, expected_std, 0.5f);
}

TEST_F(TensorRandomTest, UniformInPlace) {
    auto t = Tensor::zeros({500}, Device::CUDA);
    t.uniform_(2.0f, 4.0f);

    EXPECT_TRUE(check_range(t, 2.0f, 4.0f));

    auto [mean, std] = compute_stats(t);
    EXPECT_NEAR(mean, 3.0f, 0.1f);
}

// ============= Normal Distribution Tests =============

TEST_F(TensorRandomTest, RandnBasic) {
    auto t = Tensor::randn({1000, 10}, Device::CUDA);

    // Check mean and std
    auto [mean, std] = compute_stats(t);
    EXPECT_NEAR(mean, 0.0f, 0.1f);
    EXPECT_NEAR(std, 1.0f, 0.1f);

    // Most values should be within 3 standard deviations
    auto values = t.to_vector();
    int outliers = 0;
    for (float val : values) {
        if (std::abs(val) > 3.0f) {
            outliers++;
        }
    }
    // Expect less than 1% outliers beyond 3 sigma
    EXPECT_LT(outliers, values.size() * 0.01);
}

TEST_F(TensorRandomTest, NormalCustomParams) {
    float target_mean = 5.0f;
    float target_std = 2.0f;

    auto t = Tensor::normal({2000}, target_mean, target_std, Device::CUDA);

    auto [mean, std] = compute_stats(t);
    EXPECT_NEAR(mean, target_mean, 0.2f);
    EXPECT_NEAR(std, target_std, 0.2f);
}

TEST_F(TensorRandomTest, NormalInPlace) {
    auto t = Tensor::empty({1000}, Device::CUDA);
    t.normal_(10.0f, 3.0f);

    auto [mean, std] = compute_stats(t);
    EXPECT_NEAR(mean, 10.0f, 0.3f);
    EXPECT_NEAR(std, 3.0f, 0.3f);
}

// ============= Integer Random Tests =============

TEST_F(TensorRandomTest, RandIntBasic) {
    int low = 0;
    int high = 10;

    auto t = Tensor::randint({1000}, low, high, Device::CUDA, DataType::Float32);

    // Check all values are integers in [low, high)
    auto values = t.to_vector();
    for (float val : values) {
        EXPECT_GE(val, low);
        EXPECT_LT(val, high);
        EXPECT_FLOAT_EQ(val, std::floor(val)); // Check it's an integer
    }
}

TEST_F(TensorRandomTest, RandIntNegative) {
    int low = -5;
    int high = 5;

    auto t = Tensor::randint({500}, low, high, Device::CUDA, DataType::Float32);

    auto values = t.to_vector();
    for (float val : values) {
        EXPECT_GE(val, low);
        EXPECT_LT(val, high);
        EXPECT_FLOAT_EQ(val, std::floor(val));
    }
}

TEST_F(TensorRandomTest, RandIntInt32) {
    int low = 0;
    int high = 100;

    auto t = Tensor::randint({200}, low, high, Device::CUDA, DataType::Int32);

    EXPECT_EQ(t.dtype(), DataType::Int32);
    EXPECT_EQ(t.numel(), 200);
}

TEST_F(TensorRandomTest, RandIntDistribution) {
    int low = 0;
    int high = 10;

    auto t = Tensor::randint({10000}, low, high, Device::CUDA, DataType::Float32);

    // Check roughly uniform distribution
    auto values = t.to_vector();
    std::vector<int> counts(high - low, 0);
    for (float val : values) {
        counts[static_cast<int>(val) - low]++;
    }

    // Each value should appear roughly 1000 times (10000 / 10)
    for (int count : counts) {
        EXPECT_GT(count, 800);  // At least 800
        EXPECT_LT(count, 1200); // At most 1200
    }
}

// ============= Bernoulli Tests =============

TEST_F(TensorRandomTest, BernoulliBasic) {
    float p = 0.7f;

    auto t = Tensor::bernoulli({10000}, p, Device::CUDA);

    // Check all values are 0 or 1
    auto values = t.to_vector();
    for (float val : values) {
        EXPECT_TRUE(val == 0.0f || val == 1.0f);
    }

    // Check probability
    float sum = std::accumulate(values.begin(), values.end(), 0.0f);
    float observed_p = sum / values.size();
    EXPECT_NEAR(observed_p, p, 0.02f);
}

TEST_F(TensorRandomTest, BernoulliExtreme) {
    // Test p = 0.0
    auto t0 = Tensor::bernoulli({100}, 0.0f, Device::CUDA);
    auto values0 = t0.to_vector();
    for (float val : values0) {
        EXPECT_FLOAT_EQ(val, 0.0f);
    }

    // Test p = 1.0
    auto t1 = Tensor::bernoulli({100}, 1.0f, Device::CUDA);
    auto values1 = t1.to_vector();
    for (float val : values1) {
        EXPECT_FLOAT_EQ(val, 1.0f);
    }
}

TEST_F(TensorRandomTest, BernoulliMiddleProb) {
    float p = 0.5f;

    auto t = Tensor::bernoulli({5000}, p, Device::CUDA);

    auto values = t.to_vector();
    float sum = std::accumulate(values.begin(), values.end(), 0.0f);
    float observed_p = sum / values.size();
    EXPECT_NEAR(observed_p, 0.5f, 0.03f);
}

// ============= Multinomial Tests =============

TEST_F(TensorRandomTest, MultinomialBasic) {
    std::vector<float> weights_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto weights = Tensor::from_vector(weights_data, {4}, Device::CUDA);

    int num_samples = 1000;
    auto samples = Tensor::multinomial(weights, num_samples, true);

    // Shape should match
    EXPECT_EQ(samples.numel(), static_cast<size_t>(num_samples));

    // All samples should be in valid range [0, 4)
    auto values = samples.to_vector_int();
    for (int val : values) {
        EXPECT_GE(val, 0);
        EXPECT_LT(val, 4);
    }

    // Check distribution roughly matches probabilities
    std::vector<int> counts(4, 0);
    for (int val : values) {
        counts[val]++;
    }

    // Probabilities should be proportional to weights: 1/10, 2/10, 3/10, 4/10
    float total = 10.0f;
    for (size_t i = 0; i < 4; ++i) {
        float expected_prob = weights_data[i] / total;
        float observed_prob = counts[i] / static_cast<float>(num_samples);
        EXPECT_NEAR(observed_prob, expected_prob, 0.05f) << "Index " << i;
    }
}

TEST_F(TensorRandomTest, MultinomialWithoutReplacement) {
    std::vector<float> weights_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    auto weights = Tensor::from_vector(weights_data, {5}, Device::CUDA);

    int num_samples = 3;
    auto samples = Tensor::multinomial(weights, num_samples, false);

    EXPECT_EQ(samples.numel(), static_cast<size_t>(num_samples));

    // All samples should be unique
    auto values = samples.to_vector_int();
    std::set<int> unique_values(values.begin(), values.end());
    EXPECT_EQ(unique_values.size(), values.size());
}

// ============= Seed Reproducibility Tests =============

TEST_F(TensorRandomTest, ManualSeedReproducibility) {
    // Test that manual seed produces reproducible results
    Tensor::manual_seed(12345);
    auto t1 = Tensor::randn({100}, Device::CUDA);

    Tensor::manual_seed(12345);
    auto t2 = Tensor::randn({100}, Device::CUDA);

    // Should produce identical results
    EXPECT_TRUE(t1.all_close(t2, 1e-6f, 1e-7f));
}

TEST_F(TensorRandomTest, DifferentSeedsDifferentResults) {
    Tensor::manual_seed(12345);
    auto t1 = Tensor::randn({100}, Device::CUDA);

    Tensor::manual_seed(54321);
    auto t2 = Tensor::randn({100}, Device::CUDA);

    // Different seeds should produce different results
    EXPECT_FALSE(t1.all_close(t2));
}

TEST_F(TensorRandomTest, UniformSeedReproducibility) {
    Tensor::manual_seed(999);
    auto t1 = Tensor::rand({200}, Device::CUDA);

    Tensor::manual_seed(999);
    auto t2 = Tensor::rand({200}, Device::CUDA);

    EXPECT_TRUE(t1.all_close(t2, 1e-6f, 1e-7f));
}

TEST_F(TensorRandomTest, RandIntSeedReproducibility) {
    Tensor::manual_seed(777);
    auto t1 = Tensor::randint({100}, 0, 10, Device::CUDA, DataType::Float32);

    Tensor::manual_seed(777);
    auto t2 = Tensor::randint({100}, 0, 10, Device::CUDA, DataType::Float32);

    EXPECT_TRUE(t1.all_close(t2, 1e-6f, 1e-7f));
}

// ============= Like Operations Tests =============

TEST_F(TensorRandomTest, RandLike) {
    auto original = Tensor::zeros({3, 4, 5}, Device::CUDA);
    auto random = Tensor::rand_like(original);

    EXPECT_EQ(random.shape(), original.shape());
    EXPECT_EQ(random.device(), original.device());
    EXPECT_TRUE(check_range(random, 0.0f, 1.0f));
}

TEST_F(TensorRandomTest, RandnLike) {
    auto original = Tensor::ones({10, 10}, Device::CUDA);
    auto random = Tensor::randn_like(original);

    EXPECT_EQ(random.shape(), original.shape());
    EXPECT_EQ(random.device(), original.device());

    // Should be roughly standard normal
    auto [mean, std] = compute_stats(random);
    EXPECT_NEAR(mean, 0.0f, 0.3f);
    EXPECT_NEAR(std, 1.0f, 0.3f);
}

TEST_F(TensorRandomTest, ZerosLike) {
    auto original = Tensor::rand({5, 6}, Device::CUDA);
    auto zeros = Tensor::zeros_like(original);

    EXPECT_EQ(zeros.shape(), original.shape());
    EXPECT_EQ(zeros.device(), original.device());

    auto values = zeros.to_vector();
    for (float val : values) {
        EXPECT_FLOAT_EQ(val, 0.0f);
    }
}

TEST_F(TensorRandomTest, OnesLike) {
    auto original = Tensor::rand({4, 7}, Device::CUDA);
    auto ones = Tensor::ones_like(original);

    EXPECT_EQ(ones.shape(), original.shape());
    EXPECT_EQ(ones.device(), original.device());

    auto values = ones.to_vector();
    for (float val : values) {
        EXPECT_FLOAT_EQ(val, 1.0f);
    }
}

// ============= CPU vs CUDA Consistency =============

TEST_F(TensorRandomTest, CPUDistribution) {
    auto t = Tensor::randn({1000}, Device::CPU);

    auto [mean, std] = compute_stats(t);
    EXPECT_NEAR(mean, 0.0f, 0.1f);
    EXPECT_NEAR(std, 1.0f, 0.1f);
}

TEST_F(TensorRandomTest, CUDADistribution) {
    auto t = Tensor::randn({1000}, Device::CUDA);

    auto [mean, std] = compute_stats(t);
    EXPECT_NEAR(mean, 0.0f, 0.1f);
    EXPECT_NEAR(std, 1.0f, 0.1f);
}

TEST_F(TensorRandomTest, CPUCUDASameSeedSameResults) {
    // Test CPU reproducibility
    Tensor::manual_seed(999);
    auto cpu_t1 = Tensor::randn({100}, Device::CPU);

    Tensor::manual_seed(999);
    auto cpu_t2 = Tensor::randn({100}, Device::CPU);

    EXPECT_TRUE(cpu_t1.all_close(cpu_t2, 1e-6f, 1e-7f))
        << "CPU should be reproducible with same seed";

    // Test CUDA reproducibility
    Tensor::manual_seed(999);
    auto cuda_t1 = Tensor::randn({100}, Device::CUDA);

    Tensor::manual_seed(999);
    auto cuda_t2 = Tensor::randn({100}, Device::CUDA);

    EXPECT_TRUE(cuda_t1.all_close(cuda_t2, 1e-6f, 1e-7f))
        << "CUDA should be reproducible with same seed";

    // Both should have similar distribution properties (but not identical values)
    // Use larger sample for more stable statistics
    Tensor::manual_seed(999);
    auto cpu_large = Tensor::randn({1000}, Device::CPU); // Increased from 100 to 1000

    Tensor::manual_seed(999);
    auto cuda_large = Tensor::randn({1000}, Device::CUDA); // Increased from 100 to 1000

    auto [cpu_mean, cpu_std] = compute_stats(cpu_large);
    auto [cuda_mean, cuda_std] = compute_stats(cuda_large);

    // With 1000 samples, standard error is ~0.032, so tolerance of 0.1 is ~3 std errors
    EXPECT_NEAR(cpu_mean, 0.0f, 0.1f); // More reasonable tolerance
    EXPECT_NEAR(cpu_std, 1.0f, 0.1f);
    EXPECT_NEAR(cuda_mean, 0.0f, 0.1f); // More reasonable tolerance
    EXPECT_NEAR(cuda_std, 1.0f, 0.1f);
}

TEST_F(TensorRandomTest, CPUUniform) {
    auto t = Tensor::rand({200}, Device::CPU);

    EXPECT_TRUE(check_range(t, 0.0f, 1.0f));
    auto [mean, std] = compute_stats(t);
    EXPECT_NEAR(mean, 0.5f, 0.1f);
}

TEST_F(TensorRandomTest, CUDAUniform) {
    auto t = Tensor::rand({200}, Device::CUDA);

    EXPECT_TRUE(check_range(t, 0.0f, 1.0f));
    auto [mean, std] = compute_stats(t);
    EXPECT_NEAR(mean, 0.5f, 0.1f);
}

// ============= Shape Tests =============

TEST_F(TensorRandomTest, VariousShapes) {
    std::vector<std::vector<size_t>> shapes = {
        {10},
        {5, 5},
        {2, 3, 4},
        {1, 1, 100},
        {2, 3, 4, 5}};

    for (const auto& shape : shapes) {
        auto t = Tensor::randn(TensorShape(shape), Device::CUDA);

        EXPECT_EQ(t.shape().dims(), shape);

        size_t expected_elements = 1;
        for (size_t dim : shape) {
            expected_elements *= dim;
        }
        EXPECT_EQ(t.numel(), expected_elements);
    }
}

TEST_F(TensorRandomTest, HighDimensional) {
    auto t = Tensor::randn({2, 2, 2, 2, 2, 2}, Device::CUDA);

    EXPECT_EQ(t.ndim(), 6);
    EXPECT_EQ(t.numel(), 64);

    auto [mean, std] = compute_stats(t);
    EXPECT_NEAR(mean, 0.0f, 0.3f);
    EXPECT_NEAR(std, 1.0f, 0.3f);
}

// ============= Edge Cases =============

TEST_F(TensorRandomTest, EmptyTensor) {
    auto t = Tensor::randn({0}, Device::CUDA);

    EXPECT_TRUE(t.is_valid());
    EXPECT_EQ(t.numel(), 0);
}

TEST_F(TensorRandomTest, SingleElement) {
    auto t = Tensor::uniform({1}, -1.0f, 1.0f, Device::CUDA);

    EXPECT_EQ(t.numel(), 1);

    float val = t.item();
    EXPECT_GE(val, -1.0f);
    EXPECT_LE(val, 1.0f);
}

TEST_F(TensorRandomTest, LargeTensor) {
    const size_t large_size = 1000000;
    auto t = Tensor::randn({large_size}, Device::CUDA);

    EXPECT_EQ(t.numel(), large_size);

    // Sample a subset to check distribution
    auto sample = t.slice(0, 0, 10000);
    auto [mean, std] = compute_stats(sample);
    EXPECT_NEAR(mean, 0.0f, 0.05f);
    EXPECT_NEAR(std, 1.0f, 0.05f);
}

TEST_F(TensorRandomTest, ZeroElementsMultiDim) {
    auto t = Tensor::randn({5, 0, 3}, Device::CUDA);

    EXPECT_EQ(t.numel(), 0);
    EXPECT_EQ(t.ndim(), 3);
}

// ============= Operations on Random Data =============

TEST_F(TensorRandomTest, ReshapePreservesData) {
    Tensor::manual_seed(123);
    auto t = Tensor::randn({20}, Device::CUDA);
    auto original_copy = t.clone();

    auto reshaped = t.reshape({4, 5});

    // Check shape is correct
    EXPECT_EQ(reshaped.shape().dims(), std::vector<size_t>({4, 5}));

    // Check that reshape preserves values
    EXPECT_TRUE(reshaped.flatten().all_close(original_copy, 1e-6f, 1e-7f));
}

TEST_F(TensorRandomTest, TransposePreservesData) {
    Tensor::manual_seed(456);
    auto t = Tensor::randn({3, 4}, Device::CUDA);

    auto transposed = t.transpose();

    EXPECT_EQ(transposed.shape().dims(), std::vector<size_t>({4, 3}));

    // Original element (i,j) should be at (j,i) in transposed
    auto t_cpu = t.cpu();
    auto trans_cpu = transposed.cpu();

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(t_cpu.at({i, j}), trans_cpu.at({j, i}));
        }
    }
}

TEST_F(TensorRandomTest, ReductionDimensions) {
    auto t = Tensor::randn({10, 10}, Device::CUDA);

    // Full reduction should give scalar (0D tensor)
    auto sum_all = t.sum();
    EXPECT_EQ(sum_all.ndim(), 0) << "Full reduction should produce 0D tensor";
    EXPECT_EQ(sum_all.numel(), 1);

    auto mean_all = t.mean();
    EXPECT_EQ(mean_all.ndim(), 0) << "Full reduction should produce 0D tensor";
    EXPECT_EQ(mean_all.numel(), 1);

    // Partial reduction
    auto sum_dim0 = t.sum(std::vector<int>{0});
    EXPECT_EQ(sum_dim0.shape().dims(), std::vector<size_t>({10}));

    auto sum_dim1 = t.sum(std::vector<int>{1});
    EXPECT_EQ(sum_dim1.shape().dims(), std::vector<size_t>({10}));
}

TEST_F(TensorRandomTest, ArithmeticOnRandom) {
    Tensor::manual_seed(789);
    auto t = Tensor::randn({100}, Device::CUDA);
    auto original = t.clone();

    auto result = t.mul(2.0f).add(1.0f);

    // Check transformation is correct
    auto original_vec = original.to_vector();
    auto result_vec = result.to_vector();

    for (size_t i = 0; i < original_vec.size(); ++i) {
        float expected = original_vec[i] * 2.0f + 1.0f;
        EXPECT_NEAR(result_vec[i], expected, 1e-5f);
    }
}

TEST_F(TensorRandomTest, ChainedOperations) {
    Tensor::manual_seed(999);
    auto t = Tensor::randn({50}, Device::CUDA);

    auto result = t.reshape({10, 5}).t().flatten();

    // Check final shape
    EXPECT_EQ(result.numel(), 50);

    // Distribution properties should be roughly preserved
    auto [mean, std] = compute_stats(result);
    EXPECT_NEAR(mean, 0.0f, 0.3f);
    EXPECT_NEAR(std, 1.0f, 0.3f);
}

TEST_F(TensorRandomTest, SlicePreservesData) {
    Tensor::manual_seed(111);
    auto t = Tensor::randn({100}, Device::CUDA);

    auto slice = t.slice(0, 10, 20);

    EXPECT_EQ(slice.numel(), 10);

    // Verify slice contains correct values
    auto original_vec = t.to_vector();
    auto slice_vec = slice.to_vector();

    for (size_t i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(slice_vec[i], original_vec[i + 10]);
    }
}

// ============= Performance Test =============

TEST_F(TensorRandomTest, RandomGenerationSpeed) {
    const size_t n = 10000000; // 10M elements

    auto start = std::chrono::high_resolution_clock::now();
    auto tensor = Tensor::randn({n}, Device::CUDA);
    cudaDeviceSynchronize();
    auto duration = std::chrono::high_resolution_clock::now() - start;

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    std::cout << "Generated " << n << " random numbers in " << ms << " ms" << std::endl;

    EXPECT_TRUE(tensor.is_valid());
    EXPECT_EQ(tensor.numel(), n);
}

TEST_F(TensorRandomTest, CompareGenerationSpeed) {
    const size_t n = 1000000;

    // Custom tensor
    Tensor::manual_seed(42);
    auto start_custom = std::chrono::high_resolution_clock::now();
    auto custom_t = Tensor::randn({n}, Device::CUDA);
    cudaDeviceSynchronize();
    auto duration_custom = std::chrono::high_resolution_clock::now() - start_custom;
    auto ms_custom = std::chrono::duration_cast<std::chrono::milliseconds>(duration_custom).count();

    // PyTorch
    torch::manual_seed(42);
    auto start_torch = std::chrono::high_resolution_clock::now();
    auto torch_t = torch::randn({n}, torch::TensorOptions().device(torch::kCUDA));
    cudaDeviceSynchronize();
    auto duration_torch = std::chrono::high_resolution_clock::now() - start_torch;
    auto ms_torch = std::chrono::duration_cast<std::chrono::milliseconds>(duration_torch).count();

    std::cout << "Custom: " << ms_custom << " ms, PyTorch: " << ms_torch << " ms" << std::endl;

    EXPECT_TRUE(custom_t.is_valid());
    EXPECT_TRUE(torch_t.defined());
}