/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tensor_hardening_test_utils.hpp"

#include "core/tensor/internal/lazy_executor.hpp"
#include "core/tensor/internal/lazy_ir.hpp"

#include <algorithm>
#include <limits>
#include <optional>
#include <set>

using namespace lfs::core;
using namespace tensor_hardening;

namespace {

    class LazyStateGuard {
    public:
        explicit LazyStateGuard(const bool enable_fusion = true) {
            internal::clear_lazy_ir_for_testing();
            internal::lazy_executor_clear_registry_for_testing();
            internal::lazy_executor_reset_diagnostics_for_testing();
            internal::lazy_executor_set_size_heuristic_override_for_testing(false);
            internal::lazy_executor_set_size_threshold_override_for_testing(std::nullopt);
            internal::lazy_executor_set_pointwise_fusion_override_for_testing(enable_fusion);
        }

        ~LazyStateGuard() {
            internal::clear_lazy_ir_for_testing();
            internal::lazy_executor_clear_registry_for_testing();
            internal::lazy_executor_reset_diagnostics_for_testing();
            internal::lazy_executor_set_pointwise_fusion_override_for_testing(std::nullopt);
            internal::lazy_executor_set_size_heuristic_override_for_testing(std::nullopt);
            internal::lazy_executor_set_size_threshold_override_for_testing(std::nullopt);
        }
    };

    std::vector<int64_t> int64_values(const Tensor& tensor) {
        return tensor.cpu().to_vector_int64();
    }

} // namespace

TEST(HardeningThemeE_Numerical, E1_AllCloseUsesTorchNaNPolicy) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const auto ours_nan = lfs_float_tensor({nan}, {1}, Device::CPU);
    const auto ours_zero = lfs_float_tensor({0.0f}, {1}, Device::CPU);
    const auto torch_nan = torch::tensor({nan});
    const auto torch_zero = torch::tensor({0.0f});

    EXPECT_EQ(ours_nan.all_close(ours_zero), torch::allclose(torch_nan, torch_zero));
    EXPECT_EQ(ours_nan.all_close(ours_nan), torch::allclose(torch_nan, torch_nan));
}

TEST_F(CudaTest, E2_IntegerTensorTrueDivisionPromotesLikeTorch_CPUAndCUDA) {
    for (const Device device : {Device::CPU, Device::CUDA}) {
        const auto lhs = lfs_int_tensor({3}, {1}, device);
        const auto rhs = lfs_int_tensor({2}, {1}, device);
        const auto ours = lhs.div(rhs);
        auto torch_lhs = torch::tensor({3}, torch::TensorOptions().dtype(torch::kInt32));
        auto torch_rhs = torch::tensor({2}, torch::TensorOptions().dtype(torch::kInt32));
        if (device == Device::CUDA) {
            torch_lhs = torch_lhs.cuda();
            torch_rhs = torch_rhs.cuda();
        }
        const auto theirs = torch_lhs / torch_rhs;
        EXPECT_EQ(ours.dtype(), DataType::Float32);
        expect_float_values_match(ours, theirs,
                                  device == Device::CPU ? "E2 tensor div CPU" : "E2 tensor div CUDA");
    }
}

TEST_F(CudaTest, E2_IntegerScalarTrueDivisionPromotesLikeTorch_CPUAndCUDA) {
    for (const Device device : {Device::CPU, Device::CUDA}) {
        const auto ours = lfs_int_tensor({3}, {1}, device).div(2.0f);
        auto theirs = torch::tensor({3}, torch::TensorOptions().dtype(torch::kInt32));
        if (device == Device::CUDA) {
            theirs = theirs.cuda();
        }
        theirs = theirs / 2.0f;
        EXPECT_EQ(ours.dtype(), DataType::Float32);
        expect_float_values_match(ours, theirs,
                                  device == Device::CPU ? "E2 scalar div CPU" : "E2 scalar div CUDA");
    }
}

TEST_F(CudaTest, E3_Int32ScalarArithmeticPreservesValuesBeyondFloatMantissa) {
    constexpr int scalar = 16'777'217;
    for (const Device device : {Device::CPU, Device::CUDA}) {
        const auto ours = lfs_int_tensor({0}, {1}, device).add(scalar);
        auto theirs = torch::tensor({0}, torch::TensorOptions().dtype(torch::kInt32));
        if (device == Device::CUDA) {
            theirs = theirs.cuda();
        }
        theirs = theirs + scalar;
        expect_int_values_match(ours, theirs,
                                device == Device::CPU ? "E3 scalar add CPU" : "E3 scalar add CUDA");
    }
}

TEST_F(CudaTest, E3_Int32ScalarComparisonPreservesValuesBeyondFloatMantissa) {
    constexpr int scalar = 16'777'217;
    for (const Device device : {Device::CPU, Device::CUDA}) {
        const auto ours = lfs_int_tensor({16'777'216}, {1}, device).eq(scalar);
        auto theirs = torch::tensor({16'777'216}, torch::TensorOptions().dtype(torch::kInt32));
        if (device == Device::CUDA) {
            theirs = theirs.cuda();
        }
        theirs = theirs.eq(scalar);
        expect_bool_values_match(ours, theirs,
                                 device == Device::CPU ? "E3 scalar compare CPU" : "E3 scalar compare CUDA");
    }
}

TEST_F(CudaTest, E4_IntegerSqrtPromotesLikeTorch_CPUAndCUDA) {
    for (const Device device : {Device::CPU, Device::CUDA}) {
        const auto ours = lfs_int_tensor({2}, {1}, device).sqrt();
        auto theirs = torch::tensor({2}, torch::TensorOptions().dtype(torch::kInt32));
        if (device == Device::CUDA) {
            theirs = theirs.cuda();
        }
        theirs = theirs.sqrt();
        EXPECT_EQ(ours.dtype(), DataType::Float32);
        expect_float_values_match(ours, theirs,
                                  device == Device::CPU ? "E4 sqrt CPU" : "E4 sqrt CUDA");
    }
}

TEST_F(CudaTest, E4_IntegerReciprocalAndExpPromoteLikeTorch_CPUAndCUDA) {
    for (const Device device : {Device::CPU, Device::CUDA}) {
        const auto input = lfs_int_tensor({2}, {1}, device);
        auto torch_input = torch::tensor({2}, torch::TensorOptions().dtype(torch::kInt32));
        if (device == Device::CUDA) {
            torch_input = torch_input.cuda();
        }

        const auto reciprocal = input.reciprocal();
        EXPECT_EQ(reciprocal.dtype(), DataType::Float32);
        expect_float_values_match(reciprocal, torch_input.reciprocal(),
                                  device == Device::CPU ? "E4 reciprocal CPU" : "E4 reciprocal CUDA");

        const auto exponential = input.exp();
        EXPECT_EQ(exponential.dtype(), DataType::Float32);
        expect_float_values_match(exponential, torch_input.exp(),
                                  device == Device::CPU ? "E4 exp CPU" : "E4 exp CUDA");
    }
}

TEST_F(CudaTest, E5_AffineFoldingPreservesSequentialOverflowSemantics) {
    LazyStateGuard guard;
    const float maximum = std::numeric_limits<float>::max();
    const auto ours = Tensor::full({1024}, maximum, Device::CUDA).mul(2.0f).div(2.0f);
    const auto theirs = torch::full({1024}, maximum,
                                    torch::TensorOptions().device(torch::kCUDA))
                            .mul(2.0f)
                            .div(2.0f);
    expect_float_values_match(ours, theirs, "E5 affine overflow folding");
}

TEST_F(CudaTest, E5_AffineFoldingPreservesSequentialDivisionByZeroSemantics) {
    LazyStateGuard guard;
    const auto ours = Tensor::zeros({1024}, Device::CUDA).add(1.0f).div(0.0f);
    const auto theirs = torch::zeros({1024}, torch::TensorOptions().device(torch::kCUDA))
                            .add(1.0f)
                            .div(0.0f);
    expect_float_values_match(ours, theirs, "E5 affine division by zero folding");
}

TEST_F(CudaTest, E6_FusedMaxMinUseInfiniteIdentities) {
    LazyStateGuard guard;
    const float maximum = std::numeric_limits<float>::max();

    const auto ours_max = Tensor::full({2048}, -maximum, Device::CUDA).mul(2.0f).max();
    const auto torch_max = torch::full({2048}, -maximum,
                                       torch::TensorOptions().device(torch::kCUDA))
                               .mul(2.0f)
                               .max();
    expect_float_values_match(ours_max, torch_max, "E6 fused max identity");

    const auto ours_min = Tensor::full({2048}, maximum, Device::CUDA).mul(2.0f).min();
    const auto torch_min = torch::full({2048}, maximum,
                                       torch::TensorOptions().device(torch::kCUDA))
                               .mul(2.0f)
                               .min();
    expect_float_values_match(ours_min, torch_min, "E6 fused min identity");
}

TEST_F(CudaTest, E6_FusedMaxMinPropagateAllNaNInputs) {
    LazyStateGuard guard;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const auto ours_max = Tensor::full({2048}, nan, Device::CUDA).mul(1.0f).max();
    const auto ours_min = Tensor::full({2048}, nan, Device::CUDA).mul(1.0f).min();
    const auto torch_input = torch::full({2048}, nan,
                                         torch::TensorOptions().device(torch::kCUDA));
    expect_float_values_match(ours_max, torch_input.mul(1.0f).max(), "E6 all-NaN max");
    expect_float_values_match(ours_min, torch_input.mul(1.0f).min(), "E6 all-NaN min");
}

TEST_F(CudaTest, E7_CUDAReluPropagatesNaNLikeTorch) {
    LazyStateGuard guard;
    std::vector<float> values(1024, -1.0f);
    values[0] = std::numeric_limits<float>::quiet_NaN();
    const auto ours = lfs_float_tensor(values, {1024}, Device::CUDA).relu();
    const auto theirs = torch::tensor(values, torch::TensorOptions().device(torch::kCUDA)).relu();
    expect_float_values_match(ours, theirs, "E7 CUDA relu NaN");
}

TEST_F(CudaTest, E7_CUDALogSqrtAndRsqrtFollowTorchDomainSemantics) {
    LazyStateGuard guard;
    std::vector<float> values(1024, 4.0f);
    values[0] = std::numeric_limits<float>::quiet_NaN();
    values[1] = -1.0f;
    values[2] = 0.0f;
    const auto ours = lfs_float_tensor(values, {1024}, Device::CUDA);
    const auto theirs = torch::tensor(values, torch::TensorOptions().device(torch::kCUDA));
    expect_float_values_match(ours.log(), theirs.log(), "E7 CUDA log domain");
    expect_float_values_match(ours.sqrt(), theirs.sqrt(), "E7 CUDA sqrt domain");
    expect_float_values_match(ours.rsqrt(), theirs.rsqrt(), "E7 CUDA rsqrt domain");
}

TEST_F(CudaTest, E8_WideRangeRandintDoesNotCollapseDistribution) {
    constexpr int count = 8192;
    const auto ours = Tensor::randint({count}, std::numeric_limits<int>::min(),
                                      std::numeric_limits<int>::max(), Device::CUDA,
                                      DataType::Int32)
                          .cpu()
                          .to_vector_int();
    const auto theirs_tensor = torch::randint(
        std::numeric_limits<int>::min(), std::numeric_limits<int>::max(), {count},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    const auto theirs_cpu = theirs_tensor.cpu().contiguous();
    const auto* theirs_data = theirs_cpu.data_ptr<int>();
    const std::set<int> ours_unique(ours.begin(), ours.end());
    const std::set<int> torch_unique(theirs_data, theirs_data + count);
    ASSERT_GT(torch_unique.size(), 8000u);
    EXPECT_GT(ours_unique.size(), 8000u)
        << "torch unique=" << torch_unique.size() << ", ours unique=" << ours_unique.size();
}

TEST_F(CudaTest, E9_MultinomialLargeFiniteWeightsMatchesTorch) {
    constexpr int samples = 20'000;
    const float maximum = std::numeric_limits<float>::max();
    const auto weights = lfs_float_tensor({maximum, maximum}, {2}, Device::CUDA);
    const auto ours = int64_values(Tensor::multinomial(weights, samples, true));
    // LibTorch 2.7.1 overflows its Float32 cumulative sum for
    // {FLT_MAX, FLT_MAX}: CUDA asserts and CPU collapses to the last bucket.
    // Multinomial probabilities are scale invariant, so normalized {1, 1}
    // supplies the torch-compatible semantic oracle without oracle overflow.
    const auto torch_weights = torch::tensor({1.0f, 1.0f});
    const auto theirs = torch::multinomial(torch_weights, samples, true);
    const auto* torch_data = theirs.data_ptr<int64_t>();

    const double ours_zero_fraction =
        static_cast<double>(std::count(ours.begin(), ours.end(), 0)) / samples;
    const double torch_zero_fraction =
        static_cast<double>(std::count(torch_data, torch_data + samples, int64_t{0})) / samples;
    EXPECT_NEAR(ours_zero_fraction, torch_zero_fraction, 0.05)
        << "ours zero fraction=" << ours_zero_fraction
        << ", torch CPU zero fraction=" << torch_zero_fraction;
}

TEST_F(CudaTest, E10_SparseNoReplacementMultinomialMatchesTorchContract) {
    const auto weights = lfs_float_tensor({1.0f, 0.0f}, {2}, Device::CUDA);
    const auto torch_cpu_weights = torch::tensor({1.0f, 0.0f});
    const auto torch_cuda_weights = torch_cpu_weights.cuda();

    bool ours_threw = false;
    bool torch_cpu_threw = false;
    bool torch_cuda_threw = false;
    std::vector<int64_t> ours_values;
    std::vector<int64_t> torch_cpu_values;
    std::vector<int64_t> torch_cuda_values;

    try {
        ours_values = int64_values(Tensor::multinomial(weights, 2, false));
    } catch (const std::exception&) {
        ours_threw = true;
    }
    try {
        const auto result = torch::multinomial(torch_cpu_weights, 2, false).contiguous();
        const auto* values = result.data_ptr<int64_t>();
        torch_cpu_values.assign(values, values + result.numel());
    } catch (const c10::Error&) {
        torch_cpu_threw = true;
    }
    try {
        const auto result = torch::multinomial(torch_cuda_weights, 2, false).cpu().contiguous();
        const auto* values = result.data_ptr<int64_t>();
        torch_cuda_values.assign(values, values + result.numel());
    } catch (const c10::Error&) {
        torch_cuda_threw = true;
    }

    EXPECT_EQ(torch_cpu_threw, torch_cuda_threw);
    EXPECT_EQ(ours_threw, torch_cuda_threw);
    if (!ours_threw && !torch_cuda_threw) {
        std::sort(ours_values.begin(), ours_values.end());
        std::sort(torch_cpu_values.begin(), torch_cpu_values.end());
        std::sort(torch_cuda_values.begin(), torch_cuda_values.end());
        EXPECT_EQ(torch_cpu_values, torch_cuda_values);
        EXPECT_EQ(ours_values, torch_cuda_values);
    }
}

TEST(HardeningThemeE_Numerical, E11_LinspaceMatchesTorchWideRangeOverflow) {
    const float maximum = std::numeric_limits<float>::max();
    const auto ours = Tensor::linspace(-maximum, maximum, 3, Device::CPU);
    const auto theirs = torch::linspace(-maximum, maximum, 3,
                                        torch::TensorOptions().dtype(torch::kFloat32));
    // LibTorch 2.7.1 computes a Float32 step and fills from both endpoints. The
    // overflowing step makes this observable result {NaN, -Inf, NaN}; it is
    // the compatibility contract here rather than a higher-precision alternative.
    const auto* torch_values = theirs.data_ptr<float>();
    ASSERT_TRUE(std::isnan(torch_values[0]));
    ASSERT_TRUE(std::isinf(torch_values[1]) && std::signbit(torch_values[1]));
    ASSERT_TRUE(std::isnan(torch_values[2]));
    expect_float_values_match(ours, theirs, "E11 linspace wide-range overflow");
}

TEST_F(CudaTest, E12_RandomEndpointStressMatchesTorchIntervalContracts) {
    // The bad cuRAND endpoint has probability about 2^-32 per draw.  This is a
    // bounded runtime probe; a pass is not proof and is reported as unverified.
    constexpr int count = 1 << 20;
    const auto uniform = Tensor::uniform({count}, 2.0f, 4.0f, Device::CUDA).cpu().to_vector();
    EXPECT_TRUE(std::all_of(uniform.begin(), uniform.end(), [](const float value) {
        return value >= 2.0f && value < 4.0f;
    }));

    const auto bernoulli = Tensor::bernoulli({count}, 1.0f, Device::CUDA).cpu().to_vector();
    EXPECT_TRUE(std::all_of(bernoulli.begin(), bernoulli.end(), [](const float value) {
        return value == 1.0f;
    }));
}
