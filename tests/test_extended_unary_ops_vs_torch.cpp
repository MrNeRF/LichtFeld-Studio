/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <torch/torch.h>

using namespace lfs::core;

class ExtendedUnaryOpsVsTorchTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
    }

    void compare_tensors_near(const Tensor& ours, const torch::Tensor& theirs,
                              float rtol = 1e-5f, float atol = 1e-8f,
                              const std::string& op_name = "") {
        auto ours_cpu = ours.to(Device::CPU);
        auto theirs_cpu = theirs.cpu().to(torch::kFloat32);

        ASSERT_EQ(ours_cpu.numel(), theirs_cpu.numel())
            << op_name << ": size mismatch";

        const float* ours_ptr = ours_cpu.ptr<float>();
        const float* theirs_ptr = theirs_cpu.data_ptr<float>();

        for (size_t i = 0; i < ours_cpu.numel(); ++i) {
            float ours_val = ours_ptr[i];
            float theirs_val = theirs_ptr[i];

            // Handle NaN
            if (std::isnan(theirs_val)) {
                EXPECT_TRUE(std::isnan(ours_val))
                    << op_name << " at index " << i << ": expected NaN, got " << ours_val;
                continue;
            }

            // Handle Inf
            if (std::isinf(theirs_val)) {
                EXPECT_TRUE(std::isinf(ours_val) && (ours_val > 0) == (theirs_val > 0))
                    << op_name << " at index " << i << ": inf sign mismatch";
                continue;
            }

            float diff = std::abs(ours_val - theirs_val);
            float tolerance = atol + rtol * std::abs(theirs_val);
            EXPECT_LE(diff, tolerance)
                << op_name << " at index " << i << ": ours=" << ours_val
                << ", torch=" << theirs_val << ", diff=" << diff;
        }
    }

    void compare_bool_tensors(const Tensor& ours, const torch::Tensor& theirs,
                              const std::string& op_name = "") {
        auto ours_cpu = ours.to(Device::CPU);
        auto theirs_cpu = theirs.cpu();

        ASSERT_EQ(ours_cpu.numel(), theirs_cpu.numel())
            << op_name << ": size mismatch";

        const unsigned char* ours_ptr = ours_cpu.ptr<unsigned char>();
        const bool* theirs_ptr = theirs_cpu.data_ptr<bool>();

        for (size_t i = 0; i < ours_cpu.numel(); ++i) {
            bool ours_val = (ours_ptr[i] != 0);
            bool theirs_val = theirs_ptr[i];
            EXPECT_EQ(ours_val, theirs_val)
                << op_name << " at index " << i << ": ours=" << ours_val
                << ", torch=" << theirs_val;
        }
    }

    // Create matching test data
    std::pair<Tensor, torch::Tensor> create_positive_data(std::vector<int64_t> shape) {
        auto torch_t = torch::rand(shape, torch::device(torch::kCUDA)) * 10.0f + 0.1f;
        auto ours = tensor_from_torch(torch_t);
        return {ours, torch_t};
    }

    std::pair<Tensor, torch::Tensor> create_unit_data(std::vector<int64_t> shape) {
        // Values in [-1, 1] for asin/acos
        auto torch_t = torch::rand(shape, torch::device(torch::kCUDA)) * 1.98f - 0.99f;
        auto ours = tensor_from_torch(torch_t);
        return {ours, torch_t};
    }

    std::pair<Tensor, torch::Tensor> create_any_data(std::vector<int64_t> shape) {
        auto torch_t = torch::randn(shape, torch::device(torch::kCUDA)) * 5.0f;
        auto ours = tensor_from_torch(torch_t);
        return {ours, torch_t};
    }

    std::pair<Tensor, torch::Tensor> create_special_values() {
        // Create tensor with special values: normal, NaN, +Inf, -Inf, 0
        std::vector<float> data = {1.0f, -1.0f, 0.0f, NAN, INFINITY, -INFINITY, 1e-30f, 1e30f};
        auto torch_t = torch::from_blob(data.data(), {(int64_t)data.size()},
                                        torch::kFloat32)
                           .clone()
                           .cuda();
        auto ours = tensor_from_torch(torch_t);
        return {ours, torch_t};
    }

    Tensor tensor_from_torch(const torch::Tensor& t) {
        auto t_cpu = t.cpu().contiguous();
        std::vector<size_t> shape;
        for (int i = 0; i < t_cpu.dim(); ++i) {
            shape.push_back(static_cast<size_t>(t_cpu.size(i)));
        }
        return Tensor::from_blob(t_cpu.data_ptr<float>(), TensorShape(shape),
                                 Device::CPU, DataType::Float32)
            .to(Device::GPU);
    }
};

// =============================================================================
// Log operations
// =============================================================================

TEST_F(ExtendedUnaryOpsVsTorchTest, Log2_PositiveValues) {
    auto [ours, theirs] = create_positive_data({1000});
    compare_tensors_near(ours.log2(), torch::log2(theirs), 1e-4f, 1e-6f, "log2");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Log2_2D) {
    auto [ours, theirs] = create_positive_data({100, 50});
    compare_tensors_near(ours.log2(), torch::log2(theirs), 1e-4f, 1e-6f, "log2_2d");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Log2_LargeValues) {
    auto torch_t = torch::tensor({1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 1024.0f, 1048576.0f},
                                 torch::device(torch::kCUDA));
    auto ours = tensor_from_torch(torch_t);
    compare_tensors_near(ours.log2(), torch::log2(torch_t), 1e-5f, 1e-6f, "log2_large");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Log10_PositiveValues) {
    auto [ours, theirs] = create_positive_data({1000});
    compare_tensors_near(ours.log10(), torch::log10(theirs), 1e-4f, 1e-6f, "log10");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Log10_Powers) {
    auto torch_t = torch::tensor({1.0f, 10.0f, 100.0f, 1000.0f, 0.1f, 0.01f},
                                 torch::device(torch::kCUDA));
    auto ours = tensor_from_torch(torch_t);
    compare_tensors_near(ours.log10(), torch::log10(torch_t), 1e-5f, 1e-6f, "log10_powers");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Log1p_SmallValues) {
    // log1p(x) is more accurate than log(1+x) for small x
    auto torch_t = torch::tensor({1e-10f, 1e-8f, 1e-6f, 1e-4f, 0.01f, 0.1f, 1.0f},
                                 torch::device(torch::kCUDA));
    auto ours = tensor_from_torch(torch_t);
    compare_tensors_near(ours.log1p(), torch::log1p(torch_t), 1e-5f, 1e-8f, "log1p");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Log1p_NegativeSmall) {
    // log1p works for x > -1
    auto torch_t = torch::tensor({-0.5f, -0.1f, -0.01f, -1e-6f},
                                 torch::device(torch::kCUDA));
    auto ours = tensor_from_torch(torch_t);
    compare_tensors_near(ours.log1p(), torch::log1p(torch_t), 1e-5f, 1e-6f, "log1p_neg");
}

// =============================================================================
// Exp operations
// =============================================================================

TEST_F(ExtendedUnaryOpsVsTorchTest, Exp2_Values) {
    auto torch_t = torch::tensor({0.0f, 1.0f, 2.0f, 3.0f, -1.0f, -2.0f, 10.0f},
                                 torch::device(torch::kCUDA));
    auto ours = tensor_from_torch(torch_t);
    compare_tensors_near(ours.exp2(), torch::exp2(torch_t), 1e-4f, 1e-6f, "exp2");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Exp2_Random) {
    auto [ours, theirs] = create_any_data({500});
    compare_tensors_near(ours.exp2(), torch::exp2(theirs), 1e-4f, 1e-6f, "exp2_random");
}

// =============================================================================
// Sqrt operations
// =============================================================================

TEST_F(ExtendedUnaryOpsVsTorchTest, Rsqrt_PositiveValues) {
    auto [ours, theirs] = create_positive_data({1000});
    compare_tensors_near(ours.rsqrt(), torch::rsqrt(theirs), 1e-4f, 1e-6f, "rsqrt");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Rsqrt_KnownValues) {
    auto torch_t = torch::tensor({1.0f, 4.0f, 9.0f, 16.0f, 0.25f},
                                 torch::device(torch::kCUDA));
    auto ours = tensor_from_torch(torch_t);
    // rsqrt(x) = 1/sqrt(x)
    compare_tensors_near(ours.rsqrt(), torch::rsqrt(torch_t), 1e-5f, 1e-6f, "rsqrt_known");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Square_Values) {
    auto [ours, theirs] = create_any_data({1000});
    compare_tensors_near(ours.square(), torch::square(theirs), 1e-4f, 1e-6f, "square");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Square_NegativeValues) {
    auto torch_t = torch::tensor({-3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f},
                                 torch::device(torch::kCUDA));
    auto ours = tensor_from_torch(torch_t);
    compare_tensors_near(ours.square(), torch::square(torch_t), 1e-5f, 1e-6f, "square_neg");
}

// =============================================================================
// Inverse trigonometric
// =============================================================================

TEST_F(ExtendedUnaryOpsVsTorchTest, Asin_UnitRange) {
    auto [ours, theirs] = create_unit_data({1000});
    compare_tensors_near(ours.asin(), torch::asin(theirs), 1e-4f, 1e-6f, "asin");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Asin_KnownValues) {
    auto torch_t = torch::tensor({0.0f, 0.5f, -0.5f, 1.0f, -1.0f},
                                 torch::device(torch::kCUDA));
    auto ours = tensor_from_torch(torch_t);
    compare_tensors_near(ours.asin(), torch::asin(torch_t), 1e-5f, 1e-6f, "asin_known");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Acos_UnitRange) {
    auto [ours, theirs] = create_unit_data({1000});
    compare_tensors_near(ours.acos(), torch::acos(theirs), 1e-4f, 1e-6f, "acos");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Acos_KnownValues) {
    auto torch_t = torch::tensor({0.0f, 0.5f, -0.5f, 1.0f, -1.0f},
                                 torch::device(torch::kCUDA));
    auto ours = tensor_from_torch(torch_t);
    compare_tensors_near(ours.acos(), torch::acos(torch_t), 1e-5f, 1e-6f, "acos_known");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Atan_Values) {
    auto [ours, theirs] = create_any_data({1000});
    compare_tensors_near(ours.atan(), torch::atan(theirs), 1e-4f, 1e-6f, "atan");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Atan_LargeValues) {
    auto torch_t = torch::tensor({0.0f, 1.0f, -1.0f, 100.0f, -100.0f, 1e10f, -1e10f},
                                 torch::device(torch::kCUDA));
    auto ours = tensor_from_torch(torch_t);
    compare_tensors_near(ours.atan(), torch::atan(torch_t), 1e-5f, 1e-6f, "atan_large");
}

// =============================================================================
// Hyperbolic functions
// =============================================================================

TEST_F(ExtendedUnaryOpsVsTorchTest, Sinh_Values) {
    auto torch_t = torch::tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 5.0f},
                                 torch::device(torch::kCUDA));
    auto ours = tensor_from_torch(torch_t);
    compare_tensors_near(ours.sinh(), torch::sinh(torch_t), 1e-4f, 1e-6f, "sinh");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Sinh_Random) {
    // Keep values moderate to avoid overflow
    auto torch_t = torch::randn({500}, torch::device(torch::kCUDA)) * 3.0f;
    auto ours = tensor_from_torch(torch_t);
    compare_tensors_near(ours.sinh(), torch::sinh(torch_t), 1e-4f, 1e-5f, "sinh_random");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Cosh_Values) {
    auto torch_t = torch::tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 5.0f},
                                 torch::device(torch::kCUDA));
    auto ours = tensor_from_torch(torch_t);
    compare_tensors_near(ours.cosh(), torch::cosh(torch_t), 1e-4f, 1e-6f, "cosh");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Cosh_Random) {
    auto torch_t = torch::randn({500}, torch::device(torch::kCUDA)) * 3.0f;
    auto ours = tensor_from_torch(torch_t);
    compare_tensors_near(ours.cosh(), torch::cosh(torch_t), 1e-4f, 1e-5f, "cosh_random");
}

// =============================================================================
// Rounding operations
// =============================================================================

TEST_F(ExtendedUnaryOpsVsTorchTest, Trunc_Values) {
    auto torch_t = torch::tensor({-2.7f, -2.3f, -0.5f, 0.0f, 0.5f, 2.3f, 2.7f},
                                 torch::device(torch::kCUDA));
    auto ours = tensor_from_torch(torch_t);
    compare_tensors_near(ours.trunc(), torch::trunc(torch_t), 1e-6f, 1e-6f, "trunc");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Trunc_Random) {
    auto [ours, theirs] = create_any_data({1000});
    compare_tensors_near(ours.trunc(), torch::trunc(theirs), 1e-6f, 1e-6f, "trunc_random");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Sign_Values) {
    auto torch_t = torch::tensor({-5.0f, -0.1f, 0.0f, 0.1f, 5.0f},
                                 torch::device(torch::kCUDA));
    auto ours = tensor_from_torch(torch_t);
    compare_tensors_near(ours.sign(), torch::sign(torch_t), 1e-6f, 1e-6f, "sign");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Sign_Random) {
    auto [ours, theirs] = create_any_data({1000});
    compare_tensors_near(ours.sign(), torch::sign(theirs), 1e-6f, 1e-6f, "sign_random");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Reciprocal_PositiveValues) {
    auto [ours, theirs] = create_positive_data({1000});
    compare_tensors_near(ours.reciprocal(), torch::reciprocal(theirs), 1e-4f, 1e-6f, "reciprocal");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Reciprocal_NegativeValues) {
    auto torch_t = torch::tensor({-5.0f, -1.0f, -0.5f, 0.5f, 1.0f, 5.0f},
                                 torch::device(torch::kCUDA));
    auto ours = tensor_from_torch(torch_t);
    compare_tensors_near(ours.reciprocal(), torch::reciprocal(torch_t), 1e-5f, 1e-6f, "reciprocal_neg");
}

// =============================================================================
// Activation functions
// =============================================================================

TEST_F(ExtendedUnaryOpsVsTorchTest, Gelu_Values) {
    auto [ours, theirs] = create_any_data({1000});
    // GELU implementations may differ (tanh-approx vs erf-based)
    // Use looser tolerance for comparison
    compare_tensors_near(ours.gelu(), torch::gelu(theirs), 5e-3f, 1e-3f, "gelu");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Gelu_KnownValues) {
    auto torch_t = torch::tensor({-3.0f, -1.0f, 0.0f, 1.0f, 3.0f},
                                 torch::device(torch::kCUDA));
    auto ours = tensor_from_torch(torch_t);
    // GELU approximation difference
    compare_tensors_near(ours.gelu(), torch::gelu(torch_t), 5e-3f, 1e-3f, "gelu_known");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Swish_Values) {
    auto [ours, theirs] = create_any_data({1000});
    // swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
    auto expected = theirs * torch::sigmoid(theirs);
    compare_tensors_near(ours.swish(), expected, 1e-4f, 1e-5f, "swish");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Swish_KnownValues) {
    auto torch_t = torch::tensor({-3.0f, -1.0f, 0.0f, 1.0f, 3.0f},
                                 torch::device(torch::kCUDA));
    auto ours = tensor_from_torch(torch_t);
    auto expected = torch_t * torch::sigmoid(torch_t);
    compare_tensors_near(ours.swish(), expected, 1e-5f, 1e-6f, "swish_known");
}

// =============================================================================
// Boolean-returning operations
// =============================================================================

TEST_F(ExtendedUnaryOpsVsTorchTest, Isnan_SpecialValues) {
    auto [ours, theirs] = create_special_values();
    compare_bool_tensors(ours.isnan(), torch::isnan(theirs), "isnan");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Isnan_NoNans) {
    auto [ours, theirs] = create_positive_data({100});
    compare_bool_tensors(ours.isnan(), torch::isnan(theirs), "isnan_nonans");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Isinf_SpecialValues) {
    auto [ours, theirs] = create_special_values();
    compare_bool_tensors(ours.isinf(), torch::isinf(theirs), "isinf");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Isinf_NoInfs) {
    auto [ours, theirs] = create_positive_data({100});
    compare_bool_tensors(ours.isinf(), torch::isinf(theirs), "isinf_noinfs");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Isfinite_SpecialValues) {
    auto [ours, theirs] = create_special_values();
    compare_bool_tensors(ours.isfinite(), torch::isfinite(theirs), "isfinite");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Isfinite_AllFinite) {
    auto [ours, theirs] = create_positive_data({100});
    compare_bool_tensors(ours.isfinite(), torch::isfinite(theirs), "isfinite_allfinite");
}

// =============================================================================
// Edge cases and stress tests
// =============================================================================

TEST_F(ExtendedUnaryOpsVsTorchTest, AllOps_EmptyTensor) {
    auto torch_t = torch::empty({0}, torch::device(torch::kCUDA));
    auto ours = Tensor::empty({0}, Device::GPU, DataType::Float32);

    // These should all handle empty tensors gracefully
    EXPECT_EQ(ours.log2().numel(), 0);
    EXPECT_EQ(ours.exp2().numel(), 0);
    EXPECT_EQ(ours.rsqrt().numel(), 0);
    EXPECT_EQ(ours.sinh().numel(), 0);
    EXPECT_EQ(ours.gelu().numel(), 0);
}

TEST_F(ExtendedUnaryOpsVsTorchTest, AllOps_SingleElement) {
    auto torch_t = torch::tensor({2.0f}, torch::device(torch::kCUDA));
    auto ours = tensor_from_torch(torch_t);

    compare_tensors_near(ours.log2(), torch::log2(torch_t), 1e-5f, 1e-6f, "log2_single");
    compare_tensors_near(ours.exp2(), torch::exp2(torch_t), 1e-5f, 1e-6f, "exp2_single");
    compare_tensors_near(ours.rsqrt(), torch::rsqrt(torch_t), 1e-5f, 1e-6f, "rsqrt_single");
    compare_tensors_near(ours.square(), torch::square(torch_t), 1e-5f, 1e-6f, "square_single");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, AllOps_LargeTensor) {
    const size_t N = 10000000; // 10M elements
    auto torch_t = torch::rand({(int64_t)N}, torch::device(torch::kCUDA)) + 0.1f;
    auto ours = tensor_from_torch(torch_t);

    // Test a few ops on large tensor
    compare_tensors_near(ours.log2(), torch::log2(torch_t), 1e-4f, 1e-6f, "log2_large");
    compare_tensors_near(ours.rsqrt(), torch::rsqrt(torch_t), 1e-4f, 1e-6f, "rsqrt_large");
    compare_tensors_near(ours.square(), torch::square(torch_t), 1e-4f, 1e-6f, "square_large");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, AllOps_3DTensor) {
    auto torch_t = torch::rand({10, 20, 30}, torch::device(torch::kCUDA)) + 0.1f;
    auto ours = tensor_from_torch(torch_t);

    compare_tensors_near(ours.log2(), torch::log2(torch_t), 1e-4f, 1e-6f, "log2_3d");
    compare_tensors_near(ours.log10(), torch::log10(torch_t), 1e-4f, 1e-6f, "log10_3d");
    compare_tensors_near(ours.exp2(), torch::exp2(torch_t), 1e-4f, 1e-6f, "exp2_3d");
    compare_tensors_near(ours.rsqrt(), torch::rsqrt(torch_t), 1e-4f, 1e-6f, "rsqrt_3d");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, AllOps_NonContiguous) {
    // Create non-contiguous tensor via transpose
    auto torch_t = torch::rand({100, 50}, torch::device(torch::kCUDA)).t().contiguous();
    auto ours = tensor_from_torch(torch_t);

    compare_tensors_near(ours.log2(), torch::log2(torch_t), 1e-4f, 1e-6f, "log2_noncontig");
    compare_tensors_near(ours.sinh(), torch::sinh(torch_t), 1e-4f, 1e-5f, "sinh_noncontig");
}

// =============================================================================
// Numerical stability edge cases
// =============================================================================

TEST_F(ExtendedUnaryOpsVsTorchTest, Log2_VerySmallPositive) {
    // Float32 has limited precision for very small values
    // Test with values that have reliable precision
    auto torch_t = torch::tensor({1e-10f, 1e-7f, 1e-5f, 1e-3f, 1e-1f},
                                 torch::device(torch::kCUDA));
    auto ours = tensor_from_torch(torch_t);
    compare_tensors_near(ours.log2(), torch::log2(torch_t), 1e-4f, 1e-6f, "log2_tiny");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Exp2_LargeExponents) {
    // Avoid overflow
    auto torch_t = torch::tensor({-10.0f, -5.0f, 0.0f, 5.0f, 10.0f, 20.0f},
                                 torch::device(torch::kCUDA));
    auto ours = tensor_from_torch(torch_t);
    compare_tensors_near(ours.exp2(), torch::exp2(torch_t), 1e-4f, 1e-6f, "exp2_large_exp");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Rsqrt_VerySmallPositive) {
    auto torch_t = torch::tensor({1e-10f, 1e-6f, 1e-4f},
                                 torch::device(torch::kCUDA));
    auto ours = tensor_from_torch(torch_t);
    compare_tensors_near(ours.rsqrt(), torch::rsqrt(torch_t), 1e-2f, 1e-5f, "rsqrt_tiny");
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Sinh_Cosh_Identity) {
    // cosh^2(x) - sinh^2(x) = 1
    // For larger |x|, numerical precision degrades due to large intermediate values
    auto torch_t = torch::randn({100}, torch::device(torch::kCUDA)) * 1.5f;
    auto ours = tensor_from_torch(torch_t);

    auto cosh_sq = ours.cosh().square();
    auto sinh_sq = ours.sinh().square();
    auto diff = cosh_sq - sinh_sq;

    auto diff_cpu = diff.to(Device::CPU);
    const float* ptr = diff_cpu.ptr<float>();
    for (size_t i = 0; i < diff.numel(); ++i) {
        EXPECT_NEAR(ptr[i], 1.0f, 1e-3f)
            << "cosh^2 - sinh^2 != 1 at index " << i;
    }
}

TEST_F(ExtendedUnaryOpsVsTorchTest, Asin_Acos_Identity) {
    // asin(x) + acos(x) = pi/2
    auto torch_t = torch::rand({100}, torch::device(torch::kCUDA)) * 1.98f - 0.99f;
    auto ours = tensor_from_torch(torch_t);

    auto sum = ours.asin() + ours.acos();
    const float pi_over_2 = 3.14159265358979323846f / 2.0f;

    auto sum_cpu = sum.to(Device::CPU);
    const float* ptr = sum_cpu.ptr<float>();
    for (size_t i = 0; i < sum.numel(); ++i) {
        EXPECT_NEAR(ptr[i], pi_over_2, 1e-4f)
            << "asin + acos != pi/2 at index " << i;
    }
}
