/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <gtest/gtest.h>
#include <numeric>
#include <torch/torch.h>

using namespace lfs::core;

// ============= Helper Functions =============

namespace {

    // Helper for comparing boolean tensors
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

    // Helper for comparing integer tensors
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
        // Handle boolean tensors specially
        if (reference.dtype() == torch::kBool) {
            compare_bool_tensors(custom, reference, msg);
            return;
        }

        // Handle integer tensors specially
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

} // anonymous namespace

class TensorConversionsShapesTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA is not available for testing";
        torch::manual_seed(42);
        Tensor::manual_seed(42);
    }
};

// ============= DataType Conversion Tests =============

TEST_F(TensorConversionsShapesTest, Float32ToInt32) {
    std::vector<float> data = {1.5f, 2.7f, 3.2f, 4.9f};

    auto custom_t = Tensor::from_vector(data, {4}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_t.to(DataType::Int32);
    auto torch_result = torch_t.to(torch::kInt32);

    EXPECT_EQ(custom_result.dtype(), DataType::Int32);
    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Float32ToInt32");
}

TEST_F(TensorConversionsShapesTest, Int32ToFloat32) {
    std::vector<int> data = {1, 2, 3, 4, 5};

    auto custom_t = Tensor::from_vector(data, {5}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    auto custom_result = custom_t.to(DataType::Float32);
    auto torch_result = torch_t.to(torch::kFloat32);

    EXPECT_EQ(custom_result.dtype(), DataType::Float32);
    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Int32ToFloat32");
}

TEST_F(TensorConversionsShapesTest, Float32ToBool) {
    std::vector<float> data = {0.0f, 1.0f, 2.0f, 0.0f, -1.0f};

    auto custom_t = Tensor::from_vector(data, {5}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_t.to(DataType::Bool);
    auto torch_result = torch_t.to(torch::kBool);

    EXPECT_EQ(custom_result.dtype(), DataType::Bool);
    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Float32ToBool");
}

TEST_F(TensorConversionsShapesTest, BoolToFloat32) {
    std::vector<bool> data = {true, false, true, false};

    auto custom_t = Tensor::from_vector(data, {4}, Device::CUDA);
    auto torch_t = torch::tensor({1, 0, 1, 0}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    auto custom_result = custom_t.to(DataType::Float32);
    auto torch_result = torch_t.to(torch::kFloat32);

    EXPECT_EQ(custom_result.dtype(), DataType::Float32);
    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "BoolToFloat32");
}

TEST_F(TensorConversionsShapesTest, ConversionPreservesShape) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

    auto custom_t = Tensor::from_vector(data, {2, 2}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 2});

    auto custom_result = custom_t.to(DataType::Int32);
    auto torch_result = torch_t.to(torch::kInt32);

    EXPECT_EQ(custom_result.shape()[0], torch_result.size(0));
    EXPECT_EQ(custom_result.shape()[1], torch_result.size(1));

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "ConversionShape");
}

TEST_F(TensorConversionsShapesTest, ConversionIdempotent) {
    auto custom_t = Tensor::ones({5}, Device::CUDA, DataType::Float32);
    auto torch_t = torch::ones({5}, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_t.to(DataType::Float32);
    auto torch_result = torch_t.to(torch::kFloat32);

    EXPECT_EQ(custom_result.dtype(), DataType::Float32);

    // Should create a copy - modifying result shouldn't affect original
    custom_result.fill_(42.0f);
    EXPECT_FLOAT_EQ(custom_t.to_vector()[0], 1.0f);
}

// ============= Shape Operations Tests =============

TEST_F(TensorConversionsShapesTest, SqueezeAll) {
    auto custom_t = Tensor::zeros({2, 1, 3, 1, 4}, Device::CUDA);
    auto torch_t = torch::zeros({2, 1, 3, 1, 4}, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_t.squeeze();
    auto torch_result = torch_t.squeeze();

    EXPECT_EQ(custom_result.ndim(), torch_result.dim());
    for (size_t i = 0; i < custom_result.ndim(); ++i) {
        EXPECT_EQ(custom_result.shape()[i], static_cast<size_t>(torch_result.size(i)));
    }
}

TEST_F(TensorConversionsShapesTest, SqueezeSpecificDim) {
    auto custom_t = Tensor::zeros({2, 1, 3, 1, 4}, Device::CUDA);
    auto torch_t = torch::zeros({2, 1, 3, 1, 4}, torch::TensorOptions().device(torch::kCUDA));

    // Squeeze dimension 1 (size 1)
    auto custom_result = custom_t.squeeze(1);
    auto torch_result = torch_t.squeeze(1);

    EXPECT_EQ(custom_result.ndim(), torch_result.dim());
    for (size_t i = 0; i < custom_result.ndim(); ++i) {
        EXPECT_EQ(custom_result.shape()[i], static_cast<size_t>(torch_result.size(i)));
    }
}

TEST_F(TensorConversionsShapesTest, SqueezeNegativeDim) {
    auto custom_t = Tensor::zeros({2, 1, 3, 1}, Device::CUDA);
    auto torch_t = torch::zeros({2, 1, 3, 1}, torch::TensorOptions().device(torch::kCUDA));

    // Squeeze last dimension (index -1) which has size 1
    auto custom_result = custom_t.squeeze(-1);
    auto torch_result = torch_t.squeeze(-1);

    EXPECT_EQ(custom_result.ndim(), torch_result.dim());
    for (size_t i = 0; i < custom_result.ndim(); ++i) {
        EXPECT_EQ(custom_result.shape()[i], static_cast<size_t>(torch_result.size(i)));
    }
}

TEST_F(TensorConversionsShapesTest, SqueezeNoDims) {
    auto custom_t = Tensor::zeros({2, 3, 4}, Device::CUDA);
    auto torch_t = torch::zeros({2, 3, 4}, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_t.squeeze();
    auto torch_result = torch_t.squeeze();

    // Should be unchanged
    EXPECT_EQ(custom_result.shape(), custom_t.shape());
    EXPECT_EQ(custom_result.ndim(), torch_result.dim());
}

TEST_F(TensorConversionsShapesTest, UnsqueezeDim) {
    auto custom_t = Tensor::zeros({2, 3}, Device::CUDA);
    auto torch_t = torch::zeros({2, 3}, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_t.unsqueeze(1);
    auto torch_result = torch_t.unsqueeze(1);

    EXPECT_EQ(custom_result.ndim(), torch_result.dim());
    for (size_t i = 0; i < custom_result.ndim(); ++i) {
        EXPECT_EQ(custom_result.shape()[i], static_cast<size_t>(torch_result.size(i)));
    }
}

TEST_F(TensorConversionsShapesTest, UnsqueezeNegativeDim) {
    auto custom_t = Tensor::zeros({2, 3}, Device::CUDA);
    auto torch_t = torch::zeros({2, 3}, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_t.unsqueeze(-1);
    auto torch_result = torch_t.unsqueeze(-1);

    EXPECT_EQ(custom_result.ndim(), torch_result.dim());
    for (size_t i = 0; i < custom_result.ndim(); ++i) {
        EXPECT_EQ(custom_result.shape()[i], static_cast<size_t>(torch_result.size(i)));
    }
}

TEST_F(TensorConversionsShapesTest, FlattenDefault) {
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 0.0f);

    auto custom_t = Tensor::from_vector(data, {2, 3, 4}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 3, 4});

    auto custom_result = custom_t.flatten();
    auto torch_result = torch_t.flatten();

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "FlattenDefault");
}

TEST_F(TensorConversionsShapesTest, FlattenPartial) {
    auto custom_t = Tensor::zeros({2, 3, 4, 5}, Device::CUDA);
    auto torch_t = torch::zeros({2, 3, 4, 5}, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_t.flatten(1, 2);
    auto torch_result = torch_t.flatten(1, 2);

    EXPECT_EQ(custom_result.ndim(), torch_result.dim());
    for (size_t i = 0; i < custom_result.ndim(); ++i) {
        EXPECT_EQ(custom_result.shape()[i], static_cast<size_t>(torch_result.size(i)));
    }
}

TEST_F(TensorConversionsShapesTest, FlattenNegativeDims) {
    auto custom_t = Tensor::zeros({2, 3, 4}, Device::CUDA);
    auto torch_t = torch::zeros({2, 3, 4}, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_t.flatten(-2, -1);
    auto torch_result = torch_t.flatten(-2, -1);

    EXPECT_EQ(custom_result.ndim(), torch_result.dim());
    for (size_t i = 0; i < custom_result.ndim(); ++i) {
        EXPECT_EQ(custom_result.shape()[i], static_cast<size_t>(torch_result.size(i)));
    }
}

// ============= Reshape Tests =============

TEST_F(TensorConversionsShapesTest, ReshapeBasic) {
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 0.0f);

    auto custom_t = Tensor::from_vector(data, {24}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_t.reshape({2, 3, 4});
    auto torch_result = torch_t.reshape({2, 3, 4});

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "ReshapeBasic");
}

TEST_F(TensorConversionsShapesTest, ReshapeWithInference) {
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 0.0f);

    auto custom_t = Tensor::from_vector(data, {24}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_t.reshape({-1, 6});
    auto torch_result = torch_t.reshape({-1, 6});

    EXPECT_EQ(custom_result.shape()[0], static_cast<size_t>(torch_result.size(0)));
    EXPECT_EQ(custom_result.shape()[1], static_cast<size_t>(torch_result.size(1)));

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "ReshapeInference");
}

TEST_F(TensorConversionsShapesTest, ReshapeInvalidSize) {
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 0.0f);

    auto custom_t = Tensor::from_vector(data, {24}, Device::CUDA);

    // Try to reshape 24 elements to 25 elements
    auto custom_result = custom_t.reshape({5, 5});

    EXPECT_FALSE(custom_result.is_valid());
}

TEST_F(TensorConversionsShapesTest, ViewAlias) {
    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 0.0f);

    auto custom_t = Tensor::from_vector(data, {12}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_t.view({3, 4});
    auto torch_result = torch_t.view({3, 4});

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "View");
}

// ============= Permute and Transpose Tests =============

TEST_F(TensorConversionsShapesTest, Permute) {
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 0.0f);

    auto custom_t = Tensor::from_vector(data, {2, 3, 4}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 3, 4});

    std::vector<int> perm = {2, 0, 1};
    auto custom_result = custom_t.permute(perm);
    auto torch_result = torch_t.permute({2, 0, 1});

    EXPECT_EQ(custom_result.ndim(), torch_result.dim());
    for (size_t i = 0; i < custom_result.ndim(); ++i) {
        EXPECT_EQ(custom_result.shape()[i], static_cast<size_t>(torch_result.size(i)));
    }

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Permute");
}

TEST_F(TensorConversionsShapesTest, Transpose2D) {
    std::vector<float> data = {1, 2, 3, 4, 5, 6};

    auto custom_t = Tensor::from_vector(data, {2, 3}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 3});

    auto custom_result = custom_t.t();
    auto torch_result = torch_t.t();

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Transpose2D");
}

TEST_F(TensorConversionsShapesTest, TransposeSpecificDims) {
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 0.0f);

    auto custom_t = Tensor::from_vector(data, {2, 3, 4}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 3, 4});

    auto custom_result = custom_t.transpose(0, 2);
    auto torch_result = torch_t.transpose(0, 2);

    EXPECT_EQ(custom_result.ndim(), torch_result.dim());
    for (size_t i = 0; i < custom_result.ndim(); ++i) {
        EXPECT_EQ(custom_result.shape()[i], static_cast<size_t>(torch_result.size(i)));
    }

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "TransposeSpecific");
}

// ============= Concatenation Tests =============

// ============= Concatenation Tests =============

TEST_F(TensorConversionsShapesTest, CatBasic) {
    std::vector<float> data_a(12, 1.0f);
    std::vector<float> data_b(8, 2.0f);

    auto custom_a = Tensor::from_vector(data_a, {3, 4}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {2, 4}, Device::CUDA);

    auto torch_a = torch::ones({3, 4}, torch::TensorOptions().device(torch::kCUDA));
    auto torch_b = torch::ones({2, 4}, torch::TensorOptions().device(torch::kCUDA)) * 2.0f;

    // Build vector manually with move semantics
    std::vector<Tensor> custom_tensors;
    custom_tensors.push_back(std::move(custom_a));
    custom_tensors.push_back(std::move(custom_b));

    auto custom_result = Tensor::cat(custom_tensors, 0);
    auto torch_result = torch::cat({torch_a, torch_b}, 0);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "CatBasic");
}

TEST_F(TensorConversionsShapesTest, CatAlongDim1) {
    std::vector<float> data_a(12, 1.0f);
    std::vector<float> data_b(8, 2.0f);

    auto custom_a = Tensor::from_vector(data_a, {4, 3}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {4, 2}, Device::CUDA);

    auto torch_a = torch::ones({4, 3}, torch::TensorOptions().device(torch::kCUDA));
    auto torch_b = torch::ones({4, 2}, torch::TensorOptions().device(torch::kCUDA)) * 2.0f;

    // Build vector manually with move semantics
    std::vector<Tensor> custom_tensors;
    custom_tensors.push_back(std::move(custom_a));
    custom_tensors.push_back(std::move(custom_b));

    auto custom_result = Tensor::cat(custom_tensors, 1);
    auto torch_result = torch::cat({torch_a, torch_b}, 1);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "CatDim1");
}

TEST_F(TensorConversionsShapesTest, CatDim1Debug) {
    auto a = Tensor::ones({2, 3}, Device::CUDA);
    auto b = Tensor::ones({2, 2}, Device::CUDA) * 2.0f;

    std::vector<Tensor> tensors;
    tensors.push_back(std::move(a));
    tensors.push_back(std::move(b));

    auto result = Tensor::cat(tensors, 1);

    EXPECT_TRUE(result.is_valid()) << "Cat returned invalid tensor!";
    EXPECT_EQ(result.shape(), TensorShape({2, 5})) << "Wrong output shape";

    if (result.is_valid()) {
        auto values = result.to_vector();
        // First row: [1, 1, 1, 2, 2]
        EXPECT_FLOAT_EQ(values[0], 1.0f);
        EXPECT_FLOAT_EQ(values[1], 1.0f);
        EXPECT_FLOAT_EQ(values[2], 1.0f);
        EXPECT_FLOAT_EQ(values[3], 2.0f);
        EXPECT_FLOAT_EQ(values[4], 2.0f);
    }
}

TEST_F(TensorConversionsShapesTest, StackBasic) {
    std::vector<float> data_a(12, 1.0f);
    std::vector<float> data_b(12, 2.0f);

    auto custom_a = Tensor::from_vector(data_a, {3, 4}, Device::CUDA);
    auto custom_b = Tensor::from_vector(data_b, {3, 4}, Device::CUDA);

    auto torch_a = torch::ones({3, 4}, torch::TensorOptions().device(torch::kCUDA));
    auto torch_b = torch::ones({3, 4}, torch::TensorOptions().device(torch::kCUDA)) * 2.0f;

    // Build vector manually with move semantics
    std::vector<Tensor> custom_tensors;
    custom_tensors.push_back(std::move(custom_a));
    custom_tensors.push_back(std::move(custom_b));

    auto custom_result = Tensor::stack(custom_tensors, 0);
    auto torch_result = torch::stack({torch_a, torch_b}, 0);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "StackBasic");
}

// ============= Clone and Copy Tests =============

TEST_F(TensorConversionsShapesTest, CloneBasic) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

    auto custom_t = Tensor::from_vector(data, {4}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_clone = custom_t.clone();
    auto torch_clone = torch_t.clone();

    compare_tensors(custom_clone, torch_clone, 1e-6f, 1e-7f, "Clone");

    // Verify independence
    custom_clone.fill_(42.0f);
    EXPECT_FLOAT_EQ(custom_t.to_vector()[0], 1.0f);
}

TEST_F(TensorConversionsShapesTest, ClonePreservesProperties) {
    auto custom_t = Tensor::ones({3, 4}, Device::CUDA, DataType::Float32);
    auto torch_t = torch::ones({3, 4}, torch::TensorOptions().device(torch::kCUDA));

    auto custom_clone = custom_t.clone();
    auto torch_clone = torch_t.clone();

    EXPECT_EQ(custom_clone.shape(), custom_t.shape());
    EXPECT_EQ(custom_clone.device(), custom_t.device());
    EXPECT_EQ(custom_clone.dtype(), custom_t.dtype());
}

TEST_F(TensorConversionsShapesTest, Contiguous) {
    std::vector<float> data(12);
    std::iota(data.begin(), data.end(), 0.0f);

    auto custom_t = Tensor::from_vector(data, {3, 4}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({3, 4});

    // Transpose makes it non-contiguous
    auto custom_transposed = custom_t.t();
    auto torch_transposed = torch_t.t();

    // Make contiguous
    auto custom_contig = custom_transposed.contiguous();
    auto torch_contig = torch_transposed.contiguous();

    compare_tensors(custom_contig, torch_contig, 1e-6f, 1e-7f, "Contiguous");
}

// ============= Slice Tests =============

TEST_F(TensorConversionsShapesTest, SliceBasic) {
    std::vector<float> data(80);
    std::iota(data.begin(), data.end(), 0.0f);

    auto custom_t = Tensor::from_vector(data, {10, 8}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({10, 8});

    auto custom_result = custom_t.slice(0, 2, 7);
    auto torch_result = torch_t.slice(0, 2, 7);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Slice");
}

TEST_F(TensorConversionsShapesTest, SliceMultipleDims) {
    std::vector<float> data(120);
    std::iota(data.begin(), data.end(), 0.0f);

    auto custom_t = Tensor::from_vector(data, {5, 6, 4}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({5, 6, 4});

    std::vector<std::pair<int, int>> ranges = {{1, 4}, {2, 5}, {0, 3}};
    auto custom_result = custom_t.slice(ranges);
    auto torch_result = torch_t.slice(0, 1, 4).slice(1, 2, 5).slice(2, 0, 3);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "SliceMulti");
}

// ============= Expand and Broadcast Tests =============

TEST_F(TensorConversionsShapesTest, ExpandBasic) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f};

    auto custom_t = Tensor::from_vector(data, {1, 3}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({1, 3});

    std::vector<int> target_shape = {4, 3};
    auto custom_result = custom_t.expand(target_shape);
    auto torch_result = torch_t.expand({4, 3});

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "Expand");
}

TEST_F(TensorConversionsShapesTest, BroadcastTo) {
    std::vector<float> data = {1.0f, 2.0f};

    auto custom_t = Tensor::from_vector(data, {2, 1}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA)).reshape({2, 1});

    auto custom_result = custom_t.broadcast_to({2, 5});
    auto torch_result = torch_t.broadcast_to({2, 5});

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "BroadcastTo");
}

// ============= Index Select Tests =============

TEST_F(TensorConversionsShapesTest, IndexSelectBasic) {
    std::vector<float> data = {0, 1, 2, 3, 4};

    auto custom_t = Tensor::from_vector(data, {5}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    std::vector<int> indices_data = {0, 2, 4};
    auto custom_indices = Tensor::from_vector(indices_data, {3}, Device::CUDA);
    auto torch_indices = torch::tensor({0, 2, 4}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

    auto custom_result = custom_t.index_select(0, custom_indices);
    auto torch_result = torch::index_select(torch_t, 0, torch_indices);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "IndexSelect");
}

// ============= Integration Tests =============

TEST_F(TensorConversionsShapesTest, ConversionChain) {
    std::vector<float> data = {1.5f, 2.7f, 3.2f};

    auto custom_t = Tensor::from_vector(data, {3}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_t.to(DataType::Int32).to(DataType::Float32);
    auto torch_result = torch_t.to(torch::kInt32).to(torch::kFloat32);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "ConversionChain");
}

TEST_F(TensorConversionsShapesTest, ShapeManipulationChain) {
    std::vector<float> data(24);
    std::iota(data.begin(), data.end(), 0.0f);

    auto custom_t = Tensor::from_vector(data, {24}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_t.reshape({2, 3, 4})
                             .squeeze()
                             .flatten(0, 1)
                             .unsqueeze(0);

    auto torch_result = torch_t.reshape({2, 3, 4})
                            .squeeze()
                            .flatten(0, 1)
                            .unsqueeze(0);

    EXPECT_EQ(custom_result.ndim(), torch_result.dim());
    for (size_t i = 0; i < custom_result.ndim(); ++i) {
        EXPECT_EQ(custom_result.shape()[i], static_cast<size_t>(torch_result.size(i)));
    }

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "ShapeChain");
}

TEST_F(TensorConversionsShapesTest, ComplexReshapeAndTranspose) {
    std::vector<float> data(120);
    std::iota(data.begin(), data.end(), 0.0f);

    auto custom_t = Tensor::from_vector(data, {120}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::TensorOptions().device(torch::kCUDA));

    auto custom_result = custom_t.reshape({2, 3, 4, 5})
                             .transpose(1, 3)
                             .flatten(0, 1);

    auto torch_result = torch_t.reshape({2, 3, 4, 5})
                            .transpose(1, 3)
                            .flatten(0, 1);

    compare_tensors(custom_result, torch_result, 1e-6f, 1e-7f, "ComplexReshapeTranspose");
}

// ============= Device Transfer Tests =============

TEST_F(TensorConversionsShapesTest, DeviceTransfer) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

    auto custom_cpu = Tensor::from_vector(data, {4}, Device::CPU);
    auto torch_cpu = torch::tensor(data, torch::TensorOptions().device(torch::kCPU));

    // CPU to CUDA
    auto custom_cuda = custom_cpu.to(Device::CUDA);
    auto torch_cuda = torch_cpu.to(torch::kCUDA);

    EXPECT_EQ(custom_cuda.device(), Device::CUDA);
    compare_tensors(custom_cuda, torch_cuda, 1e-6f, 1e-7f, "CPUtoCUDA");

    // CUDA to CPU
    auto custom_back = custom_cuda.to(Device::CPU);
    auto torch_back = torch_cuda.to(torch::kCPU);

    EXPECT_EQ(custom_back.device(), Device::CPU);
    compare_tensors(custom_back, torch_back, 1e-6f, 1e-7f, "CUDAtoCPU");
}

TEST_F(TensorConversionsShapesTest, CudaCpuAliases) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f};

    auto custom_cpu = Tensor::from_vector(data, {3}, Device::CPU);
    auto torch_cpu = torch::tensor(data, torch::TensorOptions().device(torch::kCPU));

    // Test .cuda() alias
    auto custom_cuda = custom_cpu.cuda();
    auto torch_cuda = torch_cpu.cuda();

    compare_tensors(custom_cuda, torch_cuda, 1e-6f, 1e-7f, "CudaAlias");

    // Test .cpu() alias
    auto custom_back = custom_cuda.cpu();
    auto torch_back = torch_cuda.cpu();

    compare_tensors(custom_back, torch_back, 1e-6f, 1e-7f, "CpuAlias");
}