/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <vector>

using namespace lfs::core;

// ============= Helper Functions =============

namespace {

    torch::Tensor to_torch(const Tensor& t) {
        auto options = torch::TensorOptions()
                           .dtype([&]() {
                               switch (t.dtype()) {
                               case DataType::Float32: return torch::kFloat32;
                               case DataType::Float16: return torch::kFloat16;
                               case DataType::Int32: return torch::kInt32;
                               case DataType::Int64: return torch::kInt64;
                               case DataType::UInt8: return torch::kUInt8;
                               case DataType::Bool: return torch::kBool;
                               default: return torch::kFloat32;
                               }
                           }())
                           .device(t.device() == Device::CPU ? torch::kCPU : torch::kCUDA);

        std::vector<int64_t> shape;
        for (size_t i = 0; i < t.ndim(); ++i) {
            shape.push_back(static_cast<int64_t>(t.size(i)));
        }

        torch::Tensor result = torch::empty(shape, options);

        if (t.device() == Device::CPU) {
            std::memcpy(result.data_ptr(), t.raw_ptr(), t.bytes());
        } else {
            cudaMemcpy(result.data_ptr(), t.raw_ptr(), t.bytes(), cudaMemcpyDeviceToDevice);
        }

        return result;
    }

    Tensor from_torch(const torch::Tensor& t, Device device = Device::CPU) {
        auto t_cont = t.contiguous();

        DataType dtype;
        switch (t_cont.scalar_type()) {
        case torch::kFloat32: dtype = DataType::Float32; break;
        case torch::kFloat16: dtype = DataType::Float16; break;
        case torch::kInt32: dtype = DataType::Int32; break;
        case torch::kInt64: dtype = DataType::Int64; break;
        case torch::kUInt8: dtype = DataType::UInt8; break;
        case torch::kBool: dtype = DataType::Bool; break;
        default: dtype = DataType::Float32; break;
        }

        std::vector<size_t> shape;
        for (int64_t i = 0; i < t_cont.dim(); ++i) {
            shape.push_back(static_cast<size_t>(t_cont.size(i)));
        }

        Tensor result = Tensor::empty(TensorShape(shape), device, dtype);

        if (device == Device::CPU) {
            std::memcpy(result.raw_ptr(), t_cont.data_ptr(), result.bytes());
        } else {
            if (t_cont.is_cpu()) {
                cudaMemcpy(result.raw_ptr(), t_cont.data_ptr(), result.bytes(), cudaMemcpyHostToDevice);
            } else {
                cudaMemcpy(result.raw_ptr(), t_cont.data_ptr(), result.bytes(), cudaMemcpyDeviceToDevice);
            }
        }

        return result;
    }

    void compare_tensors(const Tensor& custom, const torch::Tensor& reference,
                         float rtol = 1e-5f, float atol = 1e-7f, const std::string& msg = "") {
        auto ref_cpu = reference.cpu().contiguous();
        auto custom_cpu = custom.cpu();

        ASSERT_EQ(custom_cpu.ndim(), ref_cpu.dim()) << msg << ": Rank mismatch";

        for (size_t i = 0; i < custom_cpu.ndim(); ++i) {
            ASSERT_EQ(custom_cpu.size(i), static_cast<size_t>(ref_cpu.size(i)))
                << msg << ": Shape mismatch at dim " << i;
        }

        ASSERT_EQ(custom_cpu.numel(), static_cast<size_t>(ref_cpu.numel()))
            << msg << ": Element count mismatch";

        auto custom_vec = custom_cpu.to_vector();
        auto ref_ptr = ref_cpu.data_ptr<float>();
        for (size_t i = 0; i < custom_vec.size(); ++i) {
            if (std::isnan(ref_ptr[i])) {
                EXPECT_TRUE(std::isnan(custom_vec[i])) << msg << ": Expected NaN at index " << i;
            } else if (std::isinf(ref_ptr[i])) {
                EXPECT_TRUE(std::isinf(custom_vec[i])) << msg << ": Expected Inf at index " << i;
            } else {
                float diff = std::abs(custom_vec[i] - ref_ptr[i]);
                float threshold = atol + rtol * std::abs(ref_ptr[i]);
                EXPECT_LE(diff, threshold)
                    << msg << ": Mismatch at index " << i
                    << " (custom=" << custom_vec[i] << ", ref=" << ref_ptr[i] << ")";
            }
        }
    }

    // Create test tensor with sequential values for both implementations
    std::pair<Tensor, torch::Tensor> create_test_tensors(const std::vector<int64_t>& shape, Device device = Device::CUDA) {
        size_t total = 1;
        for (auto dim : shape) {
            total *= dim;
        }

        std::vector<float> data(total);
        for (size_t i = 0; i < total; ++i) {
            data[i] = static_cast<float>(i);
        }

        std::vector<size_t> shape_custom;
        for (auto s : shape) {
            shape_custom.push_back(static_cast<size_t>(s));
        }

        auto tensor_custom = Tensor::from_vector(data, TensorShape(shape_custom), device);
        auto tensor_torch = torch::tensor(data, device == Device::CUDA ? torch::kCUDA : torch::kCPU)
                                .reshape(shape);

        return {std::move(tensor_custom), tensor_torch};
    }

} // anonymous namespace

class TensorViewTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(42);
        Tensor::manual_seed(42);
    }
};

// ============= Basic View Tests =============

TEST_F(TensorViewTest, BasicView) {
    auto [tensor_custom, tensor_torch] = create_test_tensors({2, 3, 4});

    // Reshape to different valid shape
    auto view1_custom = tensor_custom.view({6, 4});
    auto view1_torch = tensor_torch.view({6, 4});

    compare_tensors(view1_custom, view1_torch, 1e-5f, 1e-7f, "BasicView1");
    EXPECT_EQ(view1_custom.numel(), tensor_custom.numel());
    EXPECT_FALSE(view1_custom.owns_memory());

    // Another reshape
    auto view2_custom = tensor_custom.view({24});
    auto view2_torch = tensor_torch.view({24});

    compare_tensors(view2_custom, view2_torch, 1e-5f, 1e-7f, "BasicView2");

    // Reshape to original
    auto view3_custom = tensor_custom.view({2, 3, 4});
    auto view3_torch = tensor_torch.view({2, 3, 4});

    compare_tensors(view3_custom, view3_torch, 1e-5f, 1e-7f, "BasicView3");
}

TEST_F(TensorViewTest, ViewWithInferredDimension) {
    auto [tensor_custom, tensor_torch] = create_test_tensors({2, 3, 4});

    // Use -1 to infer dimension
    auto view1_custom = tensor_custom.view({-1, 4});
    auto view1_torch = tensor_torch.view({-1, 4});
    compare_tensors(view1_custom, view1_torch, 1e-5f, 1e-7f, "ViewInferred1");

    auto view2_custom = tensor_custom.view({2, -1});
    auto view2_torch = tensor_torch.view({2, -1});
    compare_tensors(view2_custom, view2_torch, 1e-5f, 1e-7f, "ViewInferred2");

    auto view3_custom = tensor_custom.view({-1});
    auto view3_torch = tensor_torch.view({-1});
    compare_tensors(view3_custom, view3_torch, 1e-5f, 1e-7f, "ViewInferred3");
}

TEST_F(TensorViewTest, InvalidView) {
    auto [tensor_custom, tensor_torch] = create_test_tensors({2, 3, 4});

    // Wrong number of elements
    auto invalid_custom = tensor_custom.view({5, 5});
    EXPECT_FALSE(invalid_custom.is_valid());

    // PyTorch throws exception
    EXPECT_THROW(tensor_torch.view({5, 5}), std::exception);

    // Multiple -1
    auto invalid2_custom = tensor_custom.view({-1, -1});
    EXPECT_FALSE(invalid2_custom.is_valid());

    EXPECT_THROW(tensor_torch.view({-1, -1}), std::exception);
}

TEST_F(TensorViewTest, Reshape) {
    auto [tensor_custom, tensor_torch] = create_test_tensors({3, 4});

    // reshape is alias for view
    auto reshaped_custom = tensor_custom.reshape({2, 6});
    auto reshaped_torch = tensor_torch.reshape({2, 6});

    compare_tensors(reshaped_custom, reshaped_torch, 1e-5f, 1e-7f, "Reshape");
}

// ============= Slice Tests =============

TEST_F(TensorViewTest, SliceBasic) {
    auto [tensor_custom, tensor_torch] = create_test_tensors({10, 5});

    // Slice along dimension 0
    auto slice1_custom = tensor_custom.slice(0, 2, 5);
    auto slice1_torch = tensor_torch.slice(0, 2, 5);
    compare_tensors(slice1_custom, slice1_torch, 1e-5f, 1e-7f, "Slice1");
    EXPECT_FALSE(slice1_custom.owns_memory());

    // Slice along dimension 1
    auto slice2_custom = tensor_custom.slice(1, 1, 4);
    auto slice2_torch = tensor_torch.slice(1, 1, 4);
    compare_tensors(slice2_custom, slice2_torch, 1e-5f, 1e-7f, "Slice2");
}

TEST_F(TensorViewTest, SliceWithPairs) {
    auto [tensor_custom, tensor_torch] = create_test_tensors({10, 5, 3});

    // Slice using pairs API
    auto slice1_custom = tensor_custom.slice({{2, 5}, {1, 4}});
    auto slice1_torch = tensor_torch.slice(0, 2, 5).slice(1, 1, 4);
    compare_tensors(slice1_custom, slice1_torch, 1e-5f, 1e-7f, "SlicePairs1");

    // Slice with negative indices
    auto slice2_custom = tensor_custom.slice({{0, -1}, {-2, -1}});
    auto slice2_torch = tensor_torch.slice(0, 0, -1).slice(1, -2, -1);
    compare_tensors(slice2_custom, slice2_torch, 1e-5f, 1e-7f, "SlicePairs2");

    // Partial slice
    auto slice3_custom = tensor_custom.slice({{2, 8}});
    auto slice3_torch = tensor_torch.slice(0, 2, 8);
    compare_tensors(slice3_custom, slice3_torch, 1e-5f, 1e-7f, "SlicePairs3");
}

TEST_F(TensorViewTest, InvalidSlice) {
    auto [tensor_custom, tensor_torch] = create_test_tensors({10, 5});

    // Out of range dimension
    auto invalid1_custom = tensor_custom.slice(2, 0, 1);
    EXPECT_FALSE(invalid1_custom.is_valid());

    // Invalid range (start > end)
    auto invalid2_custom = tensor_custom.slice(0, 5, 3);
    EXPECT_FALSE(invalid2_custom.is_valid());

    // end > size - PyTorch allows this and clamps
    auto slice_custom = tensor_custom.slice(0, 0, 11);
    auto slice_torch = tensor_torch.slice(0, 0, 11); // PyTorch clamps to 10

    // Either should fail or clamp to size
    if (slice_custom.is_valid()) {
        compare_tensors(slice_custom, slice_torch, 1e-5f, 1e-7f, "SliceClamp");
    }
}

// ============= Squeeze/Unsqueeze Tests =============

TEST_F(TensorViewTest, SqueezeAll) {
    auto [tensor_custom, tensor_torch] = create_test_tensors({1, 3, 1, 4});

    // Squeeze all dimensions of size 1
    auto squeezed_custom = tensor_custom.squeeze();
    auto squeezed_torch = tensor_torch.squeeze();
    compare_tensors(squeezed_custom, squeezed_torch, 1e-5f, 1e-7f, "SqueezeAll");
}

TEST_F(TensorViewTest, SqueezeSpecific) {
    auto [tensor_custom, tensor_torch] = create_test_tensors({1, 3, 1, 4});

    // Squeeze specific dimension
    auto squeezed0_custom = tensor_custom.squeeze(0);
    auto squeezed0_torch = tensor_torch.squeeze(0);
    compare_tensors(squeezed0_custom, squeezed0_torch, 1e-5f, 1e-7f, "Squeeze0");

    auto squeezed2_custom = tensor_custom.squeeze(2);
    auto squeezed2_torch = tensor_torch.squeeze(2);
    compare_tensors(squeezed2_custom, squeezed2_torch, 1e-5f, 1e-7f, "Squeeze2");

    // Squeeze with negative index
    auto squeezed_neg_custom = tensor_custom.squeeze(-2);
    auto squeezed_neg_torch = tensor_torch.squeeze(-2);
    compare_tensors(squeezed_neg_custom, squeezed_neg_torch, 1e-5f, 1e-7f, "SqueezeNeg");

    // Try to squeeze non-1 dimension (should return copy/view unchanged)
    auto not_squeezed_custom = tensor_custom.squeeze(1);
    auto not_squeezed_torch = tensor_torch.squeeze(1);
    compare_tensors(not_squeezed_custom, not_squeezed_torch, 1e-5f, 1e-7f, "SqueezeNoOp");
}

TEST_F(TensorViewTest, Unsqueeze) {
    auto [tensor_custom, tensor_torch] = create_test_tensors({3, 4});

    // Add dimension at position 0
    auto unsqueezed0_custom = tensor_custom.unsqueeze(0);
    auto unsqueezed0_torch = tensor_torch.unsqueeze(0);
    compare_tensors(unsqueezed0_custom, unsqueezed0_torch, 1e-5f, 1e-7f, "Unsqueeze0");

    // Add dimension at position 1
    auto unsqueezed1_custom = tensor_custom.unsqueeze(1);
    auto unsqueezed1_torch = tensor_torch.unsqueeze(1);
    compare_tensors(unsqueezed1_custom, unsqueezed1_torch, 1e-5f, 1e-7f, "Unsqueeze1");

    // Add dimension at end
    auto unsqueezed2_custom = tensor_custom.unsqueeze(2);
    auto unsqueezed2_torch = tensor_torch.unsqueeze(2);
    compare_tensors(unsqueezed2_custom, unsqueezed2_torch, 1e-5f, 1e-7f, "Unsqueeze2");

    // Negative index
    auto unsqueezed_neg_custom = tensor_custom.unsqueeze(-1);
    auto unsqueezed_neg_torch = tensor_torch.unsqueeze(-1);
    compare_tensors(unsqueezed_neg_custom, unsqueezed_neg_torch, 1e-5f, 1e-7f, "UnsqueezeNeg");
}

// ============= Flatten Tests =============

TEST_F(TensorViewTest, FlattenAll) {
    auto [tensor_custom, tensor_torch] = create_test_tensors({2, 3, 4, 5});

    // Flatten all dimensions
    auto flat_custom = tensor_custom.flatten();
    auto flat_torch = tensor_torch.flatten();
    compare_tensors(flat_custom, flat_torch, 1e-5f, 1e-7f, "FlattenAll");
}

TEST_F(TensorViewTest, FlattenPartial) {
    auto [tensor_custom, tensor_torch] = create_test_tensors({2, 3, 4, 5});

    // Flatten from dimension 1 to 2
    auto flat_custom = tensor_custom.flatten(1, 2);
    auto flat_torch = tensor_torch.flatten(1, 2);
    compare_tensors(flat_custom, flat_torch, 1e-5f, 1e-7f, "FlattenPartial");

    // Flatten with negative indices
    auto flat_neg_custom = tensor_custom.flatten(-2, -1);
    auto flat_neg_torch = tensor_torch.flatten(-2, -1);
    compare_tensors(flat_neg_custom, flat_neg_torch, 1e-5f, 1e-7f, "FlattenNeg");

    // Flatten single dimension (should not change)
    auto flat_single_custom = tensor_custom.flatten(1, 1);
    auto flat_single_torch = tensor_torch.flatten(1, 1);
    compare_tensors(flat_single_custom, flat_single_torch, 1e-5f, 1e-7f, "FlattenSingle");
}

// ============= Expand Tests =============

TEST_F(TensorViewTest, Expand) {
    auto [tensor_custom, tensor_torch] = create_test_tensors({1, 3, 1});

    // Expand dimensions of size 1
    auto expanded_custom = tensor_custom.expand({2, 3, 4});
    auto expanded_torch = tensor_torch.expand({2, 3, 4});
    compare_tensors(expanded_custom, expanded_torch, 1e-5f, 1e-7f, "Expand");

    // Use -1 to keep original dimension
    auto expanded2_custom = tensor_custom.expand({5, -1, 3});
    auto expanded2_torch = tensor_torch.expand({5, 3, 3}); // PyTorch doesn't use -1
    compare_tensors(expanded2_custom, expanded2_torch, 1e-5f, 1e-7f, "ExpandWith-1");
}

TEST_F(TensorViewTest, ExpandAddDims) {
    auto [tensor_custom, tensor_torch] = create_test_tensors({3, 1});

    // Expand to more dimensions
    auto expanded_custom = tensor_custom.expand({2, 3, 4});
    auto expanded_torch = tensor_torch.expand({2, 3, 4});
    compare_tensors(expanded_custom, expanded_torch, 1e-5f, 1e-7f, "ExpandAddDims");
}

TEST_F(TensorViewTest, ExpandErrors) {
    auto [tensor_custom, tensor_torch] = create_test_tensors({3, 4});

    // Cannot expand non-singleton dimension to different size
    auto invalid_custom = tensor_custom.expand({3, 5});
    EXPECT_FALSE(invalid_custom.is_valid());
    EXPECT_THROW(tensor_torch.expand({3, 5}), std::exception);

    // Valid expand from singleton
    auto [tensor2_custom, tensor2_torch] = create_test_tensors({3, 1});
    auto valid_custom = tensor2_custom.expand({3, 5});
    auto valid_torch = tensor2_torch.expand({3, 5});
    compare_tensors(valid_custom, valid_torch, 1e-5f, 1e-7f, "ExpandValid");
}

// ============= Transpose/Permute Tests =============

TEST_F(TensorViewTest, TransposeBasic) {
    auto [tensor_custom, tensor_torch] = create_test_tensors({3, 4, 5});

    // Transpose dimensions 0 and 1
    auto transposed_custom = tensor_custom.transpose(0, 1);
    auto transposed_torch = tensor_torch.transpose(0, 1);
    compare_tensors(transposed_custom, transposed_torch, 1e-5f, 1e-7f, "Transpose");

    EXPECT_TRUE(transposed_custom.is_valid());
    EXPECT_EQ(transposed_custom.numel(), tensor_custom.numel());
}

TEST_F(TensorViewTest, PermuteBasic) {
    auto [tensor_custom, tensor_torch] = create_test_tensors({2, 3, 4});

    // Permute dimensions
    auto permuted_custom = tensor_custom.permute({2, 0, 1});
    auto permuted_torch = tensor_torch.permute({2, 0, 1});
    compare_tensors(permuted_custom, permuted_torch, 1e-5f, 1e-7f, "Permute");

    EXPECT_TRUE(permuted_custom.is_valid());
    EXPECT_EQ(permuted_custom.numel(), tensor_custom.numel());
}

TEST_F(TensorViewTest, TFunction) {
    // Test the .t() function for 2D tensors
    auto [tensor2d_custom, tensor2d_torch] = create_test_tensors({3, 4});
    auto transposed_custom = tensor2d_custom.t();
    auto transposed_torch = tensor2d_torch.t();
    compare_tensors(transposed_custom, transposed_torch, 1e-5f, 1e-7f, "T2D");

    // For 1D tensor, t() should be no-op
    auto [tensor1d_custom, tensor1d_torch] = create_test_tensors({5});
    auto transposed1d_custom = tensor1d_custom.t();
    auto transposed1d_torch = tensor1d_torch.t();
    compare_tensors(transposed1d_custom, transposed1d_torch, 1e-5f, 1e-7f, "T1D");

    // For 3D+ tensor behavior may differ - check your implementation
    auto [tensor3d_custom, tensor3d_torch] = create_test_tensors({2, 3, 4});
    auto transposed3d_custom = tensor3d_custom.t();
    // PyTorch .t() only works for <= 2D, so we test transpose(-2, -1) instead
    auto transposed3d_torch = tensor3d_torch.transpose(-2, -1);
    compare_tensors(transposed3d_custom, transposed3d_torch, 1e-5f, 1e-7f, "T3D");
}

// ============= Chained Operations =============

TEST_F(TensorViewTest, ChainedViewOperations) {
    auto [tensor_custom, tensor_torch] = create_test_tensors({2, 3, 4});

    // Chain multiple view operations
    auto result_custom = tensor_custom.view({6, 4})
                             .unsqueeze(0)
                             .squeeze(0)
                             .reshape({2, 12})
                             .flatten();

    auto result_torch = tensor_torch.view({6, 4})
                            .unsqueeze(0)
                            .squeeze(0)
                            .reshape({2, 12})
                            .flatten();

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "ChainedOps");
    EXPECT_FALSE(result_custom.owns_memory());
}

// ============= Memory Sharing Tests =============

TEST_F(TensorViewTest, ViewsShareMemory) {
    auto [tensor_custom, tensor_torch] = create_test_tensors({4, 4});

    auto view_custom = tensor_custom.view({2, 8});
    auto view_torch = tensor_torch.view({2, 8});

    // Modify view
    view_custom.fill_(1.0f);
    view_torch.fill_(1.0f);

    // Original should be modified
    compare_tensors(tensor_custom, tensor_torch, 1e-5f, 1e-7f, "ViewShareMemory");
}

TEST_F(TensorViewTest, SliceSharesMemory) {
    auto [tensor_custom, tensor_torch] = create_test_tensors({5, 5});

    auto slice_custom = tensor_custom.slice(0, 1, 3);
    auto slice_torch = tensor_torch.slice(0, 1, 3);

    // Modify slice
    slice_custom.fill_(99.0f);
    slice_torch.fill_(99.0f);

    // Check original tensor
    compare_tensors(tensor_custom, tensor_torch, 1e-5f, 1e-7f, "SliceShareMemory");
}

// ============= Complex Reshaping =============

TEST_F(TensorViewTest, ComplexReshaping) {
    auto [tensor_custom, tensor_torch] = create_test_tensors({2, 3, 4, 5});

    // Multiple valid reshapes
    std::vector<std::vector<int64_t>> valid_shapes = {
        {120},
        {2, 60},
        {6, 20},
        {2, 3, 20},
        {24, 5},
        {2, 12, 5},
        {2, 3, 2, 10}};

    for (const auto& shape : valid_shapes) {
        auto view_custom = tensor_custom.view(std::vector<int>(shape.begin(), shape.end()));
        auto view_torch = tensor_torch.view(shape);

        EXPECT_TRUE(view_custom.is_valid());
        compare_tensors(view_custom, view_torch, 1e-5f, 1e-7f,
                        "ComplexReshape_" + std::to_string(shape[0]));
        EXPECT_EQ(view_custom.numel(), tensor_custom.numel());
    }
}

// ============= CPU Tensor Tests =============

TEST_F(TensorViewTest, ViewOnCPUTensor) {
    auto [cpu_tensor_custom, cpu_tensor_torch] = create_test_tensors({3, 4}, Device::CPU);

    // View operations should work on CPU tensors
    auto view_custom = cpu_tensor_custom.view({12});
    auto view_torch = cpu_tensor_torch.view({12});
    compare_tensors(view_custom, view_torch, 1e-5f, 1e-7f, "CPUView");

    EXPECT_EQ(view_custom.device(), Device::CPU);
    EXPECT_FALSE(view_custom.owns_memory());

    auto slice_custom = cpu_tensor_custom.slice(0, 1, 3);
    auto slice_torch = cpu_tensor_torch.slice(0, 1, 3);
    compare_tensors(slice_custom, slice_torch, 1e-5f, 1e-7f, "CPUSlice");

    EXPECT_EQ(slice_custom.device(), Device::CPU);
}

// ============= Edge Cases =============

TEST_F(TensorViewTest, EmptyTensor) {
    auto empty_custom = Tensor::empty({0}, Device::CUDA);
    auto empty_torch = torch::empty({0}, torch::kCUDA);

    auto view_custom = empty_custom.view({0});
    auto view_torch = empty_torch.view({0});

    EXPECT_TRUE(view_custom.is_valid());
    EXPECT_EQ(view_custom.numel(), 0);
    EXPECT_EQ(view_torch.numel(), 0);
}

TEST_F(TensorViewTest, SingleElement) {
    auto [single_custom, single_torch] = create_test_tensors({1});

    auto view_custom = single_custom.view({1, 1, 1});
    auto view_torch = single_torch.view({1, 1, 1});
    compare_tensors(view_custom, view_torch, 1e-5f, 1e-7f, "SingleElement");
}

TEST_F(TensorViewTest, LargeTensor) {
    auto [large_custom, large_torch] = create_test_tensors({100, 100});

    auto view_custom = large_custom.view({10000});
    auto view_torch = large_torch.view({10000});
    compare_tensors(view_custom, view_torch, 1e-5f, 1e-7f, "LargeTensor");
}

TEST_F(TensorViewTest, SqueezeEdgeCases) {
    // All dimensions are 1
    auto [all_ones_custom, all_ones_torch] = create_test_tensors({1, 1, 1});
    auto squeezed_custom = all_ones_custom.squeeze();
    auto squeezed_torch = all_ones_torch.squeeze();

    // Known difference: Custom implementation keeps at least rank 1,
    // PyTorch goes to rank 0 (scalar tensor)
    EXPECT_EQ(squeezed_custom.ndim(), 1); // Custom: shape {1}
    EXPECT_EQ(squeezed_torch.dim(), 0);   // PyTorch: 0-D scalar

    // Values should still match
    EXPECT_FLOAT_EQ(squeezed_custom.item(), squeezed_torch.item<float>());

    std::cout << "WARNING: squeeze() on all-1 dims keeps rank 1, PyTorch goes to rank 0 (known difference)\n";

    // No dimensions of size 1
    auto [no_ones_custom, no_ones_torch] = create_test_tensors({2, 3, 4});
    auto squeezed2_custom = no_ones_custom.squeeze();
    auto squeezed2_torch = no_ones_torch.squeeze();
    compare_tensors(squeezed2_custom, squeezed2_torch, 1e-5f, 1e-7f, "SqueezeNoOnes");
}

// ============= Integration Test =============

TEST_F(TensorViewTest, CompleteWorkflow) {
    // Simulate a complete workflow with various view operations
    auto [data_custom, data_torch] = create_test_tensors({32, 10, 8});

    // Reshape for processing
    auto reshaped_custom = data_custom.reshape({32, 80});
    auto reshaped_torch = data_torch.reshape({32, 80});
    compare_tensors(reshaped_custom, reshaped_torch, 1e-5f, 1e-7f, "Workflow1");

    // Add batch dimension
    auto batched_custom = reshaped_custom.unsqueeze(0);
    auto batched_torch = reshaped_torch.unsqueeze(0);
    compare_tensors(batched_custom, batched_torch, 1e-5f, 1e-7f, "Workflow2");

    // Transpose for different layout
    auto transposed_custom = batched_custom.transpose(1, 2);
    auto transposed_torch = batched_torch.transpose(1, 2);
    compare_tensors(transposed_custom, transposed_torch, 1e-5f, 1e-7f, "Workflow3");

    // Flatten back
    auto final_custom = transposed_custom.flatten(1);
    auto final_torch = transposed_torch.flatten(1);
    compare_tensors(final_custom, final_torch, 1e-5f, 1e-7f, "WorkflowFinal");
}

// ============= Verify Data Consistency =============

TEST_F(TensorViewTest, DataConsistencyAfterViews) {
    auto [tensor_custom, tensor_torch] = create_test_tensors({6, 4});

    // Original data
    compare_tensors(tensor_custom, tensor_torch, 1e-5f, 1e-7f, "OriginalData");

    // After view
    auto view_custom = tensor_custom.view({2, 3, 4});
    auto view_torch = tensor_torch.view({2, 3, 4});
    compare_tensors(view_custom, view_torch, 1e-5f, 1e-7f, "ViewedData");

    // Flattened view should give original data
    auto flat_custom = view_custom.flatten();
    auto flat_torch = view_torch.flatten();
    compare_tensors(flat_custom, flat_torch, 1e-5f, 1e-7f, "FlattenedView");

    // Should match original flattened
    auto orig_flat_custom = tensor_custom.flatten();
    auto orig_flat_torch = tensor_torch.flatten();
    compare_tensors(orig_flat_custom, orig_flat_torch, 1e-5f, 1e-7f, "OriginalFlat");
}