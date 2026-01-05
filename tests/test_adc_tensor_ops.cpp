/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Exhaustive tests for tensor operations used in default_strategy.cpp
// Verified against LibTorch reference implementations

#include "core/tensor.hpp"
#include <gtest/gtest.h>
#include <random>
#include <torch/torch.h>

using namespace lfs::core;

namespace {

    // Convert lfs::core::Tensor to torch::Tensor for comparison
    torch::Tensor to_torch(const Tensor& t) {
        if (!t.is_valid()) {
            return torch::Tensor();
        }
        auto cpu_tensor = t.to(Device::CPU);
        std::vector<int64_t> sizes;
        for (size_t i = 0; i < t.ndim(); ++i) {
            sizes.push_back(static_cast<int64_t>(t.shape()[i]));
        }

        torch::Tensor result;
        if (t.dtype() == DataType::Float32) {
            auto ptr = cpu_tensor.ptr<float>();
            result = torch::from_blob(const_cast<float*>(ptr), sizes, torch::kFloat32).clone();
        } else if (t.dtype() == DataType::Bool) {
            auto ptr = cpu_tensor.ptr<unsigned char>();
            result = torch::from_blob(const_cast<unsigned char*>(ptr), sizes, torch::kUInt8).clone().to(torch::kBool);
        } else if (t.dtype() == DataType::Int32) {
            auto ptr = cpu_tensor.ptr<int32_t>();
            result = torch::from_blob(const_cast<int32_t*>(ptr), sizes, torch::kInt32).clone();
        } else if (t.dtype() == DataType::Int64) {
            auto ptr = cpu_tensor.ptr<int64_t>();
            result = torch::from_blob(const_cast<int64_t*>(ptr), sizes, torch::kInt64).clone();
        }
        return result.cuda();
    }

    // Convert torch::Tensor to lfs::core::Tensor
    Tensor from_torch(const torch::Tensor& t) {
        if (!t.defined()) {
            return Tensor();
        }
        auto cpu_t = t.cpu().contiguous();
        std::vector<size_t> shape;
        for (int64_t i = 0; i < cpu_t.dim(); ++i) {
            shape.push_back(static_cast<size_t>(cpu_t.size(i)));
        }

        if (cpu_t.dtype() == torch::kFloat32) {
            std::vector<float> data(cpu_t.data_ptr<float>(),
                                    cpu_t.data_ptr<float>() + cpu_t.numel());
            return Tensor::from_vector(data, TensorShape(shape), Device::CUDA);
        } else if (cpu_t.dtype() == torch::kBool) {
            auto uint8_tensor = cpu_t.to(torch::kUInt8);
            auto result = Tensor::zeros_bool(TensorShape(shape), Device::CPU);
            auto ptr = result.ptr<unsigned char>();
            std::copy(uint8_tensor.data_ptr<uint8_t>(),
                      uint8_tensor.data_ptr<uint8_t>() + uint8_tensor.numel(), ptr);
            return result.to(Device::CUDA);
        } else if (cpu_t.dtype() == torch::kInt64) {
            // Convert int64 to int32 (Tensor::from_vector doesn't support int64)
            auto int64_ptr = cpu_t.data_ptr<int64_t>();
            std::vector<int> data(cpu_t.numel());
            for (int64_t i = 0; i < cpu_t.numel(); ++i) {
                data[i] = static_cast<int>(int64_ptr[i]);
            }
            auto result = Tensor::from_vector(data, TensorShape(shape), Device::CUDA);
            return result.to(DataType::Int64); // Convert back to int64 on GPU
        } else if (cpu_t.dtype() == torch::kInt32) {
            std::vector<int32_t> data(cpu_t.data_ptr<int32_t>(),
                                      cpu_t.data_ptr<int32_t>() + cpu_t.numel());
            return Tensor::from_vector(data, TensorShape(shape), Device::CUDA);
        }
        throw std::runtime_error("Unsupported dtype in from_torch");
    }

    // Compare tensors for equality within tolerance
    bool tensors_close(const Tensor& lfs, const torch::Tensor& torch_ref,
                       float rtol = 1e-5f, float atol = 1e-6f,
                       const std::string& msg = "") {
        auto lfs_torch = to_torch(lfs);

        if (lfs_torch.sizes() != torch_ref.sizes()) {
            std::cout << msg << " Shape mismatch: [";
            for (int i = 0; i < lfs_torch.dim(); ++i) {
                std::cout << lfs_torch.size(i) << (i < lfs_torch.dim() - 1 ? ", " : "");
            }
            std::cout << "] vs [";
            for (int i = 0; i < torch_ref.dim(); ++i) {
                std::cout << torch_ref.size(i) << (i < torch_ref.dim() - 1 ? ", " : "");
            }
            std::cout << "]" << std::endl;
            return false;
        }

        // Handle bool tensors
        if (torch_ref.dtype() == torch::kBool) {
            auto lfs_bool = lfs_torch.to(torch::kBool);
            auto match = (lfs_bool == torch_ref).all().item<bool>();
            if (!match) {
                auto diff_count = (~(lfs_bool == torch_ref)).sum().item<int64_t>();
                std::cout << msg << " Bool tensor mismatch: " << diff_count << " elements differ" << std::endl;
            }
            return match;
        }

        // Handle int tensors
        if (torch_ref.dtype() == torch::kInt64 || torch_ref.dtype() == torch::kInt32) {
            auto match = (lfs_torch == torch_ref).all().item<bool>();
            if (!match) {
                auto diff_count = (~(lfs_torch == torch_ref)).sum().item<int64_t>();
                std::cout << msg << " Int tensor mismatch: " << diff_count << " elements differ" << std::endl;
            }
            return match;
        }

        // Float comparison
        auto diff = torch::abs(lfs_torch.to(torch::kFloat32) - torch_ref.to(torch::kFloat32));
        auto threshold = atol + rtol * torch::abs(torch_ref.to(torch::kFloat32));
        auto close = diff <= threshold;
        auto all_close = close.all().item<bool>();

        if (!all_close) {
            auto max_diff = diff.max().item<float>();
            auto mean_diff = diff.mean().item<float>();
            auto num_bad = (~close).sum().item<int64_t>();
            std::cout << msg << " Float mismatch: " << num_bad << " elements, max_diff=" << max_diff
                      << ", mean_diff=" << mean_diff << std::endl;
        }
        return all_close;
    }

} // anonymous namespace

// ============= Test Fixture =============

class DefaultStrategyTensorOpsTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA required";
        torch::manual_seed(42);
        Tensor::manual_seed(42);
    }
};

// =============================================================================
// CREATION OPERATIONS
// Used in default_strategy.cpp for initializing tensors
// =============================================================================

TEST_F(DefaultStrategyTensorOpsTest, Zeros_Basic) {
    // Usage: _splat_data->_densification_info = Tensor::zeros({2, N}, device)
    auto lfs_zeros = Tensor::zeros({2, 100}, Device::CUDA);
    auto torch_zeros = torch::zeros({2, 100}, torch::kCUDA);

    EXPECT_TRUE(tensors_close(lfs_zeros, torch_zeros, 0, 0, "Zeros_Basic"));
}

TEST_F(DefaultStrategyTensorOpsTest, ZerosBool_Basic) {
    // Usage: _free_mask = Tensor::zeros_bool({capacity}, device)
    auto lfs_zeros = Tensor::zeros_bool({100}, Device::CUDA);
    auto torch_zeros = torch::zeros({100}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    EXPECT_TRUE(tensors_close(lfs_zeros, torch_zeros, 0, 0, "ZerosBool_Basic"));
}

TEST_F(DefaultStrategyTensorOpsTest, OnesBool_Basic) {
    // Usage: true_vals = Tensor::ones_bool({n}, device)
    auto lfs_ones = Tensor::ones_bool({50}, Device::CUDA);
    auto torch_ones = torch::ones({50}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    EXPECT_TRUE(tensors_close(lfs_ones, torch_ones, 0, 0, "OnesBool_Basic"));
}

TEST_F(DefaultStrategyTensorOpsTest, Empty_Basic) {
    // Usage: second_positions = Tensor::empty({num_split, 3}, device)
    auto lfs_empty = Tensor::empty({10, 3}, Device::CUDA);
    auto torch_empty = torch::empty({10, 3}, torch::kCUDA);

    // Can't compare values (uninitialized), just check shape
    EXPECT_EQ(lfs_empty.shape()[0], 10);
    EXPECT_EQ(lfs_empty.shape()[1], 3);
    EXPECT_EQ(lfs_empty.numel(), 30);
}

TEST_F(DefaultStrategyTensorOpsTest, Randn_ShapeMatch) {
    // Usage: random_noise = Tensor::randn({2, num_split, 3}, device)
    // Note: Can't compare values since random, but shape and stats should match
    torch::manual_seed(123);
    Tensor::manual_seed(123);

    auto lfs_randn = Tensor::randn({2, 50, 3}, Device::CUDA);
    auto torch_randn = torch::randn({2, 50, 3}, torch::kCUDA);

    EXPECT_EQ(lfs_randn.shape()[0], 2);
    EXPECT_EQ(lfs_randn.shape()[1], 50);
    EXPECT_EQ(lfs_randn.shape()[2], 3);

    // Statistical properties should be similar (mean ~0, std ~1)
    auto lfs_mean = to_torch(lfs_randn).mean().item<float>();
    auto lfs_std = to_torch(lfs_randn).std().item<float>();
    EXPECT_NEAR(lfs_mean, 0.0f, 0.3f); // Loose bound for random
    EXPECT_NEAR(lfs_std, 1.0f, 0.3f);
}

TEST_F(DefaultStrategyTensorOpsTest, FromVector_Int) {
    // Usage: new_indices = Tensor::from_vector(new_indices_vec, shape, device)
    std::vector<int> data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto lfs_tensor = Tensor::from_vector(data, TensorShape({10}), Device::CUDA);
    auto torch_tensor = torch::tensor(std::vector<int64_t>(data.begin(), data.end()),
                                      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    auto lfs_torch = to_torch(lfs_tensor);
    EXPECT_TRUE((lfs_torch.to(torch::kInt64) == torch_tensor.to(torch::kInt64)).all().item<bool>());
}

// =============================================================================
// INDEXING OPERATIONS
// Critical for densification (grow_gs, prune_gs, duplicate, split)
// =============================================================================

TEST_F(DefaultStrategyTensorOpsTest, NonzeroSqueeze_BasicPattern) {
    // Usage: sampled_idxs = is_duplicated.nonzero().squeeze(-1)
    // This is the key pattern for converting a mask to indices

    std::vector<bool> mask_data = {true, false, true, false, false, true, true, false};
    auto torch_mask = torch::tensor(std::vector<int8_t>{1, 0, 1, 0, 0, 1, 1, 0},
                                    torch::kCUDA)
                          .to(torch::kBool);
    auto lfs_mask = from_torch(torch_mask);

    auto torch_indices = torch_mask.nonzero().squeeze(-1);
    auto lfs_indices = lfs_mask.nonzero().squeeze(-1);

    EXPECT_TRUE(tensors_close(lfs_indices, torch_indices, 0, 0, "NonzeroSqueeze"));

    // Expected: [0, 2, 5, 6]
    EXPECT_EQ(lfs_indices.numel(), 4);
}

TEST_F(DefaultStrategyTensorOpsTest, NonzeroSqueeze_AllTrue) {
    auto torch_mask = torch::ones({10}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    auto lfs_mask = from_torch(torch_mask);

    auto torch_indices = torch_mask.nonzero().squeeze(-1);
    auto lfs_indices = lfs_mask.nonzero().squeeze(-1);

    EXPECT_TRUE(tensors_close(lfs_indices, torch_indices, 0, 0, "NonzeroSqueeze_AllTrue"));
    EXPECT_EQ(lfs_indices.numel(), 10);
}

TEST_F(DefaultStrategyTensorOpsTest, NonzeroSqueeze_AllFalse) {
    auto torch_mask = torch::zeros({10}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    auto lfs_mask = from_torch(torch_mask);

    auto torch_indices = torch_mask.nonzero().squeeze(-1);
    auto lfs_indices = lfs_mask.nonzero().squeeze(-1);

    // Both should be empty
    EXPECT_EQ(lfs_indices.numel(), 0);
    EXPECT_EQ(torch_indices.numel(), 0);
}

TEST_F(DefaultStrategyTensorOpsTest, Slice_Basic) {
    // Usage: append_src_indices = sampled_idxs.slice(0, num_filled, num_duplicated)
    auto torch_tensor = torch::arange(0, 100, torch::kCUDA).to(torch::kInt64);
    auto lfs_tensor = from_torch(torch_tensor);

    auto torch_slice = torch_tensor.slice(0, 10, 30);
    auto lfs_slice = lfs_tensor.slice(0, 10, 30);

    EXPECT_TRUE(tensors_close(lfs_slice, torch_slice, 0, 0, "Slice_Basic"));
    EXPECT_EQ(lfs_slice.numel(), 20);
}

TEST_F(DefaultStrategyTensorOpsTest, Slice_2D) {
    // Usage: slicing 2D tensors for positions, rotations, etc.
    auto torch_tensor = torch::randn({50, 3}, torch::kCUDA);
    auto lfs_tensor = from_torch(torch_tensor);

    auto torch_slice = torch_tensor.slice(0, 5, 15);
    auto lfs_slice = lfs_tensor.slice(0, 5, 15);

    EXPECT_TRUE(tensors_close(lfs_slice, torch_slice, 1e-5f, 1e-6f, "Slice_2D"));
    EXPECT_EQ(lfs_slice.shape()[0], 10);
    EXPECT_EQ(lfs_slice.shape()[1], 3);
}

TEST_F(DefaultStrategyTensorOpsTest, IndexSelect_Basic) {
    // Usage: _splat_data->means().index_select(0, src_indices)
    auto torch_data = torch::randn({100, 3}, torch::kCUDA);
    auto torch_indices = torch::tensor({5, 10, 15, 20, 25}, torch::kCUDA).to(torch::kInt64);

    auto lfs_data = from_torch(torch_data);
    auto lfs_indices = from_torch(torch_indices);

    auto torch_result = torch_data.index_select(0, torch_indices);
    auto lfs_result = lfs_data.index_select(0, lfs_indices);

    EXPECT_TRUE(tensors_close(lfs_result, torch_result, 1e-5f, 1e-6f, "IndexSelect_Basic"));
    EXPECT_EQ(lfs_result.shape()[0], 5);
    EXPECT_EQ(lfs_result.shape()[1], 3);
}

TEST_F(DefaultStrategyTensorOpsTest, IndexSelect_1D) {
    // Usage: for 1D tensors like opacity
    auto torch_data = torch::randn({100}, torch::kCUDA);
    auto torch_indices = torch::tensor({0, 50, 99}, torch::kCUDA).to(torch::kInt64);

    auto lfs_data = from_torch(torch_data);
    auto lfs_indices = from_torch(torch_indices);

    auto torch_result = torch_data.index_select(0, torch_indices);
    auto lfs_result = lfs_data.index_select(0, lfs_indices);

    EXPECT_TRUE(tensors_close(lfs_result, torch_result, 1e-5f, 1e-6f, "IndexSelect_1D"));
}

TEST_F(DefaultStrategyTensorOpsTest, IndexPut_Basic) {
    // Usage: state->exp_avg.index_put_(split_idxs, zeros)
    auto torch_data = torch::randn({100, 3}, torch::kCUDA);
    auto torch_indices = torch::tensor({5, 10, 15}, torch::kCUDA).to(torch::kInt64);
    auto torch_values = torch::zeros({3, 3}, torch::kCUDA);

    auto lfs_data = from_torch(torch_data);
    auto lfs_indices = from_torch(torch_indices);
    auto lfs_values = from_torch(torch_values);

    torch_data.index_put_({torch_indices}, torch_values);
    lfs_data.index_put_(lfs_indices, lfs_values);

    EXPECT_TRUE(tensors_close(lfs_data, torch_data, 1e-5f, 1e-6f, "IndexPut_Basic"));

    // Verify the specific rows are zero
    auto lfs_torch = to_torch(lfs_data);
    EXPECT_TRUE((lfs_torch.index({torch_indices}) == torch::zeros({3, 3}, torch::kCUDA)).all().item<bool>());
}

TEST_F(DefaultStrategyTensorOpsTest, IndexPut_Bool) {
    // Usage: _free_mask.index_put_(target_indices, false_vals)
    auto torch_mask = torch::ones({50}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    auto torch_indices = torch::tensor({0, 10, 20, 30, 40}, torch::kCUDA).to(torch::kInt64);
    auto torch_false = torch::zeros({5}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    auto lfs_mask = from_torch(torch_mask);
    auto lfs_indices = from_torch(torch_indices);
    auto lfs_false = from_torch(torch_false);

    torch_mask.index_put_({torch_indices}, torch_false);
    lfs_mask.index_put_(lfs_indices, lfs_false);

    EXPECT_TRUE(tensors_close(lfs_mask, torch_mask, 0, 0, "IndexPut_Bool"));
}

// =============================================================================
// LOGICAL OPERATIONS
// Used for mask manipulation in grow_gs, prune_gs
// =============================================================================

TEST_F(DefaultStrategyTensorOpsTest, LogicalAnd_Basic) {
    // Usage: is_duplicated = is_grad_high.logical_and(is_small)
    auto torch_a = torch::tensor({true, true, false, false}, torch::kCUDA).to(torch::kBool);
    auto torch_b = torch::tensor({true, false, true, false}, torch::kCUDA).to(torch::kBool);

    auto lfs_a = from_torch(torch_a);
    auto lfs_b = from_torch(torch_b);

    auto torch_result = torch_a & torch_b;
    auto lfs_result = lfs_a.logical_and(lfs_b);

    EXPECT_TRUE(tensors_close(lfs_result, torch_result, 0, 0, "LogicalAnd_Basic"));
}

TEST_F(DefaultStrategyTensorOpsTest, LogicalAnd_Large) {
    // Test with realistic size
    torch::manual_seed(42);
    auto torch_a = torch::randint(0, 2, {10000}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    auto torch_b = torch::randint(0, 2, {10000}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    auto lfs_a = from_torch(torch_a);
    auto lfs_b = from_torch(torch_b);

    auto torch_result = torch_a & torch_b;
    auto lfs_result = lfs_a.logical_and(lfs_b);

    EXPECT_TRUE(tensors_close(lfs_result, torch_result, 0, 0, "LogicalAnd_Large"));
}

TEST_F(DefaultStrategyTensorOpsTest, LogicalOr_Basic) {
    // Usage: is_prune = is_prune.logical_or(is_too_big)
    auto torch_a = torch::tensor({true, true, false, false}, torch::kCUDA).to(torch::kBool);
    auto torch_b = torch::tensor({true, false, true, false}, torch::kCUDA).to(torch::kBool);

    auto lfs_a = from_torch(torch_a);
    auto lfs_b = from_torch(torch_b);

    auto torch_result = torch_a | torch_b;
    auto lfs_result = lfs_a.logical_or(lfs_b);

    EXPECT_TRUE(tensors_close(lfs_result, torch_result, 0, 0, "LogicalOr_Basic"));
}

TEST_F(DefaultStrategyTensorOpsTest, LogicalNot_Basic) {
    // Usage: is_active = active_free_mask.logical_not()
    auto torch_mask = torch::tensor({true, false, true, false, true}, torch::kCUDA).to(torch::kBool);
    auto lfs_mask = from_torch(torch_mask);

    auto torch_result = ~torch_mask;
    auto lfs_result = lfs_mask.logical_not();

    EXPECT_TRUE(tensors_close(lfs_result, torch_result, 0, 0, "LogicalNot_Basic"));
}

TEST_F(DefaultStrategyTensorOpsTest, LogicalChain_GrowGsPattern) {
    // Simulate the exact pattern from grow_gs:
    // is_grad_high = grads > threshold
    // is_small = max_values <= threshold * scale
    // is_duplicated = is_grad_high.logical_and(is_small)
    // is_large = is_small.logical_not()
    // is_split = is_grad_high.logical_and(is_large)

    auto torch_grads = torch::randn({100}, torch::kCUDA);
    auto torch_max_values = torch::randn({100}, torch::kCUDA).abs();

    float grad_threshold = 0.5f;
    float scale_threshold = 1.0f;

    auto torch_is_grad_high = torch_grads > grad_threshold;
    auto torch_is_small = torch_max_values <= scale_threshold;
    auto torch_is_duplicated = torch_is_grad_high & torch_is_small;
    auto torch_is_large = ~torch_is_small;
    auto torch_is_split = torch_is_grad_high & torch_is_large;

    auto lfs_grads = from_torch(torch_grads);
    auto lfs_max_values = from_torch(torch_max_values);

    auto lfs_is_grad_high = lfs_grads > grad_threshold;
    auto lfs_is_small = lfs_max_values <= scale_threshold;
    auto lfs_is_duplicated = lfs_is_grad_high.logical_and(lfs_is_small);
    auto lfs_is_large = lfs_is_small.logical_not();
    auto lfs_is_split = lfs_is_grad_high.logical_and(lfs_is_large);

    EXPECT_TRUE(tensors_close(lfs_is_grad_high, torch_is_grad_high, 0, 0, "GrowGs_IsGradHigh"));
    EXPECT_TRUE(tensors_close(lfs_is_small, torch_is_small, 0, 0, "GrowGs_IsSmall"));
    EXPECT_TRUE(tensors_close(lfs_is_duplicated, torch_is_duplicated, 0, 0, "GrowGs_IsDuplicated"));
    EXPECT_TRUE(tensors_close(lfs_is_large, torch_is_large, 0, 0, "GrowGs_IsLarge"));
    EXPECT_TRUE(tensors_close(lfs_is_split, torch_is_split, 0, 0, "GrowGs_IsSplit"));
}

// =============================================================================
// REDUCTION OPERATIONS
// Used for counting, thresholding, gradient computation
// =============================================================================

TEST_F(DefaultStrategyTensorOpsTest, Sum_Scalar) {
    // Usage: num_duplicates = is_duplicated.sum_scalar()
    auto torch_tensor = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, torch::kCUDA);
    auto lfs_tensor = from_torch(torch_tensor);

    float torch_sum = torch_tensor.sum().item<float>();
    float lfs_sum = lfs_tensor.sum_scalar();

    EXPECT_NEAR(lfs_sum, torch_sum, 1e-5f);
    EXPECT_NEAR(lfs_sum, 15.0f, 1e-5f);
}

TEST_F(DefaultStrategyTensorOpsTest, Sum_BoolToInt) {
    // Usage: mask.to(Int32).sum() for counting true values
    auto torch_mask = torch::tensor({true, false, true, true, false}, torch::kCUDA).to(torch::kBool);
    auto lfs_mask = from_torch(torch_mask);

    int torch_sum = torch_mask.to(torch::kInt32).sum().item<int>();
    int lfs_sum = lfs_mask.to(DataType::Int32).sum().template item<int>();

    EXPECT_EQ(lfs_sum, torch_sum);
    EXPECT_EQ(lfs_sum, 3);
}

TEST_F(DefaultStrategyTensorOpsTest, Sum_AlongDim) {
    // Usage: (rotation_raw * rotation_raw).sum(-1, false)
    auto torch_tensor = torch::randn({100, 4}, torch::kCUDA);
    auto lfs_tensor = from_torch(torch_tensor);

    auto torch_sq = torch_tensor * torch_tensor;
    auto torch_sum = torch_sq.sum(-1);

    auto lfs_sq = lfs_tensor * lfs_tensor;
    auto lfs_sum = lfs_sq.sum(-1, false);

    EXPECT_TRUE(tensors_close(lfs_sum, torch_sum, 1e-4f, 1e-5f, "Sum_AlongDim"));
    EXPECT_EQ(lfs_sum.ndim(), 1);
    EXPECT_EQ(lfs_sum.shape()[0], 100);
}

TEST_F(DefaultStrategyTensorOpsTest, Max_AlongDim) {
    // Usage: max_values = _splat_data->get_scaling().max(-1, false)
    auto torch_tensor = torch::randn({100, 3}, torch::kCUDA);
    auto lfs_tensor = from_torch(torch_tensor);

    auto torch_max = std::get<0>(torch::max(torch_tensor, -1));
    auto lfs_max = lfs_tensor.max(-1, false);

    EXPECT_TRUE(tensors_close(lfs_max, torch_max, 1e-5f, 1e-6f, "Max_AlongDim"));
    EXPECT_EQ(lfs_max.ndim(), 1);
    EXPECT_EQ(lfs_max.shape()[0], 100);
}

TEST_F(DefaultStrategyTensorOpsTest, ClampMin_Basic) {
    // Usage: grads = numer / denom.clamp_min(1.0f)
    auto torch_tensor = torch::tensor({-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f}, torch::kCUDA);
    auto lfs_tensor = from_torch(torch_tensor);

    auto torch_clamped = torch::clamp_min(torch_tensor, 0.5f);
    auto lfs_clamped = lfs_tensor.clamp_min(0.5f);

    EXPECT_TRUE(tensors_close(lfs_clamped, torch_clamped, 1e-6f, 1e-7f, "ClampMin_Basic"));
}

TEST_F(DefaultStrategyTensorOpsTest, ClampMaxInplace_Basic) {
    // Usage: _splat_data->opacity_raw().clamp_max_(logit_threshold)
    auto torch_tensor = torch::tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 5.0f}, torch::kCUDA);
    auto lfs_tensor = from_torch(torch_tensor);

    torch_tensor.clamp_max_(1.5f);
    lfs_tensor.clamp_max_(1.5f);

    EXPECT_TRUE(tensors_close(lfs_tensor, torch_tensor, 1e-6f, 1e-7f, "ClampMaxInplace_Basic"));
}

TEST_F(DefaultStrategyTensorOpsTest, DivisionWithClamp_GradPattern) {
    // Full pattern: grads = numer / denom.clamp_min(1.0f)
    auto torch_numer = torch::randn({100}, torch::kCUDA).abs();
    auto torch_denom = torch::randint(0, 10, {100}, torch::kCUDA).to(torch::kFloat32);

    auto lfs_numer = from_torch(torch_numer);
    auto lfs_denom = from_torch(torch_denom);

    auto torch_grads = torch_numer / torch::clamp_min(torch_denom, 1.0f);
    auto lfs_grads = lfs_numer / lfs_denom.clamp_min(1.0f);

    EXPECT_TRUE(tensors_close(lfs_grads, torch_grads, 1e-4f, 1e-5f, "DivisionWithClamp"));
}

// =============================================================================
// SHAPE OPERATIONS
// Used for concatenation and reshaping
// =============================================================================

TEST_F(DefaultStrategyTensorOpsTest, Cat_1D) {
    // Usage: is_split = is_split.cat(zeros_to_concat, 0)
    auto torch_a = torch::randn({50}, torch::kCUDA);
    auto torch_b = torch::zeros({20}, torch::kCUDA);

    auto lfs_a = from_torch(torch_a);
    auto lfs_b = from_torch(torch_b);

    auto torch_cat = torch::cat({torch_a, torch_b}, 0);
    auto lfs_cat = lfs_a.cat(lfs_b, 0);

    EXPECT_TRUE(tensors_close(lfs_cat, torch_cat, 1e-5f, 1e-6f, "Cat_1D"));
    EXPECT_EQ(lfs_cat.numel(), 70);
}

TEST_F(DefaultStrategyTensorOpsTest, Cat_2D) {
    // Usage: concatenating 2D parameter tensors
    auto torch_a = torch::randn({50, 3}, torch::kCUDA);
    auto torch_b = torch::randn({20, 3}, torch::kCUDA);

    auto lfs_a = from_torch(torch_a);
    auto lfs_b = from_torch(torch_b);

    auto torch_cat = torch::cat({torch_a, torch_b}, 0);
    auto lfs_cat = lfs_a.cat(lfs_b, 0);

    EXPECT_TRUE(tensors_close(lfs_cat, torch_cat, 1e-5f, 1e-6f, "Cat_2D"));
    EXPECT_EQ(lfs_cat.shape()[0], 70);
    EXPECT_EQ(lfs_cat.shape()[1], 3);
}

TEST_F(DefaultStrategyTensorOpsTest, Cat_Bool) {
    // Usage: is_split = is_split.cat(zeros_to_concat, 0) where both are bool
    auto torch_a = torch::randint(0, 2, {50}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    auto torch_b = torch::zeros({20}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    auto lfs_a = from_torch(torch_a);
    auto lfs_b = from_torch(torch_b);

    auto torch_cat = torch::cat({torch_a, torch_b}, 0);
    auto lfs_cat = lfs_a.cat(lfs_b, 0);

    EXPECT_TRUE(tensors_close(lfs_cat, torch_cat, 0, 0, "Cat_Bool"));
}

TEST_F(DefaultStrategyTensorOpsTest, Reshape_Basic) {
    // Usage: append_sh0_reshaped = append_sh0_flat.reshape({n_remaining, 1, 3})
    auto torch_tensor = torch::randn({30}, torch::kCUDA);
    auto lfs_tensor = from_torch(torch_tensor);

    auto torch_reshaped = torch_tensor.reshape({10, 3});
    auto lfs_reshaped = lfs_tensor.reshape(TensorShape({10, 3}));

    EXPECT_TRUE(tensors_close(lfs_reshaped, torch_reshaped, 1e-5f, 1e-6f, "Reshape_Basic"));
}

TEST_F(DefaultStrategyTensorOpsTest, Reshape_SH0Pattern) {
    // Usage: sh0_reshaped = sh0.slice(0, 0, slots_to_fill).reshape({slots_to_fill, 1, 3})
    auto torch_tensor = torch::randn({20, 3}, torch::kCUDA);
    auto lfs_tensor = from_torch(torch_tensor);

    auto torch_reshaped = torch_tensor.reshape({20, 1, 3});
    auto lfs_reshaped = lfs_tensor.reshape(TensorShape({20, 1, 3}));

    EXPECT_TRUE(tensors_close(lfs_reshaped, torch_reshaped, 1e-5f, 1e-6f, "Reshape_SH0"));
    EXPECT_EQ(lfs_reshaped.ndim(), 3);
    EXPECT_EQ(lfs_reshaped.shape()[0], 20);
    EXPECT_EQ(lfs_reshaped.shape()[1], 1);
    EXPECT_EQ(lfs_reshaped.shape()[2], 3);
}

// =============================================================================
// IN-PLACE MODIFICATION OPERATIONS
// Critical for memory-efficient training without reallocations
// =============================================================================

TEST_F(DefaultStrategyTensorOpsTest, AppendGather_Basic) {
    // Usage: _splat_data->means().append_gather(append_src_indices)
    // This appends gathered rows from itself to the end
    // Note: In-place append ops require pre-allocated capacity via reserve()

    auto torch_data = torch::arange(0, 30, torch::kCUDA).to(torch::kFloat32).reshape({10, 3});
    auto torch_indices = torch::tensor({0, 2, 5}, torch::kCUDA).to(torch::kInt64);

    auto lfs_data = from_torch(torch_data);
    auto lfs_indices = from_torch(torch_indices);

    // Reserve capacity for appending (required for in-place operations)
    lfs_data.reserve(20); // Reserve space for 10 + some extra

    // Simulate append_gather: gather and concat
    auto torch_gathered = torch_data.index_select(0, torch_indices);
    auto torch_result = torch::cat({torch_data, torch_gathered}, 0);

    lfs_data.append_gather(lfs_indices);

    EXPECT_TRUE(tensors_close(lfs_data, torch_result, 1e-5f, 1e-6f, "AppendGather_Basic"));
    EXPECT_EQ(lfs_data.shape()[0], 13); // 10 + 3
}

TEST_F(DefaultStrategyTensorOpsTest, AppendZeros_Basic) {
    // Usage: _splat_data->means().append_zeros(n_remaining)
    // Note: In-place append ops require pre-allocated capacity via reserve()
    auto torch_data = torch::randn({50, 3}, torch::kCUDA);
    auto lfs_data = from_torch(torch_data);

    // Reserve capacity for appending
    lfs_data.reserve(70);

    auto torch_zeros = torch::zeros({10, 3}, torch::kCUDA);
    auto torch_result = torch::cat({torch_data, torch_zeros}, 0);

    lfs_data.append_zeros(10);

    EXPECT_TRUE(tensors_close(lfs_data, torch_result, 1e-5f, 1e-6f, "AppendZeros_Basic"));
    EXPECT_EQ(lfs_data.shape()[0], 60);
}

TEST_F(DefaultStrategyTensorOpsTest, AppendZeros_1D) {
    // Usage: for 1D tensors like opacity_raw
    // Note: In-place append ops require pre-allocated capacity via reserve()
    auto torch_data = torch::randn({50}, torch::kCUDA);
    auto lfs_data = from_torch(torch_data);

    // Reserve capacity for appending
    lfs_data.reserve(70);

    auto torch_zeros = torch::zeros({10}, torch::kCUDA);
    auto torch_result = torch::cat({torch_data, torch_zeros}, 0);

    lfs_data.append_zeros(10);

    EXPECT_TRUE(tensors_close(lfs_data, torch_result, 1e-5f, 1e-6f, "AppendZeros_1D"));
    EXPECT_EQ(lfs_data.numel(), 60);
}

TEST_F(DefaultStrategyTensorOpsTest, ZeroInplace_Basic) {
    // Usage: state->exp_avg.zero_()
    auto torch_data = torch::randn({50, 3}, torch::kCUDA);
    auto lfs_data = from_torch(torch_data);

    torch_data.zero_();
    lfs_data.zero_();

    EXPECT_TRUE(tensors_close(lfs_data, torch_data, 0, 0, "ZeroInplace_Basic"));
    EXPECT_EQ(to_torch(lfs_data).sum().item<float>(), 0.0f);
}

// =============================================================================
// COMBINED PATTERNS FROM DEFAULT_STRATEGY
// Test exact sequences of operations used in densification
// =============================================================================

TEST_F(DefaultStrategyTensorOpsTest, DuplicatePattern_Full) {
    // Full duplicate pattern from default_strategy.cpp:
    // 1. sampled_idxs = is_duplicated.nonzero().squeeze(-1)
    // 2. new_param = param.index_select(0, sampled_idxs)
    // 3. result = param.cat(new_param, 0)

    torch::manual_seed(123);
    auto torch_means = torch::randn({100, 3}, torch::kCUDA);
    auto torch_mask = torch::randint(0, 2, {100}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    auto lfs_means = from_torch(torch_means);
    auto lfs_mask = from_torch(torch_mask);

    // Step 1: Get indices
    auto torch_indices = torch_mask.nonzero().squeeze(-1);
    auto lfs_indices = lfs_mask.nonzero().squeeze(-1);

    EXPECT_TRUE(tensors_close(lfs_indices, torch_indices, 0, 0, "Duplicate_Indices"));

    if (torch_indices.numel() > 0) {
        // Step 2: Gather
        auto torch_new_means = torch_means.index_select(0, torch_indices);
        auto lfs_new_means = lfs_means.index_select(0, lfs_indices);

        EXPECT_TRUE(tensors_close(lfs_new_means, torch_new_means, 1e-5f, 1e-6f, "Duplicate_Gather"));

        // Step 3: Concatenate
        auto torch_result = torch::cat({torch_means, torch_new_means}, 0);
        auto lfs_result = lfs_means.cat(lfs_new_means, 0);

        EXPECT_TRUE(tensors_close(lfs_result, torch_result, 1e-5f, 1e-6f, "Duplicate_Result"));
    }
}

TEST_F(DefaultStrategyTensorOpsTest, RemovePattern_Full) {
    // Full remove pattern from default_strategy.cpp:
    // 1. prune_indices = is_prune.nonzero().squeeze(-1)
    // 2. Zero out quaternions at prune_indices
    // 3. Zero out optimizer states at prune_indices

    torch::manual_seed(456);
    auto torch_rotation = torch::randn({100, 4}, torch::kCUDA);
    auto torch_mask = torch::randint(0, 2, {100}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    auto lfs_rotation = from_torch(torch_rotation);
    auto lfs_mask = from_torch(torch_mask);

    // Get prune indices
    auto torch_prune_indices = torch_mask.nonzero().squeeze(-1);
    auto lfs_prune_indices = lfs_mask.nonzero().squeeze(-1);

    int64_t num_pruned = torch_prune_indices.numel();

    if (num_pruned > 0) {
        // Create zero rotation tensor
        auto torch_zero_rot = torch::zeros({num_pruned, 4}, torch::kCUDA);
        auto lfs_zero_rot = Tensor::zeros({static_cast<size_t>(num_pruned), 4}, Device::CUDA);

        // Apply in-place update
        torch_rotation.index_put_({torch_prune_indices}, torch_zero_rot);
        lfs_rotation.index_put_(lfs_prune_indices, lfs_zero_rot);

        EXPECT_TRUE(tensors_close(lfs_rotation, torch_rotation, 1e-5f, 1e-6f, "Remove_IndexPut"));

        // Verify zeroed rows
        auto lfs_torch = to_torch(lfs_rotation);
        auto zeroed = lfs_torch.index({torch_prune_indices});
        EXPECT_TRUE((zeroed == 0).all().item<bool>());
    }
}

TEST_F(DefaultStrategyTensorOpsTest, PruneGsPattern_Full) {
    // Full prune_gs pattern:
    // 1. is_prune = get_opacity() < threshold
    // 2. is_prune = is_prune.logical_or((rot * rot).sum(-1) < 1e-8)
    // 3. is_prune = is_prune.logical_and(is_active)

    torch::manual_seed(789);
    auto torch_opacity = torch::randn({100}, torch::kCUDA);
    auto torch_rotation = torch::randn({100, 4}, torch::kCUDA);
    auto torch_free_mask = torch::randint(0, 2, {100}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    float prune_threshold = 0.01f;

    auto lfs_opacity = from_torch(torch_opacity);
    auto lfs_rotation = from_torch(torch_rotation);
    auto lfs_free_mask = from_torch(torch_free_mask);

    // Step 1: Low opacity check
    auto torch_is_prune = torch_opacity < prune_threshold;
    auto lfs_is_prune = lfs_opacity < prune_threshold;

    EXPECT_TRUE(tensors_close(lfs_is_prune, torch_is_prune, 0, 0, "Prune_LowOpacity"));

    // Step 2: Invalid rotation check
    auto torch_rot_mag = (torch_rotation * torch_rotation).sum(-1);
    auto torch_invalid_rot = torch_rot_mag < 1e-8f;
    torch_is_prune = torch_is_prune | torch_invalid_rot;

    auto lfs_rot_mag = (lfs_rotation * lfs_rotation).sum(-1, false);
    auto lfs_invalid_rot = lfs_rot_mag < 1e-8f;
    lfs_is_prune = lfs_is_prune.logical_or(lfs_invalid_rot);

    EXPECT_TRUE(tensors_close(lfs_is_prune, torch_is_prune, 0, 0, "Prune_InvalidRot"));

    // Step 3: Exclude free slots
    auto torch_is_active = ~torch_free_mask;
    torch_is_prune = torch_is_prune & torch_is_active;

    auto lfs_is_active = lfs_free_mask.logical_not();
    lfs_is_prune = lfs_is_prune.logical_and(lfs_is_active);

    EXPECT_TRUE(tensors_close(lfs_is_prune, torch_is_prune, 0, 0, "Prune_Final"));
}

TEST_F(DefaultStrategyTensorOpsTest, GrowGsCapEnforcement_Full) {
    // Test the max_cap enforcement pattern:
    // 1. indices = is_duplicated.nonzero().squeeze(-1)
    // 2. keep_indices = indices.slice(0, 0, available)
    // 3. is_duplicated = zeros_bool
    // 4. is_duplicated.index_put_(keep_indices, ones_bool)

    torch::manual_seed(321);
    int current_n = 100;
    int max_cap = 110;
    int available = max_cap - current_n; // 10

    auto torch_is_duplicated = torch::randint(0, 2, {current_n},
                                              torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    auto lfs_is_duplicated = from_torch(torch_is_duplicated);

    // Get indices and limit
    auto torch_indices = torch_is_duplicated.nonzero().squeeze(-1);
    auto lfs_indices = lfs_is_duplicated.nonzero().squeeze(-1);

    if (torch_indices.numel() > available) {
        auto torch_keep = torch_indices.slice(0, 0, available);
        auto lfs_keep = lfs_indices.slice(0, 0, available);

        // Create limited mask
        auto torch_limited = torch::zeros({current_n}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
        auto torch_true_vals = torch::ones({available}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
        torch_limited.index_put_({torch_keep}, torch_true_vals);

        auto lfs_limited = Tensor::zeros_bool({static_cast<size_t>(current_n)}, Device::CUDA);
        auto lfs_true_vals = Tensor::ones_bool({static_cast<size_t>(available)}, Device::CUDA);
        lfs_limited.index_put_(lfs_keep, lfs_true_vals);

        EXPECT_TRUE(tensors_close(lfs_limited, torch_limited, 0, 0, "GrowGs_Limited"));

        // Verify exactly 'available' true values
        EXPECT_EQ(torch_limited.sum().item<int>(), available);
        EXPECT_EQ(static_cast<int>(lfs_limited.sum_scalar()), available);
    }
}

TEST_F(DefaultStrategyTensorOpsTest, FillFreeSlotsPattern) {
    // Test the fill_free_slots pattern:
    // 1. free_indices = free_mask.slice(0, 0, current_size).nonzero().squeeze(-1)
    // 2. target_indices = free_indices.slice(0, 0, slots_to_fill)
    // 3. data.index_put_(target_indices, new_data)
    // 4. free_mask.index_put_(target_indices, false)

    torch::manual_seed(654);
    size_t capacity = 150;
    size_t current_size = 100;

    auto torch_data = torch::randn({static_cast<int64_t>(current_size), 3}, torch::kCUDA);
    auto torch_free_mask = torch::randint(0, 2, {static_cast<int64_t>(capacity)},
                                          torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    auto torch_new_data = torch::randn({20, 3}, torch::kCUDA);

    auto lfs_data = from_torch(torch_data);
    auto lfs_free_mask = from_torch(torch_free_mask);
    auto lfs_new_data = from_torch(torch_new_data);

    // Get active region and find free indices
    auto torch_active_region = torch_free_mask.slice(0, 0, current_size);
    auto torch_free_indices = torch_active_region.nonzero().squeeze(-1);

    auto lfs_active_region = lfs_free_mask.slice(0, 0, current_size);
    auto lfs_free_indices = lfs_active_region.nonzero().squeeze(-1);

    EXPECT_TRUE(tensors_close(lfs_free_indices, torch_free_indices, 0, 0, "FillSlots_FindFree"));

    int64_t num_free = torch_free_indices.numel();
    int64_t slots_to_fill = std::min(num_free, static_cast<int64_t>(20));

    if (slots_to_fill > 0) {
        auto torch_target = torch_free_indices.slice(0, 0, slots_to_fill);
        auto lfs_target = lfs_free_indices.slice(0, 0, slots_to_fill);

        // Fill data
        auto torch_fill_data = torch_new_data.slice(0, 0, slots_to_fill);
        torch_data.index_put_({torch_target}, torch_fill_data);

        auto lfs_fill_data = lfs_new_data.slice(0, 0, slots_to_fill);
        lfs_data.index_put_(lfs_target, lfs_fill_data);

        EXPECT_TRUE(tensors_close(lfs_data, torch_data, 1e-5f, 1e-6f, "FillSlots_Data"));

        // Mark as not free
        auto torch_false = torch::zeros({slots_to_fill}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
        torch_free_mask.index_put_({torch_target}, torch_false);

        auto lfs_false = Tensor::zeros_bool({static_cast<size_t>(slots_to_fill)}, Device::CUDA);
        lfs_free_mask.index_put_(lfs_target, lfs_false);

        EXPECT_TRUE(tensors_close(lfs_free_mask, torch_free_mask, 0, 0, "FillSlots_Mask"));
    }
}

TEST_F(DefaultStrategyTensorOpsTest, ActiveCountPattern) {
    // Test the active_count pattern:
    // active_region = free_mask.slice(0, 0, current_size)
    // free_count = active_region.sum_scalar()
    // active_count = current_size - free_count

    torch::manual_seed(987);
    size_t capacity = 200;
    size_t current_size = 150;

    auto torch_free_mask = torch::randint(0, 2, {static_cast<int64_t>(capacity)},
                                          torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    auto lfs_free_mask = from_torch(torch_free_mask);

    auto torch_active_region = torch_free_mask.slice(0, 0, current_size);
    auto lfs_active_region = lfs_free_mask.slice(0, 0, current_size);

    int64_t torch_free_count = torch_active_region.sum().item<int64_t>();
    auto lfs_free_count = static_cast<size_t>(lfs_active_region.sum_scalar());

    EXPECT_EQ(lfs_free_count, static_cast<size_t>(torch_free_count));

    size_t torch_active_count = current_size - torch_free_count;
    size_t lfs_active_count = current_size - lfs_free_count;

    EXPECT_EQ(lfs_active_count, torch_active_count);
}

TEST_F(DefaultStrategyTensorOpsTest, GetActiveIndicesPattern) {
    // Test get_active_indices pattern:
    // is_active = free_mask.slice(0, 0, current_size).logical_not()
    // return is_active.nonzero().squeeze(-1)

    torch::manual_seed(111);
    size_t current_size = 100;

    auto torch_free_mask = torch::randint(0, 2, {static_cast<int64_t>(current_size)},
                                          torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    auto lfs_free_mask = from_torch(torch_free_mask);

    auto torch_is_active = ~torch_free_mask;
    auto torch_active_indices = torch_is_active.nonzero().squeeze(-1);

    auto lfs_is_active = lfs_free_mask.logical_not();
    auto lfs_active_indices = lfs_is_active.nonzero().squeeze(-1);

    EXPECT_TRUE(tensors_close(lfs_active_indices, torch_active_indices, 0, 0, "ActiveIndices"));
}

// =============================================================================
// DENSIFICATION_INFO OPERATIONS
// Test the [2, N] tensor operations for gradient tracking
// =============================================================================

TEST_F(DefaultStrategyTensorOpsTest, DensificationInfo_SubscriptOperator) {
    // Pattern from grow_gs():
    // numer = _densification_info[1]  // Get row 1 (gradient norms)
    // denom = _densification_info[0]  // Get row 0 (counts)
    // grads = numer / denom.clamp_min(1.0f)

    torch::manual_seed(42);
    const int N = 100000;

    // Create [2, N] tensor simulating densification_info
    auto torch_dens_info = torch::zeros({2, N}, torch::kCUDA);
    // Row 0: counts (integers stored as float)
    torch_dens_info[0] = torch::randint(0, 100, {N}, torch::kCUDA).to(torch::kFloat32);
    // Row 1: accumulated gradient norms
    torch_dens_info[1] = torch::rand({N}, torch::kCUDA) * 0.1f;

    auto lfs_dens_info = from_torch(torch_dens_info);

    // Extract rows using subscript operator
    auto torch_denom = torch_dens_info[0];
    auto torch_numer = torch_dens_info[1];

    // Must use explicit Tensor type (not auto) to trigger conversion from TensorRowProxy
    Tensor lfs_denom = lfs_dens_info[0];
    Tensor lfs_numer = lfs_dens_info[1];

    EXPECT_TRUE(tensors_close(lfs_denom, torch_denom, 1e-6f, 1e-7f, "DensInfo_Denom"));
    EXPECT_TRUE(tensors_close(lfs_numer, torch_numer, 1e-6f, 1e-7f, "DensInfo_Numer"));

    // Compute gradients
    auto torch_grads = torch_numer / torch::clamp_min(torch_denom, 1.0f);
    auto lfs_grads = lfs_numer / lfs_denom.clamp_min(1.0f);

    EXPECT_TRUE(tensors_close(lfs_grads, torch_grads, 1e-5f, 1e-6f, "DensInfo_Grads"));
}

TEST_F(DefaultStrategyTensorOpsTest, DensificationInfo_FullGrowGsPattern) {
    // Complete pattern from grow_gs using densification_info:
    // 1. numer = densification_info[1], denom = densification_info[0]
    // 2. grads = numer / denom.clamp_min(1.0f)
    // 3. is_grad_high = grads > threshold
    // 4. is_grad_high = is_grad_high.logical_and(is_active)
    // 5. is_small = max_scale <= scale_threshold
    // 6. is_duplicated = is_grad_high.logical_and(is_small)
    // 7. is_split = is_grad_high.logical_and(is_small.logical_not())

    torch::manual_seed(42);
    const int N = 100000;

    // Densification info [2, N]
    auto torch_dens_info = torch::zeros({2, N}, torch::kCUDA);
    torch_dens_info[0] = torch::randint(1, 50, {N}, torch::kCUDA).to(torch::kFloat32); // counts
    torch_dens_info[1] = torch::rand({N}, torch::kCUDA) * 0.05f;                       // grad norms

    // Scales [N, 3]
    auto torch_scales = torch::randn({N, 3}, torch::kCUDA);

    // Free mask [N]
    auto torch_free_mask = torch::randint(0, 10, {N}, torch::kCUDA) == 0; // ~10% free

    float grad_threshold = 0.002f;
    float scale_threshold = 1.0f;

    auto lfs_dens_info = from_torch(torch_dens_info);
    auto lfs_scales = from_torch(torch_scales);
    auto lfs_free_mask = from_torch(torch_free_mask);

    // Torch implementation
    auto torch_denom = torch_dens_info[0];
    auto torch_numer = torch_dens_info[1];
    auto torch_grads = torch_numer / torch::clamp_min(torch_denom, 1.0f);
    auto torch_is_grad_high = torch_grads > grad_threshold;
    auto torch_is_active = ~torch_free_mask;
    torch_is_grad_high = torch_is_grad_high & torch_is_active;
    auto torch_max_scale = std::get<0>(torch::max(torch_scales, -1));
    auto torch_is_small = torch_max_scale <= scale_threshold;
    auto torch_is_duplicated = torch_is_grad_high & torch_is_small;
    auto torch_is_split = torch_is_grad_high & (~torch_is_small);

    // LFS implementation (explicit Tensor type triggers conversion from TensorRowProxy)
    Tensor lfs_denom = lfs_dens_info[0];
    Tensor lfs_numer = lfs_dens_info[1];
    auto lfs_grads = lfs_numer / lfs_denom.clamp_min(1.0f);
    auto lfs_is_grad_high = lfs_grads > grad_threshold;
    auto lfs_is_active = lfs_free_mask.logical_not();
    lfs_is_grad_high = lfs_is_grad_high.logical_and(lfs_is_active);
    auto lfs_max_scale = lfs_scales.max(-1, false);
    auto lfs_is_small = lfs_max_scale <= scale_threshold;
    auto lfs_is_duplicated = lfs_is_grad_high.logical_and(lfs_is_small);
    auto lfs_is_split = lfs_is_grad_high.logical_and(lfs_is_small.logical_not());

    EXPECT_TRUE(tensors_close(lfs_grads, torch_grads, 1e-5f, 1e-6f, "DensInfo_GrowGs_Grads"));
    EXPECT_TRUE(tensors_close(lfs_is_duplicated, torch_is_duplicated, 0, 0, "DensInfo_GrowGs_Dup"));
    EXPECT_TRUE(tensors_close(lfs_is_split, torch_is_split, 0, 0, "DensInfo_GrowGs_Split"));

    // Verify counts match
    EXPECT_EQ(static_cast<int64_t>(lfs_is_duplicated.sum_scalar()),
              torch_is_duplicated.sum().item<int64_t>());
    EXPECT_EQ(static_cast<int64_t>(lfs_is_split.sum_scalar()),
              torch_is_split.sum().item<int64_t>());
}

TEST_F(DefaultStrategyTensorOpsTest, DensificationInfo_ResetAfterRefine) {
    // Pattern from post_backward():
    // densification_info = Tensor::zeros({2, N}, device)
    // This resets the accumulator after refinement

    const int N = 50000;
    auto device = Device::CUDA;

    auto lfs_dens_info = Tensor::zeros({2, static_cast<size_t>(N)}, device);
    auto torch_dens_info = torch::zeros({2, N}, torch::kCUDA);

    EXPECT_TRUE(tensors_close(lfs_dens_info, torch_dens_info, 0, 0, "DensInfo_Reset"));
    EXPECT_EQ(lfs_dens_info.shape()[0], 2);
    EXPECT_EQ(lfs_dens_info.shape()[1], static_cast<size_t>(N));
}

// =============================================================================
// PRUNING OPERATIONS - COMPLETE TEST
// Verifies the full prune_gs -> remove chain works correctly
// =============================================================================

TEST_F(DefaultStrategyTensorOpsTest, Pruning_CompletePruneGsChain) {
    // Complete prune_gs chain:
    // 1. is_prune = get_opacity() < prune_opacity
    // 2. is_prune = is_prune.logical_or((rotation * rotation).sum(-1) < 1e-8f)
    // 3. is_prune = is_prune.logical_or(max_scale > prune_scale)  [if iter > reset_every]
    // 4. is_prune = is_prune.logical_and(is_active)  [exclude free slots]
    // 5. remove(is_prune) if count > 0

    torch::manual_seed(42);
    const int N = 100000;

    // Create test data
    auto torch_opacity = torch::randn({N}, torch::kCUDA); // raw logit opacity
    auto torch_rotation = torch::randn({N, 4}, torch::kCUDA);
    auto torch_scales = torch::randn({N, 3}, torch::kCUDA);
    auto torch_free_mask = torch::randint(0, 20, {N}, torch::kCUDA) == 0; // ~5% free

    // Set some rotations to near-zero (should be pruned)
    torch_rotation.index_put_({torch::arange(0, 100, torch::kCUDA)},
                              torch::zeros({100, 4}, torch::kCUDA));

    float prune_opacity = 0.01f;
    float prune_scale = 2.0f;

    auto lfs_opacity = from_torch(torch_opacity);
    auto lfs_rotation = from_torch(torch_rotation);
    auto lfs_scales = from_torch(torch_scales);
    auto lfs_free_mask = from_torch(torch_free_mask);

    // ===== TORCH IMPLEMENTATION =====
    // Step 1: Low opacity
    auto torch_is_prune = torch_opacity < prune_opacity;

    // Step 2: Invalid rotation (quaternion magnitude near zero)
    auto torch_rot_mag_sq = (torch_rotation * torch_rotation).sum(-1);
    torch_is_prune = torch_is_prune | (torch_rot_mag_sq < 1e-8f);

    // Step 3: Too large scale
    auto torch_max_scale = std::get<0>(torch::max(torch_scales, -1));
    torch_is_prune = torch_is_prune | (torch_max_scale > prune_scale);

    // Step 4: Exclude already-free slots
    auto torch_is_active = ~torch_free_mask;
    torch_is_prune = torch_is_prune & torch_is_active;

    // ===== LFS IMPLEMENTATION =====
    // Step 1: Low opacity
    auto lfs_is_prune = lfs_opacity < prune_opacity;

    // Step 2: Invalid rotation
    auto lfs_rot_mag_sq = (lfs_rotation * lfs_rotation).sum(-1, false);
    lfs_is_prune = lfs_is_prune.logical_or(lfs_rot_mag_sq < 1e-8f);

    // Step 3: Too large scale
    auto lfs_max_scale = lfs_scales.max(-1, false);
    lfs_is_prune = lfs_is_prune.logical_or(lfs_max_scale > prune_scale);

    // Step 4: Exclude already-free slots
    auto lfs_is_active = lfs_free_mask.logical_not();
    lfs_is_prune = lfs_is_prune.logical_and(lfs_is_active);

    // ===== VERIFY =====
    EXPECT_TRUE(tensors_close(lfs_is_prune, torch_is_prune, 0, 0, "Pruning_FinalMask"));

    int64_t torch_prune_count = torch_is_prune.sum().item<int64_t>();
    auto lfs_prune_count = static_cast<int64_t>(lfs_is_prune.sum_scalar());
    EXPECT_EQ(lfs_prune_count, torch_prune_count);

    // The near-zero rotations should be detected
    EXPECT_GT(lfs_prune_count, 50) << "Should prune at least 50 Gaussians (100 zero rotations minus free)";
}

TEST_F(DefaultStrategyTensorOpsTest, Pruning_RemoveAndZeroOptimizer) {
    // Complete remove() operation:
    // 1. prune_indices = is_prune.nonzero().squeeze(-1)
    // 2. mark_as_free(prune_indices)  // free_mask.index_put_(indices, true)
    // 3. rotation_raw.index_put_(prune_indices, zeros)  // Zero quaternion
    // 4. For each optimizer state: state.index_put_(prune_indices, zeros)

    torch::manual_seed(42);
    const int N = 10000;

    // Create test data
    auto torch_rotation = torch::randn({N, 4}, torch::kCUDA);
    auto torch_exp_avg_means = torch::randn({N, 3}, torch::kCUDA);
    auto torch_exp_avg_sq_means = torch::randn({N, 3}, torch::kCUDA).abs();
    auto torch_exp_avg_opacity = torch::randn({N}, torch::kCUDA);
    auto torch_free_mask = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

    // Create prune mask (~20% to prune)
    auto torch_is_prune = torch::randint(0, 5, {N}, torch::kCUDA) == 0;

    auto lfs_rotation = from_torch(torch_rotation);
    auto lfs_exp_avg_means = from_torch(torch_exp_avg_means);
    auto lfs_exp_avg_sq_means = from_torch(torch_exp_avg_sq_means);
    auto lfs_exp_avg_opacity = from_torch(torch_exp_avg_opacity);
    auto lfs_free_mask = from_torch(torch_free_mask);
    auto lfs_is_prune = from_torch(torch_is_prune);

    // Get indices
    auto torch_prune_indices = torch_is_prune.nonzero().squeeze(-1);
    auto lfs_prune_indices = lfs_is_prune.nonzero().squeeze(-1);

    int64_t num_pruned = torch_prune_indices.numel();
    ASSERT_GT(num_pruned, 0) << "Need some Gaussians to prune";

    // ===== TORCH IMPLEMENTATION =====
    // Mark as free
    auto torch_true_vals = torch::ones({num_pruned}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    torch_free_mask.index_put_({torch_prune_indices}, torch_true_vals);

    // Zero rotation
    auto torch_zero_rot = torch::zeros({num_pruned, 4}, torch::kCUDA);
    torch_rotation.index_put_({torch_prune_indices}, torch_zero_rot);

    // Zero optimizer states
    auto torch_zero_means = torch::zeros({num_pruned, 3}, torch::kCUDA);
    auto torch_zero_opacity = torch::zeros({num_pruned}, torch::kCUDA);
    torch_exp_avg_means.index_put_({torch_prune_indices}, torch_zero_means);
    torch_exp_avg_sq_means.index_put_({torch_prune_indices}, torch_zero_means);
    torch_exp_avg_opacity.index_put_({torch_prune_indices}, torch_zero_opacity);

    // ===== LFS IMPLEMENTATION =====
    // Mark as free
    auto lfs_true_vals = Tensor::ones_bool({static_cast<size_t>(num_pruned)}, Device::CUDA);
    lfs_free_mask.index_put_(lfs_prune_indices, lfs_true_vals);

    // Zero rotation
    auto lfs_zero_rot = Tensor::zeros({static_cast<size_t>(num_pruned), 4}, Device::CUDA);
    lfs_rotation.index_put_(lfs_prune_indices, lfs_zero_rot);

    // Zero optimizer states
    auto lfs_zero_means = Tensor::zeros({static_cast<size_t>(num_pruned), 3}, Device::CUDA);
    auto lfs_zero_opacity = Tensor::zeros({static_cast<size_t>(num_pruned)}, Device::CUDA);
    lfs_exp_avg_means.index_put_(lfs_prune_indices, lfs_zero_means);
    lfs_exp_avg_sq_means.index_put_(lfs_prune_indices, lfs_zero_means);
    lfs_exp_avg_opacity.index_put_(lfs_prune_indices, lfs_zero_opacity);

    // ===== VERIFY =====
    EXPECT_TRUE(tensors_close(lfs_free_mask, torch_free_mask, 0, 0, "Pruning_FreeMask"));
    EXPECT_TRUE(tensors_close(lfs_rotation, torch_rotation, 1e-6f, 1e-7f, "Pruning_Rotation"));
    EXPECT_TRUE(tensors_close(lfs_exp_avg_means, torch_exp_avg_means, 1e-6f, 1e-7f, "Pruning_ExpAvgMeans"));
    EXPECT_TRUE(tensors_close(lfs_exp_avg_sq_means, torch_exp_avg_sq_means, 1e-6f, 1e-7f, "Pruning_ExpAvgSqMeans"));
    EXPECT_TRUE(tensors_close(lfs_exp_avg_opacity, torch_exp_avg_opacity, 1e-6f, 1e-7f, "Pruning_ExpAvgOpacity"));

    // Verify pruned rotations are zero
    auto lfs_rot_torch = to_torch(lfs_rotation);
    auto pruned_rotations = lfs_rot_torch.index({torch_prune_indices});
    EXPECT_TRUE((pruned_rotations == 0).all().item<bool>()) << "Pruned rotations should be zero";

    // Verify free mask is set
    auto lfs_free_torch = to_torch(lfs_free_mask);
    auto pruned_free = lfs_free_torch.index({torch_prune_indices});
    EXPECT_TRUE(pruned_free.all().item<bool>()) << "Pruned slots should be marked as free";
}

TEST_F(DefaultStrategyTensorOpsTest, Pruning_LargeScale) {
    // Large-scale pruning test with 1M Gaussians
    torch::manual_seed(42);
    const int N = 1000000;

    auto torch_opacity = torch::randn({N}, torch::kCUDA);
    auto torch_rotation = torch::randn({N, 4}, torch::kCUDA);
    auto torch_free_mask = torch::randint(0, 50, {N}, torch::kCUDA) == 0; // ~2% free

    float prune_opacity = 0.01f;

    auto lfs_opacity = from_torch(torch_opacity);
    auto lfs_rotation = from_torch(torch_rotation);
    auto lfs_free_mask = from_torch(torch_free_mask);

    // Full prune mask calculation
    auto torch_is_prune = torch_opacity < prune_opacity;
    auto torch_rot_mag = (torch_rotation * torch_rotation).sum(-1);
    torch_is_prune = torch_is_prune | (torch_rot_mag < 1e-8f);
    torch_is_prune = torch_is_prune & (~torch_free_mask);

    auto lfs_is_prune = lfs_opacity < prune_opacity;
    auto lfs_rot_mag = (lfs_rotation * lfs_rotation).sum(-1, false);
    lfs_is_prune = lfs_is_prune.logical_or(lfs_rot_mag < 1e-8f);
    lfs_is_prune = lfs_is_prune.logical_and(lfs_free_mask.logical_not());

    EXPECT_TRUE(tensors_close(lfs_is_prune, torch_is_prune, 0, 0, "Pruning_LargeScale_Mask"));

    // Verify counts
    int64_t torch_count = torch_is_prune.sum().item<int64_t>();
    auto lfs_count = static_cast<int64_t>(lfs_is_prune.sum_scalar());
    EXPECT_EQ(lfs_count, torch_count);
}

// =============================================================================
// EXACT COMBINATION TESTS FROM DEFAULT_STRATEGY.CPP
// These test the precise sequences used in actual training code
// =============================================================================

TEST_F(DefaultStrategyTensorOpsTest, FillFreeSlots_IndexPutIndexSelect) {
    // Exact pattern from fill_free_slots():
    // means().index_put_(target_indices, means().index_select(0, src_indices))
    // This is a scatter operation where we gather from one set of indices
    // and scatter to another set of indices

    torch::manual_seed(42);
    const int N = 10000;
    const int num_free = 500;
    const int num_sources = 500;

    auto torch_data = torch::randn({N, 3}, torch::kCUDA);
    // Target indices (where to write) - simulate free slots
    auto torch_target = torch::randperm(N, torch::kCUDA).slice(0, 0, num_free).to(torch::kInt64);
    // Source indices (where to read from) - simulate source Gaussians
    auto torch_source = torch::randperm(N, torch::kCUDA).slice(0, 0, num_sources).to(torch::kInt64);

    auto lfs_data = from_torch(torch_data);
    auto lfs_target = from_torch(torch_target);
    auto lfs_source = from_torch(torch_source);

    // Apply the pattern: index_put_(target, index_select(0, source))
    auto torch_selected = torch_data.index_select(0, torch_source);
    torch_data.index_put_({torch_target}, torch_selected);

    auto lfs_selected = lfs_data.index_select(0, lfs_source);
    lfs_data.index_put_(lfs_target, lfs_selected);

    EXPECT_TRUE(tensors_close(lfs_data, torch_data, 1e-5f, 1e-6f, "FillFreeSlots_IndexPutIndexSelect"));
}

TEST_F(DefaultStrategyTensorOpsTest, Split_AppendZerosThenIndexPut) {
    // Exact pattern from split():
    // means().append_zeros(n_remaining);
    // means().index_put_(new_indices, append_positions);
    // This extends the tensor then writes to the new positions

    torch::manual_seed(42);
    const int original_size = 1000;
    const int n_remaining = 100;

    auto torch_data = torch::randn({original_size, 3}, torch::kCUDA);
    auto torch_new_data = torch::randn({n_remaining, 3}, torch::kCUDA);

    auto lfs_data = from_torch(torch_data);
    auto lfs_new_data = from_torch(torch_new_data);

    // Reserve capacity for append
    lfs_data.reserve(original_size + n_remaining + 100);

    // Create new indices [original_size, original_size+1, ..., original_size+n_remaining-1]
    std::vector<int> new_indices_vec(n_remaining);
    for (int i = 0; i < n_remaining; ++i) {
        new_indices_vec[i] = original_size + i;
    }
    auto torch_new_indices = torch::tensor(std::vector<int64_t>(new_indices_vec.begin(), new_indices_vec.end()),
                                           torch::kCUDA);
    auto lfs_new_indices = Tensor::from_vector(new_indices_vec,
                                               TensorShape({static_cast<size_t>(n_remaining)}),
                                               Device::CUDA);

    // Torch: cat then index_put (equivalent to append_zeros + index_put)
    auto torch_zeros = torch::zeros({n_remaining, 3}, torch::kCUDA);
    auto torch_extended = torch::cat({torch_data, torch_zeros}, 0);
    torch_extended.index_put_({torch_new_indices}, torch_new_data);

    // LFS: append_zeros then index_put
    lfs_data.append_zeros(n_remaining);
    lfs_data.index_put_(lfs_new_indices, lfs_new_data);

    EXPECT_TRUE(tensors_close(lfs_data, torch_extended, 1e-5f, 1e-6f, "Split_AppendZerosThenIndexPut"));
}

TEST_F(DefaultStrategyTensorOpsTest, GrowGs_FullMaskCalculation) {
    // Full grow_gs mask calculation:
    // grads = numer / denom.clamp_min(1.0f)
    // is_grad_high = grads > threshold
    // is_grad_high = is_grad_high.logical_and(is_active)
    // is_small = max_values <= scale_threshold
    // is_duplicated = is_grad_high.logical_and(is_small)
    // is_large = is_small.logical_not()
    // is_split = is_grad_high.logical_and(is_large)

    torch::manual_seed(42);
    const int N = 100000;

    auto torch_numer = torch::rand({N}, torch::kCUDA) * 0.01f;
    auto torch_denom = torch::randint(0, 10, {N}, torch::kCUDA).to(torch::kFloat32);
    auto torch_scales = torch::randn({N, 3}, torch::kCUDA);
    auto torch_free_mask = torch::randint(0, 5, {N}, torch::kCUDA) == 0; // ~20% free

    float grad_threshold = 0.002f;
    float scale_threshold = 1.0f;

    auto lfs_numer = from_torch(torch_numer);
    auto lfs_denom = from_torch(torch_denom);
    auto lfs_scales = from_torch(torch_scales);
    auto lfs_free_mask = from_torch(torch_free_mask);

    // Torch implementation
    auto torch_grads = torch_numer / torch::clamp_min(torch_denom, 1.0f);
    auto torch_is_grad_high = torch_grads > grad_threshold;
    auto torch_is_active = ~torch_free_mask;
    torch_is_grad_high = torch_is_grad_high & torch_is_active;
    auto torch_max_values = std::get<0>(torch::max(torch_scales, -1));
    auto torch_is_small = torch_max_values <= scale_threshold;
    auto torch_is_duplicated = torch_is_grad_high & torch_is_small;
    auto torch_is_large = ~torch_is_small;
    auto torch_is_split = torch_is_grad_high & torch_is_large;

    // LFS implementation
    auto lfs_grads = lfs_numer / lfs_denom.clamp_min(1.0f);
    auto lfs_is_grad_high = lfs_grads > grad_threshold;
    auto lfs_is_active = lfs_free_mask.logical_not();
    lfs_is_grad_high = lfs_is_grad_high.logical_and(lfs_is_active);
    auto lfs_max_values = lfs_scales.max(-1, false);
    auto lfs_is_small = lfs_max_values <= scale_threshold;
    auto lfs_is_duplicated = lfs_is_grad_high.logical_and(lfs_is_small);
    auto lfs_is_large = lfs_is_small.logical_not();
    auto lfs_is_split = lfs_is_grad_high.logical_and(lfs_is_large);

    EXPECT_TRUE(tensors_close(lfs_grads, torch_grads, 1e-5f, 1e-6f, "GrowGs_Grads"));
    EXPECT_TRUE(tensors_close(lfs_is_duplicated, torch_is_duplicated, 0, 0, "GrowGs_IsDuplicated"));
    EXPECT_TRUE(tensors_close(lfs_is_split, torch_is_split, 0, 0, "GrowGs_IsSplit"));

    // Verify counts match
    int64_t torch_dup_count = torch_is_duplicated.sum().item<int64_t>();
    int64_t torch_split_count = torch_is_split.sum().item<int64_t>();
    auto lfs_dup_count = static_cast<int64_t>(lfs_is_duplicated.sum_scalar());
    auto lfs_split_count = static_cast<int64_t>(lfs_is_split.sum_scalar());

    EXPECT_EQ(lfs_dup_count, torch_dup_count);
    EXPECT_EQ(lfs_split_count, torch_split_count);
}

TEST_F(DefaultStrategyTensorOpsTest, PruneGs_FullMaskCalculation) {
    // Full prune_gs mask calculation:
    // is_prune = opacity < threshold
    // is_prune = is_prune.logical_or((rotation * rotation).sum(-1) < 1e-8f)
    // is_prune = is_prune.logical_or(max_scale > scale_threshold)
    // is_prune = is_prune.logical_and(is_active)

    torch::manual_seed(42);
    const int N = 100000;

    auto torch_opacity = torch::randn({N}, torch::kCUDA);
    auto torch_rotation = torch::randn({N, 4}, torch::kCUDA);
    auto torch_scales = torch::randn({N, 3}, torch::kCUDA);
    auto torch_free_mask = torch::randint(0, 10, {N}, torch::kCUDA) == 0; // ~10% free

    float opacity_threshold = 0.01f;
    float scale_threshold = 2.0f;

    auto lfs_opacity = from_torch(torch_opacity);
    auto lfs_rotation = from_torch(torch_rotation);
    auto lfs_scales = from_torch(torch_scales);
    auto lfs_free_mask = from_torch(torch_free_mask);

    // Torch implementation
    auto torch_is_prune = torch_opacity < opacity_threshold;
    auto torch_rot_mag = (torch_rotation * torch_rotation).sum(-1);
    torch_is_prune = torch_is_prune | (torch_rot_mag < 1e-8f);
    auto torch_max_scale = std::get<0>(torch::max(torch_scales, -1));
    torch_is_prune = torch_is_prune | (torch_max_scale > scale_threshold);
    auto torch_is_active = ~torch_free_mask;
    torch_is_prune = torch_is_prune & torch_is_active;

    // LFS implementation
    auto lfs_is_prune = lfs_opacity < opacity_threshold;
    auto lfs_rot_mag = (lfs_rotation * lfs_rotation).sum(-1, false);
    lfs_is_prune = lfs_is_prune.logical_or(lfs_rot_mag < 1e-8f);
    auto lfs_max_scale = lfs_scales.max(-1, false);
    lfs_is_prune = lfs_is_prune.logical_or(lfs_max_scale > scale_threshold);
    auto lfs_is_active = lfs_free_mask.logical_not();
    lfs_is_prune = lfs_is_prune.logical_and(lfs_is_active);

    EXPECT_TRUE(tensors_close(lfs_is_prune, torch_is_prune, 0, 0, "PruneGs_Final"));

    // Verify counts match
    int64_t torch_prune_count = torch_is_prune.sum().item<int64_t>();
    auto lfs_prune_count = static_cast<int64_t>(lfs_is_prune.sum_scalar());
    EXPECT_EQ(lfs_prune_count, torch_prune_count);
}

TEST_F(DefaultStrategyTensorOpsTest, Remove_ZeroRotationAndOptimizerState) {
    // Pattern from remove():
    // prune_indices = is_prune.nonzero().squeeze(-1)
    // rotation_raw.index_put_(prune_indices, zeros)
    // exp_avg.index_put_(prune_indices, zeros)
    // exp_avg_sq.index_put_(prune_indices, zeros)

    torch::manual_seed(42);
    const int N = 10000;

    auto torch_rotation = torch::randn({N, 4}, torch::kCUDA);
    auto torch_exp_avg = torch::randn({N, 3}, torch::kCUDA);
    auto torch_exp_avg_sq = torch::randn({N, 3}, torch::kCUDA).abs();
    auto torch_is_prune = torch::randint(0, 5, {N}, torch::kCUDA) == 0; // ~20% prune

    auto lfs_rotation = from_torch(torch_rotation);
    auto lfs_exp_avg = from_torch(torch_exp_avg);
    auto lfs_exp_avg_sq = from_torch(torch_exp_avg_sq);
    auto lfs_is_prune = from_torch(torch_is_prune);

    // Get prune indices
    auto torch_prune_indices = torch_is_prune.nonzero().squeeze(-1);
    auto lfs_prune_indices = lfs_is_prune.nonzero().squeeze(-1);

    int64_t num_pruned = torch_prune_indices.numel();
    if (num_pruned > 0) {
        // Create zero tensors
        auto torch_zero_rot = torch::zeros({num_pruned, 4}, torch::kCUDA);
        auto torch_zero_state = torch::zeros({num_pruned, 3}, torch::kCUDA);

        auto lfs_zero_rot = Tensor::zeros({static_cast<size_t>(num_pruned), 4}, Device::CUDA);
        auto lfs_zero_state = Tensor::zeros({static_cast<size_t>(num_pruned), 3}, Device::CUDA);

        // Apply zero operations
        torch_rotation.index_put_({torch_prune_indices}, torch_zero_rot);
        torch_exp_avg.index_put_({torch_prune_indices}, torch_zero_state);
        torch_exp_avg_sq.index_put_({torch_prune_indices}, torch_zero_state);

        lfs_rotation.index_put_(lfs_prune_indices, lfs_zero_rot);
        lfs_exp_avg.index_put_(lfs_prune_indices, lfs_zero_state);
        lfs_exp_avg_sq.index_put_(lfs_prune_indices, lfs_zero_state);

        EXPECT_TRUE(tensors_close(lfs_rotation, torch_rotation, 1e-5f, 1e-6f, "Remove_Rotation"));
        EXPECT_TRUE(tensors_close(lfs_exp_avg, torch_exp_avg, 1e-5f, 1e-6f, "Remove_ExpAvg"));
        EXPECT_TRUE(tensors_close(lfs_exp_avg_sq, torch_exp_avg_sq, 1e-5f, 1e-6f, "Remove_ExpAvgSq"));
    }
}

TEST_F(DefaultStrategyTensorOpsTest, ResetOpacity_ClampAndZero) {
    // Pattern from reset_opacity():
    // opacity_raw.clamp_max_(logit_threshold)
    // exp_avg.zero_()
    // exp_avg_sq.zero_()

    torch::manual_seed(42);
    const int N = 10000;

    auto torch_opacity = torch::randn({N}, torch::kCUDA) * 5.0f; // Wide range
    auto torch_exp_avg = torch::randn({N}, torch::kCUDA);
    auto torch_exp_avg_sq = torch::randn({N}, torch::kCUDA).abs();

    auto lfs_opacity = from_torch(torch_opacity);
    auto lfs_exp_avg = from_torch(torch_exp_avg);
    auto lfs_exp_avg_sq = from_torch(torch_exp_avg_sq);

    float logit_threshold = -2.0f;

    // Apply operations
    torch_opacity.clamp_max_(logit_threshold);
    torch_exp_avg.zero_();
    torch_exp_avg_sq.zero_();

    lfs_opacity.clamp_max_(logit_threshold);
    lfs_exp_avg.zero_();
    lfs_exp_avg_sq.zero_();

    EXPECT_TRUE(tensors_close(lfs_opacity, torch_opacity, 1e-6f, 1e-7f, "ResetOpacity_Clamp"));
    EXPECT_TRUE(tensors_close(lfs_exp_avg, torch_exp_avg, 0, 0, "ResetOpacity_ZeroAvg"));
    EXPECT_TRUE(tensors_close(lfs_exp_avg_sq, torch_exp_avg_sq, 0, 0, "ResetOpacity_ZeroSq"));
}

TEST_F(DefaultStrategyTensorOpsTest, MaxCapEnforcement_LimitMaskBySlice) {
    // Pattern from grow_gs max_cap enforcement:
    // indices = is_duplicated.nonzero().squeeze(-1)
    // keep_indices = indices.slice(0, 0, available)
    // is_duplicated = zeros_bool
    // is_duplicated.index_put_(keep_indices, ones_bool)

    torch::manual_seed(42);
    const int N = 10000;
    const int max_cap = 10500;
    const int available = max_cap - N; // 500

    auto torch_is_duplicated = torch::randint(0, 5, {N}, torch::kCUDA) == 0; // ~20% = 2000
    auto lfs_is_duplicated = from_torch(torch_is_duplicated);

    auto torch_indices = torch_is_duplicated.nonzero().squeeze(-1);
    auto lfs_indices = lfs_is_duplicated.nonzero().squeeze(-1);

    int64_t num_candidates = torch_indices.numel();

    if (num_candidates > available) {
        // Limit to available
        auto torch_keep = torch_indices.slice(0, 0, available);
        auto lfs_keep = lfs_indices.slice(0, 0, available);

        // Create limited mask
        auto torch_limited = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
        auto torch_true_vals = torch::ones({available}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
        torch_limited.index_put_({torch_keep}, torch_true_vals);

        auto lfs_limited = Tensor::zeros_bool({static_cast<size_t>(N)}, Device::CUDA);
        auto lfs_true_vals = Tensor::ones_bool({static_cast<size_t>(available)}, Device::CUDA);
        lfs_limited.index_put_(lfs_keep, lfs_true_vals);

        EXPECT_TRUE(tensors_close(lfs_limited, torch_limited, 0, 0, "MaxCap_LimitedMask"));

        // Verify exactly 'available' true values
        EXPECT_EQ(torch_limited.sum().item<int>(), available);
        EXPECT_EQ(static_cast<int>(lfs_limited.sum_scalar()), available);
    }
}

// =============================================================================
// STRESS TESTS
// Test operations with large tensors (realistic training sizes)
// =============================================================================

TEST_F(DefaultStrategyTensorOpsTest, LargeScale_Densification) {
    // Test with 1M Gaussians (realistic training size)
    const int N = 1000000;

    torch::manual_seed(42);

    auto torch_grads = torch::randn({N}, torch::kCUDA).abs();
    auto torch_scales = torch::randn({N, 3}, torch::kCUDA);

    auto lfs_grads = from_torch(torch_grads);
    auto lfs_scales = from_torch(torch_scales);

    float grad_threshold = 0.5f;
    float scale_threshold = 1.0f;

    // Run grow_gs pattern
    auto torch_is_grad_high = torch_grads > grad_threshold;
    auto torch_max_scales = std::get<0>(torch::max(torch_scales, -1));
    auto torch_is_small = torch_max_scales <= scale_threshold;
    auto torch_is_duplicated = torch_is_grad_high & torch_is_small;

    auto lfs_is_grad_high = lfs_grads > grad_threshold;
    auto lfs_max_scales = lfs_scales.max(-1, false);
    auto lfs_is_small = lfs_max_scales <= scale_threshold;
    auto lfs_is_duplicated = lfs_is_grad_high.logical_and(lfs_is_small);

    // Compare results
    EXPECT_TRUE(tensors_close(lfs_is_duplicated, torch_is_duplicated, 0, 0, "LargeScale_Mask"));

    // Compare counts
    int64_t torch_count = torch_is_duplicated.sum().item<int64_t>();
    auto lfs_count = static_cast<int64_t>(lfs_is_duplicated.sum_scalar());
    EXPECT_EQ(lfs_count, torch_count);
}

TEST_F(DefaultStrategyTensorOpsTest, LargeScale_IndexSelect) {
    // Test index_select with large tensors
    const int N = 500000;
    const int num_select = 50000;

    torch::manual_seed(42);

    auto torch_data = torch::randn({N, 4}, torch::kCUDA);
    auto torch_indices = torch::randint(0, N, {num_select}, torch::kCUDA).to(torch::kInt64);

    auto lfs_data = from_torch(torch_data);
    auto lfs_indices = from_torch(torch_indices);

    auto torch_result = torch_data.index_select(0, torch_indices);
    auto lfs_result = lfs_data.index_select(0, lfs_indices);

    EXPECT_TRUE(tensors_close(lfs_result, torch_result, 1e-5f, 1e-6f, "LargeScale_IndexSelect"));
}

TEST_F(DefaultStrategyTensorOpsTest, LargeScale_IndexPut) {
    // Test index_put with large tensors (realistic training scale)
    const int N = 500000;
    const int num_put = 50000;

    torch::manual_seed(42);

    // Generate data and indices in torch
    auto torch_data = torch::randn({N, 3}, torch::kCUDA);
    auto torch_indices = std::get<0>(torch::randperm(N, torch::kCUDA).slice(0, 0, num_put).sort());
    torch_indices = torch_indices.to(torch::kInt64);
    auto torch_values = torch::zeros({num_put, 3}, torch::kCUDA);

    // Verify indices are valid and unique
    ASSERT_EQ(torch_indices.numel(), num_put);
    ASSERT_TRUE((torch_indices >= 0).all().item<bool>());
    ASSERT_TRUE((torch_indices < N).all().item<bool>());

    // Convert data to lfs format BEFORE any modifications
    auto lfs_data = from_torch(torch_data);

    // Verify initial data matches
    ASSERT_TRUE(tensors_close(lfs_data, torch_data, 1e-6f, 1e-7f, "Initial data"))
        << "Data conversion failed before index_put";

    // Convert indices - create directly as Int64 to avoid conversion issues
    auto lfs_indices = from_torch(torch_indices);

    // Verify indices match
    auto lfs_indices_torch = to_torch(lfs_indices);
    ASSERT_TRUE((lfs_indices_torch == torch_indices).all().item<bool>())
        << "Index conversion mismatch";

    // Create zero values
    auto lfs_values = Tensor::zeros({static_cast<size_t>(num_put), 3}, Device::CUDA);

    // Apply index_put on both
    torch_data.index_put_({torch_indices}, torch_values);
    lfs_data.index_put_(lfs_indices, lfs_values);

    // Compare results
    EXPECT_TRUE(tensors_close(lfs_data, torch_data, 1e-5f, 1e-6f, "LargeScale_IndexPut"));

    // Verify zeroed rows in lfs result
    auto lfs_torch = to_torch(lfs_data);
    auto zeroed = lfs_torch.index({torch_indices});
    EXPECT_TRUE((zeroed == 0).all().item<bool>()) << "Some indexed rows were not zeroed";
}
