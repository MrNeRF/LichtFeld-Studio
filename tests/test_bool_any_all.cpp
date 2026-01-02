/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * Comprehensive tests for Bool tensor any/all operations.
 * Compares LFS implementation against LibTorch for correctness.
 */

#include "core/tensor.hpp"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <random>

using namespace lfs::core;

class BoolAnyAllTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Seed for reproducibility
        gen_.seed(42);
    }

    std::mt19937 gen_;

    // Helper to create random bool tensor
    Tensor random_bool_tensor(std::vector<size_t> shape, float true_probability = 0.5f) {
        size_t numel = 1;
        for (auto s : shape) numel *= s;

        std::vector<unsigned char> data(numel);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < numel; ++i) {
            data[i] = dist(gen_) < true_probability ? 1 : 0;
        }

        return Tensor::from_blob(data.data(), TensorShape(shape), Device::CPU, DataType::Bool).clone();
    }

    // Helper to convert LFS Bool tensor to Torch bool tensor
    torch::Tensor lfs_to_torch_bool(const Tensor& t) {
        auto cpu_tensor = t.to(Device::CPU);
        std::vector<int64_t> sizes;
        for (size_t i = 0; i < cpu_tensor.shape().rank(); ++i) {
            sizes.push_back(static_cast<int64_t>(cpu_tensor.shape()[i]));
        }

        auto options = torch::TensorOptions().dtype(torch::kBool);
        auto result = torch::empty(sizes, options);

        // Copy data
        const unsigned char* src = cpu_tensor.ptr<unsigned char>();
        bool* dst = result.data_ptr<bool>();
        for (size_t i = 0; i < cpu_tensor.numel(); ++i) {
            dst[i] = src[i] != 0;
        }

        return result;
    }

    // Helper to compare LFS Bool result with Torch bool result
    void compare_bool_tensors(const Tensor& lfs, const torch::Tensor& torch_ref, const std::string& context) {
        auto lfs_cpu = lfs.to(Device::CPU);
        auto torch_cpu = torch_ref.to(torch::kCPU).contiguous();

        // Compare shapes
        ASSERT_EQ(lfs_cpu.shape().rank(), static_cast<size_t>(torch_cpu.dim()))
            << context << ": rank mismatch";

        for (size_t i = 0; i < lfs_cpu.shape().rank(); ++i) {
            ASSERT_EQ(lfs_cpu.shape()[i], static_cast<size_t>(torch_cpu.size(i)))
                << context << ": shape mismatch at dim " << i;
        }

        // Compare values
        const unsigned char* lfs_data = lfs_cpu.ptr<unsigned char>();
        const bool* torch_data = torch_cpu.data_ptr<bool>();

        for (size_t i = 0; i < lfs_cpu.numel(); ++i) {
            bool lfs_val = lfs_data[i] != 0;
            bool torch_val = torch_data[i];
            EXPECT_EQ(lfs_val, torch_val)
                << context << ": value mismatch at index " << i
                << " (LFS: " << lfs_val << ", Torch: " << torch_val << ")";
        }
    }
};

// ============= Full Reduction Tests =============

TEST_F(BoolAnyAllTest, Any_FullReduction_AllFalse) {
    auto lfs_tensor = Tensor::zeros({3, 4, 5}, Device::CPU, DataType::Bool);
    auto torch_tensor = torch::zeros({3, 4, 5}, torch::kBool);

    auto lfs_result = lfs_tensor.any();
    auto torch_result = torch_tensor.any();

    EXPECT_FALSE(lfs_result.item() != 0) << "LFS any() should be false for all-zeros";
    EXPECT_FALSE(torch_result.item<bool>()) << "Torch any() should be false for all-zeros";
}

TEST_F(BoolAnyAllTest, Any_FullReduction_AllTrue) {
    auto lfs_tensor = Tensor::ones({3, 4, 5}, Device::CPU, DataType::Bool);
    auto torch_tensor = torch::ones({3, 4, 5}, torch::kBool);

    auto lfs_result = lfs_tensor.any();
    auto torch_result = torch_tensor.any();

    EXPECT_TRUE(lfs_result.item() != 0) << "LFS any() should be true for all-ones";
    EXPECT_TRUE(torch_result.item<bool>()) << "Torch any() should be true for all-ones";
}

TEST_F(BoolAnyAllTest, Any_FullReduction_OneTrue) {
    auto lfs_tensor = Tensor::zeros({3, 4, 5}, Device::CPU, DataType::Bool);
    lfs_tensor.set_bool({1, 2, 3}, true);

    auto torch_tensor = torch::zeros({3, 4, 5}, torch::kBool);
    torch_tensor.index_put_({1, 2, 3}, true);

    auto lfs_result = lfs_tensor.any();
    auto torch_result = torch_tensor.any();

    EXPECT_TRUE(lfs_result.item() != 0) << "LFS any() should be true with one true value";
    EXPECT_TRUE(torch_result.item<bool>()) << "Torch any() should be true with one true value";
}

TEST_F(BoolAnyAllTest, All_FullReduction_AllFalse) {
    auto lfs_tensor = Tensor::zeros({3, 4, 5}, Device::CPU, DataType::Bool);
    auto torch_tensor = torch::zeros({3, 4, 5}, torch::kBool);

    auto lfs_result = lfs_tensor.all();
    auto torch_result = torch_tensor.all();

    EXPECT_FALSE(lfs_result.item() != 0) << "LFS all() should be false for all-zeros";
    EXPECT_FALSE(torch_result.item<bool>()) << "Torch all() should be false for all-zeros";
}

TEST_F(BoolAnyAllTest, All_FullReduction_AllTrue) {
    auto lfs_tensor = Tensor::ones({3, 4, 5}, Device::CPU, DataType::Bool);
    auto torch_tensor = torch::ones({3, 4, 5}, torch::kBool);

    auto lfs_result = lfs_tensor.all();
    auto torch_result = torch_tensor.all();

    EXPECT_TRUE(lfs_result.item() != 0) << "LFS all() should be true for all-ones";
    EXPECT_TRUE(torch_result.item<bool>()) << "Torch all() should be true for all-ones";
}

TEST_F(BoolAnyAllTest, All_FullReduction_OneFalse) {
    auto lfs_tensor = Tensor::ones({3, 4, 5}, Device::CPU, DataType::Bool);
    lfs_tensor.set_bool({1, 2, 3}, false);

    auto torch_tensor = torch::ones({3, 4, 5}, torch::kBool);
    torch_tensor.index_put_({1, 2, 3}, false);

    auto lfs_result = lfs_tensor.all();
    auto torch_result = torch_tensor.all();

    EXPECT_FALSE(lfs_result.item() != 0) << "LFS all() should be false with one false value";
    EXPECT_FALSE(torch_result.item<bool>()) << "Torch all() should be false with one false value";
}

// ============= Axis-Specific Reduction Tests =============

TEST_F(BoolAnyAllTest, Any_Dim0_2D) {
    // Create a 3x4 tensor with specific pattern
    auto lfs_tensor = Tensor::zeros({3, 4}, Device::CPU, DataType::Bool);
    lfs_tensor.set_bool({0, 1}, true);  // Column 1 has at least one true
    lfs_tensor.set_bool({2, 3}, true);  // Column 3 has at least one true

    auto torch_tensor = torch::zeros({3, 4}, torch::kBool);
    torch_tensor.index_put_({0, 1}, true);
    torch_tensor.index_put_({2, 3}, true);

    auto lfs_result = lfs_tensor.any(0);       // Reduce along rows -> [4]
    auto torch_result = torch_tensor.any(0);   // Should give [false, true, false, true]

    compare_bool_tensors(lfs_result, torch_result, "any(dim=0) 2D");
}

TEST_F(BoolAnyAllTest, Any_Dim1_2D) {
    auto lfs_tensor = Tensor::zeros({3, 4}, Device::CPU, DataType::Bool);
    lfs_tensor.set_bool({0, 2}, true);  // Row 0 has at least one true
    lfs_tensor.set_bool({1, 0}, true);  // Row 1 has at least one true
    // Row 2 has no true values

    auto torch_tensor = torch::zeros({3, 4}, torch::kBool);
    torch_tensor.index_put_({0, 2}, true);
    torch_tensor.index_put_({1, 0}, true);

    auto lfs_result = lfs_tensor.any(1);       // Reduce along columns -> [3]
    auto torch_result = torch_tensor.any(1);   // Should give [true, true, false]

    compare_bool_tensors(lfs_result, torch_result, "any(dim=1) 2D");
}

TEST_F(BoolAnyAllTest, All_Dim0_2D) {
    auto lfs_tensor = Tensor::ones({3, 4}, Device::CPU, DataType::Bool);
    lfs_tensor.set_bool({1, 0}, false);  // Column 0 has a false
    lfs_tensor.set_bool({0, 2}, false);  // Column 2 has a false

    auto torch_tensor = torch::ones({3, 4}, torch::kBool);
    torch_tensor.index_put_({1, 0}, false);
    torch_tensor.index_put_({0, 2}, false);

    auto lfs_result = lfs_tensor.all(0);       // Reduce along rows -> [4]
    auto torch_result = torch_tensor.all(0);   // Should give [false, true, false, true]

    compare_bool_tensors(lfs_result, torch_result, "all(dim=0) 2D");
}

TEST_F(BoolAnyAllTest, All_Dim1_2D) {
    auto lfs_tensor = Tensor::ones({3, 4}, Device::CPU, DataType::Bool);
    lfs_tensor.set_bool({0, 1}, false);  // Row 0 has a false
    // Row 1 all true
    lfs_tensor.set_bool({2, 3}, false);  // Row 2 has a false

    auto torch_tensor = torch::ones({3, 4}, torch::kBool);
    torch_tensor.index_put_({0, 1}, false);
    torch_tensor.index_put_({2, 3}, false);

    auto lfs_result = lfs_tensor.all(1);       // Reduce along columns -> [3]
    auto torch_result = torch_tensor.all(1);   // Should give [false, true, false]

    compare_bool_tensors(lfs_result, torch_result, "all(dim=1) 2D");
}

TEST_F(BoolAnyAllTest, Any_Dim0_3D) {
    auto lfs_tensor = Tensor::zeros({2, 3, 4}, Device::CPU, DataType::Bool);
    lfs_tensor.set_bool({0, 1, 2}, true);
    lfs_tensor.set_bool({1, 0, 3}, true);

    auto torch_tensor = torch::zeros({2, 3, 4}, torch::kBool);
    torch_tensor.index_put_({0, 1, 2}, true);
    torch_tensor.index_put_({1, 0, 3}, true);

    auto lfs_result = lfs_tensor.any(0);       // -> [3, 4]
    auto torch_result = torch_tensor.any(0);

    compare_bool_tensors(lfs_result, torch_result, "any(dim=0) 3D");
}

TEST_F(BoolAnyAllTest, Any_Dim1_3D) {
    auto lfs_tensor = Tensor::zeros({2, 3, 4}, Device::CPU, DataType::Bool);
    lfs_tensor.set_bool({0, 1, 2}, true);
    lfs_tensor.set_bool({1, 2, 0}, true);

    auto torch_tensor = torch::zeros({2, 3, 4}, torch::kBool);
    torch_tensor.index_put_({0, 1, 2}, true);
    torch_tensor.index_put_({1, 2, 0}, true);

    auto lfs_result = lfs_tensor.any(1);       // -> [2, 4]
    auto torch_result = torch_tensor.any(1);

    compare_bool_tensors(lfs_result, torch_result, "any(dim=1) 3D");
}

TEST_F(BoolAnyAllTest, Any_Dim2_3D) {
    auto lfs_tensor = Tensor::zeros({2, 3, 4}, Device::CPU, DataType::Bool);
    lfs_tensor.set_bool({0, 0, 1}, true);
    lfs_tensor.set_bool({1, 2, 3}, true);

    auto torch_tensor = torch::zeros({2, 3, 4}, torch::kBool);
    torch_tensor.index_put_({0, 0, 1}, true);
    torch_tensor.index_put_({1, 2, 3}, true);

    auto lfs_result = lfs_tensor.any(2);       // -> [2, 3]
    auto torch_result = torch_tensor.any(2);

    compare_bool_tensors(lfs_result, torch_result, "any(dim=2) 3D");
}

TEST_F(BoolAnyAllTest, All_Dim2_3D) {
    auto lfs_tensor = Tensor::ones({2, 3, 4}, Device::CPU, DataType::Bool);
    lfs_tensor.set_bool({0, 1, 2}, false);
    lfs_tensor.set_bool({1, 0, 0}, false);

    auto torch_tensor = torch::ones({2, 3, 4}, torch::kBool);
    torch_tensor.index_put_({0, 1, 2}, false);
    torch_tensor.index_put_({1, 0, 0}, false);

    auto lfs_result = lfs_tensor.all(2);       // -> [2, 3]
    auto torch_result = torch_tensor.all(2);

    compare_bool_tensors(lfs_result, torch_result, "all(dim=2) 3D");
}

// ============= Keepdim Tests =============

TEST_F(BoolAnyAllTest, Any_Dim0_Keepdim) {
    auto lfs_tensor = Tensor::zeros({3, 4}, Device::CPU, DataType::Bool);
    lfs_tensor.set_bool({0, 1}, true);
    lfs_tensor.set_bool({2, 3}, true);

    auto torch_tensor = torch::zeros({3, 4}, torch::kBool);
    torch_tensor.index_put_({0, 1}, true);
    torch_tensor.index_put_({2, 3}, true);

    auto lfs_result = lfs_tensor.any(0, true);       // -> [1, 4]
    auto torch_result = torch_tensor.any(0, true);

    compare_bool_tensors(lfs_result, torch_result, "any(dim=0, keepdim=true)");
    EXPECT_EQ(lfs_result.shape().rank(), 2);
    EXPECT_EQ(lfs_result.shape()[0], 1);
    EXPECT_EQ(lfs_result.shape()[1], 4);
}

TEST_F(BoolAnyAllTest, All_Dim1_Keepdim) {
    auto lfs_tensor = Tensor::ones({3, 4}, Device::CPU, DataType::Bool);
    lfs_tensor.set_bool({1, 2}, false);

    auto torch_tensor = torch::ones({3, 4}, torch::kBool);
    torch_tensor.index_put_({1, 2}, false);

    auto lfs_result = lfs_tensor.all(1, true);       // -> [3, 1]
    auto torch_result = torch_tensor.all(1, true);

    compare_bool_tensors(lfs_result, torch_result, "all(dim=1, keepdim=true)");
    EXPECT_EQ(lfs_result.shape().rank(), 2);
    EXPECT_EQ(lfs_result.shape()[0], 3);
    EXPECT_EQ(lfs_result.shape()[1], 1);
}

// ============= Random Tests =============

TEST_F(BoolAnyAllTest, Any_Random_FullReduction) {
    for (int trial = 0; trial < 10; ++trial) {
        auto lfs_tensor = random_bool_tensor({5, 6, 7}, 0.3f);
        auto torch_tensor = lfs_to_torch_bool(lfs_tensor);

        auto lfs_result = lfs_tensor.any();
        auto torch_result = torch_tensor.any();

        bool lfs_val = lfs_result.item() != 0;
        bool torch_val = torch_result.item<bool>();

        EXPECT_EQ(lfs_val, torch_val) << "Trial " << trial << ": any() mismatch";
    }
}

TEST_F(BoolAnyAllTest, All_Random_FullReduction) {
    for (int trial = 0; trial < 10; ++trial) {
        auto lfs_tensor = random_bool_tensor({5, 6, 7}, 0.9f);  // High probability to test all()
        auto torch_tensor = lfs_to_torch_bool(lfs_tensor);

        auto lfs_result = lfs_tensor.all();
        auto torch_result = torch_tensor.all();

        bool lfs_val = lfs_result.item() != 0;
        bool torch_val = torch_result.item<bool>();

        EXPECT_EQ(lfs_val, torch_val) << "Trial " << trial << ": all() mismatch";
    }
}

TEST_F(BoolAnyAllTest, Any_Random_AxisReduction) {
    for (int trial = 0; trial < 10; ++trial) {
        auto lfs_tensor = random_bool_tensor({4, 5, 6}, 0.3f);
        auto torch_tensor = lfs_to_torch_bool(lfs_tensor);

        for (int dim = 0; dim < 3; ++dim) {
            auto lfs_result = lfs_tensor.any(dim);
            auto torch_result = torch_tensor.any(dim);

            compare_bool_tensors(lfs_result, torch_result,
                                 "Trial " + std::to_string(trial) + ", any(dim=" + std::to_string(dim) + ")");
        }
    }
}

TEST_F(BoolAnyAllTest, All_Random_AxisReduction) {
    for (int trial = 0; trial < 10; ++trial) {
        auto lfs_tensor = random_bool_tensor({4, 5, 6}, 0.8f);
        auto torch_tensor = lfs_to_torch_bool(lfs_tensor);

        for (int dim = 0; dim < 3; ++dim) {
            auto lfs_result = lfs_tensor.all(dim);
            auto torch_result = torch_tensor.all(dim);

            compare_bool_tensors(lfs_result, torch_result,
                                 "Trial " + std::to_string(trial) + ", all(dim=" + std::to_string(dim) + ")");
        }
    }
}

// ============= Edge Cases =============

TEST_F(BoolAnyAllTest, Any_EmptyTensor) {
    auto lfs_tensor = Tensor::empty({0, 5}, Device::CPU, DataType::Bool);
    auto torch_tensor = torch::empty({0, 5}, torch::kBool);

    // any() on empty should be false (no true elements)
    auto lfs_result = lfs_tensor.any();
    auto torch_result = torch_tensor.any();

    EXPECT_FALSE(lfs_result.item() != 0);
    EXPECT_FALSE(torch_result.item<bool>());
}

TEST_F(BoolAnyAllTest, All_EmptyTensor) {
    auto lfs_tensor = Tensor::empty({0, 5}, Device::CPU, DataType::Bool);
    auto torch_tensor = torch::empty({0, 5}, torch::kBool);

    // all() on empty should be true (vacuous truth)
    auto lfs_result = lfs_tensor.all();
    auto torch_result = torch_tensor.all();

    EXPECT_TRUE(lfs_result.item() != 0);
    EXPECT_TRUE(torch_result.item<bool>());
}

TEST_F(BoolAnyAllTest, Any_SingleElement) {
    auto lfs_true = Tensor::ones({1}, Device::CPU, DataType::Bool);
    auto lfs_false = Tensor::zeros({1}, Device::CPU, DataType::Bool);

    EXPECT_TRUE(lfs_true.any().item() != 0);
    EXPECT_FALSE(lfs_false.any().item() != 0);
}

TEST_F(BoolAnyAllTest, All_SingleElement) {
    auto lfs_true = Tensor::ones({1}, Device::CPU, DataType::Bool);
    auto lfs_false = Tensor::zeros({1}, Device::CPU, DataType::Bool);

    EXPECT_TRUE(lfs_true.all().item() != 0);
    EXPECT_FALSE(lfs_false.all().item() != 0);
}

// ============= Non-Contiguous Tensor Tests =============

TEST_F(BoolAnyAllTest, Any_NonContiguousSlice) {
    auto base = Tensor::zeros({4, 4}, Device::CPU, DataType::Bool);
    base.set_bool({2, 2}, true);

    auto slice = base.slice(0, 0, 3).slice(1, 0, 3);  // 3x3 slice
    ASSERT_FALSE(slice.is_contiguous());

    auto torch_base = torch::zeros({4, 4}, torch::kBool);
    torch_base.index_put_({2, 2}, true);
    auto torch_slice = torch_base.slice(0, 0, 3).slice(1, 0, 3);

    // Full reduction
    auto lfs_result = slice.any();
    auto torch_result = torch_slice.any();

    EXPECT_EQ(lfs_result.item() != 0, torch_result.item<bool>());
}

TEST_F(BoolAnyAllTest, All_NonContiguousSlice) {
    auto base = Tensor::ones({4, 4}, Device::CPU, DataType::Bool);
    base.set_bool({1, 1}, false);

    auto slice = base.slice(0, 0, 3).slice(1, 0, 3);  // 3x3 slice
    ASSERT_FALSE(slice.is_contiguous());

    auto torch_base = torch::ones({4, 4}, torch::kBool);
    torch_base.index_put_({1, 1}, false);
    auto torch_slice = torch_base.slice(0, 0, 3).slice(1, 0, 3);

    // Full reduction
    auto lfs_result = slice.all();
    auto torch_result = torch_slice.all();

    EXPECT_EQ(lfs_result.item() != 0, torch_result.item<bool>());
}

TEST_F(BoolAnyAllTest, Any_NonContiguousSlice_AxisReduction) {
    auto base = Tensor::zeros({4, 4}, Device::CPU, DataType::Bool);
    base.set_bool({0, 1}, true);
    base.set_bool({2, 0}, true);

    auto slice = base.slice(0, 0, 3).slice(1, 0, 3);  // 3x3 slice
    ASSERT_FALSE(slice.is_contiguous());

    auto torch_base = torch::zeros({4, 4}, torch::kBool);
    torch_base.index_put_({0, 1}, true);
    torch_base.index_put_({2, 0}, true);
    auto torch_slice = torch_base.slice(0, 0, 3).slice(1, 0, 3);

    // Axis reduction
    auto lfs_result = slice.any(0);
    auto torch_result = torch_slice.any(0);

    compare_bool_tensors(lfs_result, torch_result, "any(dim=0) on non-contiguous slice");
}

// ============= CUDA Tests =============

TEST_F(BoolAnyAllTest, CUDA_Any_FullReduction) {
    auto cpu_tensor = random_bool_tensor({5, 6, 7}, 0.3f);
    auto cuda_tensor = cpu_tensor.to(Device::CUDA);
    auto torch_tensor = lfs_to_torch_bool(cpu_tensor).to(torch::kCUDA);

    auto lfs_result = cuda_tensor.any();
    auto torch_result = torch_tensor.any();

    bool lfs_val = lfs_result.to(Device::CPU).item() != 0;
    bool torch_val = torch_result.to(torch::kCPU).item<bool>();

    EXPECT_EQ(lfs_val, torch_val) << "CUDA any() mismatch";
}

TEST_F(BoolAnyAllTest, CUDA_All_FullReduction) {
    auto cpu_tensor = random_bool_tensor({5, 6, 7}, 0.9f);
    auto cuda_tensor = cpu_tensor.to(Device::CUDA);
    auto torch_tensor = lfs_to_torch_bool(cpu_tensor).to(torch::kCUDA);

    auto lfs_result = cuda_tensor.all();
    auto torch_result = torch_tensor.all();

    bool lfs_val = lfs_result.to(Device::CPU).item() != 0;
    bool torch_val = torch_result.to(torch::kCPU).item<bool>();

    EXPECT_EQ(lfs_val, torch_val) << "CUDA all() mismatch";
}

TEST_F(BoolAnyAllTest, CUDA_Any_AxisReduction) {
    auto cpu_tensor = random_bool_tensor({4, 5, 6}, 0.3f);
    auto cuda_tensor = cpu_tensor.to(Device::CUDA);
    auto torch_tensor = lfs_to_torch_bool(cpu_tensor).to(torch::kCUDA);

    for (int dim = 0; dim < 3; ++dim) {
        auto lfs_result = cuda_tensor.any(dim).to(Device::CPU);
        auto torch_result = torch_tensor.any(dim).to(torch::kCPU);

        compare_bool_tensors(lfs_result, torch_result, "CUDA any(dim=" + std::to_string(dim) + ")");
    }
}

TEST_F(BoolAnyAllTest, CUDA_All_AxisReduction) {
    auto cpu_tensor = random_bool_tensor({4, 5, 6}, 0.8f);
    auto cuda_tensor = cpu_tensor.to(Device::CUDA);
    auto torch_tensor = lfs_to_torch_bool(cpu_tensor).to(torch::kCUDA);

    for (int dim = 0; dim < 3; ++dim) {
        auto lfs_result = cuda_tensor.all(dim).to(Device::CPU);
        auto torch_result = torch_tensor.all(dim).to(torch::kCPU);

        compare_bool_tensors(lfs_result, torch_result, "CUDA all(dim=" + std::to_string(dim) + ")");
    }
}

