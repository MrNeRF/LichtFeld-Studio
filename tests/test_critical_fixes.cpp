/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <torch/torch.h>

using namespace lfs::core;

// ============================================================================
// Helper Functions
// ============================================================================

class TensorCriticalFixtures : public ::testing::Test {
protected:
    void SetUp() override {
        // Don't set seed here - do it per test if needed
    }

    bool tensors_equal(const Tensor& a, const torch::Tensor& b,
                       float rtol = 1e-5f, float atol = 1e-8f) {
        if (a.numel() != b.numel()) {
            std::cout << "Size mismatch: " << a.numel() << " vs " << b.numel() << std::endl;
            return false;
        }

        auto a_cpu = a.device() == Device::CUDA ? a.cpu() : a.clone();
        auto b_cpu = b.cpu();

        auto a_vec = a_cpu.to_vector();
        auto b_ptr = b_cpu.data_ptr<float>();

        int failures = 0;
        for (size_t i = 0; i < a_vec.size(); ++i) {
            float diff = std::abs(a_vec[i] - b_ptr[i]);
            float threshold = atol + rtol * std::abs(b_ptr[i]);
            if (diff > threshold) {
                if (failures < 5) {  // Only print first 5 failures
                    std::cout << "Mismatch at index " << i << ": " << a_vec[i]
                              << " vs " << b_ptr[i] << " (diff=" << diff
                              << ", threshold=" << threshold << ")" << std::endl;
                }
                failures++;
            }
        }
        if (failures > 0) {
            std::cout << "Total mismatches: " << failures << " out of " << a_vec.size() << std::endl;
        }
        return failures == 0;
    }

    bool shapes_equal(const Tensor& a, const torch::Tensor& b) {
        if (a.ndim() != b.dim())
            return false;
        for (size_t i = 0; i < a.ndim(); ++i) {
            if (a.size(i) != b.size(i))
                return false;
        }
        return true;
    }

    // Helper to copy GS tensor data to PyTorch tensor
    void copy_to_torch(const Tensor& gs, torch::Tensor& torch_t) {
        auto gs_cpu = (gs.device() == Device::CUDA) ? gs.cpu() : gs.clone();
        auto gs_vec = gs_cpu.to_vector();
        auto torch_cpu = torch_t.cpu();
        std::memcpy(torch_cpu.data_ptr<float>(), gs_vec.data(), gs_vec.size() * sizeof(float));
        if (torch_t.is_cuda()) {
            torch_t = torch_cpu.to(torch::kCUDA);
        } else {
            torch_t = torch_cpu;
        }
    }
};

// ============================================================================
// TEST SUITE 1: Row Assignment
// ============================================================================

TEST_F(TensorCriticalFixtures, RowAssignment_2D_CPU) {
    auto gs_tensor = Tensor::zeros({3, 4}, Device::CPU);
    auto gs_row = Tensor::ones({4}, Device::CPU);

    auto torch_tensor = torch::zeros({3, 4});
    auto torch_row = torch::ones({4});

    gs_tensor[0] = gs_row;
    torch_tensor[0] = torch_row;

    EXPECT_TRUE(tensors_equal(gs_tensor, torch_tensor))
        << "2D CPU row assignment doesn't match PyTorch";
}

TEST_F(TensorCriticalFixtures, RowAssignment_3D_CPU) {
    auto gs_tensor = Tensor::zeros({2, 3, 4}, Device::CPU);
    auto gs_slice = Tensor::full({3, 4}, 5.0f, Device::CPU);

    auto torch_tensor = torch::zeros({2, 3, 4});
    auto torch_slice = torch::full({3, 4}, 5.0f);

    gs_tensor[1] = gs_slice;
    torch_tensor[1] = torch_slice;

    EXPECT_TRUE(tensors_equal(gs_tensor, torch_tensor))
        << "3D CPU row assignment doesn't match PyTorch";
}

TEST_F(TensorCriticalFixtures, RowAssignment_2D_CUDA) {
    auto gs_tensor = Tensor::zeros({5, 10}, Device::CUDA);
    auto gs_row = Tensor::full({10}, 3.14f, Device::CUDA);

    auto torch_tensor = torch::zeros({5, 10}, torch::device(torch::kCUDA));
    auto torch_row = torch::full({10}, 3.14f, torch::device(torch::kCUDA));

    gs_tensor[2] = gs_row;
    torch_tensor[2] = torch_row;

    EXPECT_TRUE(tensors_equal(gs_tensor, torch_tensor))
        << "2D CUDA row assignment doesn't match PyTorch";
}

TEST_F(TensorCriticalFixtures, RowAssignment_CrossDevice) {
    auto gs_tensor = Tensor::zeros({3, 4}, Device::CUDA);
    auto gs_row = Tensor::ones({4}, Device::CPU);

    auto torch_tensor = torch::zeros({3, 4}, torch::device(torch::kCUDA));
    auto torch_row = torch::ones({4});

    gs_tensor[0] = gs_row; // Should auto-convert CPU->CUDA
    torch_tensor[0] = torch_row.to(torch::kCUDA);

    EXPECT_TRUE(tensors_equal(gs_tensor, torch_tensor))
        << "Cross-device row assignment doesn't match PyTorch";
}

TEST_F(TensorCriticalFixtures, RowAssignment_Multiple) {
    auto gs_tensor = Tensor::zeros({5, 8}, Device::CPU);
    auto torch_tensor = torch::zeros({5, 8});

    for (int i = 0; i < 5; ++i) {
        auto gs_row = Tensor::full({8}, float(i * 2), Device::CPU);
        auto torch_row = torch::full({8}, float(i * 2));

        gs_tensor[i] = gs_row;
        torch_tensor[i] = torch_row;
    }

    EXPECT_TRUE(tensors_equal(gs_tensor, torch_tensor))
        << "Multiple row assignments don't match PyTorch";
}

TEST_F(TensorCriticalFixtures, RowAssignment_1D_Scalar) {
    auto gs_tensor = Tensor::zeros({10}, Device::CPU);
    auto torch_tensor = torch::zeros({10});

    gs_tensor[5] = 42.0f;
    torch_tensor[5] = 42.0f;

    EXPECT_TRUE(tensors_equal(gs_tensor, torch_tensor))
        << "1D scalar assignment doesn't match PyTorch";
}

// ============================================================================
// TEST SUITE 2: Reduction Operations with keepdim
// ============================================================================

TEST_F(TensorCriticalFixtures, Reduction_Mean_SingleDim_Keepdim_CPU) {
    // Create GS tensor with known values
    auto gs_x = Tensor::randn({3, 4, 5}, Device::CPU);

    // Create PyTorch tensor and copy exact same data
    auto torch_x = torch::zeros({3, 4, 5});
    copy_to_torch(gs_x, torch_x);

    auto gs_mean = gs_x.mean(0, true);
    auto torch_mean = torch_x.mean(c10::IntArrayRef{0}, true);

    EXPECT_TRUE(shapes_equal(gs_mean, torch_mean))
        << "mean(0, true) shape mismatch";
    EXPECT_TRUE(tensors_equal(gs_mean, torch_mean))
        << "mean(0, true) values don't match PyTorch";
}

TEST_F(TensorCriticalFixtures, Reduction_Mean_MultiDim_Keepdim_CPU) {
    auto gs_x = Tensor::randn({2, 3, 4, 5}, Device::CPU);

    auto torch_x = torch::zeros({2, 3, 4, 5});
    copy_to_torch(gs_x, torch_x);

    auto gs_mean = gs_x.mean({0, 2}, true);
    auto torch_mean = torch_x.mean(c10::IntArrayRef{0, 2}, true);

    EXPECT_EQ(gs_mean.shape(), TensorShape({1, 3, 1, 5}))
        << "mean({0,2}, true) shape is wrong";
    EXPECT_TRUE(tensors_equal(gs_mean, torch_mean))
        << "mean({0,2}, true) values don't match PyTorch";
}

TEST_F(TensorCriticalFixtures, Reduction_Sum_Keepdim_CPU) {
    auto gs_x = Tensor::randn({4, 5, 6}, Device::CPU);

    auto torch_x = torch::zeros({4, 5, 6});
    copy_to_torch(gs_x, torch_x);

    auto gs_sum = gs_x.sum(1, true);
    auto torch_sum = torch_x.sum(c10::IntArrayRef{1}, true);

    EXPECT_EQ(gs_sum.shape(), TensorShape({4, 1, 6}))
        << "sum(1, true) shape is wrong";
    EXPECT_TRUE(tensors_equal(gs_sum, torch_sum))
        << "sum(1, true) values don't match PyTorch";
}

TEST_F(TensorCriticalFixtures, Reduction_Max_Keepdim_CPU) {
    auto gs_x = Tensor::randn({3, 4, 5}, Device::CPU);

    auto torch_x = torch::zeros({3, 4, 5});
    copy_to_torch(gs_x, torch_x);

    auto gs_max = gs_x.max(1, true);
    auto torch_max_tuple = torch_x.max(1, true);
    auto torch_max = std::get<0>(torch_max_tuple);

    EXPECT_EQ(gs_max.shape(), TensorShape({3, 1, 5}))
        << "max(1, true) shape is wrong";
    EXPECT_TRUE(tensors_equal(gs_max, torch_max))
        << "max(1, true) values don't match PyTorch";
}

TEST_F(TensorCriticalFixtures, Reduction_Min_Keepdim_CPU) {
    auto gs_x = Tensor::randn({3, 4, 5}, Device::CPU);

    auto torch_x = torch::zeros({3, 4, 5});
    copy_to_torch(gs_x, torch_x);

    auto gs_min = gs_x.min(2, true);
    auto torch_min_tuple = torch_x.min(2, true);
    auto torch_min = std::get<0>(torch_min_tuple);

    EXPECT_EQ(gs_min.shape(), TensorShape({3, 4, 1}))
        << "min(2, true) shape is wrong";
    EXPECT_TRUE(tensors_equal(gs_min, torch_min))
        << "min(2, true) values don't match PyTorch";
}

TEST_F(TensorCriticalFixtures, Reduction_Std_Keepdim_CPU) {
    auto gs_x = Tensor::randn({2, 6, 8}, Device::CPU);

    auto torch_x = torch::zeros({2, 6, 8});
    copy_to_torch(gs_x, torch_x);

    auto gs_std = gs_x.std(0, true);
    auto torch_std = torch_x.std(c10::IntArrayRef{0}, true);

    EXPECT_EQ(gs_std.shape(), TensorShape({1, 6, 8}))
        << "std(0, true) shape is wrong";
    EXPECT_TRUE(tensors_equal(gs_std, torch_std, 1e-4f, 1e-6f))
        << "std(0, true) values don't match PyTorch";
}

TEST_F(TensorCriticalFixtures, Reduction_Var_Keepdim_CPU) {
    auto gs_x = Tensor::randn({2, 6, 8}, Device::CPU);

    auto torch_x = torch::zeros({2, 6, 8});
    copy_to_torch(gs_x, torch_x);

    auto gs_var = gs_x.var(1, true);
    auto torch_var = torch_x.var(c10::IntArrayRef{1}, true);

    EXPECT_EQ(gs_var.shape(), TensorShape({2, 1, 8}))
        << "var(1, true) shape is wrong";
    EXPECT_TRUE(tensors_equal(gs_var, torch_var, 1e-4f, 1e-6f))
        << "var(1, true) values don't match PyTorch";
}

TEST_F(TensorCriticalFixtures, Reduction_Mean_Keepdim_CUDA) {
    auto gs_x = Tensor::randn({10, 20, 30}, Device::CUDA);

    auto torch_x = torch::zeros({10, 20, 30}, torch::device(torch::kCUDA));
    copy_to_torch(gs_x, torch_x);

    auto gs_mean = gs_x.mean({0, 2}, true);
    auto torch_mean = torch_x.mean(c10::IntArrayRef{0, 2}, true);

    std::cout << "GS mean shape: [";
    for (size_t i = 0; i < gs_mean.ndim(); ++i) {
        std::cout << gs_mean.size(i);
        if (i < gs_mean.ndim() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Torch mean shape: [";
    for (int64_t i = 0; i < torch_mean.dim(); ++i) {
        std::cout << torch_mean.size(i);
        if (i < torch_mean.dim() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    auto gs_mean_cpu = gs_mean.cpu();
    auto torch_mean_cpu = torch_mean.cpu();
    auto gs_vec = gs_mean_cpu.to_vector();
    auto torch_ptr = torch_mean_cpu.data_ptr<float>();

    std::cout << "First 10 GS values: ";
    for (int i = 0; i < std::min(10, (int)gs_vec.size()); ++i) {
        std::cout << gs_vec[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "First 10 Torch values: ";
    for (int i = 0; i < std::min(10, (int)torch_mean.numel()); ++i) {
        std::cout << torch_ptr[i] << " ";
    }
    std::cout << std::endl;

    EXPECT_EQ(gs_mean.shape(), TensorShape({1, 20, 1}))
        << "CUDA mean({0,2}, true) shape is wrong";
    // Use slightly relaxed tolerance for reduction operations (many fp accumulations)
    EXPECT_TRUE(tensors_equal(gs_mean, torch_mean, 1e-4f, 1e-7f))
        << "CUDA mean({0,2}, true) values don't match PyTorch";
}

// ============================================================================
// TEST SUITE 3: Boolean Tensor Access
// ============================================================================

TEST_F(TensorCriticalFixtures, BoolAccess_Basic_2D_CPU) {
    auto gs_mask = Tensor::zeros_bool({3, 4}, Device::CPU);
    auto torch_mask = torch::zeros({3, 4}, torch::dtype(torch::kBool));

    gs_mask.set_bool({0, 0}, true);
    gs_mask.set_bool({1, 2}, true);
    gs_mask.set_bool({2, 3}, true);

    torch_mask[0][0] = true;
    torch_mask[1][2] = true;
    torch_mask[2][3] = true;

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            bool gs_val = gs_mask.get_bool({i, j});
            bool torch_val = torch_mask[i][j].item<bool>();
            EXPECT_EQ(gs_val, torch_val)
                << "Mismatch at [" << i << "," << j << "]";
        }
    }
}

TEST_F(TensorCriticalFixtures, BoolAccess_3D_CPU) {
    auto gs_mask = Tensor::zeros_bool({2, 3, 4}, Device::CPU);
    auto torch_mask = torch::zeros({2, 3, 4}, torch::dtype(torch::kBool));

    gs_mask.set_bool({1, 2, 3}, true);
    gs_mask.set_bool({0, 1, 2}, true);

    torch_mask[1][2][3] = true;
    torch_mask[0][1][2] = true;

    EXPECT_EQ(gs_mask.get_bool({1, 2, 3}), torch_mask[1][2][3].item<bool>());
    EXPECT_EQ(gs_mask.get_bool({0, 1, 2}), torch_mask[0][1][2].item<bool>());
    EXPECT_EQ(gs_mask.get_bool({0, 0, 0}), torch_mask[0][0][0].item<bool>());
}

TEST_F(TensorCriticalFixtures, BoolAccess_Span_CPU) {
    auto gs_mask = Tensor::zeros_bool({5, 6}, Device::CPU);

    std::vector<size_t> idx1 = {2, 3};
    std::vector<size_t> idx2 = {4, 5};

    gs_mask.set_bool(std::span<const size_t>(idx1), true);
    gs_mask.set_bool(std::span<const size_t>(idx2), true);

    EXPECT_TRUE(gs_mask.get_bool(std::span<const size_t>(idx1)));
    EXPECT_TRUE(gs_mask.get_bool(std::span<const size_t>(idx2)));
    EXPECT_FALSE(gs_mask.get_bool({0, 0}));
}

TEST_F(TensorCriticalFixtures, BoolAccess_CUDA) {
    auto gs_mask = Tensor::zeros_bool({4, 5}, Device::CUDA);
    auto torch_mask = torch::zeros({4, 5},
                                   torch::dtype(torch::kBool).device(torch::kCUDA));

    gs_mask.set_bool({1, 1}, true);
    gs_mask.set_bool({3, 4}, true);

    torch_mask[1][1] = true;
    torch_mask[3][4] = true;

    EXPECT_EQ(gs_mask.get_bool({1, 1}), torch_mask[1][1].item<bool>());
    EXPECT_EQ(gs_mask.get_bool({3, 4}), torch_mask[3][4].item<bool>());
    EXPECT_EQ(gs_mask.get_bool({0, 0}), torch_mask[0][0].item<bool>());
}

TEST_F(TensorCriticalFixtures, BoolAccess_Boundaries_CPU) {
    auto gs_mask = Tensor::zeros_bool({10, 10}, Device::CPU);

    // Corner cases
    gs_mask.set_bool({0, 0}, true);
    gs_mask.set_bool({9, 9}, true);
    gs_mask.set_bool({0, 9}, true);
    gs_mask.set_bool({9, 0}, true);

    EXPECT_TRUE(gs_mask.get_bool({0, 0}));
    EXPECT_TRUE(gs_mask.get_bool({9, 9}));
    EXPECT_TRUE(gs_mask.get_bool({0, 9}));
    EXPECT_TRUE(gs_mask.get_bool({9, 0}));
    EXPECT_FALSE(gs_mask.get_bool({5, 5}));
}

// ============================================================================
// TEST SUITE 4: nonzero() Operation
// ============================================================================

TEST_F(TensorCriticalFixtures, Nonzero_1D_CPU) {
    std::vector<float> data = {0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 3.0f};
    auto gs_t = Tensor::from_vector(data, {6}, Device::CPU);
    auto torch_t = torch::from_blob(data.data(), {6}, torch::kFloat32).clone();

    auto gs_idx = gs_t.nonzero();
    auto torch_idx = torch_t.nonzero().squeeze(-1);

    EXPECT_EQ(gs_idx.dtype(), DataType::Int64)
        << "nonzero() should return Int64";

    auto gs_vec = gs_idx.to_vector_int64();
    auto torch_vec = torch_idx.cpu().data_ptr<int64_t>();

    ASSERT_EQ(gs_vec.size(), static_cast<size_t>(torch_idx.size(0)));
    for (size_t i = 0; i < gs_vec.size(); ++i) {
        EXPECT_EQ(gs_vec[i], torch_vec[i]) << "Mismatch at index " << i;
    }
}

TEST_F(TensorCriticalFixtures, Nonzero_2D_CPU) {
    std::vector<float> data = {
        0.0f, 1.0f, 0.0f,
        2.0f, 0.0f, 3.0f};
    auto gs_t = Tensor::from_vector(data, {2, 3}, Device::CPU);

    auto gs_idx = gs_t.nonzero();

    EXPECT_EQ(gs_idx.shape(), TensorShape({3, 2}))
        << "2D nonzero() shape should be [3, 2]";
    EXPECT_EQ(gs_idx.dtype(), DataType::Int64)
        << "2D nonzero() should return Int64";

    // Check that we found the right positions (order may differ)
    auto gs_vec = gs_idx.to_vector_int64();

    // Expected indices: [0,1], [1,0], [1,2]
    // Verify each row
    std::vector<std::pair<int64_t, int64_t>> expected = {{0, 1}, {1, 0}, {1, 2}};
    std::vector<std::pair<int64_t, int64_t>> got;
    for (size_t i = 0; i < 3; ++i) {
        got.push_back({gs_vec[i * 2], gs_vec[i * 2 + 1]});
    }

    // Sort both for comparison
    std::sort(expected.begin(), expected.end());
    std::sort(got.begin(), got.end());

    EXPECT_EQ(got, expected) << "nonzero() indices don't match expected positions";
}

TEST_F(TensorCriticalFixtures, Nonzero_3D_CPU) {
    auto gs_t = Tensor::zeros({2, 3, 4}, Device::CPU);

    gs_t.at({0, 1, 2}) = 1.0f;
    gs_t.at({1, 0, 3}) = 2.0f;
    gs_t.at({1, 2, 1}) = 3.0f;

    auto gs_idx = gs_t.nonzero();

    EXPECT_EQ(gs_idx.shape(), TensorShape({3, 3}))
        << "3D nonzero() shape should be [3, 3]";

    auto gs_vec = gs_idx.to_vector_int64();

    // Expected: [0,1,2], [1,0,3], [1,2,1]
    std::vector<std::tuple<int64_t, int64_t, int64_t>> expected = {
        {0, 1, 2},
        {1, 0, 3},
        {1, 2, 1}};
    std::vector<std::tuple<int64_t, int64_t, int64_t>> got;
    for (size_t i = 0; i < 3; ++i) {
        got.push_back({gs_vec[i * 3], gs_vec[i * 3 + 1], gs_vec[i * 3 + 2]});
    }

    std::sort(expected.begin(), expected.end());
    std::sort(got.begin(), got.end());

    EXPECT_EQ(got, expected) << "3D nonzero() indices don't match";
}

TEST_F(TensorCriticalFixtures, Nonzero_Empty_CPU) {
    auto gs_t = Tensor::zeros({3, 4}, Device::CPU);
    auto torch_t = torch::zeros({3, 4});

    auto gs_idx = gs_t.nonzero();
    auto torch_idx = torch_t.nonzero();

    EXPECT_EQ(gs_idx.numel(), 0);
    EXPECT_EQ(torch_idx.numel(), 0);
    EXPECT_EQ(gs_idx.shape(), TensorShape({0, 2}))
        << "Empty 2D nonzero() should have shape [0, 2]";
}

TEST_F(TensorCriticalFixtures, Nonzero_Bool_CPU) {
    auto gs_mask = Tensor::zeros_bool({4, 5}, Device::CPU);

    gs_mask.set_bool({0, 0}, true);
    gs_mask.set_bool({1, 2}, true);
    gs_mask.set_bool({3, 4}, true);

    auto gs_idx = gs_mask.nonzero();

    auto gs_vec = gs_idx.to_vector_int64();

    std::vector<std::pair<int64_t, int64_t>> expected = {{0, 0}, {1, 2}, {3, 4}};
    std::vector<std::pair<int64_t, int64_t>> got;
    for (size_t i = 0; i < 3; ++i) {
        got.push_back({gs_vec[i * 2], gs_vec[i * 2 + 1]});
    }

    std::sort(expected.begin(), expected.end());
    std::sort(got.begin(), got.end());

    EXPECT_EQ(got, expected) << "Bool nonzero() indices don't match";
}

TEST_F(TensorCriticalFixtures, Nonzero_CUDA) {
    std::vector<float> data = {0, 1, 0, 2, 0, 3, 0, 4};
    auto gs_t = Tensor::from_vector(data, {2, 4}, Device::CUDA);

    auto gs_idx = gs_t.nonzero();

    auto gs_vec = gs_idx.cpu().to_vector_int64();

    // Expected: [0,1], [0,3], [1,1], [1,3]
    std::vector<std::pair<int64_t, int64_t>> expected = {{0, 1}, {0, 3}, {1, 1}, {1, 3}};
    std::vector<std::pair<int64_t, int64_t>> got;
    for (size_t i = 0; i < 4; ++i) {
        got.push_back({gs_vec[i * 2], gs_vec[i * 2 + 1]});
    }

    std::sort(expected.begin(), expected.end());
    std::sort(got.begin(), got.end());

    EXPECT_EQ(got, expected) << "CUDA nonzero() indices don't match";
}

// ============================================================================
// TEST SUITE 5: sort() Returns Int64 Indices
// ============================================================================

TEST_F(TensorCriticalFixtures, Sort_1D_Int64_CPU) {
    std::vector<float> data = {5.0f, 2.0f, 8.0f, 1.0f, 9.0f, 3.0f};
    auto gs_t = Tensor::from_vector(data, {6}, Device::CPU);
    auto torch_t = torch::from_blob(data.data(), {6}, torch::kFloat32).clone();

    auto [gs_vals, gs_idx] = gs_t.sort(0, false);
    auto torch_result = torch_t.sort(0, false);
    auto torch_vals = std::get<0>(torch_result);
    auto torch_idx = std::get<1>(torch_result);

    EXPECT_EQ(gs_idx.dtype(), DataType::Int64)
        << "sort() indices should be Int64";
    EXPECT_EQ(torch_idx.dtype(), torch::kInt64)
        << "PyTorch sort() should also return Int64";

    EXPECT_TRUE(tensors_equal(gs_vals, torch_vals))
        << "sort() values don't match PyTorch";

    auto gs_idx_vec = gs_idx.to_vector_int64();
    auto torch_idx_vec = torch_idx.data_ptr<int64_t>();

    for (size_t i = 0; i < gs_idx_vec.size(); ++i) {
        EXPECT_EQ(gs_idx_vec[i], torch_idx_vec[i])
            << "Index mismatch at position " << i;
    }
}

TEST_F(TensorCriticalFixtures, Sort_2D_Dim0_Int64_CPU) {
    std::vector<float> data = {
        5.0f, 2.0f, 8.0f,
        1.0f, 9.0f, 3.0f,
        7.0f, 4.0f, 6.0f};
    auto gs_t = Tensor::from_vector(data, {3, 3}, Device::CPU);
    auto torch_t = torch::from_blob(data.data(), {3, 3}, torch::kFloat32).clone();

    auto [gs_vals, gs_idx] = gs_t.sort(0, false);
    auto torch_result = torch_t.sort(0, false);
    auto torch_vals = std::get<0>(torch_result);

    EXPECT_EQ(gs_idx.dtype(), DataType::Int64)
        << "2D sort(0) indices should be Int64";
    EXPECT_TRUE(tensors_equal(gs_vals, torch_vals))
        << "2D sort(0) values don't match PyTorch";
}

TEST_F(TensorCriticalFixtures, Sort_2D_Dim1_Int64_CPU) {
    std::vector<float> data = {
        9.0f, 3.0f, 6.0f, 1.0f,
        5.0f, 8.0f, 2.0f, 7.0f};
    auto gs_t = Tensor::from_vector(data, {2, 4}, Device::CPU);
    auto torch_t = torch::from_blob(data.data(), {2, 4}, torch::kFloat32).clone();

    auto [gs_vals, gs_idx] = gs_t.sort(1, false);
    auto torch_result = torch_t.sort(1, false);
    auto torch_vals = std::get<0>(torch_result);
    auto torch_idx = std::get<1>(torch_result);

    EXPECT_EQ(gs_idx.dtype(), DataType::Int64)
        << "2D sort(1) indices should be Int64";
    EXPECT_TRUE(tensors_equal(gs_vals, torch_vals))
        << "2D sort(1) values don't match PyTorch";

    auto gs_idx_vec = gs_idx.to_vector_int64();
    auto torch_idx_vec = torch_idx.data_ptr<int64_t>();

    for (size_t i = 0; i < gs_idx_vec.size(); ++i) {
        EXPECT_EQ(gs_idx_vec[i], torch_idx_vec[i])
            << "2D sort(1) index mismatch at " << i;
    }
}

TEST_F(TensorCriticalFixtures, Sort_Descending_Int64_CPU) {
    std::vector<float> data = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f};
    auto gs_t = Tensor::from_vector(data, {6}, Device::CPU);
    auto torch_t = torch::from_blob(data.data(), {6}, torch::kFloat32).clone();

    auto [gs_vals, gs_idx] = gs_t.sort(0, true);
    auto torch_result = torch_t.sort(0, true);
    auto torch_vals = std::get<0>(torch_result);
    auto torch_idx = std::get<1>(torch_result);

    EXPECT_EQ(gs_idx.dtype(), DataType::Int64)
        << "Descending sort() indices should be Int64";
    EXPECT_TRUE(tensors_equal(gs_vals, torch_vals))
        << "Descending sort() values don't match PyTorch";

    auto gs_idx_vec = gs_idx.to_vector_int64();
    auto torch_idx_vec = torch_idx.data_ptr<int64_t>();

    for (size_t i = 0; i < gs_idx_vec.size(); ++i) {
        EXPECT_EQ(gs_idx_vec[i], torch_idx_vec[i])
            << "Descending sort index mismatch at " << i;
    }
}

TEST_F(TensorCriticalFixtures, Sort_CUDA_Int64) {
    std::vector<float> data = {7.0f, 2.0f, 9.0f, 4.0f, 1.0f, 8.0f, 3.0f, 6.0f};
    auto gs_t = Tensor::from_vector(data, {8}, Device::CUDA);
    auto torch_t = torch::from_blob(data.data(), {8}, torch::kFloat32)
                       .clone()
                       .to(torch::kCUDA);

    auto [gs_vals, gs_idx] = gs_t.sort(0, false);
    auto torch_result = torch_t.sort(0, false);
    auto torch_vals = std::get<0>(torch_result);
    auto torch_idx = std::get<1>(torch_result);

    EXPECT_EQ(gs_idx.dtype(), DataType::Int64)
        << "CUDA sort() indices should be Int64";
    EXPECT_TRUE(tensors_equal(gs_vals, torch_vals))
        << "CUDA sort() values don't match PyTorch";

    auto gs_idx_vec = gs_idx.cpu().to_vector_int64();
    auto torch_idx_cpu = torch_idx.cpu();
    auto torch_idx_vec = torch_idx_cpu.data_ptr<int64_t>();

    for (size_t i = 0; i < gs_idx_vec.size(); ++i) {
        EXPECT_EQ(gs_idx_vec[i], torch_idx_vec[i])
            << "CUDA sort index mismatch at " << i;
    }
}

TEST_F(TensorCriticalFixtures, Sort_3D_Int64_CPU) {
    auto gs_t = Tensor::randn({2, 3, 4}, Device::CPU);

    auto torch_t = torch::zeros({2, 3, 4});
    copy_to_torch(gs_t, torch_t);

    auto [gs_vals, gs_idx] = gs_t.sort(2, false);
    auto torch_result = torch_t.sort(2, false);
    auto torch_vals = std::get<0>(torch_result);

    EXPECT_EQ(gs_idx.dtype(), DataType::Int64)
        << "3D sort() indices should be Int64";
    EXPECT_EQ(gs_idx.shape(), TensorShape({2, 3, 4}))
        << "3D sort() indices shape is wrong";
    EXPECT_TRUE(tensors_equal(gs_vals, torch_vals))
        << "3D sort() values don't match PyTorch";
}

// ============================================================================
// TEST SUITE 6: Integration Tests - Combined Operations
// ============================================================================

TEST_F(TensorCriticalFixtures, Integration_RowAssignment_Then_Reduction) {
    auto gs_tensor = Tensor::zeros({4, 5}, Device::CPU);

    for (int i = 0; i < 4; ++i) {
        auto row = Tensor::full({5}, float(i + 1), Device::CPU);
        gs_tensor[i] = row;
    }

    auto mean = gs_tensor.mean(0, true);
    EXPECT_EQ(mean.shape(), TensorShape({1, 5}));

    // After row assignment: rows are [1,1,1,1,1], [2,2,2,2,2], [3,3,3,3,3], [4,4,4,4,4]
    // Mean of [1,2,3,4] should be 2.5
    auto mean_vec = mean.to_vector();
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_NEAR(mean_vec[i], 2.5f, 1e-5f) << "Column " << i << " mean is wrong";
    }
}

TEST_F(TensorCriticalFixtures, Integration_BoolMask_Nonzero) {
    auto mask = Tensor::zeros_bool({3, 4}, Device::CPU);
    mask.set_bool({0, 1}, true);
    mask.set_bool({2, 3}, true);

    auto idx = mask.nonzero();
    EXPECT_EQ(idx.shape(), TensorShape({2, 2}));
    EXPECT_EQ(idx.dtype(), DataType::Int64);

    auto idx_vec = idx.to_vector_int64();

    std::vector<std::pair<int64_t, int64_t>> expected = {{0, 1}, {2, 3}};
    std::vector<std::pair<int64_t, int64_t>> got;
    for (size_t i = 0; i < 2; ++i) {
        got.push_back({idx_vec[i * 2], idx_vec[i * 2 + 1]});
    }

    std::sort(expected.begin(), expected.end());
    std::sort(got.begin(), got.end());

    EXPECT_EQ(got, expected);
}

TEST_F(TensorCriticalFixtures, Integration_Sort_Then_Index) {
    auto data = Tensor::from_vector({5.0f, 2.0f, 8.0f, 1.0f}, {4}, Device::CPU);
    auto [sorted_vals, sorted_idx] = data.sort(0, false);

    EXPECT_EQ(sorted_idx.dtype(), DataType::Int64);

    auto idx_vec = sorted_idx.to_vector_int64();
    EXPECT_EQ(idx_vec[0], 3); // index of 1.0
    EXPECT_EQ(idx_vec[1], 1); // index of 2.0
    EXPECT_EQ(idx_vec[2], 0); // index of 5.0
    EXPECT_EQ(idx_vec[3], 2); // index of 8.0
}

// Add this test to see what's actually in the tensors
TEST_F(TensorCriticalFixtures, DEBUG_RowAssignment) {
    auto gs_tensor = Tensor::zeros({4, 5}, Device::CPU);

    std::cout << "Initial tensor:" << std::endl;
    gs_tensor.print_formatted("gs_tensor");

    for (int i = 0; i < 4; ++i) {
        auto row = Tensor::full({5}, float(i + 1), Device::CPU);
        std::cout << "\nAssigning row " << i << " with value " << (i + 1) << std::endl;
        gs_tensor[i] = row;
        gs_tensor.print_formatted("gs_tensor after row " + std::to_string(i));
    }

    std::cout << "\nFinal tensor:" << std::endl;
    gs_tensor.print_formatted("gs_tensor");

    std::cout << "\nComputing mean(0, true)..." << std::endl;
    auto mean = gs_tensor.mean(0, true);
    std::cout << "Mean shape: " << mean.shape().str() << std::endl;
    mean.print_formatted("mean");

    auto mean_vec = mean.to_vector();
    std::cout << "Mean values: [";
    for (size_t i = 0; i < mean_vec.size(); ++i) {
        if (i > 0)
            std::cout << ", ";
        std::cout << mean_vec[i];
    }
    std::cout << "]" << std::endl;
}

TEST_F(TensorCriticalFixtures, DEBUG_Reduction_Keepdim) {
    // Simple test case
    auto gs_x = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, Device::CPU);

    std::cout << "Input tensor (2x3):" << std::endl;
    gs_x.print_formatted("gs_x");

    std::cout << "\nmean(0, false):" << std::endl;
    auto mean_no_keepdim = gs_x.mean(0, false);
    std::cout << "Shape: " << mean_no_keepdim.shape().str() << std::endl;
    mean_no_keepdim.print_formatted("mean");

    std::cout << "\nmean(0, true):" << std::endl;
    auto mean_keepdim = gs_x.mean(0, true);
    std::cout << "Shape: " << mean_keepdim.shape().str() << std::endl;
    mean_keepdim.print_formatted("mean");

    // Compare with PyTorch
    auto torch_x = torch::from_blob(const_cast<float*>(gs_x.ptr<float>()),
                                    {2, 3}, torch::kFloat32)
                       .clone();
    auto torch_mean = torch_x.mean(c10::IntArrayRef{0}, true);

    std::cout << "\nPyTorch mean(0, true):" << std::endl;
    std::cout << "Shape: [" << torch_mean.size(0) << ", " << torch_mean.size(1) << "]" << std::endl;
    std::cout << "Values: " << torch_mean << std::endl;
}
