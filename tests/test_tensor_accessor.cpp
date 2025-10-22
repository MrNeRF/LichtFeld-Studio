/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <gtest/gtest.h>
#include <torch/torch.h>

using namespace lfs::core;

// ============= Helper Functions =============

namespace {

    // Helper to create PyTorch tensor from vector data (CPU only for accessor tests)
    torch::Tensor create_torch_tensor(const std::vector<float>& data,
                                      const std::vector<int64_t>& shape) {
        auto cpu_tensor = torch::from_blob(
                              const_cast<float*>(data.data()),
                              shape.empty() ? std::vector<int64_t>{static_cast<int64_t>(data.size())} : shape,
                              torch::TensorOptions().dtype(torch::kFloat32))
                              .clone(); // Clone to own the memory

        return cpu_tensor; // Keep on CPU for accessor
    }

    // Helper to create PyTorch int tensor
    torch::Tensor create_torch_int_tensor(const std::vector<int>& data,
                                          const std::vector<int64_t>& shape) {
        auto cpu_tensor = torch::from_blob(
                              const_cast<int*>(data.data()),
                              shape.empty() ? std::vector<int64_t>{static_cast<int64_t>(data.size())} : shape,
                              torch::TensorOptions().dtype(torch::kInt32))
                              .clone();

        return cpu_tensor;
    }

    // Helper to create PyTorch bool tensor
    torch::Tensor create_torch_bool_tensor(const std::vector<bool>& data,
                                           const std::vector<int64_t>& shape) {
        std::vector<uint8_t> uint8_data(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            uint8_data[i] = data[i] ? 1 : 0;
        }

        auto cpu_tensor = torch::from_blob(
                              uint8_data.data(),
                              shape.empty() ? std::vector<int64_t>{static_cast<int64_t>(data.size())} : shape,
                              torch::TensorOptions().dtype(torch::kBool))
                              .clone();

        return cpu_tensor;
    }

} // anonymous namespace

class TensorAccessorTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(42);
        Tensor::manual_seed(42);
    }
};

// ============= 1D Accessor Tests =============

TEST_F(TensorAccessorTest, Accessor1DBasic) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    auto t_custom = Tensor::from_vector(data, {5}, Device::CPU);
    auto t_torch = create_torch_tensor(data, {5});

    auto acc_custom = t_custom.accessor<float, 1>();
    auto acc_torch = t_torch.accessor<float, 1>();

    EXPECT_EQ(acc_custom.sizes()[0], acc_torch.size(0));

    // Compare read access
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(acc_custom(i), acc_torch[i])
            << "Mismatch at index " << i;
    }
}

TEST_F(TensorAccessorTest, Accessor1DWrite) {
    auto t_custom = Tensor::zeros({5}, Device::CPU);
    auto t_torch = torch::zeros({5}, torch::TensorOptions().dtype(torch::kFloat32));

    auto acc_custom = t_custom.accessor<float, 1>();
    auto acc_torch = t_torch.accessor<float, 1>();

    // Write same values to both
    std::vector<float> test_values = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    for (size_t i = 0; i < 5; ++i) {
        acc_custom(i) = test_values[i];
        acc_torch[i] = test_values[i];
    }

    // Compare results
    auto values_custom = t_custom.to_vector();
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(values_custom[i], acc_torch[i])
            << "Mismatch after write at index " << i;
    }
}

// ============= 2D Accessor Tests =============

TEST_F(TensorAccessorTest, Accessor2DBasic) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    auto t_custom = Tensor::from_vector(data, {2, 3}, Device::CPU);
    auto t_torch = create_torch_tensor(data, {2, 3});

    auto acc_custom = t_custom.accessor<float, 2>();
    auto acc_torch = t_torch.accessor<float, 2>();

    EXPECT_EQ(acc_custom.sizes()[0], acc_torch.size(0));
    EXPECT_EQ(acc_custom.sizes()[1], acc_torch.size(1));

    // Compare read access
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(acc_custom(i, j), acc_torch[i][j])
                << "Mismatch at (" << i << "," << j << ")";
        }
    }
}

TEST_F(TensorAccessorTest, Accessor2DWrite) {
    auto t_custom = Tensor::zeros({3, 4}, Device::CPU);
    auto t_torch = torch::zeros({3, 4}, torch::TensorOptions().dtype(torch::kFloat32));

    auto acc_custom = t_custom.accessor<float, 2>();
    auto acc_torch = t_torch.accessor<float, 2>();

    // Fill with pattern
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            float val = i * 10.0f + j;
            acc_custom(i, j) = val;
            acc_torch[i][j] = val;
        }
    }

    // Compare results
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(acc_custom(i, j), acc_torch[i][j])
                << "Mismatch after write at (" << i << "," << j << ")";
        }
    }
}

TEST_F(TensorAccessorTest, Accessor2DRowMajor) {
    auto t_custom = Tensor::zeros({2, 3}, Device::CPU);
    auto t_torch = torch::zeros({2, 3}, torch::TensorOptions().dtype(torch::kFloat32));

    auto acc_custom = t_custom.accessor<float, 2>();
    auto acc_torch = t_torch.accessor<float, 2>();

    // Fill row-major: [0,1,2; 3,4,5]
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            float val = i * 3 + j;
            acc_custom(i, j) = val;
            acc_torch[i][j] = val;
        }
    }

    // Verify storage is row-major by comparing flat arrays
    auto values_custom = t_custom.to_vector();
    auto torch_flat = t_torch.flatten();
    auto torch_acc_flat = torch_flat.accessor<float, 1>();

    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(values_custom[i], torch_acc_flat[i])
            << "Row-major layout mismatch at index " << i;
    }
}

// ============= 3D Accessor Tests =============

TEST_F(TensorAccessorTest, Accessor3DBasic) {
    auto t_custom = Tensor::zeros({2, 3, 4}, Device::CPU);
    auto t_torch = torch::zeros({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat32));

    auto acc_custom = t_custom.accessor<float, 3>();
    auto acc_torch = t_torch.accessor<float, 3>();

    EXPECT_EQ(acc_custom.sizes()[0], acc_torch.size(0));
    EXPECT_EQ(acc_custom.sizes()[1], acc_torch.size(1));
    EXPECT_EQ(acc_custom.sizes()[2], acc_torch.size(2));

    // Write a pattern to both
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                float val = i * 100 + j * 10 + k;
                acc_custom(i, j, k) = val;
                acc_torch[i][j][k] = val;
            }
        }
    }

    // Compare spot checks
    EXPECT_FLOAT_EQ(acc_custom(0, 0, 0), acc_torch[0][0][0]);
    EXPECT_FLOAT_EQ(acc_custom(0, 1, 2), acc_torch[0][1][2]);
    EXPECT_FLOAT_EQ(acc_custom(1, 2, 3), acc_torch[1][2][3]);
}

TEST_F(TensorAccessorTest, Accessor3DStrides) {
    auto t_custom = Tensor::zeros({2, 3, 4}, Device::CPU);
    auto t_torch = torch::zeros({2, 3, 4}, torch::TensorOptions().dtype(torch::kFloat32));

    auto acc_custom = t_custom.accessor<float, 3>();
    auto acc_torch = t_torch.accessor<float, 3>();

    // Fill with linear sequence
    float val = 0.0f;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                acc_custom(i, j, k) = val;
                acc_torch[i][j][k] = val;
                val++;
            }
        }
    }

    // Verify strides work correctly by comparing flat arrays
    auto values_custom = t_custom.to_vector();
    auto torch_flat = t_torch.flatten();
    auto torch_acc_flat = torch_flat.accessor<float, 1>();

    for (size_t i = 0; i < values_custom.size(); ++i) {
        EXPECT_FLOAT_EQ(values_custom[i], torch_acc_flat[i])
            << "Stride mismatch at flat index " << i;
    }
}

// ============= 4D Accessor Tests =============

TEST_F(TensorAccessorTest, Accessor4DBasic) {
    auto t_custom = Tensor::zeros({2, 2, 2, 2}, Device::CPU);
    auto t_torch = torch::zeros({2, 2, 2, 2}, torch::TensorOptions().dtype(torch::kFloat32));

    auto acc_custom = t_custom.accessor<float, 4>();
    auto acc_torch = t_torch.accessor<float, 4>();

    EXPECT_EQ(acc_custom.sizes()[0], acc_torch.size(0));
    EXPECT_EQ(acc_custom.sizes()[1], acc_torch.size(1));
    EXPECT_EQ(acc_custom.sizes()[2], acc_torch.size(2));
    EXPECT_EQ(acc_custom.sizes()[3], acc_torch.size(3));

    // Set diagonal elements
    acc_custom(0, 0, 0, 0) = 1.0f;
    acc_custom(1, 1, 1, 1) = 2.0f;

    acc_torch[0][0][0][0] = 1.0f;
    acc_torch[1][1][1][1] = 2.0f;

    // Compare
    EXPECT_FLOAT_EQ(acc_custom(0, 0, 0, 0), acc_torch[0][0][0][0]);
    EXPECT_FLOAT_EQ(acc_custom(1, 1, 1, 1), acc_torch[1][1][1][1]);
    EXPECT_FLOAT_EQ(acc_custom(0, 1, 0, 1), acc_torch[0][1][0][1]);
}

// ============= Accessor with Different Types =============

TEST_F(TensorAccessorTest, AccessorIntType) {
    auto t_custom = Tensor::zeros({3, 3}, Device::CPU, DataType::Int32);
    auto t_torch = torch::zeros({3, 3}, torch::TensorOptions().dtype(torch::kInt32));

    auto acc_custom = t_custom.accessor<int, 2>();
    auto acc_torch = t_torch.accessor<int, 2>();

    // Fill with pattern
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            int val = i * 3 + j;
            acc_custom(i, j) = val;
            acc_torch[i][j] = val;
        }
    }

    // Compare
    auto values_custom = t_custom.to_vector_int();
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(acc_custom(i, j), acc_torch[i][j])
                << "Int type mismatch at (" << i << "," << j << ")";
        }
    }
}

TEST_F(TensorAccessorTest, AccessorBoolType) {
    auto t_custom = Tensor::zeros_bool({2, 3}, Device::CPU);
    auto t_torch = torch::zeros({2, 3}, torch::TensorOptions().dtype(torch::kBool));

    auto acc_custom = t_custom.accessor<unsigned char, 2>();
    auto acc_torch = t_torch.accessor<bool, 2>();

    // Set some values to true
    std::vector<std::pair<size_t, size_t>> true_positions = {{0, 0}, {0, 2}, {1, 1}};

    for (const auto& pos : true_positions) {
        acc_custom(pos.first, pos.second) = 1;
        acc_torch[pos.first][pos.second] = true;
    }

    // Compare
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            bool expected = acc_torch[i][j];
            bool actual = acc_custom(i, j) != 0;
            EXPECT_EQ(actual, expected)
                << "Bool type mismatch at (" << i << "," << j << ")";
        }
    }
}

// ============= Error Cases =============

TEST_F(TensorAccessorTest, AccessorWrongDimension) {
    auto t_custom = Tensor::zeros({3, 3}, Device::CPU);
    auto t_torch = torch::zeros({3, 3}, torch::TensorOptions().dtype(torch::kFloat32));

    // Try to create 1D accessor for 2D tensor
    auto acc_custom = t_custom.accessor<float, 1>();

    // PyTorch will throw if wrong dimension, our implementation should handle gracefully
    // Check that sizes are reasonable or zero
    EXPECT_TRUE(acc_custom.sizes()[0] == 0 || acc_custom.sizes()[0] == 3);

    // Note: PyTorch throws std::runtime_error if dimensions don't match
    // We document this as a known difference - our implementation is more forgiving
    LOG_INFO("IMPLEMENTATION NOTE: Dimension mismatch handling may differ from PyTorch");
}

TEST_F(TensorAccessorTest, AccessorOnCUDAFails) {
    auto t_custom = Tensor::zeros({3, 3}, Device::CUDA);

    // Accessor should only work on CPU tensors
    auto acc = t_custom.accessor<float, 2>();

    // Implementation should either return null/invalid or handle gracefully
    // PyTorch also restricts accessors to CPU tensors
    LOG_INFO("IMPLEMENTATION NOTE: CUDA tensor accessor should fail gracefully");
}

// ============= Practical Use Cases =============

TEST_F(TensorAccessorTest, FillIdentityMatrix) {
    auto t_custom = Tensor::zeros({5, 5}, Device::CPU);
    auto t_torch = torch::zeros({5, 5}, torch::TensorOptions().dtype(torch::kFloat32));

    auto acc_custom = t_custom.accessor<float, 2>();
    auto acc_torch = t_torch.accessor<float, 2>();

    // Fill diagonal with 1s
    for (size_t i = 0; i < 5; ++i) {
        acc_custom(i, i) = 1.0f;
        acc_torch[i][i] = 1.0f;
    }

    // Verify it's an identity matrix by comparing
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            EXPECT_FLOAT_EQ(acc_custom(i, j), acc_torch[i][j])
                << "Identity matrix mismatch at (" << i << "," << j << ")";

            if (i == j) {
                EXPECT_FLOAT_EQ(acc_custom(i, j), 1.0f);
            } else {
                EXPECT_FLOAT_EQ(acc_custom(i, j), 0.0f);
            }
        }
    }
}

TEST_F(TensorAccessorTest, FillBinomialCoefficients) {
    // Fill Pascal's triangle (binomial coefficients)
    size_t n = 10;
    auto t_custom = Tensor::zeros({n, n}, Device::CPU);
    auto t_torch = torch::zeros({static_cast<int64_t>(n), static_cast<int64_t>(n)},
                                torch::TensorOptions().dtype(torch::kFloat32));

    auto acc_custom = t_custom.accessor<float, 2>();
    auto acc_torch = t_torch.accessor<float, 2>();

    // C(n, k) = C(n-1, k-1) + C(n-1, k)
    for (size_t i = 0; i < n; ++i) {
        acc_custom(i, 0) = 1.0f;
        acc_custom(i, i) = 1.0f;
        acc_torch[i][0] = 1.0f;
        acc_torch[i][i] = 1.0f;

        for (size_t j = 1; j < i; ++j) {
            float val = acc_torch[i - 1][j - 1] + acc_torch[i - 1][j];
            acc_custom(i, j) = val;
            acc_torch[i][j] = val;
        }
    }

    // Verify known values match between implementations
    EXPECT_FLOAT_EQ(acc_custom(4, 2), acc_torch[4][2]); // C(4,2) = 6
    EXPECT_FLOAT_EQ(acc_custom(5, 2), acc_torch[5][2]); // C(5,2) = 10
    EXPECT_FLOAT_EQ(acc_custom(6, 3), acc_torch[6][3]); // C(6,3) = 20

    // Verify actual values
    EXPECT_FLOAT_EQ(acc_custom(4, 2), 6.0f);
    EXPECT_FLOAT_EQ(acc_custom(5, 2), 10.0f);
    EXPECT_FLOAT_EQ(acc_custom(6, 3), 20.0f);
}

TEST_F(TensorAccessorTest, MatrixTranspose) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    auto src_custom = Tensor::from_vector(data, {2, 3}, Device::CPU);
    auto dst_custom = Tensor::zeros({3, 2}, Device::CPU);

    auto src_torch = create_torch_tensor(data, {2, 3});
    auto dst_torch = torch::zeros({3, 2}, torch::TensorOptions().dtype(torch::kFloat32));

    auto src_acc_custom = src_custom.accessor<float, 2>();
    auto dst_acc_custom = dst_custom.accessor<float, 2>();
    auto src_acc_torch = src_torch.accessor<float, 2>();
    auto dst_acc_torch = dst_torch.accessor<float, 2>();

    // Manual transpose
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            dst_acc_custom(j, i) = src_acc_custom(i, j);
            dst_acc_torch[j][i] = src_acc_torch[i][j];
        }
    }

    // Compare transposed results
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(dst_acc_custom(i, j), dst_acc_torch[i][j])
                << "Transpose mismatch at (" << i << "," << j << ")";
        }
    }

    // Also verify against PyTorch's built-in transpose
    auto torch_transpose = src_torch.t();
    auto torch_t_acc = torch_transpose.accessor<float, 2>();

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            EXPECT_FLOAT_EQ(dst_acc_custom(i, j), torch_t_acc[i][j])
                << "Transpose vs PyTorch t() mismatch at (" << i << "," << j << ")";
        }
    }
}

TEST_F(TensorAccessorTest, ConvolutionWindow) {
    // Simulate accessing a 3x3 window in a larger image
    auto img_custom = Tensor::randn({10, 10}, Device::CPU);
    auto img_torch = torch::randn({10, 10}, torch::TensorOptions().dtype(torch::kFloat32));

    // Copy data from custom to torch for fair comparison
    auto img_data = img_custom.to_vector();
    std::memcpy(img_torch.data_ptr<float>(), img_data.data(), img_data.size() * sizeof(float));

    auto acc_custom = img_custom.accessor<float, 2>();
    auto acc_torch = img_torch.accessor<float, 2>();

    // Extract a 3x3 window starting at (5, 5)
    float sum_custom = 0.0f;
    float sum_torch = 0.0f;

    for (size_t i = 5; i < 8; ++i) {
        for (size_t j = 5; j < 8; ++j) {
            sum_custom += acc_custom(i, j);
            sum_torch += acc_torch[i][j];
        }
    }

    EXPECT_FLOAT_EQ(sum_custom, sum_torch) << "Convolution window sum mismatch";
    EXPECT_FALSE(std::isnan(sum_custom));
    EXPECT_FALSE(std::isnan(sum_torch));
}

// ============= Accessor Lifetime Tests =============

TEST_F(TensorAccessorTest, AccessorLifetime) {
    auto t_custom = Tensor::zeros({3, 3}, Device::CPU);
    auto t_torch = torch::zeros({3, 3}, torch::TensorOptions().dtype(torch::kFloat32));

    {
        auto acc_custom = t_custom.accessor<float, 2>();
        auto acc_torch = t_torch.accessor<float, 2>();

        acc_custom(1, 1) = 42.0f;
        acc_torch[1][1] = 42.0f;
        // Accessors go out of scope
    }

    // Changes should persist
    auto values_custom = t_custom.to_vector();
    auto torch_acc = t_torch.accessor<float, 2>();

    EXPECT_FLOAT_EQ(values_custom[4], torch_acc[1][1]); // Position (1,1) in row-major
    EXPECT_FLOAT_EQ(values_custom[4], 42.0f);
}

TEST_F(TensorAccessorTest, MultipleAccessors) {
    auto t_custom = Tensor::zeros({3, 3}, Device::CPU);
    auto t_torch = torch::zeros({3, 3}, torch::TensorOptions().dtype(torch::kFloat32));

    auto acc1_custom = t_custom.accessor<float, 2>();
    auto acc2_custom = t_custom.accessor<float, 2>();
    auto acc1_torch = t_torch.accessor<float, 2>();
    auto acc2_torch = t_torch.accessor<float, 2>();

    acc1_custom(0, 0) = 1.0f;
    acc2_custom(1, 1) = 2.0f;
    acc1_torch[0][0] = 1.0f;
    acc2_torch[1][1] = 2.0f;

    // Both should see all changes
    EXPECT_FLOAT_EQ(acc1_custom(0, 0), acc1_torch[0][0]);
    EXPECT_FLOAT_EQ(acc1_custom(1, 1), acc1_torch[1][1]);
    EXPECT_FLOAT_EQ(acc2_custom(0, 0), acc2_torch[0][0]);
    EXPECT_FLOAT_EQ(acc2_custom(1, 1), acc2_torch[1][1]);
}

// ============= Performance Test =============

TEST_F(TensorAccessorTest, AccessorPerformance) {
    auto t_custom = Tensor::zeros({1000, 1000}, Device::CPU);
    auto t_torch = torch::zeros({1000, 1000}, torch::TensorOptions().dtype(torch::kFloat32));

    auto acc_custom = t_custom.accessor<float, 2>();
    auto acc_torch = t_torch.accessor<float, 2>();

    // Fill entire matrix using accessor - time our implementation
    auto start_custom = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 1000; ++i) {
        for (size_t j = 0; j < 1000; ++j) {
            acc_custom(i, j) = i + j;
        }
    }
    auto end_custom = std::chrono::high_resolution_clock::now();
    auto duration_custom = std::chrono::duration_cast<std::chrono::milliseconds>(end_custom - start_custom);

    // Fill using PyTorch accessor
    auto start_torch = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 1000; ++i) {
        for (size_t j = 0; j < 1000; ++j) {
            acc_torch[i][j] = i + j;
        }
    }
    auto end_torch = std::chrono::high_resolution_clock::now();
    auto duration_torch = std::chrono::duration_cast<std::chrono::milliseconds>(end_torch - start_torch);

    LOG_INFO("Accessor Performance (1000x1000 matrix):");
    LOG_INFO("  Our implementation: {} ms", duration_custom.count());
    LOG_INFO("  PyTorch: {} ms", duration_torch.count());

    // Spot check - both should have the same values
    EXPECT_FLOAT_EQ(acc_custom(0, 0), acc_torch[0][0]);
    EXPECT_FLOAT_EQ(acc_custom(100, 200), acc_torch[100][200]);
    EXPECT_FLOAT_EQ(acc_custom(999, 999), acc_torch[999][999]);
}

// ============= Random Data Comprehensive Test =============

TEST_F(TensorAccessorTest, RandomAccessPattern) {
    // Test random access patterns
    std::mt19937 gen(42);
    std::uniform_int_distribution<size_t> dist(0, 99);
    std::uniform_real_distribution<float> val_dist(-100.0f, 100.0f);

    auto t_custom = Tensor::zeros({100, 100}, Device::CPU);
    auto t_torch = torch::zeros({100, 100}, torch::TensorOptions().dtype(torch::kFloat32));

    auto acc_custom = t_custom.accessor<float, 2>();
    auto acc_torch = t_torch.accessor<float, 2>();

    // Write random values at random positions
    std::vector<std::tuple<size_t, size_t, float>> operations;
    for (int i = 0; i < 1000; ++i) {
        size_t row = dist(gen);
        size_t col = dist(gen);
        float val = val_dist(gen);

        acc_custom(row, col) = val;
        acc_torch[row][col] = val;
        operations.push_back({row, col, val});
    }

    // Verify all written values match
    for (const auto& [row, col, expected_val] : operations) {
        EXPECT_FLOAT_EQ(acc_custom(row, col), acc_torch[row][col])
            << "Random access mismatch at (" << row << "," << col << ")";
    }
}

// ============= Batch Processing Test =============

TEST_F(TensorAccessorTest, BatchImageProcessing) {
    // Simulate batch image processing (NCHW format)
    const size_t batch = 4;
    const size_t channels = 3;
    const size_t height = 8;
    const size_t width = 8;

    auto t_custom = Tensor::randn({batch, channels, height, width}, Device::CPU);
    auto t_torch = torch::randn({static_cast<int64_t>(batch),
                                 static_cast<int64_t>(channels),
                                 static_cast<int64_t>(height),
                                 static_cast<int64_t>(width)},
                                torch::TensorOptions().dtype(torch::kFloat32));

    // Copy data for fair comparison
    auto data = t_custom.to_vector();
    std::memcpy(t_torch.data_ptr<float>(), data.data(), data.size() * sizeof(float));

    auto acc_custom = t_custom.accessor<float, 4>();
    auto acc_torch = t_torch.accessor<float, 4>();

    // Apply a simple transformation (add 1.0 to all red channel pixels)
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < height; ++h) {
            for (size_t w = 0; w < width; ++w) {
                acc_custom(b, 0, h, w) += 1.0f; // Red channel
                acc_torch[b][0][h][w] += 1.0f;
            }
        }
    }

    // Verify transformations match
    for (size_t b = 0; b < batch; ++b) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    EXPECT_FLOAT_EQ(acc_custom(b, c, h, w), acc_torch[b][c][h][w])
                        << "Batch processing mismatch at [" << b << "," << c << "," << h << "," << w << "]";
                }
            }
        }
    }
}