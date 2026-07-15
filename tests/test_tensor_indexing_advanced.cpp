/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <array>
#include <gtest/gtest.h>
#include <numeric>
#include <torch/torch.h>

using namespace lfs::core;

// Helper functions to convert between custom Tensor and torch::Tensor
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
            std::memcpy(result.data_ptr(), t.data_ptr(), t.bytes());
        } else {
            cudaMemcpy(result.data_ptr(), t.data_ptr(), t.bytes(), cudaMemcpyDeviceToDevice);
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
            std::memcpy(result.data_ptr(), t_cont.data_ptr(), result.bytes());
        } else {
            if (t_cont.is_cpu()) {
                cudaMemcpy(result.data_ptr(), t_cont.data_ptr(), result.bytes(), cudaMemcpyHostToDevice);
            } else {
                cudaMemcpy(result.data_ptr(), t_cont.data_ptr(), result.bytes(), cudaMemcpyDeviceToDevice);
            }
        }

        return result;
    }

    void compare_tensors(const Tensor& custom, const torch::Tensor& reference,
                         float rtol = 1e-5f, float atol = 1e-7f, const std::string& msg = "") {
        auto ref_cpu = reference.cpu();
        auto custom_cpu = custom.cpu();

        ASSERT_EQ(custom_cpu.ndim(), ref_cpu.dim()) << msg << ": Rank mismatch";

        for (size_t i = 0; i < custom_cpu.ndim(); ++i) {
            ASSERT_EQ(custom_cpu.size(i), static_cast<size_t>(ref_cpu.size(i)))
                << msg << ": Shape mismatch at dim " << i;
        }

        ASSERT_EQ(custom_cpu.numel(), static_cast<size_t>(ref_cpu.numel()))
            << msg << ": Element count mismatch";

        if (custom_cpu.dtype() == DataType::Bool || ref_cpu.scalar_type() == torch::kBool) {
            auto custom_vec = custom_cpu.to_vector_bool();
            auto ref_ptr = ref_cpu.data_ptr<bool>();
            for (size_t i = 0; i < custom_vec.size(); ++i) {
                EXPECT_EQ(custom_vec[i], ref_ptr[i]) << msg << ": Mismatch at index " << i;
            }
        } else if (custom_cpu.dtype() == DataType::Int32 || custom_cpu.dtype() == DataType::Int64) {
            auto custom_vec = custom_cpu.to_vector_int();
            if (ref_cpu.scalar_type() == torch::kInt32) {
                auto ref_ptr = ref_cpu.data_ptr<int32_t>();
                for (size_t i = 0; i < custom_vec.size(); ++i) {
                    EXPECT_EQ(custom_vec[i], ref_ptr[i]) << msg << ": Mismatch at index " << i;
                }
            } else if (ref_cpu.scalar_type() == torch::kInt64) {
                auto ref_ptr = ref_cpu.data_ptr<int64_t>();
                for (size_t i = 0; i < custom_vec.size(); ++i) {
                    EXPECT_EQ(custom_vec[i], static_cast<int>(ref_ptr[i]))
                        << msg << ": Mismatch at index " << i;
                }
            }
        } else {
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
    }

} // anonymous namespace

class TensorIndexingAdvancedTest : public ::testing::Test {
protected:
    void SetUp() override {
        Tensor::manual_seed(42);
        torch::manual_seed(42);
    }
};

TEST_F(TensorIndexingAdvancedTest, RowProxyAssignmentAcrossRanksAndDevices) {
    auto matrix = Tensor::zeros({3, 4}, Device::CPU);
    matrix[0] = Tensor::from_vector(
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, {4}, Device::CPU);
    matrix[2] = Tensor::full({4}, 7.0f, Device::CPU);
    EXPECT_EQ(matrix.to_vector(),
              (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f,
                                  7.0f, 7.0f, 7.0f, 7.0f}));

    auto volume = Tensor::zeros({2, 2, 3}, Device::CPU);
    volume[1] = Tensor::from_vector(
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
        {2, 3}, Device::CPU);
    EXPECT_EQ(volume.to_vector(),
              (std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));

    auto cross_device = Tensor::zeros({2, 3}, Device::CUDA);
    cross_device[1] = Tensor::from_vector(
        std::vector<float>{8.0f, 9.0f, 10.0f}, {3}, Device::CPU);
    EXPECT_EQ(cross_device.cpu().to_vector(),
              (std::vector<float>{0.0f, 0.0f, 0.0f, 8.0f, 9.0f, 10.0f}));

    auto vector = Tensor::zeros({4}, Device::CPU);
    vector[2] = 42.0f;
    EXPECT_EQ(vector.to_vector(), (std::vector<float>{0.0f, 0.0f, 42.0f, 0.0f}));
}

// ============= index_add_ Tests =============

TEST_F(TensorIndexingAdvancedTest, IndexAddBasic) {
    // Create test data
    auto t_custom = Tensor::zeros({5}, Device::CPU);
    auto t_torch = torch::zeros({5}, torch::kFloat32);

    auto indices = std::vector<int>{0, 2, 4};
    auto indices_custom = Tensor::from_vector(indices, {3}, Device::CPU);
    auto indices_torch = torch::tensor(indices, torch::kInt64);

    auto values_vec = std::vector<float>{1.0f, 2.0f, 3.0f};
    auto values_custom = Tensor::from_vector(values_vec, {3}, Device::CPU);
    auto values_torch = torch::tensor(values_vec);

    // Apply operation
    t_custom.index_add_(0, indices_custom, values_custom);
    t_torch.index_add_(0, indices_torch, values_torch);

    compare_tensors(t_custom, t_torch, 1e-5f, 1e-7f, "IndexAddBasic");
}

TEST_F(TensorIndexingAdvancedTest, IndexAddAccumulate) {
    auto t_custom = Tensor::ones({5}, Device::CPU);
    auto t_torch = torch::ones({5}, torch::kFloat32);

    auto indices = std::vector<int>{0, 0, 1};
    auto indices_custom = Tensor::from_vector(indices, {3}, Device::CPU);
    auto indices_torch = torch::tensor(indices, torch::kInt64);

    auto values_vec = std::vector<float>{10.0f, 20.0f, 30.0f};
    auto values_custom = Tensor::from_vector(values_vec, {3}, Device::CPU);
    auto values_torch = torch::tensor(values_vec);

    t_custom.index_add_(0, indices_custom, values_custom);
    t_torch.index_add_(0, indices_torch, values_torch);

    compare_tensors(t_custom, t_torch, 1e-5f, 1e-7f, "IndexAddAccumulate");
}

TEST_F(TensorIndexingAdvancedTest, IndexAdd2D) {
    auto t_custom = Tensor::zeros({3, 4}, Device::CPU);
    auto t_torch = torch::zeros({3, 4}, torch::kFloat32);

    auto indices = std::vector<int>{0, 2};
    auto indices_custom = Tensor::from_vector(indices, {2}, Device::CPU);
    auto indices_torch = torch::tensor(indices, torch::kInt64);

    auto values_custom = Tensor::ones({2, 4}, Device::CPU);
    auto values_torch = torch::ones({2, 4}, torch::kFloat32);

    t_custom.index_add_(0, indices_custom, values_custom);
    t_torch.index_add_(0, indices_torch, values_torch);

    compare_tensors(t_custom, t_torch, 1e-5f, 1e-7f, "IndexAdd2D");
}

TEST_F(TensorIndexingAdvancedTest, IndexAddCUDA) {
    auto t_custom = Tensor::zeros({5}, Device::CUDA);
    auto t_torch = torch::zeros({5}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    auto indices_vec = std::vector<int>{0, 2, 4};
    auto indices_custom = Tensor::from_vector(indices_vec, {3}, Device::CUDA);
    auto indices_torch = torch::tensor(indices_vec, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

    auto values_vec = std::vector<float>{1.0f, 2.0f, 3.0f};
    auto values_custom = Tensor::from_vector(values_vec, {3}, Device::CUDA);
    auto values_torch = torch::tensor(values_vec, torch::TensorOptions().device(torch::kCUDA));

    t_custom.index_add_(0, indices_custom, values_custom);
    t_torch.index_add_(0, indices_torch, values_torch);

    compare_tensors(t_custom, t_torch, 1e-5f, 1e-7f, "IndexAddCUDA");
}

// ============= index_put_ Tests =============

TEST_F(TensorIndexingAdvancedTest, IndexPutSingle) {
    auto t_custom = Tensor::zeros({5}, Device::CPU);
    auto t_torch = torch::zeros({5}, torch::kFloat32);

    auto indices_vec = std::vector<int>{1, 3};
    auto indices_custom = Tensor::from_vector(indices_vec, {2}, Device::CPU);
    auto indices_torch = torch::tensor(indices_vec, torch::kInt64);

    auto values_vec = std::vector<float>{10.0f, 20.0f};
    auto values_custom = Tensor::from_vector(values_vec, {2}, Device::CPU);
    auto values_torch = torch::tensor(values_vec);

    t_custom.index_put_(indices_custom, values_custom);
    t_torch.index_put_({indices_torch}, values_torch);

    compare_tensors(t_custom, t_torch, 1e-5f, 1e-7f, "IndexPutSingle");
}

TEST_F(TensorIndexingAdvancedTest, IndexPutNegativeIndices) {
    auto t_custom = Tensor::zeros({5}, Device::CPU);
    auto t_torch = torch::zeros({5}, torch::kFloat32);

    auto indices_vec = std::vector<int>{-1, -2};
    auto indices_custom = Tensor::from_vector(indices_vec, {2}, Device::CPU);
    auto indices_torch = torch::tensor(indices_vec, torch::kInt64);

    auto values_vec = std::vector<float>{100.0f, 200.0f};
    auto values_custom = Tensor::from_vector(values_vec, {2}, Device::CPU);
    auto values_torch = torch::tensor(values_vec);

    t_custom.index_put_(indices_custom, values_custom);
    t_torch.index_put_({indices_torch}, values_torch);

    compare_tensors(t_custom, t_torch, 1e-5f, 1e-7f, "IndexPutNegativeIndices");
}

TEST_F(TensorIndexingAdvancedTest, IndexPutMultiDimensional) {
    auto t_custom = Tensor::zeros({3, 4}, Device::CPU);
    auto t_torch = torch::zeros({3, 4}, torch::kFloat32);

    auto indices_vec = std::vector<int>{0, 5, 11};
    auto indices_custom = Tensor::from_vector(indices_vec, {3}, Device::CPU);
    auto indices_torch = torch::tensor(indices_vec, torch::kInt64);

    auto values_vec = std::vector<float>{1.0f, 2.0f, 3.0f};
    auto values_custom = Tensor::from_vector(values_vec, {3}, Device::CPU);
    auto values_torch = torch::tensor(values_vec);

    // Flatten for index_put_
    auto t_custom_flat = t_custom.flatten();
    auto t_torch_flat = t_torch.flatten();

    t_custom_flat.index_put_(indices_custom, values_custom);
    t_torch_flat.index_put_({indices_torch}, values_torch);

    // Reshape back
    auto t_custom_reshaped = t_custom_flat.reshape({3, 4});
    auto t_torch_reshaped = t_torch_flat.reshape({3, 4});

    compare_tensors(t_custom_reshaped, t_torch_reshaped, 1e-5f, 1e-7f, "IndexPutMultiDimensional");
}

TEST_F(TensorIndexingAdvancedTest, IndexPutVectorOfTensors) {
    auto t_custom = Tensor::zeros({3, 3}, Device::CPU);
    auto t_torch = torch::zeros({3, 3}, torch::kFloat32);

    auto row_vec = std::vector<int>{0, 1, 2};
    auto col_vec = std::vector<int>{0, 1, 2};

    auto row_idx_custom = Tensor::from_vector(row_vec, {3}, Device::CPU);
    auto col_idx_custom = Tensor::from_vector(col_vec, {3}, Device::CPU);

    auto row_idx_torch = torch::tensor(row_vec, torch::kInt64);
    auto col_idx_torch = torch::tensor(col_vec, torch::kInt64);

    auto values_vec = std::vector<float>{1.0f, 2.0f, 3.0f};
    auto values_custom = Tensor::from_vector(values_vec, {3}, Device::CPU);
    auto values_torch = torch::tensor(values_vec);

    std::vector<Tensor> indices_custom;
    indices_custom.push_back(std::move(row_idx_custom));
    indices_custom.push_back(std::move(col_idx_custom));

    t_custom.index_put_(indices_custom, values_custom);
    t_torch.index_put_({row_idx_torch, col_idx_torch}, values_torch);

    compare_tensors(t_custom, t_torch, 1e-5f, 1e-7f, "IndexPutVectorOfTensors");
}

TEST_F(TensorIndexingAdvancedTest, IndexPutVectorOfTensorsInt64CUDA) {
    auto t_custom = Tensor::zeros({3, 3}, Device::CUDA);
    auto t_torch = torch::zeros({3, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    auto row_idx_custom = Tensor::from_vector({0, -1, 1}, {3}, Device::CUDA).to(DataType::Int64);
    auto col_idx_custom = Tensor::from_vector({1, 0, -1}, {3}, Device::CUDA).to(DataType::Int64);

    auto row_idx_torch = torch::tensor({0, -1, 1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
    auto col_idx_torch = torch::tensor({1, 0, -1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));

    auto values_custom = Tensor::from_vector({10.0f, 20.0f, 30.0f}, {3}, Device::CUDA);
    auto values_torch = torch::tensor({10.0f, 20.0f, 30.0f}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    std::vector<Tensor> indices_custom{row_idx_custom, col_idx_custom};
    t_custom.index_put_(indices_custom, values_custom);
    t_torch.index_put_({row_idx_torch, col_idx_torch}, values_torch);

    compare_tensors(t_custom, t_torch, 1e-5f, 1e-7f, "IndexPutVectorOfTensorsInt64CUDA");
}

// ============= Boundary Mode Tests =============

TEST_F(TensorIndexingAdvancedTest, IndexSelectWithClampMode) {
    auto t_custom = Tensor::arange(0.0f, 5.0f);
    auto t_torch = torch::arange(0, 5, torch::kFloat32);

    auto indices_vec = std::vector<int>{-1, 0, 3, 10};
    auto indices_custom = Tensor::from_vector(indices_vec, {4}, Device::CUDA);

    // PyTorch clamps by default in index_select when indices are clamped manually
    std::vector<int64_t> indices_clamped;
    for (int idx : indices_vec) {
        indices_clamped.push_back(std::clamp(static_cast<int64_t>(idx), static_cast<int64_t>(0), static_cast<int64_t>(4)));
    }
    auto indices_torch = torch::tensor(indices_clamped, torch::kInt64);

    auto result_custom = t_custom.index_select(0, indices_custom, BoundaryMode::Clamp);
    auto result_torch = t_torch.index_select(0, indices_torch);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "IndexSelectClampMode");
}

TEST_F(TensorIndexingAdvancedTest, IndexSelectWithWrapMode) {
    auto t_custom = Tensor::arange(0.0f, 5.0f);
    auto t_torch = torch::arange(0, 5, torch::kFloat32);

    auto indices_vec = std::vector<int>{-1, 0, 5, 7};
    auto indices_custom = Tensor::from_vector(indices_vec, {4}, Device::CUDA);

    // PyTorch wraps with modulo
    std::vector<int64_t> indices_wrapped;
    for (int idx : indices_vec) {
        int wrapped = idx % 5;
        if (wrapped < 0)
            wrapped += 5;
        indices_wrapped.push_back(wrapped);
    }
    auto indices_torch = torch::tensor(indices_wrapped, torch::kInt64);

    auto result_custom = t_custom.index_select(0, indices_custom, BoundaryMode::Wrap);
    auto result_torch = t_torch.index_select(0, indices_torch);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "IndexSelectWrapMode");
}

TEST_F(TensorIndexingAdvancedTest, GatherWithClampMode) {
    std::vector<float> data_vec = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto t_custom = Tensor::from_vector(data_vec, {2, 3}, Device::CPU);
    auto t_torch = torch::tensor(data_vec).reshape({2, 3});

    auto indices_vec = std::vector<int>{-1, 1, 10};
    auto indices_custom = Tensor::from_vector(indices_vec, {3}, Device::CPU);

    // Clamp indices for PyTorch
    std::vector<int64_t> indices_clamped;
    for (int idx : indices_vec) {
        indices_clamped.push_back(std::clamp(static_cast<int64_t>(idx), static_cast<int64_t>(0), static_cast<int64_t>(2)));
    }
    // Expand for gather
    auto indices_torch = torch::tensor(indices_clamped, torch::kInt64).reshape({1, 3}).expand({2, 3});

    auto result_custom = t_custom.gather(1, indices_custom, BoundaryMode::Clamp);
    auto result_torch = t_torch.gather(1, indices_torch);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "GatherClampMode");
}

TEST_F(TensorIndexingAdvancedTest, GatherWithWrapMode) {
    std::vector<float> data_vec = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto t_custom = Tensor::from_vector(data_vec, {2, 3}, Device::CPU);
    auto t_torch = torch::tensor(data_vec).reshape({2, 3});

    auto indices_vec = std::vector<int>{-1, 0, 3};
    auto indices_custom = Tensor::from_vector(indices_vec, {3}, Device::CPU);

    // Wrap indices for PyTorch
    std::vector<int64_t> indices_wrapped;
    for (int idx : indices_vec) {
        int wrapped = idx % 3;
        if (wrapped < 0)
            wrapped += 3;
        indices_wrapped.push_back(wrapped);
    }
    auto indices_torch = torch::tensor(indices_wrapped, torch::kInt64).reshape({1, 3}).expand({2, 3});

    auto result_custom = t_custom.gather(1, indices_custom, BoundaryMode::Wrap);
    auto result_torch = t_torch.gather(1, indices_torch);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "GatherWrapMode");
}

// ============= nonzero Tests =============

TEST_F(TensorIndexingAdvancedTest, NonzeroBasic) {
    std::vector<float> data_vec = {0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 3.0f};
    auto t_custom = Tensor::from_vector(data_vec, {6}, Device::CPU);
    auto t_torch = torch::tensor(data_vec);

    auto result_custom = t_custom.nonzero();
    auto result_torch = t_torch.nonzero(); // Keep 2D: [count, 1]

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "NonzeroBasic");
}

TEST_F(TensorIndexingAdvancedTest, NonzeroBool) {
    std::vector<bool> data_vec = {true, false, true, false, true};
    auto t_custom = Tensor::from_vector(data_vec, {5}, Device::CPU);
    auto t_torch = torch::tensor({1, 0, 1, 0, 1}, torch::kBool);

    auto result_custom = t_custom.nonzero();
    auto result_torch = t_torch.nonzero(); // Keep 2D: [count, 1]

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "NonzeroBool");
}

TEST_F(TensorIndexingAdvancedTest, NonzeroAllOnes) {
    auto t_custom = Tensor::ones({10}, Device::CPU);
    auto t_torch = torch::ones({10});

    auto result_custom = t_custom.nonzero();
    auto result_torch = t_torch.nonzero(); // Keep 2D: [10, 1]

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "NonzeroAllOnes");
}

TEST_F(TensorIndexingAdvancedTest, NonzeroCUDA) {
    std::vector<float> data_vec = {0.0f, 1.0f, 0.0f, 2.0f};
    auto t_custom = Tensor::from_vector(data_vec, {4}, Device::CUDA);
    auto t_torch = torch::tensor(data_vec, torch::kCUDA);

    auto result_custom = t_custom.nonzero();
    auto result_torch = t_torch.nonzero(); // Keep 2D: [count, 1]

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "NonzeroCUDA");
}

TEST_F(TensorIndexingAdvancedTest, NonzeroPreservesCoordinatesAcrossRanksAndEmptyResults) {
    const auto matrix = Tensor::from_vector(
        std::vector<float>{0.0f, 1.0f, 0.0f, 2.0f, 3.0f, 0.0f},
        {2, 3}, Device::CPU);
    const auto matrix_indices = matrix.nonzero();
    EXPECT_EQ(matrix_indices.shape(), TensorShape({3, 2}));
    EXPECT_EQ(matrix_indices.to_vector_int64(),
              (std::vector<int64_t>{0, 1, 1, 0, 1, 1}));

    const auto volume = Tensor::from_vector(
        std::vector<float>{0.0f, 0.0f, 4.0f, 0.0f, 0.0f, 5.0f, 0.0f, 0.0f},
        {2, 2, 2}, Device::CPU);
    const auto volume_indices = volume.nonzero();
    EXPECT_EQ(volume_indices.shape(), TensorShape({2, 3}));
    EXPECT_EQ(volume_indices.to_vector_int64(),
              (std::vector<int64_t>{0, 1, 0, 1, 0, 1}));

    const auto empty_indices = Tensor::zeros({2, 3}, Device::CPU).nonzero();
    EXPECT_EQ(empty_indices.shape(), TensorShape({0, 2}));
    EXPECT_EQ(empty_indices.dtype(), DataType::Int64);
}

// ============= Masked Select Tests =============

TEST_F(TensorIndexingAdvancedTest, MaskedSelectBasic) {
    std::vector<float> data_vec = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto t_custom = Tensor::from_vector(data_vec, {5}, Device::CPU);
    auto t_torch = torch::tensor(data_vec);

    std::vector<bool> mask_vec = {true, false, true, false, true};
    auto mask_custom = Tensor::from_vector(mask_vec, {5}, Device::CPU);
    auto mask_torch = torch::tensor({1, 0, 1, 0, 1}, torch::kBool);

    auto result_custom = t_custom.masked_select(mask_custom);
    auto result_torch = t_torch.masked_select(mask_torch);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "MaskedSelectBasic");
}

TEST_F(TensorIndexingAdvancedTest, MaskedFillBasic) {
    std::vector<float> data_vec = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto t_custom = Tensor::from_vector(data_vec, {5}, Device::CPU);
    auto t_torch = torch::tensor(data_vec);

    std::vector<bool> mask_vec = {true, false, true, false, true};
    auto mask_custom = Tensor::from_vector(mask_vec, {5}, Device::CPU);
    auto mask_torch = torch::tensor({1, 0, 1, 0, 1}, torch::kBool);

    t_custom.masked_fill_(mask_custom, 100.0f);
    t_torch.masked_fill_(mask_torch, 100.0f);

    compare_tensors(t_custom, t_torch, 1e-5f, 1e-7f, "MaskedFillBasic");
}

// ============= Scatter/Gather Tests =============

TEST_F(TensorIndexingAdvancedTest, ScatterBasic) {
    auto t_custom = Tensor::zeros({5}, Device::CPU);
    auto t_torch = torch::zeros({5});

    auto indices_vec = std::vector<int>{0, 2, 4};
    auto indices_custom = Tensor::from_vector(indices_vec, {3}, Device::CPU);
    auto indices_torch = torch::tensor(indices_vec, torch::kInt64);

    auto values_vec = std::vector<float>{10.0f, 20.0f, 30.0f};
    auto values_custom = Tensor::from_vector(values_vec, {3}, Device::CPU);
    auto values_torch = torch::tensor(values_vec);

    t_custom.scatter_(0, indices_custom, values_custom);
    t_torch.scatter_(0, indices_torch, values_torch);

    compare_tensors(t_custom, t_torch, 1e-5f, 1e-7f, "ScatterBasic");
}

TEST_F(TensorIndexingAdvancedTest, GatherBasic) {
    auto t_custom = Tensor::arange(0.0f, 10.0f);
    auto t_torch = torch::arange(0, 10, torch::kFloat32);

    auto indices_vec = std::vector<int>{0, 2, 4, 6, 8};
    auto indices_custom = Tensor::from_vector(indices_vec, {5}, Device::CUDA);
    auto indices_torch = torch::tensor(indices_vec, torch::kInt64);

    auto result_custom = t_custom.gather(0, indices_custom);
    auto result_torch = t_torch.gather(0, indices_torch);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "GatherBasic");
}

TEST_F(TensorIndexingAdvancedTest, Gather2D) {
    std::vector<float> data_vec;
    for (int i = 0; i < 12; ++i)
        data_vec.push_back(static_cast<float>(i));

    auto t_custom = Tensor::from_vector(data_vec, {3, 4}, Device::CPU);
    auto t_torch = torch::tensor(data_vec).reshape({3, 4});

    auto indices_vec = std::vector<int>{0, 1, 2, 1};
    auto indices_custom = Tensor::from_vector(indices_vec, {4}, Device::CPU);

    // For 2D gather, need to expand indices properly
    auto indices_torch = torch::tensor(indices_vec, torch::kInt64).reshape({1, 4}).expand({3, 4});

    auto result_custom = t_custom.gather(1, indices_custom);
    auto result_torch = t_torch.gather(1, indices_torch);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "Gather2D");
}

// ============= Take Tests =============

TEST_F(TensorIndexingAdvancedTest, TakeBasic) {
    std::vector<float> data_vec;
    for (int i = 0; i < 12; ++i)
        data_vec.push_back(static_cast<float>(i));

    auto t_custom = Tensor::from_vector(data_vec, {3, 4}, Device::CPU);
    auto t_torch = torch::tensor(data_vec).reshape({3, 4});

    auto indices_vec = std::vector<int>{0, 5, 11, 3};
    auto indices_custom = Tensor::from_vector(indices_vec, {4}, Device::CPU);
    auto indices_torch = torch::tensor(indices_vec, torch::kInt64);

    auto result_custom = t_custom.take(indices_custom);
    auto result_torch = t_torch.take(indices_torch);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "TakeBasic");
}

// ============= Integration Tests =============

TEST_F(TensorIndexingAdvancedTest, ComplexIndexingChain) {
    auto t_custom = Tensor::arange(0.0f, 100.0f).reshape({10, 10});
    auto t_torch = torch::arange(0, 100, torch::kFloat32).reshape({10, 10});

    // Select specific rows
    auto row_vec = std::vector<int>{0, 5, 9};
    auto row_idx_custom = Tensor::from_vector(row_vec, {3}, Device::CUDA);
    auto row_idx_torch = torch::tensor(row_vec, torch::kInt64);

    auto rows_custom = t_custom.index_select(0, row_idx_custom);
    auto rows_torch = t_torch.index_select(0, row_idx_torch);

    // Then select specific columns
    auto col_vec = std::vector<int>{0, 5, 9};
    auto col_idx_custom = Tensor::from_vector(col_vec, {3}, Device::CUDA);
    auto col_idx_torch = torch::tensor(col_vec, torch::kInt64);

    auto result_custom = rows_custom.index_select(1, col_idx_custom);
    auto result_torch = rows_torch.index_select(1, col_idx_torch);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "ComplexIndexingChain");
}

TEST_F(TensorIndexingAdvancedTest, ScatterGatherRoundtrip) {
    auto original_custom = Tensor::arange(0.0f, 10.0f);
    auto original_torch = torch::arange(0, 10, torch::kFloat32);

    auto indices_vec = std::vector<int>{0, 2, 4, 6, 8};
    auto indices_custom = Tensor::from_vector(indices_vec, {5}, Device::CUDA);
    auto indices_torch = torch::tensor(indices_vec, torch::kInt64);

    // Gather
    auto gathered_custom = original_custom.gather(0, indices_custom);
    auto gathered_torch = original_torch.gather(0, indices_torch);

    // Scatter back
    auto result_custom = Tensor::zeros({10}, Device::CUDA);
    auto result_torch = torch::zeros({10});

    result_custom.scatter_(0, indices_custom, gathered_custom);
    result_torch.scatter_(0, indices_torch, gathered_torch);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "ScatterGatherRoundtrip");
}

// ============= Edge Cases =============

TEST_F(TensorIndexingAdvancedTest, EmptyIndices) {
    auto t_custom = Tensor::arange(0.0f, 10.0f);
    auto t_torch = torch::arange(0, 10, torch::kFloat32);

    auto indices_custom = Tensor::from_vector(std::vector<int>{}, {0}, Device::CUDA);
    auto indices_torch = torch::tensor({}, torch::kInt64);

    auto result_custom = t_custom.gather(0, indices_custom);
    auto result_torch = t_torch.gather(0, indices_torch);

    EXPECT_EQ(result_custom.numel(), 0);
    EXPECT_EQ(result_torch.numel(), 0);
}

TEST_F(TensorIndexingAdvancedTest, SingleElement) {
    auto t_custom = Tensor::from_vector(std::vector<float>{42.0f}, {1}, Device::CPU);
    auto t_torch = torch::tensor({42.0f});

    auto indices_custom = Tensor::from_vector(std::vector<int>{0}, {1}, Device::CPU);
    auto indices_torch = torch::tensor({0}, torch::kInt64);

    auto result_custom = t_custom.gather(0, indices_custom);
    auto result_torch = t_torch.gather(0, indices_torch);

    compare_tensors(result_custom, result_torch, 1e-5f, 1e-7f, "SingleElement");
}

TEST_F(TensorIndexingAdvancedTest, IndexCopyUpdatesColumnsFromSlicedExpressions) {
    auto tensor = Tensor::from_vector(
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f,
                           5.0f, 6.0f, 7.0f, 8.0f,
                           9.0f, 10.0f, 11.0f, 12.0f},
        {3, 4}, Device::CUDA);

    for (int column = 0; column < 4; ++column) {
        const auto index = Tensor::from_vector(
                               std::vector<int>{column}, {1}, Device::CUDA)
                               .to(DataType::Int32);
        const auto values = tensor.slice(1, column, column + 1).mul(2.0f).add(1.0f);
        tensor.index_copy_(1, index, values);
    }

    EXPECT_EQ(tensor.cpu().to_vector(),
              (std::vector<float>{3.0f, 5.0f, 7.0f, 9.0f,
                                  11.0f, 13.0f, 15.0f, 17.0f,
                                  19.0f, 21.0f, 23.0f, 25.0f}));
}

TEST_F(TensorIndexingAdvancedTest, IndexCopyQuaternionCompositionRegression) {
    constexpr float rw = 0.92388f;
    constexpr float ry = 0.382683f;
    const std::vector<float> input = {
        -1.2682f, -0.0383f, -0.1029f, 1.4400f,
        -0.4705f, 1.1624f, 0.3058f, 0.5276f,
        -0.5726f, 1.8732f, -0.6816f, -0.2104f};
    auto rotation = Tensor::from_vector(input, {3, 4}, Device::CUDA);

    const auto w = rotation.slice(1, 0, 1).squeeze(1);
    const auto x = rotation.slice(1, 1, 2).squeeze(1);
    const auto y = rotation.slice(1, 2, 3).squeeze(1);
    const auto z = rotation.slice(1, 3, 4).squeeze(1);
    const std::array components = {
        w.mul(rw).sub(y.mul(ry)),
        x.mul(rw).add(z.mul(ry)),
        w.mul(ry).add(y.mul(rw)),
        z.mul(rw).sub(x.mul(ry))};

    for (int column = 0; column < 4; ++column) {
        const auto index = Tensor::from_vector(
                               std::vector<int>{column}, {1}, Device::CUDA)
                               .to(DataType::Int32);
        rotation.index_copy_(1, index, components[column].unsqueeze(1));
    }

    const auto actual = rotation.cpu().to_vector();
    ASSERT_EQ(actual.size(), input.size());
    for (size_t row = 0; row < 3; ++row) {
        const float expected[] = {
            rw * input[row * 4] - ry * input[row * 4 + 2],
            rw * input[row * 4 + 1] + ry * input[row * 4 + 3],
            ry * input[row * 4] + rw * input[row * 4 + 2],
            rw * input[row * 4 + 3] - ry * input[row * 4 + 1]};
        for (size_t column = 0; column < 4; ++column) {
            EXPECT_NEAR(actual[row * 4 + column], expected[column], 1e-5f);
        }
    }
}

TEST_F(TensorIndexingAdvancedTest, IndexSelectHundredThousandRowsPreservesValues) {
    constexpr size_t source_rows = 100'000;
    constexpr size_t selected_rows = 99'880;
    const auto source = Tensor::arange(0.0f, static_cast<float>(source_rows * 4))
                            .reshape({static_cast<int>(source_rows), 4});
    std::vector<int> index_values(selected_rows);
    std::iota(index_values.begin(), index_values.end(), 0);
    const auto indices = Tensor::from_vector(
        index_values, {selected_rows}, Device::CUDA);

    const auto selected = source.index_select(0, indices).cpu().to_vector();

    ASSERT_EQ(selected.size(), selected_rows * 4);
    for (size_t i = 0; i < selected.size(); ++i) {
        EXPECT_FLOAT_EQ(selected[i], static_cast<float>(i));
    }
}
