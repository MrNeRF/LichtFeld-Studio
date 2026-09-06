/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cmath>
#include <core/tensor.hpp>
#include <gtest/gtest.h>
#include <vector>

using namespace lfs::core;

class TensorReductionAlignmentTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaDeviceSynchronize();
        cudaGetLastError();
    }

    void TearDown() override {
        cudaDeviceSynchronize();
        const cudaError_t err = cudaGetLastError();
        ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);
    }

    std::vector<float> cpu_sum_axis1(const std::vector<float>& data, size_t rows, size_t cols) {
        std::vector<float> result(rows, 0.0f);
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                result[r] += data[r * cols + c];
            }
        }
        return result;
    }
};

TEST_F(TensorReductionAlignmentTest, SumDim1_MisalignedSegments) {
    const std::vector<size_t> TEST_COLS = {50, 51, 53, 55, 61, 67, 73, 97, 101, 127};

    for (const size_t cols : TEST_COLS) {
        constexpr size_t ROWS = 100;

        std::vector<float> data(ROWS * cols);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<float>(i % 100) * 0.01f;
        }

        const auto expected = cpu_sum_axis1(data, ROWS, cols);

        const Tensor t = Tensor::from_blob(data.data(), TensorShape({ROWS, cols}),
                                           Device::CPU, DataType::Float32)
                             .gpu();
        const Tensor result = t.sum(1);

        cudaDeviceSynchronize();
        const cudaError_t err = cudaGetLastError();
        ASSERT_EQ(err, cudaSuccess) << "cols=" << cols << ": " << cudaGetErrorString(err);

        ASSERT_EQ(result.shape().rank(), 1);
        ASSERT_EQ(result.shape()[0], ROWS);

        const Tensor result_cpu = result.cpu();
        const float* const result_ptr = result_cpu.ptr<float>();

        for (size_t r = 0; r < ROWS; ++r) {
            EXPECT_NEAR(result_ptr[r], expected[r], 1e-3f)
                << "row " << r << ", cols=" << cols;
        }
    }
}

TEST_F(TensorReductionAlignmentTest, MeanDim1_MisalignedSegments) {
    const std::vector<size_t> TEST_COLS = {50, 53, 67, 101};

    for (const size_t cols : TEST_COLS) {
        constexpr size_t ROWS = 100;

        std::vector<float> data(ROWS * cols, 1.0f);

        const Tensor t = Tensor::from_blob(data.data(), TensorShape({ROWS, cols}),
                                           Device::CPU, DataType::Float32)
                             .gpu();
        const Tensor result = t.mean(1);

        cudaDeviceSynchronize();
        const cudaError_t err = cudaGetLastError();
        ASSERT_EQ(err, cudaSuccess) << "cols=" << cols << ": " << cudaGetErrorString(err);

        const Tensor result_cpu = result.cpu();
        const float* const result_ptr = result_cpu.ptr<float>();

        for (size_t r = 0; r < ROWS; ++r) {
            EXPECT_NEAR(result_ptr[r], 1.0f, 1e-5f) << "row " << r << ", cols=" << cols;
        }
    }
}

TEST_F(TensorReductionAlignmentTest, MaxDim1_MisalignedSegments) {
    const std::vector<size_t> TEST_COLS = {50, 53, 67, 101};

    for (const size_t cols : TEST_COLS) {
        constexpr size_t ROWS = 100;
        constexpr float MAX_VAL = 100.0f;

        std::vector<float> data(ROWS * cols);
        for (size_t r = 0; r < ROWS; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                data[r * cols + c] = (c == (r % cols)) ? MAX_VAL : static_cast<float>(c);
            }
        }

        const Tensor t = Tensor::from_blob(data.data(), TensorShape({ROWS, cols}),
                                           Device::CPU, DataType::Float32)
                             .gpu();
        const Tensor result = t.max(1);

        cudaDeviceSynchronize();
        const cudaError_t err = cudaGetLastError();
        ASSERT_EQ(err, cudaSuccess) << "cols=" << cols << ": " << cudaGetErrorString(err);

        const Tensor result_cpu = result.cpu();
        const float* const result_ptr = result_cpu.ptr<float>();

        for (size_t r = 0; r < ROWS; ++r) {
            EXPECT_NEAR(result_ptr[r], MAX_VAL, 1e-5f) << "row " << r << ", cols=" << cols;
        }
    }
}

TEST_F(TensorReductionAlignmentTest, MinDim1_MisalignedSegments) {
    const std::vector<size_t> TEST_COLS = {50, 53, 67, 101};

    for (const size_t cols : TEST_COLS) {
        constexpr size_t ROWS = 100;
        constexpr float MIN_VAL = -100.0f;

        std::vector<float> data(ROWS * cols);
        for (size_t r = 0; r < ROWS; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                data[r * cols + c] = (c == (r % cols)) ? MIN_VAL : static_cast<float>(c);
            }
        }

        const Tensor t = Tensor::from_blob(data.data(), TensorShape({ROWS, cols}),
                                           Device::CPU, DataType::Float32)
                             .gpu();
        const Tensor result = t.min(1);

        cudaDeviceSynchronize();
        const cudaError_t err = cudaGetLastError();
        ASSERT_EQ(err, cudaSuccess) << "cols=" << cols << ": " << cudaGetErrorString(err);

        const Tensor result_cpu = result.cpu();
        const float* const result_ptr = result_cpu.ptr<float>();

        for (size_t r = 0; r < ROWS; ++r) {
            EXPECT_NEAR(result_ptr[r], MIN_VAL, 1e-5f) << "row " << r << ", cols=" << cols;
        }
    }
}

TEST_F(TensorReductionAlignmentTest, CudaStateNotCorrupted) {
    constexpr size_t ROWS = 100;
    constexpr size_t COLS = 50;
    constexpr size_t ALLOC_SIZE = 1024 * 1024;

    {
        std::vector<float> data(ROWS * COLS, 1.0f);
        const Tensor t = Tensor::from_blob(data.data(), TensorShape({ROWS, COLS}),
                                           Device::CPU, DataType::Float32)
                             .gpu();
        const Tensor result = t.sum(1);
        cudaDeviceSynchronize();
    }

    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    void* ptr = nullptr;
    err = cudaMalloc(&ptr, ALLOC_SIZE);
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
    ASSERT_NE(ptr, nullptr);

    const Tensor new_tensor = Tensor::zeros({256, 256}, Device::GPU, DataType::Float32);
    const Tensor exp_result = new_tensor.exp();

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    cudaFree(ptr);
}

TEST_F(TensorReductionAlignmentTest, SumDim1_KernelBoundaries) {
    const std::vector<size_t> TEST_COLS = {31, 32, 33, 63, 64, 65};

    for (const size_t cols : TEST_COLS) {
        constexpr size_t ROWS = 100;

        std::vector<float> data(ROWS * cols, 1.0f);

        const Tensor t = Tensor::from_blob(data.data(), TensorShape({ROWS, cols}),
                                           Device::CPU, DataType::Float32)
                             .gpu();
        const Tensor result = t.sum(1);

        cudaDeviceSynchronize();
        const cudaError_t err = cudaGetLastError();
        ASSERT_EQ(err, cudaSuccess) << "cols=" << cols << ": " << cudaGetErrorString(err);

        const Tensor result_cpu = result.cpu();
        const float* const result_ptr = result_cpu.ptr<float>();

        for (size_t r = 0; r < ROWS; ++r) {
            EXPECT_NEAR(result_ptr[r], static_cast<float>(cols), 1e-3f)
                << "row " << r << ", cols=" << cols;
        }
    }
}

TEST_F(TensorReductionAlignmentTest, SumDim1_VariousRowCounts) {
    constexpr size_t COLS = 50;
    const std::vector<size_t> TEST_ROWS = {1, 7, 32, 100, 256, 1000, 10000};

    for (const size_t rows : TEST_ROWS) {
        std::vector<float> data(rows * COLS, 1.0f);

        const Tensor t = Tensor::from_blob(data.data(), TensorShape({rows, COLS}),
                                           Device::CPU, DataType::Float32)
                             .gpu();
        const Tensor result = t.sum(1);

        cudaDeviceSynchronize();
        const cudaError_t err = cudaGetLastError();
        ASSERT_EQ(err, cudaSuccess) << "rows=" << rows << ": " << cudaGetErrorString(err);

        const Tensor result_cpu = result.cpu();
        const float* const result_ptr = result_cpu.ptr<float>();

        for (size_t r = 0; r < rows; ++r) {
            EXPECT_NEAR(result_ptr[r], static_cast<float>(COLS), 1e-3f)
                << "row " << r << ", rows=" << rows;
        }
    }
}
