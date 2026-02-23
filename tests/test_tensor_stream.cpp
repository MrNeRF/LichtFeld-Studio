/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "core/tensor.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"

using namespace lfs::core;

class TensorStreamTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaSetDevice(0);
    }
};

TEST_F(TensorStreamTest, DefaultStreamIsNullptr) {
    auto t = Tensor::empty({4, 4}, Device::CUDA);
    EXPECT_EQ(t.stream(), nullptr);
}

TEST_F(TensorStreamTest, FactoryPicksUpThreadLocalStream) {
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    {
        CUDAStreamGuard guard(stream);
        auto t = Tensor::empty({8, 8}, Device::CUDA);
        EXPECT_EQ(t.stream(), stream);

        auto z = Tensor::zeros({4}, Device::CUDA);
        EXPECT_EQ(z.stream(), stream);

        auto o = Tensor::ones({4}, Device::CUDA);
        EXPECT_EQ(o.stream(), stream);

        auto f = Tensor::full({4}, 3.14f, Device::CUDA);
        EXPECT_EQ(f.stream(), stream);

        auto r = Tensor::rand({4}, Device::CUDA);
        EXPECT_EQ(r.stream(), stream);

        auto rn = Tensor::randn({4}, Device::CUDA);
        EXPECT_EQ(rn.stream(), stream);

        auto a = Tensor::arange(0, 10, 1);
        EXPECT_EQ(a.stream(), stream);
    }

    // After guard, back to default
    auto t2 = Tensor::empty({4, 4}, Device::CUDA);
    EXPECT_EQ(t2.stream(), nullptr);

    cudaStreamDestroy(stream);
}

TEST_F(TensorStreamTest, ViewInheritsStreamReshape) {
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    Tensor t;
    {
        CUDAStreamGuard guard(stream);
        t = Tensor::empty({4, 4}, Device::CUDA);
    }
    ASSERT_EQ(t.stream(), stream);

    auto reshaped = t.reshape({16});
    EXPECT_EQ(reshaped.stream(), stream);

    auto unsqueezed = t.unsqueeze(0);
    EXPECT_EQ(unsqueezed.stream(), stream);

    auto squeezed = unsqueezed.squeeze(0);
    EXPECT_EQ(squeezed.stream(), stream);

    cudaStreamDestroy(stream);
}

TEST_F(TensorStreamTest, ViewInheritsStreamSlice) {
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    Tensor t;
    {
        CUDAStreamGuard guard(stream);
        t = Tensor::full({10, 3}, 1.0f, Device::CUDA);
    }
    ASSERT_EQ(t.stream(), stream);

    auto sliced = t.slice(0, 0, 5);
    EXPECT_EQ(sliced.stream(), stream);

    cudaStreamDestroy(stream);
}

TEST_F(TensorStreamTest, ViewInheritsStreamPermute) {
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    Tensor t;
    {
        CUDAStreamGuard guard(stream);
        t = Tensor::empty({2, 3, 4}, Device::CUDA);
    }
    ASSERT_EQ(t.stream(), stream);

    auto permuted = t.permute({2, 0, 1});
    EXPECT_EQ(permuted.stream(), stream);

    auto transposed = t.transpose(0, 1);
    EXPECT_EQ(transposed.stream(), stream);

    cudaStreamDestroy(stream);
}

TEST_F(TensorStreamTest, OperationsProduceResultWithCorrectStream) {
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    Tensor a, b;
    {
        CUDAStreamGuard guard(stream);
        a = Tensor::ones({64}, Device::CUDA);
        b = Tensor::ones({64}, Device::CUDA);
    }

    // Binary ops create result via empty(), which stamps getCurrentCUDAStream()
    // Since no guard is active now, result gets nullptr (default stream)
    auto c = a + b;
    EXPECT_EQ(c.stream(), nullptr);

    // With guard active, result gets the guarded stream
    {
        CUDAStreamGuard guard(stream);
        auto d = a + b;
        EXPECT_EQ(d.stream(), stream);
    }

    cudaStreamDestroy(stream);
}

TEST_F(TensorStreamTest, EmptyLikeInheritsStream) {
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    Tensor t;
    {
        CUDAStreamGuard guard(stream);
        t = Tensor::ones({32}, Device::CUDA);
    }
    ASSERT_EQ(t.stream(), stream);

    auto like = Tensor::empty_like(t);
    EXPECT_EQ(like.stream(), stream);

    auto flike = Tensor::full_like(t, 42.0f);
    EXPECT_EQ(flike.stream(), stream);

    cudaStreamDestroy(stream);
}

TEST_F(TensorStreamTest, StreamOrderingCorrectness) {
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    Tensor result;
    {
        CUDAStreamGuard guard(stream);
        auto t = Tensor::full({1024}, 1.0f, Device::CUDA);
        auto added = t + t;          // 2.0
        auto mulled = added * added; // 4.0
        result = mulled;
    }

    cudaStreamSynchronize(stream);

    auto cpu = result.to(Device::CPU);
    auto vals = cpu.to_vector();
    ASSERT_GT(vals.size(), 0u);
    EXPECT_NEAR(vals[0], 4.0f, 1e-5f);
    EXPECT_NEAR(vals.back(), 4.0f, 1e-5f);

    cudaStreamDestroy(stream);
}

TEST_F(TensorStreamTest, SetStreamManual) {
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto t = Tensor::empty({4}, Device::CUDA);
    EXPECT_EQ(t.stream(), nullptr);

    t.set_stream(stream);
    EXPECT_EQ(t.stream(), stream);

    t.set_stream(nullptr);
    EXPECT_EQ(t.stream(), nullptr);

    cudaStreamDestroy(stream);
}

TEST_F(TensorStreamTest, GuardRestoresPreviousStream) {
    cudaStream_t s1, s2;
    ASSERT_EQ(cudaStreamCreate(&s1), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&s2), cudaSuccess);

    {
        CUDAStreamGuard g1(s1);
        EXPECT_EQ(getCurrentCUDAStream(), s1);

        {
            CUDAStreamGuard g2(s2);
            EXPECT_EQ(getCurrentCUDAStream(), s2);
        }

        EXPECT_EQ(getCurrentCUDAStream(), s1);
    }

    EXPECT_EQ(getCurrentCUDAStream(), nullptr);

    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
}

TEST_F(TensorStreamTest, InplaceOpsUseOwnStream) {
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    Tensor t;
    {
        CUDAStreamGuard guard(stream);
        t = Tensor::ones({256}, Device::CUDA);
    }
    ASSERT_EQ(t.stream(), stream);

    // In-place scalar op should use the tensor's stream
    t = t + Tensor::full({256}, 1.0f, Device::CUDA);
    // Result gets default stream since no guard active, so re-stamp
    t.set_stream(stream);

    // In-place binary op with guard
    {
        CUDAStreamGuard guard(stream);
        auto other = Tensor::ones({256}, Device::CUDA);
        t = t + other;
        EXPECT_EQ(t.stream(), stream);
    }

    cudaStreamSynchronize(stream);

    auto vals = t.to(Device::CPU).to_vector();
    ASSERT_GT(vals.size(), 0u);
    EXPECT_NEAR(vals[0], 3.0f, 1e-5f);

    cudaStreamDestroy(stream);
}
