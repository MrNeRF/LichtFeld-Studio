/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "core/tensor/internal/stream_lifetime.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

using namespace lfs::core;

namespace {

    class CudaPoolStreamTeardownTest : public ::testing::Test {
    protected:
        void SetUp() override {
            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
            ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
        }
    };

} // namespace

TEST_F(CudaPoolStreamTeardownTest,
       TensorOutlivesReleasedD2HStreamDeallocatesSafely) {
    auto tensor = Tensor::empty({1 << 20}, Device::CUDA);

    cudaStream_t d2h = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&d2h, cudaStreamNonBlocking), cudaSuccess);
    prepare_inputs_for_stream({&tensor}, d2h);

    CudaMemoryPool::instance().release_stream(d2h);
    ASSERT_EQ(cudaStreamDestroy(d2h), cudaSuccess);

    tensor = Tensor{};
}

TEST_F(CudaPoolStreamTeardownTest,
       RecycledStreamHandleIsLiveAgainAfterRecordStream) {
    auto& pool = CudaMemoryPool::instance();

    cudaStream_t retired = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&retired, cudaStreamNonBlocking), cudaSuccess);
    pool.release_stream(retired);
    EXPECT_TRUE(is_stream_retired(retired));
    ASSERT_EQ(cudaStreamDestroy(retired), cudaSuccess);

    cudaStream_t recycled = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&recycled, cudaStreamNonBlocking), cudaSuccess);
    auto tensor = Tensor::empty({1 << 20}, Device::CUDA);
    prepare_inputs_for_stream({&tensor}, recycled);

    EXPECT_FALSE(is_stream_retired(recycled));
    tensor = Tensor{};

    pool.release_stream(recycled);
    ASSERT_EQ(cudaStreamDestroy(recycled), cudaSuccess);
}

TEST_F(CudaPoolStreamTeardownTest, ReserveCopiesAfterWritesQueuedOnTheTensorsStream) {
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    {
        const CUDAStreamGuard guard(stream);
        // Queue enough work that a copy ignoring the stream would run before the fill below.
        auto busy = Tensor::ones({2048, 2048}, Device::CUDA);
        for (int i = 0; i < 40; ++i)
            busy = busy.matmul(busy) * 1e-3f;
        // A value no earlier run left in the reused storage.
        static float run = 0.0f;
        const float value = 7.0f + ++run;
        auto values = Tensor::full({1 << 20}, value, Device::CUDA);
        values.reserve(1 << 21);
        const auto host = values.cpu().to_vector();
        ASSERT_EQ(host.size(), size_t{1} << 20);
        for (size_t i = 0; i < host.size(); ++i)
            ASSERT_EQ(host[i], value) << "element " << i;
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    }
    CudaMemoryPool::instance().release_stream(stream);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}
