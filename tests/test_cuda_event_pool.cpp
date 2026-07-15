/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

#include "core/tensor/internal/cuda_event_pool.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"

using namespace lfs::core;

class CudaEventPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    }
};

TEST_F(CudaEventPoolTest, AcquireReleaseReusesEvents) {
    auto& pool = CudaEventPool::instance();

    cudaEvent_t first = pool.acquire();
    ASSERT_NE(first, nullptr);
    pool.release(first);

    const uint64_t reused_before = pool.stats().reused.load();
    cudaEvent_t second = pool.acquire();
    ASSERT_NE(second, nullptr);
    EXPECT_GT(pool.stats().reused.load(), reused_before);
    pool.release(second);
}

TEST_F(CudaEventPoolTest, WaitForCUDAStreamStressDoesNotCreatePerCall) {
    auto& pool = CudaEventPool::instance();

    cudaStream_t a, b;
    ASSERT_EQ(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking), cudaSuccess);

    waitForCUDAStream(a, b);
    const uint64_t created_before = pool.stats().created.load();

    constexpr int kIterations = 256;
    for (int i = 0; i < kIterations; ++i) {
        waitForCUDAStream(a, b);
        waitForCUDAStream(b, a);
    }

    EXPECT_EQ(pool.stats().created.load(), created_before);

    ASSERT_EQ(cudaStreamSynchronize(a), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(b), cudaSuccess);
    cudaStreamDestroy(a);
    cudaStreamDestroy(b);
}

TEST_F(CudaEventPoolTest, ConcurrentAcquireReleaseIsSafe) {
    auto& pool = CudaEventPool::instance();

    constexpr int kThreads = 4;
    constexpr int kIterations = 200;
    std::vector<std::thread> threads;
    std::atomic<bool> failed{false};

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&pool, &failed] {
            for (int i = 0; i < kIterations; ++i) {
                cudaEvent_t event = pool.acquire();
                if (!event) {
                    failed.store(true);
                    return;
                }
                pool.release(event);
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_FALSE(failed.load());
}

TEST_F(CudaEventPoolTest, WaitOrdersCrossStreamWork) {
    cudaStream_t producer, consumer;
    ASSERT_EQ(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking), cudaSuccess);

    constexpr size_t kBytes = 1ull * 1024 * 1024;
    void* buffer = nullptr;
    ASSERT_EQ(cudaMalloc(&buffer, kBytes), cudaSuccess);

    ASSERT_EQ(cudaMemsetAsync(buffer, 0xAB, kBytes, producer), cudaSuccess);
    waitForCUDAStream(consumer, producer);

    std::vector<unsigned char> host(kBytes);
    ASSERT_EQ(cudaMemcpyAsync(host.data(), buffer, kBytes, cudaMemcpyDeviceToHost, consumer), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(consumer), cudaSuccess);

    EXPECT_EQ(host.front(), 0xAB);
    EXPECT_EQ(host[kBytes / 2], 0xAB);
    EXPECT_EQ(host.back(), 0xAB);

    cudaFree(buffer);
    cudaStreamDestroy(producer);
    cudaStreamDestroy(consumer);
}
