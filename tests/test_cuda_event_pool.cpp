/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <algorithm>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <thread>
#include <vector>

#include "core/tensor.hpp"
#include "core/tensor/backend/cuda/runtime/cuda_event_pool.hpp"
#include "core/tensor/backend/cuda/runtime/cuda_stream_context.hpp"
#include "core/tensor/backend/cuda/runtime/memory_pool.hpp"
#include "core/tensor/backend/cuda/runtime/stream_lifetime.hpp"

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

TEST_F(CudaEventPoolTest, FreshStreamHandlesReuseRetiredValuesUntilTheyEnterTheApi) {
    // The driver hands out the handle values of destroyed streams again, so the
    // retired-stream registry (which lets bridgeStreams skip dead home streams
    // without touching the driver) can name a live stream. Every entry point a
    // caller-live handle passes through must drop it from the registry;
    // otherwise a bridge from that stream is skipped and its consumer races.
    std::vector<cudaStream_t> seeds(8);
    for (auto& seed : seeds) {
        ASSERT_EQ(cudaStreamCreateWithFlags(&seed, cudaStreamNonBlocking), cudaSuccess);
        auto touched = Tensor::empty({64}, Device::CUDA);
        touched.set_stream(seed);
        touched.fill_(1.0f);
    }
    for (auto& seed : seeds) {
        CudaMemoryPool::instance().release_stream(seed);
        cudaStreamDestroy(seed);
    }
    cudaStream_t producer = nullptr;
    cudaStream_t consumer = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking), cudaSuccess);
    const bool producer_reused =
        std::find(seeds.begin(), seeds.end(), producer) != seeds.end();
    if (producer_reused) {
        EXPECT_TRUE(is_stream_retired(producer))
            << "a reused handle still sits in the registry until it enters the API";
    }

    constexpr size_t kBytes = 64ull * 1024 * 1024;
    void* buffer = nullptr;
    ASSERT_EQ(cudaMalloc(&buffer, kBytes), cudaSuccess);
    for (int pass = 0; pass < 16; ++pass) {
        ASSERT_EQ(cudaMemsetAsync(buffer, 0x11, kBytes, producer), cudaSuccess);
    }
    ASSERT_EQ(cudaMemsetAsync(buffer, 0x6d, kBytes, producer), cudaSuccess);
    waitForCUDAStream(consumer, producer);
    EXPECT_FALSE(is_stream_retired(producer));
    EXPECT_FALSE(is_stream_retired(consumer));

    std::vector<unsigned char> host(kBytes);
    ASSERT_EQ(cudaMemcpyAsync(host.data(), buffer, kBytes, cudaMemcpyDeviceToHost, consumer),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(consumer), cudaSuccess);
    EXPECT_EQ(host.front(), 0x6d);
    EXPECT_EQ(host[kBytes / 2], 0x6d);
    EXPECT_EQ(host.back(), 0x6d);

    {
        cudaStream_t guarded = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&guarded, cudaStreamNonBlocking), cudaSuccess);
        CUDAStreamGuard guard(guarded);
        EXPECT_FALSE(is_stream_retired(guarded));
        cudaStreamDestroy(guarded);
    }

    cudaFree(buffer);
    cudaStreamDestroy(producer);
    cudaStreamDestroy(consumer);
}

TEST_F(CudaEventPoolTest, EventAcquireFailureSynchronizesProducerFallback) {
    cudaStream_t producer, consumer;
    ASSERT_EQ(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking), cudaSuccess);

    constexpr size_t kBytes = 1ull * 1024 * 1024;
    void* buffer = nullptr;
    ASSERT_EQ(cudaMalloc(&buffer, kBytes), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(buffer, 0x6d, kBytes, producer), cudaSuccess);

    // The public entry: bridgeStreams itself takes stored home streams and
    // skips retired handles, and a fresh handle may reuse a retired value.
    set_cuda_event_acquire_failure_for_testing(true);
    waitForCUDAStream(consumer, producer);
    set_cuda_event_acquire_failure_for_testing(false);

    std::vector<unsigned char> host(kBytes);
    ASSERT_EQ(cudaMemcpyAsync(host.data(), buffer, kBytes, cudaMemcpyDeviceToHost, consumer),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(consumer), cudaSuccess);
    EXPECT_EQ(host.front(), 0x6d);
    EXPECT_EQ(host[kBytes / 2], 0x6d);
    EXPECT_EQ(host.back(), 0x6d);

    cudaFree(buffer);
    cudaStreamDestroy(producer);
    cudaStreamDestroy(consumer);
}
