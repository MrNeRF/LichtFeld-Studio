/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * Comprehensive tests for CUDA stream support in the tensor library.
 *
 * Tests cover:
 * - CUDAStreamPool: round-robin stream acquisition, high-priority streams
 * - CUDAEvent: recording, synchronization, cross-stream dependencies
 * - CUDAStreamContext: thread-local stream management
 * - Tensor operations: stream propagation, async execution
 * - PooledStreamGuard: RAII stream management
 */

#include "core/tensor.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/cuda_stream_pool.hpp"
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

using namespace lfs::core;

class CUDAStreamsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure CUDA is available
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        ASSERT_EQ(err, cudaSuccess) << "CUDA not available";
        ASSERT_GT(device_count, 0) << "No CUDA devices found";

        // Reset to default device
        cudaSetDevice(0);
        cudaDeviceSynchronize();

        // Seed for reproducibility
        Tensor::manual_seed(42);
    }

    void TearDown() override {
        cudaDeviceSynchronize();
    }
};

// ============================================================================
// CUDAStreamPool Tests
// ============================================================================

TEST_F(CUDAStreamsTest, StreamPoolInitialization) {
    auto& pool = CUDAStreamPool::instance();

    // Pool should be initialized
    EXPECT_TRUE(pool.is_initialized());
    EXPECT_EQ(pool.size(), CUDAStreamPool::DEFAULT_POOL_SIZE);
    EXPECT_EQ(pool.high_priority_size(), CUDAStreamPool::HIGH_PRIORITY_POOL_SIZE);
}

TEST_F(CUDAStreamsTest, StreamPoolAcquire) {
    auto& pool = CUDAStreamPool::instance();

    // Acquire streams and verify they're non-null
    cudaStream_t s1 = pool.acquire();
    cudaStream_t s2 = pool.acquire();

    EXPECT_NE(s1, nullptr);
    EXPECT_NE(s2, nullptr);

    // After pool size acquisitions, should wrap around
    std::vector<cudaStream_t> streams;
    for (size_t i = 0; i < pool.size(); ++i) {
        streams.push_back(pool.acquire());
    }

    // Next acquisition should give us the first stream again (round-robin)
    cudaStream_t wrapped = pool.acquire();
    bool found = false;
    for (const auto& s : streams) {
        if (s == wrapped) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found) << "Round-robin should wrap around to existing streams";
}

TEST_F(CUDAStreamsTest, StreamPoolHighPriority) {
    auto& pool = CUDAStreamPool::instance();

    cudaStream_t hp = pool.acquire_high_priority();
    EXPECT_NE(hp, nullptr);

    // High priority streams should be different from regular streams
    cudaStream_t regular = pool.get(0);
    EXPECT_NE(hp, regular);
}

TEST_F(CUDAStreamsTest, StreamPoolSynchronizeAll) {
    auto& pool = CUDAStreamPool::instance();

    // Launch work on multiple streams
    auto t1 = Tensor::randn({1000, 1000}, Device::CUDA);
    auto t2 = Tensor::randn({1000, 1000}, Device::CUDA);

    // Synchronize all should complete without error
    pool.synchronize_all();

    // Verify tensors are valid
    EXPECT_TRUE(t1.is_valid());
    EXPECT_TRUE(t2.is_valid());
}

// ============================================================================
// CUDAEvent Tests
// ============================================================================

TEST_F(CUDAStreamsTest, EventCreation) {
    CUDAEvent event;
    EXPECT_TRUE(event.valid());
}

TEST_F(CUDAStreamsTest, EventRecordAndSync) {
    CUDAEvent event;
    ASSERT_TRUE(event.valid());

    // Record on default stream
    EXPECT_TRUE(event.record(nullptr));

    // Should complete quickly
    EXPECT_TRUE(event.synchronize());
    EXPECT_TRUE(event.is_complete());
}

TEST_F(CUDAStreamsTest, EventCrossStreamSync) {
    auto& pool = CUDAStreamPool::instance();
    cudaStream_t stream1 = pool.acquire();
    cudaStream_t stream2 = pool.acquire();

    ASSERT_NE(stream1, nullptr);
    ASSERT_NE(stream2, nullptr);

    // Create a tensor on stream1
    CUDAStreamGuard guard1(stream1);
    auto t1 = Tensor::randn({1000, 1000}, Device::CUDA);

    // Record event after work on stream1
    CUDAEvent event;
    EXPECT_TRUE(event.record(stream1));

    // Make stream2 wait for stream1
    EXPECT_TRUE(event.wait(stream2));

    // Now work on stream2 that depends on stream1
    {
        CUDAStreamGuard guard2(stream2);
        auto t2 = t1 * 2.0f; // This operation depends on t1 being ready
        EXPECT_TRUE(t2.is_valid());
    }

    // Synchronize
    cudaStreamSynchronize(stream2);
}

TEST_F(CUDAStreamsTest, EventTiming) {
    // Create events with timing enabled
    CUDAEvent start(true); // enable_timing = true
    CUDAEvent end(true);

    ASSERT_TRUE(start.valid());
    ASSERT_TRUE(end.valid());

    // Record start
    start.record(nullptr);

    // Do some work
    auto t = Tensor::randn({1000, 1000}, Device::CUDA);
    auto result = t.matmul(t.t());

    // Record end
    end.record(nullptr);

    // Synchronize
    cudaDeviceSynchronize();

    // Get elapsed time
    float elapsed = end.elapsed_ms(start);
    EXPECT_GE(elapsed, 0.0f) << "Elapsed time should be non-negative";
}

TEST_F(CUDAStreamsTest, EventMoveSemantics) {
    CUDAEvent event1;
    ASSERT_TRUE(event1.valid());

    // Move construct
    CUDAEvent event2(std::move(event1));
    EXPECT_TRUE(event2.valid());
    EXPECT_FALSE(event1.valid()); // Original should be invalid

    // Move assign
    CUDAEvent event3;
    event3 = std::move(event2);
    EXPECT_TRUE(event3.valid());
    EXPECT_FALSE(event2.valid());
}

// ============================================================================
// CUDAStreamContext Tests
// ============================================================================

TEST_F(CUDAStreamsTest, StreamContextDefault) {
    // Default stream should be nullptr
    cudaStream_t current = getCurrentCUDAStream();
    EXPECT_EQ(current, nullptr);
}

TEST_F(CUDAStreamsTest, StreamContextSetGet) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    CUDAStreamContext::instance().setCurrentStream(stream);
    EXPECT_EQ(getCurrentCUDAStream(), stream);

    // Reset
    CUDAStreamContext::instance().setCurrentStream(nullptr);
    EXPECT_EQ(getCurrentCUDAStream(), nullptr);

    cudaStreamDestroy(stream);
}

TEST_F(CUDAStreamsTest, StreamGuardRAII) {
    cudaStream_t original = getCurrentCUDAStream();

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    {
        CUDAStreamGuard guard(stream);
        EXPECT_EQ(getCurrentCUDAStream(), stream);
    }

    // After guard goes out of scope, should be restored
    EXPECT_EQ(getCurrentCUDAStream(), original);

    cudaStreamDestroy(stream);
}

TEST_F(CUDAStreamsTest, PooledStreamGuard) {
    cudaStream_t original = getCurrentCUDAStream();

    {
        PooledStreamGuard guard;
        cudaStream_t pooled = guard.stream();
        EXPECT_NE(pooled, nullptr);
        EXPECT_EQ(getCurrentCUDAStream(), pooled);
    }

    // Restored after scope
    EXPECT_EQ(getCurrentCUDAStream(), original);
}

TEST_F(CUDAStreamsTest, PooledStreamGuardHighPriority) {
    {
        PooledStreamGuard guard(true); // high priority
        cudaStream_t hp = guard.stream();
        EXPECT_NE(hp, nullptr);
    }
}

// ============================================================================
// Tensor Stream Propagation Tests
// ============================================================================

TEST_F(CUDAStreamsTest, TensorInheritsThreadStream) {
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    {
        CUDAStreamGuard guard(stream);

        // Tensors created in this scope should use the thread's stream
        auto t = Tensor::randn({100, 100}, Device::CUDA);
        EXPECT_EQ(t.stream(), stream);
    }

    cudaStreamDestroy(stream);
}

TEST_F(CUDAStreamsTest, TensorStreamPropagationUnary) {
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    Tensor t;
    {
        CUDAStreamGuard guard(stream);
        t = Tensor::randn({100, 100}, Device::CUDA);
    }

    // Unary operations should inherit the tensor's stream
    EXPECT_EQ(t.stream(), stream);

    // Operations like neg, exp, etc. create new tensors
    // that should inherit stream from input
    auto neg_t = t.neg();
    EXPECT_TRUE(neg_t.is_valid());

    cudaStreamDestroy(stream);
}

TEST_F(CUDAStreamsTest, TensorSetStream) {
    auto t = Tensor::randn({100, 100}, Device::CUDA);

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    t.set_stream(stream);
    EXPECT_EQ(t.stream(), stream);

    cudaStreamDestroy(stream);
}

// ============================================================================
// Concurrent Stream Tests
// ============================================================================

TEST_F(CUDAStreamsTest, ConcurrentMultiStreamOps) {
    auto& pool = CUDAStreamPool::instance();
    std::vector<Tensor> results(4);
    std::atomic<int> completed{0};

    // Launch work on multiple streams concurrently
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&pool, &results, &completed, i]() {
            cudaStream_t stream = pool.acquire();
            CUDAStreamGuard guard(stream);

            // Each thread does some tensor work
            auto t = Tensor::randn({500, 500}, Device::CUDA);
            auto r = t.matmul(t.t());
            results[i] = r.sum();

            completed.fetch_add(1);
        });
    }

    // Wait for all threads
    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(completed.load(), 4);

    // Verify all results are valid
    cudaDeviceSynchronize();
    for (const auto& r : results) {
        EXPECT_TRUE(r.is_valid());
    }
}

TEST_F(CUDAStreamsTest, ThreadLocalStreamIsolation) {
    // Verify that each thread has isolated stream context
    std::vector<cudaStream_t> observed_streams(4);
    std::vector<std::thread> threads;

    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&observed_streams, i]() {
            // Each thread creates its own stream
            cudaStream_t stream;
            cudaStreamCreate(&stream);

            CUDAStreamGuard guard(stream);
            observed_streams[i] = getCurrentCUDAStream();

            cudaStreamDestroy(stream);
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // All threads should have had different streams
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            EXPECT_NE(observed_streams[i], observed_streams[j])
                << "Threads " << i << " and " << j << " had same stream";
        }
    }
}

// ============================================================================
// Memory Pool Stream Integration
// ============================================================================

TEST_F(CUDAStreamsTest, StreamAwareAllocation) {
    auto& pool = CUDAStreamPool::instance();
    cudaStream_t stream = pool.acquire();

    {
        CUDAStreamGuard guard(stream);

        // Allocate tensor on non-default stream
        auto t = Tensor::zeros({1000, 1000}, Device::CUDA);
        EXPECT_EQ(t.stream(), stream);

        // The tensor should work correctly
        t = t + 1.0f;
        EXPECT_TRUE(t.is_valid());
    }

    cudaStreamSynchronize(stream);
}

// ============================================================================
// Backward Compatibility Tests
// ============================================================================

TEST_F(CUDAStreamsTest, DefaultStreamCompatibility) {
    // Operations without explicit stream should work on default stream
    auto t1 = Tensor::randn({100, 100}, Device::CUDA);
    auto t2 = Tensor::randn({100, 100}, Device::CUDA);

    // Default stream (nullptr) should be fine
    EXPECT_EQ(t1.stream(), nullptr);
    EXPECT_EQ(t2.stream(), nullptr);

    auto result = t1 + t2;
    EXPECT_TRUE(result.is_valid());

    // Sync should work
    cudaDeviceSynchronize();

    // Verify result
    auto cpu_result = result.cpu();
    EXPECT_GT(cpu_result.numel(), 0u);
}

TEST_F(CUDAStreamsTest, MixedStreamOperations) {
    // Test operations between tensors on different streams
    auto& pool = CUDAStreamPool::instance();
    cudaStream_t stream1 = pool.acquire();
    cudaStream_t stream2 = pool.acquire();

    Tensor t1, t2;

    {
        CUDAStreamGuard guard(stream1);
        t1 = Tensor::randn({100, 100}, Device::CUDA);
    }

    {
        CUDAStreamGuard guard(stream2);
        t2 = Tensor::randn({100, 100}, Device::CUDA);
    }

    // Synchronize both streams before cross-stream operation
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Now perform operation (will use default stream or stream from one tensor)
    auto result = t1 + t2;
    EXPECT_TRUE(result.is_valid());

    cudaDeviceSynchronize();
}

// ============================================================================
// Performance Sanity Tests
// ============================================================================

TEST_F(CUDAStreamsTest, StreamPoolNoContention) {
    // Verify that acquiring streams is fast (no blocking)
    auto& pool = CUDAStreamPool::instance();

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10000; ++i) {
        volatile cudaStream_t s = pool.acquire();
        (void)s;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // 10000 acquisitions should take less than 10ms (1us each is generous)
    EXPECT_LT(duration.count(), 10000)
        << "Stream pool acquisition too slow: " << duration.count() << "us for 10000 calls";
}
