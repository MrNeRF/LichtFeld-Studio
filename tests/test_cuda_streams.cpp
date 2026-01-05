/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Tests for CUDA stream pool, events, context, and tensor stream propagation

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
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        ASSERT_EQ(err, cudaSuccess) << "CUDA not available";
        ASSERT_GT(device_count, 0) << "No CUDA devices found";
        cudaSetDevice(0);
        cudaDeviceSynchronize();
        Tensor::manual_seed(42);
    }

    void TearDown() override {
        cudaDeviceSynchronize();
    }
};

// === CUDAStreamPool ===

TEST_F(CUDAStreamsTest, StreamPoolInitialization) {
    auto& pool = CUDAStreamPool::instance();
    EXPECT_TRUE(pool.is_initialized());
    EXPECT_EQ(pool.size(), CUDAStreamPool::DEFAULT_POOL_SIZE);
    EXPECT_EQ(pool.high_priority_size(), CUDAStreamPool::HIGH_PRIORITY_POOL_SIZE);
}

TEST_F(CUDAStreamsTest, StreamPoolAcquire) {
    auto& pool = CUDAStreamPool::instance();
    cudaStream_t s1 = pool.acquire();
    cudaStream_t s2 = pool.acquire();
    EXPECT_NE(s1, nullptr);
    EXPECT_NE(s2, nullptr);

    // Verify round-robin wraps around
    std::vector<cudaStream_t> streams;
    for (size_t i = 0; i < pool.size(); ++i)
        streams.push_back(pool.acquire());
    cudaStream_t wrapped = pool.acquire();
    bool found = std::find(streams.begin(), streams.end(), wrapped) != streams.end();
    EXPECT_TRUE(found) << "Round-robin should wrap";
}

TEST_F(CUDAStreamsTest, StreamPoolHighPriority) {
    auto& pool = CUDAStreamPool::instance();
    cudaStream_t hp = pool.acquire_high_priority();
    EXPECT_NE(hp, nullptr);
    EXPECT_NE(hp, pool.get(0)); // Different from regular
}

TEST_F(CUDAStreamsTest, StreamPoolSynchronizeAll) {
    auto& pool = CUDAStreamPool::instance();
    auto t1 = Tensor::randn({1000, 1000}, Device::CUDA);
    auto t2 = Tensor::randn({1000, 1000}, Device::CUDA);
    pool.synchronize_all();
    EXPECT_TRUE(t1.is_valid());
    EXPECT_TRUE(t2.is_valid());
}

// === CUDAEvent ===

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

// === CUDAStreamContext ===

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

// === Tensor Stream Propagation ===

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

// === Concurrent Streams ===

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

// === Memory Pool Integration ===

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

// === Backward Compatibility ===

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

// === Performance ===

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

// === Regression: GPU-native fill/arange ===

TEST_F(CUDAStreamsTest, TensorConstantFillInt32Correctness) {
    constexpr int TEST_VALUE = 42;
    constexpr size_t SIZE = 10000;

    const auto tensor = Tensor::full({SIZE}, static_cast<float>(TEST_VALUE), Device::CUDA, DataType::Int32);
    ASSERT_TRUE(tensor.is_valid());
    ASSERT_EQ(tensor.numel(), SIZE);

    const auto cpu_tensor = tensor.to(Device::CPU);
    const int* data = cpu_tensor.ptr<int>();
    for (size_t i = 0; i < SIZE; ++i) {
        EXPECT_EQ(data[i], TEST_VALUE) << "Mismatch at index " << i;
    }
}

TEST_F(CUDAStreamsTest, TensorConstantFillInt64Correctness) {
    constexpr int64_t TEST_VALUE = 123456789LL;
    constexpr size_t SIZE = 10000;

    const auto tensor = Tensor::full({SIZE}, static_cast<float>(TEST_VALUE), Device::CUDA, DataType::Int64);
    ASSERT_TRUE(tensor.is_valid());
    ASSERT_EQ(tensor.numel(), SIZE);

    const auto cpu_tensor = tensor.to(Device::CPU);
    const int64_t* data = cpu_tensor.ptr<int64_t>();
    for (size_t i = 0; i < SIZE; ++i) {
        EXPECT_EQ(data[i], static_cast<int64_t>(static_cast<float>(TEST_VALUE))) << "Mismatch at index " << i;
    }
}

TEST_F(CUDAStreamsTest, TensorArangeCorrectnessFloat32) {
    constexpr float START = 0.0f, END = 1000.0f, STEP = 1.0f;

    const auto tensor = Tensor::arange(START, END, STEP);
    ASSERT_TRUE(tensor.is_valid());
    ASSERT_EQ(tensor.numel(), static_cast<size_t>(END - START));

    const auto cpu_tensor = tensor.to(Device::CPU);
    const float* data = cpu_tensor.ptr<float>();
    for (size_t i = 0; i < tensor.numel(); ++i) {
        const float expected = START + i * STEP;
        EXPECT_FLOAT_EQ(data[i], expected) << "Mismatch at index " << i;
    }
}

TEST_F(CUDAStreamsTest, TensorConstantFillStressTest) {
    // Stress test for async race conditions
    constexpr int ITERATIONS = 100;
    constexpr size_t SIZE = 5000;

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        const int test_value = iter * 7 + 13;
        const auto tensor = Tensor::full({SIZE}, static_cast<float>(test_value), Device::CUDA, DataType::Int32);
        const auto cpu_tensor = tensor.to(Device::CPU);
        const int* data = cpu_tensor.ptr<int>();

        for (size_t i = 0; i < SIZE; ++i) {
            if (data[i] != test_value) {
                FAIL() << "Iter " << iter << " idx " << i << ": got " << data[i] << " expected " << test_value;
            }
        }
    }
}
