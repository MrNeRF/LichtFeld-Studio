/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <atomic>
#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <mutex>
#include <thread>
#include <vector>

#include "core/cuda/memory_arena.hpp"
#include "core/tensor.hpp"
#include "core/tensor/internal/cuda_event_pool.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/gpu_slab_allocator.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "core/tensor/internal/size_bucketed_pool.hpp"

using namespace lfs::core;

namespace {

    constexpr size_t SLAB_BYTES = 64 * 1024;
    constexpr size_t BUCKET_BYTES = 4 * 1024 * 1024;

    void seedPinnedCache(const size_t bytes) {
        auto& pinned = PinnedMemoryAllocator::instance();
        void* first = pinned.allocate(bytes);
        void* second = pinned.allocate(bytes);
        ASSERT_NE(first, nullptr);
        ASSERT_NE(second, nullptr);
        pinned.deallocate(first);
        pinned.deallocate(second);
        ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    }

    void destroyStreamSafely(cudaStream_t stream) {
        CudaMemoryPool::instance().release_stream(stream);
        cudaStreamDestroy(stream);
    }

    class GateStream {
    public:
        GateStream() {
            EXPECT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), cudaSuccess);
            EXPECT_EQ(cudaEventCreateWithFlags(&gate_, cudaEventDisableTiming), cudaSuccess);
            EXPECT_EQ(cudaStreamCreateWithFlags(&gate_holder_, cudaStreamNonBlocking), cudaSuccess);
        }

        ~GateStream() {
            release();
            destroyStreamSafely(stream_);
            cudaStreamDestroy(gate_holder_);
            cudaEventDestroy(gate_);
        }

        // Blocks `stream_` behind a host-controlled gate so work enqueued after
        // close() cannot run until release() — widens race windows deterministically.
        void close() {
            EXPECT_EQ(cudaLaunchHostFunc(
                          gate_holder_,
                          [](void* userData) {
                              auto* released = static_cast<std::atomic<bool>*>(userData);
                              while (!released->load(std::memory_order_acquire)) {
                              }
                          },
                          &released_),
                      cudaSuccess);
            EXPECT_EQ(cudaEventRecord(gate_, gate_holder_), cudaSuccess);
            EXPECT_EQ(cudaStreamWaitEvent(stream_, gate_, 0), cudaSuccess);
        }

        void release() {
            released_.store(true, std::memory_order_release);
        }

        cudaStream_t get() const { return stream_; }

    private:
        cudaStream_t stream_ = nullptr;
        cudaStream_t gate_holder_ = nullptr;
        cudaEvent_t gate_ = nullptr;
        std::atomic<bool> released_{false};
    };

} // namespace

class TensorMultiStreamTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        auto& pinned = PinnedMemoryAllocator::instance();
        original_cache_limit_ = pinned.cache_limit_bytes();
        pinned.set_force_fallback_for_testing(false);
        pinned.set_enabled(true);
    }

    void TearDown() override {
        auto& pinned = PinnedMemoryAllocator::instance();
        pinned.set_force_fallback_for_testing(false);
        pinned.set_enabled(true);
        pinned.set_cache_limit_for_testing(original_cache_limit_);
        pinned.empty_cache();
    }

    size_t original_cache_limit_ = 0;
};

TEST_F(TensorMultiStreamTest, SlabSameStreamReuseIsImmediateAndStealFree) {
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    auto& pool = CudaMemoryPool::instance();
    const uint64_t steals_before = GPUSlabAllocator::instance().stats().steal_count.load();

    void* first = pool.allocate(SLAB_BYTES, stream);
    ASSERT_NE(first, nullptr);
    pool.deallocate(first, stream);

    void* second = pool.allocate(SLAB_BYTES, stream);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(second, first);
    EXPECT_EQ(GPUSlabAllocator::instance().stats().steal_count.load(), steals_before);

    pool.deallocate(second, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    destroyStreamSafely(stream);
}

TEST_F(TensorMultiStreamTest, SlabCrossStreamStealIsOrdered) {
    GateStream producer;
    cudaStream_t consumer;
    ASSERT_EQ(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking), cudaSuccess);

    auto& pool = CudaMemoryPool::instance();

    void* block = pool.allocate(SLAB_BYTES, producer.get());
    ASSERT_NE(block, nullptr);

    producer.close();
    ASSERT_EQ(cudaMemsetAsync(block, 0xAA, SLAB_BYTES, producer.get()), cudaSuccess);
    pool.deallocate(block, producer.get());

    const uint64_t steals_before = GPUSlabAllocator::instance().stats().steal_count.load();
    void* stolen = pool.allocate(SLAB_BYTES, consumer);
    ASSERT_NE(stolen, nullptr);

    if (stolen == block) {
        EXPECT_GT(GPUSlabAllocator::instance().stats().steal_count.load(), steals_before);
        ASSERT_EQ(cudaMemsetAsync(stolen, 0xBB, SLAB_BYTES, consumer), cudaSuccess);

        producer.release();
        ASSERT_EQ(cudaStreamSynchronize(consumer), cudaSuccess);

        std::vector<unsigned char> host(SLAB_BYTES);
        ASSERT_EQ(cudaMemcpy(host.data(), stolen, SLAB_BYTES, cudaMemcpyDeviceToHost), cudaSuccess);
        for (size_t i = 0; i < SLAB_BYTES; i += 4096) {
            ASSERT_EQ(host[i], 0xBB) << "producer write ordered after steal at offset " << i;
        }
    } else {
        producer.release();
    }

    pool.deallocate(stolen, consumer);
    ASSERT_EQ(cudaStreamSynchronize(consumer), cudaSuccess);
    destroyStreamSafely(consumer);
}

TEST_F(TensorMultiStreamTest, BucketCrossStreamReuseIsOrdered) {
    GateStream producer;
    cudaStream_t consumer;
    ASSERT_EQ(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking), cudaSuccess);

    auto& pool = CudaMemoryPool::instance();

    void* block = pool.allocate(BUCKET_BYTES, producer.get());
    ASSERT_NE(block, nullptr);

    producer.close();
    ASSERT_EQ(cudaMemsetAsync(block, 0xAA, BUCKET_BYTES, producer.get()), cudaSuccess);
    pool.deallocate(block, producer.get());

    const uint64_t cross_before = SizeBucketedPool::instance().stats().cross_stream_reuse.load();
    void* reused = pool.allocate(BUCKET_BYTES, consumer);
    ASSERT_NE(reused, nullptr);

    if (reused == block) {
        EXPECT_GT(SizeBucketedPool::instance().stats().cross_stream_reuse.load(), cross_before);
        ASSERT_EQ(cudaMemsetAsync(reused, 0xBB, BUCKET_BYTES, consumer), cudaSuccess);

        producer.release();
        ASSERT_EQ(cudaStreamSynchronize(consumer), cudaSuccess);

        std::vector<unsigned char> host(BUCKET_BYTES);
        ASSERT_EQ(cudaMemcpy(host.data(), reused, BUCKET_BYTES, cudaMemcpyDeviceToHost), cudaSuccess);
        for (size_t i = 0; i < BUCKET_BYTES; i += 65536) {
            ASSERT_EQ(host[i], 0xBB) << "producer write ordered after reuse at offset " << i;
        }
    } else {
        producer.release();
    }

    pool.deallocate(reused, consumer);
    ASSERT_EQ(cudaStreamSynchronize(consumer), cudaSuccess);
    destroyStreamSafely(consumer);
}

TEST_F(TensorMultiStreamTest, RecordStreamBridgesReaderIntoFree) {
    GateStream reader;
    cudaStream_t owner;
    ASSERT_EQ(cudaStreamCreateWithFlags(&owner, cudaStreamNonBlocking), cudaSuccess);

    auto& pool = CudaMemoryPool::instance();

    void* block = pool.allocate(SLAB_BYTES, owner);
    ASSERT_NE(block, nullptr);
    ASSERT_EQ(cudaMemsetAsync(block, 0xCC, SLAB_BYTES, owner), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(owner), cudaSuccess);

    std::vector<unsigned char> readback(SLAB_BYTES, 0);
    void* staging = nullptr;
    ASSERT_EQ(cudaMalloc(&staging, SLAB_BYTES), cudaSuccess);

    reader.close();
    ASSERT_EQ(cudaMemcpyAsync(staging, block, SLAB_BYTES, cudaMemcpyDeviceToDevice, reader.get()),
              cudaSuccess);
    pool.record_stream(block, reader.get());
    pool.deallocate(block, owner);

    void* reused = pool.allocate(SLAB_BYTES, owner);
    if (reused == block) {
        ASSERT_EQ(cudaMemsetAsync(reused, 0xDD, SLAB_BYTES, owner), cudaSuccess);
    }

    reader.release();
    ASSERT_EQ(cudaStreamSynchronize(reader.get()), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(owner), cudaSuccess);

    ASSERT_EQ(cudaMemcpy(readback.data(), staging, SLAB_BYTES, cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t i = 0; i < SLAB_BYTES; i += 4096) {
        ASSERT_EQ(readback[i], 0xCC) << "reader saw overwrite from recycled block at offset " << i;
    }

    if (reused) {
        pool.deallocate(reused, owner);
    }
    cudaFree(staging);
    destroyStreamSafely(owner);
}

TEST_F(TensorMultiStreamTest, PinnedBlockReusedOnlyAfterAllStreamsDone) {
    GateStream h2d_stream;
    cudaStream_t d2h_stream;
    ASSERT_EQ(cudaStreamCreateWithFlags(&d2h_stream, cudaStreamNonBlocking), cudaSuccess);

    auto& pinned = PinnedMemoryAllocator::instance();
    constexpr size_t kBytes = 1 * 1024 * 1024;

    void* host_block = pinned.allocate(kBytes);
    ASSERT_NE(host_block, nullptr);
    std::memset(host_block, 0xEE, kBytes);

    void* device_buffer = nullptr;
    ASSERT_EQ(cudaMalloc(&device_buffer, kBytes), cudaSuccess);

    h2d_stream.close();
    ASSERT_EQ(cudaMemcpyAsync(device_buffer, host_block, kBytes, cudaMemcpyHostToDevice, h2d_stream.get()),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(host_block, device_buffer, kBytes, cudaMemcpyDeviceToHost, d2h_stream),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(d2h_stream), cudaSuccess);

    pinned.record_stream(host_block, h2d_stream.get());
    pinned.deallocate(host_block, d2h_stream);

    // The gated H2D copy is still pending, so the block must not be handed out.
    void* reused = pinned.allocate(kBytes);
    ASSERT_NE(reused, nullptr);
    EXPECT_NE(reused, host_block);

    h2d_stream.release();
    ASSERT_EQ(cudaStreamSynchronize(h2d_stream.get()), cudaSuccess);

    // All recorded uses complete — now the cached block is reusable.
    void* reused_after = pinned.allocate(kBytes);
    ASSERT_NE(reused_after, nullptr);
    EXPECT_EQ(reused_after, host_block);

    pinned.deallocate(reused, nullptr);
    pinned.deallocate(reused_after, nullptr);
    cudaFree(device_buffer);
    destroyStreamSafely(d2h_stream);
}

TEST_F(TensorMultiStreamTest, PinnedFallbackUsesMatchingDeallocator) {
    auto& pinned = PinnedMemoryAllocator::instance();
    pinned.empty_cache();
    pinned.reset_stats();
    pinned.set_force_fallback_for_testing(true);

    constexpr size_t kBytes = 64 * 1024;
    void* ptr = pinned.allocate(kBytes);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(pinned.get_stats().malloc_fallback_allocs, 1u);

    // Provenance belongs to the allocation, not the allocator's current mode.
    pinned.set_force_fallback_for_testing(false);
    pinned.deallocate(ptr);

    const auto stats = pinned.get_stats();
    EXPECT_EQ(stats.allocated_bytes, 0u);
    EXPECT_EQ(stats.cached_bytes, 0u);
    EXPECT_EQ(stats.malloc_fallback_allocs, 1u);
    EXPECT_EQ(stats.malloc_fallback_frees, 1u);
    EXPECT_EQ(stats.cuda_host_frees, 0u);
}

TEST_F(TensorMultiStreamTest, PinnedCacheEvictsLeastRecentlyUsedBlocksToBudget) {
    auto& pinned = PinnedMemoryAllocator::instance();
    pinned.empty_cache();
    pinned.reset_stats();

    constexpr size_t kBytes = 1 * 1024 * 1024;
    pinned.set_cache_limit_for_testing(2 * kBytes);

    void* first = pinned.allocate(kBytes);
    void* second = pinned.allocate(kBytes);
    void* third = pinned.allocate(kBytes);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    ASSERT_NE(third, nullptr);

    pinned.deallocate(first);
    pinned.deallocate(second);
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);

    void* touched = pinned.allocate(kBytes);
    ASSERT_EQ(touched, first);
    pinned.deallocate(touched);
    pinned.deallocate(third);
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);

    const auto after_eviction = pinned.get_stats();
    EXPECT_EQ(after_eviction.cached_bytes, 2 * kBytes);
    EXPECT_EQ(after_eviction.evicted_blocks, 1u);
    EXPECT_EQ(after_eviction.evicted_bytes, kBytes);
    EXPECT_EQ(after_eviction.cuda_host_frees, 1u);

    void* retained_a = pinned.allocate(kBytes);
    void* retained_b = pinned.allocate(kBytes);
    ASSERT_NE(retained_a, nullptr);
    ASSERT_NE(retained_b, nullptr);
    EXPECT_NE(retained_a, second);
    EXPECT_NE(retained_b, second);
    EXPECT_TRUE((retained_a == first && retained_b == third) ||
                (retained_a == third && retained_b == first));

    pinned.deallocate(retained_a);
    pinned.deallocate(retained_b);
}

TEST_F(TensorMultiStreamTest, PinnedCacheDoesNotRetainBlockLargerThanBudget) {
    auto& pinned = PinnedMemoryAllocator::instance();
    pinned.empty_cache();
    pinned.reset_stats();
    pinned.set_cache_limit_for_testing(512 * 1024);

    void* ptr = pinned.allocate(1 * 1024 * 1024);
    ASSERT_NE(ptr, nullptr);
    pinned.deallocate(ptr);

    const auto stats = pinned.get_stats();
    EXPECT_EQ(stats.allocated_bytes, 0u);
    EXPECT_EQ(stats.cached_bytes, 0u);
    EXPECT_EQ(stats.evicted_blocks, 1u);
    EXPECT_EQ(stats.evicted_bytes, 1 * 1024 * 1024u);
    EXPECT_EQ(stats.cuda_host_frees, 1u);
}

TEST_F(TensorMultiStreamTest, TrimMemoryPoolReleasesPinnedCache) {
    auto& pinned = PinnedMemoryAllocator::instance();
    pinned.empty_cache();
    pinned.reset_stats();

    void* ptr = pinned.allocate(128 * 1024);
    ASSERT_NE(ptr, nullptr);
    pinned.deallocate(ptr);
    ASSERT_GT(pinned.get_stats().cached_bytes, 0u);

    Tensor::trim_memory_pool();
    const auto stats = pinned.get_stats();
    EXPECT_EQ(stats.cached_bytes, 0u);
    EXPECT_EQ(stats.cuda_host_frees, 1u);
}

TEST_F(TensorMultiStreamTest, ExplicitH2DTransferGuardsDroppedPinnedSource) {
    auto& pinned = PinnedMemoryAllocator::instance();
    pinned.empty_cache();
    pinned.reset_stats();
    seedPinnedCache(256 * 1024 * sizeof(float));

    GateStream transfer;
    transfer.close();
    constexpr size_t kElements = 256 * 1024;

    Tensor gpu;
    void* source_ptr = nullptr;
    {
        auto source = Tensor::ones({kElements}, Device::CPU);
        source_ptr = source.data_ptr();
        gpu = source.to(Device::CUDA, transfer.get());
    }

    auto replacement = Tensor::empty({kElements}, Device::CPU);
    ASSERT_NE(replacement.data_ptr(), nullptr);
    EXPECT_NE(replacement.data_ptr(), source_ptr);
    EXPECT_GE(pinned.get_stats().cache_hits, 2u);

    transfer.release();
    ASSERT_EQ(cudaStreamSynchronize(transfer.get()), cudaSuccess);

    auto reused_after_completion = Tensor::empty({kElements}, Device::CPU);
    EXPECT_EQ(reused_after_completion.data_ptr(), source_ptr);
    EXPECT_GE(pinned.get_stats().cache_hits, 1u);

    const auto values = gpu.to(Device::CPU).to_vector();
    ASSERT_EQ(values.size(), kElements);
    EXPECT_FLOAT_EQ(values.front(), 1.0f);
    EXPECT_FLOAT_EQ(values.back(), 1.0f);
}

TEST_F(TensorMultiStreamTest, ExplicitD2HTransferGuardsDroppedPinnedDestination) {
    auto& pinned = PinnedMemoryAllocator::instance();
    pinned.empty_cache();
    pinned.reset_stats();
    seedPinnedCache(256 * 1024 * sizeof(float));

    GateStream transfer;
    constexpr size_t kElements = 256 * 1024;

    auto gpu = Tensor::full({kElements}, 2.0f, Device::CUDA);
    ASSERT_EQ(cudaStreamSynchronize(gpu.stream()), cudaSuccess);
    transfer.close();

    void* destination_ptr = nullptr;
    {
        auto destination = gpu.to(Device::CPU, transfer.get());
        destination_ptr = destination.data_ptr();
    }

    auto replacement = Tensor::empty({kElements}, Device::CPU);
    ASSERT_NE(replacement.data_ptr(), nullptr);
    EXPECT_NE(replacement.data_ptr(), destination_ptr);

    transfer.release();
    ASSERT_EQ(cudaStreamSynchronize(transfer.get()), cudaSuccess);

    auto reused_after_completion = Tensor::empty({kElements}, Device::CPU);
    EXPECT_EQ(reused_after_completion.data_ptr(), destination_ptr);
}

TEST_F(TensorMultiStreamTest, ExplicitH2DViewMoveAssignmentGuardsDroppedPinnedSource) {
    auto& pinned = PinnedMemoryAllocator::instance();
    pinned.empty_cache();
    pinned.reset_stats();
    seedPinnedCache(256 * 1024 * sizeof(float));

    GateStream transfer;
    constexpr size_t kElements = 256 * 1024;

    auto destination = Tensor::zeros({kElements + 2}, Device::CUDA);
    auto destination_view = destination.slice(0, 1, kElements + 1);
    ASSERT_TRUE(destination_view.is_view());
    destination_view.set_stream(transfer.get());
    transfer.close();

    void* source_ptr = nullptr;
    {
        auto source = Tensor::full({kElements}, 3.0f, Device::CPU);
        source_ptr = source.data_ptr();
        ASSERT_EQ(source.shape(), destination_view.shape());
        ASSERT_FLOAT_EQ(source.ptr<float>()[0], 3.0f);
        destination_view = std::move(source);
        EXPECT_EQ(destination_view.device(), Device::CUDA);
        EXPECT_TRUE(destination_view.is_view());
    }

    auto replacement = Tensor::empty({kElements}, Device::CPU);
    ASSERT_NE(replacement.data_ptr(), nullptr);
    EXPECT_NE(replacement.data_ptr(), source_ptr);

    transfer.release();
    ASSERT_EQ(cudaStreamSynchronize(transfer.get()), cudaSuccess);

    const auto values = destination.to(Device::CPU).to_vector();
    ASSERT_EQ(values.size(), kElements + 2);
    EXPECT_FLOAT_EQ(values.front(), 0.0f);
    EXPECT_FLOAT_EQ(values[1], 3.0f);
    EXPECT_FLOAT_EQ(values[kElements], 3.0f);
    EXPECT_FLOAT_EQ(values.back(), 0.0f);
}

TEST_F(TensorMultiStreamTest, ExplicitD2HViewAssignmentGuardsDroppedPinnedDestination) {
    auto& pinned = PinnedMemoryAllocator::instance();
    pinned.empty_cache();
    pinned.reset_stats();
    seedPinnedCache(256 * 1024 * sizeof(float));

    GateStream transfer;
    constexpr size_t kElements = 256 * 1024;

    auto source = Tensor::full({kElements}, 4.0f, Device::CUDA);
    ASSERT_EQ(cudaStreamSynchronize(source.stream()), cudaSuccess);

    void* destination_ptr = nullptr;
    {
        auto destination = Tensor::empty({kElements}, Device::CPU);
        destination_ptr = destination.data_ptr();
        auto destination_view = destination.slice(0, 0, kElements);
        destination_view.set_stream(transfer.get());
        transfer.close();
        destination_view = source;
    }

    auto replacement = Tensor::empty({kElements}, Device::CPU);
    ASSERT_NE(replacement.data_ptr(), nullptr);
    EXPECT_NE(replacement.data_ptr(), destination_ptr);

    transfer.release();
    ASSERT_EQ(cudaStreamSynchronize(transfer.get()), cudaSuccess);

    auto reused_after_completion = Tensor::empty({kElements}, Device::CPU);
    EXPECT_EQ(reused_after_completion.data_ptr(), destination_ptr);
}

TEST_F(TensorMultiStreamTest, ArenaCrossStreamFrameHandoff) {
    GateStream training;
    cudaStream_t render;
    ASSERT_EQ(cudaStreamCreateWithFlags(&render, cudaStreamNonBlocking), cudaSuccess);

    constexpr size_t kBytes = 1 * 1024 * 1024;
    auto& arena = GlobalArenaManager::instance().get_arena();

    // Training frame writes a pattern; the gate keeps the write pending past end_frame.
    const uint64_t training_frame = arena.begin_frame(training.get(), false);
    char* training_buf = arena.get_allocator(training_frame)(kBytes);
    ASSERT_NE(training_buf, nullptr);

    void* staging = nullptr;
    ASSERT_EQ(cudaMalloc(&staging, kBytes), cudaSuccess);

    training.close();
    ASSERT_EQ(cudaMemsetAsync(training_buf, 0xAA, kBytes, training.get()), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(staging, training_buf, kBytes, cudaMemcpyDeviceToDevice, training.get()),
              cudaSuccess);
    arena.end_frame(training_frame, training.get(), false);

    // Render frame reuses the same arena range; its writes must be GPU-ordered
    // after the still-pending training frame via the chained completion event.
    const uint64_t render_frame = arena.begin_frame(render, true);
    char* render_buf = arena.get_allocator(render_frame)(kBytes);
    ASSERT_NE(render_buf, nullptr);
    EXPECT_EQ(static_cast<void*>(render_buf), static_cast<void*>(training_buf));
    ASSERT_EQ(cudaMemsetAsync(render_buf, 0xBB, kBytes, render), cudaSuccess);
    arena.end_frame(render_frame, render, true);

    training.release();
    ASSERT_EQ(cudaStreamSynchronize(training.get()), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(render), cudaSuccess);

    std::vector<unsigned char> host(kBytes);
    ASSERT_EQ(cudaMemcpy(host.data(), staging, kBytes, cudaMemcpyDeviceToHost), cudaSuccess);
    for (size_t i = 0; i < kBytes; i += 65536) {
        ASSERT_EQ(host[i], 0xAA) << "render frame overwrote memory before training frame finished, offset " << i;
    }

    cudaFree(staging);
    destroyStreamSafely(render);
}

// Mirrors the trainer↔viewer protocol: writer records params_ready after each
// in-place rewrite and waits the reader's done-event before the next one; the
// reader brackets its snapshot with the matching wait/record. Every snapshot
// must be one consistent iteration — without the handshake, the overlapping
// memset tears it.
TEST_F(TensorMultiStreamTest, TrainerViewerHandshakeYieldsConsistentSnapshots) {
    constexpr size_t kBytes = 32 * 1024 * 1024;
    constexpr int kMaxIterations = 200000;
    constexpr int kTargetSnapshots = 50;

    cudaStream_t train_stream, render_stream;
    ASSERT_EQ(cudaStreamCreateWithFlags(&train_stream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&render_stream, cudaStreamNonBlocking), cudaSuccess);

    void* params = nullptr;
    void* staging = nullptr;
    ASSERT_EQ(cudaMalloc(&params, kBytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&staging, kBytes), cudaSuccess);
    ASSERT_EQ(cudaMemset(params, 0, kBytes), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    cudaEvent_t params_ready, reader_done;
    ASSERT_EQ(cudaEventCreateWithFlags(&params_ready, cudaEventDisableTiming), cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&reader_done, cudaEventDisableTiming), cudaSuccess);

    std::mutex sync_mutex;
    bool params_recorded = false;
    bool reader_pending = false;
    std::atomic<bool> writer_done{false};
    std::atomic<bool> stop_writer{false};

    std::thread writer([&] {
        cudaSetDevice(0);
        for (int i = 1; i <= kMaxIterations && !stop_writer.load(std::memory_order_acquire); ++i) {
            {
                std::lock_guard<std::mutex> lock(sync_mutex);
                if (reader_pending) {
                    cudaStreamWaitEvent(train_stream, reader_done, 0);
                    reader_pending = false;
                }
            }
            cudaMemsetAsync(params, i & 0xFF, kBytes, train_stream);
            {
                std::lock_guard<std::mutex> lock(sync_mutex);
                cudaEventRecord(params_ready, train_stream);
                params_recorded = true;
            }
        }
        cudaStreamSynchronize(train_stream);
        writer_done.store(true, std::memory_order_release);
    });

    std::vector<unsigned char> host(kBytes / 65536);
    int snapshots = 0;
    int torn_snapshots = 0;
    while (!writer_done.load(std::memory_order_acquire)) {
        {
            std::lock_guard<std::mutex> lock(sync_mutex);
            if (params_recorded) {
                cudaStreamWaitEvent(render_stream, params_ready, 0);
            }
        }
        ASSERT_EQ(cudaMemcpyAsync(staging, params, kBytes, cudaMemcpyDeviceToDevice, render_stream),
                  cudaSuccess);
        {
            std::lock_guard<std::mutex> lock(sync_mutex);
            cudaEventRecord(reader_done, render_stream);
            reader_pending = true;
        }
        ASSERT_EQ(cudaStreamSynchronize(render_stream), cudaSuccess);

        for (size_t i = 0; i < host.size(); ++i) {
            ASSERT_EQ(cudaMemcpy(&host[i], static_cast<unsigned char*>(staging) + i * 65536, 1,
                                 cudaMemcpyDeviceToHost),
                      cudaSuccess);
        }
        ++snapshots;
        for (size_t i = 1; i < host.size(); ++i) {
            if (host[i] != host[0]) {
                ++torn_snapshots;
                break;
            }
        }
        if (snapshots >= kTargetSnapshots) {
            stop_writer.store(true, std::memory_order_release);
        }
    }
    writer.join();

    EXPECT_EQ(torn_snapshots, 0) << "of " << snapshots << " snapshots";
    EXPECT_GE(snapshots, kTargetSnapshots);

    cudaFree(params);
    cudaFree(staging);
    cudaEventDestroy(params_ready);
    cudaEventDestroy(reader_done);
    destroyStreamSafely(train_stream);
    destroyStreamSafely(render_stream);
}

TEST_F(TensorMultiStreamTest, MultiThreadMultiStreamHammer) {
    constexpr int kThreads = 4;
    constexpr int kIterations = 12000;
    static constexpr size_t kSizes[] = {1024, 96 * 1024, 512 * 1024, 3 * 1024 * 1024};

    std::vector<std::thread> threads;
    std::atomic<int> failures{0};

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([t, &failures] {
            cudaSetDevice(0);
            cudaStream_t stream;
            if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
                failures.fetch_add(1);
                return;
            }
            auto& pool = CudaMemoryPool::instance();
            const unsigned char pattern = static_cast<unsigned char>(0x10 + t);

            for (int i = 0; i < kIterations; ++i) {
                const size_t bytes = kSizes[(t + i) % 4];
                void* ptr = pool.allocate(bytes, stream);
                if (!ptr) {
                    failures.fetch_add(1);
                    break;
                }
                if (cudaMemsetAsync(ptr, pattern, bytes, stream) != cudaSuccess) {
                    failures.fetch_add(1);
                    pool.deallocate(ptr, stream);
                    break;
                }
                if (i % 512 == 0) {
                    unsigned char probe = 0;
                    if (cudaMemcpyAsync(&probe, ptr, 1, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
                        cudaStreamSynchronize(stream) != cudaSuccess || probe != pattern) {
                        failures.fetch_add(1);
                        pool.deallocate(ptr, stream);
                        break;
                    }
                }
                pool.deallocate(ptr, stream);
            }

            cudaStreamSynchronize(stream);
            destroyStreamSafely(stream);
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(failures.load(), 0);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

// Mirrors PyTensor::dlpack(stream=consumer): the export records the producer's
// home-stream work and makes the consumer stream wait on it via bridgeStreams.
// A gated producer write must therefore be visible to the consumer's readback.
TEST_F(TensorMultiStreamTest, DLPackExportBridgesHomeStreamOntoConsumer) {
    GateStream producer;
    cudaStream_t consumer;
    ASSERT_EQ(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking), cudaSuccess);

    auto& pool = CudaMemoryPool::instance();
    void* buffer = pool.allocate(SLAB_BYTES, producer.get());
    ASSERT_NE(buffer, nullptr);

    // Pinned so the consumer readback stays async — a pageable D2H would block
    // the host inside the copy, deadlocking against the still-closed gate.
    unsigned char* host = nullptr;
    ASSERT_EQ(cudaMallocHost(&host, SLAB_BYTES), cudaSuccess);
    std::memset(host, 0, SLAB_BYTES);

    producer.close();
    ASSERT_EQ(cudaMemsetAsync(buffer, 0xCD, SLAB_BYTES, producer.get()), cudaSuccess);

    bridgeStreams(producer.get(), consumer);
    ASSERT_EQ(cudaMemcpyAsync(host, buffer, SLAB_BYTES, cudaMemcpyDeviceToHost, consumer), cudaSuccess);

    producer.release();
    ASSERT_EQ(cudaStreamSynchronize(consumer), cudaSuccess);

    for (size_t i = 0; i < SLAB_BYTES; i += 4096) {
        ASSERT_EQ(host[i], 0xCD) << "consumer readback raced the producer write at offset " << i;
    }

    cudaFreeHost(host);
    pool.deallocate(buffer, producer.get());
    ASSERT_EQ(cudaStreamSynchronize(producer.get()), cudaSuccess);
    destroyStreamSafely(consumer);
}
