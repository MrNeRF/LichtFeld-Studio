/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Reproduces the trainer↔metrics arena/lock contention that the bounded
// begin_frame (ScopedBeginFrameTimeout) must keep deadlock-free:
//   trainer:  holds the arena frame, then wants the exclusive render lock
//   metrics:  holds the shared render lock, then wants the arena frame
// With a wait-forever metrics begin_frame this cycles; bounded, the metrics
// acquisition bails and the cycle breaks.

#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include <future>
#include <gtest/gtest.h>
#include <shared_mutex>
#include <thread>

#include "core/cuda/memory_arena.hpp"

using lfs::core::RasterizerMemoryArena;

namespace {
    // Runs fn on a thread and fails (rather than hanging the suite) if it does
    // not finish within the budget — a deadlock manifests as the timeout.
    template <typename Fn>
    bool completes_within(std::chrono::milliseconds budget, Fn&& fn) {
        std::packaged_task<void()> task(std::forward<Fn>(fn));
        auto future = task.get_future();
        std::thread runner(std::move(task));
        const bool ok = future.wait_for(budget) == std::future_status::ready;
        if (ok) {
            runner.join();
        } else {
            runner.detach(); // leak the wedged thread; the test already failed
        }
        return ok;
    }
} // namespace

class ArenaMetricsContentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    }
};

TEST_F(ArenaMetricsContentionTest, BoundedBeginFrameBailsWhileFrameHeld) {
    RasterizerMemoryArena arena;
    const uint64_t held = arena.begin_frame(nullptr, /*from_rendering=*/false);

    // A second thread with a bounded timeout must throw (not hang) because the
    // single arena frame is held.
    const bool finished = completes_within(std::chrono::milliseconds(2000), [&] {
        const RasterizerMemoryArena::ScopedBeginFrameTimeout timeout(50);
        bool threw = false;
        try {
            arena.begin_frame(nullptr, false);
        } catch (const std::exception&) {
            threw = true;
        }
        EXPECT_TRUE(threw) << "bounded begin_frame should bail while the frame is held";
    });
    EXPECT_TRUE(finished) << "bounded begin_frame hung instead of timing out";

    arena.end_frame(held, nullptr, false);
}

TEST_F(ArenaMetricsContentionTest, TrainerMetricsOppositeOrderNoDeadlock) {
    RasterizerMemoryArena arena;
    std::shared_mutex render_mutex;
    std::atomic<bool> trainer_done{false};
    std::atomic<int> metrics_attempts{0};
    std::atomic<int> metrics_acquired{0};

    const bool finished = completes_within(std::chrono::milliseconds(20000), [&] {
        // Trainer: frame-then-lock, exclusive on "refine" iterations.
        std::thread trainer([&] {
            cudaSetDevice(0);
            for (int i = 1; i <= 400; ++i) {
                const uint64_t frame = arena.begin_frame(nullptr, false);
                std::optional<std::unique_lock<std::shared_mutex>> excl;
                if (i % 5 == 0) {
                    excl.emplace(render_mutex); // hold lock while holding frame
                }
                arena.end_frame(frame, nullptr, false);
            }
            trainer_done.store(true, std::memory_order_release);
        });

        // Metrics: lock-then-(bounded)-frame, the opposite order.
        std::thread metrics([&] {
            cudaSetDevice(0);
            while (!trainer_done.load(std::memory_order_acquire)) {
                std::shared_lock<std::shared_mutex> shared(render_mutex);
                const RasterizerMemoryArena::ScopedBeginFrameTimeout timeout(20);
                metrics_attempts.fetch_add(1, std::memory_order_relaxed);
                try {
                    const uint64_t frame = arena.begin_frame(nullptr, false);
                    metrics_acquired.fetch_add(1, std::memory_order_relaxed);
                    arena.end_frame(frame, nullptr, false);
                } catch (const std::exception&) {
                    // Bounded bail under contention — expected, no deadlock.
                }
            }
        });

        trainer.join();
        metrics.join();
    });

    EXPECT_TRUE(finished) << "trainer/metrics contention deadlocked";
    EXPECT_GT(metrics_attempts.load(), 0);
    // It should manage at least some real acquisitions across 400 iterations.
    EXPECT_GT(metrics_acquired.load(), 0);
}

TEST_F(ArenaMetricsContentionTest, ExternalGrowWaitsForPendingRender) {
    constexpr size_t MiB = 1024 * 1024;
    constexpr size_t physical_bytes = 8 * MiB;
    constexpr size_t initial_committed_bytes = 1 * MiB;

    void* device_ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&device_ptr, physical_bytes), cudaSuccess);
    auto owner = std::shared_ptr<void>(device_ptr, [](void* ptr) {
        EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    });

    RasterizerMemoryArena::Config config;
    config.max_physical = physical_bytes;
    config.enable_vmm = false;
    RasterizerMemoryArena arena(config);

    std::atomic<bool> render_pending{false};
    std::atomic<bool> callback_while_pending{false};
    std::atomic<int> grow_calls{0};
    RasterizerMemoryArena::ExternalBacking backing{
        .device_ptr = device_ptr,
        .size = initial_committed_bytes,
        .device = 0,
        .owner = owner,
        .label = "test.external",
        .grow = [&](const size_t need) -> size_t {
            grow_calls.fetch_add(1, std::memory_order_relaxed);
            if (render_pending.load(std::memory_order_acquire)) {
                callback_while_pending.store(true, std::memory_order_release);
            }
            return need <= physical_bytes ? physical_bytes : 0;
        },
    };
    ASSERT_TRUE(arena.install_external_backing(std::move(backing)));

    const uint64_t frame = arena.begin_frame(nullptr, /*from_rendering=*/false);
    auto allocate = arena.get_allocator(frame);

    // Reproduce the queued-render window while the trainer owns the only active
    // frame. The render's bounded acquisition eventually clears the pending bit;
    // external growth must retry until then and only invoke the mapping callback
    // after the frame gate is safe.
    render_pending.store(true, std::memory_order_release);
    arena.set_rendering_active(true);
    std::atomic<bool> allocation_started{false};
    std::thread clear_pending([&] {
        while (!allocation_started.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        render_pending.store(false, std::memory_order_release);
        arena.set_rendering_active(false);
    });

    allocation_started.store(true, std::memory_order_release);
    char* const allocation = allocate(2 * MiB);
    clear_pending.join();

    EXPECT_NE(allocation, nullptr);
    EXPECT_EQ(allocation, device_ptr);
    EXPECT_FALSE(callback_while_pending.load(std::memory_order_acquire));
    EXPECT_EQ(grow_calls.load(std::memory_order_relaxed), 1);
    EXPECT_EQ(arena.get_statistics().capacity, physical_bytes);

    arena.end_frame(frame, nullptr, /*from_rendering=*/false);
}

TEST_F(ArenaMetricsContentionTest, FullResetDecommitsVmmHighWater) {
    constexpr size_t MiB = 1024 * 1024;
    RasterizerMemoryArena::Config config;
    config.virtual_size = 1ULL * 1024 * MiB;
    config.max_physical = 512 * MiB;
    config.granularity = 2 * MiB;
    RasterizerMemoryArena arena(config);

    const uint64_t frame = arena.begin_frame(nullptr, false);
    auto allocate = arena.get_allocator(frame);
    ASSERT_NE(allocate(192 * MiB), nullptr);
    arena.end_frame(frame, nullptr, false);

    const auto grown = arena.get_statistics();
    ASSERT_EQ(grown.capacity, 192 * MiB);
    const auto grown_info = arena.get_memory_info();
    EXPECT_EQ(grown_info.required_bytes, grown_info.peak_usage);
    EXPECT_EQ(grown_info.required_bytes, grown_info.arena_capacity);

    arena.full_reset();

    const auto reset = arena.get_statistics();
    EXPECT_EQ(reset.current_usage, 0u);
    EXPECT_EQ(reset.capacity, 0u);
    const auto reset_info = arena.get_memory_info();
    EXPECT_EQ(reset_info.peak_usage, 0u);
    EXPECT_EQ(reset_info.required_bytes, 0u);
    EXPECT_EQ(reset_info.arena_capacity, 0u);
}

TEST_F(ArenaMetricsContentionTest, FallbackGrowthUsesExactMeasuredRequirement) {
    constexpr size_t MiB = 1024 * 1024;
    RasterizerMemoryArena::Config config;
    config.max_physical = 512 * MiB;
    config.enable_vmm = false;
    RasterizerMemoryArena arena(config);

    const uint64_t frame = arena.begin_frame(nullptr, false);
    auto allocate = arena.get_allocator(frame);
    ASSERT_NE(allocate(128 * MiB), nullptr);

    const auto grown = arena.get_statistics();
    EXPECT_EQ(grown.capacity, 128 * MiB);
    const auto grown_info = arena.get_memory_info();
    EXPECT_EQ(grown_info.required_bytes, grown_info.peak_usage);
    EXPECT_EQ(grown_info.required_bytes, grown_info.arena_capacity);
    arena.end_frame(frame, nullptr, false);
}

TEST_F(ArenaMetricsContentionTest, RetainedFallbackPublishesLogicalRequiredAndClassCSlack) {
    constexpr size_t MiB = 1024 * 1024;
    RasterizerMemoryArena::Config config;
    config.max_physical = 512 * MiB;
    config.enable_vmm = false;
    RasterizerMemoryArena arena(config);

    const uint64_t large_frame = arena.begin_frame(nullptr, false);
    auto allocate_large = arena.get_allocator(large_frame);
    ASSERT_NE(allocate_large(64 * MiB), nullptr);
    arena.end_frame(large_frame, nullptr, false);

    const auto large = arena.get_memory_info();
    ASSERT_EQ(large.required_bytes, 64 * MiB);
    ASSERT_EQ(large.arena_capacity, 64 * MiB);

    const uint64_t small_frame = arena.begin_frame(nullptr, false);
    auto allocate_small = arena.get_allocator(small_frame);
    ASSERT_NE(allocate_small(10 * MiB), nullptr);
    arena.end_frame(small_frame, nullptr, false);
    ASSERT_TRUE(arena.shrink_to_current_at_boundary());

    const auto retained = arena.get_memory_info();
    EXPECT_EQ(retained.current_usage, 10 * MiB);
    EXPECT_EQ(retained.peak_usage, 10 * MiB);
    EXPECT_EQ(retained.required_bytes, 10 * MiB);
    EXPECT_EQ(retained.arena_capacity, 64 * MiB);
    EXPECT_EQ(retained.arena_capacity - retained.required_bytes, 54 * MiB);
}
