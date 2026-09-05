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
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

#include "core/cuda/memory_arena.hpp"
#include "core/logger.hpp"

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
    std::atomic<bool> trainer_has_frame{false};
    std::atomic<bool> metrics_has_render_lock{false};
    std::atomic<bool> metrics_timed_out{false};
    std::atomic<bool> trainer_acquired_render_lock{false};

    const bool finished = completes_within(std::chrono::milliseconds(2000), [&] {
        // Reproduce one deterministic lock inversion:
        // trainer: arena frame -> exclusive render lock
        // metrics: shared render lock -> bounded arena frame
        std::thread trainer([&] {
            cudaSetDevice(0);
            const uint64_t frame = arena.begin_frame(nullptr, false);
            trainer_has_frame.store(true, std::memory_order_release);
            while (!metrics_has_render_lock.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            {
                std::unique_lock<std::shared_mutex> exclusive(render_mutex);
                trainer_acquired_render_lock.store(true, std::memory_order_release);
            }
            arena.end_frame(frame, nullptr, false);
        });

        std::thread metrics([&] {
            cudaSetDevice(0);
            while (!trainer_has_frame.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            std::shared_lock<std::shared_mutex> shared(render_mutex);
            metrics_has_render_lock.store(true, std::memory_order_release);
            const RasterizerMemoryArena::ScopedBeginFrameTimeout timeout(20);
            try {
                const uint64_t frame = arena.begin_frame(nullptr, false);
                arena.end_frame(frame, nullptr, false);
            } catch (const std::exception&) {
                metrics_timed_out.store(true, std::memory_order_release);
            }
        });

        trainer.join();
        metrics.join();
    });

    EXPECT_TRUE(finished) << "trainer/metrics contention deadlocked";
    EXPECT_TRUE(metrics_timed_out.load());
    EXPECT_TRUE(trainer_acquired_render_lock.load());
    const auto next = arena.try_begin_frame(nullptr, false);
    ASSERT_TRUE(next.has_value());
    arena.end_frame(*next, nullptr, false);
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

TEST_F(ArenaMetricsContentionTest, ExternalGrowFailureRetainsCommittedCapacity) {
    constexpr size_t MiB = 1024 * 1024;
    constexpr size_t physical_bytes = 4 * MiB;
    constexpr size_t initial_committed_bytes = 1 * MiB;

    void* device_ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&device_ptr, physical_bytes), cudaSuccess);
    auto owner = std::shared_ptr<void>(device_ptr, [](void* ptr) {
        EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    });

    RasterizerMemoryArena::Config config;
    config.max_physical = physical_bytes;
    config.enable_vmm = false;
    config.granularity = MiB;
    RasterizerMemoryArena arena(config);
    std::atomic<int> grow_calls{0};
    std::atomic<size_t> requested_bytes{0};
    RasterizerMemoryArena::ExternalBacking backing{
        .device_ptr = device_ptr,
        .size = initial_committed_bytes,
        .device = 0,
        .owner = owner,
        .label = "test.external.failure",
        .grow = [&](const size_t requested) {
            grow_calls.fetch_add(1, std::memory_order_relaxed);
            requested_bytes.store(requested, std::memory_order_release);
            return size_t{0};
        },
    };
    ASSERT_TRUE(arena.install_external_backing(std::move(backing)));

    std::mutex log_mutex;
    std::vector<std::string> error_logs;
    const auto log_handler = lfs::core::Logger::get().add_log_handler(
        [&](const lfs::core::LogLevel level, const lfs::core::SourceSite&, const std::string_view message) {
            if (level == lfs::core::LogLevel::Error) {
                std::scoped_lock lock(log_mutex);
                error_logs.emplace_back(message);
            }
        });

    const uint64_t frame = arena.begin_frame(nullptr, false);
    auto allocate = arena.get_allocator(frame, "arena.failure.identity");
    EXPECT_EQ(allocate(2 * MiB), nullptr);
    EXPECT_EQ(grow_calls.load(std::memory_order_relaxed), 1);
    EXPECT_EQ(requested_bytes.load(std::memory_order_acquire), 2 * MiB);
    EXPECT_EQ(arena.get_statistics().capacity, initial_committed_bytes);
    arena.end_frame(frame, nullptr, false);
    lfs::core::Logger::get().remove_log_handler(log_handler);

    std::string joined_logs;
    for (const auto& message : error_logs) {
        joined_logs += message;
        joined_logs += '\n';
    }
    EXPECT_NE(joined_logs.find("test.external.failure"), std::string::npos);
    EXPECT_NE(joined_logs.find("arena.failure.identity"), std::string::npos);
    EXPECT_NE(joined_logs.find("capacity=1 MiB"), std::string::npos);
    EXPECT_NE(joined_logs.find("Arena VRAM failure snapshot"), std::string::npos);
    EXPECT_NE(joined_logs.find("cuda_free="), std::string::npos);
}

TEST_F(ArenaMetricsContentionTest, ExternalGrowCommitsExactAlignedNeed) {
    constexpr size_t MiB = 1024 * 1024;
    constexpr size_t initial_committed_bytes = 1 * MiB;
    constexpr size_t aligned_need = 2 * MiB;

    void* device_ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&device_ptr, 4 * MiB), cudaSuccess);
    auto owner = std::shared_ptr<void>(device_ptr, [](void* ptr) {
        EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    });

    RasterizerMemoryArena::Config config;
    config.max_physical = 4 * MiB;
    config.enable_vmm = false;
    config.granularity = MiB;
    RasterizerMemoryArena arena(config);
    RasterizerMemoryArena::ExternalBacking backing{
        .device_ptr = device_ptr,
        .size = initial_committed_bytes,
        .device = 0,
        .owner = owner,
        .label = "test.external.exact_policy",
    };
    ASSERT_TRUE(arena.install_external_backing(std::move(backing)));

    std::vector<size_t> requests;
    const bool grew = arena.grow_external_backing(
        device_ptr, aligned_need,
        [&](const size_t requested) {
            requests.push_back(requested);
            return requested <= aligned_need;
        });
    ASSERT_TRUE(grew);
    ASSERT_EQ(requests.size(), 1u);
    EXPECT_EQ(requests[0], aligned_need);
    EXPECT_EQ(arena.get_statistics().capacity, aligned_need);
}

TEST_F(ArenaMetricsContentionTest, DetachedViewerBackingCanBeReinstalledAndGrown) {
    constexpr size_t MiB = 1024 * 1024;
    void* device_ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&device_ptr, 2 * MiB), cudaSuccess);
    auto owner = std::shared_ptr<void>(device_ptr, [](void* ptr) {
        EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    });
    RasterizerMemoryArena arena;
    const RasterizerMemoryArena::ExternalBacking backing{
        .device_ptr = device_ptr,
        .size = MiB,
        .device = 0,
        .owner = owner,
        .label = "test.external.pause_resume",
    };
    ASSERT_TRUE(arena.install_external_backing(backing));
    ASSERT_TRUE(arena.using_external_backing(device_ptr));
    EXPECT_FALSE(arena.using_external_backing(static_cast<char*>(device_ptr) + 1));

    // Trainer B3 detaches independently of the viewer's retained block/import.
    arena.clear_external_backing();
    EXPECT_FALSE(arena.using_external_backing(device_ptr));
    bool committed = false;
    const auto commit = [&](size_t) {
        committed = true;
        return true;
    };
    using GrowFailure = RasterizerMemoryArena::ExternalGrowFailure;
    GrowFailure failure = GrowFailure::None;
    EXPECT_FALSE(arena.grow_external_backing(device_ptr, 2 * MiB, commit, 0, &failure));
    EXPECT_EQ(failure, GrowFailure::BackingMissing);
    EXPECT_FALSE(committed);

    // Reinstallation must defer while resumed training owns an arena frame.
    const auto held = arena.begin_frame(nullptr, false);
    EXPECT_FALSE(arena.try_install_external_backing(backing));
    arena.end_frame(held, nullptr, false);
    ASSERT_TRUE(arena.try_install_external_backing(backing));
    EXPECT_TRUE(arena.using_external_backing(device_ptr));
    EXPECT_FALSE(arena.grow_external_backing(device_ptr, 2 * MiB, [](size_t) { return false; }, 0, &failure));
    EXPECT_EQ(failure, GrowFailure::CommitFailure);
    EXPECT_EQ(arena.get_statistics().capacity, MiB);
    ASSERT_TRUE(arena.grow_external_backing(device_ptr, 2 * MiB, commit, 0, &failure));
    EXPECT_EQ(failure, GrowFailure::None);
    EXPECT_TRUE(committed);
    EXPECT_EQ(arena.get_statistics().capacity, 2 * MiB);

    arena.clear_external_backing();
    EXPECT_FALSE(arena.using_external_backing(device_ptr));
}

TEST_F(ArenaMetricsContentionTest, ViewerGrowTimeoutReleasesReservation) {
    RasterizerMemoryArena arena;
    const auto held = arena.begin_frame(nullptr, false);
    bool committed = false;
    using GrowFailure = RasterizerMemoryArena::ExternalGrowFailure;
    GrowFailure failure = GrowFailure::None;
    const auto started = std::chrono::steady_clock::now();
    EXPECT_FALSE(arena.grow_external_backing(nullptr, 1, [&](size_t) {
        committed = true;
        return true; }, 15, &failure));
    EXPECT_EQ(failure, GrowFailure::Busy);
    EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::seconds(2));
    EXPECT_FALSE(committed);
    EXPECT_FALSE(arena.is_rendering_active());
    arena.end_frame(held, nullptr, false);
    // A timed-out viewer must not leave training gated behind a stale request.
    const auto next = arena.try_begin_frame(nullptr, false);
    ASSERT_TRUE(next.has_value());
    arena.end_frame(*next, nullptr, false);
}

TEST_F(ArenaMetricsContentionTest, ViewerGrowPreservesExistingRenderReservation) {
    RasterizerMemoryArena arena;
    arena.set_rendering_active(true);
    bool committed = false;
    EXPECT_FALSE(arena.grow_external_backing(nullptr, 1, [&](size_t) {
        committed = true;
        return true; }, 15));
    EXPECT_FALSE(committed);
    EXPECT_TRUE(arena.is_rendering_active());
    arena.set_rendering_active(false);
    const auto next = arena.try_begin_frame(nullptr, false);
    ASSERT_TRUE(next.has_value());
    arena.end_frame(*next, nullptr, false);
}

TEST_F(ArenaMetricsContentionTest, ViewerGrowReservesIdleWindowAfterTrainingFrame) {
    constexpr size_t MiB = 1024 * 1024;
    void* device_ptr = nullptr;
    ASSERT_EQ(cudaMalloc(&device_ptr, 2 * MiB), cudaSuccess);
    auto owner = std::shared_ptr<void>(device_ptr, [](void* ptr) {
        EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    });
    RasterizerMemoryArena::Config config;
    config.max_physical = 2 * MiB;
    config.enable_vmm = false;
    config.granularity = MiB;
    RasterizerMemoryArena arena(config);
    ASSERT_TRUE(arena.install_external_backing({
        .device_ptr = device_ptr,
        .size = MiB,
        .device = 0,
        .owner = owner,
        .label = "test.external.viewer_growth",
    }));
    const auto held = arena.begin_frame(nullptr, false);
    std::atomic<bool> finished{false};
    bool grew = false;
    size_t committed = 0;
    std::thread viewer([&] {
        EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
        grew = arena.grow_external_backing(device_ptr, 2 * MiB, [&](size_t requested) {
            committed = requested;
            return true; }, 2000);
        finished.store(true, std::memory_order_release);
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    bool reserved = false;
    while (!(reserved = arena.is_rendering_active()) &&
           !finished.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    // Release only after growth has registered interest, avoiding a sleep-based
    // race that could let the old try-only implementation pass accidentally.
    arena.end_frame(held, nullptr, false);
    viewer.join();
    EXPECT_TRUE(reserved);
    EXPECT_TRUE(grew);
    EXPECT_EQ(committed, 2 * MiB);
    EXPECT_EQ(arena.get_statistics().capacity, 2 * MiB);
    EXPECT_FALSE(arena.is_rendering_active());
    const auto next = arena.try_begin_frame(nullptr, false);
    ASSERT_TRUE(next.has_value());
    arena.end_frame(*next, nullptr, false);
}

TEST_F(ArenaMetricsContentionTest, FullResetRetainsExternalBackingAndCapacity) {
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
    config.granularity = MiB;
    RasterizerMemoryArena arena(config);
    RasterizerMemoryArena::ExternalBacking backing{
        .device_ptr = device_ptr,
        .size = initial_committed_bytes,
        .device = 0,
        .owner = owner,
        .label = "test.external.reset",
        .grow = [physical_bytes](const size_t requested) {
            return requested <= physical_bytes ? physical_bytes : size_t{0};
        },
    };
    ASSERT_TRUE(arena.install_external_backing(std::move(backing)));

    const uint64_t frame = arena.begin_frame(nullptr, false);
    auto allocate = arena.get_allocator(frame);
    ASSERT_NE(allocate(2 * MiB), nullptr);
    arena.end_frame(frame, nullptr, false);
    ASSERT_EQ(arena.get_statistics().capacity, physical_bytes);

    arena.full_reset();
    EXPECT_TRUE(arena.using_external_backing());
    EXPECT_EQ(arena.get_statistics().capacity, physical_bytes);

    const uint64_t post_reset_frame = arena.begin_frame(nullptr, false);
    auto post_reset_allocate = arena.get_allocator(post_reset_frame);
    EXPECT_NE(post_reset_allocate(3 * MiB), nullptr);
    arena.end_frame(post_reset_frame, nullptr, false);
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
