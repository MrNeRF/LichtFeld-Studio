/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// #1566 — host-only GpuResourcePool lifecycle with a fake payload
// (payload-aware producer pred, evict, trim, force, timeline bookkeeping).

#include "rendering/cuda_vulkan_interop.hpp"
#include "rendering/output_image_pool.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace {

    using lfs::vis::GpuResourcePool;
    using lfs::vis::GpuResourcePoolKey;

    struct FakeUnit {
        std::uint64_t id = 0;
        std::uint64_t timeline_value = 0;
        std::size_t byte_size = 0;
        bool destroyed = false;
    };

    using Pool = GpuResourcePool<FakeUnit>;

    GpuResourcePoolKey makeKey(const std::uint32_t w = 128,
                               const std::uint32_t h = 64,
                               const VkFormat format = VK_FORMAT_R8G8B8A8_UNORM) {
        return GpuResourcePoolKey{
            .format = format,
            .extent = VkExtent2D{w, h},
            .usage = VK_IMAGE_USAGE_STORAGE_BIT,
            .external = true,
        };
    }

    FakeUnit makeUnit(const std::uint64_t id,
                      const std::uint64_t timeline = 0,
                      const std::size_t bytes = 1024) {
        return FakeUnit{.id = id, .timeline_value = timeline, .byte_size = bytes};
    }

    struct DestroyCapture {
        std::vector<std::uint64_t> destroyed_ids;

        Pool::DestroyFn fn() {
            return [this](FakeUnit& unit) {
                destroyed_ids.push_back(unit.id);
                unit.destroyed = true;
            };
        }
    };

    Pool makePool() {
        return Pool([](const FakeUnit& u) { return u.byte_size; });
    }

} // namespace

TEST(GpuResourcePool, AcquireMissOnEmpty) {
    auto pool = makePool();
    EXPECT_FALSE(pool.acquire(makeKey()).has_value());
    EXPECT_EQ(pool.liveCount(), 0u);
}

TEST(GpuResourcePool, RegisterReleaseDrainReusePreservesPayload) {
    auto pool = makePool();
    DestroyCapture cap;
    const auto key = makeKey();

    auto live = pool.registerCreated(key, makeUnit(42, /*timeline=*/7));
    ASSERT_NE(live.payload, nullptr);
    EXPECT_EQ(live.payload->id, 42u);
    EXPECT_EQ(live.payload->timeline_value, 7u);
    EXPECT_EQ(pool.liveCount(), 1u);

    // Producer watermark = unit timeline at release.
    pool.release(live.acquisition_serial, live.payload->timeline_value, /*consumer=*/1);
    EXPECT_EQ(pool.retiredCount(), 1u);

    // Payload-aware producer: only done when unit.timeline_value >= watermark
    // (mirrors vkGetSemaphoreCounterValue semantics used by interop drain).
    auto producer_done = [](const FakeUnit& unit, const std::uint64_t value) {
        return unit.timeline_value >= value;
    };
    auto consumer_done = [](const std::uint64_t serial) { return serial <= 1; };

    pool.drain(false, producer_done, consumer_done, cap.fn());
    EXPECT_EQ(pool.freeCount(), 1u);
    EXPECT_TRUE(cap.destroyed_ids.empty());

    auto hit = pool.acquire(key);
    ASSERT_TRUE(hit.has_value());
    ASSERT_NE(hit->payload, nullptr);
    EXPECT_EQ(hit->payload->id, 42u);
    // Timeline monotonicity bookkeeping: reuse continues from stored value.
    EXPECT_EQ(hit->payload->timeline_value, 7u);
    EXPECT_NE(hit->acquisition_serial, live.acquisition_serial);

    // Advance timeline on reuse (strictly monotonic).
    hit->payload->timeline_value = 11;
    EXPECT_EQ(pool.tryGetLive(hit->acquisition_serial)->timeline_value, 11u);
}

TEST(GpuResourcePool, ProducerPredSeesPayload) {
    auto pool = makePool();
    DestroyCapture cap;
    auto live = pool.registerCreated(makeKey(), makeUnit(1, /*timeline=*/5));
    pool.release(live.acquisition_serial, /*producer_value=*/10, /*consumer=*/0);

    // Watermark 10 but payload timeline is still 5 → hold.
    auto producer_hold = [](const FakeUnit& unit, const std::uint64_t value) {
        return unit.timeline_value >= value;
    };
    pool.drain(
        false, producer_hold, [](std::uint64_t) { return true; }, cap.fn());
    EXPECT_EQ(pool.retiredCount(), 1u);
    EXPECT_EQ(pool.freeCount(), 0u);

    // Simulate GPU completing more work on the retired unit (timeline advanced).
    // Retired entries are not tryGetLive; force-free by relaxing pred.
    auto producer_pass = [](const FakeUnit&, const std::uint64_t) { return true; };
    pool.drain(
        false, producer_pass, [](std::uint64_t) { return true; }, cap.fn());
    EXPECT_EQ(pool.freeCount(), 1u);
}

TEST(GpuResourcePool, EvictDestroysOnDrain) {
    auto pool = makePool();
    DestroyCapture cap;
    auto live = pool.registerCreated(makeKey(), makeUnit(99, 0, 2048));
    pool.release(live.acquisition_serial, 0, 0, /*evict=*/true);
    EXPECT_EQ(pool.idleBytes(), 2048u);

    pool.drain(
        false, [](const FakeUnit&, std::uint64_t) { return true; },
        [](std::uint64_t) { return true; }, cap.fn());
    ASSERT_EQ(cap.destroyed_ids.size(), 1u);
    EXPECT_EQ(cap.destroyed_ids[0], 99u);
    EXPECT_EQ(pool.freeCount(), 0u);
    EXPECT_EQ(pool.idleBytes(), 0u);
}

TEST(GpuResourcePool, ForceDrainSkipsLive) {
    auto pool = makePool();
    DestroyCapture cap;
    auto live = pool.registerCreated(makeKey(), makeUnit(1));
    auto free_src = pool.registerCreated(makeKey(), makeUnit(2));
    pool.release(free_src.acquisition_serial, 0, 0);
    pool.drain(
        false, [](const FakeUnit&, std::uint64_t) { return true; },
        [](std::uint64_t) { return true; }, cap.fn());
    EXPECT_EQ(pool.freeCount(), 1u);
    EXPECT_EQ(pool.liveCount(), 1u);
    cap.destroyed_ids.clear();

    pool.drain(
        true, [](const FakeUnit&, std::uint64_t) { return true; },
        [](std::uint64_t) { return true; }, cap.fn());
    ASSERT_EQ(cap.destroyed_ids.size(), 1u);
    EXPECT_EQ(cap.destroyed_ids[0], 2u);
    EXPECT_EQ(pool.liveCount(), 1u);
    EXPECT_EQ(live.payload->id, 1u);
}

TEST(GpuResourcePool, TrimIdleAndAged) {
    auto pool = makePool();
    DestroyCapture cap;
    const auto key = makeKey();

    auto a = pool.registerCreated(key, makeUnit(10));
    pool.release(a.acquisition_serial, 0, 0);
    pool.drain(
        false, [](const FakeUnit&, std::uint64_t) { return true; },
        [](std::uint64_t) { return true; }, cap.fn());
    EXPECT_EQ(pool.freeCount(), 1u);

    pool.trimIdle(cap.fn());
    ASSERT_EQ(cap.destroyed_ids.size(), 1u);
    EXPECT_EQ(cap.destroyed_ids[0], 10u);
    EXPECT_EQ(pool.freeCount(), 0u);

    cap.destroyed_ids.clear();
    auto stale = pool.registerCreated(key, makeUnit(20));
    pool.release(stale.acquisition_serial, 0, 0);
    pool.drain(
        false, [](const FakeUnit&, std::uint64_t) { return true; },
        [](std::uint64_t) { return true; }, cap.fn());

    auto fresh = pool.registerCreated(key, makeUnit(21));
    pool.release(fresh.acquisition_serial, 0, 0);
    for (std::uint64_t i = 0; i < Pool::kIdleTrimTicks; ++i) {
        pool.drain(
            false, [](const FakeUnit&, std::uint64_t) { return false; },
            [](std::uint64_t) { return false; }, cap.fn());
    }
    pool.drain(
        false, [](const FakeUnit&, std::uint64_t) { return false; },
        [](std::uint64_t) { return false; }, cap.fn());
    pool.drain(
        false, [](const FakeUnit&, std::uint64_t) { return true; },
        [](std::uint64_t) { return true; }, cap.fn());
    EXPECT_EQ(pool.freeCount(), 2u);
    cap.destroyed_ids.clear();

    pool.trimAged(cap.fn());
    ASSERT_EQ(cap.destroyed_ids.size(), 1u);
    EXPECT_EQ(cap.destroyed_ids[0], 20u);
    EXPECT_EQ(pool.freeCount(), 1u);
}

TEST(GpuResourcePool, TimelineMonotonicBookkeepingAcrossReuse) {
    auto pool = makePool();
    DestroyCapture cap;
    const auto key = makeKey();

    auto a = pool.registerCreated(key, makeUnit(1, /*timeline=*/0));
    a.payload->timeline_value = 3;
    a.payload->timeline_value = 5; // monotonic advance while live
    pool.release(a.acquisition_serial, a.payload->timeline_value, 0);
    pool.drain(
        false, [](const FakeUnit&, std::uint64_t) { return true; },
        [](std::uint64_t) { return true; }, cap.fn());

    auto b = pool.acquire(key);
    ASSERT_TRUE(b.has_value());
    EXPECT_EQ(b->payload->timeline_value, 5u);
    // Never reset on reuse.
    b->payload->timeline_value = 6;
    EXPECT_GT(b->payload->timeline_value, 5u);
}

TEST(CudaVulkanTensorFitsImport, SubrectAndExactAndReject) {
    using lfs::rendering::cudaVulkanTensorFitsImport;
    EXPECT_TRUE(cudaVulkanTensorFitsImport(100, 50, 128, 64));
    EXPECT_TRUE(cudaVulkanTensorFitsImport(128, 64, 128, 64));
    EXPECT_TRUE(cudaVulkanTensorFitsImport(1, 1, 64, 64));
    EXPECT_FALSE(cudaVulkanTensorFitsImport(129, 64, 128, 64));
    EXPECT_FALSE(cudaVulkanTensorFitsImport(128, 65, 128, 64));
    EXPECT_FALSE(cudaVulkanTensorFitsImport(0, 50, 128, 64));
    EXPECT_FALSE(cudaVulkanTensorFitsImport(100, 0, 128, 64));
}
