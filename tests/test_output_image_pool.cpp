/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Epic #1497 R1 — OutputImagePool host bookkeeping (GPU-free).

#include "rendering/output_image_pool.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <glm/glm.hpp>
#include <vector>
#include <vulkan/vulkan.h>

namespace {

    using lfs::vis::ceil64;
    using lfs::vis::OutputImagePool;
    using lfs::vis::VulkanContext;

    VkImage fakeImage(std::uintptr_t id) {
        return reinterpret_cast<VkImage>(id);
    }

    VkImageView fakeView(std::uintptr_t id) {
        return reinterpret_cast<VkImageView>(id);
    }

    VkDeviceMemory fakeMemory(std::uintptr_t id) {
        return reinterpret_cast<VkDeviceMemory>(id);
    }

    OutputImagePool::Key makeKey(VkFormat format = VK_FORMAT_R8G8B8A8_UNORM,
                                 std::uint32_t w = 128,
                                 std::uint32_t h = 64,
                                 VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT,
                                 bool external = true) {
        return OutputImagePool::Key{
            .format = format,
            .extent = VkExtent2D{w, h},
            .usage = usage,
            .external = external,
        };
    }

    VulkanContext::ExternalImage makeImage(std::uintptr_t id,
                                           VkFormat format = VK_FORMAT_R8G8B8A8_UNORM,
                                           std::uint32_t w = 128,
                                           std::uint32_t h = 64) {
        VulkanContext::ExternalImage img{};
        img.image = fakeImage(id);
        img.view = fakeView(id + 0x1000);
        img.memory = fakeMemory(id + 0x2000);
        img.extent = VkExtent2D{w, h};
        img.format = format;
        img.allocation_size = static_cast<VkDeviceSize>(w) * h * 4;
        img.diagnostic_scope = "vulkan.vksplat.output_pool";
        img.diagnostic_label = "test";
        img.census_counted = true;
        return img;
    }

    struct DestroyCapture {
        std::vector<VkImage> destroyed;

        OutputImagePool::DestroyFn fn() {
            return [this](VulkanContext::ExternalImage& image) {
                destroyed.push_back(image.image);
                image = {};
            };
        }
    };

    // Payload-aware producer predicates (GpuResourcePool API); value-only lambdas
    // still work as FramePred for the consumer side.
    const auto always_done = [](const VulkanContext::ExternalImage&, std::uint64_t) {
        return true;
    };
    const auto never_done = [](const VulkanContext::ExternalImage&, std::uint64_t) {
        return false;
    };
    const auto always_frame = [](std::uint64_t) { return true; };
    const auto never_frame = [](std::uint64_t) { return false; };

} // namespace

TEST(OutputImagePool, AcquireMissOnEmptyPool) {
    OutputImagePool pool;
    EXPECT_FALSE(pool.acquire(makeKey()).has_value());
    EXPECT_EQ(pool.liveCount(), 0u);
    EXPECT_EQ(pool.retiredCount(), 0u);
    EXPECT_EQ(pool.freeCount(), 0u);
}

TEST(OutputImagePool, RegisterCreatedThenReleaseDrainBothPredicatesFreesForReuse) {
    OutputImagePool pool;
    DestroyCapture cap;
    const auto key = makeKey();
    const VkImage handle = fakeImage(0xA001);

    auto acquired = pool.registerCreated(key, makeImage(0xA001));
    ASSERT_NE(acquired.acquisition_serial, 0u);
    EXPECT_EQ(acquired.image.image, handle);
    EXPECT_EQ(pool.liveCount(), 1u);

    pool.release(acquired.acquisition_serial, /*producer_value=*/10, /*consumer_serial=*/20);
    EXPECT_EQ(pool.liveCount(), 0u);
    EXPECT_EQ(pool.retiredCount(), 1u);

    pool.drain(/*force=*/false, always_done, always_frame, cap.fn());
    EXPECT_EQ(pool.retiredCount(), 0u);
    EXPECT_EQ(pool.freeCount(), 1u);
    EXPECT_TRUE(cap.destroyed.empty());

    auto reused = pool.acquire(key);
    ASSERT_TRUE(reused.has_value());
    EXPECT_EQ(reused->image.image, handle);
    EXPECT_NE(reused->acquisition_serial, acquired.acquisition_serial);
    EXPECT_EQ(pool.liveCount(), 1u);
    EXPECT_EQ(pool.freeCount(), 0u);
}

TEST(OutputImagePool, DrainProducerIncompleteHolds) {
    OutputImagePool pool;
    DestroyCapture cap;
    auto acquired = pool.registerCreated(makeKey(), makeImage(0xB001));
    pool.release(acquired.acquisition_serial, 5, 6);

    pool.drain(false, never_done, always_frame, cap.fn());
    EXPECT_EQ(pool.retiredCount(), 1u);
    EXPECT_EQ(pool.freeCount(), 0u);
    EXPECT_TRUE(cap.destroyed.empty());
    EXPECT_FALSE(pool.acquire(makeKey()).has_value());
}

TEST(OutputImagePool, DrainConsumerIncompleteHolds) {
    OutputImagePool pool;
    DestroyCapture cap;
    auto acquired = pool.registerCreated(makeKey(), makeImage(0xB002));
    pool.release(acquired.acquisition_serial, 5, 6);

    pool.drain(false, always_done, never_frame, cap.fn());
    EXPECT_EQ(pool.retiredCount(), 1u);
    EXPECT_EQ(pool.freeCount(), 0u);
    EXPECT_TRUE(cap.destroyed.empty());
    EXPECT_FALSE(pool.acquire(makeKey()).has_value());
}

TEST(OutputImagePool, ExternalFlagPartitionsKeys) {
    OutputImagePool pool;
    DestroyCapture cap;
    auto key_ext = makeKey(VK_FORMAT_R8G8B8A8_UNORM, 128, 64, VK_IMAGE_USAGE_STORAGE_BIT, true);
    auto key_int = makeKey(VK_FORMAT_R8G8B8A8_UNORM, 128, 64, VK_IMAGE_USAGE_STORAGE_BIT, false);

    auto a = pool.registerCreated(key_ext, makeImage(0xC001));
    pool.release(a.acquisition_serial, 1, 1);
    pool.drain(false, always_done, always_frame, cap.fn());

    EXPECT_FALSE(pool.acquire(key_int).has_value());
    auto hit = pool.acquire(key_ext);
    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(hit->image.image, fakeImage(0xC001));
}

TEST(OutputImagePool, FormatPartitionsKeys) {
    OutputImagePool pool;
    DestroyCapture cap;
    auto key_rgba = makeKey(VK_FORMAT_R8G8B8A8_UNORM);
    auto key_r32f = makeKey(VK_FORMAT_R32_SFLOAT);

    auto a = pool.registerCreated(key_rgba, makeImage(0xC010, VK_FORMAT_R8G8B8A8_UNORM));
    pool.release(a.acquisition_serial, 1, 1);
    pool.drain(false, always_done, always_frame, cap.fn());

    EXPECT_FALSE(pool.acquire(key_r32f).has_value());
    EXPECT_TRUE(pool.acquire(key_rgba).has_value());
}

TEST(OutputImagePool, ExtentPartitionsKeys) {
    OutputImagePool pool;
    DestroyCapture cap;
    auto key_a = makeKey(VK_FORMAT_R8G8B8A8_UNORM, 128, 64);
    auto key_b = makeKey(VK_FORMAT_R8G8B8A8_UNORM, 256, 64);

    auto a = pool.registerCreated(key_a, makeImage(0xC020, VK_FORMAT_R8G8B8A8_UNORM, 128, 64));
    pool.release(a.acquisition_serial, 1, 1);
    pool.drain(false, always_done, always_frame, cap.fn());

    EXPECT_FALSE(pool.acquire(key_b).has_value());
    EXPECT_TRUE(pool.acquire(key_a).has_value());
}

TEST(OutputImagePool, UsagePartitionsKeys) {
    OutputImagePool pool;
    DestroyCapture cap;
    auto key_a = makeKey(VK_FORMAT_R8G8B8A8_UNORM, 128, 64, VK_IMAGE_USAGE_STORAGE_BIT);
    auto key_b = makeKey(VK_FORMAT_R8G8B8A8_UNORM, 128, 64, VK_IMAGE_USAGE_SAMPLED_BIT);

    auto a = pool.registerCreated(key_a, makeImage(0xC030));
    pool.release(a.acquisition_serial, 1, 1);
    pool.drain(false, always_done, always_frame, cap.fn());

    EXPECT_FALSE(pool.acquire(key_b).has_value());
    EXPECT_TRUE(pool.acquire(key_a).has_value());
}

TEST(OutputImagePool, AcquisitionSerialsMonotonicNeverReused) {
    OutputImagePool pool;
    DestroyCapture cap;
    const auto key = makeKey();

    auto a = pool.registerCreated(key, makeImage(0xD001));
    auto b = pool.registerCreated(key, makeImage(0xD002));
    EXPECT_GT(b.acquisition_serial, a.acquisition_serial);

    pool.release(a.acquisition_serial, 1, 1);
    pool.release(b.acquisition_serial, 1, 1);
    pool.drain(false, always_done, always_frame, cap.fn());

    auto c = pool.acquire(key);
    auto d = pool.acquire(key);
    ASSERT_TRUE(c.has_value());
    ASSERT_TRUE(d.has_value());
    EXPECT_GT(c->acquisition_serial, b.acquisition_serial);
    EXPECT_GT(d->acquisition_serial, c->acquisition_serial);

    const std::uint64_t serials[] = {
        a.acquisition_serial,
        b.acquisition_serial,
        c->acquisition_serial,
        d->acquisition_serial,
    };
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = i + 1; j < 4; ++j) {
            EXPECT_NE(serials[i], serials[j]);
        }
    }
}

// Release path is the asserted behavior; debug LFS_VK_DEBUG_ASSERT is noreturn on misuse.
TEST(OutputImagePool, DoubleReleaseFlaggedAndIgnored) {
#if !defined(NDEBUG)
    GTEST_SKIP() << "Debug builds assert on double release (LFS_VK_DEBUG_ASSERT); release path covered here.";
#else
    OutputImagePool pool;
    DestroyCapture cap;
    auto acquired = pool.registerCreated(makeKey(), makeImage(0xE001));
    pool.release(acquired.acquisition_serial, 1, 1);
    EXPECT_FALSE(pool.misuseFlagged());
    EXPECT_EQ(pool.retiredCount(), 1u);

    pool.release(acquired.acquisition_serial, 1, 1);
    EXPECT_TRUE(pool.misuseFlagged());
    EXPECT_EQ(pool.retiredCount(), 1u);
    EXPECT_EQ(pool.liveCount(), 0u);

    pool.drain(false, always_done, always_frame, cap.fn());
    EXPECT_TRUE(cap.destroyed.empty());
    EXPECT_EQ(pool.freeCount(), 1u);
#endif
}

TEST(OutputImagePool, ReleaseUnknownSerialFlagged) {
#if !defined(NDEBUG)
    GTEST_SKIP() << "Debug builds assert on unknown release (LFS_VK_DEBUG_ASSERT); release path covered here.";
#else
    OutputImagePool pool;
    pool.release(/*unknown=*/999, 1, 1);
    EXPECT_TRUE(pool.misuseFlagged());
    EXPECT_EQ(pool.liveCount(), 0u);
    EXPECT_EQ(pool.retiredCount(), 0u);
#endif
}

TEST(OutputImagePool, EvictReleaseDestroysOnDrainWhenBothPredsPass) {
    OutputImagePool pool;
    DestroyCapture cap;
    const auto key = makeKey();
    const VkImage handle = fakeImage(0xE100);

    auto acquired = pool.registerCreated(key, makeImage(0xE100));
    pool.release(acquired.acquisition_serial, /*producer_value=*/10, /*consumer_serial=*/20,
                 /*evict=*/true);
    EXPECT_EQ(pool.liveCount(), 0u);
    EXPECT_EQ(pool.retiredCount(), 1u);
    EXPECT_EQ(pool.idleBytes(), 128u * 64u * 4u);

    pool.drain(/*force=*/false, always_done, always_frame, cap.fn());
    ASSERT_EQ(cap.destroyed.size(), 1u);
    EXPECT_EQ(cap.destroyed[0], handle);
    EXPECT_EQ(pool.retiredCount(), 0u);
    EXPECT_EQ(pool.freeCount(), 0u);
    EXPECT_EQ(pool.idleBytes(), 0u);
    EXPECT_FALSE(pool.acquire(key).has_value());
}

TEST(OutputImagePool, EvictReleaseHoldsWhileEitherPredFails) {
    OutputImagePool pool;
    DestroyCapture cap;
    const auto key = makeKey();

    auto acquired = pool.registerCreated(key, makeImage(0xE101));
    pool.release(acquired.acquisition_serial, 5, 6, /*evict=*/true);

    pool.drain(false, never_done, always_frame, cap.fn());
    EXPECT_EQ(pool.retiredCount(), 1u);
    EXPECT_TRUE(cap.destroyed.empty());
    EXPECT_FALSE(pool.acquire(key).has_value());

    pool.drain(false, always_done, never_frame, cap.fn());
    EXPECT_EQ(pool.retiredCount(), 1u);
    EXPECT_TRUE(cap.destroyed.empty());
    EXPECT_FALSE(pool.acquire(key).has_value());
    EXPECT_EQ(pool.idleBytes(), 128u * 64u * 4u);
}

TEST(OutputImagePool, EvictEntriesNeverReappearViaAcquire) {
    OutputImagePool pool;
    DestroyCapture cap;
    const auto key = makeKey();

    auto acquired = pool.registerCreated(key, makeImage(0xE102));
    pool.release(acquired.acquisition_serial, 1, 1, /*evict=*/true);
    pool.drain(false, always_done, always_frame, cap.fn());

    EXPECT_EQ(cap.destroyed.size(), 1u);
    EXPECT_FALSE(pool.acquire(key).has_value());
    EXPECT_EQ(pool.freeCount(), 0u);
    EXPECT_EQ(pool.liveCount(), 0u);
}

TEST(OutputImagePool, NonEvictReleaseStillFreeLists) {
    OutputImagePool pool;
    DestroyCapture cap;
    const auto key = makeKey();
    const VkImage handle = fakeImage(0xE103);

    auto acquired = pool.registerCreated(key, makeImage(0xE103));
    pool.release(acquired.acquisition_serial, 1, 1, /*evict=*/false);
    pool.drain(false, always_done, always_frame, cap.fn());
    EXPECT_TRUE(cap.destroyed.empty());
    EXPECT_EQ(pool.freeCount(), 1u);

    auto reused = pool.acquire(key);
    ASSERT_TRUE(reused.has_value());
    EXPECT_EQ(reused->image.image, handle);
}

TEST(OutputImagePool, ForceDrainDestroysEvictedRetiredEntries) {
    OutputImagePool pool;
    DestroyCapture cap;
    const auto key = makeKey();

    auto live = pool.registerCreated(key, makeImage(0xE104));
    auto free_src = pool.registerCreated(key, makeImage(0xE105));
    pool.release(free_src.acquisition_serial, 1, 1, /*evict=*/false);
    pool.drain(false, always_done, always_frame, cap.fn());

    auto retired_evict = pool.registerCreated(key, makeImage(0xE106));
    pool.release(retired_evict.acquisition_serial, 1, 1, /*evict=*/true);
    pool.drain(false, never_done, never_frame, cap.fn());

    auto retired_keep = pool.registerCreated(key, makeImage(0xE107));
    pool.release(retired_keep.acquisition_serial, 1, 1, /*evict=*/false);
    pool.drain(false, never_done, never_frame, cap.fn());

    EXPECT_EQ(pool.freeCount(), 1u);
    EXPECT_EQ(pool.retiredCount(), 2u);
    EXPECT_EQ(pool.liveCount(), 1u);
    cap.destroyed.clear();

    pool.drain(/*force=*/true, always_done, always_frame, cap.fn());
    EXPECT_EQ(cap.destroyed.size(), 3u);
    EXPECT_EQ(pool.liveCount(), 1u);
    EXPECT_EQ(pool.retiredCount(), 0u);
    EXPECT_EQ(pool.freeCount(), 0u);
    EXPECT_EQ(live.image.image, fakeImage(0xE104));
}

TEST(OutputImagePool, ForceDrainDestroysRetiredAndFreeNotLive) {
    OutputImagePool pool;
    DestroyCapture cap;
    const auto key = makeKey();

    auto live = pool.registerCreated(key, makeImage(0xF001));

    // Free-list entry via release + successful drain.
    auto free_src = pool.registerCreated(key, makeImage(0xF002));
    pool.release(free_src.acquisition_serial, 1, 1);
    pool.drain(false, always_done, always_frame, cap.fn());

    // Retired entry held by incomplete predicates (not yet free).
    auto retired = pool.registerCreated(key, makeImage(0xF003));
    pool.release(retired.acquisition_serial, 1, 1);
    pool.drain(false, never_done, never_frame, cap.fn());

    EXPECT_EQ(pool.freeCount(), 1u);
    EXPECT_EQ(pool.retiredCount(), 1u);
    EXPECT_EQ(pool.liveCount(), 1u);
    cap.destroyed.clear();

    pool.drain(/*force=*/true, always_done, always_frame, cap.fn());
    EXPECT_EQ(cap.destroyed.size(), 2u);
    EXPECT_EQ(pool.liveCount(), 1u);
    EXPECT_EQ(pool.retiredCount(), 0u);
    EXPECT_EQ(pool.freeCount(), 0u);
    EXPECT_EQ(live.image.image, fakeImage(0xF001));
}

TEST(OutputImagePool, TrimIdleDestroysOnlyFree) {
    OutputImagePool pool;
    DestroyCapture cap;
    const auto key = makeKey();

    auto live = pool.registerCreated(key, makeImage(0xF010));

    auto free_src = pool.registerCreated(key, makeImage(0xF012));
    pool.release(free_src.acquisition_serial, 1, 1);
    pool.drain(false, always_done, always_frame, cap.fn());

    auto retired = pool.registerCreated(key, makeImage(0xF011));
    pool.release(retired.acquisition_serial, 1, 1);
    pool.drain(false, never_done, never_frame, cap.fn());
    cap.destroyed.clear();

    pool.trimIdle(cap.fn());
    ASSERT_EQ(cap.destroyed.size(), 1u);
    EXPECT_EQ(cap.destroyed[0], fakeImage(0xF012));
    EXPECT_EQ(pool.freeCount(), 0u);
    EXPECT_EQ(pool.liveCount(), 1u);
    EXPECT_EQ(pool.retiredCount(), 1u);
    (void)live;
}

TEST(OutputImagePool, TrimAgedDestroysOnlyStaleEntries) {
    OutputImagePool pool;
    DestroyCapture cap;
    const auto key = makeKey();

    auto stale = pool.registerCreated(key, makeImage(0xF020));
    pool.release(stale.acquisition_serial, 1, 1);
    pool.drain(false, always_done, always_frame, cap.fn());

    auto fresh = pool.registerCreated(key, makeImage(0xF021));
    pool.release(fresh.acquisition_serial, 1, 1);

    // Advance drain ticks without freeing the fresh retired entry.
    for (std::uint64_t i = 0; i < OutputImagePool::kIdleTrimTicks; ++i) {
        pool.drain(false, never_done, never_frame, cap.fn());
    }
    // One more tick so stale free age exceeds kIdleTrimTicks.
    pool.drain(false, never_done, never_frame, cap.fn());

    // Free the fresh one now — its free_since_tick is current.
    pool.drain(false, always_done, always_frame, cap.fn());
    EXPECT_EQ(pool.freeCount(), 2u);
    cap.destroyed.clear();

    pool.trimAged(cap.fn());
    ASSERT_EQ(cap.destroyed.size(), 1u);
    EXPECT_EQ(cap.destroyed[0], fakeImage(0xF020));
    EXPECT_EQ(pool.freeCount(), 1u);

    // Fresh entry still reusable.
    auto hit = pool.acquire(key);
    ASSERT_TRUE(hit.has_value());
    EXPECT_EQ(hit->image.image, fakeImage(0xF021));
}

TEST(OutputImagePool, CountersConsistent) {
    OutputImagePool pool;
    DestroyCapture cap;
    const auto key = makeKey();

    EXPECT_EQ(pool.liveCount(), 0u);
    EXPECT_EQ(pool.retiredCount(), 0u);
    EXPECT_EQ(pool.freeCount(), 0u);

    auto a = pool.registerCreated(key, makeImage(0xF030));
    auto b = pool.registerCreated(key, makeImage(0xF031));
    EXPECT_EQ(pool.liveCount(), 2u);

    pool.release(a.acquisition_serial, 1, 1);
    EXPECT_EQ(pool.liveCount(), 1u);
    EXPECT_EQ(pool.retiredCount(), 1u);

    pool.drain(false, always_done, always_frame, cap.fn());
    EXPECT_EQ(pool.liveCount(), 1u);
    EXPECT_EQ(pool.retiredCount(), 0u);
    EXPECT_EQ(pool.freeCount(), 1u);

    auto reused = pool.acquire(key);
    ASSERT_TRUE(reused.has_value());
    EXPECT_EQ(pool.liveCount(), 2u);
    EXPECT_EQ(pool.freeCount(), 0u);

    pool.release(b.acquisition_serial, 1, 1);
    pool.release(reused->acquisition_serial, 1, 1);
    pool.drain(true, always_done, always_frame, cap.fn());
    EXPECT_EQ(pool.liveCount(), 0u);
    EXPECT_EQ(pool.retiredCount(), 0u);
    EXPECT_EQ(pool.freeCount(), 0u);
    EXPECT_EQ(cap.destroyed.size(), 2u);
}

TEST(OutputImagePool, IdleBytesTracksRetiredAndFreeEntries) {
    OutputImagePool pool;
    DestroyCapture cap;
    const auto key = makeKey();

    auto first = pool.registerCreated(key, makeImage(0xF040, VK_FORMAT_R8G8B8A8_UNORM, 128, 64));
    auto second = pool.registerCreated(key, makeImage(0xF041, VK_FORMAT_R8G8B8A8_UNORM, 128, 64));
    const std::size_t bytes_each = 128u * 64u * 4u;
    EXPECT_EQ(pool.idleBytes(), 0u);

    pool.release(first.acquisition_serial, 1, 1);
    EXPECT_EQ(pool.idleBytes(), bytes_each);

    pool.drain(false, always_done, always_frame, cap.fn());
    EXPECT_EQ(pool.idleBytes(), bytes_each);

    pool.release(second.acquisition_serial, 1, 1);
    pool.drain(false, never_done, never_frame, cap.fn());
    EXPECT_EQ(pool.idleBytes(), 2u * bytes_each);

    auto reused = pool.acquire(key);
    ASSERT_TRUE(reused.has_value());
    EXPECT_EQ(pool.idleBytes(), bytes_each);

    pool.trimIdle(cap.fn());
    EXPECT_EQ(pool.idleBytes(), bytes_each);
    pool.drain(true, always_done, always_frame, cap.fn());
    EXPECT_EQ(pool.idleBytes(), 0u);
}

TEST(OutputImagePool, Ceil64Bucketing) {
    EXPECT_EQ(ceil64(0), 0u);
    EXPECT_EQ(ceil64(1), 64u);
    EXPECT_EQ(ceil64(63), 64u);
    EXPECT_EQ(ceil64(64), 64u);
    EXPECT_EQ(ceil64(65), 128u);
    EXPECT_EQ(ceil64(128), 128u);
    EXPECT_EQ(ceil64(129), 192u);
    EXPECT_EQ(ceil64(1920), 1920u);
    EXPECT_EQ(ceil64(1921), 1984u);
}

TEST(OutputImagePool, OutputUvScaleDefaultsAndBucketing) {
    using lfs::vis::outputUvScale;

    const auto s = outputUvScale({100, 50}, {128, 64});
    EXPECT_FLOAT_EQ(s.x, 100.0f / 128.0f);
    EXPECT_FLOAT_EQ(s.y, 50.0f / 64.0f);

    EXPECT_EQ(outputUvScale({64, 64}, {64, 64}), glm::vec2(1.0f, 1.0f));
    EXPECT_EQ(outputUvScale({128, 64}, {64, 64}), glm::vec2(1.0f, 1.0f));
    EXPECT_EQ(outputUvScale({100, 50}, {0, 64}), glm::vec2(1.0f, 50.0f / 64.0f));
    EXPECT_EQ(outputUvScale({100, 50}, {128, 0}), glm::vec2(100.0f / 128.0f, 1.0f));
    EXPECT_EQ(outputUvScale({0, 0}, {128, 64}), glm::vec2(1.0f, 1.0f));
}

TEST(OutputImagePool, OutputUvClampMaxHalfTexel) {
    using lfs::vis::outputUvClampMax;

    const auto c = outputUvClampMax({100, 50}, {128, 64});
    EXPECT_FLOAT_EQ(c.x, (100.0f - 0.5f) / 128.0f);
    EXPECT_FLOAT_EQ(c.y, (50.0f - 0.5f) / 64.0f);

    const auto exact = outputUvClampMax({64, 64}, {64, 64});
    EXPECT_FLOAT_EQ(exact.x, 63.5f / 64.0f);
    EXPECT_FLOAT_EQ(exact.y, 63.5f / 64.0f);

    EXPECT_EQ(outputUvClampMax({0, 50}, {128, 64}), glm::vec2(1.0f, (50.0f - 0.5f) / 64.0f));
    EXPECT_EQ(outputUvClampMax({100, 0}, {128, 64}), glm::vec2((100.0f - 0.5f) / 128.0f, 1.0f));
    EXPECT_EQ(outputUvClampMax({100, 50}, {0, 0}), glm::vec2(1.0f, 1.0f));
}
