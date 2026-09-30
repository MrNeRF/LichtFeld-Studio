/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Epic #1568 / #1567 — OutputSlotRing host bookkeeping (GPU-free).

#include "rendering/output_slot_ring.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace {

    using lfs::vis::OutputImageSlot;
    using lfs::vis::OutputSlotRing;

    VkImage fakeImage(std::uintptr_t id) {
        return reinterpret_cast<VkImage>(id);
    }

    OutputImageSlot makeSlot(std::uintptr_t id, std::uint64_t completion = 0) {
        OutputImageSlot slot{};
        slot.image.image = fakeImage(id);
        slot.depth_image.image = fakeImage(id + 0x1000);
        slot.size = {64, 32};
        slot.alloc_size = {64, 64};
        slot.completion_value = completion;
        slot.color_pool_serial = id;
        slot.depth_pool_serial = id + 1;
        return slot;
    }

} // namespace

TEST(OutputSlotRing, AcquireRoundRobinWraps) {
    OutputSlotRing ring;
    EXPECT_EQ(ring.acquire({1}), 0u);
    EXPECT_EQ(ring.acquire({1}), 1u);
    EXPECT_EQ(ring.acquire({1}), 2u);
    EXPECT_EQ(ring.acquire({1}), 0u);
    EXPECT_EQ(ring.acquire({1}), 1u);
}

TEST(OutputSlotRing, WaitNoOpOnZeroWatermark) {
    OutputSlotRing ring;
    (void)ring.acquire({1});
    bool complete_called = false;
    bool wait_called = false;
    auto status = ring.waitUntilReusable(
        0,
        "test",
        [&](std::uint64_t) {
            complete_called = true;
            return false;
        },
        [&](std::uint64_t) -> lfs::Status {
            wait_called = true;
            return {};
        });
    EXPECT_TRUE(status);
    EXPECT_FALSE(complete_called);
    EXPECT_FALSE(wait_called);
}

TEST(OutputSlotRing, WaitClearsWatermarkWhenCompletePredTrue) {
    OutputSlotRing ring;
    (void)ring.acquire({1});
    ring.publishCompletion(1, 42);
    EXPECT_EQ(ring.ringCompletionValue(1), 42u);

    bool wait_called = false;
    auto status = ring.waitUntilReusable(
        1,
        "test",
        [](std::uint64_t value) {
            EXPECT_EQ(value, 42u);
            return true;
        },
        [&](std::uint64_t) -> lfs::Status {
            wait_called = true;
            return {};
        });
    EXPECT_TRUE(status);
    EXPECT_FALSE(wait_called);
    EXPECT_EQ(ring.ringCompletionValue(1), 0u);
}

TEST(OutputSlotRing, NonReadyWaitLeavesWatermarkIntact) {
    OutputSlotRing ring;
    (void)ring.acquire({1});
    ring.publishCompletion(2, 99);
    EXPECT_EQ(ring.ringCompletionValue(2), 99u);

    auto status = ring.waitUntilReusable(
        2,
        "selection overlay",
        [](std::uint64_t) { return false; },
        [](std::uint64_t value) -> lfs::Status {
            EXPECT_EQ(value, 99u);
            return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::DeadlineExceeded,
                .domain = lfs::ErrorDomain::Rendering,
                .user_message = "not ready",
                .detection = LFS_SOURCE_SITE_CURRENT(),
            }));
        });
    EXPECT_FALSE(status);
    EXPECT_EQ(status.error().user_message(), "not ready");
    // Critical: never manufacture a free slot on non-Ready.
    EXPECT_EQ(ring.ringCompletionValue(2), 99u);
}

TEST(OutputSlotRing, ThrowingWaitFnBecomesFailureAndLeavesWatermark) {
    OutputSlotRing ring;
    (void)ring.acquire({1});
    ring.publishCompletion(1, 42);

    auto status = ring.waitUntilReusable(
        1,
        "render",
        [](std::uint64_t) { return false; },
        [](std::uint64_t) -> lfs::Status { throw std::runtime_error("device lost"); });
    EXPECT_FALSE(status);
    EXPECT_NE(status.error().user_message().find("device lost"), std::string::npos);
    EXPECT_EQ(ring.ringCompletionValue(1), 42u);
}

TEST(OutputSlotRing, WaitClearsWatermarkOnReadyWaitFn) {
    OutputSlotRing ring;
    (void)ring.acquire({1});
    ring.publishCompletion(0, 7);
    auto status = ring.waitUntilReusable(
        0,
        "render",
        [](std::uint64_t) { return false; },
        [](std::uint64_t) -> lfs::Status { return {}; });
    EXPECT_TRUE(status);
    EXPECT_EQ(ring.ringCompletionValue(0), 0u);
}

TEST(OutputSlotRing, CompleteFnExceptionBecomesStatusAndLeavesWatermark) {
    OutputSlotRing ring;
    (void)ring.acquire({1});
    ring.publishCompletion(0, 11);
    auto status = ring.waitUntilReusable(
        0,
        "selection query",
        [](std::uint64_t) -> bool { throw std::runtime_error("poll failed"); },
        [](std::uint64_t) -> lfs::Status { return {}; });
    EXPECT_FALSE(status);
    EXPECT_NE(std::string(status.error().user_message()).find("selection query"), std::string::npos);
    EXPECT_NE(std::string(status.error().user_message()).find("poll failed"), std::string::npos);
    EXPECT_EQ(ring.ringCompletionValue(0), 11u);
}

TEST(OutputSlotRing, SparseTargetsDoNotWaitOnNeighbours) {
    OutputSlotRing ring;
    for (std::uint32_t id = 1; id < 100; ++id) {
        const auto cell = ring.acquire({id * 101});
        ASSERT_TRUE(ring.waitUntilReusable(cell, "render", [](auto) { return false; }, [](auto) -> lfs::Status { ADD_FAILURE() << "Cross-target wait"; return {}; }));
        ring.publishCompletion(cell, id);
    }
    EXPECT_EQ(ring.table().size(), 99u);
}

TEST(OutputSlotRing, ReleaseKeepsNeighbourAndDropsLatePublication) {
    OutputSlotRing ring;
    const lfs::vis::RenderTargetId a{17}, b{9001};
    auto acell = ring.acquire(a);
    auto bcell = ring.acquire(b);
    ring.slotAt(a, acell) = makeSlot(12, 77);
    ring.slotAt(b, bcell) = makeSlot(13, 78);
    ring.publishCompletion(acell, 77);
    std::vector<OutputImageSlot> retired;
    ASSERT_TRUE(ring.releaseRenderTarget(a, [&](auto& slot) { retired.push_back(slot); }));
    EXPECT_EQ(retired.front().completion_value, 77u);
    EXPECT_EQ(ring.slotAt(b, bcell).image.image, fakeImage(13));
    EXPECT_EQ(ring.ringCompletionValue(acell), 77u);
    ring.markLatest(a, acell);
    EXPECT_EQ(ring.bumpGeneration(a), 0u);
    EXPECT_THROW((void)ring.acquire(a), std::invalid_argument);
    EXPECT_FALSE(ring.contains(a));
    EXPECT_NE(ring.acquire({18}), acell);
}

TEST(RenderTargetRegistry, NeverReusesReleasedIds) {
    lfs::vis::RenderTargetRegistry registry;
    auto a = registry.allocate();
    EXPECT_TRUE(a.valid());
    EXPECT_TRUE(registry.release(a));
    auto b = registry.allocate();
    EXPECT_GT(b.value, a.value);
    EXPECT_FALSE(registry.contains(a));
    EXPECT_TRUE(registry.contains(b));
}
