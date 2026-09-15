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
    EXPECT_EQ(ring.acquire(), 0u);
    EXPECT_EQ(ring.acquire(), 1u);
    EXPECT_EQ(ring.acquire(), 2u);
    EXPECT_EQ(ring.acquire(), 0u);
    EXPECT_EQ(ring.acquire(), 1u);
    EXPECT_EQ(ring.nextRingSlot(), 2u);
}

TEST(OutputSlotRing, WaitNoOpOnZeroWatermark) {
    OutputSlotRing ring;
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

TEST(OutputSlotRing, PublishCompletionAndClearOnComposeStart) {
    OutputSlotRing ring;
    constexpr std::size_t logical = 0;
    constexpr std::size_t ring_i = 1;
    ring.slotAt(logical, ring_i).completion_value = 5;
    ring.publishCompletion(ring_i, 50);
    EXPECT_EQ(ring.ringCompletionValue(ring_i), 50u);
    EXPECT_EQ(ring.slotAt(logical, ring_i).completion_value, 5u);

    ring.clearSlotCompletion(logical, ring_i);
    EXPECT_EQ(ring.slotAt(logical, ring_i).completion_value, 0u);
    // Ring watermark is independent of per-slot clear.
    EXPECT_EQ(ring.ringCompletionValue(ring_i), 50u);
}

TEST(OutputSlotRing, LatestPublishAndBoundsCheck) {
    OutputSlotRing ring;
    ring.markLatest(/*logical=*/2, /*ring=*/1);
    EXPECT_EQ(ring.latestRingSlot(2), 1u);
    ring.slotAt(2, 1) = makeSlot(0xBEEF);
    EXPECT_EQ(ring.latestSlot(2).image.image, fakeImage(0xBEEF));

    // Corrupt the stored latest beyond ring size → throw.
    // Direct write via reset-path simulation: mark then poke table.
    // latest_output_ring_slot_ is private; force via clearLogical + manual
    // poke is not available — use a known-good then verify out_of_range on
    // bogus logical index.
    EXPECT_THROW((void)ring.latestRingSlot(OutputSlotRing::kOutputSlotCount), std::out_of_range);
    EXPECT_THROW((void)ring.slotAt(0, OutputSlotRing::kFrameRingSize), std::out_of_range);
    EXPECT_THROW((void)ring.slotAt(OutputSlotRing::kOutputSlotCount, 0), std::out_of_range);
}

TEST(OutputSlotRing, ClearLogicalInvokesPerSlotCallbackAndZerosColumn) {
    OutputSlotRing ring;
    constexpr std::size_t logical = 1;
    ring.slotAt(logical, 0) = makeSlot(0x10, 1);
    ring.slotAt(logical, 1) = makeSlot(0x20, 2);
    ring.slotAt(logical, 2) = makeSlot(0x30, 3);
    ring.markLatest(logical, 2);
    (void)ring.bumpGeneration(logical);
    (void)ring.bumpGeneration(logical);
    EXPECT_EQ(ring.generation(logical), 2u);

    // Unrelated logical column must survive.
    ring.slotAt(0, 0) = makeSlot(0xAA);
    ring.markLatest(0, 0);

    std::vector<std::uint64_t> released_serials;
    ring.clearLogical(logical, [&](OutputImageSlot& slot) {
        if (slot.color_pool_serial != 0) {
            released_serials.push_back(slot.color_pool_serial);
        }
    });

    ASSERT_EQ(released_serials.size(), 3u);
    EXPECT_EQ(released_serials[0], 0x10u);
    EXPECT_EQ(released_serials[1], 0x20u);
    EXPECT_EQ(released_serials[2], 0x30u);

    for (std::size_t r = 0; r < OutputSlotRing::kFrameRingSize; ++r) {
        EXPECT_EQ(ring.slotAt(logical, r).image.image, VK_NULL_HANDLE);
        EXPECT_EQ(ring.slotAt(logical, r).completion_value, 0u);
        EXPECT_EQ(ring.slotAt(logical, r).color_pool_serial, 0u);
    }
    EXPECT_EQ(ring.latestRingSlot(logical), 0u);
    EXPECT_EQ(ring.generation(logical), 0u);

    // Other column intact.
    EXPECT_EQ(ring.slotAt(0, 0).image.image, fakeImage(0xAA));
    EXPECT_EQ(ring.latestRingSlot(0), 0u);
}

TEST(OutputSlotRing, ResetZerosEverything) {
    OutputSlotRing ring;
    ring.slotAt(0, 0) = makeSlot(1);
    ring.slotAt(3, 2) = makeSlot(2, 9);
    ring.publishCompletion(0, 100);
    ring.publishCompletion(2, 200);
    ring.markLatest(0, 1);
    (void)ring.bumpGeneration(0);
    (void)ring.acquire();
    (void)ring.acquire();
    EXPECT_NE(ring.nextRingSlot(), 0u);

    ring.reset();

    EXPECT_EQ(ring.nextRingSlot(), 0u);
    for (std::size_t r = 0; r < OutputSlotRing::kFrameRingSize; ++r) {
        EXPECT_EQ(ring.ringCompletionValue(r), 0u);
    }
    for (std::size_t L = 0; L < OutputSlotRing::kOutputSlotCount; ++L) {
        EXPECT_EQ(ring.generation(L), 0u);
        EXPECT_EQ(ring.latestRingSlot(L), 0u);
        for (std::size_t r = 0; r < OutputSlotRing::kFrameRingSize; ++r) {
            EXPECT_EQ(ring.slotAt(L, r).image.image, VK_NULL_HANDLE);
            EXPECT_EQ(ring.slotAt(L, r).completion_value, 0u);
        }
    }
}

TEST(OutputSlotRing, GenerationBumpMonotonic) {
    OutputSlotRing ring;
    EXPECT_EQ(ring.generation(0), 0u);
    EXPECT_EQ(ring.bumpGeneration(0), 1u);
    EXPECT_EQ(ring.bumpGeneration(0), 2u);
    EXPECT_EQ(ring.bumpGeneration(0), 3u);
    EXPECT_EQ(ring.generation(0), 3u);
    // Independent per logical slot.
    EXPECT_EQ(ring.bumpGeneration(1), 1u);
    EXPECT_EQ(ring.generation(0), 3u);
    EXPECT_EQ(ring.generation(1), 1u);
}

TEST(OutputSlotRing, OutOfRangeWaitIsNoOpSuccess) {
    OutputSlotRing ring;
    ring.publishCompletion(0, 5);
    auto status = ring.waitUntilReusable(
        OutputSlotRing::kFrameRingSize,
        "test",
        [](std::uint64_t) { return true; },
        [](std::uint64_t) -> lfs::Status { return {}; });
    EXPECT_TRUE(status);
    EXPECT_EQ(ring.ringCompletionValue(0), 5u);
}

TEST(OutputSlotRing, LegacyColumnsRemainTheDefaultRegistry) {
    OutputSlotRing ring;

    EXPECT_EQ(ring.logicalCount(), OutputSlotRing::kOutputSlotCount);
    EXPECT_EQ(ring.activeLogicalCount(), OutputSlotRing::kOutputSlotCount);
    EXPECT_EQ(ring.findKey(lfs::vis::kLegacyMainOutputKey), 0u);
    EXPECT_EQ(ring.findKey(lfs::vis::kLegacyPreviewOutputKey), 3u);
    EXPECT_EQ(ring.keyAt(0), lfs::vis::kLegacyMainOutputKey);
    EXPECT_FALSE(ring.findKey(lfs::vis::kInvalidViewOutputKey));
}

TEST(OutputSlotRing, SparseSceneKeysOnlyAllocateRegisteredColumns) {
    OutputSlotRing ring;
    constexpr lfs::vis::ViewId view_id = 0x7fff'ffff'ffff'0000ull;

    const auto first_view = ring.registerSceneView(1);
    EXPECT_NE(ring.findKey(lfs::vis::sceneOutputKey(1)),
              ring.findKey(lfs::vis::kLegacyMainOutputKey));
    const auto logical = ring.registerSceneView(view_id);
    EXPECT_EQ(first_view, OutputSlotRing::kOutputSlotCount);
    EXPECT_EQ(logical, OutputSlotRing::kOutputSlotCount + 1);
    EXPECT_EQ(ring.logicalCount(), OutputSlotRing::kOutputSlotCount + 2);
    EXPECT_EQ(ring.activeLogicalCount(), OutputSlotRing::kOutputSlotCount + 2);
    EXPECT_EQ(ring.findKey(lfs::vis::sceneOutputKey(view_id)), logical);
    EXPECT_EQ(ring.registerSceneView(view_id), logical);
}

TEST(OutputSlotRing, SceneColumnsKeepIndependentLatestGenerationAndImages) {
    OutputSlotRing ring;
    const auto left = ring.registerSceneView(101);
    const auto right = ring.registerSceneView(202);

    ring.slotAt(left, 0) = makeSlot(0x10);
    ring.slotAt(right, 2) = makeSlot(0x20);
    ring.markLatest(left, 0);
    ring.markLatest(right, 2);
    EXPECT_EQ(ring.bumpGeneration(left), 1u);
    EXPECT_EQ(ring.bumpGeneration(right), 1u);
    EXPECT_EQ(ring.bumpGeneration(right), 2u);

    EXPECT_EQ(ring.latestSlot(left).image.image, fakeImage(0x10));
    EXPECT_EQ(ring.latestSlot(right).image.image, fakeImage(0x20));
    EXPECT_EQ(ring.generation(left), 1u);
    EXPECT_EQ(ring.generation(right), 2u);
}

TEST(OutputSlotRing, RetiringColumnDoesNotMoveOrAliasLiveNeighbor) {
    OutputSlotRing ring;
    const auto retired = ring.registerSceneView(303);
    const auto live = ring.registerSceneView(404);
    ring.slotAt(retired, 0) = makeSlot(0x30);
    ring.slotAt(live, 1) = makeSlot(0x40);

    std::size_t callback_count = 0;
    EXPECT_TRUE(ring.unregisterKey(lfs::vis::sceneOutputKey(303), [&](OutputImageSlot& slot) {
        ++callback_count;
    }));
    EXPECT_EQ(callback_count, OutputSlotRing::kFrameRingSize);
    EXPECT_FALSE(ring.findKey(lfs::vis::sceneOutputKey(303)));
    ASSERT_EQ(ring.findKey(lfs::vis::sceneOutputKey(404)), live);
    EXPECT_EQ(ring.slotAt(live, 1).image.image, fakeImage(0x40));
    EXPECT_EQ(ring.activeLogicalCount(), OutputSlotRing::kOutputSlotCount + 1);

    const auto replacement = ring.registerSceneView(505);
    EXPECT_EQ(replacement, retired);
    EXPECT_NE(replacement, live);
    EXPECT_EQ(ring.findKey(lfs::vis::sceneOutputKey(404)), live);
    EXPECT_EQ(ring.slotAt(replacement, 0).image.image, VK_NULL_HANDLE);
}

TEST(OutputSlotRing, UnknownOrContractlessRetirementFailsWithoutClearing) {
    OutputSlotRing ring;
    const auto logical = ring.registerSceneView(606);
    ring.slotAt(logical, 0) = makeSlot(0x60);

    EXPECT_FALSE(ring.unregisterKey(lfs::vis::sceneOutputKey(999), [&](OutputImageSlot&) {}));
    EXPECT_FALSE(ring.unregisterKey(lfs::vis::sceneOutputKey(606), {}));
    EXPECT_TRUE(ring.findKey(lfs::vis::sceneOutputKey(606)));
    EXPECT_EQ(ring.slotAt(logical, 0).image.image, fakeImage(0x60));
    EXPECT_THROW((void)ring.registerKey(lfs::vis::kInvalidViewOutputKey), std::invalid_argument);
    EXPECT_THROW((void)ring.registerSceneView(0), std::invalid_argument);

    const lfs::vis::ViewOutputKey unknown_kind{
        static_cast<lfs::vis::ViewOutputKey::Kind>(0xff), 1};
    EXPECT_FALSE(unknown_kind.valid());
    EXPECT_THROW((void)ring.registerKey(unknown_kind), std::invalid_argument);
}

TEST(OutputSlotRing, ReRegisterAndResetPreserveStableKeyState) {
    OutputSlotRing ring;
    const auto key = lfs::vis::sceneOutputKey(707);
    const auto logical = ring.registerKey(key);
    ring.slotAt(logical, 2) = makeSlot(0x70);
    ring.markLatest(logical, 2);
    EXPECT_EQ(ring.bumpGeneration(logical), 1u);

    ring.reset();
    EXPECT_EQ(ring.findKey(key), logical);
    EXPECT_EQ(ring.registerKey(key), logical);
    EXPECT_EQ(ring.latestRingSlot(logical), 0u);
    EXPECT_EQ(ring.generation(logical), 0u);
    EXPECT_EQ(ring.slotAt(logical, 2).image.image, VK_NULL_HANDLE);
}

TEST(OutputSlotRing, ThrowingRetirementCallbackKeepsKeyLive) {
    OutputSlotRing ring;
    const auto key = lfs::vis::sceneOutputKey(808);
    const auto logical = ring.registerKey(key);
    ring.slotAt(logical, 0) = makeSlot(0x80);

    EXPECT_THROW(
        (void)ring.retireKey(key, [](OutputImageSlot&) { throw std::runtime_error("retire failed"); }),
        std::runtime_error);
    EXPECT_EQ(ring.findKey(key), logical);
    EXPECT_EQ(ring.activeLogicalCount(), ring.logicalCount());
    EXPECT_EQ(ring.slotAt(logical, 0).image.image, fakeImage(0x80));
}
