/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Epic #1577 / #1574 — ReadbackTicketRing host bookkeeping (GPU-free).

#include "rendering/readback_ticket_ring.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <unordered_set>

namespace {

    using lfs::vis::ReadbackTicketRing;

    ReadbackTicketRing::TicketMeta makeMeta(const std::uint64_t ticket,
                                            const std::size_t ring_cell,
                                            VkImage source_image = VK_NULL_HANDLE,
                                            VkImage source_depth = VK_NULL_HANDLE) {
        ReadbackTicketRing::TicketMeta meta{};
        meta.ticket_value = ticket;
        meta.ring_cell = ring_cell;
        meta.source_image = source_image;
        meta.source_depth_image = source_depth;
        meta.byte_count = 64;
        meta.delivery = ReadbackTicketRing::DeliveryKind::DepthFloatPlane;
        return meta;
    }

} // namespace

TEST(ReadbackTicketRing, TicketLifecycleSubmitPollDeliver) {
    ReadbackTicketRing ring;
    EXPECT_EQ(ring.outstandingCount(), 0u);
    ASSERT_TRUE(ring.tryAcquireCell().has_value());
    EXPECT_EQ(*ring.tryAcquireCell(), 0u);

    ring.markSubmitted(0, makeMeta(1, /*ring_cell=*/2));
    EXPECT_EQ(ring.outstandingCount(), 1u);
    ASSERT_NE(ring.findByTicket(1), nullptr);
    EXPECT_EQ(ring.findByTicket(1)->state, ReadbackTicketRing::State::Outstanding);
    EXPECT_EQ(ring.findByTicket(99), nullptr);

    // Deliver: free the cell (renderer does invalidate+memcpy first).
    ring.freeCell(0);
    EXPECT_EQ(ring.outstandingCount(), 0u);
    EXPECT_EQ(ring.findByTicket(1), nullptr);
    EXPECT_EQ(ring.cell(0).state, ReadbackTicketRing::State::Free);
}

TEST(ReadbackTicketRing, RingFullBoundedWaitOldest) {
    ReadbackTicketRing ring;
    ring.markSubmitted(0, makeMeta(10, 0));
    ring.markSubmitted(1, makeMeta(20, 1));
    ring.markSubmitted(2, makeMeta(30, 2));
    EXPECT_FALSE(ring.tryAcquireCell().has_value());

    const auto oldest = ring.oldestOutstandingCell();
    ASSERT_TRUE(oldest.has_value());
    EXPECT_EQ(*oldest, 0u);
    EXPECT_EQ(ring.cell(*oldest).ticket_value, 10u);

    ring.noteRingFullWait();
    EXPECT_EQ(ring.ringFullWaitCount(), 1u);

    ring.freeCell(0);
    ASSERT_TRUE(ring.tryAcquireCell().has_value());
    EXPECT_EQ(*ring.tryAcquireCell(), 0u);
}

TEST(ReadbackTicketRing, CellPinRegistryBlocksReusePredicate) {
    ReadbackTicketRing ring;
    // Two tickets can source the same frame-ring cell over time; pin uses max.
    ring.markSubmitted(0, makeMeta(5, /*ring_cell=*/1));
    ring.markSubmitted(1, makeMeta(8, /*ring_cell=*/1));
    ring.markSubmitted(2, makeMeta(3, /*ring_cell=*/0));

    EXPECT_EQ(ring.maxTicketForFrameRingCell(1), 8u);
    EXPECT_EQ(ring.maxTicketForFrameRingCell(0), 3u);
    EXPECT_EQ(ring.maxTicketForFrameRingCell(2), 0u);

    ring.freeCell(1); // drop ticket 8
    EXPECT_EQ(ring.maxTicketForFrameRingCell(1), 5u);

    ring.noteCellPinWait();
    EXPECT_EQ(ring.cellPinWaitCount(), 1u);
}

TEST(ReadbackTicketRing, PoolPinPredicateHoldsImageHandle) {
    ReadbackTicketRing ring;
    auto meta0 = makeMeta(1, 0, reinterpret_cast<VkImage>(0xA11));
    auto meta1 = makeMeta(2, 1, VK_NULL_HANDLE, reinterpret_cast<VkImage>(0xD22));
    ring.markSubmitted(0, meta0);
    ring.markSubmitted(1, meta1);

    EXPECT_TRUE(ring.hasOutstandingForImage(reinterpret_cast<VkImage>(0xA11)));
    EXPECT_TRUE(ring.hasOutstandingForImage(reinterpret_cast<VkImage>(0xD22)));
    EXPECT_FALSE(ring.hasOutstandingForImage(reinterpret_cast<VkImage>(0xBEEF)));
    EXPECT_FALSE(ring.hasOutstandingForImage(VK_NULL_HANDLE));

    ring.freeCell(0);
    EXPECT_FALSE(ring.hasOutstandingForImage(reinterpret_cast<VkImage>(0xA11)));
    EXPECT_TRUE(ring.hasOutstandingForImage(reinterpret_cast<VkImage>(0xD22)));
}

TEST(ReadbackTicketRing, ResetWithOutstandingTicketsFailsThem) {
    ReadbackTicketRing ring;
    ring.markSubmitted(0, makeMeta(1, 0));
    ring.markSubmitted(1, makeMeta(2, 1));
    EXPECT_EQ(ring.outstandingCount(), 2u);

    const std::size_t failed = ring.failAllOutstanding("device idle on reset");
    EXPECT_EQ(failed, 2u);
    EXPECT_EQ(ring.outstandingCount(), 0u);
    EXPECT_EQ(ring.failedCount(), 2u);
    ASSERT_NE(ring.findByTicket(1), nullptr);
    EXPECT_EQ(ring.findByTicket(1)->state, ReadbackTicketRing::State::Failed);
    EXPECT_EQ(ring.findByTicket(1)->error, "device idle on reset");

    ring.reset();
    EXPECT_EQ(ring.findByTicket(1), nullptr);
    EXPECT_EQ(ring.cell(0).state, ReadbackTicketRing::State::Free);
    EXPECT_EQ(ring.ringFullWaitCount(), 0u);
}

// Failed cells keep pins until freeCell; reclaimFailedIf frees them when complete.
TEST(ReadbackTicketRing, FailedCellKeepsPinsUntilReclaim) {
    ReadbackTicketRing ring;
    auto meta = makeMeta(7, /*ring_cell=*/1, reinterpret_cast<VkImage>(0xC01));
    ring.markSubmitted(0, meta);
    ring.markFailed(0, "AMB-4 quarantine");

    EXPECT_EQ(ring.outstandingCount(), 0u);
    EXPECT_EQ(ring.failedCount(), 1u);
    // Pins still honor Failed.
    EXPECT_EQ(ring.maxTicketForFrameRingCell(1), 7u);
    EXPECT_TRUE(ring.hasOutstandingForImage(reinterpret_cast<VkImage>(0xC01)));
    // Cells 1–2 are still free; fill them to prove Failed occupies its slot.
    ring.markSubmitted(1, makeMeta(8, 0));
    ring.markSubmitted(2, makeMeta(9, 0));
    EXPECT_FALSE(ring.tryAcquireCell().has_value());

    // Incomplete reclaim does nothing.
    const std::size_t not_yet = ring.reclaimFailedIf(
        [](std::uint64_t, void*) { return false; }, nullptr);
    EXPECT_EQ(not_yet, 0u);
    EXPECT_EQ(ring.failedCount(), 1u);
    EXPECT_TRUE(ring.hasOutstandingForImage(reinterpret_cast<VkImage>(0xC01)));

    // Complete reclaim frees Failed and drops pins.
    std::unordered_set<std::uint64_t> done{7};
    const std::size_t freed = ring.reclaimFailedIf(
        [](const std::uint64_t ticket, void* ctx) {
            return static_cast<const std::unordered_set<std::uint64_t>*>(ctx)->count(ticket) > 0;
        },
        &done);
    EXPECT_EQ(freed, 1u);
    EXPECT_EQ(ring.failedCount(), 0u);
    EXPECT_FALSE(ring.hasOutstandingForImage(reinterpret_cast<VkImage>(0xC01)));
    EXPECT_EQ(ring.maxTicketForFrameRingCell(1), 0u);
    ASSERT_TRUE(ring.tryAcquireCell().has_value());
    EXPECT_EQ(*ring.tryAcquireCell(), 0u);
}

// oldestActiveCell includes failed cells so a full ring can wait for them.
TEST(ReadbackTicketRing, OldestActiveIncludesFailed) {
    ReadbackTicketRing ring;
    ring.markSubmitted(0, makeMeta(10, 0));
    ring.markSubmitted(1, makeMeta(20, 1));
    ring.markSubmitted(2, makeMeta(30, 2));
    ring.markFailed(0, "wait failed");
    ring.freeCell(1); // only 0 Failed + 2 Outstanding

    const auto oldest_out = ring.oldestOutstandingCell();
    ASSERT_TRUE(oldest_out.has_value());
    EXPECT_EQ(*oldest_out, 2u);

    const auto oldest_active = ring.oldestActiveCell();
    ASSERT_TRUE(oldest_active.has_value());
    EXPECT_EQ(*oldest_active, 0u); // Failed ticket 10 is older than Outstanding 30
    EXPECT_EQ(ring.cell(*oldest_active).state, ReadbackTicketRing::State::Failed);
}

// hasOutstandingForImage is an unlocked query; the caller serializes access.
// Drain predicates must call it only while holding the mutex or using a snapshot.
TEST(ReadbackTicketRing, PinPredicateIsUnlockedQuery) {
    ReadbackTicketRing ring;
    auto meta = makeMeta(1, 0, reinterpret_cast<VkImage>(0xB1D0));
    ring.markSubmitted(0, meta);
    // Pure function: no internal locking; safe to call under already-held mutex.
    EXPECT_TRUE(ring.hasOutstandingForImage(reinterpret_cast<VkImage>(0xB1D0)));
    ring.markFailed(0, "fail");
    EXPECT_TRUE(ring.hasOutstandingForImage(reinterpret_cast<VkImage>(0xB1D0)));
    ring.freeCell(0);
    EXPECT_FALSE(ring.hasOutstandingForImage(reinterpret_cast<VkImage>(0xB1D0)));
}
