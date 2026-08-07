/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Epic #1577 / #1574 — ReadbackTicketRing host bookkeeping (GPU-free).

#include "rendering/readback_ticket_ring.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

namespace {

    using lfs::vis::ReadbackTicketRing;

    ReadbackTicketRing::TicketMeta makeMeta(const std::uint64_t ticket,
                                            const std::size_t ring_cell,
                                            const std::uint64_t color_serial = 0,
                                            const std::uint64_t depth_serial = 0) {
        ReadbackTicketRing::TicketMeta meta{};
        meta.ticket_value = ticket;
        meta.logical_slot = 3; // Preview
        meta.ring_cell = ring_cell;
        meta.image_generation = color_serial;
        meta.completion_value = ticket * 10;
        meta.color_pool_serial = color_serial;
        meta.depth_pool_serial = depth_serial;
        meta.byte_count = 64;
        return meta;
    }

} // namespace

TEST(ReadbackTicketRing, TicketLifecycleSubmitPollDeliver) {
    ReadbackTicketRing ring;
    EXPECT_EQ(ring.outstandingCount(), 0u);
    ASSERT_TRUE(ring.tryAcquireCell().has_value());
    EXPECT_EQ(*ring.tryAcquireCell(), 0u);

    ring.markSubmitted(0, makeMeta(1, /*ring_cell=*/2, /*color=*/100));
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

TEST(ReadbackTicketRing, PoolPinPredicateHoldsAcquisitionSerial) {
    ReadbackTicketRing ring;
    ring.markSubmitted(0, makeMeta(1, 0, /*color=*/42, /*depth=*/0));
    ring.markSubmitted(1, makeMeta(2, 1, /*color=*/0, /*depth=*/99));

    EXPECT_TRUE(ring.hasOutstandingForPoolSerial(42));
    EXPECT_TRUE(ring.hasOutstandingForPoolSerial(99));
    EXPECT_FALSE(ring.hasOutstandingForPoolSerial(7));
    EXPECT_FALSE(ring.hasOutstandingForPoolSerial(0));

    ring.freeCell(0);
    EXPECT_FALSE(ring.hasOutstandingForPoolSerial(42));
    EXPECT_TRUE(ring.hasOutstandingForPoolSerial(99));
}

TEST(ReadbackTicketRing, PoolPinPredicateHoldsImageHandle) {
    ReadbackTicketRing ring;
    auto meta0 = makeMeta(1, 0, 42, 0);
    meta0.source_image = reinterpret_cast<VkImage>(0xA11);
    auto meta1 = makeMeta(2, 1, 0, 99);
    meta1.source_depth_image = reinterpret_cast<VkImage>(0xD22);
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
    ASSERT_NE(ring.findByTicket(1), nullptr);
    EXPECT_EQ(ring.findByTicket(1)->state, ReadbackTicketRing::State::Failed);
    EXPECT_EQ(ring.findByTicket(1)->error, "device idle on reset");

    ring.reset();
    EXPECT_EQ(ring.findByTicket(1), nullptr);
    EXPECT_EQ(ring.cell(0).state, ReadbackTicketRing::State::Free);
    EXPECT_EQ(ring.ringFullWaitCount(), 0u);
}
