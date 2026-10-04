/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "rendering/generic_output_slot_ring.hpp"
#include "rendering/generic_readback_ticket_ring.hpp"
#include "rendering/gpu_resource_pool.hpp"
#include <gtest/gtest.h>

namespace {
    struct Key {
        int format, width, height;
        bool operator==(const Key&) const = default;
    };
    struct Payload {
        uint64_t completion_value = 0;
        int data = 0;
    };

    TEST(GpuBookkeeping, OutputRingHasNoGraphicsPayloadDependency) {
        lfs::vis::BasicOutputSlotRing<Payload> ring;
        const lfs::vis::RenderTargetId target{7};
        auto first = ring.acquire(target);
        ring.slotAt(target, first).data = 19;
        ring.publishCompletion(first, 9);
        ring.markLatest(target, first);
        EXPECT_EQ(ring.latestSlot(target).data, 19);
        auto result = ring.waitUntilReusable(first, "test", [](uint64_t) { return false; }, [](uint64_t value) -> lfs::Status { EXPECT_EQ(value, 9); return {}; });
        ASSERT_TRUE(result);
        EXPECT_EQ(ring.ringCompletionValue(first), 0);
        EXPECT_TRUE(ring.releaseRenderTarget(target, {}));
        EXPECT_TRUE(ring.released(target));
    }

    TEST(GpuBookkeeping, TicketPinsUseBackendSuppliedKeys) {
        lfs::vis::BasicReadbackTicketRing<uint64_t> ring;
        using Ring = decltype(ring);
        ring.markSubmitted(0, {.ticket_value = 12, .ring_cell = 4, .source_image = 37});
        EXPECT_TRUE(ring.hasOutstandingForImage(37));
        EXPECT_FALSE(ring.hasOutstandingForImage(38));
        ring.markFailed(0, "cancelled delivery");
        EXPECT_EQ(ring.cell(0).state, Ring::State::Failed);
        EXPECT_EQ(ring.maxTicketForFrameRingCell(4), 12);
        EXPECT_TRUE(ring.hasOutstandingForImage(37));
        ring.freeCell(0);
        EXPECT_FALSE(ring.hasOutstandingForImage(37));
    }

    TEST(GpuBookkeeping, PoolUsesBackendSuppliedAllocationKey) {
        lfs::vis::BasicGpuResourcePool<Payload, Key> pool;
        const auto created = pool.registerCreated({1, 31, 17}, Payload{0, 42});
        EXPECT_EQ(created.payload->data, 42);
        EXPECT_FALSE(pool.acquire({2, 31, 17}));
    }
} // namespace
