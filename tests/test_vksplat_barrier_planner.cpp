/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Epic #1496: BufferBarrierPlanner GPU-free suite (spec §2.3 + §5 item 1).

#include "rendering/rasterizer/vulkan/src/barrier_planner.h"
#include "rendering/rasterizer/vulkan/src/gs_pipeline.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

// Free functions defined in gs_pipeline.cpp (external linkage; not in the header).
VkAccessFlags2 toAccessMask(VulkanGSPipeline::BarrierMask barrierMask);
VkPipelineStageFlags2 toStageMask(VulkanGSPipeline::BarrierMask barrierMask);

namespace {

    using namespace lfs::rendering::vulkan;

    constexpr uint32_t kQueueFamily = 7;

    template <typename Handle>
    [[nodiscard]] Handle fakeVkHandle(const std::uintptr_t value) {
        if constexpr (std::is_pointer_v<Handle>) {
            return reinterpret_cast<Handle>(value);
        } else {
            return static_cast<Handle>(value);
        }
    }

    [[nodiscard]] _VulkanBuffer makeBuffer(std::uintptr_t id, VkDeviceSize size = 4096) {
        _VulkanBuffer b;
        b.buffer = fakeVkHandle<VkBuffer>(id);
        b.allocSize = static_cast<size_t>(size);
        b.capacity = static_cast<size_t>(size);
        b.size = static_cast<size_t>(size);
        b.offset = 0;
        return b;
    }

    [[nodiscard]] Scope scopeFor(BufferUse use) {
        using BM = VulkanGSPipeline::BarrierMask;
        BM mask = BM::COMPUTE_SHADER_READ;
        switch (use) {
        case BufferUse::ComputeRead:
            mask = BM::COMPUTE_SHADER_READ;
            break;
        case BufferUse::ComputeWrite:
            mask = BM::COMPUTE_SHADER_WRITE;
            break;
        case BufferUse::ComputeReadWrite:
            mask = BM::COMPUTE_SHADER_READ_WRITE;
            break;
        case BufferUse::TransferRead:
            mask = BM::TRANSFER_READ;
            break;
        case BufferUse::TransferWrite:
            mask = BM::TRANSFER_WRITE;
            break;
        case BufferUse::IndirectRead:
            mask = BM::INDIRECT_DISPATCH_READ;
            break;
        case BufferUse::HostRead:
            mask = BM::HOST_READ;
            break;
        case BufferUse::ConditionalRead:
            mask = BM::CONDITIONAL_RENDERING_READ;
            break;
        }
        return Scope{toStageMask(mask), toAccessMask(mask)};
    }

    [[nodiscard]] Scope conservativeSrc() {
        using BM = VulkanGSPipeline::BarrierMask;
        return Scope{
            toStageMask(BM::TRANSFER_COMPUTE_SHADER_WRITE),
            toAccessMask(BM::TRANSFER_COMPUTE_SHADER_WRITE),
        };
    }

    [[nodiscard]] const VkBufferMemoryBarrier2* findBarrier(
        const std::vector<VkBufferMemoryBarrier2>& barriers,
        VkBuffer buffer) {
        for (const auto& b : barriers) {
            if (b.buffer == buffer) {
                return &b;
            }
        }
        return nullptr;
    }

    void expectBarrierShape(const VkBufferMemoryBarrier2& b, VkBuffer buffer) {
        EXPECT_EQ(b.sType, VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2);
        EXPECT_EQ(b.pNext, nullptr);
        EXPECT_EQ(b.buffer, buffer);
        EXPECT_EQ(b.offset, 0u);
        EXPECT_EQ(b.size, VK_WHOLE_SIZE);
        EXPECT_EQ(b.srcQueueFamilyIndex, kQueueFamily);
        EXPECT_EQ(b.dstQueueFamilyIndex, kQueueFamily);
    }

    void expectSrcDst(const VkBufferMemoryBarrier2& b, Scope src, Scope dst) {
        EXPECT_EQ(b.srcStageMask, src.stage);
        EXPECT_EQ(b.srcAccessMask, src.access);
        EXPECT_EQ(b.dstStageMask, dst.stage);
        EXPECT_EQ(b.dstAccessMask, dst.access);
    }

} // namespace

// ---------------------------------------------------------------------------
// §2.3 hazard table rows
// ---------------------------------------------------------------------------

// Catches treating untracked external buffers as empty state (no barrier) instead of conservative.
TEST(BarrierPlanner, UntrackedAnyAccessEmitsConservativeBarrier) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0xA001);
    // Not tracked.
    auto barriers = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeRead}});
    ASSERT_EQ(barriers.size(), 1u);
    expectBarrierShape(barriers[0], buf.buffer);
    expectSrcDst(barriers[0], conservativeSrc(), scopeFor(BufferUse::ComputeRead));
    EXPECT_EQ(planner.stats().conservative_fallbacks, 1u);
}

// Catches re-emitting a barrier when the write is already visible to the same read scope.
TEST(BarrierPlanner, WriterVisibleReadElidesBarrier) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0xA002);
    planner.track(buf.buffer);

    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeWrite}}).empty());
    auto first_read = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeRead}});
    ASSERT_EQ(first_read.size(), 1u);
    expectSrcDst(first_read[0], scopeFor(BufferUse::ComputeWrite), scopeFor(BufferUse::ComputeRead));

    auto second_read = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeRead}});
    EXPECT_TRUE(second_read.empty());
    EXPECT_GE(planner.stats().accesses_elided, 1u);
}

// Catches eliding a second read at a different stage after the first made only that stage visible.
TEST(BarrierPlanner, WriterNotVisibleReadEmitsBarrierFromWriter) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0xA003);
    planner.track(buf.buffer);

    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeWrite}}).empty());
    auto compute_read = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeRead}});
    ASSERT_EQ(compute_read.size(), 1u);

    // Transfer read is not ⊆ visible (only compute was made visible).
    auto transfer_read = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::TransferRead}});
    ASSERT_EQ(transfer_read.size(), 1u);
    expectBarrierShape(transfer_read[0], buf.buffer);
    expectSrcDst(transfer_read[0], scopeFor(BufferUse::ComputeWrite), scopeFor(BufferUse::TransferRead));
}

// Catches inventing a barrier for a first tracked read when no writer is outstanding.
TEST(BarrierPlanner, WriterNoneReadNoBarrier) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0xA004);
    planner.track(buf.buffer);

    auto barriers = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeRead}});
    EXPECT_TRUE(barriers.empty());
}

// Catches omitting the WAW/RAW/WAR barrier when writing after a prior writer.
TEST(BarrierPlanner, WriterThenWriteEmitsWawRawWarMergedBarrier) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0xA005);
    planner.track(buf.buffer);

    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::TransferWrite}}).empty());
    auto barriers = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeWrite}});
    ASSERT_EQ(barriers.size(), 1u);
    expectBarrierShape(barriers[0], buf.buffer);
    expectSrcDst(barriers[0], scopeFor(BufferUse::TransferWrite), scopeFor(BufferUse::ComputeWrite));
}

// Catches dropping WAR when a write follows readers with no prior writer, or setting src access ≠ NONE.
TEST(BarrierPlanner, WriterNoneReadersThenWriteEmitsWarExecutionOnly) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0xA006);
    planner.track(buf.buffer);

    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeRead}}).empty());
    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::TransferRead}}).empty());

    auto barriers = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeWrite}});
    ASSERT_EQ(barriers.size(), 1u);
    expectBarrierShape(barriers[0], buf.buffer);
    const Scope expected_src{
        scopeFor(BufferUse::ComputeRead).stage | scopeFor(BufferUse::TransferRead).stage,
        VK_ACCESS_2_NONE,
    };
    expectSrcDst(barriers[0], expected_src, scopeFor(BufferUse::ComputeWrite));
}

// Catches emitting a barrier on the first write after track when nothing is outstanding.
TEST(BarrierPlanner, WriterNoneReadersNoneWriteNoBarrier) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0xA007);
    planner.track(buf.buffer);

    auto barriers = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeWrite}});
    EXPECT_TRUE(barriers.empty());
}

// ---------------------------------------------------------------------------
// Conservative-unknown mask equivalence (pinned to legacy BarrierMask helpers)
// ---------------------------------------------------------------------------

// Catches hardcoding conservative masks that drift from toStageMask/toAccessMask(TRANSFER_COMPUTE_SHADER_WRITE).
TEST(BarrierPlanner, ConservativeMasksMatchTransferComputeShaderWrite) {
    const Scope expected = conservativeSrc();
    EXPECT_EQ(expected.stage,
              toStageMask(VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE));
    EXPECT_EQ(expected.access,
              toAccessMask(VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE));
    EXPECT_NE(expected.stage, VK_PIPELINE_STAGE_2_NONE);
    EXPECT_NE(expected.access, VK_ACCESS_2_NONE);

    // Sanity: legacy masks are ALL_TRANSFER|COMPUTE + TRANSFER_WRITE|SHADER_WRITE.
    EXPECT_EQ(expected.stage,
              VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    EXPECT_EQ(expected.access,
              VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0xB001);
    auto barriers = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::HostRead}});
    ASSERT_EQ(barriers.size(), 1u);
    EXPECT_EQ(barriers[0].srcStageMask, expected.stage);
    EXPECT_EQ(barriers[0].srcAccessMask, expected.access);
}

// ---------------------------------------------------------------------------
// WAR after N readers unions stages; src access NONE
// ---------------------------------------------------------------------------

// Catches keeping only the last reader stage in WAR src instead of OR-ing all readers.
TEST(BarrierPlanner, WarAfterNReadersUnionsAllReaderStagesSrcAccessNone) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0xC001);
    planner.track(buf.buffer);

    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeRead}}).empty());
    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::IndirectRead}}).empty());
    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::TransferRead}}).empty());

    auto barriers = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeReadWrite}});
    ASSERT_EQ(barriers.size(), 1u);
    const auto expected_stages =
        scopeFor(BufferUse::ComputeRead).stage |
        scopeFor(BufferUse::IndirectRead).stage |
        scopeFor(BufferUse::TransferRead).stage;
    EXPECT_EQ(barriers[0].srcStageMask, expected_stages);
    EXPECT_EQ(barriers[0].srcAccessMask, VK_ACCESS_2_NONE);
    expectSrcDst(barriers[0],
                 Scope{expected_stages, VK_ACCESS_2_NONE},
                 scopeFor(BufferUse::ComputeReadWrite));
}

// ---------------------------------------------------------------------------
// Visibility-set elision
// ---------------------------------------------------------------------------

// Catches failing to record visible stages after a read so a repeated same-scope read re-barriers.
TEST(BarrierPlanner, SecondReadSameScopeElides) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0xD001);
    planner.track(buf.buffer);
    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeWrite}}).empty());
    ASSERT_EQ(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeRead}}).size(), 1u);
    EXPECT_TRUE(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeRead}}).empty());
}

// Catches treating any subsequent read as elided regardless of stage membership in visible.
TEST(BarrierPlanner, ReadAtNewStageEmitsBarrier) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0xD002);
    planner.track(buf.buffer);
    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeWrite}}).empty());
    ASSERT_EQ(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeRead}}).size(), 1u);

    auto barriers = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::HostRead}});
    ASSERT_EQ(barriers.size(), 1u);
    expectSrcDst(barriers[0], scopeFor(BufferUse::ComputeWrite), scopeFor(BufferUse::HostRead));
}

// Catches ⊆-visible using only stage bits (or treating RW as elidable when stage ⊆ visible).
// Critique A2: after a narrow read visibility, a later access whose access bits are not ⊆
// visible_access must re-barrier. BufferUse has no pure-read with access ⊃ SHADER_READ at
// COMPUTE, so this uses the write path (ComputeReadWrite) after a ComputeRead visibility set:
// stage may look "visible" but the access is a superset and the access is a write.
TEST(BarrierPlanner, AccessSupersetReadAfterNarrowerVisibilityRebarriers) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0xD003);
    planner.track(buf.buffer);

    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::TransferWrite}}).empty());
    auto first = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeRead}});
    ASSERT_EQ(first.size(), 1u);
    // writer still TransferWrite; reader_stages = COMPUTE; visible = compute-read only.
    auto rw = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeReadWrite}});
    ASSERT_EQ(rw.size(), 1u);
    const Scope expected_src{
        scopeFor(BufferUse::TransferWrite).stage | scopeFor(BufferUse::ComputeRead).stage,
        scopeFor(BufferUse::TransferWrite).access,
    };
    expectSrcDst(rw[0], expected_src, scopeFor(BufferUse::ComputeReadWrite));
}

// ---------------------------------------------------------------------------
// ReadWrite merge / subsumption / duplicate merge
// ---------------------------------------------------------------------------

// Catches keeping separate R and W barriers (or wrong merge) when one plan() sees read+write.
TEST(BarrierPlanner, ReadWriteSubsumptionInSinglePlan) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0xE001);
    planner.track(buf.buffer);
    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::TransferWrite}}).empty());

    std::array accesses{
        DeclaredAccess{&buf, BufferUse::ComputeRead},
        DeclaredAccess{&buf, BufferUse::ComputeWrite},
    };
    auto barriers = planner.plan(accesses);
    ASSERT_EQ(barriers.size(), 1u);
    expectSrcDst(barriers[0], scopeFor(BufferUse::TransferWrite), scopeFor(BufferUse::ComputeReadWrite));
}

// Catches emitting one barrier per duplicate binding of the same buffer in one plan().
TEST(BarrierPlanner, DuplicateBufferInOnePlanMergesUses) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0xE002);
    planner.track(buf.buffer);
    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeWrite}}).empty());

    std::array accesses{
        DeclaredAccess{&buf, BufferUse::ComputeRead},
        DeclaredAccess{&buf, BufferUse::ComputeRead},
        DeclaredAccess{&buf, BufferUse::ComputeRead},
    };
    auto barriers = planner.plan(accesses);
    ASSERT_EQ(barriers.size(), 1u);
    expectSrcDst(barriers[0], scopeFor(BufferUse::ComputeWrite), scopeFor(BufferUse::ComputeRead));
}

// ---------------------------------------------------------------------------
// fill(TransferWrite) → compute-read is TRANSFER-only src
// ---------------------------------------------------------------------------

// Catches using TRANSFER_COMPUTE_SHADER_WRITE as src after a pure transfer fill producer.
TEST(BarrierPlanner, FillTransferWriteThenComputeReadIsTransferOnlySrc) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0xF001);
    planner.track(buf.buffer);

    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::TransferWrite}}).empty());
    auto barriers = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeRead}});
    ASSERT_EQ(barriers.size(), 1u);
    expectSrcDst(barriers[0], scopeFor(BufferUse::TransferWrite), scopeFor(BufferUse::ComputeRead));
    EXPECT_EQ(barriers[0].srcStageMask, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT);
    EXPECT_EQ(barriers[0].srcAccessMask, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    EXPECT_EQ((barriers[0].srcStageMask & VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT), 0u);
}

// ---------------------------------------------------------------------------
// forget / invalidate / null handle
// ---------------------------------------------------------------------------

// Catches leaving state after forget so a re-used handle inherits the previous writer's scope.
TEST(BarrierPlanner, ForgetThenSameHandleReuseIsConservative) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0x1001);
    planner.track(buf.buffer);
    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeWrite}}).empty());
    planner.forget(buf.buffer);

    // Same handle value, untracked again (as if destroy + recreate without track, or track later).
    auto barriers = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeRead}});
    ASSERT_EQ(barriers.size(), 1u);
    expectSrcDst(barriers[0], conservativeSrc(), scopeFor(BufferUse::ComputeRead));
}

// Catches forgetting to mark invalidate so a legacy barrier path leaves planner state hot.
TEST(BarrierPlanner, InvalidateMakesNextAccessConservative) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0x1002);
    planner.track(buf.buffer);
    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeWrite}}).empty());
    planner.invalidate(buf.buffer);

    auto barriers = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeRead}});
    ASSERT_EQ(barriers.size(), 1u);
    expectSrcDst(barriers[0], conservativeSrc(), scopeFor(BufferUse::ComputeRead));
}

// Catches conservative READ after-state dropping the unknown writer (writer=NONE), so a later
// read at a new stage elides instead of chaining from TRANSFER_COMPUTE_SHADER_WRITE.
TEST(BarrierPlanner, InvalidateReadThenNewStageReadStillBarriers) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0x1004);
    planner.track(buf.buffer);
    planner.invalidate(buf.buffer);

    auto first = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeRead}});
    ASSERT_EQ(first.size(), 1u);
    expectSrcDst(first[0], conservativeSrc(), scopeFor(BufferUse::ComputeRead));

    auto second = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::IndirectRead}});
    ASSERT_EQ(second.size(), 1u);
    expectSrcDst(second[0], conservativeSrc(), scopeFor(BufferUse::IndirectRead));
    EXPECT_EQ(second[0].srcStageMask,
              toStageMask(VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE));
    EXPECT_EQ(second[0].srcAccessMask,
              toAccessMask(VulkanGSPipeline::TRANSFER_COMPUTE_SHADER_WRITE));
    EXPECT_EQ(second[0].dstStageMask, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
    EXPECT_EQ(second[0].dstAccessMask, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
}

// Catches over-fixing the conservative-read after-state so same-scope follow-up reads re-barrier.
TEST(BarrierPlanner, ReadAfterConservativeReadSameScopeElides) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0x1005);
    planner.track(buf.buffer);
    planner.invalidate(buf.buffer);

    ASSERT_EQ(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeRead}}).size(), 1u);
    // Second same-scope read must elide (visible already holds COMPUTE_READ).
    EXPECT_TRUE(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeRead}}).empty());
}

// Catches crashing or emitting a barrier for VK_NULL_HANDLE (legacy parity: skip).
TEST(BarrierPlanner, NullHandleSkipped) {
    BufferBarrierPlanner planner(kQueueFamily);
    _VulkanBuffer null_buf;
    null_buf.buffer = VK_NULL_HANDLE;
    auto live = makeBuffer(0x1003);
    planner.track(live.buffer);

    std::array accesses{
        DeclaredAccess{&null_buf, BufferUse::ComputeWrite},
        DeclaredAccess{&live, BufferUse::ComputeWrite},
    };
    auto barriers = planner.plan(accesses);
    EXPECT_TRUE(barriers.empty()); // first write on live after track: no barrier; null skipped
    EXPECT_EQ(findBarrier(barriers, VK_NULL_HANDLE), nullptr);
}

// ---------------------------------------------------------------------------
// Emitted struct shape (G10)
// ---------------------------------------------------------------------------

// Catches using view offset/size instead of WHOLE_SIZE, or wrong queue-family indices.
TEST(BarrierPlanner, EmittedStructsUseOffsetZeroWholeSizeAndQueueFamily) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0x1101, 8192);
    buf.offset = 128; // views may carry offsets; planner still emits WHOLE_SIZE
    buf.size = 256;
    // Untracked → conservative barrier for shape snapshot.
    auto barriers = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeReadWrite}});
    ASSERT_EQ(barriers.size(), 1u);
    expectBarrierShape(barriers[0], buf.buffer);
}

// ---------------------------------------------------------------------------
// External parent + region views collapse to parent state
// ---------------------------------------------------------------------------

// Catches keying state on view identity instead of bare VkBuffer so sibling views miss WAR/RAW.
TEST(BarrierPlanner, TrackedParentTwoRegionViewsShareState) {
    BufferBarrierPlanner planner(kQueueFamily);
    const VkBuffer parent = fakeVkHandle<VkBuffer>(0x1201);
    planner.track(parent);

    _VulkanBuffer view_a;
    view_a.buffer = parent;
    view_a.offset = 0;
    view_a.size = 1024;
    view_a.capacity = 1024;
    view_a.allocSize = 4096;

    _VulkanBuffer view_b;
    view_b.buffer = parent;
    view_b.offset = 1024;
    view_b.size = 1024;
    view_b.capacity = 1024;
    view_b.allocSize = 4096;

    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&view_a, BufferUse::ComputeWrite}}).empty());
    auto barriers = planner.plan(std::array{DeclaredAccess{&view_b, BufferUse::ComputeRead}});
    ASSERT_EQ(barriers.size(), 1u);
    EXPECT_EQ(barriers[0].buffer, parent);
    expectSrcDst(barriers[0], scopeFor(BufferUse::ComputeWrite), scopeFor(BufferUse::ComputeRead));
}

// Catches leaving parent state after untrack so a re-import without track stays exact (must be conservative).
TEST(BarrierPlanner, ForgetExternalParentThenAccessIsConservative) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto parent = makeBuffer(0x1202);
    planner.track(parent.buffer);
    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&parent, BufferUse::ComputeWrite}}).empty());
    planner.forget(parent.buffer);

    auto barriers = planner.plan(std::array{DeclaredAccess{&parent, BufferUse::ComputeRead}});
    ASSERT_EQ(barriers.size(), 1u);
    expectSrcDst(barriers[0], conservativeSrc(), scopeFor(BufferUse::ComputeRead));
}

// ---------------------------------------------------------------------------
// onBatchBegin reset (§2.4 / G2 / G6)
// ---------------------------------------------------------------------------

// Catches batch reset that fails to set visible to the reuse-barrier dst so compute re-barriers.
TEST(BarrierPlanner, OnBatchBeginComputeReadElides) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0x1301);
    planner.track(buf.buffer);
    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeWrite}}).empty());

    planner.onBatchBegin();
    auto barriers = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeRead}});
    EXPECT_TRUE(barriers.empty());
}

// Catches batch visible including CONDITIONAL_RENDERING so ConditionalRead wrongly elides.
TEST(BarrierPlanner, OnBatchBeginConditionalReadDoesNotElide) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0x1302);
    planner.track(buf.buffer);
    planner.onBatchBegin();

    auto barriers = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ConditionalRead}});
    ASSERT_EQ(barriers.size(), 1u);
    // src = conservative writer from batch reset; dst = conditional scope
    expectSrcDst(barriers[0], conservativeSrc(), scopeFor(BufferUse::ConditionalRead));
}

// Catches batch visible including HOST so HostRead wrongly elides.
TEST(BarrierPlanner, OnBatchBeginHostReadDoesNotElide) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0x1303);
    planner.track(buf.buffer);
    planner.onBatchBegin();

    auto barriers = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::HostRead}});
    ASSERT_EQ(barriers.size(), 1u);
    expectSrcDst(barriers[0], conservativeSrc(), scopeFor(BufferUse::HostRead));
}

// ---------------------------------------------------------------------------
// Indirect / multi-phase / HostRead (G1 / G3 / G4)
// ---------------------------------------------------------------------------

// Catches missing DRAW_INDIRECT as dst when compute produces indirect args.
TEST(BarrierPlanner, ComputeWriteThenIndirectReadBarrier) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0x1401);
    planner.track(buf.buffer);
    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeWrite}}).empty());

    auto barriers = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::IndirectRead}});
    ASSERT_EQ(barriers.size(), 1u);
    expectSrcDst(barriers[0], scopeFor(BufferUse::ComputeWrite), scopeFor(BufferUse::IndirectRead));
    EXPECT_EQ(barriers[0].dstStageMask, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
    EXPECT_EQ(barriers[0].dstAccessMask, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
}

// Catches failing to barrier between successive ComputeReadWrite phases (cumsum-style ping-pong).
TEST(BarrierPlanner, CumsumStyleMultiPhaseReadWritePingPong) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0x1402);
    planner.track(buf.buffer);

    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeReadWrite}}).empty());
    auto phase2 = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeReadWrite}});
    ASSERT_EQ(phase2.size(), 1u);
    expectSrcDst(phase2[0], scopeFor(BufferUse::ComputeReadWrite), scopeFor(BufferUse::ComputeReadWrite));

    auto phase3 = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeReadWrite}});
    ASSERT_EQ(phase3.size(), 1u);
    expectSrcDst(phase3[0], scopeFor(BufferUse::ComputeReadWrite), scopeFor(BufferUse::ComputeReadWrite));
}

// Catches wrong HostRead stage/access, and documents that the barrier alone is not host coherence.
TEST(BarrierPlanner, HostReadBarrierShapeNoCoherenceWithoutFence) {
    // Host coherence still requires the existing fence/timeline wait (endCommandBatch); this
    // test only pins the barrier shape. A green that "promotes" HostRead to imply coherence
    // without a wait would be a separate contract violation outside this suite.
    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0x1403);
    planner.track(buf.buffer);
    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::TransferWrite}}).empty());

    auto barriers = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::HostRead}});
    ASSERT_EQ(barriers.size(), 1u);
    expectSrcDst(barriers[0], scopeFor(BufferUse::TransferWrite), scopeFor(BufferUse::HostRead));
    EXPECT_EQ(barriers[0].dstStageMask, VK_PIPELINE_STAGE_2_HOST_BIT);
    EXPECT_EQ(barriers[0].dstAccessMask, VK_ACCESS_2_HOST_READ_BIT);
}

// ---------------------------------------------------------------------------
// Stats counters
// ---------------------------------------------------------------------------

// Catches never incrementing stats, or counting elisions as emissions (or the reverse).
TEST(BarrierPlanner, StatsCountersTrackEmittedElidedAndConservative) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto tracked = makeBuffer(0x1501);
    auto untracked = makeBuffer(0x1502);
    planner.track(tracked.buffer);

    // First write: elide (no outstanding).
    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&tracked, BufferUse::ComputeWrite}}).empty());
    // Read: emit one RAW barrier.
    ASSERT_EQ(planner.plan(std::array{DeclaredAccess{&tracked, BufferUse::ComputeRead}}).size(), 1u);
    // Same-scope read: elide.
    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&tracked, BufferUse::ComputeRead}}).empty());
    // Untracked: conservative fallback + emit.
    ASSERT_EQ(planner.plan(std::array{DeclaredAccess{&untracked, BufferUse::ComputeRead}}).size(), 1u);

    const auto s = planner.stats();
    EXPECT_EQ(s.barriers_emitted, 2u);
    EXPECT_GE(s.accesses_elided, 2u);
    EXPECT_EQ(s.conservative_fallbacks, 1u);
}

// Catches reset() leaving stats or tracked state so later plans inherit prior counters/handles.
TEST(BarrierPlanner, ResetClearsStateAndStats) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0x1503);
    planner.track(buf.buffer);
    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeWrite}}).empty());
    ASSERT_EQ(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeRead}}).size(), 1u);

    planner.reset();
    const auto s = planner.stats();
    EXPECT_EQ(s.barriers_emitted, 0u);
    EXPECT_EQ(s.accesses_elided, 0u);
    EXPECT_EQ(s.conservative_fallbacks, 0u);

    // After reset, buffer is untracked → conservative.
    auto barriers = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeRead}});
    ASSERT_EQ(barriers.size(), 1u);
    expectSrcDst(barriers[0], conservativeSrc(), scopeFor(BufferUse::ComputeRead));
}

// ---------------------------------------------------------------------------
// Writer+readers then write: src unions writer stage with reader_stages
// ---------------------------------------------------------------------------

// Catches WAW-only src that drops intervening readers (WAR) when both writer and readers exist.
TEST(BarrierPlanner, WriterAndReadersThenWriteUnionsSrcStages) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto buf = makeBuffer(0x1601);
    planner.track(buf.buffer);

    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::TransferWrite}}).empty());
    ASSERT_EQ(planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeRead}}).size(), 1u);
    // reader_stages has compute; writer is still TransferWrite; visible has compute-read.

    auto barriers = planner.plan(std::array{DeclaredAccess{&buf, BufferUse::ComputeWrite}});
    ASSERT_EQ(barriers.size(), 1u);
    const Scope expected_src{
        scopeFor(BufferUse::TransferWrite).stage | scopeFor(BufferUse::ComputeRead).stage,
        scopeFor(BufferUse::TransferWrite).access,
    };
    expectSrcDst(barriers[0], expected_src, scopeFor(BufferUse::ComputeWrite));
}

// ---------------------------------------------------------------------------
// Simultaneous multi-buffer plan
// ---------------------------------------------------------------------------

// Catches global single-barrier or wrong pairing when one plan() covers multiple buffers.
TEST(BarrierPlanner, MultipleDistinctBuffersInOnePlan) {
    BufferBarrierPlanner planner(kQueueFamily);
    auto a = makeBuffer(0x1701);
    auto b = makeBuffer(0x1702);
    planner.track(a.buffer);
    planner.track(b.buffer);

    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&a, BufferUse::ComputeWrite}}).empty());
    ASSERT_TRUE(planner.plan(std::array{DeclaredAccess{&b, BufferUse::TransferWrite}}).empty());

    std::array accesses{
        DeclaredAccess{&a, BufferUse::ComputeRead},
        DeclaredAccess{&b, BufferUse::ComputeRead},
    };
    auto barriers = planner.plan(accesses);
    ASSERT_EQ(barriers.size(), 2u);
    const auto* ba = findBarrier(barriers, a.buffer);
    const auto* bb = findBarrier(barriers, b.buffer);
    ASSERT_NE(ba, nullptr);
    ASSERT_NE(bb, nullptr);
    expectSrcDst(*ba, scopeFor(BufferUse::ComputeWrite), scopeFor(BufferUse::ComputeRead));
    expectSrcDst(*bb, scopeFor(BufferUse::TransferWrite), scopeFor(BufferUse::ComputeRead));
}
