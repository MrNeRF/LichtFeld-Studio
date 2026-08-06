/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Epic #1496 §5.3 — generation-keyed image barrier tracker + reader accumulation.

#include "window/vulkan_image_barrier_tracker.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>

namespace {

    using lfs::vis::VulkanImageBarrierTracker;

    VkImage fakeImage(std::uintptr_t id) {
        return reinterpret_cast<VkImage>(id);
    }

    VkCommandBuffer fakeCmd(std::uintptr_t id = 0xBEEF) {
        return reinterpret_cast<VkCommandBuffer>(id);
    }

    thread_local std::vector<VkImageMemoryBarrier2> g_captured_barriers;

    VKAPI_ATTR void VKAPI_CALL capturePipelineBarrier2(VkCommandBuffer /*commandBuffer*/,
                                                       const VkDependencyInfo* pDependencyInfo) {
        if (pDependencyInfo == nullptr) {
            return;
        }
        for (std::uint32_t i = 0; i < pDependencyInfo->imageMemoryBarrierCount; ++i) {
            g_captured_barriers.push_back(pDependencyInfo->pImageMemoryBarriers[i]);
        }
    }

    void installCapture(VulkanImageBarrierTracker& tracker) {
        g_captured_barriers.clear();
        tracker.setPipelineBarrierEmitter(capturePipelineBarrier2);
    }

} // namespace

// Catches: bare-handle map reusing prior ImageState across a recreated VkImage with a new
// generation (issue #1478 handle-reuse inheritance).
TEST(VulkanImageTracker, ReRegisterNewGenerationNoStateInheritance) {
    VulkanImageBarrierTracker tracker;
    installCapture(tracker);
    const VkImage image = fakeImage(0x1001);

    tracker.registerImage(image, /*generation=*/1, VK_IMAGE_ASPECT_COLOR_BIT,
                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    // Recreate same handle under a new generation without an intervening forget.
    tracker.registerImage(image, /*generation=*/2, VK_IMAGE_ASPECT_COLOR_BIT,
                          VK_IMAGE_LAYOUT_UNDEFINED);

    tracker.transitionImage(fakeCmd(), image, /*generation=*/2, VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    ASSERT_EQ(g_captured_barriers.size(), 1u);
    EXPECT_EQ(g_captured_barriers[0].oldLayout, VK_IMAGE_LAYOUT_UNDEFINED);
    EXPECT_NE(g_captured_barriers[0].oldLayout, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
}

// Catches: forgetImage keyed only on VkImage, so a stale owner forget evicts the new entry.
TEST(VulkanImageTracker, StaleForgetDoesNotEvictCurrentEntry) {
    VulkanImageBarrierTracker tracker;
    const VkImage image = fakeImage(0x1002);

    tracker.registerImage(image, /*generation=*/2, VK_IMAGE_ASPECT_COLOR_BIT,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    tracker.forgetImage(image, /*generation=*/1); // stale

    EXPECT_EQ(tracker.imageLayout(image, /*generation=*/2, VK_IMAGE_LAYOUT_UNDEFINED),
              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

// Catches: generation-mismatched transition still mutating the live entry (should be untracked).
// Release path is the asserted behavior; debug LFS_VK_DEBUG_ASSERT is noreturn on mismatch.
TEST(VulkanImageTracker, GenerationMismatchTransitionIsUntracked) {
#if !defined(NDEBUG)
    GTEST_SKIP() << "Debug builds assert on generation mismatch (LFS_VK_DEBUG_ASSERT); release path covered here.";
#else
    VulkanImageBarrierTracker tracker;
    installCapture(tracker);
    const VkImage image = fakeImage(0x1003);

    tracker.registerImage(image, /*generation=*/2, VK_IMAGE_ASPECT_COLOR_BIT,
                          VK_IMAGE_LAYOUT_UNDEFINED);
    // Mismatched generation: must not advance the gen-2 entry.
    tracker.transitionImage(fakeCmd(), image, /*generation=*/1, VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    EXPECT_EQ(tracker.imageLayout(image, /*generation=*/2, VK_IMAGE_LAYOUT_PREINITIALIZED),
              VK_IMAGE_LAYOUT_UNDEFINED);
#endif
}

// Catches: same-layout early-return dropping additional reader stages, so a later write barrier
// only sees the first reader instead of the union (sub-task 4 sharp edge).
TEST(VulkanImageTracker, ReaderAccumulationSameLayoutThenWriteUnionsReaders) {
    VulkanImageBarrierTracker tracker;
    installCapture(tracker);
    const VkImage image = fakeImage(0x1004);
    const VkCommandBuffer cmd = fakeCmd();

    tracker.registerImage(image, /*generation=*/1, VK_IMAGE_ASPECT_COLOR_BIT,
                          VK_IMAGE_LAYOUT_UNDEFINED);

    // First reader: layout-derived transition into SHADER_READ_ONLY.
    tracker.transitionImage(cmd, image, /*generation=*/1, VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    g_captured_barriers.clear();

    // Second reader at the SAME layout via explicit-scope overload (early-return accumulates).
    constexpr VulkanImageBarrierTracker::AccessScope compute_read{
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_READ_BIT,
    };
    tracker.transitionImage(cmd, image, /*generation=*/1, VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            /*source=*/{},
                            compute_read);
    EXPECT_TRUE(g_captured_barriers.empty()) << "same-layout second reader must not emit a barrier";

    // Write transition: src must be the UNION of both readers.
    g_captured_barriers.clear();
    tracker.transitionImage(cmd, image, /*generation=*/1, VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    ASSERT_EQ(g_captured_barriers.size(), 1u);
    const auto& barrier = g_captured_barriers[0];
    EXPECT_EQ(barrier.srcStageMask,
              VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    EXPECT_EQ(barrier.srcAccessMask, VK_ACCESS_2_SHADER_READ_BIT);
}

// Catches: explicit-scope overload OR-ing accumulated readers into a caller-provided empty
// external src (critique D8 — cross-queue src must stay verbatim).
TEST(VulkanImageTracker, ExplicitScopeWriteAfterReadUsesCallerSrcVerbatim) {
    VulkanImageBarrierTracker tracker;
    installCapture(tracker);
    const VkImage image = fakeImage(0x1005);
    const VkCommandBuffer cmd = fakeCmd();

    tracker.registerImage(image, /*generation=*/1, VK_IMAGE_ASPECT_COLOR_BIT,
                          VK_IMAGE_LAYOUT_UNDEFINED);
    tracker.transitionImage(cmd, image, /*generation=*/1, VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // Same-layout second reader accumulates under reader_*.
    constexpr VulkanImageBarrierTracker::AccessScope compute_read{
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_READ_BIT,
    };
    tracker.transitionImage(cmd, image, /*generation=*/1, VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, {}, compute_read);

    g_captured_barriers.clear();
    constexpr VulkanImageBarrierTracker::AccessScope empty_external_src{};
    constexpr VulkanImageBarrierTracker::AccessScope transfer_write{
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        VK_ACCESS_2_TRANSFER_WRITE_BIT,
    };
    tracker.transitionImage(cmd, image, /*generation=*/1, VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            empty_external_src,
                            transfer_write);

    ASSERT_EQ(g_captured_barriers.size(), 1u);
    EXPECT_EQ(g_captured_barriers[0].srcStageMask, VK_PIPELINE_STAGE_2_NONE);
    EXPECT_EQ(g_captured_barriers[0].srcAccessMask, VK_ACCESS_2_NONE);
    EXPECT_EQ(g_captured_barriers[0].dstStageMask, VK_PIPELINE_STAGE_2_TRANSFER_BIT);
    EXPECT_EQ(g_captured_barriers[0].dstAccessMask, VK_ACCESS_2_TRANSFER_WRITE_BIT);
}

// Catches: clearSwapchainOnly wiping external/interop entries that must survive swapchain rebuild.
TEST(VulkanImageTracker, ClearSwapchainOnlyKeepsExternal) {
    VulkanImageBarrierTracker tracker;
    const VkImage external = fakeImage(0x2001);
    const VkImage swapchain = fakeImage(0x2002);

    tracker.registerImage(external, /*generation=*/1, VK_IMAGE_ASPECT_COLOR_BIT,
                          VK_IMAGE_LAYOUT_GENERAL, /*external=*/true);
    tracker.registerImage(swapchain, /*generation=*/3, VK_IMAGE_ASPECT_COLOR_BIT,
                          VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, /*external=*/false);

    tracker.clearSwapchainOnly();

    EXPECT_EQ(tracker.imageLayout(external, /*generation=*/1, VK_IMAGE_LAYOUT_UNDEFINED),
              VK_IMAGE_LAYOUT_GENERAL);
    EXPECT_EQ(tracker.imageLayout(swapchain, /*generation=*/3, VK_IMAGE_LAYOUT_UNDEFINED),
              VK_IMAGE_LAYOUT_UNDEFINED);
}

// Catches: layout-changing transition to a READ destination using only the last writer as src,
// missing WAR against accumulated readers (sub-task 4 race on the transition itself).
TEST(VulkanImageTracker, ReadLayoutTransitionWaitsOnPriorReaders) {
    VulkanImageBarrierTracker tracker;
    installCapture(tracker);
    const VkImage image = fakeImage(0x1006);
    const VkCommandBuffer cmd = fakeCmd();

    tracker.registerImage(image, /*generation=*/1, VK_IMAGE_ASPECT_COLOR_BIT,
                          VK_IMAGE_LAYOUT_UNDEFINED);
    tracker.transitionImage(cmd, image, /*generation=*/1, VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    constexpr VulkanImageBarrierTracker::AccessScope compute_read{
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_READ_BIT,
    };
    tracker.transitionImage(cmd, image, /*generation=*/1, VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, {}, compute_read);

    g_captured_barriers.clear();
    tracker.transitionImage(cmd, image, /*generation=*/1, VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    ASSERT_EQ(g_captured_barriers.size(), 1u);
    EXPECT_EQ(g_captured_barriers[0].srcStageMask,
              VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    EXPECT_EQ(g_captured_barriers[0].srcAccessMask, VK_ACCESS_2_SHADER_READ_BIT);
}

// Catches: after a layout-changing transition, stale reader_* (and old writer) surviving so the
// next barrier over-syncs with unbounded accumulated reader stages.
TEST(VulkanImageTracker, LayoutTransitionResetsReaderAccumulation) {
    VulkanImageBarrierTracker tracker;
    installCapture(tracker);
    const VkImage image = fakeImage(0x1007);
    const VkCommandBuffer cmd = fakeCmd();

    tracker.registerImage(image, /*generation=*/1, VK_IMAGE_ASPECT_COLOR_BIT,
                          VK_IMAGE_LAYOUT_UNDEFINED);
    tracker.transitionImage(cmd, image, /*generation=*/1, VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    constexpr VulkanImageBarrierTracker::AccessScope compute_read{
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_READ_BIT,
    };
    tracker.transitionImage(cmd, image, /*generation=*/1, VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, {}, compute_read);

    tracker.transitionImage(cmd, image, /*generation=*/1, VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    g_captured_barriers.clear();
    tracker.transitionImage(cmd, image, /*generation=*/1, VK_IMAGE_ASPECT_COLOR_BIT,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    ASSERT_EQ(g_captured_barriers.size(), 1u);
    const auto& barrier = g_captured_barriers[0];
    EXPECT_EQ(barrier.srcStageMask, VK_PIPELINE_STAGE_2_TRANSFER_BIT);
    EXPECT_EQ(barrier.srcAccessMask, VK_ACCESS_2_TRANSFER_READ_BIT);
    EXPECT_EQ(barrier.srcStageMask &
                  (VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT),
              0u);
}

// G9 note (spec §5.3): compose content-generation must not key tracker calls — production
// migration uses OutputImageSlot::image_generation / OutputSlotResources::image_generation
// exclusively; no unit assertion here (call-site discipline).
