/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Epic #1568 / #1567 — OutputSlotRing host bookkeeping (GPU-free).

#include "rendering/gpu_lod_target_feedback.hpp"
#include "rendering/output_slot_ring.hpp"
#include "rendering/passes/vulkan_viewport_pass.hpp"
#include "rendering/point_cloud_vulkan_renderer.hpp"

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

TEST(GpuLodTargetFeedback, IsolatesControllersAndUnionsRecentDemand) {
    lfs::vis::GpuLodTargetFeedbackTable table;
    auto& a = table.touch({1}, 10);
    a.pixel_scale_feedback = 7.0f;
    a.protected_chunks = {1, 3};
    a.prefetch_requests = {{4, 2}, {5, 3}};
    auto& b = table.touch({2}, 11);
    EXPECT_EQ(b.pixel_scale_feedback, 1.0f);
    b.protected_chunks = {2, 3};
    b.prefetch_requests = {{4, 9}};
    auto demand = table.demand(12);
    EXPECT_EQ(demand.protected_chunks, (std::vector<std::uint32_t>{1, 2, 3}));
    ASSERT_EQ(demand.prefetch_requests.size(), 2u);
    EXPECT_EQ(demand.prefetch_requests[0].priority, 9u);
    demand = table.demand(14);
    EXPECT_EQ(demand.protected_chunks, (std::vector<std::uint32_t>{2, 3}));
    table.release({2});
    EXPECT_EQ(table.find({2}), nullptr);
    EXPECT_TRUE(table.demand(14).protected_chunks.empty());
}

TEST(PointCloudRenderTargets, SparseOutputsStayIndependentAfterRelease) {
    using Access = lfs::vis::PointCloudOutputOwnershipTestAccess;
    lfs::vis::PointCloudVulkanRenderer renderer;
    const auto a = Access::createEmptyOutput(renderer, {17});
    const auto b = Access::createEmptyOutput(renderer, {9001});
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_NE(a, b);
    EXPECT_TRUE(renderer.releaseRenderTarget({17}));
    EXPECT_EQ(Access::outputIdentity(renderer, {17}), nullptr);
    EXPECT_EQ(Access::outputIdentity(renderer, {9001}), b);
    EXPECT_EQ(Access::createEmptyOutput(renderer, {17}), nullptr);
    EXPECT_NE(Access::createEmptyOutput(renderer, {18}), b);
}

TEST(SharedViewportGpuAssets, RemainAliveUntilLastPassReleasesOwnership) {
    auto assets = std::make_shared<lfs::vis::SharedViewportGpuAssets>();
    std::weak_ptr<lfs::vis::SharedViewportGpuAssets> weak = assets;
    auto a = std::make_unique<lfs::vis::VulkanViewportPass>(assets);
    auto b = std::make_unique<lfs::vis::VulkanViewportPass>(assets);
    assets.reset();
    EXPECT_FALSE(weak.expired());
    a.reset();
    EXPECT_FALSE(weak.expired());
    b.reset();
    EXPECT_TRUE(weak.expired());
}

TEST(OutputSlotRing, LatestGenerationAndResetKeepTargetIdentity) {
    OutputSlotRing ring;
    const lfs::vis::RenderTargetId a{101}, b{99991};
    auto cell = ring.acquire(a);
    auto other = ring.acquire(b);
    ring.slotAt(a, cell) = makeSlot(0x101);
    ring.slotAt(b, other) = makeSlot(0x102);
    ring.markLatest(a, cell);
    EXPECT_EQ(ring.latestRingSlot(a), cell);
    EXPECT_EQ(ring.bumpGeneration(a), 1u);
    EXPECT_EQ(ring.bumpGeneration(a), 2u);
    EXPECT_EQ(ring.bumpGeneration(b), 1u);
    EXPECT_EQ(ring.generation(a), 2u);
    EXPECT_EQ(ring.latestSlot(a).image.image, fakeImage(0x101));
    EXPECT_THROW((void)ring.slotAt(a, other), std::out_of_range);
    EXPECT_THROW((void)ring.acquire({}), std::invalid_argument);
    EXPECT_TRUE(ring.releaseRenderTarget(a, [](auto&) {}));
    ring.reset();
    EXPECT_TRUE(ring.table().empty());
    EXPECT_EQ(ring.submissionCount(), 0u);
    EXPECT_THROW((void)ring.acquire(a), std::invalid_argument);
    EXPECT_EQ(ring.acquire(b), 0u);
}
