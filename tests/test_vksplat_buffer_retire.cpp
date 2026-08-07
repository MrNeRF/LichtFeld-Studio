/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// #1576: retire-based device buffer growth (no HOST_GUARD fence on reallocation).
// Scripted VulkanDispatch forge — same fixture style as test_vksplat_tagged_dispatch.cpp.

#include "rendering/rasterizer/vulkan/src/barrier_planner.h"
#include "rendering/rasterizer/vulkan/src/gs_pipeline.h"
#include "rendering/vulkan_wait.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

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

    enum class RecordedOp : std::uint8_t {
        BeginCb,
        EndCb,
        QueueSubmit,
        ResetCb,
        Barrier2,
        ResetQuery,
    };

    struct DispatchScript {
        std::vector<RecordedOp> ops;
        int submit_calls = 0;
        int begin_calls = 0;
        int end_calls = 0;
        int reset_fences_calls = 0;
        std::uint64_t timeline_counter = 0;

        static DispatchScript*& active() {
            static DispatchScript* ptr = nullptr;
            return ptr;
        }
        void bind() { active() = this; }
        void unbind() {
            if (active() == this) {
                active() = nullptr;
            }
        }

        void clear_recording() { ops.clear(); }

        static VKAPI_ATTR VkResult VKAPI_CALL begin_cb(VkCommandBuffer, const VkCommandBufferBeginInfo*) {
            EXPECT_NE(active(), nullptr);
            ++active()->begin_calls;
            active()->ops.push_back(RecordedOp::BeginCb);
            return VK_SUCCESS;
        }
        static VKAPI_ATTR void VKAPI_CALL barrier2(VkCommandBuffer, const VkDependencyInfo*) {
            EXPECT_NE(active(), nullptr);
            active()->ops.push_back(RecordedOp::Barrier2);
        }
        static VKAPI_ATTR void VKAPI_CALL reset_query(VkCommandBuffer, VkQueryPool, uint32_t, uint32_t) {
            EXPECT_NE(active(), nullptr);
            active()->ops.push_back(RecordedOp::ResetQuery);
        }
        static VKAPI_ATTR VkResult VKAPI_CALL end_cb(VkCommandBuffer) {
            EXPECT_NE(active(), nullptr);
            ++active()->end_calls;
            active()->ops.push_back(RecordedOp::EndCb);
            return VK_SUCCESS;
        }
        static VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue,
                                                           uint32_t,
                                                           const VkSubmitInfo*,
                                                           VkFence) {
            EXPECT_NE(active(), nullptr);
            ++active()->submit_calls;
            active()->ops.push_back(RecordedOp::QueueSubmit);
            return VK_SUCCESS;
        }
        static VKAPI_ATTR VkResult VKAPI_CALL queue_wait_idle(VkQueue) { return VK_SUCCESS; }
        static VKAPI_ATTR VkResult VKAPI_CALL reset_cb(VkCommandBuffer, VkCommandBufferResetFlags) {
            EXPECT_NE(active(), nullptr);
            active()->ops.push_back(RecordedOp::ResetCb);
            return VK_SUCCESS;
        }
        static VKAPI_ATTR VkResult VKAPI_CALL get_sem(VkDevice, VkSemaphore, uint64_t* value) {
            if (value != nullptr) {
                *value = active() != nullptr ? active()->timeline_counter : 0;
            }
            return VK_SUCCESS;
        }
        static VKAPI_ATTR VkResult VKAPI_CALL reset_fences(VkDevice, uint32_t, const VkFence*) {
            EXPECT_NE(active(), nullptr);
            ++active()->reset_fences_calls;
            return VK_SUCCESS;
        }
        static VKAPI_ATTR VkResult VKAPI_CALL wait_for_fences(VkDevice,
                                                              uint32_t,
                                                              const VkFence*,
                                                              VkBool32,
                                                              uint64_t) {
            return VK_SUCCESS;
        }
    };

    struct BindScript {
        DispatchScript& script;
        explicit BindScript(DispatchScript& s) : script(s) { script.bind(); }
        ~BindScript() { script.unbind(); }
        BindScript(const BindScript&) = delete;
        BindScript& operator=(const BindScript&) = delete;
    };

    class TestablePipeline final : public VulkanGSPipeline {
    public:
        ~TestablePipeline() { disarm_for_destruction(); }

        void install_fake_handles() {
            device = fakeVkHandle<VkDevice>(0x1001);
            command_queue = fakeVkHandle<VkQueue>(0x1002);
            command_pool = fakeVkHandle<VkCommandPool>(0x1003);
            fence = fakeVkHandle<VkFence>(0x1004);
            queue_family_index = kQueueFamily;
            barrier_planner_ = BufferBarrierPlanner(kQueueFamily);

            for (std::uint32_t i = 0; i < kCommandBatchSlotCount; ++i) {
                command_batch_slots_[i].command_buffer =
                    fakeVkHandle<VkCommandBuffer>(0x2000 + i);
                command_batch_slots_[i].timestamp_query_pool =
                    fakeVkHandle<VkQueryPool>(0x3000 + i);
                command_batch_slots_[i].pending_signal = VK_NULL_HANDLE;
                command_batch_slots_[i].pending_signal_value = 0;
            }
            command_buffer = command_batch_slots_[0].command_buffer;
            timestamp_query_pool = command_batch_slots_[0].timestamp_query_pool;
            next_command_batch_slot_ = 0;
            active_command_batch_slot_ = 0;
            commandBatchInProgress = false;
            last_timeline_signal_values_.clear();
            pending_timeline_waits_.clear();
            buffer_retire_timeline_ = fakeVkHandle<VkSemaphore>(0x5001);
            next_buffer_retire_value_ = 1;
            retired_buffer_shells_.clear();
            test_buffer_handle_counter_ = 0xB1000;
            current_vram = 0;
            peak_vram = 0;
            allocator = VK_NULL_HANDLE;

            deviceInfo.subgroupSize = 32;
            deviceInfo.sharedSize = 48 * 1024;
            deviceInfo.maxGroupsX = 65535;
            deviceInfo.maxGroupsY = 65535;
            deviceInfo.maxGroupsZ = 65535;
        }

        void disarm_for_destruction() {
            commandBatchInProgress = false;
            pending_timeline_waits_.clear();
            last_timeline_signal_values_.clear();
            for (auto& slot : command_batch_slots_) {
                slot.pending_signal = VK_NULL_HANDLE;
                slot.pending_signal_value = 0;
                slot.pending_timestamp_count = 0;
                slot.pending_timestamp_marks.clear();
                slot.command_buffer = VK_NULL_HANDLE;
                slot.timestamp_query_pool = VK_NULL_HANDLE;
            }
            retired_buffer_shells_.clear();
            buffer_retire_timeline_ = VK_NULL_HANDLE;
            next_buffer_retire_value_ = 1;
            command_buffer = VK_NULL_HANDLE;
            timestamp_query_pool = VK_NULL_HANDLE;
            fence = VK_NULL_HANDLE;
            command_pool = VK_NULL_HANDLE;
            command_queue = VK_NULL_HANDLE;
            device = VK_NULL_HANDLE;
            instance = VK_NULL_HANDLE;
            physical_device = VK_NULL_HANDLE;
            allocator = VK_NULL_HANDLE;
            barrier_planner_.reset();
            current_vram = 0;
        }

        [[nodiscard]] std::size_t retired_shell_count() const {
            return retired_buffer_shells_.size();
        }
        [[nodiscard]] VkSemaphore retire_timeline() const { return buffer_retire_timeline_; }
        [[nodiscard]] std::uint64_t next_retire_value() const { return next_buffer_retire_value_; }
        [[nodiscard]] VkSemaphore pending_signal_of(const std::uint32_t slot) const {
            return command_batch_slots_[slot].pending_signal;
        }
        [[nodiscard]] std::uint64_t pending_value_of(const std::uint32_t slot) const {
            return command_batch_slots_[slot].pending_signal_value;
        }
        using VulkanGSPipeline::cleanup;
        using VulkanGSPipeline::cleanupBuffers;
    };

    [[nodiscard]] lfs::rendering::VulkanDispatch make_scripted_dispatch() {
        lfs::rendering::VulkanDispatch d{};
        d.begin_command_buffer = &DispatchScript::begin_cb;
        d.cmd_pipeline_barrier2 = &DispatchScript::barrier2;
        d.cmd_reset_query_pool = &DispatchScript::reset_query;
        d.end_command_buffer = &DispatchScript::end_cb;
        d.queue_submit = &DispatchScript::queue_submit;
        d.queue_wait_idle = &DispatchScript::queue_wait_idle;
        d.reset_command_buffer = &DispatchScript::reset_cb;
        d.get_semaphore_counter_value = &DispatchScript::get_sem;
        d.reset_fences = &DispatchScript::reset_fences;
        d.wait_for_fences = &DispatchScript::wait_for_fences;
        return d;
    }

} // namespace

// (a) Mid-batch growth splits without a fence wait; old shell queues until timeline completes.
TEST(VkSplatBufferRetire, MidBatchGrowthSplitsWithoutFenceAndRetires) {
    DispatchScript script;
    BindScript bind(script);

    TestablePipeline pipeline;
    pipeline.install_fake_handles();
    pipeline.setVulkanDispatch(make_scripted_dispatch());

    _VulkanBuffer buf;
    buf.label = "retire_test";
    pipeline.createBuffer(256, buf);
    const VkBuffer old_handle = buf.buffer;
    const size_t baseline_vram = pipeline.getCurrentAllocSize();
    EXPECT_EQ(baseline_vram, 256u);

    pipeline.beginCommandBatch();
    const int submits_before = script.submit_calls;
    const int reset_fences_before = script.reset_fences_calls;

    pipeline.resizeDeviceBuffer(buf, 512, /*no_shrink=*/true);

    EXPECT_EQ(script.reset_fences_calls, reset_fences_before)
        << "growth must not take the HOST_GUARD fence path";
    EXPECT_GT(script.submit_calls, submits_before)
        << "mid-batch growth must submit the split batch";
    EXPECT_EQ(pipeline.retired_shell_count(), 1u);
    EXPECT_NE(buf.buffer, old_handle);
    EXPECT_EQ(buf.capacity, 512u);
    // Old + new live until drain.
    EXPECT_EQ(pipeline.getCurrentAllocSize(), 256u + 512u);
    EXPECT_TRUE(pipeline.isCommandBatchInProgress());

    // Not complete yet — shell stays.
    script.timeline_counter = 0;
    pipeline.endCommandBatch(/*use_fence=*/false);
    pipeline.beginCommandBatch(); // drains
    EXPECT_EQ(pipeline.retired_shell_count(), 1u);

    // Complete retire timeline value 1 (first growth split).
    script.timeline_counter = 1;
    pipeline.endCommandBatch(/*use_fence=*/false);
    pipeline.beginCommandBatch();
    EXPECT_EQ(pipeline.retired_shell_count(), 0u);
    EXPECT_EQ(pipeline.getCurrentAllocSize(), 512u);

    pipeline.endCommandBatch(/*use_fence=*/false);
    pipeline.destroyBufferRetired(buf);
}

// (b) Growth with nothing in flight frees the old allocation immediately.
TEST(VkSplatBufferRetire, GrowthWithNothingInFlightFreesImmediately) {
    DispatchScript script;
    BindScript bind(script);

    TestablePipeline pipeline;
    pipeline.install_fake_handles();
    pipeline.setVulkanDispatch(make_scripted_dispatch());

    _VulkanBuffer buf;
    pipeline.createBuffer(128, buf);
    EXPECT_EQ(pipeline.getCurrentAllocSize(), 128u);

    pipeline.resizeDeviceBuffer(buf, 256, /*no_shrink=*/true);

    EXPECT_EQ(pipeline.retired_shell_count(), 0u);
    EXPECT_EQ(pipeline.getCurrentAllocSize(), 256u);
    EXPECT_EQ(buf.capacity, 256u);
    EXPECT_EQ(script.submit_calls, 0)
        << "outside-batch growth with empty pending slots must not submit";

    pipeline.destroyBufferRetired(buf);
}

// (c) Two growths before drain produce two shells; both free; VRAM returns to baseline.
TEST(VkSplatBufferRetire, TwoGrowthsQueueTwoShellsThenDrain) {
    DispatchScript script;
    BindScript bind(script);

    TestablePipeline pipeline;
    pipeline.install_fake_handles();
    pipeline.setVulkanDispatch(make_scripted_dispatch());

    _VulkanBuffer buf;
    pipeline.createBuffer(64, buf);

    pipeline.beginCommandBatch();
    pipeline.resizeDeviceBuffer(buf, 128, /*no_shrink=*/true);
    EXPECT_EQ(pipeline.retired_shell_count(), 1u);
    pipeline.resizeDeviceBuffer(buf, 256, /*no_shrink=*/true);
    EXPECT_EQ(pipeline.retired_shell_count(), 2u);
    // 64 + 128 retired shells + 256 live
    EXPECT_EQ(pipeline.getCurrentAllocSize(), 64u + 128u + 256u);

    script.timeline_counter = 100; // past both retire signals
    pipeline.endCommandBatch(/*use_fence=*/false);
    pipeline.beginCommandBatch();
    EXPECT_EQ(pipeline.retired_shell_count(), 0u);
    EXPECT_EQ(pipeline.getCurrentAllocSize(), 256u);

    pipeline.endCommandBatch(/*use_fence=*/false);
    pipeline.destroyBufferRetired(buf);
    EXPECT_EQ(pipeline.getCurrentAllocSize(), 0u);
}

// (d) Force-drain empties a non-empty retire queue (cleanupBuffers path shape).
TEST(VkSplatBufferRetire, ForceDrainEmptiesRetireQueue) {
    DispatchScript script;
    BindScript bind(script);

    TestablePipeline pipeline;
    pipeline.install_fake_handles();
    pipeline.setVulkanDispatch(make_scripted_dispatch());

    _VulkanBuffer buf;
    pipeline.createBuffer(100, buf);
    pipeline.beginCommandBatch();
    pipeline.resizeDeviceBuffer(buf, 200, /*no_shrink=*/true);
    EXPECT_EQ(pipeline.retired_shell_count(), 1u);

    pipeline.endCommandBatch(/*use_fence=*/false);
    // Timeline not advanced — non-force would hold.
    script.timeline_counter = 0;
    pipeline.drainRetiredBufferShells(/*force=*/false);
    EXPECT_EQ(pipeline.retired_shell_count(), 1u);

    pipeline.drainRetiredBufferShells(/*force=*/true);
    EXPECT_EQ(pipeline.retired_shell_count(), 0u);
    EXPECT_EQ(pipeline.getCurrentAllocSize(), 200u);

    pipeline.destroyBufferRetired(buf);
}

// (e) Handle-reuse safety: forget at growth enqueue so old handle is untracked immediately.
TEST(VkSplatBufferRetire, GrowthForgetAtEnqueueIsHandleReuseSafe) {
    DispatchScript script;
    BindScript bind(script);

    TestablePipeline pipeline;
    pipeline.install_fake_handles();
    pipeline.setVulkanDispatch(make_scripted_dispatch());

    _VulkanBuffer buf;
    pipeline.createBuffer(64, buf);
    const VkBuffer old_handle = buf.buffer;

    pipeline.beginCommandBatch();
    // TransferWrite seeds a writer on the live buffer before growth.
    {
        lfs::rendering::vulkan::DeclaredAccess write{
            .buffer = &buf,
            .use = BufferUse::TransferWrite,
        };
        pipeline.planTransfer(std::span{&write, 1});
    }

    pipeline.resizeDeviceBuffer(buf, 128, /*no_shrink=*/true);
    EXPECT_EQ(pipeline.retired_shell_count(), 1u);
    EXPECT_NE(buf.buffer, old_handle);

    // Reuse the old handle bits (Vulkan can recycle VkBuffer values after free).
    // After forget-at-enqueue, re-track must start EMPTY — not inherit the pre-growth writer.
    _VulkanBuffer reused;
    reused.buffer = old_handle;
    reused.allocation = fakeVkHandle<VmaAllocation>(0xDEAD);
    reused.allocSize = 32;
    reused.capacity = 32;
    reused.size = 32;
    pipeline.trackExternalParent(reused.buffer);

    script.clear_recording();
    {
        lfs::rendering::vulkan::DeclaredAccess write{
            .buffer = &reused,
            .use = BufferUse::TransferWrite,
        };
        pipeline.planTransfer(std::span{&write, 1});
    }
    // First write after re-track on a forgotten handle emits no buffer barrier.
    // (planTransfer only emits when hazards exist; EMPTY prior writer → none.)
    int barrier_ops = 0;
    for (const auto op : script.ops) {
        if (op == RecordedOp::Barrier2) {
            ++barrier_ops;
        }
    }
    EXPECT_EQ(barrier_ops, 0)
        << "first write after growth-forget+retrack must not inherit prior writer";

    script.timeline_counter = 99;
    pipeline.endCommandBatch(/*use_fence=*/false);
    pipeline.beginCommandBatch();
    EXPECT_EQ(pipeline.retired_shell_count(), 0u);
    pipeline.endCommandBatch(/*use_fence=*/false);

    pipeline.destroyBufferRetired(buf);
    pipeline.untrackExternalParent(reused.buffer);
}

// CleanupBuffers force-drains (device idle / wait path shape without real Vulkan teardown).
TEST(VkSplatBufferRetire, DrainForceMatchesCleanupContract) {
    DispatchScript script;
    BindScript bind(script);

    TestablePipeline pipeline;
    pipeline.install_fake_handles();
    pipeline.setVulkanDispatch(make_scripted_dispatch());

    _VulkanBuffer a;
    pipeline.createBuffer(16, a);
    pipeline.beginCommandBatch();
    pipeline.resizeDeviceBuffer(a, 32, true);
    pipeline.resizeDeviceBuffer(a, 64, true);
    EXPECT_EQ(pipeline.retired_shell_count(), 2u);
    pipeline.endCommandBatch(/*use_fence=*/false);

    // cleanupBuffers shape: after wait, force-drain.
    pipeline.drainRetiredBufferShells(/*force=*/true);
    EXPECT_EQ(pipeline.retired_shell_count(), 0u);

    pipeline.destroyBufferRetired(a);
}
