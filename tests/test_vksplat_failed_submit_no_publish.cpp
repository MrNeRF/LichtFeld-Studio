/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// QW-6 (Phase 7A): force vkQueueSubmit fail-then-success via VulkanDispatch;
// assert wasTimelineSignalSubmitted==false on failure, no pending/published
// completion, exactly one publication on retry. use_fence=false timeline route.
// Spec §4.3; Amendment 2 single seam (no parallel test-double of endCommandBatch).

#include "rendering/rasterizer/vulkan/src/gs_pipeline.h"
#include "rendering/vulkan_wait.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

    template <typename Handle>
    [[nodiscard]] Handle fakeVkHandle(const std::uintptr_t value) {
        if constexpr (std::is_pointer_v<Handle>) {
            return reinterpret_cast<Handle>(value);
        } else {
            return static_cast<Handle>(value);
        }
    }

    // Scripted submit: first call fails, subsequent succeed.
    struct SubmitScript {
        int submit_calls = 0;
        int end_calls = 0;
        int begin_calls = 0;
        int reset_cb_calls = 0;
        VkResult first_submit_result = VK_ERROR_DEVICE_LOST;

        static SubmitScript*& active() {
            static SubmitScript* ptr = nullptr;
            return ptr;
        }
        void bind() { active() = this; }
        void unbind() {
            if (active() == this) {
                active() = nullptr;
            }
        }

        static VKAPI_ATTR VkResult VKAPI_CALL begin_cb(VkCommandBuffer, const VkCommandBufferBeginInfo*) {
            EXPECT_NE(active(), nullptr);
            ++active()->begin_calls;
            return VK_SUCCESS;
        }
        static VKAPI_ATTR void VKAPI_CALL barrier2(VkCommandBuffer, const VkDependencyInfo*) {
            // no-op
        }
        static VKAPI_ATTR void VKAPI_CALL reset_query(VkCommandBuffer, VkQueryPool, uint32_t, uint32_t) {
            // no-op
        }
        static VKAPI_ATTR VkResult VKAPI_CALL end_cb(VkCommandBuffer) {
            EXPECT_NE(active(), nullptr);
            ++active()->end_calls;
            return VK_SUCCESS;
        }
        static VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue,
                                                           uint32_t,
                                                           const VkSubmitInfo*,
                                                           VkFence) {
            EXPECT_NE(active(), nullptr);
            const int n = active()->submit_calls++;
            if (n == 0) {
                return active()->first_submit_result;
            }
            return VK_SUCCESS;
        }
        static VKAPI_ATTR VkResult VKAPI_CALL reset_cb(VkCommandBuffer, VkCommandBufferResetFlags) {
            EXPECT_NE(active(), nullptr);
            ++active()->reset_cb_calls;
            return VK_SUCCESS;
        }
        static VKAPI_ATTR VkResult VKAPI_CALL get_sem(VkDevice, VkSemaphore, uint64_t* value) {
            if (value != nullptr) {
                *value = 0;
            }
            return VK_SUCCESS;
        }
    };

    struct BindSubmit {
        SubmitScript& script;
        explicit BindSubmit(SubmitScript& s) : script(s) { script.bind(); }
        ~BindSubmit() { script.unbind(); }
        BindSubmit(const BindSubmit&) = delete;
        BindSubmit& operator=(const BindSubmit&) = delete;
    };

    // Subclass to install fake device/queue/slot handles without real Vulkan.
    class TestablePipeline final : public VulkanGSPipeline {
    public:
        ~TestablePipeline() {
            // Base dtor calls cleanup() which would invoke real Vulkan destroy
            // paths on our forged handles. Disarm first.
            disarm_for_destruction();
        }

        void install_fake_handles() {
            device = fakeVkHandle<VkDevice>(0x1001);
            command_queue = fakeVkHandle<VkQueue>(0x1002);
            command_pool = fakeVkHandle<VkCommandPool>(0x1003);
            fence = fakeVkHandle<VkFence>(0x1004);
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
            command_buffer = VK_NULL_HANDLE;
            timestamp_query_pool = VK_NULL_HANDLE;
            fence = VK_NULL_HANDLE;
            command_pool = VK_NULL_HANDLE;
            command_queue = VK_NULL_HANDLE;
            device = VK_NULL_HANDLE;
            instance = VK_NULL_HANDLE;
            physical_device = VK_NULL_HANDLE;
            allocator = VK_NULL_HANDLE;
        }

        [[nodiscard]] std::uint64_t pending_signal_value_of_active() const {
            return command_batch_slots_[active_command_batch_slot_].pending_signal_value;
        }
        [[nodiscard]] VkSemaphore pending_signal_of_active() const {
            return command_batch_slots_[active_command_batch_slot_].pending_signal;
        }
        // After a failed end, active slot may have advanced on the next begin.
        // Inspect all slots for any pending publication.
        [[nodiscard]] bool any_pending_publication() const {
            for (const auto& slot : command_batch_slots_) {
                if (slot.pending_signal != VK_NULL_HANDLE || slot.pending_signal_value != 0) {
                    return true;
                }
            }
            return false;
        }
    };

    lfs::rendering::VulkanDispatch make_scripted_dispatch() {
        // Pure mock table — never fall through to real Vulkan with fake handles.
        lfs::rendering::VulkanDispatch d{};
        d.begin_command_buffer = &SubmitScript::begin_cb;
        d.cmd_pipeline_barrier2 = &SubmitScript::barrier2;
        d.cmd_reset_query_pool = &SubmitScript::reset_query;
        d.end_command_buffer = &SubmitScript::end_cb;
        d.queue_submit = &SubmitScript::queue_submit;
        d.reset_command_buffer = &SubmitScript::reset_cb;
        d.get_semaphore_counter_value = &SubmitScript::get_sem;
        // wait_semaphores unused when pending is empty / counter is 0.
        return d;
    }

} // namespace

TEST(VkSplatFailedSubmitNoPublish, QW6_FailThenSuccessNoFalsePublication) {
    SubmitScript script;
    BindSubmit bind(script);

    TestablePipeline pipeline;
    pipeline.install_fake_handles();
    pipeline.setVulkanDispatch(make_scripted_dispatch());

    const VkSemaphore semaphore = fakeVkHandle<VkSemaphore>(0xABCD);
    const std::uint64_t value_fail = 1;
    const std::uint64_t value_ok = 2; // strictly increasing

    // --- Attempt 1: vkQueueSubmit fails ---
    pipeline.beginCommandBatch();
    EXPECT_TRUE(pipeline.isCommandBatchInProgress());
    // P4 retired _THROW_ERROR: the rejected submit now surfaces as the typed
    // lfs::Exception (DeviceLost/Vulkan with native code and detail).
    EXPECT_THROW(
        pipeline.endCommandBatch(/*use_fence=*/false, semaphore, value_fail),
        lfs::Exception);

    EXPECT_FALSE(pipeline.isCommandBatchInProgress());
    EXPECT_FALSE(pipeline.wasTimelineSignalSubmitted(semaphore, value_fail));
    EXPECT_FALSE(pipeline.any_pending_publication());
    EXPECT_FALSE(pipeline.lastSubmissionState().submit_accepted);
    EXPECT_FALSE(pipeline.lastSubmissionState().timeline_published);
    EXPECT_EQ(script.submit_calls, 1);
    EXPECT_EQ(script.end_calls, 1);
    EXPECT_GE(script.reset_cb_calls, 1); // cancelCommandBatch resets CB

    // --- Attempt 2: same seam, success → exactly one publication ---
    pipeline.beginCommandBatch();
    EXPECT_NO_THROW(
        pipeline.endCommandBatch(/*use_fence=*/false, semaphore, value_ok));

    EXPECT_FALSE(pipeline.isCommandBatchInProgress());
    // Timeline semaphores are monotonic: publishing value_ok legitimately
    // covers every smaller value, so value_fail now reads as signaled. The
    // no-false-readiness property was asserted between failure and retry.
    EXPECT_TRUE(pipeline.wasTimelineSignalSubmitted(semaphore, value_fail));
    EXPECT_TRUE(pipeline.wasTimelineSignalSubmitted(semaphore, value_ok));
    EXPECT_TRUE(pipeline.lastSubmissionState().submit_accepted);
    EXPECT_TRUE(pipeline.lastSubmissionState().timeline_published);
    EXPECT_EQ(pipeline.lastSubmissionState().candidate_timeline, value_ok);
    EXPECT_EQ(script.submit_calls, 2);
    EXPECT_EQ(script.end_calls, 2);

    // Pending slot holds the published completion for ring retirement — not a
    // false pre-submit publication; value_fail must still be absent.
    EXPECT_NE(pipeline.pending_signal_of_active(), VK_NULL_HANDLE);
    EXPECT_EQ(pipeline.pending_signal_value_of_active(), value_ok);
}
