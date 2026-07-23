/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

// Phase 7A: bounded Vulkan waits + closed SubmissionState transition table +
// injectable VulkanDispatch / platform clock. Contract is pinned by
// .codex_tmp/phase-7a-vulkan-waits-spec.md §0.1 + §9 Amendments 1–2.

#include "core/error.hpp"
#include "core/export.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <stop_token>
#include <string_view>
#include <vulkan/vulkan.h>

namespace lfs::rendering {

    // ---------------------------------------------------------------------------
    // Frozen wait policy / outcomes (spec §0.1 verbatim semantics)
    // ---------------------------------------------------------------------------

    struct VulkanWaitPolicy {
        std::chrono::milliseconds slice{100};
        std::chrono::seconds stall_notice{2};
        std::chrono::seconds quarantine_after{10};
    };

    enum class WaitOutcome : std::uint8_t {
        Ready,
        Cancelled,
        Shutdown,
        Quarantined,
    };

    // ---------------------------------------------------------------------------
    // Submission state (spec §0.1 / §2.1). Transition table is Amendment-1
    // pinned: no T2 pre-submit-cancel event; replacement only on rejected
    // submit (T3) for the reset/pre-wait row.
    // ---------------------------------------------------------------------------

    struct SubmissionState {
        bool fence_reset = false;
        bool submit_accepted = false;
        bool timeline_published = false;
        std::uint64_t candidate_timeline = 0;
    };

    // Events map to the closed table T0,T1,T3–T7 (T2 removed by Amendment 1).
    enum class SubmissionTransition : std::uint8_t {
        BeginLifecycle,  // T0
        FenceReset,      // T1
        SubmitRejected,  // T3
        SubmitAccepted,  // T4
        PublishTimeline, // T5
        GpuReady,        // T6
        DeviceLost,      // T7
    };

    // Row assignment (spec §2.3).
    enum class SubmissionFencePolicy : std::uint8_t {
        NoResetNoReplacement,    // VulkanGSPipeline / VkSplat timeline
        ResetPreWaitReplacement, // point-cloud
    };

    struct SubmissionTransitionEffects {
        // Set on T3 when policy is ResetPreWaitReplacement and fence_reset was
        // true — caller may invoke replaceFenceSignaled. Never set for
        // NoResetNoReplacement (gs_pipeline).
        bool replace_fence_signaled = false;
        // T7: owner should latch terminal quarantine / stop new submits.
        bool terminal_device_lost = false;
    };

    // Apply one closed-table transition. Illegal transitions return
    // ErrorCode::ContractViolation (domain Vulkan). Legal transitions return
    // side-effect flags; they never host-signal timelines.
    // Fully qualify lfs::Result — lfs::rendering::Result is an unrelated
    // std::expected<T,std::string> alias in rendering.hpp and must not win
    // when that header is included before this one (point-cloud TU).
    [[nodiscard]] LFS_RENDERING_API lfs::Result<SubmissionTransitionEffects>
    apply_submission_transition(SubmissionState& state,
                                SubmissionTransition event,
                                SubmissionFencePolicy policy = SubmissionFencePolicy::NoResetNoReplacement,
                                std::uint64_t candidate_timeline = 0) noexcept;

    // ---------------------------------------------------------------------------
    // Vulkan PFN dispatch seam (Amendment 2: every begin→submit path call)
    // ---------------------------------------------------------------------------

    struct LFS_RENDERING_API VulkanDispatch {
        // Wait / acquire family
        PFN_vkWaitForFences wait_for_fences = nullptr;
        PFN_vkWaitSemaphores wait_semaphores = nullptr;
        PFN_vkAcquireNextImageKHR acquire_next_image_khr = nullptr;
        PFN_vkGetSemaphoreCounterValue get_semaphore_counter_value = nullptr;

        // beginCommandBatch prologue → endCommandBatch submit
        PFN_vkBeginCommandBuffer begin_command_buffer = nullptr;
        PFN_vkEndCommandBuffer end_command_buffer = nullptr;
        PFN_vkResetCommandBuffer reset_command_buffer = nullptr;
        PFN_vkCmdPipelineBarrier2 cmd_pipeline_barrier2 = nullptr;
        PFN_vkCmdResetQueryPool cmd_reset_query_pool = nullptr;
        PFN_vkCmdWriteTimestamp cmd_write_timestamp = nullptr;
        PFN_vkQueueSubmit queue_submit = nullptr;
        PFN_vkQueueWaitIdle queue_wait_idle = nullptr;

        // Fence lifecycle (reset/pre-wait row + post-submit fence path)
        PFN_vkResetFences reset_fences = nullptr;
        PFN_vkCreateFence create_fence = nullptr;
        PFN_vkDestroyFence destroy_fence = nullptr;

        // Production symbols. Safe to call once; returns a fully filled table.
        [[nodiscard]] static VulkanDispatch real() noexcept;
    };

    // Injectable clock for fake-time unit tests (spec §4.2 / AMB 9).
    using ClockNow = std::function<std::chrono::steady_clock::time_point()>;

    [[nodiscard]] inline std::chrono::steady_clock::time_point steady_clock_now() {
        return std::chrono::steady_clock::now();
    }

    // Stall / quarantine observer (pre-ErrorBus). Default = LOG_WARN / LOG_ERROR.
    struct WaitObserver {
        virtual ~WaitObserver() = default;
        // Called at most once per wait invocation after stall_notice.
        virtual void on_stall(std::string_view fingerprint) noexcept = 0;
        // Called once when transitioning owner into quarantine.
        virtual void on_quarantine(std::string_view fingerprint) noexcept = 0;
    };

    // Default observer: 2 s notice = LOG_WARN heartbeat (AMB 2 binding).
    struct LFS_RENDERING_API LogWaitObserver final : WaitObserver {
        void on_stall(std::string_view fingerprint) noexcept override;
        void on_quarantine(std::string_view fingerprint) noexcept override;
    };

    // Optional hooks for a single wait invocation.
    struct WaitContext {
        const VulkanDispatch* dispatch = nullptr; // nullptr → VulkanDispatch::real()
        ClockNow now;                             // empty → steady_clock::now
        std::function<bool()> shutdown_latched;   // empty → false
        std::function<bool()> is_quarantined;     // empty → false
        // Called once when returning Quarantined (owner latch). May be empty.
        std::function<void()> quarantine_owner;
        WaitObserver* observer = nullptr; // nullptr → process-local LogWaitObserver
        std::string_view fingerprint = "vulkan.wait";
        std::atomic<bool>* owner_quarantine_flag = nullptr; // optional atomic latch
    };

    // ---------------------------------------------------------------------------
    // Bounded wait family
    // ---------------------------------------------------------------------------

    [[nodiscard]] LFS_RENDERING_API lfs::Result<WaitOutcome>
    wait_fence_bounded(VkDevice device,
                       VkFence fence,
                       std::stop_token stop,
                       VulkanWaitPolicy policy = {},
                       WaitContext context = {});

    [[nodiscard]] LFS_RENDERING_API lfs::Result<WaitOutcome>
    wait_semaphores_bounded(VkDevice device,
                            const VkSemaphoreWaitInfo& wait_info,
                            std::stop_token stop,
                            VulkanWaitPolicy policy = {},
                            WaitContext context = {});

    // Ready → image index. VK_SUCCESS and VK_SUBOPTIMAL_KHR both yield the
    // image index (SUBOPTIMAL is success-with-flag, not an error). Optional
    // out_suboptimal receives true only for VK_SUBOPTIMAL_KHR; nullptr skips
    // the side channel. OUT_OF_DATE and other hard failures are Result errors
    // with ErrorCode::Unavailable / DeviceLost / etc. Timeout path uses the
    // same slice/stall/quarantine policy as fence waits.
    [[nodiscard]] LFS_RENDERING_API lfs::Result<std::uint32_t>
    acquire_next_image_bounded(VkDevice device,
                               VkSwapchainKHR swapchain,
                               VkSemaphore semaphore,
                               VkFence fence,
                               std::stop_token stop,
                               VulkanWaitPolicy policy = {},
                               WaitContext context = {},
                               bool* out_suboptimal = nullptr);

    // Map a native VkResult that is not TIMEOUT/SUCCESS into an lfs::Error.
    // DeviceLost is always ErrorCode::DeviceLost / ErrorDomain::Vulkan.
    [[nodiscard]] LFS_RENDERING_API Error
    vulkan_result_to_error(VkResult result,
                           std::string_view operation,
                           core::SourceSite site) noexcept;

} // namespace lfs::rendering
