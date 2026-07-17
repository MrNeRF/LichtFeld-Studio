/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/vulkan_wait.hpp"

#include "core/logger.hpp"
#include "rendering/vulkan_result.hpp"

#include <algorithm>
#include <format>
#include <limits>

namespace lfs::rendering {
    namespace {

        LogWaitObserver& default_observer() noexcept {
            static LogWaitObserver observer;
            return observer;
        }

        [[nodiscard]] const VulkanDispatch& resolve_dispatch(const WaitContext& context) {
            if (context.dispatch != nullptr) {
                return *context.dispatch;
            }
            static const VulkanDispatch kReal = VulkanDispatch::real();
            return kReal;
        }

        [[nodiscard]] std::chrono::steady_clock::time_point
        resolve_now(const WaitContext& context) {
            if (context.now) {
                return context.now();
            }
            return steady_clock_now();
        }

        [[nodiscard]] bool resolve_shutdown(const WaitContext& context) {
            return context.shutdown_latched ? context.shutdown_latched() : false;
        }

        [[nodiscard]] bool resolve_quarantined(const WaitContext& context) {
            if (context.owner_quarantine_flag != nullptr &&
                context.owner_quarantine_flag->load(std::memory_order_acquire)) {
                return true;
            }
            return context.is_quarantined ? context.is_quarantined() : false;
        }

        void latch_quarantine(const WaitContext& context) {
            if (context.owner_quarantine_flag != nullptr) {
                context.owner_quarantine_flag->store(true, std::memory_order_release);
            }
            if (context.quarantine_owner) {
                context.quarantine_owner();
            }
        }

        [[nodiscard]] WaitObserver& resolve_observer(const WaitContext& context) noexcept {
            return context.observer != nullptr ? *context.observer : default_observer();
        }

        [[nodiscard]] std::uint64_t timeout_ns_for_slice(
            const VulkanWaitPolicy& policy,
            const std::chrono::steady_clock::time_point now,
            const std::chrono::steady_clock::time_point deadline_quar) {
            using namespace std::chrono;
            const auto remaining = duration_cast<nanoseconds>(deadline_quar - now);
            if (remaining.count() <= 0) {
                return 0;
            }
            const auto slice_ns = duration_cast<nanoseconds>(policy.slice);
            return static_cast<std::uint64_t>(
                std::min(remaining, slice_ns).count());
        }

        [[nodiscard]] const char* vk_result_name(const VkResult result) noexcept {
            return vkResultToString(result);
        }

        // Shared slice / stall / quarantine loop. `wait_once` returns the native
        // VkResult for one slice. On SUCCESS the loop returns Ready.
        template <class WaitOnce>
        [[nodiscard]] Result<WaitOutcome>
        bounded_wait_loop(std::stop_token stop,
                          VulkanWaitPolicy policy,
                          WaitContext context,
                          WaitOnce&& wait_once) {
            const auto start = resolve_now(context);
            const auto deadline_stall = start + policy.stall_notice;
            const auto deadline_quar = start + policy.quarantine_after;
            bool stall_published = false;
            auto& observer = resolve_observer(context);
            const VulkanDispatch& dispatch = resolve_dispatch(context);
            (void)dispatch; // wait_once captures the resolved table

            for (;;) {
                if (stop.stop_requested()) {
                    return WaitOutcome::Cancelled;
                }
                if (resolve_shutdown(context)) {
                    return WaitOutcome::Shutdown;
                }
                if (resolve_quarantined(context)) {
                    return WaitOutcome::Quarantined;
                }

                const auto now = resolve_now(context);
                if (now >= deadline_quar) {
                    latch_quarantine(context);
                    observer.on_quarantine(context.fingerprint);
                    return WaitOutcome::Quarantined;
                }

                const std::uint64_t timeout_ns =
                    timeout_ns_for_slice(policy, now, deadline_quar);
                const VkResult result = wait_once(timeout_ns);

                if (result == VK_SUCCESS) {
                    return WaitOutcome::Ready;
                }
                if (result == VK_TIMEOUT) {
                    const auto after = resolve_now(context);
                    if (after >= deadline_quar) {
                        latch_quarantine(context);
                        observer.on_quarantine(context.fingerprint);
                        return WaitOutcome::Quarantined;
                    }
                    if (after >= deadline_stall && !stall_published) {
                        observer.on_stall(context.fingerprint);
                        stall_published = true;
                    }
                    continue;
                }
                if (result == VK_ERROR_DEVICE_LOST) {
                    return vulkan_result_to_error(
                        result, "bounded wait", LFS_SOURCE_SITE_CURRENT());
                }
                return vulkan_result_to_error(
                    result, "bounded wait", LFS_SOURCE_SITE_CURRENT());
            }
        }

        [[nodiscard]] Error illegal_transition(std::string_view detail) noexcept {
            return make_error(ErrorInit{
                .code = ErrorCode::ContractViolation,
                .domain = ErrorDomain::Vulkan,
                .severity = Severity::Error,
                .retryability = Retryability::NotRetryable,
                .detail = std::string(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }

    } // namespace

    VulkanDispatch VulkanDispatch::real() noexcept {
        VulkanDispatch d;
        d.wait_for_fences = ::vkWaitForFences;
        d.wait_semaphores = ::vkWaitSemaphores;
        d.acquire_next_image_khr = ::vkAcquireNextImageKHR;
        d.get_semaphore_counter_value = ::vkGetSemaphoreCounterValue;
        d.begin_command_buffer = ::vkBeginCommandBuffer;
        d.end_command_buffer = ::vkEndCommandBuffer;
        d.reset_command_buffer = ::vkResetCommandBuffer;
        d.cmd_pipeline_barrier2 = ::vkCmdPipelineBarrier2;
        d.cmd_reset_query_pool = ::vkCmdResetQueryPool;
        d.cmd_write_timestamp = ::vkCmdWriteTimestamp;
        d.queue_submit = ::vkQueueSubmit;
        d.queue_wait_idle = ::vkQueueWaitIdle;
        d.reset_fences = ::vkResetFences;
        d.create_fence = ::vkCreateFence;
        d.destroy_fence = ::vkDestroyFence;
        return d;
    }

    void LogWaitObserver::on_stall(const std::string_view fingerprint) noexcept {
        try {
            LOG_WARN("GPU stalled (vulkan wait heartbeat): {}", fingerprint);
        } catch (...) { // LFS-CENSUS-OK(empty-catch): noexcept observer; logging must not escape
        }
    }

    void LogWaitObserver::on_quarantine(const std::string_view fingerprint) noexcept {
        try {
            LOG_ERROR("GPU wait quarantined after {}s policy window: {}", 10, fingerprint);
        } catch (...) { // LFS-CENSUS-OK(empty-catch): noexcept observer; logging must not escape
        }
    }

    Error vulkan_result_to_error(const VkResult result,
                                 const std::string_view operation,
                                 const core::SourceSite site) noexcept {
        ErrorCode code = ErrorCode::Internal;
        Severity severity = Severity::Error;
        Retryability retry = Retryability::NotRetryable;
        switch (result) {
        case VK_ERROR_DEVICE_LOST:
            code = ErrorCode::DeviceLost;
            severity = Severity::Fatal;
            break;
        case VK_ERROR_OUT_OF_HOST_MEMORY:
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:
        case VK_ERROR_OUT_OF_POOL_MEMORY:
        case VK_ERROR_FRAGMENTED_POOL:
        case VK_ERROR_FRAGMENTATION:
            code = ErrorCode::ResourceExhausted;
            retry = Retryability::RetryableWithBackoff;
            break;
        case VK_ERROR_OUT_OF_DATE_KHR:
        case VK_SUBOPTIMAL_KHR:
        case VK_ERROR_SURFACE_LOST_KHR:
        case VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT:
            code = ErrorCode::Unavailable;
            retry = Retryability::Retryable;
            break;
        case VK_TIMEOUT:
            // Slice timeouts are flow (WaitOutcome) and must not reach this
            // mapper; one that does is a missed deadline, not an outcome.
            code = ErrorCode::DeadlineExceeded;
            break;
        case VK_NOT_READY:
            code = ErrorCode::Unavailable;
            retry = Retryability::Retryable;
            break;
        default:
            code = ErrorCode::Internal;
            break;
        }

        return make_error(ErrorInit{
            .code = code,
            .domain = ErrorDomain::Vulkan,
            .severity = severity,
            .retryability = retry,
            .detail = std::format("{} failed: {} ({})",
                                  operation,
                                  vk_result_name(result),
                                  static_cast<int>(result)),
            .detection = site,
            .native = NativeError{
                .domain = ErrorDomain::Vulkan,
                .code = static_cast<std::int64_t>(result),
                .name = vk_result_name(result),
            },
        });
    }

    Result<SubmissionTransitionEffects>
    apply_submission_transition(SubmissionState& state,
                                const SubmissionTransition event,
                                const SubmissionFencePolicy policy,
                                const std::uint64_t candidate_timeline) noexcept {
        SubmissionTransitionEffects effects{};

        switch (event) {
        case SubmissionTransition::BeginLifecycle: {
            // T0 — fresh / after successful retire.
            state.fence_reset = false;
            state.submit_accepted = false;
            state.timeline_published = false;
            state.candidate_timeline = candidate_timeline;
            return effects;
        }
        case SubmissionTransition::FenceReset: {
            // T1 — pre-submit only.
            if (state.submit_accepted || state.timeline_published) {
                return illegal_transition(
                    "SubmissionState T1 FenceReset requires pre-submit "
                    "(submit_accepted/timeline_published must be false)");
            }
            state.fence_reset = true;
            state.submit_accepted = false;
            state.timeline_published = false;
            return effects;
        }
        case SubmissionTransition::SubmitRejected: {
            // T3 — pre-submit only. Replacement only on reset/pre-wait row
            // when fence_reset is already true (Amendment 1: no T2).
            if (state.submit_accepted || state.timeline_published) {
                return illegal_transition(
                    "SubmissionState T3 SubmitRejected requires pre-submit "
                    "(submit_accepted/timeline_published must be false)");
            }
            if (policy == SubmissionFencePolicy::ResetPreWaitReplacement &&
                state.fence_reset) {
                effects.replace_fence_signaled = true;
                state.fence_reset = false; // fence is replaced (signaled); not still "reset"
            }
            state.submit_accepted = false;
            state.timeline_published = false;
            return effects;
        }
        case SubmissionTransition::SubmitAccepted: {
            // T4 — pre-submit; set submit_accepted BEFORE any publish.
            if (state.submit_accepted || state.timeline_published) {
                return illegal_transition(
                    "SubmissionState T4 SubmitAccepted requires pre-submit "
                    "(submit_accepted/timeline_published must be false)");
            }
            state.submit_accepted = true;
            if (candidate_timeline != 0) {
                state.candidate_timeline = candidate_timeline;
            }
            return effects;
        }
        case SubmissionTransition::PublishTimeline: {
            // T5 — ordering: submit_accepted must already be true.
            if (!state.submit_accepted) {
                return illegal_transition(
                    "SubmissionState T5 PublishTimeline requires submit_accepted "
                    "(invariant: timeline_published implies submit_accepted)");
            }
            if (state.timeline_published) {
                return illegal_transition(
                    "SubmissionState T5 PublishTimeline forbids a second publish "
                    "of the same lifecycle");
            }
            if (candidate_timeline != 0) {
                state.candidate_timeline = candidate_timeline;
            }
            state.timeline_published = true;
            return effects;
        }
        case SubmissionTransition::GpuReady: {
            // T6 — proven completion; may clear pending / retire lifecycle.
            // Freeing resources is the caller's responsibility after Ready.
            state.fence_reset = false;
            state.submit_accepted = false;
            state.timeline_published = false;
            state.candidate_timeline = 0;
            return effects;
        }
        case SubmissionTransition::DeviceLost: {
            // T7 — terminal from any state. No host-signal; retain in-flight.
            effects.terminal_device_lost = true;
            return effects;
        }
        }

        return illegal_transition("SubmissionState: unknown transition event");
    }

    Result<WaitOutcome>
    wait_fence_bounded(const VkDevice device,
                       const VkFence fence,
                       const std::stop_token stop,
                       const VulkanWaitPolicy policy,
                       WaitContext context) {
        if (device == VK_NULL_HANDLE || fence == VK_NULL_HANDLE) {
            return make_error(ErrorInit{
                .code = ErrorCode::InvalidArgument,
                .domain = ErrorDomain::Vulkan,
                .detail = "wait_fence_bounded requires a non-null device and fence",
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }
        const VulkanDispatch& dispatch = resolve_dispatch(context);
        if (dispatch.wait_for_fences == nullptr) {
            return make_error(ErrorInit{
                .code = ErrorCode::FailedPrecondition,
                .domain = ErrorDomain::Vulkan,
                .detail = "wait_fence_bounded: VulkanDispatch::wait_for_fences is null",
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }
        return bounded_wait_loop(
            stop, policy, std::move(context),
            [&](const std::uint64_t timeout_ns) -> VkResult {
                return dispatch.wait_for_fences(device, 1, &fence, VK_TRUE, timeout_ns);
            });
    }

    Result<WaitOutcome>
    wait_semaphores_bounded(const VkDevice device,
                            const VkSemaphoreWaitInfo& wait_info,
                            const std::stop_token stop,
                            const VulkanWaitPolicy policy,
                            WaitContext context) {
        if (device == VK_NULL_HANDLE || wait_info.semaphoreCount == 0 ||
            wait_info.pSemaphores == nullptr || wait_info.pValues == nullptr) {
            return make_error(ErrorInit{
                .code = ErrorCode::InvalidArgument,
                .domain = ErrorDomain::Vulkan,
                .detail = "wait_semaphores_bounded requires device + non-empty wait info",
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }
        const VulkanDispatch& dispatch = resolve_dispatch(context);
        if (dispatch.wait_semaphores == nullptr) {
            return make_error(ErrorInit{
                .code = ErrorCode::FailedPrecondition,
                .domain = ErrorDomain::Vulkan,
                .detail = "wait_semaphores_bounded: VulkanDispatch::wait_semaphores is null",
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }
        return bounded_wait_loop(
            stop, policy, std::move(context),
            [&](const std::uint64_t timeout_ns) -> VkResult {
                return dispatch.wait_semaphores(device, &wait_info, timeout_ns);
            });
    }

    Result<std::uint32_t>
    acquire_next_image_bounded(const VkDevice device,
                               const VkSwapchainKHR swapchain,
                               const VkSemaphore semaphore,
                               const VkFence fence,
                               const std::stop_token stop,
                               const VulkanWaitPolicy policy,
                               WaitContext context,
                               bool* out_suboptimal) {
        if (out_suboptimal != nullptr) {
            *out_suboptimal = false;
        }
        if (device == VK_NULL_HANDLE || swapchain == VK_NULL_HANDLE) {
            return make_error(ErrorInit{
                .code = ErrorCode::InvalidArgument,
                .domain = ErrorDomain::Vulkan,
                .detail = "acquire_next_image_bounded requires device + swapchain",
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }
        const VulkanDispatch& dispatch = resolve_dispatch(context);
        if (dispatch.acquire_next_image_khr == nullptr) {
            return make_error(ErrorInit{
                .code = ErrorCode::FailedPrecondition,
                .domain = ErrorDomain::Vulkan,
                .detail = "acquire_next_image_bounded: acquire_next_image_khr is null",
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }

        // Reuse the bounded loop shape but acquire returns an image index.
        const auto start = resolve_now(context);
        const auto deadline_stall = start + policy.stall_notice;
        const auto deadline_quar = start + policy.quarantine_after;
        bool stall_published = false;
        auto& observer = resolve_observer(context);

        for (;;) {
            if (stop.stop_requested()) {
                return make_error(ErrorInit{
                    .code = ErrorCode::Cancelled,
                    .domain = ErrorDomain::Vulkan,
                    .detail = "acquire_next_image_bounded cancelled by stop_token",
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                });
            }
            if (resolve_shutdown(context)) {
                return make_error(ErrorInit{
                    .code = ErrorCode::Cancelled,
                    .domain = ErrorDomain::Vulkan,
                    .detail = "acquire_next_image_bounded interrupted by shutdown latch",
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                });
            }
            if (resolve_quarantined(context)) {
                return make_error(ErrorInit{
                    .code = ErrorCode::Unavailable,
                    .domain = ErrorDomain::Vulkan,
                    .detail = "acquire_next_image_bounded: owner already quarantined",
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                });
            }

            const auto now = resolve_now(context);
            if (now >= deadline_quar) {
                latch_quarantine(context);
                observer.on_quarantine(context.fingerprint);
                return make_error(ErrorInit{
                    .code = ErrorCode::Unavailable,
                    .domain = ErrorDomain::Vulkan,
                    .detail = "acquire_next_image_bounded quarantined after policy window",
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                });
            }

            const std::uint64_t timeout_ns =
                timeout_ns_for_slice(policy, now, deadline_quar);
            std::uint32_t image_index = std::numeric_limits<std::uint32_t>::max();
            const VkResult result = dispatch.acquire_next_image_khr(
                device, swapchain, timeout_ns, semaphore, fence, &image_index);

            if (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) {
                // SUBOPTIMAL is success-with-flag (AMB-B1); optional out bit.
                if (out_suboptimal != nullptr) {
                    *out_suboptimal = (result == VK_SUBOPTIMAL_KHR);
                }
                return image_index;
            }
            if (result == VK_TIMEOUT || result == VK_NOT_READY) {
                const auto after = resolve_now(context);
                if (after >= deadline_quar) {
                    latch_quarantine(context);
                    observer.on_quarantine(context.fingerprint);
                    return make_error(ErrorInit{
                        .code = ErrorCode::Unavailable,
                        .domain = ErrorDomain::Vulkan,
                        .detail = "acquire_next_image_bounded quarantined after policy window",
                        .detection = LFS_SOURCE_SITE_CURRENT(),
                    });
                }
                if (after >= deadline_stall && !stall_published) {
                    observer.on_stall(context.fingerprint);
                    stall_published = true;
                }
                continue;
            }
            if (result == VK_ERROR_DEVICE_LOST) {
                return vulkan_result_to_error(
                    result, "vkAcquireNextImageKHR", LFS_SOURCE_SITE_CURRENT());
            }
            return vulkan_result_to_error(
                result, "vkAcquireNextImageKHR", LFS_SOURCE_SITE_CURRENT());
        }
    }

} // namespace lfs::rendering
