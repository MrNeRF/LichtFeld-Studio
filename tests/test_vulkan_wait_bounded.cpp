/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Phase 7A: fake-clock bounded-wait matrix (spec §4.4) — GPU-free.

#include "rendering/vulkan_wait.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <stop_token>
#include <string>
#include <string_view>
#include <vector>

using namespace std::chrono_literals;
using lfs::ErrorCode;
using lfs::ErrorDomain;
using lfs::rendering::acquire_next_image_bounded;
using lfs::rendering::ClockNow;
using lfs::rendering::VulkanDispatch;
using lfs::rendering::VulkanWaitPolicy;
using lfs::rendering::wait_fence_bounded;
using lfs::rendering::wait_semaphores_bounded;
using lfs::rendering::WaitContext;
using lfs::rendering::WaitOutcome;

namespace {

    struct ScriptedObserver final : lfs::rendering::WaitObserver {
        int stall_count = 0;
        int quarantine_count = 0;
        std::string last_stall;
        std::string last_quarantine;

        void on_stall(const std::string_view fingerprint) noexcept override {
            ++stall_count;
            last_stall.assign(fingerprint);
        }
        void on_quarantine(const std::string_view fingerprint) noexcept override {
            ++quarantine_count;
            last_quarantine.assign(fingerprint);
        }
    };

    struct FakeClock {
        std::chrono::steady_clock::time_point t{};

        ClockNow fn() {
            return [this]() { return t; };
        }
        void advance(const std::chrono::milliseconds d) { t += d; }
    };

    // Process-local script for the PFN trampoline (serial unit tests only).
    struct FenceWaitScript {
        FakeClock* clock = nullptr;
        std::int64_t ready_at_ms = 0;
        int call_count = 0;
        int device_lost_at_call = -1;
        bool advance_on_timeout = true;
        std::vector<std::uint64_t> timeouts;

        static FenceWaitScript*& active() {
            static FenceWaitScript* ptr = nullptr;
            return ptr;
        }

        void bind() { active() = this; }
        void unbind() {
            if (active() == this) {
                active() = nullptr;
            }
        }

        VkResult on_wait(const uint64_t timeout) {
            timeouts.push_back(timeout);
            const int idx = call_count++;
            if (device_lost_at_call >= 0 && idx == device_lost_at_call) {
                return VK_ERROR_DEVICE_LOST;
            }
            if (clock != nullptr && advance_on_timeout && timeout > 0) {
                clock->t += std::chrono::nanoseconds(static_cast<std::int64_t>(timeout));
            }
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                        clock->t - std::chrono::steady_clock::time_point{})
                                        .count();
            if (elapsed_ms >= ready_at_ms) {
                return VK_SUCCESS;
            }
            return VK_TIMEOUT;
        }

        static VKAPI_ATTR VkResult VKAPI_CALL trampoline(VkDevice /*device*/,
                                                         uint32_t /*fenceCount*/,
                                                         const VkFence* /*pFences*/,
                                                         VkBool32 /*waitAll*/,
                                                         uint64_t timeout) {
            EXPECT_NE(active(), nullptr);
            return active()->on_wait(timeout);
        }
    };

    struct AcquireScript {
        FakeClock* clock = nullptr;
        std::int64_t ready_at_ms = 0;
        int call_count = 0;
        VkResult ready_result = VK_SUCCESS;
        std::uint32_t image_index = 7;
        int device_lost_at_call = -1;
        int out_of_date_at_call = -1;
        bool advance_on_timeout = true;

        static AcquireScript*& active() {
            static AcquireScript* ptr = nullptr;
            return ptr;
        }

        void bind() { active() = this; }
        void unbind() {
            if (active() == this) {
                active() = nullptr;
            }
        }

        VkResult on_acquire(const uint64_t timeout, uint32_t* pImageIndex) {
            const int idx = call_count++;
            if (device_lost_at_call >= 0 && idx == device_lost_at_call) {
                return VK_ERROR_DEVICE_LOST;
            }
            if (out_of_date_at_call >= 0 && idx == out_of_date_at_call) {
                return VK_ERROR_OUT_OF_DATE_KHR;
            }
            if (clock != nullptr && advance_on_timeout && timeout > 0) {
                clock->t += std::chrono::nanoseconds(static_cast<std::int64_t>(timeout));
            }
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                        clock->t - std::chrono::steady_clock::time_point{})
                                        .count();
            if (elapsed_ms >= ready_at_ms) {
                if (pImageIndex != nullptr) {
                    *pImageIndex = image_index;
                }
                return ready_result;
            }
            return VK_TIMEOUT;
        }

        static VKAPI_ATTR VkResult VKAPI_CALL trampoline(VkDevice /*device*/,
                                                         VkSwapchainKHR /*swapchain*/,
                                                         uint64_t timeout,
                                                         VkSemaphore /*semaphore*/,
                                                         VkFence /*fence*/,
                                                         uint32_t* pImageIndex) {
            EXPECT_NE(active(), nullptr);
            return active()->on_acquire(timeout, pImageIndex);
        }
    };

    struct SemaphoreWaitScript {
        FakeClock* clock = nullptr;
        std::int64_t ready_at_ms = 0;
        int call_count = 0;
        bool advance_on_timeout = true;

        static SemaphoreWaitScript*& active() {
            static SemaphoreWaitScript* ptr = nullptr;
            return ptr;
        }

        void bind() { active() = this; }
        void unbind() {
            if (active() == this) {
                active() = nullptr;
            }
        }

        VkResult on_wait(const uint64_t timeout) {
            const int idx = call_count++;
            (void)idx;
            if (clock != nullptr && advance_on_timeout && timeout > 0) {
                clock->t += std::chrono::nanoseconds(static_cast<std::int64_t>(timeout));
            }
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                        clock->t - std::chrono::steady_clock::time_point{})
                                        .count();
            if (elapsed_ms >= ready_at_ms) {
                return VK_SUCCESS;
            }
            return VK_TIMEOUT;
        }

        static VKAPI_ATTR VkResult VKAPI_CALL trampoline(VkDevice /*device*/,
                                                         const VkSemaphoreWaitInfo* /*pWaitInfo*/,
                                                         uint64_t timeout) {
            EXPECT_NE(active(), nullptr);
            return active()->on_wait(timeout);
        }
    };

    struct BindFenceScript {
        FenceWaitScript& script;
        explicit BindFenceScript(FenceWaitScript& s) : script(s) { script.bind(); }
        ~BindFenceScript() { script.unbind(); }
        BindFenceScript(const BindFenceScript&) = delete;
        BindFenceScript& operator=(const BindFenceScript&) = delete;
    };

    // Back-compat alias used by existing 7A tests.
    using BindScript = BindFenceScript;

    struct BindAcquireScript {
        AcquireScript& script;
        explicit BindAcquireScript(AcquireScript& s) : script(s) { script.bind(); }
        ~BindAcquireScript() { script.unbind(); }
        BindAcquireScript(const BindAcquireScript&) = delete;
        BindAcquireScript& operator=(const BindAcquireScript&) = delete;
    };

    struct BindSemaphoreScript {
        SemaphoreWaitScript& script;
        explicit BindSemaphoreScript(SemaphoreWaitScript& s) : script(s) { script.bind(); }
        ~BindSemaphoreScript() { script.unbind(); }
        BindSemaphoreScript(const BindSemaphoreScript&) = delete;
        BindSemaphoreScript& operator=(const BindSemaphoreScript&) = delete;
    };

    VulkanDispatch make_dispatch() {
        VulkanDispatch d{};
        d.wait_for_fences = &FenceWaitScript::trampoline;
        return d;
    }

    VulkanDispatch make_acquire_dispatch() {
        VulkanDispatch d{};
        d.acquire_next_image_khr = &AcquireScript::trampoline;
        return d;
    }

    VulkanDispatch make_semaphore_dispatch() {
        VulkanDispatch d{};
        d.wait_semaphores = &SemaphoreWaitScript::trampoline;
        return d;
    }

    // Opaque fake handles — never dereferenced by the wait primitive.
    VkDevice fake_device() {
        return reinterpret_cast<VkDevice>(static_cast<std::uintptr_t>(0xD1));
    }
    VkFence fake_fence() {
        return reinterpret_cast<VkFence>(static_cast<std::uintptr_t>(0xF1));
    }
    VkSwapchainKHR fake_swapchain() {
        return reinterpret_cast<VkSwapchainKHR>(static_cast<std::uintptr_t>(0x51));
    }
    VkSemaphore fake_semaphore() {
        return reinterpret_cast<VkSemaphore>(static_cast<std::uintptr_t>(0xA1));
    }

    // Spec §1 outcome mapping table (option b): locks the Phase 7B contract
    // without linking VulkanContext private helpers. Soft = no last_error;
    // fail = set last_error; recreate only for true OUT_OF_DATE.
    enum class MapAction : std::uint8_t {
        Continue,
        SoftFalse,
        Fail,
        FailAndLatch,
        SoftFalseRecreate, // OUT_OF_DATE only
    };

    [[nodiscard]] MapAction map_fence_outcome_for_begin_frame(const WaitOutcome o) {
        switch (o) {
        case WaitOutcome::Ready: return MapAction::Continue;
        case WaitOutcome::Cancelled:
        case WaitOutcome::Shutdown: return MapAction::SoftFalse;
        case WaitOutcome::Quarantined: return MapAction::Fail;
        }
        return MapAction::Fail;
    }

    [[nodiscard]] MapAction map_fence_error_for_begin_frame(const ErrorCode code) {
        if (code == ErrorCode::DeviceLost) {
            return MapAction::FailAndLatch;
        }
        return MapAction::Fail;
    }

    [[nodiscard]] MapAction map_acquire_error_for_begin_frame(const ErrorCode code,
                                                              const bool is_out_of_date) {
        if (is_out_of_date) {
            return MapAction::SoftFalseRecreate;
        }
        if (code == ErrorCode::Cancelled) {
            return MapAction::SoftFalse;
        }
        if (code == ErrorCode::DeviceLost) {
            return MapAction::FailAndLatch;
        }
        // Quarantine Unavailable and other hard errors.
        return MapAction::Fail;
    }

} // namespace

TEST(VulkanWaitBounded, ReadyWithin99msNoStallNotice) {
    FakeClock clock;
    FenceWaitScript script;
    script.clock = &clock;
    script.ready_at_ms = 99;
    BindScript bind(script);
    ScriptedObserver observer;

    const VulkanDispatch dispatch = make_dispatch();
    WaitContext ctx;
    ctx.dispatch = &dispatch;
    ctx.now = clock.fn();
    ctx.observer = &observer;
    ctx.fingerprint = "test.99ms";

    std::stop_source stop;
    auto result = wait_fence_bounded(fake_device(), fake_fence(), stop.get_token(),
                                     VulkanWaitPolicy{}, ctx);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, WaitOutcome::Ready);
    EXPECT_EQ(observer.stall_count, 0);
    EXPECT_EQ(observer.quarantine_count, 0);
    EXPECT_GE(script.call_count, 1);
}

TEST(VulkanWaitBounded, StallNoticeAt2sThenReady) {
    FakeClock clock;
    FenceWaitScript script;
    script.clock = &clock;
    script.ready_at_ms = 2100; // past 2s stall_notice, before 10s quarantine
    BindScript bind(script);
    ScriptedObserver observer;

    const VulkanDispatch dispatch = make_dispatch();
    WaitContext ctx;
    ctx.dispatch = &dispatch;
    ctx.now = clock.fn();
    ctx.observer = &observer;
    ctx.fingerprint = "test.2s";

    std::stop_source stop;
    auto result = wait_fence_bounded(fake_device(), fake_fence(), stop.get_token(),
                                     VulkanWaitPolicy{}, ctx);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, WaitOutcome::Ready);
    EXPECT_EQ(observer.stall_count, 1);
    EXPECT_EQ(observer.last_stall, "test.2s");
    EXPECT_EQ(observer.quarantine_count, 0);
}

TEST(VulkanWaitBounded, QuarantineAt10sRetainsAndDoesNotDestroy) {
    FakeClock clock;
    FenceWaitScript script;
    script.clock = &clock;
    script.ready_at_ms = 100'000; // never Ready within 10s
    BindScript bind(script);
    ScriptedObserver observer;
    std::atomic<bool> quarantine_flag{false};
    int destroy_spy = 0;

    const VulkanDispatch dispatch = make_dispatch();
    WaitContext ctx;
    ctx.dispatch = &dispatch;
    ctx.now = clock.fn();
    ctx.observer = &observer;
    ctx.fingerprint = "test.10s";
    ctx.owner_quarantine_flag = &quarantine_flag;
    ctx.quarantine_owner = [&]() {
        // Owner retains resources; only latches. Spy would fire on free.
        (void)destroy_spy;
    };

    std::stop_source stop;
    auto result = wait_fence_bounded(fake_device(), fake_fence(), stop.get_token(),
                                     VulkanWaitPolicy{}, ctx);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, WaitOutcome::Quarantined);
    EXPECT_EQ(observer.quarantine_count, 1);
    EXPECT_TRUE(quarantine_flag.load());
    EXPECT_EQ(destroy_spy, 0); // timeout never authorizes destruction
    EXPECT_GE(observer.stall_count, 1);
}

TEST(VulkanWaitBounded, StopTokenYieldsCancelled) {
    FakeClock clock;
    FenceWaitScript script;
    script.clock = &clock;
    script.ready_at_ms = 100'000;
    script.advance_on_timeout = false; // stay mid-loop
    BindScript bind(script);
    ScriptedObserver observer;

    // First call: TIMEOUT without advancing; second check sees stop requested.
    // Force stop before the wait so the loop exits on the first iteration.
    std::stop_source stop;
    stop.request_stop();

    const VulkanDispatch dispatch = make_dispatch();
    WaitContext ctx;
    ctx.dispatch = &dispatch;
    ctx.now = clock.fn();
    ctx.observer = &observer;

    auto result = wait_fence_bounded(fake_device(), fake_fence(), stop.get_token(),
                                     VulkanWaitPolicy{}, ctx);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, WaitOutcome::Cancelled);
    EXPECT_EQ(script.call_count, 0); // cancelled before any native wait
    EXPECT_EQ(observer.stall_count, 0);
}

TEST(VulkanWaitBounded, ShutdownLatchYieldsShutdown) {
    FakeClock clock;
    FenceWaitScript script;
    script.clock = &clock;
    script.ready_at_ms = 100'000;
    BindScript bind(script);
    ScriptedObserver observer;

    const VulkanDispatch dispatch = make_dispatch();
    WaitContext ctx;
    ctx.dispatch = &dispatch;
    ctx.now = clock.fn();
    ctx.observer = &observer;
    ctx.shutdown_latched = []() { return true; };

    std::stop_source stop;
    auto result = wait_fence_bounded(fake_device(), fake_fence(), stop.get_token(),
                                     VulkanWaitPolicy{}, ctx);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, WaitOutcome::Shutdown);
    EXPECT_EQ(script.call_count, 0);
}

TEST(VulkanWaitBounded, DeviceLostIsResultErrorNotWaitOutcome) {
    FakeClock clock;
    FenceWaitScript script;
    script.clock = &clock;
    script.ready_at_ms = 100'000;
    script.device_lost_at_call = 0;
    BindScript bind(script);
    ScriptedObserver observer;

    const VulkanDispatch dispatch = make_dispatch();
    WaitContext ctx;
    ctx.dispatch = &dispatch;
    ctx.now = clock.fn();
    ctx.observer = &observer;

    std::stop_source stop;
    auto result = wait_fence_bounded(fake_device(), fake_fence(), stop.get_token(),
                                     VulkanWaitPolicy{}, ctx);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), ErrorCode::DeviceLost);
    EXPECT_EQ(result.error().domain(), ErrorDomain::Vulkan);
    ASSERT_TRUE(result.error().native().has_value());
    EXPECT_EQ(result.error().native()->code,
              static_cast<std::int64_t>(VK_ERROR_DEVICE_LOST));
}

TEST(VulkanWaitBounded, PolicyDefaultsMatchPinnedContract) {
    const VulkanWaitPolicy policy{};
    EXPECT_EQ(policy.slice, 100ms);
    EXPECT_EQ(policy.stall_notice, 2s);
    EXPECT_EQ(policy.quarantine_after, 10s);
}

TEST(VulkanWaitBounded, AlreadyQuarantinedReturnsQuarantinedWithoutWait) {
    FakeClock clock;
    FenceWaitScript script;
    script.clock = &clock;
    BindScript bind(script);
    ScriptedObserver observer;
    std::atomic<bool> flag{true};

    const VulkanDispatch dispatch = make_dispatch();
    WaitContext ctx;
    ctx.dispatch = &dispatch;
    ctx.now = clock.fn();
    ctx.observer = &observer;
    ctx.owner_quarantine_flag = &flag;

    std::stop_source stop;
    auto result = wait_fence_bounded(fake_device(), fake_fence(), stop.get_token(),
                                     VulkanWaitPolicy{}, ctx);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, WaitOutcome::Quarantined);
    EXPECT_EQ(script.call_count, 0);
}

// ---------------------------------------------------------------------------
// Phase 7B: UI-frame fingerprints + acquire out_suboptimal + mapping table
// ---------------------------------------------------------------------------

TEST(VulkanWaitBounded, UiFrameFenceHeartbeatAt2sUsesContextFingerprint) {
    FakeClock clock;
    FenceWaitScript script;
    script.clock = &clock;
    script.ready_at_ms = 2100;
    BindScript bind(script);
    ScriptedObserver observer;
    std::atomic<bool> quarantine{false};
    std::atomic<bool> shutdown{false};

    const VulkanDispatch dispatch = make_dispatch();
    WaitContext ctx;
    ctx.dispatch = &dispatch;
    ctx.now = clock.fn();
    ctx.observer = &observer;
    ctx.fingerprint = "vulkan.context.frame_fence";
    ctx.owner_quarantine_flag = &quarantine;
    ctx.shutdown_latched = [&]() { return shutdown.load(); };
    ctx.is_quarantined = [&]() { return quarantine.load(); };

    std::stop_source stop;
    auto result = wait_fence_bounded(fake_device(), fake_fence(), stop.get_token(),
                                     VulkanWaitPolicy{}, ctx);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, WaitOutcome::Ready);
    EXPECT_EQ(observer.stall_count, 1);
    EXPECT_EQ(observer.last_stall, "vulkan.context.frame_fence");
    EXPECT_EQ(observer.quarantine_count, 0);
}

TEST(VulkanWaitBounded, UiFrameFenceCancelledAndShutdownMapSoft) {
    EXPECT_EQ(map_fence_outcome_for_begin_frame(WaitOutcome::Cancelled), MapAction::SoftFalse);
    EXPECT_EQ(map_fence_outcome_for_begin_frame(WaitOutcome::Shutdown), MapAction::SoftFalse);
    EXPECT_EQ(map_fence_outcome_for_begin_frame(WaitOutcome::Quarantined), MapAction::Fail);
    EXPECT_EQ(map_fence_outcome_for_begin_frame(WaitOutcome::Ready), MapAction::Continue);
    EXPECT_EQ(map_fence_error_for_begin_frame(ErrorCode::DeviceLost), MapAction::FailAndLatch);
    EXPECT_EQ(map_fence_error_for_begin_frame(ErrorCode::Internal), MapAction::Fail);
}

TEST(VulkanWaitBounded, UiFrameFenceShutdownLatchViaContext) {
    FakeClock clock;
    FenceWaitScript script;
    script.clock = &clock;
    script.ready_at_ms = 100'000;
    BindScript bind(script);
    ScriptedObserver observer;
    std::atomic<bool> shutdown{true};

    const VulkanDispatch dispatch = make_dispatch();
    WaitContext ctx;
    ctx.dispatch = &dispatch;
    ctx.now = clock.fn();
    ctx.observer = &observer;
    ctx.fingerprint = "vulkan.context.current_slot";
    ctx.shutdown_latched = [&]() { return shutdown.load(); };

    std::stop_source stop;
    auto result = wait_fence_bounded(fake_device(), fake_fence(), stop.get_token(),
                                     VulkanWaitPolicy{}, ctx);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, WaitOutcome::Shutdown);
    EXPECT_EQ(map_fence_outcome_for_begin_frame(*result), MapAction::SoftFalse);
    EXPECT_EQ(script.call_count, 0);
}

TEST(VulkanWaitBounded, AcquireOutSuboptimalFalseOnSuccess) {
    FakeClock clock;
    AcquireScript script;
    script.clock = &clock;
    script.ready_at_ms = 0;
    script.ready_result = VK_SUCCESS;
    script.image_index = 3;
    BindAcquireScript bind(script);
    ScriptedObserver observer;

    const VulkanDispatch dispatch = make_acquire_dispatch();
    WaitContext ctx;
    ctx.dispatch = &dispatch;
    ctx.now = clock.fn();
    ctx.observer = &observer;
    ctx.fingerprint = "vulkan.context.acquire";

    bool suboptimal = true; // must be cleared on success
    std::stop_source stop;
    auto result = acquire_next_image_bounded(
        fake_device(), fake_swapchain(), fake_semaphore(), VK_NULL_HANDLE,
        stop.get_token(), VulkanWaitPolicy{}, ctx, &suboptimal);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 3u);
    EXPECT_FALSE(suboptimal);
}

TEST(VulkanWaitBounded, AcquireOutSuboptimalTrueOnSuboptimalKhr) {
    FakeClock clock;
    AcquireScript script;
    script.clock = &clock;
    script.ready_at_ms = 0;
    script.ready_result = VK_SUBOPTIMAL_KHR;
    script.image_index = 5;
    BindAcquireScript bind(script);
    ScriptedObserver observer;

    const VulkanDispatch dispatch = make_acquire_dispatch();
    WaitContext ctx;
    ctx.dispatch = &dispatch;
    ctx.now = clock.fn();
    ctx.observer = &observer;
    ctx.fingerprint = "vulkan.context.acquire";

    bool suboptimal = false;
    std::stop_source stop;
    auto result = acquire_next_image_bounded(
        fake_device(), fake_swapchain(), fake_semaphore(), VK_NULL_HANDLE,
        stop.get_token(), VulkanWaitPolicy{}, ctx, &suboptimal);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 5u);
    EXPECT_TRUE(suboptimal);
    // SUBOPTIMAL is success-with-flag, not ErrorCode::Unavailable.
}

TEST(VulkanWaitBounded, AcquireNullOutSuboptimalStillReturnsIndex) {
    FakeClock clock;
    AcquireScript script;
    script.clock = &clock;
    script.ready_at_ms = 0;
    script.ready_result = VK_SUBOPTIMAL_KHR;
    script.image_index = 1;
    BindAcquireScript bind(script);

    const VulkanDispatch dispatch = make_acquire_dispatch();
    WaitContext ctx;
    ctx.dispatch = &dispatch;
    ctx.now = clock.fn();

    std::stop_source stop;
    auto result = acquire_next_image_bounded(
        fake_device(), fake_swapchain(), fake_semaphore(), VK_NULL_HANDLE,
        stop.get_token(), VulkanWaitPolicy{}, ctx, nullptr);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, 1u);
}

TEST(VulkanWaitBounded, AcquireHeartbeatAt2sThenReady) {
    FakeClock clock;
    AcquireScript script;
    script.clock = &clock;
    script.ready_at_ms = 2100;
    script.ready_result = VK_SUCCESS;
    BindAcquireScript bind(script);
    ScriptedObserver observer;

    const VulkanDispatch dispatch = make_acquire_dispatch();
    WaitContext ctx;
    ctx.dispatch = &dispatch;
    ctx.now = clock.fn();
    ctx.observer = &observer;
    ctx.fingerprint = "vulkan.context.acquire";

    std::stop_source stop;
    auto result = acquire_next_image_bounded(
        fake_device(), fake_swapchain(), fake_semaphore(), VK_NULL_HANDLE,
        stop.get_token(), VulkanWaitPolicy{}, ctx, nullptr);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(observer.stall_count, 1);
    EXPECT_EQ(observer.last_stall, "vulkan.context.acquire");
    EXPECT_EQ(observer.quarantine_count, 0);
}

TEST(VulkanWaitBounded, AcquireQuarantineAt10sIsNotOutOfDate) {
    FakeClock clock;
    AcquireScript script;
    script.clock = &clock;
    script.ready_at_ms = 100'000;
    BindAcquireScript bind(script);
    ScriptedObserver observer;
    std::atomic<bool> quarantine{false};

    const VulkanDispatch dispatch = make_acquire_dispatch();
    WaitContext ctx;
    ctx.dispatch = &dispatch;
    ctx.now = clock.fn();
    ctx.observer = &observer;
    ctx.fingerprint = "vulkan.context.acquire";
    ctx.owner_quarantine_flag = &quarantine;

    std::stop_source stop;
    auto result = acquire_next_image_bounded(
        fake_device(), fake_swapchain(), fake_semaphore(), VK_NULL_HANDLE,
        stop.get_token(), VulkanWaitPolicy{}, ctx, nullptr);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), ErrorCode::Unavailable);
    EXPECT_FALSE(result.error().native().has_value()); // quarantine has no VkResult
    EXPECT_TRUE(quarantine.load());
    EXPECT_EQ(observer.quarantine_count, 1);
    EXPECT_EQ(map_acquire_error_for_begin_frame(result.error().code(),
                                                /*is_out_of_date=*/false),
              MapAction::Fail);
}

TEST(VulkanWaitBounded, AcquireOutOfDateMapsToSoftRecreate) {
    FakeClock clock;
    AcquireScript script;
    script.clock = &clock;
    script.ready_at_ms = 0;
    script.out_of_date_at_call = 0;
    BindAcquireScript bind(script);

    const VulkanDispatch dispatch = make_acquire_dispatch();
    WaitContext ctx;
    ctx.dispatch = &dispatch;
    ctx.now = clock.fn();
    ctx.fingerprint = "vulkan.context.acquire";

    std::stop_source stop;
    auto result = acquire_next_image_bounded(
        fake_device(), fake_swapchain(), fake_semaphore(), VK_NULL_HANDLE,
        stop.get_token(), VulkanWaitPolicy{}, ctx, nullptr);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code(), ErrorCode::Unavailable);
    ASSERT_TRUE(result.error().native().has_value());
    EXPECT_EQ(result.error().native()->code,
              static_cast<std::int64_t>(VK_ERROR_OUT_OF_DATE_KHR));
    const bool is_ood =
        result.error().native().has_value() &&
        static_cast<VkResult>(result.error().native()->code) == VK_ERROR_OUT_OF_DATE_KHR;
    EXPECT_EQ(map_acquire_error_for_begin_frame(result.error().code(), is_ood),
              MapAction::SoftFalseRecreate);
}

TEST(VulkanWaitBounded, AcquireCancelledAndShutdownMapSoftNoRecreate) {
    FakeClock clock;
    AcquireScript script;
    script.clock = &clock;
    script.ready_at_ms = 100'000;
    BindAcquireScript bind(script);

    {
        const VulkanDispatch dispatch = make_acquire_dispatch();
        WaitContext ctx;
        ctx.dispatch = &dispatch;
        ctx.now = clock.fn();
        std::stop_source stop;
        stop.request_stop();
        auto result = acquire_next_image_bounded(
            fake_device(), fake_swapchain(), fake_semaphore(), VK_NULL_HANDLE,
            stop.get_token(), VulkanWaitPolicy{}, ctx, nullptr);
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code(), ErrorCode::Cancelled);
        EXPECT_EQ(map_acquire_error_for_begin_frame(result.error().code(), false),
                  MapAction::SoftFalse);
    }
    {
        const VulkanDispatch dispatch = make_acquire_dispatch();
        WaitContext ctx;
        ctx.dispatch = &dispatch;
        ctx.now = clock.fn();
        ctx.shutdown_latched = []() { return true; };
        std::stop_source stop;
        auto result = acquire_next_image_bounded(
            fake_device(), fake_swapchain(), fake_semaphore(), VK_NULL_HANDLE,
            stop.get_token(), VulkanWaitPolicy{}, ctx, nullptr);
        ASSERT_FALSE(result.has_value());
        EXPECT_EQ(result.error().code(), ErrorCode::Cancelled);
        EXPECT_EQ(map_acquire_error_for_begin_frame(result.error().code(), false),
                  MapAction::SoftFalse);
    }
}

TEST(VulkanWaitBounded, ValidationSemaphoreWaitObserveFingerprintAndQuarantine) {
    FakeClock clock;
    SemaphoreWaitScript script;
    script.clock = &clock;
    script.ready_at_ms = 100'000;
    BindSemaphoreScript bind(script);
    ScriptedObserver observer;
    std::atomic<bool> quarantine{false};

    const VulkanDispatch dispatch = make_semaphore_dispatch();
    WaitContext ctx;
    ctx.dispatch = &dispatch;
    ctx.now = clock.fn();
    ctx.observer = &observer;
    ctx.fingerprint = "vulkan.context.validation.wait_observe";
    ctx.owner_quarantine_flag = &quarantine;

    VkSemaphore sem = fake_semaphore();
    std::uint64_t value = 42;
    VkSemaphoreWaitInfo wait_info{};
    wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wait_info.semaphoreCount = 1;
    wait_info.pSemaphores = &sem;
    wait_info.pValues = &value;

    std::stop_source stop;
    auto result = wait_semaphores_bounded(fake_device(), wait_info, stop.get_token(),
                                          VulkanWaitPolicy{}, ctx);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, WaitOutcome::Quarantined);
    EXPECT_TRUE(quarantine.load());
    EXPECT_EQ(observer.last_quarantine, "vulkan.context.validation.wait_observe");
    EXPECT_GE(observer.stall_count, 1);
    EXPECT_EQ(observer.last_stall, "vulkan.context.validation.wait_observe");
}

TEST(VulkanWaitBounded, ValidationSemaphoreSignalPublishHeartbeatAt2s) {
    FakeClock clock;
    SemaphoreWaitScript script;
    script.clock = &clock;
    script.ready_at_ms = 2100;
    BindSemaphoreScript bind(script);
    ScriptedObserver observer;

    const VulkanDispatch dispatch = make_semaphore_dispatch();
    WaitContext ctx;
    ctx.dispatch = &dispatch;
    ctx.now = clock.fn();
    ctx.observer = &observer;
    ctx.fingerprint = "vulkan.context.validation.signal_publish";

    VkSemaphore sem = fake_semaphore();
    std::uint64_t value = 7;
    VkSemaphoreWaitInfo wait_info{};
    wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wait_info.semaphoreCount = 1;
    wait_info.pSemaphores = &sem;
    wait_info.pValues = &value;

    std::stop_source stop;
    auto result = wait_semaphores_bounded(fake_device(), wait_info, stop.get_token(),
                                          VulkanWaitPolicy{}, ctx);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, WaitOutcome::Ready);
    EXPECT_EQ(observer.stall_count, 1);
    EXPECT_EQ(observer.last_stall, "vulkan.context.validation.signal_publish");
}

TEST(VulkanWaitBounded, ImageFenceHardErrorSetsResizeFlagQuarantineDoesNot) {
    // Spec site 3: framebuffer_resized_ only on hard Error, never on
    // Quarantined/Cancelled/Shutdown.
    EXPECT_EQ(map_fence_outcome_for_begin_frame(WaitOutcome::Quarantined), MapAction::Fail);
    EXPECT_EQ(map_fence_outcome_for_begin_frame(WaitOutcome::Cancelled), MapAction::SoftFalse);
    // Resize flag is a site-3 compose rule applied only for Error results.
    EXPECT_EQ(map_fence_error_for_begin_frame(ErrorCode::Internal), MapAction::Fail);
    EXPECT_EQ(map_fence_error_for_begin_frame(ErrorCode::DeviceLost), MapAction::FailAndLatch);
}
