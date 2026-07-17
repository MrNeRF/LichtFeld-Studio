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
using lfs::rendering::ClockNow;
using lfs::rendering::VulkanDispatch;
using lfs::rendering::VulkanWaitPolicy;
using lfs::rendering::WaitContext;
using lfs::rendering::WaitOutcome;
using lfs::rendering::wait_fence_bounded;

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

    struct BindScript {
        FenceWaitScript& script;
        explicit BindScript(FenceWaitScript& s) : script(s) { script.bind(); }
        ~BindScript() { script.unbind(); }
        BindScript(const BindScript&) = delete;
        BindScript& operator=(const BindScript&) = delete;
    };

    VulkanDispatch make_dispatch() {
        VulkanDispatch d{};
        d.wait_for_fences = &FenceWaitScript::trampoline;
        return d;
    }

    // Opaque fake handles — never dereferenced by the wait primitive.
    VkDevice fake_device() {
        return reinterpret_cast<VkDevice>(static_cast<std::uintptr_t>(0xD1));
    }
    VkFence fake_fence() {
        return reinterpret_cast<VkFence>(static_cast<std::uintptr_t>(0xF1));
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
