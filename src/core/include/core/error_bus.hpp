/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "core/export.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// lfs::ErrorBus — the native, C++-first surfacing bus for the error
// architecture (Phase 8; see .codex_tmp/phase-8-errorbus-spec.md). Publishers
// across modules translate a failure into an ErrorNotification and publish it;
// a native RmlUi consumer (registered before Python and available when Python
// is Failed) renders it. When no consumer delivers (startup/shutdown/headless),
// publish routes the error through ErrorReporter's durable fallback instead of
// dropping it. publish never blocks a worker on the GUI thread: the consumer's
// on_error only enqueues to a thread-safe queue drained on the UI frame.
//
// The public ErrorSurface / ErrorNotification / ErrorBus shape is frozen
// verbatim by the master architecture doc (Section 7.5); do not reshape it.
namespace lfs {

    // FROZEN (Section 7.5). Which native surface renders a notification. Phase 8
    // P1 renders Modal only; Toast/StatusOnly/Panel land in P2 (until then the
    // consumer falls Toast/Panel back to Modal and treats StatusOnly as silent).
    enum class ErrorSurface : std::uint8_t {
        Modal,
        Toast,
        Panel,
        StatusOnly,
    };

    // A default label is derived from the kind when `label` is empty; the
    // consumer maps each action to one modal/toast button.
    enum class ErrorActionKind : std::uint8_t {
        Dismiss,
        Retry,
        ChoosePath,
        OpenLog,
        StopRenderer,
        Custom,
    };

    // One offered recovery/branch. `on_invoke` runs on the UI thread when the
    // matching button is pressed; per the frozen contract it starts a new
    // operation id and must not mutate the source error (the notification's
    // error is const and copied into the closure by value if needed).
    struct ErrorAction {
        ErrorActionKind kind = ErrorActionKind::Dismiss;
        std::string label;
        std::function<void()> on_invoke;
    };

    // FROZEN (Section 7.5). `error` is a required lfs::Error, never a success
    // handle. `operation_id` correlates repeated faults for dedup.
    struct ErrorNotification {
        Error error;
        ErrorSurface surface = ErrorSurface::Modal;
        std::vector<ErrorAction> actions;
        OperationId operation_id;
    };

    // Implemented by the native GUI consumer. on_error is called synchronously
    // on the PUBLISHING (worker) thread and must be noexcept, non-blocking, and
    // enqueue-only to a thread-safe queue drained on the UI frame. It must never
    // touch RmlUi documents on the worker thread.
    class LFS_CORE_API NativeErrorConsumer {
    public:
        virtual ~NativeErrorConsumer() = default;
        virtual void on_error(const ErrorNotification& notification) noexcept = 0;
    };

    class ErrorBus;

    // RAII unsubscribe handle (SubscriptionToken shape). Destroying it removes
    // the consumer from the bus. The consumer must outlive its Subscription.
    class LFS_CORE_API Subscription {
    public:
        Subscription() noexcept = default;
        Subscription(ErrorBus* bus, std::uint64_t id) noexcept;
        ~Subscription();

        Subscription(const Subscription&) = delete;
        Subscription& operator=(const Subscription&) = delete;

        Subscription(Subscription&& other) noexcept;
        Subscription& operator=(Subscription&& other) noexcept;

        void reset() noexcept;
        [[nodiscard]] bool active() const noexcept { return bus_ != nullptr; }

    private:
        ErrorBus* bus_ = nullptr;
        std::uint64_t id_ = 0;
    };

    class LFS_CORE_API ErrorBus {
    public:
        // Process-wide bus. Singleton for the same exe/Python-module reason
        // EventBridge is: publishers span modules and must reach one instance
        // across the module boundary. Defined in the .cpp (liblfs_core) so there
        // is exactly one instance.
        static ErrorBus& instance();

        void publish(ErrorNotification notification) noexcept;
        [[nodiscard]] Subscription subscribe(NativeErrorConsumer& consumer);

        // Test-only: an isolated bus with its own subscriber and dedup state, so
        // unit tests do not share the process singleton's window/subscribers.
        [[nodiscard]] static std::unique_ptr<ErrorBus> create_isolated_for_testing();

        ErrorBus(const ErrorBus&) = delete;
        ErrorBus& operator=(const ErrorBus&) = delete;

    private:
        ErrorBus() = default;

        void unsubscribe(std::uint64_t id) noexcept;
        // NOT noexcept: the dedup map insert can bad_alloc under extreme OOM,
        // and publish()'s catch-all must be able to catch it and degrade rather
        // than terminate (a noexcept boundary here would defeat that contract).
        [[nodiscard]] bool should_suppress(std::uint64_t key);

        friend class Subscription;

        struct Subscriber {
            std::uint64_t id = 0;
            NativeErrorConsumer* consumer = nullptr;
        };

        struct DedupEntry {
            std::uint32_t count = 0;
            std::chrono::steady_clock::time_point last{};
        };

        std::mutex subscribers_mutex_;
        std::vector<Subscriber> subscribers_;
        std::uint64_t next_id_ = 1;

        std::mutex dedup_mutex_;
        std::unordered_map<std::uint64_t, DedupEntry> dedup_;
    };

} // namespace lfs
