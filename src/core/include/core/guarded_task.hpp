/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error.hpp"
#include "core/error_codes.hpp"
#include "core/export.hpp"
#include "core/source_site.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

// lfs::core::run_guarded — the thread/task-boundary adapter. See
// .codex_tmp/error-architecture-analysis.md Section 5.7 (design prose) and
// Section 7.3 (frozen TaskBody/TaskCompletion/run_guarded signature) and
// .codex_tmp/phase-4a-run-guarded-spec.md Section 0 for the reasoning behind
// every choice not shown in that pseudocode, including the owner-approved
// amendment (spec Section 0.3) that drops `noexcept` from TaskCompletion's
// call signature. Summary of the load-bearing invariants:
//   * body() may throw anything; run_guarded normalizes lfs::Exception,
//     std::exception, and unknown exceptions into a Result<T> exactly once,
//     then invokes complete() exactly once with that Result.
//   * complete's declared type (TaskCompletion<T>) is NOT noexcept-qualified
//     (spec Section 0.3): a throw escaping the stored callable is caught by
//     run_guarded, reported through ErrorReporter's fixed ProcessBoundary
//     fallback, and the task still reaches Settled. A completion body is
//     still documented to be non-throwing in practice — this path exists so
//     a violation degrades instead of terminating the process.
//   * The Pending/Completing/Settled settlement (Section 7.3) is owned by
//     run_guarded via TaskContext::settlement: a fresh, call-local one when
//     absent, or a shared one (set by the packaged posted-work primitive,
//     visualizer/post_work_utils.hpp) when two alternative bodies — a
//     WorkItem's run()/cancel() — must settle exactly once between them.
//   * A set expected_thread is asserted (LFS_ASSERT_MSG, always-on) before
//     body() runs. This is a programmer-contract check, not a task failure;
//     like every other LFS_ASSERT_MSG site it throws on violation, and since
//     run_guarded is noexcept, a real violation terminates the process
//     immediately rather than being silently absorbed.
namespace lfs::core {

    // Three-state settlement for one logical task (Section 7.3: "run_guarded
    // owns a three-state atomic settlement {Pending, Completing, Settled}").
    // Exactly one caller ever transitions Pending -> Completing; a second
    // caller sharing the same TaskSettlement (see TaskContext::settlement)
    // observes the failed claim and must not run its body or invoke
    // complete.
    enum class TaskSettlementState : std::uint8_t { Pending,
                                                    Completing,
                                                    Settled };

    class TaskSettlement {
    public:
        TaskSettlement() noexcept = default;
        TaskSettlement(const TaskSettlement&) = delete;
        TaskSettlement& operator=(const TaskSettlement&) = delete;

        // True iff this call is the one that must run the body and invoke
        // completion. False means another call already claimed this task
        // (already Completing or Settled) — the caller must return without
        // running its body or touching complete.
        [[nodiscard]] bool try_begin_completing() noexcept {
            TaskSettlementState expected = TaskSettlementState::Pending;
            return state_.compare_exchange_strong(expected, TaskSettlementState::Completing,
                                                  std::memory_order_acq_rel,
                                                  std::memory_order_acquire);
        }

        void mark_settled() noexcept {
            state_.store(TaskSettlementState::Settled, std::memory_order_release);
        }

    private:
        std::atomic<TaskSettlementState> state_{TaskSettlementState::Pending};
    };

    // One owning-action's context for a run_guarded call. Aggregate, not
    // polymorphic: cheap to construct per task, copyable (name and the
    // settlement shared_ptr are the only non-trivial members) so the
    // packaged posted-work primitive may share one context — and therefore
    // one TaskSettlement — across its run()/cancel() alternative bodies.
    struct TaskContext {
        // Task family for logs/context frames, e.g. "training-worker",
        // "gui.viewer-queue", "tcp.responder-thread". Becomes the operation
        // string in with_context() for every normalized Error.
        std::string name;

        // Domain used only when normalizing a std::exception/unknown throw
        // (an lfs::Exception already carries its own domain and passes
        // through unchanged).
        ErrorDomain domain = ErrorDomain::Core;

        // Generated once at the owning boundary (OperationId::generate()) —
        // callers write this explicitly; there is no implicit default,
        // matching ErrorInit's existing style (no implicit SourceSite
        // default either, for the same reason: LFS_SOURCE_SITE_CURRENT()
        // must be evaluated at the actual call site, not at this struct's
        // definition point).
        OperationId operation_id;

        // Launch site, always LFS_SOURCE_SITE_CURRENT() written explicitly
        // by the caller constructing this TaskContext.
        core::SourceSite site;

        // nullopt = no thread-affinity check. When set, run_guarded asserts
        // std::this_thread::get_id() == *expected_thread before invoking
        // body().
        std::optional<std::thread::id> expected_thread;

        // nullptr = run_guarded creates and owns a private TaskSettlement
        // for the duration of this one call (every single-shot call site in
        // this phase). Non-null = the caller (post_guarded_and_wait) shares
        // one TaskSettlement across two alternative run_guarded calls so
        // exactly one of them settles (spec Section 0.4).
        std::shared_ptr<TaskSettlement> settlement;
    };

    template <class T>
    using TaskBody = std::move_only_function<Result<T>()>;

    // NOT noexcept-qualified — spec Section 0.3 records this as an
    // owner-approved amendment to 7.3's original noexcept-qualified alias,
    // which was internally self-contradictory (a throw through a noexcept
    // call signature terminates the process before run_guarded's own
    // try/catch could run, making "route to the fixed fallback" and
    // "process does not terminate" both false at once).
    template <class T>
    using TaskCompletion = std::move_only_function<void(Result<T>&&)>;

    // Wraps error in Result<T>'s failure state, dispatching on whether T is
    // void (Result<void>::failure(...)) or a value type (Result<T>(Error)) —
    // the same void-vs-T split from_legacy_expected<T> already uses. Shared
    // (not run_guarded-private) because the packaged posted-work primitive
    // (post_guarded_and_wait, visualizer/post_work_utils.hpp) needs the same
    // dispatch for its cancellation/rejection results.
    template <class T>
    [[nodiscard]] Result<T> failure_result(Error error) noexcept {
        if constexpr (std::is_void_v<T>) {
            return Result<T>::failure(std::move(error));
        } else {
            static_assert(std::is_nothrow_move_constructible_v<T>,
                          "run_guarded<T>'s Result<T> must cross the body/complete "
                          "boundary without throwing");
            return Result<T>(std::move(error));
        }
    }

    namespace detail {

        // Out-of-line (guarded_task.cpp): classifies the currently-active
        // exception via rethrow-dispatch into an Error, tagging it with
        // context's domain/operation_id/site/name. Must be called only from
        // directly within a catch block (uses `throw;`). Never throws.
        [[nodiscard]] LFS_CORE_API Error normalize_current_exception(const TaskContext& context) noexcept;

        template <class T>
        [[nodiscard]] Result<T> task_failure_from_current_exception(const TaskContext& context) noexcept {
            return failure_result<T>(normalize_current_exception(context));
        }

        // The fixed fallback for a completion sink that violates its
        // documented no-throw contract (spec Section 0.3). Reuses
        // ErrorReporter's existing ProcessBoundary guarantee (Phase 2)
        // instead of inventing a second fixed-buffer writer. Never throws.
        LFS_CORE_API void report_completion_violation(const TaskContext& context) noexcept;

    } // namespace detail

    template <class T>
    void run_guarded(TaskContext context, TaskBody<T> body, TaskCompletion<T> complete) noexcept {
        static_assert(std::is_void_v<T> || std::is_nothrow_move_constructible_v<T>,
                      "run_guarded<T>: T must be nothrow-move-constructible to "
                      "cross the body/complete boundary safely");

        if (context.expected_thread) {
            LFS_ASSERT_MSG(*context.expected_thread == std::this_thread::get_id(),
                           "run_guarded: body invoked on unexpected thread");
        }

        TaskSettlement local_settlement;
        TaskSettlement& settlement = context.settlement ? *context.settlement : local_settlement;

        if (!settlement.try_begin_completing()) {
            return; // another call sharing this settlement already claimed it
        }

        Result<T> result = [&context, &body]() noexcept -> Result<T> {
            try {
                return std::move(body)();
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): this is the task boundary itself;
                // every exception is normalized into the task's Result<T>.
                return detail::task_failure_from_current_exception<T>(context);
            }
        }();

        try {
            std::move(complete)(std::move(result));
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): a broken completion sink degrades to
            // the fixed ProcessBoundary fallback and settlement still finishes.
            detail::report_completion_violation(context);
        }

        settlement.mark_settled();
    }

} // namespace lfs::core
