/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/guarded_task.hpp"
#include "visualizer/visualizer.hpp"

#include <atomic>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

namespace lfs::vis {

    // Runs work on a posted-work queue and carries either its value or exception
    // back to the submitting thread. WorkItem::run must never leak an exception:
    // viewer queues execute on the GUI thread and are an exception boundary.
    template <typename PostFn, typename F, typename CancelFn>
    auto post_work_and_wait(PostFn&& post_fn, F&& fn, CancelFn&& cancel_fn) {
        using Result = std::invoke_result_t<F>;
        static_assert(std::is_same_v<Result, std::invoke_result_t<CancelFn>>,
                      "work and cancellation handlers must return the same type");

        auto task = std::make_shared<std::decay_t<F>>(std::forward<F>(fn));
        auto cancel_task = std::make_shared<std::decay_t<CancelFn>>(std::forward<CancelFn>(cancel_fn));
        auto promise = std::make_shared<std::promise<Result>>();
        auto completed = std::make_shared<std::atomic_bool>(false);
        auto future = promise->get_future();

        auto finish = [promise, completed](auto& callable) mutable {
            if (completed->exchange(true))
                return;

            try {
                if constexpr (std::is_void_v<Result>) {
                    std::invoke(callable);
                    promise->set_value();
                } else {
                    promise->set_value(std::invoke(callable));
                }
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        };

        const bool posted = std::invoke(
            std::forward<PostFn>(post_fn),
            Visualizer::WorkItem{
                .run = [task, finish]() mutable { finish(*task); },
                .cancel = [cancel_task, finish]() mutable { finish(*cancel_task); },
            });

        if (!posted)
            finish(*cancel_task);

        if constexpr (std::is_void_v<Result>) {
            future.get();
        } else {
            return future.get();
        }
    }

    // The Phase 4A packaged posted-work primitive. Guarantees a WorkItem
    // posted through Visualizer::postWork settles its Result<T> exactly
    // once, and that a throwing body normalizes through run_guarded instead
    // of leaving the waiting future abandoned — the exact gap
    // post_clear_scene_to_viewer had (module.cpp:141-175, fixed in Section
    // 3.1 of the spec). `cancelled_error` becomes the result if postWork
    // rejects the item outright, or if the item is cancelled instead of run
    // (viewer shutdown, or an earlier queued item's failure).
    //
    // The run()/cancel() alternative bodies below share one TaskContext —
    // and therefore one TaskSettlement (spec Section 0.4) — instead of a
    // separate atomic claim flag: run_guarded's own Pending/Completing/
    // Settled machinery already guarantees exactly one of them proceeds, so
    // this primitive does not need a second, parallel exactly-once
    // mechanism. This is defense-in-depth, not a fix for an observed race:
    // today's WorkItem contract already guarantees exactly one of
    // run()/cancel() fires per posted item (spec Section 0.9) — sharing the
    // settlement makes that guarantee enforced by the type system, not only
    // by the queue's convention (Section 5.6).
    template <class T>
    [[nodiscard]] lfs::Result<T> post_guarded_and_wait(
        Visualizer& viewer,
        core::TaskContext context,
        core::TaskBody<T> body,
        lfs::Error cancelled_error) {

        if (!context.settlement) {
            context.settlement = std::make_shared<core::TaskSettlement>();
        }

        if (viewer.isOnViewerThread()) {
            if (!viewer.acceptsPostedWork()) {
                return core::failure_result<T>(std::move(cancelled_error));
            }
            std::optional<lfs::Result<T>> settled;
            core::run_guarded<T>(context, std::move(body),
                                 [&settled](lfs::Result<T>&& r) { settled = std::move(r); });
            return std::move(*settled);
        }

        auto promise = std::make_shared<std::promise<lfs::Result<T>>>();
        auto shared_body = std::make_shared<core::TaskBody<T>>(std::move(body));
        auto future = promise->get_future();

        const bool posted = viewer.postWork(Visualizer::WorkItem{
            .run =
                [context, shared_body, promise]() mutable {
                    core::run_guarded<T>(context, std::move(*shared_body),
                                         [promise](lfs::Result<T>&& r) { promise->set_value(std::move(r)); });
                },
            .cancel =
                [context, cancelled_error, promise]() mutable {
                    core::run_guarded<T>(
                        context,
                        [cancelled_error]() -> lfs::Result<T> {
                            return core::failure_result<T>(cancelled_error);
                        },
                        [promise](lfs::Result<T>&& r) { promise->set_value(std::move(r)); });
                },
        });

        if (!posted) {
            return core::failure_result<T>(std::move(cancelled_error));
        }

        return future.get();
    }

} // namespace lfs::vis
