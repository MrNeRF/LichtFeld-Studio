/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "visualizer/visualizer.hpp"

#include <atomic>
#include <exception>
#include <functional>
#include <future>
#include <memory>
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

} // namespace lfs::vis
