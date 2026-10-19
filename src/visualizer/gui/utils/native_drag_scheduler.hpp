/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "native_file_drag.hpp"
#include <future>
#include <stdexcept>
#include <utility>

namespace lfs::vis::gui {
    // Owned by the window, not by an Rml Element. A native drag can outlive
    // the originating panel, and is dispatched only at a completed frame.
    class NativeDragScheduler {
        std::function<FileDragResult()> pending_;
        std::shared_ptr<std::promise<FileDragResult>> result_;
        bool active_ = false;
        static inline thread_local unsigned active_count_ = 0;

    public:
        [[nodiscard]] bool busy() const { return active_ || bool(pending_); }
        [[nodiscard]] static bool outgoingDragActive() { return active_count_ != 0; }
        [[nodiscard]] static bool ownsNativeCursor() { return active_count_ != 0; }
        std::future<FileDragResult> enqueue(std::function<FileDragResult()> action) {
            if (!action)
                throw std::invalid_argument("Missing native drag action");
            if (busy())
                throw std::runtime_error("A native drag is already scheduled");
            auto result = std::make_shared<std::promise<FileDragResult>>();
            auto future = result->get_future();
            pending_ = std::move(action);
            result_ = std::move(result);
            return future;
        }
        void dispatch() {
            if (active_ || !pending_)
                return;
            auto action = std::move(pending_);
            auto result = std::move(result_);
            struct ActiveScope {
                bool& active;
                unsigned& count;
                ActiveScope(bool& value, unsigned& total) : active(value), count(total) {
                    active = true;
                    ++count;
                }
                ~ActiveScope() {
                    --count;
                    active = false;
                }
            } scope(active_, active_count_);
            try {
                result->set_value(action());
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): propagate the native action error through its future.
                result->set_exception(std::current_exception());
            }
        }
    };
} // namespace lfs::vis::gui
