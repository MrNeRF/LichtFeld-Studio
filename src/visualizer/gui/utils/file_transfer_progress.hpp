/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace lfs::vis::gui {
    // Worker callbacks only touch atomics. The existing export panel belongs to
    // the UI thread and reads the same export job as normal exports.
    struct FileTransferProgressState {
        std::shared_ptr<std::atomic<bool>> cancelled;
        std::atomic<int> completed{0};
        std::atomic<size_t> current_item{0};
        std::string error_message; // Written/read only by the native UI thread.
        std::atomic<bool> user_cancelled{false};

        explicit FileTransferProgressState(std::shared_ptr<std::atomic<bool>> token) : cancelled(std::move(token)) {}
        void requestCancel() noexcept {
            user_cancelled.store(true);
            cancelled->store(true);
        }
        bool report(size_t index, size_t count, float fraction) noexcept {
            if (count && index < count && std::isfinite(fraction)) {
                auto item = current_item.load();
                while (index > item && !current_item.compare_exchange_weak(item, index)) {}
                const int next = static_cast<int>(1000.0 * (static_cast<double>(index) + std::clamp(fraction, 0.0f, 1.0f)) / static_cast<double>(count));
                int previous = completed.load();
                while (next > previous && !completed.compare_exchange_weak(previous, next)) {}
            }
            return !cancelled->load();
        }
    };

    // The caller may advance complete frames only from the deferred native
    // action, never from an Rml update. Drain the worker even if a frame fails.
    template <class Tick>
    auto waitForFileTransfer(std::future<std::vector<std::filesystem::path>>& future,
                             FileTransferProgressState& state, Tick tick,
                             std::chrono::milliseconds poll_interval = std::chrono::milliseconds(25)) {
        std::exception_ptr ui_error;
        while (future.wait_for(poll_interval) != std::future_status::ready) {
            if (!ui_error) {
                try {
                    tick(state.completed.load());
                } catch (...) {
                    // LFS-CENSUS-OK(empty-catch): cancel/drain the writer before rethrowing UI failure.
                    state.cancelled->store(true);
                    ui_error = std::current_exception();
                }
            }
        }
        auto files = future.get();
        if (ui_error)
            std::rethrow_exception(ui_error);
        return files;
    }

} // namespace lfs::vis::gui
