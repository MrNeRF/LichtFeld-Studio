/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "gui/utils/file_transfer_progress.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

int main() {
    using namespace lfs::vis::gui;
    const auto check = [](bool value) {
        if (!value)
            throw std::runtime_error("File transfer progress contract failed");
    };
    auto cancelled = std::make_shared<std::atomic<bool>>(false);
    FileTransferProgressState state(cancelled);
    check(state.report(0, 2, 1));
    check(state.completed.load() == 500);
    state.report(0, 2, 0); // A writer may restart a stage's own fraction.
    state.report(1, 2, std::numeric_limits<float>::quiet_NaN());
    check(state.completed.load() == 500);
    state.report(1, 2, .5f);
    check(state.completed.load() == 750);
    state.report(1, 2, 2);
    check(state.completed.load() == 1000);
    cancelled->store(true);
    check(!state.report(1, 2, 1));

    // A completed export must not create or pump a progress window.
    cancelled->store(false);
    std::promise<std::vector<std::filesystem::path>> ready;
    auto result = ready.get_future();
    ready.set_value({"committed.ply"});
    int ticks = 0;
    check(waitForFileTransfer(result, state, [&](int) { ++ticks; }).size() == 1);
    check(ticks == 0);

    // Clicking cancel reaches the writer, which must finish rollback before
    // returning from the native data request. No application event pump used.
    std::promise<std::vector<std::filesystem::path>> pending;
    result = pending.get_future();
    bool saw_failure = false;
    try {
        waitForFileTransfer(result, state, [&](int) {
            state.requestCancel();
            pending.set_exception(std::make_exception_ptr(std::runtime_error("writer cancelled")));
        });
    } catch (const std::runtime_error&) {
        saw_failure = true;
    }
    check(saw_failure && cancelled->load() && state.user_cancelled.load());

    // A failing progress UI cancels and drains a live worker instead of
    // abandoning it with snapshots, callbacks or a partial write still active.
    cancelled->store(false);
    state.user_cancelled.store(false);
    std::promise<std::vector<std::filesystem::path>> working;
    result = working.get_future();
    std::atomic<bool> drained{false};
    std::jthread worker([&] {
        while (!cancelled->load())
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        drained.store(true);
        working.set_exception(std::make_exception_ptr(std::runtime_error("writer rolled back")));
    });
    saw_failure = false;
    try {
        waitForFileTransfer(result, state, [](int) { throw std::runtime_error("native UI unavailable"); });
    } catch (const std::runtime_error&) {
        saw_failure = true;
    }
    check(saw_failure && drained.load() && !state.user_cancelled.load());
    std::cout << "PASS: aggregate progress, completed fast path, cancellation, worker drain after UI failure\n";
}
