/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/main_loop.hpp"
#include "gui/utils/file_transfer_progress.hpp"
#include "gui/utils/native_drag_scheduler.hpp"
#include <iostream>
#include <stdexcept>

int main() {
    using namespace lfs::vis;
    using namespace lfs::vis::gui;
    const auto check = [](bool yes) {
        if (!yes)
            throw std::runtime_error("Modal frame contract failed");
    };
    MainLoop loop;
    NativeDragScheduler scheduler;
    int updates = 0, renders = 0, completions = 0, rejected_reentry = 0;
    int shutdowns = 0;
    loop.setShutdownCallback([&] { ++shutdowns; });
    bool close = false;
    loop.setShouldCloseCallback([&] { return close; });
    loop.setUpdateCallback([&] {
        ++updates;
        try {
            loop.runModalFrame();
        } catch (const std::logic_error&) {
            ++rejected_reentry;
        }
    });
    loop.setRenderCallback([&] { ++renders; });
    loop.setFrameCompletedCallback([&] {
        ++completions;
        scheduler.dispatch();
    });
    auto result = scheduler.enqueue([&] {
        check(updates == 1 && renders == 1 && completions == 1);
        check(scheduler.busy());
        std::promise<std::vector<std::filesystem::path>> pending_io;
        auto io = pending_io.get_future();
        auto cancelled = std::make_shared<std::atomic<bool>>(false);
        FileTransferProgressState progress(cancelled);
        int ticks = 0;
        // Hold the export unresolved across three complete application frames.
        auto paths = waitForFileTransfer(io, progress, [&](int) {
            check(loop.runModalFrame());
            if (++ticks == 3)
                pending_io.set_value({"committed.ply"});
        });
        check(paths.size() == 1 && ticks == 3);
        return FileDragResult{FileDragStatus::Copied, {}};
    });
    check(loop.runModalFrame());
    check(result.get().status == FileDragStatus::Copied);
    check(updates == 4 && renders == 4 && completions == 4 && rejected_reentry == 4);
    check(!scheduler.busy());

    // Dropping the original panel's future does not drop the queued transfer.
    bool ran = false;
    {
        auto forgotten = scheduler.enqueue([&] { ran = true; return FileDragResult{}; });
    }
    bool busy_rejected = false;
    try {
        scheduler.enqueue([] { return FileDragResult{}; });
    } catch (const std::runtime_error&) { busy_rejected = true; }
    scheduler.dispatch();
    check(ran && busy_rejected && !scheduler.busy());

    auto failed = scheduler.enqueue([]() -> FileDragResult { throw std::runtime_error("native error"); });
    scheduler.dispatch();
    bool propagated = false;
    try {
        failed.get();
    } catch (const std::runtime_error&) { propagated = true; }
    check(propagated && !scheduler.busy() && !NativeDragScheduler::ownsNativeCursor());

    close = true;
    check(!loop.runModalFrame());
    check(updates == 4 && renders == 4); // No shutdown/render after close.
    close = false;
    int faults = 0;
    loop.setRenderCallback([] { throw std::runtime_error("frame fault"); });
    loop.setFrameErrorCallback([&](std::exception_ptr) { ++faults; });
    check(loop.runModalFrame() && faults == 1);
    loop.setRenderCallback([&] { ++renders; });
    check(loop.runModalFrame());
    check(renders == 5);   // Frame guard released after an exception too.
    check(shutdowns == 0); // Only the outer run loop may tear down the application.
    std::cout << "PASS: dispatch after frame, responsive modal frames, no render reentry, detached panel, errors and close\n";
}
