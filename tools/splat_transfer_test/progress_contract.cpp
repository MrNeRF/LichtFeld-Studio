/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "gui/utils/file_transfer_progress.hpp"
#include "gui/utils/native_drag_scheduler.hpp"
#include "gui/utils/splat_drag_selection.hpp"
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <thread>

int main() {
    using namespace lfs::vis::gui;
    const auto check = [](bool value) {
        if (!value)
            throw std::runtime_error("File transfer progress contract failed");
    };
    struct Node {
        SplatDragKind kind;
        bool loaded = true;
        std::vector<int> children;
    };
    std::map<int, Node> tree = {
        {1, {SplatDragKind::Group, true, {2, 3, 6}}},
        {2, {SplatDragKind::Splat, true, {}}},
        {3, {SplatDragKind::Group, true, {4, 5}}},
        {4, {SplatDragKind::Splat, true, {}}},
        {5, {SplatDragKind::Other, true, {}}},
        {6, {SplatDragKind::Splat, true, {}}},
        {7, {SplatDragKind::Group, true, {}}},
        {8, {SplatDragKind::Splat, false, {}}},
    };
    const auto collect = [&](std::vector<int> roots) {
        return collectSplatDragSelection<int>(roots, [&](int id) -> std::optional<SplatDragNode<int>> {
            const auto found = tree.find(id);
            if (found == tree.end())
                return std::nullopt;
            return SplatDragNode<int>{found->second.kind, found->second.loaded, found->second.children};
        });
    };
    check(collect({1}).splats == std::vector<int>({2, 4, 6}));
    check(collect({1, 3, 2, 1}).splats == std::vector<int>({2, 4, 6}));
    check(collect({4, 1}).splats == std::vector<int>({4, 2, 6}));
    check(collect({2, 6}).splats == std::vector<int>({2, 6}));
    check(collect({7}).splats.empty() && collect({7}).error == SplatDragError::None);
    check(collect({1, 5}).splats.empty() && collect({1, 5}).error == SplatDragError::UnsupportedRoot);
    check(collect({999}).error == SplatDragError::MissingNode);
    check(collect({8}).error == SplatDragError::UnloadedSplat && collect({8}).failed_node == 8);
    tree[3].children.push_back(8);
    check(collect({1}).splats.empty() && collect({1}).error == SplatDragError::UnloadedSplat && collect({1}).failed_node == 8);
    tree[3].children.pop_back();
    tree[3].children.push_back(1);
    check(collect({1}).splats.empty() && collect({1}).error == SplatDragError::Cycle && collect({1}).failed_node == 1);
    tree[3].children.back() = 999;
    check(collect({1}).splats.empty() && collect({1}).error == SplatDragError::MissingNode);
    tree.clear();
    for (int id = 1; id < 20000; ++id)
        tree.emplace(id, Node{SplatDragKind::Group, true, {id + 1}});
    tree.emplace(20000, Node{SplatDragKind::Splat, true, {}});
    check(collect({1}).splats == std::vector<int>({20000}));

    auto cancelled = std::make_shared<std::atomic<bool>>(false);
    FileTransferProgressState state(cancelled);
    check(state.report(0, 2, 1));
    check(state.completed.load() == 500);
    state.report(0, 2, 0); // A writer may restart a stage's own fraction.
    state.report(1, 2, std::numeric_limits<float>::quiet_NaN());
    check(state.completed.load() == 500);
    state.report(1, 2, .5f);
    check(state.completed.load() == 750 && state.current_item.load() == 1);
    state.report(0, 2, 1);
    check(state.current_item.load() == 1);
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
    NativeDragScheduler scheduler;
    int actions = 0;
    auto scheduled = scheduler.enqueue([&] {
        ++actions;
        check(NativeDragScheduler::ownsNativeCursor());
        scheduler.dispatch(); // Modal frames cannot reenter this native action.
        return FileDragResult{FileDragStatus::Copied, {}};
    });
    check(actions == 0 && scheduler.busy());
    scheduler.dispatch();
    check(actions == 1 && scheduled.get().status == FileDragStatus::Copied && !scheduler.busy() && !NativeDragScheduler::ownsNativeCursor());
    std::cout << "PASS: aggregate progress, completed fast path, cancellation, worker drain after UI failure\n";
}
