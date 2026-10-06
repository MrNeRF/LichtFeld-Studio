/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "py_ui.hpp"
#include "visualizer/post_work_utils.hpp"

#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace lfs::python {
    // Set while a Python node's execute() runs on the evaluation worker. The viewer may be waiting for that
    // worker, so reaching the viewer from there could deadlock.
    inline thread_local bool g_in_node_execute = false;

    inline void reject_viewer_work_in_node_execute() {
        if (g_in_node_execute)
            throw std::runtime_error("Viewer functions are unavailable inside a node's execute(); read inputs from its context");
    }

    template <typename T>
    struct ViewerResultContainsPython : std::bool_constant<std::is_base_of_v<nb::handle, std::remove_cvref_t<T>>> {};

    template <typename T>
    struct ViewerResultContainsPython<std::optional<T>> : ViewerResultContainsPython<T> {};

    template <typename T, typename Allocator>
    struct ViewerResultContainsPython<std::vector<T, Allocator>> : ViewerResultContainsPython<T> {};

    template <typename F>
        requires(!std::is_void_v<std::invoke_result_t<F>>)
    auto invoke_on_viewer(F&& fn, std::invoke_result_t<F> fallback) {
        static_assert(!ViewerResultContainsPython<std::invoke_result_t<F>>::value,
                      "Viewer work must return C++ snapshots; construct Python objects with the GIL held after dispatch");
        reject_viewer_work_in_node_execute();
        auto* const viewer = get_visualizer();
        if (!viewer || viewer->isOnViewerThread())
            return std::invoke(std::forward<F>(fn));
        if (!viewer->acceptsPostedWork())
            return fallback;

        nb::gil_scoped_release release;
        return lfs::vis::post_work_and_wait(
            [viewer](lfs::vis::Visualizer::WorkItem work) {
                return viewer->postWork(std::move(work));
            },
            std::forward<F>(fn),
            [fallback]() { return fallback; });
    }

    template <typename F>
        requires(std::is_void_v<std::invoke_result_t<F>>)
    void invoke_on_viewer(F&& fn) {
        reject_viewer_work_in_node_execute();
        auto* const viewer = get_visualizer();
        if (!viewer || viewer->isOnViewerThread()) {
            std::invoke(std::forward<F>(fn));
            return;
        }
        if (!viewer->acceptsPostedWork())
            return;

        nb::gil_scoped_release release;
        lfs::vis::post_work_and_wait(
            [viewer](lfs::vis::Visualizer::WorkItem work) {
                return viewer->postWork(std::move(work));
            },
            std::forward<F>(fn), [] {});
    }

} // namespace lfs::python
