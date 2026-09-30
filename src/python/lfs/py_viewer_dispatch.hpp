/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "py_ui.hpp"
#include "visualizer/post_work_utils.hpp"

namespace lfs::python {
    template <typename F>
        requires(!std::is_void_v<std::invoke_result_t<F>>)
    auto invoke_on_viewer(F&& fn, std::invoke_result_t<F> fallback) {
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
