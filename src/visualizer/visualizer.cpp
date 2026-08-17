/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/visualizer.hpp"
#include "visualizer_impl.hpp"

#include <atomic>
#include <mutex>
#include <utility>

namespace lfs::vis {

    namespace {
        std::mutex g_runtime_service_controls_mutex;
        RuntimeServiceControls g_runtime_service_controls;
        std::atomic<std::uint64_t> g_runtime_service_revision{0};
    } // namespace

    void setRuntimeServiceControls(RuntimeServiceControls controls) {
        std::lock_guard lock(g_runtime_service_controls_mutex);
        g_runtime_service_controls = std::move(controls);
    }

    bool toggleMcpRuntimeEnabled() {
        std::function<bool()> action;
        {
            std::lock_guard lock(g_runtime_service_controls_mutex);
            action = g_runtime_service_controls.toggle_mcp_enabled;
        }
        if (!action || !action())
            return false;
        g_runtime_service_revision.fetch_add(1, std::memory_order_release);
        return true;
    }

    bool toggleMcpRuntimeBinding() {
        std::function<bool()> action;
        {
            std::lock_guard lock(g_runtime_service_controls_mutex);
            action = g_runtime_service_controls.toggle_mcp_binding;
        }
        if (!action || !action())
            return false;
        g_runtime_service_revision.fetch_add(1, std::memory_order_release);
        return true;
    }

    std::uint64_t runtimeServiceRevision() {
        return g_runtime_service_revision.load(std::memory_order_acquire);
    }

    std::unique_ptr<Visualizer> Visualizer::create(const ViewerOptions& options) {
        return std::make_unique<VisualizerImpl>(options);
    }

} // namespace lfs::vis
