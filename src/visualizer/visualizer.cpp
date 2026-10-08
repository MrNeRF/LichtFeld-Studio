/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "visualizer/visualizer.hpp"
#include "visualizer_impl.hpp"

#include "rendering/scene_upscaler_plugin.hpp"

#include <atomic>
#include <mutex>
#include <utility>

namespace lfs::vis {

    namespace {
        std::mutex g_runtime_service_controls_mutex;
        RuntimeServiceControls g_runtime_service_controls;
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
        return true;
    }

    std::optional<int> mcpPortOverride() {
        std::lock_guard lock(g_runtime_service_controls_mutex);
        return g_runtime_service_controls.mcp_port_override;
    }

    std::unique_ptr<Visualizer> Visualizer::create(const ViewerOptions& options) {
        configureSceneUpscalerPluginLoading(!options.safe_mode);
        return std::make_unique<VisualizerImpl>(options);
    }

} // namespace lfs::vis
