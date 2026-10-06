/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <nlohmann/json.hpp>

namespace lfs::mcp {
    class ToolRegistry;
    class ResourceRegistry;
} // namespace lfs::mcp
namespace lfs::vis {
    class Visualizer;
    class VisualizerImpl;
} // namespace lfs::vis
namespace lfs::app {
    void register_gui_node_tools(mcp::ToolRegistry& registry, vis::Visualizer* viewer);
    void register_gui_node_editor_tools(mcp::ToolRegistry& registry, vis::Visualizer* viewer);
    void register_gui_node_resources(mcp::ResourceRegistry& registry, vis::Visualizer* viewer);
    nlohmann::json node_evaluation_job(vis::VisualizerImpl& viewer);
} // namespace lfs::app
