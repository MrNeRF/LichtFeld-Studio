/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

namespace lfs::mcp {
    class ResourceRegistry;
    class ToolRegistry;
} // namespace lfs::mcp

namespace lfs::vis {
    class Visualizer;
}

namespace lfs::app {

    void register_gui_workspace_tools(mcp::ToolRegistry& registry,
                                      vis::Visualizer* viewer);
    void register_gui_workspace_resources(mcp::ResourceRegistry& registry,
                                          vis::Visualizer* viewer);

} // namespace lfs::app
