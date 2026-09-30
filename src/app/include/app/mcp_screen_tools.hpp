/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

namespace lfs::mcp {
    class ToolRegistry;
    class ResourceRegistry;
} // namespace lfs::mcp

namespace lfs::vis {
    class Visualizer;
}

namespace lfs::app {

    void register_gui_screen_tools(lfs::mcp::ToolRegistry& registry, lfs::vis::Visualizer* viewer);
    void register_gui_screen_resources(lfs::mcp::ResourceRegistry& registry, lfs::vis::Visualizer* viewer);

} // namespace lfs::app
