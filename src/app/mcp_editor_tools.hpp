/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
namespace lfs::mcp {
    class ToolRegistry;
}
namespace lfs::vis {
    class VisualizerImpl;
}
namespace lfs::app {
    void register_editor_tools(mcp::ToolRegistry& registry, vis::VisualizerImpl* viewer);
}
