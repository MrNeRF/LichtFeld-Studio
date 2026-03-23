/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "mcp/mcp_tools.hpp"

namespace lfs::vis {
    class Visualizer;
}

namespace lfs::app {

    void register_generic_gui_runtime_tools(mcp::ToolRegistry& registry,
                                            vis::Visualizer* viewer);

    void register_generic_gui_runtime_resources(mcp::ResourceRegistry& registry,
                                                vis::Visualizer* viewer);

} // namespace lfs::app
