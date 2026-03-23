/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

namespace lfs::vis {
    class Visualizer;
}

namespace lfs::app {

    void register_gui_scene_tools(lfs::vis::Visualizer* viewer);
    void register_gui_scene_resources(lfs::vis::Visualizer* viewer);

} // namespace lfs::app
