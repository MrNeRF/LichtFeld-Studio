/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "splat_drag_axes.hpp"
#include <string>

namespace lfs::vis {
    struct SplatDragPreferenceState {
        std::string format = "ply";
        int sh_degree = 3;
        SplatDragAxes axes;
    };
} // namespace lfs::vis
