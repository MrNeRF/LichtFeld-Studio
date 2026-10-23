/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <core/export.hpp>
#include <cstddef>
#include <string>
#include <vector>

namespace lfs::vis::gui {
    enum class ProgressMarkerStyle { Classic,
                                     Miner,
                                     Tessle };

    struct ProgressMarkerRenderState {
        bool dragging = false;
        bool adding = false;
        size_t original_step = 0;
        size_t preview_step = 0;
        size_t hover_step = 0;
    };

    [[nodiscard]] LFS_VIS_API std::string buildProgressMarkersRml(
        std::vector<size_t> save_steps, int total_iterations, int current_iteration,
        const ProgressMarkerRenderState& state, ProgressMarkerStyle style);
} // namespace lfs::vis::gui
