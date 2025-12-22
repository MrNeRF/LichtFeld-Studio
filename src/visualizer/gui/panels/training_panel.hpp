/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/ui_context.hpp"
#include <chrono>
#ifdef WIN32
#include <shobjidl.h>
#include <windows.h>
#endif

namespace lfs::vis::gui::panels {

    void DrawTrainingControls(const UIContext& ctx);
    void DrawTrainingStatus(const UIContext& ctx);
    void DrawTrainingParams(const UIContext& ctx);

    // Training panel state
    struct TrainingPanelState {
        bool save_in_progress = false;
        std::chrono::steady_clock::time_point save_start_time;

        static TrainingPanelState& getInstance() {
            static TrainingPanelState instance;
            return instance;
        }
    };
} // namespace lfs::vis::gui::panels
