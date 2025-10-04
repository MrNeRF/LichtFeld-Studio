/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/ui_context.hpp"
#include "gui/windows/save_project_browser.hpp"
#include <chrono>
#ifdef WIN32
#include <shobjidl.h>
#include <windows.h>
#endif

namespace gs::gui::panels {

    void DrawTrainingControls(const UIContext& ctx);

    // Training panel state
    struct TrainingPanelState {
        bool save_in_progress = false;
        std::chrono::steady_clock::time_point save_start_time;

        SaveProjectBrowser save_browser;

        static TrainingPanelState& getInstance() {
            static TrainingPanelState instance;
            return instance;
        }
    };
} // namespace gs::gui::panels