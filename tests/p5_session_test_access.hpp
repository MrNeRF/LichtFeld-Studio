/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "visualizer_impl.hpp"

#include <memory>

namespace lfs::vis {

    class P5SessionCaptureTestAccess {
    public:
        static void initializeInputController(VisualizerImpl& viewer) {
            // The constructor creates GUI/render owners, but input normally
            // arrives during initialize(). Capture needs its real state, not
            // a window, an event loop, or GPU initialization.
            viewer.input_controller_ = std::make_unique<InputController>(
                nullptr, viewer);
        }
    };

} // namespace lfs::vis
