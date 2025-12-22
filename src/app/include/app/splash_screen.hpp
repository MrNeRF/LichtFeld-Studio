/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <functional>

namespace lfs::app {

    class SplashScreen {
    public:
        // Run task; show splash only if it takes longer than delay_ms
        static int runWithDelay(std::function<int()> task, int delay_ms = 300);
    };

} // namespace lfs::app
