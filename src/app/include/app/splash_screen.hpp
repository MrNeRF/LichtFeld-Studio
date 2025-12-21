/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <functional>

namespace lfs::app {

struct MonitorInfo {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
};

class SplashScreen {
public:
    static int run(std::function<int()> task);
    static int runAndGetMonitor(std::function<int()> task, MonitorInfo& monitor);
};

} // namespace lfs::app
