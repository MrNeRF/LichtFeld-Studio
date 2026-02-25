/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>

namespace lfs::vis::gui {

    struct GpuMemoryInfo {
        size_t process_used = 0;
        size_t total_used = 0;
        size_t total = 0;
    };

    GpuMemoryInfo queryGpuMemory();

} // namespace lfs::vis::gui
