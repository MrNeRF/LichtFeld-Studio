/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <cstddef>
#include <string>

namespace lfs::vis::gui {

    struct GpuMemoryInfo {
        size_t process_used = 0;
        size_t total_used = 0;
        size_t total = 0;
        float gpu_utilization_percent = -1.f;
        bool gpu_utilization_valid = false;
        std::string device_name;
    };

    LFS_VIS_API GpuMemoryInfo queryGpuMemory();
    LFS_VIS_API float queryGpuUtilization();

} // namespace lfs::vis::gui
