/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <cstddef>
#include <span>
#include <string>

namespace lfs::vis::gui {

    struct GpuMemoryInfo {
        size_t process_used = 0;
        size_t total_used = 0;
        size_t total = 0;
        float gpu_utilization_percent = -1.f;
        bool gpu_utilization_valid = false;
        bool process_estimated = false;
        bool device_estimated = false;
        std::string device_name;
    };

    struct LFS_VIS_API GpuProcessUsage {
        unsigned int pid = 0;
        unsigned long long bytes = 0;
    };

    LFS_VIS_API GpuMemoryInfo queryGpuMemory();
    LFS_VIS_API float queryGpuUtilization();
    LFS_VIS_API GpuMemoryInfo selectGpuMemory(size_t compute_bytes, size_t graphics_bytes,
                                              size_t dxgi_bytes, size_t cuda_used,
                                              size_t cuda_total, size_t nvml_used,
                                              size_t nvml_total);
    LFS_VIS_API std::string formatGpuGiB(size_t bytes);
    LFS_VIS_API size_t parseGpuProcessBytes(unsigned int pid,
                                            std::span<const GpuProcessUsage> processes);

} // namespace lfs::vis::gui
