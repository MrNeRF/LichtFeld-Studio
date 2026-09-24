/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/gpu_backend_fwd.hpp"

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
        // Vulkan exposes a process budget, not whole-device usage. Keep the
        // two measurements distinct for the status bar and diagnostics.
        bool uses_process_budget = false;
        size_t process_budget = 0;
        size_t process_budget_used = 0;
    };

    LFS_VIS_API GpuMemoryInfo queryGpuMemory(lfs::core::GpuBackend backend = lfs::core::default_gpu_backend());
    LFS_VIS_API float queryGpuUtilization();

} // namespace lfs::vis::gui
