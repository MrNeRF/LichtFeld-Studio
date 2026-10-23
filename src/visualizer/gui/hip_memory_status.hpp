/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "gpu_memory_query.hpp"
#include "core/gpu_device_info.hpp"

#ifdef LFS_TENSOR_HIP
namespace lfs::vis::gui {
    // Map a device-matched native sample without querying any GPU runtime.
    // Missing telemetry stays unknown; another adapter's CUDA usage is never
    // substituted for HIP. Keep this policy testable on CPU-only CI runners.
    inline GpuMemoryInfo selectHipGpuMemory(const core::GpuDeviceInfo* device) {
        GpuMemoryInfo info;
        info.uses_process_budget = true;
        info.native_process_budget = device && device->supports_process_memory_budget;
        info.native_metrics_unavailable = !info.native_process_budget;
        if (device) {
            info.device_name = device->name;
            info.total = device->total_memory_bytes;
        }
        if (info.native_process_budget) {
            info.process_budget = device->process_memory_budget_bytes;
            info.process_budget_used = device->process_memory_used_bytes;
            info.process_valid = true;
            info.process_used = info.process_budget_used;
            info.process_over_budget = info.process_used > info.process_budget;
        }
        return info;
    }
}
#endif
