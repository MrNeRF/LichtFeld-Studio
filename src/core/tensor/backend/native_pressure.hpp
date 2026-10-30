/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/memory_pressure.hpp"
namespace lfs::core {
    class MemoryInfo;
}
namespace lfs::core::internal {
    MemoryInfo cuda_pressure_memory_info(bool free_query);
    void register_cuda_pressure_clients(MemoryPressureCoordinator&);
    void record_pressure_breadcrumb();
} // namespace lfs::core::internal
