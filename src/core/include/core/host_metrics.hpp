/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <cstddef>
#include <vector>

namespace lfs::core::host_metrics {

    struct LFS_CORE_API Sample {
        std::size_t process_rss_bytes = 0;
        std::size_t system_total_bytes = 0;
        std::size_t system_used_bytes = 0;
        bool ram_valid = false;
        float process_cpu_percent = -1.f;
        std::vector<float> per_core_cpu_percent;
        bool cpu_valid = false;
    };

    [[nodiscard]] LFS_CORE_API Sample sample();

} // namespace lfs::core::host_metrics
