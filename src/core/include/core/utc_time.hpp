/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>

namespace lfs::core {

    // ISO-8601 UTC timestamp: YYYY-MM-DDTHH:MM:SSZ
    [[nodiscard]] inline std::string utc_now() {
        const auto now = std::chrono::system_clock::now();
        const std::time_t timestamp = std::chrono::system_clock::to_time_t(now);
        std::tm utc{};
#ifdef _WIN32
        gmtime_s(&utc, &timestamp);
#else
        gmtime_r(&timestamp, &utc);
#endif
        std::ostringstream output;
        output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
        return output.str();
    }

} // namespace lfs::core
