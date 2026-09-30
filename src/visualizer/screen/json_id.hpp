/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <limits>
#include <nlohmann/json.hpp>

namespace lfs::vis::screen::detail {
    inline bool readId(const nlohmann::json& json, std::uint32_t& out, const bool allow_zero) {
        if (!json.is_number_integer() && !json.is_number_unsigned())
            return false;
        const auto raw = json.get<std::int64_t>();
        // The largest id would leave no room to allocate another.
        if (raw < 0 || raw >= static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max()))
            return false;
        if (!allow_zero && raw == 0)
            return false;
        out = static_cast<std::uint32_t>(raw);
        return true;
    }

} // namespace lfs::vis::screen::detail
