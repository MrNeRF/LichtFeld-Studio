/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <string>

#include <nlohmann/json.hpp>

namespace lfs::core {

    inline bool add_bounded_json_cost(
        const nlohmann::json& value,
        std::size_t& cost,
        const std::size_t limit) {
        constexpr std::size_t NODE_OVERHEAD = 64;
        if (cost > limit || NODE_OVERHEAD > limit - cost)
            return false;
        cost += NODE_OVERHEAD;

        if (value.is_string()) {
            const auto& text = value.get_ref<const std::string&>();
            if (text.size() > limit - cost)
                return false;
            cost += text.size();
            return true;
        }
        if (value.is_array()) {
            for (const auto& item : value) {
                if (!add_bounded_json_cost(item, cost, limit))
                    return false;
            }
            return true;
        }
        if (value.is_object()) {
            for (const auto& [key, item] : value.items()) {
                if (key.size() > limit - cost)
                    return false;
                cost += key.size();
                if (!add_bounded_json_cost(item, cost, limit))
                    return false;
            }
        }
        return true;
    }

} // namespace lfs::core
