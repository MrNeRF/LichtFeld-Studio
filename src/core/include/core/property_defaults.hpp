/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "property_system.hpp"

#include <algorithm>
#include <stdexcept>

namespace lfs::core::prop {

    // Enum defaults store EnumItem::value, never an item's position. Consumers
    // using string-valued controls resolve that value explicitly to an identifier.
    [[nodiscard]] inline std::string enum_identifier(const PropertyMeta& meta, const int64_t value) {
        if (meta.type != PropType::Enum)
            throw std::invalid_argument("Property is not an enum: " + meta.id);
        const auto item = std::ranges::find_if(meta.enum_items, [value](const auto& entry) {
            return entry.value == value;
        });
        if (item == meta.enum_items.end())
            throw std::invalid_argument("Unknown enum value for property: " + meta.id);
        if (std::ranges::count_if(meta.enum_items, [value, &item](const auto& entry) {
                return entry.value == value || entry.identifier == item->identifier;
            }) != 1)
            throw std::invalid_argument("Ambiguous enum value or identifier for property: " + meta.id);
        return item->identifier;
    }

    // Missing or imprecise legacy metadata must not prevent ordinary access.
    [[nodiscard]] inline std::optional<std::string> enum_default_identifier(const PropertyMeta& meta) {
        if (meta.type != PropType::Enum || !meta.default_value)
            return std::nullopt;
        const auto value = std::get_if<int64_t>(&*meta.default_value);
        if (!value)
            return std::nullopt;
        try {
            return enum_identifier(meta, *value);
        } catch (const std::invalid_argument&) {
            return std::nullopt;
        }
    }

} // namespace lfs::core::prop
