/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace lfs::core {

    struct LFS_CORE_API Uuid {
        std::array<std::uint8_t, 16> bytes{};

        [[nodiscard]] bool is_nil() const noexcept;
        [[nodiscard]] std::string to_string() const;
        [[nodiscard]] static std::optional<Uuid> from_string(std::string_view value);

        friend bool operator==(const Uuid&, const Uuid&) = default;
    };

    class LFS_CORE_API UuidGenerationError final : public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
    };

    // UUIDv4 backed by the operating system CSPRNG. This generator is independent
    // of all training and tensor pseudo-random number generators.
    [[nodiscard]] LFS_CORE_API Uuid generate_uuid_v4();

} // namespace lfs::core

template <>
struct std::hash<lfs::core::Uuid> {
    [[nodiscard]] std::size_t operator()(const lfs::core::Uuid& uuid) const noexcept {
        std::size_t hash = sizeof(std::size_t) == 8
                               ? static_cast<std::size_t>(14695981039346656037ull)
                               : static_cast<std::size_t>(2166136261u);
        constexpr std::size_t PRIME = sizeof(std::size_t) == 8
                                          ? static_cast<std::size_t>(1099511628211ull)
                                          : static_cast<std::size_t>(16777619u);
        for (const std::uint8_t byte : uuid.bytes) {
            hash ^= byte;
            hash *= PRIME;
        }
        return hash;
    }
};
