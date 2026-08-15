/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ranges>
#include <span>
#include <string_view>

namespace lfs::io::project::chapter_binary {

    template <std::unsigned_integral T>
    bool checked_add(const T lhs, const T rhs, T& output) {
        if (rhs > std::numeric_limits<T>::max() - lhs) {
            return false;
        }
        output = lhs + rhs;
        return true;
    }

    template <std::unsigned_integral T>
    bool checked_mul(const T lhs, const T rhs, T& output) {
        if (lhs != 0 && rhs > std::numeric_limits<T>::max() / lhs) {
            return false;
        }
        output = lhs * rhs;
        return true;
    }

    inline bool align_up(const std::uint64_t value,
                         const std::uint64_t alignment,
                         std::uint64_t& output) {
        std::uint64_t adjusted = 0;
        if (!std::has_single_bit(alignment) ||
            !checked_add(value, alignment - 1, adjusted)) {
            return false;
        }
        output = adjusted & ~(alignment - 1);
        return true;
    }

    inline std::uint16_t read_u16(const std::span<const std::byte> bytes,
                                  const std::size_t offset) {
        return static_cast<std::uint16_t>(
                   std::to_integer<std::uint8_t>(bytes[offset])) |
               static_cast<std::uint16_t>(
                   std::to_integer<std::uint8_t>(bytes[offset + 1]))
                   << 8u;
    }

    inline std::uint32_t read_u32(const std::span<const std::byte> bytes,
                                  const std::size_t offset) {
        std::uint32_t value = 0;
        for (std::size_t index = 0; index < sizeof(value); ++index) {
            value |= static_cast<std::uint32_t>(
                         std::to_integer<std::uint8_t>(bytes[offset + index]))
                     << (index * 8u);
        }
        return value;
    }

    inline std::int32_t read_i32(const std::span<const std::byte> bytes,
                                 const std::size_t offset) {
        return std::bit_cast<std::int32_t>(read_u32(bytes, offset));
    }

    inline std::uint64_t read_u64(const std::span<const std::byte> bytes,
                                  const std::size_t offset) {
        std::uint64_t value = 0;
        for (std::size_t index = 0; index < sizeof(value); ++index) {
            value |= static_cast<std::uint64_t>(
                         std::to_integer<std::uint8_t>(bytes[offset + index]))
                     << (index * 8u);
        }
        return value;
    }

    inline float read_f32(const std::span<const std::byte> bytes,
                          const std::size_t offset) {
        return std::bit_cast<float>(read_u32(bytes, offset));
    }

    inline void write_u16(const std::span<std::byte> bytes,
                          const std::size_t offset,
                          const std::uint16_t value) {
        for (std::size_t index = 0; index < sizeof(value); ++index) {
            bytes[offset + index] =
                static_cast<std::byte>(value >> (index * 8u));
        }
    }

    inline void write_u32(const std::span<std::byte> bytes,
                          const std::size_t offset,
                          const std::uint32_t value) {
        for (std::size_t index = 0; index < sizeof(value); ++index) {
            bytes[offset + index] =
                static_cast<std::byte>(value >> (index * 8u));
        }
    }

    inline void write_i32(const std::span<std::byte> bytes,
                          const std::size_t offset,
                          const std::int32_t value) {
        write_u32(bytes, offset, std::bit_cast<std::uint32_t>(value));
    }

    inline void write_u64(const std::span<std::byte> bytes,
                          const std::size_t offset,
                          const std::uint64_t value) {
        for (std::size_t index = 0; index < sizeof(value); ++index) {
            bytes[offset + index] =
                static_cast<std::byte>(value >> (index * 8u));
        }
    }

    inline void write_f32(const std::span<std::byte> bytes,
                          const std::size_t offset,
                          const float value) {
        write_u32(bytes, offset, std::bit_cast<std::uint32_t>(value));
    }

    inline bool all_zero(const std::span<const std::byte> bytes) {
        return std::ranges::all_of(bytes, [](const std::byte value) {
            return value == std::byte{0};
        });
    }

    inline bool valid_utf8(const std::string_view text) {
        std::size_t index = 0;
        while (index < text.size()) {
            const auto lead = static_cast<std::uint8_t>(
                static_cast<unsigned char>(text[index]));
            if (lead <= 0x7fu) {
                ++index;
                continue;
            }

            std::size_t count = 0;
            std::uint32_t codepoint = 0;
            if ((lead & 0xe0u) == 0xc0u) {
                count = 2;
                codepoint = lead & 0x1fu;
            } else if ((lead & 0xf0u) == 0xe0u) {
                count = 3;
                codepoint = lead & 0x0fu;
            } else if ((lead & 0xf8u) == 0xf0u) {
                count = 4;
                codepoint = lead & 0x07u;
            } else {
                return false;
            }
            if (index + count > text.size()) {
                return false;
            }
            for (std::size_t continuation = 1; continuation < count;
                 ++continuation) {
                const auto byte = static_cast<std::uint8_t>(
                    static_cast<unsigned char>(text[index + continuation]));
                if ((byte & 0xc0u) != 0x80u) {
                    return false;
                }
                codepoint = (codepoint << 6u) | (byte & 0x3fu);
            }
            const std::uint32_t minimum =
                count == 2 ? 0x80u : count == 3 ? 0x800u
                                                : 0x10000u;
            if (codepoint < minimum || codepoint > 0x10ffffu ||
                (codepoint >= 0xd800u && codepoint <= 0xdfffu)) {
                return false;
            }
            index += count;
        }
        return true;
    }

} // namespace lfs::io::project::chapter_binary
