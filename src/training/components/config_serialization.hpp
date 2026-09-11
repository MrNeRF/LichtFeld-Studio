/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <string_view>
#include <type_traits>

namespace lfs::training::config_serialization_detail {

    constexpr uint32_t MAX_CONFIG_PAYLOAD_BYTES = 4096;

    [[noreturn]] void throw_unsupported_component_version(
        std::string_view component,
        uint32_t version,
        uint32_t minimum_version,
        uint32_t current_version);

    [[noreturn]] void throw_config_data_loss(
        std::string_view field,
        std::string_view problem);

    [[noreturn]] void throw_config_write_failure(std::string_view field);

    template <typename T>
    concept LittleEndianScalar =
        std::is_trivially_copyable_v<T> &&
        (std::is_integral_v<T> || std::is_floating_point_v<T>) &&
        (sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t));

    template <LittleEndianScalar T>
    using ScalarBits = std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint64_t>;

    template <LittleEndianScalar T>
    void write_little_endian(std::ostream& os, const T value, const std::string_view field) {
        const ScalarBits<T> bits = std::bit_cast<ScalarBits<T>>(value);
        std::array<char, sizeof(T)> bytes{};
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            bytes[i] = static_cast<char>((bits >> (i * 8)) & 0xffU);
        }
        os.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!os) {
            throw_config_write_failure(field);
        }
    }

    template <LittleEndianScalar T>
    [[nodiscard]] T read_little_endian(std::istream& is, const std::string_view field) {
        std::array<char, sizeof(T)> bytes{};
        is.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!is) {
            throw_config_data_loss(field, "truncated");
        }

        ScalarBits<T> bits = 0;
        for (std::size_t i = 0; i < bytes.size(); ++i) {
            bits |= static_cast<ScalarBits<T>>(static_cast<unsigned char>(bytes[i])) << (i * 8);
        }
        return std::bit_cast<T>(bits);
    }

    inline void skip_bytes(std::istream& is, uint32_t bytes, const std::string_view field) {
        std::array<char, 256> buffer{};
        while (bytes > 0) {
            const auto chunk = std::min<uint32_t>(bytes, static_cast<uint32_t>(buffer.size()));
            is.read(buffer.data(), static_cast<std::streamsize>(chunk));
            if (!is) {
                throw_config_data_loss(field, "truncated trailing fields");
            }
            bytes -= chunk;
        }
    }

} // namespace lfs::training::config_serialization_detail
