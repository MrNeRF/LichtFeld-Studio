/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::core {

    inline std::string base64_encode(const uint8_t* data, size_t len) {
        static constexpr char CHARS[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

        std::string result;
        result.reserve(((len + 2) / 3) * 4);

        for (size_t i = 0; i < len; i += 3) {
            const uint32_t b0 = data[i];
            const uint32_t b1 = (i + 1 < len) ? data[i + 1] : 0;
            const uint32_t b2 = (i + 2 < len) ? data[i + 2] : 0;

            result += CHARS[(b0 >> 2) & 0x3F];
            result += CHARS[((b0 << 4) | (b1 >> 4)) & 0x3F];
            result += (i + 1 < len) ? CHARS[((b1 << 2) | (b2 >> 6)) & 0x3F] : '=';
            result += (i + 2 < len) ? CHARS[b2 & 0x3F] : '=';
        }
        return result;
    }

    inline std::string base64_encode(const std::vector<uint8_t>& data) {
        return base64_encode(data.data(), data.size());
    }

    inline std::vector<uint8_t> base64_decode(std::string_view encoded) {
        constexpr std::string_view alphabet =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        if (encoded.size() % 4 != 0)
            throw std::invalid_argument(
                std::format("Base64 length must be a multiple of four (length={})", encoded.size()));
        std::vector<uint8_t> result;
        result.reserve(encoded.size() / 4 * 3);
        for (size_t offset = 0; offset < encoded.size(); offset += 4) {
            uint32_t bits = 0;
            size_t padding = 0;
            for (size_t channel = 0; channel < 4; ++channel) {
                const char character = encoded[offset + channel];
                bits <<= 6;
                if (character == '=') {
                    if (channel < 2 || offset + 4 != encoded.size())
                        throw std::invalid_argument(std::format("Base64 padding must end the final quartet "
                                                                "(offset={}, channel={}, length={})",
                                                                offset, channel, encoded.size()));
                    ++padding;
                } else {
                    const size_t digit = alphabet.find(character);
                    if (padding || digit == std::string_view::npos)
                        throw std::invalid_argument(
                            std::format("Base64 requires alphabet characters before padding "
                                        "(offset={}, channel={}, character={}, padding={})",
                                        offset, channel, static_cast<unsigned char>(character), padding));
                    bits |= static_cast<uint32_t>(digit);
                }
            }
            if ((padding == 2 && (bits & 0xffffU)) || (padding == 1 && (bits & 0xffU)))
                throw std::invalid_argument(
                    std::format("Base64 padding bits must be zero (offset={}, bits={}, padding={})", offset,
                                bits, padding));
            result.push_back(static_cast<uint8_t>(bits >> 16));
            if (padding < 2)
                result.push_back(static_cast<uint8_t>(bits >> 8));
            if (padding == 0)
                result.push_back(static_cast<uint8_t>(bits));
        }
        return result;
    }

} // namespace lfs::core
