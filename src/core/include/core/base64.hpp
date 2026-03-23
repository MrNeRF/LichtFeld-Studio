/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace lfs::core {

    inline std::string base64_encode(const uint8_t* data, size_t len) {
        static constexpr char CHARS[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

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

} // namespace lfs::core
