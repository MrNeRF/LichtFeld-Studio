// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: MIT
// Linux uses element-wise conversion because glibc wide-character helpers
// assume its native four-byte wchar_t, while the FidelityFX ABI requires two.

#include "utils.h"

std::string WCharToUTF8(const std::wstring& value) {
    std::string result;
    result.reserve(value.size());
    for (const wchar_t character : value)
        result.push_back(static_cast<uint32_t>(character) < 0x80u ? static_cast<char>(character) : '?');
    return result;
}

std::wstring UTF8ToWChar(const std::string& value) {
    std::wstring result;
    result.reserve(value.size());
    for (const unsigned char character : value)
        result.push_back(character < 0x80u ? static_cast<wchar_t>(character) : L'?');
    return result;
}
