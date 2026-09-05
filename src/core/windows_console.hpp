/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

namespace lfs::core::detail {

    // Inspect the CRT stream, not GetStdHandle: capture/redirect code can replace
    // the descriptor without replacing the process standard handle.
    inline HANDLE console_output_handle(FILE* stream) {
        const int fd = _fileno(stream);
        if (fd < 0)
            return INVALID_HANDLE_VALUE;
        const auto handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
        DWORD mode = 0;
        return handle != INVALID_HANDLE_VALUE && GetConsoleMode(handle, &mode)
                   ? handle
                   : INVALID_HANDLE_VALUE;
    }

    // False means nothing was written and the caller may use its byte fallback.
    // Never replay the whole message after a partial console write.
    inline bool write_console_utf8(HANDLE handle, std::string_view text) {
        if (text.empty())
            return true;
        if (text.size() > static_cast<size_t>((std::numeric_limits<int>::max)()))
            return false;
        const int bytes = static_cast<int>(text.size());
        const int length = MultiByteToWideChar(CP_UTF8, 0, text.data(), bytes, nullptr, 0);
        if (length <= 0)
            return false;
        std::wstring wide(static_cast<size_t>(length), L'\0');
        if (MultiByteToWideChar(CP_UTF8, 0, text.data(), bytes, wide.data(), length) != length)
            return false;

        // Match the CRT text stream's LF -> CRLF translation, including multiline
        // tracebacks. WriteConsoleW bypasses that translation.
        if (wide.find(L'\n') != std::wstring::npos) {
            std::wstring translated;
            translated.reserve(wide.size());
            for (const wchar_t character : wide) {
                if (character == L'\n')
                    translated += L'\r';
                translated += character;
            }
            wide = std::move(translated);
        }

        size_t offset = 0;
        while (offset < wide.size()) {
            size_t count = (std::min)(size_t{16 * 1024}, wide.size() - offset);
            // Do not divide a UTF-16 surrogate pair at a chunk boundary.
            const wchar_t last = wide[offset + count - 1];
            if (offset + count < wide.size() && last >= 0xd800 && last <= 0xdbff)
                --count;
            DWORD written = 0;
            if (!WriteConsoleW(handle, wide.data() + offset, static_cast<DWORD>(count), &written, nullptr) || written == 0)
                return offset != 0;
            offset += written;
        }
        return true;
    }

} // namespace lfs::core::detail
#endif
