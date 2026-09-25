// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: MIT
//
// Linux-only compatibility shims for the FidelityFX SDK sources and tool.
// The SDK requires two-byte wchar_t and uses Windows-only string helpers; the
// wide helpers below intentionally operate on elements rather than glibc w*.
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>
#ifdef __cplusplus
#include <filesystem>
#include <new>
#endif

#ifndef FFX_UNUSED
#define FFX_UNUSED(x) ((void)(x))
#endif
#include <math.h>

#ifndef _WIN32
typedef unsigned char BYTE;
typedef unsigned long DWORD;
typedef int BOOL;
#ifndef TRUE
#define TRUE  1
#define FALSE 0
#endif

static inline int ffx_linux_strcpy_s(char* destination, size_t destination_size, const char* source) {
    if (!destination || !source || destination_size == 0)
        return 1;
    const size_t length = strlen(source);
    if (length >= destination_size) {
        destination[0] = '\0';
        return 1;
    }
    memcpy(destination, source, length + 1);
    return 0;
}
#define strcpy_s(destination, destination_size, source) \
    ffx_linux_strcpy_s((destination), (destination_size), (source))
#define strcat_s(destination, destination_size, source) \
    ((strlen(destination) + strlen(source) >= (destination_size)) ? 1 : (strcat((destination), (source)), 0))
#define sprintf_s(destination, destination_size, format, ...) \
    snprintf((destination), (destination_size), (format), ##__VA_ARGS__)

#ifdef __cplusplus
static inline int ffx_linux_wcscpy_s(wchar_t* destination, size_t destination_size, const wchar_t* source) {
    if (!destination || !source || destination_size == 0)
        return 1;

    size_t index = 0;
    for (; index + 1 < destination_size && source[index] != L'\0'; ++index)
        destination[index] = source[index];
    destination[index] = L'\0';

    return source[index] == L'\0' ? 0 : 1;
}
static inline int ffx_linux_wcscmp(const wchar_t* left, const wchar_t* right) {
    if (!left || !right)
        return left == right ? 0 : (left ? 1 : -1);

    size_t index = 0;
    while (left[index] != L'\0' && left[index] == right[index])
        ++index;
    if (left[index] == right[index])
        return 0;
    return left[index] < right[index] ? -1 : 1;
}
#define wcscmp(left, right)                      ffx_linux_wcscmp((left), (right))
#define FFX_WCSCPY_SELECT(_1, _2, _3, NAME, ...) NAME
#define FFX_WCSCPY_ARRAY(destination, source) \
    ffx_linux_wcscpy_s((destination), sizeof(destination) / sizeof((destination)[0]), (source))
#define FFX_WCSCPY_SIZED(destination, destination_size, source) \
    ffx_linux_wcscpy_s((destination), (destination_size), (source))
#define wcscpy_s(...) \
    FFX_WCSCPY_SELECT(__VA_ARGS__, FFX_WCSCPY_SIZED, FFX_WCSCPY_ARRAY)(__VA_ARGS__)

static inline int ffx_linux_wcstombs_s(size_t* converted, char* destination, size_t destination_size,
                                       const wchar_t* source, size_t count) {
    if (converted)
        *converted = 0;
    if (!destination || !source || destination_size == 0)
        return 1;

    size_t written = 0;
    for (size_t index = 0; index < count && source[index] != L'\0' && written + 1 < destination_size; ++index) {
        const uint32_t value = (uint32_t)source[index];
        destination[written++] = value < 0x80u ? (char)value : '?';
    }
    destination[written] = '\0';
    if (converted)
        *converted = written + 1;
    return 0;
}
#define wcstombs_s(converted, destination, destination_size, source, count) \
    ffx_linux_wcstombs_s((converted), (destination), (destination_size), (source), (count))

// The SDK's Vulkan source still spells its debug-label conversion through
// std::wstring_convert. Define a local element-wise substitute after loading
// filesystem, since libstdc++ uses its own wide codecvt types there.
#ifdef __cplusplus
#define wstring_convert ffx_linux_wstring_convert
#define codecvt_utf8    ffx_linux_codecvt_utf8
namespace std {
    template <typename Elem, unsigned long Maxcode = 0x10ffff, int Mode = 0>
    class ffx_linux_codecvt_utf8 {
    };

    template <typename Elem>
    class ffx_linux_wide_string {
    public:
        ffx_linux_wide_string() : value{} {}

        void push_back(Elem character, size_t& length) {
            if (length + 1 < sizeof(value) / sizeof(value[0]))
                value[length++] = character;
            value[length] = (Elem)0;
        }

        const Elem* c_str() const {
            return value;
        }

    private:
        Elem value[4096];
    };

    class ffx_linux_byte_string {
    public:
        ffx_linux_byte_string() : value{} {}

        void push_back(char character, size_t& length) {
            if (length + 1 < sizeof(value) / sizeof(value[0]))
                value[length++] = character;
            value[length] = '\0';
        }

        const char* c_str() const {
            return value;
        }

    private:
        char value[4096];
    };

    template <typename Codecvt, typename Elem = wchar_t>
    class ffx_linux_wstring_convert {
    public:
        ffx_linux_wide_string<Elem> from_bytes(const char* source) const {
            ffx_linux_wide_string<Elem> result;
            size_t length = 0;
            if (!source)
                return result;
            for (size_t index = 0; source[index] != '\0'; ++index) {
                const unsigned char value = (unsigned char)source[index];
                result.push_back((Elem)(value < 0x80u ? value : '?'), length);
            }
            return result;
        }

        ffx_linux_byte_string to_bytes(const Elem* source) const {
            ffx_linux_byte_string result;
            size_t length = 0;
            if (!source)
                return result;
            for (size_t index = 0; source[index] != (Elem)0; ++index) {
                const uint32_t value = (uint32_t)source[index];
                result.push_back((char)(value < 0x80u ? value : '?'), length);
            }
            return result;
        }
    };
} // namespace std
#endif
#endif
#define _countof(array) (sizeof(array) / sizeof((array)[0]))
#endif
