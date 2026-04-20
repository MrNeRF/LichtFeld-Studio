/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
// clang-format off
#include <windows.h>
#include <shellapi.h>
// clang-format on
#endif

namespace lfs::core {

    /**
     * @brief Convert filesystem path to UTF-8 string for use with external libraries
     *
     * On Windows, std::filesystem::path::string() returns a string in the system codepage,
     * not UTF-8. This function ensures the returned string is always UTF-8 encoded.
     * On Linux/Mac, the native encoding is already UTF-8.
     *
     * @param p The filesystem path to convert
     * @return UTF-8 encoded string representation of the path
     */
    namespace detail {
        inline constexpr char32_t replacement_codepoint = 0xFFFD;

        inline bool is_valid_utf8(const std::string& s) {
            const auto* bytes = reinterpret_cast<const uint8_t*>(s.data());
            const size_t len = s.size();
            for (size_t i = 0; i < len;) {
                if (bytes[i] < 0x80) {
                    ++i;
                } else if ((bytes[i] & 0xE0) == 0xC0) {
                    if (i + 1 >= len || (bytes[i + 1] & 0xC0) != 0x80)
                        return false;
                    i += 2;
                } else if ((bytes[i] & 0xF0) == 0xE0) {
                    if (i + 2 >= len || (bytes[i + 1] & 0xC0) != 0x80 || (bytes[i + 2] & 0xC0) != 0x80)
                        return false;
                    i += 3;
                } else if ((bytes[i] & 0xF8) == 0xF0) {
                    if (i + 3 >= len || (bytes[i + 1] & 0xC0) != 0x80 || (bytes[i + 2] & 0xC0) != 0x80 ||
                        (bytes[i + 3] & 0xC0) != 0x80)
                        return false;
                    i += 4;
                } else {
                    return false;
                }
            }
            return true;
        }

        inline std::string sanitize_utf8(const std::string& s) {
            std::string result;
            result.reserve(s.size());
            const auto* bytes = reinterpret_cast<const uint8_t*>(s.data());
            const size_t len = s.size();
            for (size_t i = 0; i < len;) {
                if (bytes[i] < 0x80) {
                    result += static_cast<char>(bytes[i]);
                    ++i;
                } else if ((bytes[i] & 0xE0) == 0xC0 && i + 1 < len && (bytes[i + 1] & 0xC0) == 0x80) {
                    result += static_cast<char>(bytes[i]);
                    result += static_cast<char>(bytes[i + 1]);
                    i += 2;
                } else if ((bytes[i] & 0xF0) == 0xE0 && i + 2 < len && (bytes[i + 1] & 0xC0) == 0x80 &&
                           (bytes[i + 2] & 0xC0) == 0x80) {
                    result += static_cast<char>(bytes[i]);
                    result += static_cast<char>(bytes[i + 1]);
                    result += static_cast<char>(bytes[i + 2]);
                    i += 3;
                } else if ((bytes[i] & 0xF8) == 0xF0 && i + 3 < len && (bytes[i + 1] & 0xC0) == 0x80 &&
                           (bytes[i + 2] & 0xC0) == 0x80 && (bytes[i + 3] & 0xC0) == 0x80) {
                    result += static_cast<char>(bytes[i]);
                    result += static_cast<char>(bytes[i + 1]);
                    result += static_cast<char>(bytes[i + 2]);
                    result += static_cast<char>(bytes[i + 3]);
                    i += 4;
                } else {
                    // U+FFFD replacement character
                    result += "\xEF\xBF\xBD";
                    ++i;
                }
            }
            return result;
        }

        inline bool is_valid_unicode_scalar(const char32_t codepoint) {
            return codepoint <= 0x10FFFF && !(codepoint >= 0xD800 && codepoint <= 0xDFFF);
        }

        inline void append_utf8_codepoint(std::string& result, char32_t codepoint) {
            if (!is_valid_unicode_scalar(codepoint)) {
                codepoint = replacement_codepoint;
            }

            if (codepoint <= 0x7F) {
                result.push_back(static_cast<char>(codepoint));
            } else if (codepoint <= 0x7FF) {
                result.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
                result.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
            } else if (codepoint <= 0xFFFF) {
                result.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
                result.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
                result.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
            } else {
                result.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
                result.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
                result.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
                result.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
            }
        }

        inline void append_wchar_codepoint(std::wstring& result, char32_t codepoint) {
            if (!is_valid_unicode_scalar(codepoint)) {
                codepoint = replacement_codepoint;
            }

            if constexpr (sizeof(wchar_t) == 2) {
                if (codepoint <= 0xFFFF) {
                    result.push_back(static_cast<wchar_t>(codepoint));
                    return;
                }

                const char32_t adjusted = codepoint - 0x10000;
                result.push_back(static_cast<wchar_t>(0xD800 + (adjusted >> 10)));
                result.push_back(static_cast<wchar_t>(0xDC00 + (adjusted & 0x3FF)));
            } else {
                result.push_back(static_cast<wchar_t>(codepoint));
            }
        }

        inline char32_t decode_wchar_codepoint(const std::wstring& text, size_t& index) {
            if constexpr (sizeof(wchar_t) == 2) {
                const auto lead = static_cast<uint16_t>(text[index]);
                if (lead >= 0xD800 && lead <= 0xDBFF) {
                    if (index + 1 < text.size()) {
                        const auto trail = static_cast<uint16_t>(text[index + 1]);
                        if (trail >= 0xDC00 && trail <= 0xDFFF) {
                            ++index;
                            return 0x10000 + (((lead - 0xD800) << 10) | (trail - 0xDC00));
                        }
                    }
                    return replacement_codepoint;
                }
                if (lead >= 0xDC00 && lead <= 0xDFFF) {
                    return replacement_codepoint;
                }
                return lead;
            } else {
                return static_cast<char32_t>(text[index]);
            }
        }

        inline char32_t decode_utf8_codepoint(const std::string& text, size_t& index) {
            const auto* bytes = reinterpret_cast<const uint8_t*>(text.data());
            const size_t len = text.size();
            const uint8_t first = bytes[index];

            if (first < 0x80) {
                return first;
            }

            if ((first & 0xE0) == 0xC0) {
                if (index + 1 < len && (bytes[index + 1] & 0xC0) == 0x80) {
                    const char32_t codepoint = ((first & 0x1F) << 6) | (bytes[index + 1] & 0x3F);
                    if (codepoint >= 0x80) {
                        index += 1;
                        return codepoint;
                    }
                }
                return replacement_codepoint;
            }

            if ((first & 0xF0) == 0xE0) {
                if (index + 2 < len && (bytes[index + 1] & 0xC0) == 0x80 && (bytes[index + 2] & 0xC0) == 0x80) {
                    const char32_t codepoint =
                        ((first & 0x0F) << 12) | ((bytes[index + 1] & 0x3F) << 6) | (bytes[index + 2] & 0x3F);
                    if (codepoint >= 0x800 && is_valid_unicode_scalar(codepoint)) {
                        index += 2;
                        return codepoint;
                    }
                }
                return replacement_codepoint;
            }

            if ((first & 0xF8) == 0xF0) {
                if (index + 3 < len && (bytes[index + 1] & 0xC0) == 0x80 && (bytes[index + 2] & 0xC0) == 0x80 &&
                    (bytes[index + 3] & 0xC0) == 0x80) {
                    const char32_t codepoint = ((first & 0x07) << 18) | ((bytes[index + 1] & 0x3F) << 12) |
                                               ((bytes[index + 2] & 0x3F) << 6) | (bytes[index + 3] & 0x3F);
                    if (codepoint >= 0x10000 && is_valid_unicode_scalar(codepoint)) {
                        index += 3;
                        return codepoint;
                    }
                }
                return replacement_codepoint;
            }

            return replacement_codepoint;
        }
    } // namespace detail

    /**
     * @brief Convert a wide string to UTF-8 text.
     *
     * On Windows this uses the Win32 UTF-16 conversion APIs. On other platforms it
     * performs an explicit Unicode scalar conversion so the helper remains correct
     * even if it is reused outside filesystem-path code.
     */
    inline std::string wstring_to_utf8(const std::wstring& wstr) {
#ifdef _WIN32
        if (wstr.empty()) {
            return std::string();
        }

        const int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(),
                                                    static_cast<int>(wstr.size()),
                                                    nullptr, 0, nullptr, nullptr);
        if (size_needed <= 0) {
            return std::string();
        }

        std::string utf8_str(size_needed, 0);
        const int converted = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(),
                                                  static_cast<int>(wstr.size()),
                                                  &utf8_str[0], size_needed, nullptr, nullptr);
        if (converted <= 0) {
            return std::string();
        }
        utf8_str.resize(converted);
        return utf8_str;
#else
        std::string result;
        result.reserve(wstr.size());
        for (size_t i = 0; i < wstr.size(); ++i) {
            detail::append_utf8_codepoint(result, detail::decode_wchar_codepoint(wstr, i));
        }
        return result;
#endif
    }

    /**
     * @brief Convert UTF-8 text to a wide string.
     *
     * On Windows this uses the Win32 UTF-16 conversion APIs. On other platforms it
     * decodes UTF-8 into Unicode scalar values and stores them in the platform wide
     * representation.
     */
    inline std::wstring utf8_to_wstring(const std::string& utf8_str) {
#ifdef _WIN32
        const char* str = utf8_str.c_str();
        const size_t len = std::strlen(str);
        if (len == 0) {
            return std::wstring();
        }

        const int size_needed = MultiByteToWideChar(CP_UTF8, 0, str,
                                                    static_cast<int>(len),
                                                    nullptr, 0);
        if (size_needed <= 0) {
            return std::wstring();
        }

        std::wstring wstr(size_needed, 0);
        const int converted = MultiByteToWideChar(CP_UTF8, 0, str,
                                                  static_cast<int>(len),
                                                  &wstr[0], size_needed);
        if (converted <= 0) {
            return std::wstring();
        }
        wstr.resize(converted);
        return wstr;
#else
        std::wstring result;
        result.reserve(utf8_str.size());
        for (size_t i = 0; i < utf8_str.size(); ++i) {
            detail::append_wchar_codepoint(result, detail::decode_utf8_codepoint(utf8_str, i));
        }
        return result;
#endif
    }

    inline std::string path_to_utf8(const std::filesystem::path& p) {
#ifdef _WIN32
        return wstring_to_utf8(p.wstring());
#else
        // Linux filesystems are byte-oriented; filenames may not be valid UTF-8.
        std::string s = p.string();
        if (detail::is_valid_utf8(s))
            return s;
        return detail::sanitize_utf8(s);
#endif
    }

    /**
     * @brief Convert UTF-8 string to filesystem path
     *
     * On Windows, std::filesystem::path constructor from std::string interprets
     * the string as being in the system codepage. This function properly converts
     * UTF-8 strings to paths by converting to wide string first on Windows.
     * On Linux/Mac, the native encoding is already UTF-8.
     *
     * @param utf8_str UTF-8 encoded string to convert
     * @return filesystem path constructed from the UTF-8 string
     */
    inline std::filesystem::path utf8_to_path(const std::string& utf8_str) {
#ifdef _WIN32
        const std::wstring wstr = utf8_to_wstring(utf8_str);
        if (wstr.empty()) {
            return std::filesystem::path();
        }
        return std::filesystem::path(wstr);
#else
        // On Linux/Mac, native encoding is UTF-8, so use string directly
        // Also handle embedded nulls by using c_str()
        return std::filesystem::path(utf8_str.c_str());
#endif
    }

    /**
     * @brief Open an output file stream with proper Unicode path handling
     *
     * On Windows, std::ofstream constructor with std::filesystem::path may not work
     * correctly with Unicode paths in all implementations. This function ensures
     * proper handling by using the path object directly which MSVC handles correctly.
     *
     * @param path The filesystem path to open
     * @param mode The open mode (default: std::ios::out)
     * @param[out] stream Reference to store the opened ofstream
     * @return true if the file was opened successfully, false otherwise
     */
    inline bool open_file_for_write(
        const std::filesystem::path& path,
        std::ios_base::openmode mode,
        std::ofstream& stream) {
#ifdef _WIN32
        // On Windows, explicitly use wstring to open with Unicode support
        // This ensures we bypass any narrow string conversion issues
        stream.open(path.wstring(), mode);
#else
        // On Linux/Mac, standard ofstream works with UTF-8 paths
        stream.open(path, mode);
#endif
        return stream.is_open();
    }

    /**
     * @brief Open an output file stream for writing (convenience overload)
     *
     * @param path The filesystem path to open
     * @param[out] stream Reference to store the opened ofstream
     * @return true if the file was opened successfully, false otherwise
     */
    inline bool open_file_for_write(
        const std::filesystem::path& path,
        std::ofstream& stream) {
        return open_file_for_write(path, std::ios::out, stream);
    }

    /**
     * @brief Open an input file stream with proper Unicode path handling
     *
     * @param path The filesystem path to open
     * @param mode The open mode (default: std::ios::in)
     * @param[out] stream Reference to store the opened ifstream
     * @return true if the file was opened successfully, false otherwise
     */
    inline bool open_file_for_read(
        const std::filesystem::path& path,
        std::ios_base::openmode mode,
        std::ifstream& stream) {
#ifdef _WIN32
        // On Windows, explicitly use wstring to open with Unicode support
        stream.open(path.wstring(), mode);
#else
        // On Linux/Mac, standard ifstream works with UTF-8 paths
        stream.open(path, mode);
#endif
        return stream.is_open();
    }

    /**
     * @brief Open an input file stream for reading (convenience overload)
     *
     * @param path The filesystem path to open
     * @param[out] stream Reference to store the opened ifstream
     * @return true if the file was opened successfully, false otherwise
     */
    inline bool open_file_for_read(
        const std::filesystem::path& path,
        std::ifstream& stream) {
        return open_file_for_read(path, std::ios::in, stream);
    }

    /**
     * @brief Reveal a file or directory in the OS file manager.
     *
     * Highlights the entry where the platform supports it, otherwise opens
     * the parent directory. Returns false when the path does not exist or
     * the platform helper could not be launched.
     */
    inline bool reveal_in_file_manager(const std::filesystem::path& path) {
        std::error_code ec;
        const std::filesystem::path absolute = std::filesystem::weakly_canonical(path, ec);
        if (ec || absolute.empty() || !std::filesystem::exists(absolute, ec))
            return false;

#ifdef _WIN32
        const std::wstring args = L"/select,\"" + absolute.wstring() + L"\"";
        const auto result = ShellExecuteW(nullptr, L"open", L"explorer.exe",
                                          args.c_str(), nullptr, SW_SHOWNORMAL);
        return reinterpret_cast<INT_PTR>(result) > 32;
#else
        const std::string utf8_path = path_to_utf8(absolute);
        const std::string dbus_cmd =
            "gdbus call --session --dest org.freedesktop.FileManager1 "
            "--object-path /org/freedesktop/FileManager1 "
            "--method org.freedesktop.FileManager1.ShowItems "
            "'[\"file://" +
            utf8_path + "\"]' '' >/dev/null 2>&1";
        if (std::system(dbus_cmd.c_str()) == 0)
            return true;

        const std::filesystem::path parent = absolute.parent_path();
        if (parent.empty())
            return false;
        const std::string fallback = "xdg-open \"" + path_to_utf8(parent) + "\" >/dev/null 2>&1 &";
        return std::system(fallback.c_str()) == 0;
#endif
    }

} // namespace lfs::core
