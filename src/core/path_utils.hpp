/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
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
    inline std::string path_to_utf8(const std::filesystem::path& p) {
#ifdef _WIN32
        // On Windows, convert wide string to UTF-8
        const std::wstring wstr = p.wstring();
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
        // On Linux/Mac, native encoding is UTF-8
        return p.string();
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
        // On Windows, convert UTF-8 string to wide string, then to path
        // Use c_str() and strlen to handle strings with embedded null characters
        // (common when using fixed-size buffers like ImGui InputText)
        const char* str = utf8_str.c_str();
        const size_t len = std::strlen(str);
        if (len == 0) {
            return std::filesystem::path();
        }

        const int size_needed = MultiByteToWideChar(CP_UTF8, 0, str,
                                                    static_cast<int>(len),
                                                    nullptr, 0);
        if (size_needed <= 0) {
            return std::filesystem::path();
        }

        std::wstring wstr(size_needed, 0);
        const int converted = MultiByteToWideChar(CP_UTF8, 0, str,
                                                  static_cast<int>(len),
                                                  &wstr[0], size_needed);
        if (converted <= 0) {
            return std::filesystem::path();
        }
        wstr.resize(converted);
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

} // namespace lfs::core
