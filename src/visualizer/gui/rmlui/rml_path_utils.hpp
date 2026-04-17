/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/path_utils.hpp"

#include <algorithm>
#include <filesystem>
#include <format>
#include <optional>
#include <string>
#include <string_view>

namespace lfs::vis::gui::rml_paths {

    namespace detail {
        inline bool isAsciiAlpha(const char ch) {
            return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z');
        }

        inline bool isAsciiAlnum(const char ch) {
            return (ch >= '0' && ch <= '9') || isAsciiAlpha(ch);
        }

        inline int hexValue(const char ch) {
            if (ch >= '0' && ch <= '9')
                return ch - '0';
            if (ch >= 'a' && ch <= 'f')
                return ch - 'a' + 10;
            if (ch >= 'A' && ch <= 'F')
                return ch - 'A' + 10;
            return -1;
        }
    } // namespace detail

    inline bool isWindowsDrivePath(const std::string_view text) {
        return text.size() >= 3 && detail::isAsciiAlpha(text[0]) && text[1] == ':' &&
               (text[2] == '/' || text[2] == '\\');
    }

    inline bool isUncPath(const std::string_view text) {
        return text.size() >= 2 &&
               ((text[0] == '/' && text[1] == '/') || (text[0] == '\\' && text[1] == '\\'));
    }

    inline bool isAbsoluteFilePath(const std::string_view text) {
#ifdef _WIN32
        return isWindowsDrivePath(text) || isUncPath(text);
#else
        return !text.empty() && text[0] == '/';
#endif
    }

    inline bool hasUriScheme(const std::string_view text) {
        const auto colon = text.find(':');
        if (colon == std::string_view::npos || colon == 0)
            return false;

#ifdef _WIN32
        if (colon == 1 && detail::isAsciiAlpha(text[0]))
            return false;
#endif

        if (!detail::isAsciiAlpha(text[0]))
            return false;

        for (size_t i = 1; i < colon; ++i) {
            const char ch = text[i];
            if (!detail::isAsciiAlnum(ch) && ch != '+' && ch != '-' && ch != '.')
                return false;
        }

        return true;
    }

    inline std::string percentDecode(const std::string_view text) {
        std::string decoded;
        decoded.reserve(text.size());

        for (size_t i = 0; i < text.size(); ++i) {
            if (text[i] != '%' || i + 2 >= text.size()) {
                decoded.push_back(text[i]);
                continue;
            }

            const int hi = detail::hexValue(text[i + 1]);
            const int lo = detail::hexValue(text[i + 2]);
            if (hi < 0 || lo < 0) {
                decoded.push_back(text[i]);
                continue;
            }

            decoded.push_back(static_cast<char>((hi << 4) | lo));
            i += 2;
        }

        return decoded;
    }

    inline std::string percentEncode(const std::string_view text,
                                     const std::string_view safe_chars = "/:._-~") {
        std::string encoded;
        encoded.reserve(text.size());

        for (const unsigned char ch : text) {
            if (detail::isAsciiAlnum(static_cast<char>(ch)) ||
                safe_chars.find(static_cast<char>(ch)) != std::string_view::npos) {
                encoded.push_back(static_cast<char>(ch));
                continue;
            }

            encoded += std::format("%{:02X}", static_cast<unsigned int>(ch));
        }

        return encoded;
    }

    inline std::string normalizeFilesystemPath(const std::filesystem::path& path) {
        std::string normalized = lfs::core::path_to_utf8(path.lexically_normal());
        std::replace(normalized.begin(), normalized.end(), '\\', '/');
        return normalized;
    }

    inline std::string filesystemPathToFileUri(const std::filesystem::path& path) {
        const std::string normalized = normalizeFilesystemPath(path);
        const std::string encoded = percentEncode(normalized);

#ifdef _WIN32
        if (normalized.starts_with("//"))
            return "file:" + encoded;

        return "file:///" + encoded;
#else
        return "file://" + encoded;
#endif
    }

    inline std::optional<std::filesystem::path> fileUriToPath(const std::string_view uri) {
        constexpr std::string_view kFileUriPrefix = "file://";
        if (!uri.starts_with(kFileUriPrefix))
            return std::nullopt;

        std::string decoded = percentDecode(uri.substr(kFileUriPrefix.size()));
        if (decoded.starts_with("localhost/")) {
            decoded.erase(0, std::string_view("localhost").size());
        }

#ifdef _WIN32
        if (decoded.size() >= 4 && decoded[0] == '/' && isWindowsDrivePath(decoded.substr(1))) {
            decoded.erase(0, 1);
        } else if (!decoded.empty() && decoded[0] != '/' && !isWindowsDrivePath(decoded)) {
            decoded = "//" + decoded;
        }
#endif

        return lfs::core::utf8_to_path(decoded);
    }

    inline std::optional<std::filesystem::path> pathReferenceToFilesystemPath(
        const std::string_view reference) {
        if (const auto file_uri_path = fileUriToPath(reference)) {
            return file_uri_path;
        }

        if (isAbsoluteFilePath(reference)) {
            return lfs::core::utf8_to_path(std::string(reference));
        }

        return std::nullopt;
    }

} // namespace lfs::vis::gui::rml_paths
