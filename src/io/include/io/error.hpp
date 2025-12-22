/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <expected>
#include <filesystem>
#include <format>
#include <string>
#include <string_view>

namespace lfs::io {

    /// Error codes for I/O operations (enables localized GUI messages)
    enum class ErrorCode {
        SUCCESS = 0,

        // Filesystem (100-199)
        PATH_NOT_FOUND = 100,
        NOT_A_DIRECTORY = 101,
        NOT_A_FILE = 102,
        PERMISSION_DENIED = 103,
        INSUFFICIENT_DISK_SPACE = 104,
        PATH_NOT_WRITABLE = 105,

        // Validation (200-299)
        INVALID_DATASET = 200,
        MISSING_REQUIRED_FILES = 201,
        CORRUPTED_DATA = 202,
        UNSUPPORTED_FORMAT = 203,
        EMPTY_DATASET = 204,
        INVALID_HEADER = 205,
        MALFORMED_JSON = 206,
        MASK_SIZE_MISMATCH = 207,

        // Save/Export (300-399)
        WRITE_FAILURE = 300,
        ENCODING_FAILED = 301,
        ARCHIVE_CREATION_FAILED = 302,

        // Load/Import (400-499)
        READ_FAILURE = 400,
        DECODING_FAILED = 401,

        // Operation (500-599)
        CANCELLED = 500,
        INTERNAL_ERROR = 502,
    };

    constexpr std::string_view error_code_to_string(ErrorCode code) {
        switch (code) {
        case ErrorCode::SUCCESS: return "Success";
        case ErrorCode::PATH_NOT_FOUND: return "Path not found";
        case ErrorCode::NOT_A_DIRECTORY: return "Not a directory";
        case ErrorCode::NOT_A_FILE: return "Not a file";
        case ErrorCode::PERMISSION_DENIED: return "Permission denied";
        case ErrorCode::INSUFFICIENT_DISK_SPACE: return "Insufficient disk space";
        case ErrorCode::PATH_NOT_WRITABLE: return "Path not writable";
        case ErrorCode::INVALID_DATASET: return "Invalid dataset";
        case ErrorCode::MISSING_REQUIRED_FILES: return "Missing required files";
        case ErrorCode::CORRUPTED_DATA: return "Corrupted data";
        case ErrorCode::UNSUPPORTED_FORMAT: return "Unsupported format";
        case ErrorCode::EMPTY_DATASET: return "Empty dataset";
        case ErrorCode::INVALID_HEADER: return "Invalid header";
        case ErrorCode::MALFORMED_JSON: return "Malformed JSON";
        case ErrorCode::MASK_SIZE_MISMATCH: return "Mask size mismatch";
        case ErrorCode::WRITE_FAILURE: return "Write failed";
        case ErrorCode::ENCODING_FAILED: return "Encoding failed";
        case ErrorCode::ARCHIVE_CREATION_FAILED: return "Archive creation failed";
        case ErrorCode::READ_FAILURE: return "Read failed";
        case ErrorCode::DECODING_FAILED: return "Decoding failed";
        case ErrorCode::CANCELLED: return "Cancelled";
        case ErrorCode::INTERNAL_ERROR: return "Internal error";
        default: return "Unknown error";
        }
    }

    /// Structured error with code, message, and optional path
    struct Error {
        ErrorCode code;
        std::string message;
        std::filesystem::path path;

        Error(ErrorCode c, std::string msg)
            : code(c),
              message(std::move(msg)) {}

        Error(ErrorCode c, std::string msg, std::filesystem::path p)
            : code(c),
              message(std::move(msg)),
              path(std::move(p)) {}

        [[nodiscard]] std::string format() const {
            if (path.empty()) {
                return std::format("[{}] {}", error_code_to_string(code), message);
            }
            return std::format("[{}] {}: {}", error_code_to_string(code), message, path.string());
        }

        [[nodiscard]] bool is(ErrorCode c) const { return code == c; }

        [[nodiscard]] bool is_filesystem_error() const {
            const int c = static_cast<int>(code);
            return c >= 100 && c < 200;
        }

        [[nodiscard]] bool is_validation_error() const {
            const int c = static_cast<int>(code);
            return c >= 200 && c < 300;
        }

        [[nodiscard]] bool is_save_error() const {
            const int c = static_cast<int>(code);
            return c >= 300 && c < 400;
        }

        [[nodiscard]] bool is_load_error() const {
            const int c = static_cast<int>(code);
            return c >= 400 && c < 500;
        }
    };

    template <typename T>
    using Result = std::expected<T, Error>;

    inline std::unexpected<Error> make_error(ErrorCode code, std::string message) {
        return std::unexpected(Error{code, std::move(message)});
    }

    inline std::unexpected<Error> make_error(ErrorCode code, std::string message,
                                             const std::filesystem::path& path) {
        return std::unexpected(Error{code, std::move(message), path});
    }

    /// Check disk space with safety margin (default 10%)
    [[nodiscard]] inline Result<std::uintmax_t> check_disk_space(
        const std::filesystem::path& path,
        std::uintmax_t required_bytes,
        float safety_margin = 1.1f) {

        std::error_code ec;
        auto check_path = path;

        if (!std::filesystem::is_directory(path, ec)) {
            check_path = path.parent_path();
            if (check_path.empty()) {
                check_path = std::filesystem::current_path(ec);
            }
        }

        if (!std::filesystem::exists(check_path, ec)) {
            auto parent = check_path;
            while (!parent.empty() && !std::filesystem::exists(parent, ec)) {
                parent = parent.parent_path();
            }
            if (parent.empty()) {
                return make_error(ErrorCode::PATH_NOT_FOUND,
                                  "Cannot determine disk space", path);
            }
            check_path = parent;
        }

        const auto space_info = std::filesystem::space(check_path, ec);
        if (ec) {
            return make_error(ErrorCode::PERMISSION_DENIED,
                              std::format("Cannot check disk space: {}", ec.message()), check_path);
        }

        const auto required_with_margin = static_cast<std::uintmax_t>(
            static_cast<double>(required_bytes) * safety_margin);

        if (space_info.available < required_with_margin) {
            constexpr double MB = 1024.0 * 1024.0;
            return make_error(ErrorCode::INSUFFICIENT_DISK_SPACE,
                              std::format("Need {:.1f} MB but only {:.1f} MB available",
                                          static_cast<double>(required_with_margin) / MB,
                                          static_cast<double>(space_info.available) / MB),
                              check_path);
        }

        return space_info.available;
    }

    /// Verify path is writable (creates parent dirs if needed)
    [[nodiscard]] inline Result<void> verify_writable(const std::filesystem::path& path) {
        std::error_code ec;

        if (std::filesystem::exists(path, ec)) {
            const auto perms = std::filesystem::status(path, ec).permissions();
            if (ec) {
                return make_error(ErrorCode::PERMISSION_DENIED,
                                  std::format("Cannot check permissions: {}", ec.message()), path);
            }
            if ((perms & std::filesystem::perms::owner_write) == std::filesystem::perms::none) {
                return make_error(ErrorCode::PATH_NOT_WRITABLE, "Path not writable", path);
            }
            return {};
        }

        auto parent = path.parent_path();
        if (parent.empty()) {
            parent = std::filesystem::current_path(ec);
        }

        if (!std::filesystem::exists(parent, ec)) {
            if (!std::filesystem::create_directories(parent, ec)) {
                return make_error(ErrorCode::PERMISSION_DENIED,
                                  std::format("Cannot create directory: {}", ec.message()), parent);
            }
        }

        return {};
    }

} // namespace lfs::io
