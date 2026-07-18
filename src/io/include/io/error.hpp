/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/export.hpp"
#include "core/path_utils.hpp"
#include <expected>
#include <filesystem>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

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
        DEPTH_SIZE_MISMATCH = 208,
        NORMAL_SIZE_MISMATCH = 209,

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

        // Resource (600-699)
        RESOURCE_EXHAUSTED = 600,
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
        case ErrorCode::DEPTH_SIZE_MISMATCH: return "Depth size mismatch";
        case ErrorCode::NORMAL_SIZE_MISMATCH: return "Normal size mismatch";
        case ErrorCode::WRITE_FAILURE: return "Write failed";
        case ErrorCode::ENCODING_FAILED: return "Encoding failed";
        case ErrorCode::ARCHIVE_CREATION_FAILED: return "Archive creation failed";
        case ErrorCode::READ_FAILURE: return "Read failed";
        case ErrorCode::DECODING_FAILED: return "Decoding failed";
        case ErrorCode::CANCELLED: return "Cancelled";
        case ErrorCode::INTERNAL_ERROR: return "Internal error";
        case ErrorCode::RESOURCE_EXHAUSTED: return "Resource exhausted";
        default: return "Unknown error";
        }
    }

    /// Structured error with code, message, and optional path
    struct LFS_IO_API Error {
        ErrorCode code;
        std::string message;
        std::filesystem::path path;
        size_t required_bytes = 0;
        size_t available_bytes = 0;

        Error(ErrorCode c, std::string msg)
            : code(c),
              message(std::move(msg)) {}

        Error(ErrorCode c, std::string msg, std::filesystem::path p)
            : code(c),
              message(std::move(msg)),
              path(std::move(p)) {}

        Error(ErrorCode c, std::string msg, std::filesystem::path p, size_t req, size_t avail)
            : code(c),
              message(std::move(msg)),
              path(std::move(p)),
              required_bytes(req),
              available_bytes(avail) {}

        [[nodiscard]] std::string format() const {
            const auto code_str = error_code_to_string(code);
            const auto path_str = lfs::core::path_to_utf8(path);

            if (message.empty() && path.empty()) {
                return std::format("[{}]", code_str);
            }
            if (message.empty()) {
                return std::format("[{}] {}", code_str, path_str);
            }
            if (path.empty()) {
                return std::format("[{}] {}", code_str, message);
            }
            return std::format("[{}] {}: {}", code_str, message, path_str);
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
            return std::unexpected(Error{
                ErrorCode::INSUFFICIENT_DISK_SPACE,
                std::format("Need {:.1f} MB but only {:.1f} MB available",
                            static_cast<double>(required_with_margin) / MB,
                            static_cast<double>(space_info.available) / MB),
                check_path,
                required_with_margin,
                static_cast<size_t>(space_info.available)});
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

    // Phase 1 error-architecture adapter: maps this legacy io::Error into the
    // stable lfs::ErrorCode/ErrorDomain taxonomy for callers migrating to
    // lfs::Result<T>. This mapping is a reasonable best-effort starting
    // point, not the frozen per-format taxonomy Phase 3A defines for PLY
    // (NotFound/InvalidHeader/TruncatedData/...); it exists so any call site
    // can bridge today without waiting for that phase. Header-only: this
    // header must not be included from a CUDA translation unit as a result
    // (core/error.hpp already enforces that with a hard #error if it is).
    // A deliberately-divergent sibling map lives in src/python/lfs/py_io.cpp
    // (map_io_code) with different NotFound/InvalidArgument biases and the
    // `requested_bytes` field-name variant. When adding an ErrorCode, update both.
    // NOTE: this map's `required_bytes` SmallField key is NOT in the wire
    // envelope allowlist (core/error_envelope.cpp kAllowlistedFields), so io
    // byte-counts never reach wire details while python's `requested_bytes`
    // does; tracked as post-campaign P11-L1.
    constexpr lfs::ErrorCode to_lfs_error_code(const ErrorCode code) noexcept {
        switch (code) {
        case ErrorCode::SUCCESS: return lfs::ErrorCode::Internal; // precondition: caller has a failure
        case ErrorCode::PATH_NOT_FOUND: return lfs::ErrorCode::NotFound;
        case ErrorCode::NOT_A_DIRECTORY: return lfs::ErrorCode::InvalidArgument;
        case ErrorCode::NOT_A_FILE: return lfs::ErrorCode::InvalidArgument;
        case ErrorCode::PERMISSION_DENIED: return lfs::ErrorCode::PermissionDenied;
        case ErrorCode::INSUFFICIENT_DISK_SPACE: return lfs::ErrorCode::ResourceExhausted;
        case ErrorCode::PATH_NOT_WRITABLE: return lfs::ErrorCode::PermissionDenied;
        case ErrorCode::INVALID_DATASET: return lfs::ErrorCode::InvalidArgument;
        case ErrorCode::MISSING_REQUIRED_FILES: return lfs::ErrorCode::NotFound;
        case ErrorCode::CORRUPTED_DATA: return lfs::ErrorCode::DataLoss;
        case ErrorCode::UNSUPPORTED_FORMAT: return lfs::ErrorCode::Unsupported;
        case ErrorCode::EMPTY_DATASET: return lfs::ErrorCode::InvalidArgument;
        case ErrorCode::INVALID_HEADER: return lfs::ErrorCode::DataLoss;
        case ErrorCode::MALFORMED_JSON: return lfs::ErrorCode::DataLoss;
        case ErrorCode::MASK_SIZE_MISMATCH: return lfs::ErrorCode::InvalidArgument;
        case ErrorCode::DEPTH_SIZE_MISMATCH: return lfs::ErrorCode::InvalidArgument;
        case ErrorCode::NORMAL_SIZE_MISMATCH: return lfs::ErrorCode::InvalidArgument;
        case ErrorCode::WRITE_FAILURE: return lfs::ErrorCode::Internal;
        case ErrorCode::ENCODING_FAILED: return lfs::ErrorCode::Internal;
        case ErrorCode::ARCHIVE_CREATION_FAILED: return lfs::ErrorCode::Internal;
        case ErrorCode::READ_FAILURE: return lfs::ErrorCode::Internal;
        case ErrorCode::DECODING_FAILED: return lfs::ErrorCode::DataLoss;
        case ErrorCode::CANCELLED: return lfs::ErrorCode::Cancelled;
        case ErrorCode::INTERNAL_ERROR: return lfs::ErrorCode::Internal;
        case ErrorCode::RESOURCE_EXHAUSTED: return lfs::ErrorCode::ResourceExhausted;
        }
        return lfs::ErrorCode::Internal;
    }

    // `legacy` must represent an actual failure (never call with
    // ErrorCode::SUCCESS). Carries path/required_bytes/available_bytes
    // through as SmallFields and records one detection frame at `site`.
    [[nodiscard]] inline lfs::Error to_lfs_error(const Error& legacy, const lfs::core::SourceSite site) {
        lfs::SmallFields fields;
        if (!legacy.path.empty()) {
            fields.add("path", lfs::core::path_to_utf8(legacy.path));
        }
        if (legacy.required_bytes != 0) {
            fields.add("required_bytes", static_cast<std::uint64_t>(legacy.required_bytes));
        }
        if (legacy.available_bytes != 0) {
            fields.add("available_bytes", static_cast<std::uint64_t>(legacy.available_bytes));
        }
        return lfs::make_error(lfs::ErrorInit{
            .code = to_lfs_error_code(legacy.code),
            .domain = lfs::ErrorDomain::IO,
            .severity = lfs::Severity::Error,
            .retryability = lfs::Retryability::NotRetryable,
            .operation_id = lfs::OperationId{},
            .user_message = legacy.message,
            .detail = legacy.format(),
            .detection = site,
            .fields = std::move(fields),
            .native = std::nullopt,
        });
    }

    // A non-fatal condition attached to an otherwise-successful load: rows
    // discarded, a feature degraded, etc. Carries the same stable ErrorCode
    // taxonomy as a failure so a diagnostic and a failure of the same class
    // render identically. Never itself a failure — see Section 5.4
    // ("Warnings are separate Diagnostic values, not failures hidden in a
    // successful result").
    struct Diagnostic {
        lfs::ErrorCode code;
        std::string message;
        lfs::SmallFields fields;
    };

    // A successful load's value plus zero or more non-fatal Diagnostics
    // collected while producing it (Section 7.1 row 3A: `Result<LoadOutcome<SplatData>>`).
    template <class T>
    struct LoadOutcome {
        T value;
        std::vector<Diagnostic> warnings;
    };

    namespace detail {

        // Linear scan is fine here: at most kMaxErrorContextFrames (16) frames,
        // each with at most kMaxFieldsPerFrame (16) fields — cold path only.
        [[nodiscard]] inline std::optional<std::string> find_error_field_string(
            const lfs::Error& error, const std::string_view key) {
            for (const auto& frame : error.frames()) {
                for (const auto& entry : frame.fields.entries()) {
                    if (entry.key == key) {
                        if (const auto* value = std::get_if<std::string>(&entry.value)) {
                            return *value;
                        }
                    }
                }
            }
            return std::nullopt;
        }

    } // namespace detail

    // Inverse of to_lfs_error_code: maps the stable lfs taxonomy back onto this
    // module's narrower legacy codes for callers that have not migrated. Not a
    // bijection — several lfs::ErrorCode values collapse onto INTERNAL_ERROR
    // because this legacy enum has no matching bucket; that is an accepted,
    // documented narrowing, not a bug to fix here.
    constexpr ErrorCode from_lfs_error_code(const lfs::ErrorCode code) noexcept {
        switch (code) {
        case lfs::ErrorCode::Cancelled: return ErrorCode::CANCELLED;
        case lfs::ErrorCode::InvalidArgument: return ErrorCode::INVALID_HEADER;
        case lfs::ErrorCode::BoundsViolation: return ErrorCode::CORRUPTED_DATA;
        case lfs::ErrorCode::FailedPrecondition: return ErrorCode::INVALID_DATASET;
        case lfs::ErrorCode::NotFound: return ErrorCode::PATH_NOT_FOUND;
        case lfs::ErrorCode::PermissionDenied: return ErrorCode::PERMISSION_DENIED;
        case lfs::ErrorCode::AlreadyExists: return ErrorCode::INTERNAL_ERROR;
        case lfs::ErrorCode::ResourceExhausted: return ErrorCode::RESOURCE_EXHAUSTED;
        case lfs::ErrorCode::DeadlineExceeded: return ErrorCode::INTERNAL_ERROR;
        case lfs::ErrorCode::Unavailable: return ErrorCode::INTERNAL_ERROR;
        case lfs::ErrorCode::DataLoss: return ErrorCode::CORRUPTED_DATA;
        case lfs::ErrorCode::Unsupported: return ErrorCode::UNSUPPORTED_FORMAT;
        case lfs::ErrorCode::DeviceLost: return ErrorCode::INTERNAL_ERROR;
        case lfs::ErrorCode::Internal: return ErrorCode::INTERNAL_ERROR;
        case lfs::ErrorCode::ContractViolation: return ErrorCode::INTERNAL_ERROR;
        }
        return ErrorCode::INTERNAL_ERROR;
    }

    // `error` must represent an actual failure. Extracts `path` from the
    // outermost-to-innermost context frame that carries one (every call site in
    // this phase attaches it via with_context at its single outer catch), and
    // prefers user_message() over detail() for the legacy Error::message field.
    [[nodiscard]] inline Error from_lfs_error(const lfs::Error& error) {
        const ErrorCode code = from_lfs_error_code(error.code());
        std::string message(error.user_message().empty() ? error.detail() : error.user_message());
        if (const auto path_str = detail::find_error_field_string(error, "path")) {
            return Error{code, std::move(message), lfs::core::utf8_to_path(*path_str)};
        }
        return Error{code, std::move(message)};
    }

} // namespace lfs::io
