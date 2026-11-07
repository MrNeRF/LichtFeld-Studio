// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include "media/frame_sink.hpp"
#include <algorithm>
#include <array>
#include <cerrno>
#include <new>
#include <system_error>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
namespace lfs::media::detail {
    inline constexpr std::size_t max_metadata_sidecar_bytes = 32u * 1024 * 1024;
    inline Error sidecarError(ErrorCode code, std::string message) {
        return make_error({.code = code, .domain = ErrorDomain::IO, .detail = std::move(message), .detection = LFS_SOURCE_SITE_CURRENT()});
    }
    inline Error sidecarIoError(std::error_code error, const char* operation, const std::filesystem::path& path) {
        const auto code = error == std::errc::no_such_file_or_directory       ? ErrorCode::NotFound
                          : error == std::errc::permission_denied             ? ErrorCode::PermissionDenied
                          : error == std::errc::too_many_symbolic_link_levels ? ErrorCode::FailedPrecondition
                                                                              : ErrorCode::Unavailable;
        // Windows category messages may use the active code page. Keep the
        // transport diagnostic UTF-8 and retain the exact native code/category.
        return make_error({.code = code, .domain = ErrorDomain::IO, .detail = std::string(operation) + " failed (path=" + core::path_to_utf8(path) + ", native=" + std::to_string(error.value()) + ")", .detection = LFS_SOURCE_SITE_CURRENT(), .native = NativeError{ErrorDomain::IO, error.value(), error.category().name()}})
            .with_context(operation, LFS_SOURCE_SITE_CURRENT());
    }
    struct SidecarObservation {
#ifdef _WIN32
        using Native = BY_HANDLE_FILE_INFORMATION;
#else
        using Native = struct stat;
#endif
        Native value{};
        bool sameFile(const SidecarObservation& other) const noexcept {
#ifdef _WIN32
            return value.dwVolumeSerialNumber == other.value.dwVolumeSerialNumber &&
                   value.nFileIndexHigh == other.value.nFileIndexHigh && value.nFileIndexLow == other.value.nFileIndexLow;
#else
            return value.st_dev == other.value.st_dev && value.st_ino == other.value.st_ino;
#endif
        }
        bool sameVersion(const SidecarObservation& other) const noexcept {
            if (!sameFile(other))
                return false;
#ifdef _WIN32
            return value.nFileSizeHigh == other.value.nFileSizeHigh && value.nFileSizeLow == other.value.nFileSizeLow &&
                   CompareFileTime(&value.ftCreationTime, &other.value.ftCreationTime) == 0 &&
                   CompareFileTime(&value.ftLastWriteTime, &other.value.ftLastWriteTime) == 0;
#elif defined(__APPLE__)
            return value.st_size == other.value.st_size &&
                   value.st_mtimespec.tv_sec == other.value.st_mtimespec.tv_sec && value.st_mtimespec.tv_nsec == other.value.st_mtimespec.tv_nsec &&
                   value.st_ctimespec.tv_sec == other.value.st_ctimespec.tv_sec && value.st_ctimespec.tv_nsec == other.value.st_ctimespec.tv_nsec;
#else
            return value.st_size == other.value.st_size &&
                   value.st_mtim.tv_sec == other.value.st_mtim.tv_sec && value.st_mtim.tv_nsec == other.value.st_mtim.tv_nsec &&
                   value.st_ctim.tv_sec == other.value.st_ctim.tv_sec && value.st_ctim.tv_nsec == other.value.st_ctim.tv_nsec;
#endif
        }
    };
    // Bounded I/O for explicit metadata companions. Format parsers validate
    // their payload separately. Validate the opened
    // object, not only a pathname checked before opening. Nonblocking open
    // prevents a substituted FIFO from hanging the caller on POSIX.
    class SidecarFile {
    public:
        SidecarFile() = default;
        SidecarFile(const SidecarFile&) = delete;
        SidecarFile& operator=(const SidecarFile&) = delete;
        ~SidecarFile() {
#ifdef _WIN32
            if (handle_ != INVALID_HANDLE_VALUE)
                CloseHandle(handle_);
#else
            if (descriptor_ >= 0)
                ::close(descriptor_);
#endif
        }
        Result<std::size_t> open(const std::filesystem::path& path, std::size_t limit, bool metadata_only = false) {
#ifdef _WIN32
            if (handle_ != INVALID_HANDLE_VALUE)
                return sidecarError(ErrorCode::FailedPrecondition, "Metadata reader already owns an open file");
            path_ = path;
            handle_ = CreateFileW(core::native_fs::native_path(path).c_str(), metadata_only ? FILE_READ_ATTRIBUTES : GENERIC_READ,
                                  metadata_only ? FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE : FILE_SHARE_READ, nullptr,
                                  OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
            if (handle_ == INVALID_HANDLE_VALUE)
                return sidecarIoError({static_cast<int>(GetLastError()), std::system_category()}, "Open metadata file", path);
            if (GetFileType(handle_) != FILE_TYPE_DISK)
                return sidecarError(ErrorCode::FailedPrecondition, "Metadata file must be regular and non-symlink (path=" + core::path_to_utf8(path) + ")");
            if (!GetFileInformationByHandle(handle_, &initial_))
                return sidecarIoError({static_cast<int>(GetLastError()), std::system_category()}, "Inspect metadata file", path);
            if (initial_.dwFileAttributes & (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DIRECTORY))
                return sidecarError(ErrorCode::FailedPrecondition, "Metadata file must be regular and non-symlink (path=" + core::path_to_utf8(path) + ")");
            const auto size = (std::uint64_t(initial_.nFileSizeHigh) << 32) | initial_.nFileSizeLow;
#else
            if (descriptor_ >= 0)
                return sidecarError(ErrorCode::FailedPrecondition, "Metadata reader already owns an open file");
            path_ = path;
            int access = O_RDONLY;
#if defined(__linux__)
            if (metadata_only)
                access = O_PATH;
#elif defined(__APPLE__)
            if (metadata_only)
                access = O_EVTONLY;
#endif
            descriptor_ = ::open(core::native_fs::native_path(path).c_str(), access | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
            if (descriptor_ < 0)
                return sidecarIoError({errno, std::generic_category()}, "Open metadata file", path);
            if (::fstat(descriptor_, &initial_) != 0)
                return sidecarIoError({errno, std::generic_category()}, "Inspect metadata file", path);
            if (!S_ISREG(initial_.st_mode) || initial_.st_size < 0)
                return sidecarError(ErrorCode::FailedPrecondition, "Metadata file must be regular and non-symlink (path=" + core::path_to_utf8(path) + ")");
            const auto size = static_cast<std::uintmax_t>(initial_.st_size);
#endif
            if (size > limit)
                return sidecarError(ErrorCode::ResourceExhausted, "Metadata file exceeds its byte limit (path=" + core::path_to_utf8(path) + ", size=" + std::to_string(size) + ", limit=" + std::to_string(limit) + ")");
            return static_cast<std::size_t>(size);
        }
        Result<std::size_t> read(std::span<std::uint8_t> output) {
#ifdef _WIN32
            DWORD count = 0;
            if (!ReadFile(handle_, output.data(), static_cast<DWORD>(output.size()), &count, nullptr))
                return sidecarIoError({static_cast<int>(GetLastError()), std::system_category()}, "Read metadata file", path_);
            return static_cast<std::size_t>(count);
#else
            ssize_t count;
            do {
                count = ::read(descriptor_, output.data(), output.size());
            } while (count < 0 && errno == EINTR);
            if (count < 0)
                return sidecarIoError({errno, std::generic_category()}, "Read metadata file", path_);
            return static_cast<std::size_t>(count);
#endif
        }
        SidecarObservation observation() const noexcept { return {initial_}; }
        bool sameFile(const SidecarFile& other) const noexcept {
#ifdef _WIN32
            return handle_ != INVALID_HANDLE_VALUE && other.handle_ != INVALID_HANDLE_VALUE && observation().sameFile(other.observation());
#else
            return descriptor_ >= 0 && other.descriptor_ >= 0 && observation().sameFile(other.observation());
#endif
        }
        SinkResult verifyUnchanged() const {
#ifdef _WIN32
            BY_HANDLE_FILE_INFORMATION current{};
            if (!GetFileInformationByHandle(handle_, &current))
                return SinkResult::failure(sidecarIoError({static_cast<int>(GetLastError()), std::system_category()}, "Inspect metadata file after read", path_));
#else
            struct stat current {};
            if (::fstat(descriptor_, &current) != 0)
                return SinkResult::failure(sidecarIoError({errno, std::generic_category()}, "Inspect metadata file after read", path_));
#endif
            if (!observation().sameVersion({current}))
                return SinkResult::failure(sidecarError(ErrorCode::DataLoss, "Metadata file changed while being read (path=" + core::path_to_utf8(path_) + ")"));
            // The Windows handle excludes writers and replacement while open.
            return {};
        }

    private:
        std::filesystem::path path_;
#ifdef _WIN32
        HANDLE handle_ = INVALID_HANDLE_VALUE;
        BY_HANDLE_FILE_INFORMATION initial_{};
#else
        int descriptor_ = -1;
        struct stat initial_ {};
#endif
    };

    // Shared by the native reader and deterministic fault contracts. The reader
    // owns its file; no callback or borrowed byte view survives this call.
    template <class Reader>
    Result<std::vector<std::uint8_t>> readSidecarBytes(Reader& file, std::size_t size) {
        if (size > max_metadata_sidecar_bytes)
            return sidecarError(ErrorCode::ResourceExhausted, "Metadata sidecar exceeds its byte limit (size=" + std::to_string(size) + ", limit=" + std::to_string(max_metadata_sidecar_bytes) + ")");
        try {
            std::vector<std::uint8_t> bytes(size);
            std::size_t offset = 0;
            while (offset < bytes.size()) {
                const auto buffer = std::span(bytes).subspan(offset, std::min(std::size_t{65536}, bytes.size() - offset));
                const auto count = file.read(buffer);
                if (!count)
                    return count.error();
                if (!*count || *count > buffer.size())
                    return sidecarError(ErrorCode::DataLoss, "Metadata sidecar was truncated or returned an invalid read size");
                offset += *count;
            }
            std::array<std::uint8_t, 1> extra{};
            const auto count = file.read(extra);
            if (!count)
                return count.error();
            if (*count)
                return sidecarError(ErrorCode::DataLoss, "Metadata sidecar grew while being read");
            if (const auto stable = file.verifyUnchanged(); !stable)
                return stable.error();
            return bytes;
        } catch (const std::bad_alloc&) {
            return sidecarError(ErrorCode::ResourceExhausted, "Metadata sidecar read allocation failed");
        }
    }
} // namespace lfs::media::detail
