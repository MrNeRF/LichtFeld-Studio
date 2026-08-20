/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/project_container.hpp"

#include "project_container_internal.hpp"

#include "core/path_utils.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <format>
#include <limits>
#include <mutex>
#include <new>
#include <system_error>
#include <thread>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif

namespace lfs::io::project {

    namespace {

        [[nodiscard]] std::filesystem::path
        normalized_lock_anchor(
            const std::filesystem::path& path) noexcept {
            std::error_code error;
            auto absolute =
                std::filesystem::absolute(path, error);
            return (error ? path : absolute)
                .lexically_normal();
        }

    } // namespace

    struct WriterLockLease::Impl {
        Impl(detail::WriterLock lock_in,
             std::filesystem::path anchor_in)
            : lock(std::move(lock_in)),
              anchor(std::move(anchor_in)) {}

        mutable std::mutex mutex;
        std::optional<detail::WriterLock> lock;
        std::filesystem::path anchor;
        bool release_requested = false;
    };

    WriterLockLease::WriterLockLease() noexcept = default;
    WriterLockLease::WriterLockLease(
        const WriterLockLease&) noexcept = default;
    WriterLockLease& WriterLockLease::operator=(
        const WriterLockLease&) noexcept = default;
    WriterLockLease::WriterLockLease(
        WriterLockLease&&) noexcept = default;
    WriterLockLease& WriterLockLease::operator=(
        WriterLockLease&&) noexcept = default;
    WriterLockLease::~WriterLockLease() {
        if (!impl_) {
            return;
        }
        const std::lock_guard lock(impl_->mutex);
        if (impl_->release_requested &&
            impl_->lock.has_value() &&
            impl_.use_count() == 1) {
            impl_->lock.reset();
        }
    }

    WriterLockLease::WriterLockLease(
        std::shared_ptr<Impl> impl) noexcept
        : impl_(std::move(impl)) {}

    lfs::Result<WriterLockLease>
    WriterLockLease::acquire(
        const std::filesystem::path& project_path) {
        auto lock =
            detail::WriterLock::acquire(project_path);
        if (!lock) {
            return std::move(lock).error();
        }
        return WriterLockLease(std::make_shared<Impl>(
            std::move(*lock),
            normalized_lock_anchor(project_path)));
    }

    bool WriterLockLease::valid() const noexcept {
        if (!impl_) {
            return false;
        }
        const std::lock_guard lock(impl_->mutex);
        return impl_->lock.has_value();
    }

    bool WriterLockLease::owns(
        const std::filesystem::path& project_path) const noexcept {
        if (!impl_) {
            return false;
        }
        const std::lock_guard lock(impl_->mutex);
        return impl_->lock.has_value() &&
               impl_->anchor ==
                   normalized_lock_anchor(project_path);
    }

    void WriterLockLease::release() noexcept {
        if (!impl_) {
            return;
        }
        const std::lock_guard lock(impl_->mutex);
        impl_->release_requested = true;
        if (impl_.use_count() == 1) {
            impl_->lock.reset();
        }
    }

} // namespace lfs::io::project

namespace lfs::io::project::detail {

    namespace {

        lfs::ErrorCode native_error_code(const int error, const bool writing) noexcept {
#ifdef _WIN32
            switch (static_cast<DWORD>(error)) {
            case ERROR_FILE_NOT_FOUND:
            case ERROR_PATH_NOT_FOUND:
                return lfs::ErrorCode::NotFound;
            case ERROR_ACCESS_DENIED:
            case ERROR_SHARING_VIOLATION:
            case ERROR_LOCK_VIOLATION:
                return lfs::ErrorCode::PermissionDenied;
            case ERROR_DISK_FULL:
            case ERROR_HANDLE_DISK_FULL:
                return lfs::ErrorCode::ResourceExhausted;
            default:
                return writing ? lfs::ErrorCode::Internal : lfs::ErrorCode::DataLoss;
            }
#else
            switch (error) {
            case ENOENT: return lfs::ErrorCode::NotFound;
            case EACCES:
            case EPERM: return lfs::ErrorCode::PermissionDenied;
            case ENOSPC:
            case EDQUOT:
            case EFBIG: return lfs::ErrorCode::ResourceExhausted;
            case EAGAIN: return lfs::ErrorCode::Unavailable;
#if EWOULDBLOCK != EAGAIN
            case EWOULDBLOCK: return lfs::ErrorCode::Unavailable;
#endif
            default: return writing ? lfs::ErrorCode::Internal : lfs::ErrorCode::DataLoss;
            }
#endif
        }

        lfs::Result<void> status_failure(lfs::Error error) {
            return lfs::Result<void>::failure(std::move(error));
        }

#ifdef _WIN32
        lfs::Result<DWORD> wait_for_overlapped(HANDLE handle, OVERLAPPED& operation,
                                               const BOOL immediate_result,
                                               const std::filesystem::path& path,
                                               const std::string_view action,
                                               const bool writing) {
            if (immediate_result) {
                DWORD transferred = 0;
                if (!GetOverlappedResult(handle, &operation, &transferred, TRUE)) {
                    const DWORD error = GetLastError();
                    return project_error(native_error_code(static_cast<int>(error), writing),
                                         writing ? "The project could not be written."
                                                 : "The project could not be read.",
                                         std::format("{} failed with Windows error {}", action, error),
                                         path, std::nullopt, {}, static_cast<std::int64_t>(error),
                                         "Win32");
                }
                return transferred;
            }
            const DWORD error = GetLastError();
            if (error != ERROR_IO_PENDING) {
                return project_error(native_error_code(static_cast<int>(error), writing),
                                     writing ? "The project could not be written."
                                             : "The project could not be read.",
                                     std::format("{} failed with Windows error {}", action, error),
                                     path, std::nullopt, {}, static_cast<std::int64_t>(error),
                                     "Win32");
            }
            DWORD transferred = 0;
            if (!GetOverlappedResult(handle, &operation, &transferred, TRUE)) {
                const DWORD completion_error = GetLastError();
                return project_error(
                    native_error_code(static_cast<int>(completion_error), writing),
                    writing ? "The project could not be written." : "The project could not be read.",
                    std::format("{} completion failed with Windows error {}", action,
                                completion_error),
                    path, std::nullopt, {}, static_cast<std::int64_t>(completion_error), "Win32");
            }
            return transferred;
        }

        OVERLAPPED operation_at(const std::uint64_t offset) noexcept {
            OVERLAPPED operation{};
            operation.Offset = static_cast<DWORD>(offset & 0xffffffffu);
            operation.OffsetHigh = static_cast<DWORD>(offset >> 32);
            return operation;
        }
#endif

    } // namespace

    lfs::Error project_error(
        const lfs::ErrorCode code, std::string user_message, std::string detail,
        const std::filesystem::path& path, const std::optional<std::uint64_t> offset,
        const std::string_view field, const std::optional<std::int64_t> native_code,
        const std::string_view native_name) {
        lfs::SmallFields fields;
        if (!path.empty()) {
            fields.add("path", lfs::core::path_to_utf8(path));
        }
        if (offset.has_value()) {
            fields.add("offset", *offset);
            detail = std::format("0x{:016x} {}: {}", *offset,
                                 field.empty() ? std::string_view{"container"} : field, detail);
        }
        if (!field.empty()) {
            fields.add("field", field);
        }

        std::optional<lfs::NativeError> native;
        if (native_code.has_value()) {
            native = lfs::NativeError{
                .domain = lfs::ErrorDomain::IO,
                .code = *native_code,
                .name = std::string(native_name),
            };
        }
        return lfs::make_error(lfs::ErrorInit{
            .code = code,
            .domain = lfs::ErrorDomain::IO,
            .operation_id = {},
            .user_message = std::move(user_message),
            .detail = std::move(detail),
            .detection = LFS_SOURCE_SITE_CURRENT(),
            .fields = std::move(fields),
            .native = std::move(native),
        });
    }

    lfs::Result<std::uint64_t>
    checked_add(const std::uint64_t lhs, const std::uint64_t rhs,
                const std::filesystem::path& path, const std::uint64_t field_offset,
                const std::string_view field) {
        if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
            return project_error(
                lfs::ErrorCode::BoundsViolation, "The project contains an invalid byte range.",
                std::format("expected addition without u64 overflow, got {} + {}", lhs, rhs), path,
                field_offset, field);
        }
        return lhs + rhs;
    }

    lfs::Result<std::uint64_t>
    checked_multiply(const std::uint64_t lhs, const std::uint64_t rhs,
                     const std::filesystem::path& path, const std::uint64_t field_offset,
                     const std::string_view field) {
        if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
            return project_error(
                lfs::ErrorCode::BoundsViolation, "The project contains an invalid byte count.",
                std::format("expected multiplication without u64 overflow, got {} * {}", lhs, rhs),
                path, field_offset, field);
        }
        return lhs * rhs;
    }

#ifdef _WIN32

    NativeFile::NativeFile(std::filesystem::path path, const HANDLE handle)
        : handle_(handle),
          path_(std::move(path)) {}

#else

    NativeFile::NativeFile(std::filesystem::path path, const int fd)
        : fd_(fd),
          path_(std::move(path)) {}

#endif

    NativeFile::~NativeFile() {
#ifdef _WIN32
        if (handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(handle_);
        }
#else
        if (fd_ >= 0) {
            ::close(fd_);
        }
#endif
    }

    lfs::Result<std::shared_ptr<NativeFile>>
    NativeFile::open_read(const std::filesystem::path& path) {
#ifdef _WIN32
        const HANDLE handle = CreateFileW(
            path.wstring().c_str(), GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
        if (handle == INVALID_HANDLE_VALUE) {
            const DWORD error = GetLastError();
            return project_error(native_error_code(static_cast<int>(error), false),
                                 "The project could not be opened.",
                                 std::format("CreateFileW for reading failed with Windows error {}",
                                             error),
                                 path, std::nullopt, {}, static_cast<std::int64_t>(error), "Win32");
        }
        return std::shared_ptr<NativeFile>(new NativeFile(path, handle));
#else
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            const int error = errno;
            return project_error(native_error_code(error, false),
                                 "The project could not be opened.",
                                 std::format("open for reading failed: {}", std::strerror(error)),
                                 path, std::nullopt, {}, error, std::strerror(error));
        }
        return std::shared_ptr<NativeFile>(new NativeFile(path, fd));
#endif
    }

    lfs::Result<std::shared_ptr<NativeFile>>
    NativeFile::open_read_write(const std::filesystem::path& path) {
#ifdef _WIN32
        const HANDLE handle = CreateFileW(
            path.wstring().c_str(), GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
        if (handle == INVALID_HANDLE_VALUE) {
            const DWORD error = GetLastError();
            return project_error(native_error_code(static_cast<int>(error), true),
                                 "The project could not be opened for writing.",
                                 std::format("CreateFileW for update failed with Windows error {}",
                                             error),
                                 path, std::nullopt, {}, static_cast<std::int64_t>(error), "Win32");
        }
        return std::shared_ptr<NativeFile>(new NativeFile(path, handle));
#else
        const int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
        if (fd < 0) {
            const int error = errno;
            return project_error(native_error_code(error, true),
                                 "The project could not be opened for writing.",
                                 std::format("open for update failed: {}", std::strerror(error)),
                                 path, std::nullopt, {}, error, std::strerror(error));
        }
        return std::shared_ptr<NativeFile>(new NativeFile(path, fd));
#endif
    }

    lfs::Result<std::shared_ptr<NativeFile>>
    NativeFile::create_new(const std::filesystem::path& path) {
#ifdef _WIN32
        const HANDLE handle = CreateFileW(
            path.wstring().c_str(), GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, CREATE_NEW,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
        if (handle == INVALID_HANDLE_VALUE) {
            const DWORD error = GetLastError();
            return project_error(native_error_code(static_cast<int>(error), true),
                                 "The project temporary file could not be created.",
                                 std::format("CreateFileW CREATE_NEW failed with Windows error {}",
                                             error),
                                 path, std::nullopt, {}, static_cast<std::int64_t>(error), "Win32");
        }
        return std::shared_ptr<NativeFile>(new NativeFile(path, handle));
#else
        const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd < 0) {
            const int error = errno;
            return project_error(native_error_code(error, true),
                                 "The project temporary file could not be created.",
                                 std::format("exclusive create failed: {}", std::strerror(error)),
                                 path, std::nullopt, {}, error, std::strerror(error));
        }
        return std::shared_ptr<NativeFile>(new NativeFile(path, fd));
#endif
    }

    lfs::Result<std::uint64_t> NativeFile::size() const {
#ifdef _WIN32
        LARGE_INTEGER size{};
        if (!GetFileSizeEx(handle_, &size) || size.QuadPart < 0) {
            const DWORD error = GetLastError();
            return project_error(native_error_code(static_cast<int>(error), false),
                                 "The project size could not be read.",
                                 std::format("GetFileSizeEx failed with Windows error {}", error),
                                 path_, std::nullopt, {}, static_cast<std::int64_t>(error),
                                 "Win32");
        }
        return static_cast<std::uint64_t>(size.QuadPart);
#else
        struct stat status {};
        if (::fstat(fd_, &status) != 0 || status.st_size < 0) {
            const int error = errno;
            return project_error(native_error_code(error, false),
                                 "The project size could not be read.",
                                 std::format("fstat failed: {}", std::strerror(error)), path_,
                                 std::nullopt, {}, error, std::strerror(error));
        }
        return static_cast<std::uint64_t>(status.st_size);
#endif
    }

    lfs::Result<void>
    NativeFile::read_exact(const std::uint64_t offset,
                           const std::span<std::byte> destination) const {
        std::size_t completed = 0;
        while (completed < destination.size()) {
            const std::size_t remaining = destination.size() - completed;
#ifdef _WIN32
            const DWORD request = static_cast<DWORD>(
                std::min<std::size_t>(remaining, std::numeric_limits<DWORD>::max()));
            OVERLAPPED operation = operation_at(offset + completed);
            const BOOL immediate = ReadFile(handle_, destination.data() + completed, request, nullptr,
                                            &operation);
            auto result = wait_for_overlapped(handle_, operation, immediate, path_, "ReadFile", false);
            if (!result) {
                return status_failure(std::move(result).error());
            }
            const DWORD transferred = *result;
            if (transferred == 0) {
                return status_failure(project_error(
                    lfs::ErrorCode::DataLoss, "The project is truncated.",
                    std::format("expected {} more bytes, got EOF", remaining), path_,
                    offset + completed, "positional_read"));
            }
            completed += transferred;
#else
            const ssize_t count =
                ::pread(fd_, destination.data() + completed, remaining,
                        static_cast<off_t>(offset + completed));
            if (count < 0) {
                if (errno == EINTR) {
                    continue;
                }
                const int error = errno;
                return status_failure(project_error(
                    native_error_code(error, false), "The project could not be read.",
                    std::format("pread failed: {}", std::strerror(error)), path_,
                    offset + completed, "positional_read", error, std::strerror(error)));
            }
            if (count == 0) {
                return status_failure(project_error(
                    lfs::ErrorCode::DataLoss, "The project is truncated.",
                    std::format("expected {} more bytes, got EOF", remaining), path_,
                    offset + completed, "positional_read"));
            }
            completed += static_cast<std::size_t>(count);
#endif
        }
        return {};
    }

    lfs::Result<void>
    NativeFile::write_exact(const std::uint64_t offset,
                            const std::span<const std::byte> source) {
        std::size_t completed = 0;
        while (completed < source.size()) {
            const std::size_t remaining = source.size() - completed;
#ifdef _WIN32
            const DWORD request = static_cast<DWORD>(
                std::min<std::size_t>(remaining, std::numeric_limits<DWORD>::max()));
            OVERLAPPED operation = operation_at(offset + completed);
            const BOOL immediate =
                WriteFile(handle_, source.data() + completed, request, nullptr, &operation);
            auto result = wait_for_overlapped(handle_, operation, immediate, path_, "WriteFile", true);
            if (!result) {
                return status_failure(std::move(result).error());
            }
            const DWORD transferred = *result;
            if (transferred == 0) {
                return status_failure(project_error(
                    lfs::ErrorCode::Internal, "The project could not be written.",
                    "WriteFile completed without writing bytes", path_, offset + completed,
                    "positional_write"));
            }
            completed += transferred;
#else
            const ssize_t count =
                ::pwrite(fd_, source.data() + completed, remaining,
                         static_cast<off_t>(offset + completed));
            if (count < 0) {
                if (errno == EINTR) {
                    continue;
                }
                const int error = errno;
                return status_failure(project_error(
                    native_error_code(error, true), "The project could not be written.",
                    std::format("pwrite failed: {}", std::strerror(error)), path_,
                    offset + completed, "positional_write", error, std::strerror(error)));
            }
            if (count == 0) {
                return status_failure(project_error(
                    lfs::ErrorCode::Internal, "The project could not be written.",
                    "pwrite completed without writing bytes", path_, offset + completed,
                    "positional_write"));
            }
            completed += static_cast<std::size_t>(count);
#endif
        }
        return {};
    }

    lfs::Result<void>
    NativeFile::write_single(const std::uint64_t offset,
                             const std::span<const std::byte> source) {
        if (source.empty()) {
            return {};
        }
#ifdef _WIN32
        if (source.size() > std::numeric_limits<DWORD>::max()) {
            return status_failure(project_error(
                lfs::ErrorCode::BoundsViolation, "The project head could not be written.",
                "single-write request exceeds DWORD", path_, offset, "head_write"));
        }
        OVERLAPPED operation = operation_at(offset);
        const BOOL immediate = WriteFile(handle_, source.data(), static_cast<DWORD>(source.size()),
                                         nullptr, &operation);
        auto result = wait_for_overlapped(handle_, operation, immediate, path_, "WriteFile(head)",
                                          true);
        if (!result) {
            return status_failure(std::move(result).error());
        }
        const std::size_t transferred = *result;
#else
        ssize_t count = -1;
        do {
            count = ::pwrite(fd_, source.data(), source.size(), static_cast<off_t>(offset));
        } while (count < 0 && errno == EINTR);
        if (count < 0) {
            const int error = errno;
            return status_failure(project_error(
                native_error_code(error, true), "The project head could not be written.",
                std::format("single pwrite failed: {}", std::strerror(error)), path_, offset,
                "head_write", error, std::strerror(error)));
        }
        const std::size_t transferred = static_cast<std::size_t>(count);
#endif
        if (transferred != source.size()) {
            return status_failure(project_error(
                lfs::ErrorCode::DataLoss, "The project head write was incomplete.",
                std::format("expected one {}-byte write, got {}", source.size(), transferred), path_,
                offset, "head_write"));
        }
        return {};
    }

    lfs::Result<void> NativeFile::truncate(const std::uint64_t size_value) {
#ifdef _WIN32
        LARGE_INTEGER offset{};
        offset.QuadPart = static_cast<LONGLONG>(size_value);
        if (!SetFilePointerEx(handle_, offset, nullptr, FILE_BEGIN) || !SetEndOfFile(handle_)) {
            const DWORD error = GetLastError();
            return status_failure(project_error(
                native_error_code(static_cast<int>(error), true),
                "The project orphan tail could not be reclaimed.",
                std::format("truncate failed with Windows error {}", error), path_, size_value,
                "truncate", static_cast<std::int64_t>(error), "Win32"));
        }
#else
        if (size_value > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
            ::ftruncate(fd_, static_cast<off_t>(size_value)) != 0) {
            const int error = errno;
            return status_failure(project_error(
                native_error_code(error, true), "The project could not be resized.",
                std::format("ftruncate failed: {}", std::strerror(error)), path_, size_value,
                "truncate", error, std::strerror(error)));
        }
#endif
        return {};
    }

    lfs::Result<void> NativeFile::sync_data() {
#ifdef _WIN32
        return sync_all();
#else
        if (::fdatasync(fd_) != 0) {
            const int error = errno;
            return status_failure(project_error(
                native_error_code(error, true), "The project data could not be made durable.",
                std::format("fdatasync failed: {}", std::strerror(error)), path_, std::nullopt,
                "fdatasync", error, std::strerror(error)));
        }
        return {};
#endif
    }

    lfs::Result<void> NativeFile::sync_all() {
#ifdef _WIN32
        if (!FlushFileBuffers(handle_)) {
            const DWORD error = GetLastError();
            return status_failure(project_error(
                native_error_code(static_cast<int>(error), true),
                "The project data could not be made durable.",
                std::format("FlushFileBuffers failed with Windows error {}", error), path_,
                std::nullopt, "FlushFileBuffers", static_cast<std::int64_t>(error), "Win32"));
        }
#else
        if (::fsync(fd_) != 0) {
            const int error = errno;
            return status_failure(project_error(
                native_error_code(error, true), "The project data could not be made durable.",
                std::format("fsync failed: {}", std::strerror(error)), path_, std::nullopt, "fsync",
                error, std::strerror(error)));
        }
#endif
        return {};
    }

#ifdef _WIN32

    WriterLock::WriterLock(std::filesystem::path path, const HANDLE handle)
        : handle_(handle),
          path_(std::move(path)) {}

#else

    WriterLock::WriterLock(std::filesystem::path path, const int fd)
        : fd_(fd),
          path_(std::move(path)) {}

#endif

    WriterLock::WriterLock(WriterLock&& other) noexcept
#ifdef _WIN32
        : handle_(std::exchange(other.handle_, INVALID_HANDLE_VALUE)),
          operation_(other.operation_),
#else
        : fd_(std::exchange(other.fd_, -1)),
#endif
          path_(std::move(other.path_)) {
    }

    WriterLock& WriterLock::operator=(WriterLock&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        this->~WriterLock();
        new (this) WriterLock(std::move(other));
        return *this;
    }

    WriterLock::~WriterLock() noexcept {
#ifdef _WIN32
        if (handle_ != INVALID_HANDLE_VALUE) {
            UnlockFileEx(handle_, 0, 1, 0, &operation_);
            CloseHandle(handle_);
            handle_ = INVALID_HANDLE_VALUE;
            // Best-effort: a contender that still has a share-delete handle
            // makes this fail, and that contender's release deletes the name.
            DeleteFileW(path_.wstring().c_str());
        }
#else
        if (fd_ >= 0) {
            // Unlink while still holding flock so a waiter cannot bind a new
            // inode until we drop the lock. acquire() retries on inode mismatch.
            ::unlink(path_.c_str());
            ::flock(fd_, LOCK_UN);
            ::close(fd_);
            fd_ = -1;
        }
#endif
    }

    lfs::Result<WriterLock> WriterLock::acquire(const std::filesystem::path& project_path) {
        auto lock_path = project_path;
        lock_path += ".lock";
        if (auto parent = ensure_parent_directory(lock_path); !parent) {
            return std::move(parent).error();
        }
        constexpr int kAcquireAttempts = 5;
#ifdef _WIN32
        const DWORD share_mode =
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
        DWORD last_error = 0;
        for (int attempt = 0; attempt < kAcquireAttempts; ++attempt) {
            HANDLE handle = CreateFileW(lock_path.wstring().c_str(),
                                        GENERIC_READ | GENERIC_WRITE, share_mode, nullptr,
                                        CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (handle == INVALID_HANDLE_VALUE && GetLastError() == ERROR_FILE_EXISTS) {
                handle = CreateFileW(lock_path.wstring().c_str(), GENERIC_READ | GENERIC_WRITE,
                                     share_mode, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                                     nullptr);
            }
            if (handle == INVALID_HANDLE_VALUE) {
                last_error = GetLastError();
                const bool retryable = last_error == ERROR_ACCESS_DENIED ||
                                       last_error == ERROR_DELETE_PENDING ||
                                       last_error == ERROR_FILE_NOT_FOUND;
                if (retryable && attempt + 1 < kAcquireAttempts) {
                    Sleep(1);
                    continue;
                }
                return project_error(
                    native_error_code(static_cast<int>(last_error), true),
                    "The project writer lock could not be opened.",
                    std::format("CreateFileW lockfile failed with Windows error {}", last_error),
                    lock_path, std::nullopt, "writer_lock",
                    static_cast<std::int64_t>(last_error), "Win32");
            }
            OVERLAPPED operation{};
            const DWORD flags = LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY;
            if (!LockFileEx(handle, flags, 0, 1, 0, &operation)) {
                const DWORD error = GetLastError();
                CloseHandle(handle);
                return project_error(
                    lfs::ErrorCode::Unavailable, "The project is already open for writing.",
                    std::format("LockFileEx denied the held lock with Windows error {}", error),
                    lock_path, std::nullopt, "writer_lock", static_cast<std::int64_t>(error),
                    "Win32");
            }
            WriterLock result(lock_path, handle);
            result.operation_ = operation;
            return result;
        }
        return project_error(native_error_code(static_cast<int>(last_error), true),
                             "The project writer lock could not be opened.",
                             std::format("CreateFileW lockfile failed with Windows error {}",
                                         last_error),
                             lock_path, std::nullopt, "writer_lock",
                             static_cast<std::int64_t>(last_error), "Win32");
#else
        for (int attempt = 0; attempt < kAcquireAttempts; ++attempt) {
            int fd = ::open(lock_path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
            if (fd < 0 && errno == EEXIST) {
                fd = ::open(lock_path.c_str(), O_RDWR | O_CLOEXEC);
            }
            if (fd < 0) {
                const int error = errno;
                if (error == ENOENT && attempt + 1 < kAcquireAttempts) {
                    continue;
                }
                return project_error(
                    native_error_code(error, true),
                    "The project writer lock could not be opened.",
                    std::format("lockfile open failed: {}", std::strerror(error)), lock_path,
                    std::nullopt, "writer_lock", error, std::strerror(error));
            }
            if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
                const int error = errno;
                ::close(fd);
                return project_error(
                    lfs::ErrorCode::Unavailable, "The project is already open for writing.",
                    std::format("flock denied the held lock: {}", std::strerror(error)),
                    lock_path, std::nullopt, "writer_lock", error, std::strerror(error));
            }
            struct stat fd_status {};
            if (::fstat(fd, &fd_status) != 0) {
                return WriterLock(lock_path, fd);
            }
            struct stat path_status {};
            if (::stat(lock_path.c_str(), &path_status) != 0 ||
                fd_status.st_dev != path_status.st_dev ||
                fd_status.st_ino != path_status.st_ino) {
                if (attempt + 1 < kAcquireAttempts) {
                    ::flock(fd, LOCK_UN);
                    ::close(fd);
                    continue;
                }
                return WriterLock(lock_path, fd);
            }
            return WriterLock(lock_path, fd);
        }
        return project_error(lfs::ErrorCode::Unavailable,
                             "The project writer lock could not be opened.",
                             "lockfile open exhausted inode-mismatch retries", lock_path,
                             std::nullopt, "writer_lock");
#endif
    }

    lfs::Result<WriterLock> WriterLock::acquire(const std::filesystem::path& project_path,
                                                const std::chrono::milliseconds wait) {
        auto acquired = acquire(project_path);
        if (acquired || wait <= std::chrono::milliseconds::zero() ||
            acquired.error().code() != lfs::ErrorCode::Unavailable) {
            return acquired;
        }
        const auto deadline = std::chrono::steady_clock::now() + wait;
        while (std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
            acquired = acquire(project_path);
            if (acquired || acquired.error().code() != lfs::ErrorCode::Unavailable) {
                return acquired;
            }
        }
        return acquired;
    }

    std::filesystem::path
    make_sibling_temp_path(const std::filesystem::path& destination, const std::string_view tag) {
        static std::atomic_uint64_t counter{0};
        const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
#ifdef _WIN32
        const auto process_id = GetCurrentProcessId();
#else
        const auto process_id = ::getpid();
#endif
        const std::string suffix =
            std::format(".{}.{}.{}.{}.tmp", tag, ticks, process_id,
                        counter.fetch_add(1, std::memory_order_relaxed));
        if (destination.has_extension()) {
            return destination.parent_path() /
                   (destination.stem().string() + suffix + destination.extension().string());
        }
        return std::filesystem::path(destination.string() + suffix);
    }

    lfs::Result<void> ensure_parent_directory(const std::filesystem::path& path) {
        const auto parent = path.parent_path();
        if (parent.empty()) {
            return {};
        }
        std::error_code error;
        std::filesystem::create_directories(parent, error);
        if (error) {
            return status_failure(project_error(
                lfs::ErrorCode::PermissionDenied, "The project directory could not be created.",
                std::format("create_directories failed: {}", error.message()), parent));
        }
        return {};
    }

    lfs::Result<void>
    preflight_disk_space(const std::filesystem::path& path, const std::uint64_t required_bytes) {
        std::error_code error;
        auto parent = path.parent_path();
        if (parent.empty()) {
            parent = std::filesystem::current_path(error);
        }
        const auto space = std::filesystem::space(parent, error);
        if (error) {
            return status_failure(project_error(
                lfs::ErrorCode::PermissionDenied, "Available project disk space is unknown.",
                std::format("filesystem::space failed: {}", error.message()), parent));
        }
        if (space.available < required_bytes) {
            lfs::SmallFields fields;
            fields.add("path", lfs::core::path_to_utf8(parent));
            fields.add("required_bytes", required_bytes);
            fields.add("available_bytes", static_cast<std::uint64_t>(space.available));
            return status_failure(lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::ResourceExhausted,
                .domain = lfs::ErrorDomain::IO,
                .operation_id = {},
                .user_message = "There is not enough disk space to save the project.",
                .detail = std::format("preflight requires {} bytes, volume reports {} available",
                                      required_bytes, space.available),
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .fields = std::move(fields),
                .native = std::nullopt,
            }));
        }
        return {};
    }

    lfs::Result<void> sync_parent_directory(const std::filesystem::path& path) {
#ifdef _WIN32
        (void)path;
        return {};
#else
        const auto parent =
            path.parent_path().empty() ? std::filesystem::path{"."} : path.parent_path();
        const int fd = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (fd < 0) {
            const int error = errno;
            return status_failure(project_error(
                native_error_code(error, true), "The project directory could not be flushed.",
                std::format("directory open failed: {}", std::strerror(error)), parent,
                std::nullopt, "directory_fsync", error, std::strerror(error)));
        }
        if (::fsync(fd) != 0) {
            const int error = errno;
            ::close(fd);
            return status_failure(project_error(
                native_error_code(error, true), "The project directory could not be flushed.",
                std::format("directory fsync failed: {}", std::strerror(error)), parent,
                std::nullopt, "directory_fsync", error, std::strerror(error)));
        }
        if (::close(fd) != 0) {
            const int error = errno;
            return status_failure(project_error(
                native_error_code(error, true), "The project directory could not be closed.",
                std::format("directory close failed: {}", std::strerror(error)), parent,
                std::nullopt, "directory_close", error, std::strerror(error)));
        }
        return {};
#endif
    }

    lfs::Result<AtomicReplaceState>
    atomic_replace(const std::filesystem::path& replacement,
                   const std::filesystem::path& destination) {
        AtomicReplaceState state;
        std::error_code exists_error;
        const bool destination_exists = std::filesystem::exists(destination, exists_error);
        if (exists_error) {
            return project_error(lfs::ErrorCode::PermissionDenied,
                                 "The existing project could not be inspected.",
                                 std::format("exists failed: {}", exists_error.message()),
                                 destination);
        }

#ifdef _WIN32
        if (destination_exists) {
            const auto backup = make_sibling_temp_path(destination, "replace-backup");
            if (!ReplaceFileW(destination.wstring().c_str(), replacement.wstring().c_str(),
                              backup.wstring().c_str(), REPLACEFILE_WRITE_THROUGH, nullptr,
                              nullptr)) {
                const DWORD error = GetLastError();
                return project_error(
                    native_error_code(static_cast<int>(error), true),
                    "The compacted project could not replace the existing project.",
                    std::format("ReplaceFileW failed with Windows error {}; candidates retained",
                                error),
                    destination, std::nullopt, "atomic_replace",
                    static_cast<std::int64_t>(error), "Win32");
            }
            state.backup_path = backup;
        } else if (!MoveFileExW(replacement.wstring().c_str(), destination.wstring().c_str(),
                                MOVEFILE_WRITE_THROUGH)) {
            const DWORD error = GetLastError();
            return project_error(
                native_error_code(static_cast<int>(error), true),
                "The project could not be published.",
                std::format("MoveFileExW first publication failed with Windows error {}", error),
                destination, std::nullopt, "atomic_replace",
                static_cast<std::int64_t>(error), "Win32");
        }
#else
        if (destination_exists) {
            const auto backup = make_sibling_temp_path(destination, "replace-backup");
            if (::link(destination.c_str(), backup.c_str()) != 0) {
                const int error = errno;
                return project_error(
                    native_error_code(error, true),
                    "A recoverable project replacement backup could not be created.",
                    std::format("link backup failed: {}", std::strerror(error)), backup,
                    std::nullopt, "atomic_replace_backup", error, std::strerror(error));
            }
            state.backup_path = backup;
        }
        if (::rename(replacement.c_str(), destination.c_str()) != 0) {
            const int error = errno;
            if (state.backup_path.has_value()) {
                ::unlink(state.backup_path->c_str());
            }
            return project_error(native_error_code(error, true),
                                 "The project could not be atomically published.",
                                 std::format("rename failed: {}", std::strerror(error)),
                                 destination, std::nullopt, "atomic_replace", error,
                                 std::strerror(error));
        }
        if (auto sync = sync_parent_directory(destination); !sync) {
            return std::move(sync).error();
        }
#endif
        return state;
    }

    lfs::Result<void>
    finish_atomic_replace(const AtomicReplaceState& state,
                          const std::filesystem::path& destination) {
        if (!state.backup_path.has_value()) {
            return {};
        }
        std::error_code error;
        const bool removed = std::filesystem::remove(*state.backup_path, error);
        if (error || !removed) {
            return status_failure(project_error(
                lfs::ErrorCode::Internal,
                "The old project backup could not be removed after validation.",
                error ? std::format("remove failed: {}", error.message())
                      : "backup path was not removed",
                *state.backup_path));
        }
        return sync_parent_directory(destination);
    }

    lfs::Result<void>
    rollback_atomic_replace(const AtomicReplaceState& state,
                            const std::filesystem::path& destination) {
        if (!state.backup_path.has_value()) {
            return status_failure(project_error(
                lfs::ErrorCode::DataLoss,
                "The newly published project failed validation and has no prior backup.",
                "first publication cannot be rolled back automatically", destination));
        }
#ifdef _WIN32
        if (!MoveFileExW(state.backup_path->wstring().c_str(), destination.wstring().c_str(),
                         MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            const DWORD error = GetLastError();
            return status_failure(project_error(
                lfs::ErrorCode::DataLoss,
                "The prior project backup could not be restored.",
                std::format("MoveFileExW rollback failed with Windows error {}", error),
                destination, std::nullopt, "atomic_replace_rollback",
                static_cast<std::int64_t>(error), "Win32"));
        }
#else
        if (::rename(state.backup_path->c_str(), destination.c_str()) != 0) {
            const int error = errno;
            return status_failure(project_error(
                lfs::ErrorCode::DataLoss,
                "The prior project backup could not be restored.",
                std::format("rename rollback failed: {}", std::strerror(error)), destination,
                std::nullopt, "atomic_replace_rollback", error, std::strerror(error)));
        }
        if (auto sync = sync_parent_directory(destination); !sync) {
            return sync;
        }
#endif
        return {};
    }

    std::uint64_t unix_time_ns() noexcept {
        const auto now = std::chrono::system_clock::now().time_since_epoch();
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
    }

} // namespace lfs::io::project::detail
