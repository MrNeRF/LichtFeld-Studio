// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/atomic_file.hpp"
#include "path_utils.hpp"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <exception>
#include <format>
#include <system_error>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <sys/stat.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/syscall.h>
#endif
#endif

namespace lfs::core {
    namespace {
        std::string nativeMessage(const std::error_code& error) {
#ifdef _WIN32
            // The narrow Windows category message uses the active code page,
            // but structured errors cross JSON and Python as UTF-8.
            wchar_t message[1024]{};
            if (error.category() == std::system_category()) {
                const auto length = FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                                   nullptr, error.value(), 0, message, 1024, nullptr);
                if (length)
                    return wstring_to_utf8(std::wstring(message, length));
            } else if (error.category() == std::generic_category() &&
                       _wcserror_s(message, 1024, error.value()) == 0) {
                return wstring_to_utf8(message);
            }
            return std::format("{} error {}", error.category().name(), error.value());
#else
            return error.message();
#endif
        }
        Error ioError(const std::filesystem::path& path, std::string_view operation,
                      int native, const std::error_category& category = std::generic_category()) {
            const std::error_code error(native, category);
            const auto code = error == std::errc::file_exists                                                             ? ErrorCode::AlreadyExists
                              : error == std::errc::permission_denied                                                     ? ErrorCode::PermissionDenied
                              : error == std::errc::operation_not_supported || error == std::errc::function_not_supported ? ErrorCode::Unsupported
                                                                                                                          : ErrorCode::Unavailable;
            return make_error({.code = code, .domain = ErrorDomain::IO, .detail = std::format("Atomic file {} failed (path='{}', native={}): {}", operation, path_to_utf8(path), native, nativeMessage(error)), .detection = LFS_SOURCE_SITE_CURRENT(), .native = NativeError{ErrorDomain::IO, native, category.name()}});
        }
        struct Temporary {
            std::filesystem::path path;
            FILE* stream = nullptr;
            bool owned = false;
            ~Temporary() {
                if (stream)
                    std::fclose(stream);
                if (owned) {
                    std::error_code ignored;
                    std::filesystem::remove(path, ignored);
                }
            }
        };
#ifndef _WIN32
        Status commitReserved(const std::filesystem::path& source, const std::filesystem::path& target) {
            // Some filesystems (notably macOS exFAT) support neither exclusive
            // rename nor hard links. O_EXCL is the atomic no-replace claim;
            // publication then replaces only the placeholder we just created.
            const int fd = ::open(target.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
            if (fd < 0)
                return Status::failure(ioError(target, "reserve destination", errno));
            struct stat owned{};
            if (::fstat(fd, &owned) != 0) {
                const int native = errno;
                ::unlink(target.c_str());
                ::close(fd);
                return Status::failure(ioError(target, "inspect reservation", native));
            }
            const int result = ::rename(source.c_str(), target.c_str());
            const int native = errno;
            if (result != 0) {
                struct stat current{};
                if (::lstat(target.c_str(), &current) == 0 &&
                    current.st_dev == owned.st_dev && current.st_ino == owned.st_ino)
                    ::unlink(target.c_str());
            }
            ::close(fd);
            return result == 0 ? Status{} : Status::failure(ioError(target, "publish reserved destination", native));
        }
#endif
        Status commit(Temporary& temporary, const std::filesystem::path& target, bool overwrite) {
#ifdef _WIN32
            if (!MoveFileExW(temporary.path.c_str(), target.c_str(),
                             MOVEFILE_WRITE_THROUGH | (overwrite ? MOVEFILE_REPLACE_EXISTING : 0)))
                return Status::failure(ioError(target, "commit", static_cast<int>(GetLastError()), std::system_category()));
            temporary.owned = false;
#else
            int result;
            if (overwrite) {
                result = ::rename(temporary.path.c_str(), target.c_str());
            } else {
#ifdef __APPLE__
                result = ::renamex_np(temporary.path.c_str(), target.c_str(), RENAME_EXCL);
#elif defined(__linux__) && defined(SYS_renameat2)
                result = static_cast<int>(::syscall(SYS_renameat2, AT_FDCWD, temporary.path.c_str(), AT_FDCWD, target.c_str(), 1u /* RENAME_NOREPLACE */));
#else
                result = -1;
                errno = ENOSYS;
#endif
                if (result != 0 && (errno == ENOSYS || errno == ENOTSUP || errno == EINVAL)) {
                    // Safe fallback for older kernels/filesystems. Never use
                    // exists()+rename(): another writer could create the target.
                    if (::link(temporary.path.c_str(), target.c_str()) == 0)
                        return {};
                    if (errno == ENOSYS || errno == ENOTSUP || errno == EINVAL) {
                        if (auto status = commitReserved(temporary.path, target); !status)
                            return status;
                        temporary.owned = false;
                        return {};
                    }
                }
            }
            if (result != 0)
                return Status::failure(ioError(target, "commit", errno));
            temporary.owned = false;
#endif
            return {};
        }
    } // namespace

    Status writeFileAtomically(const std::filesystem::path& target, const AtomicFileWriter& writer,
                               const AtomicFileOptions& options) {
        try {
            if (!writer || target.empty() || target.filename().empty() ||
                target.native().find(std::filesystem::path::value_type{}) != std::filesystem::path::string_type::npos)
                return Status::failure(make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = std::format("Atomic write requires a file path and writer (path_bytes={}, writer_present={})", target.native().size(), bool(writer)), .detection = LFS_SOURCE_SITE_CURRENT()}));
            const auto cancelled = [&]() -> Status {
                if (options.cancelled && options.cancelled())
                    return Status::failure(make_error({.code = ErrorCode::Cancelled, .domain = ErrorDomain::IO, .detail = "Atomic write cancelled: " + path_to_utf8(target), .detection = LFS_SOURCE_SITE_CURRENT()}));
                return {};
            };
            if (auto status = cancelled(); !status)
                return status;
            auto directory = target.parent_path();
            if (directory.empty())
                directory = ".";
            if (options.create_directories) {
                std::error_code error;
                std::filesystem::create_directories(directory, error);
                if (error)
                    return Status::failure(ioError(directory, "create directory", error.value(), error.category()));
            }
            Temporary temporary;
            static std::atomic<uint64_t> sequence{0};
            for (int attempt = 0; attempt < 32; ++attempt) {
                temporary.path = target;
                temporary.path += std::format(".tmp-{}-{}", std::chrono::steady_clock::now().time_since_epoch().count(), sequence.fetch_add(1, std::memory_order_relaxed));
#ifdef _WIN32
                int fd = -1;
                const auto native = _wsopen_s(&fd, temporary.path.c_str(), _O_RDWR | _O_CREAT | _O_EXCL | _O_BINARY, _SH_DENYRW, _S_IREAD | _S_IWRITE);
                if (native)
                    errno = native;
#else
                const int fd = ::open(temporary.path.c_str(), O_RDWR | O_CREAT | O_EXCL, 0666);
#endif
                if (fd < 0) {
                    if (errno == EEXIST)
                        continue;
                    return Status::failure(ioError(temporary.path, "create temporary", errno));
                }
                temporary.owned = true;
#ifdef _WIN32
                temporary.stream = _fdopen(fd, "w+b");
                if (!temporary.stream)
                    _close(fd);
#else
                temporary.stream = fdopen(fd, "w+b");
                if (!temporary.stream)
                    ::close(fd);
#endif
                if (!temporary.stream)
                    return Status::failure(ioError(temporary.path, "open stream", errno));
                break;
            }
            if (!temporary.stream)
                return Status::failure(ioError(target, "reserve temporary", EEXIST));
            if (auto status = writer(temporary.stream); !status)
                return status;
            if (std::fflush(temporary.stream) != 0)
                return Status::failure(ioError(target, "flush", errno));
            if (options.durable) {
#ifdef _WIN32
                const int result = _commit(_fileno(temporary.stream));
#else
                const int result = ::fsync(fileno(temporary.stream));
#endif
                if (result != 0)
                    return Status::failure(ioError(target, "synchronize", errno));
            }
            const int close_result = std::fclose(temporary.stream);
            temporary.stream = nullptr;
            if (close_result != 0)
                return Status::failure(ioError(target, "close", errno));
            if (auto status = cancelled(); !status)
                return status;
            if (auto status = commit(temporary, target, options.overwrite); !status)
                return status;
#ifndef _WIN32
            if (options.durable) {
                const int fd = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY);
                if (fd < 0)
                    return Status::failure(ioError(directory, "open for synchronization", errno));
                const int result = ::fsync(fd);
                const int native = errno;
                ::close(fd);
                if (result != 0)
                    return Status::failure(ioError(directory, "synchronize", native));
            }
#endif
            return {};
        } catch (const Exception& error) {
            return Status::failure(error.error());
        } catch (const std::exception& error) {
            // LFS-CENSUS-OK(empty-catch): callbacks/filesystem failures retain their message in a structured error; RAII cleans only our temporary.
            return Status::failure(make_error({.code = ErrorCode::Internal, .domain = ErrorDomain::IO, .detail = std::format("Atomic write failed (path='{}', exception={})", path_to_utf8(target), error.what()), .detection = LFS_SOURCE_SITE_CURRENT()}));
        }
    }
    Status writeTextFileAtomically(const std::filesystem::path& target, std::string_view contents,
                                   const AtomicFileOptions& options) {
        return writeFileAtomically(target, [&](FILE* stream) -> Status {
            if (std::fwrite(contents.data(), 1, contents.size(), stream) != contents.size())
                return Status::failure(ioError(target, "write bytes", errno));
            return {}; }, options);
    }
} // namespace lfs::core
