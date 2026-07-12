#pragma once

#include "core/path_utils.hpp"
#include "io/error.hpp"
#include "io/exporter.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <format>
#include <functional>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace lfs::io {

    enum class AtomicOutputTempName {
        AppendSuffix,
        PreserveExtension
    };

    enum class AtomicOutputDurability {
        Atomic,
        Durable
    };

    enum class AtomicOutputCommitStage {
        FileSynced,
        DestinationReplaced,
        DirectorySynced
    };

    using AtomicOutputCommitObserver = std::function<void(AtomicOutputCommitStage)>;

    inline std::filesystem::path make_atomic_temp_output_path(
        const std::filesystem::path& output_path,
        AtomicOutputTempName name_style = AtomicOutputTempName::AppendSuffix) {
        static std::atomic_uint64_t counter{0};

        const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
#ifdef _WIN32
        const auto process_id = GetCurrentProcessId();
#else
        const auto process_id = ::getpid();
#endif
        const auto unique_suffix =
            std::format(".{}.{}.{}.tmp", ticks, process_id, counter.fetch_add(1, std::memory_order_relaxed));

        if (name_style == AtomicOutputTempName::PreserveExtension && output_path.has_extension()) {
            const auto temp_name = output_path.stem().string() + unique_suffix + output_path.extension().string();
            return output_path.parent_path() / temp_name;
        }

        return output_path.string() + unique_suffix;
    }

    inline Result<void> ensure_output_parent_directory(const std::filesystem::path& output_path) {
        if (output_path.parent_path().empty()) {
            return {};
        }

        std::error_code ec;
        std::filesystem::create_directories(output_path.parent_path(), ec);
        if (ec) {
            return std::unexpected(Error{
                ErrorCode::WRITE_FAILURE,
                std::format("Failed to create output directory '{}': {}", output_path.parent_path().string(), ec.message())});
        }

        return {};
    }

    namespace detail {
        inline Result<void> sync_file_for_durable_replace(const std::filesystem::path& path) {
#ifdef _WIN32
            const HANDLE handle = CreateFileW(path.wstring().c_str(),
                                              GENERIC_READ | GENERIC_WRITE,
                                              FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                              nullptr,
                                              OPEN_EXISTING,
                                              FILE_ATTRIBUTE_NORMAL,
                                              nullptr);
            if (handle == INVALID_HANDLE_VALUE) {
                return std::unexpected(Error{
                    ErrorCode::WRITE_FAILURE,
                    std::format("Failed to open temporary output '{}' for durable flush: Windows error {}",
                                path.string(), GetLastError())});
            }
            if (!FlushFileBuffers(handle)) {
                const auto error = GetLastError();
                CloseHandle(handle);
                return std::unexpected(Error{
                    ErrorCode::WRITE_FAILURE,
                    std::format("Failed to flush temporary output '{}': Windows error {}",
                                path.string(), error)});
            }
            CloseHandle(handle);
#else
            const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
            if (fd < 0) {
                return std::unexpected(Error{
                    ErrorCode::WRITE_FAILURE,
                    std::format("Failed to open temporary output '{}' for durable flush: {}",
                                path.string(), std::strerror(errno))});
            }
            if (::fsync(fd) != 0) {
                const int error = errno;
                ::close(fd);
                return std::unexpected(Error{
                    ErrorCode::WRITE_FAILURE,
                    std::format("Failed to flush temporary output '{}': {}",
                                path.string(), std::strerror(error))});
            }
            if (::close(fd) != 0) {
                return std::unexpected(Error{
                    ErrorCode::WRITE_FAILURE,
                    std::format("Failed to close temporary output '{}' after durable flush: {}",
                                path.string(), std::strerror(errno))});
            }
#endif
            return {};
        }

        inline Result<void> sync_parent_directory(const std::filesystem::path& output_path) {
#ifdef _WIN32
            // MOVEFILE_WRITE_THROUGH below durably publishes the replacement on Windows.
            (void)output_path;
            return {};
#else
            const auto parent = output_path.parent_path().empty() ? std::filesystem::path{"."}
                                                                  : output_path.parent_path();
            const int fd = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
            if (fd < 0) {
                return std::unexpected(Error{
                    ErrorCode::WRITE_FAILURE,
                    std::format("Failed to open output directory '{}' for durable flush: {}",
                                parent.string(), std::strerror(errno))});
            }
            if (::fsync(fd) != 0) {
                const int error = errno;
                ::close(fd);
                return std::unexpected(Error{
                    ErrorCode::WRITE_FAILURE,
                    std::format("Failed to flush output directory '{}': {}",
                                parent.string(), std::strerror(error))});
            }
            if (::close(fd) != 0) {
                return std::unexpected(Error{
                    ErrorCode::WRITE_FAILURE,
                    std::format("Failed to close output directory '{}' after durable flush: {}",
                                parent.string(), std::strerror(errno))});
            }
            return {};
#endif
        }
    } // namespace detail

    inline Result<void> replace_atomic_output_file(
        const std::filesystem::path& temp_path,
        const std::filesystem::path& output_path,
        const AtomicOutputDurability durability = AtomicOutputDurability::Atomic,
        const AtomicOutputCommitObserver& observer = {}) {
        if (durability == AtomicOutputDurability::Durable) {
            if (auto result = detail::sync_file_for_durable_replace(temp_path); !result) {
                return result;
            }
            if (observer) {
                observer(AtomicOutputCommitStage::FileSynced);
            }
        }

#ifdef _WIN32
        const auto temp_w = temp_path.wstring();
        const auto output_w = output_path.wstring();
        // Preserve the existing Windows write-through behavior for all atomic
        // outputs. Durable mode additionally flushes the file handle first.
        const DWORD flags = MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH;
        if (!MoveFileExW(temp_w.c_str(), output_w.c_str(), flags)) {
            return std::unexpected(Error{
                ErrorCode::WRITE_FAILURE,
                std::format("Failed to replace '{}' with temporary export '{}': Windows error {}",
                            output_path.string(), temp_path.string(), GetLastError())});
        }
#else
        std::error_code ec;
        std::filesystem::rename(temp_path, output_path, ec);
        if (ec) {
            return std::unexpected(Error{
                ErrorCode::WRITE_FAILURE,
                std::format("Failed to replace '{}' with temporary export '{}': {}",
                            output_path.string(), temp_path.string(), ec.message())});
        }
#endif

        if (observer) {
            observer(AtomicOutputCommitStage::DestinationReplaced);
        }
        if (durability == AtomicOutputDurability::Durable) {
            if (auto result = detail::sync_parent_directory(output_path); !result) {
                return result;
            }
            if (observer) {
                observer(AtomicOutputCommitStage::DirectorySynced);
            }
        }

        return {};
    }

    class ScopedAtomicOutputFile {
    public:
        explicit ScopedAtomicOutputFile(
            std::filesystem::path output_path,
            AtomicOutputTempName name_style = AtomicOutputTempName::AppendSuffix,
            AtomicOutputDurability durability = AtomicOutputDurability::Atomic)
            : output_path_(std::move(output_path)),
              temp_path_(make_atomic_temp_output_path(output_path_, name_style)),
              durability_(durability) {}

        ScopedAtomicOutputFile(const ScopedAtomicOutputFile&) = delete;
        ScopedAtomicOutputFile& operator=(const ScopedAtomicOutputFile&) = delete;

        ScopedAtomicOutputFile(ScopedAtomicOutputFile&&) = delete;
        ScopedAtomicOutputFile& operator=(ScopedAtomicOutputFile&&) = delete;

        ~ScopedAtomicOutputFile() {
            if (!committed_) {
                std::error_code ec;
                std::filesystem::remove(temp_path_, ec);
            }
        }

        const std::filesystem::path& output_path() const { return output_path_; }
        const std::filesystem::path& temp_path() const { return temp_path_; }

        Result<void> commit(const AtomicOutputCommitObserver& observer = {}) {
            auto result = replace_atomic_output_file(temp_path_, output_path_, durability_, observer);
            if (!result) {
                return result;
            }

            committed_ = true;
            return {};
        }

    private:
        std::filesystem::path output_path_;
        std::filesystem::path temp_path_;
        AtomicOutputDurability durability_;
        bool committed_ = false;
    };

    inline bool report_export_progress(const ExportProgressCallback& callback, float progress, const std::string& stage) {
        if (!callback) {
            return true;
        }
        return callback(progress, stage);
    }

    inline ExportProgressCallback scale_export_progress(
        ExportProgressCallback callback,
        float start,
        float end) {
        if (!callback) {
            return {};
        }

        return [callback = std::move(callback), start, end](float progress, const std::string& stage) {
            const float scaled = start + (end - start) * progress;
            return callback(scaled, stage);
        };
    }

} // namespace lfs::io
