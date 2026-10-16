/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/filename_key.hpp"
#include "core/native_fs.hpp"
#include "core/path_utils.hpp"
#include <cstdio>
#include <format>
#include <memory>
#include <mutex>
#include <spdlog/sinks/base_sink.h>
#include <xxhash.h>

namespace lfs::core {
    // spdlog's installed narrow-filename ABI cannot open Unicode Windows paths.
    // Keep formatting/locking in spdlog and put only file I/O through our adapter.
    class NativeRotatingFileSink final : public spdlog::sinks::base_sink<std::mutex> {
    public:
        NativeRotatingFileSink(native_fs::path path, size_t max_bytes, size_t max_files)
            : path_(std::move(path)), max_bytes_(max_bytes), max_files_(max_files) {
            if (!max_bytes_ || max_files_ > 200000)
                spdlog::throw_spdlog_ex("Invalid native rotating log limits");
            open(false);
            std::error_code error;
            size_ = native_fs::file_size(path_, error);
            if (error) {
                std::fclose(file_);
                file_ = nullptr;
                spdlog::throw_spdlog_ex("Cannot stat log: " + path_to_utf8(path_));
            }
        }
        ~NativeRotatingFileSink() override {
            if (file_)
                std::fclose(file_);
        }

        static native_fs::path rotated_path(const native_fs::path& path, size_t index) {
            if (!index)
                return path;
            auto name = path.stem();
            name += std::format(".{}", index);
            name += path.extension();
            if (name.native().size() > 255) {
#ifdef _WIN32
                const auto original = path_to_utf8(lfs::core::detail::artifact_filename_key(path));
#else
                const auto original = lfs::core::detail::artifact_filename_key(path).native();
#endif
                const auto key = XXH3_128bits(original.data(), original.size());
                name = utf8_to_path(std::format(".lfs-log-{:016x}{:016x}.{}.log", key.high64, key.low64, index));
            }
            return path.parent_path() / name;
        }

    protected:
        void sink_it_(const spdlog::details::log_msg& message) override {
            spdlog::memory_buf_t formatted;
            formatter_->format(message, formatted);
            std::string rotation_error;
            if (!rotation_failed_ && size_ && (formatted.size() > max_bytes_ || size_ > max_bytes_ - formatted.size()))
                rotation_error = rotate();
            if (!file_ || std::fwrite(formatted.data(), 1, formatted.size(), file_) != formatted.size())
                spdlog::throw_spdlog_ex("Cannot write log: " + path_to_utf8(path_));
            size_ += formatted.size();
            // Report once, after preserving the triggering line. Keep appending
            // until restart instead of repeatedly shifting/deleting history.
            if (!rotation_error.empty())
                spdlog::throw_spdlog_ex(rotation_error);
        }
        void flush_() override {
            if (!file_ || std::fflush(file_) != 0)
                spdlog::throw_spdlog_ex("Cannot flush log: " + path_to_utf8(path_));
        }

    private:
        void open(bool truncate) {
            if (!path_.parent_path().empty())
                native_fs::create_directories(path_.parent_path());
            file_ = open_file(path_, truncate ? "wb" : "ab");
            if (!file_)
                spdlog::throw_spdlog_ex("Cannot open log: " + path_to_utf8(path_));
        }
        std::string rotate() {
            flush_();
            std::fclose(file_);
            file_ = nullptr;
            try {
#ifdef _WIN32
                // Fail before touching history when another process holds the
                // active log without delete sharing. Retain DELETE access while
                // shifting history so a new incompatible opener cannot race us.
                const auto handle = CreateFileW(native_fs::native_path(path_).c_str(), DELETE,
                                                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                                nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
                if (handle == INVALID_HANDLE_VALUE)
                    throw std::filesystem::filesystem_error("Cannot rotate active log", path_,
                                                            std::error_code(GetLastError(), std::system_category()));
                std::unique_ptr<void, decltype(&CloseHandle)> rotation_guard(handle, &CloseHandle);
#endif
                for (size_t index = max_files_; index > 0; --index) {
                    const auto source = rotated_path(path_, index - 1);
                    if (!native_fs::exists(source))
                        continue;
                    const auto destination = rotated_path(path_, index);
                    native_fs::remove(destination);
                    native_fs::rename(source, destination);
                }
#ifdef _WIN32
                rotation_guard.reset();
#endif
                open(true);
                size_ = 0;
                return {};
            } catch (const std::exception& failure) {
                // LFS-CENSUS-OK(empty-catch): Return the diagnostic to sink_it_,
                // which reports it through spdlog after writing the pending line.
                // Preserve the current log on failure and leave the sink usable.
                rotation_failed_ = true;
                if (!file_)
                    open(false);
                std::error_code error;
                size_ = native_fs::file_size(path_, error);
                if (error)
                    spdlog::throw_spdlog_ex("Cannot stat log after rotation failure: " + path_to_utf8(path_));
                return "Log rotation failed; appending without rotation until restart: " + std::string(failure.what());
            }
        }
        native_fs::path path_;
        size_t max_bytes_, max_files_, size_ = 0;
        std::FILE* file_ = nullptr;
        bool rotation_failed_ = false;
    };
} // namespace lfs::core
