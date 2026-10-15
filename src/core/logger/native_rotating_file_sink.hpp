/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/filename_key.hpp"
#include "core/filesystem.hpp"
#include "core/path_utils.hpp"
#include <cstdio>
#include <format>
#include <mutex>
#include <spdlog/sinks/base_sink.h>
#include <xxhash.h>

namespace lfs::core {
    // spdlog's installed narrow-filename ABI cannot open Unicode Windows paths.
    // Keep formatting/locking in spdlog and put only file I/O through our adapter.
    class NativeRotatingFileSink final : public spdlog::sinks::base_sink<std::mutex> {
    public:
        NativeRotatingFileSink(fs::path path, size_t max_bytes, size_t max_files)
            : path_(std::move(path)), max_bytes_(max_bytes), max_files_(max_files) {
            if (!max_bytes_ || max_files_ > 200000)
                spdlog::throw_spdlog_ex("Invalid native rotating log limits");
            open(false);
            std::error_code error;
            size_ = fs::file_size(path_, error);
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

        static fs::path rotated_path(const fs::path& path, size_t index) {
            if (!index)
                return path;
            auto name = path.stem();
            name += std::format(".{}", index);
            name += path.extension();
            if (name.native().size() > 255) {
                const auto original = path_to_utf8(lfs::core::detail::artifact_filename_key(path));
                const auto key = XXH3_128bits(original.data(), original.size());
                name = utf8_to_path(std::format(".lfs-log-{:016x}{:016x}.{}.log", key.high64, key.low64, index));
            }
            return path.parent_path() / name;
        }

    protected:
        void sink_it_(const spdlog::details::log_msg& message) override {
            spdlog::memory_buf_t formatted;
            formatter_->format(message, formatted);
            if (size_ && (formatted.size() > max_bytes_ || size_ > max_bytes_ - formatted.size()))
                rotate();
            if (!file_ || std::fwrite(formatted.data(), 1, formatted.size(), file_) != formatted.size())
                spdlog::throw_spdlog_ex("Cannot write log: " + path_to_utf8(path_));
            size_ += formatted.size();
        }
        void flush_() override {
            if (!file_ || std::fflush(file_) != 0)
                spdlog::throw_spdlog_ex("Cannot flush log: " + path_to_utf8(path_));
        }

    private:
        void open(bool truncate) {
            if (!path_.parent_path().empty())
                fs::create_directories(path_.parent_path());
            file_ = open_file(path_, truncate ? "wb" : "ab");
            if (!file_)
                spdlog::throw_spdlog_ex("Cannot open log: " + path_to_utf8(path_));
        }
        void rotate() {
            flush_();
            std::fclose(file_);
            file_ = nullptr;
            try {
                for (size_t index = max_files_; index > 0; --index) {
                    const auto source = rotated_path(path_, index - 1);
                    if (!fs::exists(source))
                        continue;
                    const auto destination = rotated_path(path_, index);
                    fs::remove(destination);
                    fs::rename(source, destination);
                }
                open(true);
                size_ = 0;
            } catch (...) {
                // Preserve the current log on failure and leave the sink usable.
                if (!file_)
                    open(false);
                std::error_code error;
                size_ = fs::file_size(path_, error);
                throw;
            }
        }
        fs::path path_;
        size_t max_bytes_, max_files_, size_ = 0;
        std::FILE* file_ = nullptr;
    };
} // namespace lfs::core
