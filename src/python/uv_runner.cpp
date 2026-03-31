/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "uv_runner.hpp"
#include "package_manager.hpp"

#include <core/logger.hpp>
#include <core/path_utils.hpp>

namespace lfs::python {

    UvRunner::~UvRunner() {
        cancel();
    }

    bool UvRunner::start(const std::vector<std::string>& args) {
        cancel();

        const auto uv = PackageManager::instance().uv_path();
        if (uv.empty()) {
            LOG_ERROR("UV not found");
            return false;
        }

        output_buffer_.clear();
        exit_code_ = -1;
        complete_ = false;

        if (!process_.start(lfs::core::path_to_utf8(uv), args)) {
            LOG_ERROR("Failed to start UV process");
            return false;
        }

        running_ = true;
        LOG_INFO("UV started with {} args", args.size());
        return true;
    }

    bool UvRunner::poll() {
        if (!running_.load()) {
            return false;
        }

        char buf[4096];
        ssize_t n;

        while ((n = process_.read(buf, sizeof(buf) - 1)) > 0) {
            buf[n] = '\0';

            if (raw_output_callback_) {
                raw_output_callback_(std::string(buf, n));
            } else {
                output_buffer_ += std::string(buf, n);
                process_buffer();
            }
        }

        if (n < 0 || !process_.is_running()) {
            // Process finished
            if (!raw_output_callback_) {
                process_buffer();
                if (!output_buffer_.empty()) {
                    emit_line(output_buffer_, false);
                    output_buffer_.clear();
                }
            }

            exit_code_ = process_.wait();
            running_ = false;
            complete_ = true;

            LOG_INFO("UV completed with exit code {}", exit_code_);

            if (completion_callback_) {
                completion_callback_(exit_code_ == 0, exit_code_);
            }
            return false;
        }

        return true;
    }

    void UvRunner::cancel() {
        if (running_.load()) {
            process_.kill();
            running_ = false;
            complete_ = true;
            exit_code_ = -1;
            LOG_INFO("UV operation cancelled");
        }
    }

    void UvRunner::process_buffer() {
        size_t pos = 0;
        size_t len = output_buffer_.size();

        while (pos < len) {
            size_t newline = output_buffer_.find('\n', pos);
            size_t cr = output_buffer_.find('\r', pos);

            if (newline == std::string::npos && cr == std::string::npos) {
                break;
            }

            size_t line_end;
            bool is_cr_only = false;

            if (newline != std::string::npos && (cr == std::string::npos || newline < cr)) {
                line_end = newline;
            } else {
                line_end = cr;
                is_cr_only = (newline == std::string::npos || cr < newline);
            }

            std::string line = output_buffer_.substr(pos, line_end - pos);
            emit_line(line, is_cr_only);

            if (is_cr_only && line_end + 1 < len && output_buffer_[line_end + 1] == '\n') {
                pos = line_end + 2;
            } else {
                pos = line_end + 1;
            }
        }

        if (pos > 0) {
            output_buffer_.erase(0, pos);
        }
    }

    void UvRunner::emit_line(const std::string& line, bool is_line_update) {
        if (output_callback_)
            output_callback_(line, false, is_line_update);
    }

} // namespace lfs::python
