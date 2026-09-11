/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "core/failure_report.hpp"
#include "core/logger.hpp"

#include <format>
#include <stdexcept>

namespace lfs::core::detail {

    [[noreturn]] void assertion_failed(
        const std::string_view contract, const std::string_view expression,
        const std::string_view message, const SourceSite location) {
        std::string error = std::format("{} failed: {}", contract, expression);
        if (!message.empty()) {
            error += " — ";
            error += message;
        }
        error += std::format(" ({}:{})", location.file_name(), location.line());
        throw std::runtime_error(error);
    }

} // namespace lfs::core::detail

namespace lfs::core {

    struct Logger::Impl {};

    Logger::Logger()
        : impl_(std::make_unique<Impl>()) {}

    Logger::~Logger() = default;

    Logger& Logger::get() {
        static Logger logger;
        return logger;
    }

    void Logger::log(LogLevel, const SourceSite&, std::string_view) {}

} // namespace lfs::core
