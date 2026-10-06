// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/failure_report.hpp"
#include <format>
#include <stdexcept>
namespace lfs::core::detail {
    [[noreturn]] void assertion_failed(std::string_view contract, std::string_view expression,
                                       std::string_view message, SourceSite location) {
        throw std::logic_error(std::format("{}: {} {} ({}:{})", contract, expression, message,
                                           location.file_name(), location.line()));
    }
} // namespace lfs::core::detail
