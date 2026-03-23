/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include "core/parameters.hpp"
#include <expected>
#include <memory>
#include <variant>

namespace lfs::core::args {

    // Parsed argument modes
    struct TrainingMode {
        std::unique_ptr<param::TrainingParameters> params;
    };
    struct ConvertMode {
        param::ConvertParameters params;
    };
    struct HelpMode {};
    struct VersionMode {};
    struct WarmupMode {}; // JIT compile PTX kernels and exit
    struct PluginMode {
        enum class Command { CREATE,
                             CHECK,
                             LIST };
        Command command;
        std::string name;
    };

    using ParsedArgs = std::variant<TrainingMode, ConvertMode, HelpMode, VersionMode, WarmupMode, PluginMode>;

    LFS_CORE_API std::expected<ParsedArgs, std::string> parse_args(int argc, const char* const argv[]);

    // Legacy interface - prefer parse_args()
    LFS_CORE_API std::expected<std::unique_ptr<param::TrainingParameters>, std::string>
    parse_args_and_params(int argc, const char* const argv[]);

} // namespace lfs::core::args
