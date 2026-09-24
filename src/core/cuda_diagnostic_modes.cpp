/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda_error.hpp"

#include "core/environment.hpp"

#include <cctype>
#include <cstdio>

namespace lfs::core {
    namespace {
        constexpr std::string_view kModeTokenCudaSync = "cuda-sync";
        constexpr std::string_view kModeTokenDeviceTrap = "device-trap";
        constexpr std::string_view kModeTokenVkFatal = "vk-fatal";

        [[nodiscard]] std::string_view trim_ascii_whitespace(std::string_view value) noexcept {
            const auto is_space = [](const char ch) noexcept {
                return std::isspace(static_cast<unsigned char>(ch)) != 0;
            };
            while (!value.empty() && is_space(value.front())) {
                value.remove_prefix(1);
            }
            while (!value.empty() && is_space(value.back())) {
                value.remove_suffix(1);
            }
            return value;
        }

        // Recognizes the legacy boolean spellings shared with environment::flag().
        [[nodiscard]] std::optional<bool> parse_legacy_bool_token(const std::string_view value) noexcept {
            using environment::detail::equals_ignore_ascii_case;
            if (equals_ignore_ascii_case(value, "1") || equals_ignore_ascii_case(value, "true") ||
                equals_ignore_ascii_case(value, "yes") || equals_ignore_ascii_case(value, "on")) {
                return true;
            }
            if (equals_ignore_ascii_case(value, "0") || equals_ignore_ascii_case(value, "false") ||
                equals_ignore_ascii_case(value, "no") || equals_ignore_ascii_case(value, "off")) {
                return false;
            }
            return std::nullopt;
        }

        struct ModeListParse {
            unsigned modes = 0;
            std::vector<std::string_view> unknown_tokens;
        };

        [[nodiscard]] ModeListParse parse_mode_list_tokens(const std::string_view value) {
            using environment::detail::equals_ignore_ascii_case;
            ModeListParse result;
            size_t pos = 0;
            while (pos <= value.size()) {
                const size_t comma = value.find(',', pos);
                const std::string_view raw_token = comma == std::string_view::npos
                                                       ? value.substr(pos)
                                                       : value.substr(pos, comma - pos);
                const std::string_view token = trim_ascii_whitespace(raw_token);
                if (!token.empty()) {
                    if (equals_ignore_ascii_case(token, kModeTokenCudaSync)) {
                        result.modes |= static_cast<unsigned>(DiagnosticMode::CudaSync);
                    } else if (equals_ignore_ascii_case(token, kModeTokenDeviceTrap)) {
                        result.modes |= static_cast<unsigned>(DiagnosticMode::DeviceTrap);
                    } else if (equals_ignore_ascii_case(token, kModeTokenVkFatal)) {
                        result.modes |= static_cast<unsigned>(DiagnosticMode::VkFatal);
                    } else {
                        result.unknown_tokens.push_back(token);
                    }
                }
                if (comma == std::string_view::npos) {
                    break;
                }
                pos = comma + 1;
            }
            return result;
        }
    } // namespace

    ParsedDiagnosticModes parse_diagnostic_modes(
        const std::optional<std::string_view> sync_debug_value,
        const std::optional<std::string_view> vk_validation_fatal_value) noexcept {
        ParsedDiagnosticModes result;
        try {
            if (sync_debug_value) {
                const std::string_view trimmed = trim_ascii_whitespace(*sync_debug_value);
                if (!trimmed.empty()) {
                    if (const auto legacy = parse_legacy_bool_token(trimmed)) {
                        if (*legacy) {
                            result.modes |= static_cast<unsigned>(DiagnosticMode::CudaSync);
                        }
                    } else {
                        const ModeListParse parsed = parse_mode_list_tokens(trimmed);
                        result.modes |= parsed.modes;
                        if (!parsed.unknown_tokens.empty()) {
                            result.unknown_tokens_present = true;
                            for (size_t i = 0; i < parsed.unknown_tokens.size(); ++i) {
                                if (i != 0) {
                                    result.unknown_tokens += ", ";
                                }
                                result.unknown_tokens += parsed.unknown_tokens[i];
                            }
                        }
                    }
                }
            }
            if (vk_validation_fatal_value) {
                const std::string_view trimmed = trim_ascii_whitespace(*vk_validation_fatal_value);
                if (!trimmed.empty()) {
                    result.legacy_alias_present = true;
                    if (const auto legacy = parse_legacy_bool_token(trimmed); legacy && *legacy) {
                        result.modes |= static_cast<unsigned>(DiagnosticMode::VkFatal);
                    }
                }
            }
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): parsing must never turn a startup env-var read
            // into a crash; fall back to whatever modes were resolved before the failure.
        }
        return result;
    }

    unsigned diagnostic_modes() noexcept {
        static const unsigned modes = [] {
            const auto sync_debug_value = environment::value("LFS_CUDA_SYNC_DEBUG");
            const auto vk_validation_fatal_value = environment::value("LFS_VK_VALIDATION_FATAL");
            const ParsedDiagnosticModes parsed = parse_diagnostic_modes(
                sync_debug_value ? std::optional<std::string_view>{*sync_debug_value} : std::nullopt,
                vk_validation_fatal_value ? std::optional<std::string_view>{*vk_validation_fatal_value}
                                          : std::nullopt);
            try {
                if (parsed.legacy_alias_present) {
                    std::fprintf(
                        stderr,
                        "LFS_VK_VALIDATION_FATAL is deprecated; set LFS_CUDA_SYNC_DEBUG=vk-fatal "
                        "(or add vk-fatal to its mode list) instead.\n");
                }
                if (parsed.unknown_tokens_present) {
                    std::fprintf(
                        stderr,
                        "LFS_CUDA_SYNC_DEBUG: ignoring unknown mode token(s) [%s]; valid modes are "
                        "cuda-sync, device-trap, vk-fatal.\n",
                        parsed.unknown_tokens.c_str());
                }
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): a diagnostics warning about bad env tokens
                // must not itself become a failure mode.
            }
            return parsed.modes;
        }();
        return modes;
    }

    bool diagnostic_mode_enabled(const DiagnosticMode mode) noexcept {
        return (diagnostic_modes() & static_cast<unsigned>(mode)) != 0;
    }
} // namespace lfs::core
