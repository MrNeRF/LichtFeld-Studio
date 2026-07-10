/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "method_session.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <format>
#include <ranges>
#include <string_view>

namespace lfs::vis {

    namespace {

        [[nodiscard]] std::string valid_option_keys(const MethodDescriptor& descriptor) {
            std::vector<std::string> keys;
            keys.reserve(descriptor.options.size());
            for (const auto& option : descriptor.options) {
                keys.push_back(option.key);
            }
            std::ranges::sort(keys);

            std::string result;
            for (const auto& key : keys) {
                if (!result.empty()) {
                    result += ", ";
                }
                result += key;
            }
            return result.empty() ? "(none)" : result;
        }

        [[nodiscard]] std::expected<std::int64_t, std::string> parse_integer(
            const MethodOptionSpec& spec,
            const std::string_view raw_value) {
            std::int64_t value = 0;
            const auto [end, error] = std::from_chars(
                raw_value.data(), raw_value.data() + raw_value.size(), value);
            if (error != std::errc{} || end != raw_value.data() + raw_value.size()) {
                return std::unexpected(std::format(
                    "Option '{}' expects an integer, got '{}'", spec.key, raw_value));
            }
            if (spec.min_value && static_cast<double>(value) < *spec.min_value) {
                return std::unexpected(std::format(
                    "Option '{}' must be at least {}, got {}", spec.key, *spec.min_value, value));
            }
            if (spec.max_value && static_cast<double>(value) > *spec.max_value) {
                return std::unexpected(std::format(
                    "Option '{}' must be at most {}, got {}", spec.key, *spec.max_value, value));
            }
            return value;
        }

        [[nodiscard]] std::expected<double, std::string> parse_float(
            const MethodOptionSpec& spec,
            const std::string_view raw_value) {
            double value = 0.0;
            const auto [end, error] = std::from_chars(
                raw_value.data(), raw_value.data() + raw_value.size(), value);
            if (error != std::errc{} || end != raw_value.data() + raw_value.size() ||
                !std::isfinite(value)) {
                return std::unexpected(std::format(
                    "Option '{}' expects a finite number, got '{}'", spec.key, raw_value));
            }
            if (spec.min_value && value < *spec.min_value) {
                return std::unexpected(std::format(
                    "Option '{}' must be at least {}, got {}", spec.key, *spec.min_value, value));
            }
            if (spec.max_value && value > *spec.max_value) {
                return std::unexpected(std::format(
                    "Option '{}' must be at most {}, got {}", spec.key, *spec.max_value, value));
            }
            return value;
        }

        [[nodiscard]] std::expected<MethodOptionValue, std::string> coerce_option(
            const MethodOptionSpec& spec,
            const std::string_view raw_value) {
            switch (spec.type) {
            case MethodOptionSpec::Type::Bool:
                if (raw_value == "true" || raw_value == "1") {
                    return MethodOptionValue{true};
                }
                if (raw_value == "false" || raw_value == "0") {
                    return MethodOptionValue{false};
                }
                return std::unexpected(std::format(
                    "Option '{}' expects true, 1, false, or 0, got '{}'",
                    spec.key,
                    raw_value));
            case MethodOptionSpec::Type::Int: {
                auto value = parse_integer(spec, raw_value);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return MethodOptionValue{*value};
            }
            case MethodOptionSpec::Type::Float: {
                auto value = parse_float(spec, raw_value);
                if (!value) {
                    return std::unexpected(value.error());
                }
                return MethodOptionValue{*value};
            }
            case MethodOptionSpec::Type::Enum:
                if (std::ranges::find(spec.choices, raw_value) == spec.choices.end()) {
                    std::string choices;
                    for (const auto& choice : spec.choices) {
                        if (!choices.empty()) {
                            choices += ", ";
                        }
                        choices += choice;
                    }
                    return std::unexpected(std::format(
                        "Option '{}' must be one of: {}; got '{}'",
                        spec.key,
                        choices.empty() ? "(none)" : choices,
                        raw_value));
                }
                return MethodOptionValue{std::string(raw_value)};
            case MethodOptionSpec::Type::String:
            case MethodOptionSpec::Type::Path:
                return MethodOptionValue{std::string(raw_value)};
            }
            return std::unexpected(std::format("Option '{}' has an unsupported type", spec.key));
        }

    } // namespace

    std::expected<MethodOptions, std::string> resolve_method_options(
        const MethodDescriptor& descriptor,
        const std::vector<std::pair<std::string, std::string>>& raw_options) {
        MethodOptions resolved;
        for (const auto& spec : descriptor.options) {
            resolved.insert_or_assign(spec.key, spec.default_value);
        }

        for (const auto& [key, raw_value] : raw_options) {
            const auto spec = std::ranges::find(descriptor.options, key, &MethodOptionSpec::key);
            if (spec == descriptor.options.end()) {
                return std::unexpected(std::format(
                    "Unknown option '{}' for method '{}'. Valid options: {}",
                    key,
                    descriptor.id,
                    valid_option_keys(descriptor)));
            }

            auto value = coerce_option(*spec, raw_value);
            if (!value) {
                return std::unexpected(value.error());
            }
            resolved.insert_or_assign(key, std::move(*value));
        }

        return resolved;
    }

} // namespace lfs::vis
