/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <format>
#include <optional>
#include <string>
#include <string_view>

namespace lfs::io::project {

    namespace detail_path {

        [[nodiscard]] inline std::string ascii_lower(std::string value) {
            std::ranges::transform(
                value, value.begin(), [](const unsigned char character) {
                    return static_cast<char>(std::tolower(character));
                });
            return value;
        }

        [[nodiscard]] inline bool is_digit_run(
            const std::string_view text, const std::size_t begin, const std::size_t end) {
            if (begin >= end) {
                return false;
            }
            for (std::size_t index = begin; index < end; ++index) {
                if (!std::isdigit(static_cast<unsigned char>(text[index]))) {
                    return false;
                }
            }
            return true;
        }

        // True when `rest` is `<digits>` or `<digits>-<digits>`.
        [[nodiscard]] inline bool is_corrupt_stamp_suffix(const std::string_view rest) {
            const auto dash = rest.find('-');
            if (dash == std::string_view::npos) {
                return is_digit_run(rest, 0, rest.size());
            }
            return is_digit_run(rest, 0, dash) &&
                   is_digit_run(rest, dash + 1, rest.size());
        }

        inline constexpr std::string_view kUnpublishedTokens[] = {
            ".project-write.",
            ".compact.",
            ".replace-backup.",
            ".recovery-session.",
            ".corrupt-",
        };

        inline constexpr std::string_view kWriteTempTags[] = {
            ".project-write.",
            ".compact.",
            ".replace-backup.",
            ".recovery-session.",
        };

    } // namespace detail_path

    [[nodiscard]] inline bool isPublishedLichtPath(const std::filesystem::path& path) {
        const auto filename = detail_path::ascii_lower(path.filename().string());
        const auto extension = detail_path::ascii_lower(path.extension().string());
        if (extension != ".licht") {
            return false;
        }
        for (const auto token : detail_path::kUnpublishedTokens) {
            if (filename.find(token) != std::string::npos) {
                return false;
            }
        }
        if (filename.ends_with(".tmp.licht")) {
            return false;
        }
        if (filename.ends_with(".autosave") ||
            filename.find(".autosave.") != std::string::npos) {
            return false;
        }
        return true;
    }

    [[nodiscard]] inline std::optional<std::filesystem::path>
    derivedPublishedMasterPath(const std::filesystem::path& path) {
        auto name = path.filename().string();
        auto lower = detail_path::ascii_lower(name);
        while (true) {
            const auto marker = lower.rfind(".corrupt-");
            if (marker == std::string::npos) {
                break;
            }
            if (!detail_path::is_corrupt_stamp_suffix(
                    std::string_view(lower).substr(marker + 9))) {
                break;
            }
            name.resize(marker);
            lower.resize(marker);
        }
        if (lower.ends_with(".autosave")) {
            name.resize(name.size() - 9);
            lower.resize(lower.size() - 9);
        }

        if (lower.ends_with(".tmp.licht")) {
            for (const auto tag : detail_path::kWriteTempTags) {
                const auto tag_at = lower.find(tag);
                if (tag_at != std::string::npos && tag_at > 0) {
                    const auto stem = name.substr(0, tag_at);
                    return path.parent_path() / (stem + ".licht");
                }
            }
        }

        const auto remainder = path.parent_path() / name;
        if (isPublishedLichtPath(remainder) && remainder != path) {
            return remainder;
        }
        return std::nullopt;
    }

    [[nodiscard]] inline std::string unpublishedLichtUserMessage(
        const std::filesystem::path& path) {
        if (const auto master = derivedPublishedMasterPath(path)) {
            return std::format(
                "This path is not a published LichtFeld project. Open '{}' instead.",
                master->generic_string());
        }
        return std::format(
            "'{}' is not a published LichtFeld project "
            "(temporary, autosave, or recovery artifact).",
            path.filename().string());
    }

} // namespace lfs::io::project
