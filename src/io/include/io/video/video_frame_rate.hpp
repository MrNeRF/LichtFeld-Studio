/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <array>
#include <charconv>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>

namespace lfs::io::video {
    struct VideoFrameRate {
        int numerator = 30;
        int denominator = 1;
        [[nodiscard]] double value() const { return static_cast<double>(numerator) / denominator; }
        bool operator==(const VideoFrameRate&) const = default;
    };

    struct VideoFrameRatePreset {
        const char* label;
        VideoFrameRate rate;
    };

    inline constexpr std::array<VideoFrameRatePreset, 22> VIDEO_FRAME_RATES{{{"1", {1, 1}}, {"5", {5, 1}}, {"10", {10, 1}}, {"12", {12, 1}}, {"15", {15, 1}}, {"23.976", {24000, 1001}}, {"24", {24, 1}}, {"25", {25, 1}}, {"29.97", {30000, 1001}}, {"30", {30, 1}}, {"48", {48, 1}}, {"50", {50, 1}}, {"59.94", {60000, 1001}}, {"60", {60, 1}}, {"90", {90, 1}}, {"100", {100, 1}}, {"119.88", {120000, 1001}}, {"120", {120, 1}}, {"144", {144, 1}}, {"200", {200, 1}}, {"239.76", {240000, 1001}}, {"240", {240, 1}}}};

    [[nodiscard]] inline std::optional<VideoFrameRate> parseVideoFrameRate(std::string_view text) {
        const auto first = text.find_first_not_of(" \t");
        if (first == std::string_view::npos)
            return std::nullopt;
        text = text.substr(first, text.find_last_not_of(" \t") - first + 1);
        std::string normalized(text);
        for (char& c : normalized)
            if (c == ',')
                c = '.';
        if (normalized.find('/') == std::string::npos && normalized.find('.') != std::string::npos) {
            while (!normalized.empty() && normalized.back() == '0')
                normalized.pop_back();
            if (!normalized.empty() && normalized.back() == '.')
                normalized.pop_back();
        }
        for (const auto& preset : VIDEO_FRAME_RATES)
            if (normalized == preset.label)
                return preset.rate;
        const auto parse = [](std::string_view part) -> std::optional<int> {
            if (part.empty() || part.front() < '0' || part.front() > '9')
                return std::nullopt;
            int value = 0;
            const auto [end, error] = std::from_chars(part.data(), part.data() + part.size(), value);
            if (error != std::errc{} || end != part.data() + part.size())
                return std::nullopt;
            return value;
        };
        int numerator = 0;
        int denominator = 1;
        const auto separator = normalized.find_first_of("./");
        if (separator == std::string::npos) {
            const auto value = parse(normalized);
            if (!value)
                return std::nullopt;
            numerator = *value;
        } else if (normalized[separator] == '/') {
            const auto n = parse(std::string_view(normalized).substr(0, separator));
            const auto d = parse(std::string_view(normalized).substr(separator + 1));
            if (!n || !d || *d == 0)
                return std::nullopt;
            numerator = *n;
            denominator = *d;
        } else {
            const auto digits = normalized.size() - separator - 1;
            if (digits == 0 || digits > 6)
                return std::nullopt;
            normalized.erase(separator, 1);
            const auto value = parse(normalized);
            if (!value)
                return std::nullopt;
            numerator = *value;
            for (size_t i = 0; i < digits; ++i)
                denominator *= 10;
        }
        if (numerator < denominator || static_cast<long long>(numerator) > 1000LL * denominator)
            return std::nullopt;
        const int divisor = std::gcd(numerator, denominator);
        return VideoFrameRate{numerator / divisor, denominator / divisor};
    }

    [[nodiscard]] inline std::string videoFrameRateLabel(VideoFrameRate rate) {
        if (rate.numerator <= 0 || rate.denominator <= 0)
            return "";
        const int divisor = std::gcd(rate.numerator, rate.denominator);
        rate = {rate.numerator / divisor, rate.denominator / divisor};
        for (const auto& preset : VIDEO_FRAME_RATES)
            if (rate == preset.rate)
                return preset.label;
        return rate.denominator == 1 ? std::to_string(rate.numerator) : std::to_string(rate.numerator) + "/" + std::to_string(rate.denominator);
    }
} // namespace lfs::io::video
