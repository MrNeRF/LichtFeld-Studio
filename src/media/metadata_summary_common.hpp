// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "media/metadata_observation.hpp"
#include <cmath>
#include <numeric>

namespace lfs::media::summary_detail {
    namespace semantic {
        inline constexpr std::string_view make = "camera.make", model = "camera.model", orientation = "orientation.exif";
        inline constexpr std::string_view lens_make = "lens.make", lens_model = "lens.model", exposure = "exposure.duration";
        inline constexpr std::string_view f_number = "lens.f_number", focal_length = "lens.focal_length";
        inline constexpr std::string_view latitude = "gps.latitude", longitude = "gps.longitude", altitude = "gps.altitude_sea_level", datum = "gps.map_datum";
        inline constexpr std::string_view original = "time.original", digitized = "time.digitized", modified = "time.modified";
    } // namespace semantic

    // Both carriers decode their own syntax, then apply the same physical bounds.
    inline bool coordinate(double degrees, double minutes, double seconds, bool latitude, char direction, double& result) {
        const double maximum = latitude ? 90 : 180;
        if (!std::isfinite(degrees) || !std::isfinite(minutes) || !std::isfinite(seconds) ||
            degrees < 0 || degrees > maximum || minutes < 0 || minutes >= 60 || seconds < 0 || seconds >= 60 ||
            (degrees == maximum && (minutes != 0 || seconds != 0)) ||
            (latitude ? direction != 'N' && direction != 'S' : direction != 'E' && direction != 'W'))
            return false;
        result = degrees + minutes / 60 + seconds / 3600;
        if (result > maximum)
            return false;
        if (direction == 'S' || direction == 'W')
            result = -result;
        return true;
    }

    inline bool altitude(double magnitude, int reference, double& result) {
        if (!std::isfinite(magnitude) || magnitude < 0 || (reference != 0 && reference != 1))
            return false;
        result = reference == 1 ? -magnitude : magnitude;
        return true;
    }

    inline std::optional<std::size_t> factCost(MetadataSummary& summary, std::size_t max_facts,
                                               std::size_t max_bytes, std::size_t charged,
                                               std::string_view semantic, const MetadataScalar& value,
                                               std::string_view diagnostic = {}, std::string_view derivation = {},
                                               std::size_t provenance_bytes = 0) {
        const auto* text = std::get_if<std::string>(&value);
        std::size_t cost = 0;
        if (summary.facts.size() < max_facts && charged <= max_bytes) {
            for (const auto size : {semantic.size(), diagnostic.size(), derivation.size(), text ? text->size() : 0, provenance_bytes}) {
                if (size > max_bytes - charged - cost) {
                    summary.limited = true;
                    return std::nullopt;
                }
                cost += size;
            }
            return cost;
        }
        summary.limited = true;
        return std::nullopt;
    }

    inline bool equalValue(const MetadataScalar& left, const MetadataScalar& right) {
        if (const auto* text = std::get_if<std::string>(&left)) {
            const auto* other = std::get_if<std::string>(&right);
            return other && *text == *other;
        }
        if (std::holds_alternative<std::string>(right))
            return false;
        // Integer/rational comparison remains exact, without cross multiplication.
        const auto ratio = [](const MetadataScalar& value) -> std::optional<ExifRational> {
            if (const auto* integer = std::get_if<int64_t>(&value))
                return ExifRational{*integer, 1};
            if (const auto* rational = std::get_if<ExifRational>(&value); rational && rational->numerator >= 0 && rational->denominator > 0) {
                const auto divisor = std::gcd(rational->numerator, rational->denominator);
                return ExifRational{rational->numerator / divisor, rational->denominator / divisor};
            }
            return std::nullopt;
        };
        const auto a = ratio(left), b = ratio(right);
        if (a && b)
            return a->numerator == b->numerator && a->denominator == b->denominator;
        const auto* x = std::get_if<double>(&left);
        const auto* y = std::get_if<double>(&right);
        if (x && y)
            return std::isfinite(*x) && std::isfinite(*y) && *x == *y;
        const auto rational = a ? a : b;
        const auto* floating = x ? x : y;
        // No epsilon: compare at the precision of the floating carrier, only
        // when converting its rational operands cannot round an integer first.
        constexpr int64_t exact_integer = INT64_C(9007199254740992);
        return rational && floating && std::isfinite(*floating) &&
               rational->numerator >= -exact_integer && rational->numerator <= exact_integer &&
               rational->denominator <= exact_integer &&
               double(rational->numerator) / double(rational->denominator) == *floating;
    }

    inline bool equivalent(const MetadataFact& left, const MetadataFact& right) {
        return left.state == MetadataValueState::Present && right.state == MetadataValueState::Present &&
               left.unit == right.unit && equalValue(left.value, right.value);
    }
} // namespace lfs::media::summary_detail
