// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media/metadata_observation.hpp"
#include <cmath>
#include <new>
#include <type_traits>

namespace lfs::media {
    namespace {
        Error invalid(const char* message) {
            return make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = message, .detection = LFS_SOURCE_SITE_CURRENT()});
        }
    } // namespace

    std::string_view metadataUnitId(MetadataUnit unit) noexcept {
        switch (unit) {
        case MetadataUnit::Unspecified: return "unspecified";
        case MetadataUnit::Unitless: return "unitless";
        case MetadataUnit::Pixels: return "pixels";
        case MetadataUnit::FramesPerSecond: return "frames_per_second";
        case MetadataUnit::Seconds: return "seconds";
        case MetadataUnit::Degrees: return "degrees";
        case MetadataUnit::Metres: return "metres";
        case MetadataUnit::Millimetres: return "millimetres";
        }
        return "unknown";
    }

    Result<MetadataSummary> summarizeSourceMetadata(const SourceMetadataCatalog& catalog, std::size_t max_facts, std::size_t max_text_bytes) {
        if (!catalog.container())
            return make_error({.code = ErrorCode::Unsupported, .domain = ErrorDomain::IO, .detail = "Static stream summary requires a container catalog", .detection = LFS_SOURCE_SITE_CURRENT()});
        try {
            MetadataSummary summary;
            std::size_t text_bytes = 0;
            const auto append = [&](const MetadataRecordHandle& source, const char* semantic, const auto& value, MetadataUnit unit) {
                if (summary.facts.size() >= max_facts) {
                    summary.limited = true;
                    return false;
                }
                const auto semantic_bytes = std::string_view(semantic).size();
                std::size_t value_bytes = 0;
                if constexpr (std::is_same_v<std::decay_t<decltype(value)>, std::string>)
                    value_bytes = value.size();
                if (semantic_bytes > max_text_bytes - text_bytes || value_bytes > max_text_bytes - text_bytes - semantic_bytes) {
                    summary.limited = true;
                    return false;
                }
                MetadataFact fact{source, semantic, value, unit};
                text_bytes += semantic_bytes + value_bytes;
                if (const auto* number = std::get_if<double>(&fact.value); number && !std::isfinite(*number))
                    fact.state = MetadataValueState::Invalid;
                if (const auto* ratio = std::get_if<ExifRational>(&fact.value); ratio && (ratio->denominator <= 0 || ratio->numerator <= 0))
                    fact.state = MetadataValueState::Invalid;
                if (const auto* integer = std::get_if<int64_t>(&fact.value); integer && unit == MetadataUnit::Pixels && *integer <= 0)
                    fact.state = MetadataValueState::Invalid;
                summary.facts.push_back(std::move(fact));
                return true;
            };
            // Traverse the bounded catalog index. Do not bypass an incomplete index.
            for (const auto& reference : catalog.records()) {
                const auto* stream_reference = std::get_if<StreamDescriptionReference>(&reference);
                if (!stream_reference)
                    continue;
                const auto handle = catalog.bind(reference);
                if (!handle)
                    return invalid("Metadata summary contains an invalid catalog reference");
                const auto& stream = *catalog.streamDescription(*stream_reference);
                summary.limited = summary.limited || stream.orientation.rotate_tag_limited;
                if (stream.width && !append(*handle, "image.width", int64_t(*stream.width), MetadataUnit::Pixels))
                    break;
                if (stream.height && !append(*handle, "image.height", int64_t(*stream.height), MetadataUnit::Pixels))
                    break;
                if (stream.nominal_frame_rate && !append(*handle, "frame_rate.nominal", ExifRational{stream.nominal_frame_rate->numerator, stream.nominal_frame_rate->denominator}, MetadataUnit::FramesPerSecond))
                    break;
                if (stream.average_frame_rate && !append(*handle, "frame_rate.average", ExifRational{stream.average_frame_rate->numerator, stream.average_frame_rate->denominator}, MetadataUnit::FramesPerSecond))
                    break;
                if (stream.orientation.clockwise_degrees && !append(*handle, "orientation.clockwise", *stream.orientation.clockwise_degrees, MetadataUnit::Degrees))
                    break;
                if (stream.color.primaries && !append(*handle, "color.primaries", *stream.color.primaries, MetadataUnit::Unspecified))
                    break;
                if (stream.color.transfer && !append(*handle, "color.transfer", *stream.color.transfer, MetadataUnit::Unspecified))
                    break;
            }
            summary.limited = summary.limited || catalog.indexStatus() == MetadataIndexStatus::Limited ||
                              (catalog.container() && catalog.container()->source_metadata && catalog.container()->source_metadata->limited);
            return summary;
        } catch (const std::bad_alloc&) {
            return make_error({.code = ErrorCode::ResourceExhausted, .domain = ErrorDomain::IO, .detail = "Metadata summary allocation failed", .detection = LFS_SOURCE_SITE_CURRENT()});
        }
    }
} // namespace lfs::media
