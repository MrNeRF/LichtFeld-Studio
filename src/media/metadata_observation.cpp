// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "media/metadata_observation.hpp"
extern "C" {
#include <libavutil/mathematics.h>
}
#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <type_traits>

namespace lfs::media {
    namespace {
        Error invalid(const char* message) {
            return make_error({.code = ErrorCode::InvalidArgument, .domain = ErrorDomain::IO, .detail = message, .detection = LFS_SOURCE_SITE_CURRENT()});
        }
        Status rejected(const char* message) { return Status::failure(invalid(message)); }
        Status exhausted(const char* message) {
            return Status::failure(make_error({.code = ErrorCode::ResourceExhausted, .domain = ErrorDomain::IO, .detail = message, .detection = LFS_SOURCE_SITE_CURRENT()}));
        }
        bool validTime(const Timestamp& value) {
            return value.ticks != std::numeric_limits<int64_t>::min() && value.time_base.numerator > 0 && value.time_base.denominator > 0;
        }
        int compare(const Timestamp& a, const Timestamp& b) {
            return av_compare_ts(a.ticks, {a.time_base.numerator, a.time_base.denominator},
                                 b.ticks, {b.time_base.numerator, b.time_base.denominator});
        }
        bool validState(MetadataValueState value) {
            switch (value) {
            case MetadataValueState::Present:
            case MetadataValueState::Absent:
            case MetadataValueState::Invalid:
            case MetadataValueState::Unsupported: return true;
            }
            return false;
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
        case MetadataUnit::Radians: return "radians";
        case MetadataUnit::Metres: return "metres";
        case MetadataUnit::MetresPerSecond: return "metres_per_second";
        case MetadataUnit::MetresPerSecondSquared: return "metres_per_second_squared";
        case MetadataUnit::DegreesPerSecond: return "degrees_per_second";
        case MetadataUnit::RadiansPerSecond: return "radians_per_second";
        }
        return "unknown";
    }

    Status validateMetadataSeries(const SourceMetadataCatalog& catalog, const MetadataNumericSeries& series, const MetadataSeriesLimits& limits) {
        if (!catalog.resolve(series.source))
            return rejected("Metadata series belongs to another observation or an invalid record");
        if (series.scope != MetadataTemporalScope::PerFrame && series.scope != MetadataTemporalScope::SampleSeries)
            return rejected("Numeric series requires per-frame or sample-series timing");
        if (series.clock.domain.empty() || (series.clock.stream_index && *series.clock.stream_index < 0))
            return rejected("Metadata series requires an explicit clock and valid stream identity");
        if (series.clock.stream_index) {
            if (!catalog.container() || std::none_of(catalog.container()->streams.begin(), catalog.container()->streams.end(),
                                                     [&](const auto& stream) { return stream.index == *series.clock.stream_index; }))
                return rejected("Metadata clock references a stream absent from its source observation");
            const auto source_stream = std::visit([&](const auto& reference) -> std::optional<int> {
                using T = std::decay_t<decltype(reference)>;
                if constexpr (std::is_same_v<T, StreamDescriptionReference>)
                    return catalog.streamDescription(reference)->index;
                else if constexpr (std::is_same_v<T, StreamTagReference> || std::is_same_v<T, StreamSideDataReference>)
                    return catalog.container()->source_metadata->streams[reference.stream].stream_index;
                else
                    return std::nullopt;
            },
                                                  series.source.reference);
            if (source_stream && *source_stream != *series.clock.stream_index)
                return rejected("Metadata clock stream differs from the source record stream");
        }
        if (series.origin != MetadataValueOrigin::InterpretedSource && series.origin != MetadataValueOrigin::DemuxerObservation && series.origin != MetadataValueOrigin::Derived)
            return rejected("Unknown metadata value origin");
        if (series.origin == MetadataValueOrigin::Derived ? series.derivation.empty() : !series.derivation.empty())
            return rejected("Metadata derivation must be explicit and reserved for derived values");
        if (series.channels.empty())
            return rejected("Metadata series requires at least one channel");
        if (series.channels.size() > limits.channels || series.timing.size() > limits.samples ||
            series.values.size() > limits.value_bytes / sizeof(double))
            return Status::failure(make_error({.code = ErrorCode::ResourceExhausted, .domain = ErrorDomain::IO, .detail = "Metadata series exceeds the channel, sample or value budget", .detection = LFS_SOURCE_SITE_CURRENT()}));
        if (series.timing.size() > std::numeric_limits<std::size_t>::max() / series.channels.size() ||
            series.values.size() != series.timing.size() * series.channels.size() || series.states.size() != series.timing.size())
            return rejected("Metadata series timing, states and channel layout do not match");
        std::size_t text_bytes = 0;
        const auto charge = [&](const std::string& value) {
            if (value.size() > limits.text_bytes - text_bytes)
                return false;
            text_bytes += value.size();
            return true;
        };
        if (!charge(series.clock.domain) || !charge(series.coordinates.axes) || !charge(series.coordinates.datum) || !charge(series.derivation))
            return exhausted("Metadata series text exceeds its budget");
        for (const auto& channel : series.channels) {
            if (channel.semantic.empty() || metadataUnitId(channel.unit) == "unknown" ||
                (channel.resolution && (!std::isfinite(*channel.resolution) || *channel.resolution <= 0)))
                return rejected("Metadata channel requires a valid semantic, unit and optional positive resolution");
            if (!charge(channel.semantic))
                return exhausted("Metadata series text exceeds its budget");
        }
        for (std::size_t i = 0; i < series.timing.size(); ++i) {
            const auto& time = series.timing[i];
            if (!validTime(time.position) || !validState(series.states[i]) ||
                (series.scope == MetadataTemporalScope::PerFrame && !time.frame_index))
                return rejected("Invalid metadata sample time, state or frame association");
            if (i) {
                const auto& previous = series.timing[i - 1];
                if (time.segment < previous.segment || (time.segment == previous.segment && compare(time.position, previous.position) < 0) ||
                    (series.scope == MetadataTemporalScope::PerFrame && time.segment == previous.segment && *time.frame_index <= *previous.frame_index))
                    return rejected("Metadata samples must preserve ordered clock segments and frame identities");
            }
            if (series.states[i] == MetadataValueState::Present) {
                for (std::size_t channel = 0; channel < series.channels.size(); ++channel)
                    if (!std::isfinite(series.values[i * series.channels.size() + channel]))
                        return rejected("Present metadata samples must be finite; invalid samples require an explicit state");
            }
        }
        return {};
    }

    namespace {
        Result<MetadataSampleRange> selectRange(const MetadataNumericSeries& series, const MetadataSampleWindow& window) {
            if (window.clock != series.clock || !validTime(window.begin) || !validTime(window.end) || compare(window.begin, window.end) > 0)
                return invalid("Metadata window requires the same explicit clock and a valid ordered interval");
            const auto first = std::lower_bound(series.timing.begin(), series.timing.end(), window.segment,
                                                [](const auto& time, auto segment) { return time.segment < segment; });
            const auto last = std::upper_bound(first, series.timing.end(), window.segment,
                                               [](auto segment, const auto& time) { return segment < time.segment; });
            const auto begin = std::lower_bound(first, last, window.begin,
                                                [](const auto& time, const auto& value) { return compare(time.position, value) < 0; });
            const auto end = std::lower_bound(begin, last, window.end,
                                              [](const auto& time, const auto& value) { return compare(time.position, value) < 0; });
            return MetadataSampleRange{static_cast<std::size_t>(begin - series.timing.begin()), static_cast<std::size_t>(end - begin)};
        }

    } // namespace
    const MetadataNumericSeries& ValidatedMetadataSeries::data() const noexcept {
        static const MetadataNumericSeries empty;
        return series_ ? *series_ : empty;
    }
    Result<MetadataSampleRange> ValidatedMetadataSeries::range(const MetadataSampleWindow& window) const {
        if (!series_)
            return invalid("Metadata series view has been moved from");
        return selectRange(*series_, window);
    }
    Result<ValidatedMetadataSeries> prepareMetadataSeries(const SourceMetadataCatalog& catalog, MetadataNumericSeries series, const MetadataSeriesLimits& limits) {
        const auto valid = validateMetadataSeries(catalog, series, limits);
        if (!valid)
            return valid.error();
        try {
            return ValidatedMetadataSeries(std::make_shared<const MetadataNumericSeries>(std::move(series)));
        } catch (const std::bad_alloc&) {
            return make_error({.code = ErrorCode::ResourceExhausted, .domain = ErrorDomain::IO, .detail = "Metadata series view allocation failed", .detection = LFS_SOURCE_SITE_CURRENT()});
        }
    }
    Result<MetadataSampleRange> metadataSampleRange(const SourceMetadataCatalog& catalog, const MetadataNumericSeries& series, const MetadataSampleWindow& window, const MetadataSeriesLimits& limits) {
        const auto valid = validateMetadataSeries(catalog, series, limits);
        if (!valid)
            return valid.error();
        return selectRange(series, window);
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
