// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "media/source_metadata.hpp"
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::media {
    enum class MetadataValueOrigin { InterpretedSource,
                                     DemuxerObservation,
                                     Derived };
    enum class MetadataValueState { Present,
                                    Absent,
                                    Invalid,
                                    Unsupported };
    enum class MetadataUnit { Unspecified,
                              Unitless,
                              Pixels,
                              FramesPerSecond,
                              Seconds,
                              Degrees,
                              Radians,
                              Metres,
                              MetresPerSecond,
                              MetresPerSecondSquared,
                              DegreesPerSecond,
                              RadiansPerSecond };
    enum class MetadataTemporalScope { Static,
                                       PerFrame,
                                       SampleSeries };
    struct MetadataCoordinates {
        // Empty means unspecified, never silently WGS84, ENU or camera axes.
        std::string axes;
        std::string datum;
    };
    struct MetadataClock {
        // Caller/provider-declared domain; labels never imply synchronization.
        std::string domain;
        std::optional<int> stream_index;
        friend bool operator==(const MetadataClock&, const MetadataClock&) = default;
    };
    struct MetadataSampleTime {
        Timestamp position;
        // Increasing ordinal explicitly separates discontinuities/reset clocks.
        std::uint64_t segment = 0;
        // Optional explicit producer association; multiple sensor samples may
        // belong to the same video frame. Never generated from nominal FPS.
        std::optional<std::uint64_t> frame_index;
    };
    struct MetadataNumericChannel {
        std::string semantic;
        MetadataUnit unit = MetadataUnit::Unspecified;
        // Reported resolution, not inferred accuracy. Absent when unknown.
        std::optional<double> resolution;
    };
    struct MetadataNumericSeries {
        MetadataRecordHandle source;
        MetadataValueOrigin origin = MetadataValueOrigin::InterpretedSource;
        MetadataTemporalScope scope = MetadataTemporalScope::SampleSeries;
        MetadataClock clock;
        MetadataCoordinates coordinates;
        // Required for derived series; originals remain in the source catalog.
        std::string derivation;
        std::vector<MetadataNumericChannel> channels;
        std::vector<MetadataSampleTime> timing;
        // Interleaved samples, exactly timing.size() * channels.size().
        std::vector<double> values;
        std::vector<MetadataValueState> states;
    };
    struct MetadataSeriesLimits {
        std::size_t channels = 64;
        std::size_t samples = 1000000;
        std::size_t value_bytes = 64 * 1024 * 1024;
        std::size_t text_bytes = 64 * 1024;
    };
    // Validation does not copy values, infer units/clocks, interpolate, or sort.
    // Limits govern submitted content, not allocations already made by callers.
    [[nodiscard]] LFS_MEDIA_API Status validateMetadataSeries(
        const SourceMetadataCatalog&, const MetadataNumericSeries&, const MetadataSeriesLimits& = {});
    struct MetadataSampleWindow {
        MetadataClock clock;
        std::uint64_t segment = 0;
        Timestamp begin;
        Timestamp end; // Half-open interval, in the explicitly selected clock.
    };
    struct MetadataSampleRange {
        std::size_t first = 0;
        std::size_t count = 0;
    };
    // Internal owning, immutable view: validate once, then O(log samples) queries.
    // Not installed as a public SDK header until a production series producer exists.
    class ValidatedMetadataSeries {
    public:
        [[nodiscard]] LFS_MEDIA_API const MetadataNumericSeries& data() const noexcept;
        [[nodiscard]] LFS_MEDIA_API Result<MetadataSampleRange> range(const MetadataSampleWindow&) const;

    private:
        explicit ValidatedMetadataSeries(std::shared_ptr<const MetadataNumericSeries> series) : series_(std::move(series)) {}
        std::shared_ptr<const MetadataNumericSeries> series_;
        friend LFS_MEDIA_API Result<ValidatedMetadataSeries> prepareMetadataSeries(
            const SourceMetadataCatalog&, MetadataNumericSeries, const MetadataSeriesLimits&);
    };
    [[nodiscard]] LFS_MEDIA_API Result<ValidatedMetadataSeries> prepareMetadataSeries(
        const SourceMetadataCatalog&, MetadataNumericSeries, const MetadataSeriesLimits& = {});
    // One-off convenience. Repeated queries should use prepareMetadataSeries.
    [[nodiscard]] LFS_MEDIA_API Result<MetadataSampleRange> metadataSampleRange(
        const SourceMetadataCatalog&, const MetadataNumericSeries&, const MetadataSampleWindow&,
        const MetadataSeriesLimits& = {});

    using MetadataScalar = std::variant<std::int64_t, double, ExifRational, std::string>;
    struct MetadataFact {
        MetadataRecordHandle source;
        std::string semantic;
        MetadataScalar value;
        MetadataUnit unit = MetadataUnit::Unspecified;
        MetadataValueState state = MetadataValueState::Present;
        MetadataValueOrigin origin = MetadataValueOrigin::DemuxerObservation;
    };
    struct MetadataSummary {
        bool limited = false;
        std::vector<MetadataFact> facts;
    };
    // Projects existing stream properties to typed, static facts with provenance.
    // Raw entries stay available; absence never becomes a fabricated zero value.
    [[nodiscard]] LFS_MEDIA_API Result<MetadataSummary> summarizeSourceMetadata(
        const SourceMetadataCatalog&, std::size_t max_facts = 4096, std::size_t max_text_bytes = 64 * 1024);
    [[nodiscard]] LFS_MEDIA_API std::string_view metadataUnitId(MetadataUnit) noexcept;
} // namespace lfs::media
