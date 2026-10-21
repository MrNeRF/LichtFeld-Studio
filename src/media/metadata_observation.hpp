// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "media/source_metadata.hpp"
#include <cstdint>
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
                              Metres,
                              Millimetres };
    using MetadataScalar = std::variant<std::int64_t, double, ExifRational, std::string>;
    struct MetadataFact {
        MetadataRecordHandle source;
        std::string semantic;
        MetadataScalar value;
        MetadataUnit unit = MetadataUnit::Unspecified;
        MetadataValueState state = MetadataValueState::Present;
        MetadataValueOrigin origin = MetadataValueOrigin::DemuxerObservation;
        // Additional operands in the same source observation, for derived facts.
        std::vector<MetadataRecordHandle> inputs;
        std::string derivation;
        std::string diagnostic;
        // Duplicate semantic occurrences in one directory are not resolved here.
        bool ambiguous = false;
    };
    struct MetadataSummary {
        bool limited = false;
        std::vector<MetadataFact> facts;
        std::string_view scope = "static_stream_properties";
    };
    // Projects existing stream properties to typed, static facts with provenance.
    // Raw entries stay available; absence never becomes a fabricated zero value.
    [[nodiscard]] LFS_MEDIA_API Result<MetadataSummary> summarizeSourceMetadata(
        const SourceMetadataCatalog&, std::size_t max_facts = 4096, std::size_t max_text_bytes = 64 * 1024);
    // Interprets standard camera/optics/GPS/civil-time fields already collected.
    // No file reads, timezone inference, datum conversion or image transforms.
    [[nodiscard]] LFS_MEDIA_API Result<MetadataSummary> summarizeImageMetadata(
        const SourceMetadataCatalog&, std::size_t max_facts = 4096, std::size_t max_text_bytes = 64 * 1024);
    [[nodiscard]] LFS_MEDIA_API std::string_view metadataUnitId(MetadataUnit) noexcept;
} // namespace lfs::media
