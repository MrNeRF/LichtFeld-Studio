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
                              Metres,
                              Millimetres };
    using MetadataScalar = std::variant<std::int64_t, double, ExifRational, std::string>;
    struct XmpPropertyName {
        std::string_view namespace_uri;
        std::string_view name;
    };
    struct XmpPropertyLocation {
        // Points into the static supported-field table; no catalog storage is borrowed.
        const XmpPropertyName* field = nullptr;
        std::size_t description = 0;
        std::size_t ordinal = 0;
        // Empty for an unqualified image; otherwise a static supported subject label.
        std::string_view subject;
    };
    struct XmpFragmentInput {
        MetadataRecordHandle source;
        std::size_t offset = 0;
        std::size_t bytes = 0;
    };
    struct XmpPacketAssembly {
        std::string guid;
        std::size_t declared_bytes = 0;
        bool md5_verified = false;
        std::optional<MetadataRecordHandle> standard_source;
        std::vector<XmpFragmentInput> fragments;
    };
    struct XmpFactProvenance {
        XmpPropertyLocation property;
        std::vector<XmpPropertyLocation> inputs;
        std::shared_ptr<const XmpPacketAssembly> assembly;
        bool extended_packet = false;
    };
    enum class XmpPacketState { Parsed,
                                Malformed,
                                Unsupported,
                                Limited };
    struct XmpPacketInterpretation {
        MetadataRecordHandle source;
        XmpPacketState state = XmpPacketState::Parsed;
        std::string diagnostic;
        std::shared_ptr<const XmpPacketAssembly> assembly;
    };
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
        // Allocate property provenance only for XMP facts; copies share it.
        std::shared_ptr<const XmpFactProvenance> xmp;
    };
    enum class MetadataSummaryScope { StreamProperties,
                                      ImageProperties };
    struct MetadataSummary {
        bool limited = false;
        std::vector<MetadataFact> facts;
        MetadataSummaryScope scope = MetadataSummaryScope::StreamProperties;
        std::vector<XmpPacketInterpretation> xmp_packets;
    };
    // Projects existing stream properties to typed, static facts with provenance.
    // Raw entries stay available; absence never becomes a fabricated zero value.
    [[nodiscard]] LFS_MEDIA_API Result<MetadataSummary> summarizeSourceMetadata(
        const SourceMetadataCatalog&, std::size_t max_facts = 4096, std::size_t max_text_bytes = 64 * 1024);
    // Interprets collected standard camera/optics/GPS/civil-time fields and
    // selected DJI photo declarations, with separate vendor semantics.
    // No file reads, timezone inference, datum conversion or image transforms.
    [[nodiscard]] LFS_MEDIA_API Result<MetadataSummary> summarizeImageMetadata(
        const SourceMetadataCatalog&, std::size_t max_facts = 4096, std::size_t max_text_bytes = 64 * 1024);
    [[nodiscard]] LFS_MEDIA_API std::string_view metadataUnitId(MetadataUnit) noexcept;
} // namespace lfs::media
