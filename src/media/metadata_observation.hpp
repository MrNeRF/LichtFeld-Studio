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
                              Metres };
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
