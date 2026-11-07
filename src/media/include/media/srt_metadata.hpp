// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "media/source_metadata.hpp"
#include <span>

namespace lfs::media {
    struct SrtMetadataLimits {
        std::size_t bytes = 32u * 1024 * 1024;
        std::size_t cues = 100000;
        // The provider splits longer physical lines, potentially inside a value.
        std::size_t line_bytes = 4095;
        std::size_t lines = 1000000;
        std::size_t facts = 500000;
        std::size_t cue_bytes = 256u * 1024;
    };
    // Offsets refer to the owned cue text, not to the input file. Unknown text
    // remains available even when no numeric interpretation is supported.
    struct DjiSrtFact {
        std::string_view semantic;
        std::string_view unit;
        std::size_t offset = 0;
        std::size_t bytes = 0;
        std::optional<double> value;
        bool ambiguous = false;
    };
    struct SrtCue {
        std::int64_t start_milliseconds = 0;
        std::int64_t end_milliseconds = 0;
        // FFmpeg packet position, not a range for the normalized text below.
        std::int64_t source_offset = -1;
        std::string text;
        std::vector<DjiSrtFact> facts;
        bool dji_bracket_profile = false;
    };
    struct SrtMetadataStorage;
    // Copies share one immutable observation. Views require a live catalog.
    class LFS_MEDIA_API SrtMetadataCatalog {
    public:
        [[nodiscard]] const MetadataSourceLocator& source() const noexcept;
        [[nodiscard]] std::span<const std::uint8_t> originalBytes() const noexcept;
        [[nodiscard]] std::span<const SrtCue> cues() const noexcept;
        // Explicit mapping: SRT time = timestamp + offset_milliseconds. Uses
        // half-open intervals, no nearest-value hold or interpolation. Multiple
        // covering cues are ambiguous and return FailedPrecondition. Zero-length
        // cues are exact point samples; optional tolerance enables nearest-point
        // fallback only when there is no exact/interval match. Ties fail.
        [[nodiscard]] Result<std::optional<std::size_t>> lookup(Timestamp, std::int64_t offset_milliseconds, std::optional<std::int64_t> point_tolerance_milliseconds = std::nullopt) const;

    private:
        explicit SrtMetadataCatalog(std::shared_ptr<const SrtMetadataStorage> storage) : storage_(std::move(storage)) {}
        std::shared_ptr<const SrtMetadataStorage> storage_;
        friend class MediaIngest;
    };
} // namespace lfs::media
