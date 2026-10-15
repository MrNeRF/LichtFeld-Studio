// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "core/source_metadata.hpp"
#include <cstdint>
#include <vector>

namespace lfs::media {
    enum class ExifParseStatus { Parsed,
                                 Limited,
                                 Malformed,
                                 Unsupported };
    struct ExifRational {
        // Signed 64-bit storage also represents the full unsigned TIFF32 range.
        int64_t numerator = 0;
        int64_t denominator = 0;
    };
    struct ExifEntry {
        // Root IFD ordinal followed by parent entry ordinals. Duplicate IDs do
        // not overwrite one another, including within the same directory.
        std::vector<uint32_t> directory_path;
        uint32_t directory_offset = 0; // Relative to the original TIFF header.
        uint32_t entry_index = 0;
        uint16_t id = 0;
        uint16_t type = 0;
        uint32_t count = 0;
        bool directory = false;
        std::vector<uint64_t> unsigned_values;
        std::vector<int64_t> signed_values;
        std::vector<double> floating_values;
        std::vector<ExifRational> rational_values;
        // ASCII retains its declared byte count, NULs and invalid UTF-8.
        // BYTE/UNDEFINED and opaque MakerNotes also use this field.
        std::vector<uint8_t> bytes;
    };
    struct ExifInterpretation {
        ExifParseStatus status = ExifParseStatus::Unsupported;
        std::string diagnostic;
        std::vector<ExifEntry> entries;
        std::size_t directory_count = 0;
        std::size_t value_storage_charge = 0;
    };
    struct ExifParseLimits {
        std::size_t entries = 4096;
        std::size_t directories = 64;
        // Aggregate decoded value storage, counted before entering FFmpeg.
        std::size_t value_bytes = 8 * 1024 * 1024;
    };
    struct ImageExifDescription {
        core::image_codecs::SourceExif original;
        // One interpretation per original block, in matching order.
        std::vector<ExifInterpretation> interpretations;
    };
    // No pixel decode or metadata writing. MakerNotes remain opaque; only
    // standard classic TIFF IFDs are interpreted, not BigTIFF/vendor structures.
    [[nodiscard]] LFS_MEDIA_API ExifInterpretation interpretExif(
        const core::image_codecs::ExifBlock&, const ExifParseLimits& = {});
} // namespace lfs::media
