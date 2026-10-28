// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "media/frame_sink.hpp"
#include "media/source_metadata.hpp"
#include <cstdio>
#include <span>
#include <string>
#include <vector>

namespace lfs::media::detail {
    inline constexpr char jpeg_metadata_policy[] = "jpeg-static-source-xmp-v1";
    // Reusable across frames. Large source provenance is an Extended XMP
    // tree; individual frames carry only its link and their own output facts.
    struct JpegMetadata {
        int applied_rotation = 0;
        std::string source_properties;
        std::string extended_guid;
        std::vector<std::uint8_t> extended_markers;
    };
    [[nodiscard]] Result<JpegMetadata> prepareJpegMetadata(const SourceMetadataCatalog&, int applied_rotation);
    // Encoded bytes are produced by our JPEG encoders. Inserts metadata without
    // recompression into the atomic writer's borrowed stream; never closes it.
    [[nodiscard]] SinkResult writeJpegMetadata(FILE*, std::span<const std::uint8_t>,
                                               const JpegMetadata&, int width, int height, const FrameInfo&);
} // namespace lfs::media::detail
