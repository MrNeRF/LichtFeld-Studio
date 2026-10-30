// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "media/file_frame_sink.hpp"
#include "media/source_metadata.hpp"
#include <cstdio>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::media::detail {
    inline constexpr char jpeg_metadata_policy[] = "jpeg-static-source-xmp-v1";
    inline constexpr char png_metadata_policy[] = "png-static-source-xmp-v1";
    inline constexpr char tiff_metadata_policy[] = "tiff16-static-source-xmp-v1";
    inline constexpr char exr_metadata_policy[] = "exr-static-source-xmp-v1";
    inline constexpr char jxl_metadata_policy[] = "jxl-static-source-xmp-v1";
    inline constexpr char avif_metadata_policy[] = "avif-static-source-xmp-v1";
    inline constexpr char sidecar_metadata_policy[] = "image-static-source-xmp-v1";
    [[nodiscard]] std::string_view embeddedMetadataPolicy(FrameFileFormat);
    // One prepared static observation per writer session. JPEG alone needs
    // Extended XMP markers when its Standard APP1 packet cannot hold the source.
    struct EmbeddedImageMetadata {
        FrameFileFormat format = FrameFileFormat::JPEG;
        int applied_rotation = 0;
        bool sidecar = false;
        std::string source_properties;
        std::string extended_guid;
        std::vector<std::uint8_t> extended_markers;
    };
    [[nodiscard]] Result<EmbeddedImageMetadata> prepareImageMetadata(const SourceMetadataCatalog&, int applied_rotation, FrameFileFormat, bool sidecar = false);
    [[nodiscard]] std::string_view imageXmpSuffix();
    [[nodiscard]] Result<std::string> imageXmpPrefix(const EmbeddedImageMetadata&, int width, int height, const FrameInfo&);
    // Contiguous representation required by TIFF/OpenEXR setters, libjxl boxes and libheif metadata items.
    [[nodiscard]] Result<std::string> imageXmpPacket(const EmbeddedImageMetadata&, int width, int height, const FrameInfo&);
    // Encoded bytes are produced by our JPEG encoders. Inserts metadata without
    // recompression into the atomic writer's borrowed stream; never closes it.
    [[nodiscard]] SinkResult writeJpegMetadata(FILE*, std::span<const std::uint8_t>,
                                               const EmbeddedImageMetadata&, int width, int height, const FrameInfo&);
} // namespace lfs::media::detail
