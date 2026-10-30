/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "image_codecs.hpp"
#include <string_view>
namespace lfs::core::image_codecs {
    enum class AvifPrimaries { Bt709,
                               Bt2020 };
    struct AvifEncodeOptions {
        bool lossless = true;
        int quality = 90;  // Effective only for lossy color; alpha stays lossless.
        int speed = 8;     // libaom speed [0,9], higher is faster.
        int bit_depth = 8; // UInt8 for 8 bits; native UInt16 code values for 10/12.
        AvifPrimaries primaries = AvifPrimaries::Bt709;
        size_t pixels = 64u * 1024 * 1024;
        size_t max_output_bytes = 256u * 1024 * 1024;
        int threads = 1; // Explicit per-call bound, not an aggregate pool.
        std::function<bool()> cancelled;
    };
    [[nodiscard]] LFS_IMAGE_CODECS_API bool avif_encoder_available();
    [[nodiscard]] LFS_IMAGE_CODECS_API std::string avif_encoder_version();
    [[nodiscard]] LFS_IMAGE_CODECS_API Status validate_avif_encode_options(const AvifEncodeOptions&);
    // Interleaved gray/gray-alpha/RGB/RGBA, straight alpha, declared sRGB.
    // UInt16 contains actual 10/12-bit code values, not normalized full-range
    // samples. No float conversion, clipping, metadata copying or tone mapping.
    // Optional UTF-8 XML packet, borrowed synchronously and copied by libheif
    // into an uncompressed application/rdf+xml MIME item on the primary image.
    // Maximum 32 MiB, no NUL; native callers supply valid XML.
    // Output byte limit bounds serialization, not libheif/libaom working RAM.
    [[nodiscard]] LFS_IMAGE_CODECS_API Status encode_avif_memory(
        std::span<const uint8_t>, int width, int height, int channels,
        size_t row_stride, std::vector<uint8_t>&, const AvifEncodeOptions& = {}, std::string_view xmp = {});
    [[nodiscard]] LFS_IMAGE_CODECS_API Status write_avif(
        const std::filesystem::path&, std::span<const uint8_t>, int width, int height,
        int channels, size_t row_stride, const AvifEncodeOptions& = {}, const AtomicFileOptions& = {}, std::string_view xmp = {});
} // namespace lfs::core::image_codecs
