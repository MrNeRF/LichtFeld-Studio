/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/source_metadata.hpp"
#include "image_codecs.hpp"
#include <string_view>

namespace lfs::core::image_codecs {
    enum class JxlEncodingColor { SRGB,
                                  LinearSRGB,
                                  LinearBT2020,
                                  SRGBBT2020 };
    struct JxlEncodeOptions {
        bool lossless = true;
        float distance = 1.f; // Lossy perceptual distance, (0,25]; ignored for lossless.
        int effort = 4;       // [1,9]; lossless Float32 requires [4,9] with libjxl 0.12.
        JxlEncodingColor color = JxlEncodingColor::SRGB;
        bool premultiplied_alpha = false; // Float32 only; output alpha is straight.
        size_t pixels = 64u * 1024 * 1024;
        size_t encoder_bytes = size_t{1} << 30;
        // 0: memory output uses 256 MiB; files use max(256 MiB, 2*raw + 1 MiB).
        // A nonzero value is an explicit hard limit for either output API.
        size_t max_output_bytes = 0;
        size_t max_threads = 0; // Same bounded worker budget as JXL decoding.
        std::function<bool()> cancelled;
    };
    [[nodiscard]] LFS_IMAGE_CODECS_API uint32_t jxl_encoder_version();
    [[nodiscard]] LFS_IMAGE_CODECS_API Status validate_jxl_encode_options(const JxlEncodeOptions&);
    [[nodiscard]] LFS_IMAGE_CODECS_API Status validate_jxl_encode_options(const JxlEncodeOptions&, SampleType);
    // Borrowed interleaved gray/gray-alpha/RGB/RGBA, native-endian UInt8/16 or
    // Float32. Explicit byte stride, finite samples and declared output color.
    // No implicit tone mapping, integer quantization, orientation or metadata
    // copying. Lossy mode explicitly applies perceptual compression.
    // Memory output is replaced only on success, including aliased input/output.
    // File encoding drains bounded chunks to the existing atomic file service.
    // Explicit UTF-8 XML packet, borrowed for this synchronous call; libjxl
    // copies it into an uncompressed standard xml box. Maximum 32 MiB, no NUL.
    // Empty packets preserve the original codestream/container selection.
    [[nodiscard]] LFS_IMAGE_CODECS_API Status encode_jxl_memory(
        std::span<const uint8_t>, int width, int height, int channels, SampleType,
        size_t row_stride, std::vector<uint8_t>&, const JxlEncodeOptions& = {}, std::string_view xmp = {});
    [[nodiscard]] LFS_IMAGE_CODECS_API Status write_jxl(
        const std::filesystem::path&, std::span<const uint8_t>, int width, int height,
        int channels, SampleType, size_t row_stride, const JxlEncodeOptions& = {},
        const AtomicFileOptions& = {}, std::string_view xmp = {});
    enum class JxlColorPolicy { SRGB,
                                Native };
    // Per-call limits, not an aggregate process RAM budget. The decoder's
    // allocations are separate from the owning output and buffered file input.
    struct JxlDecodeLimits {
        size_t pixels = 64u * 1024 * 1024;
        size_t output_bytes = size_t{1} << 30;
        size_t decoder_bytes = size_t{1} << 30;
        size_t icc_bytes = 1u << 20;
        size_t buffered_input_bytes = 4u << 20;
        size_t encoded_bytes = size_t{2} << 30;
        JxlColorPolicy color_policy = JxlColorPolicy::SRGB;
        // 0 selects a bounded automatic pool; 1 explicitly selects serial.
        size_t max_threads = 0;
    };

    [[nodiscard]] SourceImageMetadata read_jxl_metadata(const std::filesystem::path&, const MetadataLimits&, const XmpLimits*);

    LFS_IMAGE_CODECS_API bool is_jxl(std::span<const uint8_t>);
    LFS_IMAGE_CODECS_API bool probe_jxl(const std::filesystem::path&, Probe&, std::string&, const JxlDecodeLimits& = {});
    // Default output is SDR sRGB at every precision, including float. Native
    // must be explicitly requested through limits for source/HDR sample access;
    // it uses libjxl's DATA encoding, which may differ from the original ICC.
    // display=true always requests SDR sRGB even with a Native limits policy.
    // Orientation is applied once, static layers are coalesced and alpha is
    // returned unassociated. Animation/additional displayed frames are rejected.
    LFS_IMAGE_CODECS_API bool decode_jxl(const std::filesystem::path&, Image&, std::string&, bool display = false, const JxlDecodeLimits& = {});
    LFS_IMAGE_CODECS_API bool decode_jxl(std::span<const uint8_t>, Image&, std::string&, bool display = false, const JxlDecodeLimits& = {});
    // Decode/validate completely before allocating or writing caller storage.
    // libjxl emits the requested sample type/channels into private staging.
    LFS_IMAGE_CODECS_API bool decode_jxl_to_buffer(const std::filesystem::path&, DecodeTarget&, Probe&, std::string&, bool display = false, const JxlDecodeLimits& = {});
    LFS_IMAGE_CODECS_API bool decode_jxl_to_buffer(std::span<const uint8_t>, DecodeTarget&, Probe&, std::string&, bool display = false, const JxlDecodeLimits& = {});
} // namespace lfs::core::image_codecs
