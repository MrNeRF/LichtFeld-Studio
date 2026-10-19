/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "image_codecs.hpp"

namespace lfs::core::image_codecs {
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
