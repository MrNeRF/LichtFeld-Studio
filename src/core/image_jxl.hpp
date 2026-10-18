/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "image_codecs.hpp"

namespace lfs::core::image_codecs {
    // Per-call limits, not an aggregate process RAM budget. The decoder's
    // allocations are separate from the owning output and buffered file input.
    struct JxlDecodeLimits {
        size_t pixels = 64u * 1024 * 1024;
        size_t output_bytes = size_t{1} << 30;
        size_t decoder_bytes = size_t{1} << 30;
        size_t icc_bytes = 1u << 20;
        size_t buffered_input_bytes = 4u << 20;
        size_t encoded_bytes = size_t{2} << 30;
    };

    LFS_IMAGE_CODECS_API bool is_jxl(std::span<const uint8_t>);
    LFS_IMAGE_CODECS_API bool probe_jxl(const std::filesystem::path&, Probe&, std::string&, const JxlDecodeLimits& = {});
    // Values retain the original color encoding unless display=true requests
    // supported SDR conversion to sRGB. Orientation is applied once and alpha
    // is returned unassociated. Animation/additional image frames are rejected.
    LFS_IMAGE_CODECS_API bool decode_jxl(const std::filesystem::path&, Image&, std::string&, bool display = false, const JxlDecodeLimits& = {});
    LFS_IMAGE_CODECS_API bool decode_jxl(std::span<const uint8_t>, Image&, std::string&, bool display = false, const JxlDecodeLimits& = {});
    // Decode/validate completely before allocating or writing caller storage.
    LFS_IMAGE_CODECS_API bool decode_jxl_to_buffer(const std::filesystem::path&, DecodeTarget&, Probe&, std::string&, bool display = false, const JxlDecodeLimits& = {});
    LFS_IMAGE_CODECS_API bool decode_jxl_to_buffer(std::span<const uint8_t>, DecodeTarget&, Probe&, std::string&, bool display = false, const JxlDecodeLimits& = {});
} // namespace lfs::core::image_codecs
