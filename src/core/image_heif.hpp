/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "image_codecs.hpp"
namespace lfs::core::image_codecs {
    LFS_IMAGE_CODECS_API bool is_heif(std::span<const std::uint8_t>);
    LFS_IMAGE_CODECS_API bool probe_heif(const std::filesystem::path&, Probe&, std::string&);
    // Raw decode preserves encoded RGB; display decode maps supported SDR profiles to sRGB.
    LFS_IMAGE_CODECS_API bool decode_heif(const std::filesystem::path&, Image&, std::string&, bool display = false);
    LFS_IMAGE_CODECS_API bool decode_heif(std::span<const std::uint8_t>, Image&, std::string&, bool display = false);
    LFS_IMAGE_CODECS_API bool decode_heif_to_buffer(const std::filesystem::path&, DecodeTarget&, Probe&, std::string&, bool display = false);
    LFS_IMAGE_CODECS_API bool decode_heif_to_buffer(std::span<const std::uint8_t>, DecodeTarget&, Probe&, std::string&, bool display = false);
} // namespace lfs::core::image_codecs
