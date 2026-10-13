/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "image_codecs.hpp"
namespace lfs::core::image_codecs {
    LFS_IMAGE_CODECS_API bool probe_heif(const std::filesystem::path&, Probe&, std::string&);
    // Raw decode preserves encoded RGB; display decode maps supported SDR profiles to sRGB.
    LFS_IMAGE_CODECS_API bool decode_heif(const std::filesystem::path&, Image&, std::string&, bool display = false);
} // namespace lfs::core::image_codecs
